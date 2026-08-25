"""ft/merge_and_export.py — Rung 3, stage 2: adapter -> Ollama model.

Chain (route A, locked in Part 6 of the handoff):
  1. merge checkpoint-87 LoRA into the base model, bf16, HF format
  2. optional: dev_v2 pass on the MERGED BF16 model  (--eval-bf16)
  3. convert merged HF -> GGUF f16   (llama.cpp convert_hf_to_gguf.py)
  4. quantise f16 -> Q4_K_M          (llama.cpp b10582 llama-quantize.exe)
  5. write Modelfile + `ollama create`
  6. single-prompt smoke test so the artifact is known to load and answer

WHY Q4_K_M: Part 6 requires fine-tuned models be served at the SAME
quantisation as their base. `ollama show qwen2.5:3b` and `:0.5b` both report
Q4_K_M. Any other level makes the rung-1 vs rung-3 latency comparison invalid.

QUANTISATION LOSS: the measured merged delta on layers.0.q_proj was max 0.00195
(one bf16 quantum at that magnitude). Q4_K_M is a coarser grid. --eval-bf16
measures the pre-quantisation model on dev_v2 so "cost of quantisation" is a
reported number rather than an unexplained gap. Costs ~90s on 0.5b.

TOOL VERSIONS FOR THE METHODS CHAPTER:
  llama.cpp b10582 (win-cpu-x64)   |   peft 0.20.0   |   transformers 5.15.1
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ROOT       = Path(__file__).resolve().parent.parent
LLAMA_CPP  = Path(r"C:\Users\USER\llama.cpp")
CONVERTER  = LLAMA_CPP / "convert_hf_to_gguf.py"
QUANTIZE   = LLAMA_CPP / "bin" / "llama-quantize.exe"
EXPORT_DIR = ROOT / "ft" / "export"

CHECKPOINT = "checkpoint-87"          # epoch 3, earliest-plateau rule, BOTH models
QUANT_TYPE = "Q4_K_M"                 # matches the base models

MODELS = {
    "0.5b": {"base": "Qwen/Qwen2.5-0.5B-Instruct", "ollama": "qwen2.5-0.5b-lora"},
    "3b":   {"base": "Qwen/Qwen2.5-3B-Instruct",   "ollama": "qwen2.5-3b-lora"},
}

SMOKE_PROMPTS = [
    "can you grab the toy car for me",
    "put the bottle in the left side middle edge",
    "stop right now",
]


def run(cmd, cwd=None):
    print(f"\n$ {' '.join(str(c) for c in cmd)}\n")
    r = subprocess.run([str(c) for c in cmd], cwd=cwd)
    if r.returncode != 0:
        raise SystemExit(f"ABORT: command failed with exit code {r.returncode}")


# ---------------------------------------------------------------- stage 1

def merge(tag, cfg, out_dir):
    ck = ROOT / "ft" / "runs" / f"qwen{tag}_lora" / CHECKPOINT
    if not ck.exists():
        raise SystemExit(f"ABORT: {ck} not found")

    print(f"[1/5] merging {ck.name} into {cfg['base']}")
    tok = AutoTokenizer.from_pretrained(cfg["base"])
    base = AutoModelForCausalLM.from_pretrained(cfg["base"], dtype=torch.bfloat16)

    probe_before = base.model.layers[0].self_attn.q_proj.weight.detach().clone()
    merged = PeftModel.from_pretrained(base, str(ck)).merge_and_unload()
    probe_after = merged.model.layers[0].self_attn.q_proj.weight.detach()
    delta = (probe_after.float() - probe_before.float()).abs().max().item()

    residual = [n for n, _ in merged.named_modules() if "lora" in n.lower()]
    print(f"      max |delta| layers.0.q_proj : {delta:.6f}")
    print(f"      residual lora modules       : {len(residual)}")
    if delta == 0:
        raise SystemExit("ABORT: merge was a no-op — weights unchanged.")
    if residual:
        raise SystemExit("ABORT: lora modules survived merge_and_unload().")

    out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out_dir), safe_serialization=True)
    tok.save_pretrained(str(out_dir))
    print(f"      merged bf16 -> {out_dir}")
    return merged, tok


# ---------------------------------------------------------------- stage 2

@torch.no_grad()
def eval_bf16(model, tok, tag):
    """Pre-quantisation dev pass. Selection-metric code, reused verbatim."""
    sys.path.insert(0, str(ROOT))
    from ft.train_lora import dev_exact_match, read_jsonl

    rows = read_jsonl(ROOT / "ft" / "data" / f"dev_sft_{tag}.jsonl")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    m = dev_exact_match(model, tok, rows, device)
    print(f"[2/5] MERGED BF16 on dev_v2: json_match {m['json_match']*100:.1f}%  "
          f"parse {m['parse_rate']*100:.1f}%  (n={m['n']})")
    print("      ^ record this. The Q4_K_M number will be compared against it.")
    model.to("cpu")
    return m


# ---------------------------------------------------------------- stages 3-4

def to_gguf(merged_dir, tag):
    f16 = EXPORT_DIR / f"qwen{tag}-lora-f16.gguf"
    print(f"[3/5] converting -> {f16.name}")
    run([sys.executable, CONVERTER, merged_dir,
         "--outfile", f16, "--outtype", "f16"], cwd=LLAMA_CPP)
    if not f16.exists():
        raise SystemExit(f"ABORT: {f16} was not produced")
    print(f"      {f16.stat().st_size / 1e9:.2f} GB")
    return f16


def quantise(f16, tag):
    q = EXPORT_DIR / f"qwen{tag}-lora-{QUANT_TYPE}.gguf"
    print(f"[4/5] quantising -> {q.name}")
    run([QUANTIZE, f16, q, QUANT_TYPE])
    if not q.exists():
        raise SystemExit(f"ABORT: {q} was not produced")
    print(f"      {q.stat().st_size / 1e9:.2f} GB  "
          f"({q.stat().st_size / f16.stat().st_size * 100:.1f}% of f16)")
    return q


# ---------------------------------------------------------------- stage 5

def to_ollama(gguf, name):
    """SYSTEM line reproduces the stock qwen2.5 Modelfile exactly. The generate
    path renders it into the prompt, so it MUST match what rung 1 saw."""
    mf = EXPORT_DIR / f"Modelfile.{name}"
    mf.write_text(
        f'FROM {gguf}\n'
        f'TEMPLATE """{{{{- if .System }}}}<|im_start|>system\n'
        f'{{{{ .System }}}}<|im_end|>\n'
        f'{{{{ end }}}}{{{{ if .Prompt }}}}<|im_start|>user\n'
        f'{{{{ .Prompt }}}}<|im_end|>\n'
        f'{{{{ end }}}}<|im_start|>assistant\n'
        f'{{{{ .Response }}}}{{{{ if .Response }}}}<|im_end|>{{{{ end }}}}"""\n'
        f'SYSTEM You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n'
        f'PARAMETER stop "<|im_end|>"\n'
        f'PARAMETER temperature 0\n',
        encoding="utf-8",
    )
    print(f"[5/5] ollama create {name}")
    print(f"      Modelfile: {mf}")
    run(["ollama", "create", name, "-f", mf])
    return name


def smoke(name):
    print(f"\n=== SMOKE TEST: {name} ===")
    sys.path.insert(0, str(ROOT))
    from llm.prompts import build_prompt, load_schema
    from llm.clients import OllamaClient

    schema = load_schema(str(ROOT / "schema" / "schema_v2.json"))
    client = OllamaClient(name)
    for cmd in SMOKE_PROMPTS:
        p = build_prompt(cmd, "zero_shot", schema, examples=None)
        try:
            out = client.complete(p)
            ok = "OK  " if _parses(out) else "BAD "
        except Exception as e:
            out, ok = f"{type(e).__name__}: {e}", "ERR "
        print(f"  {ok}{cmd!r}\n       -> {out.strip()!r}")


def _parses(s):
    try:
        json.loads(s.strip())
        return True
    except Exception:
        return False


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--eval-bf16", action="store_true",
                    help="dev_v2 pass on the merged bf16 model before quantising")
    ap.add_argument("--keep-merged", action="store_true",
                    help="keep the merged HF dir (default: delete after GGUF)")
    ap.add_argument("--keep-f16", action="store_true",
                    help="keep the f16 GGUF (default: delete after quantise)")
    args = ap.parse_args()

    if not CONVERTER.exists():
        raise SystemExit(f"ABORT: converter not found at {CONVERTER}")
    if not QUANTIZE.exists():
        raise SystemExit(f"ABORT: llama-quantize.exe not found at {QUANTIZE}")

    tag = args.model
    cfg = MODELS[tag]
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    merged_dir = EXPORT_DIR / f"qwen{tag}-merged-bf16"

    print("=" * 72)
    print(f"MODEL      : {cfg['base']}")
    print(f"CHECKPOINT : {CHECKPOINT}  (epoch 3, earliest-plateau rule)")
    print(f"QUANT      : {QUANT_TYPE}  (matches base)")
    print(f"OLLAMA TAG : {cfg['ollama']}")
    print(f"llama.cpp  : b10582 win-cpu-x64")
    print("=" * 72)

    model, tok = merge(tag, cfg, merged_dir)
    if args.eval_bf16:
        eval_bf16(model, tok, tag)
    del model
    torch.cuda.empty_cache()

    f16 = to_gguf(merged_dir, tag)
    if not args.keep_merged:
        shutil.rmtree(merged_dir)
        print(f"      removed {merged_dir.name}")

    q = quantise(f16, tag)
    if not args.keep_f16:
        f16.unlink()
        print(f"      removed {f16.name}")

    to_ollama(q, cfg["ollama"])
    smoke(cfg["ollama"])

    print("\n" + "=" * 72)
    print(f"DONE — `ollama run {cfg['ollama']}` is live.")
    print("Record for methods: checkpoint-87, Q4_K_M, llama.cpp b10582.")
    print("=" * 72)


if __name__ == "__main__":
    main()