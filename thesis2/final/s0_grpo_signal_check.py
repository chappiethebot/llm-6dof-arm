"""GRPO inference in thesis2gpu only; signal uses train_v2, selection uses dev_v2.

Default: eight temperature-1 completions per train command with checkpoint-87.
--evaluate-adapter PATH --output PATH: greedy dev selection for the RL callback.
No held-out loader exists here. The V5 prompt is imported unchanged.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
from s0_grpo_adapter_probe import BASE, PROBE, SYSTEM

GPU_PYTHON = Path("D:/condaenvs/thesis2gpu/python.exe")
SFT = C.REPO / "ft/runs/qwen3b_lora/checkpoint-87"
SIGNAL = C.OUT / "grpo_signal_check.json"


def require_env(python):
    if Path(sys.executable).resolve() != Path(python).resolve():
        raise SystemExit("[Certain] Run this command with %s" % python)


def require_probe():
    record = C.OUT / "grpo_adapter_probe.json"
    if not record.exists():
        raise SystemExit("Run s0_grpo_adapter_probe.py --save in thesis2rl, then --load in thesis2gpu.")
    proof = json.loads(record.read_text(encoding="utf-8"))
    if not proof.get("passed") or proof.get("decoded") != 5 or proof.get("total") != 5:
        raise SystemExit("STOP: cross-environment adapter probe did not pass 5/5.")
    if proof["adapter_sha256"] != C.sha256_of(PROBE / "adapter_model.safetensors"):
        raise SystemExit("STOP: probe adapter changed; repeat the compatibility check.")


def prompt(command):
    from bench.run_cell import apply_legend
    from llm import prompts as P
    apply_legend("v5")
    text = P.build_prompt(command, "zero_shot", P.load_schema(str(C.SCHEMA_FILE)))
    return ("<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n"
            "<|im_start|>assistant\n" % (SYSTEM, text))


def exact_reward(raw, expected):
    try:
        obj = json.loads(raw)
    except (ValueError, TypeError):
        return 0.0
    return float(isinstance(obj, dict) and C.score(obj, expected).strict_match)


def rewards(completions, expected, **kwargs):
    # Dataset prompts are plain strings, so TRL returns plain string completions.
    return [exact_reward(raw, json.loads(truth)) for raw, truth in zip(completions, expected)]


def load_adapter(adapter):
    require_env(GPU_PYTHON)
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(BASE)
    base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16).to("cuda")
    model = PeftModel.from_pretrained(base, str(adapter)).merge_and_unload().eval()
    return model, tok


def generate(model, tok, command, sample=False):
    import torch
    inputs = tok(prompt(command), add_special_tokens=False, return_tensors="pt").to("cuda")
    args = dict(max_new_tokens=64, do_sample=sample, use_cache=True,
                pad_token_id=tok.pad_token_id,
                eos_token_id=list({tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}))
    if sample:
        args.update(temperature=1.0, top_p=1.0, top_k=0)
    with torch.inference_mode():
        ids = model.generate(**inputs, **args)
    return tok.decode(ids[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)


def build_predict(adapter):
    model, tok = load_adapter(adapter)

    def predict(command):
        start = time.perf_counter()
        raw = generate(model, tok, command)
        try:
            obj = json.loads(raw)
            if not isinstance(obj, dict):
                obj = {}
        except ValueError:
            obj = {}
        return obj, dict(raw=raw, latency_ms=(time.perf_counter() - start) * 1000)
    return predict


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--evaluate-adapter", type=Path)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    require_env(GPU_PYTHON)
    require_probe()
    C.start_log("s0_grpo_dev" if args.evaluate_adapter else "s0_grpo_signal_check")
    # Rule 3, at startup. GRPO adds no legend of its own: the prompt is the
    # pre-existing V5 legend (which legitimately contains v3's `upper`/`bottom`
    # and is excluded, as in s0_salsa_full.py) and the training data is train_v2
    # as it is (rule 4). What THIS script contributes is the chat wrapper and the
    # system line, and those must carry no held-out row vocabulary.
    C.assert_no_heldout_vocab(SYSTEM, "<|im_start|>system", "<|im_start|>user",
                              "<|im_start|>assistant",
                              where="the GRPO chat wrapper and system line")
    import torch
    from transformers import set_seed
    set_seed(C.SEED)
    if args.evaluate_adapter:
        if args.output is None or not args.output.resolve().is_relative_to(C.OUT.resolve()):
            ap.error("--output must be under thesis2/out/final")
        model, tok = load_adapter(args.evaluate_adapter)
        rows = C.load_split(C.DEV)
        hits = sum(exact_reward(generate(model, tok, r["command"]), C.truth_of(r)) for r in rows)
        result = dict(dev_hits=int(hits), dev_total=len(rows), dev_action_match=hits / len(rows),
                      adapter_sha256=C.sha256_of(args.evaluate_adapter / "adapter_model.safetensors"),
                      inference_python=sys.executable)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        C.gate(True, ["[Certain] dev_v2 greedy: %d/%d" % (hits, len(rows))])
    if SIGNAL.exists():
        C.gate(True, ["[Certain] Existing signal check preserved: %s" % SIGNAL])
    model, tok = load_adapter(SFT)
    records = []
    rows = C.load_split(C.TRAIN)
    for i, row in enumerate(rows):
        outputs = [generate(model, tok, row["command"], sample=True) for _ in range(8)]
        scores = [exact_reward(raw, C.truth_of(row)) for raw in outputs]
        records.append(dict(command_id=i, command=row["command"], outputs=outputs,
                            rewards=scores, mixed=len(set(scores)) == 2))
        if (i + 1) % 10 == 0:
            print("[Certain] Signal check %d/%d prompts" % (i + 1, len(rows)), flush=True)
    mixed = sum(r["mixed"] for r in records)
    result = dict(seed=C.SEED, split=C.TRAIN, prompts=len(rows), group_size=8,
                  temperature=1.0, mixed_prompts=mixed, mixed_fraction=mixed / len(rows),
                  eligible=mixed * 10 >= len(rows), records=records,
                  sft_sha256=C.sha256_of(SFT / "adapter_model.safetensors"),
                  train_sha256=C.sha256_of(C.DATA / "train_v2.jsonl"),
                  peak_vram_mb=torch.cuda.max_memory_allocated() / 1e6)
    SIGNAL.write_text(json.dumps(result, indent=2), encoding="utf-8")
    C.gate(True, ["[Certain] Mixed rewards: %d/%d = %.4f" % (mixed, len(rows), result["mixed_fraction"]),
                  "[Certain] GRPO-on-SFT eligible" if result["eligible"]
                  else "[Certain] SKIP GRPO-on-SFT: fewer than 10% mixed prompts."])


if __name__ == "__main__":
    main()
