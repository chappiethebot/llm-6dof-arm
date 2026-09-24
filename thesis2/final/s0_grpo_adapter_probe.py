"""thesis2/final/s0_grpo_adapter_probe.py -- can a thesis2rl adapter load in thesis2gpu?

GRPO trains in the NEW env `thesis2rl` (it has TRL) and saves ONLY the LoRA
adapter. Every GRPO inference -- dev selection and both held-out splits -- runs
in `thesis2gpu` via peft. That only works if an adapter written by thesis2rl
loads in thesis2gpu, so this probe proves it BEFORE any GRPO training is run.
If the probe fails, STOP: training in an env whose artifact the inference env
cannot read would waste the run.

    # 1. in thesis2rl -- write a fresh, UNTRAINED adapter with the GRPO config
    D:\condaenvs\thesis2rl\python.exe thesis2\final\s0_grpo_adapter_probe.py --save

    # 2. in thesis2gpu -- load it and decode 5 train_v2 commands
    D:\condaenvs\thesis2gpu\python.exe thesis2\final\s0_grpo_adapter_probe.py --load

The adapter written by --save is UNTRAINED (LoRA B is zero-initialised, so it is
the identity and the model's output equals the base model's). That is deliberate:
the probe tests the SERIALISATION PATH, not quality. It is written to
thesis2/models/final/grpo_probe_adapter/ and is never registered as an arm.

LoRA config is the brief's, so the probe exercises exactly the tensor shapes the
real run will produce: r=16, alpha=32, dropout 0.05, target q/k/v/o/gate/up/down,
lm_head EXCLUDED.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C                                                      # noqa: E402

BASE = "Qwen/Qwen2.5-3B-Instruct"
PROBE = C.MODELS / "grpo_probe_adapter"
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LORA = dict(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=TARGETS, bias="none")
SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


def mode_save():
    import torch
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    import peft, transformers
    print("  env: peft %s, transformers %s, torch %s"
          % (peft.__version__, transformers.__version__, torch.__version__))
    try:
        import trl
        print("  trl %s (this env is the TRAINING env)" % trl.__version__)
    except ImportError:
        C.gate(False, ["--save must run in thesis2rl, which has TRL; "
                       "this env does not import trl"])
    m = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16)
    cfg = LoraConfig(task_type="CAUSAL_LM", **LORA)
    m = get_peft_model(m, cfg)
    n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print("  trainable LoRA parameters: %.2f M" % (n_train / 1e6))
    named = [n for n, _ in m.named_modules() if "lora_A" in n]
    print("  LoRA sites: %d" % len(named))
    assert not any("lm_head" in n for n in named), "lm_head must be excluded"
    print("  lm_head excluded: True")
    PROBE.mkdir(parents=True, exist_ok=True)
    m.save_pretrained(str(PROBE))
    files = sorted(p.name for p in PROBE.iterdir())
    print("  wrote %s: %s" % (PROBE.relative_to(C.REPO), files))
    C.gate(True, [
        "UNTRAINED probe adapter written by this env (trl present)",
        "files: %s" % ", ".join(files),
        "next: run --load in thesis2gpu; it must load and decode 5 train_v2 commands",
    ])


def mode_load():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    import peft, transformers
    print("  env: peft %s, transformers %s, torch %s"
          % (peft.__version__, transformers.__version__, torch.__version__))
    if "trl" in sys.modules:
        print("  note: trl is importable here too; the probe is still valid")
    if not PROBE.exists():
        C.gate(False, ["no probe adapter at %s -- run --save in thesis2rl first"
                       % PROBE.relative_to(C.REPO)])
    cfg = json.loads((PROBE / "adapter_config.json").read_text(encoding="utf-8"))
    print("  adapter_config: r=%s alpha=%s dropout=%s targets=%s"
          % (cfg.get("r"), cfg.get("lora_alpha"), cfg.get("lora_dropout"),
             sorted(cfg.get("target_modules", []))))
    unknown = [k for k in cfg
               if k not in peft.LoraConfig.__dataclass_fields__ and not k.startswith("_")]
    if unknown:
        print("  keys this peft does not know: %s" % unknown)

    from bench.run_cell import apply_legend
    from llm import prompts as P
    apply_legend("v5")
    schema = P.load_schema(str(C.SCHEMA_FILE))
    tok = AutoTokenizer.from_pretrained(BASE)
    base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16).to("cuda")
    try:
        model = PeftModel.from_pretrained(base, str(PROBE))
    except Exception as e:                                              # noqa: BLE001
        C.gate(False, ["ADAPTER DID NOT LOAD: %s: %s" % (type(e).__name__, e),
                       "STOP: do not train GRPO in thesis2rl until this loads."])
    model = model.merge_and_unload().eval()
    print("  adapter loaded and merged OK")

    IM_END = tok.convert_tokens_to_ids("<|im_end|>")
    STOP = {IM_END, tok.eos_token_id}
    rows = C.load_split(C.TRAIN)[:5]
    ok = 0
    for r in rows:
        txt = ("<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n"
               "<|im_start|>assistant\n" % (SYSTEM, P.build_prompt(r["command"],
                                                                   "zero_shot", schema)))
        ids = tok(txt, add_special_tokens=False)["input_ids"]
        with torch.no_grad():
            out = model(input_ids=torch.tensor([ids], device="cuda"), use_cache=True)
            past, nxt, gen = out.past_key_values, int(out.logits[0, -1].argmax()), []
            while len(gen) < 64 and nxt not in STOP:
                gen.append(nxt)
                out = model(input_ids=torch.tensor([[nxt]], device="cuda"),
                            past_key_values=past, use_cache=True)
                past, nxt = out.past_key_values, int(out.logits[0, -1].argmax())
        raw = tok.decode(gen, skip_special_tokens=True)
        try:
            obj = json.loads(raw)
            good = isinstance(obj, dict) and obj.get("action") in C.VERBS
        except Exception:
            obj, good = {}, False
        ok += good
        print("     %-46s -> %s %s" % (r["command"][:46], raw[:64],
                                       "OK" if good else "*** NOT A SCHEMA OBJECT ***"))
    (C.OUT / "grpo_adapter_probe.json").write_text(json.dumps(dict(
        passed=ok == len(rows), decoded=ok, total=len(rows),
        adapter_sha256=C.sha256_of(PROBE / "adapter_model.safetensors"),
        inference_python=sys.executable, peft=peft.__version__,
        transformers=transformers.__version__, torch=torch.__version__,
    ), indent=2), encoding="utf-8")
    C.gate(ok == len(rows), [
        "adapter written in thesis2rl LOADED in thesis2gpu via peft %s" % peft.__version__,
        "decoded %d/%d train_v2 commands into schema objects" % (ok, len(rows)),
        "the probe adapter is UNTRAINED, so output equals the base model -- this "
        "tests the serialisation path only",
        "GO: GRPO may train in thesis2rl and infer in thesis2gpu" if ok == len(rows)
        else "STOP: the load path is broken; do not train.",
    ])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--save", action="store_true", help="run in thesis2rl")
    g.add_argument("--load", action="store_true", help="run in thesis2gpu")
    a = ap.parse_args()
    C.start_log("s0_grpo_adapter_probe_%s" % ("save" if a.save else "load"))
    mode_save() if a.save else mode_load()


if __name__ == "__main__":
    main()
