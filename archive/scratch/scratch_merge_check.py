import torch, json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

TAG, BASE = "0.5b", "Qwen/Qwen2.5-0.5B-Instruct"
CK = Path("ft/runs") / f"qwen{TAG}_lora" / "checkpoint-87"

print("=== chat template comparison ===")
saved = (CK / "chat_template.jinja").read_text(encoding="utf-8")
tok_ck = AutoTokenizer.from_pretrained(str(CK))
tok_base = AutoTokenizer.from_pretrained(BASE)

msgs = [{"role": "user", "content": "PROMPT_BODY"}]
a = tok_ck.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
b = tok_base.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
print("CHECKPOINT:", repr(a))
print("BASE      :", repr(b))
print("TEMPLATE UNCHANGED:", a == b)
print()

print("=== merge sanity ===")
base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16)
before = base.model.layers[0].self_attn.q_proj.weight.detach().clone()

peft = PeftModel.from_pretrained(base, str(CK))
merged = peft.merge_and_unload()
after = merged.model.layers[0].self_attn.q_proj.weight.detach()

delta = (after.float() - before.float()).abs()
print("layers.0.self_attn.q_proj")
print(f"  max |delta| : {delta.max().item():.6f}")
print(f"  mean |delta|: {delta.mean().item():.6f}")
print(f"  WEIGHTS CHANGED: {delta.max().item() > 0}")
print(f"  merged type : {type(merged).__name__}")
print(f"  dtype       : {next(merged.parameters()).dtype}")

has_lora = [n for n, _ in merged.named_modules() if "lora" in n.lower()]
print(f"  residual lora modules: {len(has_lora)} (must be 0)")
