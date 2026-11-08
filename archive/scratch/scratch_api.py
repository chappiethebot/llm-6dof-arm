import inspect, torch, transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

print("transformers", transformers.__version__)
print("bf16 supported:", torch.cuda.is_bf16_supported())
print("=" * 72)

print("Trainer.__init__ params:")
print(" ", sorted(p for p in inspect.signature(Trainer.__init__).parameters if p != "self"))
print()
for k in ["tokenizer", "processing_class", "data_collator", "train_dataset",
          "eval_dataset", "compute_metrics", "args", "model"]:
    print(f"  {k:20s} {'YES' if k in inspect.signature(Trainer.__init__).parameters else '-- MISSING --'}")
print("=" * 72)

MID = "Qwen/Qwen2.5-0.5B-Instruct"
model = None
for kw in ["dtype", "torch_dtype"]:
    try:
        model = AutoModelForCausalLM.from_pretrained(MID, **{kw: torch.bfloat16})
        print(f"from_pretrained accepts: {kw}")
        break
    except TypeError as e:
        print(f"from_pretrained REJECTS {kw}: {e}")
if model is None:
    raise SystemExit("could not load model")

print("param dtype:", next(model.parameters()).dtype)
print("=" * 72)

names = set()
for n, m in model.named_modules():
    if isinstance(m, torch.nn.Linear):
        names.add(n.split(".")[-1])
print("Linear leaf module names:", sorted(names))
print()
print("first decoder layer, full paths:")
for n, m in model.named_modules():
    if n.startswith("model.layers.0.") and isinstance(m, torch.nn.Linear):
        print("   ", n, tuple(m.weight.shape))
print("=" * 72)

from peft import LoraConfig, get_peft_model
cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
)
pm = get_peft_model(model, cfg)
pm.print_trainable_parameters()
print("=" * 72)

try:
    ta = TrainingArguments(output_dir="ft/_probe", num_train_epochs=1, bf16=True,
                           eval_strategy="epoch", save_strategy="no",
                           warmup_steps=10, report_to="none", seed=20260820)
    print("TrainingArguments constructed OK")
except Exception as e:
    print("TrainingArguments FAILED:", type(e).__name__, e)
