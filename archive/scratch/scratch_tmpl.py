import inspect, transformers, peft, torch
from transformers import AutoTokenizer

print("transformers", transformers.__version__, "| peft", peft.__version__)
print("=" * 70)

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

got = tok.apply_chat_template(
    [{"role": "user", "content": "PROMPT_BODY"}],
    tokenize=False,
    add_generation_prompt=True,
)

expected = (
    "<|im_start|>system\n"
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n"
    "PROMPT_BODY<|im_end|>\n"
    "<|im_start|>assistant\n"
)

print("HF   :", repr(got))
print()
print("OLLAMA:", repr(expected))
print()
print("BYTE-IDENTICAL:", got == expected)
print("=" * 70)

print("eos:", repr(tok.eos_token), tok.eos_token_id)
print("pad:", repr(tok.pad_token), tok.pad_token_id)
print("im_end id:", tok.convert_tokens_to_ids("<|im_end|>"))
print("=" * 70)

from transformers import TrainingArguments
sig = inspect.signature(TrainingArguments.__init__)
for k in ["output_dir","num_train_epochs","per_device_train_batch_size",
          "gradient_accumulation_steps","learning_rate","bf16","logging_steps",
          "eval_strategy","evaluation_strategy","save_strategy","warmup_ratio",
          "lr_scheduler_type","report_to","seed"]:
    print(f"  {k:32s} {'YES' if k in sig.parameters else '-- MISSING --'}")
print("=" * 70)

from peft import LoraConfig
psig = inspect.signature(LoraConfig.__init__)
print("LoraConfig params:", sorted(p for p in psig.parameters if p != "self"))
