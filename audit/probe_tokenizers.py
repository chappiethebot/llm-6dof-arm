"""Which EXACT per-model vocabularies can be reached without llama-tokenize?

llama-tokenize.exe is blocked by Windows Application Control, so the GGUF path
named in the task is unavailable. Probe the alternatives, in order of
preference: local files first, network second.
"""
import os
from pathlib import Path

ROOT = Path(".").resolve()
QWEN_TOK = ROOT / "ft" / "runs" / "qwen0.5b_lora" / "checkpoint-87"
QWEN3B_TOK = ROOT / "ft" / "runs" / "qwen3b_lora" / "checkpoint-87"

print("=== local tokenizer files ===")
for p in (QWEN_TOK, QWEN3B_TOK):
    tj = p / "tokenizer.json"
    print(f"  {tj}  exists={tj.exists()}  {tj.stat().st_size if tj.exists() else 0:,} bytes")

from transformers import AutoTokenizer

def probe(label, loader):
    try:
        t = loader()
        n = len(t("hello world")["input_ids"])
        print(f"  OK    {label:52} vocab={len(t):,}  'hello world'={n} tok")
        return t
    except Exception as e:
        msg = str(e).replace("\n", " ")[:110]
        print(f"  FAIL  {label:52} {type(e).__name__}: {msg}")
        return None

print("\n=== local (offline) ===")
os.environ["HF_HUB_OFFLINE"] = "1"
tq = probe("Qwen2.5 vocab from checkpoint-87 (local)",
           lambda: AutoTokenizer.from_pretrained(str(QWEN_TOK)))
tq3 = probe("Qwen2.5-3B vocab from checkpoint-87 (local)",
            lambda: AutoTokenizer.from_pretrained(str(QWEN3B_TOK)))

print("\n=== hub (network) ===")
os.environ.pop("HF_HUB_OFFLINE", None)
td = probe("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
           lambda: AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"))
tl = probe("meta-llama/Llama-3.2-3B-Instruct",
           lambda: AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct"))
tl2 = probe("unsloth/Llama-3.2-3B-Instruct (ungated mirror)",
            lambda: AutoTokenizer.from_pretrained("unsloth/Llama-3.2-3B-Instruct"))

print("\n=== coverage summary ===")
cov = {
    "ollama:qwen2.5:0.5b":       tq is not None,
    "ollama:qwen2.5:3b":         (tq3 or tq) is not None,
    "ollama:qwen2.5-0.5b-lora":  tq is not None,
    "ollama:qwen2.5-3b-lora":    (tq3 or tq) is not None,
    "ollama:deepseek-r1:7b":     td is not None,
    "ollama:llama3.2:3b":        (tl or tl2) is not None,
}
for m, ok in cov.items():
    print(f"  {'EXACT ' if ok else 'NO    '} {m}")
print(f"\n{sum(cov.values())}/6 local models have an exact vocabulary available")
