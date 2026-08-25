import inspect, json, transformers
from transformers import TrainingArguments

print("=== warmup args present in 5.15.1 ===")
sig = inspect.signature(TrainingArguments.__init__)
print([p for p in sig.parameters if "warm" in p.lower()])
print()

print("=== schema files on disk ===")
import os
for f in sorted(os.listdir("schema")):
    print("  schema/" + f, os.path.getsize(os.path.join("schema", f)), "bytes")
print()

print("=== every mention of load_schema / schema path in the repo ===")
import pathlib
for p in pathlib.Path(".").rglob("*.py"):
    if ".git" in str(p):
        continue
    for i, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if "schema" in line and (".json" in line or "load_schema" in line):
            print(f"  {p}:{i}: {line.strip()}")
print()

print("=== first 3 rows of train_v2.jsonl ===")
with open("data/train_v2.jsonl", encoding="utf-8") as f:
    for i, line in zip(range(3), f):
        row = json.loads(line)
        print("  keys:", list(row.keys()))
        print("  raw :", json.dumps(row, ensure_ascii=False))
        print()
