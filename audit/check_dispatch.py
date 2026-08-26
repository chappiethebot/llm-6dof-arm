"""Verify every model in the data maps to the intended vocabulary."""
import csv, collections
from bench.backfill_tokens_v3 import Tokenizers
csv.field_size_limit(2**31 - 1)

models = set()
for f in ["rung1", "rag", "lora", "cls"]:
    for r in csv.DictReader(open(f"bench/results_heldout_v3_{f}_scored.csv",
                                 encoding="utf-8", newline="")):
        models.add(r["model"])

t = Tokenizers()
print(f"{'model':28}{'vocabulary label':40}{'hello world'}")
for m in sorted(models):
    enc, src = t.for_model(m)
    print(f"{m:28}{src:40}{enc('hello world')}")
