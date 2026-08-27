"""Check 1: does the demo's retrieval + few-shot path match the measurement's?"""
import os, sys, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "sim"))

from llm.retrieve import Retriever, CORPUS_PATH, DEFAULT_K
from bench.dataset import load_seen_split

print("=== RAG CORPUS ===")
print(f"llm/retrieve.py CORPUS_PATH  = {CORPUS_PATH}")
rt = Retriever(quiet=True)
print(f"demo Retriever(quiet=True)   -> corpus_path = {rt.corpus_path}")
print(f"                             -> rows        = {len(rt.rows)}")
print(f"                             -> resolved    = {os.path.abspath(rt.corpus_path)}")

for f in ["train_v2.jsonl", "seen_v2.jsonl", "dev_v2.jsonl", "held_out_v3.jsonl"]:
    p = os.path.join(ROOT, "data", f)
    n = sum(1 for l in open(p, encoding="utf-8") if l.strip())
    same = "  <== THIS ONE" if n == len(rt.rows) and f == os.path.basename(str(rt.corpus_path)) else ""
    print(f"   data/{f:20} {n:4} rows{same}")

corpus_cmds = {r.get("command", "") for r in rt.rows}
seen = {json.loads(l)["command"] for l in open(os.path.join(ROOT, "data", "seen_v2.jsonl"), encoding="utf-8") if l.strip()}
held = {json.loads(l)["command"] for l in open(os.path.join(ROOT, "data", "held_out_v3.jsonl"), encoding="utf-8") if l.strip()}
print(f"\n   corpus  INTERSECT  held_out_v3 = {len(corpus_cmds & held)}  (must be 0)")
print(f"   corpus  SUBSET-OF  seen_v2     = {corpus_cmds <= seen}  (train_v2 is carved from seen_v2)")

print(f"\n=== k PARITY ===")
print(f"   retrieve.DEFAULT_K = {DEFAULT_K}")
print(f"   demo calls retrieve(command, k=3); few-shot loads k=3 -> same shot count")

print(f"\n=== FEW-SHOT EXEMPLARS ===")
ex, _ = load_seen_split(k=3, seen_path=os.path.join(ROOT, "data", "seen_v2.jsonl"))
print(f"   source: data/seen_v2.jsonl, {len(ex)} exemplars")
for e in ex:
    print(f"     {e['command']!r}  ->  {e['expected'].get('location','-')}")
ex_cmds = {e["command"] for e in ex}
print(f"\n   exemplars  INTERSECT  held_out_v3 = {len(ex_cmds & held)}  (must be 0)")

print("\n=== CAN THE EXECUTED COMMAND APPEAR AMONG THE EXAMPLES? ===")
print("   rag      : retrieve(exclude_exact=True) is the DEFAULT and the demo")
print("              relies on it -> an exact self-match is dropped.")
print("   few_shot : exemplars are FIXED. In the harness the scored rows come")
print("              from held_out_v3, which is disjoint from seen_v2 (0 above),")
print("              so it cannot happen. In the DEMO the user types free text,")
print("              so typing an exemplar verbatim would show the model its own")
print("              answer. Demo-only; affects no measured number.")
for c in sorted(ex_cmds):
    print(f"              avoid typing verbatim: {c!r}")
