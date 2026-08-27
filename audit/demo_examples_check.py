"""Prove few_shot and rag are no longer silently dead in run_demo.

BEFORE: run_demo hardcoded `examples = None`, and build_prompt raises for both
strategies when examples are empty -> run_once records prompt_error -> the arm
never moves, with no error shown to the user.
"""
import os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "sim"))

from llm.prompts import load_schema
from llm.pipeline import run_once
from llm.clients import OllamaClient
from bench.dataset import load_seen_split

COMMAND = "put the bottle in the upper left corner of the table"
schema = load_schema(os.path.join(ROOT, "schema", "schema_v2.json"))
client = OllamaClient("qwen2.5:3b")

print(f"command: {COMMAND!r}\n")

# --- what the OLD code did: examples=None ---
r = run_once(COMMAND, client, "few_shot", schema, examples=None)
print(f"few_shot, examples=None  (old behaviour)")
print(f"   stage={r.stage}  ok={r.ok}  obj={r.obj}")
print(f"   reasons={r.reasons}\n")

# --- what the demo does now ---
ex, _ = load_seen_split(k=3, seen_path=os.path.join(ROOT, "data", "seen_v2.jsonl"))
r = run_once(COMMAND, client, "few_shot", schema, examples=ex)
print(f"few_shot, {len(ex)} exemplars from seen_v2  (new behaviour)")
print(f"   stage={r.stage}  ok={r.ok}  obj={r.obj}\n")

from llm.retrieve import Retriever
rt = Retriever(quiet=True)
res = rt.retrieve(COMMAND, k=3)
r = run_once(COMMAND, client, "rag", schema, examples=res.examples)
print(f"rag, {len(res.examples)} retrieved (ids={res.ids}, {res.latency_s*1000:.0f} ms)")
print(f"   stage={r.stage}  ok={r.ok}  obj={r.obj}")
