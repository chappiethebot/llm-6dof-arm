"""Drive the three NEW demo clients through run_demo's STAGE 2 exactly.

The arm cannot be driven on this machine: mujoco.dll is blocked by Windows
Application Control (WinError 4551), the same policy that blocks
llama-tokenize.exe. This exercises everything run_demo does up to the motion
call -- client construction, strategy constraint, run_once, extract, validate --
using the same command the task asked for.
"""
import os, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "sim"))

from llm.prompts import load_schema
from llm.pipeline import run_once
from sim.run_demo import build_roster, strategies_for

COMMAND = "put the bottle in the upper left corner of the table"
TARGETS = ["ollama:qwen2.5-0.5b-lora", "ollama:qwen2.5-3b-lora", "classifier:distilbert"]

schema = load_schema(os.path.join(ROOT, "schema", "schema_v2.json"))
roster = {c.name: c for c, _ in build_roster()}

print(f"command: {COMMAND!r}")
print(f"schema : schema_v2.json ({len(schema['locations'])} locations)\n")

for name in TARGETS:
    client = roster[name]
    strategy = strategies_for(name)[0]      # each has exactly one
    t0 = time.perf_counter()
    r = run_once(COMMAND, client, strategy, schema, examples=None)
    ms = (time.perf_counter() - t0) * 1000
    print(f"{name}  [{strategy}]")
    print(f"   understood as: {r.obj}")
    print(f"   stage={r.stage}  ok={r.ok}  {ms:.0f} ms")
    if r.obj:
        loc = r.obj.get("location")
        print(f"   -> would drive arm to: {loc}"
              f"   (expected grid_A1 for 'upper left corner')")
    if not r.ok:
        print(f"   reasons: {r.reasons}")
    print()
