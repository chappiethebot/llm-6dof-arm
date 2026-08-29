"""2d: smoke-test the demo config across all 8 clients on 10 commands.

Demo config = V5 legend + constrained decoding. NOT a measurement: 10 commands,
1 run each, and the commands are drawn from held_out_v3 for realism only.
"""
import json, os, sys, time, collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "sim"))
os.environ.setdefault("OLLAMA_MODELS", r"D:\LLM")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from bench.run_cell import apply_legend
apply_legend("v5")

from llm.prompts import load_schema
from llm.pipeline import run_once
from llm.schema_json import to_json_schema, to_flat_json_schema
from sim.run_demo import build_roster, strategies_for
from bench.dataset import load_seen_split

S = load_schema(os.path.join(ROOT, "schema", "schema_v2.json"))
JS, FJS = to_json_schema(S), to_flat_json_schema(S)

# 10 commands: every verb, both grid and container locations, both row
# vocabularies. Drawn from held_out_v3 where possible.
CMDS = [
    ("put the bottle in the upper left corner of the table",
     {"action": "pick_place", "target": "bottle", "location": "grid_A1", "modifiers": []}),
    ("drop the rubiks cube in box 1",
     {"action": "pick_place", "target": "rubiks_cube", "location": "box_1", "modifiers": []}),
    ("move to the bottom right side corner of the table",
     {"action": "move_to", "location": "grid_C3", "modifiers": []}),
    ("go to box 2",
     {"action": "move_to", "location": "box_2", "modifiers": []}),
    ("bring me the toy car",
     {"action": "fetch", "target": "toy_car", "modifiers": []}),
    ("hold the bottle",
     {"action": "hold", "target": "bottle", "modifiers": []}),
    ("take a look at the rubiks cube",
     {"action": "inspect", "target": "rubiks_cube", "modifiers": []}),
    ("stop right now",
     {"action": "stop", "modifiers": []}),
    ("put the car in the middle of the table",
     {"action": "pick_place", "target": "toy_car", "location": "grid_B2", "modifiers": []}),
    ("teleport the bottle to the kitchen counter",
     {"action": "reject", "modifiers": []}),
]

exemplars, _ = load_seen_split(k=3, seen_path=os.path.join(ROOT, "data", "seen_v2.jsonl"))
roster = build_roster()


def canon(o):
    return {k: v for k, v in o.items() if not (k == "modifiers" and not v)}


print(f"demo config: legend v5 + constrained decoding | {len(CMDS)} commands")
print(f"{'client':26}{'strategy':11}{'valid':>7}{'correct':>9}{'p50 s':>8}  notes")
print("-" * 84)
summary = []
for client, strats in roster:
    strat = "zero_shot" if len(strats) > 1 else strats[0]
    if client.name.startswith("openai:"):
        client.json_schema = FJS
    elif client.name.startswith("ollama:"):
        client.json_schema = JS
    ex = exemplars if strat in ("few_shot", "rag") else None
    lat, valid, correct, notes = [], 0, 0, collections.Counter()
    for cmd, truth in CMDS:
        t = time.perf_counter()
        r = run_once(cmd, client, strat, S, examples=ex)
        lat.append(time.perf_counter() - t)
        if r.ok:
            valid += 1
            if canon(r.obj) == canon(truth):
                correct += 1
            else:
                notes[f"wrong:{r.obj.get('action')}"] += 1
        else:
            notes[r.stage] += 1
    p50 = sorted(lat)[len(lat) // 2]
    top = ", ".join(f"{k}x{v}" for k, v in notes.most_common(2))
    print(f"{client.name:26}{strat:11}{valid:>5}/10{correct:>7}/10{p50:>8.2f}  {top}")
    summary.append((client.name, valid, correct))

print()
ok = [n for n, v, c in summary if v == 10]
print(f"produced 10/10 STRUCTURALLY VALID: {len(ok)}/8 clients")
for n, v, c in summary:
    if v < 10:
        print(f"   NOT valid on all 10: {n} ({v}/10)")
