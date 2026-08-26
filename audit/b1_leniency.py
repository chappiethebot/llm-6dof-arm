"""B1 -- enumerate every leniency in the STRICT policy by probing llm.score."""
from llm.score import score

truth = {"action": "pick_place", "target": "bottle", "location": "box_1", "modifiers": []}

probes = {
    # --- claimed leniency ---
    "modifiers absent":            {"action": "pick_place", "target": "bottle", "location": "box_1"},
    "modifiers []":                {"action": "pick_place", "target": "bottle", "location": "box_1", "modifiers": []},
    # --- UNDOCUMENTED: _IGNORE_IF_EMPTY drops ANY falsy value, not just [] ---
    "modifiers null":              {"action": "pick_place", "target": "bottle", "location": "box_1", "modifiers": None},
    "modifiers empty string":      {"action": "pick_place", "target": "bottle", "location": "box_1", "modifiers": ""},
    "modifiers empty dict":        {"action": "pick_place", "target": "bottle", "location": "box_1", "modifiers": {}},
    "modifiers 0":                 {"action": "pick_place", "target": "bottle", "location": "box_1", "modifiers": 0},
    "modifiers False":             {"action": "pick_place", "target": "bottle", "location": "box_1", "modifiers": False},
    # --- key order (dict equality: should be free) ---
    "key order reversed":          {"modifiers": [], "location": "box_1", "target": "bottle", "action": "pick_place"},
    # --- things strict must REJECT ---
    "case variant BOX_1":          {"action": "pick_place", "target": "bottle", "location": "BOX_1"},
    "separator variant 'box 1'":   {"action": "pick_place", "target": "bottle", "location": "box 1"},
    "alias key 'verb'":            {"verb": "pick_place", "target": "bottle", "location": "box_1"},
    "nested action object":        {"action": {"verb": "pick_place", "target": "bottle", "location": "box_1"}},
    "extra unknown key":           {"action": "pick_place", "target": "bottle", "location": "box_1", "confidence": 0.9},
    "extra key falsy value":       {"action": "pick_place", "target": "bottle", "location": "box_1", "note": ""},
    "wrong verb":                  {"action": "move_to", "target": "bottle", "location": "box_1"},
    "wrong location":              {"action": "pick_place", "target": "bottle", "location": "box_2"},
    "missing location":            {"action": "pick_place", "target": "bottle"},
    "whitespace in value":         {"action": "pick_place", "target": "bottle ", "location": "box_1"},
}

print(f"TRUTH: {truth}\n")
print(f"{'probe':30}{'strict':>8}{'norm':>7}   category")
print("-" * 70)
for name, pred in probes.items():
    r = score(pred, truth)
    print(f"{name:30}{str(r.strict_match):>8}{str(r.normalized_match):>7}   {r.category}")

print()
print("--- truth-side symmetry: does a NON-EMPTY modifiers truth still compare? ---")
truth_mod = {"action": "pick_place", "target": "bottle", "location": "box_1", "modifiers": ["gently"]}
for name, pred in {
    "pred omits modifiers":   {"action": "pick_place", "target": "bottle", "location": "box_1"},
    "pred has ['gently']":    {"action": "pick_place", "target": "bottle", "location": "box_1", "modifiers": ["gently"]},
    "pred has ['slowly']":    {"action": "pick_place", "target": "bottle", "location": "box_1", "modifiers": ["slowly"]},
    "pred order differs":     {"action": "pick_place", "target": "bottle", "location": "box_1", "modifiers": ["gently", "fast"]},
}.items():
    r = score(pred, truth_mod)
    print(f"  {name:24} strict={str(r.strict_match):5} norm={str(r.normalized_match):5} cat={r.category}")

print()
print("--- does a non-empty modifiers row exist anywhere in held_out_v3? ---")
import json
n = tot = 0
for line in open("data/held_out_v3.jsonl", encoding="utf-8"):
    line = line.strip()
    if not line:
        continue
    tot += 1
    row = json.loads(line)
    exp = row.get("expected", row)
    if exp.get("modifiers"):
        n += 1
print(f"  {n} of {tot} held_out_v3 rows carry a NON-EMPTY modifiers list")
