"""Verify V4 changes the legend and NOTHING else. Read-only."""
import difflib, os, sys, json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from llm import prompts as v3, prompts_v4 as v4
from llm.prompts import load_schema

schema = load_schema(os.path.join(ROOT, "schema", "schema_v2.json"))
A, B = v4.legend_text_v3(schema), v4.legend_text(schema)

print("=" * 96)
print("V3 LEGEND")
print("=" * 96)
print(A)
print("=" * 96)
print("V4 LEGEND")
print("=" * 96)
print(B)
print("=" * 96)
print("UNIFIED DIFF")
print("=" * 96)
for line in difflib.unified_diff(A.splitlines(), B.splitlines(),
                                 "legend_v3", "legend_v4", lineterm="", n=1):
    print(line)

# --- no word may serve as BOTH a row and a column label -------------------
print()
print("=" * 96)
print("CHECK 1: does any word label BOTH a row and a column?")
print("=" * 96)
import re
for name, txt in (("v3", A), ("v4", B)):
    rows = re.search(r"Rows[^\n]*", txt).group(0)
    cols = re.search(r"Columns[^\n]*", txt).group(0)
    rw = set(re.findall(r"=\s*([a-z]+)", rows))
    cw = set(re.findall(r"=\s*([a-z]+)", cols))
    overlap = rw & cw
    print(f"  {name}: row labels={sorted(rw)}  column labels={sorted(cw)}")
    print(f"      overlap = {sorted(overlap) or 'NONE'}  "
          f"-> {'FAIL' if overlap else 'PASS'}")

# --- runtime-symbol vocabulary actually occurs in commands ----------------
print()
print("=" * 96)
print("CHECK 2: does the legend define the runtime symbols in words commands use?")
print("=" * 96)
SPLITS = ["seen_v2", "train_v2", "dev_v2", "held_out_v3"]
cmds = []
for s in SPLITS:
    p = os.path.join(ROOT, "data", f"{s}.jsonl")
    cmds += [json.loads(l)["command"].lower()
             for l in open(p, encoding="utf-8") if l.strip()]
for name, txt in (("v3", A), ("v4", B)):
    print(f"  {name}:")
    for wd in ("furthest", "farthest", "maximum", "distance", "reach",
               "empty", "free", "want", "anywhere", "nearest"):
        in_legend = re.search(rf"\b{wd}\b", txt) is not None
        n_cmds = sum(1 for c in cmds if re.search(rf"\b{wd}\b", c))
        if in_legend:
            verdict = "OK" if n_cmds else "<-- IN LEGEND, IN ZERO COMMANDS"
            print(f"      {wd:10} in legend, {n_cmds:4} commands  {verdict}")

# --- everything except the legend is byte-identical -----------------------
print()
print("=" * 96)
print("CHECK 3: is build_prompt identical apart from the legend block?")
print("=" * 96)
rows = [json.loads(l) for l in
        open(os.path.join(ROOT, "data", "held_out_v3.jsonl"), encoding="utf-8")
        if l.strip()]
ex = [{"command": "put the bottle in the far left corner",
       "expected": {"action": "pick_place", "target": "bottle",
                    "location": "grid_A1", "modifiers": []}}] * 3
STRATS = ["zero_shot", "few_shot", "cot", "role_based", "rag", "classifier"]
bad = 0
checked = 0
for strat in STRATS:
    e = ex if strat in ("few_shot", "rag") else None
    for r in rows:
        p3 = v3.build_prompt(r["command"], strat, schema, examples=e)
        p4 = v4.build_prompt(r["command"], strat, schema, examples=e)
        checked += 1
        # remove the legend from each and require the remainder to match exactly
        if p3.replace(A, "") != p4.replace(B, ""):
            bad += 1
            if bad == 1:
                print("  MISMATCH beyond the legend:")
                for l in difflib.unified_diff(p3.replace(A, "").splitlines(),
                                              p4.replace(B, "").splitlines(),
                                              "v3-minus-legend", "v4-minus-legend",
                                              lineterm="", n=1):
                    print("   ", l)
print(f"  prompts compared: {checked:,}  "
      f"({len(rows)} commands x {len(STRATS)} strategies)")
print(f"  differing outside the legend block: {bad}  "
      f"-> {'FAIL' if bad else 'PASS'}")

# --- the legend really is present/absent where expected -------------------
print()
print("=" * 96)
print("CHECK 4: v3 and v4 prompts DO differ (the change is actually applied)")
print("=" * 96)
cmd = rows[0]["command"]
p3 = v3.build_prompt(cmd, "zero_shot", schema)
p4 = v4.build_prompt(cmd, "zero_shot", schema)
print(f"  identical? {p3 == p4}  -> {'FAIL (no change applied)' if p3 == p4 else 'PASS'}")
print(f"  v3 prompt {len(p3)} chars | v4 prompt {len(p4)} chars "
      f"| delta {len(p4)-len(p3):+d}")

# --- the swap is reversible ----------------------------------------------
print()
print("=" * 96)
print("CHECK 5: calling v4 does not leave prompts.py permanently patched")
print("=" * 96)
before = v3.build_prompt(cmd, "zero_shot", schema)
_ = v4.build_prompt(cmd, "zero_shot", schema)
after = v3.build_prompt(cmd, "zero_shot", schema)
print(f"  v3 output stable across a v4 call: {before == after}  "
      f"-> {'PASS' if before == after else 'FAIL (v3 was mutated)'}")
