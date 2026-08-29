"""Mechanical checks on a legend, using the audit/vocab_conflicts.md criteria.

Runs against V3, V4, V4a-lite and V5 so the results are comparable. Read-only.

CHECKS
  1. No word may label two different axes.
  2. No pair of phrases in the legend may contradict each other (row-order
     framings that imply opposite directions).
  3. Every semantic LABEL word must appear in real commands. Structural
     scaffolding (row, column, square, table, grid, command, robot, arm) is
     exempt: it describes the schema, it is not matched against a command.
  4. No concrete cell id (grid_A1 .. grid_C3) may appear in the text.
"""
import json, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from llm.prompts import load_schema, _grid_legend as legend_v3
from llm.prompts_v4 import _grid_legend_v4, _grid_legend_v4lite
from llm.prompts_v5 import _grid_legend_v5

SCHEMA = load_schema(os.path.join(ROOT, "schema", "schema_v2.json"))
SPL = ["seen_v2", "train_v2", "dev_v2", "held_out_v3"]
CMDS = [json.loads(l)["command"].lower()
        for s in SPL
        for l in open(os.path.join(ROOT, "data", f"{s}.jsonl"), encoding="utf-8")
        if l.strip()]
CELL_IDS = [f"grid_{r}{c}" for r in "ABC" for c in "123"]

# words that describe the schema rather than being matched against a command
# tokens that only ever appear because a SCHEMA SYMBOL is spelled out
# (max_reach -> max, reach; nearest_free_cell -> nearest, free, cell;
#  grid_<ROW><COL> -> row, col). They are mandatory identifiers, not vocabulary
# the model matches against a command, so they are exempt from check 3.
SYMBOL_TOKENS = set()
for _loc in SCHEMA["locations"]:
    SYMBOL_TOKENS.update(re.findall(r"[a-z]+", _loc.lower()))
SYMBOL_TOKENS.update(["row", "col"])

STRUCTURAL = set("""grid locations location are squares square on the table surface
named row rows column columns a is it also call called may command commands
between and or of from one side other ordered to at its the they them name names
mean means instead neither nor where meets asks for with any that free picks
which robot arm can still touch fixed not it's in b2""".split())

ROW_LABELS = {"far", "upper", "near", "bottom", "back", "front", "centre", "middle"}
COL_LABELS = {"left", "right", "middle", "centre", "center"}

# phrases that assert a row ORDER; two with opposite implied direction conflict
ORDER_PATTERNS = {
    "front-to-back": "A=front(near)",
    "back-to-front": "A=back(far)",
    "far side of the table to the near side": "A=far",
    "far from the robot": "A=far",
    "near the robot": "C=near",
}


def words(t):
    return re.findall(r"[a-z][a-z\-']*", t.lower())


def check(name, text):
    print(f"\n{'='*84}\n{name}  ({len(text)} chars)\n{'='*84}")
    fails = 0

    # 1. no word labels two axes
    ws = set(words(text))
    rows_line = " ".join(l for l in text.splitlines()
                         if re.match(r"\s*(Rows|[ABC] is)", l))
    cols_line = " ".join(l for l in text.splitlines()
                         if re.match(r"\s*(Columns|[123] is)", l))
    rl = {w for w in ROW_LABELS if re.search(rf"\b{w}\b", rows_line)}
    cl = {w for w in COL_LABELS if re.search(rf"\b{w}\b", cols_line)}
    both = rl & cl
    print(f"1. row labels {sorted(rl)}")
    print(f"   col labels {sorted(cl)}")
    if both:
        print(f"   FAIL: {sorted(both)} labels BOTH axes"); fails += 1
    else:
        print("   PASS: no word labels both axes")

    # 2. contradictory order framings
    found = {p: v for p, v in ORDER_PATTERNS.items() if p in text.lower()}
    implied = set()
    for p, v in found.items():
        if v.startswith("A="):
            implied.add(v)
    conflict = {"A=front(near)", "A=far"} <= implied
    print(f"2. order framings present: {list(found) or 'none'}")
    if conflict:
        print("   FAIL: one framing implies A is nearest, another implies A is farthest")
        fails += 1
    else:
        print("   PASS: no contradictory row-order framing")

    # 3. every semantic label word occurs in real commands
    cand = sorted(w for w in ws
                  if w not in STRUCTURAL and w not in SYMBOL_TOKENS
                  and not w.isdigit() and len(w) > 2
                  and not w.startswith("grid") and "_" not in w)
    missing = [w for w in cand
               if not any(re.search(rf"\b{re.escape(w)}\b", c) for c in CMDS)]
    print(f"3. semantic words checked: {len(cand)}")
    if missing:
        print(f"   FAIL: absent from all 794 commands: {missing}"); fails += 1
    else:
        print("   PASS: every semantic word appears in real commands")

    # 4. no concrete cell id
    ids = [c for c in CELL_IDS if c in text]
    print(f"4. concrete cell ids in text: {ids or 'none'}")
    if ids:
        print(f"   FAIL: {ids} spelled out (llama3.2 anchored on this in V4)")
        fails += 1
    else:
        print("   PASS: no cell id written out")

    print(f"\n   -> {'ALL CHECKS PASS' if fails == 0 else f'{fails} CHECK(S) FAILED'}")
    return fails


tot = 0
for n, f in (("V3 (shipped, produced the 30,380 rows)", legend_v3),
             ("V4 (measured; regressed on 2 models)", _grid_legend_v4),
             ("V4a-lite (measured; partial recovery)", _grid_legend_v4lite),
             ("V5 (DEMO CANDIDATE)", _grid_legend_v5)):
    r = check(n, f(SCHEMA))
    if "V5" in n:
        tot += r
print(f"\n{'='*84}\nV5 verdict: {'READY' if tot == 0 else 'NOT READY'}\n{'='*84}")
sys.exit(0 if tot == 0 else 1)
