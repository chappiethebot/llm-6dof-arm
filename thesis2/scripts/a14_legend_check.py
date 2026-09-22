"""thesis2/scripts/a14_legend_check.py -- A14.2: what row vocabulary does each
legend actually define, and what did the splits actually contain?

RECOVERED 2026-09-22 from the session scratchpad where it was first run
(`g1_legends.py` plus the split word-counts from `f1_v4c_stats.py`); committed
here so the check has a script and an output file in the repository. It writes
thesis2/out/a14_legend_vocabulary.csv.

It prints, verbatim:
  1. the full V5 zero-shot prompt as sent to qwen2.5-3b-lora, and its legend
     fragment alone (llm/prompts_v5.py::_grid_legend_v5, rebound by
     bench/run_cell.py::apply_legend('v5'));
  2. the SALSA row legend and the full letters-mapping prompt as scored
     (thesis2/scripts/salsa_row.py).

Then it reports, for ten row-bearing words, whether each legend DEFINES the
word and how often the word occurs in train_v2, dev_v2, held_out_v3 and
held_out_v4c. Nothing is trained and no model is loaded.

Run:  python thesis2/scripts/a14_legend_check.py
"""
import csv
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "thesis2" / "scripts"))
OUT = REPO / "thesis2" / "out"

WORDS = ["back", "front", "middle", "center", "upper", "bottom", "top", "lower", "far", "near"]
SPLITS = ["train_v2", "dev_v2", "held_out_v3", "held_out_v4c"]
PROBE = "put the bottle in the upper left corner of the table"

from bench.run_cell import apply_legend                                # noqa: E402
from llm import prompts as P                                           # noqa: E402
from llm import prompts_v5 as P5                                       # noqa: E402
import salsa_row as SR                                                 # noqa: E402

note = apply_legend("v5")
schema = P.load_schema(str(REPO / "schema" / "schema_v2.json"))
v5_prompt = P.build_prompt(PROBE, "zero_shot", schema)
salsa_prompt = SR.build_prompt(PROBE, "letters")

print("=" * 100)
print("1. V5 LEGEND AS SENT TO qwen2.5-3b-lora, strategy=zero_shot   (%s)" % note)
print("   source: llm/prompts_v5.py::_grid_legend_v5, rebound by apply_legend('v5')")
print("=" * 100)
print("--- BEGIN full zero_shot prompt (command: %r) ---" % PROBE)
print(v5_prompt)
print("--- END full zero_shot prompt ---")
print()
print("   the legend fragment alone:")
print("-" * 100)
print(P5._grid_legend_v5(schema))
print("-" * 100)

print()
print("=" * 100)
print("2. SALSA ROW LEGEND   (thesis2/scripts/salsa_row.py::ROW_LEGEND + build_prompt)")
print("=" * 100)
print("--- BEGIN ROW_LEGEND ---")
print(SR.ROW_LEGEND, end="")
print("--- END ROW_LEGEND ---")
print()
print("--- BEGIN full letters-mapping prompt as scored ---")
print(salsa_prompt)
print("--- END ---")

print()
print("=" * 100)
print("3. WHICH ROW WORDS DOES EACH LEGEND DEFINE?")
print("=" * 100)
legends = {"v5_zero_shot_prompt": v5_prompt, "salsa_row_prompt": salsa_prompt}
defined = {}
for nm, text in legends.items():
    low = text.lower()
    print("  %s" % nm)
    for w in WORDS:
        hits = [m.start() for m in re.finditer(r"\b%s\b" % w, low)]
        defined[(nm, w)] = len(hits)
        ctx = ""
        if hits:
            i = hits[0]
            ctx = "  first context: ...%s..." % " ".join(text[max(0, i - 55):i + 55].split())
        print("     %-7s %d%s%s" % (w, len(hits), "   <-- NOT DEFINED" if not hits else "", ctx))
    print()
print("  NOTE -- collision: the SALSA legend's only occurrence of 'front' is in")
print("  \"A table in front of a robot is divided into three rows.\", which describes")
print("  where the table sits, NOT a row name. held_out_v4c uses 'front' as the")
print("  name of row C, so the one prior use of the token in that prompt points the")
print("  wrong way.")

print()
print("=" * 100)
print("4. ROW-WORD OCCURRENCES BY SPLIT  (commands containing the word / total occurrences)")
print("=" * 100)
rows = {nm: [json.loads(l) for l in open(REPO / "data" / ("%s.jsonl" % nm), encoding="utf-8")
             if l.strip()] for nm in SPLITS}
print("  %-13s %6s %s" % ("split", "n", " ".join("%11s" % w for w in WORDS)))
counts = {}
for nm in SPLITS:
    cells = []
    for w in WORDS:
        c = sum(1 for r in rows[nm] if re.search(r"\b%s\b" % w, r["command"].lower()))
        o = sum(len(re.findall(r"\b%s\b" % w, r["command"].lower())) for r in rows[nm])
        counts[(nm, w)] = (c, o)
        cells.append("%4d/%-5d" % (c, o))
    print("  %-13s %6d %s" % (nm, len(rows[nm]), " ".join("%11s" % x for x in cells)))
zero = [w for w in WORDS if sum(counts[(s, w)][1] for s in ("train_v2", "dev_v2")) == 0]
print()
print("  ZERO occurrences across train_v2 + dev_v2 combined: %s" % ", ".join(zero))

# grid-only counts, the ones that matter for the row factor
print()
print("  restricted to grid-cell commands only:")
grid_counts = {}
for nm in SPLITS:
    g = [r for r in rows[nm] if (r["expected"].get("location") or "").startswith("grid_")]
    cells = []
    for w in WORDS:
        c = sum(1 for r in g if re.search(r"\b%s\b" % w, r["command"].lower()))
        grid_counts[(nm, w)] = c
        cells.append("%4d" % c)
    print("  %-13s %6d %s" % (nm, len(g), " ".join("%11s" % x for x in cells)))

out = []
for w in WORDS:
    rec = dict(word=w,
               defined_in_v5_prompt=defined[("v5_zero_shot_prompt", w)],
               defined_in_salsa_prompt=defined[("salsa_row_prompt", w)])
    for nm in SPLITS:
        c, o = counts[(nm, w)]
        rec["%s_commands" % nm] = c
        rec["%s_occurrences" % nm] = o
        rec["%s_grid_commands" % nm] = grid_counts[(nm, w)]
    out.append(rec)
p = OUT / "a14_legend_vocabulary.csv"
with open(p, "w", newline="", encoding="utf-8") as fh:
    w_ = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
    w_.writeheader()
    w_.writerows(out)
print()
print("wrote %s (%d rows)" % (p.relative_to(REPO), len(out)))
print("=" * 100)
