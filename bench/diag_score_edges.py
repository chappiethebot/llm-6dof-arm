#!/usr/bin/env python
"""
bench/diag_score_edges.py -- READ-ONLY. Quantify two scoring edge cases found by
inspecting llm/score.py. Re-queries nothing, writes no results, changes no numbers.

WHY THIS EXISTS
---------------
score.py's `strict_match` is NOT exact match, and its normalized/category logic
has two behaviours that may be misattributing failures. Both are measurable from
the stored `pred_json` column of the existing CSVs.

  EDGE 1 -- NULL PLACEHOLDER FIELDS
    Truth for a `stop` row canonicalises to {"action": "stop"}. A model that emits
    {"action": "stop", "target": null} fails strict (correct: extra key) but ALSO
    fails normalized, and _categorize labels it "wrong_or_missing_target".
    The model did not get the target wrong. It emitted a null placeholder for a
    field the verb does not have. If this is common on stop/reject, the failure
    breakdown behind the SAFETY KPI is mislabelled.

  EDGE 2 -- ALIAS ORDER CLOBBER
    _apply_aliases claims it will not clobber a real "action" with an alias
    ("verb" / "command"). It does, when the alias key appears FIRST:
        {"verb": "move_to", "action": "pick_place"} -> {"action": "move_to"}
    Insertion order decides the winner. Affects normalized scoring only.

WHAT COUNTS AS "MATERIAL"
-------------------------
Decide the threshold BEFORE reading the output, so the result cannot be
rationalised afterwards. Suggested: if edge 1 accounts for more than ~2% of any
verb's failures -- or ANY meaningful share of stop/reject failures -- it belongs
in the thesis as a stated limitation, or is fixed by a rescore at a NEW frozen
commit applied uniformly to every file (seen, held-out, base, fine-tuned).

If it is negligible, record the number and move on. A measured "we checked, it
was 0.3%" is a good answer at a viva. "We never checked" is not.

DO NOT let this become a reason to edit score.py mid-experiment. Every existing
number was produced by the current scorer.

USAGE (PowerShell, repo root)
-----------------------------
    python -m bench.diag_score_edges --csv bench/results_grid_v2.csv
    python -m bench.diag_score_edges --csv bench/results_grid_v2.csv --show 5
    python -m bench.diag_score_edges --csv bench/results_grid_v2.csv --csv bench/results_heldout_v2.csv
"""
import argparse, csv, json, sys
from collections import Counter, defaultdict
from pathlib import Path

# deepseek `thinking` traces exceed the 128 KB default. 2**31-1, NOT sys.maxsize
# (which overflows on Windows).
csv.field_size_limit(2**31 - 1)

ALIAS_KEYS = ("verb", "command")


def _truth_verb(truth_json):
    try:
        return json.loads(truth_json).get("action")
    except Exception:
        return None


def _null_placeholder_keys(pred, truth):
    """Keys present in pred with a null/empty value that are ABSENT from truth.

    These are placeholder fields for slots the verb does not have. They break
    strict (correctly, it is an extra key) but they also break normalized, and
    they get categorised as a semantic error, which they are not."""
    if not isinstance(pred, dict) or not isinstance(truth, dict):
        return []
    return [k for k, v in pred.items()
            if k not in truth and (v is None or v == "" or v == [] or v == {})]


def _alias_clobber(pred):
    """True if pred carries BOTH an alias key and a real 'action', with the alias
    FIRST in insertion order -- the case where _apply_aliases silently wins."""
    if not isinstance(pred, dict) or "action" not in pred:
        return False
    keys = list(pred.keys())
    for ak in ALIAS_KEYS:
        if ak in keys and keys.index(ak) < keys.index("action"):
            return True
    return False


def scan(paths, show=0):
    total = 0
    parse_fail = 0
    fails = 0                                    # rows where strict_match is False
    edge1_rows = 0                               # failures carrying a null placeholder
    edge2_rows = 0
    edge1_by_verb = Counter()
    fails_by_verb = Counter()
    edge1_keys = Counter()
    edge1_cats = Counter()
    samples = defaultdict(list)

    for path in paths:
        p = Path(path)
        if not p.exists():
            print(f"  ! not found, skipped: {p}")
            continue
        with p.open(encoding="utf-8", newline="") as f:
            rdr = csv.DictReader(f)
            if "pred_json" not in (rdr.fieldnames or []):
                print(f"  ! {p} has no pred_json column, skipped")
                continue
            for row in rdr:
                total += 1
                verb = _truth_verb(row.get("truth_json", ""))
                strict = str(row.get("strict_match", "")).strip().lower() == "true"
                if strict:
                    continue
                fails += 1
                fails_by_verb[verb] += 1

                raw_pred = row.get("pred_json") or ""
                if not raw_pred:
                    continue                     # pipeline failure, not a scoring edge
                try:
                    pred = json.loads(raw_pred)
                    truth = json.loads(row.get("truth_json") or "{}")
                except Exception:
                    parse_fail += 1
                    continue

                ph = _null_placeholder_keys(pred, truth)
                if ph:
                    edge1_rows += 1
                    edge1_by_verb[verb] += 1
                    for k in ph:
                        edge1_keys[k] += 1
                    edge1_cats[row.get("category", "")] += 1
                    if show and len(samples["edge1"]) < show:
                        samples["edge1"].append((row.get("model"), row.get("strategy"),
                                                 verb, raw_pred, row.get("category")))
                if _alias_clobber(pred):
                    edge2_rows += 1
                    if show and len(samples["edge2"]) < show:
                        samples["edge2"].append((row.get("model"), row.get("strategy"),
                                                 verb, raw_pred, row.get("category")))

    # ---------------------------------------------------------------- report
    def pct(n, d):
        return f"{100.0*n/d:5.2f}%" if d else "  n/a"

    print(f"\nrows scanned            : {total:,}")
    print(f"strict failures         : {fails:,}  ({pct(fails, total)} of all rows)")
    print(f"unparseable pred_json   : {parse_fail:,}")

    print(f"\n--- EDGE 1: null placeholder field for a slot the verb lacks ---")
    print(f"affected rows           : {edge1_rows:,}  "
          f"({pct(edge1_rows, fails)} of strict failures, {pct(edge1_rows, total)} of all rows)")
    if edge1_keys:
        print(f"placeholder keys        : {dict(edge1_keys.most_common())}")
    if edge1_cats:
        print(f"currently categorised as: {dict(edge1_cats.most_common())}")

    if fails_by_verb:
        print(f"\n  per-verb (SAFETY VERBS FIRST -- these carry the reject/stop KPI):")
        print(f"  {'verb':<12} {'failures':>9} {'edge1':>7} {'share':>8}")
        print("  " + "-" * 40)
        order = ["reject", "stop", "fetch", "hold", "inspect", "move_to", "pick_place", None]
        seen_verbs = [v for v in order if v in fails_by_verb]
        seen_verbs += [v for v in fails_by_verb if v not in seen_verbs]
        for v in seen_verbs:
            fv, ev = fails_by_verb[v], edge1_by_verb.get(v, 0)
            flag = "  <-- look" if fv and ev / fv > 0.02 else ""
            print(f"  {str(v):<12} {fv:>9,} {ev:>7,} {pct(ev, fv):>8}{flag}")

    print(f"\n--- EDGE 2: alias key before 'action' (normalized-only clobber) ---")
    print(f"affected rows           : {edge2_rows:,}  ({pct(edge2_rows, total)} of all rows)")

    for tag, title in (("edge1", "EDGE 1 samples"), ("edge2", "EDGE 2 samples")):
        if samples[tag]:
            print(f"\n{title}:")
            for m, s, v, pj, cat in samples[tag]:
                print(f"  {m}/{s}  truth_verb={v}  category={cat}")
                print(f"      pred: {pj}")

    print("\nREAD THIS: this diagnostic changed nothing. score.py is untouched and every")
    print("existing number stands. If edge 1 is material, it is either a stated limitation")
    print("or a rescore at a NEW frozen commit applied uniformly to ALL files.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="append", required=True,
                    help="results CSV to scan; repeat the flag for several files")
    ap.add_argument("--show", type=int, default=3, help="sample rows to print per edge")
    args = ap.parse_args()
    scan(args.csv, show=args.show)


if __name__ == "__main__":
    main()
