#!/usr/bin/env python
"""
bench/rescore.py -- recompute score columns on an existing grid CSV using the
CURRENT llm.score.score(), without re-running any model. Reads pred_json /
truth_json from each row, recomputes strict/normalized/category, and writes a
new CSV. Also DEDUPES on (dataset, model, strategy, command_id, run_index)
keeping the LAST occurrence, so resumed-cell orphans are dropped.

Why this exists: scoring is a pure function of (pred, truth), both of which we
stored on every row. So a scorer fix is a re-SCORE (seconds), never a re-RUN.

WHAT RESCORING CHANGES -- READ THIS
-----------------------------------
`strict_match` is gated on stage == "validated": a row the schema validator
rejected can never be strict-correct, exactly as in the live run.

`normalized_match` and `category` are NOT gated. score() is called on any row
whose pred_json parses, including rows that failed validation. That is
deliberate and it is the whole point of the dual axis: normalized asks "was the
model semantically right, ignoring how it was shaped", and a nested action
object is the canonical case -- structurally rejected, semantically correct.
deepseek-r1's +12 to +19 point strict->normalized gap is precisely this.

CONSEQUENCE, AND IT IS EASY TO MISS: in the output file, `category` no longer
carries "invalid:schema_failed". Any schema-rejected row whose JSON parses gets
the semantic verdict instead. The raw file's 6,991 schema_failed rows do not
disappear -- they are still there -- but you must key on `stage` to find them.
bench/summary.py section 7 keys on `stage` for this reason. Never count
invalid rows off `category` in a rescored file.

`stage`, `ok`, `latency_s`, `raw`, `thinking` and every other measured column are
copied through untouched. Rescoring never alters a measurement, only a verdict.

FROZEN COMMIT: every file compared to another must be rescored at the SAME
commit of llm/score.py. Record the hash. As of this writing that is 3000198
("corrected normalized scorer (representational unwrap)"), and both V2 files
were produced under it.

MIXED SCHEMAS: 25-column V2 and 28-column V3 rows both read fine; the output
header follows the input file.

FIELD SIZE: the V2 schema added a `thinking` column holding deepseek-r1's full
reasoning trace. Python's csv module caps a single field at 128 KB by default,
and at least one trace exceeds that -- a runaway reasoning loop, which is itself
a measurement, not corruption. The limit is raised below. sys.maxsize is NOT
used: on Windows it overflows the underlying C long and raises OverflowError.

Usage:
    python -m bench.rescore --in bench/results_grid_v2.csv --out bench/results_grid_v2_scored.csv
    python -m bench.rescore --in bench/results_grid_v2.csv --out ... --dry-run
"""
import argparse, csv, json, subprocess
from collections import Counter
from pathlib import Path

from llm.score import score

# 2**31 - 1: comfortably larger than any trace, and safe for the C long on Windows.
csv.field_size_limit(2**31 - 1)


def parse(s):
    if not s:
        return None
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return None


def scorer_commit():
    """Best-effort hash of llm/score.py's last commit, printed for the record."""
    try:
        out = subprocess.run(["git", "log", "-1", "--format=%h %s", "--", "llm/score.py"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "(no git info)"
    except Exception:
        return "(git unavailable)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="bench/results_grid.csv")
    ap.add_argument("--out", default="bench/results_grid_scored.csv")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change; write nothing")
    args = ap.parse_args()

    if not Path(args.inp).exists():
        raise SystemExit(f"not found: {args.inp}")

    print(f"scorer commit: {scorer_commit()}")
    print("  record this hash. every file you compare must be scored under it.\n")

    with open(args.inp, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"no rows in {args.inp}")

    # dedupe: keep LAST occurrence per identity key (resumed-cell orphan removal).
    # `dataset` is in the key because command_id is an index into THAT dataset's
    # test rows -- seen row 7 and held-out row 7 are different commands.
    seen = {}
    for r in rows:
        seen[(r.get("dataset", ""), r["model"], r["strategy"],
              r["command_id"], r["run_index"])] = r
    deduped = list(seen.values())
    print(f"read {len(rows)} rows, {len(deduped)} after dedupe "
          f"({len(rows) - len(deduped)} orphans dropped)")

    fieldnames = list(rows[0].keys())
    changed = 0
    moved = Counter()          # (old_category -> new_category) for the report
    stage_counts = Counter()
    rescued = 0                # invalid rows that normalized-match

    out_rows = []
    for r in deduped:
        pred = parse(r.get("pred_json"))
        truth = parse(r.get("truth_json"))
        stage = r.get("stage", "")
        stage_counts[stage] += 1

        if pred is not None and truth is not None:
            sc = score(pred, truth)
            # strict requires a genuinely validated object AND a match; a row that
            # never validated cannot be strict-correct. normalized/category may
            # still rescue a mis-formatted-but-correct object -- see the module
            # docstring: this is the dual axis, not a leak.
            new_strict = bool(sc.strict_match) and (stage == "validated")
            new_norm = bool(sc.normalized_match)
            new_cat = sc.category
        else:
            new_strict, new_norm, new_cat = False, False, f"invalid:{stage}"

        if stage != "validated" and new_norm:
            rescued += 1
        if new_cat != r.get("category"):
            moved[(r.get("category"), new_cat)] += 1
        if (str(new_strict) != r.get("strict_match")
                or str(new_norm) != r.get("normalized_match")
                or new_cat != r.get("category")):
            changed += 1

        r["strict_match"] = new_strict
        r["normalized_match"] = new_norm
        r["category"] = new_cat
        out_rows.append(r)

    print(f"\nstages in this file: {dict(stage_counts.most_common())}")
    print(f"{changed} row(s) had at least one score column change")
    if rescued:
        print(f"{rescued} row(s) failed validation but are NORMALIZED-correct "
              f"-- the strict->normalized gap lives here")
    invalid_relabelled = sum(v for (o, n), v in moved.items()
                             if str(o).startswith("invalid:") and not str(n).startswith("invalid:"))
    if invalid_relabelled:
        print(f"\n  NOTE: {invalid_relabelled} row(s) had an invalid:* category replaced by a")
        print(f"  semantic one. Those rows are still present and `stage` still says what")
        print(f"  happened -- but do NOT count pipeline failures off `category` in the")
        print(f"  output file. bench/summary.py section 7 keys on `stage` for this reason.")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nwrote {args.out}  ({len(out_rows)} rows, {len(fieldnames)} columns)")


if __name__ == "__main__":
    main()
