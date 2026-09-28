"""thesis2/final/s1_repro_check.py -- the held_out_v3 reproducibility gate. NO MODEL RUNS.

    python thesis2/final/s1_repro_check.py [--split held_out_v3]

Compares the prediction files this project has already written
(thesis2/out/final/pred_<arm>_<split>.csv) against the predictions saved by the
EARLIER run in thesis2/out/. Reads CSVs only: no model is loaded, no arm is
built, nothing under thesis2/out/ is written.

One line per arm:

    arm                        compared/217  identical  differing command_ids

WHY THIS SCRIPT EXISTS
----------------------
The reproducibility check inside s1_run_arm.py printed "0/0 identical" for
flat_gpu, flat_cpu, factorized_fp16, lora_greedy, rules_open,
cascade_local_tauA, pipeline_always_local and cascade_oracle -- and PASSED,
because `same != len(shared)` is false when `shared` is empty.

The cause was the JOIN KEY, not the data. `common.score_row` sets
`command_id=cid`, an int from `enumerate()`, so a freshly computed row carries
an int while `saved_predictions()` reads its keys with `csv.DictReader` and
carries a str. `set(saved) & set(first)` intersected `{'0','1',...}` with
`{0,1,...}` and came back empty. The aggregate accuracies did match the earlier
run to four decimals for all 17 pre-existing arms, so the predictions were
right all along; only the check was broken.

Here the key is `str(command_id)` on both sides, and:

  * comparing ZERO commands is GATE FAIL, never a pass;
  * comparing FEWER than the split's commands is GATE FAIL;
  * any differing command on a deterministic, non-Ollama, non-cloud arm is
    GATE FAIL, with the differing command_ids printed.

TOTALS-ONLY ARMS
----------------
factorized_int4 and rules_firewalled were never saved per command: Phase 3 and
A9 persisted only their aggregate. For those two the TOTAL is compared instead,
against the aggregate the earlier run reported -- 0.6083 for INT4 and 0.4470
for rules_firewalled -- to four decimals. That is a weaker check than a
per-command one and is labelled "totals only" in the output so it is never read
as a per-command reproduction.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C                                                      # noqa: E402
import s1_run_arm as R                                                  # noqa: E402

csv.field_size_limit(2 ** 31 - 1)

# Arms whose earlier run persisted only an aggregate. arm -> (accuracy, source).
# Arms whose earlier run persisted only an aggregate, PER SPLIT (2026-09-28: the
# v3 totals were previously applied to v4c as well). arm -> (accuracy, source).
TOTALS_BY_SPLIT = {
    "held_out_v3": {
        "factorized_int4": (0.6083, "quantization_gate.csv (Phase 3 aggregate)"),
        "rules_firewalled": (0.4470, "baseline_rules_firewall.csv, held_out_v3 row"),
    },
    "held_out_v4c": {
        "rules_firewalled": (0.4470, "baseline_rules_firewall.csv, held_out_v4c row"),
    },
}
NO_TOTAL = {
    ("factorized_int4", "held_out_v4c"): "the earlier run saved no held_out_v4c "
    "aggregate for INT4 (quantization_gate.csv is held_out_v3 only)",
}
TOL = 5e-5          # 4 decimals, as the earlier run reported them


def mine(arm, split):
    """{str(command_id): canonical pred_json} from this project's own file.

    Repeat 0 only, to match saved_predictions, and parsed with csv.DictReader --
    never a line count, because raw outputs contain newlines.
    """
    p = C.pred_path(arm, split)
    rows = C.pred_rows_of(p)
    out, correct = {}, {}
    for r in rows:
        cid = str(r["command_id"])
        if str(r.get("repeat", "0")) != "0":
            continue
        out[cid] = R.canon(json.loads(r["pred_json"] or "{}"))
        correct[cid] = str(r["correct"]).lower() == "true"
    acc = sum(correct.values()) / len(correct) if correct else float("nan")
    return p, out, acc, len(rows), correct


def census(split, n_cmd):
    """Audit EVERY pred_* file on the split with the fixed completeness rule.

    COMPLETE means exactly n_cmd x R parsed CSV rows over exactly n_cmd distinct
    command_ids. The old rule counted PHYSICAL LINES and used `>=`, so
    pred_grpo_base_held_out_v3.csv read 986/217 rows and was declared COMPLETE;
    csv.DictReader gives exactly 217. Returns the list of failures.
    """
    cells = C.v5_census()
    bad = []
    files = sorted(C.OUT.glob("pred_*_%s.csv" % split))
    print()
    print("  COMPLETENESS AUDIT -- exactly n x R parsed CSV rows over n distinct "
          "command_ids")
    print("  %-44s %7s %7s %6s %9s  %s"
          % ("file", "rows", "lines", "uniq", "expected", "verdict"))
    print("  " + "-" * 122)
    for f in files:
        name = f.name[len("pred_"):-len("_%s.csv" % split)]
        arm = R.A.REGISTRY.get(name)
        lines = sum(1 for _ in f.open(encoding="utf-8", newline="")) - 1
        if arm is None:
            bad.append("%s: no arm named %r in the registry" % (f.name, name))
            print("  %-44s %7s %7d %6s %9s  NOT IN THE REGISTRY"
                  % (f.name, "?", lines, "?", "?"))
            continue
        rep = C.v5_repeats_for(arm.kind, cells)
        expected = n_cmd * rep
        ok_, n, u, why = C.pred_census(name, split, expected, n_commands=n_cmd)
        print("  %-44s %7d %7d %6d %9d  %s"
              % (f.name, n, lines, u, expected,
                 "COMPLETE" if ok_ else "*** FAIL: %s ***" % why))
        if not ok_:
            bad.append("%s: %s (R=%d for a %s arm)" % (f.name, why, rep, arm.kind))
    print("  " + "-" * 122)
    infl = [(f.name, sum(1 for _ in f.open(encoding="utf-8", newline="")) - 1,
             len(C.pred_rows_of(f))) for f in files]
    infl = [x for x in infl if x[1] != x[2]]
    print("  %d file(s) audited; %d FAIL" % (len(files), len(bad)))
    print("  %d file(s) have more physical LINES than CSV ROWS -- exactly the files "
          "the old line-count rule mis-read:" % len(infl))
    for nm, li, ro in infl:
        print("     %-44s %d lines vs %d rows" % (nm, li, ro))
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", default="held_out_v3", choices=list(C.SPLITS))
    ap.add_argument("--arm", default=None, help="one arm only (default: all)")
    ap.add_argument("--no-census", action="store_true",
                    help="skip the completeness audit of every pred_* file")
    a = ap.parse_args()
    C.start_log("s1_repro_check_%s" % a.split)

    n_cmd = len(C.load_split(a.split))
    TOTALS_ONLY = TOTALS_BY_SPLIT[a.split]
    # every registry arm gets a line; one without saved predictions says why
    names = [a.arm] if a.arm else sorted(R.A.REGISTRY)

    print("  split %s, %d commands. Reading CSVs only: no model is loaded."
          % (a.split, n_cmd))
    print("  ours  : thesis2/out/final/pred_<arm>_%s.csv" % a.split)
    print("  theirs: thesis2/out/ (the earlier run)")
    print()
    print("=" * 124)
    print("%-28s %12s %10s  %s"
          % ("arm", "compared/%d" % n_cmd, "identical", "differing command_ids"))
    print("=" * 124)

    fails, skipped, done = [], [], 0
    records = []          # one per named arm -> repro_check_<split>.csv
    if not a.no_census and not a.arm:
        fails += census(a.split, n_cmd)
        print()
    for name in names:
        arm = R.A.REGISTRY.get(name)
        p, ours, acc, n_rows, ours_correct = mine(name, a.split)
        if not p.exists():
            skipped.append((name, "no prediction file %s -- run s1_run_arm.py first"
                            % p.name))
            print("%-28s %12s %10s  %s" % (name, "-", "-", "NOT RUN YET (%s)" % p.name))
            records.append(dict(arm=name, split=a.split, compared="", n_commands=n_cmd,
                                identical="", status="not run",
                                result="no prediction file %s" % p.name))
            continue

        # ---- totals-only arms
        if (name, a.split) in NO_TOTAL:
            why = NO_TOTAL[(name, a.split)]
            skipped.append((name, why))
            records.append(dict(arm=name, split=a.split, compared="", n_commands=n_cmd,
                                identical="", status="not comparable", result=why))
            print("%-28s %12s %10s  NOT COMPARABLE: %s" % (name, "-", "-", why[:58]))
            continue
        if name in TOTALS_ONLY:
            want, src = TOTALS_ONLY[name]
            delta = abs(acc - want)
            okk = delta <= TOL
            print("%-28s %12s %10s  totals only: ours %.4f vs saved %.4f (%s)%s"
                  % (name, "totals", "yes" if okk else "NO", acc, want, src,
                     "" if okk else "  *** FAIL, delta %.4f ***" % delta))
            if not okk:
                fails.append("%s: aggregate %.4f != the earlier run's %.4f (%s); "
                             "no per-command file exists for this arm"
                             % (name, acc, want, src))
            records.append(dict(arm=name, split=a.split, compared="totals",
                                n_commands=n_cmd, identical="",
                                status="PASS" if okk else "FAIL",
                                result="totals only: ours %.4f vs saved %.4f (%s)"
                                % (acc, want, src)))
            done += 1
            continue

        saved = R.saved_predictions(name, a.split)
        if not saved:
            why = R.NO_SAVED.get(name, "no saved per-command file for this arm x split")
            skipped.append((name, why))
            print("%-28s %12s %10s  NOT COMPARABLE: %s" % (name, "-", "-", why[:58]))
            records.append(dict(arm=name, split=a.split, compared="", n_commands=n_cmd,
                                identical="", status="not comparable", result=why))
            continue
        # the SAME rule as s1_run_arm.py gate 5 -- one function, two callers
        truths = {str(i): C.truth_of(rw) for i, rw in enumerate(C.load_split(a.split))}
        okc, line, n_sh, same = R.repro_compare(arm, a.split, ours, saved,
                                                ours_correct, truths)
        tag = line.split("identical", 1)[1].strip()
        if not okc:
            fails.append("%s: %s" % (name, line))
        print("%-28s %12s %10d  %s"
              % (name, "%d/%d" % (n_sh, n_cmd), same, tag or "all identical"))
        records.append(dict(arm=name, split=a.split, compared=n_sh, n_commands=n_cmd,
                            identical=same, status="PASS" if okc else "FAIL",
                            result=tag or "all identical"))
        done += 1

    print("=" * 124)
    if skipped:
        print()
        print("  NOT COMPARABLE (%d arms) -- each with the reason from the earlier run:"
              % len(skipped))
        for name, why in skipped:
            print("     %-28s %s" % (name, why))

    if not a.arm:
        rp = C.OUT / ("repro_check_%s.csv" % a.split)
        with rp.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["arm", "split", "compared", "n_commands",
                                               "identical", "status", "result"])
            w.writeheader()
            w.writerows(records)
        print("  wrote %s (%d arms) -- the report's Section 3 is built from it"
              % (rp.relative_to(C.REPO), len(records)))
    C.gate(not fails, ([
        "NO MODEL RUNS: prediction CSVs and the earlier run's CSVs were read; the "
        "only file written is this check's own repro_check_<split>.csv",
        "%d arm(s) compared on %s; %d not comparable" % (done, a.split, len(skipped)),
        "join key is str(command_id) on both sides; comparing ZERO commands is a FAIL",
        ("completeness: exactly n x R PARSED CSV rows over n distinct command_ids "
         "(never a line count)") if not a.no_census and not a.arm
        else "completeness audit: skipped",
        "totals-only arms: %s" % ", ".join("%s %.4f" % (k, v[0])
                                           for k, v in sorted(TOTALS_ONLY.items())),
    ] + fails))


if __name__ == "__main__":
    main()
