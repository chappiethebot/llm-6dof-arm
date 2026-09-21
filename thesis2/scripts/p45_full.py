"""thesis2/scripts/p45_full.py -- P4.5 end-to-end: action match, joint correctness, memory.

The sweep in p45_sweep.py measures the ROW FACTOR. This measures the whole
system: reconstruct the full 14-class symbol from the cascade's factors, emit
the action object through the schema mask, and score it with llm/score.py.

CASCADE ARCHITECTURE (as specified):
  local, always   verb, target, loc_type, col, container_which, runtime_which
  escalated       ROW only, and only when its confidence < tau
  reconstruction  grid_<row><col> from the escalated row and the LOCAL col

LATENCY AND ENERGY IN THIS FILE ARE MODELLED CONSTANTS, NOT PER-COMMAND
MEASUREMENTS (Part B S-3/S-4): LOCAL_MS and ESC_MS below are single p50s from
p45_collect.py added per command, so p50/p95/p99 collapse to the same value.
They are not quoted in FINAL_REPORT.md.
"""
import csv
import json
import os
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "thesis2" / "scripts"))

from llm.score import score                                        # noqa: E402
from p2_train_factorized import (build_label_space, reconstruct, emit,  # noqa: E402
                                 ROWS, COLS, CONTAINERS, RUNTIMES)

OUT = REPO / "thesis2" / "out"
GRID = [round(0.05 * i, 2) for i in range(21)]
TARGETS = ["none", "qwen3b", "qwen3blora", "cloud", "oracle"]
SIGNALS = {"factorized_row_conf": "fact_row_conf",
           "flat_location_conf": "flat_location_conf"}

# single p50 values from p45_collect.py (GPU, warm) -- a MODEL, see docstring
LOCAL_MS = 3.2
ESC_MS = {"none": 0.0, "qwen3b": 35.5, "qwen3blora": 35.4, "cloud": 657.0,
          "oracle": 0.0}
# model sizes on disk / resident, for P4.4
MEM_MB = {"none": 0.0, "qwen3b": 6335.0, "qwen3blora": 6593.0,
          "cloud": 0.0, "oracle": 0.0}


def rd(ds):
    with open(OUT / ("p45_predictions_%s.csv" % ds), encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def esc_row(r, target, mapping="letters"):
    if target == "none":
        return ""
    if target == "oracle":
        return r["truth_row"]
    if target == "cloud":
        return r["cloud_row"]
    return r["salsa_%s_%s_row" % (target, mapping)]


def run_point(recs, space, col, target, tau, mapping="letters"):
    """One (signal, target, tau) configuration over the whole dataset."""
    n_esc = 0
    lat = []
    hits = dict(action=0, joint=0, verb=0, target_=0, location=0, row=0)
    den = dict(action=0, joint=0, verb=0, target_=0, location=0, row=0)
    for r in recs:
        truth = json.loads(r["truth_json"])
        truth.setdefault("modifiers", [])
        ms = LOCAL_MS

        lt = int(r["fact_loc_type_idx"])
        row_i = int(r["fact_row_idx"])
        col_i = int(r["fact_col_idx"])
        cw_i = int(r["fact_container_which_idx"])
        rw_i = int(r["fact_runtime_which_idx"])

        # escalate ONLY the row factor, and only on grid-typed predictions:
        # asking a row question about a container prediction is meaningless
        escalated = False
        if target != "none" and lt == 0 and float(r[col]) < tau:
            e = esc_row(r, target, mapping)
            if e in ROWS:
                row_i = ROWS.index(e)
                escalated = True
                n_esc += 1
                ms += ESC_MS[target]
        lat.append(ms)

        sym = reconstruct(lt, row_i, col_i, cw_i, rw_i)
        obj = emit(int(r["fact_verb_idx"]), int(r["fact_target_idx"]), sym, space)

        den["action"] += 1
        hits["action"] += int(score(obj, truth).strict_match)
        den["verb"] += 1
        hits["verb"] += int(obj.get("action") == truth.get("action"))
        if truth.get("target"):
            den["target_"] += 1
            hits["target_"] += int(obj.get("target") == truth.get("target"))
        if truth.get("location"):
            den["location"] += 1
            hits["location"] += int(sym == truth["location"])
        if truth.get("target") and truth.get("location"):
            den["joint"] += 1
            hits["joint"] += int(all(obj.get(k) == truth.get(k)
                                     for k in ("action", "target", "location")))
        if r["truth_row"]:
            den["row"] += 1
            hits["row"] += int(ROWS[row_i] == r["truth_row"])

    lat.sort()
    q = lambda p: lat[min(len(lat) - 1, int(round(p * (len(lat) - 1))))]
    return dict(escalation_rate=round(n_esc / len(recs), 4), escalated=n_esc,
                action_match=round(hits["action"] / den["action"], 4),
                joint_correct=round(hits["joint"] / den["joint"], 4),
                verb_acc=round(hits["verb"] / den["verb"], 4),
                target_acc=round(hits["target_"] / den["target_"], 4),
                location_acc=round(hits["location"] / den["location"], 4),
                row_acc=round(hits["row"] / den["row"], 4),
                n_action=den["action"], n_joint=den["joint"],
                n_location=den["location"], n_row=den["row"],
                latency_p50_ms=round(q(0.50), 2), latency_p95_ms=round(q(0.95), 2),
                latency_p99_ms=round(q(0.99), 2),
                mean_latency_ms=round(sum(lat) / len(lat), 2))


def main():
    space = build_label_space()
    allrows = []
    for ds in ("held_out_v3", "held_out_v4c"):
        recs = rd(ds)
        print("=" * 78)
        print("%s  n=%d commands" % (ds, len(recs)))
        print("=" * 78)
        base = run_point(recs, space, "fact_row_conf", "none", 0.0)
        print("  no escalation: action %.4f | joint %.4f | row %.4f | p50 %.1f ms"
              % (base["action_match"], base["joint_correct"], base["row_acc"],
                 base["latency_p50_ms"]))
        for sig, cl in SIGNALS.items():
            for target in TARGETS:
                if target == "none":
                    continue
                for tau in GRID:
                    p = run_point(recs, space, cl, target, tau)
                    p.update(dataset=ds, signal=sig, target=target, tau=tau,
                             extra_resident_mb=MEM_MB[target])
                    allrows.append(p)

        print()
        print("  ALWAYS-ESCALATE (tau=1.0) and the pre-registered tau_A=0.50")
        print("  %-12s %-6s %8s %8s %8s %8s %9s" %
              ("target", "tau", "esc", "action", "joint", "row", "p50 ms"))
        for target in TARGETS[1:]:
            for tau in (1.0, 0.50):
                m = [r for r in allrows if r["dataset"] == ds
                     and r["signal"] == "factorized_row_conf"
                     and r["target"] == target and abs(r["tau"] - tau) < 1e-9]
                if m:
                    r = m[0]
                    print("  %-12s %-6.2f %8.2f %8.4f %8.4f %8.4f %9.1f" %
                          (target, tau, r["escalation_rate"], r["action_match"],
                           r["joint_correct"], r["row_acc"], r["latency_p50_ms"]))

        print()
        print("  MATCHED BUDGET, action match, escalate row -> qwen3blora (SALSA)")
        print("  %-8s %-26s %-26s" % ("budget", "factorized_row_conf", "flat_location_conf"))
        for b in (0.1, 0.2, 0.3, 0.4, 0.5):
            line = "  %-8.0f%%" % (100 * b)
            for sig in SIGNALS:
                c = [r for r in allrows if r["dataset"] == ds and r["signal"] == sig
                     and r["target"] == "qwen3blora" and r["escalation_rate"] <= b + 1e-9]
                bst = max(c, key=lambda r: r["escalation_rate"]) if c else None
                line += " %-26s" % ("act %.4f @ %.2f" % (bst["action_match"],
                                                         bst["escalation_rate"])
                                    if bst else "-")
            print(line)
        print()

    p = OUT / "p45_cascade_full.csv"
    with open(p, "w", newline="", encoding="utf-8") as fh:
        cols = ["dataset", "signal", "target", "tau", "escalation_rate", "escalated",
                "action_match", "joint_correct", "verb_acc", "target_acc",
                "location_acc", "row_acc", "n_action", "n_joint", "n_location",
                "n_row", "latency_p50_ms", "latency_p95_ms", "latency_p99_ms",
                "mean_latency_ms", "extra_resident_mb"]
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(allrows)
    print("wrote %s (%d rows)" % (p.relative_to(REPO), len(allrows)))

    # ------------------------------------------------------- P4.4 memory
    mem = [
        dict(arm="per-factor cascade, local only (INT4 weights)",
             resident_mb=69.20,
             note="p3_verify_int4.json whole-model INT4 incl. fp32 scales (one size definition, Part B S-10)"),
        dict(arm="per-factor cascade, local only (FP16 weights)",
             resident_mb=132.76, note="p3_verify_int4.json whole-model FP16 (66,381,336 params x 2 B)"),
        dict(arm="per-factor cascade + qwen2.5-3b-lora escalation (co-resident)",
             resident_mb=132.76 + 6593.0,
             note="DistilBERT FP16 132.76 MB + Qwen3B FP16 peak VRAM measured 6593 MB"),
        dict(arm="two-model cascade: DistilBERT + Qwen2.5-3B (CascadeCNN-style)",
             resident_mb=132.76 + 6593.0,
             note="identical footprint -- the 3B model dominates either way"),
        dict(arm="per-factor cascade + gpt-4o-mini escalation",
             resident_mb=132.76,
             note="no second model resident; requires network. Cloud energy not "
                  "measurable from this machine"),
        dict(arm="flat classifier alone (the V5 rung-4 cell)",
             resident_mb=132.76, note="no escalation path"),
    ]
    p = OUT / "memory_footprint.csv"
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["arm", "resident_mb", "note"])
        w.writeheader()
        w.writerows(mem)
    print("wrote %s" % p.relative_to(REPO))
    print()
    print("P4.4 MEMORY -- the edge argument, concretely")
    for m in mem:
        print("  %-58s %8.1f MB" % (m["arm"][:58], m["resident_mb"]))
    print()
    print("  The honest reading: escalating to a LOCAL 3B model costs 6.6 GB")
    print("  resident, which is 50x the DistilBERT and defeats the edge claim.")
    print("  Escalating to the CLOUD keeps the footprint at 130 MB but adds a")
    print("  network dependency. INT4 local-only is 66 MB. There is no")
    print("  configuration that is both self-contained and strong on rows.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
