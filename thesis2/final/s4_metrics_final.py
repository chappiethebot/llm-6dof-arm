"""thesis2/final/s4_metrics_final.py -- STAGE 4: the one final table.

    python thesis2/final/s4_metrics_final.py

thesis2/out/final/metrics_final.csv: ONE ROW PER (arm, split) carrying every
Stage 2 metric, every Stage 3 efficiency figure, the significance verdicts that
mention the arm, a status (valid / withdrawn / not measurable / not run) and the
notes that must travel with the number.

This is the only file the thesis tables are built from for the final run, the
way metrics_all_v2.csv is for the earlier one. Nothing is blank: a cell that
cannot exist carries the reason.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C                                                      # noqa: E402
import arms as A                                                        # noqa: E402

WITHDRAWN_NOTE = ("withdrawn upstream (FINAL_REPORT B-3/B-4): A1.7 field-level "
                  "FLMD and every SALSA digits mapping are not registered")


def main():
    C.start_log("s4_metrics_final")
    s2 = {}
    p2 = C.OUT / "metrics_stage2.csv"
    if not p2.exists():
        C.gate(False, ["metrics_stage2.csv not found -- run s2_metrics.py first"])
    for r in csv.DictReader(p2.open(encoding="utf-8", newline="")):
        s2[(r["arm"], r["split"])] = r

    eff = {}
    pe = C.OUT / "unified_efficiency.csv"
    if pe.exists():
        for r in csv.DictReader(pe.open(encoding="utf-8", newline="")):
            eff[(r["arm"], r["split"])] = r

    sig = defaultdict(list)
    ps = C.OUT / "final_significance.csv"
    if ps.exists():
        for r in csv.DictReader(ps.open(encoding="utf-8", newline="")):
            if r["family"] != "primary" or r["verdict"] in ("", "NOT RUN"):
                continue
            for arm in (r["arm_a"], r["arm_b"]):
                sig[(arm, r["split"])].append(
                    "%s %s %s pts p_holm %s -> %s" % (r["id"], r["metric"],
                                                      r["diff_pts"], r["p_holm"], r["verdict"]))

    rows = []
    for (arm_name, split), m in sorted(s2.items()):
        arm = A.REGISTRY.get(arm_name)
        e = eff.get((arm_name, split), {})
        status = m.get("status") or "not run"
        reason = m.get("reason", "")
        if arm and arm.blocked:
            status, reason = "not run", arm.blocked
        rec = dict(m)
        rec.update(
            status=status, reason=reason,
            measured_latency_p50_ms=e.get("latency_p50_ms", "not measured (Stage 3 not run)"),
            measured_latency_p95_ms=e.get("latency_p95_ms", "not measured (Stage 3 not run)"),
            measured_latency_p99_ms=e.get("latency_p99_ms", "not measured (Stage 3 not run)"),
            joules_per_command=e.get("joules_per_command",
                                     m.get("joules_per_command", "not measured")),
            mean_power_w=e.get("mean_power_w", "not measured"),
            idle_median_w=e.get("idle_median_w", "not measured"),
            max_gpu_temp_c=e.get("max_gpu_temp_c", "not measured"),
            peak_vram_mb=e.get("peak_vram_mb", m.get("peak_vram_mb", "not measured")),
            cpu_ms_per_command=e.get("cpu_ms_per_command", "not measured"),
            energy_n_commands=e.get("n_commands", "not measured"),
            energy_wall_s=e.get("wall_s", "not measured"),
            ollama_version=e.get("ollama_version", ""),
            model_digest=e.get("model_digest", ""),
            primary_family_verdicts="; ".join(sig.get((arm_name, split), [])) or
                                    "not in the primary family",
            arm_note=arm.note if arm else "",
            withdrawn_policy=WITHDRAWN_NOTE)
        rows.append(rec)

    cols = list(rows[0].keys()) if rows else []
    p = C.OUT / "metrics_final.csv"
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: (r.get(c, "") if r.get(c, "") != "" else "n/a") for c in cols})

    valid = [r for r in rows if r["status"] == "valid"]
    C.gate(True, [
        "%d (arm, split) rows; %d valid, %d not run" % (len(rows), len(valid),
                                                        len(rows) - len(valid)),
        "no blank cells: every impossible cell carries its reason",
        "this file is the ONLY source for the final run's thesis tables",
        "output: %s" % p.relative_to(C.REPO),
    ])


if __name__ == "__main__":
    main()
