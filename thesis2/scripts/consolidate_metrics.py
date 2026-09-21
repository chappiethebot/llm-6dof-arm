"""thesis2/scripts/consolidate_metrics.py -- one row per arm, all metrics.

Pulls from every phase's CSV. Every row states its own denominator. No number
is recomputed here -- this only gathers what the phase scripts measured, so a
discrepancy between this file and a phase CSV is a bug in this file, not a
second opinion.
"""
import csv
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)
OUT = REPO / "thesis2" / "out"

F = ["arm", "dataset", "n_commands", "action_match", "joint_correct",
     "location_acc", "row_acc", "schema_validity", "reject_recall",
     "latency_p50", "latency_unit", "joules_per_command", "resident_mb",
     "escalation_rate", "note"]


def rd(name):
    p = OUT / name
    if not p.exists():
        return []
    with open(p, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def g(rows, **kw):
    for r in rows:
        if all(str(r.get(k, "")) == str(v) for k, v in kw.items()):
            return r
    return None


rows = []
energy = {r["arm"]: r for r in rd("energy.csv")}


def E(arm):
    r = energy.get(arm)
    return r["joules_per_command"] if r else ""


# ---- Phase 2: the two DistilBERT heads, n=217 fresh pass ----
fvf = {r["metric"]: r for r in rd("factorized_vs_flat.csv")}
for who, col in (("flat", "flat"), ("factorized", "factorized")):
    rows.append(dict(
        arm="distilbert_%s_head" % who, dataset="held_out_v3", n_commands=217,
        action_match=fvf["action_match"][col],
        joint_correct=fvf["joint_all_three"][col],
        location_acc=fvf["field_location"][col],
        row_acc=fvf["factor_row"][col],
        schema_validity="1.0",
        reject_recall=fvf["reject_recall"][col],
        latency_p50=fvf["latency_p50_ms"][col], latency_unit="ms_CPU",
        joules_per_command=E("%s_distilbert_gpu" %
                             ("flat" if who == "flat" else "factorized")),
        resident_mb="130.0", escalation_rate="0",
        note="Phase 2.2, fresh single pass, no warmup row"))

# ---- Phase 3: INT4 ----
qg = {r["quantity"]: r for r in rd("quantization_gate.csv")}
if qg:
    rows.append(dict(
        arm="distilbert_factorized_INT4", dataset="held_out_v3", n_commands=217,
        action_match=qg["action_match"]["int4"], joint_correct="",
        location_acc=qg["location"]["int4"], row_acc=qg["row"]["int4"],
        schema_validity="1.0", reject_recall="",
        latency_p50="3.22", latency_unit="ms_GPU", joules_per_command="",
        resident_mb="66.4", escalation_rate="0",
        note="GATE FAILED: INT4 costs 0.00 action points; dequantized matmul, "
             "not an INT4 kernel"))

# ---- Phase 4.5 cascade ----
for ds in ("held_out_v3", "held_out_v4c"):
    for target, tau, label in (("qwen3blora", 1.0, "always"),
                               ("qwen3blora", 0.5, "tau_A=0.50"),
                               ("cloud", 1.0, "always"),
                               ("cloud", 0.5, "tau_A=0.50"),
                               ("oracle", 1.0, "ceiling")):
        r = g(rd("p45_cascade_full.csv"), dataset=ds,
              signal="factorized_row_conf", target=target, tau=tau)
        if not r:
            continue
        rows.append(dict(
            arm="cascade_row->%s_%s" % (target, label), dataset=ds,
            n_commands=r["n_action"], action_match=r["action_match"],
            joint_correct=r["joint_correct"], location_acc=r["location_acc"],
            row_acc=r["row_acc"], schema_validity="1.0", reject_recall="",
            latency_p50=r["latency_p50_ms"], latency_unit="ms",
            joules_per_command=(E("salsa_qwen3blora") if target == "qwen3blora"
                                else ("NOT MEASURABLE" if target == "cloud" else "")),
            resident_mb=r["extra_resident_mb"],
            escalation_rate=r["escalation_rate"],
            note="per-factor gate on factorized row confidence; "
                 "escalates ONLY the 3-way row sub-question"))

# ---- Phase 5 baselines ----
for r in rd("baselines_summary.csv"):
    rows.append(dict(
        arm="baseline_%s" % r["arm"], dataset="held_out_v3", n_commands=r["n"],
        action_match=r["action_match"], joint_correct=r["joint"],
        location_acc=r["location"], row_acc="", schema_validity="1.0",
        reject_recall=r["reject_recall"], latency_p50=r["lat_p50_us"],
        latency_unit="us_CPU", joules_per_command="~0 GPU",
        resident_mb="", escalation_rate="0",
        note=("VOCABULARY-UNFAIR: rules saw upper/bottom/back/front which "
              "train_v2 lacks; firewalled variant scores 0.4470"
              if r["arm"] == "rules" else r["note"])))
for r in rd("baseline_rules_firewall.csv"):
    if r["arm"] != "rules_firewalled":
        continue
    rows.append(dict(
        arm="baseline_rules_firewalled", dataset=r["dataset"],
        n_commands=r["n"], action_match=r["action_match"], joint_correct="",
        location_acc=r["location_acc"], row_acc="", schema_validity="1.0",
        reject_recall="", latency_p50=r["lat_p50_us"], latency_unit="us_CPU",
        joules_per_command="~0 GPU", resident_mb="", escalation_rate="0",
        note="HONEST rules floor: only row words present in train_v2"))

# ---- Phase 6 SALSA, row factor only ----
for ds in ("held_out_v3", "held_out_v4c"):
    for r in rd("logit_calibration.csv"):
        if r["dataset"] != ds or not r["scorer"].startswith("salsa"):
            continue
        rows.append(dict(
            arm="P6_%s" % r["scorer"], dataset=ds, n_commands=r["n"],
            action_match="", joint_correct="", location_acc="",
            row_acc=r["accuracy"], schema_validity="n/a (row factor only)",
            reject_recall="", latency_p50="35.5", latency_unit="ms_GPU",
            joules_per_command=E("salsa_qwen3b" if "qwen3b_" in r["scorer"]
                                 else "salsa_qwen3blora"),
            resident_mb="6335" if "qwen3b_" in r["scorer"] else "6593",
            escalation_rate="1.0",
            note="SALSA 1 forward pass; ECE %s Brier %s" % (r["ECE"], r["Brier"])))

# ---- Phase 7 MASSIVE ----
sp = OUT / "generalization_massive_summary.json"
if sp.exists():
    s = json.loads(sp.read_text(encoding="utf-8"))
    for who, key in (("flat", "flat_accuracy"), ("factorized", "factorized_accuracy")):
        rows.append(dict(
            arm="massive_%s_head" % who, dataset="massive_en_US_test",
            n_commands=s["n_test"], action_match=s[key], joint_correct="",
            location_acc="", row_acc="", schema_validity="", reject_recall="",
            latency_p50="", latency_unit="", joules_per_command="",
            resident_mb="", escalation_rate="0",
            note="60-way composite intent; AUROC per-factor %.4f vs flat %.4f "
                 "-- NO advantage on non-disjoint data"
                 % (s["auroc_per_factor"], s["auroc_flat"])))

p = OUT / "metrics_all.csv"
with open(p, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=F)
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in F})

print("=" * 100)
print("metrics_all.csv -- %d arms" % len(rows))
print("=" * 100)
print("%-42s %-18s %5s %8s %8s %9s" %
      ("arm", "dataset", "n", "action", "row", "J/cmd"))
for r in rows:
    print("%-42s %-18s %5s %8s %8s %9s" %
          (r["arm"][:42], r["dataset"][:18], r["n_commands"],
           r["action_match"], r["row_acc"], r["joules_per_command"]))
print()
print("wrote %s" % p.relative_to(REPO))
print("=" * 100)
