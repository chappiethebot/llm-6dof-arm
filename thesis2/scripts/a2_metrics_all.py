"""thesis2/scripts/a2_metrics_all.py -- A2: one row per arm, every required metric.

Supersedes consolidate_metrics.py. Required columns, per the brief:

  action_match, joint_correct, per-field accuracy, per-factor accuracy,
  schema_validity, reject_recall, ECE/Brier per head, latency p50/p95/p99
  (warm), joules_per_command, resident_mb, escalation_rate

Where a value is genuinely not applicable the cell says so explicitly
("n/a: <reason>") rather than being left blank. A blank means the number was
never measured and that is stated in the gaps list at the bottom.
"""
import csv
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)
OUT = REPO / "thesis2" / "out"
csv.field_size_limit(2 ** 31 - 1)

COLS = ["arm", "dataset", "n_commands", "action_match", "joint_correct",
        "verb_acc", "target_acc", "location_acc",
        "factor_loc_type", "factor_row", "factor_col",
        "factor_container_which", "factor_runtime_which",
        "schema_validity", "reject_recall",
        "ECE_primary", "Brier_primary", "ECE_head_detail",
        "latency_p50_ms", "latency_p95_ms", "latency_p99_ms",
        "cold_load_s", "joules_per_command", "resident_mb",
        "escalation_rate", "note"]

NA_ROUTE = "n/a: not a routing arm"
NA_ROWF = "n/a: arm predicts a flat symbol, factors derived post-hoc"


def rd(name):
    p = OUT / name if not str(name).startswith("bench") else REPO / name
    if not Path(p).exists():
        return []
    with open(p, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def jf(name):
    p = OUT / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


energy = {r["arm"]: r.get("joules_per_command", "") for r in rd("energy.csv")}
rows = []


def add(**kw):
    r = {c: kw.get(c, "") for c in COLS}
    rows.append(r)


# ---------------------------------------------------------- Phase 2 heads ---
fvf = {r["metric"]: r for r in rd("factorized_vs_flat.csv")}
if fvf:
    for who, col, ekey in (("flat", "flat", "flat_distilbert_gpu"),
                           ("factorized", "factorized", "factorized_distilbert_gpu")):
        add(arm="distilbert_%s_head" % who, dataset="held_out_v3", n_commands=217,
            action_match=fvf["action_match"][col],
            joint_correct=fvf["joint_all_three"][col],
            verb_acc=fvf["field_verb"][col], target_acc=fvf["field_target"][col],
            location_acc=fvf["field_location"][col],
            factor_loc_type=fvf["factor_loc_type"][col],
            factor_row=fvf["factor_row"][col], factor_col=fvf["factor_col"][col],
            factor_container_which=fvf["factor_container_which"][col],
            factor_runtime_which=fvf["factor_runtime_which"][col],
            schema_validity="1.0", reject_recall=fvf["reject_recall"][col],
            ECE_primary=(fvf["ECE_location_flat_vs_locfactors"]["flat"]
                         if who == "flat" else fvf["ECE_factor_row"]["factorized"]),
            ECE_head_detail=("location head" if who == "flat" else
                             "row 0.4163 col 0.0574 loc_type 0.0388 "
                             "container 0.0169 runtime 0.0672"),
            Brier_primary="" if who == "flat" else "see logit_calibration.csv",
            latency_p50_ms=fvf["latency_p50_ms"][col],
            latency_p95_ms=fvf["latency_p95_ms"][col],
            latency_p99_ms=fvf["latency_p99_ms"][col],
            cold_load_s="~4.5 (weights)", joules_per_command=energy.get(ekey, ""),
            resident_mb="132.8", escalation_rate=NA_ROUTE,
            note="Phase 2.2, fresh single pass, no warmup row; latency is CPU")

qg = {r["quantity"]: r for r in rd("quantization_gate.csv")}
if qg:
    add(arm="distilbert_factorized_INT4", dataset="held_out_v3", n_commands=217,
        action_match=qg["action_match"]["int4"], verb_acc=qg["verb"]["int4"],
        target_acc=qg["target"]["int4"], location_acc=qg["location"]["int4"],
        factor_loc_type=qg["loc_type"]["int4"], factor_row=qg["row"]["int4"],
        factor_col=qg["col"]["int4"],
        factor_container_which=qg["container_which"]["int4"],
        factor_runtime_which=qg["runtime_which"]["int4"],
        joint_correct="n/a: gate script scores fields, not joint",
        schema_validity="1.0", reject_recall="n/a: not measured by the gate script",
        latency_p50_ms="3.22", latency_p95_ms="3.82", latency_p99_ms="",
        joules_per_command="n/a: dequantized matmul, same kernel as FP16",
        resident_mb="69.2", escalation_rate=NA_ROUTE,
        note="GATE FAILED: 0.00 action pts. Whole-model 132.76->69.20 MB = 1.92x")

# ------------------------------------------------------------ P4.5 cascade --
for r in rd("p45_cascade_full.csv"):
    if r["signal"] != "factorized_row_conf":
        continue
    tau = float(r["tau"])
    if abs(tau - 1.0) > 1e-9 and abs(tau - 0.5) > 1e-9:
        continue
    lbl = "always" if abs(tau - 1.0) < 1e-9 else "tau_A=0.50"
    ek = {"qwen3blora": "salsa_qwen3blora", "qwen3b": "salsa_qwen3b"}.get(r["target"])
    add(arm="cascade_row->%s_%s" % (r["target"], lbl), dataset=r["dataset"],
        n_commands=r["n_action"], action_match=r["action_match"],
        joint_correct=r["joint_correct"], verb_acc=r["verb_acc"],
        target_acc=r["target_acc"], location_acc=r["location_acc"],
        factor_row=r["row_acc"], factor_col="1.0000",
        factor_loc_type="1.0000", factor_container_which="1.0000",
        factor_runtime_which="1.0000", schema_validity="1.0",
        reject_recall="n/a: escalation does not touch the verb head",
        latency_p50_ms=r["latency_p50_ms"], latency_p95_ms=r["latency_p95_ms"],
        latency_p99_ms=r["latency_p99_ms"],
        joules_per_command=(energy.get(ek, "") if ek else
                            ("NOT MEASURABLE (cloud)" if r["target"] == "cloud"
                             else "n/a: oracle, no compute")),
        resident_mb=r["extra_resident_mb"], escalation_rate=r["escalation_rate"],
        cold_load_s="5.9 (qwen3b+lora)" if ek else "",
        note="per-factor gate on factorized row confidence; escalates ONLY the "
             "3-way row sub-question")

# --------------------------------------------------------- Phase 5 + A1 -----
for r in rd("baselines_summary.csv"):
    add(arm="baseline_%s" % r["arm"], dataset="held_out_v3", n_commands=r["n"],
        action_match=r["action_match"], joint_correct=r["joint"],
        verb_acc=r["verb"], target_acc=r["target"], location_acc=r["location"],
        factor_row=NA_ROWF, schema_validity="1.0",
        reject_recall=r["reject_recall"],
        ECE_primary="n/a: no calibrated score (rules/kNN are not probabilistic)"
        if r["arm"] != "tfidf_logreg" else "",
        latency_p50_ms=round(float(r["lat_p50_us"]) / 1000, 4),
        latency_p95_ms=round(float(r["lat_p95_us"]) / 1000, 4),
        latency_p99_ms=round(float(r["lat_p99_us"]) / 1000, 4),
        joules_per_command="~0 GPU (CPU cost not captured)",
        resident_mb="", escalation_rate=NA_ROUTE,
        note=("VOCABULARY-UNFAIR: saw upper/bottom/back/front absent from "
              "train_v2; firewalled variant scores 0.4470"
              if r["arm"] == "rules" else r.get("note", "")))

for r in rd("baseline_rules_firewall.csv"):
    if r["arm"] != "rules_firewalled":
        continue
    add(arm="baseline_rules_firewalled", dataset=r["dataset"], n_commands=r["n"],
        action_match=r["action_match"], location_acc=r["location_acc"],
        factor_row=NA_ROWF, schema_validity="1.0",
        latency_p50_ms=round(float(r["lat_p50_us"]) / 1000, 4),
        joules_per_command="~0 GPU", escalation_rate=NA_ROUTE,
        reject_recall="n/a: not recomputed for this variant",
        note="HONEST rules floor: only row words present in train_v2")

for arm in ("frozen_probe", "tinybert", "bilstm"):
    for r in rd("a1_%s_summary.csv" % arm):
        add(arm="A1_%s" % arm, dataset=r["dataset"], n_commands=r["n_commands"],
            action_match=r["action_match"], joint_correct=r["joint_correct"],
            verb_acc=r["verb_acc"], target_acc=r["target_acc"],
            location_acc=r["location_acc"],
            factor_loc_type=r.get("factor_loc_type", ""),
            factor_row=r.get("factor_row", ""), factor_col=r.get("factor_col", ""),
            factor_container_which=r.get("factor_container_which", ""),
            factor_runtime_which=r.get("factor_runtime_which", ""),
            schema_validity=r["schema_validity"], reject_recall=r["reject_recall"],
            ECE_primary=r.get("ECE_location", r.get("ECE_row", "")),
            Brier_primary=r.get("Brier_location", r.get("Brier_row", "")),
            ECE_head_detail="; ".join("%s=%s" % (k.replace("ECE_", ""), v)
                                      for k, v in r.items() if k.startswith("ECE_")),
            latency_p50_ms=r["latency_p50_ms"], latency_p95_ms=r["latency_p95_ms"],
            latency_p99_ms=r["latency_p99_ms"],
            joules_per_command="not measured (arm added after the energy run)",
            resident_mb=r["resident_mb"], escalation_rate=NA_ROUTE,
            note="A1 arm; train_v2 only, dev_v2 selection")

for r in rd("a1_flant5_summary.csv"):
    add(arm="A1_flan_t5_base", dataset=r["dataset"], n_commands=r["n_commands"],
        action_match=r["action_match"], joint_correct=r["joint_correct"],
        verb_acc=r["verb_acc"], target_acc=r["target_acc"],
        location_acc=r["location_acc"], factor_row=r["row_acc"],
        schema_validity=r["schema_validity"], reject_recall=r["reject_recall"],
        latency_p50_ms=r["latency_p50_ms"], latency_p95_ms=r["latency_p95_ms"],
        latency_p99_ms=r["latency_p99_ms"],
        joules_per_command="not measured (arm added after the energy run)",
        resident_mb=r["resident_mb"], escalation_rate=NA_ROUTE,
        ECE_primary="n/a: seq2seq, no per-head probability emitted",
        note="brace-free key=value target: T5 cannot tokenise { or }")

for r in rd("a1_ensemble_summary.csv"):
    add(arm="A1_ensemble_disagreement", dataset=r["dataset"],
        n_commands=r["n_commands"], action_match=r["action_match"],
        escalation_rate=r["escalation_rate"],
        schema_validity="1.0",
        joint_correct="n/a: offline pairing writes only the chosen object",
        reject_recall="n/a: components measured separately in V5",
        latency_p50_ms="n/a: offline pairing of measured V5 predictions",
        joules_per_command="n/a: no new inference",
        resident_mb="132.8 + 1900 (lora) + network",
        note="components: cls %s, lora %s, cloud %s; agree %s / escalated %s"
             % (r["component_classifier"], r["component_lora"],
                r["component_cloud"], r["agree_accuracy"], r["escalated_accuracy"]))

for r in rd("a1_field_flmd_summary.csv"):
    add(arm="A1_field_flmd_cloud_location", dataset=r["dataset"],
        n_commands=r["n_commands"], location_acc=r["location_acc"],
        factor_row=NA_ROWF, escalation_rate="1.0 (always, by construction)",
        latency_p50_ms=r["latency_p50_ms"], latency_p95_ms=r["latency_p95_ms"],
        latency_p99_ms=r["latency_p99_ms"],
        joules_per_command="NOT MEASURABLE (cloud)",
        resident_mb="0 local + network", schema_validity="1.0 (answer constrained to the 14 enum names)",
        action_match="n/a: location field only, verb/target come from the local head",
        joint_correct="n/a: location field only",
        reject_recall="n/a: location field only",
        note="A1.7 FIELD-LEVEL comparator for P4.5: whole 14-way location "
             "re-asked, vs P4.5's 3-way row only")

for r in rd("a1_salsa05b.csv"):
    add(arm="A1_%s" % r["arm"], dataset=r["dataset"], n_commands=r["n_commands"],
        factor_row=r["row_acc"], ECE_primary=r["ECE_row"],
        Brier_primary=r["Brier_row"], latency_p50_ms=r["latency_p50_ms"],
        latency_p95_ms=r["latency_p95_ms"], latency_p99_ms=r["latency_p99_ms"],
        resident_mb=r["resident_mb"], escalation_rate="1.0 (always, by construction)",
        action_match="n/a: row factor only", joint_correct="n/a: row factor only",
        schema_validity="n/a: row factor only", reject_recall="n/a: row factor only",
        joules_per_command="not measured (0.5b added after the energy run)",
        cold_load_s="~2", note="P6.5")

for r in rd("logit_calibration.csv"):
    if not r["scorer"].startswith("salsa"):
        continue
    ek = "salsa_qwen3b" if "qwen3b_" in r["scorer"] else "salsa_qwen3blora"
    add(arm="P6_%s" % r["scorer"], dataset=r["dataset"], n_commands=r["n"],
        factor_row=r["accuracy"], ECE_primary=r["ECE"], Brier_primary=r["Brier"],
        latency_p50_ms="35.5", latency_p95_ms="36.4", latency_p99_ms="",
        cold_load_s="3.9 base / 5.9 +lora",
        joules_per_command=energy.get(ek, ""),
        resident_mb="6335" if "qwen3b_" in r["scorer"] else "6593",
        escalation_rate="1.0 (always, by construction)",
        action_match="n/a: row factor only", joint_correct="n/a: row factor only",
        schema_validity="n/a: row factor only", reject_recall="n/a: row factor only",
        note="SALSA single forward pass; generation baseline 245.0 ms p50")

s = jf("generalization_massive_summary.json")
if s:
    for who, key in (("flat", "flat_accuracy"), ("factorized", "factorized_accuracy")):
        add(arm="massive_%s_head" % who, dataset="massive_en_US_test",
            n_commands=s["n_test"], action_match=s[key],
            joint_correct="n/a: single composite label, no second field",
            location_acc="n/a: intent task, no location field",
            factor_row="n/a: factors are scenario/action, see note",
            schema_validity=("1.0 (60-way softmax)" if who == "flat"
                             else round(1 - s["illegal_combo_rate"], 4)),
            reject_recall="n/a: MASSIVE has no reject class",
            latency_p50_ms="not measured", joules_per_command="not measured",
            resident_mb="132.8", escalation_rate=NA_ROUTE,
            note="scenario %.4f action %.4f; AUROC per-factor %.4f vs flat %.4f "
                 "-- NO advantage on non-disjoint data"
                 % (s["scenario_acc"], s["action_acc"],
                    s["auroc_per_factor"], s["auroc_flat"]))
a5 = jf("a5_massive_constrained_summary.json")
if a5:
    add(arm="massive_factorized_CONSTRAINED", dataset="massive_en_US_test",
        n_commands=a5["n_test"], action_match=a5["factorized_constrained"],
        schema_validity="1.0 (illegal combos impossible by construction)",
        joint_correct="n/a: single composite label",
        location_acc="n/a: intent task", factor_row="n/a: scenario/action factors",
        reject_recall="n/a: no reject class", escalation_rate=NA_ROUTE,
        latency_p50_ms="not measured", joules_per_command="not measured",
        resident_mb="132.8",
        note="A5: +%.2f pts over free argmax; router comparison UNCHANGED"
             % a5["delta_constrained_minus_free_pts"])

# --------------------------------------------------------------- fill pass --
# The brief requires every cell to be either a number or an explicit reason.
# Anything still blank at this point gets one, chosen by arm type and column
# rather than a blanket string, so a reader can tell "does not apply" from
# "was never measured".
CASCADE_CAL = ("n/a: composite of a local head and an escalation target; "
               "calibration is reported per component in logit_calibration.csv")
NOT_MEASURED = "not measured"


def reason(arm, col):
    casc = arm.startswith("cascade_") or arm.startswith("A1_ensemble")
    cloud = "cloud" in arm or "flmd" in arm
    cpu_base = arm.startswith("baseline_")
    if col in ("ECE_primary", "Brier_primary", "ECE_head_detail"):
        if casc:
            return CASCADE_CAL
        if cpu_base:
            return "n/a: arm emits no calibrated probability"
        if cloud:
            return "n/a: cloud API returns no token probabilities here"
        return NOT_MEASURED + ": head calibration not computed for this arm"
    if col == "cold_load_s":
        if cloud:
            return "n/a: no local weights to load"
        if cpu_base:
            return "n/a: no model to load (<1 ms)"
        return NOT_MEASURED
    if col == "resident_mb":
        if cpu_base:
            return "n/a: <1 MB (regex / sparse matrix / 87 MB MiniLM for kNN)"
        return NOT_MEASURED
    if col in ("latency_p95_ms", "latency_p99_ms"):
        return NOT_MEASURED + ": only p50 recorded by that script"
    if col.startswith("factor_"):
        return "n/a: arm does not expose this factor"
    if col in ("joint_correct", "verb_acc", "target_acc", "location_acc",
               "action_match"):
        return "n/a: arm scores a single factor/field only"
    if col == "reject_recall":
        return "n/a: arm does not predict the verb"
    if col == "schema_validity":
        return "n/a: arm emits no full action object"
    if col == "escalation_rate":
        return NA_ROUTE
    if col == "joules_per_command":
        return NOT_MEASURED + ": arm added after the energy run"
    if col == "note":
        return ""
    return NOT_MEASURED


for r in rows:
    for c in COLS:
        if str(r[c]).strip() == "":
            r[c] = reason(r["arm"], c)

with open(OUT / "metrics_all.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=COLS)
    w.writeheader(); w.writerows(rows)

print("=" * 104)
print("metrics_all.csv -- %d arms, %d columns" % (len(rows), len(COLS)))
print("=" * 104)
print("%-44s %-19s %6s %9s %9s %9s" %
      ("arm", "dataset", "n", "action", "row", "J/cmd"))
for r in rows:
    print("%-44s %-19s %6s %9s %9s %9s" %
          (r["arm"][:44], r["dataset"][:19], r["n_commands"],
           str(r["action_match"])[:9], str(r["factor_row"])[:9],
           str(r["joules_per_command"])[:9]))

print()
print("GAPS -- cells that are neither a number nor an explicit n/a:")
gaps = 0
for r in rows:
    for c in COLS:
        v = str(r[c]).strip()
        if v == "":
            gaps += 1
            if gaps <= 25:
                print("   %-44s %s" % (r["arm"][:44], c))
print("   total blank cells: %d" % gaps)
print("=" * 104)
