"""thesis2/scripts/p1_digest.py -- print the headline numbers from the P1 CSVs."""
import csv
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
OUT = os.path.join("thesis2", "out")


def rd(name):
    with open(os.path.join(OUT, name), encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


print("=" * 78)
print("P1.1 TIDE -- pooled accuracy points recoverable, brief order vs anchor-first")
tide = rd("tide_contributions.csv")
for order in ("brief", "anchor_first"):
    agg = {}
    for r in tide:
        if r["order"] != order:
            continue
        agg[r["error_type"]] = agg.get(r["error_type"], 0.0) + float(r["contribution_pts"])
    n = len(set((r["model"], r["strategy"]) for r in tide if r["order"] == order))
    print("  order=%s  (mean over %d cells)" % (order, n))
    for k, v in sorted(agg.items(), key=lambda kv: -kv[1]):
        print("     %-20s %6.2f pts" % (k, v / n))

print()
print("=" * 78)
print("P1.6 calibration of the EXISTING flat classifier (n=217, no repeats)")
for r in rd("calibration_baseline.csv"):
    print("  %-9s n=%-4s acc=%-7s meanconf=%-7s ECE=%-7s Brier=%s"
          % (r["head"], r["n"], r["accuracy"], r["mean_confidence"], r["ECE"], r["Brier"]))

print()
print("=" * 78)
print("P1.4 location-error rate by location-head margin decile (n=217)")
for r in rd("confidence_split.csv"):
    print("  %-34s n=%-5s errors=%-4s rate=%s"
          % (r["decile"], r["n_commands"], r["location_errors"], r["location_error_rate"]))

print()
print("=" * 78)
print("P1.5 confusion signatures -- classifier and the two headline cells")
want = {("classifier:distilbert", "classifier"),
        ("ollama:qwen2.5-3b-lora", "zero_shot"),
        ("openai:gpt-4o-mini", "rag"),
        ("openai:gpt-4o-mini", "role_based")}
for r in rd("confusion_signatures.csv"):
    if (r["model"], r["strategy"]) not in want:
        continue
    print("  %s / %s  (%s location errors)" % (r["model"], r["strategy"], r["location_errors"]))
    for k in ("axis_inversion_row", "row_only_other", "col_only", "both_axes",
              "anchor_b2", "container_leakage", "runtime_confusion", "other"):
        if r[k] != "0":
            print("      %-22s %-5s %s" % (k, r[k], r["%s_frac" % k]))

print()
print("=" * 78)
print("P1.3 vocabulary buckets -- classifier + gpt-4o-mini/rag")
for r in rd("vocab_buckets.csv"):
    if (r["model"], r["strategy"]) in {("classifier:distilbert", "classifier"),
                                       ("openai:gpt-4o-mini", "rag")}:
        print("  %-24s %-11s %-11s n=%-4s acc=%s"
              % (r["model"], r["strategy"], r["bucket"], r["n_commands"], r["action_match"]))
print("=" * 78)
