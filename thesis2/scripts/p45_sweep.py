"""thesis2/scripts/p45_sweep.py -- P4.5 threshold sweep, offline from the cache.

Two things are swept and they answer different questions.

  ROUTER SIGNAL (the Idea 1 evidence). Which confidence better identifies the
  commands whose ROW is wrong? Both signals escalate the SAME sub-question to
  the SAME model, so only the gate differs:
      A  factorized row confidence   (per-factor)
      B  flat head location confidence (single, whole-field)
  Compared at MATCHED escalation rate, which is the only fair axis.

  ESCALATION TARGET. Who answers the escalated row question:
      qwen3b (SALSA)   qwen3blora (SALSA)   gpt-4o-mini   oracle (upper bound)

NO THRESHOLD IS CHOSEN HERE. The full sweep is reported. The two operating
points quoted come from thesis2/out/THRESHOLD_PREREGISTRATION.md:
    tau_A = 0.50 fixed, uses no data
    tau_B = smallest tau with escalation rate <= 20% on dev_v2
Both were fixed before any of these numbers existed.
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

OUT = REPO / "thesis2" / "out"
GRID = [round(0.05 * i, 2) for i in range(21)]
TARGETS = ["qwen3b", "qwen3blora", "cloud", "oracle"]
SIGNALS = {"factorized_row_conf": "fact_row_conf",
           "flat_location_conf": "flat_location_conf"}

# measured in p45_collect.py, reused so the sweep stays offline
LAT_LOCAL_MS = 3.2          # factorized DistilBERT forward, GPU p50
LAT = {"qwen3b": 35.5, "qwen3blora": 35.4, "cloud": 657.0, "oracle": 0.0}


def rd(ds):
    with open(OUT / ("p45_predictions_%s.csv" % ds), encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def escalated_row(r, target, mapping="letters"):
    if target == "oracle":
        return r["truth_row"]
    if target == "cloud":
        return r["cloud_row"]
    return r["salsa_%s_%s_row" % (target, mapping)]


def dev_tau_b():
    """tau_B from dev_v2 only: smallest tau with dev escalation rate <= 20%."""
    import torch
    from transformers import AutoTokenizer
    from p2_train_factorized import (FactorizedClassifier, build_label_space,
                                     factorize, IGNORE, MAX_LEN)
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    space = build_label_space()
    m = FactorizedClassifier(space)
    m.load_state_dict(torch.load(
        REPO / "thesis2" / "models" / "distilbert_factorized" / "epoch_11" / "model.pt",
        map_location="cpu"))
    m.eval()
    confs = []
    rows = [json.loads(l) for l in open(REPO / "data" / "dev_v2.jsonl",
                                        encoding="utf-8") if l.strip()]
    with torch.no_grad():
        for r in rows:
            if factorize(r["expected"].get("location"))[1] == IGNORE:
                continue
            enc = tok(r["command"], truncation=True, max_length=MAX_LEN,
                      padding="max_length", return_tensors="pt")
            lg = m(enc["input_ids"], enc["attention_mask"])
            confs.append(float(torch.softmax(lg["row"], -1)[0].max()))
    for tau in GRID:
        rate = sum(1 for c in confs if c < tau) / len(confs)
        if rate <= 0.20:
            best = tau
    # smallest tau meeting the rule is simply the smallest in grid with rate<=.2
    ok = [t for t in GRID if sum(1 for c in confs if c < t) / len(confs) <= 0.20]
    return (min(ok) if ok else 0.0), len(confs), confs


def sweep(recs, ds):
    grid = [r for r in recs if r["truth_row"]]
    n = len(grid)
    out = []
    for sig_name, col in SIGNALS.items():
        for target in TARGETS:
            for tau in GRID:
                esc = [r for r in grid if float(r[col]) < tau]
                rowacc = 0
                for r in grid:
                    pred = (escalated_row(r, target) if r in esc else r["fact_row"])
                    if not pred:
                        pred = r["fact_row"]
                    rowacc += int(pred == r["truth_row"])
                rate = len(esc) / n
                lat_mean = LAT_LOCAL_MS + rate * LAT[target]
                out.append(dict(dataset=ds, signal=sig_name, target=target,
                                tau=tau, n_grid=n,
                                escalation_rate=round(rate, 4),
                                escalated=len(esc),
                                row_accuracy=round(rowacc / n, 4),
                                mean_latency_ms=round(lat_mean, 2)))
    return out


def baseline_rows(recs, ds):
    grid = [r for r in recs if r["truth_row"]]
    n = len(grid)
    b = []
    b.append(("local factorized row head (no escalation)",
              sum(int(r["fact_row"] == r["truth_row"]) for r in grid) / n))
    b.append(("local flat head row component (no escalation)",
              sum(int(r["flat_row"] == r["truth_row"]) for r in grid) / n))
    for t in TARGETS:
        if t == "oracle":
            continue
        b.append(("always escalate row -> %s" % t,
                  sum(int(escalated_row(r, t) == r["truth_row"]) for r in grid) / n))
    for t in ("qwen3b", "qwen3blora"):
        b.append(("always escalate row -> %s (digits mapping)" % t,
                  sum(int(escalated_row(r, t, "digits") == r["truth_row"])
                      for r in grid) / n))
    return n, b


def auroc(pairs):
    """pairs = [(score, positive_bool)]; positive = CORRECT. Higher score
    should mean more likely correct."""
    pos = [s for s, k in pairs if k]
    neg = [s for s, k in pairs if not k]
    if not pos or not neg:
        return float("nan")
    wins = sum((1.0 if p > q else 0.5 if p == q else 0.0)
               for p in pos for q in neg)
    return wins / (len(pos) * len(neg))


def main():
    tau_b, ndev, devconfs = dev_tau_b()
    print("=" * 78)
    print("PRE-REGISTERED OPERATING POINTS")
    print("  tau_A = 0.50 (fixed, no data used)")
    print("  tau_B = %.2f  (smallest tau with dev_v2 row-escalation rate <= 20%%; "
          "n_dev_grid=%d)" % (tau_b, ndev))
    print("  dev row confidence: min %.4f  median %.4f  max %.4f"
          % (min(devconfs), statistics.median(devconfs), max(devconfs)))
    print("  -> as pre-registered, dev contains NO updown/backfront commands, so")
    print("     the dev confidences are all high and tau_B lands at the grid floor.")
    print("=" * 78)

    allrows = []
    for ds in ("held_out_v3", "held_out_v4c"):
        recs = rd(ds)
        n, base = baseline_rows(recs, ds)
        print()
        print("=" * 78)
        print("%s -- row factor, n=%d grid commands" % (ds, n))
        print("=" * 78)
        for name, acc in base:
            print("  %-52s %.4f" % (name, acc))

        grid = [r for r in recs if r["truth_row"]]
        print()
        print("  ROUTER SIGNAL QUALITY (AUROC, does the score rank correct above wrong?)")
        for sig_name, col in SIGNALS.items():
            pairs = [(float(r[col]), r["fact_row"] == r["truth_row"]) for r in grid]
            a1 = auroc(pairs)
            pairs2 = [(float(r[col]), r["flat_row"] == r["truth_row"]) for r in grid]
            print("    %-22s AUROC vs factorized-row-correct %.4f | "
                  "vs flat-row-correct %.4f" % (sig_name, a1, auroc(pairs2)))

        sw = sweep(recs, ds)
        allrows += sw
        print()
        print("  MATCHED-BUDGET COMPARISON, escalating row -> gpt-4o-mini")
        print("  %-8s %-24s %-24s" % ("budget", "factorized_row_conf", "flat_location_conf"))
        for budget in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
            line = "  %-8.0f%%" % (100 * budget)
            for sig in SIGNALS:
                cand = [r for r in sw if r["signal"] == sig and r["target"] == "cloud"
                        and r["escalation_rate"] <= budget + 1e-9]
                best = max(cand, key=lambda r: r["escalation_rate"]) if cand else None
                line += " %-24s" % ("acc %.4f @ rate %.2f" %
                                    (best["row_accuracy"], best["escalation_rate"])
                                    if best else "-")
            print(line)

        print()
        print("  PRE-REGISTERED POINTS (row accuracy | escalation rate | mean latency ms)")
        print("  %-22s %-8s %-28s %-28s" % ("signal", "tau", "-> qwen3blora (SALSA)", "-> gpt-4o-mini"))
        for sig in SIGNALS:
            for label, tau in (("A", 0.50), ("B", tau_b)):
                cells = []
                for t in ("qwen3blora", "cloud"):
                    m = [r for r in sw if r["signal"] == sig and r["target"] == t
                         and abs(r["tau"] - tau) < 1e-9]
                    cells.append("%.4f | %.2f | %.1f" % (m[0]["row_accuracy"],
                                                        m[0]["escalation_rate"],
                                                        m[0]["mean_latency_ms"])
                                 if m else "-")
                print("  %-22s tau_%s=%.2f %-28s %-28s" % (sig, label, tau, cells[0], cells[1]))

    p = OUT / "p45_row_escalation_sweep.csv"
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(allrows[0]))
        w.writeheader()
        w.writerows(allrows)
    print()
    print("wrote %s (%d rows)" % (p.relative_to(REPO), len(allrows)))
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
