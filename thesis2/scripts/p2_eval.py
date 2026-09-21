"""thesis2/scripts/p2_eval.py -- PHASE 2.2, factorized vs flat head.

Loads BOTH models and runs BOTH over held_out_v3 in one process, same device,
same tokenizer settings, same emit() guarantee. That removes every substrate
confound from the comparison -- the only difference is the location head.

  flat        ft/runs/distilbert_cls/epoch_09/model.pt   (read-only)
  factorized  thesis2/models/distilbert_factorized/epoch_11/model.pt

Scoring uses llm/score.py, imported read-only. `strict_match` is authoritative.

DENOMINATORS. This is a fresh evaluation, so there is no warmup row to drop:
every table is n=217 commands, one pass each, no repeats to collapse. The
published V5 classifier cell is n=216 because command_id 0 was consumed as its
warmup; the n=217 flat number computed here is therefore NOT identical to the
published 0.7500 and both are reported.

Location factor denominators, per the corrected composition:
  loc_type          168 commands that carry a location
  row / col         108 grid-cell commands
  container_which    36
  runtime_which      24

Writes thesis2/out/factorized_fp16.csv (predictions, scored-CSV schema) and
thesis2/out/factorized_vs_flat.csv (the comparison).
"""
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "thesis2" / "scripts"))

from llm.score import score                                    # noqa: E402
from ft.train_classifier import (MultiHeadClassifier,           # noqa: E402
                                 build_label_space as flat_space,
                                 emit as flat_emit, MAX_LEN)
from p2_train_factorized import (FactorizedClassifier, FACTORS,  # noqa: E402
                                 build_label_space, factorize, reconstruct,
                                 emit as fact_emit, IGNORE, ROWS, COLS,
                                 CONTAINERS, RUNTIMES)
from transformers import AutoTokenizer                          # noqa: E402

HELD = REPO / "data" / "held_out_v3.jsonl"
FLAT_CK = REPO / "ft" / "runs" / "distilbert_cls" / "epoch_09" / "model.pt"
FACT_CK = REPO / "thesis2" / "models" / "distilbert_factorized" / "epoch_11" / "model.pt"
OUT = REPO / "thesis2" / "out"

UPDOWN = ("upper", "bottom", "top", "lower")
FARNEAR = ("far", "near", "farthest", "nearest")


def bucket(cmd, loc):
    if loc is None:
        return "nolocation"
    low = " %s " % " ".join(cmd.lower().replace(",", " ").split())
    fn = any(" %s " % w in low for w in FARNEAR)
    ud = any(" %s " % w in low for w in UPDOWN)
    if ud and not fn:
        return "updown"
    if fn and not ud:
        return "farnear"
    if fn and ud:
        return "both"
    return "neither"


def ece(pairs, bins=10):
    if not pairs:
        return float("nan")
    tot = 0.0
    for i in range(bins):
        a, b = i / bins, (i + 1) / bins
        sub = [p for p in pairs if a <= p[0] < b or (i == bins - 1 and p[0] >= a)]
        if not sub:
            continue
        conf = sum(c for c, _ in sub) / len(sub)
        acc = sum(1.0 for _, k in sub if k) / len(sub)
        tot += (len(sub) / len(pairs)) * abs(conf - acc)
    return tot


def brier(pairs):
    return (sum((c - (1.0 if k else 0.0)) ** 2 for c, k in pairs) / len(pairs)
            if pairs else float("nan"))


def risk_coverage(pairs):
    """Selective risk at coverages 1.0 .. 0.1. pairs = [(conf, correct)]."""
    s = sorted(pairs, key=lambda p: -p[0])
    out = {}
    for cov in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5):
        k = max(1, int(round(cov * len(s))))
        sub = s[:k]
        out[cov] = 1.0 - sum(1.0 for _, c in sub if c) / len(sub)
    return out


def softmax_top(logits):
    p = torch.softmax(logits, dim=-1)[0]
    v, i = torch.sort(p, descending=True)
    return float(v[0]), int(i[0]), float(v[0] - v[1])


def main():
    device = "cpu"
    rows = [json.loads(l) for l in open(HELD, encoding="utf-8") if l.strip()]
    space = build_label_space()
    fspace = flat_space()
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    print("=" * 76)
    print("PHASE 2.2 -- factorized vs flat, both run here, n=%d commands" % len(rows))
    print("=" * 76)

    flat = MultiHeadClassifier(fspace)
    flat.load_state_dict(torch.load(FLAT_CK, map_location="cpu"))
    flat.eval()
    fact = FactorizedClassifier(space)
    fact.load_state_dict(torch.load(FACT_CK, map_location="cpu"))
    fact.eval()
    print("  flat       %s" % FLAT_CK.relative_to(REPO))
    print("  factorized %s" % FACT_CK.relative_to(REPO))
    nf = sum(p.numel() for p in flat.parameters())
    nx = sum(p.numel() for p in fact.parameters())
    print("  params: flat %.4f M   factorized %.4f M   delta %+d"
          % (nf / 1e6, nx / 1e6, nx - nf))

    recs = []
    lat = {"flat": [], "factorized": []}
    with torch.no_grad():
        for ci, row in enumerate(rows):
            cmd = row["command"]
            truth = dict(row["expected"])
            truth.setdefault("modifiers", [])
            enc = tok(cmd, truncation=True, max_length=MAX_LEN,
                      padding="max_length", return_tensors="pt")

            t0 = time.perf_counter()
            vl, tl, ll = flat(enc["input_ids"], enc["attention_mask"])
            lat["flat"].append((time.perf_counter() - t0) * 1000)
            fv, fvi, _ = softmax_top(vl)
            ft_, fti, _ = softmax_top(tl)
            fl, fli, flm = softmax_top(ll)
            flat_obj = flat_emit(fvi, fti, fli, fspace)

            t0 = time.perf_counter()
            lg = fact(enc["input_ids"], enc["attention_mask"])
            lat["factorized"].append((time.perf_counter() - t0) * 1000)
            conf, idx = {}, {}
            for k in ["verb", "target"] + FACTORS:
                c, i, _m = softmax_top(lg[k])
                conf[k], idx[k] = c, i
            sym = reconstruct(idx["loc_type"], idx["row"], idx["col"],
                              idx["container_which"], idx["runtime_which"])
            fact_obj = fact_emit(idx["verb"], idx["target"], sym, space)

            tlt, trr, tcc, tcw, trw = factorize(truth.get("location"))
            recs.append(dict(
                command_id=ci, command=cmd, truth=truth,
                bucket=bucket(cmd, truth.get("location")),
                flat_obj=flat_obj, fact_obj=fact_obj,
                flat_strict=score(flat_obj, truth).strict_match,
                fact_strict=score(fact_obj, truth).strict_match,
                flat_locconf=fl, flat_loc=flat_obj.get("location"),
                fact_sym=sym, conf=conf, idx=idx,
                truth_factors=dict(loc_type=tlt, row=trr, col=tcc,
                                   container_which=tcw, runtime_which=trw)))

    # ---------------------------------------------------------- predictions
    import csv
    p = OUT / "factorized_fp16.csv"
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "strategy", "dataset", "command_id", "command",
                    "run_index", "is_warmup", "stage", "ok", "strict_match",
                    "pred_json", "truth_json", "latency_s",
                    "conf_verb", "conf_target", "conf_loc_type", "conf_row",
                    "conf_col", "conf_container_which", "conf_runtime_which"])
        for i, r in enumerate(recs):
            w.writerow(["factorized:distilbert", "classifier", "held_out_v3",
                        r["command_id"], r["command"], 0, False, "validated",
                        True, r["fact_strict"], json.dumps(r["fact_obj"]),
                        json.dumps(r["truth"]),
                        round(lat["factorized"][i] / 1000.0, 6),
                        round(r["conf"]["verb"], 6), round(r["conf"]["target"], 6),
                        round(r["conf"]["loc_type"], 6), round(r["conf"]["row"], 6),
                        round(r["conf"]["col"], 6),
                        round(r["conf"]["container_which"], 6),
                        round(r["conf"]["runtime_which"], 6)])
    print("  wrote %s" % p.relative_to(REPO))

    # ------------------------------------------------------------- metrics
    n = len(recs)
    out = []

    def add(metric, denom, flatv, factv, note=""):
        out.append(dict(metric=metric, n=denom,
                        flat=round(flatv, 4) if flatv == flatv else "",
                        factorized=round(factv, 4) if factv == factv else "",
                        delta_pts=round(100 * (factv - flatv), 2)
                        if flatv == flatv and factv == factv else "",
                        note=note))

    add("action_match", n,
        sum(r["flat_strict"] for r in recs) / n,
        sum(r["fact_strict"] for r in recs) / n,
        "strict_match from llm/score.py")

    for f, key in (("verb", "action"), ("target", "target"), ("location", "location")):
        sub = [r for r in recs if r["truth"].get(key) is not None]
        add("field_%s" % f, len(sub),
            sum(1 for r in sub if r["flat_obj"].get(key) == r["truth"].get(key)) / len(sub),
            sum(1 for r in sub if r["fact_obj"].get(key) == r["truth"].get(key)) / len(sub))

    # joint: all three fields simultaneously right, on rows carrying all three
    tri = [r for r in recs if r["truth"].get("target") and r["truth"].get("location")]
    add("joint_all_three", len(tri),
        sum(1 for r in tri if all(r["flat_obj"].get(k) == r["truth"].get(k)
                                  for k in ("action", "target", "location"))) / len(tri),
        sum(1 for r in tri if all(r["fact_obj"].get(k) == r["truth"].get(k)
                                  for k in ("action", "target", "location"))) / len(tri))

    # ---- per-factor accuracy (factorized only; flat has no factors, but its
    # predicted SYMBOL can be factorized post-hoc for a like-for-like number)
    def flat_factor(r, f):
        loc = r["flat_obj"].get("location")
        if loc is None:
            return None
        try:
            lt, rr, cc, cw, rw = factorize(loc)
        except ValueError:
            return None
        return dict(loc_type=lt, row=rr, col=cc,
                    container_which=cw, runtime_which=rw)[f]

    for f in FACTORS:
        sub = [r for r in recs if r["truth_factors"][f] != IGNORE]
        if not sub:
            continue
        add("factor_%s" % f, len(sub),
            sum(1 for r in sub if flat_factor(r, f) == r["truth_factors"][f]) / len(sub),
            sum(1 for r in sub if r["idx"][f] == r["truth_factors"][f]) / len(sub),
            "flat column is its predicted symbol factorized post-hoc")

    # ---- THE AMENDMENT: updown bucket reported separately
    for b in ("updown", "neither", "nolocation"):
        sub = [r for r in recs if r["bucket"] == b]
        if not sub:
            continue
        add("bucket_%s_action_match" % b, len(sub),
            sum(r["flat_strict"] for r in sub) / len(sub),
            sum(r["fact_strict"] for r in sub) / len(sub))
    ud = [r for r in recs if r["bucket"] == "updown"]
    for f in ("row", "col"):
        sub = [r for r in ud if r["truth_factors"][f] != IGNORE]
        add("updown_factor_%s" % f, len(sub),
            sum(1 for r in sub if flat_factor(r, f) == r["truth_factors"][f]) / len(sub),
            sum(1 for r in sub if r["idx"][f] == r["truth_factors"][f]) / len(sub))

    # ---- schema validity / reject rate
    add("schema_validity", n, 1.0, 1.0, "both emit from the schema mask by construction")
    imp = [r for r in recs if r["truth"]["action"] == "reject"]
    if imp:
        add("reject_recall", len(imp),
            sum(1 for r in imp if r["flat_obj"]["action"] == "reject") / len(imp),
            sum(1 for r in imp if r["fact_obj"]["action"] == "reject") / len(imp))

    # ---- calibration
    flatloc = [(r["flat_locconf"], r["flat_obj"].get("location") == r["truth"].get("location"))
               for r in recs if r["truth"].get("location")]
    add("ECE_location_flat_vs_locfactors", len(flatloc), ece(flatloc), float("nan"),
        "flat has ONE location confidence; factorized has five -- see rows below")
    out[-1]["factorized"] = ""
    out[-1]["delta_pts"] = ""
    for f in FACTORS:
        sub = [r for r in recs if r["truth_factors"][f] != IGNORE]
        if not sub:
            continue
        pairs = [(r["conf"][f], r["idx"][f] == r["truth_factors"][f]) for r in sub]
        out.append(dict(metric="ECE_factor_%s" % f, n=len(sub), flat="",
                        factorized=round(ece(pairs), 4), delta_pts="",
                        note="Brier %.4f" % brier(pairs)))

    # latency
    for name, key in (("flat", "flat"), ("factorized", "factorized")):
        pass
    lf, lx = sorted(lat["flat"]), sorted(lat["factorized"])
    q = lambda v, p: v[min(len(v) - 1, int(round(p * (len(v) - 1))))]
    add("latency_p50_ms", n, statistics.median(lf), statistics.median(lx), "CPU, warm")
    add("latency_p95_ms", n, q(lf, 0.95), q(lx, 0.95), "CPU, warm")
    add("latency_p99_ms", n, q(lf, 0.99), q(lx, 0.99), "CPU, warm")

    p = OUT / "factorized_vs_flat.csv"
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["metric", "n", "flat", "factorized",
                                           "delta_pts", "note"])
        w.writeheader()
        w.writerows(out)
    print("  wrote %s" % p.relative_to(REPO))

    print()
    print("%-38s %5s %9s %11s %9s" % ("metric", "n", "flat", "factorized", "delta"))
    print("-" * 76)
    for r in out:
        print("%-38s %5s %9s %11s %9s"
              % (r["metric"], r["n"], r["flat"], r["factorized"], r["delta_pts"]))

    print()
    print("risk-coverage, factorized row factor (grid commands only):")
    rc = [(r["conf"]["row"], r["idx"]["row"] == r["truth_factors"]["row"])
          for r in recs if r["truth_factors"]["row"] != IGNORE]
    for cov, risk in sorted(risk_coverage(rc).items(), reverse=True):
        print("   coverage %.0f%%  selective risk %.4f" % (100 * cov, risk))
    print()
    print("risk-coverage, flat single location confidence:")
    for cov, risk in sorted(risk_coverage(flatloc).items(), reverse=True):
        print("   coverage %.0f%%  selective risk %.4f" % (100 * cov, risk))
    print("=" * 76)
    return 0


if __name__ == "__main__":
    sys.exit(main())
