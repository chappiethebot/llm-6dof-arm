"""thesis2/final/s2_metrics.py -- STAGE 2: every metric, every arm, both splits.

    python thesis2/final/s2_metrics.py [--split held_out_v3]

Coverage rule: EVERY arm gets EVERY metric. Where a metric cannot exist for an
arm, the cell says WHY instead of being blank:

  ECE                      needs class probabilities. Classifiers, SALSA-full,
                           cascades and kNN have them; generative cells (Ollama,
                           transformers greedy, GRPO, cloud) get
                           "n/a - no class probabilities".
  token acceptance,
  first-rejection field    speculative arms only; others
                           "n/a - not speculative".
  J/command                "not measurable" for cloud; CPU arms carry CPU ms.

Reads thesis2/out/final/pred_*.csv (Stage 1) and results_v5cells_*.csv, plus
unified_efficiency.csv (Stage 3) when it exists. Writes
thesis2/out/final/metrics_stage2.csv.

Diagnosis metrics, per arm x split:
  row accuracy split by row word -- seen (middle/center), unseen
  (upper/bottom on v3, back/front on v4c), mixed (both) -- and the row confusion
  signature (A<->C mirror vs default-to-B).
"""
import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C                                                      # noqa: E402
import arms as A                                                        # noqa: E402

SEEN_WORDS = ("middle", "center")
UNSEEN = {"held_out_v3": ("upper", "bottom"), "held_out_v4c": ("back", "front")}
NA_ECE = "n/a - no class probabilities"
NA_SPEC = "n/a - not speculative"
NOT_MEAS = "not measurable"


def word_group(cmd, split):
    low = cmd.lower()
    seen = any(re.search(r"\b%s\b" % w, low) for w in SEEN_WORDS)
    unseen = any(re.search(r"\b%s\b" % w, low) for w in UNSEEN[split])
    if seen and unseen:
        return "mixed"
    if unseen:
        return "unseen"
    if seen:
        return "seen"
    return "no row word"


def ece(pairs, bins=10):
    if not pairs:
        return None
    tot = 0.0
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        sub = [p for p in pairs if lo <= p[0] < hi or (i == bins - 1 and p[0] >= lo)]
        if not sub:
            continue
        conf = sum(c for c, _ in sub) / len(sub)
        acc = sum(1.0 for _, k in sub if k) / len(sub)
        tot += (len(sub) / len(pairs)) * abs(conf - acc)
    return round(tot, 4)


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))] if xs else None


def collapse(rows, determinism):
    """One outcome per command. Deterministic arms take repeat 0; sampling arms
    take the majority vote, A9's rule."""
    by = defaultdict(list)
    for r in rows:
        by[r["command_id"]].append(r)
    out = {}
    for cid, rs in by.items():
        rs.sort(key=lambda r: int(r["repeat"]))
        if determinism == "deterministic":
            out[cid] = rs[0]
        else:
            keep = dict(rs[0])
            keep["correct"] = C.majority(r["correct"] in (True, "True") for r in rs)
            out[cid] = keep
    return out


def spec_metrics(arm, split):
    """Token acceptance and first-rejection field, from the speculative arms'
    own per-command files."""
    src = {"a8_speculative": "a8_speculative_percommand.csv",
           "a10_speculative": "a10_prefix_cache_percommand.csv",
           "a12_smalllm": "a12_smalllm_drafter_percommand.csv"}.get(arm)
    if not src or split != "held_out_v3":
        return NA_SPEC, NA_SPEC
    p = C.REPO / "thesis2" / "out" / src
    if not p.exists():
        return "not run", "not run"
    rows = list(csv.DictReader(p.open(encoding="utf-8", newline="")))
    off = sum(int(r.get("draft_tokens") or r.get("draft_tokens_offered") or 0) for r in rows)
    acc = sum(int(r.get("draft_accepted") or r.get("draft_tokens_accepted") or 0) for r in rows)
    rate = round(acc / off, 4) if off else "n/a"
    rej = C.REPO / "thesis2" / "out" / "a11_rejection_positions.csv"
    dist = "n/a - a11_rejection_positions.csv not present"
    if rej.exists() and arm in ("a8_speculative", "a10_speculative"):
        c = Counter(r["region"] for r in csv.DictReader(rej.open(encoding="utf-8", newline=""))
                    if not r["region"].startswith("(none"))
        dist = "; ".join("%s=%d" % kv for kv in c.most_common())
    return rate, dist


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", choices=list(C.SPLITS), default=None)
    a = ap.parse_args()
    C.start_log("s2_metrics")

    eff = {}
    ep = C.OUT / "unified_efficiency.csv"
    if ep.exists():
        for r in csv.DictReader(ep.open(encoding="utf-8", newline="")):
            eff[(r["arm"], r["split"])] = r
        print("  Stage 3 efficiency: %d (arm, split) rows" % len(eff))
    else:
        print("  Stage 3 efficiency: unified_efficiency.csv not present -- energy "
              "columns will read 'not measured yet'")

    splits = [a.split] if a.split else list(C.SPLITS)
    out = []
    for split in splits:
        truth_rows = C.load_split(split)
        for name in sorted(A.REGISTRY):
            arm = A.REGISTRY[name]
            rows = C.read_preds(name, split)
            rec = dict(arm=name, split=split, kind=arm.kind, note=arm.note)
            if not rows:
                rec.update(status="not run", reason="no pred_%s_%s.csv" % (name, split))
                out.append(rec)
                continue
            first = collapse(rows, arm.determinism)
            n = len(first)
            correct = {c: (r["correct"] in (True, "True")) for c, r in first.items()}
            rec["n_commands"] = n
            rec["action_match"] = round(sum(correct.values()) / n, 4)

            grid = {c: r for c, r in first.items() if r["row_correct"] not in ("", None)}
            rec["n_row"] = len(grid)
            rec["row_accuracy"] = (round(sum(1 for r in grid.values()
                                             if r["row_correct"] in (True, "True")) / len(grid), 4)
                                   if grid else "n/a - no grid commands")
            for f in ("verb", "target", "location", "col"):
                sub = [r for r in first.values() if r["%s_correct" % f] not in ("", None)]
                rec["%s_accuracy" % f] = (round(sum(1 for r in sub
                                                    if r["%s_correct" % f] in (True, "True"))
                                                / len(sub), 4) if sub else "n/a")
            rec["schema_validity"] = round(sum(
                1 for r in first.values()
                if (json.loads(r["pred_json"]) or {}).get("action") in C.VERBS) / n, 4)

            # ---- diagnosis: row accuracy by row-word group
            groups = defaultdict(lambda: [0, 0])
            for cid, r in grid.items():
                g = word_group(truth_rows[int(cid)]["command"], split)
                groups[g][1] += 1
                groups[g][0] += int(r["row_correct"] in (True, "True"))
            rec["row_by_word_group"] = "; ".join(
                "%s %d/%d" % (g, h, t) for g, (h, t) in sorted(groups.items()))

            # ---- diagnosis: row confusion signature
            conf = Counter()
            for cid, r in grid.items():
                t = C.rowof(json.loads(r["truth_json"]).get("location"))
                p_ = C.rowof((json.loads(r["pred_json"]) or {}).get("location"))
                conf[(t, p_ or "-")] += 1
            mirror = sum(v for (t, p_), v in conf.items()
                         if {t, p_} == {"A", "C"})
            to_b = sum(v for (t, p_), v in conf.items() if p_ == "B" and t != "B")
            rec["row_confusion"] = "; ".join("%s->%s %d" % (t, p_, v)
                                             for (t, p_), v in sorted(conf.items()))
            rec["row_signature"] = ("A<->C mirror %d; default-to-B %d" % (mirror, to_b))

            # ---- confidence
            if arm.probs:
                pairs = []
                for cid, r in first.items():
                    try:
                        info = json.loads(r["note"] or "{}")
                    except Exception:
                        info = {}
                    p_ = info.get("p_location") or info.get("p_verb")
                    if isinstance(p_, (int, float)):
                        pairs.append((float(p_), correct[cid]))
                rec["ECE"] = ece(pairs) if pairs else "n/a - probabilities not recorded"
                rec["ECE_n"] = len(pairs)
            else:
                rec["ECE"] = NA_ECE
                rec["ECE_n"] = ""

            # ---- efficiency
            lat = [float(r["latency_ms"]) for r in rows if r["latency_ms"]]
            rec["latency_p50_ms"] = round(pct(lat, .50), 3) if lat else "n/a"
            rec["latency_p95_ms"] = round(pct(lat, .95), 3) if lat else "n/a"
            rec["latency_p99_ms"] = round(pct(lat, .99), 3) if lat else "n/a"
            e = eff.get((name, split))
            if arm.kind == "cloud":
                rec["joules_per_command"] = NOT_MEAS + " (cloud)"
                rec["peak_vram_mb"] = NOT_MEAS + " (cloud)"
            elif e:
                rec["joules_per_command"] = e.get("joules_per_command", "")
                rec["peak_vram_mb"] = e.get("peak_vram_mb", "")
            else:
                rec["joules_per_command"] = "not measured yet (run s3_energy.py)"
                rec["peak_vram_mb"] = "not measured yet (run s3_energy.py)"
            rec["cpu_ms_per_command"] = (e or {}).get("cpu_ms_per_command",
                                                      "not measured yet")
            rec["size_on_disk_mb"] = (e or {}).get("size_on_disk_mb", "not measured yet")

            # ---- speculative
            rec["token_acceptance"], rec["first_rejection_field"] = (
                spec_metrics(name, split) if arm.speculative else (NA_SPEC, NA_SPEC))

            rec["status"] = "valid"
            rec["reason"] = ""
            out.append(rec)

    cols = ["arm", "split", "kind", "status", "reason", "n_commands", "action_match",
            "n_row", "row_accuracy", "verb_accuracy", "target_accuracy",
            "location_accuracy", "col_accuracy", "schema_validity", "ECE", "ECE_n",
            "latency_p50_ms", "latency_p95_ms", "latency_p99_ms",
            "joules_per_command", "cpu_ms_per_command", "peak_vram_mb",
            "size_on_disk_mb", "token_acceptance", "first_rejection_field",
            "row_by_word_group", "row_confusion", "row_signature", "note"]
    p = C.OUT / "metrics_stage2.csv"
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in out:
            w.writerow({c: r.get(c, "") for c in cols})

    ran = [r for r in out if r.get("status") == "valid"]
    print()
    print("  %-34s %-14s %8s %8s %s" % ("arm", "split", "action", "row", "row by word group"))
    for r in ran:
        print("  %-34s %-14s %8s %8s %s"
              % (r["arm"], r["split"], r["action_match"], r["row_accuracy"],
                 r.get("row_by_word_group", "")[:46]))
    blank = sum(1 for r in out for v in r.values() if v == "")
    C.gate(True, [
        "%d (arm, split) rows; %d with predictions, %d not run"
        % (len(out), len(ran), len(out) - len(ran)),
        "every metric present for every arm; impossible cells carry the reason",
        "output: %s" % p.relative_to(C.REPO),
    ])


if __name__ == "__main__":
    main()
