#!/usr/bin/env python
"""
bench/summary.py -- KPI summary over scored grid CSVs. Read-only; computes
nothing new, just reports. Works on scored files; the cost/throughput sections
additionally need the token columns filled (see bench/backfill_tokens.py).

Sections:
  1. Accuracy      -- strict / normalized / gap, per cell
  2. Latency       -- median, p95, p99, plus generation throughput (tokens/sec)
  3. Reject-rate   -- unmappable commands correctly routed to `reject` (safety KPI)
  4. Cost          -- cloud USD per command by strategy (cloud rows only)

Usage:
    python -m bench.summary --in bench/results_grid_tokens.csv
    python -m bench.summary --in bench/results_grid_tokens.csv --held bench/results_heldout_tokens.csv
"""
import argparse, csv, json, math, statistics
from collections import defaultdict

MODELS = ["openai:gpt-4o-mini", "ollama:llama3.2:3b", "ollama:qwen2.5:3b",
          "ollama:qwen2.5:0.5b", "ollama:deepseek-r1:7b"]
STRATS = ["zero_shot", "few_shot", "cot", "role_based"]
CLOUD = "openai:gpt-4o-mini"


def short(m):
    return m.split(":", 1)[1] if ":" in m else m


def load(path):
    """Read a scored CSV, dedupe on identity key keeping LAST (resume orphans)."""
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    seen = {}
    for r in rows:
        seen[(r["model"], r["strategy"], r["command_id"], r["run_index"])] = r
    return list(seen.values())


def pct(vals, p):
    v = sorted(vals)
    k = (len(v) - 1) * p / 100
    f, c = math.floor(k), math.ceil(k)
    return v[f] if f == c else v[f] + (v[c] - v[f]) * (k - f)


def has_tokens(rows):
    return any(r.get("tokens_out") not in (None, "") for r in rows)


def sec_accuracy(rows):
    print("1. ACCURACY  strict / normalized / gap")
    print(f'{"model":16}{"strategy":11}{"strict":>8}{"norm":>8}{"gap":>7}')
    for m in MODELS:
        for s in STRATS:
            cell = [r for r in rows if r["model"] == m and r["strategy"] == s]
            if not cell:
                continue
            st = sum(r["strict_match"] == "True" for r in cell) / len(cell)
            nm = sum(r["normalized_match"] == "True" for r in cell) / len(cell)
            flag = "  <==" if (nm - st) > 0.10 else ""
            print(f'{short(m):16}{s:11}{st*100:7.1f}%{nm*100:7.1f}%{(nm-st)*100:+6.1f}%{flag}')


def sec_latency(rows):
    tok = has_tokens(rows)
    print("\n2. LATENCY (s), warmup excluded, pooled across strategies"
          + ("  + generation throughput" if tok else ""))
    hdr = f'{"model":16}{"median":>8}{"p95":>8}{"p99":>8}{"max":>8}'
    if tok:
        hdr += f'{"med tok":>9}{"tok/s":>8}'
    print(hdr)
    for m in MODELS:
        c = [r for r in rows if r["model"] == m and r["is_warmup"] != "True" and r["latency_s"]]
        if not c:
            continue
        lats = [float(r["latency_s"]) for r in c]
        line = (f'{short(m):16}{statistics.median(lats):8.2f}{pct(lats,95):8.2f}'
                f'{pct(lats,99):8.2f}{max(lats):8.2f}')
        if tok:
            ct = [r for r in c if r.get("tokens_out") not in (None, "")]
            if ct:
                mt = statistics.median(int(r["tokens_out"]) for r in ct)
                tps = statistics.median(int(r["tokens_out"]) / float(r["latency_s"])
                                        for r in ct if float(r["latency_s"]) > 0)
                line += f'{mt:9.0f}{tps:8.1f}'
        print(line)
    if tok:
        print("   (low tok/s => slow generation, not verbosity. Note whether the model")
        print("    fits in VRAM before attributing this to architecture.)")


def _reject_stats(rows):
    stat = defaultdict(lambda: [0, 0])
    for r in rows:
        try:
            t = json.loads(r["truth_json"])
        except (json.JSONDecodeError, TypeError, KeyError):
            continue
        if t.get("action") != "reject":
            continue
        stat[r["model"]][1] += 1
        try:
            p = json.loads(r["pred_json"]) if r["pred_json"] else None
        except (json.JSONDecodeError, TypeError):
            p = None
        if isinstance(p, dict) and p.get("action") == "reject":
            stat[r["model"]][0] += 1
    return stat


def sec_reject(rows, held=None):
    print("\n3. REJECT-RATE  (unmappable/unsafe commands correctly routed to `reject`)")
    a = _reject_stats(rows)
    b = _reject_stats(held) if held else None
    hdr = f'{"model":16}{"seen":>12}'
    if b:
        hdr += f'{"held-out":>12}'
    print(hdr)
    for m in MODELS:
        c, n = a.get(m, [0, 0])
        line = f'{short(m):16}' + (f'{c/n*100:11.1f}%' if n else f'{"n/a":>12}')
        if b:
            c2, n2 = b.get(m, [0, 0])
            line += (f'{c2/n2*100:11.1f}%' if n2 else f'{"n/a":>12}')
        print(line)
    print("   A safety KPI: 0% means the model NEVER refuses an impossible command,")
    print("   and instead emits a plausible-looking action.")


def sec_cost(rows):
    if not has_tokens(rows):
        print("\n4. COST -- skipped (token columns empty; run bench/backfill_tokens.py first)")
        return
    print("\n4. CLOUD COST per command by strategy  (gpt-4o-mini rows only)")
    print(f'{"strategy":12}{"tok_in":>8}{"tok_out":>9}{"USD/cmd":>12}{"USD/1k cmd":>12}')
    total = 0.0
    for s in STRATS:
        c = [r for r in rows if r["model"] == CLOUD and r["strategy"] == s
             and r.get("cost_usd") not in (None, "")]
        if not c:
            continue
        ti = sum(int(r["tokens_in"]) for r in c) / len(c)
        to = sum(int(r["tokens_out"]) for r in c) / len(c)
        cost = sum(float(r["cost_usd"]) for r in c) / len(c)
        total += sum(float(r["cost_usd"]) for r in c)
        print(f'{s:12}{ti:8.0f}{to:9.0f}{cost:12.6f}{cost*1000:12.3f}')
    print(f'   total cloud spend in this file: ${total:.4f}')
    print("   (local models: no API cost; their token counts use the OpenAI")
    print("    tokenizer and are approximate.)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="bench/results_grid_scored.csv")
    ap.add_argument("--held", default=None, help="optional held-out file, for the reject-rate column")
    args = ap.parse_args()

    rows = load(args.inp)
    held = load(args.held) if args.held else None
    print(f"loaded {len(rows)} rows from {args.inp}"
          + (f" | {len(held)} from {args.held}" if held else "") + "\n")

    sec_accuracy(rows)
    sec_latency(rows)
    sec_reject(rows, held)
    sec_cost(rows)


if __name__ == "__main__":
    main()
