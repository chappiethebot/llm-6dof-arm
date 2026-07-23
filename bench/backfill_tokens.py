#!/usr/bin/env python
"""
bench/backfill_tokens.py -- fill the reserved tokens_in / tokens_out / cost_usd
columns on an already-scored results CSV. No model is called: the prompt is
RECONSTRUCTED with build_prompt() from the stored command+strategy, and the
response is the stored `raw` column. This is why those columns were reserved
empty rather than dropped.

CAVEAT (state this in the methods chapter): token counts use the OpenAI
o200k_base tokenizer for every row. That is exact for gpt-4o-mini and an
APPROXIMATION for the Ollama models, which use their own tokenizers. Cost is
therefore computed for the cloud model ONLY; local rows get cost_usd = 0 and
their token counts should be read as indicative, not billed quantities.

Usage:
    python -m bench.backfill_tokens --in bench/results_grid_scored.csv --out bench/results_grid_tokens.csv
"""
import argparse, csv
from pathlib import Path

from llm.pipeline import load_schema
from llm.prompts import build_prompt
from bench.dataset import load_seen_split, DEFAULT_K

# gpt-4o-mini list price, USD per 1M tokens (verified July 2026 -- re-check before final write-up)
CLOUD_MODEL = "openai:gpt-4o-mini"
USD_PER_M_IN = 0.15
USD_PER_M_OUT = 0.60


def get_encoder():
    import tiktoken
    return tiktoken.get_encoding("o200k_base")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    args = ap.parse_args()

    enc = get_encoder()
    try:
        schema = load_schema()
    except TypeError:
        schema = load_schema("schema/schema.json")
    exemplars, _ = load_seen_split(k=args.k)

    rows = list(csv.DictReader(open(args.inp, encoding="utf-8")))
    if not rows:
        raise SystemExit("no rows")

    prompt_cache = {}   # (command, strategy) -> token count; prompts repeat across repeats
    tot_in = tot_out = 0
    cloud_cost = 0.0

    for r in rows:
        key = (r["command"], r["strategy"])
        if key not in prompt_cache:
            examples = exemplars if r["strategy"] == "few_shot" else None
            prompt = build_prompt(r["command"], r["strategy"], schema, examples=examples)
            prompt_cache[key] = len(enc.encode(prompt))
        t_in = prompt_cache[key]
        t_out = len(enc.encode(r["raw"])) if r["raw"] else 0

        r["tokens_in"] = t_in
        r["tokens_out"] = t_out
        if r["model"] == CLOUD_MODEL:
            cost = t_in / 1e6 * USD_PER_M_IN + t_out / 1e6 * USD_PER_M_OUT
            r["cost_usd"] = f"{cost:.8f}"
            cloud_cost += cost
        else:
            r["cost_usd"] = "0"
        tot_in += t_in; tot_out += t_out

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print(f"wrote {args.out} | {len(rows)} rows")
    print(f"total tokens: in={tot_in:,}  out={tot_out:,}")
    print(f"cloud (gpt-4o-mini) cost for this file: ${cloud_cost:.4f}")
    print("NOTE: local-model token counts use the OpenAI tokenizer and are approximate.")


if __name__ == "__main__":
    main()
