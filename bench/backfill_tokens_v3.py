#!/usr/bin/env python
r"""
bench/backfill_tokens_v3.py -- token/cost backfill for the held_out_v3 files,
using EACH MODEL'S OWN VOCABULARY.

WHY A SEPARATE SCRIPT. bench/backfill_tokens.py is V1-shaped and produces
SILENTLY WRONG numbers on v3 rows. Four defects, all confirmed by audit:
  1. (line 45-47) schema defaults to schema/schema.json; V3 needs schema_v2.
  2. (line 48) exemplars default to data/seen.jsonl, not seen_v2.jsonl.
  3. (line 61) examples passed only for few_shot, so rag prompts count as
     having zero retrieved examples.
  4. (line 65) tokens_out = encode(raw) only; `thinking` excluded entirely.
The original is left untouched: it is the provenance of the V1 token files.

TOKENIZER POLICY -- ONE VOCABULARY PER MODEL, NEVER ONE FOR ALL
----------------------------------------------------------------
Counting Qwen or Llama output with OpenAI's o200k_base is wrong: the vocabularies
differ, and the error is not a constant factor. Measured during the audit,
o200k_base gives ~2.0 chars/token on dense JSON and ~5.1 on prose reasoning, so
a single tokenizer distorts the visible/hidden ratio in a direction that depends
on what the model wrote.

  openai:gpt-4o-mini    tiktoken o200k_base                      EXACT
  ollama:qwen2.5:*      Qwen2.5 vocab, LOCAL, read from
                        ft/runs/qwen0.5b_lora/checkpoint-87/      EXACT
  ollama:*-lora         same Qwen2.5 vocab (merged adapters do
                        not change the vocabulary)                EXACT
  ollama:deepseek-r1:7b deepseek-ai/DeepSeek-R1-Distill-Qwen-7B   EXACT
  ollama:llama3.2:3b    Llama-3.2-3B-Instruct vocab               EXACT*
  classifier:distilbert DistilBERT vocab for the INPUT; output is
                        CONSTRUCTED from the schema, not generated

  * meta-llama/Llama-3.2-3B-Instruct is gated on the Hub; an ungated mirror of
    the same tokenizer is used. Vocabulary size 128,256 matches the published
    Llama 3.2 vocabulary. [Likely] identical to Meta's; not cryptographically
    verified against it.

NOT llama-tokenize: the task specified llama.cpp's llama-tokenize.exe against
the GGUF blobs in D:\LLM\blobs. That binary is present at
C:\Users\USER\llama.cpp\bin but is BLOCKED BY WINDOWS APPLICATION CONTROL and
will not execute. The HF tokenizers above are the same vocabularies reached by
a different route; no attempt was made to bypass the policy.

THREE OUTPUT COLUMNS, NOT ONE
-----------------------------
  tokens_out_visible   what the caller receives
  tokens_thinking      reasoning trace; 0 for non-reasoning models
  tokens_out_total     visible + thinking -- what the model actually produced
The GAP between visible and total is itself a finding about reasoning models.
Legacy `tokens_out` is filled with the VISIBLE count so bench/summary.py keeps
working; read tokens_out_total for any cost or throughput claim.
`token_source` records which vocabulary produced each row's counts.

WRITES NEW FILES. The _scored inputs are opened read-only and never modified.

Usage:
    python -m bench.backfill_tokens_v3
    python -m bench.backfill_tokens_v3 --check-only
"""
import argparse, csv, json, os, sys
from pathlib import Path

from llm.prompts import build_prompt, load_schema
from bench.dataset import load_seen_split

csv.field_size_limit(2**31 - 1)

ROOT = Path(__file__).resolve().parent.parent
FILES = ["rung1", "rag", "lora", "cls"]
IN_TMPL = "bench/results_heldout_v3_{}_scored.csv"
OUT_TMPL = "bench/results_heldout_v3_{}_tokens.csv"

SCHEMA_PATH = "schema/schema_v2.json"      # fix 1
EXEMPLAR_SRC = "data/seen_v2.jsonl"        # fix 2
RAG_CORPUS = "data/train_v2.jsonl"         # fix 3

QWEN_VOCAB = ROOT / "ft" / "runs" / "qwen0.5b_lora" / "checkpoint-87"
DISTILBERT_VOCAB = ROOT / "ft" / "runs" / "distilbert_cls" / "epoch_09"
DEEPSEEK_REPO = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
LLAMA_REPO = "unsloth/Llama-3.2-3B-Instruct"

CLOUD_MODEL = "openai:gpt-4o-mini"
USD_PER_M_IN = 0.15
USD_PER_M_OUT = 0.60

FROZEN = ("stage", "ok", "strict_match", "normalized_match", "category",
          "latency_s", "raw", "thinking", "pred_json", "truth_json",
          "command_id", "run_index", "model", "strategy", "dataset")

NEW_COLS = ("tokens_out_visible", "tokens_thinking", "tokens_out_total",
            "token_source")


class Tokenizers:
    """Lazy per-model vocabularies. Each is loaded at most once."""

    def __init__(self):
        self._cache = {}
        self._counts = {}

    def _hf(self, key, src, label):
        if key not in self._cache:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(src)
            self._cache[key] = (lambda s: len(tok(s)["input_ids"]) if s else 0, label)
            print(f"  vocab loaded: {label:34} size={len(tok):,}")
        return self._cache[key]

    def _tiktoken(self):
        if "openai" not in self._cache:
            import tiktoken
            enc = tiktoken.get_encoding("o200k_base")
            self._cache["openai"] = (lambda s: len(enc.encode(s)) if s else 0,
                                     "tiktoken_o200k_base")
            print(f"  vocab loaded: {'tiktoken o200k_base':34} size={enc.n_vocab:,}")
        return self._cache["openai"]

    def for_model(self, model):
        if model.startswith("openai:"):
            return self._tiktoken()
        if model.startswith("classifier:"):
            return self._hf("distilbert", str(DISTILBERT_VOCAB),
                            "distilbert_vocab_local")
        # Match on the model NAME only, never the whole string: the provider
        # prefix "ollama:" itself contains the substring "llama", so a naive
        # `"llama" in model` routes every Qwen model to the Llama vocabulary.
        # That bug was live in this file and silently mis-tokenised four of the
        # six local models before audit/check_dispatch.py caught it.
        name = model.split(":", 1)[1] if ":" in model else model
        if "deepseek" in name:
            return self._hf("deepseek", DEEPSEEK_REPO,
                            "deepseek_r1_distill_qwen_vocab")
        if "qwen" in name:
            return self._hf("qwen", str(QWEN_VOCAB), "qwen2.5_vocab_local")
        if "llama" in name:
            return self._hf("llama", LLAMA_REPO, "llama3.2_vocab_mirror")
        raise ValueError(f"no vocabulary registered for model {model!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()

    toks = Tokenizers()
    schema = load_schema(SCHEMA_PATH)
    exemplars, _ = load_seen_split(k=3, seen_path=EXEMPLAR_SRC)
    train = [json.loads(l) for l in open(RAG_CORPUS, encoding="utf-8") if l.strip()]
    print(f"schema: {SCHEMA_PATH} | exemplars: {len(exemplars)} | rag corpus: {len(train)}")

    def ex_of(row):
        return {"command": row.get("command", ""), "expected": row.get("expected", {})}

    grand = {"in": 0, "visible": 0, "thinking": 0, "cost": 0.0}
    drift_total = 0
    str_cache = {}

    for tag in FILES:
        inp, out = IN_TMPL.format(tag), OUT_TMPL.format(tag)
        with open(inp, encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            print(f"  {tag}: no rows"); continue

        before = [{k: r.get(k) for k in FROZEN} for r in rows]
        f_in = f_vis = f_think = 0
        cost = 0.0

        for r in rows:
            model, strat = r["model"], r["strategy"]
            encode, source = toks.for_model(model)

            def enc(s):
                if not s:
                    return 0
                key = (source, s)
                if key not in str_cache:
                    str_cache[key] = encode(s)
                return str_cache[key]

            if strat == "rag" and r.get("retrieved_ids"):
                pkey = (source, r["command"], strat, r["retrieved_ids"])
            else:
                pkey = (source, r["command"], strat)
            if pkey not in str_cache:
                if strat == "few_shot":
                    ex = exemplars
                elif strat == "rag" and r.get("retrieved_ids"):
                    ids = json.loads(r["retrieved_ids"])
                    ex = [ex_of(train[i]) for i in ids if i < len(train)]
                else:
                    ex = None
                prompt = (r["command"] if strat == "classifier"
                          else build_prompt(r["command"], strat, schema, examples=ex))
                str_cache[pkey] = encode(prompt)
            t_in = str_cache[pkey]

            t_vis = enc(r.get("raw") or "")
            t_think = enc(r.get("thinking") or "")       # fix 4
            t_tot = t_vis + t_think

            r["tokens_in"] = t_in
            r["tokens_out"] = t_vis
            r["tokens_out_visible"] = t_vis
            r["tokens_thinking"] = t_think
            r["tokens_out_total"] = t_tot
            # the classifier does not GENERATE its output; it is constructed
            # from schema_v2.json's required-field list for the predicted verb.
            r["token_source"] = (f"{source}|output_constructed_not_generated"
                                 if model.startswith("classifier:") else source)

            if model == CLOUD_MODEL:
                c = t_in / 1e6 * USD_PER_M_IN + t_tot / 1e6 * USD_PER_M_OUT
                r["cost_usd"] = f"{c:.8f}"
                cost += c
            else:
                r["cost_usd"] = "0"

            f_in += t_in; f_vis += t_vis; f_think += t_think

        drift = 0
        for b, r in zip(before, rows):
            for k in FROZEN:
                if b[k] != r.get(k):
                    drift += 1
                    if drift <= 3:
                        print(f"  !! DRIFT {tag} col={k} before={b[k]!r} after={r.get(k)!r}")
        drift_total += drift
        status = "OK (0 frozen-column changes)" if drift == 0 else f"DRIFT {drift}"

        print(f"  {tag:6} n={len(rows):6} in={f_in:9,} visible={f_vis:9,} "
              f"thinking={f_think:11,} total={f_vis+f_think:11,} "
              f"cost=${cost:.4f}  [{status}]")

        grand["in"] += f_in; grand["visible"] += f_vis
        grand["thinking"] += f_think; grand["cost"] += cost

        if drift:
            print(f"  ABORT: {tag} frozen columns changed. Nothing written.", file=sys.stderr)
            return 1
        if not args.check_only:
            fieldnames = list(rows[0].keys())
            for c in NEW_COLS:
                if c not in fieldnames:
                    fieldnames.append(c)
            with open(out, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader(); w.writerows(rows)
            print(f"         -> {out}")

    tot = grand["visible"] + grand["thinking"]
    print(f"\nGRAND TOTAL  prompt={grand['in']:,}  visible={grand['visible']:,}  "
          f"thinking={grand['thinking']:,}  total_out={tot:,}")
    if grand["visible"]:
        print(f"study-wide hidden:visible = {grand['thinking']/grand['visible']:.2f}:1")
    print(f"cloud cost (gpt-4o-mini, billed on tokens_out_TOTAL): ${grand['cost']:.4f}")
    print(f"frozen-column drift across all files: {drift_total}")
    print(f"distinct strings tokenised: {len(str_cache):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
