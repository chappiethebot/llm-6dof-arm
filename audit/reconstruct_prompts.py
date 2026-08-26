"""Recover tokens_in without re-running any model.

The token columns are 0% populated on every held_out_v3 file, so context
pressure cannot be read off the CSVs. But the prompt is a DETERMINISTIC
function of (command, strategy, schema, examples):
  - zero_shot / cot / role_based : no examples
  - few_shot                     : the k fixed exemplars from bench/dataset.py
  - rag                          : the retrieved rows, and `retrieved_ids` is
                                   stored on all 5,420 rag rows
So every prompt is reconstructible offline. No model call, no re-run, the
sealed set is only READ.

Tokenised with the real tokenizer where one is cached locally; otherwise a
chars/4 bound is reported and labelled as such.
"""
import csv, json, collections
from pathlib import Path

csv.field_size_limit(2**31 - 1)

from llm.prompts import build_prompt, load_schema
from bench.dataset import load_seen_split

SCHEMA = load_schema("schema/schema_v2.json")
ROWS = [json.loads(l) for l in open("data/held_out_v3.jsonl", encoding="utf-8") if l.strip()]
TRAIN = [json.loads(l) for l in open("data/train_v2.jsonl", encoding="utf-8") if l.strip()]
EXEMPLARS, _ = load_seen_split(k=3, seen_path="data/seen_v2.jsonl")

print(f"held_out_v3 rows : {len(ROWS)}")
print(f"train_v2 rows    : {len(TRAIN)}")
print(f"fixed exemplars  : {len(EXEMPLARS)}")


def cmd_of(row):
    return row.get("command") or row.get("cmd") or ""


def ex_of(row):
    return {"command": cmd_of(row), "expected": row.get("expected", {})}


# --- tokenizer: try real ones, fall back to a chars/4 bound -------------------
TOK = {}
try:
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoTokenizer
    for label, repo in [("qwen", "Qwen/Qwen2.5-0.5B-Instruct"),
                        ("llama", "meta-llama/Llama-3.2-3B-Instruct")]:
        try:
            TOK[label] = AutoTokenizer.from_pretrained(repo)
            print(f"tokenizer OK     : {label} ({repo})")
        except Exception as e:
            print(f"tokenizer MISS   : {label} -- {type(e).__name__}")
except Exception as e:
    print(f"transformers unavailable: {e}")

try:
    import tiktoken
    TOK["openai"] = tiktoken.encoding_for_model("gpt-4o-mini")
    print("tokenizer OK     : openai (tiktoken)")
except Exception as e:
    print(f"tokenizer MISS   : openai -- {type(e).__name__}")


def ntok(text, label):
    t = TOK.get(label)
    if t is None:
        return None
    try:
        return len(t.encode(text)) if label == "openai" else len(t(text)["input_ids"])
    except Exception:
        return None


# --- retrieved ids per (command_id) for rag ----------------------------------
rag_ids = {}
for f in ["rag"]:
    for r in csv.DictReader(open(f"bench/results_heldout_v3_{f}_scored.csv",
                                 encoding="utf-8", newline="")):
        cid = int(r["command_id"])
        if r.get("retrieved_ids"):
            rag_ids.setdefault(cid, json.loads(r["retrieved_ids"]))

print(f"rag command_ids with stored retrieved_ids: {len(rag_ids)}")
print()

STRATS = ["zero_shot", "few_shot", "cot", "role_based", "rag"]
print("=" * 96)
print("RECONSTRUCTED PROMPT SIZE PER STRATEGY  (over all 217 held_out_v3 commands)")
print("=" * 96)
print(f"{'strategy':12}{'max_chars':>11}{'mean_chars':>12}"
      f"{'~tok(ch/4)':>12}{'qwen_tok':>10}{'llama_tok':>11}{'oai_tok':>9}")

results = {}
for s in STRATS:
    lens, qt, lt, ot = [], [], [], []
    for cid, row in enumerate(ROWS):
        cmd = cmd_of(row)
        if s == "few_shot":
            ex = [ex_of(e) if not isinstance(e, dict) or "expected" not in e else e
                  for e in EXEMPLARS]
        elif s == "rag":
            ids = rag_ids.get(cid)
            if ids is None:
                continue
            ex = [ex_of(TRAIN[i]) for i in ids if i < len(TRAIN)]
        else:
            ex = None
        p = build_prompt(cmd, s, SCHEMA, examples=ex)
        lens.append(len(p))
        for lbl, acc in (("qwen", qt), ("llama", lt), ("openai", ot)):
            v = ntok(p, lbl)
            if v is not None:
                acc.append(v)
    mx = max(lens)
    results[s] = dict(max_chars=mx, max_qwen=max(qt) if qt else None,
                      max_llama=max(lt) if lt else None,
                      max_oai=max(ot) if ot else None)
    f = lambda v: f"{max(v):>10,}" if v else f"{'n/a':>10}"
    print(f"{s:12}{mx:>11,}{sum(lens)/len(lens):>12,.0f}{mx/4:>12,.0f}"
          f"{(max(qt) if qt else 0) or 'n/a':>10}{(max(lt) if lt else 0) or 'n/a':>11}"
          f"{(max(ot) if ot else 0) or 'n/a':>9}")

print()
print("=" * 96)
print("VERDICT: worst-case TOTAL context per cell = max prompt + max generated")
print("=" * 96)
d = {}
for f in ["rung1", "rag", "lora", "cls"]:
    for r in csv.DictReader(open(f"bench/results_heldout_v3_{f}_scored.csv",
                                 encoding="utf-8", newline="")):
        d[(r["model"], r["strategy"], r["command_id"], r["run_index"])] = r
cells = collections.defaultdict(list)
for r in d.values():
    if r["is_warmup"] not in ("True", "true", "1"):
        cells[(r["model"], r["strategy"])].append(r)

print(f"{'model':26}{'strategy':11}{'prompt_tok':>11}{'gen_tok_max':>12}"
      f"{'TOTAL':>9}{'vs 2048':>10}")
for (m, s), c in sorted(cells.items()):
    if s == "classifier":
        continue
    lbl = "openai" if m.startswith("openai") else ("llama" if "llama" in m else "qwen")
    r_ = results.get(s, {})
    ptok = r_.get(f"max_{'oai' if lbl=='openai' else lbl}") or (r_.get("max_chars", 0) / 4)
    gen = max(len(x.get("raw") or "") + len(x.get("thinking") or "") for x in c) / 4
    tot = ptok + gen
    flag = "OVER" if tot > 2048 else "ok"
    print(f"{m:26}{s:11}{ptok:>11,.0f}{gen:>12,.0f}{tot:>9,.0f}{flag:>10}")
