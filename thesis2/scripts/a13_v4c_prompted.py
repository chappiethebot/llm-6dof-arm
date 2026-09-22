"""thesis2/scripts/a13_v4c_prompted.py -- A13: the two arms that read the V5
legend, evaluated on held_out_v4c.

WHY. held_out_v4c is the back/front vocabulary probe: a third row wording that
appears in neither train_v2 nor held_out_v3. Everything the project has said
about vocabulary generalisation for PROMPTED arms so far came from v3, where
the legend does define the row words in play. This runs the two arms that
actually consume the V5 legend text on v4c:

  qwen2.5-3b-lora, zero_shot, V5 legend   -- transformers fp16, adapter merged,
                                             greedy, no prefix cache (the plain
                                             baseline A8/A10 reproduce exactly)
  flat DistilBERT classifier              -- reads no legend at all; included as
                                             the no-prompt control

ACCURACY AND ROW ACCURACY ONLY. Latency, energy and acceptance are not measured
here and are not reported.

Scoring is llm/score.py strict_match, the same grader as every other arm. Row
accuracy is over the 108 grid-cell commands; a prediction that is not a
grid_<ROW><COL> symbol counts as a row miss and is also counted separately.

Run (conda env thesis2gpu):  python thesis2/scripts/a13_v4c_prompted.py
"""
import csv
import json
import os
import re
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "thesis2" / "scripts"))
OUT = REPO / "thesis2" / "out"
csv.field_size_limit(2 ** 31 - 1)

from llm.score import score                                          # noqa: E402

BASE = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER = REPO / "ft" / "runs" / "qwen3b_lora" / "checkpoint-87"
SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
MAX_NEW = 64
ROWS = ["A", "B", "C"]


def load(name):
    return [json.loads(l) for l in
            open(REPO / "data" / ("%s.jsonl" % name), encoding="utf-8") if l.strip()]


def rowof(loc):
    return loc[5] if isinstance(loc, str) and loc.startswith("grid_") and len(loc) == 7 else None


print("=" * 100)
print("A13 -- V5-legend prompted arm and the flat classifier on held_out_v4c")
print("=" * 100)

rows_v4c = load("held_out_v4c")
rows_v3 = load("held_out_v3")
print("  held_out_v4c n=%d   held_out_v3 n=%d" % (len(rows_v4c), len(rows_v3)))

# ---------------------------------------------------------------- vocabulary
ROWWORDS = ["upper", "lower", "top", "bottom", "far", "near", "farthest",
            "nearest", "back", "front", "middle", "center"]
print()
print("  row-bearing words in the 108 grid commands of each set:")
for nm, rr in (("held_out_v3", rows_v3), ("held_out_v4c", rows_v4c)):
    cnt = {}
    for r in rr:
        if not (r["expected"].get("location") or "").startswith("grid_"):
            continue
        for w in ROWWORDS:
            if re.search(r"\b%s\b" % w, r["command"].lower()):
                cnt[w] = cnt.get(w, 0) + 1
    print("    %-13s %s" % (nm, dict(sorted(cnt.items(), key=lambda x: -x[1]))))

# ---------------------------------------------------------- the legend text
from bench.run_cell import apply_legend                              # noqa: E402
from llm import prompts as P                                         # noqa: E402

note = apply_legend("v5")
schema = P.load_schema(str(REPO / "schema" / "schema_v2.json"))
legend = P.build_prompt("X", "zero_shot", schema)
print()
print("  legend in force: %s" % note)
print("  which v4c row words does the V5 legend DEFINE?")
for w in ("back", "front", "upper", "bottom", "top", "lower", "far", "near"):
    n = len(re.findall(r"\b%s\b" % w, legend.lower()))
    print("     %-8s occurrences in the prompt: %d%s"
          % (w, n, "   <-- NOT DEFINED" if n == 0 else ""))

# ------------------------------------------------------------- flat classifier
print()
print("  running the flat DistilBERT classifier on held_out_v4c (reads no legend)")
from llm.classifier_client import DistilBertClassifierClient           # noqa: E402

clf = DistilBertClassifierClient()
clf.complete("warm up")
flat = []
for r in rows_v4c:
    obj = json.loads(clf.complete(r["command"]))
    truth = dict(r["expected"]); truth.setdefault("modifiers", [])
    flat.append(dict(pred=obj, truth=truth, command=r["command"],
                     ok=bool(score(obj, truth).strict_match)))

# Cross-check against the saved p45 v4c predictions. NOTE: compare only on rows
# whose predicted verb REQUIRES a location -- emit() discards the location for
# verbs that do not take one, while p45's flat_location column is the raw
# location argmax regardless of verb. Comparing all 217 rows measures that
# difference, not model disagreement (it shows 46 spurious "mismatches").
try:
    p45 = list(csv.DictReader(open(OUT / "p45_predictions_held_out_v4c.csv",
                                   encoding="utf-8", newline="")))
    cmp_rows = [(f, p) for f, p in zip(flat, p45) if f["pred"].get("location")]
    agree = sum(1 for f, p in cmp_rows if f["pred"]["location"] == p["flat_location"])
    print("     cross-check vs saved p45_predictions_held_out_v4c.csv flat_location: "
          "%d/%d rows that carry a location (the other %d have a verb that takes none)"
          % (agree, len(cmp_rows), len(flat) - len(cmp_rows)))
except Exception as e:                                               # noqa: BLE001
    print("     (cross-check skipped: %s)" % e)

# ------------------------------------------------------------------- the LoRA
print()
print("  loading %s + %s (fp16, merged) -- greedy, V5 legend, no prefix cache"
      % (BASE, ADAPTER.name))
from transformers import AutoTokenizer, AutoModelForCausalLM            # noqa: E402
from peft import PeftModel                                              # noqa: E402

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16).to("cuda")
model = PeftModel.from_pretrained(model, str(ADAPTER)).merge_and_unload().eval()
IM_END = tok.convert_tokens_to_ids("<|im_end|>")
STOP = {IM_END, tok.eos_token_id}


def chat_wrap(user):
    return ("<|im_start|>system\n%s<|im_end|>\n"
            "<|im_start|>user\n%s<|im_end|>\n"
            "<|im_start|>assistant\n" % (SYSTEM, user))


@torch.no_grad()
def greedy(pids):
    out = model(input_ids=torch.tensor([pids], device="cuda"), use_cache=True)
    past = out.past_key_values
    nxt = int(out.logits[0, -1].argmax())
    gen = []
    while len(gen) < MAX_NEW and nxt not in STOP:
        gen.append(nxt)
        out = model(input_ids=torch.tensor([[nxt]], device="cuda"),
                    past_key_values=past, use_cache=True)
        past = out.past_key_values
        nxt = int(out.logits[0, -1].argmax())
    return tok.decode(gen, skip_special_tokens=True)


lora = []
for i, r in enumerate(rows_v4c):
    ptxt = chat_wrap(P.build_prompt(r["command"], "zero_shot", schema))
    txt = greedy(tok(ptxt, add_special_tokens=False)["input_ids"])
    try:
        obj = json.loads(txt)
        parsed = True
    except Exception:                                                # noqa: BLE001
        obj, parsed = {}, False
    truth = dict(r["expected"]); truth.setdefault("modifiers", [])
    lora.append(dict(pred=obj, raw=txt, parsed=parsed, truth=truth,
                     command=r["command"], ok=bool(score(obj, truth).strict_match)))
    if (i + 1) % 50 == 0:
        print("     %3d / %d" % (i + 1, len(rows_v4c)))

# --------------------------------------------------------------------- report
def summarise(name, recs):
    n = len(recs)
    acc = sum(r["ok"] for r in recs) / n
    grid = [r for r in recs if rowof(r["truth"].get("location"))]
    hit = nongrid = 0
    for r in grid:
        pr = rowof(r["pred"].get("location"))
        if pr is None:
            nongrid += 1
        elif pr == rowof(r["truth"]["location"]):
            hit += 1
    return dict(arm=name, n=n, action_match=round(acc, 4), n_row=len(grid),
                row_acc=round(hit / len(grid), 4), row_hits=hit,
                pred_not_a_grid_cell=nongrid,
                unparseable=sum(1 for r in recs if not r.get("parsed", True)))


res = [summarise("qwen2.5-3b-lora zero_shot (V5 legend, transformers fp16)", lora),
       summarise("flat DistilBERT classifier (no legend)", flat)]
print()
print("=" * 100)
print("held_out_v4c -- ACCURACY AND ROW ACCURACY ONLY")
print("=" * 100)
print("  %-56s %6s %10s %6s %10s" % ("arm", "n", "action", "n_row", "row acc"))
for s in res:
    print("  %-56s %6d %10.4f %6d %10.4f" % (s["arm"], s["n"], s["action_match"],
                                             s["n_row"], s["row_acc"]))
print()
for s in res:
    print("  %-56s rows predicted as a non-grid symbol: %d   unparseable output: %d"
          % (s["arm"][:56], s["pred_not_a_grid_cell"], s["unparseable"]))

# row confusion for the LoRA
conf = {}
for r in lora:
    t = rowof(r["truth"].get("location"))
    if not t:
        continue
    p_ = rowof(r["pred"].get("location")) or "-"
    conf[(t, p_)] = conf.get((t, p_), 0) + 1
print()
print("  LoRA row confusion on v4c (truth -> predicted): %s"
      % dict(sorted(conf.items())))

with open(OUT / "a13_v4c_prompted.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(res[0].keys()))
    w.writeheader(); w.writerows(res)
with open(OUT / "a13_v4c_lora_percommand.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["command_id", "command", "raw", "pred_json", "truth_json",
                "strict_match", "truth_row", "pred_row"])
    for i, r in enumerate(lora):
        w.writerow([i, r["command"], r["raw"], json.dumps(r["pred"]),
                    json.dumps(r["truth"]), r["ok"],
                    rowof(r["truth"].get("location")) or "",
                    rowof(r["pred"].get("location")) or ""])
print()
print("wrote thesis2/out/a13_v4c_prompted.csv and a13_v4c_lora_percommand.csv")
print("=" * 100)
