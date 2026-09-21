"""thesis2/scripts/p45_collect.py -- P4.5 / P6.1 / P6.2: collect every prediction once.

For each command in a dataset, records:

  local factorized head   verb, target, loc_type, row, col, container_which,
                          runtime_which -- each with its softmax confidence
  local flat head         the 14-way location argmax and its confidence
                          (the rival router signal, for the Idea 1 comparison)
  SALSA row, qwen base    3-way row logits, letters AND digits mappings
  SALSA row, qwen+LoRA    same
  gpt-4o-mini row         one-character row answer

Escalation targets are asked ONLY the 3-way ROW sub-question. That is the whole
point of factorization: the sub-query is 3 logits from one forward pass, not a
14-way decision and not a generated JSON object.

Models are loaded and freed SEQUENTIALLY so peak VRAM stays near one model.

Writes a per-command cache so every threshold sweep afterwards is offline and
costs nothing. gpt-4o-mini is called at most once per command per dataset.
"""
import argparse
import csv
import gc
import json
import os
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "thesis2" / "scripts"))

from p2_train_factorized import (FactorizedClassifier, FACTORS,        # noqa: E402
                                 build_label_space, factorize, IGNORE,
                                 MAX_LEN, ROWS)
from ft.train_classifier import (MultiHeadClassifier,                  # noqa: E402
                                 build_label_space as flat_space)
from salsa_row import SalsaRowScorer, MAPPINGS, build_prompt           # noqa: E402
from transformers import AutoTokenizer                                 # noqa: E402

FACT_CK = REPO / "thesis2" / "models" / "distilbert_factorized" / "epoch_11" / "model.pt"
FLAT_CK = REPO / "ft" / "runs" / "distilbert_cls" / "epoch_09" / "model.pt"
LORA_CK = REPO / "ft" / "runs" / "qwen3b_lora" / "checkpoint-87"
OUT = REPO / "thesis2" / "out"

UPDOWN = ("upper", "bottom", "top", "lower")
FARNEAR = ("far", "near", "farthest", "nearest")
BACKFRONT = ("back", "front")


def bucket(cmd, loc):
    if loc is None:
        return "nolocation"
    low = " %s " % " ".join(cmd.lower().replace(",", " ").split())
    hits = [n for n, ws in (("farnear", FARNEAR), ("updown", UPDOWN),
                            ("backfront", BACKFRONT))
            if any(" %s " % w in low for w in ws)]
    if len(hits) == 1:
        return hits[0]
    return "mixed" if hits else "neither"


def softmax_top(logits):
    p = torch.softmax(logits, dim=-1)[0]
    v, i = torch.sort(p, descending=True)
    return float(v[0]), int(i[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="held_out_v3")
    ap.add_argument("--no-cloud", action="store_true",
                    help="skip gpt-4o-mini (costs money / needs network)")
    args = ap.parse_args()

    path = REPO / "data" / ("%s.jsonl" % args.dataset)
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    space = build_label_space()
    fsp = flat_space()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 76)
    print("P4.5 COLLECT -- %s, n=%d commands, device=%s" % (args.dataset, len(rows), device))
    print("=" * 76)

    recs = []
    for ci, r in enumerate(rows):
        loc = r["expected"].get("location")
        tf = dict(zip(FACTORS, factorize(loc)))
        recs.append(dict(command_id=ci, command=r["command"],
                         truth_json=json.dumps(r["expected"]),
                         truth_location=loc if loc else "",
                         truth_row=("" if tf["row"] == IGNORE else ROWS[tf["row"]]),
                         bucket=bucket(r["command"], loc)))

    # ------------------------------------------------- 1. local DistilBERTs
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    fact = FactorizedClassifier(space)
    fact.load_state_dict(torch.load(FACT_CK, map_location="cpu"))
    fact.eval()
    flat = MultiHeadClassifier(fsp)
    flat.load_state_dict(torch.load(FLAT_CK, map_location="cpu"))
    flat.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        for rec, r in zip(recs, rows):
            enc = tok(r["command"], truncation=True, max_length=MAX_LEN,
                      padding="max_length", return_tensors="pt")
            lg = fact(enc["input_ids"], enc["attention_mask"])
            for k in ["verb", "target"] + FACTORS:
                c, i = softmax_top(lg[k])
                rec["fact_%s_idx" % k] = i
                rec["fact_%s_conf" % k] = round(c, 6)
            rec["fact_row"] = ROWS[rec["fact_row_idx"]]
            vl, tl, ll = flat(enc["input_ids"], enc["attention_mask"])
            c, i = softmax_top(ll)
            sym = fsp["locations"][i]
            rec["flat_location"] = sym
            rec["flat_location_conf"] = round(c, 6)
            rec["flat_row"] = (sym[5] if sym.startswith("grid_") and len(sym) == 7
                               else "")
            rec["flat_verb_idx"] = int(vl.argmax(-1).item())
            rec["flat_target_idx"] = int(tl.argmax(-1).item())
    print("  DistilBERT (factorized + flat) done in %.1fs" % (time.perf_counter() - t0))
    del fact, flat
    gc.collect()

    # ------------------------------------------------------ 2. SALSA arms
    for tag, base, adapter in (("qwen3b", "Qwen/Qwen2.5-3B-Instruct", None),
                               ("qwen3blora", "Qwen/Qwen2.5-3B-Instruct", str(LORA_CK))):
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        s = SalsaRowScorer(base=base, adapter=adapter)
        load_s = time.perf_counter() - t0
        for mp in MAPPINGS:
            lat = []
            t0 = time.perf_counter()
            for rec in recs:
                pred, probs, ms = s.score(rec["command"], mp)
                rec["salsa_%s_%s_row" % (tag, mp)] = pred
                rec["salsa_%s_%s_conf" % (tag, mp)] = round(max(probs), 6)
                rec["salsa_%s_%s_probs" % (tag, mp)] = json.dumps(
                    [round(p, 5) for p in probs])
                lat.append(ms)
            lat.sort()
            print("  SALSA %-11s mapping=%-8s n=%d  p50 %.1f ms  p95 %.1f ms  "
                  "load %.1fs  peakVRAM %.0f MB"
                  % (tag, mp, len(lat), lat[len(lat) // 2],
                     lat[int(0.95 * (len(lat) - 1))], load_s, s.memory_mb()))
        # generation baseline for P6.3, letters only, first 40 commands
        gl = []
        for rec in recs[:40]:
            _p, _t, ms = s.generate_baseline(rec["command"], "letters")
            gl.append(ms)
        gl.sort()
        print("     generation baseline (same prompt, 8 new tokens, n=40): "
              "p50 %.1f ms  p95 %.1f ms" % (gl[len(gl) // 2], gl[int(0.95 * (len(gl) - 1))]))
        (OUT / ("p6_latency_%s_%s.json" % (tag, args.dataset))).write_text(
            json.dumps({"salsa_p50_ms": lat[len(lat) // 2],
                        "salsa_p95_ms": lat[int(0.95 * (len(lat) - 1))],
                        "generate_p50_ms": gl[len(gl) // 2],
                        "generate_p95_ms": gl[int(0.95 * (len(gl) - 1))],
                        "generate_n": len(gl), "load_s": round(load_s, 2),
                        "peak_vram_mb": round(s.memory_mb(), 1)}, indent=2),
            encoding="utf-8")
        del s
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    # --------------------------------------------------- 3. gpt-4o-mini row
    if not args.no_cloud:
        try:
            from dotenv import load_dotenv
            load_dotenv(REPO / ".env")
            from openai import OpenAI
            client = OpenAI()
            lat = []
            n_err = 0
            for rec in recs:
                prompt = build_prompt(rec["command"], "letters")
                t0 = time.perf_counter()
                try:
                    resp = client.chat.completions.create(
                        model="gpt-4o-mini", temperature=0.0, max_tokens=2,
                        messages=[{"role": "user", "content": prompt}])
                    txt = (resp.choices[0].message.content or "").strip().upper()
                except Exception as e:
                    n_err += 1
                    txt = ""
                lat.append((time.perf_counter() - t0) * 1000)
                rec["cloud_row"] = txt[:1] if txt[:1] in ROWS else ""
            lat.sort()
            print("  gpt-4o-mini row  n=%d  p50 %.0f ms  p95 %.0f ms  errors %d"
                  % (len(lat), lat[len(lat) // 2], lat[int(0.95 * (len(lat) - 1))], n_err))
            (OUT / ("p45_cloud_latency_%s.json" % args.dataset)).write_text(
                json.dumps({"p50_ms": lat[len(lat) // 2],
                            "p95_ms": lat[int(0.95 * (len(lat) - 1))],
                            "n": len(lat), "errors": n_err}, indent=2), encoding="utf-8")
        except Exception as e:
            print("  gpt-4o-mini SKIPPED: %s: %s" % (type(e).__name__, e))
            for rec in recs:
                rec.setdefault("cloud_row", "")
    else:
        for rec in recs:
            rec["cloud_row"] = ""

    cols = sorted({k for r in recs for k in r})
    lead = ["command_id", "command", "bucket", "truth_location", "truth_row"]
    cols = lead + [c for c in cols if c not in lead]
    p = OUT / ("p45_predictions_%s.csv" % args.dataset)
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in recs:
            w.writerow({c: r.get(c, "") for c in cols})
    print("  wrote %s (%d rows, %d cols)" % (p.relative_to(REPO), len(recs), len(cols)))
    print("=" * 76)
    return 0


if __name__ == "__main__":
    sys.exit(main())
