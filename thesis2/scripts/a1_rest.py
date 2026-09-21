"""thesis2/scripts/a1_rest.py -- A1 arms 3, 5, 6, 7.

  --arm salsa05b     P6.5: qwen2.5:0.5b SALSA row scorer (cached weights)
  --arm flant5       A1.5: Flan-T5-base fine-tuned to emit the JSON directly
  --arm ensemble     A1.6: flat classifier + qwen2.5-3b-lora in parallel;
                     agree -> emit, disagree -> escalate to gpt-4o-mini.
                     OFFLINE: pairs predictions already measured in V5.
  --arm field_flmd   A1.7: escalate the WHOLE 14-way location field to
                     gpt-4o-mini, the field-level comparator for P4.5's
                     factor-level row escalation.
"""
import argparse
import collections
import csv
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "thesis2" / "scripts"))

from llm.score import score                                        # noqa: E402
from p2_train_factorized import (build_label_space, emit, factorize,  # noqa: E402
                                 IGNORE, ROWS)

OUT = REPO / "thesis2" / "out"
SPACE = build_label_space()
V, T, L = SPACE["verbs"], SPACE["targets"], SPACE["locations"]
SEED = 20260820
csv.field_size_limit(2 ** 31 - 1)


def load(n):
    return [json.loads(l) for l in open(REPO / "data" / ("%s.jsonl" % n),
                                        encoding="utf-8") if l.strip()]


def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(p * (len(v) - 1))))]


# ============================================================ arm 3 =========
def run_salsa05b():
    from salsa_row import SalsaRowScorer, MAPPINGS
    out = []
    for ds in ("held_out_v3", "held_out_v4c"):
        with open(OUT / ("p45_predictions_%s.csv" % ds), encoding="utf-8",
                  newline="") as fh:
            recs = [r for r in csv.DictReader(fh) if r["truth_row"]]
        s = SalsaRowScorer(base="Qwen/Qwen2.5-0.5B-Instruct")
        for mp in MAPPINGS:
            hit, lat, pairs = 0, [], []
            for r in recs:
                pred, probs, ms = s.score(r["command"], mp)
                lat.append(ms)
                ok = pred == r["truth_row"]
                hit += int(ok)
                pairs.append((max(probs), ok))
            acc = hit / len(recs)
            e = 0.0
            for i in range(10):
                a, b = i / 10, (i + 1) / 10
                sub = [p for p in pairs if a <= p[0] < b or (i == 9 and p[0] >= a)]
                if sub:
                    e += (len(sub) / len(pairs)) * abs(
                        sum(c for c, _ in sub) / len(sub)
                        - sum(1.0 for _, k in sub if k) / len(sub))
            br = sum((c - (1.0 if k else 0.0)) ** 2 for c, k in pairs) / len(pairs)
            print("  qwen2.5-0.5b %-8s %-13s row %.4f  ECE %.4f  Brier %.4f  "
                  "p50 %.1f ms  peakVRAM %.0f MB"
                  % (mp, ds, acc, e, br, q(lat, .5), s.memory_mb()))
            out.append(dict(arm="salsa_qwen05b_%s" % mp, dataset=ds,
                            n_commands=len(recs), row_acc=round(acc, 4),
                            ECE_row=round(e, 4), Brier_row=round(br, 4),
                            latency_p50_ms=round(q(lat, .5), 2),
                            latency_p95_ms=round(q(lat, .95), 2),
                            latency_p99_ms=round(q(lat, .99), 2),
                            resident_mb=round(s.memory_mb(), 1)))
        del s
        torch.cuda.empty_cache()
    return out, "a1_salsa05b.csv"


# ============================================================ arm 5 =========
def run_flant5():
    from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    base = "google/flan-t5-base"
    tok = AutoTokenizer.from_pretrained(base)
    m = AutoModelForSeq2SeqLM.from_pretrained(base).to(dev)
    print("  flan-t5-base parameters: %.2f M"
          % (sum(p.numel() for p in m.parameters()) / 1e6))
    tr, dv = load("train_v2"), load("dev_v2")

    # BRACE-FREE TARGET FORMAT. T5's SentencePiece vocabulary cannot represent
    # "{" or "}" -- both encode to [3, 2] and decode to the empty string
    # (verified in a1_flant5_diag.py). Training it to emit JSON braces asks for
    # characters it has no tokens for, which is what produced 0.0000 schema
    # validity on the first attempt. The object is emitted as key=value pairs
    # and reconstructed afterwards; the schema check is applied to the
    # reconstructed object exactly as for every other arm.
    def tgt(exp):
        parts = ["action=%s" % exp["action"]]
        if exp.get("target"):
            parts.append("target=%s" % exp["target"])
        if exp.get("location"):
            parts.append("location=%s" % exp["location"])
        return " | ".join(parts)

    def parse_kv(txt):
        o = {}
        for piece in txt.split("|"):
            piece = piece.strip()
            if "=" not in piece:
                continue
            k, v = piece.split("=", 1)
            k, v = k.strip(), v.strip()
            if k in ("action", "target", "location") and v:
                o[k] = v
        return o

    PRE = "translate command to action json: "
    Xtr = tok([PRE + r["command"] for r in tr], padding=True, truncation=True,
              max_length=64, return_tensors="pt")
    Ytr = tok([tgt(r["expected"]) for r in tr], padding=True, truncation=True,
              max_length=48, return_tensors="pt")
    labels = Ytr["input_ids"].clone()
    labels[labels == tok.pad_token_id] = -100
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.01)
    n, bs, EP = len(tr), 8, 12
    best, hist = None, []
    for ep in range(1, EP + 1):
        m.train(); perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            loss = m(input_ids=Xtr["input_ids"][idx].to(dev),
                     attention_mask=Xtr["attention_mask"][idx].to(dev),
                     labels=labels[idx].to(dev)).loss
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval(); hit = 0
        with torch.no_grad():
            for r in dv:
                e = tok(PRE + r["command"], return_tensors="pt").to(dev)
                o = m.generate(**e, max_new_tokens=48, do_sample=False)
                txt = tok.decode(o[0], skip_special_tokens=True)
                p = parse_kv(txt)
                t = dict(r["expected"]); t.setdefault("modifiers", [])
                p.setdefault("modifiers", [])
                hit += int(p == t)
        hist.append((ep, hit))
        if best is None or hit > best[1]:
            best = (ep, hit, {k: v.detach().clone() for k, v in m.state_dict().items()})
        print("    epoch %2d dev %2d/%d" % (ep, hit, len(dv)))
    bh = max(h for _e, h in hist)
    chosen = min([(e, h) for e, h in hist if h >= bh - 1])
    print("  dev plateau %d/%d, plateau-earliest epoch %d; LOADED argmax epoch %d"
          % (bh, len(dv), chosen[0], best[0]))   # Part B S-7: argmax is what runs
    m.load_state_dict(best[2]); m.eval()

    out = []
    for ds in ("held_out_v3", "held_out_v4c"):
        rows = load(ds)
        hits = dict(action=0, joint=0, verb=0, target=0, location=0)
        den = dict(action=0, joint=0, verb=0, target=0, location=0)
        valid = 0
        lat, recs = [], []
        rowh = [0, 0]
        for ci, r in enumerate(rows):
            truth = dict(r["expected"]); truth.setdefault("modifiers", [])
            e = tok(PRE + r["command"], return_tensors="pt").to(dev)
            t0 = time.perf_counter()
            with torch.no_grad():
                o = m.generate(**e, max_new_tokens=48, do_sample=False)
            lat.append((time.perf_counter() - t0) * 1000)
            txt = tok.decode(o[0], skip_special_tokens=True)
            obj = parse_kv(txt)
            # schema validity: legal verb, legal fields for that verb
            ok_schema = (obj.get("action") in V
                         and set(k for k in obj if k not in ("modifiers",))
                         <= {"action", "target", "location"}
                         and (obj.get("target") in T or "target" not in obj)
                         and (obj.get("location") in L or "location" not in obj))
            valid += int(ok_schema)
            obj.setdefault("modifiers", [])
            den["action"] += 1
            hits["action"] += int(bool(score(obj, truth).strict_match))
            den["verb"] += 1
            hits["verb"] += int(obj.get("action") == truth.get("action"))
            if truth.get("target"):
                den["target"] += 1
                hits["target"] += int(obj.get("target") == truth.get("target"))
            if truth.get("location"):
                den["location"] += 1
                hits["location"] += int(obj.get("location") == truth.get("location"))
            if truth.get("target") and truth.get("location"):
                den["joint"] += 1
                hits["joint"] += int(all(obj.get(k) == truth.get(k)
                                         for k in ("action", "target", "location")))
            tf = factorize(truth.get("location"))[1]
            if tf != IGNORE:
                rowh[1] += 1
                pl = obj.get("location")
                pf = factorize(pl)[1] if pl in L else IGNORE
                rowh[0] += int(pf == tf)
            recs.append(dict(model="flan_t5_base", strategy="seq2seq", dataset=ds,
                             command_id=ci, command=r["command"], run_index=0,
                             is_warmup=False,
                             stage="validated" if ok_schema else "schema_failed",
                             ok=ok_schema,
                             strict_match=bool(score(obj, truth).strict_match),
                             pred_json=json.dumps(obj),
                             truth_json=json.dumps(truth),
                             latency_s=round(lat[-1] / 1000, 8)))
        with open(OUT / ("a1_flant5_%s.csv" % ds), "w", newline="",
                  encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(recs[0]))
            w.writeheader(); w.writerows(recs)
        rej = [r for r in rows if r["expected"]["action"] == "reject"]
        rejhit = sum(1 for rc in recs
                     if json.loads(rc["pred_json"]).get("action") == "reject"
                     and json.loads(rc["truth_json"])["action"] == "reject")
        s = dict(arm="flan_t5_base", dataset=ds, n_commands=den["action"],
                 action_match=round(hits["action"] / den["action"], 4),
                 joint_correct=round(hits["joint"] / den["joint"], 4),
                 verb_acc=round(hits["verb"] / den["verb"], 4),
                 target_acc=round(hits["target"] / den["target"], 4),
                 location_acc=round(hits["location"] / den["location"], 4),
                 row_acc=round(rowh[0] / rowh[1], 4), n_row=rowh[1],
                 schema_validity=round(valid / den["action"], 4),
                 reject_recall=round(rejhit / len(rej), 4) if rej else "",
                 latency_p50_ms=round(q(lat, .5), 2),
                 latency_p95_ms=round(q(lat, .95), 2),
                 latency_p99_ms=round(q(lat, .99), 2),
                 resident_mb=round(sum(p.numel() for p in m.parameters()) * 4 / 1e6, 1))
        print("  flan_t5_base %-13s action %.4f  loc %.4f  row %.4f  "
              "schema_valid %.4f  p50 %.0f ms"
              % (ds, s["action_match"], s["location_acc"], s["row_acc"],
                 s["schema_validity"], s["latency_p50_ms"]))
        out.append(s)
    return out, "a1_flant5_summary.csv"


# ============================================================ arm 6 =========
def _v5(tag):
    rows = {}
    with open(REPO / ("bench/results_v5_%s_scored.csv" % tag), encoding="utf-8",
              newline="") as fh:
        for r in csv.DictReader(fh):
            if str(r["is_warmup"]).lower() == "true":
                continue
            rows.setdefault((r["model"], r["strategy"]), {})[r["command_id"]] = r
    return rows


def run_ensemble():
    """flat classifier + qwen2.5-3b-lora in parallel; disagree -> gpt-4o-mini.

    OFFLINE. Every prediction already exists in the V5 measurement; nothing is
    re-inferred. The cloud arbiter is gpt-4o-mini/rag, the best measured cloud
    cell.
    """
    cls = _v5("cls")[("classifier:distilbert", "classifier")]
    lora = _v5("lora")[("ollama:qwen2.5-3b-lora", "zero_shot")]
    rag = _v5("rag")[("openai:gpt-4o-mini", "rag")]
    common = sorted(set(cls) & set(lora) & set(rag), key=int)
    print("  commands present in all three measured cells: %d" % len(common))

    def obj(r):
        try:
            return json.loads(r["pred_json"])
        except Exception:
            return {}
    agree = dis = 0
    hit = dis_hit = agree_hit = 0
    recs = []
    for cid in common:
        a, b, c = obj(cls[cid]), obj(lora[cid]), obj(rag[cid])
        truth = json.loads(cls[cid]["truth_json"])
        key = lambda o: (o.get("action"), o.get("target"), o.get("location"))
        if key(a) == key(b):
            agree += 1
            pick, route = a, "agree"
        else:
            dis += 1
            pick, route = c, "escalate_cloud"
        ok = bool(score(pick, truth).strict_match)
        hit += int(ok)
        if route == "agree":
            agree_hit += int(ok)
        else:
            dis_hit += int(ok)
        recs.append(dict(command_id=cid, route=route, strict_match=ok,
                         pred_json=json.dumps(pick),
                         truth_json=json.dumps(truth)))
    with open(OUT / "a1_ensemble_held_out_v3.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(recs[0]))
        w.writeheader(); w.writerows(recs)
    n = len(common)
    base_cls = sum(1 for cid in common
                   if str(cls[cid]["strict_match"]).lower() == "true") / n
    base_lora = sum(1 for cid in common
                    if str(lora[cid]["strict_match"]).lower() == "true") / n
    base_rag = sum(1 for cid in common
                   if str(rag[cid]["strict_match"]).lower() == "true") / n
    print("  components  classifier %.4f | qwen3b-lora %.4f | gpt-4o-mini/rag %.4f"
          % (base_cls, base_lora, base_rag))
    print("  agreement rate %.4f (%d)  disagreement %.4f (%d)"
          % (agree / n, agree, dis / n, dis))
    print("  accuracy when they AGREE     %.4f (%d/%d)"
          % (agree_hit / agree, agree_hit, agree))
    print("  accuracy on ESCALATED rows   %.4f (%d/%d)"
          % (dis_hit / dis, dis_hit, dis))
    print("  ENSEMBLE action match        %.4f" % (hit / n))
    return [dict(arm="ensemble_disagreement", dataset="held_out_v3",
                 n_commands=n, action_match=round(hit / n, 4),
                 escalation_rate=round(dis / n, 4),
                 agree_accuracy=round(agree_hit / agree, 4),
                 escalated_accuracy=round(dis_hit / dis, 4),
                 component_classifier=round(base_cls, 4),
                 component_lora=round(base_lora, 4),
                 component_cloud=round(base_rag, 4),
                 note="offline pairing of measured V5 predictions; "
                      "no new inference")], "a1_ensemble_summary.csv"


# ============================================================ arm 7 =========
LOC_LEGEND = (
    "A table in front of a robot has 14 named places.\n"
    "Grid cells: grid_A1 grid_A2 grid_A3 grid_B1 grid_B2 grid_B3 "
    "grid_C1 grid_C2 grid_C3.\n"
    "Row A is FURTHEST from the robot (far side, upper side, back).\n"
    "Row B is the MIDDLE row.\n"
    "Row C is NEAREST the robot (near side, bottom side, front).\n"
    "Column 1 is LEFT, column 2 is MIDDLE, column 3 is RIGHT.\n"
    "Containers: box_1 box_2 box_3.\n"
    "Runtime symbols: max_reach (as far as the arm can reach), "
    "nearest_free_cell (any free square).\n")


def run_field_flmd():
    """A1.7 -- escalate the WHOLE 14-way location field to gpt-4o-mini.

    The field-level comparator for P4.5. Same cloud model, same command set,
    same gate signal; the only difference is that the ENTIRE location is
    re-asked instead of only the 3-way row factor.
    """
    from dotenv import load_dotenv
    load_dotenv(REPO / ".env")
    from openai import OpenAI
    client = OpenAI()
    out = []
    for ds in ("held_out_v3", "held_out_v4c"):
        with open(OUT / ("p45_predictions_%s.csv" % ds), encoding="utf-8",
                  newline="") as fh:
            recs = list(csv.DictReader(fh))
        lat, n_err = [], 0
        for r in recs:
            if not r["truth_location"]:
                r["cloud_location"] = ""
                continue
            prompt = (LOC_LEGEND + "\nWhich place does this command refer to?\n"
                      "Answer with exactly one name from the list above.\n\n"
                      "Command: %s\nPlace:" % r["command"])
            t0 = time.perf_counter()
            try:
                resp = client.chat.completions.create(
                    model="gpt-4o-mini", temperature=0.0, max_tokens=8,
                    messages=[{"role": "user", "content": prompt}])
                txt = (resp.choices[0].message.content or "").strip()
            except Exception:
                n_err += 1
                txt = ""
            lat.append((time.perf_counter() - t0) * 1000)
            r["cloud_location"] = txt if txt in L else ""
        cols = list(recs[0])
        with open(OUT / ("a1_field_flmd_%s.csv" % ds), "w", newline="",
                  encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in recs:
                w.writerow({c: r.get(c, "") for c in cols})
        withloc = [r for r in recs if r["truth_location"]]
        acc = sum(1 for r in withloc
                  if r["cloud_location"] == r["truth_location"]) / len(withloc)
        print("  field-level cloud location, %-13s n=%d  acc %.4f  "
              "p50 %.0f ms  errors %d  unparseable %d"
              % (ds, len(withloc), acc, q(lat, .5), n_err,
                 sum(1 for r in withloc if not r["cloud_location"])))
        out.append(dict(arm="field_flmd_cloud_location", dataset=ds,
                        n_commands=len(withloc), location_acc=round(acc, 4),
                        latency_p50_ms=round(q(lat, .5), 1),
                        latency_p95_ms=round(q(lat, .95), 1),
                        latency_p99_ms=round(q(lat, .99), 1),
                        api_errors=n_err,
                        unparseable=sum(1 for r in withloc if not r["cloud_location"])))
    return out, "a1_field_flmd_summary.csv"


ARMS = {"salsa05b": run_salsa05b, "flant5": run_flant5,
        "ensemble": run_ensemble, "field_flmd": run_field_flmd}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    a = ap.parse_args()
    print("=" * 78)
    print("A1 ARM: %s" % a.arm)
    print("=" * 78)
    rows, fname = ARMS[a.arm]()
    p = OUT / fname
    with open(p, "w", newline="", encoding="utf-8") as fh:
        keys = sorted({k for r in rows for k in r})
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print("  wrote %s" % p.relative_to(REPO))
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
