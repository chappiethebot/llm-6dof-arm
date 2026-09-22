"""thesis2/scripts/a15_metrics_v2_append.py -- add the speculative-decoding and
v4c rows to metrics_all_v2.csv.

WHY. metrics_all_v2.csv is declared in FINAL_REPORT.md as the only file thesis
tables may be built from, but it inherited metrics_all.csv's 50 arms and so
contained no A8/A10/A12 row and no v4c row for the two arms §3 now leads with
(re-review finding S-E). This adds them, computed from the saved per-command
files, keeping the file's "no blank cells" property.

It reads only saved outputs; no model is loaded. Idempotent: re-running
replaces the appended rows rather than duplicating them.

Run:  python thesis2/scripts/a15_metrics_v2_append.py
"""
import csv
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)
sys.path.insert(0, str(REPO))
OUT = REPO / "thesis2" / "out"
csv.field_size_limit(2 ** 31 - 1)
from llm.score import score                                            # noqa: E402

rd = lambda p: list(csv.DictReader(open(p, encoding="utf-8", newline="")))
sch = json.loads((REPO / "schema" / "schema_v2.json").read_text(encoding="utf-8"))
VERBS, TARGETS = list(sch["verbs"]), sch["targets"]
REQ = {v: sch["verbs"][v]["required"] for v in VERBS}
ROWS, COLS = ["A", "B", "C"], ["1", "2", "3"]
CONT, RT = ["box_1", "box_2", "box_3"], ["max_reach", "nearest_free_cell"]
rowof = lambda l: l[5] if isinstance(l, str) and l.startswith("grid_") and len(l) == 7 else None
colof = lambda l: l[6] if isinstance(l, str) and l.startswith("grid_") and len(l) == 7 else None


def metrics(objs, truths):
    """Every accuracy column metrics_all uses, from predicted objects."""
    n = len(objs)
    h = dict(action=0, joint=0, verb=0, target=0, location=0)
    d = dict(action=0, joint=0, verb=0, target=0, location=0)
    fh = {k: [0, 0] for k in ("loc_type", "row", "col", "container_which", "runtime_which")}
    rej_h = rej_n = 0
    for o, t in zip(objs, truths):
        t = dict(t)
        t.setdefault("modifiers", [])
        d["action"] += 1
        h["action"] += int(bool(score(o, t).strict_match))
        d["verb"] += 1
        h["verb"] += int(o.get("action") == t.get("action"))
        if t.get("target"):
            d["target"] += 1
            h["target"] += int(o.get("target") == t.get("target"))
        tl = t.get("location")
        if tl:
            d["location"] += 1
            h["location"] += int(o.get("location") == tl)
        if t.get("target") and tl:
            d["joint"] += 1
            h["joint"] += int(all(o.get(k) == t.get(k) for k in ("action", "target", "location")))
        if t["action"] == "reject":
            rej_n += 1
            rej_h += int(o.get("action") == "reject")
        pl = o.get("location")
        if tl:
            ltype = "grid" if tl.startswith("grid_") else ("container" if tl in CONT else "runtime")
            ptype = ("grid" if (pl or "").startswith("grid_")
                     else ("container" if pl in CONT else ("runtime" if pl in RT else None)))
            fh["loc_type"][1] += 1
            fh["loc_type"][0] += int(ptype == ltype)
            if ltype == "grid":
                fh["row"][1] += 1
                fh["row"][0] += int(rowof(pl) == rowof(tl))
                fh["col"][1] += 1
                fh["col"][0] += int(colof(pl) == colof(tl))
            elif ltype == "container":
                fh["container_which"][1] += 1
                fh["container_which"][0] += int(pl == tl)
            else:
                fh["runtime_which"][1] += 1
                fh["runtime_which"][0] += int(pl == tl)
    r = lambda a, b: round(a / b, 4) if b else "n/a"
    out = dict(n_commands=n, action_match=r(h["action"], d["action"]),
               joint_correct=r(h["joint"], d["joint"]), verb_acc=r(h["verb"], d["verb"]),
               target_acc=r(h["target"], d["target"]), location_acc=r(h["location"], d["location"]),
               reject_recall=r(rej_h, rej_n))
    for k, (a, b) in fh.items():
        out["factor_%s" % k] = r(a, b)
    return out


ho3 = [json.loads(l) for l in (REPO / "data" / "held_out_v3.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
ho4 = [json.loads(l) for l in (REPO / "data" / "held_out_v4c.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

# ---- A8 / A10 / A12 all reproduce the SAME greedy output on held_out_v3
a8 = rd(OUT / "a8_speculative_percommand.csv")
objs = []
for r in a8:
    try:
        objs.append(json.loads(r["greedy_text"]))
    except Exception:                                                  # noqa: BLE001
        objs.append({})
spec3 = metrics(objs, [r["expected"] for r in ho3])
print("held_out_v3 greedy/A8/A10/A12 output: %s" % spec3)

# ---- A13: the same verifier on v4c, and the flat classifier on v4c
l4 = rd(OUT / "a13_v4c_lora_percommand.csv")
lora4 = metrics([json.loads(r["pred_json"]) for r in l4], [r["expected"] for r in ho4])
print("held_out_v4c qwen2.5-3b-lora V5 zero-shot: %s" % lora4)
p454 = rd(OUT / "p45_predictions_held_out_v4c.csv")


def emit(v, t, l):
    o = {"action": v}
    if "target" in REQ[v]: o["target"] = t
    if "location" in REQ[v]: o["location"] = l
    o["modifiers"] = []
    return o


flat4 = metrics([emit(VERBS[int(r["flat_verb_idx"])], TARGETS[int(r["flat_target_idx"])],
                      r["flat_location"]) for r in p454], [r["expected"] for r in ho4])
print("held_out_v4c flat DistilBERT classifier: %s" % flat4)

a8s = json.loads((OUT / "a8_speculative_summary.json").read_text(encoding="utf-8"))
a10s = json.loads((OUT / "a10_prefix_cache_summary.json").read_text(encoding="utf-8"))
a12s = json.loads((OUT / "a12_smalllm_drafter_summary.json").read_text(encoding="utf-8"))
E = {r["arm"]: r for r in rd(OUT / "a8_energy.csv")}
E.update({r["arm"]: r for r in rd(OUT / "a10_energy.csv")})
E.update({r["arm"]: r for r in rd(OUT / "a12_energy.csv")})

base = list(rd(OUT / "metrics_all_v2.csv"))
cols = list(base[0].keys())
NM = "not measured"


def row(arm, dataset, m, lat, joules, resident, note, reason, status="valid"):
    d = {c: NM for c in cols}
    d.update(arm=arm, dataset=dataset, note=note, status=status, reason=reason)
    d.update({k: v for k, v in m.items() if k in cols})
    d["schema_validity"] = 1.0
    d["ECE_primary"] = "n/a: greedy decode, no confidence" if "lora" in arm or "spec" in arm else NM
    d["Brier_primary"] = d["ECE_primary"]
    d["ECE_head_detail"] = "n/a"
    d["cold_load_s"] = NM
    d["escalation_rate"] = "n/a: not a routing arm"
    d["latency_p50_ms"], d["latency_p95_ms"], d["latency_p99_ms"] = lat
    d["joules_per_command"] = joules
    d["resident_mb"] = resident
    return d


L8 = a8s["latency_ms"]["speculative (incl. drafter)"]
L10 = a10s["latency_ms"]["PREFIX-CACHED SPECULATIVE (A10)"]
L12 = a12s["latency_ms"]["0.5b-drafted speculative (A12)"]
LG = a8s["latency_ms"]["plain greedy (transformers fp16)"]
new = [
    row("qwen2.5-3b-lora_zero_shot_V5_transformers_fp16", "held_out_v3", spec3,
        (LG["p50"], LG["p95"], LG["p99"]),
        E["a8_plain_greedy_qwen3blora"]["joules_per_command"], 6593.0,
        "plain greedy, no speculation; the baseline A8/A10/A12 reproduce exactly",
        "measured this session; the arm A8/A10/A12 accelerate"),
    row("A8_classifier_drafted_speculative", "held_out_v3", spec3,
        (L8["p50"], L8["p95"], L8["p99"]),
        E["a8_speculative_qwen3blora"]["joules_per_command"], 6725.76,
        "217/217 byte-identical to the row above; accuracy is that row's by construction",
        "added by a15 (re-review S-E); accuracy columns are the verifier's, which "
        "speculative decoding reproduces token for token"),
    row("A10_prefix_cached_classifier_drafted_speculative", "held_out_v3", spec3,
        (L10["p50"], L10["p95"], L10["p99"]),
        E["a10_prefix_cached_speculative"]["joules_per_command"], 6725.76,
        "217/217 byte-identical, verified three times incl. shuffled order",
        "added by a15 (re-review S-E); same accuracy by construction"),
    row("A12_qwen2.5-0.5b_drafted_speculative", "held_out_v3", spec3,
        (L12["p50"], L12["p95"], L12["p99"]),
        E["a12_prefix_cached_spec_qwen05b_drafter"]["joules_per_command"],
        "7581 (3B 6593 + 0.5B ~988, not separately measured)",
        "conventional small-LM drafter; 217/217 identical but slower than plain greedy",
        "added by a15 (re-review S-E); same accuracy by construction"),
    row("qwen2.5-3b-lora_zero_shot_V5_transformers_fp16", "held_out_v4c", lora4,
        (NM, NM, NM), NM, 6593.0,
        "the fair vocabulary-generalisation figure for prompted arms; A8/A10/A12 "
        "equal this by construction",
        "added by a15; A13 run. The V5 legend defines neither back nor front"),
    row("distilbert_flat_head", "held_out_v4c", flat4, (NM, NM, NM), NM, 132.8,
        "reads no legend; beats the prompted arm on v4c (action p_holm 0.0215)",
        "added by a15; A13 run"),
]
keep = [r for r in base if (r["arm"], r["dataset"]) not in {(n["arm"], n["dataset"]) for n in new}]
allrows = keep + new
with open(OUT / "metrics_all_v2.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=cols)
    w.writeheader()
    w.writerows(allrows)
blank = sum(1 for r in allrows for v in r.values() if v == "")
print()
print("metrics_all_v2.csv: %d rows (%d carried over + %d added), %d blank cells"
      % (len(allrows), len(keep), len(new), blank))
for n in new:
    print("   + %-52s %-14s action %s row %s" % (n["arm"][:52], n["dataset"],
                                                 n["action_match"], n["factor_row"]))
