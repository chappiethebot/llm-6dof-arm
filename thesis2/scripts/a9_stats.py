"""thesis2/scripts/a9_stats.py -- A9: clustered bootstrap CIs and McNemar tests
for every headline comparison in FINAL_REPORT.md.

PROTOCOL (fixed before looking at any result)
---------------------------------------------
  * Clustered bootstrap over command_id. 2,000 resamples, seed 20260827.
    Clusters (command_ids) are resampled with replacement; every row belonging
    to a sampled cluster is included. On held_out_v3 each command contributes
    one row per arm, so this reduces to a paired bootstrap over commands -- it
    is implemented generally anyway so repeated-measure cells are handled.
  * McNemar's exact test on the paired binary outcomes. b = A right / B wrong,
    c = A wrong / B right; two-sided exact binomial on the b+c discordant
    pairs against p=0.5. Exact rather than chi-square because several of these
    comparisons have single-digit discordance.
  * DECISION RULE, applied mechanically: a claimed difference is NOT CLAIMABLE
    if the 95% bootstrap CI of the paired difference contains 0, OR McNemar
    p >= 0.05. Both must separate for the claim to stand.

Repeated V5 cells are collapsed to one outcome per command_id by majority vote
over the repeats (ties -> the first repeat). Stated per arm in the output.

Run (conda env thesis2gpu):  python thesis2/scripts/a9_stats.py
"""
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "thesis2" / "scripts"))
OUT = REPO / "thesis2" / "out"
csv.field_size_limit(2 ** 31 - 1)

B = 2000
SEED = 20260827
ALPHA = 0.05

VERBS = ["pick_place", "fetch", "hold", "move_to", "inspect", "stop", "reject"]
TARGETS = ["rubiks_cube", "bottle", "toy_car"]
ROWS_, COLS_ = ["A", "B", "C"], ["1", "2", "3"]
CONTAINERS = ["box_1", "box_2", "box_3"]
RUNTIMES = ["max_reach", "nearest_free_cell"]
NEEDS = {"pick_place": ("target", "location"), "fetch": ("target",),
         "hold": ("target",), "move_to": ("location",), "inspect": ("target",),
         "stop": (), "reject": ()}


def rd(p):
    with open(p, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def T(v):
    return str(v).strip().lower() == "true"


def norm(d):
    if not isinstance(d, dict):
        return None
    v = d.get("action")
    o = {"action": v}
    for f in NEEDS.get(v, ()):
        o[f] = d.get(f)
    mods = d.get("modifiers") or []
    o["modifiers"] = sorted(mods) if isinstance(mods, list) else []
    return json.dumps(o, sort_keys=True)


# ---------------------------------------------------------------- statistics
def mcnemar_exact(a, b):
    """a, b: equal-length 0/1 lists. Returns (b_count, c_count, two-sided p)."""
    bb = sum(1 for x, y in zip(a, b) if x and not y)
    cc = sum(1 for x, y in zip(a, b) if y and not x)
    n = bb + cc
    if n == 0:
        return bb, cc, 1.0
    k = min(bb, cc)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n)
    return bb, cc, min(1.0, 2.0 * tail)


def clustered_bootstrap(clusters, a, b, n_boot=B, seed=SEED):
    """Paired difference mean(a)-mean(b), resampling CLUSTERS with replacement.

    clusters: cluster id per row. a, b: 0/1 per row.
    Returns (point, lo, hi, frac_of_resamples_favouring_a).
    """
    clusters = np.asarray(clusters)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    uniq, inv = np.unique(clusters, return_inverse=True)
    idx_by_cluster = [np.where(inv == i)[0] for i in range(len(uniq))]
    point = a.mean() - b.mean()
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        rows = np.concatenate([idx_by_cluster[j] for j in pick])
        diffs[i] = a[rows].mean() - b[rows].mean()
    lo, hi = np.percentile(diffs, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return point, lo, hi, float((diffs > 0).mean())


def auroc(scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    pos = np.sort(scores[labels])
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r1 = np.searchsorted(pos, neg, side="left")
    r2 = np.searchsorted(pos, neg, side="right")
    return float(np.sum(len(pos) - r2) + 0.5 * np.sum(r2 - r1)) / (len(pos) * len(neg))


def bootstrap_auroc_diff(clusters, s_a, s_b, labels, n_boot=B, seed=SEED):
    """Paired AUROC(s_a) - AUROC(s_b) against the SAME labels."""
    clusters = np.asarray(clusters)
    s_a, s_b = np.asarray(s_a, float), np.asarray(s_b, float)
    labels = np.asarray(labels, bool)
    uniq, inv = np.unique(clusters, return_inverse=True)
    idx_by_cluster = [np.where(inv == i)[0] for i in range(len(uniq))]
    point = auroc(s_a, labels) - auroc(s_b, labels)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        rows = np.concatenate([idx_by_cluster[j] for j in pick])
        d = auroc(s_a[rows], labels[rows]) - auroc(s_b[rows], labels[rows])
        if not math.isnan(d):
            diffs.append(d)
    diffs = np.asarray(diffs)
    lo, hi = np.percentile(diffs, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return point, lo, hi, float((diffs > 0).mean())


RESULTS = []


def compare(label, family, arm_a, arm_b, d_a, d_b, kind="accuracy"):
    """Run both tests on the commands the two arms share, and rule."""
    keys = sorted(set(d_a) & set(d_b), key=lambda k: (len(k), k))
    a = [1 if d_a[k] else 0 for k in keys]
    b = [1 if d_b[k] else 0 for k in keys]
    point, lo, hi, _ = clustered_bootstrap(keys, a, b)
    bb, cc, p = mcnemar_exact(a, b)
    ci_sep = not (lo <= 0.0 <= hi)
    claimable = ci_sep and p < ALPHA
    RESULTS.append(dict(family=family, comparison=label, arm_a=arm_a, arm_b=arm_b,
                        metric=kind, n=len(keys),
                        acc_a=round(sum(a) / len(a), 4), acc_b=round(sum(b) / len(b), 4),
                        diff_pts=round(100 * point, 2),
                        ci_lo_pts=round(100 * lo, 2), ci_hi_pts=round(100 * hi, 2),
                        mcnemar_b=bb, mcnemar_c=cc, mcnemar_p=round(p, 6),
                        ci_excludes_zero=ci_sep, verdict="CLAIMABLE" if claimable
                        else "NOT CLAIMABLE"))
    print("  %-46s n=%-5d %+7.2f pts  [%+6.2f, %+6.2f]  b=%-4d c=%-4d p=%-9s %s"
          % (label, len(keys), 100 * point, 100 * lo, 100 * hi, bb, cc,
             ("%.2e" % p) if p < 1e-4 else ("%.4f" % p),
             "CLAIMABLE" if claimable else "*** NOT CLAIMABLE ***"))
    return claimable


print("=" * 118)
print("A9 -- CLUSTERED BOOTSTRAP CIs AND McNEMAR TESTS")
print("=" * 118)
print("  %d resamples, seed %d, clusters = command_id, alpha = %.2f" % (B, SEED, ALPHA))
print("  RULE: claimable only if the 95%% CI excludes 0 AND McNemar p < %.2f" % ALPHA)
print()

# ================================================================ held_out_v3
p45 = rd(OUT / "p45_predictions_held_out_v3.csv")
truth_of = {r["command_id"]: json.loads(r["truth_json"]) for r in p45}

arms = {}

flat = {}
for r in p45:
    v = VERBS[int(r["flat_verb_idx"])]
    d = {"action": v}
    if "target" in NEEDS[v]:
        d["target"] = TARGETS[int(r["flat_target_idx"])]
    if "location" in NEEDS[v]:
        d["location"] = r["flat_location"]
    flat[r["command_id"]] = norm(d) == norm(truth_of[r["command_id"]])
arms["flat_distilbert"] = flat

cas = {}
for r in p45:
    lt = int(r["fact_loc_type_idx"])
    row_i, col_i = int(r["fact_row_idx"]), int(r["fact_col_idx"])
    if lt == 0:
        e = r["salsa_qwen3blora_letters_row"]
        if e in ROWS_:
            row_i = ROWS_.index(e)
        sym = "grid_%s%s" % (ROWS_[row_i], COLS_[col_i])
    elif lt == 1:
        sym = CONTAINERS[int(r["fact_container_which_idx"])]
    else:
        sym = RUNTIMES[int(r["fact_runtime_which_idx"])]
    v = VERBS[int(r["fact_verb_idx"])]
    d = {"action": v}
    if "target" in NEEDS[v]:
        d["target"] = TARGETS[int(r["fact_target_idx"])]
    if "location" in NEEDS[v]:
        d["location"] = sym
    cas[r["command_id"]] = norm(d) == norm(truth_of[r["command_id"]])
arms["cascade_row_qwen3blora"] = cas

for name, fn in (("factorized", "factorized_fp16.csv"),
                 ("bilstm", "a1_bilstm_held_out_v3.csv"),
                 ("frozen_probe", "a1_frozen_probe_held_out_v3.csv"),
                 ("tinybert", "a1_tinybert_held_out_v3.csv"),
                 ("flant5", "a1_flant5_held_out_v3.csv"),
                 ("rules_open_saved", "baseline_rules.csv")):
    d = {}
    for r in rd(OUT / fn):
        if T(r.get("is_warmup", "")):
            continue
        d[r["command_id"]] = T(r["strict_match"])
    arms[name] = d

ens = {r["command_id"]: T(r["strict_match"]) for r in rd(OUT / "a1_ensemble_held_out_v3.csv")}
arms["ensemble"] = ens


def collapse(path, model, strategy):
    """Majority vote over repeats, per command_id. Warmup rows excluded."""
    by = {}
    for r in rd(REPO / "bench" / path):
        if r["model"] != model or r["strategy"] != strategy:
            continue
        if T(r.get("is_warmup", "")):
            continue
        by.setdefault(r["command_id"], []).append(T(r["strict_match"]))
    reps = max(len(v) for v in by.values())
    return {k: (sum(v) * 2 > len(v)) if len(v) > 1 else v[0] for k, v in by.items()}, reps


arms["qwen3b_lora_v5"], rep_lora = collapse("results_v5_qwen3.csv",
                                            "ollama:qwen2.5-3b-lora", "zero_shot")
arms["gpt4omini_rag_v5"], rep_cloud = collapse("results_v5_cloud.csv",
                                               "openai:gpt-4o-mini", "rag")
print("  repeat collapsing: qwen3b-lora %d repeat(s)/command, gpt-4o-mini/rag %d"
      % (rep_lora, rep_cloud))

# --- rules, recomputed so per-command outcomes exist (deterministic grammar)
print("  recomputing the two rule grammars for per-command outcomes...")
import io
import contextlib
_buf = io.StringIO()
with contextlib.redirect_stdout(_buf):
    import p5_rules_firewalled as RF          # module-level code re-runs; identical output
rows_v3 = [json.loads(l) for l in
           open(REPO / "data" / "held_out_v3.jsonl", encoding="utf-8") if l.strip()]
for nm, roww in (("rules_open", RF.ROW_OPEN), ("rules_firewalled", RF.ROW_FW)):
    fn = RF.make_predict(roww)
    d = {}
    for i, r in enumerate(rows_v3):
        t = dict(r["expected"]); t.setdefault("modifiers", [])
        d[str(i)] = RF.score(fn(r["command"]), t).strict_match
    arms[nm] = d

print("  arm accuracies (n / mean):")
for k in sorted(arms):
    v = arms[k]
    print("    %-26s n=%-5d %.4f" % (k, len(v), sum(v.values()) / len(v)))

print()
print("-" * 118)
print("FAMILY 1 -- held_out_v3 command accuracy")
print("-" * 118)
compare("ensemble vs qwen2.5-3b-lora (V5)", "held_out_v3", "ensemble",
        "qwen3b_lora_v5", arms["ensemble"], arms["qwen3b_lora_v5"])
compare("ensemble vs gpt-4o-mini/rag (V5)", "held_out_v3", "ensemble",
        "gpt4omini_rag_v5", arms["ensemble"], arms["gpt4omini_rag_v5"])
compare("ensemble vs cascade(row->qwen3b-lora)", "held_out_v3", "ensemble",
        "cascade_row_qwen3blora", arms["ensemble"], arms["cascade_row_qwen3blora"])
compare("cascade vs flat DistilBERT", "held_out_v3", "cascade_row_qwen3blora",
        "flat_distilbert", arms["cascade_row_qwen3blora"], arms["flat_distilbert"])
compare("factorized vs flat DistilBERT", "held_out_v3", "factorized",
        "flat_distilbert", arms["factorized"], arms["flat_distilbert"])
compare("flat vs BiLSTM", "held_out_v3", "flat_distilbert", "bilstm",
        arms["flat_distilbert"], arms["bilstm"])
compare("BiLSTM vs factorized", "held_out_v3", "bilstm", "factorized",
        arms["bilstm"], arms["factorized"])
compare("frozen probe vs factorized", "held_out_v3", "frozen_probe", "factorized",
        arms["frozen_probe"], arms["factorized"])
compare("factorized vs TinyBERT-4", "held_out_v3", "factorized", "tinybert",
        arms["factorized"], arms["tinybert"])
compare("factorized vs Flan-T5-base", "held_out_v3", "factorized", "flant5",
        arms["factorized"], arms["flant5"])
compare("flat vs rules (open/unfair)", "held_out_v3", "flat_distilbert",
        "rules_open", arms["flat_distilbert"], arms["rules_open"])
compare("rules open vs rules FIREWALLED", "held_out_v3", "rules_open",
        "rules_firewalled", arms["rules_open"], arms["rules_firewalled"])

# ---------------------------------------------------------------- row factor
print()
print("-" * 118)
print("FAMILY 2 -- row factor accuracy (grid commands only)")
print("-" * 118)


def rowof(loc):
    return loc[5] if isinstance(loc, str) and loc.startswith("grid_") and len(loc) == 7 else None


row_flat, row_fact, row_cas = {}, {}, {}
for r in p45:
    tr = rowof(truth_of[r["command_id"]].get("location"))
    if not tr:
        continue
    cid = r["command_id"]
    row_flat[cid] = (r["flat_row"] == tr)
    row_fact[cid] = (ROWS_[int(r["fact_row_idx"])] == tr)
    e = r["salsa_qwen3blora_letters_row"]
    row_cas[cid] = ((e if e in ROWS_ else ROWS_[int(r["fact_row_idx"])]) == tr)
compare("flat row vs factorized row", "row_factor", "flat_row", "factorized_row",
        row_flat, row_fact, kind="row_accuracy")
compare("cascade row vs flat row", "row_factor", "cascade_row", "flat_row",
        row_cas, row_flat, kind="row_accuracy")

# ---------------------------------------------------------------- A1.7
print()
print("-" * 118)
print("FAMILY 3 -- A1.7 factor-level vs field-level escalation (location accuracy)")
print("-" * 118)
fld = rd(OUT / "a1_field_flmd_held_out_v3.csv")
fac_loc, fld_loc = {}, {}
for r in fld:
    tl = (json.loads(r["truth_json"]) or {}).get("location")
    if not tl:
        continue
    cid = r["command_id"]
    lt = int(r["fact_loc_type_idx"])
    if lt == 0:
        # MUST be the same cloud model as the field-level arm, or the comparison
        # is escalation-target vs escalation-target instead of factor vs field.
        e = r["cloud_row"]
        ri = ROWS_.index(e) if e in ROWS_ else int(r["fact_row_idx"])
        sym = "grid_%s%s" % (ROWS_[ri], COLS_[int(r["fact_col_idx"])])
    elif lt == 1:
        sym = CONTAINERS[int(r["fact_container_which_idx"])]
    else:
        sym = RUNTIMES[int(r["fact_runtime_which_idx"])]
    fac_loc[cid] = (sym == tl)
    fld_loc[cid] = (r["cloud_location"] == tl)
compare("factor vs field level, BOTH gpt-4o-mini", "A1.7", "factor_level_row_only",
        "field_level_whole_location", fac_loc, fld_loc, kind="location_accuracy")

# ---------------------------------------------------------------- INT4
print()
print("-" * 118)
print("FAMILY 4 -- INT4 vs FP16 (the Phase-3 gate; the CLAIM IS NO DIFFERENCE)")
print("-" * 118)
print("  re-running the gate to obtain per-command outcomes (not persisted in Phase 3)...")
import torch
_buf = io.StringIO()
with contextlib.redirect_stdout(_buf):
    import p3_quantization_gate as QG
dev = "cuda" if torch.cuda.is_available() else "cpu"
from transformers import AutoTokenizer                                # noqa: E402

space = QG.build_label_space()
tokq = AutoTokenizer.from_pretrained("distilbert-base-uncased")
modelq = QG.FactorizedClassifier(space)
modelq.load_state_dict(torch.load(QG.CK, map_location="cpu"))
modelq.eval().to(dev)
if dev == "cuda":
    modelq.half()


def eval_percommand(mdl, sw, mode):
    sw.use(mode)
    d = {}
    for i, row in enumerate(rows_v3):
        t = dict(row["expected"]); t.setdefault("modifiers", [])
        enc = tokq(row["command"], truncation=True, max_length=QG.MAX_LEN,
                   padding="max_length", return_tensors="pt").to(dev)
        with torch.no_grad():
            lg = mdl(enc["input_ids"], enc["attention_mask"])
        idx = {k: int(lg[k].argmax(-1).item()) for k in ["verb", "target"] + QG.FACTORS}
        sym = QG.reconstruct(idx["loc_type"], idx["row"], idx["col"],
                             idx["container_which"], idx["runtime_which"])
        obj = QG.emit(idx["verb"], idx["target"], sym, space)
        d[str(i)] = (norm(obj) == norm(t))
    return d


try:
    sw = QG.PrecisionSwitch(modelq)
    fp16 = eval_percommand(modelq, sw, "fp16")
    int4 = eval_percommand(modelq, sw, "int4")
    compare("FP16 vs INT4 (factorized head)", "quantization", "fp16", "int4",
            fp16, int4, kind="accuracy")
except Exception as e:                                        # noqa: BLE001
    print("  COULD NOT RE-RUN THE GATE: %s: %s" % (type(e).__name__, e))
    print("  Per-command INT4 outcomes were not persisted in Phase 3; this row is")
    print("  reported as NOT TESTED rather than guessed.")
    RESULTS.append(dict(family="quantization", comparison="FP16 vs INT4",
                        arm_a="fp16", arm_b="int4", metric="accuracy", n="",
                        acc_a=0.6083, acc_b=0.6083, diff_pts=0.0, ci_lo_pts="",
                        ci_hi_pts="", mcnemar_b="", mcnemar_c="", mcnemar_p="",
                        ci_excludes_zero=False, verdict="NOT TESTED"))

# ================================================================ MASSIVE
print()
print("=" * 118)
print("FAMILY 5 -- MASSIVE (Phase 7 saved predictions; FREE decode only)")
print("=" * 118)
mp = rd(OUT / "generalization_massive_preds.csv")
cl = [str(i) for i in range(len(mp))]
flat_ok = [r["flat_pred"] == r["truth"] for r in mp]
fact_ok = [r["fact_sym"] == r["truth"] for r in mp]
print("  n = %d   flat %.4f   factorized(free) %.4f   illegal %.4f"
      % (len(mp), np.mean(flat_ok), np.mean(fact_ok),
         np.mean([not T(r["fact_legal"]) for r in mp])))
compare("MASSIVE flat vs factorized (free)", "MASSIVE_phase7", "flat",
        "factorized_free", dict(zip(cl, flat_ok)), dict(zip(cl, fact_ok)))

print()
print("  ROUTER SIGNAL -- AUROC of per-factor min-conf vs flat conf, SAME labels")
minf = [min(float(r["scen_conf"]), float(r["act_conf"])) for r in mp]
fltc = [float(r["flat_conf"]) for r in mp]
pt, lo, hi, _ = bootstrap_auroc_diff(cl, minf, fltc, fact_ok)
sep = not (lo <= 0 <= hi)
print("    AUROC per-factor %.4f  flat %.4f" % (auroc(minf, fact_ok), auroc(fltc, fact_ok)))
print("    diff %+.4f  95%% CI [%+.4f, %+.4f]   %s"
      % (pt, lo, hi, "CLAIMABLE" if sep else "*** NOT CLAIMABLE ***"))
RESULTS.append(dict(family="MASSIVE_phase7", comparison="router AUROC per-factor vs flat (free)",
                    arm_a="per_factor_minconf", arm_b="flat_conf", metric="AUROC",
                    n=len(mp), acc_a=round(auroc(minf, fact_ok), 4),
                    acc_b=round(auroc(fltc, fact_ok), 4), diff_pts=round(pt, 4),
                    ci_lo_pts=round(lo, 4), ci_hi_pts=round(hi, 4), mcnemar_b="",
                    mcnemar_c="", mcnemar_p="", ci_excludes_zero=sep,
                    verdict="CLAIMABLE" if sep else "NOT CLAIMABLE"))

# --- fresh run that persists free AND constrained per-row
print()
print("=" * 118)
print("FAMILY 6 -- MASSIVE (fresh run, persists FREE and CONSTRAINED per row)")
print("=" * 118)
print("  Phase 7 / A5 did not save the constrained per-row decode, so the model is")
print("  retrained once here with the same seed and recipe. Point estimates may")
print("  differ in the last decimal from A5 (cuDNN nondeterminism); the VERDICTS")
print("  are what matter and are cross-checked against Family 5 above.")
try:
    from p7_massive import Model, BASE, MAX_LEN, SEED as M_SEED, set_seed
    from datasets import load_dataset
    from transformers import AutoTokenizer
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset

    set_seed(M_SEED)
    dd = load_dataset("mteb/amazon_massive_intent", "en")
    tr, te = dd["train"], dd["test"]
    intents = sorted(set(tr["label_text"]))
    scen = sorted({i.split("_")[0] for i in intents})
    acts = sorted({i.split("_", 1)[1] for i in intents})
    i2i = {s: k for k, s in enumerate(intents)}
    s2i = {s: k for k, s in enumerate(scen)}
    a2i = {s: k for k, s in enumerate(acts)}
    tkm = AutoTokenizer.from_pretrained(BASE)

    class DS(Dataset):
        def __init__(self, sp):
            self.t = sp["text"]
            self.f = [i2i[x] for x in sp["label_text"]]
            self.s = [s2i[x.split("_")[0]] for x in sp["label_text"]]
            self.a = [a2i[x.split("_", 1)[1]] for x in sp["label_text"]]

        def __len__(self):
            return len(self.t)

        def __getitem__(self, i):
            e = tkm(self.t[i], truncation=True, max_length=MAX_LEN,
                    padding="max_length", return_tensors="pt")
            return dict(input_ids=e["input_ids"].squeeze(0),
                        attention_mask=e["attention_mask"].squeeze(0),
                        flat=torch.tensor(self.f[i]), scen=torch.tensor(self.s[i]),
                        act=torch.tensor(self.a[i]))

    dtr, dte = DS(tr), DS(te)
    dl = DataLoader(dtr, batch_size=32, shuffle=True,
                    generator=torch.Generator().manual_seed(M_SEED))
    mm = Model(len(intents), len(scen), len(acts)).to(dev)
    opt = torch.optim.AdamW(mm.parameters(), lr=5e-5, weight_decay=0.01)
    total = 4 * math.ceil(len(dtr) / 32)
    sch = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s / 100) if s < 100 else max(0.0, 1.0 - (s - 100) / max(1, total - 100)))
    ce = nn.CrossEntropyLoss()
    for ep in range(4):
        mm.train()
        for bt in dl:
            fl, sc, ac = mm(bt["input_ids"].to(dev), bt["attention_mask"].to(dev))
            loss = (ce(fl, bt["flat"].to(dev)) + ce(sc, bt["scen"].to(dev))
                    + ce(ac, bt["act"].to(dev)))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(mm.parameters(), 1.0)
            opt.step(); sch.step()
        print("    epoch %d done" % (ep + 1))

    LEGAL = [(s2i[i.split("_")[0]], a2i[i.split("_", 1)[1]]) for i in intents]
    mm.eval()
    res = []
    with torch.no_grad():
        for bt in DataLoader(dte, batch_size=128):
            fl, sc, ac = mm(bt["input_ids"].to(dev), bt["attention_mask"].to(dev))
            pf, ps, pa = (torch.softmax(fl, -1), torch.softmax(sc, -1), torch.softmax(ac, -1))
            for i in range(fl.shape[0]):
                si, ai = int(ps[i].argmax()), int(pa[i].argmax())
                lp, la = ps[i].float().log(), pa[i].float().log()
                best = max(LEGAL, key=lambda q: float(lp[q[0]] + la[q[1]]))
                res.append(dict(truth=intents[int(bt["flat"][i])],
                                flat_pred=intents[int(pf[i].argmax())],
                                flat_conf=float(pf[i].max()),
                                free_sym="%s_%s" % (scen[si], acts[ai]),
                                constrained_sym="%s_%s" % (scen[best[0]], acts[best[1]]),
                                scen_conf=float(ps[i].max()), act_conf=float(pa[i].max())))
    pth = OUT / "a9_massive_rows.csv"
    with open(pth, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(res[0].keys()))
        w.writeheader(); w.writerows(res)
    print("    wrote %s (%d rows)" % (pth.relative_to(REPO), len(res)))

    c2 = [str(i) for i in range(len(res))]
    fl_ok = [r["flat_pred"] == r["truth"] for r in res]
    fr_ok = [r["free_sym"] == r["truth"] for r in res]
    co_ok = [r["constrained_sym"] == r["truth"] for r in res]
    print("    flat %.4f   free %.4f   constrained %.4f"
          % (np.mean(fl_ok), np.mean(fr_ok), np.mean(co_ok)))
    print()
    compare("MASSIVE flat vs factorized (free)", "MASSIVE_fresh", "flat",
            "factorized_free", dict(zip(c2, fl_ok)), dict(zip(c2, fr_ok)))
    compare("MASSIVE flat vs factorized (CONSTRAINED)", "MASSIVE_fresh", "flat",
            "factorized_constrained", dict(zip(c2, fl_ok)), dict(zip(c2, co_ok)))
    compare("MASSIVE constrained vs free", "MASSIVE_fresh", "factorized_constrained",
            "factorized_free", dict(zip(c2, co_ok)), dict(zip(c2, fr_ok)))

    print()
    print("  ROUTER SIGNAL under both decodes")
    mn = [min(r["scen_conf"], r["act_conf"]) for r in res]
    fc = [r["flat_conf"] for r in res]
    for lbl, lab in (("free", fr_ok), ("constrained", co_ok)):
        pt, lo, hi, _ = bootstrap_auroc_diff(c2, mn, fc, lab)
        sep = not (lo <= 0 <= hi)
        print("    %-12s AUROC per-factor %.4f  flat %.4f  diff %+.4f  "
              "CI [%+.4f, %+.4f]  %s"
              % (lbl, auroc(mn, lab), auroc(fc, lab), pt, lo, hi,
                 "CLAIMABLE" if sep else "*** NOT CLAIMABLE ***"))
        RESULTS.append(dict(family="MASSIVE_fresh",
                            comparison="router AUROC per-factor vs flat (%s)" % lbl,
                            arm_a="per_factor_minconf", arm_b="flat_conf",
                            metric="AUROC", n=len(res),
                            acc_a=round(auroc(mn, lab), 4), acc_b=round(auroc(fc, lab), 4),
                            diff_pts=round(pt, 4), ci_lo_pts=round(lo, 4),
                            ci_hi_pts=round(hi, 4), mcnemar_b="", mcnemar_c="",
                            mcnemar_p="", ci_excludes_zero=sep,
                            verdict="CLAIMABLE" if sep else "NOT CLAIMABLE"))
except Exception as e:                                        # noqa: BLE001
    print("  FRESH MASSIVE RUN FAILED: %s: %s" % (type(e).__name__, e))
    print("  Family 5 (saved Phase-7 predictions, free decode) still stands.")

# ---------------------------------------------------------------- write
print()
print("=" * 118)
print("VERDICT SUMMARY")
print("=" * 118)
nc = [r for r in RESULTS if r["verdict"] != "CLAIMABLE"]
print("  %d comparisons tested, %d CLAIMABLE, %d NOT CLAIMABLE/NOT TESTED"
      % (len(RESULTS), len(RESULTS) - len(nc), len(nc)))
for r in nc:
    print("    %-14s %-48s diff %+8s  %s"
          % (r["family"], r["comparison"], r["diff_pts"], r["verdict"]))

p = OUT / "a9_significance.csv"
with open(p, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(RESULTS[0].keys()))
    w.writeheader(); w.writerows(RESULTS)
(OUT / "a9_significance_summary.json").write_text(json.dumps(dict(
    n_boot=B, seed=SEED, alpha=ALPHA, cluster="command_id",
    rule="claimable iff 95% CI excludes 0 AND McNemar exact p < alpha",
    n_comparisons=len(RESULTS), n_claimable=len(RESULTS) - len(nc),
    not_claimable=[r["comparison"] for r in nc], results=RESULTS), indent=2),
    encoding="utf-8")
print()
print("wrote %s" % p.relative_to(REPO))
print("wrote thesis2/out/a9_significance_summary.json")
print("=" * 118)
