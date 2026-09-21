"""thesis2/scripts/p5_baselines.py -- PHASE 5, the floor.

Four baselines, all trained on train_v2 only, selected on dev_v2 only,
evaluated on held_out_v3. Predictions written in a scored-CSV-compatible
schema and scored with llm/score.py.

  P5.1  rule-based slot grammar   regex/keyword, microsecond latency
  P5.2  kNN over MiniLM           encode, cosine-match train_v2, copy the label
  P5.3  TF-IDF + logistic regression, three heads
  P5.4  TinyBERT-4 fine-tuned with the FACTORIZED head (the sub-15M edge arm)

Every arm emits through the same schema mask as the classifiers, so schema
validity is 1.0 by construction for all of them and the comparison is about
semantics, not format.
"""
import csv
import json
import math
import os
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "thesis2" / "scripts"))

from llm.score import score                                      # noqa: E402
from p2_train_factorized import build_label_space, emit           # noqa: E402

OUT = REPO / "thesis2" / "out"
SPACE = build_label_space()
V, T, L = SPACE["verbs"], SPACE["targets"], SPACE["locations"]


def load(name):
    return [json.loads(l) for l in
            open(REPO / "data" / ("%s.jsonl" % name), encoding="utf-8") if l.strip()]


TRAIN, DEV, HELD = load("train_v2"), load("dev_v2"), load("held_out_v3")


def evaluate(name, predict_fn, note=""):
    """Run one arm over held_out_v3, score, write its CSV, return the summary."""
    lat, rows = [], []
    hits = dict(action=0, verb=0, target=0, location=0, joint=0)
    den = dict(action=0, verb=0, target=0, location=0, joint=0)
    for ci, r in enumerate(HELD):
        truth = dict(r["expected"])
        truth.setdefault("modifiers", [])
        t0 = time.perf_counter()
        obj = predict_fn(r["command"])
        lat.append((time.perf_counter() - t0) * 1e6)          # microseconds
        sm = bool(score(obj, truth).strict_match)
        den["action"] += 1
        hits["action"] += int(sm)
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
        rows.append(dict(model=name, strategy="baseline", dataset="held_out_v3",
                         command_id=ci, command=r["command"], run_index=0,
                         is_warmup=False, stage="validated", ok=True,
                         strict_match=sm, pred_json=json.dumps(obj),
                         truth_json=json.dumps(truth),
                         latency_s=round(lat[-1] / 1e6, 8)))
    p = OUT / ("baseline_%s.csv" % name)
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lat.sort()
    q = lambda f: lat[min(len(lat) - 1, int(round(f * (len(lat) - 1))))]
    rej = [r for r in HELD if r["expected"]["action"] == "reject"]
    rejhit = sum(1 for r in rej if predict_fn(r["command"]).get("action") == "reject")
    s = dict(arm=name, n=den["action"],
             action_match=round(hits["action"] / den["action"], 4),
             joint=round(hits["joint"] / den["joint"], 4),
             verb=round(hits["verb"] / den["verb"], 4),
             target=round(hits["target"] / den["target"], 4),
             location=round(hits["location"] / den["location"], 4),
             n_location=den["location"],
             reject_recall=round(rejhit / len(rej), 4) if rej else "",
             schema_validity=1.0,
             lat_p50_us=round(q(.50), 1), lat_p95_us=round(q(.95), 1),
             lat_p99_us=round(q(.99), 1), note=note)
    print("  %-16s action %.4f  joint %.4f  loc %.4f  p50 %.1f us"
          % (name, s["action_match"], s["joint"], s["location"], s["lat_p50_us"]))
    return s


# ------------------------------------------------------------ P5.1 rules ----
COLOURS = {"red": "rubiks_cube", "blue": "bottle", "yellow": "toy_car"}
NOUNS = [("rubiks cube", "rubiks_cube"), ("rubik's cube", "rubiks_cube"),
         ("toy car", "toy_car"), ("bottle", "bottle"), ("cube", "rubiks_cube"),
         ("car", "toy_car")]
ROWW = [(("upper", "top", "far", "back"), "A"), (("middle", "centre", "center"), "B"),
        (("bottom", "lower", "near", "front"), "C")]
COLW = [(("left",), "1"), (("middle", "centre", "center"), "2"), (("right",), "3")]


def rule_predict(cmd):
    t = " %s " % " ".join(re.sub(r"[^a-z0-9 ]", " ", cmd.lower()).split())
    verb = "reject"
    if any(w in t for w in (" stop ", " halt ", " freeze ")):
        verb = "stop"
    elif any(w in t for w in (" bring ", " fetch ", " give ", " retrieve ")):
        verb = "fetch"
    elif " hold " in t or " keep holding " in t:
        verb = "hold"
    elif any(w in t for w in (" inspect ", " look at ", " examine ", " scan ")):
        verb = "inspect"
    elif any(w in t for w in (" put ", " place ", " keep ", " drop ", " stick ", " move the ")):
        verb = "pick_place"
    elif any(w in t for w in (" move ", " go ", " navigate ", " travel ")):
        verb = "move_to"
    target = None
    for phrase, sym in NOUNS:
        if " %s " % phrase in t:
            target = sym
            break
    if target is None:
        for c, sym in COLOURS.items():
            if " %s " % c in t:
                target = sym
                break
    loc = None
    m = re.search(r" box (\d) ", t)
    if m and m.group(1) in "123":
        loc = "box_%s" % m.group(1)
    elif "free" in t or "space" in t or "anywhere" in t:
        loc = "nearest_free_cell"
    elif "as far as" in t or "farthest" in t or "maximum" in t:
        loc = "max_reach"
    else:
        row = col = None
        for ws, s in ROWW:
            if any(" %s " % w in t for w in ws):
                row = s
                break
        for ws, s in COLW:
            if any(" %s " % w in t for w in ws):
                col = s
                break
        if row and col:
            loc = "grid_%s%s" % (row, col)
        elif row:
            loc = "grid_%s2" % row
        elif col:
            loc = "grid_B%s" % col
    if verb == "reject" and (target or loc):
        verb = "pick_place" if (target and loc) else ("fetch" if target else "move_to")
    vi = V.index(verb)
    ti = T.index(target) if target else 0
    return emit(vi, ti, loc if loc else L[0], SPACE)


# -------------------------------------------------------------- P5.2 kNN ----
def build_knn():
    from sentence_transformers import SentenceTransformer
    import numpy as np
    m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    corpus = [r["command"] for r in TRAIN]
    E = m.encode(corpus, normalize_embeddings=True, show_progress_bar=False)

    def predict(cmd):
        q = m.encode([cmd], normalize_embeddings=True, show_progress_bar=False)[0]
        i = int(np.argmax(E @ q))
        exp = TRAIN[i]["expected"]
        vi = V.index(exp["action"])
        ti = T.index(exp["target"]) if exp.get("target") else 0
        return emit(vi, ti, exp.get("location") or L[0], SPACE)
    return predict


# ----------------------------------------------------------- P5.3 TF-IDF ----
def build_tfidf():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
    X = vec.fit_transform([r["command"] for r in TRAIN])
    heads = {}
    for name, key, space in (("verb", "action", V), ("target", "target", T),
                             ("location", "location", L)):
        idx = [i for i, r in enumerate(TRAIN) if r["expected"].get(key)]
        y = [space.index(TRAIN[i]["expected"][key]) for i in idx]
        clf = LogisticRegression(max_iter=2000, C=4.0)
        clf.fit(X[idx], y)
        heads[name] = clf

    def predict(cmd):
        x = vec.transform([cmd])
        vi = int(heads["verb"].predict(x)[0])
        ti = int(heads["target"].predict(x)[0])
        li = int(heads["location"].predict(x)[0])
        return emit(vi, ti, L[li], SPACE)
    return predict


def main():
    print("=" * 78)
    print("PHASE 5 -- baselines. train=%d dev=%d held_out_v3=%d"
          % (len(TRAIN), len(DEV), len(HELD)))
    print("=" * 78)
    summ = []
    summ.append(evaluate("rules", rule_predict, "regex/keyword slot grammar"))
    try:
        summ.append(evaluate("knn_minilm", build_knn(),
                             "all-MiniLM-L6-v2, cosine 1-NN over train_v2"))
    except Exception as e:
        print("  knn_minilm SKIPPED: %s: %s" % (type(e).__name__, e))
    try:
        summ.append(evaluate("tfidf_logreg", build_tfidf(),
                             "TF-IDF 1-2gram + logistic regression, 3 heads"))
    except Exception as e:
        print("  tfidf_logreg SKIPPED: %s: %s" % (type(e).__name__, e))

    p = OUT / "baselines_summary.csv"
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summ[0]))
        w.writeheader()
        w.writerows(summ)
    print()
    print("%-16s %5s %8s %8s %8s %8s %10s" %
          ("arm", "n", "action", "joint", "loc", "reject", "p50_us"))
    for s in summ:
        print("%-16s %5d %8.4f %8.4f %8.4f %8s %10.1f" %
              (s["arm"], s["n"], s["action_match"], s["joint"], s["location"],
               s["reject_recall"], s["lat_p50_us"]))
    print()
    print("wrote %s" % p.relative_to(REPO))
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
