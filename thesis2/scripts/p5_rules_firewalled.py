"""thesis2/scripts/p5_rules_firewalled.py -- the honest rules baseline.

P5.1's grammar scores 0.7235 on held_out_v3, beating every trained arm. That
number is NOT comparable to the trained arms, and this script shows why.

I wrote its row vocabulary as ("upper","top","far","back") -- covering
upper/bottom (held_out_v3), far/near (train_v2) AND back/front (held_out_v4c).
A rule author naturally writes down every word they can think of. But
train_v2 contains ONLY far/near, so the trained models never saw upper, bottom,
back or front. The rules baseline is therefore using vocabulary knowledge that
was firewalled away from every neural arm.

This script runs both:

  rules_open        every row word I could think of (the P5.1 number)
  rules_firewalled  ONLY row words that actually occur in train_v2

The gap between them is the size of the unfair advantage, measured rather than
argued.
"""
import collections
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "thesis2" / "scripts"))

from llm.score import score                                    # noqa: E402
from p2_train_factorized import build_label_space, emit         # noqa: E402
import p5_baselines as B                                        # noqa: E402

SPACE = build_label_space()
V, T, L = SPACE["verbs"], SPACE["targets"], SPACE["locations"]
OUT = REPO / "thesis2" / "out"

CANDIDATE_ROW_WORDS = ["upper", "top", "far", "back", "bottom", "lower",
                       "near", "front", "middle", "centre", "center"]
CANDIDATE_COL_WORDS = ["left", "right", "middle", "centre", "center"]


def words_in(rows):
    seen = collections.Counter()
    for r in rows:
        for w in re.sub(r"[^a-z0-9 ]", " ", r["command"].lower()).split():
            seen[w] += 1
    return seen


TRAIN_WORDS = words_in(B.TRAIN)
print("=" * 78)
print("WHICH SPATIAL WORDS ACTUALLY OCCUR IN train_v2?")
print("=" * 78)
allowed_row, allowed_col = [], []
for w in CANDIDATE_ROW_WORDS:
    n = TRAIN_WORDS.get(w, 0)
    print("  row word  %-8s train_v2 count %4d   %s"
          % (w, n, "ALLOWED" if n else "FIREWALLED OUT"))
    if n:
        allowed_row.append(w)
for w in CANDIDATE_COL_WORDS:
    n = TRAIN_WORDS.get(w, 0)
    if n:
        allowed_col.append(w)
print("  -> firewalled row vocabulary: %s" % allowed_row)
print("  -> firewalled col vocabulary: %s" % allowed_col)

ROW_OPEN = [(("upper", "top", "far", "back"), "A"),
            (("middle", "centre", "center"), "B"),
            (("bottom", "lower", "near", "front"), "C")]
ROW_FW = [(tuple(w for w in ("upper", "top", "far", "back") if w in allowed_row), "A"),
          (tuple(w for w in ("middle", "centre", "center") if w in allowed_row), "B"),
          (tuple(w for w in ("bottom", "lower", "near", "front") if w in allowed_row), "C")]


def make_predict(roww):
    def predict(cmd):
        t = " %s " % " ".join(re.sub(r"[^a-z0-9 ]", " ", cmd.lower()).split())
        verb = "reject"
        if any(w in t for w in (" stop ", " halt ", " freeze ")):
            verb = "stop"
        elif any(w in t for w in (" bring ", " fetch ", " give ", " retrieve ")):
            verb = "fetch"
        elif " hold " in t:
            verb = "hold"
        elif any(w in t for w in (" inspect ", " look at ", " examine ", " scan ")):
            verb = "inspect"
        elif any(w in t for w in (" put ", " place ", " keep ", " drop ", " stick ", " move the ")):
            verb = "pick_place"
        elif any(w in t for w in (" move ", " go ", " navigate ", " travel ")):
            verb = "move_to"
        target = None
        for phrase, sym in B.NOUNS:
            if " %s " % phrase in t:
                target = sym
                break
        if target is None:
            for c, sym in B.COLOURS.items():
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
            for ws, s in roww:
                if ws and any(" %s " % w in t for w in ws):
                    row = s
                    break
            for ws, s in B.COLW:
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
        return emit(V.index(verb), T.index(target) if target else 0,
                    loc if loc else L[0], SPACE)
    return predict


def run(name, fn, dataset):
    rows = [json.loads(l) for l in
            open(REPO / "data" / ("%s.jsonl" % dataset), encoding="utf-8") if l.strip()]
    hit = loc_hit = loc_n = 0
    lat = []
    for r in rows:
        truth = dict(r["expected"])
        truth.setdefault("modifiers", [])
        t0 = time.perf_counter()
        obj = fn(r["command"])
        lat.append((time.perf_counter() - t0) * 1e6)
        hit += int(score(obj, truth).strict_match)
        if truth.get("location"):
            loc_n += 1
            loc_hit += int(obj.get("location") == truth["location"])
    lat.sort()
    return dict(arm=name, dataset=dataset, n=len(rows),
                action_match=round(hit / len(rows), 4),
                location_acc=round(loc_hit / loc_n, 4), n_location=loc_n,
                lat_p50_us=round(lat[len(lat) // 2], 1))


print()
print("=" * 78)
print("OPEN vs FIREWALLED rule grammar")
print("=" * 78)
out = []
for ds in ("held_out_v3", "held_out_v4c"):
    for nm, roww in (("rules_open", ROW_OPEN), ("rules_firewalled", ROW_FW)):
        s = run(nm, make_predict(roww), ds)
        out.append(s)
        print("  %-18s %-14s action %.4f  location %.4f  p50 %.1f us"
              % (s["arm"], s["dataset"], s["action_match"], s["location_acc"],
                 s["lat_p50_us"]))

p = OUT / "baseline_rules_firewall.csv"
with open(p, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0]))
    w.writeheader()
    w.writerows(out)
print()
for ds in ("held_out_v3", "held_out_v4c"):
    o = [r for r in out if r["dataset"] == ds and r["arm"] == "rules_open"][0]
    f = [r for r in out if r["dataset"] == ds and r["arm"] == "rules_firewalled"][0]
    print("  %s: the unfair advantage is %+.1f action points, %+.1f location points"
          % (ds, 100 * (o["action_match"] - f["action_match"]),
             100 * (o["location_acc"] - f["location_acc"])))
print("wrote %s" % p.relative_to(REPO))
print("=" * 78)
