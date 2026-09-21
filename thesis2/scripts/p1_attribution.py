"""thesis2/scripts/p1_attribution.py -- PHASE 1, offline attribution.

READ-ONLY over the existing tree. Reads bench/results_v5_*_scored.csv,
audit/cls_margins.csv, data/held_out_v3.jsonl and schema/schema_v2.json.
Writes ONLY under thesis2/out and thesis2/figures. No model is called.

CONVENTIONS, applied everywhere and restated on every table:

  warmup      EXCLUDED. is_warmup == "True" rows are dropped before anything.
              That makes the 31 local cells n=216 commands and the 5 cloud
              cells n=217, because command_id 0 is consumed as the warmup at
              --repeats 1. Every table carries its own n.

  dedupe      (dataset, model, strategy, command_id, run_index), keep LAST --
              the same 5-field key bench/summary.py, analyze.py, figures.py and
              rescore.py all use. Verified zero duplicates on the V5 set.

  per-command Metrics are the mean over COMMANDS of the per-command mean over
              repeats. Local cells have one repeat after warmup exclusion so
              this is identical to per-row for them; cloud cells have 4-5, and
              a command contributes once regardless. Confusion matrices POOL
              repeats, because they describe a distribution rather than a rate.

  correctness `strict_match` from llm/score.py is authoritative for action
              match. Field-level and factor-level correctness are recomputed
              here from pred_json/truth_json, because no existing column
              carries them.
"""
import collections
import csv
import json
import os
import sys

csv.field_size_limit(2 ** 31 - 1)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
OUT = os.path.join("thesis2", "out")
FIG = os.path.join("thesis2", "figures")

SCORED = {
    "rung1": "bench/results_v5_rung1_scored.csv",
    "rag": "bench/results_v5_rag_scored.csv",
    "lora": "bench/results_v5_lora_scored.csv",
    "cls": "bench/results_v5_cls_scored.csv",
}

T = lambda v: str(v).strip().lower() == "true"


# --------------------------------------------------------------- loading ----
def load_rows():
    raw = []
    for tag, path in SCORED.items():
        with open(path, encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                r["_file"] = tag
                raw.append(r)
    ded = {}
    for r in raw:
        ded[(r.get("dataset", ""), r["model"], r["strategy"],
             r["command_id"], r["run_index"])] = r
    dup = len(raw) - len(ded)
    rows = [r for r in ded.values() if not T(r["is_warmup"])]
    return raw, dup, rows


def obj(r, key):
    try:
        o = json.loads(r[key]) if r[key] else None
        return o if isinstance(o, dict) else None
    except Exception:
        return None


def field(o, name):
    """Schema field from an action object. `action` is the verb key."""
    if o is None:
        return None
    if name == "verb":
        v = o.get("action")
        if v is None:
            v = o.get("verb")
        return v if isinstance(v, str) else None
    v = o.get(name)
    return v if isinstance(v, str) else None


# --------------------------------------------------------------- factors ----
CONTAINERS = {"box_1", "box_2", "box_3"}
RUNTIME = {"max_reach", "nearest_free_cell"}


def loc_type(loc):
    if loc is None:
        return None
    if loc in CONTAINERS:
        return "container"
    if loc in RUNTIME:
        return "runtime"
    if loc.startswith("grid_") and len(loc) == 7:
        return "grid"
    return "other"


def loc_row(loc):
    return loc[5] if loc_type(loc) == "grid" else None


def loc_col(loc):
    return loc[6] if loc_type(loc) == "grid" else None


# ------------------------------------------------------------- P1.1 TIDE ----
# MUTUALLY EXCLUSIVE classification, in the order the brief fixes. Order
# matters and is stated in the output: a row falls in the FIRST bucket it
# matches, so an earlier bucket absorbs overlapping evidence from a later one.
TIDE_ORDER = [
    "verb_only", "target_only",
    "loc_row", "loc_col", "loc_both",
    "loc_container_grid", "loc_anchor_b2", "loc_runtime",
    "multi_field", "structural",
]


def tide_bucket(r, anchor_first=False):
    """Which single error bucket this row belongs to, or None if correct."""
    if T(r["strict_match"]):
        return None
    if r.get("stage") != "validated":
        return "structural"

    p, t = obj(r, "pred_json"), obj(r, "truth_json")
    if p is None or t is None:
        return "structural"

    wrong = [f for f in ("verb", "target", "location")
             if field(p, f) != field(t, f)]
    if len(wrong) == 0:
        # strict_match false but all three fields agree -> modifiers or shape.
        return "structural"
    if len(wrong) > 1:
        return "multi_field"
    if wrong == ["verb"]:
        return "verb_only"
    if wrong == ["target"]:
        return "target_only"

    pl, tl = field(p, "location"), field(t, "location")
    pt, tt = loc_type(pl), loc_type(tl)

    if anchor_first and pl == "grid_B2" and tl != "grid_B2":
        return "loc_anchor_b2"

    if pt == "grid" and tt == "grid":
        rw = loc_row(pl) != loc_row(tl)
        cw = loc_col(pl) != loc_col(tl)
        if rw and cw:
            return "loc_both"
        if rw:
            return "loc_row"
        return "loc_col"
    if "runtime" in (pt, tt):
        return "loc_runtime"
    if {pt, tt} == {"grid", "container"}:
        return "loc_container_grid"
    if pl == "grid_B2" and tl != "grid_B2":
        return "loc_anchor_b2"
    return "loc_container_grid"


def per_command(rows, fn):
    """Mean over commands of the mean over that command's repeats."""
    by = collections.defaultdict(list)
    for r in rows:
        by[r["command_id"]].append(fn(r))
    if not by:
        return float("nan"), 0
    return sum(sum(v) / len(v) for v in by.values()) / len(by), len(by)


def cells_of(rows):
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["model"], r["strategy"])].append(r)
    return by


def p11_tide(rows):
    by = cells_of(rows)
    out = []
    for (model, strat), cell in sorted(by.items()):
        acc, ncmd = per_command(cell, lambda r: T(r["strict_match"]))
        for order_name, anchor_first in (("brief", False), ("anchor_first", True)):
            counts = collections.Counter()
            cmds = collections.defaultdict(lambda: collections.Counter())
            for r in cell:
                b = tide_bucket(r, anchor_first=anchor_first)
                if b:
                    cmds[r["command_id"]][b] += 1
            # per-command contribution: fraction of that command's repeats in
            # the bucket, averaged over commands -> accuracy points recoverable
            nrep = collections.Counter(r["command_id"] for r in cell)
            for cid, c in cmds.items():
                for b, k in c.items():
                    counts[b] += k / nrep[cid]
            for b in TIDE_ORDER:
                out.append(dict(model=model, strategy=strat, order=order_name,
                                n_commands=ncmd, action_match=round(acc, 4),
                                error_type=b,
                                contribution_pts=round(100.0 * counts[b] / ncmd, 2)))
    path = os.path.join(OUT, "tide_contributions.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print("P1.1 wrote %s (%d rows)" % (path, len(out)))
    return out


# ------------------------------------------------- P1.2 factor accuracy -----
def p12_factors(rows):
    by = cells_of(rows)
    out = []
    for (model, strat), cell in sorted(by.items()):
        rec = dict(model=model, strategy=strat)

        # loc_type over every command whose TRUTH has a location
        withloc = [r for r in cell if field(obj(r, "truth_json"), "location")]
        acc, n = per_command(withloc, lambda r: loc_type(field(obj(r, "pred_json"), "location"))
                             == loc_type(field(obj(r, "truth_json"), "location")))
        rec["n_with_location"] = n
        rec["loc_type_acc"] = round(acc, 4)

        # exact location, same denominator
        acc, _ = per_command(withloc, lambda r: field(obj(r, "pred_json"), "location")
                             == field(obj(r, "truth_json"), "location"))
        rec["location_exact_acc"] = round(acc, 4)

        # row / col over commands whose TRUTH is a grid cell
        grid = [r for r in cell if loc_type(field(obj(r, "truth_json"), "location")) == "grid"]
        rec["n_grid"] = len(set(r["command_id"] for r in grid))
        for fname, fgetter in (("row", loc_row), ("col", loc_col)):
            acc, _ = per_command(grid, lambda r, g=fgetter:
                                 g(field(obj(r, "pred_json"), "location"))
                                 == g(field(obj(r, "truth_json"), "location")))
            rec["%s_acc" % fname] = round(acc, 4)
        acc, _ = per_command(grid, lambda r:
                             field(obj(r, "pred_json"), "location")
                             == field(obj(r, "truth_json"), "location"))
        rec["grid_exact_acc"] = round(acc, 4)
        rec["row_minus_col_pts"] = round(100.0 * (rec["row_acc"] - rec["col_acc"]), 2)

        cont = [r for r in cell if loc_type(field(obj(r, "truth_json"), "location")) == "container"]
        rec["n_container"] = len(set(r["command_id"] for r in cont))
        acc, _ = per_command(cont, lambda r: field(obj(r, "pred_json"), "location")
                             == field(obj(r, "truth_json"), "location"))
        rec["container_acc"] = round(acc, 4)

        run = [r for r in cell if loc_type(field(obj(r, "truth_json"), "location")) == "runtime"]
        rec["n_runtime"] = len(set(r["command_id"] for r in run))
        acc, _ = per_command(run, lambda r: field(obj(r, "pred_json"), "location")
                             == field(obj(r, "truth_json"), "location"))
        rec["runtime_acc"] = round(acc, 4)

        for f in ("verb", "target"):
            sub = [r for r in cell if field(obj(r, "truth_json"), f)]
            acc, n = per_command(sub, lambda r, ff=f:
                                 field(obj(r, "pred_json"), ff) == field(obj(r, "truth_json"), ff))
            rec["%s_acc" % f] = round(acc, 4)
            rec["n_%s" % f] = n
        out.append(rec)

    path = os.path.join(OUT, "factor_accuracy.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print("P1.2 wrote %s (%d cells)" % (path, len(out)))
    return out


# ------------------------------------------------ P1.3 vocabulary buckets ---
FARNEAR = ("far", "near", "farthest", "nearest")
UPDOWN = ("upper", "bottom", "top", "lower")


def vocab_bucket(command, truth_loc):
    if truth_loc is None:
        return "nolocation"
    low = " %s " % " ".join(command.lower().replace(",", " ").split())
    has_fn = any(" %s " % w in low for w in FARNEAR)
    has_ud = any(" %s " % w in low for w in UPDOWN)
    if has_fn and not has_ud:
        return "farnear"
    if has_ud and not has_fn:
        return "updown"
    if has_fn and has_ud:
        return "both"
    return "neither"


def p13_vocab(rows):
    by = cells_of(rows)
    out = []
    for (model, strat), cell in sorted(by.items()):
        groups = collections.defaultdict(list)
        for r in cell:
            groups[vocab_bucket(r["command"], field(obj(r, "truth_json"), "location"))].append(r)
        for b in ("farnear", "updown", "both", "neither", "nolocation"):
            sub = groups.get(b, [])
            if not sub:
                continue
            acc, n = per_command(sub, lambda r: T(r["strict_match"]))
            out.append(dict(model=model, strategy=strat, bucket=b,
                            n_commands=n, action_match=round(acc, 4)))
    path = os.path.join(OUT, "vocab_buckets.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print("P1.3 wrote %s (%d rows)" % (path, len(out)))
    return out


# ----------------------------------------------- P1.4 confidence split ------
def p14_confidence():
    path = "audit/cls_margins.csv"
    with open(path, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    recs = []
    for r in rows:
        p, t = json.loads(r["pred_json"]), json.loads(r["truth_json"])
        wrong = {f: (p.get(k) != t.get(k))
                 for f, k in (("verb", "action"), ("target", "target"),
                              ("location", "location"))}
        # responsible head: the first wrong field, location last since it is
        # the bottleneck and we want its margin when it is the cause.
        head = None
        for f in ("verb", "target", "location"):
            if wrong[f]:
                head = f
                break
        recs.append(dict(command_id=int(r["command_id"]), command=r["command"],
                         head=head,
                         margin=float(r["%s_margin" % head]) if head else None,
                         any_wrong=any(wrong.values()),
                         pred=json.dumps(p), truth=json.dumps(t)))
    errs = [x for x in recs if x["any_wrong"]]
    print("P1.4 cls_margins rows=%d  errors=%d" % (len(recs), len(errs)))

    out = []
    for lo in range(10):
        a, b = lo / 10.0, (lo + 1) / 10.0
        inb = [x for x in recs
               if x["margin"] is not None and a <= x["margin"] < b] \
            if lo < 9 else [x for x in recs if x["margin"] is not None and x["margin"] >= a]
        # every row has a margin for each head; bucket ALL rows by the
        # location-head margin, which is the bottleneck head
        allb = [r for r in rows
                if a <= float(r["location_margin"]) < b] if lo < 9 else \
               [r for r in rows if float(r["location_margin"]) >= a]
        nerr = 0
        for r in allb:
            p, t = json.loads(r["pred_json"]), json.loads(r["truth_json"])
            if p.get("location") != t.get("location"):
                nerr += 1
        out.append(dict(decile="%.1f-%.1f" % (a, b), n_commands=len(allb),
                        location_errors=nerr,
                        location_error_rate=round(nerr / len(allb), 4) if allb else ""))
    hi = [r for r in rows if float(r["location_margin"]) > 0.8
          and json.loads(r["pred_json"]).get("location") != json.loads(r["truth_json"]).get("location")]
    lo_ = [r for r in rows if float(r["location_margin"]) < 0.3
           and json.loads(r["pred_json"]).get("location") != json.loads(r["truth_json"]).get("location")]
    out.append(dict(decile="SUMMARY margin>0.8 (systematic)", n_commands=len(hi),
                    location_errors=len(hi), location_error_rate=""))
    out.append(dict(decile="SUMMARY margin<0.3 (ambiguous)", n_commands=len(lo_),
                    location_errors=len(lo_), location_error_rate=""))
    p = os.path.join(OUT, "confidence_split.csv")
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print("P1.4 wrote %s" % p)

    hi.sort(key=lambda r: -float(r["location_margin"]))
    top = []
    for r in hi[:20]:
        p_, t_ = json.loads(r["pred_json"]), json.loads(r["truth_json"])
        top.append(dict(command_id=r["command_id"], command=r["command"],
                        location_margin=round(float(r["location_margin"]), 4),
                        pred_location=p_.get("location"),
                        truth_location=t_.get("location"),
                        pred_json=r["pred_json"], truth_json=r["truth_json"]))
    p2 = os.path.join(OUT, "confident_wrong_top20.csv")
    with open(p2, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(top[0]))
        w.writeheader()
        w.writerows(top)
    print("P1.4 wrote %s (%d of %d high-margin errors)" % (p2, len(top), len(hi)))
    return out, len(hi), len(lo_)


# ------------------------------------------ P1.5 confusion signatures -------
def p15_confusion(rows):
    by = cells_of(rows)
    out = []
    mats = {}
    for (model, strat), cell in sorted(by.items()):
        mat = collections.Counter()
        errs = 0
        sig = collections.Counter()
        for r in cell:
            tl = field(obj(r, "truth_json"), "location")
            pl = field(obj(r, "pred_json"), "location")
            if tl is None:
                continue
            mat[(tl, pl if pl else "<none>")] += 1
            if tl == pl:
                continue
            errs += 1
            tt, pt = loc_type(tl), loc_type(pl)
            if tt == "grid" and pt == "grid":
                rw = loc_row(pl) != loc_row(tl)
                cw = loc_col(pl) != loc_col(tl)
                mirrored = (loc_row(pl), loc_row(tl)) in (("A", "C"), ("C", "A"))
                if mirrored and not cw:
                    sig["axis_inversion_row"] += 1
                elif rw and not cw:
                    sig["row_only_other"] += 1
                elif cw and not rw:
                    sig["col_only"] += 1
                else:
                    sig["both_axes"] += 1
                if pl == "grid_B2":
                    sig["anchor_b2"] += 1
            elif {tt, pt} == {"grid", "container"}:
                sig["container_leakage"] += 1
            elif "runtime" in (tt, pt):
                sig["runtime_confusion"] += 1
            else:
                sig["other"] += 1
        mats[(model, strat)] = mat
        rec = dict(model=model, strategy=strat, location_errors=errs)
        for k in ("axis_inversion_row", "row_only_other", "col_only", "both_axes",
                  "anchor_b2", "container_leakage", "runtime_confusion", "other"):
            rec[k] = sig[k]
            rec["%s_frac" % k] = round(sig[k] / errs, 4) if errs else ""
        out.append(rec)
    path = os.path.join(OUT, "confusion_signatures.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print("P1.5 wrote %s (%d cells)  NOTE: pooled over repeats" % (path, len(out)))
    return out, mats


# ------------------------------------------- P1.6 calibration baseline ------
def ece(pairs, bins=10):
    """pairs = [(confidence, correct_bool)]"""
    if not pairs:
        return float("nan")
    tot = 0.0
    for i in range(bins):
        a, b = i / bins, (i + 1) / bins
        sub = [pc for pc in pairs if (a <= pc[0] < b or (i == bins - 1 and pc[0] >= a))]
        if not sub:
            continue
        conf = sum(c for c, _ in sub) / len(sub)
        acc = sum(1.0 for _, k in sub if k) / len(sub)
        tot += (len(sub) / len(pairs)) * abs(conf - acc)
    return tot


def brier(pairs):
    if not pairs:
        return float("nan")
    return sum((c - (1.0 if k else 0.0)) ** 2 for c, k in pairs) / len(pairs)


def p16_calibration():
    with open("audit/cls_margins.csv", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    out = []
    curves = {}
    for head, top1col, predkey, truthkey in (
            ("verb", "verb_top1", "action", "action"),
            ("target", "target_top1", "target", "target"),
            ("location", "location_top1", "location", "location")):
        pairs = []
        for r in rows:
            p, t = json.loads(r["pred_json"]), json.loads(r["truth_json"])
            if t.get(truthkey) is None:
                continue
            pairs.append((float(r[top1col]), p.get(predkey) == t.get(truthkey)))
        out.append(dict(head=head, n=len(pairs),
                        accuracy=round(sum(1 for _, k in pairs if k) / len(pairs), 4),
                        mean_confidence=round(sum(c for c, _ in pairs) / len(pairs), 4),
                        ECE=round(ece(pairs), 4), Brier=round(brier(pairs), 4)))
        curves[head] = pairs
    path = os.path.join(OUT, "calibration_baseline.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print("P1.6 wrote %s" % path)
    return out, curves


# ------------------------------------------------------------- figures ------
def figures(fa, tide, vocab, curves, mats):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("figures SKIPPED (matplotlib unavailable: %s)" % e)
        return

    # factor accuracy: row vs col per cell
    fa2 = sorted(fa, key=lambda d: d["row_acc"] - d["col_acc"])
    labels = ["%s/%s" % (d["model"].split(":", 1)[-1], d["strategy"]) for d in fa2]
    fig, ax = plt.subplots(figsize=(11, 9))
    y = range(len(fa2))
    ax.barh([i - 0.2 for i in y], [d["row_acc"] for d in fa2], height=0.4, label="row (A/B/C)")
    ax.barh([i + 0.2 for i in y], [d["col_acc"] for d in fa2], height=0.4, label="col (1/2/3)")
    ax.set_yticks(list(y)); ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("accuracy on grid-truth commands (per-command, warmup excluded)")
    ax.set_title("P1.2 row vs column accuracy per cell -- the factorization evidence")
    ax.legend(); ax.grid(axis="x", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "factor_accuracy.png"), dpi=140)
    plt.close(fig)

    # TIDE stacked, brief order only
    types = [t for t in TIDE_ORDER]
    cells = sorted(set((d["model"], d["strategy"]) for d in tide))
    idx = {(d["model"], d["strategy"], d["error_type"]): d["contribution_pts"]
           for d in tide if d["order"] == "brief"}
    fig, ax = plt.subplots(figsize=(13, 8))
    bottom = [0.0] * len(cells)
    for t in types:
        vals = [idx.get((m, s, t), 0.0) for m, s in cells]
        ax.bar(range(len(cells)), vals, bottom=bottom, label=t)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels(["%s/%s" % (m.split(":", 1)[-1], s) for m, s in cells],
                       rotation=90, fontsize=6)
    ax.set_ylabel("accuracy points recoverable")
    ax.set_title("P1.1 TIDE per-field contribution (mutually exclusive, brief order)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "tide_contributions.png"), dpi=140)
    plt.close(fig)

    # vocab heatmap
    buckets = ["farnear", "updown", "both", "neither", "nolocation"]
    cl = sorted(set((d["model"], d["strategy"]) for d in vocab))
    vi = {(d["model"], d["strategy"], d["bucket"]): d["action_match"] for d in vocab}
    data = [[vi.get((m, s, b), float("nan")) for b in buckets] for m, s in cl]
    fig, ax = plt.subplots(figsize=(7, 11))
    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(buckets))); ax.set_xticklabels(buckets, rotation=45)
    ax.set_yticks(range(len(cl)))
    ax.set_yticklabels(["%s/%s" % (m.split(":", 1)[-1], s) for m, s in cl], fontsize=6)
    ax.set_title("P1.3 action match by spatial-vocabulary bucket")
    fig.colorbar(im); fig.tight_layout()
    fig.savefig(os.path.join(FIG, "vocab_buckets.png"), dpi=140); plt.close(fig)

    # reliability
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, head in zip(axes, ("verb", "target", "location")):
        pairs = curves[head]
        xs, ys = [], []
        for i in range(10):
            a, b = i / 10.0, (i + 1) / 10.0
            sub = [pc for pc in pairs if a <= pc[0] < b or (i == 9 and pc[0] >= a)]
            if sub:
                xs.append(sum(c for c, _ in sub) / len(sub))
                ys.append(sum(1.0 for _, k in sub if k) / len(sub))
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.plot(xs, ys, "o-")
        ax.set_title("%s (ECE %.3f)" % (head, ece(pairs)))
        ax.set_xlabel("confidence"); ax.set_ylabel("accuracy")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(alpha=0.3)
    fig.suptitle("P1.6 reliability, existing flat classifier")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "calibration_baseline.png"), dpi=140)
    plt.close(fig)

    # confusion matrices
    locs = ["grid_A1", "grid_A2", "grid_A3", "grid_B1", "grid_B2", "grid_B3",
            "grid_C1", "grid_C2", "grid_C3", "box_1", "box_2", "box_3",
            "max_reach", "nearest_free_cell"]
    n = 0
    for (model, strat), mat in mats.items():
        data = [[mat.get((t, p), 0) for p in locs] for t in locs]
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.imshow(data, cmap="viridis")
        ax.set_xticks(range(len(locs))); ax.set_xticklabels(locs, rotation=90, fontsize=6)
        ax.set_yticks(range(len(locs))); ax.set_yticklabels(locs, fontsize=6)
        ax.set_xlabel("predicted"); ax.set_ylabel("truth")
        ax.set_title("%s / %s" % (model, strat), fontsize=8)
        fig.tight_layout()
        safe = ("%s_%s" % (model, strat)).replace(":", "-").replace("/", "-")
        fig.savefig(os.path.join(FIG, "confusion", "%s.png" % safe), dpi=120)
        plt.close(fig)
        n += 1
    print("figures: wrote factor_accuracy, tide_contributions, vocab_buckets, "
          "calibration_baseline and %d confusion matrices" % n)


def main():
    raw, dup, rows = load_rows()
    ncmd = len(set(r["command_id"] for r in rows))
    print("=" * 78)
    print("loaded %d raw rows from 4 scored CSVs; duplicates removed by 5-field key: %d"
          % (len(raw), dup))
    print("after dropping warmup: %d rows, %d distinct command_ids, %d cells"
          % (len(rows), ncmd, len(cells_of(rows))))
    ds = collections.Counter(r["dataset"] for r in rows)
    print("datasets: %s" % dict(ds))
    print("=" * 78)

    tide = p11_tide(rows)
    fa = p12_factors(rows)
    vocab = p13_vocab(rows)
    conf, nhi, nlo = p14_confidence()
    sigs, mats = p15_confusion(rows)
    cal, curves = p16_calibration()
    figures(fa, tide, vocab, curves, mats)

    print()
    print("=" * 78)
    print("P1.2 HEADLINE -- row vs column accuracy (grid-truth commands only)")
    print("%-28s %-11s %5s %8s %8s %9s" % ("model", "strategy", "n", "row", "col", "row-col"))
    for d in sorted(fa, key=lambda x: x["row_acc"] - x["col_acc"]):
        print("%-28s %-11s %5d %8.4f %8.4f %+8.2f" %
              (d["model"], d["strategy"], d["n_grid"], d["row_acc"], d["col_acc"],
               d["row_minus_col_pts"]))
    gaps = [d["row_minus_col_pts"] for d in fa]
    print("row-col gap: min %+.2f  median %+.2f  max %+.2f  (negative = row harder)"
          % (min(gaps), sorted(gaps)[len(gaps) // 2], max(gaps)))
    print("cells where row is >=10 pts worse than col: %d of %d"
          % (sum(1 for g in gaps if g <= -10), len(gaps)))
    print("=" * 78)
    print("P1.4 high-margin (>0.8) location errors: %d    low-margin (<0.3): %d" % (nhi, nlo))
    print("=" * 78)


if __name__ == "__main__":
    sys.exit(main())
