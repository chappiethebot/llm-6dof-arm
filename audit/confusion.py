"""Task 1 -- confusion matrices over the scored held_out_v3 CSVs. READ-ONLY.

Builds location (14x14), verb (7x7) and target (3x3) confusion matrices for
  (a) results_heldout_v3_cls_scored.csv    -- rung 4, the classifier
  (b) the 20 rung-1 cells pooled           -- 5 models x 4 prompting strategies
and answers the specific questions in the task.

Rows with stage != "validated" are excluded: a row the validator rejected has
no trustworthy predicted symbol to put in a cell. Counts reported.
"""
import csv, json, collections, os, sys

csv.field_size_limit(2**31 - 1)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "audit")
T = lambda v: v in ("True", "true", "1")

SCHEMA = json.load(open(os.path.join(ROOT, "schema", "schema_v2.json"), encoding="utf-8"))
LOCS = list(SCHEMA["locations"])            # 14, in schema order
VERBS = list(SCHEMA["verbs"])               # 7
TARGETS = list(SCHEMA["targets"])           # 3
ABSENT = "(absent)"
OTHER = "(other)"

GRID = [l for l in LOCS if l.startswith("grid_")]
ROW_OF = {l: l.split("_")[1][0] for l in GRID}     # grid_A1 -> A
COL_OF = {l: l.split("_")[1][1] for l in GRID}     # grid_A1 -> 1


def load(tag):
    p = os.path.join(ROOT, "bench", f"bench/results_heldout_v3_{tag}_scored.csv"
                     .replace("bench/", ""))
    d = {}
    with open(p, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            d[(r.get("dataset", ""), r["model"], r["strategy"],
               r["command_id"], r["run_index"])] = r
    return list(d.values())


def parse(s):
    try:
        return json.loads(s) if s else None
    except (json.JSONDecodeError, TypeError):
        return None


def field(obj, name, allowed):
    if not isinstance(obj, dict):
        return OTHER
    v = obj.get(name)
    if v is None:
        return ABSENT
    if isinstance(v, (dict, list)):
        return OTHER
    return v if v in allowed else OTHER


def build(rows, name, allowed, require_truth=True):
    """Return (matrix, n_used, per_model) keyed [truth][pred]."""
    m = collections.defaultdict(collections.Counter)
    per_model = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    n = 0
    for r in rows:
        t = parse(r.get("truth_json"))
        p = parse(r.get("pred_json"))
        tv = field(t, name, allowed)
        if require_truth and tv in (ABSENT, OTHER):
            continue
        pv = field(p, name, allowed)
        m[tv][pv] += 1
        per_model[r["model"]][tv][pv] += 1
        n += 1
    return m, n, per_model


def write_matrix(m, labels, path, title):
    cols = labels + [ABSENT, OTHER]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([f"truth\\pred ({title})"] + cols + ["row_total"])
        for t in labels:
            row = [m[t].get(c, 0) for c in cols]
            w.writerow([t] + row + [sum(row)])
    return path


def pct(a, b):
    return (a / b * 100) if b else 0.0


def report(tag, rows, fh):
    def out(s=""):
        print(s)
        fh.write(s + "\n")

    total = len(rows)
    val = [r for r in rows if r.get("stage") == "validated"]
    out(f"\n{'='*100}")
    out(f"{tag}")
    out(f"{'='*100}")
    out(f"rows: {total:,}   validated: {len(val):,}   "
        f"EXCLUDED (stage != validated): {total-len(val):,} "
        f"({pct(total-len(val), total):.1f}%)")
    stages = collections.Counter(r["stage"] for r in rows if r["stage"] != "validated")
    if stages:
        out(f"  excluded by stage: {dict(stages)}")

    loc, nloc, loc_pm = build(val, "location", LOCS)
    verb, nverb, verb_pm = build(val, "action", VERBS)
    tgt, ntgt, tgt_pm = build(val, "target", TARGETS)
    out(f"location-bearing rows used: {nloc:,} | verb rows: {nverb:,} | target rows: {ntgt:,}")

    slug = tag.split()[0].lower().replace(":", "")
    write_matrix(loc, LOCS, os.path.join(OUT, f"confusion_location_{slug}.csv"), "location")
    write_matrix(verb, VERBS, os.path.join(OUT, f"confusion_verb_{slug}.csv"), "verb")
    write_matrix(tgt, TARGETS, os.path.join(OUT, f"confusion_target_{slug}.csv"), "target")
    out(f"  -> audit/confusion_{{location,verb,target}}_{slug}.csv")

    # accuracy per head
    for label, m, n in (("location", loc, nloc), ("verb", verb, nverb), ("target", tgt, ntgt)):
        corr = sum(m[t].get(t, 0) for t in m)
        out(f"  {label:9} head accuracy: {corr:,}/{n:,} = {pct(corr,n):.1f}%")

    # ---- Q1: collapse onto grid_B2 ----
    out(f"\n-- Q1. COLLAPSE ONTO grid_B2 --")
    onto = {t: loc[t].get("grid_B2", 0) for t in LOCS if t != "grid_B2"}
    tot_onto = sum(onto.values())
    rt = {t: sum(loc[t].values()) for t in LOCS}
    b2_correct = loc["grid_B2"].get("grid_B2", 0)
    b2_total = rt.get("grid_B2", 0)
    out(f"   grid_B2 itself: {b2_correct}/{b2_total} correct ({pct(b2_correct,b2_total):.1f}%)")
    out(f"   OTHER symbols predicted as grid_B2: {tot_onto:,} rows")
    for t, c in sorted(onto.items(), key=lambda x: -x[1]):
        if c:
            out(f"     {t:20} {c:5}  = {pct(c, rt[t]):5.1f}% of its {rt[t]} rows")

    # ---- Q2: row inversion A<->C, split by phrasing ----
    out(f"\n-- Q2. ROW INVERSION (A<->C) --")
    ph = collections.Counter()
    for r in val:
        c = r.get("command", "").lower()
        for w in ("far", "near", "upper", "bottom", "top", "lower"):
            if f" {w} " in f" {c} ":
                ph[w] += 1
    out(f"   phrasing inventory in these rows: {dict(ph)}")
    out(f"   NOTE: held_out_v3 contains ZERO far/near commands (seen_v2 has 38/37,")
    out(f"   held_out_v3 has 0/0 and uses upper/bottom instead). The far-vs-upper")
    out(f"   split CANNOT be computed on this dataset: the train and eval splits are")
    out(f"   phrasing-disjoint on the row axis by design. Any inversion measured here")
    out(f"   is an UPPER/BOTTOM effect, not a far/near one.")
    def rowclass(cmd):
        c = cmd.lower()
        has_fn = any(w in c for w in ("far", "near"))
        has_ub = any(w in c for w in ("upper", "bottom", "top", "lower"))
        if has_fn and not has_ub: return "far/near"
        if has_ub and not has_fn: return "upper/bottom"
        if has_fn and has_ub:     return "both"
        return "neither"
    inv = collections.defaultdict(lambda: [0, 0])   # class -> [inverted, total]
    for r in val:
        t = parse(r.get("truth_json")); p = parse(r.get("pred_json"))
        tv = field(t, "location", LOCS); pv = field(p, "location", LOCS)
        if tv not in GRID or pv not in GRID:
            continue
        k = rowclass(r.get("command", ""))
        inv[k][1] += 1
        if {ROW_OF[tv], ROW_OF[pv]} == {"A", "C"}:
            inv[k][0] += 1
    for k in ("far/near", "upper/bottom", "both", "neither"):
        i, n = inv[k]
        if n:
            out(f"   {k:14} {i:5}/{n:5} = {pct(i,n):5.1f}% A<->C inverted")

    # ---- Q3: column inversion 1<->3, per row ----
    out(f"\n-- Q3. COLUMN INVERSION (1<->3) --")
    col = collections.defaultdict(lambda: [0, 0])
    for r in val:
        t = parse(r.get("truth_json")); p = parse(r.get("pred_json"))
        tv = field(t, "location", LOCS); pv = field(p, "location", LOCS)
        if tv not in GRID or pv not in GRID:
            continue
        col["ALL"][1] += 1
        col[f"row {ROW_OF[tv]}"][1] += 1
        if {COL_OF[tv], COL_OF[pv]} == {"1", "3"}:
            col["ALL"][0] += 1
            col[f"row {ROW_OF[tv]}"][0] += 1
    for k in ("ALL", "row A", "row B", "row C"):
        i, n = col[k]
        if n:
            out(f"   {k:14} {i:5}/{n:5} = {pct(i,n):5.1f}% col 1<->3 swapped")

    # ---- Q4: any other pair > 5% of that symbol's rows ----
    out(f"\n-- Q4. OTHER CONFUSION PAIRS > 5% OF THE TRUTH SYMBOL'S ROWS --")
    named = {("grid_B2",)}
    found = []
    for t in LOCS:
        n = rt.get(t, 0)
        if not n:
            continue
        for p, c in loc[t].items():
            if p == t or not c:
                continue
            share = pct(c, n)
            if share > 5.0:
                found.append((share, t, p, c, n))
    for share, t, p, c, n in sorted(found, reverse=True):
        flag = ""
        if p == "grid_B2":
            flag = "  [Q1 B2-collapse]"
        elif t in GRID and p in GRID and {ROW_OF[t], ROW_OF[p]} == {"A", "C"}:
            flag = "  [Q2 row inversion]"
        elif t in GRID and p in GRID and {COL_OF[t], COL_OF[p]} == {"1", "3"}:
            flag = "  [Q3 col inversion]"
        out(f"   {t:20} -> {p:20} {c:5} = {share:5.1f}% of {n}{flag}")
    if not found:
        out("   (none)")

    return loc, loc_pm, rt


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "confusion_summary.txt"), "w", encoding="utf-8") as fh:
        cls_rows = load("cls")
        r1_rows = load("rung1")
        report("classifier:distilbert (rung 4)", cls_rows, fh)
        loc_r1, loc_pm, rt_r1 = report("rung-1 pooled (20 cells, 5 models x 4 strategies)",
                                       r1_rows, fh)

        # cross-model check for the three named effects
        fh.write("\n")
        print(f"\n{'='*100}\nPER-MODEL: is each effect general or classifier-specific?\n{'='*100}")
        fh.write(f"\n{'='*100}\nPER-MODEL: is each effect general or classifier-specific?\n{'='*100}\n")
        val_all = [r for r in (cls_rows + r1_rows) if r.get("stage") == "validated"]
        pm = collections.defaultdict(lambda: collections.Counter())
        for r in val_all:
            t = parse(r.get("truth_json")); p = parse(r.get("pred_json"))
            tv = field(t, "location", LOCS); pv = field(p, "location", LOCS)
            if tv not in LOCS:
                continue
            m = r["model"]
            pm[m]["n"] += 1
            if pv == "grid_B2" and tv != "grid_B2":
                pm[m]["b2"] += 1
            if tv in GRID and pv in GRID:
                pm[m]["grid_pairs"] += 1
                if {ROW_OF[tv], ROW_OF[pv]} == {"A", "C"}:
                    pm[m]["rowinv"] += 1
                if {COL_OF[tv], COL_OF[pv]} == {"1", "3"}:
                    pm[m]["colinv"] += 1
        hdr = f"{'model':26}{'n':>7}{'->B2':>8}{'%':>7}{'rowinv':>8}{'%':>7}{'colinv':>8}{'%':>7}"
        print(hdr); fh.write(hdr + "\n")
        for m in sorted(pm, key=lambda x: -pct(pm[x]['b2'], pm[x]['n'])):
            c = pm[m]
            line = (f"{m:26}{c['n']:>7}{c['b2']:>8}{pct(c['b2'],c['n']):>7.1f}"
                    f"{c['rowinv']:>8}{pct(c['rowinv'],c['grid_pairs']):>7.1f}"
                    f"{c['colinv']:>8}{pct(c['colinv'],c['grid_pairs']):>7.1f}")
            print(line); fh.write(line + "\n")
    print(f"\nwrote audit/confusion_summary.txt")


if __name__ == "__main__":
    main()
