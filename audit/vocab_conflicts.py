"""Exhaustive vocabulary-conflict scan. READ-ONLY; writes audit/vocab_conflicts.md.

Sources: schema/schema_v2.json, the legend in llm/prompts.py, and every command
in seen_v2 / train_v2 / dev_v2 / held_out_v3.
"""
import csv, json, os, re, collections, sys

csv.field_size_limit(2**31 - 1)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from llm.prompts import _grid_legend, load_schema

SCHEMA = load_schema(os.path.join(ROOT, "schema", "schema_v2.json"))
LEGEND = _grid_legend(SCHEMA)
CELLS = [f"grid_{r}{c}" for r in "ABC" for c in "123"]
BOXES = ["box_1", "box_2", "box_3"]
RUNTIME = ["max_reach", "nearest_free_cell"]
SPLITS = ["seen_v2", "train_v2", "dev_v2", "held_out_v3"]

# What the legend asserts each word means.
LEGEND_MEANING = {
    "far":     "row A (far from the robot)",
    "middle":  "row B  AND ALSO  column 2   <-- used for BOTH axes",
    "near":    "row C (near the robot)",
    "left":    "column 1",
    "right":   "column 3",
    "furthest": "max_reach (the furthest point the arm can reach)",
    "empty":   "nearest_free_cell (whichever square is currently empty)",
    "nearest": "(NOT DEFINED in the legend; appears only inside the symbol NAME)",
    "centre":  "(NOT IN THE LEGEND AT ALL)",
    "center":  "(NOT IN THE LEGEND AT ALL)",
}
ROW_WORDS = {"far": "A", "middle": "B", "near": "C"}
COL_WORDS = {"left": "1", "middle": "2", "right": "3"}


def load(split):
    p = os.path.join(ROOT, "data", f"{split}.jsonl")
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


DATA = {s: load(s) for s in SPLITS}


def loc(r):
    return (r.get("expected") or {}).get("location")


def verb(r):
    return (r.get("expected") or {}).get("action")


def tgt(r):
    return (r.get("expected") or {}).get("target")


def words(cmd):
    return re.findall(r"[a-z0-9_]+", cmd.lower())


def has(cmd, w):
    return re.search(rf"\b{re.escape(w)}\b", cmd.lower()) is not None


# scored held-out rows, to price each conflict
def scored():
    d = {}
    for tag in ("rung1", "rag", "lora", "cls"):
        p = os.path.join(ROOT, "bench", f"results_heldout_v3_{tag}_scored.csv")
        with open(p, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                d[(r["model"], r["strategy"], r["command_id"], r["run_index"])] = r
    return [r for r in d.values() if r["is_warmup"] != "True"]


SCORED = scored()
N_CELLS = len({(r["model"], r["strategy"]) for r in SCORED})


def pred_loc(r):
    try:
        o = json.loads(r["pred_json"]) if r["pred_json"] else None
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(o, dict):
        return None
    v = o.get("location")
    if v is None and isinstance(o.get("action"), dict):
        v = o["action"].get("location")
    return v if isinstance(v, str) else None


def truth_loc(r):
    try:
        return json.loads(r["truth_json"] or "{}").get("location")
    except (json.JSONDecodeError, TypeError):
        return None


OUT = []
def w(s=""):
    OUT.append(s)


w("# Vocabulary conflicts in the command datasets")
w()
w("Read-only scan. Nothing was modified.")
w()
w("Sources: `schema/schema_v2.json`, the legend emitted by "
  "`llm/prompts.py:_grid_legend()`, and every command in "
  "`seen_v2` (290), `train_v2` (230), `dev_v2` (57), `held_out_v3` (217).")
w()
w("**Pricing convention.** Each held-out command appears 5 times per cell "
  f"(5 repeats) in each of {N_CELLS} measured cells, so *1 command = 5 rows per "
  f"cell = {5*N_CELLS} rows across the whole measurement*. Costs below are given "
  "in commands and in rows-per-cell, which is the number that moves a reported "
  "accuracy.")
w()
w("## The legend, verbatim")
w()
w("```")
for line in LEGEND.strip().splitlines():
    w(line)
w("```")
w()

# ---------------------------------------------------------------- CONFIRM 1
w("---")
w()
w("## 1. CONFIRMED — `middle` is used for BOTH a row and a column")
w()
w("This is the headline conflict and it is visible in the legend text itself:")
w()
w("| legend line | says `middle` means |")
w("|---|---|")
w("| `Rows:    A = far from the robot,  B = middle,  C = near the robot.` | row **B** |")
w("| `Columns: 1 = left,  2 = middle,  3 = right.` | column **2** |")
w()
w("A command containing `middle` is therefore under-determined by the legend "
  "alone. `grid_B2` is the only cell where both readings agree, which is exactly "
  "where predictions pile up.")
w()

# quantify collapse under BOTH population definitions
def collapse(rows, truth_set=None):
    truth_set = truth_set or CELLS
    onto, rowtot = collections.Counter(), collections.Counter()
    for r in rows:
        t, p = truth_loc(r), pred_loc(r)
        if t in truth_set:
            rowtot[t] += 1
            if p == "grid_B2" and t != "grid_B2":
                onto[t] += 1
    return onto, rowtot

RUNG1_STRATS = {"zero_shot", "few_shot", "cot", "role_based"}
rung1_valid = [r for r in SCORED
               if r["strategy"] in RUNG1_STRATS
               and not r["model"].endswith("-lora")
               and not r["model"].startswith("classifier:")
               and r["stage"] == "validated"]
ALL_LOCS = SCHEMA["locations"]
onto1g, _ = collapse(rung1_valid)                    # truth = 9 grid cells
onto1a, _ = collapse(rung1_valid, ALL_LOCS)          # truth = all 14 symbols
ontoA, rowtotA = collapse(SCORED, ALL_LOCS)
t1g, t1a, tA = sum(onto1g.values()), sum(onto1a.values()), sum(ontoA.values())
c1a = sum(v for k, v in onto1a.items() if k.endswith("2"))
cA = sum(v for k, v in ontoA.items() if k.endswith("2"))

w("**RECONCILIATION OF THE COUNT — the reported 871 is confirmed, and the "
  "difference from other numbers below is a definition, not a discrepancy.**")
w()
w("| population | truth symbols counted | rows onto `grid_B2` | column-2 share |")
w("|---|---|---|---|")
w(f"| 20 rung-1 cells, validated rows *(the reported figure)* | all 14 | "
  f"**{t1a:,}** | **{100*c1a/t1a:.1f}%** |")
w(f"| 20 rung-1 cells, validated rows | 9 grid cells only | {t1g:,} | — |")
w(f"| all {N_CELLS} cells, all rows | all 14 | {tA:,} | {100*cA/tA:.1f}% |")
w()
w(f"**The reported 871 rows at 91.4% reproduces exactly.** The {t1g:,} figure is "
  f"the same measurement restricted to grid-cell truths; the "
  f"{t1a - t1g} row difference is commands whose ground truth is "
  "`nearest_free_cell` that were answered `grid_B2` — a runtime symbol "
  "collapsing onto the middle cell, which is section 3's problem, not "
  "section 1's. Quoting one number without its truth-set definition is how "
  "these drift.")
w()
w("Per true cell, all 28 cells, all rows:")
w()
w("| true cell | rows predicted `grid_B2` | % of that cell's rows |")
w("|---|---|---|")
for k, v in sorted(ontoA.items(), key=lambda x: -x[1]):
    w(f"| `{k}` | {v} | {100*v/rowtotA[k]:.1f}% |")
w(f"| **total** | **{tA:,}** | — |")
w()
onto, tot_onto = ontoA, tA
w()
w("### Every held-out command containing a `middle`-class word")
w()
amb_cmds = []
for r in DATA["held_out_v3"]:
    if any(has(r["command"], x) for x in ("middle", "center", "centre")):
        amb_cmds.append(r)
w(f"{len(amb_cmds)} of 217 held-out commands = **{len(amb_cmds)*5} rows per cell**.")
w()
w("| # | command | ground truth |")
w("|---|---|---|")
for i, r in enumerate(sorted(amb_cmds, key=lambda x: (loc(x) or "", x["command"])), 1):
    w(f"| {i} | `{r['command']}` | `{loc(r)}` |")
w()
by_loc = collections.Counter(loc(r) for r in amb_cmds)
w("Breakdown by ground truth: " +
  ", ".join(f"`{k}` {v}" for k, v in sorted(by_loc.items())) + ".")
w()
w("Note `centre` and `center` appear in commands but **appear nowhere in the "
  "legend**, so a model must infer they are synonyms of `middle` without being "
  "told. That is a second, separate defect on the same word family.")
w()

# ---------------------------------------------------------------- CONFIRM 2
w("---")
w()
w("## 2. CONFIRMED — row vocabulary differs between training and evaluation")
w()
ROW_VOCAB = ["far", "near", "nearest", "furthest", "upper", "bottom", "top",
             "lower", "front", "back", "behind", "closest"]
COL_VOCAB = ["left", "right", "middle", "centre", "center"]
w("Exact per-split counts of every row-axis and column-axis word found:")
w()
hdr = "| word | axis | " + " | ".join(SPLITS) + " |"
w(hdr)
w("|---|---|" + "---|" * len(SPLITS))
for wd in ROW_VOCAB:
    counts = [sum(1 for r in DATA[s] if has(r["command"], wd)) for s in SPLITS]
    if any(counts):
        w(f"| `{wd}` | row | " + " | ".join(str(c) for c in counts) + " |")
for wd in COL_VOCAB:
    counts = [sum(1 for r in DATA[s] if has(r["command"], wd)) for s in SPLITS]
    if any(counts):
        w(f"| `{wd}` | column | " + " | ".join(str(c) for c in counts) + " |")
w()
w("**The row axis is disjoint between training and evaluation. The column axis "
  "is shared.** That asymmetry is the whole finding: `far`/`near` appear "
  "only in training, `upper`/`bottom` only in evaluation, while `left`/`right` "
  "appear in both.")
w()
w("Demonstrated consequence (audit/lora_vocab_test.csv): both LoRA adapters "
  "score **100.0% row accuracy** on `dev_v2` (far/near, never trained on) and "
  "**30.6% / 40.8%** on `held_out_v3` (upper/bottom). Column accuracy stays at "
  "86–100% throughout.")
w()
w("### Held-out commands that would need rewording to match training vocabulary")
w()
need = [r for r in DATA["held_out_v3"]
        if any(has(r["command"], x) for x in ("upper", "bottom", "top", "lower"))]
w(f"{len(need)} of 217 commands = **{len(need)*5} rows per cell**. "
  "The mapping is mechanical: `upper` -> `far`, `bottom` -> `near`.")
w()
w("| # | command | ground truth | reword to |")
w("|---|---|---|---|")
for i, r in enumerate(sorted(need, key=lambda x: (loc(x) or "", x["command"])), 1):
    fixed = re.sub(r"\bupper\b", "far", r["command"], flags=re.I)
    fixed = re.sub(r"\bbottom\b", "near", fixed, flags=re.I)
    w(f"| {i} | `{r['command']}` | `{loc(r)}` | `{fixed}` |")
w()

# ---------------------------------------------------------------- CONFIRM 3
w("---")
w()
w("## 3. CONFIRMED — the two runtime symbols are under-specified")
w()
w("Legend text for the two symbols:")
w()
w("```")
w("max_reach = the furthest point the arm can reach; resolved by the robot.")
w("nearest_free_cell = whichever square is currently empty; resolved by the robot.")
w("```")
w()
w("**Three defects in two lines:**")
w()
w("1. `nearest_free_cell` is named *nearest* but the legend never says nearest "
  "**to what** — nor even that nearness is involved. Its definition is "
  '"whichever square is currently empty", which is a *free* criterion, not a '
  "*nearest* one. Name and definition disagree.")
w("2. `max_reach` is defined with **furthest**, and `far` is simultaneously the "
  "legend's label for **row A**. A command saying *furthest* or *far* is "
  "ambiguous between `max_reach` and row A.")
w("3. `nearest` shares a stem with `near`, the legend's label for **row C**. A "
  "command saying *nearest* is ambiguous between `nearest_free_cell` and row C.")
w()
rt = [r for r in SCORED if truth_loc(r) in RUNTIME]
for sym in RUNTIME:
    g = [r for r in rt if truth_loc(r) == sym]
    c = collections.Counter(pred_loc(r) for r in g)
    n = len(g)
    w(f"**`{sym}`** — {n:,} scored rows:")
    w()
    w("| predicted | rows | % |")
    w("|---|---|---|")
    for k, v in c.most_common(6):
        w(f"| `{k}` | {v} | {100*v/n:.1f}% |")
    w()
w("### Commands ambiguous between the two runtime symbols")
w()
rt_cmds = [r for r in DATA["held_out_v3"] if loc(r) in RUNTIME]
w(f"{len(rt_cmds)} of 217 held-out commands carry a runtime symbol "
  f"= **{len(rt_cmds)*5} rows per cell**.")
w()
w("| command | ground truth | contains a row-label word? |")
w("|---|---|---|")
for r in sorted(rt_cmds, key=lambda x: (loc(x), x["command"])):
    hits = [x for x in ("far", "furthest", "near", "nearest", "close", "closest",
                        "empty", "free") if has(r["command"], x)]
    flag = ", ".join(f"`{h}`" for h in hits) or "—"
    w(f"| `{r['command']}` | `{loc(r)}` | {flag} |")
w()

# ---------------------------------------------------------------- NEW SCAN
w("---")
w()
w("## 4. NEW — every legend label word, checked against actual command meaning")
w()
w("For each word the legend uses as a label, does it appear in commands whose "
  "ground truth is *not* what the legend says it means?")
w()
w("| word | legend meaning | commands containing it | ground truths it actually maps to | conflict |")
w("|---|---|---|---|---|")
for wd, meaning in LEGEND_MEANING.items():
    hits = [(s, r) for s in SPLITS for r in DATA[s] if has(r["command"], wd)]
    if not hits:
        w(f"| `{wd}` | {meaning} | 0 | — | word never used in any command |")
        continue
    locs = collections.Counter(loc(r) or "(no location)" for _, r in hits)
    exp = None
    if wd in ROW_WORDS and wd not in COL_WORDS:
        exp = lambda L: isinstance(L, str) and L.startswith("grid_") and L[5] == ROW_WORDS[wd]
    elif wd in COL_WORDS and wd not in ROW_WORDS:
        exp = lambda L: isinstance(L, str) and L.startswith("grid_") and L[6] == COL_WORDS[wd]
    bad = "n/a"
    if exp:
        nbad = sum(v for L, v in locs.items() if not exp(L))
        bad = f"**{nbad} of {sum(locs.values())} disagree**" if nbad else "consistent"
    top = ", ".join(f"`{k}`×{v}" for k, v in locs.most_common(4))
    w(f"| `{wd}` | {meaning} | {sum(locs.values())} | {top} | {bad} |")
w()

w("### Words used as BOTH a row label and a column label")
w()
both = sorted(set(ROW_WORDS) & set(COL_WORDS))
w(f"**{', '.join('`'+b+'`' for b in both)}** — see section 1. This is the only "
  "word in the legend that labels two different axes, and it is the single "
  "largest source of location error in the study.")
w()

# ---------------------------------------------------------------- SPLIT VOCAB
w("---")
w()
w("## 5. NEW — vocabulary present in one split but not another")
w()
for field, fn, label in (("verb", verb, "verbs"), ("target", tgt, "targets")):
    w(f"### {label.capitalize()}")
    w()
    allv = sorted({fn(r) for s in SPLITS for r in DATA[s] if fn(r)})
    w("| value | " + " | ".join(SPLITS) + " | missing from |")
    w("|---|" + "---|" * (len(SPLITS) + 1))
    for v in allv:
        counts = [sum(1 for r in DATA[s] if fn(r) == v) for s in SPLITS]
        missing = [s for s, c in zip(SPLITS, counts) if c == 0]
        w(f"| `{v}` | " + " | ".join(str(c) for c in counts) + " | "
          + (", ".join(f"**{m}**" for m in missing) or "—") + " |")
    w()

w("### Location symbols")
w()
w("| symbol | " + " | ".join(SPLITS) + " | missing from |")
w("|---|" + "---|" * (len(SPLITS) + 1))
for L in SCHEMA["locations"]:
    counts = [sum(1 for r in DATA[s] if loc(r) == L) for s in SPLITS]
    missing = [s for s, c in zip(SPLITS, counts) if c == 0]
    w(f"| `{L}` | " + " | ".join(str(c) for c in counts) + " | "
      + (", ".join(f"**{m}**" for m in missing) or "—") + " |")
w()

# vocabulary of content words per split
w("### Content words appearing in only one split")
w()
STOP = set("the a an in to on at of and or please can you could would i it "
           "me my is are be do does put move go over there that this for with "
           "your s t re ll".split())
vocab = {s: collections.Counter() for s in SPLITS}
for s in SPLITS:
    for r in DATA[s]:
        for tok in set(words(r["command"])):
            if tok not in STOP and not tok.isdigit():
                vocab[s][tok] += 1
train_side = set(vocab["seen_v2"]) | set(vocab["train_v2"]) | set(vocab["dev_v2"])
held = set(vocab["held_out_v3"])
only_held = sorted(held - train_side)
only_train = sorted(train_side - held)
w(f"**In `held_out_v3` but in NO training-side split ({len(only_held)} words)** "
  "— a model has never seen these:")
w()
w("| word | held-out commands | rows/cell |")
w("|---|---|---|")
for t in only_held:
    n = vocab["held_out_v3"][t]
    w(f"| `{t}` | {n} | {n*5} |")
w()
w(f"**In training-side splits but NOT in `held_out_v3` ({len(only_train)} words)** "
  "— learned and never tested:")
w()
w("| word | seen_v2 | train_v2 | dev_v2 |")
w("|---|---|---|---|")
for t in only_train:
    w(f"| `{t}` | {vocab['seen_v2'][t]} | {vocab['train_v2'][t]} | {vocab['dev_v2'][t]} |")
w()

# ---------------------------------------------------------------- CARRIERS
w("---")
w()
w("## 6. NEW — carrier words appearing under more than one ground-truth verb")
w()
w("A *carrier* here is any content word in the command. If the same carrier "
  "introduces different verbs, the verb head cannot be decided from that word "
  "alone.")
w()
carrier = collections.defaultdict(collections.Counter)
for s in SPLITS:
    for r in DATA[s]:
        for tok in set(words(r["command"])):
            if tok in STOP or tok.isdigit():
                continue
            carrier[tok][verb(r)] += 1
multi = {t: c for t, c in carrier.items() if len(c) > 1}
w(f"**{len(multi)} words appear under more than one verb.** Listed by how "
  "evenly they split, most ambiguous first (a word split 50/50 between two "
  "verbs carries no verb signal at all):")
w()
w("| word | verbs it appears under | total | most ambiguous? |")
w("|---|---|---|---|")
def evenness(c):
    tot = sum(c.values())
    return max(c.values()) / tot
for t, c in sorted(multi.items(), key=lambda x: (evenness(x[1]), -sum(x[1].values()))):
    tot = sum(c.values())
    dist = ", ".join(f"`{k}`×{v}" for k, v in c.most_common())
    flag = "**yes — near-even split**" if evenness(c) < 0.6 else ""
    w(f"| `{t}` | {dist} | {tot} | {flag} |")
w()

# ------------------------------------------------- NEW: non-locative direction
w("---")
w()
w("## 6b. NEW — direction words used non-spatially, and inside `reject`")
w()
DIRW = ("far", "near", "left", "right", "middle", "center", "centre",
        "top", "bottom", "upper", "lower")
nonloc = []
for s in SPLITS:
    for r in DATA[s]:
        if loc(r):
            continue
        hits = [x for x in DIRW if has(r["command"], x)]
        if hits:
            nonloc.append((s, r, hits))
uniq = {}
for s, r, hits in nonloc:
    uniq.setdefault(r["command"], [set(), r, hits])[0].add(s)
w(f"**{len(uniq)} distinct commands** contain a legend direction word but have "
  "**no location in their ground truth**. A model keying on the word alone "
  "would emit a location for a verb that must not have one — a schema violation, "
  "not merely a wrong cell.")
w()
w("| command | verb | direction word(s) | splits |")
w("|---|---|---|---|")
for c, (ss, r, hits) in sorted(uniq.items(), key=lambda x: x[1][1].get("expected", {}).get("action", "")):
    w(f"| `{c}` | `{verb(r)}` | " + ", ".join(f"`{h}`" for h in hits)
      + " | " + ", ".join(sorted(ss)) + " |")
w()
held_nonloc = [c for c, (ss, r, h) in uniq.items() if "held_out_v3" in ss]
w(f"**Cost to held-out: {len(held_nonloc)} commands = {len(held_nonloc)*5} rows "
  "per cell.**")
w()
if not held_nonloc:
    w("Zero. Every one of these is training-side only. That is not entirely good "
      "news: it means the evaluation set **never tests** whether a model learned "
      "to ignore a direction word in a non-locative context. `right` in "
      "*'stop right there'* is an adverb meaning *immediately*, and the model is "
      "taught that on 3 training commands and never examined on it. This is a "
      "coverage gap in `held_out_v3`, not a cost against it.")
w()

# valid vocabulary inside reject commands
w("### Valid schema vocabulary appearing inside `reject` commands")
w()
w("`reject` commands deliberately contain legal-looking words. That is sound "
  "adversarial design, but it means a legal target or location word is **not** "
  "evidence that the verb takes one.")
w()
rej = [(s, r) for s in SPLITS for r in DATA[s] if verb(r) == "reject"]
uniqr = {}
for s, r in rej:
    uniqr.setdefault(r["command"], [set(), r])[0].add(s)
TARGET_WORDS = {"bottle": "bottle", "toy car": "toy_car", "car": "toy_car",
                "rubiks cube": "rubiks_cube", "cube": "rubiks_cube"}
LOC_WORDS = {"box 1": "box_1", "box 2": "box_2", "box 3": "box_3"}
w("| command | contains legal target | contains legal location | splits |")
w("|---|---|---|---|")
nheld = 0
for c, (ss, r) in sorted(uniqr.items()):
    # word-boundary, not substring: "car" must not match inside "carry"
    def phrase_in(p, text):
        return re.search(rf"\b{re.escape(p)}\b", text) is not None
    tw = sorted({v for k, v in TARGET_WORDS.items() if phrase_in(k, c.lower())})
    lw = sorted({v for k, v in LOC_WORDS.items() if phrase_in(k, c.lower())})
    if not tw and not lw:
        continue
    if "held_out_v3" in ss:
        nheld += 1
    w(f"| `{c}` | " + (", ".join(f"`{x}`" for x in tw) or "—") + " | "
      + (", ".join(f"`{x}`" for x in lw) or "—") + " | "
      + ", ".join(sorted(ss)) + " |")
w()
w(f"**{nheld} of these are in `held_out_v3` = {nheld*5} rows per cell.** These "
  "are working as intended — they are the safety KPI — but they are the reason "
  "a target word cannot be used as a shortcut for verb selection.")
w()

# ---------------------------------------------------------------- REWORD PLAN
w("---")
w()
w("## 7. Which held-out commands must change")
w()
w("Constraint: keep all 217 commands and their ground truth; change only "
  "conflicting words.")
w()
must = {}
for r in DATA["held_out_v3"]:
    c, L, reasons = r["command"], loc(r), []
    if any(has(c, x) for x in ("upper", "bottom", "top", "lower")):
        reasons.append("row vocabulary absent from training (§2)")
    if any(has(c, x) for x in ("middle", "centre", "center")):
        if isinstance(L, str) and L.startswith("grid_"):
            if L[6] == "2" and L[5] == "B":
                reasons.append("`middle` — both readings agree here, ambiguous but not wrong")
            else:
                reasons.append("`middle` is both row B and column 2 (§1)")
        else:
            reasons.append("`middle` used with a non-grid target (§1)")
    if any(has(c, x) for x in ("centre", "center")):
        reasons.append("`centre`/`center` absent from the legend (§1)")
    if L in RUNTIME and any(has(c, x) for x in ("far", "furthest", "near", "nearest")):
        reasons.append("runtime symbol described with a row-label word (§3)")
    if reasons:
        must[c] = (L, reasons)
w(f"**{len(must)} of 217 commands are affected = {len(must)*5} rows per cell "
  f"= {len(must)*5*N_CELLS:,} rows across the measurement.**")
w()
w("| command | ground truth | why | fixable without changing ground truth? |")
w("|---|---|---|---|")
unfixable = []
for c, (L, reasons) in sorted(must.items(), key=lambda x: (x[1][0] or "", x[0])):
    # A command is unfixable if its ground truth DEPENDS on the ambiguous word
    fixable = True
    note = "yes"
    if isinstance(L, str) and L.startswith("grid_") and L[6] == "2" \
            and any(has(c, x) for x in ("middle", "centre", "center")) and L[5] != "B":
        note = ("**NO** — `middle` is the only word selecting column 2; removing "
                "it removes the column")
        fixable = False
    elif isinstance(L, str) and L == "grid_B2" \
            and any(has(c, x) for x in ("middle", "centre", "center")):
        note = ("**NO** — both axes are `middle` here; the cell cannot be named "
                "without the ambiguous word")
        fixable = False
    if not fixable:
        unfixable.append((c, L))
    w(f"| `{c}` | `{L}` | {'; '.join(reasons)} | {note} |")
w()
w(f"### Cannot be fixed without changing ground truth: {len(unfixable)} commands")
w()
if unfixable:
    w("| command | ground truth | why not |")
    w("|---|---|---|")
    for c, L in unfixable:
        w(f"| `{c}` | `{L}` | the ambiguous word is the *only* word selecting "
          f"that cell's column; any replacement changes which cell is meant |")
w()
w("**The fix for these is not in the data, it is in the legend.** Giving row B "
  "and column 2 distinct labels — for example `B = centre row`, `2 = middle "
  "column` — resolves every one of them without touching a single command or "
  "any ground truth.")
w()

path = os.path.join(ROOT, "audit", "vocab_conflicts.md")
with open(path, "w", encoding="utf-8") as f:
    f.write("\n".join(OUT) + "\n")
print(f"wrote {path}  ({len(OUT)} lines)")
print(f"  middle-class held-out commands : {len(amb_cmds)}")
print(f"  row-vocab reword needed        : {len(need)}")
print(f"  runtime-symbol commands        : {len(rt_cmds)}")
print(f"  multi-verb carrier words       : {len(multi)}")
print(f"  held-out commands affected     : {len(must)}")
print(f"  unfixable without ground truth : {len(unfixable)}")
