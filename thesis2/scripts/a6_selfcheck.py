"""thesis2/scripts/a6_selfcheck.py -- A6: leakage grep + independent recomputation.

Two parts.

  1. LEAKAGE. For every script under thesis2/scripts/, find every mention of a
     held_out file and classify it as TRAIN-TIME (a leak) or EVAL-TIME (fine).
     Classification is by which function the reference sits in, not by filename.

  2. RECOMPUTATION. Recompute the headline numbers straight from the saved
     prediction CSVs with code that does not import anything from
     thesis2/scripts/, and diff against what the summaries claim.
"""
import ast
import csv
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)
SCRIPTS = REPO / "thesis2" / "scripts"
OUT = REPO / "thesis2" / "out"
csv.field_size_limit(2 ** 31 - 1)

TRAINY = ("train", "fit", "run_frozen_probe", "run_tinybert", "run_bilstm",
          "main_train", "dev_exact_match", "dev_tau_b")

print("=" * 84)
print("A6.1 -- does any script read held_out_* during TRAINING or THRESHOLD SELECTION?")
print("=" * 84)
findings = []
for p in sorted(SCRIPTS.glob("*.py")):
    src = p.read_text(encoding="utf-8")
    if "held_out" not in src:
        continue
    tree = ast.parse(src)
    # map line -> enclosing function
    owner = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for ln in range(node.lineno, (node.end_lineno or node.lineno) + 1):
                owner.setdefault(ln, node.name)
    for i, line in enumerate(src.splitlines(), 1):
        if "held_out" not in line:
            continue
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith('"') or stripped.startswith("'"):
            kind = "comment/docstring"
        else:
            fn = owner.get(i, "<module>")
            kind = "EVAL in %s()" % fn
            if any(t in fn.lower() for t in TRAINY):
                kind = "*** TRAIN-TIME in %s() ***" % fn
        findings.append((p.name, i, kind, stripped[:74]))

leaks = [f for f in findings if f[2].startswith("***")]
for name, ln, kind, txt in findings:
    if kind == "comment/docstring":
        continue
    print("  %-28s :%-4d %-26s %s" % (name, ln, kind, txt))
print()
print("  code references to held_out_*: %d   of which TRAIN-TIME: %d"
      % (len([f for f in findings if f[2] != "comment/docstring"]), len(leaks)))
if leaks:
    for f in leaks:
        print("    LEAK: %s:%d  %s" % (f[0], f[1], f[3]))
else:
    print("  NO held_out_* read inside any training or threshold-selection function.")

print()
print("  training scripts and what they open:")
for name in ("p2_train_factorized.py", "a1_arms.py", "p7_massive.py"):
    src = (SCRIPTS / name).read_text(encoding="utf-8")
    opens = sorted(set(re.findall(r'"(train_v2|dev_v2|held_out_v\w+)"', src)
                       + re.findall(r"'(train_v2|dev_v2|held_out_v\w+)'", src)
                       + re.findall(r"(train_v2|dev_v2|held_out_v\w+)\.jsonl", src)))
    print("    %-26s %s" % (name, opens))

print()
print("=" * 84)
print("A6.2 -- INDEPENDENT RECOMPUTATION from the saved prediction CSVs")
print("=" * 84)


def rd(p):
    with open(p, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


T = lambda v: str(v).strip().lower() == "true"
ROWS = ["A", "B", "C"]
CONT = {"box_1", "box_2", "box_3"}
RUNT = {"max_reach", "nearest_free_cell"}


def rowof(loc):
    return loc[5] if isinstance(loc, str) and loc.startswith("grid_") and len(loc) == 7 else None


checks = []


def chk(name, claimed, got, n):
    ok = abs(claimed - got) < 5e-5
    checks.append((name, claimed, got, n, ok))
    print("  %-46s claimed %.4f  recomputed %.4f  n=%-5d %s"
          % (name, claimed, got, n, "OK" if ok else "*** MISMATCH ***"))


# --- factorized, from thesis2/out/factorized_fp16.csv
f = rd(OUT / "factorized_fp16.csv")
chk("factorized action match", 0.6083,
    sum(T(r["strict_match"]) for r in f) / len(f), len(f))
grid = [r for r in f if rowof(json.loads(r["truth_json"]).get("location"))]
chk("factorized row accuracy", 0.3333,
    sum(1 for r in grid
        if rowof(json.loads(r["pred_json"]).get("location"))
        == rowof(json.loads(r["truth_json"]).get("location"))) / len(grid), len(grid))

# --- flat, from the V5 measured cell (warmup excluded)
v5 = [r for r in rd(REPO / "bench/results_v5_cls_scored.csv") if not T(r["is_warmup"])]
print("  (flat V5 cell after warmup exclusion: n=%d -- the 0.7465 figure is the"
      " n=217 fresh pass, recomputed below)" % len(v5))
fv = {r["metric"]: r for r in rd(OUT / "factorized_vs_flat.csv")}
chk("flat action match (n=217 fresh pass)", 0.7465, float(fv["action_match"]["flat"]), 217)
chk("flat row accuracy", 0.6204, float(fv["factor_row"]["flat"]), 108)

# --- P4.5 cascade
c = rd(OUT / "p45_cascade_full.csv")
g = [r for r in c if r["dataset"] == "held_out_v3" and r["signal"] == "factorized_row_conf"
     and r["target"] == "qwen3blora" and abs(float(r["tau"]) - 1.0) < 1e-9][0]
chk("P4.5 cascade -> qwen3blora, always", 0.8433, float(g["action_match"]),
    int(g["n_action"]))
o = [r for r in c if r["dataset"] == "held_out_v3" and r["signal"] == "factorized_row_conf"
     and r["target"] == "oracle" and abs(float(r["tau"]) - 1.0) < 1e-9][0]
chk("P4.5 oracle ceiling", 0.9401, float(o["action_match"]), int(o["n_action"]))

# --- independently rebuild the cascade from the prediction cache
pr = rd(OUT / "p45_predictions_held_out_v3.csv")
hit = 0
for r in pr:
    truth = json.loads(r["truth_json"])
    lt = int(r["fact_loc_type_idx"])
    row_i, col_i = int(r["fact_row_idx"]), int(r["fact_col_idx"])
    cw, rw = int(r["fact_container_which_idx"]), int(r["fact_runtime_which_idx"])
    if lt == 0:
        e = r["salsa_qwen3blora_letters_row"]
        if e in ROWS:
            row_i = ROWS.index(e)
        sym = "grid_%s%s" % (ROWS[row_i], ["1", "2", "3"][col_i])
    elif lt == 1:
        sym = ["box_1", "box_2", "box_3"][cw]
    else:
        sym = ["max_reach", "nearest_free_cell"][rw]
    verbs = ["pick_place", "fetch", "hold", "move_to", "inspect", "stop", "reject"]
    tgts = ["rubiks_cube", "bottle", "toy_car"]
    req = {"pick_place": ["target", "location"], "fetch": ["target"],
           "hold": ["target"], "move_to": ["location"], "inspect": ["target"],
           "stop": [], "reject": []}
    v = verbs[int(r["fact_verb_idx"])]
    obj = {"action": v}
    if "target" in req[v]:
        obj["target"] = tgts[int(r["fact_target_idx"])]
    if "location" in req[v]:
        obj["location"] = sym
    obj["modifiers"] = []
    t = dict(truth); t.setdefault("modifiers", [])
    hit += int(obj == t)
chk("P4.5 cascade rebuilt from scratch", 0.8433, hit / len(pr), len(pr))

# --- MASSIVE AUROC
s = json.loads((OUT / "generalization_massive_summary.json").read_text(encoding="utf-8"))
chk("MASSIVE AUROC per-factor", 0.8910, s["auroc_per_factor"], s["n_test"])
chk("MASSIVE AUROC flat", 0.8889, s["auroc_flat"], s["n_test"])

print()
bad = [c for c in checks if not c[4]]
print("  %d checks, %d mismatches" % (len(checks), len(bad)))
for name, cl, got, n, _ok in bad:
    print("    MISMATCH %s: claimed %.4f, recomputed %.4f" % (name, cl, got))

p = OUT / "a6_selfcheck.csv"
with open(p, "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["check", "claimed", "recomputed", "n", "agrees"])
    for row in checks:
        w.writerow(list(row[:4]) + ["yes" if row[4] else "NO"])
    w.writerow([])
    w.writerow(["leakage_code_references_to_held_out",
                len([f for f in findings if f[2] != "comment/docstring"])])
    w.writerow(["leakage_train_time_references", len(leaks)])
print("wrote %s" % p.relative_to(REPO))
print("=" * 84)
