"""Block B audit checks. READ-ONLY over bench/ and data/. Writes nothing."""
import csv, json, collections, hashlib, sys
csv.field_size_limit(2**31 - 1)

T = lambda v: v in ("True", "true", "1")
FILES = ["rung1", "rag", "lora", "cls"]
SCORED = {f: f"bench/results_heldout_v3_{f}_scored.csv" for f in FILES}
RAW = {f: f"bench/results_heldout_v3_{f}.csv" for f in FILES}

print("=" * 70)
print("B3 -- DEDUPE KEY: does omitting `dataset` drop rows?")
print("=" * 70)
rows_all = []
for f in FILES:
    with open(SCORED[f], encoding="utf-8", newline="") as fh:
        rs = list(csv.DictReader(fh))
    rows_all += rs
    print(f"  {f:6} rows={len(rs)}")
k4 = {(r["model"], r["strategy"], r["command_id"], r["run_index"]) for r in rows_all}
k5 = {(r["dataset"], r["model"], r["strategy"], r["command_id"], r["run_index"]) for r in rows_all}
print(f"  total rows            : {len(rows_all)}")
print(f"  unique 4-key (scratch): {len(k4)}")
print(f"  unique 5-key (summary): {len(k5)}")
print(f"  ROWS DROPPED BY 4-KEY : {len(rows_all) - len(k4)}")
print(f"  datasets present      : {sorted({r['dataset'] for r in rows_all})}")

print()
print("=" * 70)
print("B2 -- INVALID RATE: stage vs category in the SCORED files")
print("=" * 70)
stage_c = collections.Counter(r["stage"] for r in rows_all)
cat_c = collections.Counter(r["category"] for r in rows_all)
print("  stage counts   :", dict(stage_c.most_common()))
print("  category invalid:* counts:",
      {k: v for k, v in cat_c.most_common() if k.startswith("invalid:")})
n_bad_stage = sum(v for k, v in stage_c.items() if k != "validated")
n_bad_cat = sum(v for k, v in cat_c.items() if k.startswith("invalid:"))
print(f"  rows with stage != validated      : {n_bad_stage}")
print(f"  rows with category invalid:*      : {n_bad_cat}")
print(f"  ROWS HIDDEN IF KEYED ON category  : {n_bad_stage - n_bad_cat}")

print()
print("=" * 70)
print("B4 -- RUNG 4 STRUCTURAL GUARANTEE over all cls rows")
print("=" * 70)
schema = json.load(open("schema/schema_v2.json", encoding="utf-8"))
VERBS = set(schema["verbs"]); TARGETS = set(schema["targets"]); LOCS = set(schema["locations"])
REQ = {v: set(d["required"]) for v, d in schema["verbs"].items()}
with open(SCORED["cls"], encoding="utf-8", newline="") as fh:
    cls = list(csv.DictReader(fh))
print(f"  rows: {len(cls)}")
bad = collections.Counter()
stages = collections.Counter(r["stage"] for r in cls)
for r in cls:
    try:
        o = json.loads(r["pred_json"]) if r["pred_json"] else None
    except Exception:
        bad["pred_json_unparseable"] += 1; continue
    if o is None:
        bad["pred_json_empty"] += 1; continue
    if not isinstance(o, dict):
        bad["not_an_object"] += 1; continue
    a = o.get("action")
    if a not in VERBS:
        bad["illegal_verb"] += 1; continue
    present = {k for k in o if k != "action" and not (k == "modifiers" and not o[k])}
    req = REQ[a]
    if present - req:
        bad[f"extra_field({a}:{sorted(present-req)})"] += 1
    if req - present:
        bad[f"missing_field({a}:{sorted(req-present)})"] += 1
    if "target" in o and o["target"] not in TARGETS:
        bad["illegal_target"] += 1
    if "location" in o and o["location"] not in LOCS:
        bad["illegal_location"] += 1
    if "modifiers" in o and o["modifiers"] != []:
        bad["nonempty_modifiers"] += 1
print(f"  stages: {dict(stages)}")
print(f"  VIOLATIONS: {dict(bad) if bad else 'NONE (0 across all rows)'}")
verbs_emitted = collections.Counter(
    (json.loads(r["pred_json"]).get("action") if r["pred_json"] else None) for r in cls)
print(f"  verbs emitted: {dict(verbs_emitted)}")

print()
print("=" * 70)
print("B5 -- RAW vs SCORED: did rescore actually run on each file?")
print("=" * 70)
for f in FILES:
    try:
        with open(RAW[f], encoding="utf-8", newline="") as fh:
            raw = list(csv.DictReader(fh))
        with open(SCORED[f], encoding="utf-8", newline="") as fh:
            sc = list(csv.DictReader(fh))
    except FileNotFoundError as e:
        print(f"  {f:6} MISSING: {e}"); continue
    rk = {(r["model"], r["strategy"], r["command_id"], r["run_index"]): r for r in raw}
    diff = collections.Counter()
    for r in sc:
        k = (r["model"], r["strategy"], r["command_id"], r["run_index"])
        o = rk.get(k)
        if o is None:
            diff["no_raw_counterpart"] += 1; continue
        for col in ("strict_match", "normalized_match", "category"):
            if o.get(col) != r.get(col):
                diff[col] += 1
    print(f"  {f:6} raw={len(raw):6} scored={len(sc):6} orphans_dropped={len(raw)-len(sc):5} "
          f"score-col diffs={dict(diff) if diff else 'NONE'}")
