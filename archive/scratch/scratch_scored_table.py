"""Results-chapter table over the four held_out_v3 scored CSVs.

INVALID-OUTPUT COLUMNS KEY OFF `stage`, NEVER `category`.
bench/rescore.py overwrites `category` with the semantic verdict for any row
whose pred_json parses, so in a _scored file the invalid:schema_failed bucket is
GONE from `category` while the rows are still present. Measured on this data:
9,536 rows have stage != validated, but only 1,004 carry category invalid:* --
keying on `category` would hide 8,532 structural failures. `stage` is never
rewritten by rescore, so these columns read identically on raw and scored files.

Columns:
  strict/norm/gap   as before (strict is gated on stage == validated upstream)
  grid/box          strict accuracy on command_id <181 / >=181
  inv%              rows the pipeline never validated       (stage != validated)
  sf%               of those, schema_failed                 (parsed, wrong shape)
  xf%               of those, extract_failed                (no JSON recoverable)
  null%             share of schema_failed rows that are NULL-PADDED BUT
                    SEMANTICALLY CORRECT -- e.g. {"action":"stop","target":null,
                    "location":null}. Rejected by the validator for null
                    placeholders in slots the verb does not have. See the
                    NULL-PADDING note in bench/summary.py; this concentrates on
                    low-arity verbs and skews the failure profile of exactly the
                    verbs carrying the safety KPI.
  p50/p99           latency; rag rows include retrieval_latency_s
"""
import csv, json, collections

csv.field_size_limit(2**31 - 1)
T = lambda v: v in ("True", "true", "1")
FILES = ["rung1", "rag", "lora", "cls"]

# Dedupe key INCLUDES `dataset`. All four files here are held_out_v3, so the
# shorter key drops nothing today (verified: 30,380 rows -> 30,380 unique under
# both). It stops being safe the instant a seen file joins FILES: command_id is
# a per-dataset index, so seen row 7 and held-out row 7 collide and one is
# silently discarded. Matches bench/summary.py, figures.py, analyze.py,
# rescore.py -- one key everywhere.
d = {}
for f in FILES:
    for r in csv.DictReader(open(f"bench/results_heldout_v3_{f}_scored.csv",
                                 encoding="utf-8", newline="")):
        d[(r.get("dataset", ""), r["model"], r["strategy"],
           r["command_id"], r["run_index"])] = r

cells = collections.defaultdict(list)
for r in d.values():
    cells[(r["model"], r["strategy"])].append(r)


def null_padded_correct(r):
    """schema_failed row that is semantically right, rejected only for nulls."""
    try:
        p = json.loads(r["pred_json"]) if r["pred_json"] else None
        t = json.loads(r["truth_json"]) if r["truth_json"] else None
    except (json.JSONDecodeError, TypeError):
        return False
    if not isinstance(p, dict) or not isinstance(t, dict):
        return False
    stripped = {k: v for k, v in p.items() if v is not None}
    stripped = {k: v for k, v in stripped.items() if not (k == "modifiers" and not v)}
    tc = {k: v for k, v in t.items() if not (k == "modifiers" and not v)}
    return stripped == tc


print(f"cells: {len(cells)}\n")
print(f"{'model':26}{'strategy':11}{'n':>6}{'strict':>8}{'norm':>7}{'gap':>7}"
      f"{'grid':>7}{'box':>7}{'inv%':>7}{'sf%':>7}{'xf%':>7}{'null%':>7}"
      f"{'p50':>8}{'p99':>8}")

tot_inv = tot_rows = 0
for (m, s), c in sorted(cells.items()):
    v = [r for r in c if not T(r["is_warmup"])]

    def lat(r):
        x = float(r["latency_s"])
        if s == "rag" and r.get("retrieval_latency_s"):
            x += float(r["retrieval_latency_s"])
        return x

    L = sorted(lat(r) for r in v)
    p = lambda q: L[min(int(q * len(L)), len(L) - 1)]
    g = [r for r in v if int(r["command_id"]) < 181]
    b = [r for r in v if int(r["command_id"]) >= 181]
    st = sum(T(r["strict_match"]) for r in v) / len(v)
    no = sum(T(r["normalized_match"]) for r in v) / len(v)

    # --- invalid decomposition, keyed on `stage` ---
    inv = [r for r in v if r.get("stage") not in (None, "", "validated")]
    sf = [r for r in inv if r.get("stage") == "schema_failed"]
    xf = [r for r in inv if r.get("stage") == "extract_failed"]
    nulls = sum(1 for r in sf if null_padded_correct(r))
    tot_inv += len(inv)
    tot_rows += len(v)

    print(f"{m:26}{s:11}{len(v):>6}{st:>8.3f}{no:>7.3f}{no-st:>7.3f}"
          f"{sum(T(r['strict_match']) for r in g)/len(g):>7.3f}"
          f"{sum(T(r['strict_match']) for r in b)/len(b):>7.3f}"
          f"{len(inv)/len(v)*100:>7.1f}"
          f"{len(sf)/len(v)*100:>7.1f}"
          f"{len(xf)/len(v)*100:>7.1f}"
          f"{(nulls/len(sf)*100) if sf else 0:>7.1f}"
          f"{p(.50):>8.3f}{p(.99):>8.3f}")

print(f"\ntotals: {tot_rows:,} rows (warmup excluded), {tot_inv:,} never validated "
      f"({tot_inv/tot_rows*100:.1f}%)")
print("inv%/sf%/xf% key on `stage`; `category` cannot be used -- rescore.py "
      "overwrites it (8,532 rows hidden).")
print("null% is a share of that cell's schema_failed rows, not of n.")
