#!/usr/bin/env python
"""
bench/summary.py -- KPI summary over scored grid CSVs. Read-only; computes
nothing new, just reports. Works on scored files; the cost/throughput sections
additionally need the token columns filled (see bench/backfill_tokens.py).

Sections:
  1. Accuracy       -- strict / normalized / gap, per cell
  2. Latency        -- median, p95, p99, throughput, and (rag) end-to-end
  3. Reject-rate    -- unmappable commands correctly routed to `reject` (safety KPI)
  4. Thinking       -- reasoning-trace length vs latency (schema v2+ files only)
  5. Cost           -- cloud USD per command by strategy (cloud rows only)
  6. Retrieval      -- rung 2 only: retrieval latency + retrieved-label precision
  7. Invalid output -- decomposition of pipeline failures (see NULL-PADDING below)

MIXED ROW SCHEMAS (25-col V2 and 28-col V3)
-------------------------------------------
V2 files have 25 columns. V3 files (rung 2 onward, and the rung-1 re-run on
held_out_v3) have 28: retrieval_latency_s, retrieved_ids, retrieved_scores.
Every column access here goes through .get() with a default, so both read fine
and --in may be passed more than once to union them.

DEDUPE KEY INCLUDES `dataset`. command_id is an index into that dataset's test
rows, so seen row 7 and held-out row 7 share an id. Without `dataset` in the key,
unioning two files through --in silently drops rows. (Passing them as --in and
--held keeps them separate and was never affected.)

COMPARABILITY: seen_v2 and held_out_v2 do NOT contain the same verb mix.
held_out_v2 varies the LOCATION PHRASING, and fetch/hold/inspect/stop/reject
carry no location -- so there was nothing to vary and those rows do not exist in
held_out_v2 at all. Comparing raw seen vs held-out accuracy therefore mixes a
phrasing effect with a verb-mix effect, and can even show held-out as EASIER.
--match-verbs restricts BOTH files to the verbs present in both, so the gap
measures phrasing alone. Use it for any number that goes in the thesis.
held_out_v3 fixes this at the source: it carries all seven verbs. --match-verbs
becomes a no-op there, which is the point.

NULL-PADDING (found by bench/diag_score_edges.py)
-------------------------------------------------
260 of 33,200 V2 rows are predictions like {"action":"stop","target":null,
"location":null} -- semantically CORRECT, rejected by the validator for carrying
null placeholders for slots the verb does not have. They never reach score.py;
run_cell records them as invalid:schema_failed.

They concentrate on the low-arity verbs: stop 11.5% of failures, inspect 5.6%,
fetch 3.0%, reject 3.0%, versus pick_place 0.89% -- pick_place has both slots, so
there is no empty slot to null-fill. That skews the failure profile of the exact
verbs carrying the safety KPI.

Section 7 splits invalid:schema_failed into "null-padded but correct" and
"genuinely wrong", from the stored pred_json. It changes NO stored number.

FIELD SIZE: the V2 schema added a `thinking` column holding deepseek-r1's full
reasoning trace, and at least one trace exceeds Python's default 128 KB field
cap. The limit is raised below. sys.maxsize is NOT used: on Windows it overflows
the underlying C long and raises OverflowError.

Usage:
    python -m bench.summary --in bench/results_grid_v2.csv
    python -m bench.summary --in bench/results_grid_v2.csv --held bench/results_heldout_v2.csv --match-verbs
    python -m bench.summary --in bench/results_heldout_v3_rung1.csv --in bench/results_heldout_v3_rag.csv
"""
import argparse, csv, json, math, statistics
from collections import defaultdict
from pathlib import Path

csv.field_size_limit(2**31 - 1)

try:
    # single source of truth; falls back if run outside the package
    from llm.prompts import ALL_STRATEGIES as _STRATS
except Exception:
    _STRATS = ["zero_shot", "few_shot", "cot", "role_based", "rag"]

MODELS = ["openai:gpt-4o-mini", "ollama:llama3.2:3b", "ollama:qwen2.5:3b",
          "ollama:qwen2.5:0.5b", "ollama:deepseek-r1:7b"]
STRATS = list(_STRATS)
CLOUD = "openai:gpt-4o-mini"
DEFAULT_CORPUS = "data/train_v2.jsonl"


def short(m):
    return m.split(":", 1)[1] if ":" in m else m


def load(paths):
    """Read scored CSV(s), dedupe on identity key keeping LAST (resume orphans).

    Key includes `dataset`: command_id is per-dataset, so without it a union of
    two files would collide and silently drop rows."""
    if isinstance(paths, str):
        paths = [paths]
    seen = {}
    for path in paths:
        p = Path(path)
        if not p.exists():
            print(f"  ! not found, skipped: {p}")
            continue
        with p.open(encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                seen[(r.get("dataset", ""), r.get("model", ""), r.get("strategy", ""),
                      r.get("command_id", ""), r.get("run_index", ""))] = r
    return list(seen.values())


def models_in(rows):
    """Known models in fixed order, then any unexpected ones, so nothing is hidden."""
    present = {r.get("model") for r in rows}
    out = [m for m in MODELS if m in present]
    out += sorted(m for m in present if m and m not in MODELS)
    return out


def strats_in(rows):
    present = {r.get("strategy") for r in rows}
    out = [s for s in STRATS if s in present]
    out += sorted(s for s in present if s and s not in STRATS)
    return out


def pct(vals, p):
    v = sorted(vals)
    k = (len(v) - 1) * p / 100
    f, c = math.floor(k), math.ceil(k)
    return v[f] if f == c else v[f] + (v[c] - v[f]) * (k - f)


def has_tokens(rows):
    return any(r.get("tokens_out") not in (None, "") for r in rows)


def has_thinking(rows):
    return any(r.get("thinking_chars") not in (None, "", "0") for r in rows)


def has_retrieval(rows):
    return any(r.get("retrieval_latency_s") not in (None, "") for r in rows)


def truth_verb(r):
    try:
        return json.loads(r.get("truth_json") or "{}").get("action")
    except (json.JSONDecodeError, TypeError, KeyError):
        return None


def verbs_in(rows):
    return {v for v in (truth_verb(r) for r in rows) if v}


def _fl(r, key):
    v = r.get(key)
    if v in (None, ""):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _acc(cell, col):
    return sum(r.get(col) == "True" for r in cell) / len(cell)


# ------------------------------------------------------------------ 1. accuracy

def sec_accuracy(rows, held=None):
    print("1. ACCURACY  strict / normalized / gap"
          + ("   + SEEN->HELD-OUT generalization gap" if held else ""))
    hdr = f'{"model":16}{"strategy":11}{"strict":>8}{"norm":>8}{"gap":>7}'
    if held:
        hdr += f'{"held":>9}{"drop":>8}'
    print(hdr)
    for m in models_in(rows):
        for s in strats_in(rows):
            cell = [r for r in rows if r.get("model") == m and r.get("strategy") == s]
            if not cell:
                continue
            st, nm = _acc(cell, "strict_match"), _acc(cell, "normalized_match")
            line = (f'{short(m):16}{s:11}{st*100:7.1f}%{nm*100:7.1f}%'
                    f'{(nm-st)*100:+6.1f}%')
            if held:
                hc = [r for r in held if r.get("model") == m and r.get("strategy") == s]
                if hc:
                    hst = _acc(hc, "strict_match")
                    line += f'{hst*100:8.1f}%{(hst-st)*100:+7.1f}'
                else:
                    line += f'{"n/a":>9}{"":>8}'
            flag = "  <==" if (nm - st) > 0.10 else ""
            print(line + flag)
    if held:
        print("   `drop` is held-out strict minus seen strict. More negative = worse")
        print("   generalization to unseen PHRASINGS of the same locations.")


# ------------------------------------------------------------------- 2. latency

def sec_latency(rows):
    tok = has_tokens(rows)
    ret = has_retrieval(rows)
    print("\n2. LATENCY (s), warmup excluded, pooled across strategies"
          + ("  + generation throughput" if tok else ""))
    hdr = f'{"model":16}{"median":>8}{"p95":>8}{"p99":>8}{"max":>8}'
    if tok:
        hdr += f'{"med tok":>9}{"tok/s":>8}'
    print(hdr)
    for m in models_in(rows):
        c = [r for r in rows if r.get("model") == m and r.get("is_warmup") != "True"
             and _fl(r, "latency_s") is not None]
        if not c:
            continue
        lats = [_fl(r, "latency_s") for r in c]
        line = (f'{short(m):16}{statistics.median(lats):8.2f}{pct(lats,95):8.2f}'
                f'{pct(lats,99):8.2f}{max(lats):8.2f}')
        if tok:
            ct = [r for r in c if r.get("tokens_out") not in (None, "")]
            if ct:
                mt = statistics.median(int(r["tokens_out"]) for r in ct)
                tps = statistics.median(int(r["tokens_out"]) / _fl(r, "latency_s")
                                        for r in ct if (_fl(r, "latency_s") or 0) > 0)
                line += f'{mt:9.0f}{tps:8.1f}'
        print(line)
    print("   latency_s is the LLM CALL ONLY. For rag cells the end-to-end figure is")
    print("   latency_s + retrieval_latency_s -- see section 6. Never quote latency_s")
    print("   alone as rag's latency.")
    if tok:
        print("   (low tok/s => slow generation, not verbosity. Note whether the model")
        print("    fits in VRAM before attributing this to architecture.)")


# -------------------------------------------------------------------- 3. reject

def _reject_stats(rows):
    stat = defaultdict(lambda: [0, 0])
    for r in rows:
        if truth_verb(r) != "reject":
            continue
        stat[r.get("model")][1] += 1
        try:
            p = json.loads(r["pred_json"]) if r.get("pred_json") else None
        except (json.JSONDecodeError, TypeError):
            p = None
        if isinstance(p, dict) and p.get("action") == "reject":
            stat[r.get("model")][0] += 1
    return stat


def sec_reject(rows, held=None):
    print("\n3. REJECT-RATE  (unmappable/unsafe commands correctly routed to `reject`)")
    a = _reject_stats(rows)
    b = _reject_stats(held) if held else None
    have_held_reject = bool(b) and any(n for _, n in b.values())
    hdr = f'{"model":16}{"seen":>12}'
    if b:
        hdr += f'{"held-out":>12}'
    print(hdr)
    for m in models_in(rows):
        c, n = a.get(m, [0, 0])
        line = f'{short(m):16}' + (f'{c/n*100:11.1f}%' if n else f'{"n/a":>12}')
        if b:
            c2, n2 = b.get(m, [0, 0])
            line += (f'{c2/n2*100:11.1f}%' if n2 else f'{"n/a":>12}')
        print(line)
    n_seen = max((n for _, n in a.values()), default=0)
    print(f"   denominator: {n_seen} reject rows per model in this file.")
    print("   A safety KPI: 0% means the model NEVER refuses an impossible command,")
    print("   and instead emits a plausible-looking action.")
    if held is not None and not have_held_reject:
        print("   This held-out file contains NO reject rows (held_out_v2: reject commands")
        print("   carry no location, so there was no phrasing to vary). Seen-only KPI.")
        print("   held_out_v3 carries 12 reject rows -- use it for the reportable number.")


# ------------------------------------------------------------------ 4. thinking

def sec_thinking(rows):
    if not has_thinking(rows):
        return
    print("\n4. REASONING TRACE  (models that return a separate `thinking` field)")
    print(f'{"model":16}{"strategy":11}{"n":>6}{"med chars":>11}{"max chars":>11}{"med lat":>9}')
    for m in models_in(rows):
        for s in strats_in(rows):
            c = [r for r in rows if r.get("model") == m and r.get("strategy") == s
                 and r.get("thinking_chars") not in (None, "")]
            tc = [int(r["thinking_chars"]) for r in c]
            if not tc or max(tc) == 0:
                continue
            lats = [_fl(r, "latency_s") for r in c if _fl(r, "latency_s") is not None]
            med_lat = statistics.median(lats) if lats else float("nan")
            print(f'{short(m):16}{s:11}{len(tc):6d}{statistics.median(tc):11.0f}'
                  f'{max(tc):11d}{med_lat:9.2f}')
    print("   A trace far above the median is a runaway reasoning loop, not verbosity.")
    print("   Report those separately: they drive the p99 latency tail.")


# ---------------------------------------------------------------------- 5. cost

def sec_cost(rows):
    if not has_tokens(rows):
        print("\n5. COST -- skipped (token columns empty; run bench/backfill_tokens.py first)")
        return
    print("\n5. CLOUD COST per command by strategy  (gpt-4o-mini rows only)")
    print(f'{"strategy":12}{"tok_in":>8}{"tok_out":>9}{"USD/cmd":>12}{"USD/1k cmd":>12}')
    total = 0.0
    for s in strats_in(rows):
        c = [r for r in rows if r.get("model") == CLOUD and r.get("strategy") == s
             and r.get("cost_usd") not in (None, "")]
        if not c:
            continue
        ti = sum(int(r["tokens_in"]) for r in c) / len(c)
        to = sum(int(r["tokens_out"]) for r in c) / len(c)
        cost = sum(float(r["cost_usd"]) for r in c) / len(c)
        total += sum(float(r["cost_usd"]) for r in c)
        print(f'{s:12}{ti:8.0f}{to:9.0f}{cost:12.6f}{cost*1000:12.3f}')
    print(f'   total cloud spend in this file: ${total:.4f}')
    print("   (local models: no API cost; their token counts use the OpenAI")
    print("    tokenizer and are approximate.)")


# ----------------------------------------------------------------- 6. retrieval

def sec_retrieval(rows, corpus_path=DEFAULT_CORPUS):
    """Rung 2 only. Retrieval cost, and whether retrieval returned USEFUL examples.

    `label precision` = share of retrieved examples whose verb matches the truth
    verb of the query. It is the direct test of the pre-registered prediction that
    retrieval helps on structured verbs and HURTS on `reject`, because the encoder
    matches surface phrasing and reject rows share no vocabulary with each other."""
    rag = [r for r in rows if r.get("strategy") == "rag"]
    if not rag:
        return
    print("\n6. RETRIEVAL (rung 2)")

    lats = [_fl(r, "retrieval_latency_s") for r in rag]
    lats = [x for x in lats if x is not None]
    if lats:
        print(f"   retrieval latency (ms): median {statistics.median(lats)*1000:6.1f}   "
              f"p95 {pct(lats,95)*1000:6.1f}   p99 {pct(lats,99)*1000:6.1f}   "
              f"max {max(lats)*1000:6.1f}")

    print(f'\n   {"model":16}{"llm p50":>9}{"retr p50":>10}{"e2e p50":>9}{"retr %":>8}')
    for m in models_in(rag):
        c = [r for r in rag if r.get("model") == m and r.get("is_warmup") != "True"]
        ll = [_fl(r, "latency_s") for r in c if _fl(r, "latency_s") is not None]
        rr = [_fl(r, "retrieval_latency_s") for r in c
              if _fl(r, "retrieval_latency_s") is not None]
        if not ll or not rr:
            continue
        l50, r50 = statistics.median(ll), statistics.median(rr)
        e2e = l50 + r50
        print(f'   {short(m):16}{l50:9.3f}{r50:10.4f}{e2e:9.3f}{r50/e2e*100:7.2f}%')
    print("   retr % = retrieval's share of end-to-end. Small share does NOT mean")
    print("   retrieval is free: it is a mandatory serial stage before the LLM call.")

    # ---- retrieved-label precision, per truth verb -----------------------
    cp = Path(corpus_path)
    if not cp.exists():
        print(f"\n   (label precision skipped: corpus {cp} not found; pass --corpus)")
        return
    corpus = [json.loads(l) for l in cp.open(encoding="utf-8") if l.strip()]

    hit = defaultdict(lambda: [0, 0])     # verb -> [matching examples, total examples]
    anyhit = defaultdict(lambda: [0, 0])  # verb -> [queries with >=1 match, queries]
    oob = 0
    for r in rag:
        tv = truth_verb(r)
        if not tv or not r.get("retrieved_ids"):
            continue
        try:
            ids = json.loads(r["retrieved_ids"])
        except (json.JSONDecodeError, TypeError):
            continue
        got = 0
        for i in ids:
            if not isinstance(i, int) or i < 0 or i >= len(corpus):
                oob += 1
                continue
            hit[tv][1] += 1
            if corpus[i]["expected"].get("action") == tv:
                hit[tv][0] += 1
                got += 1
        anyhit[tv][1] += 1
        if got:
            anyhit[tv][0] += 1

    if not hit:
        return
    print(f'\n   RETRIEVED-LABEL PRECISION  (corpus: {cp}, {len(corpus)} rows)')
    print(f'   {"truth verb":12}{"examples":>10}{"same verb":>11}{"precision":>11}'
          f'{"queries w/ >=1":>16}')
    order = ["reject", "stop", "fetch", "hold", "inspect", "move_to", "pick_place"]
    verbs = [v for v in order if v in hit] + [v for v in hit if v not in order]
    for v in verbs:
        good, tot = hit[v]
        a, b = anyhit[v]
        flag = "  <-- retrieval is misleading here" if tot and good / tot < 0.5 else ""
        print(f'   {v:12}{tot:>10}{good:>11}{good/tot*100:>10.1f}%'
              f'{a/b*100:>15.1f}%{flag}')
    if oob:
        print(f"   WARNING: {oob} retrieved id(s) out of range for this corpus. The rag")
        print(f"   rows were produced against a DIFFERENT corpus file than {cp}.")
    print("   PRE-REGISTERED PREDICTION: precision is high on the structured verbs and")
    print("   low on `reject`, because the encoder matches surface phrasing and reject")
    print("   rows share no vocabulary with each other. Check it here.")


# ----------------------------------------------------- 7. invalid decomposition

def _null_padded_but_correct(pred, truth):
    """True if pred equals truth once null/empty placeholder keys are dropped.

    These are the {"action":"stop","target":null,"location":null} cases: the model
    understood the command and padded slots the verb does not have. The validator
    rejects them, so they never reach score.py and land in invalid:schema_failed."""
    if not isinstance(pred, dict) or not isinstance(truth, dict):
        return False
    p = {k: v for k, v in pred.items()
         if not (k not in truth and (v is None or v == "" or v == [] or v == {}))}
    t = {k: v for k, v in truth.items() if not (k == "modifiers" and not v)}
    p = {k: v for k, v in p.items() if not (k == "modifiers" and not v)}
    return p == t


def sec_invalid(rows):
    """Split pipeline failures. Changes no stored number -- reads stage + pred_json.

    KEYS OFF `stage`, NOT `category`. bench/rescore.py overwrites `category` with
    the semantic verdict for ANY row whose pred_json parses, including rows the
    schema validator rejected -- so in a _scored file the invalid:schema_failed
    bucket is gone from `category` even though those rows are still there. `stage`
    is never rewritten, so this section reads the same on raw and scored files."""
    inv = [r for r in rows if r.get("stage") not in (None, "", "validated")]
    if not inv:
        return
    print("\n7. INVALID OUTPUT  (rows the pipeline never validated; keyed on `stage`)")
    stages = defaultdict(int)
    for r in inv:
        stages[r.get("stage")] += 1
    for k, v in sorted(stages.items(), key=lambda x: -x[1]):
        print(f'   {k:28}{v:>8,}')

    # In a rescored file, `normalized_match` is NOT gated on validation, so a row
    # can be schema-invalid and still normalized-correct. That is the deepseek
    # strict->normalized gap, located precisely.
    rescued = sum(1 for r in inv if r.get("normalized_match") == "True")
    if rescued:
        print(f'\n   of which NORMALIZED-CORRECT despite failing validation: {rescued:,} '
              f'({rescued/len(inv)*100:.1f}%)')
        print('   These are semantically right, structurally rejected -- mostly nested')
        print('   action objects. This is what the strict->normalized gap measures.')
        by = defaultdict(lambda: [0, 0])
        for r in inv:
            by[r.get("model")][1] += 1
            if r.get("normalized_match") == "True":
                by[r.get("model")][0] += 1
        print(f'   {"model":16}{"invalid":>10}{"rescued":>10}{"share":>9}')
        for m in models_in(rows):
            g, n = by.get(m, [0, 0])
            if n:
                print(f'   {short(m):16}{n:>10,}{g:>10,}{g/n*100:>8.1f}%')

    padded_by_verb = defaultdict(int)
    inv_by_verb = defaultdict(int)
    padded_total = 0
    for r in inv:
        v = truth_verb(r)
        inv_by_verb[v] += 1
        try:
            pred = json.loads(r["pred_json"]) if r.get("pred_json") else None
            truth = json.loads(r.get("truth_json") or "{}")
        except (json.JSONDecodeError, TypeError):
            continue
        if pred is not None and _null_padded_but_correct(pred, truth):
            padded_by_verb[v] += 1
            padded_total += 1

    print(f'\n   of which SEMANTICALLY CORRECT but null-padded: {padded_total:,} '
          f'({padded_total/len(inv)*100:.2f}% of invalid rows)')
    print(f'   {"verb":12}{"invalid":>10}{"null-padded":>13}{"share":>9}')
    order = ["reject", "stop", "fetch", "hold", "inspect", "move_to", "pick_place"]
    verbs = [v for v in order if v in inv_by_verb] + \
            [v for v in inv_by_verb if v not in order]
    for v in verbs:
        n, p = inv_by_verb[v], padded_by_verb.get(v, 0)
        flag = "  <-- look" if n and p / n > 0.05 else ""
        print(f'   {str(v):12}{n:>10,}{p:>13,}{(p/n*100 if n else 0):>8.1f}%{flag}')
    print("   Null-padding is a FORMATTING failure, not a comprehension failure: the")
    print("   model named the right verb and filled slots the verb does not have. It")
    print("   concentrates on low-arity verbs (stop, inspect, fetch, reject), which are")
    print("   the verbs carrying the safety KPI. PREDICTION: rung 3 removes it entirely,")
    print("   because LoRA trains on the exact output shape.")


# ---------------------------------------------------------------------- driver

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", action="append",
                    help="scored CSV; repeat to union several (25- and 28-col both fine)")
    ap.add_argument("--held", action="append", default=None,
                    help="optional held-out file(s); enables the generalization-gap column")
    ap.add_argument("--match-verbs", action="store_true",
                    help="restrict BOTH files to verbs present in both, so the "
                         "generalization gap is not confounded by verb mix")
    ap.add_argument("--corpus", default=DEFAULT_CORPUS,
                    help="retrieval corpus, for section 6 label precision")
    args = ap.parse_args()

    if not args.inp:
        args.inp = ["bench/results_grid_scored.csv"]

    rows = load(args.inp)
    if not rows:
        raise SystemExit("no rows loaded -- check the --in path(s)")
    held = load(args.held) if args.held else None
    print(f"loaded {len(rows)} rows from {', '.join(args.inp)}"
          + (f" | {len(held)} from {', '.join(args.held)}" if held else ""))

    datasets = sorted({r.get("dataset", "") for r in rows})
    ncols = sorted({len(r) for r in rows})
    print(f"  datasets present: {datasets}   row widths: {ncols}")

    if args.match_verbs:
        if not held:
            raise SystemExit("--match-verbs needs --held")
        common = verbs_in(rows) & verbs_in(held)
        before = (len(rows), len(held))
        rows = [r for r in rows if truth_verb(r) in common]
        held = [r for r in held if truth_verb(r) in common]
        print(f"--match-verbs: common verbs {sorted(common)}")
        print(f"  seen {before[0]} -> {len(rows)} rows | held {before[1]} -> {len(held)} rows")
        if before == (len(rows), len(held)):
            print("  (no rows dropped: both files already carry the same verbs)")
    elif held:
        only_seen = verbs_in(rows) - verbs_in(held)
        if only_seen:
            print(f"  WARNING: verbs present ONLY in seen: {sorted(only_seen)}. "
                  f"The `drop` column mixes phrasing and verb-mix effects. "
                  f"Re-run with --match-verbs for the thesis number.")
    print()

    sec_accuracy(rows, held)
    sec_latency(rows)
    sec_reject(rows, held)
    sec_thinking(rows)
    sec_cost(rows)
    sec_retrieval(rows, corpus_path=args.corpus)
    sec_invalid(rows)


if __name__ == "__main__":
    main()
