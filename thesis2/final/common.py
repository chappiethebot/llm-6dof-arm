"""thesis2/final/common.py -- shared backbone for the final run.

Everything here is imported by the s0..s6 scripts. It owns, in one place:

  * the repo paths and the read-only rule (nothing outside thesis2/ is written);
  * the SHARED label space, field masks, emit() and scorer -- IMPORTED from
    ft/train_classifier.py and llm/score.py, never copied;
  * the held-out vocabulary assertion (rule 3);
  * the frozen-checkpoint gate (rule 2);
  * the V5 repeat-count census (rule: R is read, never chosen);
  * logging to thesis2/out/final/logs/, resumability, and the GATE verdict.

NOTHING in this module trains, and nothing reads a held-out split.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DATA = REPO / "data"
OUT = REPO / "thesis2" / "out" / "final"
LOGS = OUT / "logs"
POWER = OUT / "power_logs"
MODELS = REPO / "thesis2" / "models" / "final"
FIGS = REPO / "thesis2" / "figures" / "final"
BENCH = REPO / "bench"
SCHEMA_FILE = REPO / "schema" / "schema_v2.json"
FROZEN = OUT / "frozen_checkpoints.csv"

for _d in (OUT, LOGS, POWER, MODELS, FIGS):
    _d.mkdir(parents=True, exist_ok=True)

csv.field_size_limit(2 ** 31 - 1)

SEED = 20260827
SPLITS = ("held_out_v3", "held_out_v4c")
TRAIN, DEV = "train_v2", "dev_v2"

# --------------------------------------------------------------- shared API
from llm.score import score                                            # noqa: E402
from ft.train_classifier import (build_label_space, emit, MAX_LEN)      # noqa: E402

SPACE = build_label_space()
VERBS = SPACE["verbs"]
TARGETS = SPACE["targets"]
LOCATIONS = SPACE["locations"]
VERB_FIELDS = SPACE["verb_fields"]

# Field masks, derived from the schema -- the same source emit() uses.
CONTAINERS = [l for l in LOCATIONS if l.startswith("box_")]
GRIDS = [l for l in LOCATIONS if l.startswith("grid_")]
RUNTIMES = [l for l in LOCATIONS if not l.startswith(("box_", "grid_"))]


def rowof(loc):
    return loc[5] if isinstance(loc, str) and loc.startswith("grid_") and len(loc) == 7 else None


def colof(loc):
    return loc[6] if isinstance(loc, str) and loc.startswith("grid_") and len(loc) == 7 else None


def load_split(name: str):
    p = DATA / ("%s.jsonl" % name)
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def truth_of(row):
    t = dict(row["expected"])
    t.setdefault("modifiers", [])
    return t


# ------------------------------------------------- V5 exemplars and retrieval
#
# ONE definition of what a V5 cell feeds run_once as `examples`, used by
# s1_run_v5_cells.py, s3_energy.py and s1_prompt_equality.py. It exists because
# three separate call sites got it wrong in the same three ways (2026-09-26):
#
#   1. `load_seen_split(DEFAULT_K)` returns a TUPLE (exemplars, test_rows);
#   2. its default seen_path is data/seen.jsonl -- the V1 file. V5 draws
#      exemplars from V5's own DATASETS[split]["exemplars_from"];
#   3. rag does not use the fixed pool at all: V5 retrieves PER COMMAND with
#      llm.retrieve.Retriever (bench/run_cell.py:279-283).
#
# Nothing here restates a path or a k. Both come from bench/run_cell.py, so if
# V5's table changes these follow it.

_RETRIEVER = None


def v5_retriever(quiet=True):
    """The rag Retriever, built ONCE and reused across cells.

    bench/run_cell.py: "Build it ONCE and reuse it across cells -- constructing
    it per cell puts a 2-3 second model load and a corpus encode into every cell
    for no reason."
    """
    global _RETRIEVER
    if _RETRIEVER is None:
        from llm.retrieve import Retriever        # lazy: pulls torch
        _RETRIEVER = Retriever(quiet=quiet)
    return _RETRIEVER


def v5_pool(split, strategy, quiet=True):
    """-> (rows, fewshot_examples, retriever) for one cell, V5's way.

    `rows` comes from bench/run_cell._load_rows so the command list and the
    exemplar path are V5's, not ours. `fewshot_examples` is the fixed pool and
    is None for every strategy except few_shot, matching run_cell.py:244.
    """
    from bench.run_cell import _load_rows
    from bench.dataset import DEFAULT_K
    rows, exemplars = _load_rows(split, DEFAULT_K)
    if strategy == "few_shot" and not exemplars:
        # run_cell raises ValueError here; we fail the gate with the same meaning.
        gate(False, ["few_shot requires seen exemplars and none were loaded for "
                     "split %r (bench/run_cell.py raises ValueError here)" % split])
    fewshot = exemplars if strategy == "few_shot" else None
    retriever = v5_retriever(quiet=quiet) if strategy == "rag" else None
    return rows, fewshot, retriever


def v5_examples_for(command, strategy, fewshot, retriever):
    """-> (examples, ret_latency_s, ret_ids_json, ret_scores_json) for ONE command.

    Byte-for-byte the choice bench/run_cell.py:279-283 makes.
    """
    if strategy != "rag":
        return fewshot, "", "", ""
    from bench.dataset import DEFAULT_K
    res = retriever.retrieve(command, k=DEFAULT_K, exclude_exact=True)
    return (res.examples, res.latency_s, json.dumps(res.ids),
            json.dumps([round(x, 4) for x in res.scores]))


# ------------------------------------------------- rule 3: vocabulary guard
HELD_OUT_ROW_WORDS = ("back", "front", "upper", "bottom", "top", "lower")


def assert_no_heldout_vocab(*texts, where="startup"):
    """Refuse to run if held-out row vocabulary appears in any legend, lexicon,
    label description, prompt or vocabulary a NEW arm is built from.

    Called at startup by every new-arm script with everything it will show the
    model. Raises SystemExit (not AssertionError) so the gate is visible.
    """
    bad = []
    for t in texts:
        if t is None:
            continue
        if isinstance(t, dict):
            t = " ".join(map(str, list(t.keys()) + list(map(str, t.values()))))
        elif not isinstance(t, str):
            t = " ".join(map(str, t))
        low = t.lower()
        for w in HELD_OUT_ROW_WORDS:
            if re.search(r"\b%s\b" % w, low):
                bad.append(w)
    if bad:
        print("GATE FAIL: held-out row vocabulary %s reached %s. Rule 3 forbids adding "
              "back/front/upper/bottom/top/lower to any legend, lexicon, label "
              "description, prompt or tokenizer vocabulary of a NEW arm."
              % (sorted(set(bad)), where))
        raise SystemExit(2)
    print("  vocab guard OK: none of %s in %s" % (list(HELD_OUT_ROW_WORDS), where))


def train_vocab_words():
    """Every whitespace token in train_v2 -- the only legal vocabulary source."""
    return {w for r in load_split(TRAIN) for w in r["command"].lower().split()}


def assert_vocab_from_train_only(vocab, where="a new arm's vocabulary"):
    """Rule 3 for a VOCABULARY built from data, as opposed to text a script adds.

    Rule 3 forbids ADDING held-out row vocabulary. Rule 4 says training data is
    train_v2 as it is -- and `top` genuinely occurs once in train_v2, so a
    vocabulary built from train_v2 legitimately contains it. A zero-tolerance
    check over the vocabulary therefore aborts a legal run (it did: the 5-command
    smoke test hid this because those 5 commands contain no `top`).

    So this asserts the two things that actually matter:
      1. every token comes from train_v2 -- nothing was added from anywhere else;
      2. any held-out row word present is present BECAUSE train_v2 contains it,
         and is reported so the reader can see which and how often.
    """
    train_words = train_vocab_words()
    tokens = set(vocab) if isinstance(vocab, dict) else set(vocab)
    special = {"<pad>", "<unk>"}
    foreign = sorted(tokens - train_words - special)
    if foreign:
        print("GATE FAIL: %d token(s) in %s are not in %s: %s"
              % (len(foreign), where, TRAIN, foreign[:20]))
        raise SystemExit(2)
    inherited = sorted(w for w in HELD_OUT_ROW_WORDS if w in tokens)
    added = sorted(w for w in HELD_OUT_ROW_WORDS
                   if w in tokens and w not in train_words)
    if added:
        print("GATE FAIL: held-out row vocabulary %s was ADDED to %s" % (added, where))
        raise SystemExit(2)
    print("  vocab guard OK: %d tokens in %s, all from %s; held-out row words present "
          "only because %s contains them: %s"
          % (len(tokens), where, TRAIN, TRAIN, inherited or "none"))


# ------------------------------------------- rule 2: frozen-checkpoint gate
FROZEN_COLS = ["arm", "kind", "path", "sha256", "selected_epoch", "dev_action_match",
               "frozen_utc"]

# Arms that predate this run: their checkpoints are already frozen upstream and
# are NOT re-registered here. The gate applies to NEW arms only.
PREEXISTING_ARMS = {
    "flat_gpu", "flat_cpu", "factorized_fp16", "factorized_int4", "tinybert",
    "minilm_probe", "knn_minilm", "rules_open", "rules_firewalled", "tfidf_logreg",
    "bilstm", "flant5", "lora_greedy", "a8_speculative", "a10_speculative",
    "a12_smalllm", "cascade_local_tauA", "pipeline_always_local",
    "cascade_cloud_tauA", "cascade_cloud_always", "cascade_oracle", "ensemble",
}
NEW_ARMS = {"textcnn", "mlpbow", "grpo_base", "grpo_sft"} | {
    "salsa_full_qwen2.5-3b-lora", "salsa_full_Qwen2.5-3B-Instruct",
    "salsa_full_qwen2.5-0.5b-lora", "salsa_full_Qwen2.5-0.5B-Instruct"}
RETRAINED_LEGACY_ARMS = {"tinybert", "minilm_probe", "knn_minilm", "tfidf_logreg", "bilstm", "flant5"}


def read_frozen():
    if not FROZEN.exists():
        return {}
    return {r["arm"]: r for r in csv.DictReader(FROZEN.open(encoding="utf-8", newline=""))}


def require_frozen(arm: str):
    """Rule 2: a script that reads held-out data refuses to run unless the arm's
    frozen-checkpoint hash is on record. Pre-existing arms are exempt (their
    provenance is the earlier run); new arms are not."""
    if arm in PREEXISTING_ARMS and arm not in RETRAINED_LEGACY_ARMS:
        return None
    frozen = read_frozen()
    if arm not in frozen:
        print("GATE FAIL: arm %r is a NEW arm and has no row in %s. Run "
              "s0_freeze.py first; held-out inference is refused until the "
              "checkpoint is frozen." % (arm, FROZEN.relative_to(REPO)))
        raise SystemExit(2)
    rec = frozen[arm]
    p = Path(rec["path"])
    if not p.exists() and not str(rec["sha256"]).startswith("n/a:"):
        raise SystemExit("GATE FAIL: frozen checkpoint is missing: %s" % p)
    if p.exists():
        from s0_freeze import dir_sha
        h = dir_sha(p) if p.is_dir() else sha256_of(p)
        if h != rec["sha256"]:
            print("GATE FAIL: %s has sha256 %s but %s records %s. The checkpoint "
                  "changed after freezing." % (p, h[:16], FROZEN.name, rec["sha256"][:16]))
            raise SystemExit(2)
    print("  frozen-checkpoint gate OK for %s (%s, dev %s)"
          % (arm, rec["sha256"][:16], rec["dev_action_match"]))
    return rec


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------- the V5 repeat-count census
V5_MAIN_FILES = ["results_v5_cloud.csv", "results_v5_cls.csv", "results_v5_deepseek.csv",
                 "results_v5_llama.csv", "results_v5_qwen05.csv", "results_v5_qwen3.csv"]
# results_v5_check.csv and results_v5_smoke.csv are smoke runs, not cells.
V5_EXPECTED_ROWS_V3 = 12152


def v5_census(verbose=True):
    """Read the exact V5 cell list and repeat count per cell from bench/.

    Returns {(model, strategy): {"file", "repeats", "n_commands", "rows",
    "warmup_ids", "kind"}}. R is READ here; no script chooses it.
    """
    cells = {}
    total = 0
    for fn in V5_MAIN_FILES:
        p = BENCH / fn
        if not p.exists():
            continue
        rows = list(csv.DictReader(p.open(encoding="utf-8", newline="")))
        total += len(rows)
        per = {}
        for r in rows:
            k = (r["model"], r["strategy"])
            per.setdefault(k, {"rows": 0, "cmd": {}, "warm": set()})
            per[k]["rows"] += 1
            if str(r.get("is_warmup", "")).lower() == "true":
                per[k]["warm"].add(r["command_id"])
            else:
                per[k]["cmd"][r["command_id"]] = per[k]["cmd"].get(r["command_id"], 0) + 1
        for k, v in per.items():
            reps = sorted({c for c in v["cmd"].values()})
            cells[k] = dict(file=fn, rows=v["rows"], n_commands=len(v["cmd"]),
                            repeats=max(reps) if reps else 0,
                            repeat_dist={r: sum(1 for c in v["cmd"].values() if c == r)
                                         for r in reps},
                            warmup_ids=sorted(v["warm"]),
                            kind="cloud" if k[0].startswith("openai:") else "local")
    if verbose:
        print("  V5 census: %d cells, %d rows in %s" % (len(cells), total, ", ".join(V5_MAIN_FILES)))
        by_kind = {}
        for k, v in cells.items():
            by_kind.setdefault((v["kind"], v["repeats"]), []).append(k)
        for (kind, reps), ks in sorted(by_kind.items()):
            print("     %-6s R=%d  %d cells" % (kind, reps, len(ks)))
    return cells


def v5_repeats_for(kind: str, cells=None) -> int:
    """R for a new arm = the repeat count V5 used for its main cells of that kind."""
    cells = cells or v5_census(verbose=False)
    reps = {v["repeats"] for v in cells.values() if v["kind"] == kind}
    if len(reps) != 1:
        print("GATE FAIL: V5 used more than one repeat count for %s cells (%s). "
              "R must be read, not chosen -- resolve this before running."
              % (kind, sorted(reps)))
        raise SystemExit(2)
    return reps.pop()


# ------------------------------------------------------------------ logging
class Tee:
    """stdout -> console AND log file.

    Delegates every other attribute to the real stdout: transformers' progress
    bars call isatty()/fileno()/encoding on sys.stdout, and a Tee without them
    raises AttributeError inside model loading (found by the parser smoke test).
    """

    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")
        self.out = sys.stdout

    def write(self, s):
        self.out.write(s)
        try:
            self.f.write(s)
        except (ValueError, OSError):
            pass
        return len(s)

    def flush(self):
        self.out.flush()
        try:
            self.f.flush()
        except (ValueError, OSError):
            pass

    def isatty(self):
        return False            # a log file is never a tty; disables progress bars

    def __getattr__(self, name):
        return getattr(self.out, name)


def start_log(script_name: str):
    ts = time.strftime("%Y%m%d_%H%M%S")
    p = LOGS / ("%s_%s.log" % (script_name, ts))
    sys.stdout = Tee(p)
    print("=" * 96)
    print("%s   started %s" % (script_name, time.strftime("%Y-%m-%d %H:%M:%S")))
    print("  log: %s" % p.relative_to(REPO))
    print("  python: %s" % sys.executable)
    print("=" * 96)
    return p


def gate(ok: bool, summary_lines=()):
    """One-screen summary, then GATE PASS / GATE FAIL and the exit code."""
    print()
    print("-" * 96)
    for l in summary_lines:
        print("  %s" % l)
    print("-" * 96)
    print("GATE PASS" if ok else "GATE FAIL")
    sys.stdout.flush()
    raise SystemExit(0 if ok else 1)


# ----------------------------------------------------- prediction file I/O
PRED_COLS = ["arm", "split", "command_id", "repeat", "command", "raw", "pred_json",
             "truth_json", "correct", "verb_correct", "target_correct",
             "location_correct", "row_correct", "col_correct", "latency_ms", "note"]


def pred_path(arm, split):
    return OUT / ("pred_%s_%s.csv" % (arm, split))


def pred_rows_of(path):
    """Parsed CSV rows of a prediction file. Never a line count.

    Fixed 2026-09-26. The previous version of pred_is_complete counted PHYSICAL
    LINES (`sum(1 for _ in open(...)) - 1`). A raw model output containing a
    newline -- which every GRPO and every Ollama arm produces -- therefore
    inflated the count: pred_grpo_base_held_out_v3.csv reported 986/217 rows and
    was declared COMPLETE, when csv.DictReader gives exactly 217 rows. A line
    count is never a row count for a file with quoted newlines in it.
    """
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def pred_census(arm, split, expected_rows, n_commands=None):
    """-> (complete, n_rows, n_unique_cids, reason).

    COMPLETE means EXACTLY expected_rows parsed CSV rows AND exactly
    n_commands distinct command_ids. Anything else is not complete and the
    reason says which of the two failed, so "complete" is never a line count
    and never an inequality.
    """
    if n_commands is None:
        n_commands = len(load_split(split))      # the split itself, not a literal
    rows = pred_rows_of(pred_path(arm, split))
    n = len(rows)
    cids = {str(r.get("command_id", "")) for r in rows}
    u = len(cids)
    if not n:
        return False, 0, 0, "file absent or empty"
    if n != expected_rows:
        return False, n, u, ("%d parsed rows, expected exactly %d" % (n, expected_rows))
    if u != n_commands:
        return False, n, u, ("%d distinct command_ids, expected exactly %d"
                             % (u, n_commands))
    return True, n, u, "exactly %d rows over %d command_ids" % (n, u)


def pred_is_complete(arm, split, expected_rows):
    complete, n, _u, _why = pred_census(arm, split, expected_rows)
    return complete, n


def score_row(arm, split, cid, rep, command, raw, obj, truth, latency_ms, note=""):
    """One prediction row with per-field correctness, scored by llm/score.py."""
    ok = bool(score(obj, truth).strict_match) if isinstance(obj, dict) else False
    tl, pl = truth.get("location"), (obj or {}).get("location")
    return dict(arm=arm, split=split, command_id=cid, repeat=rep, command=command,
                raw=raw, pred_json=json.dumps(obj if isinstance(obj, dict) else {}),
                truth_json=json.dumps(truth), correct=ok,
                verb_correct=((obj or {}).get("action") == truth.get("action")),
                target_correct=("" if not truth.get("target")
                                else (obj or {}).get("target") == truth.get("target")),
                location_correct=("" if not tl else pl == tl),
                row_correct=("" if not rowof(tl) else rowof(pl) == rowof(tl)),
                col_correct=("" if not colof(tl) else colof(pl) == colof(tl)),
                # None = the arm has no latency of its own (the offline ensemble);
                # written empty so Stage 2 reports n/a, never a fake 0
                latency_ms="" if latency_ms is None else round(latency_ms, 3),
                note=note)


def write_preds(arm, split, rows):
    p = pred_path(arm, split)
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=PRED_COLS)
        w.writeheader()
        w.writerows(rows)
    return p


def preds_from_percommand(arm, split, recs, text_col, ms_col, note="", outdir=None):
    """Write pred_<arm>_<split>.csv from an A8/A10/A12 per-command record list.

    Added 2026-09-26. a8_speculative.py, a10_prefix_cache.py and
    a12_smalllm_drafter.py wrote only their own `*_percommand.csv`, so
    s1_repro_check.py and Stage 2 -- which read pred_<arm>_<split>.csv -- showed
    all three arms as NOT RUN YET even though the measurements existed. They now
    emit the standard file as well, from the same records, so there is one format
    for every arm and no second scoring path.

    `recs` may be the in-memory record list or an iterable of CSV rows: the two
    have the same keys. `text_col` is the arm's own output text (the VERIFIED
    output, not the greedy baseline) and `ms_col` its per-command latency, so the
    latency in the standard file is this arm's, measured on this run.
    """
    rows_in = load_split(split)
    cmds = [r["command"] for r in rows_in]
    truths = [truth_of(r) for r in rows_in]
    out = []
    for r in recs:
        cid = int(r["command_id"])
        raw_text = r.get(text_col) or ""
        try:
            obj = json.loads(raw_text)
        except Exception:
            obj = {}
        if not isinstance(obj, dict):
            obj = {}
        try:
            ms = float(r.get(ms_col) or 0.0)
        except (TypeError, ValueError):
            ms = 0.0
        out.append(score_row(arm, split, cid, 0, r.get("command") or cmds[cid],
                             raw_text, obj, truths[cid], ms, note=note))
    out.sort(key=lambda x: int(x["command_id"]))
    p = (Path(outdir) if outdir else OUT) / ("pred_%s_%s.csv" % (arm, split))
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=PRED_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(out)
    return p, len(out)


# ------------------------------------------------- cloud retries (gpt-4o-mini)
#
# Every gpt-4o-mini call in the final run -- the five V5 cloud cells and both
# cloud cascades -- goes through with_retries(). Added 2026-09-26.
#
#   * retried: connection errors, timeouts, HTTP 429 and HTTP 5xx;
#   * NOT retried: any other error (400, 401, 403, 404, a bug in our code) --
#     those fail at once and are recorded as model_error;
#   * up to RETRY_ATTEMPTS attempts, sleeping 2, 4, 8, 16, 32 s between them;
#   * the call is re-issued UNCHANGED: same prompt, same model, same
#     temperature. A retry repeats a request that never produced an answer; it
#     is not a second sample.
#
# The openai SDK retries twice on its own by default (max_retries=2), which
# would make one "attempt" up to three billed requests and the logged count an
# undercount. Its retries are switched OFF (max_retries=0) wherever we hold the
# client, so the retry count logged per row is the exact number of extra
# requests.

RETRY_ATTEMPTS = 6
RETRY_BASE_S = 2.0


def retryable(e):
    """True for a connection error, a timeout, HTTP 429 or HTTP 5xx."""
    try:
        import openai
    except ImportError:
        openai = None
    if openai is not None:
        if isinstance(e, (openai.APIConnectionError, openai.RateLimitError,
                          openai.InternalServerError)):   # APITimeoutError is a
            return True                                   # APIConnectionError
        if isinstance(e, openai.APIStatusError):
            code = getattr(e, "status_code", None)
            return code == 429 or (code is not None and code >= 500)
    return isinstance(e, (TimeoutError, ConnectionError))


def with_retries(call, what="gpt-4o-mini"):
    """-> (result, retries). Re-raises the last error with `.retries` attached."""
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            return call(), attempt - 1
        except Exception as e:                            # noqa: BLE001
            if attempt == RETRY_ATTEMPTS or not retryable(e):
                e.retries = attempt - 1
                raise
            wait = RETRY_BASE_S * 2 ** (attempt - 1)
            print("     retry %d/%d for %s in %.0f s after %s: %s"
                  % (attempt, RETRY_ATTEMPTS - 1, what, wait, type(e).__name__,
                     str(e)[:120]))
            time.sleep(wait)


def sdk_no_retries(openai_client):
    """The same client with the SDK's own retries off (see above)."""
    return openai_client.with_options(max_retries=0)


class RetryingClient:
    """A V5 cloud client whose complete() goes through with_retries().

    Wraps llm/clients.OpenAIClient without editing it (llm/ is read-only).
    Everything except complete() is delegated, so run_once sees the same name,
    temperature and last_thinking. `last_retries` holds the retry count of the
    most recent call, read by the runner IMMEDIATELY after run_once.
    """

    def __init__(self, inner):
        self._inner = inner
        self.last_retries = 0
        if hasattr(inner, "_client") and hasattr(inner._client, "with_options"):
            inner._client = sdk_no_retries(inner._client)

    def complete(self, prompt):
        try:
            out, self.last_retries = with_retries(lambda: self._inner.complete(prompt),
                                                  self._inner.name)
            return out
        except Exception as e:                            # noqa: BLE001
            self.last_retries = getattr(e, "retries", 0)
            raise

    def __getattr__(self, k):
        return getattr(self._inner, k)


def row_stage(r):
    """The stage of a pred_<arm>_<split>.csv row, read from its `note` JSON.

    A cloud cascade that exhausts its retries writes stage "model_error" there.
    Rows written before 2026-09-26 carry no stage and read as "".
    """
    note = r.get("note") or ""
    try:
        return (json.loads(note) or {}).get("stage", "") or ""
    except Exception:                                     # noqa: BLE001
        # a note truncated at 400 chars is not valid JSON; the marker survives
        return "model_error" if '"stage": "model_error"' in note else ""


def log_rerun(label, split, n_rows):
    """One line in the tracked force_rerun_log.csv. Its 4-column schema is kept;
    the flag that caused the rerun is part of `label`."""
    log = OUT / "force_rerun_log.csv"
    new = not log.exists()
    with log.open("a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["utc", "arm", "split", "rows_discarded"])
        w.writerow([time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    label, split, n_rows])
    return log


# ------------------------------------------------- efficiency display helpers
#
# One definition of how an energy or VRAM figure is SHOWN, used by s2_metrics,
# s4_metrics_final, s4_figures, s6_report and DEFENCE_NUMBERS (2026-09-28).

J_FLOOR_TEXT = "\u2248 0 (below GPU floor)"


def j_display(j):
    """A measured J/command for tables. <= 0 means the arm's load was inside
    the GPU's idle noise: shown as '~0 (below GPU floor)', never as a negative
    number and never dropped. Non-numeric cells pass through unchanged."""
    try:
        v = float(j)
    except (TypeError, ValueError):
        return j
    return J_FLOOR_TEXT if v <= 0 else j


_IDLE_LOGS = None


def idle_logs():
    """[(start_epoch, file_name, median_power_w, median_mem_used_mb)] for every
    idle baseline in power_logs/, oldest first. Timestamps are the epoch values
    inside the files, so no timezone is guessed."""
    global _IDLE_LOGS
    if _IDLE_LOGS is None:
        import statistics
        out = []
        for f in sorted(POWER.glob("idle_*.csv")):
            with f.open(encoding="utf-8", newline="") as fh:
                rs = list(csv.DictReader(fh))
            if rs:
                out.append((float(rs[0]["timestamp"]), f.name,
                            statistics.median(float(r["power_w"]) for r in rs),
                            statistics.median(float(r["mem_used_mb"]) for r in rs)))
        _IDLE_LOGS = sorted(out)
    return _IDLE_LOGS


def idle_for(eff_row):
    """The idle baseline in force when an efficiency row was measured: the
    latest idle log that started before the row's utc. Returns (file, median W,
    median memory.used MB), or None. Its median W must equal the row's recorded
    idle_median_w, which s2_metrics checks for every row."""
    import calendar
    try:
        t = calendar.timegm(time.strptime(eff_row["utc"], "%Y-%m-%dT%H:%M:%SZ"))
    except (KeyError, ValueError):
        return None
    prior = [l for l in idle_logs() if l[0] <= t]
    return prior[-1][1:] if prior else None


def vram_above_idle(eff_row):
    """peak_vram_mb minus the median idle memory.used of the baseline in force.
    The model's own VRAM; a CPU-only arm reads ~0 (small negatives are noise in
    the desktop's own allocation and are returned as measured)."""
    idle = idle_for(eff_row)
    try:
        return round(float(eff_row["peak_vram_mb"]) - idle[2], 1) if idle else None
    except (KeyError, TypeError, ValueError):
        return None


def vram_display(v):
    """VRAM above idle for tables: a value <= 0 (no model on the GPU; small
    negatives are the desktop's own allocation moving) is shown as "~0"."""
    try:
        return "\u2248 0" if float(v) <= 0 else v
    except (TypeError, ValueError):
        return v


# ------------------------------------------------- V5 cells: one outcome per command

def v5_cell_outcomes(rows, metric="action"):
    """{command_id: bool} for ONE V5 cell's rows, collapsed as A9 did:
    warm-up rows excluded, repeats collapsed by majority vote (sum*2 > n, a tie
    is False). metric 'row' keeps grid commands only."""
    by = {}
    for r in rows:
        if str(r.get("is_warmup", "")).lower() == "true":
            continue
        truth = json.loads(r["truth_json"] or "{}")
        if metric == "row":
            tl = truth.get("location")
            if not rowof(tl):
                continue
            try:
                pred = json.loads(r["pred_json"] or "{}")
            except ValueError:
                pred = {}
            ok = rowof((pred or {}).get("location") if isinstance(pred, dict) else None) == rowof(tl)
        else:
            ok = str(r["strict_match"]).lower() == "true"
        by.setdefault(r["command_id"], []).append(ok)
    return {c: (v[0] if len(v) == 1 else majority(v)) for c, v in by.items()}


def read_preds(arm, split):
    p = pred_path(arm, split)
    if not p.exists():
        return []
    return list(csv.DictReader(p.open(encoding="utf-8", newline="")))


def majority(vals):
    """A9's collapse rule: sum*2 > n (a tie is False). Stated, not silent."""
    vals = list(vals)
    return (sum(1 for v in vals if v) * 2) > len(vals)


def env_banner():
    import torch
    print("  torch %s cuda=%s | transformers %s"
          % (torch.__version__, torch.cuda.is_available(),
             __import__("transformers").__version__))
    if torch.cuda.is_available():
        print("  gpu: %s" % torch.cuda.get_device_name(0))
