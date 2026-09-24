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


def pred_is_complete(arm, split, expected_rows):
    p = pred_path(arm, split)
    if not p.exists():
        return False, 0
    n = sum(1 for _ in p.open(encoding="utf-8", newline="")) - 1
    return n >= expected_rows, max(n, 0)


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
                latency_ms=round(latency_ms, 3), note=note)


def write_preds(arm, split, rows):
    p = pred_path(arm, split)
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=PRED_COLS)
        w.writeheader()
        w.writerows(rows)
    return p


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
