"""
bench/dataset.py -- the one place the seen/exemplar split is defined.

Why a shared module: few-shot draws its in-prompt examples from seen.jsonl. If
those same commands also appear as TEST items, the model is shown the answer to
its own question -- within-seen contamination. So we carve off a small, FIXED
set of exemplars and remove them from the test set FOR EVERY STRATEGY, so all
four strategies are scored on the identical test set (same denominator = a fair
comparison table).

Fixed exemplars (first k rows, no shuffle) -- NOT per-command nearest-neighbour
retrieval. Per-command retrieval is RAG, which the thesis plan rejected from the
critical path. Keeping the exemplars fixed keeps few-shot as few-shot.

held_out is never touched here. This module reads seen.jsonl only.
"""
import json
from pathlib import Path

# How many seen rows are reserved as few-shot exemplars and excluded from the
# test set. k=3: enough to demonstrate the format; leaves 128/131 rows for test.
DEFAULT_K = 3


def load_seen_split(k: int = DEFAULT_K, seen_path: str = "data/seen.jsonl"):
    """
    Return (exemplars, test_rows).

    exemplars : the first k rows of seen.jsonl, used as few-shot shots ONLY.
    test_rows : the remaining rows, used as the scored test set for ALL strategies.

    Deterministic: first k rows, no shuffle, so every run and every machine gets
    the identical split. command_id in the harness is the index INTO test_rows,
    so it stays stable as long as seen.jsonl is append-only.
    """
    path = Path(seen_path)
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    if k < 0 or k >= len(rows):
        raise ValueError(f"k={k} invalid for {len(rows)} seen rows")
    exemplars = rows[:k]
    test_rows = rows[k:]
    return exemplars, test_rows
