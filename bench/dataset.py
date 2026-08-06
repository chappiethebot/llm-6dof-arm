"""
bench/dataset.py -- the one place the seen/exemplar split is defined.

Why a shared module: few-shot draws its in-prompt examples from seen.jsonl. If
those same commands also appear as TEST items, the model is shown the answer to
its own question -- within-seen contamination. So we carve off a small, FIXED
set of exemplars and remove them from the test set FOR EVERY STRATEGY, so all
four strategies are scored on the identical test set (same denominator = a fair
comparison table).

Fixed exemplars -- NOT per-command nearest-neighbour retrieval. Per-command
retrieval is RAG, which the thesis plan rejected from the critical path. Keeping
the exemplars fixed keeps few-shot as few-shot.

held_out is never touched here. This module reads a seen file only.

V2 SELECTION CHANGE
-------------------
V1 took the first k rows and that was fine: seen.jsonl was homogeneous, so the
first 3 rows were representative. In seen_v2.jsonl the carried-over V1 container
rows come FIRST, so `rows[:3]` yields three box commands and demonstrates ZERO
grid locations -- for an experiment whose entire subject is grid locations.
Few-shot would then be no better informed than zero-shot, and that cell of the
grid would be measuring nothing.

So for schema-v2 files, exemplars are chosen by location CLASS: one corner, one
edge, one runtime symbol. Selection is still fixed and deterministic (first match
per class, in file order, no shuffle, no dependence on the command being scored).

DELIBERATELY NOT REPRESENTED in the exemplars:
  - interior (grid_B2) and container (box_*) locations
  - the move_to / fetch / hold / inspect / stop / reject verbs
Showing every class would make few-shot a lookup table rather than a test of
generalization. Three shots cannot cover fourteen locations; the point is to
demonstrate the FORMAT and the grid naming scheme, not to enumerate the answers.
"""
import json
from pathlib import Path

# How many seen rows are reserved as few-shot exemplars and excluded from the
# test set. k=3: enough to demonstrate the format; leaves 128/131 rows for test
# in V1, 200/203 in V2.
DEFAULT_K = 3

# For schema-v2 files: which location classes the k exemplars must cover, in
# order. Truncated to k. Kept explicit so the choice is auditable in the thesis.
V2_EXEMPLAR_CLASSES = ["corner", "edge", "runtime_symbol"]

_SCHEMA_V2 = Path("schema/schema_v2.json")


def _location_classes():
    """Return {location: class} from schema_v2.json, or {} if it is absent."""
    if not _SCHEMA_V2.exists():
        return {}
    return json.loads(_SCHEMA_V2.read_text(encoding="utf-8")).get("location_classes", {})


def _pick_by_class(rows, k):
    """First row per class in V2_EXEMPLAR_CLASSES, in file order. Returns indices
    or None if the classes cannot be covered (caller falls back to first-k)."""
    classes = _location_classes()
    if not classes:
        return None
    wanted = V2_EXEMPLAR_CLASSES[:k]
    chosen = {}
    for i, r in enumerate(rows):
        loc = r.get("expected", {}).get("location")
        cls = classes.get(loc)
        if cls in wanted and cls not in chosen:
            chosen[cls] = i
        if len(chosen) == len(wanted):
            break
    if len(chosen) < len(wanted):
        return None
    return [chosen[c] for c in wanted]


def load_seen_split(k: int = DEFAULT_K, seen_path: str = "data/seen.jsonl"):
    """
    Return (exemplars, test_rows).

    exemplars : k rows used as few-shot shots ONLY.
    test_rows : the remaining rows, used as the scored test set for ALL strategies.

    V1 files: the first k rows.
    V2 files: one row per location class (corner, edge, runtime_symbol), taken in
              file order. Falls back to first-k if the classes cannot be covered.

    Deterministic either way: no shuffle, no randomness, so every run and every
    machine gets the identical split. command_id in the harness is the index INTO
    test_rows, so it stays stable as long as the dataset file is append-only.
    """
    path = Path(seen_path)
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    if k < 0 or k >= len(rows):
        raise ValueError(f"k={k} invalid for {len(rows)} seen rows")

    idx = _pick_by_class(rows, k) if "_v2" in path.name else None
    if idx is None:
        exemplars = rows[:k]
        test_rows = rows[k:]
    else:
        picked = set(idx)
        exemplars = [rows[i] for i in idx]
        test_rows = [r for i, r in enumerate(rows) if i not in picked]
    return exemplars, test_rows


if __name__ == "__main__":
    import sys
    for p in ("data/seen.jsonl", "data/seen_v2.jsonl"):
        if not Path(p).exists():
            print(f"{p}: not found, skipped")
            continue
        ex, test = load_seen_split(seen_path=p)
        classes = _location_classes()
        print(f"\n{p}   exemplars={len(ex)}  test={len(test)}")
        for e in ex:
            loc = e["expected"].get("location", "-")
            print(f"   [{classes.get(loc, 'v1'):14s}] {e['command']}  ->  {loc}")
        # contamination check: no exemplar may also appear as a test item
        overlap = {e["command"] for e in ex} & {t["command"] for t in test}
        print(f"   exemplar/test overlap: {len(overlap)}"
              f"{'  <-- CONTAMINATION' if overlap else ''}")
