"""llm/score.py — Phase 3, step 3: grade ONE extracted object against ground truth.

Scores the SAME object two ways (frozen policy, decided by inspecting the data):
  - strict:      keys must be exactly as the schema specifies ('action').
  - normalized:  key aliases 'verb'/'command' are treated as 'action'; nothing else changes.
Empty modifiers (absent or []) are ignored in both. A non-empty modifiers list IS compared.

This is scoring only. It does NOT call a model and does NOT re-generate — it grades
text that llm/pipeline.py already produced. The Phase 5 harness will call this across
the full dataset x 16 cells x >=5 runs. This file is the grader, not the harness.
"""
from dataclasses import dataclass

_KEY_ALIASES = {"verb": "action", "command": "action"}
_IGNORE_IF_EMPTY = {"modifiers"}


@dataclass
class ScoreResult:
    strict_match: bool
    normalized_match: bool
    category: str  # why it missed (worst-case across policies): see _categorize


def _apply_aliases(obj: dict) -> dict:
    """Return a copy with alias keys renamed to 'action' (only if 'action' not already set)."""
    out = {}
    for k, v in obj.items():
        nk = _KEY_ALIASES.get(k, k)
        if nk in out:            # don't clobber a real 'action' with an alias
            continue
        out[nk] = v
    return out


def _canonical(obj: dict) -> dict:
    """Drop empty-ignorable fields so absent == [] compares equal."""
    return {k: v for k, v in obj.items() if not (k in _IGNORE_IF_EMPTY and not v)}


def _equal(pred: dict, truth: dict) -> bool:
    return _canonical(pred) == _canonical(truth)


def _categorize(pred: dict, truth: dict) -> str:
    """Best-effort reason the prediction differs from truth. For failure breakdowns."""
    p, t = _canonical(pred), _canonical(truth)
    if p == t:
        return "correct"
    # action present but wrong?
    pa, ta = p.get("action"), t.get("action")
    if pa is None:
        return "no_action_key"           # used 'verb'/'command'/none instead of 'action'
    if pa != ta:
        return "wrong_verb"              # e.g. predicted 'put' or 'move_to' for a pick_place
    # right verb — must be a field problem
    for f in ("target", "location"):
        if t.get(f) != p.get(f):
            return f"wrong_or_missing_{f}"
    return "other_field_mismatch"


def score(pred: dict, truth: dict) -> ScoreResult:
    strict = _equal(pred, truth)
    normalized = _equal(_apply_aliases(pred), truth)
    category = _categorize(_apply_aliases(pred), truth)  # categorize on the lenient view
    return ScoreResult(strict, normalized, category)


if __name__ == "__main__":
    truth = {"action": "pick_place", "target": "bottle", "location": "box_1", "modifiers": []}
    cases = {
        "perfect_no_mods":   {"action": "pick_place", "target": "bottle", "location": "box_1"},
        "perfect_with_mods": {"action": "pick_place", "target": "bottle", "location": "box_1", "modifiers": []},
        "alias_verb_key":    {"verb": "pick_place", "target": "bottle", "location": "box_1"},   # llama CoT failure
        "wrong_verb_put":    {"action": "put", "target": "bottle", "location": "box_1"},        # qwen CoT failure
        "wrong_location":    {"action": "pick_place", "target": "bottle", "location": "box_2"},
        "missing_location":  {"action": "pick_place", "target": "bottle"},
    }
    print(f"TRUTH: {truth}\n")
    for name, pred in cases.items():
        r = score(pred, truth)
        print(f"{name:18} strict={str(r.strict_match):5} normalized={str(r.normalized_match):5} category={r.category}")