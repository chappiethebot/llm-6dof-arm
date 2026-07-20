"""llm/score.py -- Phase 3, step 3: grade ONE extracted object against ground truth.

Scores the SAME object two ways (frozen policy, decided by inspecting the data):
  - strict:      keys AND structure must be exactly as the schema specifies
                 (flat, "action" is a string). Exact match after ignoring empty modifiers.
  - normalized:  REPRESENTATIONAL forgiveness only. Unwraps a nested action object
                 ({"action": {"verb": X, ...}} -> flat), treats key aliases
                 verb/command as action, and canonicalizes value separators/case
                 (box 1 == box_1). It forgives FORMAT/STRUCTURE, never MEANING:
                 a wrong verb or wrong target still fails under normalized.
Empty modifiers (absent or []) are ignored in both. A non-empty modifiers list IS compared.

This is scoring only. It does NOT call a model and does NOT re-generate -- it grades
text that llm/pipeline.py already produced. The Phase 5 harness calls this across
the full dataset x 20 cells x >=5 runs. This file is the grader, not the harness.
"""
import re
from dataclasses import dataclass

_KEY_ALIASES = {"verb": "action", "command": "action"}
_IGNORE_IF_EMPTY = {"modifiers"}


@dataclass
class ScoreResult:
    strict_match: bool
    normalized_match: bool
    category: str  # why it missed (semantic view, after representational normalization)


def _apply_aliases(obj: dict) -> dict:
    """Return a copy with alias keys renamed to 'action' (only if 'action' not already set)."""
    out = {}
    for k, v in obj.items():
        nk = _KEY_ALIASES.get(k, k)
        if nk in out:            # don't clobber a real 'action' with an alias
            continue
        out[nk] = v
    return out


def _canon_value(v):
    """Canonicalize a leaf value for representational comparison: lowercase and
    strip non-alphanumerics, so 'box 1' == 'box_1' == 'Box1'. Lists canon each item.
    Forgives SEPARATOR/CASE only -- it never merges distinct vocabulary
    (box_1..box_3, toy_car, bottle all stay distinct)."""
    if v is None:
        return ""
    if isinstance(v, str):
        return re.sub(r"[^a-z0-9]", "", v.lower())
    if isinstance(v, list):
        return tuple(sorted(str(_canon_value(x)) for x in v))
    if isinstance(v, dict):
        return tuple(sorted((str(k), str(_canon_value(x))) for k, x in v.items()))
    return str(v)


def _normalize_repr(obj: dict) -> dict:
    """Representational normalization for the NORMALIZED policy only.
    Unwraps a nested action object, applies key aliases, canonicalizes values.
    Forgives STRUCTURE and FORMAT, never MEANING."""
    if not isinstance(obj, dict):
        return obj
    # Unwrap: if 'action' (or an alias key) holds a dict, lift its verb + fields up.
    act = obj.get("action")
    if act is None:
        for ak in _KEY_ALIASES:
            if isinstance(obj.get(ak), dict):
                act = obj[ak]
                break
    if isinstance(act, dict):
        flat = {k: v for k, v in obj.items() if _KEY_ALIASES.get(k, k) != "action"}
        verb = act.get("action") or act.get("verb") or act.get("command")
        if verb is not None:
            flat["action"] = verb
        for k, v in act.items():
            nk = _KEY_ALIASES.get(k, k)
            if nk == "action":
                continue
            flat.setdefault(nk, v)
        obj = flat
    obj = _apply_aliases(obj)
    obj = {k: _canon_value(v) for k, v in obj.items()}
    return obj


def _canonical(obj: dict) -> dict:
    """Drop empty-ignorable fields so absent == [] compares equal."""
    return {k: v for k, v in obj.items() if not (k in _IGNORE_IF_EMPTY and not v)}


def _equal(pred: dict, truth: dict) -> bool:
    return _canonical(pred) == _canonical(truth)


def _categorize(pred: dict, truth: dict) -> str:
    """Best-effort reason the prediction differs from truth. For failure breakdowns.
    Runs on the representationally-normalized view, so a merely mis-formatted but
    correct object categorizes as 'correct'; remaining categories are SEMANTIC errors."""
    p, t = _canonical(pred), _canonical(truth)
    if p == t:
        return "correct"
    pa, ta = p.get("action"), t.get("action")
    if pa is None:
        return "no_action_key"
    if pa != ta:
        return "wrong_verb"
    for f in ("target", "location"):
        if t.get(f) != p.get(f):
            return f"wrong_or_missing_{f}"
    return "other_field_mismatch"


def score(pred: dict, truth: dict) -> ScoreResult:
    strict = _equal(pred, truth)
    normalized = _equal(_normalize_repr(pred), _normalize_repr(truth))
    category = _categorize(_normalize_repr(pred), _normalize_repr(truth))  # semantic view
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
        # --- representational normalization cases (the score.py fix) ---
        "nested_correct":    {"action": {"verb": "pick_place", "target": "bottle", "location": "box_1"}},  # gpt CoT: right content, wrong shape -> strict F, norm T
        "nested_wrong_verb": {"action": {"verb": "move_to", "location": "box_1"}},  # deepseek: wrong verb under nesting -> strict F, norm F
        "sep_variant":       {"action": "pick_place", "target": "bottle", "location": "box 1"},  # 'box 1' == 'box_1' -> norm T
    }
    print(f"TRUTH: {truth}\n")
    for name, pred in cases.items():
        r = score(pred, truth)
        print(f"{name:18} strict={str(r.strict_match):5} normalized={str(r.normalized_match):5} category={r.category}")
