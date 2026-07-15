"""schema/validate_object.py — Phase 3, step 3: validate ONE command dict against schema.json.
Single source of truth for "is this a legal schema object?". The rules mirror the
per-row checks in the Phase 1 validate.py, lifted out so both the dataset-checker
and the LLM-scorer can share them.

This is step 3 (schema-legality), NOT step 2 (JSON parseability). Input here is
already a dict from llm/extract.py.
"""
import json
from dataclasses import dataclass, field


@dataclass
class ValidationResult:
    """ok=True -> obj is a legal schema command. ok=False -> reasons lists every rule broken."""
    ok: bool
    reasons: list[str] = field(default_factory=list)


def load_schema(path: str = "schema/schema.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def validate_object(obj: dict, schema: dict) -> ValidationResult:
    verbs = schema["verbs"]
    targets = set(schema["targets"])
    locations = set(schema["locations"])
    reasons = []

    # 0. Must actually be a dict (extractor guarantees this, but step 3 shouldn't assume it).
    if not isinstance(obj, dict):
        return ValidationResult(False, ["not_a_dict"])

    act = obj.get("action")

    # 1. action must be a known verb. If it's not, no point checking fields for a verb we don't know.
    if act not in verbs:
        return ValidationResult(False, [f"unknown_verb:{act!r}"])

    # 2. required fields present and non-empty.
    for req in verbs[act]["required"]:
        if req not in obj or obj[req] in (None, "", []):
            reasons.append(f"missing_required:{req}")

    # 3. target, if present, must be legal.
    if "target" in obj and obj["target"] not in targets:
        reasons.append(f"bad_target:{obj['target']!r}")

    # 4. location, if present, must be legal.
    if "location" in obj and obj["location"] not in locations:
        reasons.append(f"bad_location:{obj['location']!r}")

    # 5. no stray fields. 'modifiers' allowed on every verb (schema modifiers_allowed=true).
    allowed = set(verbs[act]["required"]) | {"action", "modifiers"}
    for key in obj:
        if key not in allowed:
            reasons.append(f"unexpected_field:{key}")

    return ValidationResult(len(reasons) == 0, reasons)


if __name__ == "__main__":
    schema = load_schema()
    cases = {
        "valid_pick_place":   {"action": "pick_place", "target": "bottle", "location": "box_1"},
        "valid_stop":         {"action": "stop"},
        "valid_with_mods":    {"action": "fetch", "target": "toy_car", "modifiers": ["gently"]},
        "missing_location":   {"action": "pick_place", "target": "bottle"},          # from your prose_wrapped case
        "unknown_verb":       {"action": "banana", "target": "bottle"},
        "bad_target":         {"action": "fetch", "target": "banana"},
        "bad_location":       {"action": "move_to", "location": "mars"},
        "stray_field":        {"action": "stop", "target": "bottle"},
        "empty_dict":         {},
    }
    for name, obj in cases.items():
        r = validate_object(obj, schema)
        status = "VALID  " if r.ok else "INVALID"
        print(f"[{status}] {name:18} -> ok={r.ok}  reasons={r.reasons}")