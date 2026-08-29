r"""Convert schema/schema_v2.json into real JSON Schema for constrained decoding.

schema_v2.json is NOT JSON Schema -- it is a bespoke dict of verbs/targets/
locations. It also CANNOT be expressed as one flat object schema, because
`required` depends on the value of `action`:

    pick_place -> target, location      move_to -> location
    fetch/hold/inspect -> target        stop/reject -> (none)

Four distinct required-sets. A flat schema can only say "these keys are always
required", so the mask would be lost and the guarantee with it. The conversion
therefore emits `oneOf` over one closed subschema per verb.

Both back ends need every property listed in `required` for their strict modes,
which the per-verb split gives for free: each branch lists exactly its own
fields and nothing else.
"""
import json


def to_flat_json_schema(schema: dict) -> dict:
    """Flat variant for OpenAI Structured Outputs.

    OpenAI rejects a root schema carrying oneOf/anyOf:
        "schema must have type 'object' and not have
         'oneOf'/'anyOf'/'allOf'/'enum'/'const'/'not' at the top level"

    So the per-verb required-field mask CANNOT be expressed there. This flat
    form still forbids illegal verbs, illegal targets, illegal location symbols
    and malformed JSON -- but it cannot enforce "stop takes no target".

    That residual check is not lost, only moved: schema/validate_object.py runs
    on every response and rejects field-mask violations. Constrained decoding is
    the belt; the validator is the braces. Ollama gets the full oneOf form via
    to_json_schema() and needs no such caveat.
    """
    return {
        "type": "object",
        "properties": {
            "action": {"enum": list(schema["verbs"])},
            "target": {"enum": schema["targets"]},
            "location": {"enum": schema["locations"]},
        },
        "required": ["action"],
        "additionalProperties": False,
    }


def to_json_schema(schema: dict, *, include_modifiers: bool = False) -> dict:
    verbs = schema["verbs"]
    targets = schema["targets"]
    locations = schema["locations"]
    branches = []
    for verb, spec in verbs.items():
        props = {"action": {"const": verb}}
        req = ["action"]
        for f in spec["required"]:
            props[f] = {"enum": targets if f == "target" else locations}
            req.append(f)
        # modifiers is ALLOWED but never REQUIRED: real models omit it, and the
        # scorer drops it when falsy. Requiring it would reject correct output.
        props["modifiers"] = {"type": "array", "items": {"type": "string"}}
        if include_modifiers:
            req.append("modifiers")
        branches.append({
            "type": "object",
            "properties": props,
            "required": req,
            "additionalProperties": False,
        })
    return {"type": "object", "oneOf": branches}


def validates(obj, js) -> bool:
    """Minimal checker so this module can self-test without a jsonschema dep."""
    if not isinstance(obj, dict):
        return False
    for b in js["oneOf"]:
        props, req = b["properties"], b["required"]
        if set(obj) - set(props):
            continue
        if any(k not in obj for k in req):
            continue
        ok = True
        for k, v in obj.items():
            spec = props[k]
            if "const" in spec and v != spec["const"]:
                ok = False
            elif "enum" in spec and v not in spec["enum"]:
                ok = False
            elif spec.get("type") == "array" and not isinstance(v, list):
                ok = False
        if ok:
            return True
    return False


if __name__ == "__main__":
    s = json.load(open("schema/schema_v2.json", encoding="utf-8"))
    js = to_json_schema(s)
    print(f"branches: {len(js['oneOf'])} (one per verb)")
    print(json.dumps(js["oneOf"][0], indent=2)[:300])
    good = [
        {"action": "stop", "modifiers": []},
        {"action": "pick_place", "target": "bottle", "location": "grid_A1", "modifiers": []},
        {"action": "move_to", "location": "box_1", "modifiers": []},
    ]
    bad = [
        {"action": "pick_place", "target": "bottle", "modifiers": []},   # missing location
        {"action": "drop", "target": "bottle", "modifiers": []},          # illegal verb
        {"action": "stop", "target": "bottle", "modifiers": []},          # field not allowed
        {"action": "move_to", "location": "grid_Z9", "modifiers": []},    # illegal symbol
    ]
    for o in good:
        assert validates(o, js), f"should pass: {o}"
    for o in bad:
        assert not validates(o, js), f"should FAIL: {o}"
    print("self-check: 3 legal accepted, 4 illegal rejected -- OK")
