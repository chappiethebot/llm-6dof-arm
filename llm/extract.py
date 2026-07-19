"""llm/extract.py -- Phase 3, step 2: raw model text -> clean dict, or a loud failure.
This layer ONLY answers "is this parseable JSON?". It does NOT check the schema -- that's step 3.
"""
import json
import re
from dataclasses import dataclass


@dataclass
class ExtractResult:
    """ok=True -> data is a dict. ok=False -> data is None and reason says why."""
    ok: bool
    data: dict | None
    reason: str  # "" when ok, else a short failure label for your metrics


def extract_json(raw: str) -> ExtractResult:
    if raw is None or not raw.strip():
        return ExtractResult(False, None, "empty_output")
    text = raw.strip()

    # 1) Strip a markdown code fence if the whole thing is wrapped in one (gpt-4o-mini did this).
    #    Handles ```json ... ``` and plain ``` ... ```
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    # 2) First attempt: parse the (de-fenced) text directly. Clean models land here.
    parsed = _try_load(text)
    if parsed is not None:
        return _finish(parsed)

    # 3) Fallback: the model wrapped JSON in prose (or reasoned first, then emitted
    #    the answer -- cot/reasoning models). Scan for ALL balanced {...} objects and
    #    take the LAST one that parses. The cot prompt puts the answer on the last
    #    line, so the last valid object is the answer; earlier braces are schema
    #    echoes or reasoning fragments. On single-object replies the last object IS
    #    the only object, so the clean strategies are unaffected.
    for cand in reversed(_find_json_objects(text)):
        parsed = _try_load(cand)
        if parsed is not None and isinstance(parsed, dict):
            return _finish(parsed)

    return ExtractResult(False, None, "no_valid_json")


def _find_json_objects(text: str) -> list[str]:
    """Return every brace-balanced {...} substring, in order of appearance.
    Balanced-brace scan (not regex) so nested objects are captured whole."""
    objs, depth, start = [], 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    objs.append(text[start:i + 1])
                    start = None
    return objs


def _try_load(s: str):
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return None


def _finish(parsed) -> ExtractResult:
    # Must be a JSON object (dict), not a list/number/string that happens to parse.
    if not isinstance(parsed, dict):
        return ExtractResult(False, None, "not_an_object")
    return ExtractResult(True, parsed, "")


if __name__ == "__main__":
    # Test cases = the four raw outputs you actually saw + the mess you'll hit later.
    cases = {
        "gpt_fenced":        '```json\n{"status": "ok"}\n```',
        "llama_clean":       '{"status": "ok"}',
        "qwen3b_clean":      '{"status": "ok"}',
        "qwen05b_pretty":    '{\n  "status": "ok"\n}',
        "prose_wrapped":     'Sure! Here is the JSON:\n{"action": "pick_place", "target": "bottle"}',
        "plain_fence":       '```\n{"action": "stop"}\n```',
        "garbage_no_json":   'I cannot help with that.',
        "empty":             '',
        "json_but_a_list":   '[1, 2, 3]',
        "cot_reasoning":     'Step 1: the verb is {pick_place}. Final answer:\n{"action": "pick_place", "target": "toy_car", "location": "box_2"}',
    }
    for name, raw in cases.items():
        r = extract_json(raw)
        status = "OK  " if r.ok else "FAIL"
        print(f"[{status}] {name:18} -> ok={r.ok}  data={r.data}  reason=\x27{r.reason}\x27")
