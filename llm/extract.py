"""llm/extract.py — Phase 3, step 2: raw model text -> clean dict, or a loud failure.
This layer ONLY answers 'is this parseable JSON?'. It does NOT check the schema — that's step 3.
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

    # 3) Fallback: the model wrapped JSON in prose. Grab the first {...} block and try that.
    #    Non-greedy from the first '{' to the last '}' — good enough for single-object replies.
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        parsed = _try_load(brace.group(0))
        if parsed is not None:
            return _finish(parsed)

    return ExtractResult(False, None, "no_valid_json")


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
    }
    for name, raw in cases.items():
        r = extract_json(raw)
        status = "OK  " if r.ok else "FAIL"
        print(f"[{status}] {name:18} -> ok={r.ok}  data={r.data}  reason='{r.reason}'")