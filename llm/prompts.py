"""llm/prompts.py — Phase 3: build the model prompt. One function, four strategies.

strategy in {"zero_shot", "few_shot", "cot", "role_based"}.

INTEGRITY: few-shot examples are INJECTED by the caller (`examples` arg), never
read from a file here. This keeps 'examples come only from seen.jsonl, and never
overlap the command being scored' enforceable at the eval layer, not buried here.

This step builds STRINGS only. No model is called here — that's the pipeline step.
"""
import json


def load_schema(path: str = "schema/schema.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _schema_spec(schema: dict) -> str:
    lines = ["Verbs (with required fields):"]
    for verb, spec in schema["verbs"].items():
        req = spec["required"]
        lines.append(f"- {verb}: requires {', '.join(req)}" if req else f"- {verb}: (no fields)")
    lines.append("")
    lines.append("Allowed targets: " + ", ".join(schema["targets"]))
    lines.append("Allowed locations: " + ", ".join(schema["locations"]))
    return "\n".join(lines)


_OUTPUT_RULE = (
    'Output ONLY a single JSON object with an "action" field plus that verb\'s '
    "required fields. No explanation, no markdown, no code fences."
)


def _format_example(ex: dict) -> str:
    # ex is a seen.jsonl-style row: {"command": ..., "expected": {...}}
    return f'Command: {ex["command"]}\nJSON: {json.dumps(ex["expected"])}'


def build_prompt(command: str, strategy: str, schema: dict, examples: list[dict] | None = None) -> str:
    spec = _schema_spec(schema)
    base = f"Convert the robot command into a schema JSON object.\n\n{spec}\n\n"

    if strategy == "zero_shot":
        return f"{base}{_OUTPUT_RULE}\n\nCommand: {command}\nJSON:"

    if strategy == "few_shot":
        if not examples:
            raise ValueError("few_shot requires injected examples (from seen.jsonl only)")
        shots = "\n\n".join(_format_example(e) for e in examples)
        return f"{base}{_OUTPUT_RULE}\n\nExamples:\n{shots}\n\nCommand: {command}\nJSON:"

    if strategy == "cot":
        return (
            f"Convert the robot command into a schema JSON object.\n\n{spec}\n\n"
            f"Think step by step:\n"
            f"1. Which verb does the command mean?\n"
            f"2. What are that verb's required fields, and what value does the command give for each?\n"
            f"Write your reasoning, then on the LAST line write the JSON object alone "
            f'(with "action" plus required fields, no code fences).\n\n'
            f"Command: {command}\n"
        )

    if strategy == "role_based":
        return (
            f"You are a strict robot command parser. You do not chat. Your only job is to "
            f"map one natural-language command to one schema JSON object.\n\n"
            f"{spec}\n\n{_OUTPUT_RULE}\n\nCommand: {command}\nJSON:"
        )

    raise ValueError(f"unknown strategy: {strategy!r}")


STRATEGIES = ["zero_shot", "few_shot", "cot", "role_based"]


if __name__ == "__main__":
    schema = load_schema()
    # TEST-ONLY examples. Real runs inject these from seen.jsonl (never held_out,
    # never the row being scored). Never hardcode few-shot examples in production.
    demo_examples = [
        {"command": "grab the toy car", "expected": {"action": "fetch", "target": "toy_car"}},
        {"command": "stop now", "expected": {"action": "stop"}},
    ]
    cmd = "put the bottle in box 1"
    for strat in STRATEGIES:
        print("=" * 70)
        print(f"STRATEGY: {strat}")
        print("-" * 70)
        ex = demo_examples if strat == "few_shot" else None
        print(build_prompt(cmd, strat, schema, examples=ex))
    print("=" * 70)