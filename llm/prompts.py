"""llm/prompts.py — Phase 3: build the model prompt. One function, six strategies.

strategy in {"zero_shot", "few_shot", "cot", "role_based", "rag", "classifier"}.

INTEGRITY: few-shot examples are INJECTED by the caller (`examples` arg), never
read from a file here. This keeps 'examples come only from seen.jsonl, and never
overlap the command being scored' enforceable at the eval layer, not buried here.

This step builds STRINGS only. No model is called here — that's the pipeline step.

V2: schema_version 2 introduces opaque location names (grid_A1 ... grid_C3). A V1
name like `table_left` is self-describing; `grid_A1` is not. Without a key, the
model cannot map "the far left corner" to a symbol and is reduced to guessing.
_grid_legend() emits that key, and ONLY for schemas that declare schema_version
>= 2 — so V1 prompts are byte-identical to what produced the existing 22,520 rows.

RUNG 2 (rag): "rag" shares the few_shot branch EXACTLY. Same wrapper, same shot
count, same formatting — the ONLY difference is that the caller selected the
examples by per-command similarity (llm/retrieve.py) instead of using the k fixed
exemplars. Written this way deliberately: if RAG had its own prompt string, any
accuracy difference between the few_shot and rag cells could be the wrapper
rather than the retrieval, and the rung-1 vs rung-2 comparison would be dead.

RUNG 4 (classifier): the DistilBERT classifier has NO prompt. It is an encoder
over the command string, not a generator conditioned on instructions. The
"classifier" strategy therefore returns the command VERBATIM. Two reasons this
is a strategy rather than a special case in the pipeline:
  1. ONE measurement path. run_once -> extract_json -> validate_object is
     unchanged, so the classifier is scored by exactly the code that scored
     every other rung.
  2. `latency_s` wraps client.complete() only, so for rung 4 it times tokenise +
     forward and nothing else — which is the CPU number the thesis reports.
NOTE: pairing "classifier" with an LLM model is an operator error. The model
would receive a bare command with no schema and no output rule. It fails loudly
(near-zero accuracy), it does not fail silently, but do not do it.

STRATEGIES is unchanged — it still lists only the four rung-1 strategies, so
bench/run_grid.py keeps producing exactly the frozen 20-cell base grid. "rag" is
listed separately in STRATEGIES_RAG and runs as its own 5 cells; "classifier" in
STRATEGIES_CLS and runs as 1 cell.
"""
import json


def load_schema(path: str = "schema/schema.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _grid_legend(schema: dict) -> str:
    """Key for the V2 grid naming scheme. Empty string for V1 schemas."""
    if schema.get("schema_version", 1) < 2:
        return ""
    if not any(loc.startswith("grid_") for loc in schema.get("locations", [])):
        return ""
    return (
        "\nGrid locations are squares on the table surface, named grid_<ROW><COL>.\n"
        "Rows:    A = far from the robot,  B = middle,  C = near the robot.\n"
        "Columns: 1 = left,  2 = middle,  3 = right.\n"
        "max_reach = the furthest point the arm can reach; resolved by the robot.\n"
        "nearest_free_cell = whichever square is currently empty; resolved by the robot.\n"
    )


def _schema_spec(schema: dict) -> str:
    lines = ["Verbs (with required fields):"]
    for verb, spec in schema["verbs"].items():
        req = spec["required"]
        lines.append(f"- {verb}: requires {', '.join(req)}" if req else f"- {verb}: (no fields)")
    lines.append("")
    lines.append("Allowed targets: " + ", ".join(schema["targets"]))
    lines.append("Allowed locations: " + ", ".join(schema["locations"]))
    legend = _grid_legend(schema)
    if legend:
        lines.append(legend)
    return "\n".join(lines)


_OUTPUT_RULE = (
    'Output ONLY a single JSON object with an "action" field plus that verb\'s '
    "required fields. No explanation, no markdown, no code fences."
)


def _format_example(ex: dict) -> str:
    # ex is a seen.jsonl-style row: {"command": ..., "expected": {...}}
    return f'Command: {ex["command"]}\nJSON: {json.dumps(ex["expected"])}'


def build_prompt(command: str, strategy: str, schema: dict, examples: list[dict] | None = None) -> str:
    # Rung 4 dispatches FIRST, before any schema string is built. The classifier
    # consumes the raw command; a legend it cannot read would only be truncated
    # away by the tokeniser at max_len, silently destroying the input.
    if strategy == "classifier":
        return command

    spec = _schema_spec(schema)
    base = f"Convert the robot command into a schema JSON object.\n\n{spec}\n\n"

    if strategy == "zero_shot":
        return f"{base}{_OUTPUT_RULE}\n\nCommand: {command}\nJSON:"

    # few_shot and rag share this branch ON PURPOSE. Identical wrapper, identical
    # shot count. Only the SELECTION of `examples` differs (fixed vs retrieved).
    if strategy in ("few_shot", "rag"):
        if not examples:
            if strategy == "rag":
                raise ValueError(
                    "rag requires retrieved examples; the caller must pass them from "
                    "llm/retrieve.py (corpus = data/train_v2.jsonl only)"
                )
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


# Rung 1. Do NOT append "rag" or "classifier" here — bench/run_grid.py iterates
# this list and the base grid is frozen at 5 models x 4 strategies = 20 cells.
STRATEGIES = ["zero_shot", "few_shot", "cot", "role_based"]

# Rung 2. Run as its own 5 cells (5 models x 1 strategy).
STRATEGIES_RAG = ["rag"]

# Rung 4. Run as its own 1 cell (1 classifier x 1 strategy).
STRATEGIES_CLS = ["classifier"]

# Everything build_prompt understands, for CLI choices and validation.
ALL_STRATEGIES = STRATEGIES + STRATEGIES_RAG + STRATEGIES_CLS

# Strategies whose prompt contains in-context examples.
EXAMPLE_STRATEGIES = {"few_shot", "rag"}


if __name__ == "__main__":
    schema = load_schema()
    # TEST-ONLY examples. Real runs inject these from seen.jsonl (never held_out,
    # never the row being scored). Never hardcode few-shot examples in production.
    demo_examples = [
        {"command": "grab the toy car", "expected": {"action": "fetch", "target": "toy_car"}},
        {"command": "stop now", "expected": {"action": "stop"}},
    ]
    cmd = "put the bottle in box 1"
    for strat in ALL_STRATEGIES:
        print("=" * 70)
        print(f"STRATEGY: {strat}")
        print("-" * 70)
        ex = demo_examples if strat in EXAMPLE_STRATEGIES else None
        print(build_prompt(cmd, strat, schema, examples=ex))
    print("=" * 70)
    # Proof that rag and few_shot differ ONLY in which examples were selected.
    a = build_prompt(cmd, "few_shot", schema, examples=demo_examples)
    b = build_prompt(cmd, "rag", schema, examples=demo_examples)
    print(f"few_shot prompt == rag prompt, given identical examples: {a == b}")
    # Proof that the classifier receives the command and nothing else.
    c = build_prompt(cmd, "classifier", schema)
    print(f"classifier prompt is the bare command: {c == cmd}  -> {c!r}")
