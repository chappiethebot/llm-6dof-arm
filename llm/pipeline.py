"""llm/pipeline.py — Phase 3: the machine. command -> prompt -> model -> extract -> validate.

Runs ONE command through ONE client with ONE strategy, and reports exactly where it
landed: got a reply, parsed to JSON, passed schema. Failures are recorded, not raised —
a failed cell is a data point (invalid/hallucination rate), not a crash.

This is NOT the eval harness (that's Phase 5: many commands, ground-truth scoring,
variance, seen/held-out). This just proves one command flows end to end.
"""
import time
from dataclasses import dataclass, field

from llm.prompts import build_prompt, load_schema
from llm.extract import extract_json
from schema.validate_object import validate_object


@dataclass
class PipelineResult:
    command: str
    model: str
    strategy: str
    stage: str            # where it ended: "validated" | "extract_failed" | "schema_failed"
    ok: bool              # True only if it passed schema validation
    obj: dict | None = None
    reasons: list[str] = field(default_factory=list)  # extract/validate failure labels
    raw: str = ""         # raw model text, kept for inspection
    latency_s: float = 0.0


def run_once(command, client, strategy, schema, examples=None) -> PipelineResult:
    prompt = build_prompt(command, strategy, schema, examples=examples)

    t0 = time.perf_counter()
    raw = client.complete(prompt)
    latency = time.perf_counter() - t0

    base = dict(command=command, model=client.name, strategy=strategy, raw=raw, latency_s=latency)

    ext = extract_json(raw)
    if not ext.ok:
        return PipelineResult(stage="extract_failed", ok=False, reasons=[ext.reason], **base)

    val = validate_object(ext.data, schema)
    if not val.ok:
        return PipelineResult(stage="schema_failed", ok=False, obj=ext.data, reasons=val.reasons, **base)

    return PipelineResult(stage="validated", ok=True, obj=ext.data, **base)


if __name__ == "__main__":
    from llm.clients import all_clients

    schema = load_schema()
    command = "put the bottle in box 1"   # correct answer: pick_place / bottle / box_1
    fewshot_examples = [
        {"command": "grab the toy car", "expected": {"action": "fetch", "target": "toy_car"}},
        {"command": "stop now", "expected": {"action": "stop"}},
    ]

    clients = all_clients()
    strategies = ["zero_shot", "few_shot", "cot", "role_based"]

    print(f"COMMAND: {command!r}\n")
    for client in clients:
        for strat in strategies:
            ex = fewshot_examples if strat == "few_shot" else None
            try:
                r = run_once(command, client, strat, schema, examples=ex)
                flag = "PASS" if r.ok else "----"
                print(f"[{flag}] {client.name:22} {strat:11} "
                      f"stage={r.stage:14} obj={r.obj} "
                      f"{'reasons='+str(r.reasons) if r.reasons else ''} "
                      f"({r.latency_s:.2f}s)")
            except Exception as e:
                print(f"[ERR ] {client.name:22} {strat:11} {type(e).__name__}: {e}")
    print("\nDone. PASS = reached valid schema. ---- = failed at extract or schema (that's data).")