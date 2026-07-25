"""llm/pipeline.py — Phase 3: the machine. command -> prompt -> model -> extract -> validate.
Runs ONE command through ONE client with ONE strategy, and reports exactly where it
landed. Failures are RECORDED, not raised — run_once never throws, so no weird model
reply or API hiccup can kill a measurement run. A failed cell is a data point.
This is NOT the eval harness (that's Phase 5).
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
    stage: str            # validated | extract_failed | schema_failed | model_error | prompt_error | pipeline_error
    ok: bool
    obj: dict | None = None
    reasons: list[str] = field(default_factory=list)
    raw: str = ""
    thinking: str = ""    # V2: reasoning trace, "" for non-reasoning models
    latency_s: float = 0.0


def run_once(command, client, strategy, schema, examples=None) -> PipelineResult:
    base = dict(command=command, model=client.name, strategy=strategy)

    # Build the prompt (local, cheap — but guard anyway).
    try:
        prompt = build_prompt(command, strategy, schema, examples=examples)
    except Exception as e:
        return PipelineResult(stage="prompt_error", ok=False,
                              reasons=[f"{type(e).__name__}:{e}"], **base)

    # Call the model. Network/API errors (timeout, rate limit) live here and must
    # never kill the run — they become a recorded model_error.
    t0 = time.perf_counter()
    try:
        raw = client.complete(prompt)
    except Exception as e:
        return PipelineResult(stage="model_error", ok=False,
                              reasons=[f"{type(e).__name__}:{e}"],
                              latency_s=time.perf_counter() - t0, **base)
    latency = time.perf_counter() - t0

    # V2: complete() parks the reasoning trace on the client. Read it immediately
    # after the call, before anything else can overwrite it.
    base = dict(**base, raw=raw, latency_s=latency,
                thinking=getattr(client, "last_thinking", ""))

    # Extract + validate. Guarded so no unexpected reply shape can crash a run.
    try:
        ext = extract_json(raw)
        if not ext.ok:
            return PipelineResult(stage="extract_failed", ok=False, reasons=[ext.reason], **base)
        val = validate_object(ext.data, schema)
        if not val.ok:
            return PipelineResult(stage="schema_failed", ok=False, obj=ext.data, reasons=val.reasons, **base)
        return PipelineResult(stage="validated", ok=True, obj=ext.data, **base)
    except Exception as e:
        return PipelineResult(stage="pipeline_error", ok=False,
                              reasons=[f"{type(e).__name__}:{e}"], **base)


if __name__ == "__main__":
    from llm.clients import all_clients

    schema = load_schema()
    command = "put the bottle in box 1"
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
            r = run_once(command, client, strat, schema, examples=ex)
            flag = "PASS" if r.ok else "----"
            think = f" think={len(r.thinking)}ch" if r.thinking else ""
            print(f"[{flag}] {client.name:22} {strat:11} "
                  f"stage={r.stage:14} obj={r.obj} "
                  f"{'reasons='+str(r.reasons) if r.reasons else ''} "
                  f"({r.latency_s:.2f}s){think}")
    print("\nDone. PASS = reached valid schema. ---- = failed (extract/schema/model — that's data).")
