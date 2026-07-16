"""llm/repl.py — convenience only: type a command, watch the pipeline handle it.

NOT part of the measurement. The Phase 5 harness drives the pipeline from the
dataset, never from the keyboard. This is a toy for manual poking. Do not tune
anything against what you see here.

Change the model/strategy by editing the two lines below.
"""
from llm.clients import OpenAIClient, OllamaClient
from llm.prompts import load_schema
from llm.pipeline import run_once

# --- edit these two lines to try a different cell ---
CLIENT = OpenAIClient("gpt-4o-mini")        # or OllamaClient("llama3.2:3b") / ("qwen2.5:3b") / ("qwen2.5:0.5b")
STRATEGY = "zero_shot"                       # or "few_shot" / "cot" / "role_based"
# ----------------------------------------------------


def main():
    schema = load_schema()
    print(f"Model: {CLIENT.name}   Strategy: {STRATEGY}")
    print("Type a command and press Enter. Type 'quit' or Ctrl+C to exit.\n")
    while True:
        try:
            command = input("command> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break
        if command.lower() in {"quit", "exit", ""}:
            print("bye")
            break

        r = run_once(command, CLIENT, STRATEGY, schema)
        print(f"  raw:    {r.raw!r}")
        print(f"  object: {r.obj}")
        print(f"  result: {'VALID' if r.ok else 'FAILED at ' + r.stage}"
              f"{'  reasons=' + str(r.reasons) if r.reasons else ''}"
              f"  ({r.latency_s:.2f}s)\n")


if __name__ == "__main__":
    main()