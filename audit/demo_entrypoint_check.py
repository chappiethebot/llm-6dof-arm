"""Check 3: invoke sim/run_demo.py's REAL entrypoint, main(), end to end.

WHY THIS EXISTS
---------------
demo_full_pass.py and demo_smoke_v5.py import `build_roster` / `strategies_for`
from run_demo and then reimplement STAGE 2 themselves. That covers the prompt
and the client roster, but it never executes main(), so every line of argument
handling, menu flow, schema selection and client setup in run_demo was untested.

That gap hid a real crash. The constrained-decoding block sat ABOVE the
load_schema() call and read `schema` before it was assigned, so `--constrain on`
-- the default -- raised UnboundLocalError for every openai: and ollama: client,
before the viewer opened. Only classifier:distilbert survived, because neither
provider branch fires for it. `python -m sim.run_demo --default` was dead and no
check noticed.

So this file drives main() itself: real argv, real menu, real MuJoCo backend,
one scripted command, one client per PROVIDER BRANCH of that block --

    openai:      -> to_flat_json_schema(schema)
    ollama:      -> to_json_schema(schema)
    classifier:  -> neither branch (the case that used to pass by accident)

plus one --constrain off run, since that is the other side of the same `if`.

Needs the `thesis` env, a reachable MuJoCo, a running Ollama, and OPENAI_API_KEY.
It calls live models, so it is slow (~1 min/case) and costs a few cents. It is a
release gate for the demo, not something to run in a loop.

    python audit/demo_entrypoint_check.py            # all cases
    python audit/demo_entrypoint_check.py openai     # substring filter

NOT a measurement. Nothing it prints is quotable.
"""
import builtins, io, os, sys, time, contextlib, traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "sim"))

COMMAND = "put the bottle in the upper left corner of the table"

# (label, argv tail). One per provider branch of the constrained-decoding block,
# plus the --constrain off side of the same condition.
CASES = [
    ("openai / constrain on",
     ["--model", "gpt-4o-mini", "--strategy", "zero_shot"]),
    ("openai / constrain off",
     ["--model", "gpt-4o-mini", "--strategy", "zero_shot", "--constrain", "off"]),
    ("ollama / constrain on",
     ["--model", "qwen2.5:3b", "--strategy", "zero_shot"]),
    ("classifier / neither branch",
     ["--model", "classifier:distilbert", "--strategy", "classifier"]),
]

# main() must print all three. "understood as" alone is not enough: a crash after
# the LLM call would still show it. "arm result" is the proof the whole chain ran.
REQUIRED = ("schema: schema_v2.json", "understood as:", "arm result:")


def scripted_input(lines):
    """Feed main()'s input() calls from a list. Anything past the end is 'quit',
    so a menu that asks one more question than expected ends the run instead of
    hanging on a closed stdin."""
    it = iter(lines)

    def _input(prompt=""):
        try:
            value = next(it)
        except StopIteration:
            value = "quit"
        print(f"{prompt}{value}")
        return value
    return _input


def run_case(label, argv_tail):
    """Call main() for real. Returns (ok, seconds, captured_stdout, error)."""
    from sim import run_demo

    real_input, real_argv = builtins.input, sys.argv
    buf = io.StringIO()
    # "1" = keyboard mode, then the command, then leave the loop.
    builtins.input = scripted_input(["1", COMMAND, "quit"])
    sys.argv = ["sim/run_demo.py"] + argv_tail
    t0 = time.perf_counter()
    try:
        with contextlib.redirect_stdout(buf):
            run_demo.main()
        err = None
    except BaseException as exc:            # SystemExit included: main() calls sys.exit
        err = "".join(traceback.format_exception_only(type(exc), exc)).strip()
    finally:
        builtins.input, sys.argv = real_input, real_argv
        secs = time.perf_counter() - t0

    out = buf.getvalue()
    if err is None:
        missing = [m for m in REQUIRED if m not in out]
        if missing:
            err = "ran without raising, but never printed: " + ", ".join(missing)
    return err is None, secs, out, err


def main():
    wanted = sys.argv[1:]
    cases = [c for c in CASES if not wanted or any(w in c[0] for w in wanted)]
    print(f"driving sim.run_demo.main() -- {len(cases)} case(s), command={COMMAND!r}\n")

    failures = []
    for label, argv_tail in cases:
        print(f"{'='*78}\n{label}\n  argv: {' '.join(argv_tail)}\n{'='*78}")
        ok, secs, out, err = run_case(label, argv_tail)
        for line in out.splitlines():
            if any(k in line for k in ("understood as:", "arm result:", "schema:",
                                       "constrained decoding:", "legend:")):
                print("  " + line.strip())
        if ok:
            print(f"  PASS  ({secs:.1f}s)\n")
        else:
            failures.append((label, err))
            print(f"  FAIL  ({secs:.1f}s)  {err}\n")
            if out:
                print("  --- last 15 lines of captured stdout ---")
                for line in out.splitlines()[-15:]:
                    print("  " + line)
                print()

    print("=" * 78)
    if failures:
        print(f"{len(failures)} of {len(cases)} FAILED:")
        for label, err in failures:
            print(f"  {label}: {err}")
        return 1
    print(f"all {len(cases)} cases drove main() to a completed arm result")
    return 0


if __name__ == "__main__":
    sys.exit(main())
