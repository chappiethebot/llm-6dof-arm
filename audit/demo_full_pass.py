"""Check 2: walk all 8 demo clients through run_demo's STAGE 2.

Base models use zero_shot; the LoRA models and the classifier take their only
strategy. The arm is NOT driven -- mujoco.dll is blocked by Windows Application
Control on this machine -- so this covers every stage up to the motion call.
"""
import os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "sim"))

from llm.prompts import load_schema
from llm.pipeline import run_once
from sim.run_demo import build_roster, strategies_for

COMMANDS = [
    "put the bottle in the upper left corner of the table",
    "drop the rubiks cube in box 1",
    "move to the far right of the table",
]

schema = load_schema(os.path.join(ROOT, "schema", "schema_v2.json"))
roster = build_roster()


def strategy_for_demo(name):
    s = strategies_for(name)
    return "zero_shot" if len(s) > 1 else s[0]


def fmt(obj):
    """Render an action object. Must tolerate a NESTED action ({"action": {...}}),
    which is a real failure mode here -- it is exactly what llm/score.py's
    normalized policy exists to unwrap."""
    if obj is None:
        return "-"
    if not isinstance(obj, dict):
        return repr(obj)[:42]
    a = obj.get("action")
    if isinstance(a, dict):
        inner = a.get("action") or a.get("verb") or a.get("command") or "?"
        rest = [str(v) for k, v in a.items()
                if k not in ("action", "verb", "command") and v]
        return f"NESTED[{inner}" + (" / " + " / ".join(rest) if rest else "") + "]"
    parts = [str(a) if a is not None else "?"]
    parts += [str(x) for x in (obj.get("target"), obj.get("location")) if x]
    return " / ".join(parts)


for ci, cmd in enumerate(COMMANDS, 1):
    print(f"\n{'='*104}")
    print(f"COMMAND {ci}: {cmd!r}")
    print(f"{'='*104}")
    print(f"{'client':26}{'strategy':11}{'action object':44}{'stage':16}{'ms':>7}")
    print("-" * 104)
    for client, _ in roster:
        strat = strategy_for_demo(client.name)
        t0 = time.perf_counter()
        r = run_once(cmd, client, strat, schema, examples=None)
        ms = (time.perf_counter() - t0) * 1000
        print(f"{client.name:26}{strat:11}{fmt(r.obj):44}{r.stage:16}{ms:>7.0f}")
        if not r.ok:
            print(f"{'':37}reasons: {r.reasons}")
