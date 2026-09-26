"""thesis2/final/s1_prompt_equality.py -- prove our prompts are V5's. NO MODEL CALLS.

    python thesis2/final/s1_prompt_equality.py [--split held_out_v3] [--n 5]

For the first N commands of the split, the few_shot and rag prompts are built
TWICE and compared byte for byte:

  ours  : common.v5_pool / common.v5_examples_for  -> llm.pipeline.run_once
  V5's  : bench/run_cell.py::run_cell()            -> llm.pipeline.run_once

Both sides call the project's real prompt builder. Neither side calls a model:
the client is a stub whose complete() RECORDS the prompt it was handed and
returns a fixed valid JSON string. That is the only thing stubbed, so every
byte of the prompt -- legend, schema text, exemplar order, retrieved neighbours
-- is produced by the code that will produce it on the real run.

V5's side is its OWN loop, not a re-statement of it: run_cell() is imported and
called. If V5 changes how it picks exemplars, this test breaks, which is the
point. Its rows go to a scratch file that is deleted afterwards; no file under
thesis2/out/ and no prediction file is touched.

rag is the reason this test exists. The bug it guards against passed the fixed
exemplar pool to rag instead of retrieving per command, and a prompt-level
check is the only thing that catches it without a model run.
"""
import argparse
import difflib
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C                                                      # noqa: E402

CANNED = '{"action": "move", "target": "gripper", "location": "grid_A1", "modifiers": []}'


class RecordingClient:
    """The stub. complete() records the prompt and returns CANNED. No network."""

    def __init__(self, name):
        self.name = name
        self.temperature = 0.0
        self.last_thinking = ""
        self.prompts = []

    def complete(self, prompt):
        self.prompts.append(prompt)
        return CANNED


def ours(split, strategy, commands):
    """Our path: the common helpers, then the real run_once."""
    from llm.pipeline import run_once
    from bench.run_cell import apply_legend
    from llm import prompts as P
    apply_legend("v5")
    schema = P.load_schema(str(C.SCHEMA_FILE))
    client = RecordingClient("stub")
    _, fewshot, retriever = C.v5_pool(split, strategy)
    for cmd in commands:
        examples, _lat, _ids, _sc = C.v5_examples_for(cmd, strategy, fewshot, retriever)
        run_once(cmd, client, strategy, schema, examples=examples)
    return client.prompts


def theirs(split, strategy, n):
    """V5's path: bench/run_cell.run_cell() itself, with limit=n."""
    from bench.run_cell import run_cell, apply_legend
    apply_legend("v5")
    client = RecordingClient("stub")
    tmp = Path(tempfile.mkdtemp(prefix="prompt_eq_")) / ("%s.csv" % strategy)
    retriever = C.v5_retriever(quiet=True) if strategy == "rag" else None
    run_cell(client, strategy, tmp, dataset=split, limit=n, repeats=1,
             quiet=True, retriever=retriever)
    try:
        tmp.unlink()
        tmp.parent.rmdir()
    except OSError:
        pass
    return client.prompts


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", default="held_out_v3", choices=list(C.SPLITS))
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--strategies", default="few_shot,rag")
    a = ap.parse_args()
    C.start_log("s1_prompt_equality_%s" % a.split)

    rows = C.load_split(a.split)[:a.n]
    commands = [r["command"] for r in rows]
    print("  split %s, first %d commands, NO model calls (stub client)"
          % (a.split, len(commands)))
    for i, cmd in enumerate(commands):
        print("     [%d] %s" % (i, cmd))

    fails = []
    lines = []
    for strategy in [s.strip() for s in a.strategies.split(",") if s.strip()]:
        print()
        print("  ---- %s" % strategy)
        mine = ours(a.split, strategy, commands)
        v5 = theirs(a.split, strategy, len(commands))
        if len(mine) != len(v5):
            fails.append("%s: %d prompts ours vs %d V5's -- the loops disagree on "
                         "how many calls a command makes" % (strategy, len(mine), len(v5)))
            continue
        same = 0
        for i, (x, y) in enumerate(zip(mine, v5)):
            xb, yb = x.encode("utf-8"), y.encode("utf-8")
            if xb == yb:
                same += 1
                print("     [%d] identical  (%d bytes)" % (i, len(xb)))
            else:
                print("     [%d] DIFFERS    ours %d bytes, V5 %d bytes"
                      % (i, len(xb), len(yb)))
                d = list(difflib.unified_diff(y.splitlines(), x.splitlines(),
                                              "V5", "ours", lineterm="", n=1))
                for ln in d[:24]:
                    print("        %s" % ln[:140])
                if len(d) > 24:
                    print("        ... %d more diff lines" % (len(d) - 24))
                fails.append("%s: prompt %d is not byte-identical to V5's" % (strategy, i))
        lines.append("%-9s %d/%d prompts byte-identical" % (strategy, same, len(mine)))
        print("     %s: %d/%d byte-identical" % (strategy, same, len(mine)))

    print()
    C.gate(not fails, ([
        "NO MODEL CALLS: the client is a stub that records the prompt",
        "no prediction file and no file under thesis2/out/ was written",
    ] + lines + fails))


if __name__ == "__main__":
    main()
