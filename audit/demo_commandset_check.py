"""audit/demo_commandset_check.py -- run the 24-command demo set, no execution.

Replays the command set through bench.llm_client.query_with_retry, which is the
same call run_one() makes, so the validator-feedback retry is exercised exactly
as the demo exercises it. Prints one markdown table per model:

    # | command | first answer | retried | final answer | correct

    python audit/demo_commandset_check.py
    python audit/demo_commandset_check.py ollama:qwen2.5-3b-lora/zero_shot

NOTHING IS EXECUTED. send() is never called, so neither the bridge nor Isaac is
needed. No dataset is written and no results CSV is touched.

THE EXPECTATIONS ARE NOT GROUND TRUTH. Of the 24, twelve are verbatim dataset
rows, eight are the nearest cited phrasing, and four are judgement (#20, #21,
#23, #24). #24 is contested: no dataset row uses a colour as a referring
expression for an in-vocabulary object, and both measured models map "red cube"
to rubiks_cube. Read a red cell as "disagrees with my expectation", not as "the
model is wrong". bench/demo_commands.txt carries the basis for every row.

DO NOT TUNE THE RETRY PROMPT AGAINST THIS SET. It exists to measure whether the
demo generalises; a prompt fitted to it measures nothing.

SETUP MIRRORS main(): isaac backend, v5 legend, constrained decoding on. If
those drift apart the numbers here stop describing the demo.
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import bench.llm_client as C            # noqa: E402

PAIRS = [("ollama:qwen2.5-3b-lora", "zero_shot"),
         ("openai:gpt-4o-mini", "role_based")]

# Kept in step with bench/demo_commands.txt, which records each expectation's
# basis. Locations are given canonically (grid_*) and translated to the attached
# backend's names below, the same way a model's answer is.
CMDS = [
 (1,  "move the toy car to box 1",                               ("pick_place", "toy_car", "box_1")),
 (2,  "put the rubiks cube in box 1",                            ("pick_place", "rubiks_cube", "box_1")),
 (3,  "place the bottle in the upper left corner of the table",  ("pick_place", "bottle", "grid_A1")),
 (4,  "put the toy car in the bottom right corner of the table", ("pick_place", "toy_car", "grid_C3")),
 (5,  "put the bottle in the middle of the table",               ("pick_place", "bottle", "grid_B2")),
 (6,  "move to the far left corner of the table",                ("move_to", None, "grid_A1")),
 (7,  "go to the near right corner of the table",                ("move_to", None, "grid_C3")),
 (8,  "move to box 2",                                           ("move_to", None, "box_2")),
 (9,  "bring me the bottle",                                     ("fetch", "bottle", None)),
 (10, "fetch the rubiks cube",                                   ("fetch", "rubiks_cube", None)),
 (11, "pick up the toy car and hold it",                         ("hold", "toy_car", None)),
 (12, "hold the bottle",                                         ("hold", "bottle", None)),
 (13, "take a look at the rubiks cube",                          ("inspect", "rubiks_cube", None)),
 (14, "inspect the toy car",                                     ("inspect", "toy_car", None)),
 (15, "stop",                                                    ("stop", None, None)),
 (16, "halt right now",                                          ("stop", None, None)),
 (17, "put the bottle as far away as you can reach",             ("pick_place", "bottle", "max_reach")),
 (18, "put the rubiks cube anywhere there is space",             ("pick_place", "rubiks_cube", "nearest_free_cell")),
 (19, "make me a sandwich",                                      ("reject", None, None)),
 (20, "paint the bottle green",                                  ("reject", None, None)),
 (21, "throw the toy car across the room",                       ("reject", None, None)),
 (22, "put the toy car in the near side, middle",                ("pick_place", "toy_car", "grid_C2")),
 (23, "move to the middle of the left column",                   ("move_to", None, "grid_B1")),
 (24, "put the red cube in box 1",                               ("reject", None, None)),
]


def expected_backend(spec):
    """The expectation in the attached backend's vocabulary."""
    verb, target, loc = spec
    out = {"verb": verb}
    if target:
        out["target"] = target
    if loc:
        out["location"] = C.GRID_TO_ISAAC.get(loc, loc)
    return out


def brief(a):
    if a is None:
        return "-"
    if isinstance(a, str):
        return a.replace("\n", " ")[:90]
    return "/".join([a.get("verb", "?")]
                    + [a[k] for k in ("target", "location") if a.get(k)])


def run(model, strategy, retry=True):
    C.RETRY_ENABLED = retry
    rows = []
    for n, cmd, spec in CMDS:
        want = expected_backend(spec)
        first = final = None
        retried = False
        try:
            final, _raw, info = C.query_with_retry(cmd, model, strategy,
                                                   allow_retry=retry)
            if info:
                retried = True
                first = info.get("first") or info.get("first_error")
            else:
                first = final
        except Exception as exc:
            # A failure is a result, not a crash: the table must show it.
            final = "FAILED: %s" % str(exc)[:70]
            if first is None:
                first = final
        rows.append(dict(n=n, cmd=cmd, want=want, first=first,
                         retried=retried, final=final,
                         ok=isinstance(final, dict) and final == want))
    return rows


def table(title, rows):
    out = ["", "### %s" % title, "",
           "| # | command | first answer | retried | final answer | correct |",
           "|---|---|---|---|---|---|"]
    for r in rows:
        out.append("| %d | %s | `%s` | %s | `%s` | %s |"
                   % (r["n"], r["cmd"], brief(r["first"]),
                      "yes" if r["retried"] else "no",
                      brief(r["final"]), "yes" if r["ok"] else "**no**"))
    before = sum(1 for r in rows
                 if isinstance(r["first"], dict) and r["first"] == r["want"])
    out += ["",
            "correct before retry %d/%d | correct after %d/%d | retries fired %d"
            % (before, len(rows), sum(1 for r in rows if r["ok"]), len(rows),
               sum(1 for r in rows if r["retried"]))]
    return "\n".join(out)


def main():
    C._set_backend("isaac")
    from bench.run_cell import apply_legend
    apply_legend("v5")
    C.apply_constraints("on")

    targets = (PAIRS if len(sys.argv) < 2
               else [tuple(a.split("/", 1)) for a in sys.argv[1:]])
    blocks = []
    for model, strategy in targets:
        t0 = time.perf_counter()
        blocks.append(table("%s / %s" % (model, strategy),
                            run(model, strategy)))
        print("  (%s done in %.0fs)" % (model, time.perf_counter() - t0),
              file=sys.stderr)
    print("\n".join(blocks))


if __name__ == "__main__":
    main()
