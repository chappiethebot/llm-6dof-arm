"""thesis2/demo/run_isaac_demo.py -- the existing Isaac demo, with the parser swapped.

    # live: the chosen parser answers each demo command, Isaac executes it
    python thesis2/demo/run_isaac_demo.py --parser flat_gpu
    python thesis2/demo/run_isaac_demo.py --parser v5_ollama        # today's behaviour

    # replay: execute predictions ALREADY SAVED by Stage 1, no new model call
    python thesis2/demo/run_isaac_demo.py --replay lora_greedy --split held_out_v4c
    python thesis2/demo/run_isaac_demo.py --replay flat_gpu --split held_out_v3 --ids 0,5,9

sim/, llm/ and asr/ are imported UNCHANGED and are never written to. The only
injection point is the `client` argument of llm/pipeline.py's run_once(): a
parser from thesis2/demo/parsers.py is a client shim with .name and
.complete(prompt), so the existing pipeline drives it with no edit.

ISAAC NUMBERS ARE LABELLED "demo, not measurement" EVERYWHERE. Latency measured
with a viewer attached is not the Stage 3 figure and must not be quoted as one.

REPLAY MODE
  Executes the rows of thesis2/out/final/pred_<arm>_<split>.csv. It makes NO new
  model call and refuses to run if that file is missing or incomplete. This is
  how held-out behaviour -- including the failures -- is shown in Isaac without a
  second inference pass over a test split.

  --ids default, FIXED HERE BEFORE ANY ISAAC RUN and not chosen by looking at
  which commands the arm got right: walk command_id 0,1,2,... and keep the first
  command that introduces a verb not yet covered, until all 7 verbs are present,
  then keep the next ids in order until 10 are selected.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (str(REPO), str(REPO / "thesis2" / "final"), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import common as C                                                      # noqa: E402
import parsers as PARSERS                                               # noqa: E402

DEMO_LABEL = "demo, not measurement"


def load_demo_commands():
    rows = []
    for line in (HERE / "demo_commands.txt").read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            cmd, exp = line.split("\t")
            rows.append((cmd, json.loads(exp)))
    return rows


def default_replay_ids(split, n=10):
    """FIXED RULE, written before any Isaac run: first ids covering all 7 verbs,
    then the next ids in order, up to n. Independent of any arm's correctness."""
    rows = C.load_split(split)
    seen, ids = set(), []
    for cid, r in enumerate(rows):
        v = r["expected"]["action"]
        if v not in seen:
            seen.add(v)
            ids.append(cid)
        if len(seen) == len(C.VERBS):
            break
    for cid in range(len(rows)):
        if len(ids) >= n:
            break
        if cid not in ids:
            ids.append(cid)
    return sorted(ids[:n])


def make_backend():
    """The existing Isaac backend, imported unchanged."""
    from sim.isaac_backend import IsaacBackend, launch
    launch(headless=False)
    return IsaacBackend()


def execute(be, obj):
    """Drive the existing backend from an action object. No sim/ code is edited."""
    verb = obj.get("action")
    try:
        if verb == "pick_place":
            r = be.pick_place(obj["target"], obj["location"])
        elif verb == "move_to":
            r = be.move_to(obj["location"])
        elif verb in ("fetch", "hold", "inspect"):
            r = getattr(be, verb)(obj["target"])
        elif verb in ("stop", "reject"):
            r = {"ok": True, "reason": "no motion for %s" % verb}
        else:
            r = {"ok": False, "reason": "unknown verb %r" % verb}
        return bool(r.get("ok")), json.dumps(r)[:200]
    except Exception as e:                                              # noqa: BLE001
        return False, "%s: %s" % (type(e).__name__, str(e)[:140])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--parser", choices=PARSERS.names(), help="live parser")
    g.add_argument("--replay", help="arm whose SAVED predictions to execute")
    ap.add_argument("--split", choices=list(C.SPLITS), help="required with --replay")
    ap.add_argument("--ids", default=None, help="comma list of command_ids (replay)")
    ap.add_argument("--no-isaac", action="store_true",
                    help="parse/replay only, do not launch Isaac (for checking)")
    a = ap.parse_args()
    tag = a.parser or ("replay_%s_%s" % (a.replay, a.split))
    C.start_log("run_isaac_demo_%s" % tag)
    print("  ISAAC NUMBERS IN THIS LOG ARE '%s'" % DEMO_LABEL)

    # ---------------------------------------------------------- what to run
    if a.replay:
        if not a.split:
            C.gate(False, ["--split is required with --replay"])
        p = C.pred_path(a.replay, a.split)
        if not p.exists():
            C.gate(False, ["no saved predictions at %s" % p.relative_to(C.REPO),
                           "replay makes no model call; run Stage 1 for this arm first"])
        rows = C.read_preds(a.replay, a.split)
        n_cmd = len(C.load_split(a.split))
        have = {r["command_id"] for r in rows}
        if len(have) < n_cmd:
            C.gate(False, ["%s has %d of %d commands -- INCOMPLETE"
                           % (p.name, len(have), n_cmd),
                           "replay refuses to run on a partial prediction file"])
        ids = ([int(x) for x in a.ids.split(",")] if a.ids
               else default_replay_ids(a.split))
        print("  replaying %s on %s, command_ids %s" % (a.replay, a.split, ids))
        print("  ids rule: %s" % ("given on the command line" if a.ids else
                                  "fixed default (first ids covering all 7 verbs, "
                                  "then in order) -- not chosen by correctness"))
        by = {}
        for r in rows:
            if r["repeat"] == "0":
                by[int(r["command_id"])] = r
        work = [(cid, by[cid]["command"], json.loads(by[cid]["pred_json"]),
                 json.loads(by[cid]["truth_json"]), float(by[cid]["latency_ms"]))
                for cid in ids if cid in by]
        parser = None
    else:
        parser = PARSERS.get(a.parser)
        print("  live parser: %s -- %s" % (parser.name, parser.note))
        work = [(i, cmd, None, dict(exp, modifiers=exp.get("modifiers", [])), None)
                for i, (cmd, exp) in enumerate(load_demo_commands())]
        print("  %d demo commands, all verbatim seen_v2" % len(work))

    # ---------------------------------------------------------------- Isaac
    be = None
    if not a.no_isaac:
        print("  launching the existing Isaac backend (sim/isaac_backend.py, unmodified)")
        be = make_backend()
    else:
        print("  --no-isaac: parsing/replaying only, nothing is executed")

    out = []
    for cid, cmd, saved_obj, truth, saved_ms in work:
        if saved_obj is None:
            t0 = time.perf_counter()
            obj = parser.parse(cmd)
            ms = (time.perf_counter() - t0) * 1000
            src = "live"
        else:
            obj, ms, src = saved_obj, saved_ms, "saved (no new model call)"
        correct = bool(C.score(obj, truth).strict_match)
        ok, detail = (execute(be, obj) if be else (None, "not executed (--no-isaac)"))
        out.append(dict(command_id=cid, command=cmd, parser=a.parser or a.replay,
                        source=src, pred_json=json.dumps(obj),
                        truth_json=json.dumps(truth), matches_expectation=correct,
                        parser_latency_ms=round(ms, 2) if ms is not None else "",
                        executed=ok, detail=detail, label=DEMO_LABEL))
        print("  [%3s] %-52s pred=%-42s match=%-5s exec=%-5s %.1f ms"
              % (cid, cmd[:52], json.dumps(obj)[:42], correct, ok,
                 ms if ms is not None else float("nan")))

    fn = C.OUT / ("demo_isaac_%s.csv" % tag)
    with fn.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)

    n_exec = sum(1 for r in out if r["executed"] is True)
    n_match = sum(1 for r in out if r["matches_expectation"])
    C.gate(True, [
        "%d commands; parser agreed with the expectation on %d; Isaac executed %d"
        % (len(out), n_match, n_exec),
        "EVERY number here is '%s' -- not a Stage 3 latency and not an accuracy" % DEMO_LABEL,
        ("replay: predictions came from %s, no model was called"
         % C.pred_path(a.replay, a.split).name) if a.replay else
        "live parse through llm/pipeline.py's client interface; sim/ and llm/ unmodified",
        "output: %s" % fn.relative_to(C.REPO),
    ])


if __name__ == "__main__":
    main()
