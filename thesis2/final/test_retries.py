"""thesis2/final/test_retries.py -- the gpt-4o-mini retry layer and --retry-errors.

    python thesis2/final/test_retries.py

Stubs only: no network, no model, no billed call. Every file is written to a
temporary directory under thesis2/final/ (common.OUT is pointed there), so no result file and not the
tracked force_rerun_log.csv is touched.

  1. with_retries: retries 429 / 5xx / connection errors / timeouts with 2, 4, 8,
     16, 32 s backoff, gives up after 6 attempts, never retries a 400.
  2. RetryingClient: the prompt reaching the inner client is identical on every
     attempt; name and temperature are delegated; last_retries is set.
  3. s1_run_arm.py --retry-errors (on held_out_v4c, where it is allowed): only
     the model_error rows are re-run, every other row is kept byte for byte,
     the use is logged with the row count; and a plain run on a complete file
     that still holds model_error rows is GATE FAIL.
  4. s1_run_v5_cells.py --retry-errors: same, for a cloud cell.
"""
import csv
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C                                                      # noqa: E402
import openai                                                           # noqa: E402
import httpx                                                            # noqa: E402

csv.field_size_limit(2 ** 31 - 1)
SLEEPS = []
C.time.sleep = SLEEPS.append            # record the backoff instead of waiting
REQ = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")


def status_err(cls, code):
    return cls("x", response=httpx.Response(code, request=REQ), body=None)


def failing(errors, result="OK"):
    calls = []

    def call():
        calls.append(1)
        if len(calls) <= len(errors):
            raise errors[len(calls) - 1]
        return result
    return call, calls


def test_with_retries():
    SLEEPS.clear()
    call, calls = failing([status_err(openai.RateLimitError, 429),
                           status_err(openai.InternalServerError, 503),
                           openai.APITimeoutError(request=REQ),
                           openai.APIConnectionError(request=REQ)])
    out, n = C.with_retries(call)
    assert (out, n, len(calls)) == ("OK", 4, 5), (out, n, len(calls))
    assert SLEEPS == [2.0, 4.0, 8.0, 16.0], SLEEPS

    SLEEPS.clear()
    call, calls = failing([status_err(openai.RateLimitError, 429)] * 10)
    try:
        C.with_retries(call)
        raise AssertionError("should have given up")
    except openai.RateLimitError as e:
        assert e.retries == 5 and len(calls) == 6, (e.retries, len(calls))
    assert SLEEPS == [2.0, 4.0, 8.0, 16.0, 32.0], SLEEPS

    SLEEPS.clear()
    call, calls = failing([status_err(openai.BadRequestError, 400)])
    try:
        C.with_retries(call)
        raise AssertionError("a 400 must not be retried")
    except openai.BadRequestError as e:
        assert e.retries == 0 and len(calls) == 1 and SLEEPS == []
    print("  1. with_retries: 429/5xx/timeout/connection retried, 6 attempts max, "
          "backoff 2-4-8-16-32 s, 400 not retried -- OK")


class StubOpenAI:
    """Looks like llm/clients.OpenAIClient to run_once. No network."""
    name = "openai:gpt-4o-mini"
    temperature = 0.0
    last_thinking = ""

    def __init__(self, fail_first=0, reply=None):
        self.fail_first, self.prompts = fail_first, []
        self.reply = reply or json.dumps({"action": "pick", "target": "bottle",
                                          "location": None, "modifiers": []})

    def complete(self, prompt):
        self.prompts.append(prompt)
        if len(self.prompts) <= self.fail_first:
            raise status_err(openai.RateLimitError, 429)
        return self.reply


def test_retrying_client():
    SLEEPS.clear()
    inner = StubOpenAI(fail_first=2)
    c = C.RetryingClient(inner)
    assert c.complete("PROMPT") == inner.reply
    assert inner.prompts == ["PROMPT"] * 3, "prompt must be identical on every attempt"
    assert c.last_retries == 2 and c.name == inner.name and c.temperature == 0.0
    print("  2. RetryingClient: same prompt on all 3 attempts, name/temperature "
          "delegated, last_retries=2 -- OK")


def run_main(module, argv):
    sys.argv = [module.__file__] + argv
    try:
        module.main()
    except SystemExit as e:
        return e.code or 0
    return 0


def test_run_arm(tmp):
    import arms as A
    import s1_run_arm as R
    split = "held_out_v4c"
    rows_in = C.load_split(split)
    calls = []

    def build():
        def predict(cmd):
            calls.append(cmd)
            return {"action": "pick", "target": "bottle", "location": None,
                    "modifiers": []}, dict(latency_ms=1.0, stage="ok", retries=1)
        return predict
    A.register("stub_cloud", "cloud", "sampling", build, "test stub")
    C.require_frozen = lambda name: None
    n_rep = C.v5_repeats_for("cloud", C.v5_census())

    # a COMPLETE file in which 3 rows are model_error
    rows = []
    for cid, row in enumerate(rows_in):
        for rep in range(n_rep):
            err = (cid, rep) in {(3, 0), (40, 2), (200, 4)}
            rows.append(C.score_row("stub_cloud", split, cid, rep, row["command"],
                                    "", {} if err else {"action": "place"},
                                    C.truth_of(row), 1.0,
                                    note=json.dumps({"stage": "model_error" if err
                                                     else "ok", "retries": 5 if err else 0})))
    C.write_preds("stub_cloud", split, rows)
    before = {(r["command_id"], r["repeat"]): r for r in C.read_preds("stub_cloud", split)}

    rc = run_main(R, ["--arm", "stub_cloud", "--split", split])
    assert rc != 0 and not calls, "plain run on a file with model_error rows must FAIL"

    rc = run_main(R, ["--arm", "stub_cloud", "--split", split, "--retry-errors"])
    after = {(r["command_id"], r["repeat"]): r for r in C.read_preds("stub_cloud", split)}
    assert len(calls) == 3, "exactly the 3 model_error rows re-run, got %d" % len(calls)
    assert len(after) == len(before) == len(rows_in) * n_rep
    changed = [k for k in before if before[k] != after[k]]
    assert sorted(changed) == sorted([("3", "0"), ("40", "2"), ("200", "4")]), changed
    assert not [r for r in after.values() if C.row_stage(r) == "model_error"]
    log = list(csv.DictReader((C.OUT / "force_rerun_log.csv").open(encoding="utf-8")))
    assert log[-1]["arm"] == "stub_cloud [--retry-errors]" and log[-1]["rows_discarded"] == "3"
    assert log[-1]["split"] == split and rc == 0, (log[-1], rc)
    print("  3. s1_run_arm --retry-errors on %s: plain run FAILS on model_error rows; "
          "3/%d rows re-run, the other %d byte-identical, logged -- OK"
          % (split, len(rows), len(rows) - 3))


def test_v5cells(tmp):
    import llm.clients
    import s1_run_v5_cells as V
    split = "held_out_v3"
    stub = StubOpenAI()
    llm.clients.all_clients = lambda: [stub]
    real = C.REPO / "thesis2" / "out" / "final" / ("results_v5cells_%s.csv" % split)
    rows = list(csv.DictReader(real.open(encoding="utf-8", newline="")))
    cell = ("openai:gpt-4o-mini", "zero_shot")
    hit = 0
    for r in rows:
        if (r["model"], r["strategy"]) == cell and r["command_id"] in ("5", "77"):
            r["stage"], r["raw"] = "model_error", ""
            hit += 1
    assert hit == 10, hit                      # 2 commands x R = 5
    with (C.OUT / real.name).open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=V.V5_ROW_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    before = {(r["model"], r["strategy"], r["command_id"], r["run_index"]): r
              for r in rows}

    rc = run_main(V, ["--split", split, "--cells", "openai:gpt-4o-mini:zero_shot",
                      "--retry-errors", "--no-check"])
    after_rows = list(csv.DictReader((C.OUT / real.name).open(encoding="utf-8",
                                                               newline="")))
    after = {(r["model"], r["strategy"], r["command_id"], r["run_index"]): r
             for r in after_rows}
    assert len(stub.prompts) == 10, "exactly the 10 model_error rows re-run"
    assert len(after_rows) == len(rows)
    changed = [k for k in before
               if {c: before[k].get(c, "") for c in V.V5_ROW_COLS if c != "retries"}
               != {c: after[k].get(c, "") for c in V.V5_ROW_COLS if c != "retries"}]
    assert len(changed) == 10 and all(k[2] in ("5", "77") for k in changed), changed
    assert all(after[k]["retries"] == "0" and after[k]["stage"] != "model_error"
               for k in changed)
    log = list(csv.DictReader((C.OUT / "force_rerun_log.csv").open(encoding="utf-8")))
    assert log[-1]["arm"] == "v5cell:openai:gpt-4o-mini/zero_shot [--retry-errors]"
    assert log[-1]["rows_discarded"] == "10" and rc == 0, (log[-1], rc)
    print("  4. s1_run_v5_cells --retry-errors: 10/%d rows re-run through the retry "
          "wrapper, the other %d unchanged, retries column written, logged -- OK"
          % (len(rows), len(rows) - 10))


def main():
    # inside the repo because the runners print paths relative to it; under
    # thesis2/final/, never thesis2/out/, and removed at the end
    tmp = Path(tempfile.mkdtemp(prefix="_test_retries_", dir=str(Path(__file__).resolve().parent)))
    C.OUT = tmp
    C.start_log = lambda *a, **k: None
    try:
        test_with_retries()
        test_retrying_client()
        test_run_arm(tmp)
        test_v5cells(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("ALL RETRY TESTS PASS (stubs only; no network, no model, no result file "
          "touched)")


if __name__ == "__main__":
    main()
