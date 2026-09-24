"""thesis2/demo/smoke_parsers.py -- every registered parser on the 10 demo
commands, WITHOUT Isaac.

    python thesis2/demo/smoke_parsers.py [--parsers a,b,c] [--allow-cloud N]

For each parser: valid JSON, agreement with the seen_v2 expectation, latency,
and peak VRAM for GPU parsers. Isaac is never imported.

SAFETY
  * the demo commands are re-checked against data/seen_v2.jsonl and against both
    held-out splits at startup: a command that is not verbatim seen_v2, or that
    appears in a held-out split, aborts the run;
  * cloud parsers are SKIPPED unless --allow-cloud N is given, and then at most
    N cloud calls are made in total (default 0);
  * parsers whose model is not trained yet (grpo_*) are skipped with the reason,
    not failed.

Exit code is non-zero if any parser that DID run produced invalid JSON.
"""
import argparse
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


def load_commands():
    rows = []
    for line in (HERE / "demo_commands.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        cmd, exp = line.split("\t")
        rows.append((cmd, json.loads(exp)))
    return rows


def verify_provenance(rows):
    seen = {}
    for l in (REPO / "data" / "seen_v2.jsonl").read_text(encoding="utf-8").splitlines():
        if l.strip():
            r = json.loads(l)
            seen[r["command"]] = r["expected"]
    held = set()
    for f in C.SPLITS:
        for l in (REPO / "data" / ("%s.jsonl" % f)).read_text(encoding="utf-8").splitlines():
            if l.strip():
                held.add(json.loads(l)["command"])
    bad = []
    for cmd, exp in rows:
        if cmd not in seen:
            bad.append("NOT VERBATIM seen_v2: %r" % cmd)
        if cmd in held:
            bad.append("HELD-OUT COMMAND IN THE DEMO SET: %r" % cmd)
    if bad:
        C.gate(False, bad)
    print("  provenance OK: %d commands, all verbatim seen_v2, none held-out" % len(rows))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--parsers", default=None, help="comma list; default = all registered")
    ap.add_argument("--allow-cloud", type=int, default=0,
                    help="maximum number of cloud calls IN TOTAL (default 0 = skip cloud)")
    a = ap.parse_args()
    C.start_log("smoke_parsers")

    rows = load_commands()
    verify_provenance(rows)
    names = a.parsers.split(",") if a.parsers else PARSERS.names()

    import torch
    results, bad_json = [], []
    cloud_budget = a.allow_cloud
    for name in names:
        if name in PARSERS.BLOCKED:
            print("  %-34s SKIPPED -- %s" % (name, PARSERS.PARSERS[name]().note))
            results.append(dict(parser=name, status="skipped: not trained"))
            continue
        if name in PARSERS.CLOUD and cloud_budget < len(rows):
            print("  %-34s SKIPPED -- cloud parser; --allow-cloud %d is below the "
                  "%d calls it needs" % (name, cloud_budget, len(rows)))
            results.append(dict(parser=name, status="skipped: cloud budget"))
            continue
        try:
            p = PARSERS.get(name)
        except SystemExit as e:
            print("  %-34s SKIPPED -- %s" % (name, str(e)[:90]))
            results.append(dict(parser=name, status="skipped: %s" % str(e)[:60]))
            continue
        if name in PARSERS.GPU_PARSERS and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        ok_json, agree, lat = 0, 0, []
        err = ""
        try:
            for cmd, exp in rows:
                t0 = time.perf_counter()
                obj = p.parse(cmd)
                lat.append((time.perf_counter() - t0) * 1000)
                if isinstance(obj, dict) and obj.get("action") in C.VERBS:
                    ok_json += 1
                agree += int(bool(C.score(obj, dict(exp, modifiers=exp.get("modifiers", []))).strict_match))
                if name in PARSERS.CLOUD:
                    cloud_budget -= 1
        except SystemExit as e:
            err = str(e)[:120]
        except Exception as e:                                          # noqa: BLE001
            err = "%s: %s" % (type(e).__name__, str(e)[:100])
        vram = (torch.cuda.max_memory_allocated() / 1e6
                if name in PARSERS.GPU_PARSERS and torch.cuda.is_available() else "")
        if err:
            print("  %-34s ERROR -- %s" % (name, err))
            results.append(dict(parser=name, status="error", note=err))
            continue
        if ok_json != len(rows):
            bad_json.append(name)
        lat.sort()
        print("  %-34s json %2d/%d  agrees %2d/%d  p50 %8.2f ms  peak VRAM %s"
              % (name, ok_json, len(rows), agree, len(rows), lat[len(lat) // 2],
                 ("%.0f MB" % vram) if vram else "n/a (CPU/cloud)"))
        results.append(dict(parser=name, status="ok", valid_json=ok_json,
                            agrees_with_seen_v2=agree, n=len(rows),
                            latency_p50_ms=round(lat[len(lat) // 2], 2),
                            peak_vram_mb=round(vram, 1) if vram else "n/a"))
        del p
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out = C.OUT / "demo_smoke_parsers.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    errored = [r["parser"] for r in results if r.get("status") == "error"]
    C.gate(not bad_json and not errored, [
        "%d parsers attempted; %d produced valid JSON on all %d commands"
        % (len(results), sum(1 for r in results if r.get("status") == "ok"), len(rows)),
        "invalid JSON from: %s" % (bad_json or "none"),
        "ERRORED (gate failure): %s" % (errored or "none"),
        "skipped (cloud budget / not trained): %s"
        % ([r["parser"] for r in results if str(r.get("status", "")).startswith("skipped")] or "none"),
        "cloud calls made: %d (budget %d)" % (a.allow_cloud - cloud_budget, a.allow_cloud),
        "agreement with seen_v2 is DEMO evidence, not a measurement",
        "output: %s" % out.relative_to(C.REPO),
    ])


if __name__ == "__main__":
    main()
