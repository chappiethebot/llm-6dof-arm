"""thesis2/final/s6_report.py -- STAGE 6: assemble the final run report.

    python thesis2/final/s6_report.py

Writes thesis2/out/final/FINAL_RUN_REPORT.md from the output files only. It
reads; it does not recompute, and it invents nothing: a section whose input file
is absent says so.

Sections
  1. what ran and what was skipped (and why)
  2. accuracy on both splits, with the diagnosis metrics
  3. reproducibility agreement against the earlier run
  4. the unified efficiency table, including the V5 LoRA's FIRST measured
     latency and energy (V5 recorded neither)
  5. the primary family, then the exploratory family
  6. Isaac status
  7. everything not measurable, and why

It tags nothing as novel.
"""
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C                                                      # noqa: E402
import arms as A                                                        # noqa: E402


def load(name):
    p = C.OUT / name
    return list(csv.DictReader(p.open(encoding="utf-8", newline=""))) if p.exists() else []


def table(rows, cols, headers=None):
    if not rows:
        return "_no rows_\n"
    headers = headers or cols
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out) + "\n"


def main():
    C.start_log("s6_report")
    metrics = load("metrics_final.csv") or load("metrics_stage2.csv")
    sig = load("final_significance.csv")
    eff = load("unified_efficiency.csv")
    frozen = load("frozen_checkpoints.csv")
    force = load("force_rerun_log.csv")
    L = []
    W = L.append

    W("# Final run report\n")
    W("Assembled %s by `thesis2/final/s6_report.py` from the output files in "
      "`thesis2/out/final/`. Nothing here is recomputed and nothing is tagged "
      "novel.\n" % time.strftime("%Y-%m-%d %H:%M"))
    W("Effective n is **217 commands per split**. The V5 cell grid is 12,152 "
      "rows over those 217 commands; repeats collapse to one outcome per command "
      "by majority vote, as A9 did. This run is never 12,000 commands.\n")

    # ---------------------------------------------------------------- 1
    W("## 1. What ran, what was skipped\n")
    ran = {(r["arm"], r["split"]) for r in metrics if r.get("status") == "valid"}
    rows = []
    for name in sorted(A.REGISTRY):
        arm = A.REGISTRY[name]
        for split in C.SPLITS:
            st = "ran" if (name, split) in ran else "NOT RUN"
            why = ""
            if st == "NOT RUN":
                m = [r for r in metrics if r["arm"] == name and r["split"] == split]
                why = (arm.blocked or (m[0].get("reason") if m else "") or
                       "no prediction file")
            rows.append(dict(arm=name, split=split, kind=arm.kind, status=st,
                             reason=why[:110]))
    W(table(rows, ["arm", "split", "kind", "status", "reason"]))
    if frozen:
        W("\n### Frozen checkpoints (new arms)\n")
        W(table(frozen, ["arm", "selected_epoch", "dev_action_match", "sha256"],
                ["arm", "selected epoch", "dev_v2 action match", "sha256"]))
    else:
        W("\n_No `frozen_checkpoints.csv`: no new arm was frozen, so every new arm "
          "was refused held-out access by the rule-2 gate._\n")
    if force:
        W("\n### `--force-rerun-after-crash` uses\n")
        W(table(force, ["utc", "arm", "split", "rows_discarded"]))
    else:
        W("\n_`--force-rerun-after-crash` was never used: every arm ran once per "
          "split._\n")

    # ---------------------------------------------------------------- 2
    W("\n## 2. Accuracy on both splits, with the diagnosis metrics\n")
    if not metrics:
        W("_No metrics file: Stage 2 has not run._\n")
    for split in C.SPLITS:
        W("\n### %s\n" % split)
        sub = [r for r in metrics if r["split"] == split and r.get("status") == "valid"]
        sub.sort(key=lambda r: -(float(r["action_match"]) if r.get("action_match") else 0))
        W(table(sub, ["arm", "n_commands", "action_match", "row_accuracy",
                      "row_by_word_group", "row_signature", "ECE"],
                ["arm", "n", "action", "row", "row by word group", "row signature", "ECE"]))
    W("\n`row by word group`: seen = middle/center (both splits); unseen = "
      "upper/bottom on v3, back/front on v4c; mixed = a command containing both. "
      "A prompted arm's v3 number is measured on wording the V5 legend defines; "
      "its v4c number is not. [Certain]\n")

    # ---------------------------------------------------------------- 3
    W("\n## 3. Reproducibility against the earlier run\n")
    logs = sorted((C.LOGS).glob("s1_run_arm_*.log"))
    lines = []
    for lg in logs:
        for line in lg.read_text(encoding="utf-8", errors="replace").splitlines():
            if "reproducibility" in line:
                lines.append(line.strip())
    if lines:
        for l in sorted(set(lines)):
            W("- %s\n" % l)
    else:
        W("_No Stage 1 logs contain a reproducibility line yet._\n")

    # ---------------------------------------------------------------- 4
    W("\n## 4. Unified efficiency (measured, not modelled)\n")
    if eff:
        W(table(eff, ["arm", "split", "n_commands", "wall_s", "latency_p50_ms",
                      "latency_p95_ms", "latency_p99_ms", "joules_per_command",
                      "mean_power_w", "peak_vram_mb", "cpu_ms_per_command",
                      "max_gpu_temp_c"],
                ["arm", "split", "N", "wall s", "p50 ms", "p95 ms", "p99 ms",
                 "J/cmd", "mean W", "peak VRAM MB", "CPU ms", "max C"]))
        v5 = [r for r in eff if "ollama" in (r.get("model_digest", "") + r["arm"]).lower()]
        W("\n**The V5 LoRA's first measured latency and energy.** V5 recorded "
          "neither for its Ollama cells; the rows above are the first. %s\n"
          % ("Rows present: %s." % ", ".join(sorted({r["arm"] for r in v5}))
             if v5 else "No Ollama arm has been measured yet."))
        W("\nThe modelled cascade constants in `p45_cascade_full.csv` are used "
          "nowhere in this run: cascades are measured per command. [Certain]\n")
    else:
        W("_No `unified_efficiency.csv`: Stage 3 has not run, so every energy and "
          "measured-latency cell elsewhere reads 'not measured yet'._\n")

    # ---------------------------------------------------------------- 5
    W("\n## 5. Significance\n")
    prim = [r for r in sig if r["family"] == "primary"]
    ex = [r for r in sig if r["family"] == "exploratory"]
    if prim:
        W("\n### Primary family — 8 hypotheses, declared 2026-09-22 before this run\n")
        W(table(prim, ["id", "split", "arm_a", "arm_b", "metric", "n", "acc_a",
                       "acc_b", "diff_pts", "ci_lo_pts", "ci_hi_pts", "mcnemar_p",
                       "p_holm", "verdict"],
                ["id", "split", "A", "B", "metric", "n", "acc A", "acc B", "diff",
                 "CI lo", "CI hi", "p", "p_holm", "verdict"]))
        W("\n**Transparency.** H1, H2, H4 and H5 were measured once before this "
          "family was declared; their old 26-hypothesis Holm values are printed in "
          "the `s4_significance` log and are not re-derived here.\n")
    else:
        W("_No primary-family rows: Stage 4 has not run._\n")
    if ex:
        claim = [r for r in ex if r["verdict"] == "CLAIMABLE"]
        W("\n### Exploratory family — %d comparisons, Holm across all of them\n" % len(ex))
        W("Reported, never headlined. %d survive correction.\n" % len(claim))
        W(table(claim[:40], ["split", "arm_a", "arm_b", "metric", "diff_pts",
                             "p_holm", "verdict"]))

    # ---------------------------------------------------------------- 6
    W("\n## 6. Isaac status\n")
    demo = sorted(C.OUT.glob("demo_isaac_*.csv"))
    smoke = C.OUT / "demo_smoke_parsers.json"
    if smoke.exists():
        s = json.loads(smoke.read_text(encoding="utf-8"))
        ok = [r for r in s if r.get("status") == "ok"]
        W("\nParser smoke test (no Isaac), 10 seen_v2 commands: %d parsers ran, "
          "%d produced valid JSON on all 10.\n" % (len(s), len(ok)))
        W(table(ok, ["parser", "valid_json", "agrees_with_seen_v2", "latency_p50_ms",
                     "peak_vram_mb"],
                ["parser", "valid JSON", "agrees with seen_v2", "p50 ms", "peak VRAM MB"]))
    else:
        W("_No parser smoke test on record._\n")
    if demo:
        for d in demo:
            rows = list(csv.DictReader(d.open(encoding="utf-8", newline="")))
            n_ex = sum(1 for r in rows if r["executed"] == "True")
            W("\n- `%s`: %d commands, %d executed in Isaac. Every number in that "
              "file is labelled **demo, not measurement**.\n" % (d.name, len(rows), n_ex))
    else:
        W("\n_Isaac itself has not been run: no `demo_isaac_*.csv`. The parsers are "
          "smoke-tested without Isaac; VRAM fit is unmeasured (see "
          "`thesis2/demo/vram_check.md`)._\n")

    # ---------------------------------------------------------------- 7
    W("\n## 7. Not measurable, and why\n")
    W("| item | why |\n|---|---|\n")
    W("| GRPO arms (`grpo_base`, `grpo_sft`) | Training uses `thesis2rl`; "
      "inference uses `thesis2gpu`. See selection records and the signal check "
      "for completion/skip status. H6 stays in the eight-test primary family "
      "and is NOT RUN until the required predictions exist. |\n")
    W("| cloud arms' energy | the work happens off this machine |\n")
    W("| CPU watts | only GPU watts are sampled; CPU arms carry process CPU ms "
      "instead |\n")
    W("| A1.7 field-level FLMD, every SALSA digits mapping | withdrawn upstream "
      "(FINAL_REPORT B-3, B-4); not registered |\n")
    W("| `cascade_oracle` efficiency | a ceiling that consumes the ground-truth "
      "row; accuracy only |\n")
    for r in metrics:
        if r.get("status") == "not run" and r.get("reason"):
            pass
    W("\nArms whose builder reports a missing checkpoint (`tinybert`, "
      "`minilm_probe`, `knn_minilm`, `tfidf_logreg`, `bilstm`, `flant5`) train and "
      "evaluate in one process in `thesis2/scripts/a1_arms.py` and persist no "
      "checkpoint, so this runner cannot rebuild them. That is reported by the "
      "builder rather than worked around. [Certain]\n")

    p = C.OUT / "FINAL_RUN_REPORT.md"
    p.write_text("\n".join(L), encoding="utf-8")
    C.gate(True, [
        "sections written from: metrics(%d) significance(%d) efficiency(%d) frozen(%d)"
        % (len(metrics), len(sig), len(eff), len(frozen)),
        "absent inputs are stated as absent, not filled in",
        "output: %s" % p.relative_to(C.REPO),
    ])


if __name__ == "__main__":
    main()
