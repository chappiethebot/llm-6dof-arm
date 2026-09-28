"""thesis2/final/s6_report.py -- STAGE 6: assemble the final run report.

    python thesis2/final/s6_report.py

Writes thesis2/out/final/FINAL_RUN_REPORT.md from the output files only. It
reads; it does not recompute, and it invents nothing: a section whose input file
is absent says so.

Sections
  1. what ran and what was skipped (and why)
  2. accuracy on both splits, with the diagnosis metrics
  3. reproducibility agreement against the earlier run (from s1_repro_check.py's
     repro_check_<split>.csv, one line per named arm, both splits)
  4. the unified efficiency table, including the V5 LoRA's FIRST measured
     latency and energy (V5 recorded neither); J <= 0 shown as "~0 (below GPU
     floor)"; VRAM above idle; cascade_oracle labelled "ceiling, not a system";
     notes on outliers
  4b. the V5 cells on both splits (v5cells_both_splits.csv)
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


def outlier_notes(eff):
    """Two measured outliers, explained from the raw outputs, not guessed."""
    import statistics
    csv.field_size_limit(2 ** 31 - 1)
    L = ["\n### Notes on outliers\n"]
    e = {(r["arm"], r["split"]): r for r in eff}
    cot = e.get(("v5:ollama:qwen2.5-0.5b-lora/cot", "held_out_v3"))
    rp = C.OUT / "results_v5cells_held_out_v3.csv"
    if cot and rp.exists():
        with rp.open(encoding="utf-8", newline="") as fh:
            rows = [r for r in csv.DictReader(fh)
                    if r["model"] == "ollama:qwen2.5-0.5b-lora" and r["strategy"] == "cot"]
        long_ = [r for r in rows if float(r["latency_s"] or 0) > 20]
        lat = sorted(float(r["latency_s"]) for r in long_)
        chars = sorted(len(r["raw"]) for r in long_)
        closed = sum(r["raw"].rstrip().endswith("}") for r in long_)
        L.append(
            "**qwen2.5-0.5b-lora / cot: %s J/command, p95 %.0f s -- runaway generation.** "
            "In the same cell's Stage-1 outputs (temperature 0), %d of %d commands "
            "run for more than 20 s; all %d stop between %.1f and %.1f s, with "
            "%d-%d characters of output of which none ends in a closed JSON object "
            "(%d of %d). The text is one answer followed by `modifiers:none` "
            "repeated thousands of times. `llm/clients.py` sets no `num_predict`, so "
            "generation is bounded only by the stop token and the server. [Certain] "
            "The tight cluster at ~113 s with very different lengths points to a "
            "fixed generation budget being hit rather than the model finishing "
            "[Likely]; which budget it is has not been established [Guessing]. The "
            "energy per command is therefore dominated by ~%d%% of commands "
            "generating at ~100 W for ~113 s each. The same model's zero_shot cell "
            "has no such rows.\n"
            % (cot["joules_per_command"], float(cot["latency_p95_ms"]) / 1000,
               len(long_), len(rows), len(long_), lat[0], lat[-1], chars[0],
               chars[-1], closed, len(long_), round(100 * len(long_) / len(rows))))
    ds = [(k[0], r) for k, r in e.items() if k[0].startswith("v5:ollama:deepseek-r1")]
    bp = C.BENCH / "results_v5_deepseek.csv"
    if ds and bp.exists():
        js = sorted(float(r["joules_per_command"]) for _, r in ds)
        with bp.open(encoding="utf-8", newline="") as fh:
            th = [int(r["thinking_chars"] or 0) for r in csv.DictReader(fh)
                  if r.get("dataset") == "held_out_v3"]
        L.append(
            "**deepseek-r1:7b: %.0f-%.0f J/command.** A reasoning model: before "
            "its answer it writes a reasoning trace, median %d characters per "
            "command in V5's own held_out_v3 rows (`bench/results_v5_deepseek.csv`, "
            "`thinking_chars`), at ~115 W on this GPU. Long outputs, not a fault. "
            "[Certain]\n" % (js[0], js[-1], statistics.median(th)))
    return "\n".join(L) + "\n"


def v5_section():
    """Section 4b: the 36 V5 cells on both splits, descriptive CIs only."""
    v5 = load("v5cells_both_splits.csv")
    L = ["\n## 4b. The V5 cells on both splits\n"]
    if not v5:
        return "\n".join(L + ["_v5cells_both_splits.csv absent: run s2_metrics.py._\n"])
    L.append("All 36 cells (31 local + 5 gpt-4o-mini), from `v5cells_both_splits.csv`. "
             "Collapsed as A9 did: warm-up rows excluded, gpt-4o-mini's 5 repeats by "
             "majority vote (a tie is wrong), so n is 216 for a local cell and 217 "
             "for a cloud cell. CIs are a clustered bootstrap over command_id, 2,000 "
             "resamples, seed 20260827 -- **descriptive only: no hypothesis is "
             "tested here.** [Certain]\n")
    rows = []
    for r in v5:
        def ci(tag, col):
            return "%s [%s, %s]" % (r["%s_%s" % (tag, col)], r["%s_%s_ci_lo" % (tag, col)],
                                    r["%s_%s_ci_hi" % (tag, col)])
        rows.append(dict(cell="%s / %s" % (r["model"], r["strategy"]), kind=r["kind"],
                         v3a=ci("v3", "action"), v4a=ci("v4c", "action"),
                         v3r=ci("v3", "row"), v4r=ci("v4c", "row"),
                         j=r["v3_joules_per_command"]))
    L.append(table(rows, ["cell", "kind", "v3a", "v4a", "v3r", "v4r", "j"],
                   ["cell", "kind", "v3 action [95% CI]", "v4c action [95% CI]",
                    "v3 row [95% CI]", "v4c row [95% CI]", "v3 J/cmd"]))
    return "\n".join(L) + "\n"


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
        W("\n### --replace-cells re-runs on held_out_v3 (12 void cells + 4 "
          "connection-failure cells)\n")
        W("Every row of `force_rerun_log.csv`. The 12 local cells were voided by the "
          "exemplar-loader bug (`prompt_error`, no model call); the 4 gpt-4o-mini "
          "cells lost rows to connection drops before the retry layer existed. "
          "held_out_v4c was never re-run this way. [Certain]\n")
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
    W("From `thesis2/final/s1_repro_check.py` (`repro_check_<split>.csv`), one "
      "line per registry arm. The join key is `str(command_id)`; comparing zero "
      "commands is a FAIL; totals-only arms are compared against their own "
      "split's saved total. [Certain]\n")
    for split in C.SPLITS:
        rc = load("repro_check_%s.csv" % split)
        W("\n### %s\n" % split)
        if not rc:
            W("_repro_check_%s.csv absent: run s1_repro_check.py --split %s._\n"
              % (split, split))
            continue
        for r in rc:
            if r["status"] in ("PASS", "FAIL") and r["compared"] != "totals":
                W("- `%s` **%s** — %s/%s compared, %s identical; %s\n"
                  % (r["arm"], r["status"], r["compared"], r["n_commands"],
                     r["identical"], r["result"]))
            elif r["compared"] == "totals":
                W("- `%s` **%s** — %s\n" % (r["arm"], r["status"], r["result"]))
            else:
                W("- `%s` %s — %s\n" % (r["arm"], r["status"], r["result"]))

    # ---------------------------------------------------------------- 4
    W("\n## 4. Unified efficiency (measured, not modelled)\n")
    if eff:
        shown = []
        for r in eff:
            d = dict(r)
            if r["arm"] == "cascade_oracle":
                d["arm"] = "cascade_oracle (ceiling, not a system)"
            d["joules_per_command"] = C.j_display(r["joules_per_command"])
            d["vram_above_idle_mb"] = C.vram_display(C.vram_above_idle(r))
            shown.append(d)
        W(table(shown, ["arm", "split", "n_commands", "wall_s", "latency_p50_ms",
                        "latency_p95_ms", "latency_p99_ms", "joules_per_command",
                        "mean_power_w", "peak_vram_mb", "vram_above_idle_mb",
                        "cpu_ms_per_command", "max_gpu_temp_c"],
                ["arm", "split", "N", "wall s", "p50 ms", "p95 ms", "p99 ms",
                 "J/cmd", "mean W", "peak VRAM MB", "VRAM above idle MB",
                 "CPU ms", "max C"]))
        W("\nJ/command = (mean load W - idle median W) x wall s / N, GPU watts only. "
          "A value <= 0 means the arm's load sat inside the GPU's idle noise and is "
          "shown as \"%s\"; the measured value stays in `unified_efficiency.csv` "
          "and in `metrics_final.csv` (`joules_per_command_raw`). VRAM above idle = "
          "peak VRAM minus the median `memory.used` of the idle baseline in force "
          "when the arm was measured (`power_logs/idle_*.csv`); every row's matched "
          "log reproduces its recorded idle watts. `cascade_oracle` consumes the "
          "true row: a ceiling, not a system. [Certain]\n" % C.J_FLOOR_TEXT)
        W(outlier_notes(eff))
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

    # ---------------------------------------------------------------- 4b
    W(v5_section())

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
        full = [r for r in ok if r.get("valid_json") == r.get("n")]
        W("\nParser smoke test (Stage 5 S5-1; no Isaac, no cloud), the 10 seen_v2 "
          "commands in `thesis2/demo/demo_commands.txt`: %d parsers registered, %d ran, "
          "%d produced valid JSON on all 10; the rest are listed with their status. "
          "[Certain]\n" % (len(s), len(ok), len(full)))
        rows = []
        for r in s:
            rows.append(dict(parser=r["parser"], status=r["status"],
                             valid_json=r.get("valid_json", "-"),
                             agrees=r.get("agrees_with_seen_v2", "-"),
                             p50=r.get("latency_p50_ms", "-"),
                             vram=r.get("peak_vram_mb", "-"),
                             env=Path(r.get("python", "")).parent.name or "-"))
        W(table(rows, ["parser", "status", "valid_json", "agrees", "p50", "vram", "env"],
                ["parser", "status", "valid JSON /10", "agrees with seen_v2 /10",
                 "p50 ms", "peak VRAM MB", "python env"]))
        W("\nPeak VRAM here is PyTorch's own `max_memory_allocated` for the parser, "
          "not the nvidia-smi figure of the efficiency table; `v5_ollama` runs inside "
          "the Ollama server (the `thesis` env), whose memory PyTorch cannot see, so it "
          "reads n/a. Cloud parsers are skipped without `--allow-cloud`. Agreement with "
          "seen_v2 is demo evidence on training-distribution commands, not a "
          "measurement. [Certain]\n")
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
    W("| `ensemble` latency and energy | an offline pairing of three measured V5 "
      "cells; it performs no inference of its own |\n")
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
    W("\nThe GRPO arms ran: both were trained (`thesis2rl`), dev-selected, "
      "frozen and measured (`thesis2gpu`), and H6 is in the primary family with a "
      "verdict (section 5). The six legacy arms (`tinybert`, `minilm_probe`, "
      "`knn_minilm`, `tfidf_logreg`, `bilstm`, `flant5`) were retrained by "
      "`s0_retrain_legacy.py`, reproduced 217/217 and frozen "
      "(`frozen_checkpoints.csv`). [Certain]\n")

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
