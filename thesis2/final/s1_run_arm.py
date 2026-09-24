"""thesis2/final/s1_run_arm.py -- STAGE 1: one runner, one registry.

    python thesis2/final/s1_run_arm.py --arm <name> --split {held_out_v3,held_out_v4c}

Writes thesis2/out/final/pred_<arm>_<split>.csv with one row per
(command_id, repeat): the raw output, the parsed JSON, correctness, per-field
correctness and the per-command latency.

GATES, in the order they fire
  1. rule 2 -- a NEW arm with no row in frozen_checkpoints.csv is refused, and
     its checkpoint hash is re-checked before any held-out row is read.
  2. R is READ from the V5 census (bench/results_v5_*.csv), never chosen:
     V5 used R = 1 for its local cells and R = 5 for its cloud cells, so a local
     arm runs 217 x 1 rows and a cloud arm 217 x 5. R is printed at start.
  3. a COMPLETE prediction file is never overwritten unless
     --force-rerun-after-crash is given, and every use of that flag is logged
     to thesis2/out/final/force_rerun_log.csv.
  4. determinism -- for a deterministic arm every repeat must be byte-identical
     (only testable at R > 1; at R = 1 the cross-run check below does the work).
     Sampling arms collapse to one outcome per command by majority vote, A9's
     rule, and the per-command repeat agreement is reported.
  5. reproducibility -- on held_out_v3, against the saved predictions of the
     earlier run: a deterministic local arm must agree 217/217 or GATE FAIL;
     Ollama and cloud arms report agreement only.
  6. A8/A10/A12 must be byte-identical to lora_greedy on the same split.

Resumable: a partial file is completed, and what was skipped is logged.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C                                                      # noqa: E402
import arms as A                                                        # noqa: E402

# Saved predictions from the earlier run, for the reproducibility check.
SAVED_V3 = {
    "flat_gpu": ("p45_predictions_held_out_v3.csv", "flat"),
    "flat_cpu": ("p45_predictions_held_out_v3.csv", "flat"),
    "factorized_fp16": ("factorized_fp16.csv", "pred_json"),
    "bilstm": ("a1_bilstm_held_out_v3.csv", "pred_json"),
    "tinybert": ("a1_tinybert_held_out_v3.csv", "pred_json"),
    "minilm_probe": ("a1_frozen_probe_held_out_v3.csv", "pred_json"),
    "flant5": ("a1_flant5_held_out_v3.csv", "pred_json"),
    "rules_open": ("baseline_rules.csv", "pred_json"),
    "tfidf_logreg": ("baseline_tfidf_logreg.csv", "pred_json"),
    "knn_minilm": ("baseline_knn_minilm.csv", "pred_json"),
    "ensemble": ("a1_ensemble_held_out_v3.csv", "pred_json"),
    "lora_greedy": ("a8_speculative_percommand.csv", "greedy_text"),
    "a8_speculative": ("a8_speculative_percommand.csv", "spec_text"),
    "a10_speculative": ("a10_prefix_cache_percommand.csv", "prefix_spec_text"),
    "a12_smalllm": ("a12_smalllm_drafter_percommand.csv", "spec_text"),
}
SAVED_V4C = {
    "lora_greedy": ("a13_v4c_lora_percommand.csv", "pred_json"),
    "flat_gpu": ("p45_predictions_held_out_v4c.csv", "flat"),
    "flat_cpu": ("p45_predictions_held_out_v4c.csv", "flat"),
    "factorized_fp16": ("p45_predictions_held_out_v4c.csv", "factorized"),
    "bilstm": ("a1_bilstm_held_out_v4c.csv", "pred_json"),
    "tinybert": ("a1_tinybert_held_out_v4c.csv", "pred_json"),
    "minilm_probe": ("a1_frozen_probe_held_out_v4c.csv", "pred_json"),
    "flant5": ("a1_flant5_held_out_v4c.csv", "pred_json"),
}
OLD = C.REPO / "thesis2" / "out"

# Arms whose earlier-run predictions were never persisted per command. They are
# named here with the reason, so "no saved predictions" is always a statement
# about the earlier run and never a silent skip.
NO_SAVED = {
    "factorized_int4": "Phase 3 and A9 computed INT4 per-command outcomes but "
                       "persisted only the aggregate (quantization_gate.csv)",
    "rules_firewalled": "p5_rules_firewalled.py persists only the summary row; "
                        "A9 recomputed its per-command outcomes in memory",
    "textcnn": "NEW arm -- first measurement",
    "mlpbow": "NEW arm -- first measurement",
    "grpo_base": "NEW arm -- first measurement",
    "grpo_sft": "NEW arm -- first measurement",
    "knn_minilm": "baseline_knn_minilm.csv exists for v3 only",
    "tfidf_logreg": "baseline_tfidf_logreg.csv exists for v3 only",
    "rules_open": "baseline_rules.csv exists for v3 only",
    "ensemble": "a1_ensemble_held_out_v3.csv exists for v3 only; the three V5 "
                "cells it pairs do not exist on v4c until s1_run_v5_cells.py runs",
}
for _k in ("qwen2.5-3b-lora", "Qwen2.5-3B-Instruct",
           "qwen2.5-0.5b-lora", "Qwen2.5-0.5B-Instruct"):
    NO_SAVED["salsa_full_%s" % _k] = "NEW arm -- first measurement"

# The composite arms were never saved as prediction files, but they are a
# DETERMINISTIC function of columns that were saved, so the earlier run's
# outcome can be rebuilt exactly and compared. tau is the arm's own threshold.
CASCADE_REBUILD = {
    "cascade_local_tauA": ("salsa_qwen3blora_letters_row", 0.50),
    "pipeline_always_local": ("salsa_qwen3blora_letters_row", 1.01),
    "cascade_cloud_tauA": ("cloud_row", 0.50),
    "cascade_cloud_always": ("cloud_row", 1.01),
    "cascade_oracle": ("truth_row", 1.01),
}


def rebuild_cascade(arm, split):
    """Rebuild a composite arm's earlier-run outcome from the saved p45 columns.

    Deterministic: the local factorized factors, the escalation target's answer
    and the arm's own tau are all columns of p45_predictions_<split>.csv, so this
    reproduces what the earlier run computed without re-running any model.
    """
    col, tau = CASCADE_REBUILD[arm]
    p = OLD / ("p45_predictions_%s.csv" % split)
    if not p.exists():
        return None
    ROWS, COLS = ["A", "B", "C"], ["1", "2", "3"]
    CONT = [l for l in C.LOCATIONS if l.startswith("box_")]
    RT = [l for l in C.LOCATIONS if not l.startswith(("box_", "grid_"))]
    out = {}
    for r in csv.DictReader(p.open(encoding="utf-8", newline="")):
        lt = int(r["fact_loc_type_idx"])
        row_i = int(r["fact_row_idx"])
        if lt == 0 and float(r["fact_row_conf"]) < tau:
            e = r.get(col, "")
            if e in ROWS:
                row_i = ROWS.index(e)
        if lt == 0:
            sym = "grid_%s%s" % (ROWS[row_i], COLS[int(r["fact_col_idx"])])
        elif lt == 1:
            sym = CONT[int(r["fact_container_which_idx"])]
        else:
            sym = RT[int(r["fact_runtime_which_idx"])]
        obj = C.emit(int(r["fact_verb_idx"]), int(r["fact_target_idx"]),
                     C.LOCATIONS.index(sym), C.SPACE)
        out[r["command_id"]] = json.dumps(obj, sort_keys=True)
    return out


def saved_predictions(arm, split):
    """{command_id: canonical json string} from the earlier run, or None."""
    if arm in CASCADE_REBUILD:
        return rebuild_cascade(arm, split)
    src = (SAVED_V3 if split == "held_out_v3" else SAVED_V4C).get(arm)
    if not src:
        return None
    fn, col = src
    p = OLD / fn
    if not p.exists():
        return None
    out = {}
    rows = list(csv.DictReader(p.open(encoding="utf-8", newline="")))
    for i, r in enumerate(rows):
        if str(r.get("is_warmup", "")).lower() == "true":
            continue
        cid = r.get("command_id", str(i))
        if col == "flat":
            obj = C.emit(C.VERBS.index(C.VERBS[int(r["flat_verb_idx"])]),
                         int(r["flat_target_idx"]),
                         C.LOCATIONS.index(r["flat_location"]), C.SPACE)
        elif col == "factorized":
            # the factorized head's own saved factors, reconstructed to a symbol
            ROWS, COLS = ["A", "B", "C"], ["1", "2", "3"]
            CONT = [l for l in C.LOCATIONS if l.startswith("box_")]
            RT = [l for l in C.LOCATIONS if not l.startswith(("box_", "grid_"))]
            lt = int(r["fact_loc_type_idx"])
            if lt == 0:
                sym = "grid_%s%s" % (ROWS[int(r["fact_row_idx"])],
                                     COLS[int(r["fact_col_idx"])])
            elif lt == 1:
                sym = CONT[int(r["fact_container_which_idx"])]
            else:
                sym = RT[int(r["fact_runtime_which_idx"])]
            obj = C.emit(int(r["fact_verb_idx"]), int(r["fact_target_idx"]),
                         C.LOCATIONS.index(sym), C.SPACE)
        elif col in ("greedy_text", "spec_text", "prefix_spec_text"):
            try:
                obj = json.loads(r[col])
            except Exception:
                obj = {}
        else:
            obj = json.loads(r[col])
        out[str(cid)] = json.dumps(obj, sort_keys=True)
    return out


def canon(obj):
    return json.dumps(obj if isinstance(obj, dict) else {}, sort_keys=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm")
    ap.add_argument("--split", choices=list(C.SPLITS))
    ap.add_argument("--force-rerun-after-crash", action="store_true",
                    help="overwrite a COMPLETE prediction file; every use is logged")
    ap.add_argument("--list", action="store_true", help="print the registry and exit")
    a = ap.parse_args()
    if a.list:
        A.summary()
        return
    if not a.arm or not a.split:
        ap.error("--arm and --split are required unless --list is used")
    C.start_log("s1_run_arm_%s_%s" % (a.arm, a.split))
    C.env_banner()

    arm = A.get(a.arm)
    print("  arm  : %s  (%s, %s)" % (arm.name, arm.kind, arm.determinism))
    print("  note : %s" % arm.note)
    print("  split: %s" % a.split)

    # ---- gate 1: frozen checkpoint (new arms only)
    C.require_frozen(arm.name)

    # ---- gate 2: R is read from the V5 census
    cells = C.v5_census()
    R = C.v5_repeats_for(arm.kind, cells)
    rows_in = C.load_split(a.split)
    n_cmd = len(rows_in)
    expected = n_cmd * R
    print("  R = %d  (read from V5's %s cells; not chosen)" % (R, arm.kind))
    print("  rows for this arm x split: %d commands x %d repeats = %d" % (n_cmd, R, expected))
    n_local = sum(1 for x in A.REGISTRY.values() if x.kind == "local" and not x.blocked)
    n_cloud = sum(1 for x in A.REGISTRY.values() if x.kind == "cloud" and not x.blocked)
    print("  planned grand total across the registry and both splits: "
          "%d local x %d x 1 + %d cloud x %d x 5 = %d rows"
          % (n_local, n_cmd * 2, n_cloud, n_cmd * 2,
             n_local * n_cmd * 2 * 1 + n_cloud * n_cmd * 2 * 5))

    # ---- gate 3: never silently overwrite a complete run
    complete, have = C.pred_is_complete(arm.name, a.split, expected)
    if complete and not a.force_rerun_after_crash:
        C.gate(True, [
            "%s already has %d/%d rows -- COMPLETE. Nothing re-run."
            % (C.pred_path(arm.name, a.split).name, have, expected),
            "Each arm runs ONCE per split. To re-run after a crash, pass "
            "--force-rerun-after-crash (logged).",
        ])
    if complete and a.force_rerun_after_crash:
        log = C.OUT / "force_rerun_log.csv"
        new = not log.exists()
        with log.open("a", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["utc", "arm", "split", "rows_discarded"])
            w.writerow([time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        arm.name, a.split, have])
        print("  *** --force-rerun-after-crash: discarding %d existing rows; "
              "logged to %s ***" % (have, log.name))
    if have and not complete:
        print("  RESUMING: %d/%d rows already present" % (have, expected))

    done = set()
    existing = []
    if have and not complete and not a.force_rerun_after_crash:
        existing = C.read_preds(arm.name, a.split)
        done = {(r["command_id"], r["repeat"]) for r in existing}
        print("  skipping %d already-complete (command, repeat) pairs" % len(done))

    # ---- run
    predict = arm.build()
    out_rows = list(existing)
    t0 = time.time()
    for cid, row in enumerate(rows_in):
        truth = C.truth_of(row)
        for rep in range(R):
            if (str(cid), str(rep)) in done:
                continue
            kw = {}
            if arm.name == "cascade_oracle":
                kw["truth_row"] = C.rowof(truth.get("location"))
            obj, info = predict(row["command"], **kw) if kw else predict(row["command"])
            out_rows.append(C.score_row(
                arm.name, a.split, cid, rep, row["command"],
                info.get("raw", json.dumps(obj)), obj, truth,
                info.get("latency_ms", 0.0),
                note=json.dumps({k: v for k, v in info.items()
                                 if k not in ("raw", "latency_ms")})[:400]))
        if (cid + 1) % 50 == 0:
            print("     %3d/%d commands (%.0fs)" % (cid + 1, n_cmd, time.time() - t0))
    out_rows.sort(key=lambda r: (int(r["command_id"]), int(r["repeat"])))
    p = C.write_preds(arm.name, a.split, out_rows)
    print("  wrote %s (%d rows)" % (p.name, len(out_rows)))

    notes, ok = [], True

    # ---- gate 4: determinism / repeat collapse
    by_cmd = {}
    for r in out_rows:
        by_cmd.setdefault(r["command_id"], []).append(r)
    if arm.determinism == "deterministic":
        if R == 1:
            notes.append("determinism across repeats: NOT TESTABLE at R = 1 "
                         "(V5's local repeat count). The cross-run check below "
                         "is the determinism evidence for this arm.")
        else:
            bad = [c for c, rs in by_cmd.items() if len({x["pred_json"] for x in rs}) > 1]
            if bad:
                ok = False
                notes.append("DETERMINISM FAIL on command_ids %s" % bad[:20])
            else:
                notes.append("determinism: all %d repeats byte-identical on %d commands"
                             % (R, len(by_cmd)))
    else:
        agree = sum(1 for rs in by_cmd.values()
                    if len({x["pred_json"] for x in rs}) == 1)
        coll = {c: C.majority(x["correct"] == "True" or x["correct"] is True for x in rs)
                for c, rs in by_cmd.items()}
        acc = sum(coll.values()) / len(coll)
        notes.append("sampling arm: %d repeats collapsed by majority vote (A9's rule); "
                     "per-command repeat agreement %d/%d; collapsed action match %.4f"
                     % (R, agree, len(by_cmd), acc))

    # ---- accuracy
    first = {c: rs[0] for c, rs in by_cmd.items()}
    acc = sum(1 for r in first.values() if r["correct"] in (True, "True")) / len(first)
    grid = [r for r in first.values() if r["row_correct"] not in ("", None)]
    rowacc = (sum(1 for r in grid if r["row_correct"] in (True, "True")) / len(grid)
              if grid else float("nan"))
    notes.append("action match %.4f (n=%d); row accuracy %.4f (n=%d)"
                 % (acc, len(first), rowacc, len(grid)))

    # ---- gate 5: reproducibility against the earlier run
    saved = saved_predictions(arm.name, a.split)
    if saved:
        shared = sorted(set(saved) & set(first), key=int)
        same = sum(1 for c in shared if canon(json.loads(first[c]["pred_json"])) == saved[c])
        line = ("reproducibility vs saved %s: %d/%d identical"
                % (a.split, same, len(shared)))
        is_ollama_or_cloud = arm.kind == "cloud" or "ollama" in arm.note.lower()
        if same != len(shared) and arm.determinism == "deterministic" and not is_ollama_or_cloud:
            ok = False
            diff = [c for c in shared if canon(json.loads(first[c]["pred_json"])) != saved[c]]
            line += "  *** FAIL, differing command_ids: %s ***" % diff[:20]
        elif same != len(shared):
            line += "  (reported only: Ollama/cloud arm)"
        notes.append(line)
    else:
        why = NO_SAVED.get(arm.name, "no saved per-command file for this arm x split")
        notes.append("reproducibility: NOT COMPARABLE on %s -- %s" % (a.split, why))

    # ---- gate 6: speculative arms are byte-identical to lora_greedy
    if arm.speculative:
        base = C.read_preds("lora_greedy", a.split)
        if not base:
            ok = False
            notes.append("GATE FAIL: lora_greedy has no predictions on %s; run it "
                         "first -- the speculative arms are checked against it." % a.split)
        else:
            b = {r["command_id"]: canon(json.loads(r["pred_json"]))
                 for r in base if r["repeat"] == "0"}
            diff = [c for c in first
                    if b.get(c) != canon(json.loads(first[c]["pred_json"]))]
            if diff:
                ok = False
                notes.append("GATE FAIL: %s differs from lora_greedy on %d commands: %s"
                             % (arm.name, len(diff), diff[:20]))
            else:
                notes.append("byte-identical to lora_greedy on %d/%d commands"
                             % (len(first), len(first)))

    C.gate(ok, notes + ["output: %s" % p.relative_to(C.REPO)])


if __name__ == "__main__":
    main()
