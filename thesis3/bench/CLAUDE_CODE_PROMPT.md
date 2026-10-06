# Prompt for Claude Code: build the thesis3 unified bench

Paste everything below the line into Claude Code, started from `C:\Users\USER\llm-6dof-arm`
after RUNBOOK.md steps 1 to 4 (the package is in `thesis3\bench\`).

---

Build `thesis3/`: one unified test bench that re-measures **every** system from the thesis
on **the same two test sets** with **one constant measurement setup**, and writes exactly the
files described in `thesis3/bench/SCHEMA.md`. Write the code only; I run the long runs myself
in PowerShell. Do not modify anything under `thesis2/`, `llm/`, `bench/` or `ft/`; import from
them. Do not modify the shipped files in `thesis3/bench/` (telemetry.py, analyze.py,
verify_campaign.py, make_synthetic.py, corrupt_test.py, systems_expected.csv, SCHEMA.md,
RUNBOOK.md); if one is wrong, stop and tell me. Read SCHEMA.md and RUNBOOK.md first.

## 0. Ground rules
- Reuse, do not re-implement: every system comes from `thesis2/final/arms.py` (and the V5
  cells from `thesis2/final/s1_run_v5_cells.py` / `llm/` clients). Scoring uses `llm/score.py`
  unchanged. Frozen weights load from `thesis2/models/final/`; check each sha256 against
  `thesis2/out/final/frozen_checkpoints.csv` and refuse to run on a mismatch.
- Nothing is tuned. No prompt, threshold, checkpoint or generation setting changes. Ollama
  exactly as V5: `OLLAMA_MODELS=D:\LLM`, `num_ctx` 4096, `temperature` as recorded, no
  `num_predict`, `OLLAMA_MAX_LOADED_MODELS=1`. Record the Ollama version and every model digest.
- **Test data, identical for every system:** `data/held_out_v3.jsonl` and
  `data/held_out_v4c.jsonl`. Assert 217 lines each, compute both sha256, write them to
  `machine.json` (`datasets`) and onto every runs.csv row. If any `data/held_out_v5*.jsonl`
  exists, stop and ask me: "V5" in this thesis is the legend version, not a test set.
- **Legend:** every system that builds a prompt uses the V5 legend on both test sets.
  `systems_expected.csv` says what each system must record (`v5`, `none`, or `recorded` =
  whatever the thesis2 code uses; show me which value that is in `--plan`).
- **Registry must match `systems_expected.csv` exactly** (66 systems; family, kind, env,
  uses_cloud, repeats). If the thesis2 code disagrees with a row, stop and tell me.
- **Environments. Never install anything into `thesis`, `thesis2gpu` or `thesis2rl`.**
  - `D:\condaenvs\thesis3meter` (RUNBOOK step 2): the parent process. It runs
    `telemetry.Telemetry` (10 Hz; do not write another sampler), the protocol, analyze.py and
    verify_campaign.py. It loads no model and imports nothing from thesis2.
  - Each system runs in a **worker subprocess** started with the `python.exe` of its env
    column: `thesis2gpu` = `D:\condaenvs\thesis2gpu\python.exe` (CUDA torch, transformers,
    no `ollama` package); `thesis` = the Ollama client env (CPU torch; find its python with
    `conda env list`). Use the env the final run used for each system (RUN_ORDER.md,
    METHODS_NOTES.md, the `thesis2/out/final/logs`); if a package is missing, stop and tell me.
  - ensemble follows the thesis2 definition (`thesis2/final/arms.py`, METHODS_NOTES item 8):
    its classifier is the V5 cell `classifier:distilbert` (CPU, single-threaded), so the whole
    ensemble runs in one `thesis` worker with no helper process.
  - Register every worker with `tel.track_pid(pid)` (Ollama's own processes are tracked by
    name) and untrack it when it ends.
- **Worker protocol:** the worker loads the system, runs warm-up and load passes and writes
  the run's files itself. It reports every phase change to the parent over stdout as one
  JSON line carrying its own `time.perf_counter_ns()`; the parent calls
  `set_phase(phase, run_id)` at once. `perf_counter_ns` is system-wide on Windows, so worker
  and sampler timestamps line up; `t_load_start/end` in runs.csv come from the worker's stamps.
  The worker stamps every call (`t_start`/`t_end` around the whole call,
  `t_infer_start`/`t_infer_end` around the model call only).
- `set_phase` raises if the sampler has died: let that stop the campaign. After each run, if
  `tel.n_errors` grew, add it to `notes`.
- **Per-run files and write order** (SCHEMA.md): outputs jsonl, calls csv.gz (pass 0 and
  passes 1..20 only), latency npy (every load call), passes csv, then the runs.csv row (the
  commit marker; the parent appends it after the worker exits cleanly). A run without its
  runs.csv row is incomplete: `--redo <run_id>` deletes its files and runs it again. Never write
  per-call rows beyond pass 20 (rules systems make ~16 million calls in 300 s); keep latencies
  in a preallocated numpy array.
- Fill every column in SCHEMA.md: `loc_true`/`loc_pred`, `pred_hash`, `confidence`,
  `tokens_thinking` for deepseek-r1, `error` (a failed call is scored wrong, never skipped),
  `extra` keys. Use `csv.field_size_limit(2**31 - 1)` wherever raw text is read back.
- PowerShell conventions in anything you print: `;` not `&&`, no multi-line `python -c`,
  `Set-Content -Encoding ascii`.

## 1. Registry: 66 systems x 2 test sets = 132 runs (`thesis3/bench/registry.py`)
Each system: name, family, kind, device, runtime, env, legend, uses_cloud, repeats,
`load()`, `predict(command)` returning raw output (+ confidence and token counts where
available) built from the existing code, `unload()`.
- The 30 final-run systems, names exactly as in `thesis2/out/final/metrics_final.csv`, and
  the 36 V5 cells `v5:<runtime>:<model>/<strategy>`; all 66 are in `systems_expected.csv`.
- **ensemble runs online now** (it was an offline pairing before): per command, run
  `v5:classifier:distilbert/classifier` + `v5:ollama:qwen2.5-3b-lora/zero_shot`; if they agree on (action, target,
  location) emit the classifier's answer, otherwise call `gpt-4o-mini/rag` 5 times and take
  the majority (METHODS_NOTES item 8). Log `escalated` and `cloud_calls` in `extra`.
- cascade_oracle (ceiling) and rules_open (vocabulary-unfair) are run and reported; the
  analysis never ranks them.
- gpt-4o-mini: 5 repeats per command, retries as in `thesis2/final/common.py`
  (`with_retries`), retry count in `extra`.
- Speculative systems: `accepted_tokens`, `draft_tokens`, `verifier_passes`,
  `first_rejection_field` in `extra` (computed after the call's timing ends). If a module can
  only be loaded by running its pipeline on a few commands, do that inside `modelload` and
  write in `notes` that model_load_s includes those calls.
- Confidence stays empty for transformers LLMs (do not add `output_scores`: it would change
  their timing). Token counts are taken outside the timed window.

## 2. Constant protocol for every system (`thesis3/bench/run_bench.py`)
Unload everything (Ollama `keep_alive=0`, free CUDA, end the previous worker), then:
1. **Thermal gate:** wait until GPU temperature <= 50 C (poll 1 s, max 10 min; log the wait).
2. **Idle baseline:** whenever the last idle window ended more than 60 min ago (or none
   exists in this session): phase `idle`, 300 s, nothing loaded, id `idle_<session>_<NNNN>`;
   append to idle.csv with mono and UTC start/end. The system's runs use the newest idle id.
3. **Model load** (local): `set_phase("modelload", <v3 run_id>)`, start the worker, load, time
   it (`model_load_s` on the v3 row). Ollama: the first `keep_alive` request, empty prompt.
4. **Loaded idle** (local): `set_phase("loadedidle", <v3 run_id>)`, 60 s, model resident, no calls.
5. For held_out_v3, then held_out_v4c, each its own run:
   a. **Warm-up** (`warmup`): 3 calls on commands 0, 1, 2, pass 0.
   b. **Load** (`load`): shuffled passes over all 217 commands (seed = 20260827 + pass + run
      index) until BOTH at least 1 full pass AND at least 300 s; finish the pass in progress.
      Cloud: 5 repeats per command in every pass.
   c. **Cool-down** (`cooldown`): 60 s.
   d. Write the run's files in the order above, flush, then its runs.csv row.
6. Unload, end the worker.

Blocks, fixed order, written to `thesis3/bench/run_order.csv` (system, split, block):
A = CPU systems; B = transformers GPU systems; C1 = Ollama cells except deepseek-r1;
C2 = the 5 deepseek-r1:7b cells (about 7.5 h on their own); D = cloud cells and systems that
call gpt-4o-mini. Resume skips a (system, test set) that has a runs.csv row; each launch
starts a new `telemetry_<session>.csv`. `--block X` runs only block X.

## 3. Gates (stop the campaign with a clear message)
- `telemetry.py --probe` must PASS before start; the CPU sensor it prints must be the CPU
  package, and its refresh must be <= 1 s.
- A8, A10, A12 pass-1 raw output byte-identical to lora_greedy on both test sets.
- Local systems: any pass-1 `error`. Systems with `uses_cloud = 1`: more than 2 % of pass-1
  calls with `error`. Either way: not all 217 commands scored in pass 1.
- Sampling below 9 Hz over a load window.
- End of every block: `python thesis3\bench\verify_campaign.py thesis3\out\c1 --partial`;
  stop if it FAILs.
Reproduction check (report, do not stop): pass-1 action_match per system vs
`thesis2/out/final/metrics_final.csv` and `v5cells_both_splits.csv` (see METHODS_NOTES);
write `repro_check.csv`. The old V5 local cells used n=216 (warm-up row excluded); here every
system is scored on all 217.

## 4. Before the campaign (show me the terminal output of all five, then stop)
Run everything below in `D:\condaenvs\thesis3meter`.
1. `python thesis3/bench/telemetry.py --selftest` and `--probe`.
2. `python thesis3/bench/make_synthetic.py thesis3/out/synth`,
   `python thesis3/bench/analyze.py thesis3/out/synth`,
   `python thesis3/bench/verify_campaign.py thesis3/out/synth --subset --partial --after-analysis --min-load-s 20 --min-idle-s 29 --cooldown-s 4 --loadedidle-s 5`.
   Expected: exactly two FAIL lines, both for r6a (its CPU sensor is missing on purpose).
3. `python thesis3/bench/corrupt_test.py thesis3/out/synth`: expected `24 of 24 corruptions caught`.
4. `run_bench.py --plan`: the 132 runs, block order, system -> env -> legend table, the
   ensemble wiring, both dataset sha256, ETA per block from
   `thesis2/out/final/unified_efficiency.csv` (wall_s / n_commands x 217, minimum 300 s,
   + warm-up + 60 s cool-down + 60 s loaded idle + model load + a 300 s idle window every
   60 min). Write `thesis3/bench/PLAN.md`.
5. `run_bench.py --smoke`: flat_gpu and lora_greedy on held_out_v3 with a 30 s minimum load
   into `thesis3/out/smoke/`; then `analyze.py` and
   `verify_campaign.py thesis3/out/smoke --subset --partial --after-analysis --min-load-s 30`
   and print action_acc, lat_p50_ms, net_wh_per_1k, energy_complete, loaded_idle_net_w,
   model_load_s (summary.csv) and sample_hz, gpu_counter_vs_integral_pct,
   cpu_sensor_refresh_s (quality.csv).
I start the real campaign myself with
`python thesis3\bench\run_bench.py --campaign c1 --block <A|B|C1|C2|D>`.
