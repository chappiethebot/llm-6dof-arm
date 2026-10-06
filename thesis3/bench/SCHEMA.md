# thesis3 data contract

The harness writes these files into `thesis3/out/<campaign>/`. `analyze.py` and
`verify_campaign.py` read only these files (plus `systems_expected.csv` next to them).
Every timestamp uses the same clocks as `telemetry.py`: `*_mono_ns` =
`time.perf_counter_ns()` (system-wide QueryPerformanceCounter on Windows, so parent and
child processes agree), `*_utc_ns` = `time.time_ns()`.

## Test data (identical for every system)
- `data/held_out_v3.jsonl` and `data/held_out_v4c.jsonl`, 217 commands each, nothing else.
- Every system that builds a prompt uses the **V5 legend** on both sets. `legend` records
  what the system used: `v5`, `none` (reads no legend), or for the 8 systems marked
  `recorded` in `systems_expected.csv` the value the thesis2 code uses: `v5` for the 4
  salsa_full systems, `salsa_row` (the SALSA row scorer's own row legend) for the 4 cascades
  that call a row model (cascade_local_tauA, pipeline_always_local, cascade_cloud_tauA,
  cascade_cloud_always).
- "V5" is the legend version, not a third test set. If a `held_out_v5*.jsonl` file exists,
  the harness stops and asks.
- The sha256 of both files goes into `machine.json` (`datasets`) and onto every run row.
  `verify_campaign.py` also checks that every run of a test set was scored against
  identical ground truth, command by command.

## systems_expected.csv (shipped, not written by the harness)
The 66 systems with family, kind, env, uses_cloud, repeats, legend_expected, system_class
(`system`, `ceiling` for cascade_oracle, `vocabulary_unfair` for rules_open). Only
`system` rows are ranked; the other two are reported but never ranked or put on a frontier.

## machine.json
`telemetry.machine_info()` plus: CPU model (`(Get-CimInstance Win32_Processor).Name`), GPU
name, driver, CUDA, torch, transformers, pandas, numpy and ollama versions, Windows build,
active power plan (`powercfg /getactivescheme`), LibreHardwareMonitor version and update
interval, git commit, conda envs, and `datasets: {held_out_v3: <sha256>, held_out_v4c: <sha256>}`.

## telemetry_<session>.csv
One continuous file per harness launch, written by `telemetry.Telemetry` at 10 Hz, never
overwritten (a reused session name refuses to start). The sampler runs in the parent
harness process; the system under test runs in a child process registered with
`tel.track_pid(child.pid)`. Phases:

| phase | run_id | length |
|---|---|---|
| `idle` | the idle window's id, `idle_<session>_<NNNN>` (unique) | 300 s, nothing loaded |
| `modelload` | the system's **held_out_v3** run id | as long as loading takes |
| `loadedidle` | the system's **held_out_v3** run id | 60 s, model resident, no calls (standby cost) |
| `warmup`, `load`, `cooldown` | the run's id | 3 calls / >= 300 s and >= 1 pass / 60 s |
| `between` | `""` | gaps, ignored |

An idle window is taken between systems whenever the last one ended more than 60 min
ago; it must end at most 60 min before the system's v3 load starts (checked in UTC). Then
one model load and loaded idle per system, its v3 run, and its v4c run with the model still
resident.

## idle.csv
idle_id, session, t_start_mono_ns, t_end_mono_ns, t_start_utc_ns, t_end_utc_ns.

## runs.csv (one row per system x test set; the row is the commit marker, written LAST)
| column | meaning |
|---|---|
| run_id, session | unique id; telemetry file it lives in |
| system, family, kind, device, runtime | as in systems_expected.csv (`kind` local/cloud, `device` gpu/cpu/cloud) |
| split | `held_out_v3` or `held_out_v4c` |
| legend, dataset_sha256 | see above |
| idle_id | the system's idle window |
| t_load_start_mono_ns, t_load_end_mono_ns, t_load_start_utc_ns, t_load_end_utc_ns | the load window |
| passes, row_passes, calls_total | all load passes; passes with per-call rows (<= 20); load calls in total |
| repeats | calls per command per pass (1, or 5 for gpt-4o-mini cells and cascade_cloud_*) |
| model_sha256, model_load_s | weights hash (or `n/a`); harness-timed load seconds (v3 row, local only) |
| uses_cloud | 1 if any call can go to gpt-4o-mini, else 0 |
| gpu_temp_start_c, notes | temperature at load start; free text incl. telemetry error counts |

## Per-run files (written in this order, then the runs.csv row)
1. `outputs/<run_id>.jsonl`: `{"run_id", "pass", "cmd_id", "repeat", "raw", "parsed"}` for
   every pass-0 and pass-1 call, and for any later call (passes 2..20) whose `pred_hash`
   differs from the same command's pass-1 hash. Raw text can be huge (one trace exceeded
   128,000 characters), so it is not in a CSV. Rescore, never re-run.
2. `calls/<run_id>.csv.gz`: one row per call for pass 0 and passes 1..min(P, 20):

| column | meaning |
|---|---|
| run_id, pass, order_idx, cmd_id, repeat | pass 0 = warm-up on commands 0, 1, 2 (not scored) |
| t_start_mono_ns, t_end_mono_ns, latency_ms | the call end to end (prompt build, inference, parsing) |
| t_infer_start_mono_ns, t_infer_end_mono_ns | the model call alone |
| valid_json, action_ok, verb_ok, target_ok, loc_ok | 0/1 from `llm/score.py`; `target_ok`/`loc_ok` empty only when the truth has no such field; the others are never empty |
| verb_true, verb_pred, loc_true, loc_pred | verb strings; location symbols (`grid_A1`..`box_3`, `max_reach`, `nearest_free_cell`), empty when none |
| row_true, row_pred, col_true, col_pred, row_word | grid commands only |
| pred_hash | first 12 hex of sha1 of the parsed prediction as sorted-key JSON |
| confidence | classifier: product of the three heads' max softmax; transformers LLM: exp(mean token log-prob); else empty |
| tokens_in, tokens_out, tokens_thinking, ttft_ms | generative systems |
| error | empty, or the exception text after all retries (scored wrong, never skipped) |
| extra | JSON: `escalated`, `cloud_calls`, `retries`, `accepted_tokens`, `draft_tokens`, `verifier_passes`, `first_rejection_field` as applicable |

3. `latency/<run_id>.npy`: float32 latency_ms of EVERY load call (all passes), in call order.
   Fast systems make millions of calls in 300 s (rules: ~16 million); per-call rows stop at
   pass 20, latency is kept for all.
4. `passes/<run_id>.csv`: pass, t_start_mono_ns, t_end_mono_ns, calls, action_ok_sum, one
   row per load pass. Every pass is complete (217 x repeats calls).

Scoring uses `llm/score.py` unchanged. Accuracy is taken from pass 1 (repeats: majority vote
over all 5 planned repeats; a missing or failed repeat is a wrong vote, a tie is wrong).
Latency uses every load call. A "command" in every per-command, per-1,000 and per-million
figure is one decided command (5 calls for gpt-4o-mini).

## How missing data is treated
- A power sensor covering < 95 % of a load window: that run's power means and energy
  totals are left empty (`energy_complete = False`), never zero.
- A telemetry sample whose sensors failed keeps its time stamp and has NaN values.
- An empty `action_ok`, `verb_ok` or `valid_json` is scored wrong and fails verification.
- `verify_campaign.py` fails the campaign on any of these, so nothing is found missing at
  writing time.
