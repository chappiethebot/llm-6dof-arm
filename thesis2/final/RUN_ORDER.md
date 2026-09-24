# RUN_ORDER.md — the final run, step by step

Written 2026-09-23. Every command is one PowerShell line, `;` separated. Run
from the repo root, `C:\Users\USER\llm-6dof-arm`.

**Runtimes tagged [Guessing] are estimates from the smoke tests, not
measurements.** Anything tagged [Certain] was timed on this machine.

**Envs**

| env | path | used for |
|---|---|---|
| `thesis2gpu` | `D:\condaenvs\thesis2gpu` | everything except Ollama cells, MuJoCo and Isaac |
| `thesis` | (Ollama client + MuJoCo) | `s1_run_v5_cells.py` local cells |
| `thesis2rl` | `D:\condaenvs\thesis2rl` | GRPO training only; dev inference is delegated to thesis2gpu |
| `isaac6` | `D:\condaenvs\isaac6` | Stage 5 Isaac only |

Activate with e.g. `conda activate D:\condaenvs\thesis2gpu`.

---

## Rulings applied on 2026-09-23

- **GRPO:** `D:\condaenvs\thesis2rl` is the separate TRL training environment.
  `thesis2gpu` and `thesis` packages remain unchanged. Save LoRA adapters only;
  every dev-selection and held-out inference uses `thesis2gpu` via PEFT.
  The adapter save/load probe passed, including 5/5 train commands. [Certain]
- **Legacy arms:** retrain with the original train/dev recipes and seeds,
  persist selected weights, then compare with all 217 saved v3 predictions.
  Freeze only at 217/217; otherwise retain "retrained, not reproduced" and the
  differing command IDs. The user runs these trainers. [Certain]
- **Local R = 1 accepted.** Cloud R = 5; both come from the V5 census.
  Repeat determinism remains untestable for local cells; use the cross-run
  comparison. [Certain]
- **SALSA:** no prompt changes. The three seen-command disagreements recovered
  from Claude's diagnostic are in `thesis2/out/final/CODEX_HANDOFF.md`.
- **H6 remains declared:** the primary family is still eight hypotheses. Missing
  GRPO predictions mean NOT RUN, never removal from the family. [Certain]

The scripts are prepared; full training and evaluation have NOT been run by
Codex. Stage-0 checks and the adapter compatibility probe are the only new
executions. Stop after Stage 0 to review the results before Stage 1. [Certain]

### Review of the 2026-09-23 additions, 2026-09-24

Verified independently: the cross-env adapter probe re-run (5/5, sha256 matches
the recorded one), the GRPO config against the brief (lr 2e-5, cosine, beta 0.04,
8 generations, temperature 1.0, 64 new tokens, 10 epochs, seed 20260827, LoRA
r=16/alpha=32/dropout 0.05 on q,k,v,o,gate,up,down with lm_head excluded), the
reward (`llm/score.py` strict match, 0.0 on invalid JSON), the OOM path (records
peak allocated/reserved and refuses to fall back to 0.5B), the legacy recipes
against the originals (hyperparameters identical, model classes IMPORTED from
`a1_arms.py`, and the original argmax-load behaviour preserved rather than
"fixed", which is what makes 217/217 reachable), and every Stage-1 gate.

**Two rule-3 defects were found and fixed on 2026-09-24:**

1. `assert_no_heldout_vocab` was applied to the TextCNN and MLP-BoW
   vocabularies, and `top` occurs once in `train_v2`. The full 230-command run
   would therefore have **aborted with GATE FAIL** — the 5-command smoke test hid
   it because those five commands contain no `top`. Rule 3 forbids *adding*
   held-out vocabulary; rule 4 says `train_v2` is used as it is. Both scripts now
   call `common.assert_vocab_from_train_only`, which fails on any token not in
   `train_v2` and reports inherited held-out words separately. [Certain]
2. None of `s0_grpo_signal_check.py`, `s0_train_grpo.py`, `s0_retrain_legacy.py`
   had the rule-3 startup assert. Added: the GRPO scripts guard the chat wrapper
   and system line they contribute (the V5 legend is pre-existing and excluded,
   as in `s0_salsa_full.py`); `s0_retrain_legacy.py` guards the `train_v2`
   vocabulary its recipes build from. [Certain]

`.vscode/extensions.json` was created outside `thesis2/`, which rule 1 forbids.
It is an editor artefact, not work product; it is untracked and left in place for
you to delete. [Certain]

---

## STAGE 0 — new arms, train/dev only

### S0-1 SALSA letter-token and greedy check

* env `thesis2gpu`
* `conda activate D:\condaenvs\thesis2gpu ; python thesis2\final\s0_salsa_full.py --check --model qwen2.5-3b-lora`
* ~1 min [Certain — already run, GATE PASS]
* output: `thesis2\out\final\logs\s0_salsa_full_check_*.log`
* check: `Select-String -Path thesis2\out\final\logs\s0_salsa_full_check_*.log -Pattern "GATE"` → **GATE PASS**. Good = `' A'..' N'` each one token, ids unique, and the greedy next token after `Answer:` is a letter on 5/5 train_v2 commands.
* **already run for you**: all 14 letters single-token, 5/5 letters.

### S0-2 TextCNN

* env `thesis2gpu`
* `conda activate D:\condaenvs\thesis2gpu ; python thesis2\final\s0_train_textcnn.py`
* ~2 min [Guessing — 0.4 s/epoch on 5 commands measured; 230 commands × 30 epochs]
* outputs: `thesis2\models\final\textcnn.pt`, `thesis2\out\final\selection_textcnn.json`
* check: `python -c "import json;d=json.load(open('thesis2/out/final/selection_textcnn.json'));print(d['selected_epoch'],d['dev_action_match'],d['vocab_size'],d['smoke'])"` → good = `smoke` is `False`, `vocab_size` ≈ 142 (train_v2 only), `dev_action_match` well above 0.

### S0-3 MLP bag-of-words

* env `thesis2gpu`
* `conda activate D:\condaenvs\thesis2gpu ; python thesis2\final\s0_train_mlpbow.py`
* ~1 min [Guessing]
* outputs: `thesis2\models\final\mlpbow.pt`, `selection_mlpbow.json`
* check: same as S0-2 with `selection_mlpbow.json`.

### S0-4 SALSA-full on dev, four models

* env `thesis2gpu`
* `conda activate D:\condaenvs\thesis2gpu ; foreach ($m in "qwen2.5-3b-lora","Qwen2.5-3B-Instruct","qwen2.5-0.5b-lora","Qwen2.5-0.5B-Instruct") { python thesis2\final\s0_salsa_full.py --dev --model $m }`
* ~6 min total [Guessing — 173 ms/command measured on the 3B, 57 dev commands × 3 passes + model load ×4]
* outputs: `selection_salsa_full_<model>.json` ×4
* check: `Get-ChildItem thesis2\out\final\selection_salsa_full_*.json | ForEach-Object { python -c "import json,sys;d=json.load(open(sys.argv[1]));print(d['arm'],d['dev_action_match'])" $_.FullName }` → good = four numbers, all four files present.

### S0-5 GRPO: adapter probe, signal check, training

All commands below run from the repository root. Run them one at a time and
stop if a command fails. Do not train until the cross-environment probe passes.

1. Adapter probe, already passed on 2026-09-23 [Certain]:
   ```powershell
   & D:\condaenvs\thesis2rl\python.exe thesis2\final\s0_grpo_adapter_probe.py --save
   & D:\condaenvs\thesis2gpu\python.exe thesis2\final\s0_grpo_adapter_probe.py --load
   ```
   The load writes `thesis2/out/final/grpo_adapter_probe.json`. This is an
   untrained serialization probe, not an accuracy result.
2. Signal check, **user runs**, train_v2 only:
   ```powershell
   & D:\condaenvs\thesis2gpu\python.exe thesis2\final\s0_grpo_signal_check.py
   ```
   `grpo_signal_check.json` stores all 230 x 8 completions and rewards.
   Fewer than 23 mixed-reward prompts means skip GRPO-on-SFT; do not lower the
   threshold. Existing signal results are preserved.
3. Train from base, **user runs**:
   ```powershell
   & D:\condaenvs\thesis2rl\python.exe thesis2\final\s0_train_grpo.py --arm grpo_base
   ```
4. Train on SFT, **user runs**, only if the signal check permits it:
   ```powershell
   & D:\condaenvs\thesis2rl\python.exe thesis2\final\s0_train_grpo.py --arm grpo_sft
   ```
   This command reports a skip without training if fewer than 10% of prompts
   had mixed rewards.

Ten epochs, original brief's hyperparameters, explicit GRPO loss. A microbatch
of one and eight accumulation steps retain groups of eight. Each epoch saves
only its LoRA adapter; the trainer moves training tensors to CPU while a
`thesis2gpu` subprocess performs greedy dev_v2 selection, then restores them.
The selected adapter is the earliest epoch within one dev hit of the best.
Outputs: `selection_grpo_<base|sft>.json`, `history_grpo_<base|sft>.json`, and
`thesis2/models/final/grpo_<base|sft>/`. No full base-model copy is saved.
3B out-of-memory aborts with peak allocated/reserved memory recorded; no smaller
model fallback. Partial training runs are retained and require review before
restarting; optimizer state is not silently reset. [Certain: implemented;
training runtime and peak memory unmeasured]

### S0-5b Persist and reproduce the six legacy arms

**The user runs this, not Codex.** No held_out_v4c data is read. Training and
selection finish and the selected checkpoint hash is written before v3 is opened.

```powershell
foreach ($a in "tinybert","minilm_probe","knn_minilm","tfidf_logreg","bilstm","flant5") { & D:\condaenvs\thesis2gpu\python.exe thesis2\final\s0_retrain_legacy.py --arm $a; if ($LASTEXITCODE -ne 0) { break } }
```

- The training bodies in `legacy_recipes.py` match the original scripts, with
  artifact returns replacing their evaluation tails. Neural recipes keep seed
  20260820; kNN and TF-IDF keep their original fixed recipes. No sweeps.
- Outputs: `models/final/legacy_<arm>/`, `selection_<arm>.json`.
- **217/217 identical:** register in `frozen_checkpoints.csv` and save the v3
  predictions under the standard Stage-1 filename, so Stage 1 skips them.
- **Mismatch:** write `retrained_<arm>_held_out_v3.csv`, retain every differing ID,
  mark "retrained, not reproduced", and do not freeze. No automatic retry or
  selection using v3. Stop and review the discrepancy.
- Model loading in Stage 1 uses these persisted checkpoints; it never retrains.

Relevant checks (no training, downloads or held-out inference):
```powershell
& D:\condaenvs\thesis2gpu\python.exe thesis2\final\test_stage0.py
& D:\condaenvs\thesis2rl\python.exe thesis2\final\s0_train_grpo.py --arm grpo_base --check-config
```

### S0-6 Freeze

* env `thesis2gpu`
* `conda activate D:\condaenvs\thesis2gpu ; python thesis2\final\s0_freeze.py`
* seconds [Certain]
* output: `thesis2\out\final\frozen_checkpoints.csv`
* check: `Import-Csv thesis2\out\final\frozen_checkpoints.csv | Format-Table arm,selected_epoch,dev_action_match` → good = rows for `textcnn`, `mlpbow` and the four `salsa_full_*`; `grpo_base` present after training, `grpo_sft` present only if eligible and trained. A `smoke: true` selection record is refused by design.

> ### ⏸ PASTE TO ADVISOR AFTER STAGE 0
> The `selection_*.json` dev numbers, `frozen_checkpoints.csv`, GRPO selection/skip status, training curves, and legacy reproducibility records. Only the approved legacy v3 reproduction check has read a held-out split; no arm has read v4c in this run.

---

## STAGE 1 — accuracy on both held-out splits, v3 FIRST

Every arm runs **once per split**. A complete prediction file is never
overwritten without `--force-rerun-after-crash`, and each use of that flag is
appended to `thesis2\out\final\force_rerun_log.csv`.

### S1-1 The registry, and the plan

* `conda activate D:\condaenvs\thesis2gpu ; python thesis2\final\s1_run_arm.py --list`
* seconds [Certain]. Good = 30 arms, `grpo_*` registered; inference requires frozen adapters.

### S1-2 Local arms on held_out_v3

* env `thesis2gpu`
* `conda activate D:\condaenvs\thesis2gpu ; foreach ($a in "flat_gpu","flat_cpu","factorized_fp16","factorized_int4","rules_open","rules_firewalled","textcnn","mlpbow","lora_greedy","salsa_full_qwen2.5-3b-lora","salsa_full_Qwen2.5-3B-Instruct","salsa_full_qwen2.5-0.5b-lora","salsa_full_Qwen2.5-0.5B-Instruct","cascade_local_tauA","pipeline_always_local","cascade_oracle") { python thesis2\final\s1_run_arm.py --arm $a --split held_out_v3 }`
* ~25 min [Guessing — classifiers seconds; `lora_greedy` 217 × 730 ms ≈ 3 min; each SALSA-full ≈ 1 min; cascades ≈ 2 min each]
* outputs: `thesis2\out\final\pred_<arm>_held_out_v3.csv`
* check: `Select-String -Path thesis2\out\final\logs\s1_run_arm_*held_out_v3.log -Pattern "GATE|reproducibility|action match"` → good = **GATE PASS** for every arm, and `reproducibility ... 217/217 identical` for `flat_gpu`, `factorized_fp16`, `rules_open`, `lora_greedy`. Any count below 217/217 on a deterministic local arm is a GATE FAIL with the differing ids — stop and send me those ids.

### S1-3 Speculative arms (A8/A10/A12)

These three are **not** re-implemented in the runner: it refuses them with a
pointer, because their verified decoders live in their own scripts. Run those,
then the runner picks the outputs up for the reproducibility and byte-identity
checks:

* `conda activate D:\condaenvs\thesis2gpu ; python thesis2\scripts\a8_speculative.py ; python thesis2\scripts\a10_prefix_cache.py ; python thesis2\scripts\a12_smalllm_drafter.py`
* ~35 min [Guessing — A8 ≈ 7 min, A10 ≈ 7 min, A12 ≈ 7 min, plus loads]
* check: each prints `217 / 217 PASS`. If any prints less, stop.

### S1-4 The V5 cell grid on held_out_v3

* **plan first** (no model calls): `conda activate D:\condaenvs\thesis2gpu ; python thesis2\final\s1_run_v5_cells.py --split held_out_v3 --plan-only`
  * seconds [Certain — already run: 36 cells, 12,152 rows, row-count gate PASS]
* **local cells**, env `thesis` (Ollama): `conda activate thesis ; $env:OLLAMA_MODELS="D:\LLM" ; python thesis2\final\s1_run_v5_cells.py --split held_out_v3 --kind local`
  * ~3 h [Guessing — 31 cells × 217 calls at ~1.5 s]
* **cloud cells**, env `thesis2gpu`: `conda activate D:\condaenvs\thesis2gpu ; python thesis2\final\s1_run_v5_cells.py --split held_out_v3 --kind cloud`
  * ~1 h and **5,425 billed gpt-4o-mini calls** [Guessing on time; Certain on the call count — 5 cells × 217 × 5 repeats]
* output: `thesis2\out\final\results_v5cells_held_out_v3.csv`
* check: `python -c "import csv;r=list(csv.DictReader(open('thesis2/out/final/results_v5cells_held_out_v3.csv',encoding='utf-8')));print(len(r))"` → good = **12152**.

### S1-5 Cloud cascades and the ensemble, held_out_v3

* `conda activate D:\condaenvs\thesis2gpu ; foreach ($a in "cascade_cloud_tauA","cascade_cloud_always") { python thesis2\final\s1_run_arm.py --arm $a --split held_out_v3 }`
* ~25 min, **2,170 billed calls** (217 × 5 repeats × 2 arms) [Certain on the count]
* `ensemble` must come after S1-4: it pairs three already-measured V5 cells and refuses to run before they exist.

> ### ⏸ PASTE TO ADVISOR AFTER STAGE 1 v3 (reproducibility)
> Every `reproducibility ...` line from the S1-2/S1-3 logs, plus the 12,152 row
> count. This is the gate that says the pipeline reproduces the earlier run
> before any new split is touched.

### S1-6 Everything again on held_out_v4c

Same commands as S1-2, S1-4, S1-5 with `--split held_out_v4c`. There are no
saved v4c predictions for most arms, so the reproducibility line will say "first
measurement, nothing to compare" — that is expected, not a failure.

* ~4.5 h and **7,595 more billed calls** [Guessing on time]
* check: `Get-ChildItem thesis2\out\final\pred_*_held_out_v4c.csv | Measure-Object` → good = one file per arm you ran, each 217 × R rows.

> ### ⏸ PASTE TO ADVISOR AFTER STAGE 1 v4c
> The per-arm action match and row accuracy on both splits, side by side.

---

## STAGE 3 — measured latency and energy (before Stage 2, so Stage 2 can read it)

### S3-1 Idle baseline

* `conda activate D:\condaenvs\thesis2gpu ; python thesis2\final\s3_energy.py --idle`
* 5 min [Certain — 300 s by construction]
* output: `thesis2\out\final\idle_baseline.json`
* check: `Get-Content thesis2\out\final\idle_baseline.json` → good = `mean_util_pct` ≤ 2. If it fails, the script prints the GPU process list; close the offender and re-run. **Nothing else may run on the GPU during this.**

### S3-2 Each local arm on each split

* `conda activate D:\condaenvs\thesis2gpu ; foreach ($s in "held_out_v3","held_out_v4c") { foreach ($a in "flat_gpu","flat_cpu","factorized_fp16","factorized_int4","textcnn","mlpbow","lora_greedy","salsa_full_qwen2.5-3b-lora","cascade_local_tauA","pipeline_always_local") { python thesis2\final\s3_energy.py --arm $a --split $s } }`
* ~70 min [Guessing — 20 runs × (120 s load + 60 s cool-down + load time)]
* The idle baseline expires after 2 hours: re-run S3-1 if you pause.
* output: `thesis2\out\final\unified_efficiency.csv`
* check: `Import-Csv thesis2\out\final\unified_efficiency.csv | Format-Table arm,split,latency_p50_ms,joules_per_command,peak_vram_mb` → good = every row has a real J/command, `wall_s` ≥ 120, and `n_commands` ≥ 217.

> ### ⏸ PASTE TO ADVISOR AFTER STAGE 3
> `unified_efficiency.csv`. This carries the V5 LoRA's first measured latency and
> energy, and the cascades' first per-command measurements (the old modelled
> constants are used nowhere).

---

## STAGE 2 — every metric, every arm, both splits

* `conda activate D:\condaenvs\thesis2gpu ; python thesis2\final\s2_metrics.py`
* ~1 min [Guessing]
* output: `thesis2\out\final\metrics_stage2.csv`
* check: `Import-Csv thesis2\out\final\metrics_stage2.csv | Where-Object {$_.status -eq "valid"} | Format-Table arm,split,action_match,row_accuracy,ECE,token_acceptance` → good = no blank cell anywhere; generative arms read `n/a - no class probabilities` for ECE and `n/a - not speculative` for acceptance.

---

## STAGE 4 — significance, final table, figures

### S4-1 Significance

* `conda activate D:\condaenvs\thesis2gpu ; python thesis2\final\s4_significance.py`
* ~3 min [Guessing — 2,000 bootstrap resamples × the exploratory family]
* output: `thesis2\out\final\final_significance.csv`
* check: `Import-Csv thesis2\out\final\final_significance.csv | Where-Object {$_.family -eq "primary"} | Format-Table id,split,arm_a,arm_b,metric,diff_pts,p_holm,verdict` → good = 8 rows; H6 `NOT RUN` until its predictions exist; the log prints the old 26-hypothesis Holm values for H1/H2/H4/H5.

### S4-2 Final table

* `conda activate D:\condaenvs\thesis2gpu ; python thesis2\final\s4_metrics_final.py`
* ~1 min [Guessing]. Output `metrics_final.csv` — **the only file the final run's thesis tables may be built from.**
* check: `Import-Csv thesis2\out\final\metrics_final.csv | Measure-Object` → good = one row per (arm, split); `python -c "import csv;print(sum(1 for r in csv.DictReader(open('thesis2/out/final/metrics_final.csv',encoding='utf-8')) for v in r.values() if v==''))"` → good = **0**.

### S4-3 Figures

* `conda activate D:\condaenvs\thesis2gpu ; python thesis2\final\s4_figures.py`
* ~1 min [Certain — already run on empty inputs, 6 files]
* output: `thesis2\figures\final\fig1..fig6*.png`
* check: `Get-ChildItem thesis2\figures\final\*.png | Format-Table Name,Length` → good = 6 files, none tiny.

> ### ⏸ PASTE TO ADVISOR AFTER STAGE 4
> The primary-family table and figures 1–3.

---

## STAGE 5 — Isaac demo, seen commands only

### S5-1 Parser smoke test, no Isaac

* `conda activate D:\condaenvs\thesis2gpu ; python thesis2\demo\smoke_parsers.py`
* ~6 min [Guessing for all parsers; the subset I ran took ~4 min]
* output: `thesis2\out\final\demo_smoke_parsers.json`
* check: `GATE PASS`, and every parser `json 10/10`. **Already run for you** on 14 parsers — see the summary below. Cloud parsers are skipped unless you pass `--allow-cloud 20` (2 cloud parsers × 10 commands).

### S5-2 Isaac's own VRAM, then the fit table

Follow `thesis2\demo\vram_check.md` — it is a procedure, and every number in it
is deliberately blank. Fill `thesis2\out\final\isaac_vram_fit.csv`.

### S5-3 Live demo

* env `isaac6`
* `conda activate D:\condaenvs\isaac6 ; python thesis2\demo\run_isaac_demo.py --parser v5_ollama`
* then any parser marked **fits**: `--parser flat_gpu`, `--parser lora_greedy`, …
* check: the log's last block; every number is labelled `demo, not measurement`.

### S5-4 Held-out replay in Isaac (no new inference)

* `conda activate D:\condaenvs\isaac6 ; python thesis2\demo\run_isaac_demo.py --replay lora_greedy --split held_out_v4c`
* refuses to run unless `pred_lora_greedy_held_out_v4c.csv` exists and is complete.
* default `--ids` is the fixed rule in the script (first ids covering all 7 verbs, then in order): **0, 1, 2, 3, 9, 132, 141, 149, 159, 166** on both splits — chosen before any Isaac run and not by correctness. [Certain]

---

## STAGE 6 — the report

* `conda activate D:\condaenvs\thesis2gpu ; python thesis2\final\s6_report.py`
* ~1 min [Certain]
* output: `thesis2\out\final\FINAL_RUN_REPORT.md`
* check: `Select-String -Path thesis2\out\final\FINAL_RUN_REPORT.md -Pattern "^## "` → good = 7 sections, and §7 reports GRPO completion/skip status, cloud energy, CPU watts and the withdrawn arms.

---

## Billed cloud calls, totalled

| step | calls |
|---|---|
| S1-4 cloud V5 cells, v3 | 5,425 |
| S1-5 cloud cascades, v3 | 2,170 |
| S1-6 the same on v4c | 7,595 |
| S5-1 optional `--allow-cloud` | ≤ 20 |
| **total** | **≈ 15,210** |

All `gpt-4o-mini`, ≤ 8 tokens per reply for the cascades' row query, a full JSON
object for the V5 cells. Decide the budget before S1-4. [Certain on the counts]
