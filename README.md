# llm-6dof-arm

Code, data and results of an MSc thesis: turning a spoken or typed command for a robot arm
("put the bottle in the back left corner of the table") into a structured, schema-checked
JSON action, and comparing 66 ways of doing it (rule baselines, small classifiers, prompted
and fine-tuned local LLMs, cloud gpt-4o-mini, cascades, speculative decoding) on accuracy,
latency and energy. Actions are executed in MuJoCo (measured) and in Isaac Sim 6.0 (demo only).

The registered thesis title includes a vision pipeline with real-time object detection. That part
was not carried out; Section 1.4 of the thesis (Scope and deviations from the task description)
explains why.

## Contents

- [Repository layout](#repository-layout)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Reproducing the results](#reproducing-the-results)
- [Demo](#demo)
- [Data and seal rules](#data-and-seal-rules)
- [Known limitations](#known-limitations)
- [Licence](#licence) · [How to cite](#how-to-cite) · [Contact](#contact)

## Repository layout

| path | what it is |
|---|---|
| `data/` | command datasets (JSONL): `train_v2`, `dev_v2`, held-out test sets `held_out_v3` and `held_out_v4c` (217 commands each) |
| `schema/` | the action grammar (`schema_v2.json`), its generator and the per-object validator |
| `llm/` | model clients (Ollama, OpenAI, DistilBERT), prompts, retrieval, JSON extraction, scoring (`llm/score.py`) |
| `bench/` | the V1-V5 evaluation harness and the V5 result CSVs |
| `ft/` | fine-tuning: LoRA SFT on Qwen2.5, DistilBERT classifier, LoRA merge and GGUF export |
| `sim/` | MuJoCo backend (measured) and Isaac Sim backend (demo), demo entry points |
| `asr/` | voice input (offline Whisper) |
| `thesis2/` | the final run: one arm registry (`final/arms.py`), stage scripts `s0`-`s6`, every final number in `out/final/` |
| `thesis3/` | the unified bench: all 66 systems re-measured on both test sets under one measurement protocol (`bench/`) |
| `audit/` | adversarial checks of data, prompts and results; dated audit reports |
| `scripts/` | `smoke_test.py`, `validate.py`, `make_split.py` |
| `envs/` | one conda environment file per environment used |
| `docs/` | `REPO_MAP.md` (detailed map and invariants), `MOVED_PATHS.md`, `DEMO_PORTABLE.md` |
| `media/isaac_demo/` | README for the Isaac demo videos (the videos are on GitHub release v1.0) |
| `archive/` | one-off development checks and placeholders of cut phases; not needed for any result |

`docs/REPO_MAP.md` describes every folder in detail, including the rules the code relies on.

## Requirements

Everything was run on one Windows 11 machine: AMD Ryzen 5 7600X (6 cores), 31 GB RAM,
NVIDIA GeForce RTX 4060 Ti 16 GB, driver 617.14, Windows build 10.0.26300. Nothing else
was tested.

| need | for | version used |
|---|---|---|
| Miniconda / Anaconda | every environment | conda 25.11.1 |
| NVIDIA GPU + driver | Tier 2 and 3, Isaac Sim | RTX 4060 Ti 16 GB, driver 617.14 (CUDA 12.6 torch builds) |
| Ollama | the 30 local V5 cells and the ensemble | 0.20.3, models stored with `OLLAMA_MODELS=D:\LLM` |
| OpenAI API key | gpt-4o-mini cells, cloud cascades, ensemble | model `gpt-4o-mini` |
| MuJoCo | measured simulator | `mujoco` 3.10.0 from pip (in `envs/thesis.yml`) |
| Isaac Sim 6.0 | demo only | `isaacsim` 6.0.0.0 from pip (in `envs/isaac6.yml`), RTX GPU |
| LibreHardwareMonitor | Tier 3 only: CPU package power | any recent release, web server on port 8085 |
| llama.cpp b10582 | only to rebuild the LoRA GGUFs yourself | `convert_hf_to_gguf.py`, `llama-quantize.exe` |

### Conda environments

| env file | used for | Python | torch |
|---|---|---|---|
| `envs/thesis.yml` | quick start, MuJoCo, Ollama/OpenAI clients, V1-V5 bench, demo; enough for Tier 1 | 3.11.15 | 2.12.1 CPU |
| `envs/thesis2gpu.yml` | thesis2 final run, thesis3 workers, Tier 1 (byte-identical figures), Tier 2 | 3.11.16 | 2.14.0+cu126 |
| `envs/thesis3meter.yml` | thesis3 parent: power sampling, `analyze.py`, `verify_campaign.py` | 3.11.17 | none |
| `envs/thesis2rl.yml` | GRPO training only | 3.11.16 | 2.14.0+cu126 |
| `envs/train.yml` | `ft/` training (LoRA, DistilBERT) | 3.11.15 | 2.11.0 CPU |
| `envs/isaac6.yml` | Isaac Sim 6.0 demo | 3.12.14 | 2.10.0 CPU |

Each file pins every pip package to the version installed on 2026-11-08. `requirements.txt`
and `requirements.lock.txt` are older (audit of 27 Aug 2026) and describe the `thesis` env only.

### Models

| model | where it comes from |
|---|---|
| `qwen2.5:3b`, `qwen2.5:0.5b`, `llama3.2:3b`, `deepseek-r1:7b` | `ollama pull <name>`; the digests that were measured are in `audit/MODEL_PROVENANCE.txt` |
| `qwen2.5-3b-lora`, `qwen2.5-0.5b-lora` | not downloadable: built from the LoRA adapters, see [Ollama LoRA models](#ollama-lora-models) |
| Qwen2.5-0.5B/3B-Instruct, distilbert-base-uncased, flan-t5-base, TinyBERT, all-MiniLM-L6-v2 | downloaded from Hugging Face on first use (revisions in `thesis2/out/final/METHODS_NOTES.md` item 6) |
| fine-tuned weights (DistilBERT classifier, LoRA adapters and their GGUFs) | not in git; GitHub release v1.0 assets, see [Files not in git](#files-not-in-git). The frozen thesis2 arms (`thesis2/models`) are not released. |

## Installation

PowerShell, from the folder you want the repository in:

```powershell
git clone <repository URL> llm-6dof-arm
cd llm-6dof-arm
conda env create -n thesis -f envs\thesis.yml
conda env create -n thesis2gpu -f envs\thesis2gpu.yml
conda env create -n thesis3meter -f envs\thesis3meter.yml
Copy-Item .env.example .env
```

Then put your OpenAI key into `.env` (only needed for cloud systems). `envs\thesis.yml`
took 99 s to create on the measurement machine, where pip already had the packages cached; on
a new machine expect longer, and the CUDA env downloads the CUDA build of torch (several GB).
Create `thesis2rl`, `train` and `isaac6` only if you need them.

If conda asks you to accept the Terms of Service for the defaults channel, run
`conda tos accept`, or change the channel to conda-forge.

### Files not in git

Model weights and the raw thesis3 campaign data are too large for git. They are attached to
GitHub release v1.0 with `SHA256SUMS.txt` and `RELEASE_NOTES.md`. Every zip holds
repository-relative paths, so unpack it at the root of the clone
(`Expand-Archive <asset>.zip -DestinationPath .`); put the two `.gguf` files into `ft\export\`.

| release asset | gives | size | needed for |
|---|---|---|---|
| `distilbert_cls_epoch_09.zip` | `ft/runs/distilbert_cls/epoch_09/` | 266 MB | Tier 2, Tier 3, demo |
| `qwen3b_lora_checkpoint-87.zip` | `ft/runs/qwen3b_lora/checkpoint-87/` (adapter, no training state) | 122 MB | Tier 3, rebuilding the 3B GGUF |
| `qwen0.5b_lora_checkpoint-87.zip` | `ft/runs/qwen0.5b_lora/checkpoint-87/` (adapter, no training state) | 37 MB | Tier 3, rebuilding the 0.5B GGUF |
| `qwen3b-lora-Q4_K_M.gguf`, `qwen0.5b-lora-Q4_K_M.gguf` | `ft/export/` | 1,930 MB, 398 MB | Ollama LoRA models |
| `modelfiles.zip` | `ft/export/Modelfile.qwen2.5-3b-lora`, `...-0.5b-lora` | 1 KB | Ollama LoRA models |
| `qwen3b_LICENSE_NOTICE.zip`, `qwen0.5b_LICENSE_NOTICE.zip` | licence, notice and change notice for the GGUFs | 4 KB each | keep with the GGUFs |
| `thesis3_c1.zip` | `thesis3/out/c1/` | 684 MB (957 MB unpacked) | Tier 1 (thesis3) |
| `isaac_demo_videos.zip` | `media/isaac_demo/*.mp4` | 267 MB | watching the demo |

`thesis2/models` (4.0 GB): not released. The final run's predictions are in
`thesis2/out/final/pred_*.csv`, and Tiers 0 to 2 need nothing else. Re-running these arms is Tier
3, which is tied to the measurement machine.

The LoRA zips hold the adapter, tokenizer and chat template only; the training state
(`optimizer.pt`, `scheduler.pt`, `rng_state.pth`, `trainer_state.json`, `training_args.bin`) is not
released. The full checkpoint folders (with training state) are not released, so the thesis2
frozen-checkpoint check does not pass on the released adapters. This check is only part of Tier 3.

SHA-256 of the main weight files:

```
dc8747fb42df10fe20b4b8e2c03d738ef228a6342b883e2c24dcf6aefea5eadc  ft/runs/distilbert_cls/epoch_09/model.pt
607076e37927839602a147f80299080011154327cf2035406046c757987a76f0  ft/runs/qwen3b_lora/checkpoint-87/adapter_model.safetensors
40fee997524939dd13e096f0dc1338d716ffe34bc0eabe899cf56dd512a3b71a  ft/runs/qwen0.5b_lora/checkpoint-87/adapter_model.safetensors
21d0a833fce4d7fef61096bd80e7e8e382e5bc497666f0a182e98c0e15bea1d2  ft/export/qwen3b-lora-Q4_K_M.gguf
ae7d182b5cfa710f4e64c36d339b9ec975133fac655c6d5bab9babc874800f15  ft/export/qwen0.5b-lora-Q4_K_M.gguf
```

Every new thesis2 arm also has its checkpoint hash in `thesis2/out/final/frozen_checkpoints.csv`;
the thesis2 and thesis3 runners refuse to run on a mismatch.

### Ollama LoRA models

The two LoRA models were merged, converted to GGUF and quantised to Q4_K_M (the same level as
the stock Qwen2.5 models) by `ft/merge_and_export.py`. With the two GGUFs and `modelfiles.zip`
from the release unpacked into `ft/export/` (the Modelfiles point at `./<file>.gguf`):

```powershell
ollama create qwen2.5-3b-lora -f ft\export\Modelfile.qwen2.5-3b-lora
ollama create qwen2.5-0.5b-lora -f ft\export\Modelfile.qwen2.5-0.5b-lora
ollama list
```

On the measurement machine the GGUF hashes above are the names of the weight blobs in the
Ollama store, so these files are exactly what was served. To rebuild the GGUFs from the
adapters instead, run `python ft\merge_and_export.py` in the `train` env; it expects llama.cpp
b10582 at `C:\Users\USER\llama.cpp` (change `LLAMA_CPP` at the top of the script).

## Quick start

```powershell
conda activate thesis
python scripts\smoke_test.py
python scripts\validate.py schema\schema_v2.json data\seen_v2.jsonl data\held_out_v3.jsonl
```

Expected output (5 s for the smoke test; Gymnasium-Robotics also prints a warning about the
Adroit environments, which is harmless):

```
[ ok ] mujoco
       -> resolved Fetch env id: FetchPickAndPlace-v4
[ ok ] gymnasium + gymnasium-robotics (Fetch)
[ ok ] stable-baselines3 (SAC + HER)
[ ok ] sb3-contrib (TQC)
[ ok ] ultralytics (YOLO)
[ ok ] opencv-python
[ ok ] faster-whisper
[ ok ] openai (cloud arm)
[ ok ] ollama (local arm)
[ ok ] numpy

10/10 components OK
Environment ready.
```

`validate.py` ends with `Exact command overlap between the two files: 0` and `RESULT: PASS`.

## Reproducing the results

Four tiers, cheapest first. Tiers 0-2 take minutes. Run them in a separate clone, or restore
afterwards with `git checkout -- thesis2`: Tier 1 and 2 rewrite tracked result files in place.

### Tier 0: the test sets are the ones the thesis used

```powershell
Get-FileHash data\held_out_v3.jsonl, data\held_out_v4c.jsonl -Algorithm SHA256
```

```
5A767445A5BCBD717895B2D30D0725A1A3163AC9A0E7B9439FF7548131E8D687  ...\data\held_out_v3.jsonl
D507992EDE161ECE9A395C8746AD6790E57062ACC2A9A245A7703939EE8E95C8  ...\data\held_out_v4c.jsonl
```

The thesis appendix gives the prefixes `5a767445a5bc` and `d507992ede16`. `.gitattributes`
forces LF line endings, so the hashes are the same on every operating system.

### Tier 1: tables, significance and figures from the stored raw outputs (no GPU)

**thesis2** (accuracy, significance, figures; about 90 s):

```powershell
conda activate thesis2gpu
foreach ($s in "s2_metrics","s4_significance","s4_metrics_final","s5_extra_analysis","s4_figures","s6_report","make_defence_numbers","check_defence_numbers") { python thesis2\final\$s.py; if ($LASTEXITCODE -ne 0) { break } }
git diff --ignore-cr-at-eol --stat -- thesis2
```

Every script ends with `GATE PASS` (`make_defence_numbers` with `641 cited numbers`). The
`git diff` then shows only two files, `DEFENCE_NUMBERS_rev3.md` and `FINAL_RUN_REPORT.md`,
whose only changes are the date they were written and the order of rows with tied scores:
every CSV and all six figures in `thesis2/figures/final/` come out identical to the committed
ones. In the `thesis` env the numbers are also identical, but the PNG bytes differ
(matplotlib 3.11.0 instead of 3.11.2). Ignore the new files in `thesis2/out/final/logs/`.

**thesis3** (the unified bench: latency, energy, accuracy of all 66 systems; needs the release
asset `thesis3_c1.zip` unpacked at the repo root, giving `thesis3\out\c1`; about 2 min):

```powershell
conda activate thesis3meter
python thesis3\bench\verify_campaign.py thesis3\out\c1 --check-manifest
python thesis3\bench\analyze.py thesis3\out\c1
python thesis3\bench\verify_campaign.py thesis3\out\c1 --after-analysis
```

Expected: `MANIFEST PASS (753 files listed)`, then `analysed 132 runs, 184 columns`, then
`VERDICT PASS; 0 warnings`. `summary.csv`, `compare.csv`, `quality.csv`, `per_loc_type.csv`,
`per_verb.csv` and `confusion.csv` are rewritten byte for byte identical.

### Tier 2: re-run the DistilBERT classifier on both test sets (local GPU)

Needs `ft\runs\distilbert_cls\epoch_09\` from the release; downloads `distilbert-base-uncased`
(about 260 MB) on first use.

```powershell
conda activate thesis2gpu
python thesis2\final\s1_run_arm.py --arm flat_gpu --split held_out_v3 --force-rerun-after-crash
python thesis2\final\s1_run_arm.py --arm flat_gpu --split held_out_v4c --force-rerun-after-crash
```

Expected (16 s and 8 s):

```
  action match 0.7465 (n=217); row accuracy 0.6204 (n=108)
  reproducibility vs saved held_out_v3: 217/217 identical
GATE PASS
  action match 0.7143 (n=217); row accuracy 0.5556 (n=108)
  reproducibility vs saved held_out_v4c: 217/217 identical
GATE PASS
```

The prediction files are rewritten with new latencies (every other column is identical) and
`--force-rerun-after-crash` appends one line to `force_rerun_log.csv`; restore both with
`git checkout -- thesis2`.

### Tier 3: full re-measurement (not run for this README)

- **thesis3 campaign:** `thesis3/bench/RUNBOOK.md` (step by step) and `RUNBOOK_FAST.md`
  (one `block.ps1` call per block). Five blocks, A, B, C1, C2, D, about 31 h of machine
  time in total (`thesis3/bench/PLAN.md`), plus `finish.ps1`. On the measurement machine it
  ran from 6 Oct 2026 18:13 to 8 Oct 2026 03:09.
- **API cost:** about 13,700 gpt-4o-mini calls, 5.5 M input and 0.5 M output tokens:
  about 1.2 USD at 0.15 / 0.60 USD per million tokens.
- **thesis2 final run** (training, frozen checkpoints, predictions, energy):
  `thesis2/final/RUN_ORDER.md`, one stage at a time.
- **Paths are hard-coded for the measurement machine:** `thesis3/bench/registry.py` (`PY`,
  the python of each env), `block.ps1`, `finish.ps1`, `campaign.ps1`. Edit them first.
  `thesis2/out/final/frozen_checkpoints.csv` stores absolute checkpoint paths
  (`C:\Users\USER\llm-6dof-arm\thesis2\models\final\...`); the frozen-checkpoint gate stops
  with `frozen checkpoint is missing` unless the repository is at that path. Tiers 0-2 are not
  affected.
- **Expect different latency and energy on other hardware.** Accuracy of the deterministic
  local systems should reproduce exactly; Ollama and gpt-4o-mini outputs can vary between
  runs (see `thesis2/out/final/METHODS_NOTES.md`, items 8, 9 and 12).

### Where each result comes from

Thesis figures and tables (numbering of the thesis version of 3 Nov 2026) and the files their
data come from:

| thesis | content | data |
|---|---|---|
| Fig 6.1 | ladder of the main steps | `thesis2/out/final/metrics_final.csv`; `thesis2/out/final/FINAL_RUN_REPORT.md` §4, §4b |
| Table 6.1 | accuracy of all final-run systems | `thesis2/out/final/metrics_final.csv` |
| Fig 6.2 | V5 cells, both test sets | `thesis2/out/final/FINAL_RUN_REPORT.md` §4b |
| Fig 6.3 | held_out_v3 vs held_out_v4c per system | `thesis2/out/final/metrics_final.csv`; `thesis2/out/final/FINAL_RUN_REPORT.md` §4b |
| Fig 6.4 | predicted row per row word, six systems | `thesis2/out/final/metrics_final.csv` |
| Table 6.2 | primary hypotheses | `thesis2/out/final/final_significance.csv` |
| Fig 6.5 | pairwise significance | `thesis2/out/final/metrics_final.csv`; `thesis2/out/final/final_significance.csv` |
| Table 6.3 | latency, energy, memory on held_out_v3 | `thesis2/out/final/metrics_final.csv` |
| Fig 6.6 | accuracy vs latency (top) and energy (bottom) | `thesis2/out/final/metrics_final.csv`; `thesis2/out/final/FINAL_RUN_REPORT.md` §4, §4b |
| Fig 6.7 | speculative decoding | `thesis2/out/final/metrics_final.csv` |
| Fig 6.8 | speculative internals | first-rejection field on held_out_v3 (162 / 41 / 13 / 1 / 0): `thesis2/out/a11_rejection_positions.csv`; draft gains vs harms (49 / 8 vs 4 / 28): `thesis2/out/final/draft_verifier_gh.csv` |
| Table 6.4 | row accuracy, seen vs unseen row words | `thesis3/out/c1/summary.csv` |
| Fig 6.9 | by verb and by location class | `thesis3/out/c1/per_verb.csv`, `thesis3/out/c1/per_loc_type.csv` |
| Fig 6.10, 6.11, 6.12 | latency, energy, tokens | `thesis3/out/c1/summary.csv` |
| Table 6.5, 6.6 | power, safety channel | `thesis3/out/c1/summary.csv` |
| Fig 7.1 | same LoRA in Ollama and transformers | `audit/figures/cell_stats.csv`; `thesis2/out/final/FINAL_RUN_REPORT.md` §4b |
| Fig A.1 | V3/V4/V5 campaigns | `audit/figures/cell_stats.csv`; `audit/v4_cells.csv`; `thesis2/out/final/FINAL_RUN_REPORT.md` §4b |
| Table B.1 | all V5 cells | `thesis2/out/final/FINAL_RUN_REPORT.md` §4b |
| Fig C.1, Table C.1 | final run vs re-measurement; all 66 systems | `thesis2/out/final/metrics_final.csv`; `thesis3/out/c1/summary.csv` |
| Table D.1 | frozen checkpoints | the ten frozen arms: `thesis2/out/final/frozen_checkpoints.csv`; DistilBERT flat (epoch 9) and the LoRA adapters (checkpoint-87): release assets `distilbert_cls_epoch_09.zip`, `qwen3b_lora_checkpoint-87.zip`, `qwen0.5b_lora_checkpoint-87.zip` |

The thesis figures were redrawn from these files by the plotting scripts in the thesis LaTeX
sources; `thesis2/figures/final/` holds the repository's own versions. The background on A8 and
A12 is in `thesis2/out/FINAL_REPORT.md` §7.1-7.2.

Files and the scripts that write them:

| result | produced by | file |
|---|---|---|
| accuracy, per-field accuracy, ECE, latency and energy of the thesis2 run, both test sets | `thesis2/final/s4_metrics_final.py` | `thesis2/out/final/metrics_final.csv` |
| V5 cells (36 model x strategy cells), both test sets | `thesis2/final/s2_metrics.py` | `thesis2/out/final/v5cells_both_splits.csv` |
| hypotheses H1-H8 and the exploratory family (bootstrap CIs, McNemar, Holm) | `thesis2/final/s4_significance.py` | `thesis2/out/final/final_significance.csv` |
| figures 1-6 (accuracy vs latency, energy, VRAM; row accuracy by word group; row confusion; A10 breakdown) | `thesis2/final/s4_figures.py` | `thesis2/figures/final/fig*.png` |
| row confusion by row word; draft vs verifier gain/harm | `thesis2/final/s5_extra_analysis.py` | `thesis2/out/final/row_confusion_by_word.csv`, `draft_verifier_gh.csv` |
| every number quotable in the defence, each cited to its source cell | `thesis2/final/make_defence_numbers.py` (checked by `check_defence_numbers.py`) | `thesis2/out/final/DEFENCE_NUMBERS_rev3.md` |
| unified bench: accuracy, latency, energy, CO2, cost per million commands, all 66 systems | `thesis3/bench/analyze.py` | `thesis3/out/c1/summary.csv`, `report.md` |
| unified bench: paired comparisons, per verb, per location type, measurement quality | `thesis3/bench/analyze.py` | `thesis3/out/c1/compare.csv`, `per_verb.csv`, `per_loc_type.csv`, `quality.csv` |
| unified bench vs thesis2 accuracy | `thesis3/bench/run_bench.py` | `thesis3/out/c1/repro_check.csv` |

## Demo

The demo is shown, not measured: no number in the thesis comes from it (`docs/REPO_MAP.md`,
invariant 6). Recordings of the Isaac Sim demo with six different parsers are attached to
GitHub release v1.0 (`media/isaac_demo/README.md`).

**MuJoCo** (in-process, any machine):

```powershell
conda activate thesis
python -m sim.run_demo --model gpt-4o-mini --strategy zero_shot
```

**Isaac Sim 6.0** (two processes; RTX GPU; Ollama running for local models):

```powershell
conda activate isaac6
python sim\isaac_bridge.py
```

and in a second terminal:

```powershell
conda activate thesis
python bench\llm_client.py --backend isaac
```

`python bench\llm_client.py --backend isaac --model rules --selftest` checks the language side
without Isaac. `docs/DEMO_PORTABLE.md` has the full setup, voice input and the model switch
commands. `thesis2/demo/run_isaac_demo.py` replays the final run's predictions in Isaac. None
of the demo commands were run for this README.

## Data and seal rules

- `train_v2.jsonl` (230 commands) is the only training and retrieval corpus;
  `dev_v2.jsonl` (57) is used only for checkpoint and epoch selection.
- `held_out_v3.jsonl` and `held_out_v4c.jsonl` (217 commands each: 108 grid, 24 runtime,
  36 container, 49 without a location) are used for inference and scoring only: never for
  training, checkpoint or threshold selection, prompt wording or any hyperparameter.
- `held_out_v4c` replaces the row words of `held_out_v3` with `back`/`front`, which appear in
  no training set and no legend. It is the fair generalisation test, because the V5 legend was
  written after seeing V3 results (`thesis2/out/final/METHODS_NOTES.md`, item 1).
- No held-out row vocabulary (`back`, `front`, `upper`, `bottom`, `top`, `lower`) may be added to
  any legend, lexicon, prompt or vocabulary; the thesis2 scripts check this at start-up.
- `scripts\validate.py` checks a dataset against the schema and that no command appears in
  both a training and a held-out file.

## Known limitations

- **Machine-specific paths.** Paths in older audit, log and result files, and some scripts,
  refer to the measurement machine (`C:\Users\USER\...`, `D:\condaenvs\...`, `D:\LLM`). They
  are kept as recorded.
- **Measurement environments of V1-V5.** `requirements.txt` describes the audit-time
  environment, not the one that produced the V1-V5 rows; no lock file was taken then
  (`audit/AUDIT_REPORT.md`, finding F-ENV). The `envs/` files describe the environments as of
  November 2026.
- **One machine.** Latency and energy were measured on a single Windows PC. Energy covers GPU
  board and CPU package power only, not RAM, disk, fans or power-supply losses.
- **Two runtimes for the same LoRA.** The V5 LoRA cells run in Ollama at Q4_K_M;
  `lora_greedy` and the speculative arms run the same adapter in transformers FP16. Their
  latency, energy and raw text differ for runtime reasons (METHODS_NOTES item 12).
- **Cloud and sampled outputs.** gpt-4o-mini does not return identical text on repeated calls;
  its five repeats per command are collapsed by majority vote.
- **held_out_v3 is not blind for prompted systems** (see above); report `held_out_v4c` as the
  generalisation result.
- **Phrasing confound in held_out_v4c.** `back` and `front` commands use different carrier
  phrasing, so per-word readings cannot be attributed to the row word alone (METHODS_NOTES
  item 23).
- **Windows only.** Nothing was tested on Linux or macOS. The thesis3 scripts and
  `bench/llm_client.py` use Windows paths and PowerShell.

## Licence

- **Code:** MIT, see [LICENSE](LICENSE).
- **Data and results:** `data/` and the result files (`thesis2/out/`, `bench/results_v5_*.csv`,
  `audit/` CSVs and figures, and the thesis3 campaign data in the release) are licensed under
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **Models:** under their own licences, below.

The models used have their own licences: Qwen2.5-0.5B-Instruct (Apache 2.0),
Qwen2.5-3B-Instruct (Qwen Research Licence, non-commercial), Llama 3.2 (Llama 3.2 Community
Licence), DeepSeek-R1-Distill-Qwen-7B (MIT), DistilBERT, Flan-T5 and all-MiniLM-L6-v2
(Apache 2.0). The fine-tuned 3B adapter and its GGUF are derivatives of Qwen2.5-3B-Instruct
and are distributed under the Qwen Research Licence: Built with Qwen. The MASSIVE dataset (Amazon Science,
CC BY 4.0) is loaded by `thesis2/scripts/p7_massive.py` through the MTEB mirror
`mteb/amazon_massive_intent` (whose card states Apache 2.0); the `thesis2/out/*massive*` files
hold intent labels and predictions derived from it, not its utterances.

## How to cite

```bibtex
@mastersthesis{dhara2026llm6dof,
  author = {Ritayan Dhara},
  title  = {LLM-Driven Control and Vision-Based Manipulation for a 6-DoF Robotic Arm: Bridging Natural Language Commands with Real-Time Object Detection},
  school = {Technische Universität Dresden},
  year   = {2026}
}
```

## Contact

Ritayan Dhara, via GitHub issues on this repository.
