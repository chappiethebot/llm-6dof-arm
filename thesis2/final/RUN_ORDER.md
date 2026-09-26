# RUN_ORDER.md — the final run, step by step

Written 2026-09-23, rewritten in full 2026-09-25. Every command is one PowerShell
line, `;` separated. Run from the repo root, `C:\Users\USER\llm-6dof-arm`.

**Runtimes tagged [Guessing] are estimates from smoke tests, not measurements.**
Anything tagged [Certain] was timed or run on this machine.

**There are no "same as above" shortcuts in this file.** Every arm, every cell
and every split is written out, because a shortcut is where an arm gets silently
dropped.

## Envs

| env | path | used for |
|---|---|---|
| `thesis2gpu` | `D:\condaenvs\thesis2gpu` | everything except Ollama cells, GRPO training and Isaac |
| `thesis` | `C:\Users\USER\.conda\envs\thesis` | Ollama client + MuJoCo |
| `thesis2rl` | `D:\condaenvs\thesis2rl` | GRPO **training only**; all GRPO inference is delegated to `thesis2gpu` |
| `isaac6` | `D:\condaenvs\isaac6` | Stage 5 Isaac only |

`thesis2rl` holds TRL 1.13.0 with peft 0.21.0 / transformers 5.17.0 /
torch 2.14.0+cu126 — identical to `thesis2gpu` for all three. `thesis2gpu` and
`thesis` were not modified. [Certain]

---

## Rulings in force

- **GRPO** trains in `thesis2rl`, saves the LoRA adapter only, and every
  dev-selection and held-out inference runs in `thesis2gpu` via peft. The
  cross-env adapter probe passed and was independently re-run: 5/5 train_v2
  commands decoded, adapter sha256 `4e076c4b…` matches the recorded one. [Certain]
- **The six legacy arms** are retrained with the original recipes, seeds and
  train/dev splits, persisted, then compared against all 217 saved v3
  predictions. Freeze only at 217/217; otherwise the arm is marked
  "retrained, not reproduced" with the differing ids and is **not** frozen.
- **Local R = 1, cloud R = 5**, both read from the V5 census. Repeat-determinism
  is untestable at R = 1; the cross-run comparison against saved predictions is
  the determinism evidence for local arms. [Certain]
- **H6 stays in the primary family.** If GRPO is not trained, H6 reports NOT RUN.
  A declared hypothesis is never removed after the fact.
- **SALSA-full location labels were relabelled D..Q** on 2026-09-25 (below). No
  other SALSA change; the question text is otherwise untouched.

### The SALSA-full letter relabel, and its result

The location option letters A, B, C collided with the row letters inside the
location symbols (`A = grid_A1`). The 14 letters that occur in **no** location
symbol are D..Q, exactly 14 for 14 locations, so each location now carries one
fixed label in schema order: `D=box_1 E=box_2 F=box_3 G=grid_A1 H=grid_A2
I=grid_A3 J=grid_B1 K=grid_B2 L=grid_B3 M=grid_C1 N=grid_C2 O=grid_C3
P=max_reach Q=nearest_free_cell`. Verb (A..G) and target (A..C) labels are
unchanged: no verb or target symbol contains an uppercase letter.

Measured on all 230 `train_v2` commands, before and after, same model
(`Qwen2.5-3B-Instruct` + `qwen3b_lora/checkpoint-87`), same prompt text:

| metric | before (A..N) | after (D..Q) | change |
|---|---|---|---|
| action match | 143/230 = 0.6217 | 142/230 = 0.6174 | −1 command |
| verb | 205/230 = 0.8913 | 205/230 = 0.8913 | 0 |
| target (truth needs it) | 152/158 = 0.9620 | 152/158 = 0.9620 | 0 |
| location (truth needs it) | 60/132 = 0.4545 | 60/132 = 0.4545 | 0 |
| **row**, grid truths | **30/85 = 0.3529** | **56/85 = 0.6588** | **+30.6 pts** |
| column, grid truths | 44/85 = 0.5176 | 43/85 = 0.5059 | −1.2 pts |
| grid symbols exact | 24/85 = 0.2824 | 26/85 = 0.3059 | +2 |
| non-grid symbols exact | 36/47 = 0.7660 | 34/47 = 0.7234 | −2 |

The collision was real and specific to the row: row accuracy went from chance
(1/3) to 0.6588. Full-symbol accuracy did not move, because the **column** read
is now the binding constraint at 0.5059. 67 commands changed their predicted
location: 11 became correct, 11 became wrong, 45 changed and stayed wrong; 26
rows became correct while 19 columns became wrong. [Certain]

Row is no longer near chance, so the run continues. The column read is a separate
semantic weakness (`"middle edge"` → column 2), not a labelling artefact, and no
further prompt change was made.

---

## STAGE 0 — new arms, train/dev only

### S0-1 SALSA letter-token and greedy check
* env `thesis2gpu`
* `conda activate D:\condaenvs\thesis2gpu ; python thesis2\final\s0_salsa_full.py --check --model qwen2.5-3b-lora`
* ~1 min [Certain — run, GATE PASS]
* output: `thesis2\out\final\logs\s0_salsa_full_check_*.log`
* check: `Select-String thesis2\out\final\logs\s0_salsa_full_check_*.log -Pattern "GATE|clashing"` → **GATE PASS**, `' A'..' Q'` all one token and unique, `location labels clashing with them: none`, greedy next token a letter on 5/5 train_v2 commands.

### S0-2 TextCNN
* env `thesis2gpu`
* `conda activate D:\condaenvs\thesis2gpu ; python thesis2\final\s0_train_textcnn.py`
* ~2 min [Guessing]
* outputs: `thesis2\models\final\textcnn.pt`, `thesis2\out\final\selection_textcnn.json`
* check: `python -c "import json;d=json.load(open('thesis2/out/final/selection_textcnn.json'));print(d['selected_epoch'],d['dev_action_match'],d['vocab_size'],d['smoke'])"` → good = `smoke` False, `vocab_size` **144**, `dev_action_match` well above 0.

### S0-3 MLP bag-of-words
* env `thesis2gpu`
* `conda activate D:\condaenvs\thesis2gpu ; python thesis2\final\s0_train_mlpbow.py`
* ~1 min [Guessing]
* outputs: `thesis2\models\final\mlpbow.pt`, `selection_mlpbow.json`
* check: as S0-2 with `selection_mlpbow.json`.

#### The TextCNN / MLP-BoW vocabulary against the BiLSTM's

Both are built by the BiLSTM's own construction: lowercase whitespace split over
`train_v2` only, `{"<pad>":0,"<unk>":1}` seeded first, OOV → `<unk>`. Verified by
running both constructions side by side:

```
TextCNN/MLP vocabulary : 144
BiLSTM (same code)     : 144
identical token sets   : True
identical ORDER        : True
only in TextCNN/MLP    : none
only in BiLSTM         : none
first 12 tokens (both) : <pad> <unk> can you grab the toy car for me fetch bottle
```

**144 vs 142 is a counting difference, not a vocabulary difference.**
`train_v2` contains **142 distinct words**; `<pad>` and `<unk>` bring the table to
**144 entries**. `thesis2/scripts/a1_arms.py:342` prints `len(vocab)` = **144**,
but `thesis2/out/FINAL_REPORT.md:813` states "142 types" — the report dropped the
two specials. The arms are identical; the committed report's figure is the one
that is off by two. [Certain]

`top` occurs once in `train_v2`, so it is legitimately in all three
vocabularies. The rule-3 guard (`common.assert_vocab_from_train_only`) fails only
on a token **not** in `train_v2`, and reports inherited held-out words
separately — a zero-tolerance check would abort this legal run.

### S0-4 SALSA-full on dev, all four models
* env `thesis2gpu`
* `conda activate D:\condaenvs\thesis2gpu ; foreach ($m in "qwen2.5-3b-lora","Qwen2.5-3B-Instruct","qwen2.5-0.5b-lora","Qwen2.5-0.5B-Instruct") { python thesis2\final\s0_salsa_full.py --dev --model $m }`
* ~6 min [Guessing]
* outputs: `selection_salsa_full_qwen2.5-3b-lora.json`, `selection_salsa_full_Qwen2.5-3B-Instruct.json`, `selection_salsa_full_qwen2.5-0.5b-lora.json`, `selection_salsa_full_Qwen2.5-0.5B-Instruct.json`
* check: all four files exist and each has a `dev_action_match`.

### S0-5 GRPO
1. Adapter probe (already passed, 2026-09-23; re-verified 2026-09-24) [Certain]:
   `& D:\condaenvs\thesis2rl\python.exe thesis2\final\s0_grpo_adapter_probe.py --save` then
   `& D:\condaenvs\thesis2gpu\python.exe thesis2\final\s0_grpo_adapter_probe.py --load`
2. Signal check, **you run**, train_v2 only, in `thesis2gpu`:
   `& D:\condaenvs\thesis2gpu\python.exe thesis2\final\s0_grpo_signal_check.py`
   → `grpo_signal_check.json`, 230 × 8 completions. Fewer than **23** mixed-reward
   prompts (10 % of 230) means grpo_sft is skipped; the threshold is not lowered.
3. `& D:\condaenvs\thesis2rl\python.exe thesis2\final\s0_train_grpo.py --arm grpo_base`
4. `& D:\condaenvs\thesis2rl\python.exe thesis2\final\s0_train_grpo.py --arm grpo_sft` — reports a skip without training if the signal check does not permit it.

Config, all from the brief: lr 2e-5, cosine, KL β 0.04, 8 generations,
temperature 1.0, 64 new tokens, 10 epochs, seed 20260827, LoRA r=16 α=32
dropout 0.05 on q/k/v/o/gate/up/down with `lm_head` excluded, reward 1.0 iff
`llm/score.py` strict match else 0.0 (invalid JSON = 0.0). OOM writes peak
allocated/reserved and refuses to fall back to a smaller model.

**Selection rule.** The codified earliest-plateau rule is
`ft/train_classifier.py:337-339`, with `PLATEAU_TOL = 1` at line 83 and the
docstring at line 36 calling it "same rule as rung 3":

```python
best_hits = max(h["dev_hits"] for h in history)
plateau = [h for h in history if h["dev_hits"] >= best_hits - PLATEAU_TOL]
chosen = min(plateau, key=lambda h: h["epoch"])
```

GRPO's `select_epoch` (`thesis2/final/s0_train_grpo.py:25-27`) is the same rule:
tested against that code over 2,000 random 10-epoch dev histories, **0
disagreements**. [Certain]

> **Provenance note on "the rung-3 rule".** `ft/train_lora.py` does not implement
> selection: it prints the dev curve and says "pick the one for epoch …"
> (line 336), recommending the **argmax**. Applying the earliest-plateau rule to
> rung 3's own `dev_metrics.json` gives **epoch 1** for `qwen3b_lora`
> (hits/57 by epoch: 55, 53, 54, 54, 54, 54, 54, 54 → best 55, plateau
> [1,3,4,5,6,7,8], earliest = 1, argmax = 1), but the project uses
> `checkpoint-87` = **epoch 3** (`ft/merge_and_export.py:40`, "epoch 3,
> earliest-plateau rule, BOTH models"). Epoch 3 is what the rule gives for the
> **0.5B** (49, 52, 54, 54, … → plateau from 3); the comment applied the 0.5B's
> answer to the 3B. This changes nothing already measured — the checkpoint is
> fixed, frozen and dev-selected, with no held-out leakage — but the 3B
> checkpoint is not what the stated rule selects, and the thesis should say so.
> [Certain on the arithmetic]

### S0-5b The six legacy arms — persist and reproduce

**You run this.** No `held_out_v4c` data is read. Training and selection complete
and the checkpoint hash is written before v3 is opened.

```powershell
foreach ($a in "tinybert","minilm_probe","knn_minilm","tfidf_logreg","bilstm","flant5") { & D:\condaenvs\thesis2gpu\python.exe thesis2\final\s0_retrain_legacy.py --arm $a; if ($LASTEXITCODE -ne 0) { break } }
```

`legacy_recipes.py` imports the model classes, seed and `PLATEAU_TOL` from
`a1_arms.py` and matches every hyperparameter (lr 1e-2 / 5e-5 / 2e-3 / 3e-4,
batch 16/16/8, epochs 30/30/12, MAXL 24). It preserves the originals' **argmax
load** rather than the plateau epoch they print, which is what makes 217/217
reachable. [Certain]

* 217/217 → registered in `frozen_checkpoints.csv`, v3 predictions saved under the Stage-1 filename so Stage 1 skips the arm.
* mismatch → `retrained_<arm>_held_out_v3.csv`, every differing id retained, marked "retrained, not reproduced", **not** frozen. Stop and review.

### S0-6 Freeze
* `conda activate D:\condaenvs\thesis2gpu ; python thesis2\final\s0_freeze.py`
* seconds [Certain]
* check: `Import-Csv thesis2\out\final\frozen_checkpoints.csv | Format-Table arm,selected_epoch,dev_action_match` → rows for `textcnn`, `mlpbow`, the four `salsa_full_*`, `grpo_base`/`grpo_sft` if trained, and each legacy arm that reproduced 217/217.

> ### ⏸ PASTE TO ADVISOR AFTER STAGE 0
> The four SALSA dev numbers, the SALSA relabel table above, the TextCNN/MLP dev
> numbers, the GRPO signal-check fraction, the six legacy 217/217 results, and
> `frozen_checkpoints.csv`. Nothing has touched a held-out split yet.

---

## STAGE 1 — accuracy on both held-out splits

Every arm runs **once per split**. A complete prediction file is never
overwritten without `--force-rerun-after-crash`, and each use is appended to
`thesis2\out\final\force_rerun_log.csv`.

### S1-1 The registry and the plan
`conda activate D:\condaenvs\thesis2gpu ; python thesis2\final\s1_run_arm.py --list`
→ 30 arms; `grpo_base`/`grpo_sft` show BLOCKED until trained. [Certain]

### S1-2 held_out_v3 — every local arm, named

```powershell
conda activate D:\condaenvs\thesis2gpu ; foreach ($a in "flat_gpu","flat_cpu","factorized_fp16","factorized_int4","tinybert","minilm_probe","knn_minilm","rules_open","rules_firewalled","tfidf_logreg","bilstm","textcnn","mlpbow","flant5","lora_greedy","salsa_full_qwen2.5-3b-lora","salsa_full_Qwen2.5-3B-Instruct","salsa_full_qwen2.5-0.5b-lora","salsa_full_Qwen2.5-0.5B-Instruct","grpo_base","grpo_sft","cascade_local_tauA","pipeline_always_local","cascade_oracle") { python thesis2\final\s1_run_arm.py --arm $a --split held_out_v3 }
```

24 arms. `grpo_base`/`grpo_sft` exit 2 if not trained — expected, not a failure.
~40 min [Guessing]. Outputs: `pred_<arm>_held_out_v3.csv` each.

With S1-3 (`"a8_speculative"`, `"a10_speculative"`, `"a12_smalllm"`) and S1-5
(`"cascade_cloud_tauA"`, `"cascade_cloud_always"`, `"ensemble"`) that is all 30
registry arms on `held_out_v3`.

### S1-3 held_out_v3 — the speculative arms

```powershell
conda activate D:\condaenvs\thesis2gpu ; python thesis2\scripts\a8_speculative.py --split held_out_v3 ; python thesis2\scripts\a10_prefix_cache.py --split held_out_v3 ; python thesis2\scripts\a12_smalllm_drafter.py --split held_out_v3
```

Each now takes `--split`, `--limit` and `--outdir` (default
`thesis2\out\final\`), so **the committed files in `thesis2\out\` are never
overwritten**. Proved with a 5-command `train_v2` smoke test: 5/5 byte-identical
to plain greedy, output written to
`thesis2\out\final\a8_speculative_percommand_train_v2.csv`, and
`git status --short thesis2/out/` excluding `final/` was empty while
`git diff --stat` showed only the four scripts changed. [Certain]

Each script ends with a **byte-identity gate against `lora_greedy` on the same
split**: it loads `pred_lora_greedy_<split>.csv` and exits non-zero on any
mismatch. Run `lora_greedy` (S1-2) before these. ~35 min [Guessing].

Each script also writes the **standard `pred_<arm>_<split>.csv`** —
`pred_a8_speculative_`, `pred_a10_speculative_`, `pred_a12_smalllm_` — in the
same format as every other arm, through the one shared emitter
`common.preds_from_percommand`. It carries the arm's own verified output
(`spec_text` / `prefix_spec_text`) and its own per-command latency
(`spec_ms` / `prefix_spec_ms`) in `latency_ms`. Before 2026-09-26 the scripts
wrote only `*_percommand.csv`, so `s1_repro_check.py` and Stage 2 showed all
three arms as NOT RUN YET.

The held_out_v3 files were generated from the per-command CSVs that already
exist, **with no re-run** (`--emit-preds-only` loads no model):

```powershell
conda activate D:\condaenvs\thesis2gpu ; python thesis2\scripts\a8_speculative.py --split held_out_v3 --emit-preds-only ; python thesis2\scripts\a10_prefix_cache.py --split held_out_v3 --emit-preds-only ; python thesis2\scripts\a12_smalllm_drafter.py --split held_out_v3 --emit-preds-only
```

→ 217 rows each, GATE PASS; `s1_repro_check.py` then reads all three as
COMPLETE and **217/217 identical** to the earlier run's per-command files
(a genuine cross-run comparison: the two files differ and 0/217 latencies match).
[Certain — run]. On held_out_v4c no conversion step is needed: a normal run
writes both files.

### S1-4 held_out_v3 — the V5 cell grid

Plan first, no model calls:
`conda activate D:\condaenvs\thesis2gpu ; python thesis2\final\s1_run_v5_cells.py --split held_out_v3 --plan-only`
→ 36 cells, **12,152 rows**, row-count gate PASS. [Certain]

Local cells (31), env `thesis`:
`conda activate thesis ; $env:OLLAMA_MODELS="D:\LLM" ; python thesis2\final\s1_run_v5_cells.py --split held_out_v3 --kind local`
~3 h [Guessing].

Cloud cells (5), env `thesis2gpu`:
`conda activate D:\condaenvs\thesis2gpu ; python thesis2\final\s1_run_v5_cells.py --split held_out_v3 --kind cloud`
~1 h, **5,425 billed calls** [Certain on the count].

check: `python -c "import csv;print(sum(1 for _ in csv.DictReader(open('thesis2/out/final/results_v5cells_held_out_v3.csv',encoding='utf-8'))))"` → **12152**.

`s1_check_v5cells.py` runs **automatically at the end of every**
`s1_run_v5_cells.py` run and its exit code fails the runner's own gate. It GATE
FAILs on:

* **every cell:** any `prompt_error` row, or more empty-raw rows than V5 had
  for the same cell;
* **local cells**, held_out_v3: raw agreement with V5 below **210/216**;
* **cloud cells** (gpt-4o-mini, R = 5): any `model_error` row, on either split;
  and on held_out_v3, action match more than **3 points** (0.03) from V5's for
  the same cell, both on non-warmup rows.

Raw agreement is **reported for cloud cells but never gated on**, as the brief
specified: a sampled API at R = 5 does not repeat its raw text byte for byte.
On the current file `cot` agrees on 10/217 raw strings yet matches V5's
action match to four decimals (0.7177 vs 0.7177); all five cloud cells are
within 3 points (largest gap +0.0083, `few_shot`), 0 `model_error` rows, GATE
PASS. [Certain — run 2026-09-26]

To audit an existing file on its own:
`conda activate D:\condaenvs\thesis2gpu ; python thesis2\final\s1_check_v5cells.py --split held_out_v3`

#### S1-4b Re-run the 12 void cells (few_shot and rag)

The first held_out_v3 local pass voided **12 of the 31 local cells** — the
`few_shot` and `rag` cell of all six models — with 217/217 `prompt_error`
rows and **no model call**. `s1_run_v5_cells.py` passed
`load_seen_split(DEFAULT_K)` as `examples`, which returns a **tuple**, defaults
to the **V1** seen file, and is not what `rag` uses at all. Fixed 2026-09-26:
exemplars and retrieval now come from `bench/run_cell.py` itself through
`common.v5_pool` / `common.v5_examples_for`, so the path, `k` and the
per-command `rag` retrieval are V5's own. The 19 healthy cells were unaffected
and reproduce V5 at **216/216** raw agreement.

Prove the prompts first — no model calls, ~2 min:
`conda activate D:\condaenvs\thesis2gpu ; python thesis2\final\s1_prompt_equality.py --split held_out_v3 --n 5`
→ **few_shot 5/5 and rag 5/5 byte-identical** to `bench/run_cell.py`'s own
builder. [Certain — run, GATE PASS]

Then re-run only those 12 cells, env `thesis`, ~1 h [Guessing]:

```powershell
conda activate thesis ; $env:OLLAMA_MODELS="D:\LLM" ; python thesis2\final\s1_run_v5_cells.py --split held_out_v3 --replace-cells --cells "ollama:qwen2.5-3b-lora:few_shot,ollama:qwen2.5-3b-lora:rag,ollama:qwen2.5:3b:few_shot,ollama:qwen2.5:3b:rag,ollama:qwen2.5-0.5b-lora:few_shot,ollama:qwen2.5-0.5b-lora:rag,ollama:qwen2.5:0.5b:few_shot,ollama:qwen2.5:0.5b:rag,ollama:llama3.2:3b:few_shot,ollama:llama3.2:3b:rag,ollama:deepseek-r1:7b:few_shot,ollama:deepseek-r1:7b:rag"
```

`--replace-cells` deletes only those 12 cells' rows, logs every use to
`thesis2/out/final/force_rerun_log.csv`, and **requires `--cells`**. It is
**held_out_v3 only**: the v4c no-overwrite rule is unchanged, so on v4c each
cell still runs exactly once and a complete file is never replaced. The run ends
with the audit, which must report `prompt_error cells: 0` and every cell at or
above 210/216.

check: `Select-String thesis2\out\final\logs\s1_run_v5_cells_held_out_v3_*.log -Pattern "GATE|prompt_error cells|raw-agreement"` → **GATE PASS**, `prompt_error cells: 0`.

#### S1-4c gpt-4o-mini retries, and `--retry-errors`

**Every gpt-4o-mini call** — the five V5 cloud cells and both cloud cascades
— goes through `common.with_retries`:

* retried on connection errors, timeouts, **HTTP 429** and **HTTP 5xx**; any
  other error (400, 401, 403, 404) fails at once;
* up to **6 attempts**, sleeping **2, 4, 8, 16, 32 s** between them;
* the request is re-issued **unchanged** — same prompt, model and temperature.
  A retry repeats a request that produced no answer; it is not a second sample;
* the openai SDK's own retries (default `max_retries=2`) are switched **off**,
  so 6 attempts is 6 requests and the logged count is exact;
* the retry count is logged **per row**: column `retries` in
  `results_v5cells_<split>.csv` (`""` for local cells), and `"retries"` in the
  `note` JSON of `pred_cascade_cloud_*_<split>.csv`. Each retry also prints one
  line to the run log.

V5 cells use a wrapper (`common.RetryingClient`) around the unmodified
`llm/clients.OpenAIClient`. The cascades call the SDK directly in `arms.py`.
There, an exhausted retry used to be an **uncaught exception that killed the
run**; it is now a row with no prediction and `"stage": "model_error"` in its
`note`.

A `model_error` row is **not a measured outcome**. `s1_check_v5cells.py` fails
on any in a cloud cell, and `s1_run_arm.py` fails on any, including on an
already-complete file. `--retry-errors` removes and re-runs **only** the rows
whose stage is `model_error`. Every other row is kept byte for byte, and no other
missing row is filled. It is **allowed on held_out_v4c** because a connection
failure produced no answer, so re-running it re-measures nothing; the v4c
no-overwrite rule is otherwise unchanged. Every use is logged to
`force_rerun_log.csv` with the row count, flagged `[--retry-errors]` in the
`arm` column (the file's 4-column schema is kept). It is exclusive with
`--replace-cells` and `--force-rerun-after-crash`.

```powershell
conda activate D:\condaenvs\thesis2gpu ; python thesis2\final\s1_run_v5_cells.py --split held_out_v3 --kind cloud --retry-errors
conda activate D:\condaenvs\thesis2gpu ; foreach ($a in "cascade_cloud_tauA","cascade_cloud_always") { python thesis2\final\s1_run_arm.py --arm $a --split held_out_v3 --retry-errors }
```

The same two lines with `--split held_out_v4c` after S1-6. Run them only when
a gate names `model_error` rows; with none, each logs the use with 0 rows and
exits GATE PASS.

Stub test, no network and no model, ~1 min [Certain — run, all 4 pass]:
`conda activate D:\condaenvs\thesis2gpu ; python thesis2\final\test_retries.py`
— backoff 2-4-8-16-32 s, 6 attempts, 400 not retried; the prompt is identical
on every attempt; `s1_run_arm.py --retry-errors` on held_out_v4c re-runs 3/1085
rows and leaves 1,082 byte-identical; a plain run on a file with `model_error`
rows is GATE FAIL; `s1_run_v5_cells.py --retry-errors` re-runs 10/12,152 rows.
It writes to a temporary directory only. The cascade's own `model_error` branch
in `arms.py` is not exercised by the stubs: that needs the factorized model
loaded.

### S1-5 held_out_v3 — cloud cascades and the ensemble

```powershell
conda activate D:\condaenvs\thesis2gpu ; foreach ($a in "cascade_cloud_tauA","cascade_cloud_always","ensemble") { python thesis2\final\s1_run_arm.py --arm $a --split held_out_v3 }
```

The two cloud cascades are R = 5 → **2,170 billed calls**. `ensemble` pairs three
already-measured V5 cells and must come after S1-4. Both cascades retry
gpt-4o-mini as in S1-4c. If the gate reports `model_error` rows, re-run only those
with `--retry-errors`.

> ### ⏸ PASTE TO ADVISOR AFTER STAGE 1 v3 (reproducibility)
> Every `reproducibility ...` line from the S1-2/S1-3 logs, plus the 12,152 row
> count.

#### Which arms have a reproducibility comparison

The check runs for **every** arm that has saved predictions — 20 of 30 on v3 and
13 on v4c, not four. Composite arms are rebuilt deterministically from the saved
`p45_predictions_<split>.csv` columns.

| arm | v3 | v4c | source |
|---|---|---|---|
| flat_gpu | 217 | 217 | `p45_predictions_<split>.csv` (flat columns) |
| flat_cpu | 217 | 217 | same file, same columns |
| factorized_fp16 | 217 | 217 | v3 `factorized_fp16.csv`; v4c rebuilt from p45 factors |
| tinybert | 217 | 217 | `a1_tinybert_held_out_<split>.csv` |
| minilm_probe | 217 | 217 | `a1_frozen_probe_held_out_<split>.csv` |
| bilstm | 217 | 217 | `a1_bilstm_held_out_<split>.csv` |
| flant5 | 217 | 217 | `a1_flant5_held_out_<split>.csv` |
| lora_greedy | 217 | 217 | v3 `a8_speculative_percommand.csv`; v4c `a13_v4c_lora_percommand.csv` |
| a8_speculative | 217 | — | `a8_speculative_percommand.csv` |
| a10_speculative | 217 | — | `a10_prefix_cache_percommand.csv` |
| a12_smalllm | 217 | — | `a12_smalllm_drafter_percommand.csv` |
| knn_minilm | 217 | — | `baseline_knn_minilm.csv` (v3 only) |
| tfidf_logreg | 217 | — | `baseline_tfidf_logreg.csv` (v3 only) |
| rules_open | 217 | — | `baseline_rules.csv` (v3 only) |
| ensemble | 216 | — | `a1_ensemble_held_out_v3.csv` |
| cascade_local_tauA | 217 | 217 | rebuilt from p45 (τ_A = 0.50, SALSA letters row) |
| pipeline_always_local | 217 | 217 | rebuilt from p45 (τ = 1.0) |
| cascade_cloud_tauA | 217 | 217 | rebuilt from p45 (τ_A = 0.50, `cloud_row`) |
| cascade_cloud_always | 217 | 217 | rebuilt from p45 (τ = 1.0, `cloud_row`) |
| cascade_oracle | 217 | 217 | rebuilt from p45 (`truth_row`) |
| factorized_int4 | — | — | Phase 3 / A9 persisted only the aggregate |
| rules_firewalled | — | — | only the summary row was persisted |
| textcnn, mlpbow, 4 × salsa_full_*, grpo_base, grpo_sft | — | — | NEW arms — first measurement |

A deterministic local arm below 217/217 is a GATE FAIL that prints the differing
ids. Ollama and cloud arms report agreement only.

#### S1-5b The v3 reproducibility gate — `s1_repro_check.py`

Run this after S1-2/S1-3 and again after S1-5. No model runs, ~10 s:

`conda activate D:\condaenvs\thesis2gpu ; python thesis2\final\s1_repro_check.py --split held_out_v3`

It is the gate, not the in-runner line, because the in-runner check was broken:
it printed **`0/0 identical` and PASSED** for flat_gpu, flat_cpu,
factorized_fp16, lora_greedy, rules_open, cascade_local_tauA,
pipeline_always_local and cascade_oracle. The cause was the **join key**, not
the data — `common.score_row` sets `command_id=cid`, an `int` from
`enumerate()`, while `saved_predictions()` reads `str` keys from
`csv.DictReader`, so `set(saved) & set(first)` was empty and `same ==
len(shared)` was `0 == 0`. Both sides now key on `str(command_id)`, and
**comparing zero commands is a GATE FAIL**, as is comparing fewer commands than
the split has.

The script prints one line per arm — `compared/217`, identical, differing
command_ids — and first audits **every** `pred_*` file for completeness.
For `factorized_int4` and `rules_firewalled`, where the earlier run saved only
totals, it compares the totals to four decimals: **0.6083** and **0.4470**.

Verified 2026-09-26 on the files already written [Certain — run, GATE PASS]:
all **17** arms with per-command saved predictions reproduce at **217/217
identical** (14 on the first check, plus A8/A10/A12 once their standard files
were emitted), and both totals-only arms match exactly. The predictions were
right all along; only the check was broken.

#### Completeness is parsed CSV rows, never a line count

`s1_run_arm.py` reported `pred_grpo_base_held_out_v3.csv already has 986/217
rows -- COMPLETE`. `csv.DictReader` gives exactly 217 rows over 217 unique
command_ids: `common.pred_is_complete` counted **physical lines**
(`sum(1 for _ in open(...)) - 1`) and GRPO raw outputs contain newlines. This
was not cosmetic — with 220 physical data lines for 217 rows, a **crashed
grpo_base run would have been declared COMPLETE after only 46 of 217
commands**, and gate 3 would have refused to resume it.

`common.pred_census` now requires **exactly** `n × R` parsed CSV rows over
**exactly** `n` distinct `command_ids`; anything else is GATE FAIL with the
reason printed. Audited across all 24 existing v3 files: **0 fail**, and
`pred_grpo_base_held_out_v3.csv` is the one file whose line count (986)
differs from its row count (217). [Certain — run]

### S1-6 held_out_v4c — every arm, named explicitly

**No "same as above".** Local arms:

```powershell
conda activate D:\condaenvs\thesis2gpu ; foreach ($a in "flat_gpu","flat_cpu","factorized_fp16","factorized_int4","tinybert","minilm_probe","knn_minilm","rules_open","rules_firewalled","tfidf_logreg","bilstm","textcnn","mlpbow","flant5","lora_greedy","salsa_full_qwen2.5-3b-lora","salsa_full_Qwen2.5-3B-Instruct","salsa_full_qwen2.5-0.5b-lora","salsa_full_Qwen2.5-0.5B-Instruct","grpo_base","grpo_sft","cascade_local_tauA","pipeline_always_local","cascade_oracle") { python thesis2\final\s1_run_arm.py --arm $a --split held_out_v4c }
```

Speculative arms — registry names `"a8_speculative"`, `"a10_speculative"`,
`"a12_smalllm"` — run through their own scripts, **with the v4c byte-identity
gate**:

```powershell
conda activate D:\condaenvs\thesis2gpu ; python thesis2\scripts\a8_speculative.py --split held_out_v4c ; python thesis2\scripts\a10_prefix_cache.py --split held_out_v4c ; python thesis2\scripts\a12_smalllm_drafter.py --split held_out_v4c
```

That is all 30 registry arms on `held_out_v4c`: 24 in the local loop, 3
speculative here, `"cascade_cloud_tauA"`, `"cascade_cloud_always"` and
`"ensemble"` below.

Each must print `BYTE-IDENTITY vs lora_greedy on held_out_v4c … identical -- PASS`.
A mismatch exits non-zero and no speed number from that run may be quoted.

The V5 cell grid — the same 36 cells and the same repeats, 12,152 rows:

```powershell
conda activate thesis ; $env:OLLAMA_MODELS="D:\LLM" ; python thesis2\final\s1_run_v5_cells.py --split held_out_v4c --kind local
conda activate D:\condaenvs\thesis2gpu ; python thesis2\final\s1_run_v5_cells.py --split held_out_v4c --kind cloud
```

Cloud cascades and the ensemble:

```powershell
conda activate D:\condaenvs\thesis2gpu ; foreach ($a in "cascade_cloud_tauA","cascade_cloud_always","ensemble") { python thesis2\final\s1_run_arm.py --arm $a --split held_out_v4c }
```

`ensemble` on v4c requires the three V5 cells it pairs to exist on v4c first, so
it must follow the v4c V5 grid. If the V5 audit or a cascade gate reports
`model_error` rows on v4c, re-run **only those rows** with `--retry-errors`
(S1-4c). It is the one re-run allowed on v4c. ~4.5 h and **7,595 billed calls** [Guessing on
time]. Most arms will report "reproducibility: NOT COMPARABLE on held_out_v4c"
with the reason — expected for a first measurement, not a failure.

> ### ⏸ PASTE TO ADVISOR AFTER STAGE 1 v4c
> Per-arm action match and row accuracy on both splits, side by side.

---

## STAGE 3 — measured latency and energy (before Stage 2, which reads it)

### S3-1 Idle baseline
`conda activate D:\condaenvs\thesis2gpu ; python thesis2\final\s3_energy.py --idle`
5 min [Certain]. GATE FAIL if mean idle utilisation > 2 %, printing the GPU
process list. Nothing else may use the GPU. The baseline expires after 2 hours —
re-run it if you pause.

### S3-2 Every non-cloud arm, both splits, V5 LoRA zero-shot FIRST

The V5 `qwen2.5-3b-lora/zero_shot` cell is measured first because V5 recorded
neither latency nor energy for it, and it is the anchor every other number is
read against.

```powershell
conda activate thesis ; $env:OLLAMA_MODELS="D:\LLM" ; foreach ($s in "held_out_v3","held_out_v4c") { python thesis2\final\s3_energy.py --v5-cell ollama:qwen2.5-3b-lora/zero_shot --split $s }
```

Then the remaining 30 local V5 Ollama cells, both splits, env `thesis`:

```powershell
conda activate thesis ; $env:OLLAMA_MODELS="D:\LLM" ; foreach ($s in "held_out_v3","held_out_v4c") { foreach ($c in "ollama:qwen2.5-3b-lora/few_shot","ollama:qwen2.5-3b-lora/cot","ollama:qwen2.5-3b-lora/role_based","ollama:qwen2.5-3b-lora/rag","ollama:qwen2.5:3b/zero_shot","ollama:qwen2.5:3b/few_shot","ollama:qwen2.5:3b/cot","ollama:qwen2.5:3b/role_based","ollama:qwen2.5:3b/rag","ollama:qwen2.5-0.5b-lora/zero_shot","ollama:qwen2.5-0.5b-lora/few_shot","ollama:qwen2.5-0.5b-lora/cot","ollama:qwen2.5-0.5b-lora/role_based","ollama:qwen2.5-0.5b-lora/rag","ollama:qwen2.5:0.5b/zero_shot","ollama:qwen2.5:0.5b/few_shot","ollama:qwen2.5:0.5b/cot","ollama:qwen2.5:0.5b/role_based","ollama:qwen2.5:0.5b/rag","ollama:llama3.2:3b/zero_shot","ollama:llama3.2:3b/few_shot","ollama:llama3.2:3b/cot","ollama:llama3.2:3b/role_based","ollama:llama3.2:3b/rag","ollama:deepseek-r1:7b/zero_shot","ollama:deepseek-r1:7b/few_shot","ollama:deepseek-r1:7b/cot","ollama:deepseek-r1:7b/role_based","ollama:deepseek-r1:7b/rag","classifier:distilbert/classifier") { python thesis2\final\s3_energy.py --v5-cell $c --split $s } }
```

Then every non-cloud registry arm, both splits, env `thesis2gpu`:

```powershell
conda activate D:\condaenvs\thesis2gpu ; foreach ($s in "held_out_v3","held_out_v4c") { foreach ($a in "flat_gpu","flat_cpu","factorized_fp16","factorized_int4","tinybert","minilm_probe","knn_minilm","rules_open","rules_firewalled","tfidf_logreg","bilstm","textcnn","mlpbow","flant5","lora_greedy","a8_speculative","a10_speculative","a12_smalllm","salsa_full_qwen2.5-3b-lora","salsa_full_Qwen2.5-3B-Instruct","salsa_full_qwen2.5-0.5b-lora","salsa_full_Qwen2.5-0.5B-Instruct","grpo_base","grpo_sft","cascade_local_tauA","pipeline_always_local","cascade_oracle") { python thesis2\final\s3_energy.py --arm $a --split $s } }
```

54 arm×split runs + 62 cell×split runs, each ≥ 120 s load + 60 s cool-down →
**~6 h** [Guessing]. Re-run `--idle` whenever more than 2 hours have passed.

That list is 27 arms: every registry arm except the two cloud cascades and
`"ensemble"`. **`ensemble` is excluded deliberately** — it performs no inference
of its own, it pairs three already-measured V5 cells offline, so it has no
latency or energy to measure. Its cost is the sum of its three components'
rows, which are measured above. Stated here rather than left as a gap.

**Cloud arms get latency only.** `cascade_cloud_tauA` and `cascade_cloud_always`
are not in the list above: `--arm` on a cloud arm records latency and writes
`joules_per_command = "not measurable (cloud)"`, and `--v5-cell` on an
`openai:` cell is refused outright with that reason. Their latency comes from
the Stage-1 per-command files.

check: `Import-Csv thesis2\out\final\unified_efficiency.csv | Format-Table arm,split,n_commands,wall_s,latency_p50_ms,joules_per_command,peak_vram_mb` → every row has `wall_s` ≥ 120, `n_commands` ≥ 217, and a real J/command for every non-cloud row.

Cascades are measured per command for real here; the modelled constants in
`p45_cascade_full.csv` are read by nothing in this run.

> ### ⏸ PASTE TO ADVISOR AFTER STAGE 3
> `unified_efficiency.csv`, with the V5 LoRA zero-shot row highlighted — its
> first measured latency and energy.

---

## STAGE 2 — every metric, every arm, both splits

`conda activate D:\condaenvs\thesis2gpu ; python thesis2\final\s2_metrics.py`
~1 min [Guessing]. Output `metrics_stage2.csv`.

check: no blank cell anywhere. Generative arms read `n/a - no class
probabilities` for ECE; non-speculative arms read `n/a - not speculative` for
token acceptance; cloud rows read `not measurable (cloud)` for J/command.

---

## STAGE 4 — significance, final table, figures

### S4-1 Significance
`conda activate D:\condaenvs\thesis2gpu ; python thesis2\final\s4_significance.py`
~3 min [Guessing]. Output `final_significance.csv`.

Primary family, 8 hypotheses declared 2026-09-22, Holm **within** these 8:
H1 v4c flat_gpu vs lora_greedy (action) · H2 v4c same pair (row) ·
H3 v3 lora_greedy vs flat_gpu (action) · H4 v3 factorized_fp16 vs flat_gpu
(action) · H5 v4c cascade_cloud_tauA vs flat_gpu (action) · H6 v4c grpo_base vs
lora_greedy (action) · H7 v3 salsa_full_qwen2.5-3b-lora vs lora_greedy (action) ·
H8 v4c textcnn vs flat_gpu (action).

H1, H2, H4 and H5 were measured once before the family was declared; the log
prints their old 26-hypothesis Holm values alongside. Every other pairwise
comparison goes in the exploratory family, Holm across all of them, reported and
never headlined.

### S4-2 Final table
`conda activate D:\condaenvs\thesis2gpu ; python thesis2\final\s4_metrics_final.py`
→ `metrics_final.csv`, the **only** file the final run's thesis tables may be
built from. check: zero empty cells.

### S4-3 Figures
`conda activate D:\condaenvs\thesis2gpu ; python thesis2\final\s4_figures.py`
→ six PNGs in `thesis2\figures\final\`. [Certain — runs]

> ### ⏸ PASTE TO ADVISOR AFTER STAGE 4
> The primary-family table and figures 1–3.

---

## STAGE 5 — Isaac demo, seen commands only

### S5-1 Parser smoke test, no Isaac
`conda activate D:\condaenvs\thesis2gpu ; python thesis2\demo\smoke_parsers.py`
~6 min [Guessing]. Cloud parsers need `--allow-cloud 20`. Already run on 14
parsers: `flat_gpu` 10/10 JSON and 10/10 agreement at 3.4 ms, `lora_greedy`
10/10 and 10/10 at 730 ms, `salsa_full_qwen2.5-3b-lora` 10/10 JSON and 7/10
agreement at 173 ms, `cascade_local_tauA` 10/10 at 4.6 ms / 6,860 MB. [Certain]

### S5-2 Isaac VRAM
Follow `thesis2\demo\vram_check.md` and fill `isaac_vram_fit.csv`. Every number
in that file is deliberately blank — it is a procedure, not a result.

### S5-3 Live demo
`conda activate D:\condaenvs\isaac6 ; python thesis2\demo\run_isaac_demo.py --parser v5_ollama`
then any parser marked **fits**. Every number is labelled `demo, not measurement`.

### S5-4 Held-out replay in Isaac, no new inference
`conda activate D:\condaenvs\isaac6 ; python thesis2\demo\run_isaac_demo.py --replay lora_greedy --split held_out_v4c`
Refuses to run unless the prediction file exists and is complete. Default
`--ids` = **0, 1, 2, 3, 9, 132, 141, 149, 159, 166** on both splits, from the
fixed rule in the script (first ids covering all 7 verbs, then in order) — not
chosen by correctness. [Certain]

---

## STAGE 6 — the report

`conda activate D:\condaenvs\thesis2gpu ; python thesis2\final\s6_report.py`
→ `FINAL_RUN_REPORT.md`, 7 sections, assembled from the output files only.

---

## Billed cloud calls

| step | calls |
|---|---|
| S1-4 V5 cloud cells, v3 | 5,425 |
| S1-5 cloud cascades, v3 | 2,170 |
| S1-6 V5 cloud cells + cloud cascades, v4c | 7,595 |
| S5-1 optional `--allow-cloud` | ≤ 20 |
| **total** | **≈ 15,210** |

All `gpt-4o-mini`. Decide the budget before S1-4. [Certain on the counts]

These counts are **first attempts**. Retries (S1-4c) add requests on top: at
most 5 per row, logged per row in `retries`, so the true number of requests
is the table plus the sum of that column. Any `--retry-errors` run adds its
re-run rows again. The cascades' 2,170 per split is an upper bound: a cascade calls
gpt-4o-mini only for a grid-typed row that escalates.

---

## Fixes applied 2026-09-25

1. **SALSA-full location labels A..N → D..Q** (collision with the row letters
   inside the location symbols). Row 0.3529 → 0.6588 on train_v2; full table
   above. Adopted as a design fix regardless of result, per instruction.
2. **A8/A10/A12** gained `--split`, `--limit` and `--outdir` (default
   `thesis2\out\final\`) and a `lora_greedy` byte-identity gate on every split.
   `thesis2\out\` committed files proved unchanged by the 5-command train_v2
   smoke test.
3. **Energy** now covers every non-cloud registry arm **and** every local V5
   Ollama cell, on both splits, V5 LoRA zero-shot first, via the new
   `--v5-cell` mode sharing one measurement loop with `--arm`.
4. **Reproducibility** extended from 4 named arms to every arm with saved
   predictions: 20 on v3, 13 on v4c, with the other 10 named and explained.
5. **Vocabulary** 144 vs 142 explained: identical token sets and order; the
   committed `FINAL_REPORT.md:813` figure omits `<pad>`/`<unk>`.
6. **Rung-3 selection** located (`ft/train_classifier.py:337-339`), GRPO proved
   to use the same rule, and the rung-3 checkpoint discrepancy recorded.
7. **This file** rewritten with every arm, cell and split named explicitly.

Earlier fixes (2026-09-24) remain: the rule-3 vocabulary guard corrected so a
legal `train_v2` run is not aborted by `top`, and rule-3 startup asserts added to
the GRPO and legacy scripts. `.vscode\extensions.json` is still outside
`thesis2\` and is yours to delete. [Certain]
