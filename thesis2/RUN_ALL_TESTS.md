# RUN_ALL_TESTS v2 - full re-run on V5 + held-out, new arms, energy, Isaac

Save as: `C:\Users\USER\llm-6dof-arm\thesis2\RUN_ALL_TESTS.md`
Replaces the v1 file from 22 Sep (which was only ever sent in chat, never committed).

Two parts. **Part 1 is for the author.** **Part 2 goes to a fresh Claude Code session** at `C:\Users\USER\llm-6dof-arm`.

Expected duration [Guessing]: 1-2 days of machine time, mostly RL training and the V5 cells on v4c. The prompt is written to be resumable.

---

# PART 1 - What the author does

## Before

1. Pull and check the tree:
   ```powershell
   cd C:\Users\USER\llm-6dof-arm; git pull; git status -sb; git log --oneline -3
   ```
2. Ollama running, models present (`OLLAMA_MODELS=D:\LLM`):
   ```powershell
   ollama list
   ```
3. OpenAI key works (new key; `.env` not on the demo USB). Cloud cost for everything here [Guessing] under 3 USD.
4. **Close every GPU application** (browser, Discord, Steam, Isaac, anything with hardware acceleration). Then:
   ```powershell
   nvidia-smi
   ```
   Utilisation must read 0 %. Last time the "idle" baseline had 20 % mean utilisation - that must not happen again.
5. Power plan: Windows "High performance", sleep disabled, screen saver off. Do not use the machine while Stages 1 and 3 run.

## During

- Claude Code stops at every **STOP**. Read the report before saying continue.
- If anything disagrees with saved predictions, it stops. Do not let it "fix" by editing earlier outputs.

## Isaac (Stage 5)

Isaac needs the GUI, so you run it with the launch commands Claude Code gives you, using the 10 seen commands it lists.

## After

```powershell
git add thesis2; git commit -m "Full re-run v3+v4c, new arms (TextCNN, MLP-BoW, SALSA-full, GRPO), unified efficiency, Isaac parsers"; git push origin HEAD; git log --oneline -3
```

Then every thesis table comes from `thesis2/out/final/metrics_final.csv` only.

---

# PART 2 - Claude Code prompt

You are working on my MSc thesis repo at `C:\Users\USER\llm-6dof-arm`. Read this whole brief before doing anything. Also read `thesis2/out/FINAL_REPORT.md` fully - it defines every existing arm, every withdrawn number and the significance protocol.

## Rules (not negotiable)

1. **No file in the repository outside `thesis2/` may be created, modified, moved or deleted** (`sim/`, `bench/`, `llm/`, `asr/`, `ft/`, `audit/`, `data/`, `docs/` are read-only). New code goes in `thesis2/final/` (scripts) and `thesis2/demo/`; outputs in `thesis2/out/final/`.
2. **Held-out data (`held_out_v3`, `held_out_v4c`) is used for inference and scoring only.** Never for training, checkpoint selection, threshold selection, prompt wording, label/letter mapping or any hyperparameter. All training uses `train_v2`; all selection uses `dev_v2`.
3. **Every new arm must be fully trained, selected on dev_v2 and frozen before it sees any held-out command.** Record the SHA-256 of each frozen checkpoint in `thesis2/out/final/frozen_checkpoints.csv` before Stage 1.
4. **Never add held-out vocabulary** (`back`, `front`, `upper`, `bottom`, `top`, `lower`) to any legend, lexicon, label description, prompt or tokenizer vocabulary. New arms build vocabulary from `train_v2` only.
5. **No LLM-generated training phrasings.** Training data is `train_v2` as it is.
6. **Each arm runs once on `held_out_v4c`.** No iteration on v4c. If an arm crashes mid-run, fix the crash, delete its partial v4c output, and re-run from scratch - report that it happened.
7. Hyperparameters for new arms are the ones written in this file. **No sweeps.**
8. Withdrawn arms stay withdrawn: A1.7 field-level FLMD, every SALSA *digits* mapping.
9. Nothing is done without terminal output. Tag every claim `[Certain]` / `[Likely]` / `[Guessing]`.
10. PowerShell: `;` not `&&`, no multi-line `python -c`, script files only, `Set-Content -Encoding ascii`.
11. Envs: `thesis2gpu` at `D:\condaenvs\thesis2gpu` (CUDA torch 2.14, transformers 5.17); `thesis` for the Ollama client and MuJoCo; `isaac6` for Isaac. **Do not upgrade or downgrade packages in `thesis2gpu` or `thesis`.** If a new package is needed and conflicts, STOP and ask before creating a new env.
12. HuggingFace TLS: `CERTIFICATE_VERIFY_FAILED` is a known intermittent proxy issue. **Never disable certificate verification.** Retry; if it persists, STOP.
13. **Resumable:** every (arm, dataset) writes its own prediction file. On restart, skip any (arm, dataset) whose output file exists and is complete (217 rows). Log what was skipped.

## Datasets

| split | n | use |
|---|---|---|
| `train_v2` | 230 | training only |
| `dev_v2` | 57 | selection only |
| `held_out_v3` | 217 | inference only (108 grid + 24 runtime + 36 container + 49 no-location) |
| `held_out_v4c` | 217 | inference only - the fair generalisation test for prompted arms |

"V5" = the V5 legend prompt and constrained decoding, exactly as in the V5 manifests. For arms that read no prompt, V5 changes nothing; they simply run on both sets.

MASSIVE is **not** re-run. Existing MASSIVE numbers in `metrics_all_v2.csv` stand.

---

## STAGE 0 - Build and freeze the new arms (train/dev only)

Read `thesis2/scripts/a1_arms.py` (BiLSTM) and the flat DistilBERT eval path first. New arms must use **the same label space, the same field masks, the same JSON emit function and the same scorer (`llm/score.py`)** as the existing arms.

### 0.1 TextCNN (new)
- Tokeniser and vocabulary: exactly the BiLSTM's (train_v2 only; OOV -> `<unk>`).
- Embedding 128, trained from scratch. Conv1d filter widths 3/4/5, 100 filters each, ReLU, max-over-time pool, dropout 0.5.
- Three heads: verb (7), target (3), location (14). Loss = sum of three cross-entropies (masked as the flat classifier does).
- Adam lr 1e-3, batch 32, 30 epochs, seed 20260827.
- Selection: argmax dev_v2 action match; **persist the selected epoch number** to `thesis2/out/final/selection_textcnn.json`.

### 0.2 MLP over bag-of-words (new)
- Same tokeniser/vocabulary. Input = binary bag-of-words over the train_v2 vocabulary.
- Linear -> 256 -> ReLU -> dropout 0.3 -> three heads as above.
- Adam lr 1e-3, batch 32, 30 epochs, seed 20260827. Same selection and persistence rule (`selection_mlpbow.json`).

### 0.3 SALSA full-command parser (new)
The existing SALSA scorer only scores the row. This arm scores **every field** with single-pass logit classification, so it produces a full command.

- For each field (verb, then target, then location), one forward pass: the V5 prompt with the command, then a fixed question listing that field's options as letters in schema order (verb A-G, target A-C, location A-N), ending with `Answer:`.
- Read the logits of the letter tokens at the position after `Answer:` and take the argmax **over legal options only**, applying the field masks (stop/reject -> no location; containers only with pick_place). Emit JSON with the shared emit function.
- **Token check before anything else (this is the B-4 lesson):** print the token ids of `' A'` ... `' N'` in the Qwen tokenizer. Every one must be a single token. Then on 5 `train_v2` commands, confirm the model's greedy next token after `Answer:` is a letter token. If either check fails, **STOP**.
- Share the prompt prefix KV cache across the three passes.
- Models: qwen2.5-3b-lora (merged, FP16), Qwen2.5-3B-Instruct base, qwen2.5-0.5b-lora, Qwen2.5-0.5B-Instruct base.
- Prompt wording is fixed after checking dev_v2 only. Record dev_v2 action match per model. No edits after the first held-out run.

### 0.4 GRPO reinforcement-learning fine-tune (new)
Library: TRL `GRPOTrainer`. First check TRL installs and imports against transformers 5.17 without changing torch/transformers. If not, **STOP** and report.

- Prompt: the V5 prompt, identical to inference.
- Reward: 1.0 if the generated action exactly matches the train_v2 label under `llm/score.py`, else 0.0 (invalid JSON = 0.0). Nothing else.
- LoRA r=16, alpha=32, dropout 0.05, same target modules as rung 3 (q/k/v/o/gate/up/down; lm_head excluded).
- Group size 8, sampling temperature 1.0, max new tokens 64, KL beta 0.04, lr 2e-5, cosine, up to 10 epochs over train_v2, seed 20260827.
- Selection: greedy dev_v2 action match after each epoch, **earliest-plateau rule as in rung 3**; persist the selected epoch.

Two arms:
- **GRPO-from-base** (Qwen2.5-3B-Instruct + fresh LoRA). This is the clean "RL instead of SFT" comparison.
- **GRPO-on-SFT** (start from the rung-3 3B adapter, `checkpoint-87`). **Signal check first:** sample 8 completions per train_v2 prompt at temperature 1.0 with the SFT model. If fewer than 10 % of prompts have mixed rewards (some 0, some 1), GRPO has no gradient - **skip this arm and report the fraction.** Do not lower the bar.

If 3B does not fit in 16 GB for GRPO, **STOP** and report peak memory. Do not silently fall back to 0.5B.

Write the frozen checkpoint hashes. **STOP after Stage 0** and report: dev_v2 action match for every new arm, selected epochs, SALSA token checks, the GRPO signal-check fraction and training curves.

---

## STAGE 1 - Accuracy on both held-out sets

Run every arm below on **`held_out_v3` and `held_out_v4c`**. Save per-command predictions as `thesis2/out/final/pred_<arm>_<split>.csv` (command_id, raw output, parsed JSON, correct, per-field correctness).

### Arm list

**A. Encoder classifiers**
- A1 flat DistilBERT (GPU) · A2 flat DistilBERT (CPU)
- A3 factorized DistilBERT FP16 · A4 factorized DistilBERT INT4 (simulated)
- A5 TinyBERT-4 · A6 MiniLM frozen + linear probe · A7 kNN over MiniLM

**B. Classical / small non-transformer**
- B1 rules, open (its grammar lists back/front - report, but mark "vocabulary-unfair")
- B2 rules, firewalled · B3 TF-IDF + logreg · B4 BiLSTM
- B5 **TextCNN (new)** · B6 **MLP-BoW (new)**

**C. Seq2seq**
- C1 Flan-T5-base

**D. Local LLMs**
- D1 **every local V5 cell** (each model x strategy x runtime in the V5 manifests; read `bench/` manifests and `bench/results_v5_*.csv` to get the exact list - do not invent or drop cells). Ollama, greedy, same `num_ctx` and prompts as V5.
- D2 qwen2.5-3b-lora plain greedy, transformers FP16 merged
- D3 A8 speculative · D4 A10 speculative + prefix cache · D5 A12 0.5B drafter (gamma = 4)
- D6 **SALSA full-command (new)**, four models
- D7 **GRPO-from-base (new)** · D8 **GRPO-on-SFT (new, only if the signal check passed)**

**E. Cloud**
- E1 **every cloud V5 cell** (gpt-4o-mini x strategies incl. RAG), 5 repeats, majority vote per command, exactly as V5.

**F. Composite**
- F1 cascade tau_A = 0.50, row -> qwen2.5-3b-lora SALSA letters (pre-registered)
- F2 always-escalate pipeline tau = 1.0 -> qwen2.5-3b-lora (label it "pipeline, not routing")
- F3 cascade row -> gpt-4o-mini, tau_A = 0.50 and tau = 1.0
- F4 cascade row -> oracle (accuracy only; ceiling)
- F5 ensemble over disagreement

### Reproducibility check (held_out_v3)
Every arm that already has saved v3 predictions must reproduce them.
- Deterministic local arms: 217/217 identical, or **STOP** and list the differing commands.
- Ollama cells: report agreement vs `bench/results_v5_*`; small GGUF differences are possible - report, never hide.
- Cloud cells: report agreement; gpt-4o-mini is known to vary across repeats.

A8/A10/A12 on v4c must be byte-identical to D2 on v4c (lossless check). Report it.

---

## STAGE 2 - Metrics per (arm, split)

Compute from the prediction files. The headline set, one or two per category:

| category | metric |
|---|---|
| correctness | action match (whole JSON); row accuracy (n = 108 grid) |
| confidence | ECE, for arms that expose probabilities (classifiers, SALSA, cascades) |
| efficiency | latency p50; J/command (Stage 3) |
| speculative decoding | token acceptance; first-rejection field distribution |
| diagnosis | row accuracy split by row word (seen: middle/center; unseen: upper/bottom on v3, back/front on v4c; mixed); row confusion signature (count of A<->C mirrors, defaults to B) |
| statistics | clustered bootstrap 95 % CI; exact McNemar; Holm p |

Also keep (table columns, not headline): per-field accuracy (verb, target, location, column), schema validity, p95/p99, peak VRAM, size on disk (FINAL_REPORT one definition), CPU ms/command for CPU arms.

---

## STAGE 3 - Unified efficiency (latency + GPU wattage)

Identical protocol for every local arm, **on both splits**:

1. `nvidia-smi` idle check (utilisation 0 %). Log `nvidia-smi --query-gpu=power.draw,utilization.gpu,memory.used,temperature.gpu --format=csv -l 1` for 300 s with no workload -> idle **median**. If mean idle utilisation > 2 %, **STOP** and report the process list.
2. Repeat the idle baseline every 2 hours; report drift.
3. Per arm: warm up with 3 commands (not counted, and state this). Loop the full split until >= 120 s wall clock. Record N and duration.
4. Latency: per-command wall clock including tokenisation, drafter, retrieval and network. p50/p95/p99.
5. Energy: (mean load W - idle median W) x wall s / N. GPU only - state it.
6. CPU arms: GPU delta ~0 does not capture CPU cost. Report **process CPU ms per command** (psutil) instead. CPU watts are not measured on this machine - state it.
7. Cloud: latency only; energy not measurable - state it.
8. One arm at a time, 60 s cool-down between arms, log max GPU temperature.
9. Ollama: `OLLAMA_MODELS=D:\LLM`, same `num_ctx` as V5; record Ollama version and model digest. Unload other models first.
10. The cascades F1-F3 are measured **for real**, per command. The old modelled constants are not used anywhere.

Outputs: `thesis2/out/final/unified_efficiency.csv`, raw logs in `thesis2/out/final/power_logs/`.

**STOP after Stages 1-3** and report the full accuracy table (both splits), the reproducibility table and the efficiency table.

---

## STAGE 4 - Significance, figures, final table

### Families (declared on 22 Sep 2026, before this run)

**Primary family (8 hypotheses, Holm within these 8):**

| id | split | comparison | metric |
|---|---|---|---|
| H1 | v4c | flat DistilBERT vs qwen2.5-3b-lora V5 zero-shot (transformers = A10 output) | action |
| H2 | v4c | same pair | row |
| H3 | v3 | qwen2.5-3b-lora V5 zero-shot (transformers) vs flat DistilBERT | action |
| H4 | v3 | factorized vs flat DistilBERT | action |
| H5 | v4c | cascade row -> gpt-4o-mini tau_A = 0.50 vs flat DistilBERT | action |
| H6 | v4c | GRPO-from-base vs qwen2.5-3b-lora SFT V5 zero-shot (transformers) | action |
| H7 | v3 | SALSA full-command (qwen2.5-3b-lora) vs qwen2.5-3b-lora generative (transformers) | action |
| H8 | v4c | TextCNN vs flat DistilBERT | action |

Transparency note to carry into the report: H1, H2, H4 and H5 were already measured once before this family was declared. Report the old 26-hypothesis Holm values alongside.

**Exploratory family:** every other pairwise comparison you report, Holm across all of them. Exploratory results are reported, never headlined.

Protocol unchanged: clustered bootstrap over command_id, 2,000 resamples, seed 20260827; exact McNemar; claimable only if CI excludes 0 AND p_holm < 0.05. Output `thesis2/out/final/final_significance.csv` with a `family` column.

### Final table
`thesis2/out/final/metrics_final.csv`: one row per (arm, split) with every Stage 2 metric, Stage 3 efficiency, status (valid / withdrawn / not measurable), and notes.

### Figures -> `thesis2/figures/final/`
1. Action match vs latency p50 (log x), both splits side by side, CI bars, Pareto frontier.
2. Action match vs J/command (log x); cloud shown separately as "energy not measurable".
3. Row accuracy by row-word group (seen / unseen / mixed), every arm, v3 and v4c - **the "why it fails" figure**.
4. Row confusion signature per arm (A<->C mirror vs default-to-B).
5. A10 latency breakdown (drafter / verification / decode) and first-rejection field, v3 vs v4c.
6. Action match vs peak VRAM.

Mark claimable differences on every figure.

---

## STAGE 5 - Isaac demo integration (seen commands only)

Do not modify `sim/`, `llm/` or `asr/`. Build `thesis2/demo/run_isaac_demo.py`: imports the existing demo unchanged and injects a parser selected by `--parser <name>`. Every parser implements `parse(text) -> action_json` through a registry in `thesis2/demo/parsers.py`.

Register every arm that can produce a full command:
- CPU / small: flat, factorized FP16, factorized INT4, tinybert, probe, knn, rules_firewalled, tfidf, bilstm, textcnn, mlpbow, flant5
- Ollama: every local V5 cell (`v5_ollama_<model>_<strategy>`); `v5_ollama` = the current demo behaviour, reproduced exactly
- transformers GPU: lora_greedy, a10_speculative, a12_smalllm, salsa_full_<model>, grpo_base, grpo_sft (if it exists)
- cloud: `cloud_<strategy>`, cascade_cloud (need network + key)
- composite: cascade_local, ensemble

**Not registered:** oracle cascade (needs the true answer), SALSA row-only (cannot produce a full command), withdrawn arms.

Steps:
1. Read `sim/isaac_bridge.py`, `sim/isaac_backend.py` and the demo launcher to find where text -> JSON is called.
2. Pick 10 commands from `data/seen_v2.jsonl` covering all 7 verbs, grid cells, containers and runtime symbols. **No held-out commands.** Write them to `thesis2/demo/demo_commands.txt`.
3. Headless smoke test: every registered parser on those 10 commands **without Isaac**; confirm valid JSON; log latency.
4. VRAM: measure each GPU parser's peak VRAM. Measure Isaac's own VRAM with the demo scene loaded (ask me to launch it if needed). Mark each GPU parser "fits" or "does not fit" on 16 GB. **Never degrade silently** to another parser.
5. **STOP and give me the exact launch command for each parser**, grouped fits / does not fit. I run Isaac myself.

Isaac numbers are demo numbers, not measurements. Label them so everywhere.

---

## STAGE 6 - Report

Write `thesis2/out/final/FINAL_RUN_REPORT.md`:
- what ran, what was skipped and why (including GRPO-on-SFT if the signal check failed)
- accuracy on both splits for every arm, with the diagnosis metrics
- reproducibility agreement per arm
- the unified efficiency table, incl. the V5 LoRA's first latency/energy figures
- primary-family results first, then exploratory
- Isaac integration status and launch commands
- everything not measurable and why

Tag every claim. Do not soften anything. Do not describe any result as novel.
