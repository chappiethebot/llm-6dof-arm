# Methods notes -- final run

2026-09-28. One paragraph per item, each with the file that supports it. Every
item was checked against the repository while writing; where no file supports an
item, the paragraph says so. Tags: [Certain] verified from the cited file,
[Likely] inference from it, [Guessing] not established.

## 1. held_out_v3 is confounded; held_out_v4c is the fair test

`held_out_v3` was measured three times (V3, V4c, V5), and the V5 legend -- the
prompt text every prompted arm uses -- was written after seeing V3 fail on it.
There is no training leakage, but the set is no longer blind for prompted arms.
`held_out_v4c` replaces the unseen row words with back/front, which appear in no
training set and no legend, so it is the fair test of generalisation.
Evidence: `audit/AUDIT_REPORT.md:74` (finding 24, BLOCKER), `audit/DEFENCE_NUMBERS.md:107`
("the legend was authored after seeing V3 fail"), `thesis2/final/s2_metrics.py`
(`UNSEEN` word groups per split). [Certain]

## 2. Pre-registration status of H1-H8

The primary family was declared on 2026-09-22. H1, H2, H4 and H5 had each been
measured once before that date, so they are not blind to a first result; H3, H6,
H7 and H8 were declared before any of their arms ran on held-out data.
Evidence: `thesis2/final/s4_significance.py` (`PRIMARY`, `PRE_MEASURED = {"H1",
"H2", "H4", "H5"}`, `DECLARED = "2026-09-22"`); `thesis2/out/final/final_significance.csv`.
[Certain]

## 3. checkpoint-87 is not what the strict earliest-plateau rule gives for the 3B

The codified rule (`ft/train_classifier.py:337-339`, `PLATEAU_TOL = 1`) applied to
rung 3's own dev curve for `qwen3b_lora` (hits/57 by epoch 55, 53, 54, 54, ...)
selects **epoch 1**. The project uses `checkpoint-87` = epoch 3, which is the rule's
answer for the **0.5B** (49, 52, 54, 54, ...); `ft/merge_and_export.py:40` applied
it to both models. Nothing measured changes -- the checkpoint is fixed, frozen and
dev-selected -- but the 3B checkpoint is not what the stated rule selects.
Evidence: `thesis2/final/RUN_ORDER.md` ("Provenance note on the rung-3 rule").
[Certain on the arithmetic]

## 4. Legacy arms use argmax epoch selection

The six legacy arms were retrained with their original recipes, which load the
**argmax** dev epoch, not the plateau epoch they print; that is what makes 217/217
reproduction possible. Example: `minilm_probe` selected epoch 100 (the first epoch
at the best 39 dev hits); the earliest-plateau rule (within 1 hit, >= 38) would
give epoch 80. Evidence: `thesis2/out/final/selection_minilm_probe.json` (`"rule":
"original argmax dev selection"`, history), `thesis2/final/legacy_recipes.py:71-72`.
[Certain]

## 5. SALSA-full letter relabel D..Q, decided on train_v2

Location options were relabelled with letters D..Q because A/B/C collided with the
row letters inside `grid_A1` etc. The relabel was adopted as a design fix and
checked on train_v2 (`s0_salsa_full.py --check`) on 2026-09-25 00:28-00:56; the
first held-out SALSA run is 2026-09-26 00:31. Evidence:
`thesis2/out/final/logs/s0_salsa_full_check_*_20260925_*.log`,
`thesis2/out/final/logs/s1_run_arm_salsa_full_*_held_out_v3_20260926_003122.log`,
`thesis2/final/RUN_ORDER.md` ("The SALSA-full letter relabel, and its result").
[Certain]

## 6. SALSA-full base-model arms use stock HF weights

`salsa_full_Qwen2.5-3B-Instruct` and `salsa_full_Qwen2.5-0.5B-Instruct` load the
unmodified Hugging Face weights. Revisions in the local HF cache (`refs/main`):
Qwen/Qwen2.5-3B-Instruct `aa8e72537993ba99e69dfaafa59ed015b17504d1`;
Qwen/Qwen2.5-0.5B-Instruct `7ae557604adf67be50417f59c2c2f167def9a775`.
Evidence: `C:\Users\USER\.cache\huggingface\hub\models--Qwen--Qwen2.5-3B-Instruct\refs\main`
and `...Qwen2.5-0.5B-Instruct\refs\main` (outside the repo). [Certain that these are
the cached revisions; Likely that they are the ones loaded, since the energy stage
ran with HF offline mode -- item 16]

## 7. GRPO: flat dev curves; the RL-on-SFT signal check

Both GRPO arms were selected at **epoch 1**: `grpo_base` scores 48/57 on dev at every
epoch except 47 at epoch 5; `grpo_sft` scores 56, 55, 57, then 56 for epochs 4-10.
The signal check before training `grpo_sft` found 30 of 230 train_v2 prompts
(13.0%) with mixed rewards across 8 samples at temperature 1.0 -- above the 10% gate.
Evidence: `thesis2/out/final/selection_grpo_base.json`,
`thesis2/out/final/selection_grpo_sft.json`, `thesis2/out/final/grpo_signal_check.json`
(`mixed_prompts: 30`, `mixed_fraction: 0.1304`). [Certain]

## 8. The ensemble

The disagreement ensemble is rebuilt offline from `results_v5cells_<split>.csv`
with the original rule: if the classifier and `qwen2.5-3b-lora/zero_shot` agree on
(action, target, location), emit the classifier's answer, otherwise
`gpt-4o-mini/rag`'s. gpt-4o-mini's 5 repeats are collapsed by majority vote over the
prediction; on V5's own files this gives the same 216/216 predictions as the
original's last-repeat rule. Command 0 is included, from the warm-up rows (217
commands). Reproducibility is outcome-gated, because the cloud input is sampled:
on held_out_v3, 215/216 predictions and 216/216 outcomes match the saved file
(command 132's repeats split 3-2 in V5 and 4-1 in the re-run, both wrong).
Evidence: `thesis2/final/arms.py` (`_ensemble`), `thesis2/final/s1_run_arm.py`
(`SAMPLED_INPUT`, `SAVED_EXCLUDES`), `thesis2/out/final/repro_check_held_out_v3.csv`.
[Certain]

## 9. Cloud cells and cascades: gating and retries

For the 5 gpt-4o-mini cells, raw agreement with V5 is reported and not gated (at
R = 5 the API does not repeat its text; `cot` agrees on 10/217 raw strings yet
matches V5's **per-row** accuracy, 0.7177 vs 0.7177); the gate is zero
`model_error` rows and, on held_out_v3, per-row action match within 3 points of
V5's. Every gpt-4o-mini call retries on connection
errors, timeouts, 429 and 5xx, up to 6 attempts with 2-4-8-16-32 s backoff and the
SDK's own retries off, and logs the retry count per row. The held_out_v4c cloud
rows carry that count (all 5,425 rows: 0 retries, 0 `model_error`); the held_out_v3
cloud cells were re-run on 2026-09-26 before the retry layer existed, so their
`retries` column is empty. Evidence: `thesis2/final/s1_check_v5cells.py`
(docstring), `thesis2/final/common.py` (`with_retries`),
`thesis2/out/final/results_v5cells_held_out_v4c.csv` (`retries`),
`thesis2/out/final/force_rerun_log.csv`. [Certain]

## 10. Energy formula and protocol

J/command = (mean load W - idle median W) x wall s / N. Protocol: an idle baseline
under 2 hours old (300 s, median), 3 warm-up commands not counted, a load loop of
at least 120 s, then a 60 s cool-down. GPU watts only (nvidia-smi); CPU arms
additionally report process CPU ms per command; CPU watts are not measured. Cloud
energy is not measurable. A J/command <= 0 means the load sat inside the idle
noise and is shown as "~0 (below GPU floor)"; the measured value is kept. VRAM
above idle = peak VRAM minus the median idle `memory.used` of the baseline in force;
all 85 efficiency rows reproduce their recorded idle watts from the matched log.
Evidence: `thesis2/final/s3_energy.py` (docstring, `measure_loop`),
`thesis2/out/final/unified_efficiency.csv`, `thesis2/out/final/power_logs/idle_*.csv`,
`thesis2/final/common.py` (`j_display`, `idle_for`, `vram_above_idle`). [Certain]

## 11. Efficiency scope

The main efficiency table is **held_out_v3**. held_out_v4c efficiency is reported
as a supplement for the speculative arms (A8, A10, A12) and their verifier only,
although most local arms were also measured on v4c. The V5 cells' energy was
measured on held_out_v3 only (31 local cells; the 5 cloud cells are not
measurable). Evidence: `thesis2/out/final/unified_efficiency.csv` (every `v5:` row
is held_out_v3), `thesis2/out/final/DEFENCE_NUMBERS_rev3.md` sections 3-4.
[Certain]

## 12. Runtime difference whenever V5 is compared with A8/A10

The V5 LoRA cells run in **Ollama at Q4_K_M**; `lora_greedy`, A8, A10 and A12 run
the same adapter in **transformers FP16** (merged). Latency, energy and even raw
text therefore differ for runtime reasons, not method reasons, and any V5-vs-A8/A10
comparison must say so. Evidence: `thesis2/scripts/a8_speculative.py` (section 4,
"QUANTISATION DIAGNOSTIC -- transformers fp16 vs the recorded ollama Q4_K_M rows"),
`thesis2/out/final/unified_efficiency.csv` (`ollama_version`, `model_digest`).
[Certain]

## 13. 12,152 rows, effective n = 217

The V5 cell grid is 31 local cells x 217 commands x 1 repeat + 5 cloud cells x 217
x 5 = 6,727 + 5,425 = 12,152 rows per split. Repeats collapse to one outcome per
command, so the effective n is 217 commands (216 for a local V5 cell under A9's
warm-up rule). Evidence: `thesis2/final/common.py` (`v5_census`),
`thesis2/out/final/results_v5cells_held_out_v3.csv`. [Certain]

## 14. MuJoCo end-to-end was not re-run for the new arms

The earlier MuJoCo end-to-end run executed six arms with **0 execution failures**
and 0 invalid symbols in every arm, and end-to-end success equals action match
(for example the flat classifier: 0.7465 both). With execution never the failure
point, end-to-end success for a new arm is its string match, so the simulator
was not re-run. Evidence: `thesis2/out/a3_mujoco_e2e.csv`. [Certain on the earlier
run; that it would hold for the new arms is Likely -- they emit the same symbol set]

## 15. Bugs caught and fixed during the final run

- **SALSA letter collision** -- option letters A/B/C collided with the row letters in
  location symbols; relabelled D..Q (item 5). `thesis2/final/RUN_ORDER.md`. [Certain]
- **Void few_shot/rag cells** -- an exemplar-loader bug (a tuple, the V1 seen file,
  and no per-command retrieval for rag) produced `prompt_error` with no model call
  in 12 of 31 local cells; fixed through V5's own loader and re-run with
  `--replace-cells`. `thesis2/final/s1_run_v5_cells.py` (docstring),
  `thesis2/out/final/force_rerun_log.csv`. [Certain]
- **0/0 reproducibility check** -- an int/str join-key mismatch compared zero
  commands and passed; fixed, and zero comparisons now FAIL.
  `thesis2/final/s1_repro_check.py` (docstring). [Certain]
- **Line-vs-row counting** -- completeness counted physical lines, so outputs with
  newlines looked complete (986 "rows" for 217); now parsed CSV rows.
  `thesis2/final/common.py` (`pred_rows_of`). [Certain]
- **The environment mix-up** -- what the logs show, and only that. On 2026-09-26
  step S1-4 (V5 cloud cells, held_out_v3) was started twice by the autorun:
  - `thesis2/out/final/logs/autorun_phaseA_20260926_152910.log`: "AUTORUN STOPPED:
    S1-4 V5 cloud cells, held_out_v3 (5,425 billed calls) (exit code 1, after
    0.1 min)". The matching `s1_run_v5_cells_held_out_v3_cloud_20260926_152917.log`
    records `python: D:\condaenvs\thesis2gpu\python.exe` and ends after the cell
    plan ("this run will execute 5 cell(s)"); it contains no error text.
  - `autorun_phaseA_20260926_153057.log`: the same step, "AUTORUN STOPPED ... (exit
    code 1, after 116.7 min)". The matching
    `s1_run_v5_cells_held_out_v3_cloud_20260926_153101.log` records
    `python: C:\Users\USER\.conda\envs\thesis\python.exe`, wrote all 12,152 rows,
    and failed its audit; the audit's own log
    (`s1_check_v5cells_held_out_v3_20260926_172739.log`) shows `model_error` rows in
    4 gpt-4o-mini cells -- cot 85, few_shot 1,050, role_based 205, zero_shot 1,085
    (rag none) -- the 4 cells re-run with `--replace-cells` at 16:53Z
    (`force_rerun_log.csv`).
  So the step ran first under `thesis2gpu` and then under `thesis`, and the second
  run's cloud calls failed in part. The files do not state why the first run
  exited or why the calls failed. [Certain on what the files show; nothing beyond it]

  **Cause, from the console output at the time** (reported by the author
  2026-09-28; not preserved in repo files after the rows were replaced):
  - attempt 1: `ModuleNotFoundError: ollama` -- the wrong environment. Consistent
    with the code: even a `--kind cloud` run calls `llm.clients.all_clients()`,
    which constructs every `OllamaClient` and so runs `import ollama`
    (`llm/clients.py:127`, called from `:172`), and `thesis2gpu` has no `ollama`
    package (the same error recurred in the 2026-09-28 parser smoke test).
    [Certain that the code path fails this way; the console text is as reported]
  - attempt 2: `APIConnectionError` on all failed gpt-4o-mini rows -- the
    network. After it, retry with backoff was added (`common.with_retries`,
    commit `bababbd`; item 9). [As reported; consistent with `net_check.py`
    having been written 16:52Z that day, just before the re-run]
- **Connection drops** -- 4 gpt-4o-mini cells on held_out_v3 lost rows; re-run with
  `--replace-cells`, then a retry layer was added for every cloud call (item 9).
  `thesis2/out/final/force_rerun_log.csv`. [Certain]
- **Ensemble lookup** -- the ensemble refused to run although its V5 inputs were
  complete; rebuilt as an offline pairing (item 8). `thesis2/final/arms.py`. [Certain]
- **v4c checker comparisons** -- the V5-cells audit compared v4c outputs with V5's
  v3 numbers; the comparison rules now run on held_out_v3 only.
  `thesis2/final/s1_check_v5cells.py` (docstring). [Certain]
- **Offline model loading** -- intermittent HF TLS failures; the energy stage runs
  with `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.
  `thesis2/final/autorun.ps1:17`. [Certain]
- **Speculative energy wiring** -- A8/A10/A12 were refused by `s3_energy.py`; they now
  measure through the shared protocol inside their own scripts, after the
  byte-identity gate. A10 moved from 9.45 to 9.54 J/command: the old path used a
  hard-coded idle of 12.12 W, no warm-up or cool-down, and excluded the drafter and
  tokenisation; (91.4 - 12.12) x 129.2 / 1085 = 9.44 vs (84.72 - 10.61) x 139.7 / 1085
  = 9.54. `thesis2/out/FINAL_REPORT.md:523`, `thesis2/out/final/unified_efficiency.csv`,
  `thesis2/final/s3_energy.py` (`mode_speculative`). [Certain on the arithmetic]

## 16. HF offline mode for the energy stage

The energy stage ran with `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`, so every
model loaded from the local cache and no weights could change mid-run.
Evidence: `thesis2/final/autorun.ps1:17`. [Certain that the script sets it; Likely
that every energy run used this script]

## 17. H6: what GRPO changed on held_out_v4c -- bias removal, not understanding

H6 (grpo_base vs lora_greedy, v4c action match) is CLAIMABLE, but the row
diagnosis says what kind of improvement it is. On the 48 grid commands whose row
word is unseen (back/front), from `metrics_final.csv` (`row_by_word_group`,
`row_signature`; the signature counts cover all 108 grid commands):

| arm | unseen-word rows correct | A<->C mirror | default to B |
|---|---|---|---|
| lora_greedy | 0/48 | 6 | 66 |
| grpo_base | 23/48 | 24 | 25 |
| flat_gpu | 24/48 | 24 | 24 |

`lora_greedy` answers row B for almost every grid command whose row it cannot
read: a systematic error. After RL, `grpo_base` chooses a far/near row instead,
and is right about as often as it mirrors -- the rate of picking A or C at random.
RL converted a systematic default-to-B error into a chance-level A/C error: bias
removal, not an understanding of back/front. Neither `grpo_base` nor `flat_gpu`
maps back/front; `flat_gpu` shows the same chance-level pattern. [Certain on the
counts; the interpretation is Likely]

## 18. Cloud aggregation: majority vote, not V5's per-row averages

Every cloud number in the thesis tables collapses gpt-4o-mini's 5 repeats per
command by **majority vote** (A9's rule: correct iff more than half the repeats are,
a tie is wrong) -- `metrics_final.csv` for the cloud cascades,
`v5cells_both_splits.csv` for the V5 cloud cells. V5's own reports averaged over
**rows** (every repeat scored separately). The bases differ: `cot` on held_out_v3 is
0.7327 by majority vote and 0.7177 per-row; they are not directly comparable.
`DEFENCE_NUMBERS_rev3.md` and `FINAL_RUN_REPORT.md` use majority vote for every
cloud cell; the only per-row cloud figures in the repository's current documents
are the V5-cells audit's gate values, labelled per-row in `RUN_ORDER.md` S1-4 and
in item 9 above. [Certain]

## 19. What stopped the runaway qwen2.5-0.5b-lora/cot generations

The repository's code sets no generation limit: `llm/clients.py:144-148` calls
`ollama.generate(model, prompt, options={"temperature": ..., "num_ctx": 4096})`
with no `num_predict` and no timeout, and `bench/run_cell.py` passes nothing else.
The 32 runaway rows returned normally (stage `extract_failed`, raw text present),
so no client timeout fired. V5's own token accounting for the same cell
(`bench/results_v5_lora_tokens.csv`, `tokens_out`, re-tokenised locally with the
Qwen2.5 vocabulary, `token_source = qwen2.5_vocab_local`) shows **28 of the 32 at
exactly 40,959-40,960 tokens = 10 x num_ctx (4,096)**; the other 4 (27,316, 27,318,
30,726, 102,379) are re-tokenisations of highly repetitive text, not server
counts. [Certain on the counts] The stop is therefore a server-side generation cap
of 10 x num_ctx applied by Ollama when `num_predict` is unset [Likely]. That cap
lives in the Ollama server, not in this repository, so no file and line in the
repo implements it -- it cannot be confirmed from the repo's code alone. At the
0.5B's speed that budget is ~113 s, which is why the stops cluster there.

## 20. Citation check: "DCCD"

A web search on 2026-09-28 found a paper that uses the acronym for structured
generation with a draft, which fits the context this project cites it in:

> Avinash Reddy, Thayne T. Walker, James S. Ide, Amrit Singh Bedi. **"The Hidden
> Cost of Structured Generation in LLMs: Draft-Conditioned Constrained Decoding."**
> arXiv:2603.03305 [cs.CL], v1 8 Feb 2026, v2 27 Jun 2026.
> https://doi.org/10.48550/arXiv.2603.03305

Details are as shown on the arXiv abstract page (https://arxiv.org/abs/2603.03305);
the page lists v1 on 8 Feb 2026 although the identifier's `2603` prefix would
normally mean March 2026 -- recorded as shown, not reconciled. A different paper
also uses the acronym: "Dual-Confidence Contrastive Decoding for
Retrieval-Augmented Generation", arXiv:2607.00570. The earlier report
(`thesis2/out/FINAL_REPORT.md:595`) recorded DCCD as not identified. **Confirm that
2603.03305 is the work the thesis means before citing it** -- the match is on
acronym and topic only. [Certain that the paper exists as described; Likely that
it is the intended DCCD]
