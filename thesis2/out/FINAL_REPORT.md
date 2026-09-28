# Factorized per-factor precision cascade — final report (revised after Part B review)

> **2026-09-28 — this file is now the historical record.** Thesis tables now come
> only from `thesis2/out/final/metrics_final.csv` (the final run; see
> `thesis2/out/final/FINAL_RUN_REPORT.md` and `DEFENCE_NUMBERS_rev3.md`). The
> source-of-record note below applied to the earlier run and is kept unchanged
> as history.

> **Source of record for thesis tables: `thesis2/out/metrics_all_v2.csv` is the only file thesis tables may be built from.** It carries every row of `metrics_all.csv` (left untouched) plus `status` (valid / withdrawn / modelled) and `reason` columns; withdrawn rows have their values blanked. 50 rows: 26 valid, 16 modelled (cascade latency/energy columns are constants; accuracy columns measured), 8 withdrawn (A1.7, all SALSA digits). It now carries **56 rows**: the original 50 plus the plain-greedy LoRA baseline, A8, A10 and A12 on `held_out_v3`, and the two `held_out_v4c` cells §3 leads with (added by `a15_metrics_v2_append.py`, re-review S-E). Per-command detail still lives in `a8_speculative_summary.json`, `a10_prefix_cache_summary.json`, `a11_rejection_positions.csv`, `a12_smalllm_drafter_summary.json` and `a13_v4c_*`; the phase summaries carry a SUPERSEDED banner and are not table sources.

Sessions 2026-09-21/22; revision 2026-09-22 after the independent Part B review
(`C:\Users\USER\thesis2_review.md`). **This is a fix pass only: no new arm was
run, no prompt or threshold was changed after seeing held-out results.** Every
number below comes from files that already existed before the review, plus
statistics recomputed on those files. Script edits in this pass are text-only
(prints, docstrings, a hand-typed size table) and are listed in §15.

All repository work is under `thesis2/`. The two files `git status` shows
modified outside it (`audit/AUDIT_REPORT.md`, `audit/DEFENCE_NUMBERS.md`) carry
mtime 17-09-2026 22:49, before this work began. [Certain]

**Read §0, §2 and §5 first.** The previous draft over-claimed in four places
that the review found blocking; each is corrected here and the change is
recorded in §0.

---

## 0. Part B findings — status

| id | finding | status | where fixed |
|---|---|---|---|
| B-1 | "Accuracy unchanged / identical to V5" for A8/A10 was false; the A8/A10 runtime scores 0.9539, V5's Ollama/GGUF cell 0.9583 | **fixed** | §2(6), §3, §6, §7 — "lossless relative to its own greedy baseline", McNemar reported |
| B-2 | Headline cascade 0.8433 was the τ = 1.0 (always-escalate) row, unlabelled; pre-registered τ_A = 0.50 gives 0.7327 < flat 0.7465 | **fixed** | §3, §5, §12 — headline is now τ_A; τ = 1.0 reported only as a labelled "always-escalate pipeline", shown to be dominated |
| B-3 | A1.7 factor-vs-field comparison invalid: field-level prompt maps every container to a grid cell (0/36); on grid rows field-level is 108/108 vs factor-level 76/108 | **withdrawn** | §5, §9 — bug and grid-only numbers recorded; not re-run |
| B-4 | SALSA "digits" mapping reads logits one position early (`' 1'` = two Qwen tokens) | **withdrawn** | every digits number and the "mapping-sensitivity" claim removed; letters stand |
| S-1 | MASSIVE flat vs factorized = two heads of one jointly trained model | **fixed (restated)** | §8 — null restated as limited to that design; no retrain |
| S-2 | MASSIVE router AUROC scores both signals against factorized-correctness labels | **fixed (restated)** | §8 |
| S-3 | Cascade latency in `p45_cascade_full.csv` / `metrics_all.csv` is a constant model, docstring said "measured" | **fixed** | constants removed from this report; docstring corrected; §3 quotes only measured per-call latencies |
| S-4 | "4.081 J/cmd" for the cascade was one SALSA call's energy | **fixed** | §3 — per-call energy labelled as such |
| S-5 | "3 warmup excluded from every latency figure" was false (all 217 measured) | **fixed** | §6, §10; prints/docstrings in `a8_speculative.py`, `a10_prefix_cache.py` |
| S-6 | Pre-registration post-dates the P2 held-out confidence analysis; quoted point was neither τ_A nor τ_B | **fixed** | §5 timestamps stated; headline now τ_A |
| S-7 | A1 arms / Flan-T5 load the argmax dev epoch while printing the plateau rule | **fixed (print)** | `a1_arms.py`, `a1_rest.py` now print what is loaded; §9 |
| S-8 | A6 leakage scan is a function-name heuristic | **fixed (restated)** | §10 — heuristic named; Part B's code trace cited as the verification |
| S-9 | INT4 flip count depends on quantiser arithmetic (9 in FP16, 8 in FP32) | **fixed** | §11 reports 8–9 |
| S-10 | Three model-size figures across outputs | **fixed** | one definition (whole model, fp32 scales included): 132.76 → 69.20 MB; `memory_footprint.csv` and `p45_full.py` aligned |
| M-1 | Idle-power baseline not idle (util mean 20 %); median idle vs mean load | **recorded** | §6 |
| M-2 | Latency ratios quoted to 4 s.f. from single runs | **fixed** | ratios rounded to the run-to-run spread |
| M-3 | A9 tie rule text vs code | **recorded** | §5 (no tie occurred) |
| M-4 | A9 own `norm()` scorer | **recorded** | §5 (agrees with `llm/score.py` on every arm) |
| M-5 | "Reproduced A5 exactly" is aggregate-only (106/2974 rows differ) | **fixed** | §8 |
| M-6 | A8 greedy gate read a different results file | **recorded** | §7 |
| M-7 | "Total" row-vocabulary disjunction overstated (row B in-vocabulary) | **fixed** | §9 |
| M-8 | Masked-CE mean reduction weights factors per batch | **recorded** | §4 |
| M-9 | Latency "delta_pts" unit artefact in `factorized_vs_flat.csv` | **recorded** | not quoted |
| M-10 | Drafter time included (confirmed) | **verified** | §6 |
| M-11 | "LoRA beats gpt-4o-mini on rows" never tested | **withdrawn** | not claimed |

---

## 1. Completion table

| # | item | status | evidence / reason |
|---|---|---|---|
| P1 | Attribution (TIDE, factor errors, vocab buckets, confidence split, confusion signatures, calibration) | DONE | `PHASE_1_SUMMARY.md`, 8 CSVs, 40 figures |
| P2 | Factorized location head | DONE | `PHASE_2_SUMMARY.md`; Part B: round-trip exact, parameter count equal, masking correct |
| P3 | INT4 gate | DONE — FAILED | `PHASE_3_SUMMARY.md`; A9 bounds the effect to ±1.84 pts |
| P4.1–4.2 | Precision cascade + τ sweep | NOT RUN | Gate failed; agreed in advance |
| P4.3 | Comparison arms | PARTIAL | Precision escalation not runnable (gate failed). Field-level model deferral (A1.7) **withdrawn** — see §0 B-3 |
| P4.4 | Memory footprint | DONE | `memory_footprint.csv` (one size definition) |
| P4.5 | Per-factor model escalation | DONE | `PHASE_4_SUMMARY.md`; headline now at τ_A = 0.50 |
| P5.1 | Rule-based slot grammar | DONE | plus a firewalled honest variant |
| P5.2 | kNN over MiniLM | DONE | 0.6452 |
| P5.3 | TF-IDF + logreg | DONE | 0.6221 |
| P5.4 | TinyBERT-4 | DONE | |
| P6.1–6.4 | SALSA logit classification | DONE (letters only) | digits mapping **withdrawn** (§0 B-4) |
| P6.5 | qwen2.5:0.5b SALSA | DONE (letters only) | `a1_salsa05b.csv` letters rows |
| P7 | Generalization (MASSIVE) | DONE | `PHASE_7_SUMMARY.md`; null restated (§8) |
| A1.1 | Frozen encoder + linear probe | DONE | `a1_frozen_probe_summary.csv` |
| A1.4 | BiLSTM joint intent+slot | DONE | `a1_bilstm_summary.csv` |
| A1.5 | Flan-T5-base seq2seq | DONE | `a1_flant5_summary.csv` |
| A1.6 | Ensemble-over-disagreement | DONE — CLAIM WITHDRAWN | 0.9630 does not separate from the LoRA's 0.9583 (p = 1.0) |
| A1.7 | Field-level FLMD cloud | **WITHDRAWN** | comparator prompt bug; §0 B-3 |
| A2 | Complete metric coverage | DONE, with caveat | `metrics_all.csv` 50 × 26, 0 blanks; cascade latency/energy columns are modelled constants (§0 S-3) |
| A3 | End-to-end MuJoCo success | DONE | `a3_mujoco_e2e.csv` |
| A4 | Legend ablation | SKIPPED | reuse confound; §13 |
| A5 | MASSIVE illegal combinations | DONE | `a5_massive_constrained.csv` |
| A6 | Self-check | DONE, heuristic | 9/9 recomputations match; leakage verified by Part B code trace, not by the scan (§10) |
| A8 | Classifier-drafted speculative decoding | DONE | 217/217 identical to its own greedy baseline (0.9539); 7.8× p50, 3.6× energy |
| A9 | Bootstrap CIs + McNemar | DONE | `a9_significance.csv` + three comparisons added in this revision (§5) |
| A10 | Prefix KV caching for A8 | DONE | 217/217 identical, also in shuffled order (Part B); further 1.9× p50, 2.1× energy |
| A11 | Rejection-position analysis for A8/A10 | DONE | `a11_rejection_positions.csv`; no new inference — the draft is rebuilt from the saved classifier predictions (§7.1) |
| A12 | Conventional small-LM drafter (qwen2.5-0.5b) for the same verifier | DONE | `a12_smalllm_drafter_summary.json`; 217/217 identical, and **slower than plain greedy** (§7.2) |
| Energy | GPU joules per command | DONE | `energy.csv`, `a8_energy.csv`, `a10_energy.csv`, `a12_energy.csv` |

---

## 2. Verdict, after significance testing and review

1. **Factorization measurably hurts on the robot task.** −13.82 pts vs the
   flat head, CI [−18.89, −8.76], p = 6.0e-07. Claimable. [Certain]
2. **The MASSIVE router advantage is indistinguishable from zero** — but only
   within the design actually run: two heads on one jointly trained encoder,
   both signals scored against factorized correctness (§8). AUROC gaps
   +0.0021, +0.0007, −0.0001, all CIs spanning 0. [Certain for that design;
   not evidence about separately trained flat vs factorized models]
3. **The ensemble does not beat the LoRA.** +0.46 pts, CI [−1.39, +2.33],
   b = 3 c = 2, p = 1.0. Withdrawn. [Certain]
4. **A1.7 is withdrawn, not "unresolved".** Its field-level comparator never
   returned a container symbol (0/36) because of the prompt in `a1_rest.py`;
   on the 108 grid rows, where both arms actually escalate, the whole-location
   query scored 1.000 and the row-only query 0.704. No factor-vs-field claim
   survives in either direction. [Certain]
5. **Every escalation result in this project is nominal only** — the
   always-escalate pipeline beating the flat head (+9.68, p = 0.0154,
   p_holm = 0.257), its row factor beating the flat row (+18.52, p = 0.0151,
   p_holm = 0.257), and all four v4c cloud-cascade comparisons
   (p_holm ≥ 0.372). None survives Holm–Bonferroni across the 26-hypothesis
   family (§5). And at the pre-registered operating point τ_A = 0.50 the
   per-factor cascade does not beat the flat head: 0.7327 vs 0.7465, −1.38 pts, CI [−7.37,
   +4.61], b = 22 c = 25, p = 0.77. It does beat the un-escalated factorized
   head it is built on (+12.44, CI [+8.29, +17.05], b = 27 c = 0, p = 1.5e-08),
   i.e. escalation recovers what factorization lost, and no more. [Certain]
   The τ = 1.0 row (0.8433) is an **always-escalate pipeline** with no routing
   in it; it is reported in §5 as such and is dominated (§3).
6. **A8 + A10 hold unconditionally as *lossless* relative to their own greedy
   baseline** — but "lossless" is a statement about the *arm being
   accelerated*, and on `held_out_v4c` that arm scores 0.6221 against the flat
   classifier's 0.7143, so there the whole speculative stack is both slower and
   less accurate than the classifier alone (§6): transformers-FP16 greedy with the merged adapter, which scores
   **0.9539 (207/217)** on `held_out_v3`. Output is byte-identical to that
   baseline 217/217 (verified three times, once in shuffled order by the
   reviewer). Against V5's Ollama/Q4_K_M cell (0.9583, n = 216) the runtime
   differs on 4 outputs and 3 correctness outcomes: −0.46 pts, CI [−2.31,
   +0.93], b = 1 c = 2, p = 1.0. Median latency 912 → 58 ms (≈16×), energy
   70.8 → 9.4 J/command (≈7.5×). [Certain]

---

## 3. Headline numbers — conditioned on legend-defined vocabulary

**There is no unconditional "best system" in this project.** Every arm is
reported on both held-out families, because the ranking changes between them
and the reason is vocabulary, not difficulty:

- **`held_out_v3`** uses `upper` (36 grid commands) and `bottom` (36) for rows
  A and C. **The V5 legend defines both**: *"A is the far row. A command may
  also call it the upper row … C is the near row. A command may also call it
  the bottom row."*
- **`held_out_v4c`** uses `back` (36) and `front` (36) for the same two rows.
  **The V5 legend defines neither** — `back` and `front` occur **0 times** in
  the entire zero-shot prompt (so do `top` and `lower`). Nor does training:
  `back`, `front`, `upper`, `bottom` and `lower` occur **0 times in `train_v2`
  and 0 times in `dev_v2`** (§9).

So `held_out_v3` measures an arm on wording the prompt explains, and
`held_out_v4c` measures the same arm on wording nothing explains. **A prompted
arm's v3 number is not a generalisation estimate; its v4c number is.** [Certain]

Row factor n = 108 on both sets; action match n = 217 unless stated. Resident
size uses one definition throughout: whole model, weights at the stated
precision, fp32 scales included (§11).

| arm | reads the legend? | v3 action | v3 row | **v4c action** | **v4c row** | resident |
|---|---|---|---|---|---|---|
| A1.6 ensemble-over-disagreement | partly (cloud arbiter) | 0.9630 (n=216) | — | not run | not run | 132.8 + 1,900 MB + net |
| qwen2.5-3b-lora (V5 cell, Ollama Q4_K_M) | **yes** | 0.9583 (n=216) | — | not run | not run | 1,900 MB |
| **qwen2.5-3b-lora zero-shot, transformers FP16 merged** (the A8/A10 baseline) | **yes** | **0.9539** | **1.0000** | **0.6221** | **0.3333** | 6,593 MB |
| **A8 / A10 speculative decoding** | **yes** (same prompt) | **0.9539** | **1.0000** | **0.6221** | **0.3333** | 6,726 MB |
| gpt-4o-mini / rag (V5 cell) | **yes** | 0.9398† | — | not run | not run | net |
| cascade row → gpt-4o-mini, τ = 1.0 | **yes** (row legend) | 0.7926 | 0.7037 | **0.7788** | **0.6759** | 132.8 MB + net |
| *always-escalate pipeline*: row → qwen2.5-3b-lora, τ = 1.0, 60.4 % escalated | **yes** (row legend) | 0.8433 | 0.8056 | **0.5438** | **0.2037** | 6,726 MB |
| **cascade, pre-registered τ_A = 0.50**, row → qwen2.5-3b-lora | **yes** (row legend) | 0.7327 | 0.5833 | **0.6452** | **0.4074** | 6,726 MB |
| **flat DistilBERT** (V5 rung-4 head, fresh pass) | **no** | 0.7465 | 0.6204 | **0.7143** | **0.5556** | 132.8 MB |
| rules baseline (vocabulary-unfair) | **no** | 0.7235 | — | 0.7235 | — | <1 MB |
| A1.4 BiLSTM from scratch | **no** | 0.7005 | 0.6852 | 0.7005 | 0.6852 | 2.1 MB |
| A1.1 frozen MiniLM + linear probe | **no** | 0.6498 | 0.6204 | 0.6544 | 0.6574 | 87 MB |
| kNN over MiniLM | **no** | 0.6452 | — | not run | not run | 87 MB |
| TF-IDF + logreg | **no** | 0.6221 | — | not run | not run | <1 MB |
| A1.5 Flan-T5-base (fine-tuned, no legend at inference) | **no** | 0.6175 | 0.4259 | 0.5714 | 0.3333 | 990 MB |
| factorized DistilBERT | **no** | 0.6083 | 0.3333 | 0.6359 | 0.3889 | 132.8 MB |
| factorized, INT4 weights | **no** | 0.6083 | 0.3426 | not run | not run | **69.2 MB** |
| A1.2 TinyBERT-4 (14.36 M) | **no** | 0.5760 | 0.3333 | 0.5806 | 0.3426 | 57.4 MB |
| rules baseline (firewalled, honest) | **no** | 0.4470 | — | 0.4470 | — | <1 MB |
| cascade row → oracle (ceiling) | — | 0.9401 | 1.0000 | 0.9401 | 1.0000 | — |

† A9 collapses the 5 repeats of the cloud cell by majority vote (0.9398 on the
216 ensemble ids; 0.9401 on all 217); A1.6 kept the last repeat (0.9352). The
two rules rows score identically on both sets because the hand-written grammar
lists `back` and `front` alongside `upper`/`bottom` — the only arm here whose
author supplied the v4c vocabulary by hand.

### What the two columns say

1. **On legend-defined vocabulary the prompted LoRA is the best system by a
   wide margin** — 0.9539 action, 1.0000 row, and with A8+A10 at 58 ms p50 and
   9.45 J/command with byte-identical output (§6, §7). Nothing else in the
   project is close. (The gap to the next arm is large enough that it was never
   in doubt, but note it was not itself entered as a hypothesis in the
   corrected family.)
2. **On vocabulary the legend does not define, the same arm is beaten by the
   classifier that reads no legend at all**: 0.6221 vs **0.7143** action
   (+9.22 pts for the classifier, CI [+4.15, +14.75], b = 28, c = 8,
   p = 1.2e-03, **claimable**) and 0.3333 vs **0.5556** row (+22.22 pts, CI
   [+13.89, +30.56], b = 24, **c = 0**, p = 1.2e-07, **claimable**) — §5.
   The LoRA's v4c row accuracy is exactly chance and is a default: its row
   confusion is `A→B` 30, `A→C` 6, `C→B` 36, `B→B` 36, so all 36 of its row
   hits are the middle row it was not being asked to generalise on.
3. **The legend-blind arms move less, but that is stability, not competence
   — and it is not a back/front result.** Aggregate movement is small (flat
   −3.2 action / −6.5 row, BiLSTM 0.0 / 0.0 with byte-identical predictions,
   frozen probe +0.5 / +3.7, TinyBERT +0.5 / +0.9, factorized +2.8 / +5.6),
   but splitting the flat classifier's 108 v4c grid commands by the row word
   they actually use shows where its 0.5556 comes from:

   | group | n | flat classifier | LoRA-V5 |
   |---|---|---|---|
   | `middle` only | 12 | 12 (1.0000) | 12 (1.0000) |
   | `center` only | 24 | 24 (1.0000) | 24 (1.0000) |
   | `back`/`front` only | 48 | 24 (0.5000) | **0 (0.0000)** |
   | `back`/`front` **and** `middle`/`center` | 24 | **0 (0.0000)** | **0 (0.0000)** |

   By truth row: `back` → A, flat **2/36**, LoRA **0/36**; `front` → C, flat
   22/36, LoRA 0/36; `middle`/`center` → B, both **36/36**.

   **Neither system maps `back`.** The classifier's advantage on v4c comes from
   the vocabulary it did see: all 36 of its `middle`/`center` commands, plus a
   partial and unexplained `front` → C signal, against 24/72 = chance on
   back/front as a whole. And when a back/front command *also* contains a seen
   word it scores **0/24**, answering B every time (confusion A→B 13,
   C→B 11) — the seen word wins. So the claimable +22.22-point row gap over
   the prompted LoRA (§5) says the LoRA is *systematically* wrong where the
   classifier is *randomly* wrong; it is **not** evidence that the classifier
   has learned back/front. [Certain]
4. **Escalating to the cloud is the only prompted configuration that holds up
   on v4c**, and it separates from the flat classifier on both metrics:

   | arm (v4c) | esc. rate | action | vs flat | row | vs flat |
   |---|---|---|---|---|---|
   | cascade row → gpt-4o-mini, **τ = 1.0 (always escalate)** | 60.4 % | 0.7788 | +6.45, CI [+0.92, +11.98], p = 0.038, p_holm = 0.456 — **nominal** | 0.6759 | +12.04, CI [+1.85, +22.22], p = 0.035, p_holm = 0.456 — **nominal** |
   | cascade row → gpt-4o-mini, **pre-registered τ_A = 0.50** | 16.1 % | 0.7696 | +5.53, CI [+0.92, +10.14], p = 0.036, p_holm = 0.456 — **nominal** | 0.6574 | +10.19, CI [+2.78, +18.52], p = 0.027, p_holm = 0.372 — **nominal** |

   **None of the four survives Holm–Bonferroni across the 26-hypothesis family
   (§5).** They are nominal: separable per-comparison, not separable family-wise.

   The 0.7788 / 0.6759 figure quoted above is the **τ = 1.0 always-escalate**
   configuration, which is a pipeline, not routing (§5). The pre-registered
   τ_A = 0.50 operating point separates nominally too, at a quarter of the
   cloud calls, and is the figure to quote for a routed system. Both margins
   are thin — the action CI reaches down to +0.92 points — and neither
   survives correction. `gpt-4o-mini` supplies the
   `back`/`front` mapping from world knowledge, which the legend does not give
   it and the local 3B does not have. (`a13_v4c_cloud_significance.csv`)
5. **Escalating the row to the local 3B is actively harmful on v4c** — 0.5438
   action, 0.2037 row, both **below** the un-escalated factorized head
   (0.6359 / 0.3889). The gate correctly flags that the local head is unsure;
   it cannot make the escalation target competent.

**So: "best system" = qwen2.5-3b-lora zero-shot with the V5 legend, on
commands whose row vocabulary the legend defines.** Off that vocabulary:

- the best **local** system is the flat DistilBERT classifier (0.7143 /
  0.5556), beating the prompted LoRA by a claimable margin — while itself
  being at chance on back/front and scoring 0/24 where a seen word competes;
- **cascade-to-cloud is nominally best; it is not separable after
  correction.** It beats the classifier per-comparison on both metrics
  (action CI [+0.92, +11.98], p = 0.038 at τ = 1.0; [+0.92, +10.14],
  p = 0.036 at the pre-registered τ_A = 0.50) but every one of those four
  tests has p_holm ≥ 0.37 across the 26-hypothesis family (§5), and it needs a
  network dependency to get there. On the evidence here the flat classifier and
  cascade-to-cloud are not distinguishable;
- **the classifier's advantage over the prompted LoRA on v4c is the one
  v4c result that survives correction** (action p_holm = 0.0215; row
  p_holm = 2.9e-06);
- **no arm in this project maps `back` to row A.** The best v4c row accuracy
  anywhere here is 0.6759, from asking a cloud model. [Certain]

Per-call latencies that *were* measured (`p6_latency_qwen3blora_held_out_v3.json`,
`p45_cloud_latency_held_out_v3.json`): local factorized forward 3.2 ms p50;
SALSA row query on the 3B LoRA 35.6 ms p50 / 36.8 ms p95; gpt-4o-mini row
query 657 ms p50 / 1,285 ms p95. **Cascade latency and energy are not quoted
as per-command figures anywhere in this report**: the columns in
`p45_cascade_full.csv` and `metrics_all.csv` are those constants added per
command (p50 = p95 = p99 by construction) and the `joules_per_command` column
for cascade arms is the SALSA per-call energy, not a cascade average. [Certain]

**Adjacent rows in this table are mostly not distinguishable** (§5).

---

## 4. A3 — end-to-end execution in MuJoCo

`FullGraspBackend`, `FetchPickAndPlace-v4`, 17 locations, `sim/` unmodified.
Object pose seeded from `command_id`; 409 distinct executions cover 1,301
arm × command pairs (217 × 5 + 216).

`MOVE_TOL = 0.05` changed nothing: `full_grasp_backend.move_to` returns
`"ok": True` unconditionally, so 0.05 was applied as a gate the backend lacks
(final gripper XY error vs the resolved cell). Errors: p50 1.6 cm, max 2.0 cm;
every `move_to` passes at 3, 5 and 9 cm.

| arm | n | e2e | n_moving | e2e (moving) | wrong prediction | execution failure |
|---|---|---|---|---|---|---|
| ensemble | 216 | 0.9630 | 194 | 0.9845 | 8 | 0 |
| always-escalate pipeline (τ = 1.0) | 217 | 0.8433 | 195 | 0.8410 | 34 | 0 |
| flat DistilBERT | 217 | 0.7465 | 195 | 0.7487 | 55 | 0 |
| BiLSTM | 217 | 0.7005 | 195 | 0.7231 | 65 | 0 |
| factorized | 217 | 0.6083 | 195 | 0.5795 | 85 | 0 |
| truth (oracle) | 217 | 1.0000 | 195 | 1.0000 | 0 | 0 |

SUCCESS requires prediction == truth *and* execution; with 0 execution
failures, end-to-end success equals strict string match for every arm, so §5's
verdicts transfer unchanged. The τ_A cascade was not among the five arms
executed; its per-command outcomes are a subset of the arms above (local
factorized predictions plus 32 escalated rows), all of whose executions
succeeded, so its e2e would also equal its string match (0.7327). [Likely —
not separately executed]

Factor supervision note (M-8): masked cross-entropy uses `reduction="mean"`
over the supervised subset of each batch, so a batch with one container
example gives that term the weight of the mean of all row terms. Not a bug;
"plain sum of 7 terms" is the literal code, this is its effective weighting.

---

## 5. A9 — confidence intervals and paired tests

Protocol: clustered bootstrap over `command_id`, 2,000 resamples, seed
20260827; McNemar's exact binomial test on discordant pairs. Repeated V5 cells
collapsed to one outcome per command by majority vote (code: `sum > n/2`, so a
tie would be False; no tie occurred). A9 scores the cascade/flat/INT4 arms with
its own `norm()` rather than `llm/score.py`; because `emit()` never adds
fields, the two agree on every arm (Part B rescored all of them).

Decision rule: claimable only if the 95 % CI excludes 0 AND McNemar p < 0.05.

Timestamps (S-6): `THRESHOLD_PREREGISTRATION.md` 19:00:50 precedes every
P3/P4/P6/P7 output (first: 19:02:00) but follows the P2 held-out
row-confidence analysis (18:51–19:00), which it acknowledges. τ_A = 0.50 is a
constant; τ_B is dev-selected and degenerated to 0.00 (never escalate).

| family | comparison | n | A | B | diff (pts) | 95 % CI | b / c | p | verdict |
|---|---|---|---|---|---|---|---|---|---|
| held_out_v3 | ensemble vs qwen2.5-3b-lora (V5) | 216 | 0.9630 | 0.9583 | +0.46 | [−1.39, +2.33] | 3/2 | 1.000 | NOT CLAIMABLE |
| held_out_v3 | ensemble vs gpt-4o-mini/rag (V5) | 216 | 0.9630 | 0.9398 | +2.31 | [0.00, +4.63] | 6/1 | 0.125 | NOT CLAIMABLE |
| held_out_v3 | ensemble vs always-escalate pipeline | 216 | 0.9630 | 0.8426 | +12.04 | [+6.94, +17.13] | 30/4 | 6.2e-06 | CLAIMABLE |
| held_out_v3 | **A8/A10 greedy baseline (transformers FP16) vs V5 Ollama Q4 cell** ‡ | 216 | 0.9537 | 0.9583 | −0.46 | [−2.31, +0.93] | 1/2 | 1.000 | NOT CLAIMABLE (no measurable runtime cost) |
| held_out_v3 | A8/A10 greedy baseline vs always-escalate pipeline ‡ | 217 | 0.9539 | 0.8433 | +11.06 | [+5.99, +16.13] | 29/5 | 3.9e-05 | CLAIMABLE (pipeline dominated) |
| held_out_v3 | **cascade τ_A = 0.50 vs flat** ‡ | 217 | 0.7327 | 0.7465 | −1.38 | [−7.37, +4.61] | 22/25 | 0.771 | **NOT CLAIMABLE** |
| held_out_v3 | cascade τ_A = 0.50 vs factorized ‡ | 217 | 0.7327 | 0.6083 | +12.44 | [+8.29, +17.05] | 27/0 | 1.5e-08 | CLAIMABLE |
| held_out_v3 | always-escalate pipeline vs flat | 217 | 0.8433 | 0.7465 | +9.68 | [+2.75, +17.05] | 45/24 | 0.0154 | **nominal** (does not survive family-wise correction); also *not routing* |
| held_out_v3 | factorized vs flat | 217 | 0.6083 | 0.7465 | −13.82 | [−18.89, −8.76] | 4/34 | 6.0e-07 | CLAIMABLE |
| held_out_v3 | flat vs BiLSTM | 217 | 0.7465 | 0.7005 | +4.61 | [−1.84, +11.06] | 32/22 | 0.220 | NOT CLAIMABLE |
| held_out_v3 | BiLSTM vs factorized | 217 | 0.7005 | 0.6083 | +9.22 | [+2.30, +16.14] | 42/22 | 0.0169 | **nominal** (does not survive family-wise correction) |
| held_out_v3 | frozen probe vs factorized | 217 | 0.6498 | 0.6083 | +4.15 | [−2.76, +11.06] | 34/25 | 0.298 | NOT CLAIMABLE |
| held_out_v3 | factorized vs TinyBERT-4 | 217 | 0.6083 | 0.5760 | +3.23 | [−0.92, +7.37] | 14/7 | 0.189 | NOT CLAIMABLE |
| held_out_v3 | factorized vs Flan-T5 | 217 | 0.6083 | 0.6175 | −0.92 | [−5.99, +4.15] | 13/15 | 0.851 | NOT CLAIMABLE |
| held_out_v3 | flat vs rules (open) | 217 | 0.7465 | 0.7235 | +2.30 | [−5.53, +10.60] | 43/38 | 0.657 | NOT CLAIMABLE |
| held_out_v3 | rules open vs firewalled | 217 | 0.7235 | 0.4470 | +27.65 | [+22.12, +33.64] | 60/0 | 1.7e-18 | CLAIMABLE |
| **held_out_v4c** | **flat classifier vs qwen2.5-3b-lora V5 zero-shot** ‡ | 217 | 0.7143 | 0.6221 | **+9.22** | [+4.15, +14.75] | 28/8 | 1.2e-03 | **CLAIMABLE** |
| **held_out_v4c** | **flat row vs qwen2.5-3b-lora V5 row** ‡ | 108 | 0.5556 | 0.3333 | **+22.22** | [+13.89, +30.56] | 24/0 | 1.2e-07 | **CLAIMABLE** |
| **held_out_v4c** | cascade row → gpt-4o-mini (τ = 1.0) vs flat classifier ‡ | 217 | 0.7788 | 0.7143 | +6.45 | [+0.92, +11.98] | 27/13 | 0.0385 | **nominal** (does not survive family-wise correction) |
| **held_out_v4c** | cascade row → gpt-4o-mini (τ = 1.0) row vs flat row ‡ | 108 | 0.6759 | 0.5556 | +12.04 | [+1.85, +22.22] | 23/10 | 0.0351 | **nominal** (does not survive family-wise correction) |
| **held_out_v4c** | cascade row → gpt-4o-mini (τ_A = 0.50) vs flat classifier ‡ | 217 | 0.7696 | 0.7143 | +5.53 | [+0.92, +10.14] | 20/8 | 0.0357 | **nominal** (does not survive family-wise correction) |
| **held_out_v4c** | cascade row → gpt-4o-mini (τ_A = 0.50) row vs flat row ‡ | 108 | 0.6574 | 0.5556 | +10.19 | [+2.78, +18.52] | 16/5 | 0.0266 | **nominal** (does not survive family-wise correction) |
| row factor | flat row vs factorized row | 108 | 0.6204 | 0.3333 | +28.70 | [+20.37, +37.96] | 31/0 | 9.3e-10 | CLAIMABLE |
| row factor | cascade τ_A row vs flat row ‡ | 108 | 0.5833 | 0.6204 | −3.70 | [−14.81, +8.33] | 18/22 | 0.636 | NOT CLAIMABLE |
| row factor | always-escalate row vs flat row | 108 | 0.8056 | 0.6204 | +18.52 | [+4.63, +32.41] | 41/21 | 0.0151 | **nominal** (does not survive family-wise correction); also *not routing* |
| A1.7 | factor vs field level | 168 | — | — | — | — | — | — | **WITHDRAWN** (comparator invalid, §9) |
| quantization | FP16 vs INT4 | 217 | 0.6083 | 0.6083 | 0.00 | [−1.84, +1.84] | 2/2 | 1.000 | NOT CLAIMABLE (bounded) |
| MASSIVE (Phase 7, joint model) | flat head vs factorized heads (free) | 2974 | 0.8880 | 0.8783 | +0.98 | [+0.54, +1.45] | 41/12 | 8.2e-05 | CLAIMABLE within design |
| MASSIVE (Phase 7) | router AUROC per-factor vs flat, factorized labels | 2974 | 0.8910 | 0.8889 | +0.0021 | [−0.0025, +0.0073] | — | — | NOT CLAIMABLE |
| MASSIVE (fresh) | flat vs factorized (free) | 2974 | 0.8860 | 0.8779 | +0.81 | [+0.40, +1.18] | 30/6 | 7.0e-05 | CLAIMABLE within design |
| MASSIVE (fresh) | flat vs factorized (constrained) | 2974 | 0.8860 | 0.8850 | +0.10 | [−0.24, +0.44] | 16/13 | 0.711 | NOT CLAIMABLE |
| MASSIVE (fresh) | constrained vs free | 2974 | 0.8850 | 0.8779 | +0.71 | [+0.40, +1.04] | 21/0 | 9.5e-07 | CLAIMABLE |
| MASSIVE (fresh) | router AUROC, free decode | 2974 | 0.8893 | 0.8886 | +0.0007 | [−0.0043, +0.0062] | — | — | NOT CLAIMABLE |
| MASSIVE (fresh) | router AUROC, constrained decode | 2974 | 0.8777 | 0.8778 | −0.0001 | [−0.0056, +0.0054] | — | — | NOT CLAIMABLE |

‡ computed after A9 on the existing per-command files with the same bootstrap
(2,000 resamples, seed 20260827) and exact McNemar; they are not in
`a9_significance.csv`. The two v4c rows are in `a13_v4c_significance.csv` and
are stable across four bootstrap seeds (action CI [+3.69, +4.15] to [+14.29,
+14.75]; row CI [+13.89, +14.81] to [+30.56]). The four cloud-cascade rows are
in `a13_v4c_cloud_significance.csv` and are likewise seed-stable; all four are
thin (every action CI reaches down to +0.92 points). The reviewer's independent implementation reproduced
every b/c/p of the original 23 and the CIs to within Monte-Carlo noise.

Of the original 23 comparisons, 10 are claimable and 13 not; A1.7 is now
withdrawn rather than "not claimable". The four added rows split 2/2.

### Family-wise correction (Holm–Bonferroni)

The tables above report **per-comparison** p-values. Across the declared family
— all 23 rows of `a9_significance.csv` plus all 6 A13 rows
(`a13_v4c_significance.csv`, `a13_v4c_cloud_significance.csv`) — that is 29
hypotheses, of which **26 carry a McNemar p** and enter the ranking; the 3
MASSIVE router-AUROC rows are CI-only, cannot be ranked, and were already NOT
CLAIMABLE, so excluding them removes nothing. Holm–Bonferroni at α = 0.05,
`claimable_corrected = (CI excludes 0) AND p_holm < 0.05`. Both columns
(`p_holm`, `claimable_corrected`) are now in all three CSVs.

**9 of the 16 nominally claimable results survive; 7 do not.**

| survives correction | p_raw | p_holm |
|---|---|---|
| rules open vs rules firewalled | 1.7e-18 † | < 1e-16 |
| flat row vs factorized row | 9.3e-10 † | < 1e-7 |
| **v4c: flat classifier row vs qwen2.5-3b-lora V5 row** | 1.2e-07 | 2.9e-06 |
| factorized vs flat DistilBERT | 6.0e-07 † | 2.3e-05 |
| MASSIVE constrained vs free | 9.5e-07 † | 2.3e-05 |
| ensemble vs always-escalate pipeline | 6.2e-06 | 1.3e-04 |
| MASSIVE (fresh) flat vs factorized, free | 7.0e-05 | 1.4e-03 |
| MASSIVE (Phase 7) flat vs factorized, free | 8.2e-05 | 1.6e-03 |
| **v4c: flat classifier vs qwen2.5-3b-lora V5 zero-shot** | 1.2e-03 | **0.0215** |

| lost to correction — nominal only | p_raw | p_holm |
|---|---|---|
| cascade row vs flat row (always-escalate) | 0.0151 | 0.257 |
| always-escalate pipeline vs flat | 0.0154 | 0.257 |
| BiLSTM vs factorized | 0.0169 | 0.257 |
| v4c cascade → gpt-4o-mini (τ_A = 0.50) row vs flat row | 0.0266 | 0.372 |
| v4c cascade → gpt-4o-mini (τ = 1.0) row vs flat row | 0.0351 | 0.456 |
| v4c cascade → gpt-4o-mini (τ_A = 0.50) vs flat | 0.0357 | 0.456 |
| v4c cascade → gpt-4o-mini (τ = 1.0) vs flat | 0.0385 | 0.456 |

† `a9_significance.csv` stores `mcnemar_p` rounded to 6 decimals, so the
smallest p-values read as 0.0 in the file; the ranking and every verdict are
unaffected, since all of them are orders of magnitude below any Holm threshold.

**What this costs the report.** The two surviving *positive* results about this
project's own contribution are both negative in direction: factorization hurts,
and the prompted arm loses to the classifier off legend-defined vocabulary.
**Every escalation result — the cascade beating the flat head, the cascade row
beating the flat row, and all four cloud-cascade rows — is nominal only.** The
escalation story rests on effects of 6–18 points measured on 108–217 commands,
which a 26-hypothesis family does not support. [Certain]

Rows marked ‡ that are *not* in the declared family (the in-report comparisons
added after A9: cascade τ_A vs flat, cascade τ_A vs factorized, A8 greedy vs
V5, A8 greedy vs always-escalate) carry **uncorrected** p-values. Two of them
would survive any correction of this size (p = 1.5e-08 and 3.9e-05); the other
two are already not claimable.

### What this changes

- **The ensemble claim is withdrawn** (p = 1.0 on 5 discordant commands).
- **The confidence-gated cascade has no measurable advantage over the flat
  head at its pre-registered operating point** (−1.38 pts, p = 0.77), and its
  row factor is no better than the flat row (p = 0.64). What it measurably
  does is undo the factorization penalty (+12.44 vs factorized, b = 27 c = 0).
- **The always-escalate pipeline (0.8433) beats the flat head, but it is not
  routing** — every grid-typed row is delegated — and it is dominated by
  simply running the LoRA (0.9539 at 58 ms with A10, same resident model).
- **The MASSIVE router comparison is a tight null within its design** (CI
  half-width ≈ ±0.005 AUROC), but the design is two heads on one jointly
  trained encoder, scored against the factorized head's correctness (§8).
- **INT4 is bounded**: true cost within ±1.84 pts.
- **The flat DistilBERT does not measurably beat a hand-written grammar**
  (+2.30, p = 0.657); the grammar's score depends on vocabulary it should not
  have had (firewalled: −27.65, p = 1.7e-18).
- **Several vocabulary-story comparisons do not separate individually**; the
  argument rests on the pattern plus the OOV analysis (§9).

---

## 6. A10 — prefix KV caching for A8

All 217 prompts share the V5 legend; the longest common token prefix is 449
tokens of 460–474 (96 % of the median prompt); per-command suffix 11–25
tokens, median 18. The prefix is processed once, its `DynamicCache` stays
resident, each command appends suffix + draft and the cache is cropped back to
449 entries before every command and after acceptance.

### Correctness first

| arm | identical to plain greedy (same runtime) |
|---|---|
| speculative, no prefix cache (A8) | 217 / 217 PASS |
| prefix-cached plain greedy | 217 / 217 PASS |
| prefix-cached speculative (A10) | 217 / 217 PASS |
| **reviewer re-run, own decoders, commands in two different shuffled orders** | **217 / 217 PASS**; prefix K/V bit-identical after all commands; 746 forward passes reproduced |

Acceptance unchanged by caching: token acceptance 0.8946 (4,704 / 5,258),
full-draft 162 / 217, forward passes 746 (= A8), accepted-token count equal to
A8 on 217 / 217 commands.

**A10 on `held_out_v4c` equals 0.6221 by construction.** Speculative decoding
is output-exact: A8 and A10 reproduce the verifier's greedy output token for
token, so on any input set their accuracy *is* the verifier's. The verifier
here is qwen2.5-3b-lora zero-shot with the V5 legend, which scores **0.6221
action / 0.3333 row on v4c** (§3, A13). No v4c speculative run is needed to
know this and none was made. The consequence is worth stating plainly:
**on `held_out_v4c` the whole speculative stack is slower *and* less accurate
than the flat classifier alone** — 0.6221 against 0.7143 (−9.22 pts,
p = 1.2e-03, claimable) at 58 ms p50 and 9.45 J/command against the
classifier's ~3 ms and 0.062 J/command. The speed-up is real and the loss is
exact, but both are properties of the arm being accelerated, and off
legend-defined vocabulary that arm is the wrong one to accelerate. [Certain]

**Accuracy of this runtime**: the plain-greedy baseline that all three arms
reproduce scores **0.9539 (207/217)** on `held_out_v3` with `llm/score.py`
(0.9537 on the 216 non-warmup V5 ids). A8/A10 are *lossless relative to that
baseline*. They are not byte-identical to V5's Ollama Q4_K_M rows (213/217
raw outputs agree; correctness differs on commands 131, 139 — V5 right — and
175 — this runtime right), and the difference is not measurable (§5 ‡ row).

### Latency (n = 217, all commands included; 3 were pre-run as GPU warm-up and are *not* excluded — S-5)

| arm | p50 | p95 | p99 | mean |
|---|---|---|---|---|
| plain greedy | 912 ms | 994 ms | 1,051 ms | 816 ms |
| prefix-cached plain greedy | 855 ms | 922 ms | 1,057 ms | 760 ms |
| speculative (A8) | 113 ms | 430 ms | 832 ms | 190 ms |
| **prefix-cached speculative (A10)** | **58 ms** | **344 ms** | **743 ms** | **133 ms** |

A10 vs A8: ≈1.9× at p50, 1.25× at p95, 1.1× at p99, 1.4× mean. A10 vs plain
greedy: ≈16× at p50. Prefix caching alone: 1.07× at p50. Ratios are rounded
to the observed run-to-run spread (A8's own run measured plain greedy at
883 ms p50 vs 912 ms here, 3 %). Excluding the three pre-run commands moves
every p50 by < 0.2 ms. All four arms run on the same `prompt_ids` in one
process, interleaved per command; the classifier's CPU draft time (p50
19.4 ms) is included in both speculative arms (`spec_ms = spec_gpu_ms +
draft_ms` on every row) and is 33 % of A10's p50.

### Energy (GPU only; idle baseline 12.12 W = *median* of `power_idle.csv`, whose mean is 14.87 W with 20 % mean GPU utilisation — the baseline was not fully idle, M-1; effect ≈ 3 % on these two rows)

| arm | N | wall | mean P | J/command |
|---|---|---|---|---|
| speculative (A8), re-measured here | 868 | 152.4 s | 127.0 W | 20.17 |
| **prefix-cached speculative (A10)** | 1085 | 129.2 s | 91.4 W | **9.45** |

≈2.1× over A8 (A8 re-measured at 20.17 vs 19.41 J/cmd in its own run, 4 %).
Cumulative vs the plain generative arm: 70.77 → 9.45 J/command (≈7.5×),
912 → 58 ms p50 (≈16×), **output identical to the same runtime's greedy
decode, which scores 0.9539**. [Certain]

---

## 7. A8 — classifier-drafted speculative decoding

Greedy was confirmed before any code was written: `llm/clients.py:147` sends
`temperature 0.0`; `bench/run_cell.py:323` records it; the V5 rows carry
`temperature == 0.0`; `Modelfile.qwen2.5-3b-lora` sets `PARAMETER temperature
0`. (The script's gate reads `results_heldout_v3_lora_scored.csv`, an earlier
run, and then compares against `results_v5_qwen3.csv`; both carry 0.0 — M-6.)

**Runtime mismatch, stated.** V5 decoded through Ollama against
`qwen3b-lora-Q4_K_M.gguf`; A8/A10 run transformers FP16 with the adapter
merged. Byte-identity with the V5 rows is neither achievable nor claimed;
agreement is 213/217 raw, and the two runtimes' accuracies (0.9537 vs 0.9583
on n = 216) do not separate (§5).

**Method.** The classifier's prediction is rendered in the LoRA's exact output
format (verified against all 217 V5 raws, 0 mismatches), tokenised (BPE
boundary clean 217/217) and appended including the trailing `<|im_end|>`. One
forward pass over prompt + draft yields the argmax at every draft position; the
longest matching prefix is accepted, the first-mismatch token comes free from
the same pass, the cache is cropped, greedy resumes.

| | value |
|---|---|
| token acceptance rate | 0.8946 (4,704 / 5,258) |
| commands where the whole draft was taken | 162 / 217 = 0.7465 |
| commands finished in one forward pass | 162 / 217 |
| forward passes, plain → speculative | 5,288 → 746 (−85.9 %) |
| A8's own run, p50 | 883 ms → 113 ms (≈7.8×) |
| A8's own energy run | 70.77 → 19.41 J/cmd (≈3.6×) |

Acceptance is bimodal: every full-accept command takes 1 pass; of the other
55, 41 take 9 passes (divergence lands on the grid-cell character). Within
those 55, acceptance p50 is 0.6786 (19 of a median 28 tokens). The 0.7465
full-accept rate coincides with the classifier's 0.7465 accuracy but is a
different event: the sets overlap on 158 of 162 (4 classifier-right /
LoRA-would-differ, 4 both-wrong-and-agreeing). [Certain]

**Bug recorded.** The first run scored 55/217 because an accepted `<|im_end|>`
was pushed into the output; the byte-identity check caught it.

### Novelty — narrow verdict

Ten queries (listed in the previous draft; unchanged). **The mechanism is not
novel** — DFlash (2602.06036), Medusa (2401.10774), blockwise parallel
decoding occupy the one-shot-drafter slot. **No prior art was found for the
drafter choice**: reusing an already-deployed discriminative classifier over a
closed label set, rendered deterministically into the target's format, with
zero extra training. [Likely — ten queries is weak evidence of absence] The
defensible claim is a systems one: in a pipeline that already ships a
classifier, that classifier is a free one-shot drafter for the generative
rung.

### Related work — seven further entries, one differing axis each

Checked 2026-09-22 by web search from this machine; each row names the **one
axis** on which A8 differs, not a general comparison.

| work | what it is | the one axis on which A8 differs |
|---|---|---|
| **TaskSpec** — *Automatic Task Detection and Heterogeneous LLM Speculative Decoding*, arXiv **2505.08600** | clusters observed traffic into tasks and fields a task-specialised autoregressive draft model per cluster; reports acceptance rising from <20 % to >60 % at γ = 4 on reasoning traffic | **the drafter is still an autoregressive LM trained for drafting.** A8 trains nothing: it reuses a discriminative classifier the pipeline already ships for routing, and A12 shows that on this workload an autoregressive drafter loses even at 0.54 acceptance |
| **SpecGR** — *Inductive Generative Recommendation via Retrieval-based Speculation*, arXiv **2410.02939**, AAAI 2026 | a retrieval/inductive model drafts candidate **items**, the generative recommender verifies, which lets the system recommend items never seen in training | **speculation is used to widen coverage, not to cut latency.** Verification changes *what can be produced*; A8's verification changes only *how fast*, and its output set is exactly the verifier's by construction |
| **C2T** — *A Classifier-Based Tree Construction Method in Speculative Decoding*, arXiv **2502.13652** | a 241-parameter three-feature classifier scores draft tokens to prune the candidate tree before verification, cutting candidate tokens ~25 % against EAGLE-2 | **its classifier selects among tokens a drafter already produced; A8's classifier *is* the drafter.** C2T still needs a draft model underneath |
| **CDSL** — *Constrained Decoding with Speculative Lookaheads*, arXiv **2412.10418**, NAACL 2025 | a small draft LM proposes lookaheads that a large target LM plus a task-specific reward function verifies, giving 2.2–12.15× over lookahead-heuristic constrained decoding | **the constraint lives in the verifier, as a reward the acceptance test consults.** In A8 the schema constraint lives in the drafter (the classifier can only emit legal symbols) and the acceptance test is plain token equality, so the output stays greedy-exact |
| **DCCD** — *The Hidden Cost of Structured Generation in LLMs: Draft-Conditioned Constrained Decoding*, Reddy, Walker, Ide, Bedi, arXiv **2603.03305** (cs.CL). _Correction 2026-09-28: this row previously read "not identified" after four searches on 2026-09-22; a search on 2026-09-28 found this paper._ | a training-free, two-step procedure for structured output (JSON, API calls): the LM first generates an unconstrained draft, then constrained decoding runs conditioned on that draft, separating semantic planning from structural enforcement | **its draft steers a second, constrained decode that may change the output; A8's draft is only verified.** A8's draft comes from a separate classifier and is accepted or rejected token by token against plain greedy, so its output is byte-identical to greedy by construction. [Likely — matched on acronym and topic; confirm it is the intended source] |
| **SemanticSpec** — *Beyond Tokens: Semantic-Aware Speculative Decoding … by Probing Internal States*, arXiv **2602.03708** | estimates semantic probability from the target's hidden states so that semantically equivalent but token-different drafts are accepted; up to 2.7× | **it relaxes the acceptance test; A8 does not.** Accepting semantic equivalents makes output non-identical at the token level, which is precisely the guarantee A8 is built to keep (217/217, three times) |
| **US20250356258A1** — *Selective Speculative Decoding*, Microsoft Technology Licensing, filed 2024-12-02, published 2025-11-20 | per-iteration logic decides which portions of the output to speculate on and which to generate directly with the primary model | **it gates *when* to speculate with the same kind of drafter; A8 changes *what* the drafter is.** The two are orthogonal: A8's one-shot classifier draft could be gated by exactly this logic |

Taken with the table above, the honest position is unchanged: **the mechanism
is owned by DFlash and Medusa**, the acceptance test is standard, and what is
new here is only the drafter's provenance — an already-deployed discriminative
classifier over a closed label set, free at deployment time. A12 is the
evidence that this choice is not merely convenient but load-bearing on this
workload. [Likely — absence of prior art rests on 17 searches in total]

---

## 7.1 A11 — where the classifier draft is rejected

**No new inference.** The A8/A10 drafter is the flat DistilBERT classifier and
its 217 predictions are already saved in `p45_predictions_held_out_v3.csv`; the
draft is a deterministic `json.dumps` of them. Reconstructing it and tokenising
with the Qwen tokenizer reproduces `a8_speculative_percommand.csv`'s
`draft_tokens` on **217/217** commands, so the reconstruction is exact. The
verifier's output is the saved `greedy_text`. Only the tokenizer was loaded.

A BPE token can straddle two schema regions — `_C` covers the `grid_` prefix
*and* the row letter — so each rejection is attributed to the region of the
**first character at which draft and verifier text differ**, not to the token's
majority region. (Attributing by token span instead mislabels all 41 row
rejections as `loc_type`.)

| schema region | first rejections | of 217 | of the 55 rejected |
|---|---|---|---|
| **row** | **41** | 18.9 % | **74.5 %** |
| **verb** | **13** | 6.0 % | **23.6 %** |
| runtime symbol | 1 | 0.5 % | 1.8 % |
| JSON syntax | **0** | — | — |
| target | **0** | — | — |
| loc_type (`grid_` / `box_`) | **0** | — | — |
| column | **0** | — | — |
| container index | **0** | — | — |
| drafted stop token | **0** | — | — |
| *(no rejection: full draft accepted)* | 162 | 74.7 % | — |

**The classifier never loses the frame.** Zero rejections in JSON syntax, key
order, field set, target, location type, column, container index or the drafted
`<|im_end|>`: rendering the classifier's label into the LoRA's format is exact,
and the only places the two models disagree are the two the rest of this report
already identifies as the weak factors. [Certain]

Row rejections are one-directional and always corrective: draft `C` → verifier
`A` ×28, `B` → `C` ×6, `A` → `C` ×6, `B` → `A` ×1, and on **41 / 41 the
verifier's row is the ground-truth row while the draft's is not**. Verb
rejections are scattered (`reject`→`fetch` ×3, `fetch`→`pick` ×2, eight
singletons) and there the verifier is right on 9 / 13 against the draft's
11 / 13 — the one region where rejecting the draft is not reliably an
improvement.

Cost follows the position. A row rejection lands late, so those commands finish
in exactly **9 forward passes** (41/41, mean acceptance 0.658); a verb
rejection lands at the second token and costs **15.8 passes on average**
(max 25, mean acceptance 0.254). The 55 rejections split 41 cheap / 13
expensive, which is what produces the bimodal pass distribution in §7.

### Frame correct vs draft accepted

Frame = verb + target + field set; the location **value** is excluded.

| | draft accepted | draft rejected |
|---|---|---|
| frame correct (vs **ground truth**) | 158 | **45** |
| frame wrong (vs ground truth) | **4** | 10 |
| frame correct (vs **verifier output**) | 162 | 42 |
| frame wrong (vs verifier output) | **0** | 13 |

- **Frame correct but rejected anyway: 45** — 41 on the row, 3 on the verb, 1
  on the runtime symbol. The draft was structurally right and lost on one
  character of the location.
- **Frame wrong but fully accepted: 4** — commands 144, 173, 174, 176. In three
  the classifier and the LoRA agree on a verb the ground truth does not have
  (`suspend the toy car in the air`, `hold the bottle while keeping your
  gripper closed`, `clutch the toy car tightly`); in 174 (`navigate to box 1 by
  moving away from box 1`) both emit `box_1` where the truth carries no
  location. Speculative decoding reproduces the verifier exactly, so these are
  accepted *and wrong* — the mechanism cannot improve on the verifier, which is
  what the byte-identity guarantee means.
- **Against the verifier's own output the bottom-left cell is 0**: no draft
  whose frame differed from the verifier's was ever fully accepted, as the
  accept rule requires.
- On the 162 fully accepted commands the draft's location equals the truth on
  **161**; on the 55 rejected, on **12**.

---

## 7.2 A12 — a conventional small-LM drafter for the same verifier

The comparison A8 needs: **Qwen2.5-0.5B-Instruct drafting for the same
Qwen2.5-3B-Instruct + `qwen3b_lora` verifier**, same V5 prompt and chat
wrapper, same 449-token prefix KV cache on **both** models (each cropped back
to the prefix before every command), same acceptance rule, same stop handling,
same byte-identity requirement, same energy protocol. The drafter is the
off-the-shelf instruct model, not fine-tuned on this task — which is the
comparison: A8's drafter is free because the pipeline already ships it, while a
conventional drafter must be trained or taken as-is.

**γ = 4 draft tokens per iteration, fixed before the run and not swept.**
Sweeping γ on `held_out_v3` and quoting the best value would be threshold
selection on the test set, which this project rules out everywhere else. The
drafter and verifier share a tokenizer (vocab 151,665, verified on a probe
string), so draft ids need no re-tokenisation.

### Correctness first

| check | result |
|---|---|
| 0.5b-drafted speculative == plain greedy, same process | **217 / 217 PASS** |
| this run's plain greedy == A10's saved `greedy_text` | **217 / 217** |

### Acceptance

| | A12 (qwen2.5-0.5b drafter) | A8/A10 (classifier drafter) |
|---|---|---|
| token acceptance rate | **0.5410** (3,971 / 7,340) | **0.8946** (4,704 / 5,258) |
| per-command acceptance | p50 0.5357, p95 0.6562, min 0.3333, max 0.7188 | bimodal: 162 at 1.0, rest p50 0.6786 |
| commands finished in ONE verifier pass | **0 / 217** | **162 / 217** |
| verifier forward passes | 1,835 (2.88× fewer than plain greedy's 5,288) | **746 (7.09× fewer)** |
| drafter forward passes | **7,340** | 217 (one classifier call each) |
| total model forward passes | 9,175 | 963 |
| verifier passes per command | 5–11, mode 9 | 1 (162 commands) or 8–25 |

Tokens confirmed per verifier pass: **2.76** (23.4 output tokens over 8.46
passes per command). With γ = 4 and 54 % acceptance that is the arithmetic
working as designed — the 0.5B is a competent drafter in the usual sense. It
simply never produces the whole object in one shot, which is the event that
makes A8 cheap.

### Latency (n = 217, all commands included; 3 pre-run as GPU warm-up)

| arm | p50 | p95 | p99 | mean |
|---|---|---|---|---|
| prefix-cached plain greedy (this run's control) | 833 ms | 938 ms | 979 ms | 749 ms |
| **0.5b-drafted speculative (A12)** | **1,000 ms** | 1,258 ms | 1,284 ms | 982 ms |
| — of which the 0.5B drafter | 692 ms | 875 ms | 898 ms | 682 ms |
| prefix-cached classifier-drafted speculative (A10) | **58 ms** | 344 ms | 743 ms | 133 ms |

**A12 is 1.20× SLOWER at p50 than the plain greedy it is meant to accelerate,
and 17.2× slower than A10.** [Certain] The reason is per-pass cost, not
acceptance: small-batch decoding on this GPU is launch-bound, so the 0.5B costs
**20.15 ms per forward pass against the 3B verifier's 35.54 ms — a ratio of
0.57, not the ≈0.1 the speculative-decoding arithmetic assumes.** Four
sequential drafter passes (80.6 ms) to save about 1.8 verifier passes (64 ms)
is a losing trade, and the drafter is **69 % of A12's p50**. A classifier that
produces the entire draft in one ~19 ms CPU call has no such term. Run-to-run
spread on this arm is larger than A8/A10's — the first A12 run measured p50
1,101 ms against 1,000 ms here (≈10 %) — so these ratios are quoted to two
significant figures.

### Energy (GPU only, idle 12.12 W, same protocol as `energy.csv` / `a8_energy.csv` / `a10_energy.csv`)

| arm | N | wall | mean P | J/command |
|---|---|---|---|---|
| prefix-cached plain greedy | 217 | 158.5 s | 95.3 W | 60.72 |
| **0.5b-drafted speculative (A12)** | 217 | 204.3 s | **60.4 W** | **45.44** |
| prefix-cached classifier-drafted (A10) | 1085 | 129.2 s | 91.4 W | **9.45** |

A12 uses **4.8× the energy of A10** per command. It does draw 25 % less than
its own plain-greedy control while being 20 % slower: the drafter phase leaves
the GPU underutilised (60.4 W against 95.3 W), and this protocol scores
`(mean load power − median idle power) × wall / N`, which rewards an arm that
runs cool even when it runs long. Stated because latency and energy point in
opposite directions for this arm and only this arm. [Certain]

### What A12 settles

The classifier drafter's advantage is **not** that it predicts the verifier's
tokens better in the usual sense — 0.89 against 0.54 acceptance is a gap, but
acceptance is not what decides this. It is that a one-shot drafter costing
~19 ms on the CPU turns 74.7 % of commands into a **single** verifier pass,
while an autoregressive drafter pays γ sequential GPU passes per iteration on
every command and, at this model scale on this hardware, never recovers them.
**A conventional small-LM drafter is a net loss on this workload; the
classifier drafter is 17× faster and 4.8× cheaper than it, and both are exactly
lossless.** [Certain]

---

## 8. MASSIVE — Phase 7 and A5, restated

**Design as run (S-1).** One DistilBERT encoder with three heads — a 60-way
intent head, an 18-way scenario head and a K-way action head — trained jointly
with loss = CE(flat) + CE(scenario) + CE(action) (`p7_massive.py:69-83`; the
A9 fresh run uses the same recipe). "Flat" and "factorized" here are two
read-outs of one representation shaped by both losses. **Nothing in this
section is evidence about a separately trained flat model versus a separately
trained factorized model**; that experiment was not run and is not run in this
fix pass. The held_out_v3 comparison in §5 (two separately trained models,
`p2_eval.py`) is the only one that tests that.

**Router-signal labels (S-2).** Both AUROCs rank commands by the *factorized*
prediction's correctness (`fact_sym == truth`): per-factor min-confidence
0.8910, flat-head confidence 0.8889. The flat confidence scored against its own
head's correctness is 0.8729 (Part B), a different question; the comparison as
run answers "which signal better flags a wrong factorized prediction", which is
the routing question if the factorized prediction is what gets executed.

| | value | verdict (within design) |
|---|---|---|
| flat head | 0.8860 | — |
| factorized heads, free argmax | 0.8779 (2.02 % illegal) | flat > free: claimable (+0.81, p = 7e-05) |
| factorized heads, constrained to 60 legal pairs | 0.8850 (0 illegal) | vs free: claimable (+0.71, p = 9.5e-07) |
| constrained vs flat | −0.10 pts | not claimable (p = 0.711) |
| router AUROC gaps (free / constrained / Phase 7) | +0.0007 / −0.0001 / +0.0021 | all CIs span 0 |

The A9 retrain reproduced A5's three aggregates to 4 decimals; per-row flat
predictions differ on 106 / 2,974 rows (cuDNN nondeterminism), so "reproduced
exactly" holds at aggregate level only (M-5).

---

## 9. Remaining arms — the vocabulary story

**A1.4 BiLSTM.** Vocabulary from `train_v2` only (142 word types (144 table entries with <pad>/<unk>)). `upper`, `bottom`,
`back`, `front` are OOV (`top` is in `train_v2` but unused in `held_out_v3`;
row B's `middle`/`center` are in-vocabulary — so the disjunction is total for
rows A and C, not for B, M-7). Predictions are byte-identical on `held_out_v3`
and `held_out_v4c` (217/217) and it reaches 0.6852 row accuracy without seeing
the row word (B 36/36, A 22/36, C 16/36) against the factorized head's 0.3333.

> **Correction 2026-09-28.** This paragraph said "142 types". The BiLSTM's
> vocabulary is 142 word types; its embedding table has 144 entries because
> `<pad>` and `<unk>` are added (`thesis2/final/RUN_ORDER.md`, the TextCNN/MLP-BoW
> vs BiLSTM vocabulary section). The count was never wrong, only unqualified.

**Row-word occurrences by split** (whole-word, case-insensitive; *commands
containing the word / total occurrences*):

| split | n | back | front | middle | center | upper | bottom | top | lower | far | near |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `train_v2` | 230 | **0/0** | **0/0** | 38/38 | 11/11 | **0/0** | **0/0** | 1/1 | **0/0** | 27/27 | 32/32 |
| `dev_v2` | 57 | **0/0** | **0/0** | 9/9 | 1/1 | **0/0** | **0/0** | **0/0** | **0/0** | 9/9 | 5/5 |
| `held_out_v3` | 217 | 1/1 | 0/0 | 36/36 | 24/24 | **36/36** | **36/36** | 0/0 | 0/0 | 0/0 | 0/0 |
| `held_out_v4c` | 217 | **37/37** | **36/36** | 36/36 | 24/24 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |

The per-group split of v4c row accuracy (§3, point 3) shows what this costs:
**no arm maps `back`** — flat 2/36, prompted LoRA 0/36 — while both are perfect
on the `middle`/`center` commands whose wording training did contain, and the
classifier collapses to 0/24 on commands that mix a new row word with a seen
one. The vocabulary story is not "the classifier is robust"; it is "the
classifier is undisturbed on what it saw and at chance on what it did not,
while the prompted model is confidently wrong on what it did not."

Across `train_v2` and `dev_v2` combined, **`back`, `front`, `upper`, `bottom`
and `lower` occur zero times**; `top` occurs once, in `train_v2`. Training saw
rows named only by `far`/`near`/`middle`/`center`. Both held-out families
therefore rename rows A and C into words training never saw — v3 into
`upper`/`bottom`, which the V5 legend defines, and v4c into `back`/`front`,
which it does not. This is the whole of the vocabulary story in one table, and
it is why §3 reports two columns rather than one. [Certain]

**A1.2 TinyBERT-4** 0.5760, row 0.3333. **A1.1 frozen probe** 0.6498, row
0.6204. **A1.5 Flan-T5-base** 0.6175, schema validity 0.9770 (brace-free target
format because T5's SentencePiece cannot represent `{`/`}`). **P6.5
qwen2.5:0.5b SALSA, letters** v3 row 0.5278, ECE 0.0962. None separates
individually from the factorized head (§5).

**Checkpoint rule for the A1 arms and Flan-T5 (S-7).** The code loads the
*argmax* dev epoch, not the plateau-earliest epoch the previous draft said it
used; the prints now say so. Selection is `dev_v2`-only either way; whether the
two rules picked different epochs was not persisted and is unknown.

**A1.7 — withdrawn (B-3).** The field-level comparator (`a1_rest.py`
`LOC_LEGEND` prompt, gpt-4o-mini asked for the whole 14-way symbol) returned a
grid cell for every container command: `box_1 → grid_A1` (12/12), `box_2 →
grid_B2` (12/12), `box_3 → grid_C3/B3` (12/12), `max_reach → grid_A*` (12/12),
`nearest_free_cell` right 12/12. On the 60 non-grid commands it scored 0.200
while the factor-level arm never calls the cloud there (local heads 1.000). On
the 108 grid commands — the only place both arms escalate — the whole-location
query scored **1.000** and the row-only query **0.704**. The reported +9.52 was
therefore an artefact of the comparator's prompt plus the factor arm's free
local heads. The prompt is not fixed and the arm is not re-run in this pass;
the comparison is withdrawn with no claim in either direction. The 336
gpt-4o-mini calls it made stand in §14.

**SALSA digits mapping — withdrawn (B-4).** In the Qwen2.5 tokenizer `' A'`,
`' B'`, `' C'` are single tokens (362/425/356) but `' 1'` is two tokens
(`[220, 16]`); `salsa_row.py` silently fell back to the bare digit and read its
logit at the position after `Row:`, where the model's next token is a space.
Every `*_digits_*` number (P6 tables, `a1_salsa05b.csv` digits rows,
`metrics_all.csv` digits arms, PHASE_4's "label-token mapping matters" table)
measures that misalignment, not the mapping, and is withdrawn along with the
mapping-sensitivity claim. Letters numbers stand; the headline cascade uses
letters.

---

## 10. A6 — self-check, restated

**Leakage.** `a6_selfcheck.py` is a *heuristic*: a `held_out` reference counts
as train-time only if its enclosing function's name contains one of eight
strings (S-8); `run_flant5`, which trains and evaluates in one function, is not
on the list, and module-level code is classed as evaluation. Its "63
references, 0 train-time" therefore does not verify leakage by itself. **The
verification is Part B's code trace** (`thesis2_review.md` §4): every training
path (`p2_train_factorized.py`, the three `a1_arms.py` arms, `run_flant5`,
`p7_massive.py` and the A9 retrain) opens only `train_v2`/`dev_v2` (or the
MASSIVE train/validation splits) before selection, no helper loads a held-out
file indirectly, threshold τ_A is a constant and τ_B is dev-only, and the
exact-command overlap train_v2 ∩ held_out_v3 and dev_v2 ∩ held_out_v3 is 0.

**Recomputation: 9/9 exact** (0.6083, 0.3333, 0.7465, 0.6204, 0.8433, 0.9401,
always-escalate pipeline rebuilt 0.8433 — labelled "cascade" in A6's own
output, which is the B-2 mislabelling, not a second result — MASSIVE AUROC
0.8910 / 0.8889), and Part B reproduced
the same nine plus the row-factor and INT4 figures with independent code.

Denominators: fresh-pass arms n = 217, no warmup row; the V5 classifier and
LoRA cells are n = 216 (`command_id 0` was the warmup); the V5 cloud cell has
217 commands (command 0 keeps 4 of its 5 repeats). **A8/A10 measure all 217
commands; the 3 pre-run commands are included** (S-5).

---

## 11. The INT4 gate

**FAIL.** Action match identical at 0.6083, CI [−1.84, +1.84] pts. Real
execution verified: 43/43 Linear layers changed; ≤ 15 distinct values per
output channel; codes integral in [−8, 7]; no head has a zero logit delta.
Behaviour changes even though accuracy does not: **5 verb flips and 8–9 row
flips** (9 with the gate's FP16 quantiser arithmetic, 8 when the same scheme is
computed in FP32 — S-9), row factor 0.3333 → 0.3426, command-level b = 2, c = 2.

Weights only, per-output-channel symmetric INT4 (`scale = max|W_o| / 7`,
`q = clamp(round(W / scale), −8, 7)`), activations FP16, weights dequantised
before every matmul — no INT4 kernel, so no latency benefit (3.14 → 3.22 ms
p50).

**One size definition, used everywhere in this report and in
`memory_footprint.csv`**: whole model, 66,381,336 parameters, weights at the
stated precision, per-channel fp32 scales included.

| | FP16 | INT4 | ratio |
|---|---|---|---|
| **whole model, resident** | **132.76 MB** | **69.20 MB** | **1.92×** |
| quantized Linear layers (42.49 M weights + 33,024 fp32 scales) | 84.97 MB | 21.41 MB | 3.97× |
| embeddings (23.84 M params, 36 %) | 47.67 MB | 47.67 MB | unquantized |
| LayerNorm + biases | 0.12 MB | 0.12 MB | unquantized |

The headline compression is 1.92×, not 4×. CascadeCNN's Low Precision Unit
quantizes weights *and* activations on an FPGA datapath — a different regime;
its 14.38 % / 18.65 % figures could not be verified from this machine.

---

## 12. What the contribution is, after review

1. **Factorization measurably hurts on the robot task** — −13.82 pts
   (claimable, two separately trained models). On MASSIVE the −0.81 is between
   two heads of one jointly trained model and says nothing beyond that design.
   [Certain]
2. **Within that MASSIVE design, the per-factor router signal is
   indistinguishable from the flat confidence**: four AUROC CIs contain zero
   with half-width ≈ ±0.005. [Certain for the design run]
3. **Confidence-gated per-factor escalation does not beat the flat head at the
   pre-registered operating point** (0.7327 vs 0.7465, p = 0.77); it recovers
   the factorization loss (+12.44 vs the factorized head, p = 1.5e-08) and no
   more. Delegating *every* grid row to the LoRA (0.8433) beats the flat head
   **nominally** (p = 0.0154, p_holm = 0.257 — it does not survive family-wise
   correction), is not routing, and is dominated by running the LoRA outright
   (0.9539).
   [Certain]
4. **Factor-level vs field-level escalation is untested** — A1.7 is withdrawn
   because its comparator was broken by its prompt. [Certain]
5. **The ensemble buys nothing measurable** over the LoRA (p = 1.0). [Certain]
6. **Execution is not a confound** — zero MuJoCo failures in 1,301 pairs.
   [Certain]
7. **The classifier is a free one-shot drafter for the generative rung**: with
   prefix caching the LoRA arm runs ≈16× faster at p50 and ≈7.5× cheaper in
   energy with output byte-identical to its own greedy decode, which scores
   0.9539 on `held_out_v3` (V5's Ollama Q4 cell: 0.9583; not separable).
   Verified 217/217 three times, once in shuffled order. [Certain]
   - **The draft never fails on structure** — of 55 rejections, 41 are the row
     factor and 13 the verb; zero are JSON syntax, target, column, location
     type, container index or the stop token (§7.1). The speculative arm's
     failure mode is the project's own failure mode. [Certain]
   - **A conventional small-LM drafter does not work here.** Qwen2.5-0.5B
     drafting for the same verifier is exactly lossless (217/217) at 0.54
     acceptance, and is 1.20× *slower* than plain greedy and 17.2× slower and
     4.8× more energy-hungry than the classifier drafter, because a 0.5B
     forward pass costs 0.57 of a 3B one on this hardware (§7.2). [Certain]
8. **AUROC = 1.000 on `held_out_v3` is an upper bound** driven by the A/C row
   vocabulary disjunction; that split predates the factorization idea.
   [Certain]
9. **τ_B degenerated to 0.00** exactly as pre-registered. [Certain]

---

## 13. Honest limitations

- `held_out_v3` is 217 commands; the 95 % CI on a proportion near 0.75 is
  roughly ±6 points. Most arms in §3 are mutually indistinguishable.
- A4 was skipped; no evidence isolates which clause of the legend carries the
  row semantics.
- P4.1–4.2 were never run (gate failed).
- `held_out_v3` has been reused for V5, Phases 1–7, A1, A3, A8, A9, A10 and
  the pre-registration itself post-dates the P2 held-out confidence analysis.
  Nothing here is a fresh held-out estimate.
- The cascade's latency and energy were never measured per command; only the
  per-call components were (§3).
- The MASSIVE experiment does not separate flat from factorized *models* (§8).
- Cloud energy is not measurable from this machine.
- The A1 arms' selected epoch (argmax vs plateau) was not persisted.
- The GPU idle-power baseline was not fully idle (M-1).

---

## 14. Blocked, skipped, deleted, and written outside `thesis2/`

No tool call was denied by a permission prompt in any session. [Certain]

A4 skipped on instruction. Corrections carried forward from earlier drafts:
TinyBERT downloads fine with verification on; the TLS interception is real but
intermittent (`CERTIFICATE_VERIFY_FAILED` on huggingface.co, `ECONNRESET` on
arxiv.org; verification never disabled).

| item | what happened | effect |
|---|---|---|
| `AmazonScience/massive`, `qanastek/MASSIVE` | `datasets` 4.x removed script loaders | used `mteb/amazon_massive_intent` |
| bitsandbytes INT4 | `Linear4bit` needs a `from_pretrained` tree | manual INT4, verified to execute |
| No CUDA torch in `thesis`/`train` | CPU-only builds | created `thesis2gpu` |
| No `gymnasium`/`mujoco` in `thesis2gpu` | A3 needs MuJoCo | ran A3 in the pre-existing `thesis` env; nothing installed into it |
| A8 first run | 55/217 — accepted `<|im_end|>` appended | fixed; 217/217 |
| A9 first run | A1.7 used qwen3b-lora for the factor level | fixed to `cloud_row`; arm since withdrawn (B-3) |
| A9 MASSIVE constrained | per-row constrained decodes never persisted | retrained once inside A9 |
| Flan-T5 first attempt | 0.0000 — T5 cannot tokenise `{`/`}` | brace-free target format |
| A1.7 field-level prompt | never returns a container symbol | arm withdrawn, not re-run |
| SALSA digits mapping | `' 1'` is two tokens; wrong-position logits | all digits numbers withdrawn |
| P4.1–4.2 | not run | gate failed |

Deletions — all under `thesis2/`: 14 checkpoint directories (3,728 MB) written
by the factorized training script minutes earlier; `epoch_11` kept.

Written outside `thesis2/`: `D:\condaenvs\thesis2gpu` (~5,100 MB); HF caches
for TinyBERT (~60 MB) and Flan-T5-base (~990 MB); `mteb___amazon_massive_intent`
(1.4 MB); 4 dirs in `~\.conda\pkgs`; the review at
`C:\Users\USER\thesis2_review.md` and its scripts in the session scratchpad.
`Qwen/Qwen2.5-3B-Instruct` was already cached.

OpenAI API calls: 434 + 336 = 770 `gpt-4o-mini` calls, ≤ 8 tokens each (the
336 A1.7 calls produced a withdrawn arm). A8, A9, A10 and this fix pass made no
billed network calls.

---

## 15. Files written and changed in this fix pass

**Changed in this pass (text only; no logic, nothing re-run):**
`thesis2/scripts/a1_arms.py` and `a1_rest.py` (checkpoint prints/docstring
now state that the argmax dev epoch is loaded); `a8_speculative.py` and
`a10_prefix_cache.py` (warm-up prints/docstrings/summary key: pre-run, not
excluded; the saved JSONs from the earlier runs still carry the old
`warmup_excluded` key and mean the same thing); `p45_full.py` (docstring says
latency/energy are modelled constants; hand-typed memory table uses the §11
definition); `thesis2/out/memory_footprint.csv` (rewritten to the §11
definition); this file.

**Added after the fix pass:** `thesis2/scripts/a13_v4c_prompted.py`,
`thesis2/out/a13_v4c_prompted.csv`, `a13_v4c_lora_percommand.csv`,
`a13_v4c_significance.csv`, `a13_v4c_cloud_significance.csv` (the v4c
prompted-arm evaluation, its two significance rows and the four cloud-cascade
rows), plus:
`thesis2/scripts/a12_smalllm_drafter.py`, `thesis2/out/a11_rejection_positions.csv`,
`thesis2/out/a12_smalllm_drafter_percommand.csv`,
`thesis2/out/a12_smalllm_drafter_summary.json`, `thesis2/out/a12_energy.csv`,
and §§7.1–7.2 plus the related-work table in §7. A11 ran no model (tokenizer
only); A12 is the one new arm and made no billed network calls. The A11
analysis script lives outside the repository in the session scratchpad
(`a11_reject.py`) and its output CSV is the artefact of record.

**Not changed:** every per-command CSV/JSON output of A1–A10, every model, every
prediction file, `metrics_all.csv`, `metrics_all_v2.csv`. **Added:** `metrics_all_v2.csv` — the
same 50 rows with `status`/`reason` columns; withdrawn rows blanked. This is
the only file thesis tables may be built from (see the line under the title).

**`thesis2/scripts/` (31)** — `p1_attribution`, `p1_verify`, `p1_digest`,
`p2_train_factorized`, `p2_eval`, `p2_diagnose_row`, `p2_row_confidence`,
`p3_quantization_gate`, `p3_verify_int4`, `p3_verify_int4_supervised`,
`salsa_row`, `p45_collect`, `p45_sweep`, `p45_full`, `p5_baselines`,
`p5_rules_firewalled`, `p6_calibration`, `p7_massive`, `p_energy`,
`consolidate_metrics`, `a1_download_probe`, `a1_arms`, `a1_bilstm_check`,
`a1_rest`, `a1_flant5_diag`, `a2_metrics_all`, `a3_mujoco_e2e`,
`a5_massive_constrained`, `a6_selfcheck`, `a8_speculative`, `a9_stats`,
`a10_prefix_cache`.

**`thesis2/out/`** — phase summaries (unrevised; superseded where they
conflict with this file), `THRESHOLD_PREREGISTRATION.md`, and ~63 CSV/JSON
outputs.

**`thesis2/figures/`** — 4 analysis figures + 36 confusion matrices.

**`thesis2/models/distilbert_factorized/`** — `epoch_11/` plus label maps,
history and selection record.
