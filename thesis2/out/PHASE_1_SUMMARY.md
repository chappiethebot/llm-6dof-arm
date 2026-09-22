> **SUPERSEDED IN PART — do not quote from this file without checking `FINAL_REPORT.md` §0.**
> Nothing in this file is withdrawn, but `FINAL_REPORT.md` is the source of
> record and `metrics_all_v2.csv` is the only file thesis tables may be built from.

# Phase 1 — Attribution

Run 2026-09-21. Offline; no model called. Read-only over the existing tree.

Scripts: `thesis2/scripts/p1_attribution.py`, `p1_verify.py`, `p1_digest.py`.

## Conventions

- **Warmup excluded everywhere.** 12,152 raw rows → 12,116 after dropping 36
  warmup rows (one per cell). 31 local cells are n=216 commands; 5 cloud cells
  are n=217.
- **Dedupe** on `(dataset, model, strategy, command_id, run_index)`, keep last.
  **Zero duplicates found.** [Certain]
- **Per-command**: mean over commands of the mean over that command's repeats.
  Local cells have one repeat post-warmup, so this equals per-row for them.
  Confusion matrices pool repeats (they describe a distribution, not a rate).
- `strict_match` from `llm/score.py` is authoritative for action match. Field
  and factor correctness are recomputed from `pred_json`/`truth_json`.

## Correction to the brief's dataset composition

The brief states `held_out_v3` is "132 grid-cell, 36 container, 49
no-location". Measured directly from `data/held_out_v3.jsonl`: [Certain]

| bucket | n |
|---|---|
| grid cell (`grid_A1`..`grid_C3`) | **108** |
| runtime symbol (`max_reach`, `nearest_free_cell`) | **24** |
| container (`box_1..3`) | 36 |
| no location | 49 |
| total | 217 |

132 = 108 + 24. The brief's figure merges grid cells with runtime symbols.
Row/column factors are only defined on the 108, so every row/col table here is
n=107 (local, warmup excluded) or n=108 (cloud).

## P1.2 — the factorization evidence

**Row is harder than column in 25 of 36 cells. Median gap −7.48 points, worst
−31.77.** 11 of 36 cells have row ≥10 points worse. [Certain]

| cell | n | row | col | row−col |
|---|---|---|---|---|
| classifier:distilbert | 107 | 0.6262 | 0.9439 | **−31.77** |
| qwen2.5:3b / cot | 107 | 0.4673 | 0.7477 | −28.04 |
| llama3.2:3b / zero_shot | 107 | 0.6075 | 0.8505 | −24.30 |
| qwen2.5-3b-lora / rag | 107 | 0.7383 | 0.8972 | −15.89 |
| gpt-4o-mini / role_based | 108 | 0.9500 | 1.0000 | −5.00 |
| qwen2.5-3b-lora / zero_shot | 107 | 1.0000 | 1.0000 | 0.00 |
| llama3.2:3b / rag | 107 | 0.7664 | 0.6542 | +11.22 |

**Eleven cells show a gap of exactly 0.00, which needed checking.** Two causes,
both benign for the argument: [Certain]

- `qwen2.5-3b-lora/zero_shot` has **zero grid-cell errors** — the gap is vacuous.
- `gpt-4o-mini/zero_shot` has 29 grid errors split **12 row-only / 12 col-only /
  0 both / 5 off-grid** — exactly balanced, not non-decomposable. 83% of its
  errors are still single-factor.

### Decomposability — the number that matters for Idea 1

Pooled over all 36 cells, 1,997 grid-cell errors: [Certain]

| | n | share |
|---|---|---|
| row only | 553 | 27.7% |
| col only | 197 | 9.9% |
| both axes | 238 | 11.9% |
| off-grid (predicted a non-cell) | 1,009 | 50.5% |

Off-grid is dominated by the collapsed `cot` cells. **Restricted to errors where
the model did predict a grid cell, 750 of 988 = 75.9% have exactly one factor
wrong.** Row-only outnumbers col-only **2.8:1**.

For the classifier specifically — the model Phase 2 builds on — the asymmetry is
near-total: **34 row-only, 0 col-only, 6 both, 0 off-grid.**

## P1.5 — confusion signatures

The classifier's 40 location errors: **33 (82.5%) are row axis inversion —
A↔C mirrored with the column preserved.** 1 other row-only, 6 both-axes, zero
column-only, zero container leakage. [Certain]

`gpt-4o-mini/role_based`: all 27 location errors are row-only **and** all 27
land on `grid_B2` — simultaneous row error and B2 anchoring.
`gpt-4o-mini/rag`: 12 errors, 9 of them runtime-symbol confusion, not grid.

## P1.1 — TIDE

Mean accuracy points recoverable, averaged over 36 cells, brief order:

| error type | pts | | error type | pts |
|---|---|---|---|---|
| structural | 30.86 | | verb_only | 0.81 |
| multi_field | 12.46 | | target_only | 0.27 |
| **loc_row** | **4.04** | | loc_container_grid | 0.19 |
| loc_both | 2.07 | | loc_runtime | 0.18 |
| **loc_col** | **1.61** | | loc_anchor_b2 | 0.00 |

Structural dominates because several `cot` cells emit unparseable output; it is
near zero for the classifier and the strong cells. **Among location errors,
`loc_row` is 2.5× `loc_col`.**

Order sensitivity, as the brief warned: moving `loc_anchor_b2` ahead of the
row/col split moves 1.50 pts into it, taken from `loc_row` (4.04→3.33),
`loc_col` (1.61→1.14) and `loc_both` (2.07→1.81). Both orderings are in the CSV
under an `order` column.

## P1.3 — vocabulary buckets

`held_out_v3` contains **no `farnear` commands at all**. Every row-axis command
uses `upper`/`bottom`. Buckets present: `updown` 71–72, `neither` 96,
`nolocation` 49. [Certain]

| cell | updown | neither | nolocation |
|---|---|---|---|
| classifier | **0.4366** | 1.0000 | 0.7143 |
| gpt-4o-mini / rag | 1.0000 | 0.9750 | 0.7755 |

The classifier's entire deficit is the `updown` bucket — a 56-point drop against
`neither`, on the axis whose training vocabulary (`far`/`near`) never appears in
the held-out set.

## P1.4 — confident-wrong vs unconfident-wrong

Location error rate by location-head margin decile, n=217: [Certain]

| margin | n | errors | rate |
|---|---|---|---|
| 0.0–0.1 | 69 | 20 | 0.290 |
| 0.1–0.2 | 27 | 10 | 0.370 |
| 0.2–0.3 | 17 | 5 | 0.294 |
| 0.3–0.4 | 14 | 5 | 0.357 |
| 0.4–0.5 | 20 | 2 | 0.100 |
| 0.5–0.8 | 44 | 1 | 0.023 |
| 0.8–1.0 | 26 | 1 | 0.038 |

**Only 1 high-margin (>0.8) location error; 35 low-margin (<0.3) errors.** The
confidence signal is informative and the errors are concentrated where
confidence is low. That is the precondition a confidence-gated cascade needs.
[Certain]

## P1.6 — calibration of the existing flat classifier

| head | n | accuracy | mean conf | ECE | Brier |
|---|---|---|---|---|---|
| verb | 217 | 0.9355 | 0.7747 | 0.1613 | 0.0790 |
| target | 153 | 0.9608 | 0.9457 | **0.0151** | 0.0398 |
| location | 168 | 0.7560 | 0.5685 | **0.1874** | 0.1699 |

Verb and location are **under**-confident (mean confidence below accuracy).
Target is well calibrated. The location head is both the least accurate and the
worst calibrated — it is the bottleneck on both axes.

## Files written

`thesis2/out/`: `tide_contributions.csv`, `factor_accuracy.csv`,
`vocab_buckets.csv`, `confidence_split.csv`, `confident_wrong_top20.csv`,
`confusion_signatures.csv`, `calibration_baseline.csv`.
`thesis2/figures/`: `tide_contributions.png`, `factor_accuracy.png`,
`vocab_buckets.png`, `calibration_baseline.png`, `confusion/` (36 matrices).
