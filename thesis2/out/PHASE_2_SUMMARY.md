> **SUPERSEDED IN PART — do not quote from this file without checking `FINAL_REPORT.md` §0.**
> Nothing in this file is withdrawn, but `FINAL_REPORT.md` is the source of
> record and `metrics_all_v2.csv` is the only file thesis tables may be built from.

# Phase 2 — Factorized location head

Run 2026-09-21, CPU (`thesis` env, torch 2.12.1+cpu). Neither conda env has a
CUDA build, so Phase 2 ran on CPU — which is also the right substrate, since the
existing classifier runs in-process on CPU single-threaded and its 17.9 ms
figure is a CPU figure.

Scripts: `p2_train_factorized.py`, `p2_eval.py`, `p2_diagnose_row.py`,
`p2_row_confidence.py`.

## P2.1 — training

Same backbone, same seed (20260820), lr 3e-5, batch 16, 15 epochs, warmup 10,
dropout 0.1, max_len 64, weight decay 0.01, and the same checkpoint rule
(earliest epoch within 1 dev row of best). Trained on `train_v2` only, selected
on `dev_v2` only; `held_out_v3` is never opened by the training script.

**Parameter count is identical to the flat head: 66.3813 M both, delta +0.**
3+3+3+3+2 = 14 output units, the same 14 the flat head has. Any accuracy
difference therefore cannot be attributed to capacity. [Certain]

Round-trip `factorize → reconstruct` over all 14 symbols: **EXACT**. [Certain]

Supervised train rows per factor: `loc_type` 132, `row` 85, `col` 85,
`container_which` 30, `runtime_which` 17.

Selected **epoch 11**, dev 54/57 = 94.7%. Plateau was epochs 11–15; argmax also
epoch 11.

## P2.2 — the answer: factorization FAILS its acceptance criterion

The brief set the test: *"Factorization must not cost accuracy."* **It costs
13.82 points of action match.** [Certain]

n=217 commands, fresh single pass, no warmup row to drop. (The published V5
classifier cell is n=216 = 0.7500 because command_id 0 was its warmup; the
n=217 flat figure recomputed here is 0.7465.)

| metric | n | flat | factorized | delta |
|---|---|---|---|---|
| **action match** | 217 | **0.7465** | **0.6083** | **−13.82** |
| field verb | 217 | 0.9355 | 0.9401 | +0.46 |
| field target | 153 | 0.9608 | 0.9477 | −1.31 |
| **field location** | 168 | **0.7560** | **0.5714** | **−18.45** |
| joint all three | 126 | 0.7460 | 0.5714 | −17.46 |
| factor loc_type | 168 | 1.0000 | 1.0000 | 0.00 |
| **factor row** | 108 | **0.6204** | **0.3333** | **−28.70** |
| factor col | 108 | 0.9444 | **1.0000** | +5.56 |
| factor container_which | 36 | 1.0000 | 1.0000 | 0.00 |
| factor runtime_which | 24 | 1.0000 | 1.0000 | 0.00 |
| reject recall | 12 | 0.5833 | 0.7500 | +16.67 |
| latency p50 (ms, CPU) | 217 | 29.19 | 28.59 | −0.60 ms |

The column factor became **perfect**. Everything lost is the row factor.
(`delta_pts` in the CSV is `100 × difference`, which is meaningful for
accuracies and meaningless for the three latency rows — read those from the
`flat`/`factorized` columns.)

## The amendment: updown bucket, confirmed and then some

**Prediction refuted in the harsher direction.** You expected factorization not
to *fix* row inversion. It does not merely fail to fix it — it makes it total.

| bucket | n | flat | factorized |
|---|---|---|---|
| updown action match | 72 | 0.4306 | **0.0000** |
| neither action match | 96 | 1.0000 | 1.0000 |
| nolocation action match | 49 | 0.7143 | 0.7347 |
| **updown factor row** | 72 | 0.4306 | **0.0000** |
| updown factor col | 72 | 0.9167 | **1.0000** |

**0 of 72.** [Certain]

## Why — and it is not a bug

Row-head confusion, `held_out_v3`:

```
FACTORIZED            FLAT (symbol's row component)
A -> A:0  B:30 C:6    A -> A:7  B:1  C:28
B -> A:0  B:36 C:0    B -> A:0  B:36 C:0
C -> A:0  B:36 C:0    C -> A:6  B:6  C:24
```

The factorized head **collapses to B** — the middle row — for every command
whose row wording it does not recognise. The flat head *inverts* (A→C) but
lands on the right row often enough to score 0.62.

On data it has seen the factorized head is flawless: **train 56/56 farnear,
29/29 neither; dev 100%.** The failure is entirely out-of-vocabulary. [Certain]

The cause is a **total vocabulary disjunction** on the row axis: [Certain]

| split | farnear | updown | neither |
|---|---|---|---|
| train_v2 | 56 | **0** | 29 |
| dev_v2 | 14 | **0** | 7 |
| held_out_v3 | **0** | 72 | 36 |

Training contains no `upper`/`bottom` at all; held-out contains no `far`/`near`
at all. The row factor is tested entirely on vocabulary it has never seen. A
narrow 3-way head has nothing to fall back on but its prior; the 14-way flat
head hedges through the rest of the symbol structure.

This is also the deeper reading of the whole V5 row/column asymmetry: **every
model's row problem is a vocabulary problem**, which P1.3 already showed
(classifier updown 0.4366 vs neither 1.0000).

## The result that rescues the contribution

Row-head confidence on `held_out_v3`:

| bucket | n | accuracy | mean conf | median | min |
|---|---|---|---|---|---|
| neither | 36 | 1.0000 | 0.9528 | 0.9523 | **0.9155** |
| updown | 72 | 0.0000 | 0.6008 | 0.5507 | 0.3832 |

**Correct predictions never fall below 0.9155. Wrong predictions never rise
above 0.8818. The two distributions do not overlap — AUROC = 1.000.** [Certain]

**NO OPERATING POINT IS QUOTED FROM THIS.** An earlier revision of this file
singled out a threshold that had been read off the held-out sweep; that is
threshold selection on the test set and it has been removed. The separation
statistic above (non-overlapping supports, AUROC = 1.000) is a property of the
whole curve and does not depend on choosing a point.

Thresholds actually used anywhere downstream are fixed a priori by the rules in
`thesis2/out/THRESHOLD_PREREGISTRATION.md`, written before any P4/P6/P7 arm was
run. Full sweeps are reported for everything.

So: **factorization is a failure as an accuracy mechanism and a near-ideal
failure detector.** The per-factor confidence does the job the cascade needs —
it identifies precisely which commands need help and which factor needs it.
That is the condition P4.5 requires.

Caveat, and it matters: this separation is partly an artifact of the *total*
vocabulary disjunction. The head is confident on wording it saw and unconfident
on wording it did not. With partial overlap the separation would be less clean,
so AUROC = 1.000 should be read as an upper bound, not a general property.
[Likely]

Note also the tension with calibration: row ECE is **0.4163**, the worst of any
factor — the head is badly calibrated (mean confidence 0.60 against accuracy
0.00 on updown) while ranking perfectly. Discrimination and calibration are
different properties and this head has one without the other. [Certain]

## Risk-coverage — the flat head is still the better selective signal overall

| coverage | factorized row | flat single location conf |
|---|---|---|
| 100% | 0.6667 | 0.2440 |
| 80% | 0.5814 | 0.1493 |
| 50% | 0.3333 | 0.0238 |

On absolute selective risk the flat head wins, because its base error rate is
far lower. The factorized row confidence wins only on *targeting*: it says
which factor is broken, which the flat head's single confidence cannot.

## Files written

`thesis2/models/distilbert_factorized/` (epoch_11 kept, 14 others deleted,
3,728 MB freed; `label_maps.json`, `history.json`, `selected.json`),
`thesis2/out/factorized_fp16.csv`, `thesis2/out/factorized_vs_flat.csv`.
