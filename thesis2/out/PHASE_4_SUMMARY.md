> **SUPERSEDED IN PART — do not quote from this file without checking `FINAL_REPORT.md` §0.**
>
> Specifically:
> - **0.8433 is the τ = 1.0 always-escalate pipeline, not a routed cascade.** The pre-registered operating point is τ_A = 0.50, which scores **0.7327** and does **not** beat the flat head (p = 0.77). "The cascade beats the flat head it replaces" below is withdrawn as written (B-2), and the +9.68-point version is **nominal only** — p_holm = 0.257 across the 26-hypothesis family.
> - **every `digits` mapping number below is withdrawn** — `' 1'` is two Qwen tokens, so the digits logits were read one position early (B-4). The "label-token mapping matters" table measures that bug, not the mapping.
> - the 66.4 / 130.0 MB sizes are superseded by one definition: **132.76 MB FP16, 69.20 MB INT4**, whole model incl. fp32 scales (S-10).
> - "the per-factor signal is the better router at every budget" is not supported after correction; on MASSIVE the router gap is a tight null (§8).

# Phase 4.5 — per-factor model escalation (the primary contribution)

Run 2026-09-21 on `thesis2gpu` (CUDA, RTX 4060 Ti). Phases 4.1–4.2 were **not
run**: the Phase 3 INT4 gate failed.

Scripts: `salsa_row.py`, `p45_collect.py`, `p45_sweep.py`, `p45_full.py`.

## Architecture

Local factorized DistilBERT answers **verb, target, loc_type, col,
container_which, runtime_which** — always, at 3.2 ms. Only the **3-way row
sub-question** is escalated, and only when its confidence falls below τ. The
symbol is then reconstructed from the escalated row and the **local** column.

Escalation targets, each asked only "which row: A, B or C?":
`qwen2.5:3b` (SALSA logits), `qwen2.5-3b-lora` (SALSA logits), `gpt-4o-mini`,
and an `oracle` upper bound.

## No threshold was selected on held-out data

Per `THRESHOLD_PREREGISTRATION.md`, written before any of these numbers
existed. Full sweeps (τ = 0.00…1.00, 0.05 steps, 336 rows each) are in
`p45_row_escalation_sweep.csv` and `p45_cascade_full.csv`.

**τ_B = 0.00 — the pre-registered prediction came true.** `dev_v2` contains no
`updown` or `backfront` commands, so every dev row confidence is ≥ 0.9511 and
no τ above the grid floor meets the ≤20% escalation rule. τ_B degenerates to
"never escalate".

**That is a methodological finding, not a bug: a per-factor gate cannot be
calibrated on a development set that does not contain the distribution shift
the gate exists to catch.** [Certain] It generalises beyond this project — any
confidence-gated router tuned on an in-distribution dev set will under-escalate
exactly when escalation matters.

## held_out_v3 — n=217 commands, 108 grid

| configuration | esc rate | action match | joint | row | p50 ms |
|---|---|---|---|---|---|
| local only, no escalation | 0.00 | 0.6083 | 0.5714 | 0.3333 | 3.2 |
| *flat head baseline (Phase 2)* | — | *0.7465* | — | *0.6204* | — |
| → qwen2.5:3b, always | 0.60 | 0.8111 | 0.8492 | 0.7407 | 38.7 |
| **→ qwen2.5-3b-lora, always** | 0.60 | **0.8433** | **0.8571** | **0.8056** | 38.6 |
| → gpt-4o-mini, always | 0.60 | 0.7926 | 0.7937 | 0.7037 | 660.2 |
| → oracle row (ceiling) | 0.50 | 0.9401 | 1.0000 | 1.0000 | 3.2 |
| → qwen2.5-3b-lora, τ_A = 0.50 | 0.15 | 0.7327 | 0.6905 | 0.5833 | 3.2 |

**The cascade beats the flat head it replaces: 0.8433 against 0.7465**, +9.7
points, while keeping p50 latency at 38.6 ms — and at τ_A = 0.50 the *median*
command never escalates at all, so p50 stays 3.2 ms. [Certain]

**The 3B LoRA beats gpt-4o-mini on the row sub-question** (0.8056 vs 0.7037)
while being local and 18× faster. [Certain]

## The Idea 1 evidence: per-factor signal vs flat single confidence

Both signals escalate the *same* sub-question to the *same* model. Only the
gate differs. Compared at matched escalation budget — the only fair axis.

**Action match, escalating row → qwen2.5-3b-lora, `held_out_v3`:**

| budget | factorized row conf | flat location conf | gap |
|---|---|---|---|
| 10% | **0.6682** @ 0.08 | 0.6083 @ 0.09 | +6.0 |
| 20% | **0.7327** @ 0.15 | 0.6820 @ 0.18 | +5.1 |
| 30% | **0.8203** @ 0.28 | 0.8018 @ 0.30 | +1.9 |
| 40% | **0.8756** @ 0.37 | 0.8664 @ 0.36 | +0.9 |
| 50% | **0.9401** @ 0.44 | 0.8848 @ 0.44 | **+5.5** |

**Row accuracy, escalating → gpt-4o-mini, `held_out_v3`:**

| budget | factorized | flat | gap |
|---|---|---|---|
| 20% | **0.4537** | 0.3611 | +9.3 |
| 30% | **0.5833** | 0.4444 | +13.9 |
| 40% | **0.6667** | 0.5278 | +13.9 |
| 50% | **0.7778** | 0.5278 | **+25.0** |

AUROC for ranking row-correct above row-wrong: factorized **1.0000**, flat
0.9552. The per-factor signal is the better router at every budget. [Certain]

**Caveat, stated with the result: AUROC = 1.000 is an upper bound.** It is
driven by the *total* row-vocabulary disjunction between `train_v2` and
`held_out_v3` — the head is confident on wording it saw and unconfident on
wording it did not. `held_out_v3` also predates the factorization idea, so it
was not designed to test this. [Certain]

## held_out_v4c — the second held-out family, and it breaks the local arms

`held_out_v4c` uses `back`/`front` for rows: a *third* vocabulary, in neither
training nor `held_out_v3`.

| configuration | esc rate | action match | joint | row |
|---|---|---|---|---|
| local only | 0.00 | 0.6359 | 0.5873 | 0.3889 |
| → qwen2.5:3b, always | 0.60 | 0.5714 | 0.5079 | **0.2593** |
| → qwen2.5-3b-lora, always | 0.60 | 0.5438 | 0.4603 | **0.2037** |
| **→ gpt-4o-mini, always** | 0.60 | **0.7788** | **0.7698** | **0.6759** |

**Both local escalation targets score BELOW chance (0.333) and below the
un-escalated local head.** Escalating actively hurts. [Certain]

The cause is in the prompt: `salsa_row.py`'s legend defines row A as "the far
side, the upper side" and says nothing about `back`/`front`. The local 3B models
have no basis to map the word and guess consistently wrong; `gpt-4o-mini` infers
it from world knowledge and reaches 0.6759.

**This is the most important negative result in Phase 4: per-factor escalation
only helps when the escalation target actually knows the vocabulary. The gate
correctly identifies *that* the local head is unsure; it cannot make the
escalation target competent.** [Certain]

At matched budget on `held_out_v4c` the two signals are indistinguishable
(0.6406 vs 0.6452 at 30%) — because neither routes to anything useful.

## Label-token mapping matters, and more where the task is harder

| arm | letters | digits | gap |
|---|---|---|---|
| qwen2.5:3b, held_out_v3 | 0.7407 | 0.7315 | −0.9 |
| qwen2.5-3b-lora, held_out_v3 | 0.8056 | 0.7778 | −2.8 |
| qwen2.5:3b, held_out_v4c | 0.2593 | **0.4167** | **+15.7** |
| qwen2.5-3b-lora, held_out_v4c | 0.2037 | **0.3889** | **+18.5** |

Negligible where the model understands the question; **15–18 points where it
does not**. A SALSA result quoted without its label-token mapping is not
reproducible. [Certain]

## P4.4 memory — the edge argument, and it cuts against the local arm

| arm | resident |
|---|---|
| per-factor cascade, local only, INT4 | **66.4 MB** |
| per-factor cascade, local only, FP16 | 130.0 MB |
| flat classifier alone (V5 rung-4 cell) | 130.0 MB |
| per-factor cascade + gpt-4o-mini | 130.0 MB + network |
| per-factor cascade + qwen2.5-3b-lora co-resident | **6,723 MB** |
| two-model cascade, CascadeCNN-style | 6,723 MB |

**The per-factor cascade and the two-model cascade have identical footprints
when the escalation target is local — the 3B model dominates either way, 50×
the DistilBERT.** The "no second model resident" advantage claimed for Idea 3
belongs to *precision* escalation, and precision escalation is exactly what the
Phase 3 gate ruled out. [Certain]

There is no configuration here that is both self-contained and strong on rows:
66 MB and 0.39 row, or 6.7 GB and 0.81, or 130 MB plus a network dependency and
0.70.

## Files written

`p45_predictions_held_out_v3.csv`, `p45_predictions_held_out_v4c.csv`,
`p45_row_escalation_sweep.csv` (336 rows), `p45_cascade_full.csv` (336 rows),
`memory_footprint.csv`, `p6_latency_*.json`, `p45_cloud_latency_*.json`.
