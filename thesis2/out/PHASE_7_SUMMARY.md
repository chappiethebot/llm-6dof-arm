> **SUPERSEDED IN PART — do not quote from this file without checking `FINAL_REPORT.md` §0.**
>
> Specifically:
> - flat and factorized here are **two heads of one jointly trained encoder**, so this is not a comparison of separately trained models (S-1).
> - both router AUROCs are scored against **factorized-head correctness** (S-2).

# Phase 7 — generalization beyond the robot arm

Run 2026-09-21 on `thesis2gpu`. Script: `thesis2/scripts/p7_massive.py`.

## Why MASSIVE, and why this is the decisive test

MASSIVE en-US via `mteb/amazon_massive_intent`: 11,514 train / 2,033 dev /
2,974 test. Every intent label is literally `<scenario>_<action>` — **60
intents over 18 scenarios and 46 action strings**, only 60 of 828 possible
combinations legal. That is the same composite-label structure as
`grid_<row><col>`, and the factorization is the dataset's own, not one I
invented. [Certain]

`AmazonScience/massive` and `qanastek/MASSIVE` both fail on this `datasets`
version ("Dataset scripts are no longer supported"). The `mteb` mirror is
intent-only, no slots — which is fine, because the intent label is itself the
composite label we need.

**The critical property: MASSIVE's train/dev/test are a random split of one
distribution. There is NO vocabulary disjunction.** `held_out_v3` has a total
row-vocabulary disjunction from `train_v2`, and that is what drove the Phase 2
AUROC to 1.000. Phase 7 is the only test of the gate where that artifact is
absent. [Certain]

## Factorization costs accuracy here too, but far less

Same backbone (DistilBERT), both heads on one encoder so the comparison shares
a substrate. 4 epochs, seed 20260820, lr 5e-5, batch 32.

| quantity | n | value |
|---|---|---|
| flat 60-way intent accuracy | 2,974 | **0.8880** |
| factorized reconstructed accuracy | 2,974 | **0.8783** |
| delta | | **−0.98 pts** |
| scenario factor accuracy | 2,974 | 0.9169 |
| action factor accuracy | 2,974 | 0.8981 |
| illegal `scenario_action` combos produced | 2,974 | 1.98% |

−0.98 points against −13.82 on the robot task. **Factorization is mildly
harmful on both, and the robot task's severity came from the vocabulary
disjunction, not from factorization as such.** [Certain]

The 1.98% illegal-combination rate is a real cost of factorization that the
robot task never exposed: `grid_<row><col>` has all 9 combinations legal, so
reconstruction there is always valid by construction. MASSIVE has only 60 of
828, so independent factor heads can and do emit combinations that do not
exist. [Certain]

## The headline: the per-factor router advantage does NOT survive

Both signals escalate the **weaker factor** to an oracle; only the gate differs.

**AUROC for ranking "factorized symbol correct":**

| signal | AUROC |
|---|---|
| per-factor, min(scenario conf, action conf) | **0.8910** |
| flat single intent confidence | **0.8889** |

A difference of **+0.0021**. On `held_out_v3` the same comparison was 1.0000
against 0.9552.

**Matched-budget accuracy, oracle escalation of the weaker factor:**

| budget | per-factor signal | flat signal | gap |
|---|---|---|---|
| 5% (149 rows) | 0.8921 | **0.8941** | −0.20 |
| 10% (297) | 0.9032 | **0.9052** | −0.20 |
| 20% (595) | 0.9126 | **0.9153** | −0.27 |
| 30% (892) | **0.9196** | 0.9193 | +0.03 |
| 40% (1,190) | **0.9223** | 0.9220 | +0.03 |
| 50% (1,487) | 0.9230 | 0.9230 | 0.00 |

**At small budgets the flat single confidence is slightly BETTER. At large
budgets they are indistinguishable. The per-factor signal never wins by more
than 0.03 points.** [Certain]

Compare the robot task, where the per-factor signal won by 5–25 points.

## What this establishes

The Idea 1 claim — *factorization yields a better router signal than a flat
head's single confidence* — **holds on `held_out_v3` and does not generalize**.

The mechanism is now clear and is a narrower, more defensible claim:

> A per-factor confidence beats a whole-field confidence as a router **when the
> errors are localized to one factor**. `held_out_v3`'s row-vocabulary
> disjunction localizes 100% of the location error into the row factor, and the
> per-factor signal exploits that exactly. MASSIVE's errors are spread across
> both factors with no shift to localize them, and the per-factor signal
> collapses to parity with the flat one.

That condition is checkable in advance — Phase 1's factor-level error analysis
is precisely the diagnostic — so the contribution is not vacuous. But it is a
**conditional** contribution, not a general improvement, and the thesis should
say so. [Certain]

## Files written

`generalization_massive.csv`, `generalization_massive_summary.json`,
`generalization_massive_preds.csv` (2,974 rows).
