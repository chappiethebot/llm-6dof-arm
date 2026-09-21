# Threshold pre-registration

**Written 2026-09-21, before any Phase 4 / 6 / 7 arm was run**, and before any
escalation result on `held_out_v3`, `held_out_v4c` or the external benchmark
existed. Committed here so the ordering is auditable.

## Why this file exists

Phase 2 produced a row-confidence separation on `held_out_v3` with
non-overlapping supports. Reading an operating point off that curve and then
quoting its accuracy is threshold selection on the test set. The earlier
`τ = 0.90` quote has been removed from `PHASE_2_SUMMARY.md` for exactly that
reason.

Everything downstream reports the **full sweep**, τ from 0.00 to 1.00 in 0.05
steps. Where a single number is quoted, it comes from one of the two rules
below, both fixed here in advance.

## Rule A — fixed, uses no data at all

**τ_A = 0.50** for every factor.

Rationale, stated before seeing results: escalate when the top class holds less
than half the probability mass. It needs no tuning set, it is the same number
for every factor regardless of arity, and it cannot leak test information. It
is the weakest possible claim and therefore the safest one to quote.

## Rule B — selected on `dev_v2` only

**τ_B = the smallest τ in the sweep grid whose escalation rate on `dev_v2` is
≤ 20%.** If several qualify, the smallest τ wins (least escalation).

Rationale: `dev_v2` is already the established selection set for this project —
it chose epoch 09 for the flat classifier and epoch 11 for the factorized one.
Using it for one more scalar is consistent with that, and it is the only
legitimate tuning set available.

### Known and recorded in advance: Rule B is expected to be weak here

`dev_v2` contains **14 farnear and 7 neither row commands and zero updown**
commands. The factorized row head scores **100% on dev**. A selection set that
contains none of the failure mode cannot calibrate a detector for it, so τ_B is
expected to land low and to escalate little.

**This is itself a finding and will be reported as one**, not quietly patched:
a per-factor gate cannot be tuned on a development set that does not contain
the distribution shift the gate exists to catch. If τ_B behaves as predicted,
that is evidence about the *methodology*, not a bug in the cascade.

## What may NOT be quoted

- Any τ read off a held-out sweep.
- Any τ chosen because it produced a good held-out number.
- `τ = 0.90`, specifically, which was the value the earlier draft quoted.

## Reporting contract

For every escalation arm:

1. The full τ sweep, as a curve and as a CSV.
2. The operating points at **τ_A** and **τ_B**, both labelled with the rule.
3. Escalation rate, action match, joint correctness, per-factor accuracy,
   latency p50/p95/p99, joules per command, and peak resident memory at each
   quoted point.
4. The same sweep run on the **flat head's single location confidence**, so the
   router-signal comparison that is the actual evidence for Idea 1 is on the
   same axes.

## Held-out families

`held_out_v3` is the primary set. Two independent checks are required because
`held_out_v3` has a total row-vocabulary disjunction from training, which
inflates any separation statistic:

- **`held_out_v4c`** — the neither-vocabulary probe, zero exact and zero
  near-duplicate overlap with `train_v2`, `dev_v2` or `seen_v2`.
  (`held_out_v4b` is explicitly excluded: it shares 56 exact commands with
  `train_v2` by construction.)
- **Phase 7 external benchmark** — the only test of the gate on data whose
  train/test vocabulary is not disjoint by construction.
