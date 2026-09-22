> **SUPERSEDED IN PART — do not quote from this file without checking `FINAL_REPORT.md` §0.**
>
> Specifically:
> - the reference to "the per-factor cascade (0.8433)" is the τ = 1.0 always-escalate pipeline, not a routed cascade, and the gap it implies is nominal only (B-2).

# Phase 5 — baselines, the floor

Run 2026-09-21. Scripts: `p5_baselines.py`, `p5_rules_firewalled.py`.
All train on `train_v2` only, select on `dev_v2` only, evaluate on
`held_out_v3` (n=217 commands, fresh single pass, no warmup row). All emit
through the same schema mask as the classifiers, so schema validity is 1.0 by
construction for every arm and the comparison is about semantics, not format.

| arm | n | action match | joint | location | reject | p50 latency |
|---|---|---|---|---|---|---|
| **rules (open vocabulary)** | 217 | **0.7235** | 0.8571 | 0.8571 | 0.25 | **7.7 µs** |
| kNN over MiniLM | 217 | 0.6452 | 0.6270 | 0.6131 | 0.25 | 5,637 µs |
| TF-IDF + logistic regression | 217 | 0.6221 | 0.6905 | 0.6964 | 0.25 | 608 µs |
| *factorized DistilBERT (P2)* | 217 | *0.6083* | *0.5714* | *0.5714* | *0.75* | *~29,000 µs* |
| *flat DistilBERT (P2)* | 217 | *0.7465* | *0.7460* | *0.7560* | *0.58* | *~29,000 µs* |

## The rules baseline beats every trained arm — and the number is not honest

At face value a 7.7 µs regex grammar scores 0.7235, beating the factorized
DistilBERT by 11.5 points and coming within 2.3 points of the flat one, at
~3,800× lower latency.

**That comparison is invalid, and the invalidity is measurable.** I wrote the
grammar's row vocabulary as `upper / top / far / back`, covering `upper/bottom`
(held_out_v3), `far/near` (train_v2) **and** `back/front` (held_out_v4c). A rule
author naturally writes down every word they can think of. But `train_v2`
contains only `far/near`, so no trained model ever saw `upper`, `bottom`,
`back` or `front`. The grammar is using vocabulary knowledge that was
firewalled away from every neural arm.

`p5_rules_firewalled.py` measures the size of that advantage by rebuilding the
grammar using **only row words that actually occur in `train_v2`**:

```
row word  upper    train_v2 count    0   FIREWALLED OUT
row word  top      train_v2 count    1   ALLOWED
row word  far      train_v2 count   27   ALLOWED
row word  back     train_v2 count    0   FIREWALLED OUT
row word  bottom   train_v2 count    0   FIREWALLED OUT
row word  near     train_v2 count   32   ALLOWED
row word  front    train_v2 count    0   FIREWALLED OUT
row word  middle   train_v2 count   38   ALLOWED
```

| arm | dataset | action match | location |
|---|---|---|---|
| rules_open | held_out_v3 | 0.7235 | 0.8571 |
| **rules_firewalled** | held_out_v3 | **0.4470** | **0.5000** |
| rules_open | held_out_v4c | 0.7235 | 0.8571 |
| **rules_firewalled** | held_out_v4c | **0.4470** | **0.5000** |

**The unfair advantage is +27.7 action points and +35.7 location points.**
[Certain]

The honest floor is therefore **0.4470**, well below the flat DistilBERT
(0.7465) and far below the per-factor cascade (0.8433).

This is the same finding as Phase 2 seen from the other side: the entire row
problem is vocabulary coverage. It is worth 27.7 action points to a rule
system and 43 points to a neural row head. Any thesis table listing a
rule-based baseline against trained models must state which vocabulary the
rules were allowed to see, or the comparison is meaningless.

Note also that `rules_open` scores **identically on `held_out_v3` and
`held_out_v4c`** (0.7235 both), because it was written to handle both
vocabularies — while every trained arm drops on v4c. That is the signature of
a baseline that is not actually being tested.

## Reject recall is poor across the board

Every baseline scores 0.25 reject recall, the flat classifier 0.58, the
factorized one 0.75. Impossible commands are not a solved part of this task for
any arm. [Certain]

## P5.4 TinyBERT-4 — NOT RUN

`huawei-noah/TinyBERT_General_4L_312D` is not in the local HuggingFace cache
and the session ran with `HF_HUB_OFFLINE=1` for everything except the MASSIVE
download. It was not attempted rather than silently swapped for a different
model. The sub-15M edge arm remains an open item. [Certain]

## Files written

`baseline_rules.csv`, `baseline_knn_minilm.csv`, `baseline_tfidf_logreg.csv`,
`baselines_summary.csv`, `baseline_rules_firewall.csv`.
