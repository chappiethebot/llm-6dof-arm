# Phase 6 — single-pass logit classification (SALSA)

Run 2026-09-21 on `thesis2gpu`. Scripts: `salsa_row.py`, `p45_collect.py`,
`p6_calibration.py`.

Per arXiv 2510.22691. Each class is mapped to a distinct **single token**, the
model is prompted for a one-token answer, and the output is projected onto only
the valid class-token logits. One forward pass, no autoregressive generation.

Ollama is not used — it does not expose arbitrary token logits. HuggingFace
transformers with the locally cached `Qwen/Qwen2.5-3B-Instruct` weights.
`bench/llm_client.py` was not imported or modified.

**Factorization is what makes this cheap.** `grid_A1` is not a single token and
the full location question needs 14 classes. The **row** sub-question needs
three, and A/B/C are single tokens. 3 logits from one forward pass replaces a
generated JSON object.

## P6.1 / P6.2 — row accuracy, n=108 grid commands per dataset

| scorer | held_out_v3 | held_out_v4c |
|---|---|---|
| DistilBERT factorized row head (local baseline) | 0.3333 | 0.3889 |
| **qwen2.5:3b base, letters** | **0.7407** | 0.2593 |
| qwen2.5:3b base, digits | 0.7315 | **0.4167** |
| **qwen2.5-3b-lora, letters** | **0.8056** | 0.2037 |
| qwen2.5-3b-lora, digits | 0.7778 | 0.3889 |

**P6.2's LoRA adapter, trained to generate JSON, is the best row scorer on
`held_out_v3` (0.8056) — better than its base model and better than
gpt-4o-mini (0.7037).** It was never trained to answer a 3-way letter question,
yet reading its logits at that position works. [Certain] That is a genuinely
useful result for the thesis: an adapter's knowledge is accessible through a
channel it was not trained on.

On `held_out_v4c` (`back`/`front` row words) **both local scorers fall below
chance**. The prompt legend in `salsa_row.py` defines row A as "the far side,
the upper side" and never mentions `back`/`front`, so the models have no basis
for the mapping and guess consistently wrong. gpt-4o-mini reaches 0.6759 on the
same prompt by inferring it. [Certain]

## Label-token mapping matters, and most where the task is hardest

| arm | letters | digits | gap |
|---|---|---|---|
| qwen2.5:3b, v3 | 0.7407 | 0.7315 | −0.9 |
| qwen2.5-3b-lora, v3 | 0.8056 | 0.7778 | −2.8 |
| qwen2.5:3b, **v4c** | 0.2593 | **0.4167** | **+15.7** |
| qwen2.5-3b-lora, **v4c** | 0.2037 | **0.3889** | **+18.5** |

Negligible where the model understands the question, **15–18 points where it
does not**. A SALSA number quoted without its label-token mapping is not
reproducible. [Certain]

## P6.3 — the headline efficiency comparison

Same model, same prompt, same machine. Generation decodes 8 new tokens.

| | SALSA (1 forward pass) | generation | speedup |
|---|---|---|---|
| qwen2.5:3b, p50 | **35.5 ms** | 245.0 ms | **6.9×** |
| qwen2.5:3b, p95 | 36.4 ms | 253.7 ms | 7.0× |
| qwen2.5-3b-lora, p50 | **35.4 ms** | 245.8 ms | **6.9×** |
| gpt-4o-mini (cloud, for scale) | 657 ms p50 | — | — |

SALSA is **6.9× faster than generation on the same weights**, and the local
SALSA call is **18× faster than the cloud round trip**. [Certain]

Energy (GPU only, protocol in `energy.csv`): DistilBERT **0.070 J/command**
against SALSA-on-3B **4.02 J/command** — **57× more energy** for the escalated
sub-question. Cheap in latency is not cheap in joules. [Certain]

## P6.4 — calibration of the logit softmax

Same 3-way question, same 108 commands, same denominator.

| dataset | scorer | acc | mean conf | ECE | Brier |
|---|---|---|---|---|---|
| v3 | DistilBERT factor row | 0.3333 | 0.7182 | **0.4163** | 0.2579 |
| v3 | salsa qwen3b letters | 0.7407 | 0.8395 | **0.1071** | 0.1110 |
| v3 | salsa qwen3b digits | 0.7315 | 0.8948 | 0.1843 | 0.1238 |
| v3 | salsa qwen3blora letters | 0.8056 | 0.8041 | 0.1325 | **0.0628** |
| v3 | salsa qwen3blora digits | 0.7778 | 0.9259 | 0.1938 | 0.1326 |
| v4c | DistilBERT factor row | 0.3889 | 0.7189 | 0.3344 | 0.2608 |
| v4c | salsa qwen3b letters | 0.2593 | 0.6736 | 0.4143 | 0.3672 |
| v4c | salsa qwen3blora digits | 0.3889 | 0.8087 | **0.4198** | 0.3911 |

**The SALSA logit softmax is far better calibrated than the DistilBERT factor
head on `held_out_v3` — ECE 0.107 against 0.416, a 4× improvement.** [Certain]
The `letters` mapping is consistently better calibrated than `digits`, which is
systematically overconfident (mean confidence 0.89–0.93 at similar accuracy).

On `held_out_v4c` every scorer is badly overconfident — mean confidence exceeds
accuracy by 0.33 to 0.42. **Out of distribution, none of these confidences can
be trusted as a gate**, which is the same conclusion Phase 4 reached from the
escalation side. [Certain]

## P6.5 — qwen2.5:0.5b — NOT RUN

`Qwen/Qwen2.5-0.5B-Instruct` is cached (953 MB) and would run, but was not
reached before the session's compute budget went to Phases 4, 5 and 7. Reported
as not run rather than estimated. [Certain]

## Files written

`p45_predictions_held_out_v3.csv`, `p45_predictions_held_out_v4c.csv`
(both carry the full per-command SALSA probability vectors for both mappings),
`logit_calibration.csv`, `p6_latency_qwen3b_*.json`,
`p6_latency_qwen3blora_*.json`.
