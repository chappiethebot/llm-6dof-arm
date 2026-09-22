> **SUPERSEDED IN PART — do not quote from this file without checking `FINAL_REPORT.md` §0.**
>
> Specifically:
> - the INT4 flip count is **8–9 row flips depending on the quantiser's arithmetic** (9 in FP16, 8 in FP32), not 9 (re-review S-9).

# Phase 3 — INT4 quantization gate

Run 2026-09-21 on `thesis2gpu` (torch 2.14.0+cu126, CUDA, RTX 4060 Ti sm_89).
Scripts: `p3_quantization_gate.py`, `p3_verify_int4.py`,
`p3_verify_int4_supervised.py`.

## GATE: **FAIL**

Criterion: at least one location factor loses ≥3.0 points FP16→INT4, **and**
verb and target each lose <1.0 point.

**No location factor lost anything.** [Certain]

| quantity | n | FP16 | INT4 | delta pts |
|---|---|---|---|---|
| **action match** | 217 | 0.6083 | 0.6083 | **+0.00** |
| verb | 217 | 0.9401 | 0.9355 | −0.46 |
| target (head, raw) | 153 | 1.0000 | 1.0000 | +0.00 |
| location (symbol) | 168 | 0.5714 | 0.5774 | +0.60 |
| loc_type | 168 | 1.0000 | 1.0000 | +0.00 |
| **row** | 108 | 0.3333 | 0.3426 | **+0.93** |
| col | 108 | 1.0000 | 1.0000 | +0.00 |
| container_which | 36 | 1.0000 | 1.0000 | +0.00 |
| runtime_which | 24 | 1.0000 | 1.0000 | +0.00 |

Row *improved* by 0.93 points (1 command of 108) — noise, not a gain.

Consequence, as pre-agreed: **Phases 4.1–4.2 are not run.**

## Verification: the result is not an inference-path bug

A zero-delta gate is exactly what a broken precision switch would produce, so
it was checked directly rather than assumed. [Certain]

**(c1) The weights really change.** All **43 of 43** `nn.Linear` layers have
different weights in the two modes. Largest single change
`encoder.transformer.layer.2.ffn.lin2`, max |ΔW| = 0.7075 against a mean |W| of
0.0337.

**(c2) The dequantized weights sit on a 4-bit grid.** Sampling 165 output
channels: **maximum 15 distinct values in any channel** (a 4-bit code allows
16). The same channels in FP16 mode hold **732–741** distinct values. This is
the decisive check — a weight matrix that was never quantized cannot have 15
distinct values.

**(c3)** All 43 code tensors are integral and inside [−8, 7].

**(c4)** `codes` are `int8` tensors holding 4-bit *values*; torch has no int4
dtype. The live `.weight` is `float16` in both modes because INT4 is
**dequantized before the matmul**.

**(a)+(b) Logit differences are non-zero everywhere.** No head had an exactly
zero delta.

| head | mean abs Δlogit | max abs Δlogit | exactly zero? |
|---|---|---|---|
| verb | 0.2447 | 1.6602 | no |
| target | 0.2735 | 1.0225 | no |
| loc_type | 0.2279 | 0.9336 | no |
| row | 0.1849 | 1.0918 | no |
| col | 0.2324 | 1.0613 | no |
| container_which | 0.1834 | 1.0137 | no |
| runtime_which | 0.1727 | 0.6456 | no |

**Verdict: INT4 executed. The gate result stands.**

### But "unchanged accuracy" is not "unchanged behaviour"

Prediction agreement measured **only on the commands where each head's output
actually reaches the emitted object** (emit() discards a head's output unless
the predicted verb requires that field): [Certain]

| head | n used | agreement |
|---|---|---|
| target | 153 | **100.00%** |
| loc_type | 168 | **100.00%** |
| col | 108 | **100.00%** |
| container_which | 36 | **100.00%** |
| runtime_which | 24 | **100.00%** |
| verb | 217 | 97.70% (5 flips) |
| **row** | 108 | **91.67% (9 flips)** |

Five of seven heads are bit-identical on every command that uses them. INT4
flips **5 verb** and **9 row** predictions, and those flips net to **+1
correct**. The aggregate accuracy is unchanged because the flips cancel, not
because nothing moved. A deployment that needs determinism across a precision
change should know this. [Certain]

(The raw all-217 agreement figures — container_which 74.65%, runtime_which
83.41% — are misleading and are reported in `p3_int4_verification.csv` only for
completeness: they count flips on commands where the head's output is thrown
away.)

## Quantization method, stated precisely

**Manual per-output-channel symmetric INT4, applied to WEIGHTS ONLY.
ACTIVATIONS ARE NOT QUANTIZED — they remain FP16 throughout.** [Certain]

```
scale_o = max(|W[o,:]|) / 7        per output channel
q       = clamp(round(W/scale), -8, 7)      int8 tensor holding 4-bit values
W_hat   = q * scale                          dequantized to fp16 for the matmul
```

bitsandbytes 0.50.2 is installed and CUDA works; it was tried first as the
brief asked. `BitsAndBytesConfig(load_in_4bit=True)` operates through
`from_pretrained`, replacing `nn.Linear` with `Linear4bit` inside a loaded hub
tree. This model is a plain `nn.Module` holding an `AutoModel` plus seven heads
restored from a local `state_dict`; there is no `from_pretrained` path that
carries the heads. Reshaping the model to suit the library would have changed
what was measured.

**Weight sharing is real, not simulated.** One module tree is resident (101
modules, verified). `PrecisionSwitch` holds the FP16 master and the INT4
codes+scale per Linear and rebinds `.weight` in place. No second model is ever
constructed. [Certain]

43 `nn.Linear` layers, 42.49 M weights quantized. **Embeddings and LayerNorms
are NOT quantized** — standard for 4-bit weight-only, and a 23.8 M embedding
table would otherwise have dominated the measurement.

## Compression: the whole-model figure, not the flattering one

| | FP16 | INT4 | ratio |
|---|---|---|---|
| **whole model, resident** | **132.76 MB** | **69.20 MB** | **1.92×** |
| quantized Linear layers only | 84.97 MB | 21.24 MB | 4.00× |
| embeddings (23,835,648 params) | 47.67 MB | **47.67 MB** | **1.00× — unquantized** |
| LayerNorm + biases | 0.12 MB | 0.12 MB | 1.00× |

Total parameters 66,381,336.

**The headline compression is 1.92×, not 4×.** The 4.00× applies only to the
42.49 M parameters actually quantized; the 23.8 M-parameter embedding table is
left at FP16 and is 36% of the model. Quoting 3.97× or 4× without that context
overstates the deployment benefit by roughly a factor of two. [Certain]

**There is no latency benefit.** The INT4 weights are **dequantized to FP16
before every matmul** — no INT4 kernel is used, and none exists in this
implementation. Measured GPU latency is FP16 p50 3.14 ms against INT4 p50
3.22 ms; INT4 is *marginally slower* because of the dequantization step. Any
speed claim for 4-bit here would require a real low-precision kernel and is
**not made**. [Certain]

So the accurate statement is: **on this task, halving the resident model size
costs nothing in aggregate accuracy** — not "4× compression is free".

## Comparison with CascadeCNN — a different regime, not the same scheme

An earlier draft of this file claimed this used "the same weight-only scheme
CascadeCNN's 4-bit stage uses". **That was wrong and has been removed.**

CascadeCNN (Kouris, Venieris & Bouganis, FPL 2018, arXiv 1805.08743) is an
FPGA toolflow. Its Low Precision Unit is a reduced-precision *datapath* —
**dynamic fixed-point applied to weights and activations together** — because
the whole point is narrower arithmetic units and therefore higher throughput on
reconfigurable hardware. A confidence evaluation unit then flags unreliable
classifications for re-processing by the High Precision Unit. [Likely — this is
my understanding of the architecture; I could not re-read the paper from this
machine, see the blocked-steps list in FINAL_REPORT.md §11]

The differences that make the degradation figures non-comparable:

| | CascadeCNN LPU | this experiment |
|---|---|---|
| weights | low-precision fixed point | INT4, per-output-channel symmetric |
| **activations** | **also quantized** | **FP16, not quantized** |
| arithmetic | genuine low-precision datapath (FPGA) | FP16 matmul on dequantized weights |
| benefit measured | throughput / latency | accuracy sensitivity only |
| task | ImageNet CNN classification | 230-example intent/slot, DistilBERT |

Quantizing activations as well as weights is substantially more destructive
than weight-only quantization, so a weights-and-activations degradation figure
is an **upper bound** on what this experiment would have seen even at matched
bit-width.

The brief cites CascadeCNN degradation of 14.38% (AlexNet) and 18.65%
(VGG-16) at 4-bit. **I could not verify those numbers from this machine** and
they are reproduced here on the brief's authority only. Given the regime
difference above, they should be used to say "a different setting behaves very
differently", not as a like-for-like baseline. [Guessing on the exact figures;
Certain on the regime difference]

## Even a PASS could not have rescued the row defect

Worth stating because it is the reason Idea 3 cannot be the contribution on
this task. Precision escalation moves a factor from INT4 to FP16. **The row
factor's FP16 accuracy is 0.3333 — chance for a 3-way head.** Escalating to a
precision that is itself at chance recovers nothing. The row defect is a
vocabulary-generalization failure (Phase 2), and no amount of numerical
precision addresses a word the model never saw in training. [Certain]

This is why the contribution moves to **per-factor *model* escalation (P4.5)**:
the row factor needs a model with different training data, not the same weights
at higher precision.

## Caveat carried forward

`target` reads 1.0000 here but 0.9477 in `factorized_vs_flat.csv`. Different
definitions, both correct: Phase 3 measures the **target head's raw argmax**;
Phase 2.2 measures the **emitted field after verb masking**, so a wrong verb
makes the emitted object lack a target and scores it wrong. [Certain]

## Files written

`quantization_gate.csv`, `p3_int4_verification.csv`,
`p3_int4_verification.json`, `p3_int4_agreement_supervised.csv`,
`env_thesis2gpu_pipfreeze.txt`.
