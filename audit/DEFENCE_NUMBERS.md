# Defence numbers — what can be quoted, with intervals and basis

**Revision 2 — 17 September 2026.** Rewritten, not patched: the V5 re-measurement
changed the *ranking*, so every headline row of revision 1 was wrong rather than
stale. Revision 1 (V3-era) is preserved in git history and its failure-mode
narrative is retained in §6 below.

**Primary basis: V5.** 12,152 rows, 36 cells, `held_out_v3` (217 commands),
legend v5. Manifests tracked and verified against their CSVs.
**V3 is superseded as a source of headline numbers** and is quoted only where
this document says so explicitly.

**Metric name.** Call it **action match** (verb, target and location all
correct). Never "exact match": the scorer drops `modifiers` when falsy, and
71% of strict-correct rows depend on that rule. `llm/score.py` is single-sourced
and behaviourally identical at `3000198` and HEAD (0 differing verdicts on
20,716 real pred/truth pairs). [Certain]

---

## 0. Two decisions that gate everything below — UNRESOLVED

Nothing in §1 is safe to put on a slide until these are settled. Both are
recorded as BLOCKERs in `audit/AUDIT_REPORT.md` (findings 2 and 10).

### 0.1 The warmup rule is undecided, and it moves the headline rounding

`bench/summary.py:176` includes the warmup row in the accuracy denominator;
`:208` excludes it from latency. For the 31 local cells, which ran
`--repeats 1`, `command_id 0` **is** the warmup, so it is the only observation
of that command.

| cell | warmup in | warmup out | previously published |
|---|---|---|---|
| `qwen2.5-3b-lora` / zero_shot | 208/217 = 0.9585 | 207/216 = 0.9583 | 95.9% |
| `classifier:distilbert` | 162/217 = 0.7465 | 162/216 = **0.7500** | 74.7% |
| `gpt-4o-mini` / rag | 1018/1085 = 0.9382 | 1017/1084 = 0.9382 | 93.8% |

**This document quotes the warmup-excluded column throughout**, because the
latency figure beside it already excludes the same row and a number cannot be
representative of one and not the other. Consequences of adopting that rule:

- 31 local cells are honestly **n=216**, not 217; grid commands are **131**, not
  132, for those cells (command 0 is a grid command).
- The headline row count becomes **12,116**, not 12,152.
- The classifier reads **75.0%**, not 74.7%, everywhere — including the slides.

If you instead re-run `command_id 0` for the 31 local cells (cheap, local, no
cloud spend), n=217 is recoverable honestly and this whole subsection collapses
to one sentence. That is the better outcome and it costs about two hours.
**Decide once, then make `summary.py`, `analyze.py` and `figures_v3.py` agree.**

### 0.2 No V5 confidence intervals exist [Certain]

`audit/figures_v3.py:104-125` implements `clustered_ci` correctly — bootstrap
over `command_id` clusters, cluster means, `n_clusters` reported — but `:92`
hardcodes the V3 filenames. **There is no V5 equivalent, so no V5 number in this
document currently has an interval.** The intervals printed in revision 1 belong
to different numbers from a different run and have been removed rather than
carried over.

This is not cosmetic. The top two V5 cells are **0.9583 and 0.9382 — 2.0 points
apart**. Revision 1's own stated rule was that the median CI half-width is 5.8
points and "any difference under ~6 points is not claimable".

> **Under the thesis's own pre-stated rule, the headline finding — that a 3B
> local LoRA beats gpt-4o-mini — is not claimable until V5 intervals exist to
> test it.** [Certain]

Fix: parameterise `figures_v3.py` onto the V5 files. The bootstrap code needs no
change. ~2 hours. Until then, present §1 as an ordering **to be confirmed**.

---

## 1. Headline — V5, warmup-excluded, NO INTERVALS YET

| claim | number | basis |
|---|---|---|
| Best system overall | qwen2.5-3b-lora / zero_shot **0.9583** | all, n=216 |
| Best cloud system | gpt-4o-mini / rag **0.9382** | all, n=217 commands (1,084 rows) |
| Best local, no GPU at inference | classifier **0.7500** | all, n=216 |

**Three-way basis split — use this, not "grid-only"** [Certain]:

| cell | all | grid cells | containers | no-location |
|---|---|---|---|---|
| gpt-4o-mini / rag | 0.9382 | **0.9848** (132) | 0.9889 (36) | 0.7755 (49) |
| qwen2.5-3b-lora / zero_shot | 0.9583 | **1.0000** (131) | 1.0000 (36) | 0.8163 (49) |
| classifier:distilbert | 0.7500 | **0.6947** (131) | 1.0000 (36) | 0.7143 (49) |

`held_out_v3` composition: **132 grid-cell, 36 container, 49 no-location**
(`stop`, `reject`, `fetch`, `hold`, `inspect`). Cloud figures are row-level over
5 repeats; local figures are command-level at 1 repeat. State which when quoting.

**Revision 1's "grid-only (n=181)" basis is retired.** It pooled the 132 grid
commands with the 49 that contain no location at all, so 27% of a basis labelled
"spatial reasoning" had no spatial content. Not defensible; do not reuse it.

### The 131/131 result needs its own paragraph, written before submission

`qwen2.5-3b-lora` scores **perfect on every grid cell**. It is not string
leakage — `train_v2 ∩ held_out_v3` is 0 exact and 0 near-duplicate on four
independent checks (`audit/AUDIT_REPORT.md`, leakage verdict). But a perfect
score is the first thing an examiner probes, and the honest reading is that the
V5 legend removed the vocabulary mismatch that was suppressing this cell in V3
(0.407 grid-only → 1.000). **That makes the V5 legend, not the adapter, the
active ingredient — and the legend was authored after seeing V3 fail. See §7.1.**
[Likely]

### Pareto frontier — INCOMPLETE, do not draw yet

| cell | action match | median latency |
|---|---|---|
| classifier:distilbert | 0.7500 | **17.9 ms** (min 14.1, max 24.4, n=216) |
| gpt-4o-mini / role_based | 0.9133 | ~699 ms (V3 figure) |
| gpt-4o-mini / rag | 0.9382 | ~806 ms end-to-end incl. 5.3 ms retrieval |
| **qwen2.5-3b-lora / zero_shot** | **0.9583** | **NOT RECORDED IN ANY DOCUMENT** |

> **The winning cell has no V5 latency or cost figure anywhere in the repo or in
> any committed document.** The cost/accuracy Pareto plot is the thesis's core
> contribution and its top-right point currently cannot be placed. Extract
> `latency_s` p50 for `qwen2.5-3b-lora/zero_shot` from
> `results_v5_lora_scored.csv` before drawing the figure. For scale, base
> `qwen2.5:3b` ran at p50 1560.6 ms in the same harness. [Certain]

**Also not yet recomputed for V5:** whether every locally-run generative model is
still strictly dominated. That was a V3 claim and the V5 ranking inverted it —
the LoRA is now the frontier point. Do not repeat revision 1's "nothing else is
on it" sentence without rechecking.

### LoRA × strategy — new in V5, never measured before

| strategy | action match |
|---|---|
| zero_shot | **0.9583** |
| role_based | 0.9491 |
| cot | 0.8148 |
| few_shot | 0.7593 |
| rag | 0.7454 |

**Prompting on top of a fine-tuned adapter degrades it** — up to 21 points. This
is a clean, quotable finding and it is the most interesting thing V5 produced.
Note the conflict with `bench/llm_client.py:220`, which restricts LoRA models to
`zero_shot` and whose docstring asserts the adapters were "measured on bare
zero-shot prompts". The manifests contradict that docstring. No measured number
is wrong; the comment is. Fix the comment, not the data. [Certain]

---

## 2. Structural results [Certain]

- **The classifier emitted zero structurally invalid objects.** 0 illegal verbs,
  0 illegal symbols, 0 field-mask violations. **V3 basis: 1,085 rows. The V5
  basis is 216 rows — restate or recompute before quoting, do not carry the
  1,085 figure across.**
- **31.4% of all generative rows never validated (9,530 of 30,352).** *V3 only.*
  No V5 equivalent has been computed. Either compute it or label it V3.
- **Rows are harder than columns for every model** — gpt-4o-mini 96.3% column vs
  82.1% row; the LoRA pair 86–90% column vs 30–41% row. *V3.*
- **Container vs grid are different tasks**: the answer symbol appears verbatim
  in 36/36 container commands and 0/132 grid commands.
- **The classifier is the control that isolates the V3 LoRA failure.** Same
  training file, same distribution shift, 62.2% row accuracy vs the 0.5B LoRA's
  30.6%. Same data, different architecture, different outcome.

**Unresolved count discrepancy:** V3 is stated as 30,352 rows in revision 1 §2
and as 30,380 rows in `audit/AUDIT_REPORT.md` §1 and §7. A 28-row gap.
Reconcile before either number appears in the thesis. [Certain]

---

## 3. Cost [Certain, with caveat]

- **V5 cloud spend for the entire 36-cell study: ~$0.43.** Only gpt-4o-mini cells
  incur cost. Confirm the exact figure from `bench/backfill_tokens_v5.py` output
  before quoting to four decimals.
- **V3 cloud spend, 5-cell gpt-4o-mini condition: $0.4236**, exact (o200k_base
  *is* its tokenizer). CoT was **$0.1602 of that — 3.0× zero_shot for 15 points
  of accuracy loss.** That remains the cleanest single cost sentence available,
  and it is a V3 sentence. Label it.
- **Hidden reasoning tokens collapsed between runs**: study-wide hidden:visible
  fell from **1.65:1 (V3) to 0.31:1 (V5)**. In V3, deepseek-r1 hid 94.7% of its
  output — 17.9 hidden tokens per visible one.
- Local token counts are exact arithmetic over an approximate tokenisation.
  Ollama's `eval_count` is the exact route and is captured only from V5 onward.

---

## 4. Latency

- Classifier **17.9 ms p50** (V5, n=216). The excluded warmup row is 14,650.9 ms
  — the cold model load, 818× the median. Say "warm median" out loud.
- `latency_s` **excludes retrieval**. End-to-end for a RAG cell is the sum.
  Measured retrieval medians are 5.3–6.7 ms, 0.1–0.7% of end-to-end.
  `bench/summary.py:341-353` sums correctly; `analyze.py:103` and
  `figures.py:273` do not, so any figure from those two understates RAG by ~5 ms.
- **TTFT and per-stage latency are unrecoverable** — the client made blocking
  calls. The original per-stage novelty claim is not supported by the data.
- **Never claim sub-100 ms end-to-end.** Plan v6 don't-do list.

---

## 5. V4 — robustness check only [Certain]

V4 is **post-hoc** (pre-registered in `audit/PREREGISTRATION_v4.md`, but after V3
was seen). It does not replace any number above.

- The legend rewrite **changed nothing overall: 20 of 27 cells inside the CI,
  with 7 real moves** — not 21 of 27 and not six real moves. Three independent
  counting methods (the CSV's own `verdict` column, a flat ±6-point band, and
  disjoint bootstrap intervals) all return 7 real / 20 noise. They disagree on
  *which* seven — the band and the CI test swap `gpt-4o-mini/cot` for
  `qwen2.5:0.5b/few_shot` — so **both of those cells are reported as ambiguous
  rather than picking whichever rule tells the tidier story.**
  *(Corrected 17 Sep 2026. The wrong figure survived in this file, in
  `audit/V4_RESULTS.md:121` and in the `fig5_v3_vs_v4a.png` caption across three
  sessions after being corrected in `audit/V3_VS_V4.md:316`. Check all four
  locations, not just this one.)*
- It **does not track model size**. deepseek-r1:7b gained (+0.213 zero_shot)
  while gpt-4o-mini, the largest, regressed (−0.159 few_shot). What predicts the
  sign is whether the model was already failing the row axis.
- **The rewrite introduced a new defect.** `Rows run front-to-back: A = far from
  the robot` contradicts itself; gpt-4o-mini's regression is dominated by A→C
  inversion (`grid_A1` → `grid_C1` ×49). Report as a finding about prompt
  engineering, not a system improvement.
- Removing the literal cell id and arrows (V4a-lite) **fully removed `grid_B2`
  anchoring** (40.3% → 11.1%) but only **halved** spurious modifier extraction
  (59.7% → 44.0%) — two failure modes, different causes. Anchoring comes from the
  tokens; modifier pollution from the extra prose itself.
- **The entire V3-vs-V4 comparison runs on the all-217 basis.** `v4_cells.csv`
  contains no grid-only decomposition of any V4 condition, so the robustness
  check runs on the weaker basis. Producing grid-only V4 numbers is a
  re-analysis of existing CSVs, no model runs.
- V4a ran **1 repeat** for local models (216 rows, and **216 commands not 217** —
  one absent, cause unrecorded) against V3's 5 repeats.

---

## 6. Limitations to state before an examiner finds them

### 6.1 `held_out_v3` is no longer a sealed set [Certain] — the biggest one

Zero training leakage is established on four independent checks. That is not the
issue. The issue is **reuse**: `held_out_v3` was measured in V3, again under V4
conditions, and again in V5 — and **the V5 legend was authored specifically to
fix failures observed on that same set.** The prompt is not data, so no file was
modified and no rule was broken in the letter. But the guarantee a held-out set
provides is spent after the first look, and plan v6's own don't-do list reads
"❌ Tune on the held-out set".

This was a deliberate, recorded decision (option C: accept the confound and
declare it). **Declare it in the methods chapter, in one paragraph, in these
terms** — the V5 numbers measure how well each method performs under a legend
tuned on this set, not how well it would perform on a fresh one. A reader who
finds this unstated will discount the whole results chapter; a reader who finds
it stated will treat it as ordinary experimental honesty.

The clean version of this experiment needs a fourth held-out set authored after
the legend was frozen. That is future work, not this thesis.

### 6.2 The winning margin is inside the noise floor

2.0 points between the top two cells, against a stated ~6-point claimability
threshold and no V5 intervals. See §0.2. **Do not say "beats" until the
intervals exist.** "Scores higher than, within a margin the study cannot
currently resolve" is the defensible phrasing.

### 6.3 Everything is simulation

MuJoCo for measurement, Isaac Sim for demo. The physical arm phase was cut with
supervisor agreement. Phase 6 (vision) and Phase 2.5 (RL backend) were formally
cut in writing by the supervisor on 17 September 2026. **The cover page still
promises vision-based manipulation — reconcile the title with the supervisor
before submission, in writing.**

### 6.4 The left/right convention is a convention, not a fact

Left and right are stated from the operator's view facing the robot across the
table. That is correct for an audience opposite and mirrored for anyone standing
behind the robot. `_GRID_COL` was settled physically (`put the red cube in cell
0 2` lands nearest the boxes at y=+0.40, so column index increases toward +y and
schema column 3 = "right"). The datasets were authored to match. **One sentence
in the thesis, no code change.** [Likely]

### 6.5 Reproducibility gaps disclosed in the audit

- V3 has no run manifests at all (the manifest feature postdates it). Any V3
  number is quoted without provenance.
- The regroup step from six per-model CSVs to four per-rung CSVs is uncommitted.
  Verified lossless (12,152 rows both sides, 0 verdict drift) but not automated.
- The four committed `bench/figures_v5/*.png` were drawn by `analyze.py` from a
  field-truncated copy (`results_v5_slim.csv`) with an unrecorded `--seen`
  argument. **Do not put those four figures in the thesis without regenerating
  them from a committed script with named inputs.**
- The rung-1 exclusion rate of 39.8% biases every confusion percentage toward the
  better models.
- `num_ctx=4096` is hardware-derived from 16 GB VRAM; reproduction on other
  hardware silently gets a different window.

---

## 7. NOT reportable — do not quote

| item | why |
|---|---|
| Any V5 number as a *significant* difference | no V5 confidence intervals exist (§0.2) |
| "3B LoRA beats gpt-4o-mini" | 2.0-point gap, below the stated ~6-point resolution |
| 74.7% / 95.9% / 19 ms | warmup-inclusive figures; superseded by 75.0% / 95.83% / 17.9 ms (§0.1) |
| The V3 headline ordering | superseded — the ranking inverted under V5 |
| Any V4 accuracy as a system result | post-hoc condition |
| The deepseek V4 gains as caused by the legend fix | mechanism not established; abandonment criterion partially fired |
| `qwen2.5-0.5b-lora` V4c row accuracy (0.495) | inconclusive — between the pre-registered bands |
| **Field-level escalation (t=0.30 dominates role_based)** | **simulated from full-object prompts; assumes cloud location accuracy is unchanged when only the location is asked — untested. Novelty claim unverified beyond two searches.** |
| Any per-token cost or throughput for local models | exact counts need `eval_count`, captured only from V5 |
| TTFT or per-stage latency | not recoverable; blocking calls |
| Energy per command | never measured |
| Vision or real-arm results | Phase 6 and 7 have zero implementation and are formally cut |
| `results_v4a_local.CONTAMINATED.csv` | two concurrent writers, 22.3% disagreement; quarantined |
| Anything from `held_out_v4b.jsonl` | 26% seen in training by construction; built and never run |
| `n=539` on slide 17 | what it counts has never been established |
| "0.1–0.5 cm placement" on slide 22 | the verified Isaac figure is **0.2–0.4 cm**, and it is `pick_place`; `move_to` is a different and much worse number |

---

## 8. Questions to expect

**1. "Your title promises vision-based manipulation — where is it?"** [Certain
this gets asked] The scope was reduced with supervisor agreement, in writing,
on 17 September 2026. Have the written confirmation available and the title
reconciled. This is now a scope answer, not a gap.

**2. "Your best system beats the cloud by two points — is that real?"** [Likely]
Answer honestly: not at the study's resolution. See §0.2 and §6.2.

**3. "The LoRA scores 100% on grid cells. Why should I believe that?"** [Likely]
Leakage is ruled out on four checks. The honest mechanism is the legend rewrite,
and the legend was tuned on this set. See §6.1.

**4. "Is 217 commands enough?"** Median CI half-width is 5.8 points (V3), so any
difference under ~6 points is not claimable. State it on every table.

**5. "Why does the demo only run LoRA zero-shot when you measured five
strategies?"** `llm_client.py:220` is a demo restriction, not a measurement
fact. Fix the docstring before anyone reads it.
