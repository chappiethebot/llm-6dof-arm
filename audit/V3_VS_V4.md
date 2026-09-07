# V3 vs V4 — three prompt-design bugs, each with a measured price, plus a fourth introduced while fixing them

Source data: `audit/v4_cells.csv` (59 cells), `audit/DEFENCE_NUMBERS.md`,
`audit/PREREGISTRATION_v4.md` (commit `48b838a`, written before any V4 run).
Figures: `audit/figures/fig5_v3_vs_v4a.png`, `fig6_llama_failure_modes.png`,
`fig7_v4c_row_vs_column.png`. No model was run to produce this document.

---

## 0. How to read every number below

**Scorer.** All cells rescored under `llm/score.py` at commit `3000198`.

**Metric.** *Action match* — verb, target and location all correct. Never call
it exact match: the scorer drops `modifiers` when falsy and 71% of
strict-correct rows depend on that rule.

**Basis — read this before quoting anything.** Every V3-vs-V4 number in this
document is on the **all 217** basis. `v4_cells.csv` contains no grid-only
decomposition of any V4 condition. This matters: `DEFENCE_NUMBERS.md` states
that grid-only (n=181, of which 132 carry a location) is the basis that carries
the thesis claim, and container rows (n=36) name the answer in the command.
**So the robustness check runs on the weaker of the two bases.** State that in
the chapter; do not let a reader assume the V4 deltas were measured on the
spatial subset. Producing grid-only V4 numbers would need a re-analysis of the
existing V4 CSVs — no new model runs — and is the single cheapest improvement
available to this section.

**Noise band.** Intervals are clustered bootstrap over `command_id`, 2,000
resamples, seed 20260827. Half-widths across these cells run 0.0–6.7 points,
median 6.0. **Any change under about 6 points is not a result.** Every delta
below is tagged `[real]` or `[noise]` on that rule, and the tag is repeated in
the figures.

**Effective n.** For the seven local models the five repeats are byte-identical,
so effective n is the command count, not the row count. gpt-4o-mini is the
exception at 77.1% identical across repeats.

**Unequal repeat counts — a real asymmetry.** V3 ran 5 repeats for everything
(1,084 scorable rows/cell). V4a ran **1 repeat for the local models** (216
rows) and 5 for gpt-4o-mini (1,084 rows). V4a local also covers **216 commands,
not 217** — one command is absent and the cause is not recorded. For local
models this costs nothing, because repeats were proven to carry no information;
for gpt-4o-mini both conditions are at 5 repeats, so the comparison is like for
like. Say so rather than letting a reader find the mismatched `n` column.

**Status.** V4 is **post-hoc**. It was pre-registered, but after V3 was seen. It
does not replace any V3 number, and no V4 accuracy is a system result. V3
remains the measurement.

---

## 1. The V3 baseline

V3 is the legend that produced the 30,380 measured rows. Its five gpt-4o-mini
cells and the two frontier points:

| cell | action match | 95% CI | basis |
|---|---|---|---|
| gpt-4o-mini / rag | 0.9235 | [0.8866, 0.9558] | all 217 |
| gpt-4o-mini / role_based | 0.8553 | [0.8055, 0.9005] | all 217 |
| gpt-4o-mini / few_shot | 0.8424 | [0.7917, 0.8885] | all 217 |
| gpt-4o-mini / zero_shot | 0.8350 | [0.7889, 0.8811] | all 217 |
| gpt-4o-mini / cot | 0.6848 | [0.6258, 0.7401] | all 217 |
| classifier:distilbert | 0.7465 | [0.6866, 0.8018] | all 217 |
| qwen2.5-3b-lora / zero_shot | 0.6544 | [0.5899, 0.7189] | all 217 |
| qwen2.5-0.5b-lora / zero_shot | 0.3963 | [0.3318, 0.4608] | all 217 |

The V3 legend contained three defects. None of them was found by proofreading.
All three were found by looking at what the models actually predicted, and each
one has a price.

---

## 2. Bug 1 — the phrase/legend collision

**The defect.** Four of nine location phrasings in the dataset contain the word
*middle*. The V3 legend also used *middle* as the label for column 2. One word
labelled two different axes.

**The price.** 871 rung-1 predictions collapsed onto `grid_B2`; 91.4% of those
were commands whose true answer was a column-2 cell. The DistilBERT classifier,
trained on the same data, produced **zero** such collapses and scored `grid_B2`
60/60 — the collision lives in the prompt, not in the task.

**Who it hurt.** Not who you would guess. llama3.2 collapsed on 15.6% of rows;
qwen2.5:0.5b on 0%. A model has to be reading the legend before the legend can
mislead it.

**Caveat, and it is load-bearing.** 8,641 of 21,700 rung-1 rows (39.8%) never
validated and are absent from the confusion matrices, so every percentage in
this section is conditional on parseable output and biased toward the better
models. These numbers carry **no CI** — they are counts over pooled rung-1 rows,
not bootstrapped cell accuracies. Do not put them on the same axis as anything
in §1.

---

## 3. Bug 2 — the row vocabulary mismatch

**The defect.** Training used *far*/*near*. Evaluation used *upper*/*bottom*.
Columns shared vocabulary across both splits; rows did not.

**The price**, measured per split:

| model | split | row words | row acc | column acc |
|---|---|---|---|---|
| qwen2.5-0.5b-lora | train_v2 | far/near | 100% | 100% |
| qwen2.5-0.5b-lora | dev_v2 | far/near | 100% | 100% |
| qwen2.5-0.5b-lora | held_out_v3 | upper/bottom | **30.6%** | 90.0% |
| qwen2.5-3b-lora | dev_v2 | far/near | 100% | 100% |
| qwen2.5-3b-lora | held_out_v3 | upper/bottom | **40.8%** | 86.1% |

Chance on rows is 33.3%. Both adapters are **at chance on rows and near ceiling
on columns**, on the same commands.

**The control that makes this a finding rather than an anecdote.** The DistilBERT
classifier was trained on the same file and faced the same shift, and did not
collapse — 62.2% row accuracy against the 0.5B adapter's 30.6%. Same data,
different architecture, different outcome. That isolates the failure to LoRA
fine-tuning of a generative model rather than to the dataset.

Two competing explanations were tested and refuted. Learning the training
marginal would over-emit row C (36.5%, the most common), not row B. Copying when
the answer is visible fails too: the 0.5B scores 0.139 on containers where the
answer *is* visible, while handling runtime symbols well (0.792, 0.917) despite
those never appearing verbatim.

**The sentence the examiner will otherwise say first:** the rung-3 held-out
number measures vocabulary generalisation, not adaptation quality.

**Task property, not artefact:** rows are harder than columns for *every* model
including gpt-4o-mini (82.1% row vs 96.3% column). `[Likely]` left/right is
egocentric and near/far allocentric.

---

## 4. Bug 3 — dead legend vocabulary for the runtime symbols

**The defect.** The symbol definitions for `max_reach` and `nearest_free_cell`
were written in words that appear in **zero** commands in any split.

**The price.** `nearest_free_cell` fails 55.1% of the time — dropped entirely on
27.9% of rows, confused with `max_reach` on 27.2%. Same exclusion caveat as §2;
no CI.

---

## 5. V4a — what the fix changed, and what it cost per model

V4a rewrote the legend to address all three bugs at once, adding an explicit
disambiguation rule. Figure: `fig5_v3_vs_v4a.png`.

**The headline is that almost nothing moved. 20 of 27 cells sit inside the
±6-point band.**

The seven cells that moved:

| model / strategy | V3 | V4a | Δ (pts) | verdict |
|---|---|---|---|---|
| deepseek-r1:7b / zero_shot | 0.0645 [0.037, 0.097] | 0.2778 [0.218, 0.338] | **+21.3** | [real] |
| deepseek-r1:7b / role_based | 0.1613 [0.115, 0.212] | 0.3519 [0.287, 0.417] | **+19.1** | [real] |
| llama3.2:3b / few_shot | 0.4009 [0.336, 0.470] | 0.2222 [0.167, 0.278] | **−17.9** | [real] |
| gpt-4o-mini / few_shot | 0.8424 [0.792, 0.889] | 0.6839 [0.623, 0.745] | **−15.9** | [real] |
| llama3.2:3b / rag | 0.3226 [0.263, 0.387] | 0.1667 [0.116, 0.218] | **−15.6** | [real] |
| deepseek-r1:7b / cot | 0.1060 [0.069, 0.152] | 0.2083 [0.153, 0.264] | **+10.2** | [real] |
| gpt-4o-mini / cot | 0.6848 [0.626, 0.740] | 0.7493 [0.694, 0.805] | +6.5 | [borderline] |

`gpt-4o-mini / cot` is the one genuinely ambiguous cell: +6.5 points clears the
flat band but its intervals still overlap. Report it as borderline rather than
picking whichever rule gives the tidier story.

**The effect does not track model size.** deepseek-r1:7b gained on three of five
strategies; gpt-4o-mini, the largest model in the study, posted the single worst
regression. What predicts the sign is whether the model was already failing the
row axis — a model with nowhere to go but up gains from any legend change, and a
model at 0.84 has only downside.

**One cell is worth naming separately.** `qwen2.5:0.5b / few_shot` went 0.0553 →
**0.0000** — an interval of [0.000, 0.000], total collapse. The delta is only
5.5 points because the baseline was already near the floor, so on the published
band rule it is `[noise]`, and this document tags it that way. Its disjoint
intervals argue the other way. Small absolute moves at the floor are where the
percentage-point rule breaks down; say so once rather than quietly picking a
side.

---

## 6. Bug 4 — the one introduced while fixing the other three

The V4a legend read:

> `Rows run front-to-back: A = far from the robot`

*Front-to-back* implies A is the front. *Far from the robot* says A is the back.
The two halves of one sentence contradict each other.

**The price: 15.9 points on gpt-4o-mini / few_shot**, 0.8424 → 0.6839, and the
regression is dominated by A→C inversion — `grid_A1` predicted as `grid_C1` 49
times. The model was not confused; it followed the half of the sentence that was
wrong.

This is a new instance of exactly the conflict class that
`audit/vocab_conflicts.md` was written to catalogue, added by the person holding
the catalogue, while fixing the conflicts already in it.

**Frame it as a finding about prompt engineering, not as a system improvement.**
Three legends were written carefully by hand; two shipped with defects. What
caught the fourth was not care, it was a mechanical checker. That is the
generalisable claim in this section: legend text needs the same automated
conflict checking as code, because reading it does not work — and V5 is the
first legend to pass all four checks, catching two defects in its own first
draft.

---

## 7. V4a-lite — separating two failure modes that looked like one

Adding the disambiguation rule caused two distinct problems in llama3.2, which
V4a alone could not tell apart:

1. **`grid_B2` anchoring.** The rule text names the cell literally, and llama
   answered `grid_B2` for commands meaning other cells.
2. **Spurious `modifiers`.** The model extracted legend vocabulary into its
   output — `{"location": "grid_A1", "modifiers": ["upper"]}`. Location correct,
   object wrong. 22 of 40 lost few_shot rows are this, not the first problem.

V4a-lite removed the literal cell id and the arrows, keeping the rule as prose.
Figure: `fig6_llama_failure_modes.png`.

| llama3.2:3b | V3 | V4a | V4a-lite | Δ lite vs V3 |
|---|---|---|---|---|
| few_shot | 0.4009 [0.336, 0.470] | 0.2222 [0.167, 0.278] | 0.3148 [0.250, 0.380] | −8.6 [noise] |
| rag | 0.3226 [0.263, 0.387] | 0.1667 [0.116, 0.218] | 0.2593 [0.204, 0.324] | −6.3 [noise] |

Both cells return to inside V3's interval — **H5 supported**.

The mechanism split is the interesting part:

- Anchoring was **fully removed**: 40.3% → 11.1% (**H6 supported**).
- Modifier pollution was only **halved**: 59.7% → 44.0% (**H7 supported**, in
  the sense that the pre-registered prediction was that it would not be fully
  fixed).

**Anchoring is caused by the specific tokens; modifier pollution is caused by the
extra prose itself.** Two failure modes with different causes, separable only
because a third condition was run. Neither of the anchoring/modifier
percentages carries a CI — they are rates over rung-1 rows, like §2 and §4.

---

## 8. V4c — the split verdict on H3

**H3:** did the LoRA adapters learn a row *concept*, or two *strings*?

V4c uses *front*/*back* — row vocabulary present in **neither** the training
split nor the evaluation split. If the adapters learned a concept, accuracy
should hold. If they memorised `far`/`near`, it should not.

Overall action match, from `v4_cells.csv` (figure: `fig7_v4c_row_vs_column.png`):

| model | V3 | V4c | Δ (pts) | verdict |
|---|---|---|---|---|
| qwen2.5-0.5b-lora | 0.3963 [0.332, 0.461] | 0.4676 [0.403, 0.528] | +7.1 | [real by band, overlapping CIs] |
| qwen2.5-3b-lora | 0.6544 [0.590, 0.719] | 0.6435 [0.574, 0.708] | −1.1 | [noise] |

**Overall accuracy is not where the answer lives.** The pre-registered
hypothesis was about *row* accuracy, plotted in `fig7_v4c_row_vs_column.png`
against the two pre-registered bands. Those row figures are **not in
`v4_cells.csv` and carry no interval**, so they cannot be put on the same
axis as the overall accuracies above.

| model | V3 row | V4c row | V3 col | V4c col |
|---|---|---|---|---|
| qwen2.5-0.5b-lora | 31% | 50% | 90% | 79% |
| qwen2.5-3b-lora | 41% | 36% | 86% | 91% |

Columns stay near ceiling under vocabulary neither adapter has seen; rows sit
at or near chance (33.3%). The adapters read the side-to-side axis and guess
the front-to-back one. On the
pre-registered bands, 3B's 0.364 supports memorisation and 0.5B's 0.495 falls
between the bands and is **inconclusive**. The verdict is a genuine split, and
the 0.5B result must be reported as inconclusive rather than rounded toward the
3B story.

**Why V4b does not exist.** Reverting the row-word substitution would have
landed on verbatim training strings — 56 present in `train_v2`, 14 in `dev_v2`,
0 novel. It would have measured memorisation and returned about 100%. The
dataset was built, registered outside SEALED, and deliberately never run. That
decision is worth one sentence in the Method chapter: a condition abandoned for
a stated reason, before it could produce a number, is stronger evidence of
discipline than any result it would have produced.

---

## 9. Hypothesis scorecard

| # | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| H1 | `grid_B2` collapse drops ≥50% | **not supported** | it rose; rag 10.1% → 40.3% |
| H2 | Runtime-symbol fix helps | **not supported** | no gain outside the band |
| H3 | LoRA memorised strings | **split** | 3B row 0.364 supports; 0.5B row 0.495 inconclusive |
| H4 | Cloud within ±5 pts | **not supported** | gpt-4o-mini few_shot −15.9 |
| H5 | V4a-lite recovers llama | **supported** | both cells back inside V3's CI |
| H6 | …via `grid_B2` anchoring | **supported** | 40.3% → 11.1% |
| H7 | Modifiers not fully fixed | **supported** | only halved, 59.7% → 44.0% |

Four of seven pre-registered hypotheses failed, including the one the whole
exercise was designed around (H1). Report that plainly. A pre-registration that
only ever confirms is not doing any work.

---

## 10. Corrections found while writing this

**`21 of 27` is wrong. The correct figure is 20 of 27.** The claim appears in
`audit/DEFENCE_NUMBERS.md` §4 and in `session_record_v3.md` §2.13, the latter as
"21 of 27 cells inside the CI. Of the six real moves…". `v4_cells.csv` marks
**seven** cells `**real**`, and all three ways of counting agree:

| method | real | inside |
|---|---|---|
| the csv's own `verdict` column | 7 | 20 |
| flat ±6-point band | 7 | 20 |
| disjoint bootstrap intervals | 7 | 20 |

The three methods disagree on *which* seven — the band and the CI test swap
`gpt-4o-mini / cot` for `qwen2.5:0.5b / few_shot` — but every method returns
seven. Fix the figure in both files before either is quoted.

**`fig5_v3_vs_v4a.png` carries the wrong count in its own caption** — it reads
"21 of 27" and "the six that move". The figure is otherwise correct; only the
caption text needs regenerating. Do not ship it as-is.

**`fig7_v4c_row_vs_column.png` has a label collision**: the `36` on the
qwen2.5-3b-lora V4c bar is overprinted by the "pre-registered: memorised
strings" annotation.

**Also flagged:** `DEFENCE_NUMBERS.md` §4 attributes gpt-4o-mini's −0.159 to
few_shot, which `v4_cells.csv` confirms. `session_record_v3.md` §2.13 states the
regression without naming the strategy. Add it there.

---

## 11. What this section licenses, and what it does not

**Claimable:**

- The legend rewrite changed nothing overall — 20 of 27 cells inside the band.
- The effect of a legend fix does not track model size; it tracks whether the
  model was already failing the axis the legend describes.
- Three hand-written legends produced three distinct defect classes, and the
  fix for them introduced a fourth. Legend text needs mechanical conflict
  checking; proofreading demonstrably does not work.
- Anchoring and modifier pollution are separate failure modes with separate
  causes, demonstrated by a third condition.

**Not claimable:**

- Any V4 accuracy as a system result. V3 is the measurement.
- The deepseek gains as *caused by* the legend fix. The mechanism is not
  established and the pre-registered abandonment criterion partially fired.
- `qwen2.5-0.5b-lora`'s V4c row accuracy as evidence either way. It is
  inconclusive by the pre-registered bands.
- Anything about the grid-only subset under V4. It was not computed.

**The contribution of this section is the price list, not the fix.** Four
prompt-design defects, three of them in the measured configuration and one
introduced while repairing them, each with a number attached and a control that
rules out the obvious alternative explanation. A section that said "we improved
the legend" would be worth less and would also be false, since on the pooled
evidence the rewrite did not improve anything.
