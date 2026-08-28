# Pre-registration — V4 legend and vocabulary conditions

**Written before any V4 measurement was run.** Committed prior to execution so
that the hypotheses below cannot be adjusted after seeing results.

Author: the thesis author · Date: 2026-08-28 · Repo state: see the commit carrying this file.

---

## 0. Status of this work: post-hoc, and reported as such

The V3 measurement — 28 cells, 30,380 rows on `held_out_v3` — is complete and
is the thesis result. Everything described here was **designed after seeing
those results**, and specifically after an audit found the vocabulary defects
listed below.

That makes every condition here **post-hoc**. It is pre-registered to constrain
what can be claimed from it, not to disguise its status. Three commitments:

1. **V4 does not replace V3.** The V3 numbers remain the reported measurement.
   V4 is a robustness check reported alongside them.
2. **V4 runs are not comparable with V3 row-for-row.** The prompt differs, so a
   V4 cell and a V3 cell are different conditions, not two samples of one.
3. **No V4 result will be used to select anything** — not a checkpoint, not a
   strategy, not a model. The V3 selections stand.

If V4 improves accuracy, the honest claim is *"a legend defect accounted for
part of the V3 error, and correcting it recovers X points"* — not *"the system
achieves V4's number"*.

---

## 1. What is being tested, and why

The audit (`audit/vocab_conflicts.md`, `audit/AUDIT_REPORT.md`) established
three defects in the V3 prompt and data:

| # | defect | measured cost under V3 |
|---|---|---|
| A | `middle` labels **both** row B and column 2 in the legend | 871 rows collapsed onto `grid_B2`, 91.4% of them column-2 cells |
| B | `max_reach` / `nearest_free_cell` defined with `furthest`, `empty`, `nearest` — words appearing in **zero** commands | `nearest_free_cell` 55% failure: 27.9% dropped, 27.2% confused with `max_reach` |
| C | row vocabulary disjoint across splits: training says `far`/`near`, evaluation says `upper`/`bottom`; column words shared | both LoRA adapters: **100.0%** row accuracy on `dev_v2` vs **30.6% / 40.8%** on `held_out_v3` |

A and B are prompt defects and are addressed by the V4 legend. C is a data
property and is addressed by the V4c dataset.

---

## 2. The three conditions

### V4a — new legend, original data
- **Data:** `held_out_v3.jsonl`, unchanged and sealed.
- **Prompt:** V4 legend (`llm/prompts_v4.py`), `--legend v4`.
- **Cells:** 25 — 5 base models × 4 prompting strategies, 5 RAG cells, 2 LoRA cells.
- **Classifier excluded:** it is an encoder over the command string and never
  receives the legend, so its number cannot change. The V3 value (0.747) is
  reused verbatim.
- **Isolates:** the legend fixes A and B, with data held constant.

### V4b — training-aligned data — **BUILT BUT WILL NOT BE RUN**
- **Data:** `held_out_v4b.jsonl` — `upper`→`far`, `bottom`→`near`.
- **Why it is not run:** the substitution that created `held_out_v3` from the
  training phrasings was exactly `far`→`upper`, `near`→`bottom`. Reverting it
  lands on the training strings. Measured: **all 72 reworded commands are
  verbatim in `seen_v2`** — 56 in `train_v2`, 14 in `dev_v2`, **0 novel**.
  A LoRA model evaluated on it would be recalling its own training data.
  Its expected result (~100%) is already known from the `dev_v2` test.
- **Kept on disk** as the record of why V4c was necessary.

### V4c — third vocabulary
- **Data:** `held_out_v4c.jsonl` — `upper`→`back`, `bottom`→`front`.
  `front` and `back` appear in **neither** training nor `held_out_v3` as row
  words. Ground truth identical on all 217 rows; `held_out_v3.jsonl` byte-identical
  after the build (`sha256 5a767445…`).
  **Verified 0 verbatim collisions** with `seen_v2`, `train_v2`, `dev_v2`.
- **Prompt:** V4 legend.
- **Cells:** 2 — both LoRA models, `zero_shot`.
- **Isolates:** whether the adapters learned a row **concept** or two specific
  strings.

---

## 3. Hypotheses, stated before measurement

**H1 — legend disambiguation reduces the `grid_B2` collapse.**
Predicted: rows collapsing onto `grid_B2` fall by **≥50%** from the V3 baseline
of 871 (rung-1, validated, all 14 truth symbols). Directional prediction:
`grid_A2` and `grid_C2` recover most, since they are the column-2 cells the
ambiguity captured.

**H2 — rewriting the runtime definitions reduces `nearest_free_cell` failure.**
Predicted: the combined drop-plus-confusion rate falls from 55.1% to **below
40%**. Weaker confidence than H1 — the underlying `free`/`want` vocabulary split
across training and evaluation is a data property the legend cannot fix.

**H3 — the LoRA row collapse is vocabulary-specific, not conceptual.**
Predicted on V4c: row accuracy stays **near the V3 level (30–45%)** and the
row-B prediction share stays **high (>70%)**.
This is the discriminating test:
- If V4c row accuracy is ~30–45% → the adapters learned *strings*
  (`far`/`near`) and have no row concept. **Predicted.**
- If V4c row accuracy is ≥70% → they learned a row concept and `upper`/`bottom`
  specifically is the problem. Would falsify H3.
- The `dev_v2` result (100%) and a V4c result near chance together bound the
  effect: perfect on trained strings, chance on any other.

**H4 — the base prompted models are less affected than the fine-tuned ones.**
Predicted: gpt-4o-mini's V4a change is within ±5 points, because its V3 row
accuracy (82.1%) shows it already resolves the row axis without help.

**Null result is a result.** If V4a moves nothing, the conclusion is that the
legend was not the binding constraint and the errors are model capability, not
prompt ambiguity. That is reportable and will be reported.

---

## 3b. AMENDMENT — V4a-lite, added 2026-08-28 after the V4a local run

**Written before the V4a-lite measurement.** The V4a local cells (20/20) had
already been run and inspected when this was added; that is disclosed here
rather than presented as an original hypothesis.

### What V4a showed

Mean delta across the 20 local cells was **-0.000** — no net effect — but that
average hides two opposite movements, six of them beyond the ~6 pt CI:

| cell | V3 | V4 | delta |
|---|---|---|---|
| deepseek-r1:7b zero_shot | 0.064 | 0.278 | **+0.214** |
| deepseek-r1:7b role_based | 0.161 | 0.352 | **+0.190** |
| deepseek-r1:7b cot | 0.106 | 0.208 | **+0.102** |
| llama3.2:3b few_shot | 0.400 | 0.222 | **-0.178** |
| llama3.2:3b rag | 0.322 | 0.167 | **-0.155** |

### Ruled out first

The obvious explanation — the legend contradicting the in-context examples —
was tested and **refuted**. Of the 3 fixed few-shot exemplars, 1 contains
middle/centre/center; of the 230 `train_v2` RAG corpus rows, 49 do. **Zero**
conflict with the V4 rule in either set.

### Diagnosis [Likely]

llama3.2's losses split into two distinct failure modes:

1. **Spurious modifier extraction** (few_shot, 22 of 40 lost rows are
   `validated` with the *correct* location). V4 causes the row word to be
   emitted as a modifier:
   `V3 {"action":"pick_place","target":"toy_car","location":"grid_A1"}` →
   `V4 {... ,"location":"grid_A1","modifiers":["upper"]}`. Non-empty
   `modifiers` is compared, so strict fails on an otherwise correct answer.
2. **grid_B2 anchoring** (rag, 38 of 42 lost rows are `validated` with a
   *wrong* location, systematically `grid_B2`).
   `'move to the upper left corner'` → V3 `grid_A1`, V4 `grid_B2`. This is the
   exact collapse V4 was built to prevent, made worse.

The V4 rule block adds the only two concrete hooks in the whole prompt: the
literal token `grid_B2` (the only cell id written anywhere) and `->` arrow
notation. [Guessing] a 3B model latches onto these where a 7B model uses the
rule as intended.

### The condition

**V4a-lite:** the same rule stated in prose. No cell id spelled out, no arrows.
Nothing else changes. v3 337 chars / v4 937 / v4lite 888.

**Cells:** llama3.2:3b `few_shot` and `rag` only — the two that regressed.

### Predictions, before running

- **H5:** llama3.2 few_shot and rag **recover toward their V3 values**
  (0.400 / 0.322), landing within 6 pts of V3 or better.
- **H6:** the recovery is driven by the `grid_B2` anchoring disappearing in
  rag; predicted `grid_B2` share on rag rows falls below the V4 level.
- **H7 [Guessing]:** spurious `modifiers` extraction in few_shot is *not* fully
  fixed by removing the hooks, because it is caused by the extra prose length
  rather than the two tokens. If few_shot recovers and rag does not, or vice
  versa, the two failure modes have different causes and should be reported
  separately.

**This is the last legend condition.** No fourth variant will be run. If
V4a-lite does not recover llama3.2, the conclusion is that the legend rewrite
helps large models and hurts small ones, and that is the reported result.

---

## 4. Analysis plan, fixed in advance

- **Primary metric:** action match (verb, target and location all correct) —
  never called "exact match"; `modifiers` is dropped when falsy and is not
  under test.
- **Intervals:** bootstrap over `command_id` clusters, 2,000 resamples,
  seed 20260827 — identical to `audit/figures_v3.py`.
- **Effective n:** 217 commands. Repeats are 1 for local models (measured 100%
  byte-identical across 5 repeats) and 5 for gpt-4o-mini (77.1% identical).
- **Reported separately:** grid (181) and container (36) rows, per the finding
  that these are structurally different tasks.
- **Scoring:** frozen scorer at commit `3000198`, unchanged.
- **A difference is only claimed if the clustered 95% intervals do not overlap.**
  Median CI half-width under V3 was 5.8 points, so differences below ~6 points
  will not be claimed regardless of direction.

---

## 5. What would make me abandon the V4 claim

- V4a shows large gains on cells whose V3 errors were **not** `grid_B2`
  collapse or runtime-symbol failures → the legend changed something other
  than the defect and the mechanism is not established.
- V4c row accuracy lands between 45% and 70% → neither hypothesis is supported
  and the result is uninterpretable at n=217.
- Any V4 run turns out to have read the wrong dataset or legend → discard and
  re-run; the manifest records both per run.

---

## 6. Known limitation carried into V4c

`held_out_v4c` contains one command where a row word is used non-spatially:
`'secure the bottle and bring it back'` (no location in ground truth). This is
the §6b conflict class the audit catalogued, newly introduced by choosing
`back` as the row-A word. It affects 1 of 217 commands and none of the 72
reworded ones. Swapping `back`→`rear` (0 occurrences in any split) removes it
and is a one-word change if the result looks sensitive to it.

---

## OUTCOMES (written after measurement, 2026-08-29)

All runs complete: V4a 27 cells, V4a-lite 2, V4c 2. Rescored under `3000198`
(verified behaviourally identical, 0/20,716 verdicts differ). Full table and
figures in `audit/V4_RESULTS.md`.

| # | verdict |
|---|---|
| H1 | NOT SUPPORTED - grid_B2 collapse rose under V4 |
| H2 | NOT SUPPORTED - no real runtime-symbol gain |
| H3 | SPLIT - 3b-lora 0.364 supported; 0.5b-lora 0.495 INCONCLUSIVE |
| H4 | NOT SUPPORTED - gpt-4o-mini few_shot moved -0.159 |
| H5 | SUPPORTED - llama3.2 recovered to within the CI of V3 |
| H6 | SUPPORTED - grid_B2 share 40.3% -> 11.1% |
| H7 | SUPPORTED - modifier pollution only halved; different causes |

**The abandonment criterion in section 5 partially fired.** V4a produced a
large gain on cells whose V3 errors were not grid_B2 collapse (deepseek
zero_shot +0.213 from a 0.065 baseline), so the mechanism is NOT established
for the gains. They are reported as an observed effect without a demonstrated
cause.

**A defect was introduced by the fix** [Certain]: V4's row line reads
`Rows run front-to-back: A = far from the robot`. `front-to-back` implies A is
the front; `far from the robot` says it is the back. The two halves contradict.
gpt-4o-mini few_shot lost 208 rows dominated by A->C inversion
(grid_A1->grid_C1 x49, grid_A3->grid_C3 x43). V4a-lite retains the phrase, so
it does not test this. No fourth variant was run, per the commitment above.
