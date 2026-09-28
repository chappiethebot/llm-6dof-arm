# V4 results

Scored under `llm/score.py` as of commit `3000198`. The rescore tool reports `b8f1385` because that is the last commit touching the file, but its diff against `3000198` is **entirely inside the module docstring**: verified behaviourally on 20,716 real (pred, truth) pairs from the V3 measurement, **0 differing verdicts**. [Certain]

Intervals: bootstrap over `command_id` clusters, 2,000 resamples, seed 20260827 — identical to `audit/figures_v3.py`. Effective n = 217 commands.

## Full cell table

| model | strategy | condition | n | action match | 95% CI | vs V3 | verdict |
|---|---|---|---|---|---|---|---|
| classifier:distilbert | classifier | V3 | 217 | 0.747 | [0.687, 0.802] ±5.8pt |  | baseline |
| ollama:deepseek-r1:7b | cot | V3 | 217 | 0.106 | [0.069, 0.152] ±4.1pt |  | baseline |
| ollama:deepseek-r1:7b | cot | V4a | 216 | 0.208 | [0.153, 0.264] ±5.6pt | +0.102 | **real** |
| ollama:deepseek-r1:7b | few_shot | V3 | 217 | 0.369 | [0.304, 0.433] ±6.5pt |  | baseline |
| ollama:deepseek-r1:7b | few_shot | V4a | 216 | 0.338 | [0.278, 0.403] ±6.2pt | -0.031 | noise |
| ollama:deepseek-r1:7b | rag | V3 | 217 | 0.562 | [0.498, 0.622] ±6.2pt |  | baseline |
| ollama:deepseek-r1:7b | rag | V4a | 216 | 0.514 | [0.449, 0.579] ±6.5pt | -0.048 | noise |
| ollama:deepseek-r1:7b | role_based | V3 | 217 | 0.161 | [0.115, 0.212] ±4.8pt |  | baseline |
| ollama:deepseek-r1:7b | role_based | V4a | 216 | 0.352 | [0.287, 0.417] ±6.5pt | +0.191 | **real** |
| ollama:deepseek-r1:7b | zero_shot | V3 | 217 | 0.065 | [0.037, 0.097] ±3.0pt |  | baseline |
| ollama:deepseek-r1:7b | zero_shot | V4a | 216 | 0.278 | [0.218, 0.338] ±6.0pt | +0.213 | **real** |
| ollama:llama3.2:3b | cot | V3 | 217 | 0.069 | [0.041, 0.106] ±3.2pt |  | baseline |
| ollama:llama3.2:3b | cot | V4a | 216 | 0.093 | [0.056, 0.134] ±3.9pt | +0.023 | noise |
| ollama:llama3.2:3b | few_shot | V3 | 217 | 0.401 | [0.336, 0.470] ±6.7pt |  | baseline |
| ollama:llama3.2:3b | few_shot | V4a | 216 | 0.222 | [0.167, 0.278] ±5.6pt | -0.179 | **real** |
| ollama:llama3.2:3b | few_shot | V4a-lite | 216 | 0.315 | [0.250, 0.380] ±6.5pt | -0.086 | noise |
| ollama:llama3.2:3b | rag | V3 | 217 | 0.323 | [0.263, 0.387] ±6.2pt |  | baseline |
| ollama:llama3.2:3b | rag | V4a | 216 | 0.167 | [0.116, 0.218] ±5.1pt | -0.156 | **real** |
| ollama:llama3.2:3b | rag | V4a-lite | 216 | 0.259 | [0.204, 0.324] ±6.0pt | -0.063 | noise |
| ollama:llama3.2:3b | role_based | V3 | 217 | 0.438 | [0.373, 0.502] ±6.5pt |  | baseline |
| ollama:llama3.2:3b | role_based | V4a | 216 | 0.417 | [0.352, 0.481] ±6.5pt | -0.021 | noise |
| ollama:llama3.2:3b | zero_shot | V3 | 217 | 0.484 | [0.419, 0.548] ±6.5pt |  | baseline |
| ollama:llama3.2:3b | zero_shot | V4a | 216 | 0.426 | [0.361, 0.491] ±6.5pt | -0.058 | noise |
| ollama:qwen2.5-0.5b-lora | zero_shot | V3 | 217 | 0.396 | [0.332, 0.461] ±6.5pt |  | baseline |
| ollama:qwen2.5-0.5b-lora | zero_shot | V4a | 216 | 0.380 | [0.315, 0.449] ±6.7pt | -0.017 | noise |
| ollama:qwen2.5-0.5b-lora | zero_shot | V4c | 216 | 0.468 | [0.403, 0.528] ±6.2pt | +0.071 | noise |
| ollama:qwen2.5-3b-lora | zero_shot | V3 | 217 | 0.654 | [0.590, 0.719] ±6.5pt |  | baseline |
| ollama:qwen2.5-3b-lora | zero_shot | V4a | 216 | 0.667 | [0.602, 0.727] ±6.2pt | +0.012 | noise |
| ollama:qwen2.5-3b-lora | zero_shot | V4c | 216 | 0.644 | [0.574, 0.708] ±6.7pt | -0.011 | noise |
| ollama:qwen2.5:0.5b | cot | V3 | 217 | 0.014 | [0.000, 0.032] ±1.6pt |  | baseline |
| ollama:qwen2.5:0.5b | cot | V4a | 216 | 0.014 | [0.000, 0.032] ±1.6pt | +0.000 | noise |
| ollama:qwen2.5:0.5b | few_shot | V3 | 217 | 0.055 | [0.028, 0.088] ±3.0pt |  | baseline |
| ollama:qwen2.5:0.5b | few_shot | V4a | 216 | 0.000 | [0.000, 0.000] ±0.0pt | -0.055 | **real** |
| ollama:qwen2.5:0.5b | rag | V3 | 217 | 0.069 | [0.037, 0.106] ±3.5pt |  | baseline |
| ollama:qwen2.5:0.5b | rag | V4a | 216 | 0.051 | [0.023, 0.079] ±2.8pt | -0.018 | noise |
| ollama:qwen2.5:0.5b | role_based | V3 | 217 | 0.124 | [0.083, 0.171] ±4.4pt |  | baseline |
| ollama:qwen2.5:0.5b | role_based | V4a | 216 | 0.088 | [0.051, 0.130] ±3.9pt | -0.036 | noise |
| ollama:qwen2.5:0.5b | zero_shot | V3 | 217 | 0.138 | [0.097, 0.184] ±4.4pt |  | baseline |
| ollama:qwen2.5:0.5b | zero_shot | V4a | 216 | 0.134 | [0.093, 0.181] ±4.4pt | -0.004 | noise |
| ollama:qwen2.5:3b | cot | V3 | 217 | 0.452 | [0.387, 0.521] ±6.7pt |  | baseline |
| ollama:qwen2.5:3b | cot | V4a | 216 | 0.407 | [0.343, 0.472] ±6.5pt | -0.044 | noise |
| ollama:qwen2.5:3b | few_shot | V3 | 217 | 0.447 | [0.382, 0.516] ±6.7pt |  | baseline |
| ollama:qwen2.5:3b | few_shot | V4a | 216 | 0.472 | [0.403, 0.537] ±6.7pt | +0.025 | noise |
| ollama:qwen2.5:3b | rag | V3 | 217 | 0.567 | [0.502, 0.631] ±6.5pt |  | baseline |
| ollama:qwen2.5:3b | rag | V4a | 216 | 0.579 | [0.514, 0.644] ±6.5pt | +0.012 | noise |
| ollama:qwen2.5:3b | role_based | V3 | 217 | 0.373 | [0.309, 0.438] ±6.5pt |  | baseline |
| ollama:qwen2.5:3b | role_based | V4a | 216 | 0.398 | [0.333, 0.463] ±6.5pt | +0.025 | noise |
| ollama:qwen2.5:3b | zero_shot | V3 | 217 | 0.392 | [0.332, 0.456] ±6.2pt |  | baseline |
| ollama:qwen2.5:3b | zero_shot | V4a | 216 | 0.444 | [0.380, 0.509] ±6.5pt | +0.053 | noise |
| openai:gpt-4o-mini | cot | V3 | 217 | 0.685 | [0.626, 0.740] ±5.7pt |  | baseline |
| openai:gpt-4o-mini | cot | V4a | 217 | 0.749 | [0.694, 0.805] ±5.5pt | +0.065 | noise |
| openai:gpt-4o-mini | few_shot | V3 | 217 | 0.842 | [0.792, 0.888] ±4.8pt |  | baseline |
| openai:gpt-4o-mini | few_shot | V4a | 217 | 0.684 | [0.623, 0.745] ±6.1pt | -0.159 | **real** |
| openai:gpt-4o-mini | rag | V3 | 217 | 0.924 | [0.887, 0.956] ±3.5pt |  | baseline |
| openai:gpt-4o-mini | rag | V4a | 217 | 0.892 | [0.852, 0.930] ±3.9pt | -0.031 | noise |
| openai:gpt-4o-mini | role_based | V3 | 217 | 0.855 | [0.806, 0.900] ±4.7pt |  | baseline |
| openai:gpt-4o-mini | role_based | V4a | 217 | 0.799 | [0.747, 0.850] ±5.2pt | -0.056 | noise |
| openai:gpt-4o-mini | zero_shot | V3 | 217 | 0.835 | [0.789, 0.881] ±4.6pt |  | baseline |
| openai:gpt-4o-mini | zero_shot | V4a | 217 | 0.798 | [0.745, 0.848] ±5.2pt | -0.037 | noise |

## V4c — row vs column accuracy (the H3 test)

| model | condition | grid rows | row acc | column acc |
|---|---|---|---|---|
| ollama:qwen2.5-0.5b-lora | V3 | 539 | **0.306** | 0.900 |
| ollama:qwen2.5-3b-lora | V3 | 539 | **0.408** | 0.861 |
| ollama:qwen2.5-0.5b-lora | V4a | 107 | **0.308** | 0.664 |
| ollama:qwen2.5-3b-lora | V4a | 107 | **0.411** | 0.888 |
| ollama:qwen2.5-0.5b-lora | V4c | 107 | **0.495** | 0.785 |
| ollama:qwen2.5-3b-lora | V4c | 107 | **0.364** | 0.907 |

Pre-registered thresholds: **30–45% = memorised strings**, **≥70% = learned a row concept**, between = inconclusive.

- `ollama:qwen2.5-0.5b-lora`: row accuracy on front/back = **0.495** -> **INCONCLUSIVE** — 49.5% falls between the bands, forced into neither
- `ollama:qwen2.5-3b-lora`: row accuracy on front/back = **0.364** -> **memorised strings** (in the 30–45% band)


---

## Hypotheses, answered

| # | hypothesis | verdict |
|---|---|---|
| **H1** | legend disambiguation cuts the `grid_B2` collapse by >=50% | **NOT SUPPORTED.** llama3.2 `grid_B2` predictions *rose* under V4 (rag 10.1% -> 40.3%). V4a-lite reversed it (11.1%), so the rule works only without the literal cell id. |
| **H2** | runtime-symbol rewrite cuts `nearest_free_cell` failure below 40% | **NOT SUPPORTED.** No cell shows a real gain attributable to the runtime lines; every LoRA delta is inside the CI. |
| **H3** | LoRA row collapse is vocabulary-specific, not conceptual (predicted 30-45% on v4c) | **SPLIT.** qwen2.5-3b-lora **0.364 -> supported** (memorised strings). qwen2.5-0.5b-lora **0.495 -> INCONCLUSIVE**, between the bands; not forced. |
| **H4** | base prompted models less affected than fine-tuned; gpt-4o-mini within +-5 pts | **NOT SUPPORTED.** gpt-4o-mini `few_shot` moved **-0.159**, the second-largest regression in the study. The fine-tuned models were the *least* affected (both inside the CI). |
| **H5** | V4a-lite recovers llama3.2 few_shot and rag to within 6 pts of V3 | **SUPPORTED.** few_shot 0.222 -> 0.315 (V3 0.401, gap -0.086) and rag 0.167 -> 0.259 (V3 0.323, gap -0.063). Both now **inside the CI of V3** i.e. noise. |
| **H6** | recovery driven by `grid_B2` anchoring disappearing | **SUPPORTED.** rag `grid_B2` share 40.3% (V4a) -> 11.1% (lite) vs 10.1% (V3) - fully back to baseline. |
| **H7** | modifier pollution is *not* fully fixed by removing the hooks, i.e. the two modes have different causes | **SUPPORTED, and this is the informative one.** Non-empty `modifiers`: rag 10.6% (V3) -> 59.7% (V4a) -> **44.0%** (lite); few_shot 13.4% -> 41.7% -> **25.5%**. Anchoring fully fixed, pollution only partly. The two failure modes have different causes. |

---

## Summary

### 1. The legend fix does NOT track parameter count [Certain]

The "helps large, hurts small" reading from the partial data is **wrong** once
the cloud cells land. Real movements (outside the CI):

| model | params | cell | delta |
|---|---|---|---|
| deepseek-r1:7b | 7B | zero_shot | **+0.213** |
| deepseek-r1:7b | 7B | role_based | **+0.191** |
| deepseek-r1:7b | 7B | cot | **+0.102** |
| gpt-4o-mini | (cloud, large) | few_shot | **-0.159** |
| llama3.2:3b | 3B | few_shot | **-0.179** |
| llama3.2:3b | 3B | rag | **-0.156** |
| qwen2.5:0.5b | 0.5B | few_shot | **-0.055** |

The largest model in the study regresses alongside the smallest. **20 of 27
cells are noise** (7 real moves; `audit/v4_cells.csv`, V4a rows: 20 noise, 7 real;
corrected 2026-09-28, previously "21 of 27"). What predicts the sign is not size but *which model was
already failing on the row axis*: deepseek's V3 zero_shot was 0.065, so almost
any change helps; gpt-4o-mini's was 0.842, so there is only downside.

### 2. Three distinct failure modes, not two [Certain]

- **grid_B2 anchoring** - the model copies the only cell id written in the
  prompt. llama3.2 rag 10.1% -> 40.3%. **Cause: the literal `grid_B2` token
  and `->` arrows.** Fully fixed by V4a-lite (11.1%).
- **Spurious modifier extraction** - correct location, but the row word is
  emitted as a modifier, so strict fails on an otherwise right answer.
  llama3.2 rag 10.6% -> 59.7%. **Cause: the extra prose itself**, not the
  tokens - V4a-lite only halves it (44.0%).
- **Row inversion, newly found in gpt-4o-mini** [Likely] - `few_shot` lost 208
  rows, 203 of them `validated`, dominated by `grid_A1 -> grid_C1` (49) and
  `grid_A3 -> grid_C3` (43). A -> C, systematically.

### 3. The row inversion is my error, introduced by the fix [Certain]

V4 rewrote the row line as:

```
Rows run front-to-back:   A = far from the robot,  B = centre,  C = near the robot.
```

`front-to-back` implies A is the front. `far from the robot` says A is the
back. **The two halves of one line contradict each other.** V3 had no such
phrase. This is precisely the class of vocabulary conflict
`audit/vocab_conflicts.md` was written to catalogue, and I introduced a new
instance of it while fixing the old ones. V4a-lite keeps the same phrase, so it
does not test this - the gpt-4o-mini regression is unexplained by the lite
condition and would need a fourth variant, which is **out of scope by
pre-registration**.

### 4. V4c verdict [Certain]

| model | V3 row acc | V4c row acc | verdict |
|---|---|---|---|
| qwen2.5-3b-lora | 0.408 | **0.364** | memorised strings (30-45% band) |
| qwen2.5-0.5b-lora | 0.306 | **0.495** | **INCONCLUSIVE** |

Column accuracy stays high on the unseen vocabulary (0.785 / 0.907), so the
row/column asymmetry survives: the adapters still read columns and guess rows.
For the 3B this is clean support that it learned strings, not a concept. The
0.5B is reported as inconclusive rather than forced.
