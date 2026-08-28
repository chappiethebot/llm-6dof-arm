# Defence numbers — what can be quoted, with intervals and basis

One page. Every number below is rescored under `llm/score.py` at commit
`3000198` (verified behaviourally identical to HEAD: 0 differing verdicts on
20,716 real pred/truth pairs). Intervals are bootstrap over `command_id`
clusters, 2,000 resamples, seed 20260827. **Effective n = 217 commands**, not
1,085 rows — the five repeats are byte-identical for every local model.

**Metric name.** Call it **action match** (verb, target and location all
correct). Never "exact match": the scorer drops `modifiers` when falsy, and
71% of strict-correct rows depend on that rule.

**Two bases, always state which.** `all 217` pools two structurally different
tasks. `grid-only` (n=181, of which 132 carry a location) is the one that
carries the thesis claim; container rows (n=36) name the answer in the command.

---

## 1. Headline — safe to quote [Certain]

| claim | number | basis |
|---|---|---|
| Best system overall | gpt-4o-mini + RAG **0.924** [0.889, 0.956] | all 217 |
| — on spatial reasoning | **0.970** [0.937, 0.996] | grid-only |
| Best local, no GPU at inference | classifier **0.747** [0.687, 0.802] | all 217 |
| — on spatial reasoning | **0.620** [0.537, 0.713] | grid-only |
| Best fine-tuned generative | qwen2.5-3b-lora **0.654** [0.590, 0.719] | all 217 |
| — on spatial reasoning | **0.407** [0.315, 0.500] | grid-only |

**Pareto frontier — three points, nothing else is on it** [Certain]:

| cell | action match | median latency |
|---|---|---|
| classifier | 0.747 | **19 ms** |
| gpt-4o-mini / role_based | 0.855 | 699 ms |
| gpt-4o-mini / RAG | 0.924 | 804 ms |

Every locally-run generative model is strictly dominated. Safe to state.

---

## 2. Structural results — safe [Certain]

- **The classifier emitted zero structurally invalid objects** across all
  1,085 held-out rows. 0 illegal verbs, 0 illegal symbols, 0 field-mask
  violations. Guarantee holds at full n.
- **31.4% of all generative rows never validated** (9,530 of 30,352).
- **deepseek-r1 hides most of what it generates**: 94.7% of its output tokens
  are reasoning the caller never sees, 17.9 hidden per visible token.
  *Caveat: exact for gpt-4o-mini's tokenizer, approximate for the local models.*
- **Rows are harder than columns for every model.** gpt-4o-mini 96.3% column
  vs 82.1% row; the LoRA pair 86–90% column vs 30–41% row.
- **Container vs grid are different tasks**: the answer symbol appears verbatim
  in 36/36 container commands and 0/132 grid commands.

---

## 3. Cost — quotable with its caveat [Certain]

Cloud spend for the whole 5-cell gpt-4o-mini condition: **$0.4236**, exact
(o200k_base *is* its tokenizer). CoT is **$0.1602 of that — 3.0× zero_shot for
15 points of accuracy loss.** That single comparison is the cleanest cost
sentence available.

Local-model token counts use each model's own vocabulary and are exact
arithmetic over an approximate tokenisation; the truly exact route is Ollama's
`eval_count`, now captured but only for future runs.

---

## 4. V4 — reportable only as a robustness check [Certain]

V4 is **post-hoc** (pre-registered in `audit/PREREGISTRATION_v4.md`, but after
V3 was seen). It does **not** replace any number above.

Safe to say:

- The legend rewrite **changed nothing overall**: 21 of 27 cells inside the CI.
- It **does not track model size**. deepseek-r1:7b gained (+0.213 zero_shot)
  while gpt-4o-mini, the largest model, regressed (−0.159 few_shot).
- **The rewrite introduced a new defect** — `Rows run front-to-back: A = far
  from the robot` contradicts itself, and gpt-4o-mini's regression is dominated
  by A→C inversion. Report this as a finding about prompt engineering, not as a
  system improvement.
- Removing the literal cell id and arrows (V4a-lite) **fully removed grid_B2
  anchoring** (40.3% → 11.1%) but only **halved** spurious modifier extraction
  (59.7% → 44.0%) — two failure modes, different causes.

---

## 5. NOT reportable — do not quote

| item | why |
|---|---|
| Any V4 accuracy as a system result | post-hoc condition; V3 stands as the measurement |
| The deepseek V4 gains as caused by the legend fix | mechanism not established; the abandonment criterion partially fired |
| qwen2.5-0.5b-lora V4c row accuracy (0.495) | **inconclusive** — between the pre-registered bands |
| Any per-token cost or throughput for local models | exact counts need `eval_count`, captured only from now on |
| Anything about TTFT / per-stage latency | not recoverable; the client made blocking calls |
| Energy per command | never measured |
| Vision or real-arm results | Phase 6 and 7 have zero lines of implementation |
| `results_v4a_local.CONTAMINATED.csv` | two concurrent writers, 22.3% disagreement; quarantined |

---

## 6. The question to expect first [Likely]

**"Your title promises vision-based manipulation — where is it?"** Phase 6 has
no implementation. No number on this page answers that; have the scope answer
ready.

Second most likely: **"Is 217 commands enough?"** Median CI half-width is
**5.8 points**, so any difference under ~6 points is not claimable. That is
stated on every table.
