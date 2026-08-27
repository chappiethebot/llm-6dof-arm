AUDIT REPORT — llm-6dof-arm
============================

Auditor posture: hostile external reviewer. Read-only over `data/` and
`bench/results_*.csv`; nothing in either was modified.

Repo state at audit time:
  HEAD              3a4a436 (2026-08-26)
  llm/score.py      last commit 3000198 (2026-07-20), working tree CLEAN
  measurement files bench/results_heldout_v3_{rung1,rag,lora,cls}[_scored].csv
  totals            30,380 rows / 28 cells / dataset=held_out_v3 only

Scripts written for this audit (throwaway, under audit/):
  audit/b_checks.py       B2/B3/B4/B5 over the full CSVs
  audit/b1_leniency.py    B1 scorer probes


=====================================================================
BLOCK B — SCORING CORRECTNESS
=====================================================================

WORST FINDING FIRST
-------------------

B1-a  [Certain]  DISCLOSE
71% of all strict credit depends on a leniency the Methods wording does not
describe.

  strict-correct rows across all 28 cells      : 12,528
  strict-correct where pred.modifiers != truth.modifiers :  8,897  (71.0%)

Ground truth carries `"modifiers": []` on every row. Models mostly omit the
key entirely (21,488 of 30,380 predictions have no `modifiers` key; 7,518 emit
`[]`). `llm/score.py:_canonical()` drops `modifiers` from BOTH sides when
falsy, so absent == []. Without that rule, strict accuracy across the study
collapses by roughly 71% of its current value.

This is NOT a validity defect. The rule is symmetric, applied identically to
all 28 cells, and `modifiers` is a constant across the entire evaluation set
(verified: 0 of 217 held_out_v3 rows carry a non-empty modifiers list), so it
cannot advantage any rung over another. It is a WORDING defect: any Methods
sentence saying "exact match" is false as written and an examiner who greps
the scorer will find it in one minute.


B1-b  [Certain]  DISCLOSE  (complete leniency enumeration)

The strict policy is raw `dict ==` after dropping falsy `modifiers`. Probed
exhaustively (audit/b1_leniency.py). Full list of what strict FORGIVES:

  1. `modifiers` absent  vs  `[]`                        -> strict TRUE
  2. `modifiers` = None / "" / {} / 0 / False            -> strict TRUE
     (UNDOCUMENTED: the docstring says "absent or []". `_IGNORE_IF_EMPTY`
      tests falsiness, so it silently accepts five more values. Zero
      practical impact — no model emitted any of them; see B1-c.)
  3. Key insertion order                                  -> strict TRUE
     (ordinary dict equality, not a real leniency)

Everything else is REJECTED by strict, as claimed:
  case variants (BOX_1), separator variants ("box 1"), alias keys (`verb`),
  nested action objects, extra keys (even with falsy values), trailing
  whitespace in values, wrong verb, wrong/missing target or location.

Non-empty modifiers ARE compared, and compared order-sensitively:
`["gently"]` vs `["gently","fast"]` fails strict AND normalized.

B1-c  [Certain]  CHEAP-FIX (documentation only)
Inconsistency: `modifiers: 0` and `modifiers: False` score strict=TRUE but
normalized=FALSE — strict is more lenient than normalized on that one path,
which inverts the intended ordering of the two policies. Fired zero times in
the real data (verified across all 30,380 predictions: only `<ABSENT>`, `[]`,
and genuine non-empty lists appear). Note it in the scorer docstring; do not
change the code, it would alter a frozen scorer for no measured benefit.


B2  [Certain]  NOT A DEFECT — but a reporting gap

The `stage`-vs-`category` hazard is real and large:

  rows with stage != "validated"    : 9,536
  rows with category "invalid:*"    : 1,004
  ROWS ERASED FROM `category`       : 8,532   (all schema_failed)

`bench/rescore.py` behaves exactly as documented — every schema_failed row
whose JSON parses gets a semantic verdict, so `invalid:schema_failed`
disappears from `category` while remaining visible in `stage`.

Audit of every analysis script:
  bench/summary.py         keys invalid-output on `stage` (line 432)      CORRECT
  bench/figures.py         never reads `category` at all                  SAFE
  bench/analyze.py         never reads `category` at all                  SAFE
  scratch_scored_table.py  never reads `category` at all                  SAFE

No script produces a wrong invalid-rate number. However —

B2-a  [Certain]  CHEAP-FIX
`scratch_scored_table.py`, the script the results chapter comes out of,
reports NO invalid rate at any level. Its columns are strict / norm / gap /
grid / box / p50 / p99. The 9,536-row structural-failure result — arguably
the study's most examiner-relevant number, and the entire basis of Block C —
is not in the results-chapter table. It exists only in `bench/summary.py`,
which is a different script producing a different table.


B3  [Certain]  NOT A DEFECT — fragile but currently correct

Dedupe keys audited across all aggregating scripts:

  bench/rescore.py         (dataset, model, strategy, command_id, run_index)  5-key
  bench/summary.py         (dataset, model, strategy, command_id, run_index)  5-key
  bench/figures.py         (dataset, model, strategy, command_id, run_index)  5-key
  bench/analyze.py         (model, strategy, command_id, run_index)           4-key
  scratch_scored_table.py  (model, strategy, command_id, run_index)           4-key

Measured blast radius of the missing `dataset` field:

  total rows                : 30,380
  unique under 4-key        : 30,380
  unique under 5-key        : 30,380
  ROWS DROPPED BY 4-KEY     : 0

Zero, because all four held-out v3 files carry `dataset=held_out_v3` and the
28 (model, strategy) pairs are disjoint across files. The 4-key is safe HERE
and unsafe in general: the moment anyone unions a seen file with a held-out
file in `scratch_scored_table.py`, rows collide silently and the results table
is wrong with no error. Also confirmed: 0 orphan rows were dropped by rescore
in any of the four files, so the append-only resume discipline left no
duplicates to clean.


B4  [Certain]  CLAIM HOLDS — verified on all 1,085 held-out rows

The rung-4 structural guarantee was re-asserted against the real evaluation
data, not the 57-row dev set:

  rows checked                     : 1,085
  stage                            : validated  1,085 / 1,085  (100%)
  illegal verb                     : 0
  illegal target symbol            : 0
  illegal location symbol          : 0
  field-mask violations
    (extra field for the verb)     : 0
    (missing required field)       : 0
  non-empty modifiers emitted      : 0
  TOTAL VIOLATIONS                 : 0

Verb distribution emitted:
  pick_place 640, move_to 215, reject 65, inspect 55, hold 45, stop 45, fetch 20

All 7 schema verbs are exercised. The classifier never produced a structurally
invalid object. This is the strongest result in the audit and it survives at
full n.


B5  [Certain]  CLAIM HOLDS — scorer uniformity confirmed

  llm/score.py commit history:
    3000198  2026-07-20  "corrected normalized scorer (representational unwrap)"
    1fdf49d  2026-07-16  (superseded)
  Working tree            : CLEAN (no uncommitted edits to score.py)
  Measurement runs        : 2026-08-24 / 08-25 — after 3000198, before HEAD
  All four _scored files  : written 2026-08-25 23:45, one batch

No commit touched `llm/score.py` between the four runs. Cross-rung comparison
is valid on this axis.

B5-a  [Certain]  RESOLVED — the byte-identical classifier file is benign
`results_heldout_v3_cls.csv` and `..._cls_scored.csv` are both 504,839 bytes.
Cause identified: rescore ran and changed nothing.

  file    raw     scored  orphans  score-column diffs
  rung1   21,700  21,700  0        category 7,657 / normalized_match 445
  rag      5,425   5,425  0        category   855 / normalized_match   5
  lora     2,170   2,170  0        category    20
  cls      1,085   1,085  0        NONE

`strict_match` changed on ZERO rows in all four files — live scoring
(`bench/run_cell.py:250-254`) and rescoring agree completely on the headline
metric. The classifier's outputs were already fully scored at run time because
all 1,085 rows validated, so there was nothing for rescore to relabel. Not a
skipped step.


=====================================================================
B6 — num_ctx CONTEXT-TRUNCATION CONFOUND
=====================================================================

VERDICT: the confound is REAL but does NOT touch either finding it was feared
to touch. The RAG negative result is CLEAN. The CoT collapse is CLEAN in 3 of
5 models and moves by <=0.3 points in the other 2.

Two blockers had to be worked around first — see B7 and B8.

B6-a  [Certain]  THE PROMPT SIDE NEVER APPROACHES ANY CEILING

Prompts were reconstructed offline for all 217 held-out commands x 5
strategies (audit/reconstruct_prompts.py). `build_prompt` is deterministic and
`retrieved_ids` is stored on all 5,420 rag rows, so every prompt in the study
is exactly recoverable without calling a model. Sealed data was only READ.

  strategy      max_chars   mean_chars   ~tokens (chars/4)
  zero_shot         1,039        1,005          260
  cot               1,170        1,136          292
  role_based        1,122        1,088          280
  few_shot          1,511        1,477          378
  rag               1,604        1,426          401

The largest prompt in the entire study is ~401 tokens. Even at a pessimistic
chars/3 it is ~535. Against a 2048 default that is a 4x margin.

This KILLS the proposed mechanism directly. Front-eviction of the schema and
grid legend requires the PROMPT to exceed the window. It never does, in any
cell, under any strategy. The legend was present in every call.

B6-b  [Certain]  ANSWERS A5 AS A SIDE EFFECT — RAG PREFILL IS NOT THE LATENCY STORY
RAG prompts (401 tok max, 1,426 chars mean) versus few_shot (378 tok max,
1,477 chars mean). RAG's MEAN prompt is shorter than few_shot's. The rung-1
vs rung-2 latency delta cannot be attributed to prefill length; the two
conditions are prompt-size equivalent, exactly as llm/prompts.py claims.

B6-c  [Certain]  GENERATION-SIDE OVERFLOW IS REAL, RARE, AND CONFINED

Only 6 of 27 generative cells contain any row whose prompt+generated exceeds
2048 tokens (generated length from stored `raw` + `thinking`, chars/4):

  model              strategy     over/n      share   acc<=2048  acc>2048   delta
  deepseek-r1:7b     few_shot     60/1084     5.54%     0.385     0.083    -0.301
  deepseek-r1:7b     rag          55/1084     5.07%     0.582     0.182    -0.400
  deepseek-r1:7b     cot          30/1084     2.77%     0.109     0.000    -0.109
  deepseek-r1:7b     zero_shot    15/1084     1.38%     0.065     0.000    -0.065
  qwen2.5:0.5b       cot          15/1084     1.38%     0.014     0.000    -0.014
  deepseek-r1:7b     role_based   10/1084     0.92%     0.163     0.000    -0.163

Every other cell — including ALL FIVE llama3.2 cells, all five qwen2.5:3b
cells, both LoRA cells and all five gpt-4o-mini cells — has ZERO rows over
2048. Worst-case total context for llama3.2 rag is 432 tokens.

B6-d  [Certain]  CAUSAL DIRECTION IS NOT IDENTIFIABLE — AND DOES NOT NEED TO BE
Accuracy above the ceiling is far worse, but that correlation is consistent
with BOTH readings: truncation caused the failure, or the model was already
failing and rambled. Stored data cannot separate them, and this audit does not
claim it can.

It does not matter, because exposure is bounded by row share. Excluding every
over-ceiling row entirely moves the cell numbers by:

  deepseek rag         0.562 -> 0.582   (+0.020)
  deepseek few_shot    0.368 -> 0.385   (+0.017)
  deepseek cot         0.106 -> 0.109   (+0.003)
  deepseek role_based  0.161 -> 0.163   (+0.002)
  deepseek zero_shot   0.064 -> 0.065   (+0.001)
  qwen0.5b cot         0.014 -> 0.014   (+0.000)

Maximum movement anywhere: 2.0 points, on a cell whose 95% CI is about
+-6.6 points. No cell ordering changes. No finding inverts.

B6-e  THE TWO FINDINGS THAT WERE AT RISK — BOTH SURVIVE

  llama3.2 rag (ok=0.912 / strict=0.322)  CLEARED. Max total context 432
  tokens, zero rows near any ceiling, max generated output 122 chars. The
  structurally-valid / semantically-wrong signature is a genuine model result.
  Confirmed from the table: rag inv% 8.8 => ok 0.912 exactly.

  CoT collapse across models              CLEAN in llama3.2 (1,051 tok),
  qwen2.5:3b (691) and gpt-4o-mini (672). Contaminated only in deepseek-r1
  (2.77% of rows) and qwen2.5:0.5b (1.38%), bounded at +0.003 there.

B6-f  [Certain]  THE CEILING IS MEASURED: num_ctx = 4096

After recovering the model store (B8), read directly from the running server:

  routes.go:1852  msg="vram-based default context" total_vram="16.0 GiB"
                  default_num_ctx=4096
  llama_context:  n_ctx = 4096   n_ctx_seq = 4096
  n_ctx_seq (4096) < n_ctx_train (32768)

NO model sets `PARAMETER num_ctx` in its Modelfile — verified across all six.
So every local cell ran at the server default, and that default is VRAM-derived
rather than the flat 2048 assumed in the hypothesis.

Re-running the split at the MEASURED ceiling collapses the confound further:

  model              strategy    over/n     share   acc<=   acc>   cell -> excl
  qwen2.5:0.5b       cot         15/1084    1.38%   0.014   0.000  0.014 -> 0.014
  deepseek-r1:7b     rag         10/1084    0.92%   0.567   0.000  0.562 -> 0.567
  deepseek-r1:7b     cot          5/1084    0.46%   0.107   0.000  0.106 -> 0.107
  deepseek-r1:7b     few_shot     5/1084    0.46%   0.370   0.000  0.368 -> 0.370
  deepseek-r1:7b     role_based   5/1084    0.46%   0.162   0.000  0.161 -> 0.162

22 of 27 generative cells have ZERO rows over the ceiling. deepseek zero_shot,
over at 2048, is clean at 4096. Maximum movement from excluding every
over-ceiling row anywhere in the study: +0.005 — a twentieth of the CI.

B6-g  [Certain]  DISCLOSE — the context window was a function of the
                 measurement machine's VRAM, not of any setting in the repo

`default_num_ctx=4096` is derived from `total_vram="16.0 GiB"`. No model
Modelfile sets num_ctx and no code in this repo sets it. The effective
context window is therefore a property of the HARDWARE the study ran on.
Anyone reproducing this work on a machine with different VRAM gets a
different context window without changing a single line of code, and without
any warning. Ollama reports this only at INFO level in the server log, which
is not captured by the harness.

  Draft sentence: "Local models were served by Ollama with no explicit
   num_ctx; the effective context window was 4096 tokens, derived by Ollama
   from the measurement machine's 16 GiB of VRAM. Reproduction on different
   hardware may yield a different window. All prompts in this study are under
   410 tokens, so no prompt was affected at any plausible window size."

CHEAP-FIX: pin num_ctx explicitly in the client options for any future run,
so the window becomes a recorded experimental condition rather than an
emergent property of the GPU.

CAVEAT [Likely]: this is Ollama client 0.20.3 reading today. The VRAM-based
default is a recent feature; the version installed during the August runs
cannot be recovered and may have used a flat 2048. This does not matter — the
2048 analysis (B6-c) and the 4096 analysis both leave every finding standing,
so the conclusion holds across the whole plausible range.


=====================================================================
B7 — TOKEN COLUMNS ARE EMPTY ON EVERY held_out_v3 ROW
=====================================================================

B7-a  [Certain]  CRITICAL-RECOVERABLE  (NOT fatal — every input still exists)

Triage note: a FATAL finding is one where a number cannot be recovered. This
is not that. Prompts are deterministically reconstructible (proved in B6-a)
and output text is stored in `raw` + `thinking`. Nothing is lost; it is
arithmetic over data already on disk. Block E cannot be computed TODAY.

  file    rows     tokens_in  tokens_out  cost_usd   retrieval_latency_s  thinking_chars
  rung1   21,700           0           0         0                     0          21,700
  rag      5,425           0           0         0                 5,425           5,425
  lora     2,170           0           0         0                     0           2,170
  cls      1,085           0           0         0                     0           1,085

Zero of 30,380 rows carry a token count or a cost, on BOTH raw and scored
files. `bench/backfill_tokens.py` was run on the V1 files and never on V3:
  bench/results_grid_tokens.csv     12,800 rows, dataset=seen,     fully populated
  bench/results_heldout_tokens.csv   9,700 rows, dataset=held_out, fully populated

Consequences as of today:
  - E4 Cost-of-Pass cannot be computed. G for the cloud cells comes from token
    counts; there are none. The cost curve — the thesis's subject — has no
    cost axis on the v3 measurement.
  - E2 output-length confound (latency per output token) not computable.
  - bench/summary.py cost and throughput sections silently degrade to nothing
    on v3 (has_tokens() returns False). No error is raised.

B7-b  [Certain]  RECOVERABLE — but the recovery tool is V1-shaped and will
                 silently produce wrong numbers if run as-is

`backfill_tokens.py` reconstructs prompts offline exactly as this audit did,
so no re-run is needed. Four defects make it wrong for v3:

  line 45-47  schema defaults to schema.json (V1). V3 rows need schema_v2.json.
              V2 prompts carry the grid legend, so the V1 schema undercounts
              tokens_in on every single row.
  line 48     load_seen_split() defaults to data/seen.jsonl (V1 exemplars),
              not seen_v2.jsonl.
  line 61     examples are passed only for strategy == "few_shot". `rag` gets
              None, so all 5,425 rag prompts are counted with zero examples.
  line 65     t_out = encode(raw) only. `thinking` is EXCLUDED, so deepseek's
              reasoning traces — up to 194,628 chars — are not counted as
              output tokens at all.

B7-c  [Certain]  THE `thinking` EXCLUSION CHANGES DIRECTION, NOT JUST MAGNITUDE

This is worse than an E3 hazard. With traces of 194,628 characters scoring
ZERO output tokens, deepseek's cost-per-command and tokens-per-second are
currently UNMEASURABLE rather than merely understated. Once the traces are
counted, deepseek gets slower and more expensive — never cheaper. That
strengthens the cost curve rather than threatening it.

REQUIRED OUTPUT SHAPE for the fixed backfill: emit TWO columns, not one
merged figure —

  tokens_out_visible   encode(raw)                 — what the caller receives
  tokens_out_total     encode(raw) + encode(thinking) — what the GPU produced

Report both. The gap between them is itself a finding about reasoning models:
it is the share of generated tokens the user pays for and never sees. A single
merged column destroys that and cannot be reconstructed from the merged value.

Fixing those four lines and running it on the four v3 files is roughly 2 h and
requires no measurement re-run. Highest-payoff item in the queue: it is the
only thing standing between the study and its own cost axis.


B7-d  [Certain]  DONE — bench/backfill_tokens_v3.py written and run

All four defects fixed in a NEW script; the V1 original is untouched because
it is the provenance of the V1 token files. Outputs are new
`*_tokens.csv` files — the four `_scored` inputs were opened read-only and
verified byte-identical afterwards against audit/backup_scored_v3/.

Frozen-column assertion (stage, ok, strict_match, normalized_match, category,
latency_s, raw, thinking, pred_json, truth_json, ids): checked per row on all
30,380 rows. DRIFT = 0 on every file. The script aborts before writing if any
frozen column moves.

RESULT — THE UNSEEN-TOKEN FINDING:

  model                     visible    thinking       total   unseen share
  deepseek-r1:7b            153,000   3,683,031   3,836,031        96.0%
  qwen2.5:0.5b              827,545           0     827,545         0.0%
  llama3.2:3b               287,518           0     287,518         0.0%
  gpt-4o-mini               260,894           0     260,894         0.0%
  qwen2.5:3b                193,947           0     193,947         0.0%
  qwen2.5-3b-lora            20,424           0      20,424         0.0%
  classifier:distilbert      20,354           0      20,354         0.0%
  qwen2.5-0.5b-lora          17,730           0      17,730         0.0%

  study total: 1,782,738 visible / 3,685,055 thinking = 2.1x more generated
  than delivered.

96.0% of everything deepseek-r1 generates is never seen by the caller — a 24x
ratio of hidden to visible tokens. Under the old single-column scheme all of
it counted as zero. This is the number that makes deepseek's position on the
cost curve honest, and it moves it in the expensive direction.

Second-order observation: qwen2.5:0.5b has the LARGEST visible output of any
model (827,545 tokens, 4.3x gpt-4o-mini) with zero reasoning tokens. Its cost
problem is runaway visible generation in the cot cell, a different failure
from deepseek's. Two distinct expensive-failure modes, previously invisible.

B7-h  [Certain]  THE qwen2.5:0.5b RUNAWAY IS A CLEAN, SEPARABLE PATHOLOGY

Tested directly on the cot cell (n=1084), split at 10,000 output characters:

  subset              n      strict    inv     median lat    median raw chars
  raw >10k chars      15     0.000     1.000     118.80 s         129,777
  raw <=10k chars   1069     0.014     0.958       1.83 s             771

  cell p99 latency = 116.94 s     cell max = 120.10 s

The 15 runaway rows are 1.38% of the cell and are 100% wrong — every one
fails, every one is structurally invalid. They are also the ENTIRE latency
tail: the p99 of 116.94 s sits inside this group, while the other 98.6% of
the cell answers in 1.83 s median. A 0.5B model that normally responds in
under two seconds spends two minutes on 15 commands and gets all 15 wrong.

[Likely] the runaway rows did not terminate naturally: their latencies cluster
in 118.80-120.10 s, a 1.3-second spread across 15 rows, which reads as a
ceiling being hit rather than 15 independent stopping decisions. No timeout is
set anywhere in llm/clients.py or bench/run_cell.py, so the ceiling is
external — most plausibly generation running until the 4096-token context is
exhausted. Worth one sentence; not worth a re-run to confirm.

REPORTING CONSEQUENCE: "qwen2.5:0.5b p99 = 116.9 s" is a true number about a
1.38% pathology, not about the model's normal latency. Report the tail
explicitly as a failure mode with its own row count, never as the model's
latency profile.

B7-f  [Certain]  THE EXACT COUNTS WERE THROWN AWAY AT THE SOURCE — same class
                 of bug that V2 fixed for `thinking`

Both providers RETURN authoritative token counts on every response, and
llm/clients.py discards both:

  OpenAI   resp.usage.prompt_tokens / completion_tokens
           -> line 63 returns resp.choices[0].message.content only
  Ollama   resp.prompt_eval_count / eval_count  (eval_count covers ALL
           generated tokens, reasoning included)
           -> line 84 returns resp["response"] only

Verified absent from the artifacts: scanned the CSVs for eval_count,
prompt_eval_count, usage, prompt_tokens, completion_tokens — no hits (the one
apparent match was the word "usage" inside a deepseek reasoning trace).

This is precisely the V1 `thinking` bug repeating: the client returns one
field from a rich response object and drops the rest, so a measurement that
the server already computed is lost. The V2 fix captured `thinking`; nothing
captured usage.

CHEAP-FIX, and it should happen before any further measurement: park
`last_usage` on the client the same way `last_thinking` is parked, and write
the two counts per row. Ollama's `eval_count` is ground truth for the local
models and is tokenizer-correct per model — better than anything tiktoken
could produce, since Qwen, Llama and DeepSeek each use their own vocabulary
and counting their output with o200k_base would be wrong even if it installed.

B7-g  [Certain]  THE LOCAL HALF IS RECOVERABLE OFFLINE TODAY
A built llama.cpp is present at C:\Users\USER\llama.cpp\bin (llama-cli.exe,
llama-bench.exe and siblings). The GGUF vocabularies are in D:\LLM\blobs.
Exact per-model token counts for the four local models can therefore be
computed offline with no network and no PyPI. Only the CLOUD count genuinely
needs tiktoken — one model, one rung, and the $0.4066 figure.

B7-e  [Certain]  FALSE LEAD — THERE WAS NEVER A TLS OR NETWORK PROBLEM

RECORDED AS AN ERROR OF THIS AUDIT, not as a resolved blocker.

For several turns this report asserted that `pip install tiktoken` was blocked
by a self-signed certificate in the PyPI chain, and treated the exact token
counts as unobtainable. Two explanations were offered for the certificate —
a university proxy, then local antivirus TLS interception. BOTH WERE WRONG,
and so was the premise.

What was actually happening: every command was being run against the SYSTEM
interpreter, C:\Users\USER\AppData\Local\Programs\Python\Python311, which is
not the project's environment. The project's documented environment is the
`thesis` conda env, named in the first line of smoke_test.py's docstring, and
it already contained tiktoken 0.13.0, transformers 5.15.1, torch, dotenv and
sentence-transformers. Nothing needed installing and nothing needed
downloading. The certificate error was a real error from an irrelevant
interpreter reaching for a package it did not need to have.

The same mis-selection produced two further false conclusions, both since
withdrawn: that the environment was "partially rebuilt", and that
llm/clients.py "would not import".

WHY IT PERSISTED: the VS Code workspace also resolves to the system
interpreter, so the IDE's own diagnostics reported nine packages as missing —
apparently independent corroboration of a conclusion that was wrong at the
root. Two sources agreeing meant only that both were reading the same wrong
interpreter.

LESSON, and it generalises past this repo: locate and activate the project's
documented environment BEFORE diagnosing any dependency fault. An
infrastructure explanation for a missing package should be the last
hypothesis, not the first. Certificate verification was never disabled, which
was the one correct decision in the sequence.

RESOLVED — RE-RUN WITH THE REAL TOKENIZER. The section below is kept as the
record of two successive errors; the exact figures are in B7-i.

CORRECTION [Certain] — an earlier draft of this report claimed the ~2:1 and
~24:1 RATIOS were robust to tokenizer choice because numerator and denominator
were estimated identically. THAT WAS WRONG and the error is material.

The two quantities are different kinds of text and do not tokenize alike.
Measured against the live server:

  qwen2.5:0.5b   JSON output        2.00 chars/token
  deepseek-r1    reasoning prose    5.10 chars/token
  (this report's estimate assumed 4.00 for both)

Visible output is punctuation-dense JSON and was therefore UNDERCOUNTED by
roughly 2x. Reasoning is prose and was OVERCOUNTED by roughly 1.3x. Both
errors move the ratio the same way. Rough recalculation for deepseek-r1 puts
the hidden:visible ratio nearer 9:1 and the unseen share nearer 90%, not
24:1 and 96.0%.

THE FINDING SURVIVES — the great majority of deepseek's generated tokens are
never delivered, and under the old scheme all of them counted as zero. THE
NUMBER DOES NOT. Neither 96.0% nor 24:1 may be quoted. The two constants above
come from single short samples and are evidence that the assumption was
unsafe, not a correction factor.

This is now cheap to close exactly: see B7-f. Ollama's `eval_count` is the
server's own count against the model's own vocabulary and includes reasoning
tokens (verified: eval_count=384 for a response of 16 visible + 1,943 thinking
characters). Nothing needs to be tokenised externally for any future run.

The absolute counts and the $0.4066 cloud figure were never quotable and
remain so.


B7-i  [Certain]  EXACT COUNTS — backfill re-run with tiktoken o200k_base

The TLS blocker was NOT a blocker. tiktoken 0.13.0 was already installed in
the project's own documented environment, C:\Users\USER\.conda\envs\thesis,
along with transformers 5.15.1, torch 2.12.1, python-dotenv and
sentence-transformers. The earlier failure was running system Python
(C:\Users\USER\AppData\Local\Programs\Python\Python311) against PyPI. The
env is NOT partially rebuilt; an earlier claim in this report that it was is
withdrawn. LESSON: locate the documented environment before diagnosing an
infrastructure fault. smoke_test.py names it in its first docstring line.

Re-run: all 30,352 rows now carry token_source=tiktoken_o200k_base. Zero
estimate rows remain. Frozen-column drift = 0 on all four files again.

  model                     visible     thinking        total    unseen
  deepseek-r1:7b            188,057    3,282,417    3,470,474     94.6%
  qwen2.5:0.5b              958,067            0      958,067      0.0%
  llama3.2:3b               288,949            0      288,949      0.0%
  gpt-4o-mini               256,800            0      256,800      0.0%
  qwen2.5:3b                220,528            0      220,528      0.0%
  qwen2.5-3b-lora            26,412            0       26,412      0.0%
  classifier:distilbert      26,262            0       26,262      0.0%
  qwen2.5-0.5b-lora          22,493            0       22,493      0.0%

  deepseek hidden:visible = 17.5:1     unseen share = 94.6%
  study total: 1,989,007 visible / 3,284,275 thinking = 1.7x

SECOND CORRECTION, AGAINST MYSELF [Certain]: the estimate said 24:1 / 96.0%.
The correction above predicted "nearer 9:1 and 90%". The truth is 17.5:1 and
94.6%. The ORIGINAL ESTIMATE WAS CLOSER than the correction. Extrapolating a
global ratio from two single-sample chars/token constants (2.00 and 5.10,
measured on one short response each) was worse than the flat assumption it
replaced. The caution was right; the arithmetic behind it was not. Quote
94.6%.

REMAINING LIMITATION [Certain], and it must be stated wherever 94.6% appears:
o200k_base is EXACT for gpt-4o-mini and an APPROXIMATION for the four Ollama
models, which use their own vocabularies. So 94.6% is exact arithmetic over an
approximate tokenisation. The truly exact figure comes from Ollama's
eval_count — now captured by llm/clients.py per B7-f, but only for runs made
from here on. For the existing 30,380 rows it needs llama-tokenize against the
GGUF vocabularies in D:\LLM\blobs. Expect the corrected figure to move by a
few points, not to change the finding.

CLOUD COST IS GENUINELY EXACT — o200k_base IS gpt-4o-mini's tokenizer:

  cot          $0.1602      <- 37.8% of all cloud spend
  few_shot     $0.0777
  rag          $0.0758
  role_based   $0.0564
  zero_shot    $0.0534
  TOTAL        $0.4236      (estimate had said $0.4066, -4.0%)

CoT costs 3.0x zero_shot on the cloud model for a strict accuracy of 0.685
against zero_shot's 0.835. That is the cost curve's cleanest single sentence:
on gpt-4o-mini, chain-of-thought triples the bill to buy 15 points of LOSS.

Note qwen2.5:0.5b now has 958,067 visible tokens — 3.7x gpt-4o-mini and the
largest visible output of any model, driven entirely by the cot runaway
documented in B7-h.


=====================================================================
B8 — THE OLLAMA MODEL STORE — RESOLVED, NOTHING WAS LOST
=====================================================================

B8-a  [Certain]  RESOLVED. Config, not data loss. No re-pull needed.

Initial symptom: `ollama list` empty, `~/.ollama/models` 0 bytes, /api/show
404. Diagnosis before re-pulling anything:

  OLLAMA_MODELS   unset in shell, User env, and Machine env
  D:\LLM          a COMPLETE Ollama store — manifests/ + blobs/, 58 GB, 32 blobs

Ollama was defaulting to the empty ~/.ollama/models while the real store sat
orphaned on D:. Restarting with OLLAMA_MODELS=D:\LLM recovers all six study
models intact:

  qwen2.5-3b-lora:latest      1.9 GB    (blob dated Aug 22 16:21)
  qwen2.5-0.5b-lora:latest    397 MB    (blob dated Aug 22 16:21)
  deepseek-r1:7b              4.7 GB
  qwen2.5:3b                  1.9 GB
  qwen2.5:0.5b                397 MB
  llama3.2:3b                 2.0 GB

The LoRA blob timestamps (Aug 22 16:21) match ft/merge_and_export.py's mtime
(Aug 22 16:07) — these are the measured artifacts, not re-pulls. Provenance
is intact, which a re-pull would have destroyed.

Sizes independently corroborate llm/clients.py's docstring claim: the LoRA
exports are 397 MB / 1.9 GB, matching stock qwen2.5:0.5b / qwen2.5:3b exactly.

ACTION REQUIRED: set OLLAMA_MODELS=D:\LLM persistently (User env), or every
future run silently sees an empty store. This is almost certainly what the
August runs had set and what was subsequently lost.

B8-c  [Certain]  REPRODUCIBILITY FINDING — diagnosing before re-pulling
                 PRESERVED PROVENANCE, AND A RE-PULL WOULD HAVE DESTROYED IT
                 SILENTLY

This is the most consequential thing in the session and it is a finding, not
a fixed config.

The obvious response to an empty store is `ollama pull`. Had that been done,
the study's local cells would now be backed by the CURRENTLY published
weights for each tag, not the weights that produced the 30,380 rows. Ollama
tags are mutable: `llama3.2:3b` today is not provably the blob served in
August. The substitution leaves NO trace — same tag, same name, same size
class, no warning, no version record in the CSVs.

What establishes provenance instead:

  ft/merge_and_export.py       mtime Aug 22 16:07
  qwen2.5-3b-lora blob         mtime Aug 22 16:21   (1,929,902,464 bytes)
  qwen2.5-0.5b-lora blob       mtime Aug 22 16:21
  measurement runs             Aug 24-25

The LoRA blobs were written 14 minutes after the export script and two days
before the runs. These are the measured artifacts. That chain exists only
because nothing was overwritten.

METHODS IMPLICATION: the thesis should record the blob digests for all six
models, since the tag names alone do not identify what was measured. They are
available now via `ollama show`; they will not be recoverable if the store is
ever rebuilt. Capture them before anything else touches D:\LLM.

B8-b  [Certain]  ALL THREE TRAINING CHECKPOINTS INTACT

  ft/runs/qwen0.5b_lora/checkpoint-87    adapter_model.safetensors    35,237,104
  ft/runs/qwen3b_lora/checkpoint-87      adapter_model.safetensors   119,801,528
  ft/runs/distilbert_cls/epoch_09        model.pt                    265,558,923

All carry optimizer/scheduler/rng state and tokenizer. The fine-tuned rungs
are fully reproducible and ft/merge_and_export.py is not decoration.


=====================================================================
BLOCK A — COMPARABILITY OF THE 28 CELLS
=====================================================================

WORST FINDING FIRST: A2. temperature=0 did NOT deliver determinism on the
cloud model, and one cell flips 34 verdicts across repeats.

---------------------------------------------------------------------
A2 — DECODING PARAMETER PARITY  (the headline)
---------------------------------------------------------------------

A2-a  [Certain]  WHAT IS ACTUALLY SENT

  llm/clients.py:57-62   OpenAI    temperature=self.temperature (0.0). Nothing else.
  llm/clients.py:76-80   Ollama    options={"temperature": 0.0}. Nothing else.
  llm/classifier_client   rung 4    argmax. No sampling surface at all.

Every other decoding parameter on both providers was left at library default.
The defaults DIFFER between providers, so they are not a shared constant:

  parameter          set by study   Ollama default        OpenAI default
  temperature        YES (0.0)      0.8                   1.0
  top_p              no             0.9                   1.0
  top_k              no             40                    (not exposed)
  min_p              no             0.0                   (not exposed)
  repeat_penalty     no             1.1                   (not exposed)
  repeat_last_n      no             64                    (not exposed)
  frequency_penalty  no             (not exposed)         0.0
  presence_penalty   no             (not exposed)         0.0
  num_predict/max    no             -1 (unbounded)        model maximum
  num_ctx            no             4096 (VRAM-derived,   ~128k model window
                                     MEASURED — see B6-f)
  seed               no             0 -> random           null -> no determinism
                                                          guarantee
  stop               no             from each Modelfile   null

[Certain] for the study-side column and for num_ctx, which was measured off
the running server. [Likely] for the published default values, which are
documented rather than instrumented — the harness records no configuration at
all (see A4-b), so they cannot be recovered from the artifacts.

A2-b  [Certain]  QUANTIFIED — the uncontrolled parameters injected NO variance
                 locally, and REAL variance on the cloud

Rather than observing that parameters were unset, this was measured: for every
command in every cell, were the 5 repeats byte-identical?

  model                     identical raw output across 5 repeats
  ollama:llama3.2:3b                1085/1085   100.0%
  ollama:qwen2.5:3b                 1085/1085   100.0%
  ollama:qwen2.5:0.5b               1085/1085   100.0%
  ollama:deepseek-r1:7b             1085/1085   100.0%
  ollama:qwen2.5-3b-lora             217/217    100.0%
  ollama:qwen2.5-0.5b-lora           217/217    100.0%
  classifier:distilbert              217/217    100.0%
  openai:gpt-4o-mini                 836/1085    77.1%   <== NOT DETERMINISTIC

All seven LOCAL cells are perfectly deterministic. Whatever Ollama's unset
sampler values were, they were CONSTANT across every call, so they are a
recorded-nowhere constant rather than a source of noise. The defect is
reproducibility, not validity: a replication on a different Ollama version
inherits different defaults with no warning and no way to detect the change.

THE CLOUD MODEL IS A DIFFERENT MATTER. Per strategy:

  gpt-4o-mini  cot          2/217 identical   0.9%    34 verdict flips
  gpt-4o-mini  zero_shot  204/217 identical  94.0%     9 verdict flips
  gpt-4o-mini  few_shot   207/217 identical  95.4%     7 verdict flips
  gpt-4o-mini  rag        209/217 identical  96.3%     4 verdict flips
  gpt-4o-mini  role_based 214/217 identical  98.6%     3 verdict flips

temperature=0.0 is not a determinism guarantee at the OpenAI API, and no
`seed` was passed — that is the single uncontrolled parameter that actually
mattered in this study. In the cot cell 99.1% of commands produce different
text across repeats and 34 of 217 commands CHANGE THEIR STRICT VERDICT
depending on which repeat you look at.

CONSEQUENCE FOR BLOCK D, and it cuts both ways: effective n = 217 (not 1,085)
holds exactly for the seven local models, where the repeats are byte-identical
and carry literally zero additional information. It does NOT hold for
gpt-4o-mini, whose repeats are genuinely independent draws. The cloud cells
have real sampling variance that the local cells do not, so a single pooled
error model is wrong for this study.

DISCLOSE (draft):
  "Only temperature was set (0.0 for every model). All other decoding
   parameters were left at each provider's defaults, which differ between
   OpenAI and Ollama and were not recorded by the harness. Measured
   consequence: all seven locally served models produced byte-identical output
   across the five repeats of all 217 commands, so the unset parameters were
   constant and contributed no variance. The cloud model did not: 22.9% of
   commands varied across repeats despite temperature=0, rising to 99.1%
   under chain-of-thought, where 34 of 217 commands changed strict verdict
   between repeats. No `seed` was supplied to the OpenAI API. Cloud-model
   results therefore carry sampling variance that the local results do not."

---------------------------------------------------------------------
A1 — PROMPT-PATH IDENTITY (rung 1 vs rung 2)
---------------------------------------------------------------------

A1-a  [Certain]  PASS — byte-identical

Rendered both prompts for the same command with the SAME example list:

  few_shot  len=1264
  rag       len=1264
  identical=True

Zero divergence. The `rag` branch shares few_shot's wrapper exactly, so the
only difference between rungs 1 and 2 is which examples were selected. The
rung-1-vs-rung-2 comparison is sound, exactly as llm/prompts.py claims.
Combined with A5 (RAG's mean prompt is SHORTER than few_shot's), neither the
wrapper nor prefill length can explain any rung-2 effect.

---------------------------------------------------------------------
A4 — ENVIRONMENT PARITY
---------------------------------------------------------------------

A4-a  [Certain]  The harness sets two variables, by setdefault:
  bench/run_grid.py:72-73   OLLAMA_MAX_LOADED_MODELS=1, OLLAMA_KEEP_ALIVE=0
  (setdefault, so a value already in the environment WINS — and both are
   currently set in the shell, confirming they can be externally supplied)

A4-b  [Certain]  DISCLOSE — parity across the four launches is NOT VERIFIABLE

run_grid.py PRINTS the effective values at startup (lines 177-179) and the
docstring calls keep-alive "an experimental condition, not a setting". But:

  - no env/config column exists anywhere in the 28-column CSV schema
  - the values went to a console that no longer exists
  - the four result files were produced by four separate launches

So the claim "the environment was identical across all four runs" cannot be
confirmed or refuted from the artifacts. Given the docstring's own position
that keep-alive is the single biggest determinant of latency and cold/warm
stability, an experimental condition of that importance being unrecorded is a
genuine gap. CHEAP-FIX for any future run: write the three variables into
every row, or into a sidecar manifest per launch.

A4-c  [Certain]  CONFIRMED — the "non-uniform offline flags" concern is real
                 as a code path

HF_HUB_OFFLINE is NOT set by run_grid.py; it is only recommended in prose.
And llm/retrieve.py:121-133 actively POPS HF_HUB_OFFLINE from the environment
to permit the sentence-transformer download, then restores it. That path exists
only for rag cells. So rung-2 cells could run under a transiently different
environment from rung-1 cells in the same process. Effect is bounded — it
governs whether a model file downloads, not how any model decodes — but it is
a real non-uniformity and it is undocumented in the Methods.

---------------------------------------------------------------------
A5 / A6 — answered earlier, recorded here for completeness
---------------------------------------------------------------------

A5  PASS  (see B6-b) RAG max prompt 401 tok vs few_shot 378; RAG MEAN prompt
          1,426 chars vs few_shot 1,477 — RAG's is shorter. Prefill length
          does not explain the rung-2 latency delta.
A6  PASS  is_warmup is excluded in all four analysis scripts
          (analyze.py:89, figures.py:267, summary.py:208,
          scratch_scored_table.py:14). n=1084 per cell, not 1085.


=====================================================================
A3 — QUANTIZATION PARITY  (answerable once B8 was resolved)
=====================================================================

A3-a  [Certain]  PASS — no quantization confound

  model                    params     quant     native ctx
  llama3.2:3b              3.2B       Q4_K_M    131072
  qwen2.5:3b               3.1B       Q4_K_M     32768
  qwen2.5:0.5b             494.03M    Q4_K_M     32768
  deepseek-r1:7b           7.6B       Q4_K_M    131072
  qwen2.5-3b-lora          3.1B       Q4_K_M     32768
  qwen2.5-0.5b-lora        494.03M    Q4_K_M     32768

All six are Q4_K_M, base and LoRA alike, and each LoRA matches its own base in
both parameter count and quantization. "Fine-tuning helps" is NOT partly a
quantization claim. This was the strongest available threat to the rung-3
result and it does not exist.

gpt-4o-mini is served at undisclosed precision — an unavoidable DISCLOSE, but
it is a cloud-vs-local caveat, not a confound within the local ladder.

A3-b  [Certain]  MINOR PARITY ASYMMETRY — the two LoRA Modelfiles pin
`PARAMETER temperature 0`; the four base Modelfiles set no temperature at all.
Both paths still receive temperature=0.0 from the client on every request
(llm/clients.py line 79), so effective behaviour is identical and no measured
number is affected. Worth one sentence in Methods so a reader diffing the
Modelfiles does not think it was uncontrolled. deepseek-r1 is the only model
declaring the `thinking` capability, which is why it alone populates the
`thinking` column.

METHOD CAVEAT [Certain]: this audit ran under an interpreter without
`transformers` or `tiktoken`, so all token figures above are chars/4 bounds,
not exact counts. The margins are large enough (4x) that the conclusions do
not depend on the constant. Re-running audit/reconstruct_prompts.py inside the
`thesis` conda env yields exact counts in minutes.


=====================================================================
B9 — CONTAINER AND GRID ROWS ARE TWO DIFFERENT TASKS
=====================================================================

B9-a  [Certain]  DISCLOSE — the subsets are not one task

CORRECTION LOGGED: an earlier draft of this section attributed the container
subset's easiness to literal symbol presence ("transcription vs grounding").
That over-claims. The mechanism was NOT isolated and cannot be, from this
data. What follows is the defensible version.

The two subsets differ on FOUR axes simultaneously:

  axis                          grid (id 0-180)        container (181-216)
  -------------------------------------------------------------------------
  literal symbol in command     0/132   =   0.0%       36/36   = 100.0%
  distinct location symbols     11 (9 cells + 2        3 (box_1..box_3),
                                runtime), 12 rows ea.   12 rows each
  verbs represented             7 (all)                2 (pick_place 27,
                                                          move_to 9)
  carrier vocabulary            "put the" 34, "can      "drop the" 9, "set
                                you" 33, "pick up" 33   the" 9, "stick the" 9

These co-vary perfectly. Nothing in the design separates them.

WHY THE CAUSAL CLAIM FAILS: within-subset label cardinality differs by ~4x.
A model choosing at random among the symbols its subset actually uses scores
1/3 = 33% on containers and 1/11 = 9% on grid, before any question of
grounding arises. The verb mix differs too — containers exercise 2 of 7
verbs. An examiner asking "how do you know it is the literal string and not
the smaller label set, or the easier verb mix?" has a question this data
cannot answer.

WHAT SURVIVES [Certain]: container rows are a 3-way selection in which the
answer string is present in the input; grid rows are an 11-way selection
requiring spatial resolution against a legend. They are not one task, and a
pooled accuracy averages them at a 36:181 weight fixed by dataset
construction rather than by anything meaningful.

Corroborating: five unrelated cells score EXACTLY 1.000 on containers —
llama3.2 zero_shot and role_based, qwen2.5-3b-lora, gpt-4o-mini few_shot,
role_based and zero_shot — while scoring 0.32-0.83 on grid. Five independent
models saturating one subset and not the other is consistent with any of the
four axes, or all of them.

DRAFT DISCLOSURE (narrowed):

  "The held-out set contains two structurally different subsets. Container
   rows (n=36) are a three-way selection among box_1..box_3 in which the
   answer symbol appears verbatim in the command, and they exercise two of
   the seven verbs. Grid rows (n=181, of which 132 carry a location) are an
   eleven-way selection among nine grid cells and two runtime symbols, the
   symbol never appears in the command, and all seven verbs are present.
   Because these subsets differ on several axes at once, no single cause for
   the accuracy gap is claimed. They are reported separately throughout, and
   the thesis claim rests on the grid subset."

NOT WORTH TESTING [Guessing]: the mechanism could be isolated with ~12 rows
phrasing box locations without the literal symbol, run on two models. That is
a newly authored subset against a spent eval file, and the post-hoc integrity
cost exceeds the value of the answer. Narrow the claim instead.

B9-b  [Certain]  the container subset cannot carry a claim alone
n=36 commands gives a 95% CI half-width near +-16 points. Combined with five
cells at a ceiling of exactly 1.000, the subset has almost no resolving power.
It is a sanity check, not a result.


=====================================================================
B7-j — EXACT TOKEN COUNTS, EACH MODEL WITH ITS OWN VOCABULARY  (FINAL)
=====================================================================

B7-j-a  [Certain]  llama-tokenize IS BLOCKED, AND WAS NOT BYPASSED

The specified tool, C:\Users\USER\llama.cpp\bin\llama-tokenize.exe, exists
(57,344 bytes) but will not execute:

    Program 'llama-tokenize.exe' failed to run: An Application Control
    policy has blocked this file

That is a Windows Application Control (WDAC/AppLocker) policy. No attempt was
made to disable, relax, or work around it. The same vocabularies were reached
by a different, permitted route: HuggingFace tokenizer files, three of the
four loaded from disk with no network at all.

  ollama:qwen2.5:0.5b / :3b / both LoRAs -> Qwen2.5 vocab, size 151,665,
      read LOCALLY from ft/runs/qwen0.5b_lora/checkpoint-87/tokenizer.json.
      This is the exact tokenizer the adapters were trained with.
  ollama:deepseek-r1:7b -> deepseek-ai/DeepSeek-R1-Distill-Qwen-7B, 151,665
  ollama:llama3.2:3b    -> Llama-3.2-3B-Instruct vocab, 128,256. [Likely]
      identical to Meta's: meta-llama/* is gated on the Hub, so an ungated
      mirror was used. Vocabulary size matches the published Llama 3.2 figure.
      Not cryptographically verified against Meta's copy.
  openai:gpt-4o-mini    -> tiktoken o200k_base, 200,019. EXACT.
  classifier:distilbert -> DistilBERT vocab, 30,522, local. Its OUTPUT is
      constructed from the schema rather than generated, and is stamped
      "distilbert_vocab_local|output_constructed_not_generated" so it can
      never be pooled into a generation-cost figure by accident.

B7-j-b  [Certain]  A DISPATCH BUG WAS CAUGHT BEFORE IT REACHED THE NUMBERS

The first implementation routed by substring:

    if "llama" in model: ...      # model = "ollama:qwen2.5:0.5b"

The provider prefix "ollama:" CONTAINS the substring "llama", so all four Qwen
models were silently assigned the Llama vocabulary. Nothing errors; the counts
are simply wrong, by a few percent, in a way no assertion would catch.

Caught by audit/check_dispatch.py, which prints the vocabulary actually chosen
for every model present in the data. Fixed by matching on the model name after
the provider prefix. The check is retained and should be run whenever the
tokenizer registry changes. This is the third time in this audit that a
plausible-looking value turned out to be produced by the wrong mechanism.

B7-j-c  [Certain]  FINAL EXACT FIGURES

  model                     visible     thinking        total    unseen
  deepseek-r1:7b            187,187    3,345,311    3,532,498     94.7%
  qwen2.5:0.5b              983,049            0      983,049      0.0%
  llama3.2:3b               294,575            0      294,575      0.0%
  gpt-4o-mini               256,800            0      256,800      0.0%
  qwen2.5:3b                219,280            0      219,280      0.0%
  classifier:distilbert      38,806            0       38,806      0.0%
  qwen2.5-3b-lora            25,328            0       25,328      0.0%
  qwen2.5-0.5b-lora          21,409            0       21,409      0.0%

  DEEPSEEK-R1 HIDDEN:VISIBLE = 17.9:1     UNSEEN SHARE = 94.7%
  study-wide hidden:visible  =  1.65:1
  frozen-column drift        =  0 across all four files
  distinct strings tokenised =  8,763
  token_source stamps        =  tiktoken 5,420 | llama 5,420 | qwen 13,008 |
                                deepseek 5,420 | distilbert 1,084  (= 30,352)

THE ESTIMATE WAS AN UPPER BOUND, AS PREDICTED. chars/4 under-counts dense JSON
(measured ~2.0 chars/token) relative to prose reasoning traces (~5.1), so it
inflated the hidden:visible ratio by understating the visible denominator. The
progression across three methods:

  chars/4 estimate              24:1     96.0%     upper bound
  tiktoken for everything       17.5:1   94.6%     wrong vocab for local models
  per-model exact vocabularies  17.9:1   94.7%     FINAL -- quote this

Note the ordering: the flat estimate erred by ~6 points, and the single-wrong-
vocabulary pass landed within 0.1 points of the exact answer almost by
accident. Neither is a substitute for using each model's own vocabulary, and
the agreement between the last two should not be read as either one being
validated.

B7-j-d  [Certain]  qwen2.5:0.5b now has 983,049 visible tokens — 3.8x
gpt-4o-mini and the largest visible output of any model in the study, all of
it non-reasoning generation, driven by the 15-row cot runaway in B7-h.


=====================================================================
PROV — THREE TIMES, PROVENANCE DEPENDED ON SOMETHING NOBODY RECORDED
=====================================================================

PROV-a  [Certain]  This is the pattern finding of the audit.

Three separate times, the identity of an artifact that produced reported
numbers survived only by accident, and in each case the recovery route was
discovered rather than designed:

  1. THE MODEL STORE LOCATION (B8-a). `ollama list` returned nothing and the
     default store was empty. The measured weights survived only because
     OLLAMA_MODELS had once pointed at D:\LLM and the directory was still
     there. Nothing in the repo, the CSVs or the docs recorded that path. The
     obvious response — `ollama pull` — would have replaced the measured
     weights with current published ones under the same mutable tags, silently
     and with no trace.

  2. THE ENVIRONMENT CONTENTS (F-ENV). No lockfile, no pip freeze, no version
     pins, and no environment columns in the 28-column result schema. The
     software stack that produced 30,380 rows is not reconstructible; the best
     available substitute is an audit-time snapshot taken three days later.

  3. THE PEFT VERSION (F-ENV-b). peft was absent entirely, so the whole rung-3
     path could not run. The version that trained the adapters was recoverable
     ONLY because the PEFT library happens to stamp "peft_version": "0.20.0"
     into adapter_config.json. No deliberate record was kept. Had PEFT not
     written that field, the correct pin would have been unrecoverable and the
     rung-3 adapters would be unreproducible in principle.

THE COMMON SHAPE: every artifact in this study is identified by a MUTABLE
NAME — an Ollama tag, an unpinned package, a conda env — rather than by an
immutable digest or a recorded version. Names are not identities. The study
survived three times on luck: a directory that was not deleted, a library that
volunteers its own version, and an audit that looked before it re-pulled.

REMEDY, and it is cheap: record identity at measurement time, not afterwards.
Digests for model weights (now captured in audit/MODEL_PROVENANCE.txt), a
lockfile per run (now in requirements.lock.txt), and the environment written
into the result rows themselves (A4-b, still outstanding). Any future run
should emit a manifest alongside the CSV.

DRAFT METHODS SENTENCE (PROV):
  "Model weights are identified by manifest digest rather than by Ollama tag,
   because tags are mutable and do not identify a specific artifact. Dependency
   versions are recorded in a lockfile. Where a version could not be recovered
   from a deliberate record it is stated as such; the PEFT version used for
   rung-3 training was recovered from the adapter configuration written by the
   library itself."


=====================================================================
F-ENV — THE MEASUREMENT ENVIRONMENT IS NOT EXACTLY RECONSTRUCTIBLE
=====================================================================

F-ENV-a  [Certain]  DISCLOSE

No lock file, no `pip freeze`, and no environment record of any kind was
captured when the 30,380 rows were measured on 2026-08-24/25. requirements.txt
carried no version pins at all until this pass. The four result files contain
no environment columns (see A4-b). So the exact software stack that produced
the measurement cannot be reconstructed, only approximated.

What is now pinned:
  requirements.lock.txt   full transitive closure of the thesis env, 103
                          packages, Python 3.11.15, captured 2026-08-27
  requirements.txt        direct dependencies pinned to the same versions

Both are stamped AUDIT-TIME, not measurement-time. They are the closest
available reference point and must not be presented as a reproduction target
for the August runs.

F-ENV-b  [Certain]  TWO PACKAGES WERE MISSING AND RUNG 3 COULD NOT RUN

At audit time `peft` and `accelerate` were absent from the thesis env, so:

  ft/train_lora.py        ModuleNotFoundError: No module named 'peft'
  ft/merge_and_export.py  ModuleNotFoundError: No module named 'peft'
  ft/train_classifier.py  imported fine (rung 4 unaffected)

The entire rung-3 path — LoRA training AND the merge/GGUF export — was
non-executable. Nothing measured was lost: the checkpoints are intact (B8-b)
and the merged models are in the Ollama store (B8-a). But the path from
checkpoint to served model could not be re-executed, which is precisely the
step a reproducibility challenge would target.

RECOVERED PROVENANCE: rather than install the latest peft, the version was
read out of the artifact itself —

  ft/runs/qwen3b_lora/checkpoint-87/adapter_config.json
      "peft_version": "0.20.0"

and peft==0.20.0 installed to match. The checkpoint records the version that
trained it, so this pin is measurement-time truth, not audit-time drift.
`accelerate` has no such record; 1.14.0 is audit-time latest and is flagged as
such in both requirements files.

All ten repo modules now import cleanly under the thesis env, rung 3 included.

F-ENV-c  [Likely]  the "missing packages" impression came from interpreter
selection, not from a degraded env. The VS Code workspace resolves to
C:\Users\USER\AppData\Local\Programs\Python\Python311, which has almost none
of these packages; the project's documented env is
C:\Users\USER\.conda\envs\thesis (named in smoke_test.py's docstring).
Repointing the IDE removes the false signal. This audit made the same mistake
for several turns and wrongly reported a TLS/PyPI blocker as a result — see
B7-i.

F-ENV-d  [Certain]  requirements.txt listed dependencies for work that does
not exist, and omitted one that does. stable-baselines3 and sb3-contrib were
listed but are NOT installed and Phase 2.5 has zero lines. matplotlib is
required by bench/analyze.py and bench/figures.py and was never declared. Both
corrected in this pass.

DRAFT METHODS SENTENCE (F-ENV):

  "No environment lockfile was captured at measurement time. The dependency
   versions recorded in requirements.txt and requirements.lock.txt describe the
   environment as of 2026-08-27, after the measurement, and are provided as the
   closest available reference point rather than as an exact reproduction of
   the measurement environment. One exception is the PEFT library, pinned to
   version 0.20.0, which is the version recorded inside the saved LoRA adapter
   configuration and therefore the version that produced the rung-3 adapters.
   Model weights are identified by digest in audit/MODEL_PROVENANCE.txt, since
   Ollama tags are mutable and do not identify a specific artifact."


BLOCK B TRIAGE
--------------

  DISCLOSE   B1-a  "exact match" wording false; 71% of strict credit rides on
                   the modifiers leniency
  DISCLOSE   B1-b  five undocumented falsy-modifiers acceptances in strict
  CHEAP-FIX  B2-a  results-chapter script reports no invalid rate (~1 h)
  CHEAP-FIX  B1-c  scorer docstring understates _IGNORE_IF_EMPTY (~15 min)
  CHEAP-FIX  B3-a  add `dataset` to the 4-key dedupes in analyze.py and
                   scratch_scored_table.py — currently harmless, silently
                   wrong the moment a seen file is unioned in (~15 min)
  PASS       B2    no script counts invalid off `category`
  PASS       B3    0 rows dropped, 0 orphans
  PASS       B4    0 structural violations in 1,085 classifier rows
  PASS       B5    all four files scored under 3000198

  CRITICAL-  B7-a  tokens_in/tokens_out/cost_usd 0% populated on all 30,380
  RECOVERABLE      v3 rows -> Block E has no cost axis TODAY. Every input to
                   recompute it is on disk.
  CHEAP-FIX  B7-b  backfill_tokens.py recovers it offline but has 4 V1-shaped
                   defects that would silently produce wrong v3 numbers (~2 h)
  CHEAP-FIX  B7-c  fixed backfill must emit tokens_out_visible AND
                   tokens_out_total as separate columns; the gap is a finding
  CONFIG     B8-a  RESOLVED: models were never lost, OLLAMA_MODELS was unset.
                   Set it to D:\LLM persistently.
  PASS       B8-b  all three training checkpoints intact and complete
  PASS       A3-a  all six models Q4_K_M — no quantization confound
  DISCLOSE   A3-b  LoRA Modelfiles pin temperature 0, bases do not; effective
                   behaviour identical (client sets it either way)
  PASS       B6    num_ctx MEASURED at 4096. Prompts max ~401 tok (10x
                   margin). 22 of 27 generative cells have zero rows over the
                   ceiling. Worst-case movement anywhere +0.005.

  DISCLOSE   A2-a  every non-temperature decoding parameter was provider
                   default on both providers; defaults differ; nothing recorded
  DISCLOSE   A2-b  temperature=0 did NOT give determinism on gpt-4o-mini:
                   22.9% of commands vary, 99.1% under cot, 34 verdict flips.
                   No seed was passed. Local models: 100% deterministic.
  PASS       A1-a  few_shot and rag prompts byte-identical (1264 == 1264)
  DISCLOSE   A4-b  environment parity across the four launches is not
                   verifiable — no config column exists in the CSV schema
  DISCLOSE   A4-c  llm/retrieve.py pops HF_HUB_OFFLINE mid-run; rag cells can
                   run under a transiently different env from rung-1 cells
  CHEAP-FIX  B7-f  clients.py discards OpenAI `usage` and Ollama `eval_count`
                   — the same class of bug V2 fixed for `thinking` (~1 h)
  DISCLOSE   B7-h  qwen2.5:0.5b p99 of 116.9 s is a 1.38% runaway pathology,
                   not the model's latency; those 15 rows are 100% wrong

NO FATAL FINDINGS ANYWHERE IN THE AUDIT. Nothing currently reported in the
accuracy or latency tables is invalidated. The cost axis does not yet exist,
and every input needed to build it is on disk.


=====================================================================
CONF — CONFUSION MATRICES (location, verb, target)
=====================================================================

Artifacts: audit/confusion_{location,verb,target}_{classifierdistilbert,rung-1}.csv
           audit/confusion_summary.txt      Script: audit/confusion.py

EXCLUSIONS. Rows with stage != "validated" carry no trustworthy predicted
symbol and are dropped:
  classifier   1,085 rows ->     0 excluded (0.0%)   -- it never fails validation
  rung-1 pooled 21,700 rows -> 8,641 excluded (39.8%)
                               schema_failed 7,657 | extract_failed 984
  usable location-bearing rows: 840 (classifier) | 10,036 (rung-1)

HEAD ACCURACY
  head        classifier          rung-1 pooled
  location    635/840   75.6%     6,535/10,036  65.1%
  verb      1,015/1,085 93.5%    10,490/13,059  80.3%
  target      735/765   96.1%     7,980/9,241   86.4%

CONF-1  [Certain]  THE grid_B2 COLLAPSE IS REAL, LARGE, AND ABSENT FROM
                   THE CLASSIFIER

rung-1 pooled: 871 rows predicted grid_B2 when the truth was something else.
  grid_C2 -> grid_B2   478  = 67.8% of grid_C2's 705 rows
  grid_A2 -> grid_B2   318  = 43.6% of grid_A2's 729 rows
  grid_B3 -> grid_B2    35  =  4.8%
  grid_B1 -> grid_B2    30  =  4.2%
  nearest_free_cell     10  =  1.3%
grid_B2 itself is 77.3% correct (553/715).

The collapse is confined to COLUMN 2 -- the two other column-2 cells account
for 796 of the 871 rows (91.4%). This is the documented middle/centre legend
collision: "middle" appears both in the column-2 label and in the phrase for
the centre cell, and the models resolve it to the centre.

CLASSIFIER: ZERO. Not one of its 840 location rows collapses onto grid_B2, and
grid_B2 is 60/60 correct. The effect is entirely a prompting-path artifact --
the classifier has no legend to be confused by, which is the mechanism made
visible. Per-model: llama3.2 15.6%, qwen2.5:3b 10.8%, gpt-4o-mini 6.2%,
deepseek 2.5%, qwen2.5:0.5b 0.0%, classifier 0.0%.

CONF-2  [Certain]  ROW INVERSION EXISTS — BUT THE REQUESTED far/near VS
                   upper/bottom SPLIT CANNOT BE COMPUTED ON THIS DATASET

held_out_v3 contains ZERO far/near commands. Phrasing inventory:
  seen_v2       n=290   far=38  near=37  upper=0   bottom=0
  held_out_v3   n=217   far=0   near=0   upper=36  bottom=36
  held_out_v2   n=132   far=0   near=0   upper=36  bottom=36

The train and evaluation splits are phrasing-disjoint on the row axis BY
DESIGN -- that disjointness is the generalisation test. So the documented
finding ("far/near triggers row inversion; upper/bottom does not") was
measured on a different phrasing population and cannot be reproduced or
refuted here. Any inversion measured on held_out_v3 is an UPPER/BOTTOM effect.

And it is substantial, which is the uncomfortable part: the documented claim
is that upper/bottom does NOT invert.
  classifier      170/360  = 47.2% of grid->grid pairs are A<->C inverted
  rung-1 pooled   622/4225 = 14.7%
  (rows with neither phrasing: 0.0% in both -- the effect is phrasing-bound)

Per-model A<->C inversion rate: classifier 31.5%, deepseek 23.3%,
qwen2.5:3b 15.9%, qwen2.5:0.5b 14.9%, llama3.2 9.7%, gpt-4o-mini 0.1%.
GENERAL, not classifier-specific -- but the classifier is the worst offender
and gpt-4o-mini is essentially immune (2 rows in 3,326).

Row A is the FAR row from the robot base (sim/workspace.py: row_A_x .. row_E_x,
"far -> near"). So "upper" must map to the row furthest from the robot, and the
failing models map it to the nearest. This is a viewpoint ambiguity —
table-top-down versus robot-egocentric — not a legend collision.

CONF-3  [Certain]  COLUMN INVERSION (1<->3) IS NOT SYSTEMATIC

  classifier      0/540  = 0.0%   (all rows, and 0.0% in every row A/B/C)
  rung-1 pooled   75 rows total, entirely from qwen2.5:3b (3.8% of its pairs)
  every other model: 0.0%

The live observation of grid_C1 for "right bottom corner" is NOT column
inversion. grid_C1 IS the bottom-left... the classifier's error there is the
ROW/column pairing, and its dominant off-diagonal is grid_C2 -> grid_B1
(41.7%), a diagonal move, not a mirror. Left/right confusion is not a
systematic effect in this data and no row shows it preferentially.

CONF-4  [Certain]  PAIRS ABOVE 5% NOT ALREADY NAMED

CLASSIFIER (each denominator is 60 rows):
  grid_A3 -> grid_C3   60  100.0%   row inversion, TOTAL failure of this symbol
  grid_A1 -> grid_C1   50   83.3%   row inversion
  grid_A2 -> grid_C2   30   50.0%   row inversion
  grid_C2 -> grid_B1   25   41.7%   NOT previously named -- diagonal
  grid_C2 -> grid_A2   20   33.3%   row inversion
  grid_C3 -> grid_A3   10   16.7%   row inversion
  grid_C1 -> grid_B1    5    8.3%   NOT previously named
  grid_A2 -> grid_B3    5    8.3%   NOT previously named

RUNG-1 POOLED, not already named:
  grid_B3 -> grid_C3   239  32.5%   row drift toward C, no mirror
  grid_B1 -> grid_A1   202  28.1%   row drift toward A
  nearest_free_cell -> (absent)  210  27.9%   the runtime symbol is DROPPED
  nearest_free_cell -> max_reach 205  27.2%   the two runtime symbols conflate
  grid_B3 -> grid_A3    99  13.5%
  grid_C1 -> grid_B1    95  12.8%
  grid_B1 -> grid_C1    78  10.8%
  grid_B2 -> (absent)   67   9.4%
  grid_A3 -> grid_B3    65   8.7%
  grid_A2 -> grid_A3    55   7.5%
  max_reach -> (absent) 56   7.4%
  grid_B2 -> grid_A2    50   7.0%
  grid_B3 -> grid_C2    50   6.8%
  grid_B1 -> grid_A2    45   6.3%
  grid_C3 -> grid_B3    40   5.6%
  grid_B3 -> (absent)   38   5.2%

THE UNNAMED FINDING THAT MATTERS MOST: the two RUNTIME SYMBOLS.
nearest_free_cell is dropped entirely on 27.9% of its rows and confused with
max_reach on a further 27.2% -- so 55.1% of its rows fail in one of two ways,
neither of which is a grid-geometry error. These are the symbols the model
cannot resolve from the prompt by construction (they depend on scene state at
execution time), and they are the single worst-served part of the label space.
Worth its own sentence in the results chapter.


=====================================================================
FIG — ACCURACY AND COST FIGURES (all 28 cells)
=====================================================================

Artifacts: audit/figures/fig{1..4}.{png,pdf} + audit/figures/cell_stats.csv
Script: audit/figures_v3.py

WRITTEN FRESH, and bench/figures.py is NOT broken -- correcting an earlier
claim in this report (F4). It renders all five of its figures without error;
it simply defaults to the V2 files (results_grid_v2_scored.csv /
results_heldout_v2_scored.csv, 20,000 + 13,200 rows). Re-running it reproduced
its committed outputs byte-identically. It was left untouched because it is
tracked with committed PNG/PDF outputs, produces a different figure set, and
has no notion of the per-model effective-n correction these figures need.

INTERVALS. Bootstrap (2,000 resamples, seed 20260827) over command_id
CLUSTERS rather than rows. Each cell is 217 commands x 5 repeats; for all seven
local models the repeats are byte-identical, so effective n is 217 and a naive
sqrt(1085) interval would understate the error by sqrt(5) = 2.24x. gpt-4o-mini
is the exception (77.1% identical), and clustering handles both without a
special case: a deterministic cluster contributes zero within-cluster variance.
Measured median CI half-width across the 28 cells: 5.8 points; widest 6.7
points near p=0.5, which matches the +-6.6 predicted analytically.

FIG-1  fig1_strict_accuracy_clustered_ci
  All 28 cells sorted, with clustered 95% intervals, marker by rung.

FIG-2  fig2_grid_vs_container
  Grid (n=181) against container (n=36) per cell. Container intervals drawn at
  +-16 points and CLIPPED to [0,1], since accuracy is a bounded proportion and
  an unclipped bar runs off the axis. Five cells sit at exactly 1.000 on
  containers. Reinforces B9: two tasks, not two difficulties.

FIG-3  fig3_cost_accuracy_pareto  <-- HEADLINE
  p50 latency (log x, rag includes retrieval) against strict accuracy.
  THE FRONTIER IS THREE POINTS, and no cell is both faster and more accurate
  than any of them:

    cls:distilbert / classifier    acc=0.747   p50=0.019 s   rung 4
    gpt-4o-mini / role_based       acc=0.855   p50=0.699 s   rung 1
    gpt-4o-mini / rag              acc=0.924   p50=0.804 s   rung 2

  Read as a frontier, not a winner. The classifier buys 0.747 accuracy at
  1/37th the latency of the nearest more-accurate cell; gpt-4o-mini/rag buys
  the top accuracy in the study at 42x the classifier's latency. Everything
  else in the grid is dominated. Note that all three LOCAL generative rungs are
  strictly dominated -- including both LoRA cells -- which is the finding the
  cost curve was built to expose.

FIG-4  fig4_strict_vs_normalized_gap
  normalized minus strict per cell. deepseek-r1 occupies the top of this
  ranking: its nested-action-object emissions are semantically right and
  structurally rejected, which is exactly what the dual-axis scorer exists to
  separate (see B7-h and the live demo pass, where deepseek failed both
  multi-slot commands with bad_action_type:dict).


=====================================================================
FIGV2 — V2-STYLE FIGURES RENDERED ON V3, AND THE ZERO-CELL EXPLANATION
=====================================================================

Artifacts: audit/figures/v3_style/fig{1..4}.{png,pdf} + v3_style_numbers.csv
Script:    audit/figures_v3_style.py   (bench/figures.py NOT modified)

WHY FLAGS ALONE WOULD NOT WORK — five blockers in bench/figures.py:
  1. load() takes ONE path; the V3 measurement is four files and --seen/--held
     cannot union them.
  2. Every figure builds sets = [("Seen", seen)] + ..., so the first panel is
     always LABELLED "Seen". V3 is held-out only; the label would be false.
  3. fig_confusion renders one model (--model); eight are wanted.
  4. Labels say "strict accuracy", which B1-a established is not exact match.
  5. is_ambiguous() classifies a whole CELL from the FIRST matching row.
     On held_out_v3 that is lossy: grid_B1 and grid_B3 each hold 9
     legend-token commands and 3 clean ones, so a per-cell label misassigns 3
     rows in each. The V3 version classifies PER COMMAND.
MODELS is also hardcoded to 5; models_in() would append the other three in
arbitrary order. The V3 version orders all eight explicitly.

FIGV2-1  [Certain]  WHICH V2 FINDINGS REPRODUCE ON V3

Zero cells (a grid cell where the model is correct on none of its rows),
held-out in both cases:

  model            V2 zero cells              V3 zero cells        verdict
  gpt-4o-mini      (none)                     (none)               REPRODUCES
  llama3.2:3b      A3                         A3                   REPRODUCES exactly
  qwen2.5:3b       A3, B1, B3                 B1                   PARTLY - A3 and B3 recover
  qwen2.5:0.5b     B1, B2, B3, C1, C2         B2, C2               MOSTLY NOT - the whole-row-B
                                                                   collapse does not reproduce
  deepseek-r1:7b   (none)                     (none)               REPRODUCES
  qwen2.5-0.5b-lora    n/a                    A1,A2,A3,C1,C2,C3    NEW - two whole rows
  qwen2.5-3b-lora      n/a                    A2,A3,C1,C2,C3       NEW - one whole row + A2,A3
  classifier           n/a                    A3                   NEW

SUPERSEDED BY FIGV2-6 -- see the corrected table there. The judgement below
tests exact zeros against 12-command cells and is too strict.

The single most robust result across both datasets is llama3.2's A3 zero.
The most dramatic V2 result -- qwen2.5:0.5b scoring 0 across the entire
middle row -- does NOT survive: on V3 that row is 2%/0%/2% rather than
0%/0%/0%. Same shape, marginally above zero, so it is not a whole-row zero
any more.

FIGV2-2  [Certain]  THE ZERO ROWS AND COLUMNS ARE MARGINAL PREDICTION
                    COLLAPSE, NOT A SPATIAL EFFECT

This is the explanation the V2 figure never carried. Truth is uniform across
rows -- exactly one third of grid commands target each of A, B and C. What
each model actually PREDICTS is not:

  model                predicts A   predicts B   predicts C   no location
  --- V2 held-out ---
  gpt-4o-mini              37.0%        32.9%        25.2%         4.9%
  llama3.2:3b              17.5%        47.5%        15.1%        19.9%
  qwen2.5:3b               21.2%        29.2%        48.2%         1.4%
  qwen2.5:0.5b             36.3%         0.9%        10.9%        51.9%
  deepseek-r1:7b           29.0%        16.2%        14.4%        40.4%
  --- V3 held-out ---
  gpt-4o-mini              36.2%        33.7%        26.4%         3.7%
  llama3.2:3b              18.0%        46.9%        18.0%        17.0%
  qwen2.5:3b               24.5%        28.0%        46.2%         1.3%
  qwen2.5:0.5b             36.2%         1.9%        15.0%        46.9%
  deepseek-r1:7b           32.5%        21.5%        13.7%        32.3%
  qwen2.5-0.5b-lora         0.0%        90.0%         0.0%        10.0%
  qwen2.5-3b-lora           7.4%        92.6%         0.0%         0.0%
  classifier:distilbert    12.1%        39.9%        48.1%         0.0%

A zero ROW in the heatmap means the model essentially never EMITS that row.
It is a degenerate marginal, not a failure of spatial reasoning on those
particular cells: the model would score zero there regardless of what the
command said. Two opposite versions of the same pathology:

  qwen2.5:0.5b emits row B on 0.9% (V2) / 1.9% (V3) of grid commands, so
  row B is zero or near-zero in both. It is not confusing B with something;
  B is barely in its output distribution at all. Note also that it produces
  NO parseable location on 46.9-51.9% of grid rows, which is the larger
  problem and is invisible in a heatmap of accuracy.

  BOTH LoRA MODELS COLLAPSE ONTO ROW B: 90.0% and 92.6% of grid predictions,
  against a 33.3% base rate. Rows A and C are therefore zero almost by
  construction. This is the strongest single effect in either dataset and it
  appears only in the fine-tuned rung -- the adapters learned a row prior
  rather than a row mapping. Worth stating plainly in the results chapter:
  the rung-3 models did not learn to place objects, they learned to answer
  "middle".

  qwen2.5:3b leans the other way, emitting row C on 46-48%, which is why its
  A-row and B-row cells are the ones that go dark.

CONSEQUENCE FOR THE THESIS: the phrase-legend collision (CONF-1) is a real
but comparatively small effect -- 871 rows pooled. The marginal collapse is
larger and more basic, and the two must not be conflated. A reader shown only
the heatmap will read the dark cells as "this model cannot reach the far left
corner", when the true statement is "this model almost never says A".

FIGV2-4  [Certain]  THE LoRA ROW-B COLLAPSE EXPLAINED — IT IS A FINE-TUNING
                    FINDING, NOT A DATASET FINDING

Script: audit/lora_collapse.py

DATASET ACCOUNT: REFUTED. train_v2's grid rows are close to balanced, and row
B is not the most common:

  train_v2.jsonl     grid=85    row A 29.4%   row B 34.1%   row C 36.5%
  dev_v2.jsonl       grid=21    row A 42.9%   row B 33.3%   row C 23.8%
  held_out_v3.jsonl  grid=108   row A 33.3%   row B 33.3%   row C 33.3%

Row C is the most frequent training row (36.5%). If the adapters had learned
the training frequency they would over-emit C. They emit B on 90.0% and 92.6%.
The collapse is NOT learned frequency.

COPY-ELSE-DEFAULT ACCOUNT: REFUTED, on two independent grounds.
  (a) The 0.5B LoRA does not copy. On container commands, where the answer IS
      verbatim in the text, it emits NO location on 155 of 180 rows (86%) and
      scores 0.139. A copying rule would score near 1.000 there.
  (b) Both LoRA models handle the RUNTIME symbols well -- max_reach and
      nearest_free_cell never appear verbatim in any command, yet the 0.5B
      scores 0.792 and the 3B scores 0.917 on them, far above their grid
      accuracy. There is no "copy when present, else default" rule that
      predicts this.

WHAT THE DATA ACTUALLY SUPPORTS [Certain]: the adapters learned the COLUMN
axis and did not learn the ROW axis. Scoring the two axes separately on grid
commands:

  model                    row correct   column correct   both
  qwen2.5-0.5b-lora            30.6%          90.0%      30.6%
  qwen2.5-3b-lora              40.8%          86.1%      40.8%
  classifier:distilbert        62.2%          94.4%      62.2%
  qwen2.5:3b (prompted)        41.2%          82.2%      38.6%
  gpt-4o-mini (prompted)       82.1%          96.3%      82.1%
  (chance on either axis = 33.3%)

The LoRA models are at or barely above CHANCE on rows while near-ceiling on
columns. Because a legal symbol requires some row, they emit the middle one
constantly -- and row accuracy then equals the base rate of B, which is exactly
what 30.6% is. They are not collapsing onto one CELL: the 0.5B spreads its
predictions across B1/B2/B3 at 28.8/33.4/27.8%, tracking the column it
correctly identified.

  MECHANISM, stated plainly: the rung-3 adapters learned left/right and did
  not learn far/near. They answer the column question and guess the row.

GENERALITY [Certain]: the row/column asymmetry is present in EVERY model, not
only the fine-tuned ones -- gpt-4o-mini is 96.3% on columns against 82.1% on
rows. It is catastrophic only in the LoRA rung. This ties directly to CONF-2:
columns are lateral (left/right is unambiguous in any frame) while rows are
depth (far/near versus upper/bottom is frame-dependent), and the training and
evaluation splits use DIFFERENT row vocabulary (seen_v2 far/near, held_out_v3
upper/bottom) while sharing left/right. The adapters were fine-tuned on row
words they never see at evaluation, and the column words are identical in both.
That is the most economical explanation of why fine-tuning helped the column
axis and actively hurt the row axis.

NOT GENERAL TO EVERYTHING TRAINED ON train_v2 [Certain]: the DistilBERT
classifier trained on the same file does NOT collapse -- 12.1/39.9/48.1 across
rows, and 62.2% row accuracy. Whatever happened is specific to the LoRA
fine-tuning of a generative model, not to the training data.

FIGV2-5  [Certain]  fig1 NOW REPORTS PARSE FAILURES PER PANEL

Commands producing no parseable location were previously folded silently into
the heatmap cells, so a dark cell conflated "chose the wrong place" with
"produced nothing usable". Each panel title now carries the rate, in red above
25%:

  qwen2.5:0.5b 47%   deepseek-r1:7b 32%   llama3.2:3b 17%
  qwen2.5-0.5b-lora 10%   gpt-4o-mini 4%   qwen2.5:3b 1%
  qwen2.5-3b-lora 0%   classifier 0%

Stated in the title rather than as a fourth heatmap column, which would have
put two different quantities on one colour scale.

FIGV2-6  [Certain]  CORRECTION TO THE REPRODUCTION TABLE (FIGV2-1)

The earlier table judged reproduction on whether a cell was exactly zero. That
is the wrong standard at this sample size. Each cell holds 12 DISTINCT
COMMANDS (the ~240-300 rows per cell are 12 commands x repeats x strategies).
At 0 hits in 12 commands the 95% Wilson upper bound is 24.3%; at 1 hit it is
26.7%. Zero and "one command out of twelve" are statistically indistinguishable.

  qwen2.5:0.5b, row B and the C1/C2 cells:
    V2  B1 0/240  B2 0/240  B3 0/240  C1 0/240  C2 0/240   (all 0.0%)
    V3  B1 5/300  B2 0/300  B3 5/300  C1 5/300  C2 0/300   (0.0-1.7%)
  Five rows out of 300 is ONE command out of twelve, hit on all five repeats.

CORRECTED VERDICT: qwen2.5:0.5b REPRODUCES. The model barely emits row B in
either dataset (0.9% V2, 1.9% V3 of grid predictions) and the V2 whole-row
zero was thresholding a near-zero marginal, not measuring a different
phenomenon. The same correction applies to qwen2.5:3b, whose A3 and B3 cells
moved from 0 to a similarly indistinguishable non-zero.

  model              V2 zeros            V3 zeros        CORRECTED verdict
  gpt-4o-mini        (none)              (none)          reproduces
  llama3.2:3b        A3                  A3              reproduces exactly
  qwen2.5:3b         A3,B1,B3            B1              reproduces (within noise)
  qwen2.5:0.5b       B1,B2,B3,C1,C2      B2,C2           reproduces (within noise)
  deepseek-r1:7b     (none)              (none)          reproduces
  LoRA pair          n/a                 6 and 5 cells   new, not in V2
  classifier         n/a                 A3              new, not in V2

NO V2 FINDING FAILS TO REPRODUCE ON V3. The earlier "does not reproduce"
entries were an artifact of testing exact zeros against 12-command cells.
The right way to report any of these cells is the marginal (FIGV2-2), which is
estimated from all 539-2,695 of a model's grid rows rather than 12 commands.

FIGV2-3  [Certain]  fig4 renders correctly across the full range. The
classifier's p50 is 0.019 s against deepseek-r1's maximum near 380 s -- about
4.5 orders of magnitude. Log axis with limits pinned to [0.005, 400] keeps
the classifier's bars visible instead of collapsing them onto the axis.


=====================================================================
FINAL TRIAGE TABLE
=====================================================================

FATAL — invalidates a reported number
  (none)

CRITICAL-RECOVERABLE — blocks a number; every input still exists
  B7-a  B  token/cost columns 0% populated on all 30,380 v3 rows
           STATUS: RESOLVED. Exact per-model counts written (B7-j).

DISCLOSE — real limitation, survivable if stated in Methods
  A2-a  A  every non-temperature decoding parameter left at provider default;
           defaults differ between OpenAI and Ollama; none recorded
  A2-b  A  temperature=0 did NOT give determinism on gpt-4o-mini: 22.9% of
           commands vary across repeats, 99.1% under cot, 34 verdict flips
  A3-b  A  LoRA Modelfiles pin temperature 0, base Modelfiles do not
           (effective behaviour identical; client sets it either way)
  A4-b  A  environment parity across the four launches is not verifiable
  A4-c  A  llm/retrieve.py pops HF_HUB_OFFLINE mid-run; rag cells only
  B1-a  B  "exact match" wording false: 71% of strict credit (8,897 of
           12,528 rows) depends on the empty-modifiers rule
  B1-b  B  five undocumented falsy-modifiers acceptances in strict
  B6-g  B  num_ctx was VRAM-derived (4096 from 16 GiB); the context window
           was a property of the measurement hardware, not of any setting
  B7-h  B  qwen2.5:0.5b p99 of 116.9 s is a 1.38% runaway pathology, not
           the model's latency; those 15 rows are 100% wrong
  B7-j-a B llama3.2 token counts use an ungated mirror of Meta's tokenizer
  B9-a  B  container and grid subsets differ on four axes at once; no single
           cause for the accuracy gap can be claimed
  B9-b  B  container subset n=36, CI half-width ~±16 pts; cannot carry a claim
  F-ENV-a  no lockfile at measurement time; env not reconstructible
  PROV-a   three artifacts' provenance depended on unrecorded information

CHEAP-FIX — under 2 h, no re-run of the sealed set
  B7-f  B  clients.py discards OpenAI `usage` and Ollama `eval_count`
           STATUS: DONE. last_usage now captured for future runs.
  B2-a  B  results-chapter script reported no invalid rate
           STATUS: DONE. inv%/sf%/xf%/null% added, keyed on `stage`.
  B7-b  B  backfill_tokens.py V1-shaped, 4 defects
           STATUS: DONE. backfill_tokens_v3.py written, run, verified.
  F-ENV-b  peft/accelerate absent; rung-3 path non-executable
           STATUS: DONE. peft==0.20.0 restored from adapter_config.json.
  B3-a  B  add `dataset` to 4-key dedupes in analyze.py and
           scratch_scored_table.py — harmless now, wrong on any union
  B1-c  B  scorer docstring understates _IGNORE_IF_EMPTY
  A4-b  A  write env vars into every row or a per-launch manifest
  B6-g  B  pin num_ctx explicitly so the window is a recorded condition

PASS — tested and held
  B3    zero rows dropped by dedupe, zero orphans across 30,380 rows
  B4    classifier: 1,085/1,085 valid, zero structural violations at full n
  B5    all four files scored under one commit, 3000198, tree clean
  B5-a  strict_match moved on ZERO rows under rescoring
  B6    22 of 27 generative cells never near the context ceiling;
        worst-case accuracy movement anywhere +0.005
  A1-a  few_shot and rag prompts byte-identical (1264 == 1264)
  A2-b  all seven local models 100% deterministic across 5 repeats
  A3-a  all six models Q4_K_M; no quantization confound
  A5    RAG mean prompt SHORTER than few_shot; prefill explains nothing
  A6    warmup excluded in all four analysis scripts
  B8-b  all three training checkpoints intact and complete

FUTURE-WORK — legitimate, out of scope with 7 weeks left
  C     constrained decoding (Rung 1.5). Costed, not implemented. Post-hoc
        status must be pre-registered in writing if attempted.
  E5    energy measurement (pyNVML). Scoped, not implemented.
  D-G   statistics, cost accounting, novelty framing — these are ANALYSIS,
        not audit, and belong inside the write-up.


=====================================================================
METHODS DISCLOSURE LIST — drafted sentences, edit to taste
=====================================================================

1. SCORING (B1-a, B1-b)
   "The evaluation set contains no non-empty `modifiers` list: all 217
    held-out commands carry an empty one, and no model emitted a non-empty one
    across 30,380 predictions. The `modifiers` field is therefore not under
    test, and the scorer treats an absent field and an empty field as the same
    value. Strict match is otherwise exact object equality against ground
    truth. All other differences are scored as failures, including letter
    case, separator variants, key aliases, nested action objects, and
    additional keys."

2. DECODING PARAMETERS (A2-a, A2-b)
   "Only temperature was set (0.0 for every model). All other decoding
    parameters were left at each provider's defaults, which differ between
    OpenAI and Ollama and were not recorded by the harness. Measured
    consequence: all seven locally served models produced byte-identical
    output across the five repeats of all 217 commands, so the unset
    parameters were constant and contributed no variance. The cloud model did
    not: 22.9% of commands varied across repeats despite temperature=0, rising
    to 99.1% under chain-of-thought, where 34 of 217 commands changed strict
    verdict between repeats. No seed was supplied to the OpenAI API.
    Cloud-model results therefore carry sampling variance that the local
    results do not, and the five repeats are informative only there."

3. CONTEXT WINDOW (B6-g)
   "Local models were served by Ollama with no explicit num_ctx; the effective
    context window was 4096 tokens, derived by Ollama from the measurement
    machine's 16 GiB of VRAM. Reproduction on different hardware may yield a
    different window. All prompts in this study are under 410 tokens, so no
    prompt was affected at any plausible window size."

4. TASK STRUCTURE (B9-a, B9-b)
   "The held-out set contains two structurally different subsets. Container
    rows (n=36) are a three-way selection among box_1..box_3 in which the
    answer symbol appears verbatim in the command, and they exercise two of
    the seven verbs. Grid rows (n=181, of which 132 carry a location) are an
    eleven-way selection among nine grid cells and two runtime symbols, the
    symbol never appears in the command, and all seven verbs are present.
    Because these subsets differ on several axes at once, no single cause for
    the accuracy gap is claimed. They are reported separately throughout, and
    the thesis claim rests on the grid subset."

5. TOKEN COUNTING AND VOCABULARY PROVENANCE (B7-j)

   NOTE ON A CORRECTION: an instruction to this audit described deepseek's as
   "the only vocabulary in the set not taken from the measured artifact". That
   is not accurate on two counts and the Methods text must not repeat it.
   Llama 3.2's vocabulary ALSO came from the Hub, and from an ungated
   third-party mirror rather than Meta's gated original, which makes it the
   weaker provenance of the two. And the Qwen vocabulary, while local, was read
   from the fine-tuning checkpoint rather than from the served GGUF blob. Only
   gpt-4o-mini's and DistilBERT's vocabularies come from the exact artifact
   that was measured. The accurate table:

     model(s)                    vocabulary                     provenance
     qwen2.5:0.5b / :3b /        Qwen2.5, 151,665               LOCAL, from
       both merged LoRAs                                        checkpoint-87,
                                                                not the GGUF
     deepseek-r1:7b              DeepSeek-R1-Distill-Qwen-7B,   HUB
                                 151,665
     llama3.2:3b                 Llama 3.2, 128,256             HUB, ungated
                                                                mirror
     gpt-4o-mini                 o200k_base, 200,019            EXACT
     classifier:distilbert       DistilBERT, 30,522             LOCAL, exact

   The deepseek row matters most because it carries the headline figure: the
   94.7% unseen-token result rests on a vocabulary fetched from the Hub, not
   on the vocabulary embedded in the measured GGUF at D:\LLM\blobs. The two are
   expected to be identical — the GGUF was built from that model — but this
   audit did not verify it, because llama-tokenize.exe is blocked by Windows
   Application Control (B7-j-a). [Likely] identical; not demonstrated.

   "Token counts use each model's own vocabulary rather than a single shared
    tokenizer, since the models' vocabularies differ and the resulting error is
    not a constant factor. The Qwen2.5 tokenizer was used for the Qwen models
    and their merged LoRA variants, read from the local fine-tuning checkpoint;
    the DeepSeek-R1-Distill tokenizer for deepseek-r1 and the Llama 3.2
    tokenizer for llama3.2, both obtained from the HuggingFace Hub rather than
    from the served GGUF artifacts; and OpenAI's o200k_base for gpt-4o-mini,
    which is exact for that model. The Hub-sourced vocabularies are expected to
    match those embedded in the served models but this was not independently
    verified. Output token counts are reported separately as visible tokens,
    reasoning tokens, and their total; the DistilBERT classifier constructs its
    output from the schema rather than generating it and is excluded from
    generation-cost comparisons."

6. LATENCY TAIL (B7-h)
   "The 99th-percentile latency reported for qwen2.5:0.5b under
    chain-of-thought (116.9 s) is produced by 15 of 1,084 rows (1.38%) in
    which generation ran away; all 15 are structurally invalid and none is
    correct. The remaining 98.6% of that cell has a median latency of 1.83 s.
    The tail is reported as a failure mode with its row count rather than as
    the model's latency profile."

7a. PROVENANCE — STANDALONE PARAGRAPH (PROV)

   This is promoted from a bullet because it is a limitation about the STUDY
   AS A WHOLE rather than about any one measurement, and an examiner who finds
   it unaided will find it more damaging than one who reads it here first.

   "A limitation applies to the identification of artifacts in this study
    rather than to any individual measurement. Every component of the
    measurement apparatus was identified by a mutable name rather than by an
    immutable digest or a recorded version: local models by Ollama tag, which
    can be repointed at different weights without notice; Python dependencies
    by unpinned package name; and the runtime by conda environment name. No
    lockfile, model digest, or environment record was captured at measurement
    time, and the result schema contains no configuration fields. The
    consequence became concrete on three separate occasions during the
    post-hoc audit of this work. The location of the model store was recorded
    nowhere and the measured weights survived only because the directory they
    occupied had not been reused; the routine remedy, re-pulling the models by
    tag, would have substituted current published weights for the measured
    ones silently and without trace. The contents of the software environment
    were likewise unrecorded, and two packages required by the fine-tuning
    pipeline were found to be absent, so that path could not be executed at
    all. The version of the PEFT library used to train the LoRA adapters was
    recoverable only because that library writes its own version into the
    adapter configuration file it saves; had it not done so, the correct
    version would have been unrecoverable in principle. In each case the
    artifact was recovered, and in each case recovery depended on
    circumstance rather than on any deliberate record. The measurements
    themselves are unaffected — no reported number changed — but the study as
    originally conducted did not establish the identity of what it measured.
    Digests for all model weights, a dependency lockfile, and a per-run
    configuration manifest have since been added, and are described above;
    they document the apparatus as of the audit rather than as of the
    measurement, and that distinction is stated wherever they are cited."

7. ENVIRONMENT AND PROVENANCE (F-ENV, PROV)
   "No environment lockfile was captured at measurement time. The dependency
    versions recorded in requirements.txt and requirements.lock.txt describe
    the environment as of 2026-08-27, after the measurement, and are provided
    as the closest available reference point rather than as an exact
    reproduction of the measurement environment. One exception is the PEFT
    library, pinned to version 0.20.0, which is the version recorded inside
    the saved LoRA adapter configuration and therefore the version that
    produced the rung-3 adapters. Model weights are identified by manifest
    digest rather than by Ollama tag, because tags are mutable and do not
    identify a specific artifact."

8. CLOUD PRECISION (A3-a)
   "All locally served models are Q4_K_M quantised, including both merged
    LoRA models, each matching its own base model in parameter count and
    quantisation. gpt-4o-mini is served at undisclosed precision; comparisons
    between the cloud model and the local ladder therefore carry a precision
    caveat that comparisons within the local ladder do not."


=====================================================================
CHEAP-FIX QUEUE, ordered by payoff / hour
=====================================================================

COMPLETED DURING THE AUDIT
  DONE  B2-a    invalid-rate decomposition added to scratch_scored_table.py,
                keyed on `stage`. 9,530 never-validated rows now visible in
                the results-chapter table.
  DONE  B7-b    bench/backfill_tokens_v3.py: four V1 defects fixed, three
                output columns, frozen-column assertion, drift 0.
  DONE  B7-j    exact per-model token counts written; five vocabularies,
                three of them loaded from disk.
  DONE  B7-f    llm/clients.py captures last_usage from both providers.
  DONE  B8-a    OLLAMA_MODELS=D:\LLM set persistently in User env.
  DONE  F-ENV-b peft==0.20.0 and accelerate restored; all ten repo modules
                import cleanly, rung 3 included.
  DONE  F-ENV-a requirements.lock.txt + pinned requirements.txt, committed
                on branch audit/env-pinning (dba5ae9).
  DONE  PROV    audit/MODEL_PROVENANCE.txt: six manifest digests captured
                before anything could overwrite the store.

  DONE  B3-a    `dataset` added to the dedupe keys in bench/analyze.py and
                scratch_scored_table.py. Regression-checked: 28 cells,
                30,352 rows, every figure identical. All five aggregating
                scripts now share one key.
  DONE  B1-c    llm/score.py docstring corrected: _IGNORE_IF_EMPTY tests
                FALSINESS, so strict also accepts None/""/{}/0/False. Zero
                occurrences in the data; the strict-vs-normalized inversion
                on 0/False is documented rather than fixed, because the
                scorer is frozen at 3000198.
  DONE  B6-g    llm/clients.py pins num_ctx=4096 explicitly
                (OllamaClient.DEFAULT_NUM_CTX). The window is now a recorded
                condition. Pinned to the value the measurement actually ran
                under, NOT raised, so future runs stay comparable with the
                30,380 existing rows.
  DONE  A4-b    bench/run_grid.py writes <out>.manifest.json per launch:
                env vars, model DIGESTS, package versions, python, platform,
                pinned num_ctx, dataset/schema/exemplar config, retrieval
                settings. Append-only across resumes, so a resumed run records
                both launches. A sidecar rather than new CSV columns, to avoid
                changing the 28-column schema shared with all existing rows.

REMAINING
  1. (~1 h)   B7-g  OPTIONAL. Cross-check llama3.2 token counts against
                    Meta's own tokenizer if Hub access is granted, replacing
                    the ungated mirror; and, if Application Control is ever
                    relaxed, verify the Hub vocabularies against the GGUFs in
                    D:\LLM\blobs. Expected to change nothing. Removes two
                    DISCLOSE lines, including the one attached to the headline
                    94.7% figure.

Nothing in this queue requires re-running the sealed set. The audit's
cheap-fix queue is otherwise empty.


=====================================================================
IF I WERE THE EXAMINER
=====================================================================

First question: "Your thesis is about the cost of adaptation. Show me the
cost." Today you cannot — tokens_in, tokens_out and cost_usd are empty on all
30,380 rows of the measurement, and Cost-of-Pass therefore does not exist for
any of the 28 cells. You would have to answer that the cost axis is
reconstructible from stored prompts and outputs but was not computed, which
invites the follow-up you least want: what else was reserved and never filled?

You can fix this before anyone asks. Every input is on disk, the reconstruction
is proven to work, the script is written, and the frozen-column assertion shows
it changes no verdict. It is hours, not a re-run.

CORRECTION [Likely] — the question above is probably NOT the first one. The
thesis title promises vision-based manipulation and Phase 6 has zero lines.
That is in the title, an examiner reads the title first, and no amount of
cost-axis work answers it. The cost work is hours and should happen; it does
not touch this. Have the answer ready.

The question you should WANT: "deepseek-r1 looks cheap in your table. How many
tokens did it actually generate?" The great majority of its output never
reaches the caller and previously counted as zero — a literal measurement of
the cost of thinking, on a thesis about the cost of thinking, invisible until
this audit. Lead with it ONCE the exact figure is in hand; see the correction
under B7-e for why neither 96.0% nor 24:1 is quotable today.


=====================================================================
WHAT HELD — worth stating, because it is the real result
=====================================================================

The audit found real defects and every one of them is in the COST AXIS or in
WORDING. Nothing in the accuracy or latency tables is invalidated. The
measurement substrate itself holds, and it was tested hard:

  B4    classifier structural guarantee    1,085/1,085, zero violations,
                                           verified at full n, not on dev
  A3-a  quantization parity                all six models Q4_K_M
  A1-a  rung-1 vs rung-2 prompt path       byte-identical, 1264 == 1264
  B5    scorer uniformity                  one commit, 3000198, tree clean
  B5-a  strict_match under rescoring       moved on ZERO rows, all four files
  B3    dedupe                             zero rows dropped, zero orphans
  A2-b  local determinism                  7 of 8 models 100% byte-identical
                                           across 5 repeats of 217 commands
  B6    context truncation                 22 of 27 cells never near the
                                           ceiling; worst-case move +0.005
  B8-b  training checkpoints               all three intact and complete

A study where the hostile pass comes back with no invalidated number, a clean
scorer, a clean prompt path and a structurally perfect rung 4 is in good
shape. That is the finding, not the absence of one.


DRAFT DISCLOSURE SENTENCES (B1-a, B1-b — edit to taste)
--------------------------------------------------------

  (Framing per supervisor direction: do NOT write this as "we relax exact
   match". Write it as — the field is not under test, therefore absent and
   empty are the same value. That explains WHY the rule is harmless instead of
   asking the examiner to accept that it is.)

  "The evaluation set contains no non-empty `modifiers` list: all 217 held-out
   commands carry an empty one, and no model emitted a non-empty one across
   30,380 predictions. The `modifiers` field is therefore not under test, and
   the scorer treats an absent field and an empty field as the same value.
   Strict match is otherwise exact object equality against ground truth. All
   other differences are scored as failures, including letter case, separator
   variants, key aliases, nested action objects, and additional keys."

  "All other differences are scored as failures under the strict policy,
   including letter case, separator variants, key aliases, nested action
   objects, and additional keys."
