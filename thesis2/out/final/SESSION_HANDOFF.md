# Session handoff — final run of the thesis2 study

Written 2026-09-28 from repository files only, to start a fresh chat. Every number
is copied from the file named next to it. Tags: [Certain] read from a file,
[Likely] an inference from one, [Guessing] not established.

Repository: `C:\Users\USER\llm-6dof-arm`. Environments: `D:\condaenvs\thesis2gpu`
(GPU work, most scripts), `C:\Users\USER\.conda\envs\thesis` (Ollama cells; the only
env with the `ollama` package), `D:\condaenvs\thesis2rl` (GRPO training only).
PowerShell conventions: `;` not `&&`, script files not multi-line `python -c`.

---

## 1. The task and the data

A natural-language command for a 6-DOF arm ("put the bottle in the upper left
corner of the table") is parsed into a JSON action: verb, target, location,
modifiers (`schema/schema_v2.json`). [Certain]

- 7 verbs: pick_place, fetch, hold, move_to, inspect, stop, reject.
- 3 targets: rubiks_cube, bottle, toy_car.
- 14 locations: a 3x3 grid (`grid_A1`..`grid_C3`; row A = far, B = middle,
  C = near; column 1 = left, 3 = right), 3 boxes (`box_1`..`box_3`), and 2 runtime
  symbols (`max_reach`, `nearest_free_cell`).

**Metric: action match** = verb, target and location all correct (`llm/score.py`,
the single scorer). **Row accuracy** = the grid row correct, over the 108 grid
commands. [Certain]

| split | commands | role |
|---|---|---|
| `data/train_v2.jsonl` | 230 | training |
| `data/dev_v2.jsonl` | 57 | checkpoint selection |
| `data/seen_v2.jsonl` | 290 | few-shot exemplars; demo commands |
| `data/held_out_v3.jsonl` | 217 | held out; unseen row words upper/bottom |
| `data/held_out_v4c.jsonl` | 217 | held out; unseen row words back/front |

held_out_v3 = 108 grid + 24 runtime (12 max_reach, 12 nearest_free_cell) + 36 box +
49 no-location (verified from the file; the old "132 grid" was 108 + 24).
[Certain]

**held_out_v3 is confounded; held_out_v4c is the fair test.** The V5 prompt
legend was written after seeing V3 fail on held_out_v3, so prompted arms are not
blind there. back/front appear in no training set and no legend.
(`METHODS_NOTES.md` item 1.) [Certain]

Rules in force: held-out data is for inference and scoring only; new arms build
vocabulary from train_v2 only; each arm runs once on held_out_v4c; withdrawn arms
(A1.7 field-level FLMD, every SALSA digits mapping) are never registered.
(`thesis2/final/RUN_ORDER.md`, "Rulings in force".) [Certain]

## 2. Every arm (`thesis2/final/arms.py`, 30 arms)

| arm | what it is |
|---|---|
| flat_gpu | V5 rung-4 flat DistilBERT classifier, GPU |
| flat_cpu | the same weights on CPU (CPU ms instead of J/command) |
| factorized_fp16 | factorized location head (type / row / column), FP16 |
| factorized_int4 | factorized head, per-channel symmetric INT4 |
| tinybert | TinyBERT-4 classifier (legacy, retrained 217/217) |
| minilm_probe | frozen MiniLM + linear probe (legacy) |
| knn_minilm | cosine 1-NN over train_v2 MiniLM embeddings (legacy) |
| tfidf_logreg | TF-IDF 1-2gram + logistic regression (legacy) |
| bilstm | BiLSTM from scratch, train_v2 vocabulary (legacy) |
| flant5 | Flan-T5-base seq2seq (legacy) |
| textcnn | NEW: TextCNN, train_v2 vocabulary |
| mlpbow | NEW: bag-of-words MLP, train_v2 vocabulary |
| rules_open | hand grammar; **vocabulary-unfair** (lists back/front) |
| rules_firewalled | hand grammar with only train_v2 row words (honest floor) |
| lora_greedy | qwen2.5-3b-lora (checkpoint-87), transformers FP16 merged, greedy |
| a8_speculative | A8: classifier-drafted speculative decoding, verifier = lora_greedy |
| a10_speculative | A10: A8 + prefix KV cache |
| a12_smalllm | A12: conventional qwen2.5-0.5b drafter, gamma = 4 |
| salsa_full_* (4) | NEW: every field read from letter-token logits in one pass; 3B/0.5B, LoRA and base |
| grpo_base | NEW: GRPO on the base 3B (trained in thesis2rl) |
| grpo_sft | NEW: GRPO on checkpoint-87 |
| cascade_local_tauA | factorized head; row < tau_A = 0.50 -> qwen2.5-3b-lora SALSA row |
| pipeline_always_local | same, tau = 1.0 (every grid row delegated) |
| cascade_cloud_tauA | row < 0.50 -> gpt-4o-mini (R = 5) |
| cascade_cloud_always | every grid row -> gpt-4o-mini (R = 5) |
| cascade_oracle | **ceiling, not a system**: the true row is supplied |
| ensemble | offline: classifier and LoRA agree -> classifier, else gpt-4o-mini/rag |

Plus the **V5 cell grid**: 36 cells (31 local Ollama + 5 gpt-4o-mini), 12,152 rows
per split, effective n = 217 commands (`results_v5cells_<split>.csv`). [Certain]

## 3. Final accuracy, both splits (`metrics_final.csv`)

Sorted by held_out_v4c action match. Cloud arms collapse 5 repeats by majority
vote. [Certain]

| arm | v3 action | v4c action | v3 row | v4c row | class |
|---|---|---|---|---|---|
| cascade_oracle | 0.9401 | 0.9401 | 1.0 | 1.0 | ceiling, not a system |
| ensemble | 0.9631 | 0.8387 | 1.0 | 0.75 | system |
| cascade_cloud_always | 0.7926 | 0.7788 | 0.7037 | 0.6759 | system |
| cascade_cloud_tauA | 0.7327 | 0.7696 | 0.5833 | 0.6574 | system |
| rules_open | 0.7235 | 0.7235 | 0.8889 | 0.8889 | vocabulary-unfair |
| flat_cpu | 0.7465 | 0.7143 | 0.6204 | 0.5556 | system |
| flat_gpu | 0.7465 | 0.7143 | 0.6204 | 0.5556 | system |
| bilstm | 0.7005 | 0.7005 | 0.6852 | 0.6852 | system |
| grpo_sft | 0.9585 | 0.7005 | 1.0 | 0.4815 | system |
| textcnn | 0.6959 | 0.6959 | 0.6759 | 0.6759 | system |
| grpo_base | 0.7926 | 0.6959 | 0.7222 | 0.5463 | system |
| knn_minilm | 0.6452 | 0.6959 | 0.5648 | 0.6852 | system |
| minilm_probe | 0.6498 | 0.6544 | 0.6204 | 0.6574 | system |
| cascade_local_tauA | 0.7327 | 0.6452 | 0.5833 | 0.4074 | system |
| mlpbow | 0.6406 | 0.6406 | 0.6667 | 0.6667 | system |
| factorized_fp16 | 0.6083 | 0.6359 | 0.3333 | 0.3889 | system |
| a10_speculative | 0.9539 | 0.6221 | 1.0 | 0.3333 | system |
| a12_smalllm | 0.9539 | 0.6221 | 1.0 | 0.3333 | system |
| a8_speculative | 0.9539 | 0.6221 | 1.0 | 0.3333 | system |
| lora_greedy | 0.9539 | 0.6221 | 1.0 | 0.3333 | system |
| factorized_int4 | 0.6083 | 0.6175 | 0.3426 | 0.3611 | system |
| tinybert | 0.576 | 0.5806 | 0.3333 | 0.3426 | system |
| flant5 | 0.6175 | 0.5714 | 0.4259 | 0.3333 | system |
| pipeline_always_local | 0.8433 | 0.5438 | 0.8056 | 0.2037 | system |
| rules_firewalled | 0.447 | 0.447 | 0.3333 | 0.3333 | system |
| salsa_full_qwen2.5-3b-lora | 0.5023 | 0.4055 | 0.6574 | 0.3333 | system |
| salsa_full_Qwen2.5-3B-Instruct | 0.3594 | 0.3502 | 0.2963 | 0.1759 | system |
| salsa_full_Qwen2.5-0.5B-Instruct | 0.0461 | 0.0599 | 0.3056 | 0.2778 | system |
| salsa_full_qwen2.5-0.5b-lora | 0.0369 | 0.0369 | 0.0093 | 0.0093 | system |

The ensemble has 217 commands here (command 0 included); the earlier run's file
has 216. The V5 cells on both splits, with descriptive 95% CIs, are in
`v5cells_both_splits.csv` and `FINAL_RUN_REPORT.md` section 4b. [Certain]

## 4. The primary family, H1-H8 (`final_significance.csv`)

Clustered bootstrap over command_id (2,000 resamples, seed 20260827), exact McNemar,
Holm across the eight; CLAIMABLE iff the CI excludes 0 and p_holm < 0.05. [Certain]

| id | split | A vs B | metric | acc A | acc B | diff pts | 95% CI | p_holm | verdict |
|---|---|---|---|---|---|---|---|---|---|
| H1 | v4c | flat_gpu vs lora_greedy | action | 0.7143 | 0.6221 | 9.22 | [4.15, 14.75] | 0.00477296 | CLAIMABLE |
| H2 | v4c | flat_gpu vs lora_greedy | row | 0.5556 | 0.3333 | 22.22 | [13.89, 30.56] | 7.15254e-07 | CLAIMABLE |
| H3 | v3 | lora_greedy vs flat_gpu | action | 0.9539 | 0.7465 | 20.74 | [14.75, 26.73] | 4.93779e-10 | CLAIMABLE |
| H4 | v3 | factorized_fp16 vs flat_gpu | action | 0.6083 | 0.7465 | -13.82 | [-19.35, -9.21] | 3.01927e-06 | CLAIMABLE |
| H5 | v4c | cascade_cloud_tauA vs flat_gpu | action | 0.7696 | 0.7143 | 5.53 | [0.92, 10.14] | 0.0713962 | nominal (does not survive correction) |
| H6 | v4c | grpo_base vs lora_greedy | action | 0.6959 | 0.6221 | 7.37 | [2.76, 12.44] | 0.0156686 | CLAIMABLE |
| H7 | v3 | salsa_full_qwen2.5-3b-lora vs lora_greedy | action | 0.5023 | 0.9539 | -45.16 | [-52.07, -38.25] | 1.47962e-25 | CLAIMABLE |
| H8 | v4c | textcnn vs flat_gpu | action | 0.6959 | 0.7143 | -1.84 | [-8.29, 5.07] | 0.683489 | NOT CLAIMABLE |

6 of 8 claimable. H1, H2, H4 and H5 were measured once before the family was
declared (2026-09-22); H3, H6, H7 and H8 are fully pre-registered
(`s4_significance.py`, `PRE_MEASURED`). The exploratory family (1,732
comparisons, Holm across all) is reported, never headlined. [Certain]

## 5. Efficiency

Protocol: idle baseline under 2 h old, 3 uncounted warm-ups, >= 120 s load, 60 s
cool-down; J/command = (mean load W - idle median W) x s / N; GPU watts only.
J <= 0 reads "~0 (below GPU floor)"; VRAM above idle <= 0 reads "~0".
(`s3_energy.py`, `unified_efficiency.csv`, `metrics_final.csv`.) [Certain]

**Main table, held_out_v3.**

| system | action | p50 ms | J/command | VRAM above idle MB |
|---|---|---|---|---|
| V5 qwen2.5-3b-lora / zero_shot (Ollama Q4_K_M; n = 216) | 0.9583 | 1623.173 | 64.02359 | 2544.0 |
| lora_greedy | 0.9539 | 878.506 | 71.40392 | 7017.0 |
| a8_speculative | 0.9539 | 111.343 | 19.98257 | 7328.0 |
| a10_speculative | 0.9539 | 56.908 | 9.54235 | 7319.0 |
| a12_smalllm | 0.9539 | 938.225 | 45.6599 | 7640.0 |
| flat_gpu | 0.7465 | 2.826 | 0.14415 | 427.0 |
| flat_cpu | 0.7465 | 8.962 | ~0 (below GPU floor) | ~0 |
| grpo_base | 0.7926 | 882.586 | 70.31138 | 6738.0 |
| grpo_sft | 0.9585 | 881.813 | 70.20741 | 6738.0 |

**Speculative supplement, held_out_v4c.**

| arm | action | p50 ms | J/command | VRAM above idle MB |
|---|---|---|---|---|
| lora_greedy | 0.6221 | 875.733 | 71.17939 | 7020.0 |
| a8_speculative | 0.6221 | 109.506 | 21.23815 | 7154.0 |
| a10_speculative | 0.6221 | 56.062 | 10.91288 | 7286.0 |
| a12_smalllm | 0.6221 | 951.604 | 47.2076 | 7640.0 |

The V5 cell runs in Ollama at Q4_K_M; lora_greedy and A8/A10/A12 run the same
adapter in transformers FP16 -- a runtime difference, not a method difference.
(`METHODS_NOTES.md` item 12.) [Certain]

## 6. Key interpretations

- **The prompt legend determines row accuracy for prompted arms.** lora_greedy,
  its three speculative arms and grpo_sft are perfect on held_out_v3's rows, where
  the V5 legend defines upper/bottom, and collapse on held_out_v4c's back/front
  (grpo_sft 1.0 -> 0.4815): lora_greedy's row accuracy goes 1.0 -> 0.3333, with
  0/48 on the unseen words and 66 defaults to B
  (`metrics_final.csv`, `row_by_word_group`, `row_signature`). [Certain on the
  numbers; that the legend is the cause is Likely -- `METHODS_NOTES.md` item 1]
- **No local arm maps back/front; H1/H2 and H6 are defaults, not understanding.**
  flat_gpu defaults to row C (43/48 on back/front; also on v3 upper/bottom);
  lora_greedy defaults to B; grpo_base splits A/C independently of the word; only
  cascade_cloud_always maps back/front (46/48). The classifier's v4c row advantage
  is mostly its C default coinciding with 'front' (22 of lora_greedy's 24 row
  losses are front commands). GRPO's H6 gain is the removal of the B default, not
  a mapping. Word-dependent predictions on v4c cannot be attributed to the row
  word, because back and front commands use different carrier phrasings.
  (`row_confusion_by_word.csv`, `draft_verifier_gh.csv`; `METHODS_NOTES.md` items
  17, 21-23.) [Certain on the counts; the interpretation is Likely]
- **Speculative decoding is lossless and fast.** A8, A10 and A12 are byte-identical
  to lora_greedy on 217/217 on both splits (the gate in each script). A10 vs
  lora_greedy on held_out_v3: **15.44x** faster at p50 (878.506 / 56.908 ms) and
  **7.48x** less energy (71.40392 / 9.54235 J) -- the "15.4x / 7.5x" figures. On
  v4c: 15.62x and 6.52x. A12's conventional LM drafter is slower than plain
  greedy at p50 (938 vs 879 ms). (`unified_efficiency.csv`.) [Certain]
- **SALSA-full does not work as a parser** (H7: -45.16 pts vs lora_greedy on v3),
  even after the D..Q letter relabel. [Certain]

## 7. Decisions and deviations (full text: `METHODS_NOTES.md`)

1. held_out_v3 confounded; held_out_v4c is the fair test (item 1).
2. H1/H2/H4/H5 pre-date the family declaration (item 2).
3. checkpoint-87 = epoch 3; the strict earliest-plateau rule gives epoch 1 for the
   3B (item 3).
4. Legacy arms keep argmax epoch selection, e.g. minilm_probe epoch 100 vs plateau
   80 (item 4).
5. SALSA-full letter relabel D..Q, decided on train_v2 before any held-out run
   (item 5).
6. SALSA base arms use stock HF weights; cache revisions recorded (item 6).
7. GRPO dev curves flat after epoch 1; RL-on-SFT signal check 30/230 (13.0%)
   (item 7).
8. Ensemble: vote over 5 predictions, command 0 included, outcome-gated
   reproducibility (item 8).
9. Cloud: raw agreement reported not gated; per-row +/-3-point accuracy gate on
   v3; retries with backoff (item 9); thesis numbers use majority vote, V5 used
   per-row averages (item 18).
10. Energy formula and protocol; GPU only; CPU arms as CPU ms; cloud not
    measurable (item 10). The energy Pareto frontier is GPU-only (`RUN_ORDER.md`).
11. Efficiency scope: main table v3; v4c supplement for the speculative arms only;
    V5-cell energy on v3 only (item 11).
12. Ollama Q4_K_M vs transformers FP16 runtime difference (item 12).
13. 12,152 rows = cells x 217 x repeats; effective n = 217 (item 13).
14. MuJoCo end-to-end not re-run: 0 execution failures earlier, so success =
    string match (item 14).
15. qwen2.5-0.5b-lora/cot runaway generation stopped at ~10 x num_ctx tokens by the
    Ollama server (item 19).

## 8. Bugs caught and fixed during the final run (`METHODS_NOTES.md` item 15)

- SALSA letter collision (option letters A/B/C inside `grid_A1`) -> relabel D..Q.
- 12 void few_shot/rag V5 cells: exemplar-loader bug, `prompt_error` with no model
  call -> V5's own loader, re-run with `--replace-cells`.
- Reproducibility check compared 0/0 and passed (int/str join key) -> fixed; zero
  comparisons now FAIL.
- Completeness counted lines, not CSV rows (986 "rows" for 217) -> parsed rows.
- Environment: S1-4 cloud step first run under `thesis2gpu` -> `ModuleNotFoundError:
  ollama` (wrong env; `all_clients()` imports ollama even for cloud cells), then
  under `thesis` (logs cited in item 15; the error text is from the console, as
  reported by the author).
- Connection drops: the second run's failed gpt-4o-mini rows were
  `APIConnectionError` (console, as reported) -> 4 cloud cells re-run with
  `--replace-cells`; retry with backoff added.
- Ensemble refused to run -> rebuilt as an offline pairing.
- V5-cells audit compared v4c with V5's v3 numbers -> comparisons v3-only.
- HF TLS failures -> offline model loading for the energy stage.
- Speculative arms refused by `s3_energy.py` -> shared protocol inside their
  scripts after the byte-identity gate (A10 9.45 -> 9.54 J).
- Smoke test: `v5_ollama` needs the `thesis` env; GRPO parsers were missing from
  `GPU_PARSERS` (VRAM unmeasured) -> fixed 2026-09-28.

## 9. Files that are the source of truth

| file | what |
|---|---|
| `thesis2/out/final/metrics_final.csv` | **the only source for thesis tables**: every metric, every arm, both splits |
| `thesis2/out/final/final_significance.csv` | H1-H8 and the exploratory family |
| `thesis2/out/final/unified_efficiency.csv` | every latency / energy / VRAM measurement (85 rows) |
| `thesis2/out/final/v5cells_both_splits.csv` | the 36 V5 cells on both splits, with descriptive CIs |
| `thesis2/out/final/pred_<arm>_<split>.csv` | per-command predictions (Stage 1) |
| `thesis2/out/final/results_v5cells_<split>.csv` | V5 cell grid rows (12,152 per split) |
| `thesis2/out/final/repro_check_<split>.csv` | reproducibility against the earlier run |
| `thesis2/out/final/DEFENCE_NUMBERS_rev3.md` | quotable numbers, each cited; checked by `check_defence_numbers.py` |
| `thesis2/out/final/METHODS_NOTES.md` | decisions, deviations, bugs, with evidence |
| `thesis2/out/final/FINAL_RUN_REPORT.md` | the assembled report (`s6_report.py`) |
| `thesis2/final/RUN_ORDER.md` | every command, status table, decisions by date |
| `thesis2/figures/final/fig1..fig6` | figures (`s4_figures.py`) |
| `thesis2/out/FINAL_REPORT.md` | **historical record** of the earlier run; not a table source |

Rebuild chain (no model, no API): `s2_metrics.py` -> `s4_metrics_final.py` ->
`s4_figures.py` -> `s6_report.py` -> `make_defence_numbers.py` ->
`check_defence_numbers.py`. Do not re-run `s4_significance.py` (the tests are
fixed). [Certain]

## 10. Open tasks

- **Stage 5 Isaac demo**: only the parser smoke test (S5-1) is done -- 18/18 local
  parsers valid JSON on the 10 seen commands (`demo_smoke_parsers.json`). Isaac
  VRAM (S5-2), the live demo (S5-3) and the held-out replay (S5-4) are not done.
- **Slides**: the deck is not in the repo; apply `SLIDE_FIXES.md` by hand (slide 22
  -> 0.2-0.4 cm `pick_place`; slide 17's n = 539 = rows, 108 commands).
- **DCCD**: arXiv:2603.03305 is a [Likely] match; confirm it is the intended
  source before citing (`METHODS_NOTES.md` item 20).
- **Ollama generation cap**: the ~10 x num_ctx stop is inferred from token counts;
  it is set in the Ollama server, not in this repo (`METHODS_NOTES.md` item 19).
- **The 3B checkpoint** is not what the stated earliest-plateau rule selects; say
  so in the thesis (item 3).
- **`sim/isaac_backend.py:49`** still says 0.1-0.5 cm in a comment (`sim/` is
  read-only; not changed).
- Commit the working tree: the final fix pass of 2026-09-28 is uncommitted.
