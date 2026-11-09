# Where each thesis result comes from

Thesis figures and tables (numbering of the thesis version of 3 Nov 2026) and the files their
data come from. Files under `thesis3/out/c1/` come from the release file `thesis3_c1.zip`.

| thesis | content | data |
|---|---|---|
| Fig 6.1 | ladder of the main steps | `thesis2/out/final/metrics_final.csv`; `thesis2/out/final/FINAL_RUN_REPORT.md` §4, §4b |
| Table 6.1 | accuracy of all final-run systems | `thesis2/out/final/metrics_final.csv` |
| Fig 6.2 | V5 cells, both test sets | `thesis2/out/final/FINAL_RUN_REPORT.md` §4b |
| Fig 6.3 | held_out_v3 vs held_out_v4c per system | `thesis2/out/final/metrics_final.csv`; `thesis2/out/final/FINAL_RUN_REPORT.md` §4b |
| Fig 6.4 | predicted row per row word, six systems | `thesis2/out/final/metrics_final.csv` |
| Table 6.2 | primary hypotheses | `thesis2/out/final/final_significance.csv` |
| Fig 6.5 | pairwise significance | `thesis2/out/final/metrics_final.csv`; `thesis2/out/final/final_significance.csv` |
| Table 6.3 | latency, energy, memory on held_out_v3 | `thesis2/out/final/metrics_final.csv` |
| Fig 6.6 | accuracy vs latency (top) and energy (bottom) | `thesis2/out/final/metrics_final.csv`; `thesis2/out/final/FINAL_RUN_REPORT.md` §4, §4b |
| Fig 6.7 | speculative decoding | `thesis2/out/final/metrics_final.csv` |
| Fig 6.8 | speculative internals | first-rejection field on held_out_v3 (162 / 41 / 13 / 1 / 0): `thesis2/out/a11_rejection_positions.csv`; draft gains vs harms (49 / 8 vs 4 / 28): `thesis2/out/final/draft_verifier_gh.csv` |
| Table 6.4 | row accuracy, seen vs unseen row words | `thesis3/out/c1/summary.csv` |
| Fig 6.9 | by verb and by location class | `thesis3/out/c1/per_verb.csv`, `thesis3/out/c1/per_loc_type.csv` |
| Fig 6.10, 6.11, 6.12 | latency, energy, tokens | `thesis3/out/c1/summary.csv` |
| Table 6.5, 6.6 | power, safety channel | `thesis3/out/c1/summary.csv` |
| Fig 7.1 | same LoRA in Ollama and transformers | `audit/figures/cell_stats.csv`; `thesis2/out/final/FINAL_RUN_REPORT.md` §4b |
| Fig A.1 | V3/V4/V5 campaigns | `audit/figures/cell_stats.csv`; `audit/v4_cells.csv`; `thesis2/out/final/FINAL_RUN_REPORT.md` §4b |
| Table B.1 | all V5 cells | `thesis2/out/final/FINAL_RUN_REPORT.md` §4b |
| Fig C.1, Table C.1 | final run vs re-measurement; all 66 systems | `thesis2/out/final/metrics_final.csv`; `thesis3/out/c1/summary.csv` |
| Table D.1 | frozen checkpoints | the ten frozen arms: `thesis2/out/final/frozen_checkpoints.csv`; DistilBERT flat (epoch 9) and the LoRA adapters (checkpoint-87): release files `distilbert_cls_epoch_09.zip`, `qwen3b_lora_checkpoint-87.zip`, `qwen0.5b_lora_checkpoint-87.zip` |

The thesis figures were redrawn from these files by the plotting scripts in the thesis LaTeX
sources; `thesis2/figures/final/` holds the repository's own versions. The background on A8 and
A12 is in `thesis2/out/FINAL_REPORT.md` §7.1-7.2.

## Which script writes which file

| result | produced by | file |
|---|---|---|
| accuracy, per-field accuracy, ECE, latency and energy of the thesis2 run, both test sets | `thesis2/final/s4_metrics_final.py` | `thesis2/out/final/metrics_final.csv` |
| V5 cells (36 model x strategy cells), both test sets | `thesis2/final/s2_metrics.py` | `thesis2/out/final/v5cells_both_splits.csv` |
| hypotheses H1-H8 and the exploratory family (bootstrap CIs, McNemar, Holm) | `thesis2/final/s4_significance.py` | `thesis2/out/final/final_significance.csv` |
| figures 1-6 (accuracy vs latency, energy, VRAM; row accuracy by word group; row confusion; A10 breakdown) | `thesis2/final/s4_figures.py` | `thesis2/figures/final/fig*.png` |
| row confusion by row word; draft vs verifier gain/harm | `thesis2/final/s5_extra_analysis.py` | `thesis2/out/final/row_confusion_by_word.csv`, `draft_verifier_gh.csv` |
| every number quotable in the defence, each cited to its source cell | `thesis2/final/make_defence_numbers.py` (checked by `check_defence_numbers.py`) | `thesis2/out/final/DEFENCE_NUMBERS_rev3.md` |
| unified bench: accuracy, latency, energy, CO2, cost per million commands, all 66 systems | `thesis3/bench/analyze.py` | `thesis3/out/c1/summary.csv`, `report.md` |
| unified bench: paired comparisons, per verb, per location type, measurement quality | `thesis3/bench/analyze.py` | `thesis3/out/c1/compare.csv`, `per_verb.csv`, `per_loc_type.csv`, `quality.csv` |
| unified bench vs thesis2 accuracy | `thesis3/bench/run_bench.py` | `thesis3/out/c1/repro_check.csv` |

## How the data were used

- `train_v2.jsonl` (230 commands) is the only training and retrieval set; `dev_v2.jsonl` (57)
  is used only to pick checkpoints and epochs.
- `held_out_v3.jsonl` and `held_out_v4c.jsonl` (217 commands each) are used only for testing.
- `held_out_v4c` uses the row words `back` and `front`, which appear in no training data. It is
  the fair generalisation test (`thesis2/out/final/METHODS_NOTES.md`, item 1).

## Notes on the measurements

- Latency and energy were measured on one Windows PC (AMD Ryzen 5 7600X, RTX 4060 Ti 16 GB).
  Energy covers GPU board and CPU package power.
- The V5 LoRA cells run in Ollama (Q4_K_M); `lora_greedy` and the speculative systems run the
  same adapter in transformers (FP16) (METHODS_NOTES item 12).
- gpt-4o-mini answers can differ between calls; its five repeats per command are combined by
  majority vote.
- Some logs and result files contain paths of the measurement machine (`C:\Users\USER\...`,
  `D:\condaenvs\...`, `D:\LLM`); they are kept as recorded.
