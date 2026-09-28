# Final run report

Assembled 2026-09-28 17:55 by `thesis2/final/s6_report.py` from the output files in `thesis2/out/final/`. Nothing here is recomputed and nothing is tagged novel.

Effective n is **217 commands per split**. The V5 cell grid is 12,152 rows over those 217 commands; repeats collapse to one outcome per command by majority vote, as A9 did. This run is never 12,000 commands.

## 1. What ran, what was skipped

| arm | split | kind | status | reason |
|---|---|---|---|---|
| a10_speculative | held_out_v3 | local | ran |  |
| a10_speculative | held_out_v4c | local | ran |  |
| a12_smalllm | held_out_v3 | local | ran |  |
| a12_smalllm | held_out_v4c | local | ran |  |
| a8_speculative | held_out_v3 | local | ran |  |
| a8_speculative | held_out_v4c | local | ran |  |
| bilstm | held_out_v3 | local | ran |  |
| bilstm | held_out_v4c | local | ran |  |
| cascade_cloud_always | held_out_v3 | cloud | ran |  |
| cascade_cloud_always | held_out_v4c | cloud | ran |  |
| cascade_cloud_tauA | held_out_v3 | cloud | ran |  |
| cascade_cloud_tauA | held_out_v4c | cloud | ran |  |
| cascade_local_tauA | held_out_v3 | local | ran |  |
| cascade_local_tauA | held_out_v4c | local | ran |  |
| cascade_oracle | held_out_v3 | local | ran |  |
| cascade_oracle | held_out_v4c | local | ran |  |
| ensemble | held_out_v3 | local | ran |  |
| ensemble | held_out_v4c | local | ran |  |
| factorized_fp16 | held_out_v3 | local | ran |  |
| factorized_fp16 | held_out_v4c | local | ran |  |
| factorized_int4 | held_out_v3 | local | ran |  |
| factorized_int4 | held_out_v4c | local | ran |  |
| flant5 | held_out_v3 | local | ran |  |
| flant5 | held_out_v4c | local | ran |  |
| flat_cpu | held_out_v3 | local | ran |  |
| flat_cpu | held_out_v4c | local | ran |  |
| flat_gpu | held_out_v3 | local | ran |  |
| flat_gpu | held_out_v4c | local | ran |  |
| grpo_base | held_out_v3 | local | ran |  |
| grpo_base | held_out_v4c | local | ran |  |
| grpo_sft | held_out_v3 | local | ran |  |
| grpo_sft | held_out_v4c | local | ran |  |
| knn_minilm | held_out_v3 | local | ran |  |
| knn_minilm | held_out_v4c | local | ran |  |
| lora_greedy | held_out_v3 | local | ran |  |
| lora_greedy | held_out_v4c | local | ran |  |
| minilm_probe | held_out_v3 | local | ran |  |
| minilm_probe | held_out_v4c | local | ran |  |
| mlpbow | held_out_v3 | local | ran |  |
| mlpbow | held_out_v4c | local | ran |  |
| pipeline_always_local | held_out_v3 | local | ran |  |
| pipeline_always_local | held_out_v4c | local | ran |  |
| rules_firewalled | held_out_v3 | local | ran |  |
| rules_firewalled | held_out_v4c | local | ran |  |
| rules_open | held_out_v3 | local | ran |  |
| rules_open | held_out_v4c | local | ran |  |
| salsa_full_Qwen2.5-0.5B-Instruct | held_out_v3 | local | ran |  |
| salsa_full_Qwen2.5-0.5B-Instruct | held_out_v4c | local | ran |  |
| salsa_full_Qwen2.5-3B-Instruct | held_out_v3 | local | ran |  |
| salsa_full_Qwen2.5-3B-Instruct | held_out_v4c | local | ran |  |
| salsa_full_qwen2.5-0.5b-lora | held_out_v3 | local | ran |  |
| salsa_full_qwen2.5-0.5b-lora | held_out_v4c | local | ran |  |
| salsa_full_qwen2.5-3b-lora | held_out_v3 | local | ran |  |
| salsa_full_qwen2.5-3b-lora | held_out_v4c | local | ran |  |
| textcnn | held_out_v3 | local | ran |  |
| textcnn | held_out_v4c | local | ran |  |
| tfidf_logreg | held_out_v3 | local | ran |  |
| tfidf_logreg | held_out_v4c | local | ran |  |
| tinybert | held_out_v3 | local | ran |  |
| tinybert | held_out_v4c | local | ran |  |


### Frozen checkpoints (new arms)

| arm | selected epoch | dev_v2 action match | sha256 |
|---|---|---|---|
| bilstm | 6 | 0.9473684210526315 | 79217a8587d6767cdd23e85d7b2371658b7c34b8fd68f78830fa70de5ab3b3c4 |
| flant5 | 5 | 0.9298245614035088 | 3de022196f12a9b512c6f9e37263833a664fac6d82dd8941f5fe416c273e698c |
| grpo_base | 1 | 0.8421052631578947 | 86acbd3da67a78ac15a69e7051fd10e28d0efc2859201b1e46a9e5b02fe88819 |
| grpo_sft | 1 | 0.9824561403508771 | d3bda156209dea17cc0fc1de51742fd8382252a03b4cfcb8d6ab2db2dd499aa7 |
| knn_minilm | n/a: no epochs | 0.43859649122807015 | 017186d7c139353e2bc474f2d8b416de9197e9df1e400ee0f16f79438714afd0 |
| minilm_probe | 100 | 0.6842105263157895 | a3739413cf5f1a2b7c6a668485494dc0297d0b8e1a8028ec0cbff7e447158cfa |
| mlpbow | 25 | 0.9298 | 1b81a2cacfac1e62dfb0becce7ec372869ee0d6966029e8348115a32a47060b5 |
| salsa_full_Qwen2.5-0.5B-Instruct | n/a: no training | 0.1228 | n/a: base weights only |
| salsa_full_Qwen2.5-3B-Instruct | n/a: no training | 0.4211 | n/a: base weights only |
| salsa_full_qwen2.5-0.5b-lora | n/a: no training | 0.0526 | f05df4d52542dd3289b314831e86d46000459665c7d0de6d2bfac36a8c9b2d24 |
| salsa_full_qwen2.5-3b-lora | n/a: no training | 0.5614 | 5b409e547527340b1bae5de046667ee5beb8b80cac11983f04193a4b299c37e3 |
| textcnn | 18 | 0.9474 | 2bb7ffa5af495a0578a9b1c4f9691238be5bf40bdf11e74b9c330518513fabe3 |
| tfidf_logreg | n/a: no epochs | 0.8245614035087719 | 22d06f3472d63ce3694f387fa90a70e59a6fe0080b9261a6e3a99a1d17feebc8 |
| tinybert | 21 | 0.9473684210526315 | b8e9a679966326f5c2dd0ac42b4fe5be92df09189268eac6e57f00e2a5a9e46f |


### --replace-cells re-runs on held_out_v3 (12 void cells + 4 connection-failure cells)

Every row of `force_rerun_log.csv`. The 12 local cells were voided by the exemplar-loader bug (`prompt_error`, no model call); the 4 gpt-4o-mini cells lost rows to connection drops before the retry layer existed. held_out_v4c was never re-run this way. [Certain]

| utc | arm | split | rows_discarded |
|---|---|---|---|
| 2026-09-26T10:37:57Z | v5cell:ollama:deepseek-r1:7b/few_shot | held_out_v3 | 217 |
| 2026-09-26T10:37:57Z | v5cell:ollama:deepseek-r1:7b/rag | held_out_v3 | 217 |
| 2026-09-26T10:37:57Z | v5cell:ollama:llama3.2:3b/few_shot | held_out_v3 | 217 |
| 2026-09-26T10:37:57Z | v5cell:ollama:llama3.2:3b/rag | held_out_v3 | 217 |
| 2026-09-26T10:37:57Z | v5cell:ollama:qwen2.5-0.5b-lora/few_shot | held_out_v3 | 217 |
| 2026-09-26T10:37:57Z | v5cell:ollama:qwen2.5-0.5b-lora/rag | held_out_v3 | 217 |
| 2026-09-26T10:37:57Z | v5cell:ollama:qwen2.5-3b-lora/few_shot | held_out_v3 | 217 |
| 2026-09-26T10:37:57Z | v5cell:ollama:qwen2.5-3b-lora/rag | held_out_v3 | 217 |
| 2026-09-26T10:37:57Z | v5cell:ollama:qwen2.5:0.5b/few_shot | held_out_v3 | 217 |
| 2026-09-26T10:37:57Z | v5cell:ollama:qwen2.5:0.5b/rag | held_out_v3 | 217 |
| 2026-09-26T10:37:57Z | v5cell:ollama:qwen2.5:3b/few_shot | held_out_v3 | 217 |
| 2026-09-26T10:37:58Z | v5cell:ollama:qwen2.5:3b/rag | held_out_v3 | 217 |
| 2026-09-26T16:53:56Z | v5cell:openai:gpt-4o-mini/cot | held_out_v3 | 1085 |
| 2026-09-26T16:53:56Z | v5cell:openai:gpt-4o-mini/few_shot | held_out_v3 | 1085 |
| 2026-09-26T16:53:57Z | v5cell:openai:gpt-4o-mini/role_based | held_out_v3 | 1085 |
| 2026-09-26T16:53:57Z | v5cell:openai:gpt-4o-mini/zero_shot | held_out_v3 | 1085 |


## 2. Accuracy on both splits, with the diagnosis metrics


### held_out_v3

| arm | n | action | row | row by word group | row signature | ECE |
|---|---|---|---|---|---|---|
| ensemble | 217 | 0.9631 | 1.0 | mixed 24/24; seen 36/36; unseen 48/48 | A<->C mirror 0; default-to-B 0 | n/a - no class probabilities |
| grpo_sft | 217 | 0.9585 | 1.0 | mixed 24/24; seen 36/36; unseen 48/48 | A<->C mirror 0; default-to-B 0 | n/a - no class probabilities |
| a10_speculative | 217 | 0.9539 | 1.0 | mixed 24/24; seen 36/36; unseen 48/48 | A<->C mirror 0; default-to-B 0 | n/a - no class probabilities |
| a12_smalllm | 217 | 0.9539 | 1.0 | mixed 24/24; seen 36/36; unseen 48/48 | A<->C mirror 0; default-to-B 0 | n/a - no class probabilities |
| a8_speculative | 217 | 0.9539 | 1.0 | mixed 24/24; seen 36/36; unseen 48/48 | A<->C mirror 0; default-to-B 0 | n/a - no class probabilities |
| lora_greedy | 217 | 0.9539 | 1.0 | mixed 24/24; seen 36/36; unseen 48/48 | A<->C mirror 0; default-to-B 0 | n/a - no class probabilities |
| cascade_oracle | 217 | 0.9401 | 1.0 | mixed 24/24; seen 36/36; unseen 48/48 | A<->C mirror 0; default-to-B 0 | 0.5253 |
| pipeline_always_local | 217 | 0.8433 | 0.8056 | mixed 24/24; seen 15/36; unseen 48/48 | A<->C mirror 0; default-to-B 0 | 0.4285 |
| cascade_cloud_always | 217 | 0.7926 | 0.7037 | mixed 1/24; seen 27/36; unseen 48/48 | A<->C mirror 0; default-to-B 23 | 0.3778 |
| grpo_base | 217 | 0.7926 | 0.7222 | mixed 7/24; seen 36/36; unseen 35/48 | A<->C mirror 13; default-to-B 17 | n/a - no class probabilities |
| flat_cpu | 217 | 0.7465 | 0.6204 | mixed 8/24; seen 36/36; unseen 23/48 | A<->C mirror 34; default-to-B 7 | 0.2651 |
| flat_gpu | 217 | 0.7465 | 0.6204 | mixed 8/24; seen 36/36; unseen 23/48 | A<->C mirror 34; default-to-B 7 | 0.2651 |
| cascade_cloud_tauA | 217 | 0.7327 | 0.5833 | mixed 0/24; seen 36/36; unseen 27/48 | A<->C mirror 0; default-to-B 45 | 0.3179 |
| cascade_local_tauA | 217 | 0.7327 | 0.5833 | mixed 0/24; seen 36/36; unseen 27/48 | A<->C mirror 0; default-to-B 45 | 0.3179 |
| rules_open | 217 | 0.7235 | 0.8889 | mixed 12/24; seen 36/36; unseen 48/48 | A<->C mirror 0; default-to-B 12 | n/a - no class probabilities |
| bilstm | 217 | 0.7005 | 0.6852 | mixed 2/24; seen 36/36; unseen 36/48 | A<->C mirror 14; default-to-B 20 | 0.2045 |
| textcnn | 217 | 0.6959 | 0.6759 | mixed 1/24; seen 36/36; unseen 36/48 | A<->C mirror 13; default-to-B 22 | 0.0777 |
| minilm_probe | 217 | 0.6498 | 0.6204 | mixed 9/24; seen 34/36; unseen 24/48 | A<->C mirror 30; default-to-B 9 | 0.2627 |
| knn_minilm | 217 | 0.6452 | 0.5648 | mixed 6/24; seen 34/36; unseen 21/48 | A<->C mirror 34; default-to-B 11 | n/a - no class probabilities |
| mlpbow | 217 | 0.6406 | 0.6667 | mixed 9/24; seen 33/36; unseen 30/48 | A<->C mirror 27; default-to-B 6 | 0.2702 |
| tfidf_logreg | 217 | 0.6221 | 0.75 | mixed 9/24; seen 36/36; unseen 36/48 | A<->C mirror 21; default-to-B 6 | 0.3363 |
| flant5 | 217 | 0.6175 | 0.4259 | mixed 0/24; seen 36/36; unseen 10/48 | A<->C mirror 0; default-to-B 62 | n/a - no class probabilities |
| factorized_fp16 | 217 | 0.6083 | 0.3333 | mixed 0/24; seen 36/36; unseen 0/48 | A<->C mirror 6; default-to-B 66 | 0.1935 |
| factorized_int4 | 217 | 0.6083 | 0.3426 | mixed 0/24; seen 36/36; unseen 1/48 | A<->C mirror 6; default-to-B 65 | 0.2009 |
| tinybert | 217 | 0.576 | 0.3333 | mixed 0/24; seen 36/36; unseen 0/48 | A<->C mirror 0; default-to-B 72 | 0.1711 |
| salsa_full_qwen2.5-3b-lora | 217 | 0.5023 | 0.6574 | mixed 24/24; seen 3/36; unseen 44/48 | A<->C mirror 2; default-to-B 2 | 0.105 |
| rules_firewalled | 217 | 0.447 | 0.3333 | mixed 0/24; seen 36/36; unseen 0/48 | A<->C mirror 0; default-to-B 72 | n/a - no class probabilities |
| salsa_full_Qwen2.5-3B-Instruct | 217 | 0.3594 | 0.2963 | mixed 9/24; seen 1/36; unseen 22/48 | A<->C mirror 19; default-to-B 13 | 0.4465 |
| salsa_full_Qwen2.5-0.5B-Instruct | 217 | 0.0461 | 0.3056 | mixed 6/24; seen 16/36; unseen 11/48 | A<->C mirror 0; default-to-B 41 | 0.336 |
| salsa_full_qwen2.5-0.5b-lora | 217 | 0.0369 | 0.0093 | mixed 0/24; seen 1/36; unseen 0/48 | A<->C mirror 0; default-to-B 36 | 0.2181 |


### held_out_v4c

| arm | n | action | row | row by word group | row signature | ECE |
|---|---|---|---|---|---|---|
| cascade_oracle | 217 | 0.9401 | 1.0 | mixed 24/24; seen 36/36; unseen 48/48 | A<->C mirror 0; default-to-B 0 | 0.5248 |
| ensemble | 217 | 0.8387 | 0.75 | mixed 12/24; seen 36/36; unseen 33/48 | A<->C mirror 9; default-to-B 18 | n/a - no class probabilities |
| cascade_cloud_always | 217 | 0.7788 | 0.6759 | mixed 0/24; seen 27/36; unseen 46/48 | A<->C mirror 2; default-to-B 24 | 0.3636 |
| cascade_cloud_tauA | 217 | 0.7696 | 0.6574 | mixed 0/24; seen 36/36; unseen 35/48 | A<->C mirror 4; default-to-B 33 | 0.3543 |
| rules_open | 217 | 0.7235 | 0.8889 | mixed 12/24; seen 36/36; unseen 48/48 | A<->C mirror 0; default-to-B 12 | n/a - no class probabilities |
| flat_cpu | 217 | 0.7143 | 0.5556 | mixed 0/24; seen 36/36; unseen 24/48 | A<->C mirror 24; default-to-B 24 | 0.256 |
| flat_gpu | 217 | 0.7143 | 0.5556 | mixed 0/24; seen 36/36; unseen 24/48 | A<->C mirror 24; default-to-B 24 | 0.256 |
| bilstm | 217 | 0.7005 | 0.6852 | mixed 2/24; seen 36/36; unseen 36/48 | A<->C mirror 14; default-to-B 20 | 0.2045 |
| grpo_sft | 217 | 0.7005 | 0.4815 | mixed 4/24; seen 36/36; unseen 12/48 | A<->C mirror 2; default-to-B 54 | n/a - no class probabilities |
| grpo_base | 217 | 0.6959 | 0.5463 | mixed 0/24; seen 36/36; unseen 23/48 | A<->C mirror 24; default-to-B 25 | n/a - no class probabilities |
| knn_minilm | 217 | 0.6959 | 0.6852 | mixed 8/24; seen 34/36; unseen 32/48 | A<->C mirror 23; default-to-B 9 | n/a - no class probabilities |
| textcnn | 217 | 0.6959 | 0.6759 | mixed 1/24; seen 36/36; unseen 36/48 | A<->C mirror 13; default-to-B 22 | 0.0777 |
| minilm_probe | 217 | 0.6544 | 0.6574 | mixed 9/24; seen 34/36; unseen 28/48 | A<->C mirror 28; default-to-B 7 | 0.261 |
| cascade_local_tauA | 217 | 0.6452 | 0.4074 | mixed 0/24; seen 36/36; unseen 8/48 | A<->C mirror 31; default-to-B 33 | 0.2299 |
| mlpbow | 217 | 0.6406 | 0.6667 | mixed 9/24; seen 33/36; unseen 30/48 | A<->C mirror 27; default-to-B 6 | 0.2702 |
| factorized_fp16 | 217 | 0.6359 | 0.3889 | mixed 0/24; seen 36/36; unseen 6/48 | A<->C mirror 20; default-to-B 46 | 0.2207 |
| a10_speculative | 217 | 0.6221 | 0.3333 | mixed 0/24; seen 36/36; unseen 0/48 | A<->C mirror 6; default-to-B 66 | n/a - no class probabilities |
| a12_smalllm | 217 | 0.6221 | 0.3333 | mixed 0/24; seen 36/36; unseen 0/48 | A<->C mirror 6; default-to-B 66 | n/a - no class probabilities |
| a8_speculative | 217 | 0.6221 | 0.3333 | mixed 0/24; seen 36/36; unseen 0/48 | A<->C mirror 6; default-to-B 66 | n/a - no class probabilities |
| lora_greedy | 217 | 0.6221 | 0.3333 | mixed 0/24; seen 36/36; unseen 0/48 | A<->C mirror 6; default-to-B 66 | n/a - no class probabilities |
| tfidf_logreg | 217 | 0.6221 | 0.75 | mixed 9/24; seen 36/36; unseen 36/48 | A<->C mirror 21; default-to-B 6 | 0.3363 |
| factorized_int4 | 217 | 0.6175 | 0.3611 | mixed 0/24; seen 36/36; unseen 3/48 | A<->C mirror 12; default-to-B 57 | 0.2108 |
| tinybert | 217 | 0.5806 | 0.3426 | mixed 0/24; seen 36/36; unseen 1/48 | A<->C mirror 0; default-to-B 71 | 0.1432 |
| flant5 | 217 | 0.5714 | 0.3333 | mixed 0/24; seen 36/36; unseen 0/48 | A<->C mirror 0; default-to-B 72 | n/a - no class probabilities |
| pipeline_always_local | 217 | 0.5438 | 0.2037 | mixed 0/24; seen 15/36; unseen 7/48 | A<->C mirror 42; default-to-B 23 | 0.1445 |
| rules_firewalled | 217 | 0.447 | 0.3333 | mixed 0/24; seen 36/36; unseen 0/48 | A<->C mirror 0; default-to-B 72 | n/a - no class probabilities |
| salsa_full_qwen2.5-3b-lora | 217 | 0.4055 | 0.3333 | mixed 12/24; seen 3/36; unseen 21/48 | A<->C mirror 35; default-to-B 4 | 0.105 |
| salsa_full_Qwen2.5-3B-Instruct | 217 | 0.3502 | 0.1759 | mixed 7/24; seen 1/36; unseen 11/48 | A<->C mirror 32; default-to-B 7 | 0.4604 |
| salsa_full_Qwen2.5-0.5B-Instruct | 217 | 0.0599 | 0.2778 | mixed 1/24; seen 16/36; unseen 13/48 | A<->C mirror 22; default-to-B 33 | 0.336 |
| salsa_full_qwen2.5-0.5b-lora | 217 | 0.0369 | 0.0093 | mixed 0/24; seen 1/36; unseen 0/48 | A<->C mirror 0; default-to-B 21 | 0.2198 |


`row by word group`: seen = middle/center (both splits); unseen = upper/bottom on v3, back/front on v4c; mixed = a command containing both. A prompted arm's v3 number is measured on wording the V5 legend defines; its v4c number is not. [Certain]


## 3. Reproducibility against the earlier run

From `thesis2/final/s1_repro_check.py` (`repro_check_<split>.csv`), one line per registry arm. The join key is `str(command_id)`; comparing zero commands is a FAIL; totals-only arms are compared against their own split's saved total. [Certain]


### held_out_v3

- `a10_speculative` **PASS** — 217/217 compared, 217 identical; all identical

- `a12_smalllm` **PASS** — 217/217 compared, 217 identical; all identical

- `a8_speculative` **PASS** — 217/217 compared, 217 identical; all identical

- `bilstm` **PASS** — 217/217 compared, 217 identical; all identical

- `cascade_cloud_always` **PASS** — 217/217 compared, 217 identical; all identical

- `cascade_cloud_tauA` **PASS** — 217/217 compared, 217 identical; all identical

- `cascade_local_tauA` **PASS** — 217/217 compared, 217 identical; all identical

- `cascade_oracle` **PASS** — 217/217 compared, 217 identical; all identical

- `ensemble` **PASS** — 216/217 compared, 215 identical; (differing command_ids ['132']: reported only -- its gpt-4o-mini/rag input is sampled (R = 5, majority vote); per-command outcome identical on 216/216); not in the saved file by design: 0 (V5's warm-up row, dropped by the original loader)

- `factorized_fp16` **PASS** — 217/217 compared, 217 identical; all identical

- `factorized_int4` **PASS** — totals only: ours 0.6083 vs saved 0.6083 (quantization_gate.csv (Phase 3 aggregate))

- `flant5` **PASS** — 217/217 compared, 217 identical; all identical

- `flat_cpu` **PASS** — 217/217 compared, 217 identical; all identical

- `flat_gpu` **PASS** — 217/217 compared, 217 identical; all identical

- `grpo_base` not comparable — NEW arm -- first measurement

- `grpo_sft` not comparable — NEW arm -- first measurement

- `knn_minilm` **PASS** — 217/217 compared, 217 identical; all identical

- `lora_greedy` **PASS** — 217/217 compared, 217 identical; all identical

- `minilm_probe` **PASS** — 217/217 compared, 217 identical; all identical

- `mlpbow` not comparable — NEW arm -- first measurement

- `pipeline_always_local` **PASS** — 217/217 compared, 217 identical; all identical

- `rules_firewalled` **PASS** — totals only: ours 0.4470 vs saved 0.4470 (baseline_rules_firewall.csv, held_out_v3 row)

- `rules_open` **PASS** — 217/217 compared, 217 identical; all identical

- `salsa_full_Qwen2.5-0.5B-Instruct` not comparable — NEW arm -- first measurement

- `salsa_full_Qwen2.5-3B-Instruct` not comparable — NEW arm -- first measurement

- `salsa_full_qwen2.5-0.5b-lora` not comparable — NEW arm -- first measurement

- `salsa_full_qwen2.5-3b-lora` not comparable — NEW arm -- first measurement

- `textcnn` not comparable — NEW arm -- first measurement

- `tfidf_logreg` **PASS** — 217/217 compared, 217 identical; all identical

- `tinybert` **PASS** — 217/217 compared, 217 identical; all identical


### held_out_v4c

- `a10_speculative` not comparable — the earlier run made no held_out_v4c speculative run; its per-command files are held_out_v3 only

- `a12_smalllm` not comparable — the earlier run made no held_out_v4c speculative run; its per-command files are held_out_v3 only

- `a8_speculative` not comparable — the earlier run made no held_out_v4c speculative run; its per-command files are held_out_v3 only

- `bilstm` **PASS** — 217/217 compared, 217 identical; all identical

- `cascade_cloud_always` **PASS** — 217/217 compared, 217 identical; all identical

- `cascade_cloud_tauA` **PASS** — 217/217 compared, 217 identical; all identical

- `cascade_local_tauA` **PASS** — 217/217 compared, 217 identical; all identical

- `cascade_oracle` **PASS** — 217/217 compared, 217 identical; all identical

- `ensemble` not comparable — a1_ensemble_held_out_v3.csv exists for v3 only; the earlier run built no held_out_v4c ensemble

- `factorized_fp16` **PASS** — 217/217 compared, 217 identical; all identical

- `factorized_int4` not comparable — the earlier run saved no held_out_v4c aggregate for INT4 (quantization_gate.csv is held_out_v3 only)

- `flant5` **PASS** — 217/217 compared, 217 identical; all identical

- `flat_cpu` **PASS** — 217/217 compared, 217 identical; all identical

- `flat_gpu` **PASS** — 217/217 compared, 217 identical; all identical

- `grpo_base` not comparable — NEW arm -- first measurement

- `grpo_sft` not comparable — NEW arm -- first measurement

- `knn_minilm` not comparable — baseline_knn_minilm.csv exists for v3 only

- `lora_greedy` **PASS** — 217/217 compared, 217 identical; all identical

- `minilm_probe` **PASS** — 217/217 compared, 217 identical; all identical

- `mlpbow` not comparable — NEW arm -- first measurement

- `pipeline_always_local` **PASS** — 217/217 compared, 217 identical; all identical

- `rules_firewalled` **PASS** — totals only: ours 0.4470 vs saved 0.4470 (baseline_rules_firewall.csv, held_out_v4c row)

- `rules_open` not comparable — baseline_rules.csv exists for v3 only

- `salsa_full_Qwen2.5-0.5B-Instruct` not comparable — NEW arm -- first measurement

- `salsa_full_Qwen2.5-3B-Instruct` not comparable — NEW arm -- first measurement

- `salsa_full_qwen2.5-0.5b-lora` not comparable — NEW arm -- first measurement

- `salsa_full_qwen2.5-3b-lora` not comparable — NEW arm -- first measurement

- `textcnn` not comparable — NEW arm -- first measurement

- `tfidf_logreg` not comparable — baseline_tfidf_logreg.csv exists for v3 only

- `tinybert` **PASS** — 217/217 compared, 217 identical; all identical


## 4. Unified efficiency (measured, not modelled)

| arm | split | N | wall s | p50 ms | p95 ms | p99 ms | J/cmd | mean W | peak VRAM MB | VRAM above idle MB | CPU ms | max C |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v5:ollama:qwen2.5-3b-lora/zero_shot | held_out_v3 | 217 | 349.9 | 1623.173 | 1687.355 | 1701.688 | 64.02359 | 49.97 | 4080.0 | 2544.0 | 4.536 | 63.0 |
| v5:ollama:qwen2.5-3b-lora/few_shot | held_out_v3 | 217 | 352.5 | 1625.114 | 1699.56 | 1747.854 | 65.49753 | 50.57 | 4080.0 | 2544.0 | 5.112 | 65.0 |
| v5:ollama:qwen2.5-3b-lora/cot | held_out_v3 | 217 | 598.0 | 2608.104 | 3987.153 | 4452.902 | 193.70754 | 80.56 | 4078.0 | 2542.0 | 8.209 | 64.0 |
| v5:ollama:qwen2.5-3b-lora/role_based | held_out_v3 | 217 | 350.8 | 1626.233 | 1694.285 | 1717.754 | 63.63995 | 49.63 | 4080.0 | 2544.0 | 4.608 | 64.0 |
| v5:ollama:qwen2.5-3b-lora/rag | held_out_v3 | 217 | 356.7 | 1654.802 | 1720.855 | 1752.063 | 66.30941 | 50.76 | 4028.0 | 2543.0 | 660.138 | 67.0 |
| v5:ollama:qwen2.5:3b/zero_shot | held_out_v3 | 217 | 345.3 | 1584.294 | 1691.536 | 1910.219 | 60.40958 | 48.38 | 4047.0 | 2562.0 | 5.832 | 65.0 |
| v5:ollama:qwen2.5:3b/few_shot | held_out_v3 | 217 | 363.6 | 1667.366 | 1910.687 | 2016.528 | 66.2779 | 49.98 | 4120.0 | 2635.0 | 6.552 | 66.0 |
| v5:ollama:qwen2.5:3b/cot | held_out_v3 | 217 | 555.0 | 2442.955 | 3464.958 | 4051.567 | 167.40018 | 75.88 | 4050.0 | 2565.0 | 8.569 | 64.0 |
| v5:ollama:qwen2.5:3b/role_based | held_out_v3 | 217 | 341.0 | 1570.84 | 1668.104 | 1716.078 | 58.84008 | 47.86 | 3951.0 | 2466.0 | 4.68 | 64.0 |
| v5:ollama:qwen2.5:3b/rag | held_out_v3 | 217 | 358.1 | 1656.432 | 1750.588 | 1946.519 | 65.73152 | 50.25 | 3951.0 | 2466.0 | 648.257 | 68.0 |
| v5:ollama:qwen2.5-0.5b-lora/zero_shot | held_out_v3 | 217 | 298.8 | 1378.406 | 1423.911 | 1443.184 | 28.42557 | 31.06 | 2393.0 | 908.0 | 4.392 | 58.0 |
| v5:ollama:qwen2.5-0.5b-lora/few_shot | held_out_v3 | 217 | 297.5 | 1371.323 | 1406.443 | 1435.998 | 28.81295 | 31.44 | 2395.0 | 910.0 | 4.608 | 58.0 |
| v5:ollama:qwen2.5-0.5b-lora/cot | held_out_v3 | 217 | 3914.8 | 1372.403 | 113775.138 | 115271.072 | 1629.43805 | 100.74 | 2393.0 | 908.0 | 49.395 | 63.0 |
| v5:ollama:qwen2.5-0.5b-lora/role_based | held_out_v3 | 217 | 299.0 | 1377.907 | 1420.99 | 1454.642 | 27.5877 | 30.95 | 2393.0 | 984.0 | 4.536 | 57.0 |
| v5:ollama:qwen2.5-0.5b-lora/rag | held_out_v3 | 217 | 300.9 | 1387.278 | 1430.209 | 1445.282 | 29.29692 | 32.06 | 2395.0 | 986.0 | 682.604 | 58.0 |
| v5:ollama:qwen2.5:0.5b/zero_shot | held_out_v3 | 217 | 303.0 | 1397.503 | 1439.336 | 1460.55 | 30.84484 | 33.02 | 2393.0 | 984.0 | 4.176 | 60.0 |
| v5:ollama:qwen2.5:0.5b/few_shot | held_out_v3 | 217 | 311.3 | 1429.778 | 1489.495 | 1523.817 | 36.25585 | 36.2 | 2395.0 | 986.0 | 5.616 | 62.0 |
| v5:ollama:qwen2.5:0.5b/cot | held_out_v3 | 217 | 498.7 | 1723.326 | 2114.151 | 2480.784 | 112.76153 | 59.99 | 2393.0 | 984.0 | 6.336 | 62.0 |
| v5:ollama:qwen2.5:0.5b/role_based | held_out_v3 | 217 | 302.8 | 1393.708 | 1429.926 | 1450.07 | 29.4118 | 32.01 | 2393.0 | 984.0 | 3.816 | 59.0 |
| v5:ollama:qwen2.5:0.5b/rag | held_out_v3 | 217 | 313.2 | 1436.073 | 1507.592 | 1631.088 | 36.2466 | 36.05 | 2395.0 | 986.0 | 676.915 | 62.0 |
| v5:ollama:llama3.2:3b/zero_shot | held_out_v3 | 217 | 404.5 | 1876.967 | 1932.107 | 1946.515 | 62.47583 | 44.44 | 4293.0 | 2884.0 | 5.688 | 68.0 |
| v5:ollama:llama3.2:3b/few_shot | held_out_v3 | 217 | 410.1 | 1889.923 | 1982.736 | 2026.194 | 67.68239 | 46.74 | 4293.0 | 2884.0 | 5.112 | 65.0 |
| v5:ollama:llama3.2:3b/cot | held_out_v3 | 217 | 697.7 | 3111.16 | 4059.184 | 4351.136 | 214.10761 | 77.52 | 4293.0 | 2884.0 | 9.865 | 65.0 |
| v5:ollama:llama3.2:3b/role_based | held_out_v3 | 217 | 405.5 | 1875.275 | 1946.363 | 2007.43 | 63.81791 | 45.08 | 4293.0 | 2884.0 | 4.752 | 67.0 |
| v5:ollama:llama3.2:3b/rag | held_out_v3 | 217 | 421.3 | 1938.784 | 2025.029 | 2072.183 | 70.85437 | 47.42 | 4295.0 | 2886.0 | 618.088 | 66.0 |
| v5:ollama:deepseek-r1:7b/zero_shot | held_out_v3 | 217 | 2238.9 | 9208.677 | 19145.479 | 28090.445 | 1060.84655 | 113.75 | 6367.0 | 4958.0 | 27.29 | 68.0 |
| v5:ollama:deepseek-r1:7b/few_shot | held_out_v3 | 217 | 3066.3 | 8760.881 | 21353.049 | 44165.128 | 1527.42823 | 118.72 | 6365.0 | 4956.0 | 39.026 | 70.0 |
| v5:ollama:deepseek-r1:7b/cot | held_out_v3 | 217 | 2846.1 | 8609.622 | 15946.017 | 35130.898 | 1405.88853 | 117.81 | 6365.0 | 4956.0 | 33.266 | 69.0 |
| v5:ollama:deepseek-r1:7b/role_based | held_out_v3 | 217 | 2292.4 | 8887.146 | 20932.989 | 30820.672 | 1091.93496 | 114.4 | 6365.0 | 4956.0 | 27.29 | 69.0 |
| v5:ollama:deepseek-r1:7b/rag | held_out_v3 | 217 | 3036.2 | 8558.891 | 22835.723 | 54521.392 | 1505.00576 | 118.59 | 6365.0 | 4956.0 | 689.804 | 68.0 |
| v5:classifier:distilbert/classifier | held_out_v3 | 6293 | 121.6 | 19.036 | 24.0 | 25.352 | ≈ 0 (below GPU floor) | 10.71 | 1409.0 | ≈ 0 | 19.136 | 41.0 |
| flat_gpu | held_out_v3 | 41447 | 120.1 | 2.826 | 3.328 | 3.954 | 0.14415 | 60.79 | 1836.0 | 427.0 | 2.862 | 58.0 |
| flat_cpu | held_out_v3 | 12586 | 120.5 | 8.962 | 13.044 | 15.055 | ≈ 0 (below GPU floor) | 10.75 | 1409.0 | ≈ 0 | 47.454 | 42.0 |
| factorized_fp16 | held_out_v3 | 34503 | 120.7 | 3.425 | 3.931 | 4.262 | 0.23356 | 77.54 | 1838.0 | 429.0 | 3.491 | 59.0 |
| factorized_int4 | held_out_v3 | 34286 | 120.7 | 3.437 | 3.992 | 4.608 | 0.05408 | 26.16 | 1884.0 | 475.0 | 3.474 | 48.0 |
| tinybert | held_out_v3 | 34286 | 120.2 | 3.435 | 3.921 | 4.448 | 0.01745 | 15.78 | 1612.0 | 203.0 | 3.495 | 48.0 |
| minilm_probe | held_out_v3 | 23653 | 120.3 | 4.844 | 6.632 | 7.593 | 7e-05 | 10.81 | 1409.0 | ≈ 0 | 26.059 | 43.0 |
| knn_minilm | held_out_v3 | 24304 | 120.5 | 4.716 | 6.502 | 7.409 | ≈ 0 (below GPU floor) | 10.77 | 1409.0 | ≈ 0 | 25.398 | 40.0 |
| rules_open | held_out_v3 | 15489243 | 120.0 | 0.008 | 0.01 | 0.012 | ≈ 0 (below GPU floor) | 11.03 | 1409.0 | ≈ 0 | 0.008 | 39.0 |
| rules_firewalled | held_out_v3 | 15786316 | 120.0 | 0.007 | 0.01 | 0.012 | ≈ 0 (below GPU floor) | 10.52 | 1409.0 | ≈ 0 | 0.007 | 39.0 |
| tfidf_logreg | held_out_v3 | 136059 | 120.1 | 0.854 | 1.094 | 1.229 | ≈ 0 (below GPU floor) | 10.01 | 1409.0 | ≈ 0 | 0.884 | 38.0 |
| bilstm | held_out_v3 | 220255 | 120.1 | 0.447 | 0.948 | 1.169 | ≈ 0 (below GPU floor) | 10.12 | 1409.0 | ≈ 0 | 2.81 | 38.0 |
| textcnn | held_out_v3 | 146692 | 120.1 | 0.802 | 0.935 | 1.212 | 0.00177 | 12.96 | 1566.0 | 157.0 | 0.815 | 41.0 |
| mlpbow | held_out_v3 | 198121 | 120.0 | 0.593 | 0.683 | 0.931 | 0.00065 | 11.87 | 1566.0 | 157.0 | 0.604 | 42.0 |
| flant5 | held_out_v3 | 651 | 153.2 | 260.32 | 326.236 | 347.534 | 4.42635 | 29.61 | 2582.0 | 1173.0 | 232.479 | 48.0 |
| lora_greedy | held_out_v3 | 217 | 168.1 | 878.506 | 910.002 | 914.003 | 71.40392 | 102.98 | 8426.0 | 7017.0 | 759.937 | 69.0 |
| salsa_full_qwen2.5-3b-lora | held_out_v3 | 868 | 155.4 | 196.44 | 198.068 | 198.699 | 21.44893 | 130.55 | 8472.0 | 7012.0 | 166.421 | 70.0 |
| salsa_full_Qwen2.5-3B-Instruct | held_out_v3 | 868 | 147.9 | 162.21 | 198.577 | 200.751 | 20.07411 | 128.57 | 8108.0 | 6648.0 | 156.988 | 70.0 |
| salsa_full_qwen2.5-0.5b-lora | held_out_v3 | 1953 | 131.0 | 64.302 | 87.135 | 90.5 | 4.5248 | 78.26 | 3016.0 | 1556.0 | 66.58 | 62.0 |
| salsa_full_Qwen2.5-0.5B-Instruct | held_out_v3 | 1736 | 129.5 | 82.423 | 86.326 | 88.968 | 4.90751 | 76.57 | 2944.0 | 1484.0 | 74.21 | 62.0 |
| grpo_base | held_out_v3 | 217 | 167.5 | 882.586 | 913.53 | 915.015 | 70.31138 | 101.87 | 8198.0 | 6738.0 | 753.744 | 69.0 |
| grpo_sft | held_out_v3 | 217 | 167.4 | 881.813 | 912.534 | 914.272 | 70.20741 | 101.8 | 8198.0 | 6738.0 | 755.256 | 69.0 |
| cascade_local_tauA | held_out_v3 | 15190 | 120.9 | 3.51 | 38.739 | 39.379 | 0.74494 | 104.37 | 8468.0 | 7008.0 | 7.908 | 65.0 |
| pipeline_always_local | held_out_v3 | 5642 | 120.2 | 38.13 | 39.393 | 40.152 | 2.28731 | 118.18 | 8468.0 | 7008.0 | 20.998 | 65.0 |
| cascade_oracle (ceiling, not a system) | held_out_v3 | 33852 | 120.8 | 3.479 | 4.067 | 4.794 | 0.23678 | 77.16 | 1884.0 | 424.0 | 3.528 | 59.0 |
| flat_gpu | held_out_v4c | 41664 | 120.4 | 2.826 | 3.29 | 3.61 | 0.14465 | 60.84 | 1882.0 | 422.0 | 2.881 | 58.0 |
| flat_cpu | held_out_v4c | 12586 | 121.2 | 9.069 | 12.985 | 14.912 | ≈ 0 (below GPU floor) | 10.61 | 1455.0 | ≈ 0 | 47.458 | 42.0 |
| factorized_fp16 | held_out_v4c | 34286 | 120.3 | 3.436 | 3.94 | 4.396 | 0.23282 | 77.16 | 1884.0 | 424.0 | 3.49 | 59.0 |
| factorized_int4 | held_out_v4c | 34503 | 120.3 | 3.417 | 3.904 | 4.273 | 0.05401 | 26.28 | 1930.0 | 470.0 | 3.477 | 48.0 |
| tinybert | held_out_v4c | 34286 | 120.7 | 3.443 | 3.952 | 4.481 | 0.0197 | 16.37 | 1658.0 | 198.0 | 3.51 | 48.0 |
| minilm_probe | held_out_v4c | 23436 | 120.7 | 4.915 | 6.661 | 7.551 | ≈ 0 (below GPU floor) | 10.62 | 1455.0 | ≈ 0 | 26.077 | 43.0 |
| knn_minilm | held_out_v4c | 23653 | 120.4 | 4.847 | 6.62 | 7.542 | ≈ 0 (below GPU floor) | 10.55 | 1455.0 | ≈ 0 | 25.846 | 40.0 |
| rules_open | held_out_v4c | 15263129 | 120.0 | 0.008 | 0.011 | 0.013 | ≈ 0 (below GPU floor) | 11.13 | 1455.0 | ≈ 0 | 0.008 | 39.0 |
| rules_firewalled | held_out_v4c | 16074275 | 120.0 | 0.007 | 0.01 | 0.012 | ≈ 0 (below GPU floor) | 11.07 | 1455.0 | ≈ 0 | 0.007 | 39.0 |
| tfidf_logreg | held_out_v4c | 124341 | 120.0 | 0.937 | 1.186 | 1.334 | ≈ 0 (below GPU floor) | 10.5 | 1455.0 | ≈ 0 | 0.967 | 38.0 |
| bilstm | held_out_v4c | 218519 | 120.1 | 0.45 | 0.947 | 1.189 | ≈ 0 (below GPU floor) | 10.64 | 1455.0 | ≈ 0 | 2.811 | 38.0 |
| textcnn | held_out_v4c | 146041 | 120.1 | 0.806 | 0.944 | 1.23 | 0.0019 | 13.09 | 1612.0 | 152.0 | 0.817 | 41.0 |
| mlpbow | held_out_v4c | 196168 | 120.0 | 0.6 | 0.687 | 0.913 | 0.00074 | 11.99 | 1612.0 | 152.0 | 0.61 | 42.0 |
| flant5 | held_out_v4c | 651 | 153.7 | 263.339 | 326.629 | 351.995 | 4.51999 | 29.92 | 2636.0 | 1176.0 | 233.439 | 50.0 |
| lora_greedy | held_out_v4c | 217 | 167.3 | 875.733 | 907.495 | 908.646 | 71.17939 | 103.12 | 8480.0 | 7020.0 | 757.2 | 69.0 |
| salsa_full_qwen2.5-3b-lora | held_out_v4c | 868 | 155.9 | 196.773 | 198.472 | 199.054 | 21.57315 | 130.9 | 8480.0 | 7020.0 | 166.997 | 70.0 |
| salsa_full_Qwen2.5-3B-Instruct | held_out_v4c | 868 | 146.2 | 161.696 | 197.927 | 198.462 | 20.04194 | 129.77 | 8116.0 | 6656.0 | 155.278 | 70.0 |
| salsa_full_qwen2.5-0.5b-lora | held_out_v4c | 1953 | 131.2 | 64.978 | 88.065 | 90.341 | 4.49071 | 77.64 | 2994.0 | 1534.0 | 66.5 | 63.0 |
| salsa_full_Qwen2.5-0.5B-Instruct | held_out_v4c | 1736 | 131.6 | 83.247 | 86.96 | 90.058 | 4.95418 | 76.13 | 2978.0 | 1518.0 | 75.362 | 62.0 |
| grpo_base | held_out_v4c | 217 | 166.9 | 879.675 | 910.763 | 912.283 | 70.08044 | 101.91 | 8206.0 | 6746.0 | 750.432 | 69.0 |
| grpo_sft | held_out_v4c | 217 | 167.5 | 881.306 | 912.181 | 913.665 | 70.28662 | 101.84 | 8206.0 | 6746.0 | 754.32 | 69.0 |
| cascade_local_tauA | held_out_v4c | 14322 | 120.6 | 3.495 | 38.735 | 39.363 | 0.80298 | 105.71 | 8476.0 | 7013.0 | 8.386 | 65.0 |
| pipeline_always_local | held_out_v4c | 5859 | 122.8 | 38.182 | 39.263 | 39.896 | 2.27149 | 118.75 | 8476.0 | 7013.0 | 21.172 | 65.0 |
| cascade_oracle (ceiling, not a system) | held_out_v4c | 34286 | 120.3 | 3.436 | 3.943 | 4.283 | 0.23747 | 78.01 | 1892.0 | 429.0 | 3.488 | 59.0 |
| a8_speculative | held_out_v3 | 868 | 159.6 | 111.343 | 384.395 | 788.819 | 19.98257 | 119.3 | 8861.0 | 7328.0 | 170.543 | 71.0 |
| a10_speculative | held_out_v3 | 1085 | 139.7 | 56.908 | 330.785 | 727.945 | 9.54235 | 84.72 | 8852.0 | 7319.0 | 127.002 | 60.0 |
| a12_smalllm | held_out_v3 | 217 | 192.6 | 938.225 | 1148.562 | 1161.51 | 45.6599 | 62.06 | 9173.0 | 7640.0 | 883.785 | 58.0 |
| a8_speculative | held_out_v4c | 651 | 128.6 | 109.506 | 358.505 | 777.291 | 21.23815 | 118.14 | 8687.0 | 7154.0 | 184.644 | 71.0 |
| a10_speculative | held_out_v4c | 868 | 124.7 | 56.062 | 295.025 | 723.45 | 10.91288 | 86.58 | 8819.0 | 7286.0 | 142.497 | 60.0 |
| a12_smalllm | held_out_v4c | 217 | 199.8 | 951.604 | 1167.336 | 1258.647 | 47.2076 | 61.87 | 9173.0 | 7640.0 | 910.426 | 58.0 |


J/command = (mean load W - idle median W) x wall s / N, GPU watts only. A value <= 0 means the arm's load sat inside the GPU's idle noise and is shown as "≈ 0 (below GPU floor)"; the measured value stays in `unified_efficiency.csv` and in `metrics_final.csv` (`joules_per_command_raw`). VRAM above idle = peak VRAM minus the median `memory.used` of the idle baseline in force when the arm was measured (`power_logs/idle_*.csv`); every row's matched log reproduces its recorded idle watts. `cascade_oracle` consumes the true row: a ceiling, not a system. [Certain]


### Notes on outliers

**qwen2.5-0.5b-lora / cot: 1629.43805 J/command, p95 114 s -- runaway generation.** In the same cell's Stage-1 outputs (temperature 0), 32 of 217 commands run for more than 20 s; all 32 stop between 113.0 and 126.2 s, with 40979-286686 characters of output of which none ends in a closed JSON object (0 of 32). The text is one answer followed by `modifiers:none` repeated thousands of times. `llm/clients.py` sets no `num_predict`, so generation is bounded only by the stop token and the server. [Certain] The tight cluster at ~113 s with very different lengths points to a fixed generation budget being hit rather than the model finishing [Likely]; which budget it is has not been established [Guessing]. The energy per command is therefore dominated by ~15% of commands generating at ~100 W for ~113 s each. The same model's zero_shot cell has no such rows.

**deepseek-r1:7b: 1061-1527 J/command.** A reasoning model: before its answer it writes a reasoning trace, median 1516 characters per command in V5's own held_out_v3 rows (`bench/results_v5_deepseek.csv`, `thinking_chars`), at ~115 W on this GPU. Long outputs, not a fault. [Certain]



**The V5 LoRA's first measured latency and energy.** V5 recorded neither for its Ollama cells; the rows above are the first. Rows present: v5:ollama:deepseek-r1:7b/cot, v5:ollama:deepseek-r1:7b/few_shot, v5:ollama:deepseek-r1:7b/rag, v5:ollama:deepseek-r1:7b/role_based, v5:ollama:deepseek-r1:7b/zero_shot, v5:ollama:llama3.2:3b/cot, v5:ollama:llama3.2:3b/few_shot, v5:ollama:llama3.2:3b/rag, v5:ollama:llama3.2:3b/role_based, v5:ollama:llama3.2:3b/zero_shot, v5:ollama:qwen2.5-0.5b-lora/cot, v5:ollama:qwen2.5-0.5b-lora/few_shot, v5:ollama:qwen2.5-0.5b-lora/rag, v5:ollama:qwen2.5-0.5b-lora/role_based, v5:ollama:qwen2.5-0.5b-lora/zero_shot, v5:ollama:qwen2.5-3b-lora/cot, v5:ollama:qwen2.5-3b-lora/few_shot, v5:ollama:qwen2.5-3b-lora/rag, v5:ollama:qwen2.5-3b-lora/role_based, v5:ollama:qwen2.5-3b-lora/zero_shot, v5:ollama:qwen2.5:0.5b/cot, v5:ollama:qwen2.5:0.5b/few_shot, v5:ollama:qwen2.5:0.5b/rag, v5:ollama:qwen2.5:0.5b/role_based, v5:ollama:qwen2.5:0.5b/zero_shot, v5:ollama:qwen2.5:3b/cot, v5:ollama:qwen2.5:3b/few_shot, v5:ollama:qwen2.5:3b/rag, v5:ollama:qwen2.5:3b/role_based, v5:ollama:qwen2.5:3b/zero_shot.


The modelled cascade constants in `p45_cascade_full.csv` are used nowhere in this run: cascades are measured per command. [Certain]


## 4b. The V5 cells on both splits

All 36 cells (31 local + 5 gpt-4o-mini), from `v5cells_both_splits.csv`. Collapsed as A9 did: warm-up rows excluded, gpt-4o-mini's 5 repeats by majority vote (a tie is wrong), so n is 216 for a local cell and 217 for a cloud cell. CIs are a clustered bootstrap over command_id, 2,000 resamples, seed 20260827 -- **descriptive only: no hypothesis is tested here.** [Certain]

| cell | kind | v3 action [95% CI] | v4c action [95% CI] | v3 row [95% CI] | v4c row [95% CI] | v3 J/cmd |
|---|---|---|---|---|---|---|
| openai:gpt-4o-mini / cot | cloud | 0.7327 [0.6728, 0.788] | 0.7143 [0.6498, 0.7742] | 0.8241 [0.75, 0.8889] | 0.7685 [0.6852, 0.8426] | not measurable (cloud) |
| openai:gpt-4o-mini / few_shot | cloud | 0.8341 [0.7788, 0.8802] | 0.6866 [0.6267, 0.7512] | 0.8796 [0.8148, 0.9352] | 0.5741 [0.4815, 0.6667] | not measurable (cloud) |
| openai:gpt-4o-mini / rag | cloud | 0.9355 [0.8986, 0.9631] | 0.8571 [0.8111, 0.9032] | 1.0 [1.0, 1.0] | 0.8426 [0.7778, 0.9074] | not measurable (cloud) |
| openai:gpt-4o-mini / role_based | cloud | 0.9171 [0.8802, 0.9493] | 0.7143 [0.6544, 0.7742] | 0.9537 [0.9074, 0.9907] | 0.537 [0.4352, 0.6296] | not measurable (cloud) |
| openai:gpt-4o-mini / zero_shot | cloud | 0.8756 [0.8295, 0.9171] | 0.6682 [0.6037, 0.7281] | 0.9722 [0.9352, 1.0] | 0.537 [0.4352, 0.6296] | not measurable (cloud) |
| classifier:distilbert / classifier | local | 0.75 [0.6898, 0.8056] | 0.7176 [0.6574, 0.7731] | 0.6262 [0.5234, 0.7103] | 0.5607 [0.4671, 0.6542] | ≈ 0 (below GPU floor) |
| ollama:deepseek-r1:7b / cot | local | 0.125 [0.0833, 0.1759] | 0.1204 [0.0787, 0.1667] | 0.1682 [0.1028, 0.243] | 0.1215 [0.0654, 0.1869] | 1405.88853 |
| ollama:deepseek-r1:7b / few_shot | local | 0.3935 [0.3287, 0.4583] | 0.3148 [0.2546, 0.375] | 0.8318 [0.757, 0.8972] | 0.5234 [0.4299, 0.6168] | 1527.42823 |
| ollama:deepseek-r1:7b / rag | local | 0.713 [0.6528, 0.7731] | 0.6019 [0.537, 0.6667] | 0.8972 [0.8409, 0.9533] | 0.6449 [0.5514, 0.729] | 1505.00576 |
| ollama:deepseek-r1:7b / role_based | local | 0.338 [0.2778, 0.4028] | 0.2963 [0.2407, 0.3611] | 0.5234 [0.4299, 0.6262] | 0.3925 [0.2991, 0.486] | 1091.93496 |
| ollama:deepseek-r1:7b / zero_shot | local | 0.1574 [0.1111, 0.2037] | 0.1481 [0.1019, 0.1991] | 0.243 [0.1682, 0.3271] | 0.2056 [0.1308, 0.2897] | 1060.84655 |
| ollama:llama3.2:3b / cot | local | 0.0787 [0.0463, 0.1157] | 0.0648 [0.0324, 0.0972] | 0.0841 [0.0374, 0.1402] | 0.0374 [0.0093, 0.0748] | 214.10761 |
| ollama:llama3.2:3b / few_shot | local | 0.4167 [0.3565, 0.4815] | 0.3148 [0.2546, 0.375] | 0.785 [0.7009, 0.8598] | 0.3364 [0.243, 0.4299] | 67.68239 |
| ollama:llama3.2:3b / rag | local | 0.3843 [0.3194, 0.4491] | 0.2315 [0.1759, 0.2872] | 0.7664 [0.6822, 0.8411] | 0.3645 [0.271, 0.4579] | 70.85437 |
| ollama:llama3.2:3b / role_based | local | 0.5324 [0.463, 0.5972] | 0.3889 [0.3241, 0.4537] | 0.6542 [0.5607, 0.7383] | 0.2523 [0.1776, 0.3364] | 63.81791 |
| ollama:llama3.2:3b / zero_shot | local | 0.5602 [0.4954, 0.625] | 0.3981 [0.3332, 0.463] | 0.6075 [0.514, 0.7009] | 0.1682 [0.1028, 0.243] | 62.47583 |
| ollama:qwen2.5-0.5b-lora / cot | local | 0.0602 [0.0324, 0.0926] | 0.0602 [0.0324, 0.0926] | 0.028 [0.0, 0.0654] | 0.0187 [0.0, 0.0467] | 1629.43805 |
| ollama:qwen2.5-0.5b-lora / few_shot | local | 0.2963 [0.2361, 0.3611] | 0.2454 [0.1898, 0.3009] | 0.271 [0.1869, 0.3551] | 0.1402 [0.0748, 0.2056] | 28.81295 |
| ollama:qwen2.5-0.5b-lora / rag | local | 0.2407 [0.1852, 0.3009] | 0.1898 [0.1389, 0.2407] | 0.4393 [0.3458, 0.5327] | 0.2243 [0.1495, 0.3084] | 29.29692 |
| ollama:qwen2.5-0.5b-lora / role_based | local | 0.4352 [0.3704, 0.5] | 0.3333 [0.2685, 0.3935] | 0.5981 [0.514, 0.6916] | 0.271 [0.1963, 0.3551] | 27.5877 |
| ollama:qwen2.5-0.5b-lora / zero_shot | local | 0.4537 [0.3889, 0.5185] | 0.3519 [0.287, 0.412] | 0.6168 [0.5327, 0.7103] | 0.2897 [0.2056, 0.3738] | 28.42557 |
| ollama:qwen2.5-3b-lora / cot | local | 0.8148 [0.7639, 0.8657] | 0.5324 [0.4676, 0.6019] | 0.8692 [0.8037, 0.9252] | 0.2804 [0.2056, 0.3645] | 193.70754 |
| ollama:qwen2.5-3b-lora / few_shot | local | 0.7593 [0.7037, 0.8194] | 0.5833 [0.5185, 0.6528] | 0.6636 [0.5701, 0.7477] | 0.3084 [0.2243, 0.3925] | 65.49753 |
| ollama:qwen2.5-3b-lora / rag | local | 0.7454 [0.6852, 0.8009] | 0.5648 [0.4954, 0.6296] | 0.7383 [0.6542, 0.8131] | 0.3832 [0.2897, 0.4766] | 66.30941 |
| ollama:qwen2.5-3b-lora / role_based | local | 0.9491 [0.9167, 0.9769] | 0.625 [0.5602, 0.6898] | 0.9907 [0.972, 1.0] | 0.3364 [0.243, 0.4299] | 63.63995 |
| ollama:qwen2.5-3b-lora / zero_shot | local | 0.9583 [0.9306, 0.9815] | 0.6296 [0.5648, 0.6944] | 1.0 [1.0, 1.0] | 0.3364 [0.243, 0.4299] | 64.02359 |
| ollama:qwen2.5:0.5b / cot | local | 0.0139 [0.0, 0.0324] | 0.0139 [0.0, 0.0324] | 0.0 [0.0, 0.0] | 0.0093 [0.0, 0.028] | 112.76153 |
| ollama:qwen2.5:0.5b / few_shot | local | 0.0046 [0.0, 0.0139] | 0.0046 [0.0, 0.0139] | 0.3645 [0.2804, 0.4579] | 0.2523 [0.1776, 0.3364] | 36.25585 |
| ollama:qwen2.5:0.5b / rag | local | 0.0602 [0.0323, 0.0926] | 0.0648 [0.0324, 0.1019] | 0.2804 [0.1963, 0.3645] | 0.2897 [0.2056, 0.3832] | 36.2466 |
| ollama:qwen2.5:0.5b / role_based | local | 0.0741 [0.0417, 0.1111] | 0.0741 [0.0417, 0.1111] | 0.1402 [0.0748, 0.2056] | 0.0467 [0.0093, 0.0935] | 29.4118 |
| ollama:qwen2.5:0.5b / zero_shot | local | 0.0926 [0.0556, 0.1343] | 0.0833 [0.0463, 0.1204] | 0.2243 [0.1495, 0.3084] | 0.0561 [0.0187, 0.1028] | 30.84484 |
| ollama:qwen2.5:3b / cot | local | 0.4074 [0.3426, 0.4769] | 0.3333 [0.2731, 0.3981] | 0.4673 [0.3645, 0.5607] | 0.1963 [0.1215, 0.271] | 167.40018 |
| ollama:qwen2.5:3b / few_shot | local | 0.5926 [0.5278, 0.6574] | 0.5185 [0.4537, 0.5833] | 0.5794 [0.4766, 0.6729] | 0.4112 [0.3178, 0.5047] | 66.2779 |
| ollama:qwen2.5:3b / rag | local | 0.662 [0.5972, 0.7222] | 0.5509 [0.4861, 0.6157] | 0.6729 [0.5794, 0.7664] | 0.486 [0.3925, 0.5794] | 65.73152 |
| ollama:qwen2.5:3b / role_based | local | 0.3704 [0.3102, 0.4352] | 0.2917 [0.2315, 0.3519] | 0.486 [0.3925, 0.5794] | 0.2056 [0.1308, 0.2897] | 58.84008 |
| ollama:qwen2.5:3b / zero_shot | local | 0.3843 [0.3241, 0.4537] | 0.2963 [0.2361, 0.3565] | 0.5421 [0.4486, 0.6262] | 0.2243 [0.1495, 0.3084] | 60.40958 |



## 5. Significance


### Primary family — 8 hypotheses, declared 2026-09-22 before this run

| id | split | A | B | metric | n | acc A | acc B | diff | CI lo | CI hi | p | p_holm | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H1 | held_out_v4c | flat_gpu | lora_greedy | action | 217 | 0.7143 | 0.6221 | 9.22 | 4.15 | 14.75 | 0.00119324 | 0.00477296 | CLAIMABLE |
| H2 | held_out_v4c | flat_gpu | lora_greedy | row | 108 | 0.5556 | 0.3333 | 22.22 | 13.89 | 30.56 | 1.19209e-07 | 7.15254e-07 | CLAIMABLE |
| H3 | held_out_v3 | lora_greedy | flat_gpu | action | 217 | 0.9539 | 0.7465 | 20.74 | 14.75 | 26.73 | 7.05398e-11 | 4.93779e-10 | CLAIMABLE |
| H4 | held_out_v3 | factorized_fp16 | flat_gpu | action | 217 | 0.6083 | 0.7465 | -13.82 | -19.35 | -9.21 | 6.03854e-07 | 3.01927e-06 | CLAIMABLE |
| H5 | held_out_v4c | cascade_cloud_tauA | flat_gpu | action | 217 | 0.7696 | 0.7143 | 5.53 | 0.92 | 10.14 | 0.0356981 | 0.0713962 | nominal (does not survive family-wise correction) |
| H6 | held_out_v4c | grpo_base | lora_greedy | action | 217 | 0.6959 | 0.6221 | 7.37 | 2.76 | 12.44 | 0.00522288 | 0.0156686 | CLAIMABLE |
| H7 | held_out_v3 | salsa_full_qwen2.5-3b-lora | lora_greedy | action | 217 | 0.5023 | 0.9539 | -45.16 | -52.07 | -38.25 | 1.84953e-26 | 1.47962e-25 | CLAIMABLE |
| H8 | held_out_v4c | textcnn | flat_gpu | action | 217 | 0.6959 | 0.7143 | -1.84 | -8.29 | 5.07 | 0.683489 | 0.683489 | NOT CLAIMABLE |


**Transparency.** H1, H2, H4 and H5 were measured once before this family was declared; their old 26-hypothesis Holm values are printed in the `s4_significance` log and are not re-derived here.


### Exploratory family — 1732 comparisons, Holm across all of them

Reported, never headlined. 1006 survive correction.

| split | arm_a | arm_b | metric | diff_pts | p_holm | verdict |
|---|---|---|---|---|---|---|
| held_out_v3 | a10_speculative | bilstm | action | 25.35 | 4.43246e-11 | CLAIMABLE |
| held_out_v3 | a10_speculative | bilstm | row | 31.48 | 1.30036e-07 | CLAIMABLE |
| held_out_v3 | a10_speculative | cascade_cloud_always | action | 16.13 | 7.48446e-05 | CLAIMABLE |
| held_out_v3 | a10_speculative | cascade_cloud_always | row | 29.63 | 5.05708e-07 | CLAIMABLE |
| held_out_v3 | a10_speculative | cascade_cloud_tauA | action | 22.12 | 4.06291e-08 | CLAIMABLE |
| held_out_v3 | a10_speculative | cascade_cloud_tauA | row | 41.67 | 7.61133e-11 | CLAIMABLE |
| held_out_v3 | a10_speculative | cascade_local_tauA | action | 22.12 | 4.06291e-08 | CLAIMABLE |
| held_out_v3 | a10_speculative | cascade_local_tauA | row | 41.67 | 7.61133e-11 | CLAIMABLE |
| held_out_v3 | a10_speculative | factorized_fp16 | action | 34.56 | 2.62378e-15 | CLAIMABLE |
| held_out_v3 | a10_speculative | factorized_fp16 | row | 66.67 | 6.61108e-19 | CLAIMABLE |
| held_out_v3 | a10_speculative | factorized_int4 | action | 34.56 | 2.62378e-15 | CLAIMABLE |
| held_out_v3 | a10_speculative | factorized_int4 | row | 65.74 | 1.29596e-18 | CLAIMABLE |
| held_out_v3 | a10_speculative | flant5 | action | 33.64 | 9.17038e-15 | CLAIMABLE |
| held_out_v3 | a10_speculative | flant5 | row | 57.41 | 6.42282e-16 | CLAIMABLE |
| held_out_v3 | a10_speculative | flat_cpu | action | 20.74 | 7.92867e-08 | CLAIMABLE |
| held_out_v3 | a10_speculative | flat_cpu | row | 37.96 | 1.16827e-09 | CLAIMABLE |
| held_out_v3 | a10_speculative | flat_gpu | action | 20.74 | 7.92867e-08 | CLAIMABLE |
| held_out_v3 | a10_speculative | flat_gpu | row | 37.96 | 1.16827e-09 | CLAIMABLE |
| held_out_v3 | a10_speculative | grpo_base | action | 16.13 | 2.96343e-06 | CLAIMABLE |
| held_out_v3 | a10_speculative | grpo_base | row | 27.78 | 1.96696e-06 | CLAIMABLE |
| held_out_v3 | a10_speculative | knn_minilm | action | 30.88 | 1.38833e-12 | CLAIMABLE |
| held_out_v3 | a10_speculative | knn_minilm | row | 43.52 | 1.93837e-11 | CLAIMABLE |
| held_out_v3 | a10_speculative | minilm_probe | action | 30.41 | 7.35942e-13 | CLAIMABLE |
| held_out_v3 | a10_speculative | minilm_probe | row | 37.96 | 1.16827e-09 | CLAIMABLE |
| held_out_v3 | a10_speculative | mlpbow | action | 31.34 | 2.12075e-13 | CLAIMABLE |
| held_out_v3 | a10_speculative | mlpbow | row | 33.33 | 3.42843e-08 | CLAIMABLE |
| held_out_v3 | a10_speculative | pipeline_always_local | action | 11.06 | 0.0289958 | CLAIMABLE |
| held_out_v3 | a10_speculative | pipeline_always_local | row | 19.44 | 0.000810623 | CLAIMABLE |
| held_out_v3 | a10_speculative | rules_firewalled | action | 50.69 | 4.82564e-25 | CLAIMABLE |
| held_out_v3 | a10_speculative | rules_firewalled | row | 66.67 | 6.61108e-19 | CLAIMABLE |
| held_out_v3 | a10_speculative | rules_open | action | 23.04 | 1.26254e-08 | CLAIMABLE |
| held_out_v3 | a10_speculative | salsa_full_Qwen2.5-0.5B-Instruct | action | 90.78 | 1.72253e-56 | CLAIMABLE |
| held_out_v3 | a10_speculative | salsa_full_Qwen2.5-0.5B-Instruct | row | 69.44 | 8.33799e-20 | CLAIMABLE |
| held_out_v3 | a10_speculative | salsa_full_Qwen2.5-3B-Instruct | action | 59.45 | 2.74292e-31 | CLAIMABLE |
| held_out_v3 | a10_speculative | salsa_full_Qwen2.5-3B-Instruct | row | 70.37 | 4.18752e-20 | CLAIMABLE |
| held_out_v3 | a10_speculative | salsa_full_qwen2.5-0.5b-lora | action | 91.71 | 2.16715e-55 | CLAIMABLE |
| held_out_v3 | a10_speculative | salsa_full_qwen2.5-0.5b-lora | row | 99.07 | 2.02146e-29 | CLAIMABLE |
| held_out_v3 | a10_speculative | salsa_full_qwen2.5-3b-lora | action | 45.16 | 2.96295e-23 | CLAIMABLE |
| held_out_v3 | a10_speculative | salsa_full_qwen2.5-3b-lora | row | 34.26 | 1.75641e-08 | CLAIMABLE |
| held_out_v3 | a10_speculative | textcnn | action | 25.81 | 2.33899e-11 | CLAIMABLE |


## 6. Isaac status


Parser smoke test (Stage 5 S5-1; no Isaac, no cloud), the 10 seen_v2 commands in `thesis2/demo/demo_commands.txt`: 20 parsers registered, 18 ran, 18 produced valid JSON on all 10; the rest are listed with their status. [Certain]

| parser | status | valid JSON /10 | agrees with seen_v2 /10 | p50 ms | peak VRAM MB | python env |
|---|---|---|---|---|---|---|
| cascade_cloud_always | skipped: cloud budget | - | - | - | - | thesis2gpu |
| cascade_cloud_tauA | skipped: cloud budget | - | - | - | - | thesis2gpu |
| cascade_local_tauA | ok | 10 | 10 | 3.6 | 6860.1 | thesis2gpu |
| factorized_fp16 | ok | 10 | 10 | 3.7 | 276.8 | thesis2gpu |
| factorized_int4 | ok | 10 | 10 | 3.53 | 321.5 | thesis2gpu |
| flat_cpu | ok | 10 | 10 | 12.31 | n/a | thesis2gpu |
| flat_gpu | ok | 10 | 10 | 3.05 | 275.1 | thesis2gpu |
| grpo_base | ok | 10 | 9 | 651.56 | 6593.4 | thesis2gpu |
| grpo_sft | ok | 10 | 10 | 691.96 | 6593.4 | thesis2gpu |
| lora_greedy | ok | 10 | 10 | 707.65 | 6593.4 | thesis2gpu |
| mlpbow | ok | 10 | 10 | 0.45 | 8.7 | thesis2gpu |
| pipeline_always_local | ok | 10 | 10 | 4.08 | 6859.6 | thesis2gpu |
| rules_firewalled | ok | 10 | 9 | 0.01 | n/a | thesis2gpu |
| rules_open | ok | 10 | 9 | 0.01 | n/a | thesis2gpu |
| salsa_full_Qwen2.5-0.5B-Instruct | ok | 10 | 2 | 76.42 | 1195.7 | thesis2gpu |
| salsa_full_Qwen2.5-3B-Instruct | ok | 10 | 6 | 173.68 | 6499.0 | thesis2gpu |
| salsa_full_qwen2.5-0.5b-lora | ok | 10 | 1 | 100.83 | 1195.7 | thesis2gpu |
| salsa_full_qwen2.5-3b-lora | ok | 10 | 7 | 162.38 | 6593.4 | thesis2gpu |
| textcnn | ok | 10 | 10 | 0.78 | 9.8 | thesis2gpu |
| v5_ollama | ok | 10 | 9 | 1715.4 | n/a | thesis |


Peak VRAM here is PyTorch's own `max_memory_allocated` for the parser, not the nvidia-smi figure of the efficiency table; `v5_ollama` runs inside the Ollama server (the `thesis` env), whose memory PyTorch cannot see, so it reads n/a. Cloud parsers are skipped without `--allow-cloud`. Agreement with seen_v2 is demo evidence on training-distribution commands, not a measurement. [Certain]


_Isaac itself has not been run: no `demo_isaac_*.csv`. The parsers are smoke-tested without Isaac; VRAM fit is unmeasured (see `thesis2/demo/vram_check.md`)._


## 7. Not measurable, and why

| item | why |
|---|---|

| `ensemble` latency and energy | an offline pairing of three measured V5 cells; it performs no inference of its own |

| cloud arms' energy | the work happens off this machine |

| CPU watts | only GPU watts are sampled; CPU arms carry process CPU ms instead |

| A1.7 field-level FLMD, every SALSA digits mapping | withdrawn upstream (FINAL_REPORT B-3, B-4); not registered |

| `cascade_oracle` efficiency | a ceiling that consumes the ground-truth row; accuracy only |


The GRPO arms ran: both were trained (`thesis2rl`), dev-selected, frozen and measured (`thesis2gpu`), and H6 is in the primary family with a verdict (section 5). The six legacy arms (`tinybert`, `minilm_probe`, `knn_minilm`, `tfidf_logreg`, `bilstm`, `flant5`) were retrained by `s0_retrain_legacy.py`, reproduced 217/217 and frozen (`frozen_checkpoints.csv`). [Certain]
