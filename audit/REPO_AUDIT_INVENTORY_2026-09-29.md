# Python implementation inventory ? 2026-09-29

Every tracked Python source was parsed. Summaries below come from module docstrings and symbols; this is a navigation/structural inventory, not a claim that every function was dynamically tested. See REPO_AUDIT_2026-09-29.md for review depth and findings.

| File | Lines | Module purpose / first docstring line | Top-level definitions |
|---|---:|---|---|
| [asr/record_test.py](../asr/record_test.py) | 76 | No module docstring | resolve_input_device, pick_samplerate, main |
| [asr/transcribe_test.py](../asr/transcribe_test.py) | 39 | No module docstring | main |
| [asr/voice.py](../asr/voice.py) | 185 | asr/voice.py -- the one voice front-end. Mic -> audio -> Whisper -> text. | resolve_input_device, pick_samplerate, load_whisper, record_push_to_talk, record_fixed, save_wav, listen |
| [audit/b1_leniency.py](../audit/b1_leniency.py) | 63 | B1 -- enumerate every leniency in the STRICT policy by probing llm.score. | (script / package marker) |
| [audit/b_checks.py](../audit/b_checks.py) | 107 | Block B audit checks. READ-ONLY over bench/ and data/. Writes nothing. | (script / package marker) |
| [audit/build_v4b.py](../audit/build_v4b.py) | 72 | Build data/held_out_v4b.jsonl -- held_out_v3 with row vocabulary aligned to training. | reword |
| [audit/build_v4c.py](../audit/build_v4c.py) | 105 | Build data/held_out_v4c.jsonl -- row vocabulary in NEITHER split. | reword, cmds |
| [audit/check_dispatch.py](../audit/check_dispatch.py) | 16 | Verify every model in the data maps to the intended vocabulary. | (script / package marker) |
| [audit/check_legend_v5.py](../audit/check_legend_v5.py) | 134 | Mechanical checks on a legend, using the audit/vocab_conflicts.md criteria. | words, check |
| [audit/cls_margins.py](../audit/cls_margins.py) | 46 | Step 9a: log the classifier's per-head softmax margins on held_out_v3. | (script / package marker) |
| [audit/confusion.py](../audit/confusion.py) | 264 | Task 1 -- confusion matrices over the scored held_out_v3 CSVs. READ-ONLY. | load, parse, field, build, write_matrix, pct, report, main |
| [audit/count_unique.py](../audit/count_unique.py) | 31 | How many DISTINCT strings must actually be tokenised per model? | (script / package marker) |
| [audit/demo_commandset_check.py](../audit/demo_commandset_check.py) | 155 | audit/demo_commandset_check.py -- run the 24-command demo set, no execution. | expected_backend, brief, run, table, main |
| [audit/demo_entrypoint_check.py](../audit/demo_entrypoint_check.py) | 139 | Check 3: invoke sim/run_demo.py's REAL entrypoint, main(), end to end. | scripted_input, run_case, main |
| [audit/demo_examples_check.py](../audit/demo_examples_check.py) | 39 | Prove few_shot and rag are no longer silently dead in run_demo. | (script / package marker) |
| [audit/demo_firewall_check.py](../audit/demo_firewall_check.py) | 49 | Check 1: does the demo's retrieval + few-shot path match the measurement's? | (script / package marker) |
| [audit/demo_full_pass.py](../audit/demo_full_pass.py) | 62 | Check 2: walk all 8 demo clients through run_demo's STAGE 2. | strategy_for_demo, fmt |
| [audit/demo_smoke_v5.py](../audit/demo_smoke_v5.py) | 92 | 2d: smoke-test the demo config across all 8 clients on 10 commands. | canon |
| [audit/escalation_sim.py](../audit/escalation_sim.py) | 169 | Step 9b: SIMULATE field-level escalation. Offline, from existing predictions. | clustered_ci, loc, match |
| [audit/estimate_v4_runs.py](../audit/estimate_v4_runs.py) | 81 | Wall-clock estimate for the V4a / V4b runs, from MEASURED p50 latencies. | fmt, plan |
| [audit/figures_v3.py](../audit/figures_v3.py) | 350 | Task 2 -- publication figures over all 28 held_out_v3 cells. READ-ONLY. | rung_of, load_cells, clustered_ci, short, cell_label, pctl, lat_of, save, fig_accuracy, fig_grid_vs_container, fig_pareto, fig_gap, main |
| [audit/figures_v3_style.py](../audit/figures_v3_style.py) | 368 | V2-style figures rendered against the V3 held-out results. READ-ONLY. | short, method_of, load_all, truth_loc, pred_loc, is_ambiguous_row, cell_acc, save, no_location_rate, fig_cell_accuracy, fig_confusion, fig_clean_vs_amb, fig_latency, main |
| [audit/figures_v4.py](../audit/figures_v4.py) | 237 | V4 condition figures, same style as audit/figures_v3.py. Read-only. | load, ci, short, save, rates, rc |
| [audit/lora_collapse.py](../audit/lora_collapse.py) | 140 | Why do both LoRA adapters put ~90% of grid predictions on row B? | jl, loc_of, pred_loc, truth_loc |
| [audit/lora_vocab_test.py](../audit/lora_vocab_test.py) | 114 | Does the row collapse disappear when the ROW VOCABULARY matches training? | grid_rows, pred_loc |
| [audit/numctx.py](../audit/numctx.py) | 112 | num_ctx confound check. READ-ONLY over bench/. | i, pctl |
| [audit/probe_tokenizers.py](../audit/probe_tokenizers.py) | 59 | Which EXACT per-model vocabularies can be reached without llama-tokenize? | probe |
| [audit/reconstruct_prompts.py](../audit/reconstruct_prompts.py) | 150 | Recover tokens_in without re-running any model. | cmd_of, ex_of, ntok |
| [audit/unseen_exact.py](../audit/unseen_exact.py) | 40 | No module docstring | (script / package marker) |
| [audit/v4_results.py](../audit/v4_results.py) | 176 | V4 results: full cell table across V3 / V4a / V4a-lite / V4c, clustered CIs. | load, ci, cells_of, tl, pl, rowcol |
| [audit/verify_legend_v4.py](../audit/verify_legend_v4.py) | 121 | Verify V4 changes the legend and NOTHING else. Read-only. | (script / package marker) |
| [audit/vocab_conflicts.py](../audit/vocab_conflicts.py) | 604 | Exhaustive vocabulary-conflict scan. READ-ONLY; writes audit/vocab_conflicts.md. | load, loc, verb, tgt, words, has, scored, pred_loc, truth_loc, w, collapse, evenness |
| [bench/__init__.py](../bench/__init__.py) | 0 | No module docstring | (script / package marker) |
| [bench/analyze.py](../bench/analyze.py) | 186 | bench/analyze.py -- Phase 5 analysis. Reads the two SCORED grid CSVs | short, load, pct, acc_grid, fig_heatmap, fig_latency, fig_generalization, fig_cot, print_tables, main |
| [bench/backfill_tokens.py](../bench/backfill_tokens.py) | 88 | bench/backfill_tokens.py -- fill the reserved tokens_in / tokens_out / cost_usd | get_encoder, main |
| [bench/backfill_tokens_v3.py](../bench/backfill_tokens_v3.py) | 263 | bench/backfill_tokens_v3.py -- token/cost backfill for the held_out_v3 files, | Tokenizers, main |
| [bench/backfill_tokens_v5.py](../bench/backfill_tokens_v5.py) | 263 | bench/backfill_tokens_v3.py -- token/cost backfill for the held_out_v3 files, | Tokenizers, main |
| [bench/build_seen_v2.py](../bench/build_seen_v2.py) | 176 | Build data/seen_v2.jsonl AND data/held_out_v2.jsonl from: | die, load_jsonl, check_required, main |
| [bench/dataset.py](../bench/dataset.py) | 125 | bench/dataset.py -- the one place the seen/exemplar split is defined. | _location_classes, _pick_by_class, load_seen_split |
| [bench/diag_cells.py](../bench/diag_cells.py) | 239 | bench/diag_cells.py -- per-cell accuracy and confusion, for the V2 grid. | short, load, truth_loc, pred_loc, phrase_for, is_ambiguous, per_cell, section_cells, section_confusion, section_matrix, main |
| [bench/diag_cold_accuracy.py](../bench/diag_cold_accuracy.py) | 27 | Is the cold first call less accurate, or just different? | tally |
| [bench/diag_coldstart.py](../bench/diag_coldstart.py) | 28 | Diagnostic A: is the variation entirely a run0 (cold-start) effect? | (script / package marker) |
| [bench/diag_determinism.py](../bench/diag_determinism.py) | 28 | Diagnostic A part 2: does GPU memory pressure cause the instability? | trial |
| [bench/diag_legend.py](../bench/diag_legend.py) | 211 | bench/diag_legend.py -- Which legend wording lets the model ground the grid? | schema_spec, build, load_grid_rows, main |
| [bench/diag_rowword.py](../bench/diag_rowword.py) | 156 | bench/diag_rowword.py -- Is the seen -> held-out difference a VOCABULARY effect | short, load, breakdown, show, main |
| [bench/diag_score_edges.py](../bench/diag_score_edges.py) | 202 | bench/diag_score_edges.py -- READ-ONLY. Quantify two scoring edge cases found by | _truth_verb, _null_placeholder_keys, _alias_clobber, scan, main |
| [bench/diag_variation.py](../bench/diag_variation.py) | 25 | Diagnostic A: WHAT varied between repeats in the seen run? | (script / package marker) |
| [bench/figures.py](../bench/figures.py) | 359 | bench/figures.py -- generate the V2 thesis figures from scored result CSVs. | short, load, models_in, _fl, truth_loc, pred_loc, phrase_for, is_ambiguous, cell_acc, save, fig_cell_accuracy, fig_confusion, fig_clean_vs_amb, fig_latency, fig_thinking, main |
| [bench/llm_client.py](../bench/llm_client.py) | 1914 | bench/llm_client.py | _load_grammar, _set_backend, roster, strategies_for, rung_of, model_names, resolve_model, uses_canonical, _to_backend_location, to_backend_action, validate, canonical_schema, warm_retriever, canonical_examples, canonical_prompt, _few_shot, _load_seen, _normalise_label, _retrieve, build_prompt, _rename_action_key, _to_backend_form, apply_constraints, _candidate_objects, _extract_json, call_rules, query, to_action, listen, send, quiet_transformers, warm_classifier, _offline_hint, complete_with_retry, names_target, diagnose, correction_prompt, _retry_prompt_for, query_with_retry, check_ollama, check_bridge, check_openai, check_offline, preconditions, send_control, do_reset, run_one, preflight, selftest, main |
| [bench/peek.py](../bench/peek.py) | 9 | No module docstring | (script / package marker) |
| [bench/rescore.py](../bench/rescore.py) | 175 | bench/rescore.py -- recompute score columns on an existing grid CSV using the | parse, scorer_commit, main |
| [bench/run_cell.py](../bench/run_cell.py) | 387 | bench/run_cell.py -- Phase 5 measurement runner (single cell), and the shared | apply_legend, resolve_clients, _load_rows, _check_header, run_cell, main |
| [bench/run_grid.py](../bench/run_grid.py) | 343 | bench/run_grid.py -- Phase 5 grid driver. Runs a grid of (model x strategy) cells | write_run_manifest, count_existing, main |
| [bench/summary.py](../bench/summary.py) | 552 | bench/summary.py -- KPI summary over scored grid CSVs. Read-only; computes | short, load, models_in, strats_in, pct, has_tokens, has_thinking, has_retrieval, truth_verb, verbs_in, _fl, _acc, sec_accuracy, sec_latency, _reject_stats, sec_reject, sec_thinking, sec_cost, sec_retrieval, _null_padded_but_correct, sec_invalid, main |
| [ft/eval_classifier.py](../ft/eval_classifier.py) | 138 | ft/eval_classifier.py -- Rung 4 diagnostic. NOT a reported number. | main |
| [ft/merge_and_export.py](../ft/merge_and_export.py) | 244 | ft/merge_and_export.py — Rung 3, stage 2: adapter -> Ollama model. | run, merge, eval_bf16, to_gguf, quantise, to_ollama, smoke, _parses, main |
| [ft/prep_data.py](../ft/prep_data.py) | 113 | ft/prep_data.py — build LoRA SFT data for rung 3. | read_jsonl, main |
| [ft/train_classifier.py](../ft/train_classifier.py) | 359 | ft/train_classifier.py -- Rung 4. DistilBERT with three classification heads. | set_seed, load_jsonl, build_label_space, CommandDataset, MultiHeadClassifier, masked_ce, emit, dev_exact_match, main |
| [ft/train_lora.py](../ft/train_lora.py) | 342 | ft/train_lora.py — Rung 3: LoRA fine-tuning on Qwen2.5 Instruct. | read_jsonl, SFTDataset, make_collator, dump_mask, _drop_empty_modifiers, dev_exact_match, DevEvalCallback, main |
| [inspect_smoke.py](../inspect_smoke.py) | 15 | No module docstring | (script / package marker) |
| [llm/__init__.py](../llm/__init__.py) | 0 | No module docstring | (script / package marker) |
| [llm/classifier_client.py](../llm/classifier_client.py) | 145 | llm/classifier_client.py — Rung 4. The DistilBERT classifier behind the same | DistilBertClassifierClient |
| [llm/clients.py](../llm/clients.py) | 199 | llm/clients.py — Phase 3, step 1: one interface to all models. | LLMClient, OpenAIClient, OllamaClient, all_clients |
| [llm/extract.py](../llm/extract.py) | 96 | llm/extract.py -- Phase 3, step 2: raw model text -> clean dict, or a loud failure. | ExtractResult, extract_json, _find_json_objects, _try_load, _finish |
| [llm/pipeline.py](../llm/pipeline.py) | 92 | llm/pipeline.py — Phase 3: the machine. command -> prompt -> model -> extract -> validate. | PipelineResult, run_once |
| [llm/prompts.py](../llm/prompts.py) | 176 | llm/prompts.py — Phase 3: build the model prompt. One function, six strategies. | load_schema, _grid_legend, _schema_spec, _format_example, build_prompt |
| [llm/prompts_v4.py](../llm/prompts_v4.py) | 183 | llm/prompts_v4.py -- V4 legend. LEGEND TEXT ONLY; no other prompt change. | _grid_legend_v4, _grid_legend_v4lite, _v4_legend, build_prompt, legend_text, legend_text_v3 |
| [llm/prompts_v5.py](../llm/prompts_v5.py) | 94 | llm/prompts_v5.py -- V5 DEMO legend. Not a measurement condition. | _grid_legend_v5, _v5_legend, build_prompt, legend_text |
| [llm/repl.py](../llm/repl.py) | 42 | llm/repl.py — convenience only: type a command, watch the pipeline handle it. | main |
| [llm/retrieve.py](../llm/retrieve.py) | 246 | llm/retrieve.py -- Rung 2: retrieve few-shot examples from the training split. | Retrieved, Retriever, _smoke, main |
| [llm/schema_json.py](../llm/schema_json.py) | 118 | Convert schema/schema_v2.json into real JSON Schema for constrained decoding. | to_flat_json_schema, to_json_schema, validates |
| [llm/score.py](../llm/score.py) | 158 | llm/score.py -- Phase 3, step 3: grade ONE extracted object against ground truth. | ScoreResult, _apply_aliases, _canon_value, _normalize_repr, _canonical, _equal, _categorize, score |
| [make_split.py](../make_split.py) | 199 | make_split.py -- carve train_v2 / dev_v2 out of seen_v2.jsonl. | load_rows_minus_exemplars, verb_of, ambiguity_of, write_jsonl, stratified_split, main |
| [schema/__init__.py](../schema/__init__.py) | 0 | No module docstring | (script / package marker) |
| [schema/gen_schema_v2.py](../schema/gen_schema_v2.py) | 138 | Generate schema/schema_v2.json from schema/grid_spec.json. | cell_id, classify, check_pitch, build, main |
| [schema/validate_object.py](../schema/validate_object.py) | 91 | schema/validate_object.py — Phase 3, step 3: validate ONE command dict against schema.json. | ValidationResult, load_schema, validate_object |
| [scratch_api.py](../scratch_api.py) | 59 | No module docstring | (script / package marker) |
| [scratch_build_container_rows.py](../scratch_build_container_rows.py) | 85 | No module docstring | ld, skel |
| [scratch_ckpt.py](../scratch_ckpt.py) | 20 | No module docstring | (script / package marker) |
| [scratch_clscheck.py](../scratch_clscheck.py) | 15 | No module docstring | (script / package marker) |
| [scratch_counts.py](../scratch_counts.py) | 13 | No module docstring | (script / package marker) |
| [scratch_gatecheck.py](../scratch_gatecheck.py) | 16 | No module docstring | (script / package marker) |
| [scratch_labelspace.py](../scratch_labelspace.py) | 14 | No module docstring | (script / package marker) |
| [scratch_matrix_integrity.py](../scratch_matrix_integrity.py) | 43 | No module docstring | (script / package marker) |
| [scratch_merge_check.py](../scratch_merge_check.py) | 39 | No module docstring | (script / package marker) |
| [scratch_modifiers.py](../scratch_modifiers.py) | 10 | No module docstring | (script / package marker) |
| [scratch_paths.py](../scratch_paths.py) | 31 | No module docstring | (script / package marker) |
| [scratch_phrasing_inventory.py](../scratch_phrasing_inventory.py) | 57 | No module docstring | ld, skel |
| [scratch_rung1_integrity.py](../scratch_rung1_integrity.py) | 19 | No module docstring | (script / package marker) |
| [scratch_scored_table.py](../scratch_scored_table.py) | 108 | Results-chapter table over the four held_out_v3 scored CSVs. | null_padded_correct |
| [scratch_smoke.py](../scratch_smoke.py) | 26 | No module docstring | (script / package marker) |
| [scratch_tmpl.py](../scratch_tmpl.py) | 46 | No module docstring | (script / package marker) |
| [sim/full_grasp_backend.py](../sim/full_grasp_backend.py) | 202 | No module docstring | FullGraspBackend |
| [sim/isaac_backend.py](../sim/isaac_backend.py) | 918 | sim/isaac_backend.py | classify_exception, cell_centre, box_centre, launch, _late_imports, IsaacBackend, _self_test |
| [sim/isaac_bridge.py](../sim/isaac_bridge.py) | 162 | sim/isaac_bridge.py | _ensure_dirs, _pending, _read_action, _write_result, _handle_control, main |
| [sim/isaac_demo_scene.py](../sim/isaac_demo_scene.py) | 207 | sim/isaac_demo_scene.py | cell_centre, add_lights, add_grid, add_box, add_objects, main |
| [sim/isaac_grasp_test.py](../sim/isaac_grasp_test.py) | 191 | sim/isaac_grasp_test.py | add_lights, main |
| [sim/isaac_scene_headless.py](../sim/isaac_scene_headless.py) | 140 | sim/isaac_scene_headless.py | cell_centre, add_lights, build_scene, main |
| [sim/isaac_scene_window.py](../sim/isaac_scene_window.py) | 141 | sim/isaac_scene_window.py | cell_centre, add_lights, build_scene, main |
| [sim/llm_demo.py](../sim/llm_demo.py) | 231 | sim/llm_demo.py -- ROUGH DEMO ONLY. Type a command, watch the arm move. | draw_box_markers, print_legend, drive, describe, main |
| [sim/mujoco_bridge.py](../sim/mujoco_bridge.py) | 138 | sim/mujoco_bridge.py | _ensure_dirs, _pending, _read_action, _write_result, dispatch, main |
| [sim/reachability.py](../sim/reachability.py) | 157 | Step 1 — reachability sweep. | table_extent, probe_reach, probe_place, main |
| [sim/run_demo.py](../sim/run_demo.py) | 422 | sim/run_demo.py — unified demo: pick keyboard or voice input at startup. | load_sim, build_roster, resolve_model, _fmt_rung, choose_client, choose_strategy, choose_mode, get_voice_command, main |
| [sim/sim_backend.py](../sim/sim_backend.py) | 200 | No module docstring | _seg_point_dist_2d, SimBackend |
| [sim/test_coords.py](../sim/test_coords.py) | 7 | No module docstring | (script / package marker) |
| [sim/test_move_to.py](../sim/test_move_to.py) | 13 | No module docstring | (script / package marker) |
| [sim/test_obstacles.py](../sim/test_obstacles.py) | 21 | No module docstring | (script / package marker) |
| [sim/test_verbs.py](../sim/test_verbs.py) | 28 | No module docstring | (script / package marker) |
| [sim/watch_full.py](../sim/watch_full.py) | 15 | No module docstring | (script / package marker) |
| [sim/workspace.py](../sim/workspace.py) | 126 | sim/workspace.py -- the one place a location SYMBOL becomes a coordinate. | _load_spec, _lin, grid_cells, table_z, location_classes, resolve, describe_cell |
| [smoke_test.py](../smoke_test.py) | 85 | smoke_test.py - proves the thesis environment before anything depends on it. | check, _mujoco, _gym_fetch, _sb3, _sb3_contrib, _ultra, _cv2, _whisper, _openai, _ollama, _numpy |
| [thesis2/demo/parsers.py](../thesis2/demo/parsers.py) | 187 | thesis2/demo/parsers.py -- text -> action JSON, one entry per arm. | Parser, _registry_parser, _v5_ollama, get, names |
| [thesis2/demo/run_isaac_demo.py](../thesis2/demo/run_isaac_demo.py) | 201 | thesis2/demo/run_isaac_demo.py -- the existing Isaac demo, with the parser swapped. | load_demo_commands, default_replay_ids, make_backend, execute, main |
| [thesis2/demo/smoke_parsers.py](../thesis2/demo/smoke_parsers.py) | 174 | thesis2/demo/smoke_parsers.py -- every registered parser on the 10 demo | load_commands, verify_provenance, main |
| [thesis2/final/arms.py](../thesis2/final/arms.py) | 539 | thesis2/final/arms.py -- the ONE arm registry. | Arm, register, _flat, _factorized, _local_pt_arm, _hf_arm, _rules, _lora_greedy, _spec, _salsa_full, _grpo, _cascade, _ensemble, get, summary |
| [thesis2/final/check_defence_numbers.py](../thesis2/final/check_defence_numbers.py) | 117 | thesis2/final/check_defence_numbers.py -- verify DEFENCE_NUMBERS_rev3.md. | load, lookup, main |
| [thesis2/final/common.py](../thesis2/final/common.py) | 760 | thesis2/final/common.py -- shared backbone for the final run. | rowof, colof, load_split, truth_of, v5_retriever, v5_pool, v5_examples_for, assert_no_heldout_vocab, train_vocab_words, assert_vocab_from_train_only, read_frozen, require_frozen, sha256_of, v5_census, v5_repeats_for, Tee, start_log, gate, pred_path, pred_rows_of, pred_census, pred_is_complete, score_row, write_preds, preds_from_percommand, retryable, with_retries, sdk_no_retries, RetryingClient, row_stage, log_rerun, j_display, idle_logs, idle_for, vram_above_idle, vram_display, v5_cell_outcomes, read_preds, majority, env_banner |
| [thesis2/final/legacy_recipes.py](../thesis2/final/legacy_recipes.py) | 297 | Original legacy training bodies, with evaluation removed and artifacts returned. | load, run_frozen_probe, run_tinybert, run_bilstm, run_flant5, build_knn, build_tfidf |
| [thesis2/final/make_defence_numbers.py](../thesis2/final/make_defence_numbers.py) | 217 | thesis2/final/make_defence_numbers.py -- write DEFENCE_NUMBERS_rev3.md. | load, cite, cite_group, row, header |
| [thesis2/final/net_check.py](../thesis2/final/net_check.py) | 16 | No module docstring | (script / package marker) |
| [thesis2/final/s0_freeze.py](../thesis2/final/s0_freeze.py) | 123 | thesis2/final/s0_freeze.py -- STAGE 0: freeze every NEW arm's checkpoint. | dir_sha, main |
| [thesis2/final/s0_grpo_adapter_probe.py](../thesis2/final/s0_grpo_adapter_probe.py) | 161 | thesis2/final/s0_grpo_adapter_probe.py -- can a thesis2rl adapter load in thesis2gpu? | mode_save, mode_load, main |
| [thesis2/final/s0_grpo_signal_check.py](../thesis2/final/s0_grpo_signal_check.py) | 156 | GRPO inference in thesis2gpu only; signal uses train_v2, selection uses dev_v2. | require_env, require_probe, prompt, exact_reward, rewards, load_adapter, generate, build_predict, main |
| [thesis2/final/s0_retrain_legacy.py](../thesis2/final/s0_retrain_legacy.py) | 255 | Persist six original legacy recipes; reproduce v3 once, freeze only at 217/217. | checkpoint, record_path, persist, load_predict, compare, freeze_reproduced, main |
| [thesis2/final/s0_salsa_diag.py](../thesis2/final/s0_salsa_diag.py) | 240 | thesis2/final/s0_salsa_diag.py -- diagnose SALSA-full. REPORT ONLY. | predict_traced, show_disagreements, per_field, main |
| [thesis2/final/s0_salsa_full.py](../thesis2/final/s0_salsa_full.py) | 352 | thesis2/final/s0_salsa_full.py -- STAGE 0: SALSA-full, every field by logits. | options_block, letters_for, field_options, legal_locations, SalsaFull, mode_check, mode_dev, main |
| [thesis2/final/s0_train_grpo.py](../thesis2/final/s0_train_grpo.py) | 167 | Train GRPO in thesis2rl; save adapters only; dev inference in thesis2gpu. | select_epoch, config, signal_allows_sft, train, main |
| [thesis2/final/s0_train_mlpbow.py](../thesis2/final/s0_train_mlpbow.py) | 167 | thesis2/final/s0_train_mlpbow.py -- STAGE 0: MLP bag-of-words arm (train_v2 only). | MLPBoW, masked_ce, main |
| [thesis2/final/s0_train_textcnn.py](../thesis2/final/s0_train_textcnn.py) | 179 | thesis2/final/s0_train_textcnn.py -- STAGE 0: TextCNN arm (train_v2 only). | TextCNN, masked_ce, main |
| [thesis2/final/s1_check_v5cells.py](../thesis2/final/s1_check_v5cells.py) | 325 | thesis2/final/s1_check_v5cells.py -- audit the replicated V5 cell grid. REPORT ONLY. | load_mine, load_v5, obj_of, main, _isfloat |
| [thesis2/final/s1_prompt_equality.py](../thesis2/final/s1_prompt_equality.py) | 137 | thesis2/final/s1_prompt_equality.py -- prove our prompts are V5's. NO MODEL CALLS. | RecordingClient, ours, theirs, main |
| [thesis2/final/s1_repro_check.py](../thesis2/final/s1_repro_check.py) | 262 | thesis2/final/s1_repro_check.py -- the held_out_v3 reproducibility gate. NO MODEL RUNS. | mine, census, main |
| [thesis2/final/s1_run_arm.py](../thesis2/final/s1_run_arm.py) | 488 | thesis2/final/s1_run_arm.py -- STAGE 1: one runner, one registry. | rebuild_cascade, saved_predictions, canon, repro_compare, main |
| [thesis2/final/s1_run_v5_cells.py](../thesis2/final/s1_run_v5_cells.py) | 367 | thesis2/final/s1_run_v5_cells.py -- STAGE 1: re-run the V5 cell grid. | out_path, parse_cells, main |
| [thesis2/final/s2_metrics.py](../thesis2/final/s2_metrics.py) | 380 | thesis2/final/s2_metrics.py -- STAGE 2: every metric, every arm, both splits. | word_group, ece, pct, collapse, spec_metrics, write_v5_table, main |
| [thesis2/final/s3_energy.py](../thesis2/final/s3_energy.py) | 405 | thesis2/final/s3_energy.py -- STAGE 3: measured latency and energy. | nvidia_smi_query, PowerLogger, gpu_processes, mode_idle, dir_size_mb, ollama_facts, require_idle, measure_loop, _eff_rows, mode_speculative, mode_arm, v5_cell_predictor, mode_v5_cell, main |
| [thesis2/final/s4_figures.py](../thesis2/final/s4_figures.py) | 587 | thesis2/final/s4_figures.py -- STAGE 4: the six figures. | load, f, pareto, numbering, place_labels, key_table, fmt, fig_scatter, arm_order, arm_label, heatmap, fig_wordgroup, fig_confusion, stage3_drafter_p50, fig_a10, main |
| [thesis2/final/s4_metrics_final.py](../thesis2/final/s4_metrics_final.py) | 114 | thesis2/final/s4_metrics_final.py -- STAGE 4: the one final table. | main |
| [thesis2/final/s4_significance.py](../thesis2/final/s4_significance.py) | 246 | thesis2/final/s4_significance.py -- STAGE 4: the declared families. | old_holm, outcomes, mcnemar_exact, clustered_bootstrap, compare, holm, main |
| [thesis2/final/s5_extra_analysis.py](../thesis2/final/s5_extra_analysis.py) | 216 | thesis2/final/s5_extra_analysis.py -- two descriptive analyses. NO MODEL RUNS. | words_in, by_command, pred_row, confusion, fields_wrong, gain_harm, write, main |
| [thesis2/final/s6_report.py](../thesis2/final/s6_report.py) | 360 | thesis2/final/s6_report.py -- STAGE 6: assemble the final run report. | load, table, outlier_notes, v5_section, main |
| [thesis2/final/test_retries.py](../thesis2/final/test_retries.py) | 230 | thesis2/final/test_retries.py -- the gpt-4o-mini retry layer and --retry-errors. | status_err, failing, test_with_retries, StubOpenAI, test_retrying_client, run_main, test_run_arm, test_v5cells, main |
| [thesis2/final/test_stage0.py](../thesis2/final/test_stage0.py) | 117 | Focused Stage-0 checks. No training, downloads or held-out reads. | raises, recipe_check, main |
| [thesis2/scripts/a10_prefix_cache.py](../thesis2/scripts/a10_prefix_cache.py) | 567 | thesis2/scripts/a10_prefix_cache.py -- A10: prefix KV caching on top of A8. | pct, chat_wrap, render, prep, build_prefix_cache, reset_cache, greedy_plain, greedy_prefix, spec_plain, spec_prefix |
| [thesis2/scripts/a11_rejection_positions.py](../thesis2/scripts/a11_rejection_positions.py) | 205 | thesis2/scripts/a11_rejection_positions.py -- A11: where does the classifier | render, regions |
| [thesis2/scripts/a12_smalllm_drafter.py](../thesis2/scripts/a12_smalllm_drafter.py) | 508 | thesis2/scripts/a12_smalllm_drafter.py -- A12: a CONVENTIONAL small-LM drafter | pct, chat_wrap, prep, build_cache, reset, greedy_prefix, spec_smalllm |
| [thesis2/scripts/a13_v4c_prompted.py](../thesis2/scripts/a13_v4c_prompted.py) | 236 | thesis2/scripts/a13_v4c_prompted.py -- A13: the two arms that read the V5 | load, rowof, chat_wrap, greedy, summarise |
| [thesis2/scripts/a13_v4c_significance.py](../thesis2/scripts/a13_v4c_significance.py) | 318 | thesis2/scripts/a13_v4c_significance.py -- A13 significance: the two v4c | emit, mcnemar_exact, clustered_bootstrap, emit, mcn, boot, group |
| [thesis2/scripts/a14_geometry_gate.py](../thesis2/scripts/a14_geometry_gate.py) | 182 | thesis2/scripts/a14_geometry_gate.py -- A14.1: does frozen embedding geometry | cos_matrix, report |
| [thesis2/scripts/a14_legend_check.py](../thesis2/scripts/a14_legend_check.py) | 149 | thesis2/scripts/a14_legend_check.py -- A14.2: what row vocabulary does each | (script / package marker) |
| [thesis2/scripts/a15_metrics_v2_append.py](../thesis2/scripts/a15_metrics_v2_append.py) | 202 | thesis2/scripts/a15_metrics_v2_append.py -- add the speculative-decoding and | metrics, emit, row |
| [thesis2/scripts/a16_holm_correction.py](../thesis2/scripts/a16_holm_correction.py) | 113 | thesis2/scripts/a16_holm_correction.py -- family-wise correction. | pval |
| [thesis2/scripts/a1_arms.py](../thesis2/scripts/a1_arms.py) | 419 | thesis2/scripts/a1_arms.py -- A1 arms 1, 2, 4: frozen probe, TinyBERT-4, BiLSTM. | set_seed, load, ece, brier, evaluate, run_frozen_probe, TinyFactorized, run_tinybert, BiLSTMJoint, run_bilstm, main |
| [thesis2/scripts/a1_bilstm_check.py](../thesis2/scripts/a1_bilstm_check.py) | 102 | thesis2/scripts/a1_bilstm_check.py -- why are v3 and v4c identical for the BiLSTM? | load, rd |
| [thesis2/scripts/a1_download_probe.py](../thesis2/scripts/a1_download_probe.py) | 57 | thesis2/scripts/a1_download_probe.py -- actually attempt the two downloads. | (script / package marker) |
| [thesis2/scripts/a1_flant5_diag.py](../thesis2/scripts/a1_flant5_diag.py) | 76 | thesis2/scripts/a1_flant5_diag.py -- why did Flan-T5 score 0.0000? | (script / package marker) |
| [thesis2/scripts/a1_rest.py](../thesis2/scripts/a1_rest.py) | 436 | thesis2/scripts/a1_rest.py -- A1 arms 3, 5, 6, 7. | load, q, run_salsa05b, run_flant5, _v5, run_ensemble, run_field_flmd, main |
| [thesis2/scripts/a2_metrics_all.py](../thesis2/scripts/a2_metrics_all.py) | 360 | thesis2/scripts/a2_metrics_all.py -- A2: one row per arm, every required metric. | rd, jf, add, reason |
| [thesis2/scripts/a3_mujoco_e2e.py](../thesis2/scripts/a3_mujoco_e2e.py) | 354 | thesis2/scripts/a3_mujoco_e2e.py -- A3: end-to-end execution success in MuJoCo. | norm, start_pose, rd |
| [thesis2/scripts/a5_massive_constrained.py](../thesis2/scripts/a5_massive_constrained.py) | 218 | thesis2/scripts/a5_massive_constrained.py -- A5: does constrained reconstruction fix it? | auroc, DS, lam |
| [thesis2/scripts/a6_selfcheck.py](../thesis2/scripts/a6_selfcheck.py) | 195 | thesis2/scripts/a6_selfcheck.py -- A6: leakage grep + independent recomputation. | rd, rowof, chk |
| [thesis2/scripts/a8_speculative.py](../thesis2/scripts/a8_speculative.py) | 598 | thesis2/scripts/a8_speculative.py -- A8: classifier-drafted speculative decoding. | pct, chat_wrap, render, draft_for, greedy_manual, speculative, salsa_pass, prep |
| [thesis2/scripts/a9_stats.py](../thesis2/scripts/a9_stats.py) | 596 | thesis2/scripts/a9_stats.py -- A9: clustered bootstrap CIs and McNemar tests | rd, T, norm, mcnemar_exact, clustered_bootstrap, auroc, bootstrap_auroc_diff, compare, collapse, rowof, eval_percommand |
| [thesis2/scripts/consolidate_metrics.py](../thesis2/scripts/consolidate_metrics.py) | 173 | thesis2/scripts/consolidate_metrics.py -- one row per arm, all metrics. | rd, g, E |
| [thesis2/scripts/p1_attribution.py](../thesis2/scripts/p1_attribution.py) | 651 | thesis2/scripts/p1_attribution.py -- PHASE 1, offline attribution. | load_rows, obj, field, loc_type, loc_row, loc_col, tide_bucket, per_command, cells_of, p11_tide, p12_factors, vocab_bucket, p13_vocab, p14_confidence, p15_confusion, ece, brier, p16_calibration, figures, main |
| [thesis2/scripts/p1_digest.py](../thesis2/scripts/p1_digest.py) | 67 | thesis2/scripts/p1_digest.py -- print the headline numbers from the P1 CSVs. | rd |
| [thesis2/scripts/p1_verify.py](../thesis2/scripts/p1_verify.py) | 146 | thesis2/scripts/p1_verify.py -- sanity checks on the P1.2 result. | ltype, obj, loc |
| [thesis2/scripts/p2_diagnose_row.py](../thesis2/scripts/p2_diagnose_row.py) | 116 | thesis2/scripts/p2_diagnose_row.py -- is the row-head collapse real or a bug? | buck, rows_of, predict |
| [thesis2/scripts/p2_eval.py](../thesis2/scripts/p2_eval.py) | 331 | thesis2/scripts/p2_eval.py -- PHASE 2.2, factorized vs flat head. | bucket, ece, brier, risk_coverage, softmax_top, main |
| [thesis2/scripts/p2_row_confidence.py](../thesis2/scripts/p2_row_confidence.py) | 103 | thesis2/scripts/p2_row_confidence.py -- can a confidence gate SEE the row failure? | buck |
| [thesis2/scripts/p2_train_factorized.py](../thesis2/scripts/p2_train_factorized.py) | 399 | thesis2/scripts/p2_train_factorized.py -- PHASE 2.1, factorized location head. | set_seed, load_jsonl, build_label_space, factorize, reconstruct, check_roundtrip, CommandDataset, FactorizedClassifier, masked_ce, emit, predict_symbol, dev_exact_match, main |
| [thesis2/scripts/p3_quantization_gate.py](../thesis2/scripts/p3_quantization_gate.py) | 294 | thesis2/scripts/p3_quantization_gate.py -- PHASE 3, the INT4 gate. | quantize_per_channel, dequantize, PrecisionSwitch, evaluate, main |
| [thesis2/scripts/p3_verify_int4.py](../thesis2/scripts/p3_verify_int4.py) | 241 | thesis2/scripts/p3_verify_int4.py -- did INT4 actually execute? | main |
| [thesis2/scripts/p3_verify_int4_supervised.py](../thesis2/scripts/p3_verify_int4_supervised.py) | 102 | thesis2/scripts/p3_verify_int4_supervised.py -- agreement where the head is USED. | (script / package marker) |
| [thesis2/scripts/p45_collect.py](../thesis2/scripts/p45_collect.py) | 229 | thesis2/scripts/p45_collect.py -- P4.5 / P6.1 / P6.2: collect every prediction once. | bucket, softmax_top, main |
| [thesis2/scripts/p45_full.py](../thesis2/scripts/p45_full.py) | 234 | thesis2/scripts/p45_full.py -- P4.5 end-to-end: action match, joint correctness, memory. | rd, esc_row, run_point, main |
| [thesis2/scripts/p45_sweep.py](../thesis2/scripts/p45_sweep.py) | 224 | thesis2/scripts/p45_sweep.py -- P4.5 threshold sweep, offline from the cache. | rd, escalated_row, dev_tau_b, sweep, baseline_rows, auroc, main |
| [thesis2/scripts/p5_baselines.py](../thesis2/scripts/p5_baselines.py) | 248 | thesis2/scripts/p5_baselines.py -- PHASE 5, the floor. | load, evaluate, rule_predict, build_knn, build_tfidf, main |
| [thesis2/scripts/p5_rules_firewalled.py](../thesis2/scripts/p5_rules_firewalled.py) | 187 | thesis2/scripts/p5_rules_firewalled.py -- the honest rules baseline. | words_in, make_predict, run |
| [thesis2/scripts/p6_calibration.py](../thesis2/scripts/p6_calibration.py) | 83 | thesis2/scripts/p6_calibration.py -- P6.4: calibration of the SALSA logit softmax. | ece, brier |
| [thesis2/scripts/p7_massive.py](../thesis2/scripts/p7_massive.py) | 277 | thesis2/scripts/p7_massive.py -- PHASE 7, generalization beyond the robot arm. | set_seed, DS, Model, auroc, main |
| [thesis2/scripts/p_energy.py](../thesis2/scripts/p_energy.py) | 196 | thesis2/scripts/p_energy.py -- GPU energy per command, per the stated protocol. | PowerLogger, measure, main |
| [thesis2/scripts/salsa_row.py](../thesis2/scripts/salsa_row.py) | 170 | thesis2/scripts/salsa_row.py -- single-pass logit classification for the ROW factor. | build_prompt, SalsaRowScorer, selftest |
| [validate.py](../validate.py) | 143 | validate.py — check a command dataset (.jsonl) against schema.json. | load_schema, validate_file, report, main |
