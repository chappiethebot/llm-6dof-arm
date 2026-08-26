llm-6dof-arm - project map
===========================

Master's thesis project. Pipeline:

    natural-language command -> LLM (or classifier) -> structured JSON action -> 6-DOF arm motion

Evaluated as a benchmark across four "rungs" of increasingly sophisticated
methods, with a MuJoCo simulator as the current backend and voice input via
Whisper.

Everything downstream is driven from schema/schema_v2.json:
  - 7 verbs: pick_place, fetch, hold, move_to, inspect, stop, reject
  - 3 targets: rubiks_cube, bottle, toy_car
  - locations: 3 containers + 9 grid cells (grid_A1..grid_C3) + 2 runtime
    symbols (max_reach, nearest_free_cell)

Grid coordinates are measured, not hardcoded:
  sim/reachability.py -> data/reachability.json -> schema/grid_spec.json
    -> schema/gen_schema_v2.py -> schema/schema_v2.json, and grid_spec.json
    -> sim/workspace.py
  (reachability.py writes data/reachability.json only; grid_spec.json is
   hand-maintained FROM that sweep, then schema_v2.json is generated from it.)


THE FOUR RUNGS (core thesis comparison)
----------------------------------------

  Rung 1 - Prompting: zero-shot / few-shot / cot / role-based
           llm/prompts.py, llm/pipeline.py

  Rung 2 - RAG: per-command retrieved few-shot examples
           llm/retrieve.py  (corpus = data/train_v2.jsonl ONLY)

  Rung 3 - LoRA fine-tuned Qwen2.5 (0.5B and 3B), served via Ollama
           ft/train_lora.py, ft/merge_and_export.py, ft/export/*.gguf

  Rung 4 - DistilBERT multi-head classifier (verb/target/location)
           ft/train_classifier.py, llm/classifier_client.py

All four rungs are scored through the SAME measurement path:

    llm/pipeline.py:run_once()
        -> llm/extract.py        (pull JSON out of raw model text)
        -> schema/validate_object.py   (schema check)

This uniformity is a deliberate, repeatedly-stated invariant in the code.
Never fork this path per-rung.


DIRECTORY GUIDE
----------------

schema/  - single source of truth for the action grammar
    schema_v2.json / schema.json   verb/target/location definitions (v2/v1)
    grid_spec.json                 measured grid geometry, hand-maintained
                                   from data/reachability.json
    gen_schema_v2.py                generates schema_v2.json from grid spec
    validate_object.py              validates model JSON against schema

llm/  - the model-calling layer
    clients.py             uniform LLMClient interface: OpenAI, Ollama
                            (incl. LoRA-merged models)
    classifier_client.py   duck-typed client wrapping rung-4 DistilBERT
                            classifier (lazy-loaded, CPU)
    prompts.py              builds prompts for all 6 strategies
                            (zero_shot, few_shot, cot, role_based, rag,
                            classifier)
    retrieve.py             rung-2 semantic retrieval of few-shot examples
    extract.py               pulls JSON object out of raw model output
    pipeline.py              run_once(): command -> prompt -> model ->
                            extract -> validate. Never raises; failures are
                            recorded data points.
    score.py                 scoring logic for pipeline results
    repl.py                  interactive command REPL

bench/  - Phase 5, the evaluation harness (separate from llm/pipeline.py,
          which is "the machine"; this is "the eval harness")
    dataset.py               defines seen/exemplar/test split; fixed
                            few-shot exemplars chosen by location class (v2)
    run_cell.py / run_grid.py   run one (model x strategy) cell / full grid,
                            resumable, append-only CSV output (never deletes)
    rescore.py, backfill_tokens.py   post-hoc rescoring / token-count passes
    analyze.py, summary.py, figures.py   produce thesis figures/tables into
                            bench/figures/ (accuracy heatmap, latency tails,
                            generalization gap, cot strict-vs-normalized)
    diag_*.py                targeted diagnostics (determinism, cold-start
                            accuracy, legend collisions, score edges, etc.)
    results_*.csv            raw + scored grid outputs per dataset/rung
                            ("_scored" = post-validation; heldout_v3 = KPI)
    peek.py                  prints rows of a results CSV (harness output
                            inspection, not dataset inspection)
    build_seen_v2.py         builds BOTH data/seen_v2.jsonl AND
                            data/held_out_v2.jsonl from seen.jsonl +
                            location_phrases_v2.json - the two v2 sets are
                            authored in one place, which is where the
                            seen/held-out firewall is established

ft/  - fine-tuning (rungs 3 and 4)
    prep_data.py             builds SFT train/dev files (ft/data/) for the
                            0.5B and 3B models
    train_lora.py            LoRA SFT on Qwen2.5-Instruct (fixed config,
                            not swept; 230 train / 57 dev rows)
    train_classifier.py      DistilBERT 3-head classifier training (no
                            "none" class; field presence enforced by masking
                            read from schema_v2.json)
    eval_classifier.py       classifier evaluation
    merge_and_export.py      merges LoRA adapters, exports GGUF for Ollama
                            (ft/export/)
    runs/                     checkpoints per epoch/step for each of the
                            three fine-tuned models (gitignored; metrics/
                            history JSON tracked)

sim/  - MuJoCo simulation backend (Gymnasium FetchPickAndPlace-v4)
    workspace.py              the one place a location SYMBOL becomes a
                            coordinate; resolves runtime symbols against
                            scene state
    sim_backend.py            V1 backend: proportional controller executing
                            the 7 schema verbs; obstacles cleared by lifting
                            over them (objects not physically carried)
    full_grasp_backend.py     backend variant with real grasp/lift
                            verification; V1 locations + V2 grid merged
    reachability.py           measures the 9x9 reachable grid (3 trials
                            per candidate), writes data/reachability.json
                            (NOT grid_spec.json - that is derived by hand)
    run_demo.py               unified interactive demo (keyboard or voice
                            input, model/strategy as CLI flags); run as
                            `python -m sim.run_demo` from repo root
    llm_demo.py               earlier typed-only demo, kept as stable
                            reference
    test_*.py                 coordinate / move / verb / obstacle checks
    watch_full.py              visualization/inspection helper

asr/  - voice input (Phase 4), offline Whisper via faster-whisper
    record_test.py            records a mic clip to clip.wav
    transcribe_test.py        offline transcription test (forces local
                            model cache, no network hit - fixed a 169s ->
                            0.4s latency bug)

data/  - command datasets (JSONL), versioned v1 -> v3
    seen.jsonl / held_out.jsonl        v1
    seen_v2.jsonl                       split via make_split.py into:
        train_v2.jsonl    LoRA/RAG training corpus
        dev_v2.jsonl      checkpoint/epoch selection only
    held_out_v2.jsonl, held_out_v3.jsonl   sealed generalization KPI sets,
                            never used for training/selection (v3 = current,
                            217 rows)
    seen_v2_backup_203.jsonl            pre-gate backup of seen_v2 (203
                            rows, before the container-row expansion that
                            took held_out_v3 to 217); kept for provenance,
                            not read by any code
    location_phrases_v2.json, reachability.json   supporting grid data

scripts/  - empty leftover directory (no files, not even .gitkeep). Nothing
    imports it; delete it or give it a purpose.

vision/, rl/, real/  - placeholders (.gitkeep only). Vision (object
    detection) and RL are explicitly deferred future work, not on the
    critical path. real/ presumably for a future real igus ReBeL arm
    backend (different base frame - see sim/workspace.py notes).

Root-level scripts:
    validate.py         schema-validates a dataset file; also checks no
                        command string leaks between seen/held-out
                        ("sealed" discipline)
    smoke_test.py        environment sanity check (imports + cheap
                        constructions only, no weight downloads); run on
                        both dev machines first
    make_split.py         carves train_v2/dev_v2 out of seen_v2.jsonl,
                        stratified by (verb, ambiguity class)
    inspect_smoke.py      inspects smoke-test / sample outputs
    scratch_*.py           one-off analysis/integrity-check scripts (label
                        space, modifiers, matrix integrity, gate checks,
                        phrasing inventory); not part of the pipeline, kept
                        for provenance of specific investigations


KEY INVARIANTS (read before touching code)
--------------------------------------------

1. One measurement path. Every rung (prompted, RAG, LoRA, classifier) is
   scored by the same run_once -> extract_json -> validate_object chain.
   Never fork this per-rung - it would invalidate cross-rung comparisons.

2. Corpus/held-out firewall. data/train_v2.jsonl is the only RAG/training
   corpus. held_out_v2/v3.jsonl is sealed and must never leak into
   training, retrieval, or exemplar selection. run_grid.py requires
   --i-understand-heldout-is-sealed to touch it.

3. Append-only results. bench/run_grid.py never deletes CSV rows on
   restart - partial cells are re-run and appended; de-duplication (keep
   last) happens at analysis time, not write time.

4. Schema follows dataset version. v1 schema for v1 data, v2 schema for v2+
   data - enforced structurally so a v2 run can't be scored against v1
   rules.

5. Weights are gitignored; metrics/history JSON files are tracked.
