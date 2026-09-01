llm-6dof-arm - project map
===========================

Master's thesis project. Pipeline:

    natural-language command -> LLM (or classifier) -> structured JSON action -> 6-DOF arm motion

Evaluated as a benchmark across four "rungs" of increasingly sophisticated
methods, with voice input via Whisper and TWO simulator backends: MuJoCo
(Gymnasium FetchPickAndPlace-v4, conda env `thesis`) and Isaac Sim 6.0
(Franka Panda, conda env `isaac6`). MuJoCo is the one every measured number was
produced against; Isaac is the demo / robotics-realism backend and has its own
location vocabulary. See THE TWO-PROCESS DEMO BRIDGE below.

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

(__init__.py in llm/, bench/ and schema/ are package markers, nothing more.
 asr/ has none and is imported as a namespace package.)

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
                            classifier). V3 legend = provenance of the
                            30,380 measured rows. FROZEN.
    prompts_v4.py            V4 legend ONLY. Reuses every prompts.py function
                            and rebinds _grid_legend, so "identical apart
                            from the legend" holds by construction rather
                            than by test. Selected with run_cell --legend v4.
    prompts_v5.py            V5 DEMO legend, same construction. Not a
                            measurement condition; no number under it is
                            comparable with V3/V4. It is the demo default.
    schema_json.py           schema_v2 -> JSON Schema for constrained
                            decoding. Ollama takes the per-verb oneOf form;
                            OpenAI rejects oneOf at the root and gets the
                            flat form, with the field mask enforced by
                            validate_object downstream. DEMO-ONLY: every
                            measured row was produced unconstrained.
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
    backfill_tokens_v3.py    same, for the held_out_v3 files, using EACH
                            MODEL OWN vocabulary rather than one tokenizer
    analyze.py, summary.py, figures.py   produce thesis figures/tables into
                            bench/figures/ (accuracy heatmap, latency tails,
                            generalization gap, cot strict-vs-normalized).
                            V1/V2 ONLY - they default to the V2 files. V3 and
                            V4 figures live in audit/figures_v3*.py and
                            audit/figures_v4.py, written fresh rather than
                            extending these (which are the provenance of the
                            committed V2 figures and are not broken).
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
    llm_client.py            NOT the harness. The LLM half of the demo
                            bridge: prompts in the ATTACHED SIMULATOR
                            vocabulary, then posts JSON actions to
                            bridge/inbox. See THE TWO-PROCESS DEMO BRIDGE
                            and invariant 6.
    results_*.csv.manifest.json   per-run provenance for the V4 runs: argv,
                            dataset, schema version, exemplar source, env
                            vars, Ollama model digests, package versions,
                            Python, platform. The CSVs themselves are
                            gitignored (219 MB); these are what is committed.

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

sim/  - the simulator backends. MuJoCo (measured) and Isaac Sim (demo).
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
    llm_demo.py               earlier typed-only demo. NOT dead: run_demo
                            imports drive / describe / print_legend /
                            draw_box_markers from it.
    test_*.py                 coordinate / move / verb / obstacle checks
                            against the V1 SimBackend
    watch_full.py              visualization/inspection helper

    --- Isaac Sim 6.0 (conda env `isaac6`, never imported from `thesis`) ---
    isaac_backend.py          the same 7 schema verbs on a Franka Panda.
                            Differences from MuJoCo are deliberate: objects
                            are real rigid bodies and are physically carried,
                            positions are QUERIED from the live stage (so an
                            object dragged in the editor is picked up where
                            you left it), and a box holds ONE object - a
                            second placement is refused, not stacked.
                            Its location vocabulary is box_1..3, handover and
                            cell_R_C (R,C in 0-2), NOT the schema_v2 grid
                            grid_A1..grid_C3. That gap is real; see
                            bench/llm_client.py for how rung-3/4 output
                            crosses it.
    isaac_bridge.py           sim-side watcher: opens the Isaac window, reads
                            bridge/inbox, executes, writes bridge/outbox
    mujoco_bridge.py          the same protocol over the V1 SimBackend, env
                            `thesis`, no GPU. Demo path only.
    isaac_demo_scene.py       Stage A scaffold: layout sanity render
    isaac_scene_headless.py   headless scene + fixed camera
    isaac_scene_window.py     windowed full-Kit scene (records the fix:
                            SimulationApp needs an ABSOLUTE .kit path)
    isaac_grasp_test.py       Stage B gate: grasp, LIFT, place. Records the
                            fix that made it work - the gripper must be
                            explicitly opened after world.reset(), or the arm
                            descends shut and shoves the object.
                            The four scene files above have no importers;
                            each is a runnable record of one Isaac gotcha.

asr/  - voice input (Phase 4), offline Whisper via faster-whisper
    voice.py                  THE voice front-end: mic resolver (by name
                            substring, never device index - indices shift),
                            push-to-talk and fixed-window recorders, cached
                            Whisper models, offline-first with one online
                            retry. sim/run_demo.py and bench/llm_client.py
                            both call it; they used to carry two copies that
                            had already drifted apart.
    record_test.py            records a mic clip to clip.wav
    transcribe_test.py        offline transcription test (forces local
                            model cache, no network hit - fixed a 169s ->
                            0.4s latency bug)

data/  - command datasets (JSONL), versioned v1 -> v4
    seen.jsonl / held_out.jsonl        v1
    seen_v2.jsonl                       split via make_split.py into:
        train_v2.jsonl    LoRA/RAG training corpus
        dev_v2.jsonl      checkpoint/epoch selection only
    held_out_v4b.jsonl    held_out_v3 with the ROW vocabulary aligned to
                        training (upper->far, bottom->near). Same 217
                        commands, ground truth unchanged. Built read-only
                        from v3 by audit/build_v4b.py.
    held_out_v4c.jsonl    held_out_v3 with row words in NEITHER split
                        (upper->back, bottom->front). audit/build_v4c.py.
    held_out_v2.jsonl, held_out_v3.jsonl   sealed generalization KPI sets,
                            never used for training/selection (v3 = current,
                            217 rows)
    seen_v2_backup_203.jsonl            pre-gate backup of seen_v2 (203
                            rows, before the container-row expansion that
                            took held_out_v3 to 217); kept for provenance,
                            not read by any code
    location_phrases_v2.json, reachability.json   supporting grid data

audit/  - the adversarial pass over everything above. Read-only over data/
    and bench/; the scripts write only into audit/. This is where a viva
    would be run from.

    REPORTS
    AUDIT_REPORT.md          the full hostile-external-reviewer pass. Findings
                            are tagged (F-ENV, etc.) and referenced elsewhere.
    DEFENCE_NUMBERS.md       one page: every number that can be quoted, with
                            its interval and its basis.
    PREREGISTRATION_v4.md    the V4 hypotheses, committed BEFORE V4 ran, so
                            they cannot be adjusted after seeing results.
    V4_RESULTS.md            V4 cell table, clustered CIs.
    MODEL_PROVENANCE.txt     Ollama blob digests. Tag names are mutable and do
                            not identify what was measured; these do.
    vocab_conflicts.md       exhaustive vocabulary-conflict scan over the
                            schema, the legend and every command. Generated
                            by vocab_conflicts.py.
    REPO_CLEANLINESS.md      per-file delete/keep/merge audit of this repo.

    CONDITIONS AND ANALYSIS
    build_v4b.py / build_v4c.py   generate the V4 datasets from held_out_v3,
                            read-only, ground truth untouched.
    verify_legend_v4.py      proves V4 changed the legend and NOTHING else.
    check_legend_v5.py       mechanical legend checks against the
                            vocab_conflicts.md criteria, V3/V4/V4a-lite/V5.
    v4_results.py            builds V4_RESULTS.md and v4_cells.csv.
    figures_v3.py, figures_v3_style.py, figures_v4.py   the V3 and V4 figure
                            sets (bench/figures.py is V2-only; these were
                            written fresh rather than extending it).
    confusion.py             location/verb/target confusion matrices.
    lora_collapse.py, lora_vocab_test.py   why both LoRA adapters put ~90% of
                            grid predictions on row B, and whether it is a
                            dataset or a vocabulary effect.
    cls_margins.py, escalation_sim.py   classifier softmax margins, and an
                            OFFLINE simulation of field-level escalation.
    numctx.py, count_unique.py, unseen_exact.py, reconstruct_prompts.py,
    probe_tokenizers.py, check_dispatch.py, b_checks.py, b1_leniency.py,
    estimate_v4_runs.py      targeted audit probes.

    DEMO GATES (the only automated cover sim/run_demo.py has)
    demo_entrypoint_check.py  invokes run_demo.main() ITSELF - real argv, real
                            menu, real backend, one scripted command, one
                            client per provider branch. The others import
                            run_demo helpers and reimplement stage 2, which is
                            how an UnboundLocalError on the demo default path
                            survived unnoticed. Live models; slow.
    demo_full_pass.py         all 8 clients through stage 2.
    demo_smoke_v5.py          8 clients x 10 commands under the demo config
                            (V5 legend + constrained decoding).
    demo_examples_check.py    proves few_shot/rag are not silently dead in the
                            demo. They were: run_demo hardcoded examples=None,
                            build_prompt raises without them, run_once recorded
                            prompt_error, and the arm never moved.
    demo_firewall_check.py    the demo retrieval path matches the measurement
                            one. Guards invariant 2.

    DERIVED DATA (tracked on purpose)
    figures/, *.csv, confusion_summary.txt   their INPUTS are the 219 MB of
                            gitignored results CSVs, so these are the only
                            committed trace of the measured numbers.

    discarded/               quarantine. results_v4a_local.CONTAMINATED.csv was
                            produced by two concurrent run_grid processes
                            appending to one file: 6,944 rows = 2 x 3,472, and
                            775 of 3,472 duplicated keys (22.3%) DISAGREE on
                            pred_json, so de-duplication cannot rescue it. The
                            CSV is gitignored; README.md and the manifest are
                            tracked. Kept as evidence, used for no number.
                            Harness gap it exposed: run_grid.py has no lockfile
                            on --out - append-only resume is safe against a
                            crashed run, not a concurrent one.

vision/, rl/, real/  - placeholders (.gitkeep only). Vision (object
    detection) and RL are explicitly deferred future work, not on the
    critical path. real/ presumably for a future real igus ReBeL arm
    backend (different base frame - see sim/workspace.py notes).

Root-level files:
    README.md           one line. This file is the real README.
    requirements.txt    56 direct pins. READ THE PROVENANCE WARNING AT THE
                        TOP: the pins describe the AUDIT-TIME environment,
                        not the one that produced the 30,380 measured rows.
                        No lock file was captured at measurement time
                        (AUDIT_REPORT.md finding F-ENV). peft is the one
                        exception - pinned to what the checkpoints record.
    requirements.lock.txt   the full transitive closure
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


THE TWO-PROCESS DEMO BRIDGE
----------------------------

The demo runs as TWO processes that never share an interpreter, so a model crash
cannot kill the simulator and Isaac never has to live in the same conda env as
torch and openai.

    bench/llm_client.py   (env `thesis`)   command -> LLM -> validated action
              |                             writes bridge/inbox/<id>.json
              v
    bridge/inbox/  ->  sim/isaac_bridge.py  (env `isaac6`)
                   ->  sim/mujoco_bridge.py (env `thesis`, no GPU)
              |                             executes, writes the result
              v
    bridge/outbox/<id>.json  ->  back to llm_client, matched by id

bridge/ is gitignored: it is runtime scratch, not state.

There are therefore TWO demo clients, and they are not interchangeable:

    sim/run_demo.py       in-process MuJoCo, through llm/pipeline.py run_once().
                          Speaks schema_v2 (grid_A1..grid_C3 + runtime symbols).
    bench/llm_client.py   out-of-process, either simulator, over the bridge.
                          Speaks the ATTACHED backend vocabulary: Isaac
                          cell_R_C + handover, or the V1 table_* names that
                          sim/sim_backend.py behind mujoco_bridge.py implements.

They share the client roster (llm.clients.all_clients()), the verb/target/
field-mask grammar (read from schema_v2.json), and the voice front-end
(asr/voice.py). They deliberately differ in exactly two places:

  * LOCATION VOCABULARY. Unifying it would mean editing schema_v2.json, which
    is the provenance of the 30,380 measured rows. Rungs 3 and 4 cannot be
    prompted in a backend vocabulary at all - the adapters were fine-tuned on
    schema_v2 zero-shot prompts and the classifier label space IS schema_v2 -
    so they are run canonically and translated at the edge by
    llm_client.to_backend_action().

  * JSON EXTRACTION POLICY. Opposite rules, both correct for their context:
    llm/extract.py takes the LAST balanced object that parses (the cot prompt
    puts the answer on the final line, and a benchmark must produce a verdict
    for every row); llm_client._extract_json REFUSES when more than one parses
    (multiple actions is ambiguity, not a menu, and a demo must not move an arm
    on a guess). Do not unify them.


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

6. Demo / measurement firewall. The demo paths - sim/run_demo.py,
   bench/llm_client.py, sim/mujoco_bridge.py, sim/isaac_bridge.py,
   llm/prompts_v5.py, llm/schema_json.py, llm/repl.py, sim/llm_demo.py - exist
   to be watched, not measured. They differ from the harness in ways that are
   individually deliberate and collectively disqualifying: the V5 legend,
   constrained decoding on, token-overlap retrieval instead of the embedding
   store, a location vocabulary that is not schema_v2, and a different JSON
   extraction rule. NO number produced through them is comparable with the
   30,380 measured rows, and none is reported. Never regenerate results through
   a demo script, and when you change one, check whether the harness twin needs
   the same change or deliberately does not.
