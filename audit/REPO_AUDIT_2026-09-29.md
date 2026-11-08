# Repository implementation audit

Date: 2026-09-29  
Reviewed commit: `3c844ece8b681a091903214300bb94215b9961bc`  
Scope: architecture, implementation correctness, experiment integrity, simulation behavior, reproducibility, and maintainability.

## Overall assessment

This is a working thesis research repository with substantial implementations and retained experimental evidence. It is not just an LLM-to-robot demo. It now contains three overlapping systems: the original four-rung benchmark, a newer 30-arm research pipeline, and interactive/replay robot demonstrations. Much of the complexity comes from preserving older experiments while adding new ones.

The strongest parts are the shared action vocabulary, training/dev separation, checkpoint selection and hashing, explicit treatment of oracle/unfair baselines, saved per-command predictions, and tests around retries and training provenance. The current saved predictions passed the consistency checks below.

There are nevertheless confirmed implementation defects. The highest-priority issues are a mislabeled floating-point precision condition, inflated schema-validity metrics, and a newer Isaac demo that bypasses the existing coordinate adapter and validation path. These should be resolved before presenting the affected precision, validity, or demo claims. This audit does **not** establish that the saved exact-match accuracy results are wrong: all 16,492 registry rows agreed with the frozen scorer when independently rescored.

Only audit artifacts were added. No implementation, dataset, checkpoint, or existing measurement file was changed. The pre-existing untracked `v5_measurement_backup.zip` was left untouched.

## Scope and verification limits

- Inventoried and AST-parsed all **184 tracked Python files / 36,391 lines**. No syntax errors; two docstring/string escape warnings.
- Read the central implementation paths in detail: extraction/validation/scoring, prompting and clients, retrieval, classifier and LoRA training/export, benchmark execution/resume, final arm registry, training/freezing, metrics/statistics/energy, speculative decoder cores, simulator resolution/dispatch, demo adapters, and ASR.
- Examined older experiment families through module structure, selected implementation sections, their consumers, and saved outputs. This is a repository-wide structural review with deeper review of active paths, **not a claim of exhaustive line-by-line or dynamic verification of every historical script**.
- Inspected existing reports as context; findings below were checked against current code and/or local outputs rather than accepted from older reports.
- Ran offline checks and two existing test suites. Loaded the cached factorized model solely to inspect dtype; no inference was performed in that check.
- Did not rerun training, paid/cloud inference, the full benchmark, microphone recording, Isaac, physical hardware, or energy acquisition. Did not install dependencies or download models. Physics behavior and full model quality were not revalidated.
- Binary model contents, generated plots, every historical log, and the ZIP backup were not exhaustively inspected. No dedicated dependency-vulnerability or external literature review was performed.

Supporting artifacts:

- [Per-file source inventory](REPO_AUDIT_INVENTORY_2026-09-29.md)
- [Machine-readable check results](REPO_AUDIT_EVIDENCE_2026-09-29.json)
- [Read-only reproducible checks](repo_audit_checks_2026_09_29.py), run from the repository root with `python audit/repo_audit_checks_2026_09_29.py`

## What the repository implements

### End-to-end architecture

```text
typed command / microphone -> Whisper transcription
    -> prompt + optional retrieved training examples
    -> OpenAI / Ollama / in-process classifier
    -> JSON extraction -> schema validation -> frozen exact-match scorer
    -> benchmark CSVs and reports

demo execution:
    validated canonical action -> backend vocabulary adapter
    -> filesystem bridge -> MuJoCo or Isaac -> execution result

new research pipeline:
    train/dev -> select/freeze checkpoint -> registry arm inference
    -> saved prediction CSVs -> metrics + energy + paired statistics -> report
```

The diagram describes the intended separation. The newer demo currently skips parts of the validation/adapter path; see F03/F04.

| Area | Implementation and role | Assessment |
|---|---|---|
| `schema/` | Seven verbs, three targets, fourteen locations; per-verb field requirements; generated V2 schema and grid specification | Good shared grammar; modifiers validation is incomplete |
| `data/`, `make_split.py` | Versioned command datasets, fixed exemplars, train/dev split, vocabulary-shift held-out variants | Current checked truths are valid and checked split pairs have no exact command overlap |
| `llm/prompts*.py`, `schema_json.py` | Zero/few-shot, CoT, role prompting, RAG prompts; V3/V4/V5 legends; constrained-output schema | Version separation matters; legend selection mutates shared prompt state |
| `llm/clients.py`, `pipeline.py` | Provider calls, raw/thinking/usage state; failure stages; common extraction and validation | Useful separation; generation budgets and usage persistence remain weak in the original harness |
| `llm/extract.py`, `score.py` | Prose/fence JSON extraction; strict and normalized matches | Frozen scoring has documented quirks; extractor mishandles braces inside strings |
| `llm/retrieve.py` | CPU MiniLM embeddings, cosine retrieval, default training-only corpus, exact-match exclusion | Sensible small-corpus implementation; constructor permits alternate corpora, so the firewall is a caller/config convention rather than absolute enforcement |
| `ft/` | LoRA SFT with prompt loss masking and dev plateau selection; merge/export to GGUF/Ollama; DistilBERT multi-head training | Masked loss and schema-driven emission are sound design choices; export has machine-specific tooling paths |
| `bench/` | Original cell/grid runners, manifests, resume, token backfill, scoring and figures; large bridge demo client | Preserves historical methodology; resume can mistake duplicated prefixes for complete cells |
| `thesis2/scripts/p*` | Attribution, factorized location classifier, quantization perturbation, confidence cascades, baselines, calibration, broader evaluation, energy | Research experiment scripts, several with execution at import time; not a clean reusable library |
| `thesis2/scripts/a*` | Additional classifiers/seq2seq baselines, MuJoCo end-to-end study, constrained decoding, self-checking, speculation/cache experiments, significance/geometry checks | Meaningful extensions; speculative implementations include greedy-equivalence checks, but full GPU decoding was not rerun here |
| `thesis2/final/s0*` | TextCNN/BoW, SALSA letter-logit decisions, GRPO signal/training, legacy reproduction, checkpoint freeze | Strong explicit selection/provenance gates; Stage-0 tests pass |
| `thesis2/final/arms.py`, `s1*` | Thirty arms; local/cloud repeat rules; retry/error repair; stored-prediction reproduction | Central registry is useful; dtype label, completeness keys, and non-atomic writes need attention |
| `thesis2/final/s2*`–`s6*` | Metric aggregation, GPU power/latency, bootstrap/McNemar/Holm, figures, diagnosis, final report | Reports are traceable, but schema validity and several gate/aggregation contracts are incomplete |
| `sim/` | V1 proportional motion backend, verified-grasp variant, workspace resolver, Isaac Franka backend and bridges | Several distinct meanings of “success”; not interchangeable evidence of real-world task completion |
| `thesis2/demo/` | Registry-backed parsers and replay of saved predictions | New demo integration is weaker than the older bridge client; F03/F04 |
| `asr/` | Microphone selection by name, recording, cached CPU Whisper, offline-first loading | Shared implementation avoids duplicate voice stacks; not included in the typed-command accuracy benchmark |
| `audit/`, `docs/`, root scratch files | Prior audits, dataset/legend diagnostics, historical figures and exploratory probes | Valuable provenance; discoverability and import safety need improvement |
| `real/`, `vision/`, `rl/` | Placeholder directories | No real robot or vision pipeline here. GRPO exists elsewhere under `thesis2/final`, despite the old RL placeholder |

### How the newer arms work

- **Flat/factorized classifiers:** DistilBERT predicts verb/target/location, or decomposes location into type, row, column, container and runtime factors. Emission masks irrelevant fields using the shared schema.
- **Other learned baselines:** TinyBERT, frozen MiniLM probe, BiLSTM, TF-IDF/logistic regression, MiniLM nearest neighbor, TextCNN, binary BoW MLP, and Flan-T5. Legacy checkpoints are persisted and checked against saved predictions.
- **Generative methods:** merged LoRA greedy generation, GRPO from base or SFT adapters, and four SALSA-full variants that choose legal field symbols from letter-token logits.
- **Speculative methods:** A8 verifies a classifier's JSON draft against greedy tokens; A10 additionally reuses a prompt-prefix KV cache; A12 drafts blocks with a smaller LM and verifies with the larger one. The primary correctness contract is equivalence to the greedy verifier, not improved task accuracy.
- **Cascades:** uncertain grid-row decisions escalate locally or to cloud; the column and other fields remain local. The oracle receives the true row and is a ceiling, not a deployable system.
- **Ensemble:** offline pairing of stored classifier, LoRA, and cloud-RAG outputs. It is not a new live end-to-end latency measurement. `rules_open` intentionally has unfair access to held-out vocabulary and is labeled accordingly.

## Findings

Priorities: **P1** affects current reported conditions/results or an active demo path; **P2** is a reproducible robustness/correctness gap with narrower or conditional impact; **P3** is documentation/maintenance work. “Confirmed” means code or a local probe establishes the behavior, not that every historical run was affected.

### F01 — P1: `factorized_fp16` actually builds an FP32 model

**Evidence:** `thesis2/final/arms.py:89–105,173–176`. The model is constructed and moved to the device, but `m.half()` is inside the `if int4` branch only. The `_factorized(False)` builder is also used by cascades.

**Runtime verification:** loaded the cached model with network access disabled in `D:/condaenvs/thesis2gpu/python.exe`. All **114 parameter tensors were `torch.float32` on `cuda:0`**. No predictions were generated.

**Impact:** the final registry's FP16 name and method description do not describe its actual dtype. The FP16-versus-INT4 comparison changes more than quantization: the baseline is FP32 and the quantized branch first converts to FP16. Matching discrete predictions does not establish matching precision, memory, or latency.

**Action:** choose the intended condition explicitly, assert parameter dtype before measurement, and record it in provenance. Preserve existing results under their actual condition; if an FP16 comparison is required, create a separately identified measurement. This audit does not establish the runtime dtype of every older Phase-2/3 result.

### F02 — P1: final schema-validity metrics overcount invalid actions

**Evidence:** `thesis2/final/s2_metrics.py:251–253` checks only whether `pred_json.action` is a recognized verb. It does not call `schema.validate_object.validate_object`.

**Observed impact:** **28 saved predictions across 14 arm/split summaries** are counted valid despite failing the shared validator. Examples include command 139 with `location="near"`, a `hold` carrying an unexpected location, and `target="toy_remote"`.

| Arms (both splits) | Reported validity | Shared-validator validity |
|---|---:|---:|
| `lora_greedy`, `a8_speculative`, `a10_speculative`, `a12_smalllm`, `grpo_sft` | 1.0000 | 0.9954 |
| `grpo_base` | 1.0000 | 0.9862 |
| `flant5` | 0.9954 | 0.9677 |

**Action:** calculate this metric with the shared validator and regenerate derived tables/reports from existing predictions. New model inference is unnecessary. Current exact-match score consistency passed independently; this finding concerns the separately reported validity metric.

### F03 — P1: newer Isaac demo omits canonical-grid translation

**Evidence:** `thesis2/demo/run_isaac_demo.py:85–102` passes `obj["location"]` directly into backend methods. `sim/isaac_backend.py:435–462` accepts `cell_R_C`, containers and runtime symbols, not `grid_A1` names. The existing correct mapping is in `bench/llm_client.py:377–400`.

**Reproduction:** executing the unchanged demo `execute` function with the real backend resolver, without launching Isaac, returned `False, KeyError: unknown schema name: 'grid_A1'`.

**Impact:** valid grid predictions in the new live/replay demo cannot resolve to an Isaac coordinate. Older bridge-client translation does not automatically protect this separate entry point.

**Action:** reuse the established canonical-action adapter before backend dispatch. Verify all nine grid mappings, containers and runtime symbols with stubbed dispatch, then perform an Isaac smoke check. Keep original saved predictions canonical.

### F04 — P1: newer demo bypasses the validation contract

**Evidence:** `thesis2/demo/run_isaac_demo.py:162–172` calls `parser.parse()` and then `execute()` directly. Despite its docstring, it does not route those objects through `run_once`. Replay also dispatches saved objects without schema validation. `execute` accesses only the fields it needs and ignores extras.

**Reproduction:** `{"action":"stop","target":"bottle"}` is rejected by the shared schema validator but is reported successful by this demo's `execute` function. A moving verb with extra fields can likewise reach its backend method.

**Impact:** the new demo can present schema-invalid predictions as executable successes, diverging from the original validated benchmark/demo contract. There is no evidence here of physical hardware being connected.

**Action:** validate every live/replayed object before adapting/dispatching; log schema failure separately from motion failure. Use backend dispatch for stop/reject semantics as well. Add tests at this entry point, since parser-only smoke checks cannot find integration defects.

### F05 — P2: result completeness and resume do not establish unique full coverage

**Original harness:** `bench/run_grid.py:194–205,315–322` counts raw rows per model/strategy and skips at `have >= expected_per_cell`. A partial cell is rerun from the beginning. Two interrupted prefixes can accumulate enough rows to skip a cell even though later command IDs are absent. Counts do not bind the existing CSV to the requested dataset/legend either. There is no writer lock; historical concurrent contamination is already documented in `audit/discarded/README.md`.

**Final runner:** `thesis2/final/common.py:424–445` improves this to exact row totals plus distinct-command counts, but still does not validate the full `(command_id, repeat)` key set. A probe containing `(0,0),(0,0),(1,0),(1,0)` is accepted as complete for two commands with two repeats, despite missing repeat 1 entirely.

**Current impact:** no duplicate keys were found in the checked final prediction/V5 outputs; this is a confirmed recovery/concurrency vulnerability, not evidence those outputs are corrupt.

**Action:** validate exact expected keys, reject duplicates/foreign IDs, check dataset and configuration identity, and enforce one writer per output. Resume missing keys rather than relying on accumulated row counts.

### F06 — P2: prediction replacement is not crash-atomic

**Evidence:** `thesis2/final/common.py:472–478` opens the destination prediction CSV with `"w"` and writes rows directly. Retry repair and final runner output use this helper.

**Impact:** a process failure or disk error during replacement can truncate an existing result file, even though inference completed and unaffected rows were supposed to be retained. The retry tests verify retained content after a successful write, not interrupted writes.

**Action:** write a sibling temporary CSV, flush/close successfully, then atomically replace the destination. Keep the old artifact intact until replacement. Add a focused injected-write-failure check.

### F07 — P2: schema and extraction edge cases remain

**Evidence and probes:**

- `schema/validate_object.py:64–70` always permits `modifiers` without checking its type/content or consulting `modifiers_allowed`. `{"action":"stop","modifiers":42}` passes.
- `llm/extract.py:47–63` counts braces without JSON string/escape awareness. `Answer: {"action":"stop","modifiers":["}"]}` contains valid embedded JSON but returns `no_valid_json`. Clean JSON succeeds via the first parse; the defect affects prose fallback.
- `llm/score.py` deliberately drops all falsy modifiers in strict scoring. `modifiers=False` therefore scores strict true but normalized false. This is already documented as frozen behavior, not a newly discovered unexplained result.

**Action:** version any scorer/validator semantic change; do not silently reinterpret historical measurements. For current/demo extraction use a JSON decoder or string-aware scanner, and explicitly define the modifiers contract. The evaluated labels all use empty modifiers, so general modifier understanding has not been established.

### F08 — P2: simulator success and free-space contracts are too permissive

**Evidence:** `sim/full_grasp_backend.py:120–124` returns `ok=True` for `move_to` even when `_reach` exhausts its step cap. The A3 measurement script explicitly works around this using a final-distance gate (`thesis2/scripts/a3_mujoco_e2e.py:7–27`), but other callers inherit unconditional success.

`sim/workspace.py:95–100` falls back to all cells when no free cell exists; a full-grid probe returns occupied `grid_C2`. Isaac repeats this fallback at `sim/isaac_backend.py:428–433`, although its current three-object/nine-cell scene does not normally fill every cell.

**Action:** report convergence from actual final pose; fail with a distinct `no_free_cell` outcome when no legal free destination exists. Preserve the fact that A3 already adds its own movement check when interpreting its saved results.

**Other semantic limits:** V1 `SimBackend` success primarily measures waypoint motion, not physical object transport. `stop` is synchronous and does not interrupt an in-flight queued bridge action. These are limitations to state in demonstrations, not evidence of a real emergency-stop capability.

### F09 — P2: energy validity gates can accept invalid acquisition conditions

**Evidence:** `thesis2/final/s3_energy.py:105–133` writes `idle_baseline.json` before checking the utilization threshold. `require_idle():160–171` checks existence and age but does not recheck utilization. A manually resumed run can thus consume a recent baseline from a failed idle gate. The supplied autorun stops on the first failure, which mitigates that path.

`measure_loop():174–265` permits no power samples to yield NaN energy, then appends the row; its returned messages are not themselves a finite-sample validity gate. `live` samples are filtered only by `timestamp >= t0`, including the two-second post-work sleep in mean power, although `wall` excludes it.

**Impact:** contaminated idle baselines or missing sensors can produce apparently completed efficiency records; end-of-load samples can bias energy estimates. No deliberate sensor-failure or new power run was performed, and this audit does not quantify a correction to existing joule values.

**Action:** persist/use only accepted idle baselines; reject missing/nonfinite samples; record load end time and integrate samples only over the defined interval. Bind each row to its baseline and raw power log explicitly. CPU watts and cloud energy are not measured by this GPU protocol.

### F10 — P2: field aggregation and report validity checks are inconsistent

**Evidence:** `thesis2/final/s2_metrics.py:90–104` majority-collapses only `correct` for sampling arms, retaining the first repeat's row/verb/target/location/column fields. `s4_significance.py:67–86` majority-collapses the selected metric itself.

**Probe:** three repeats with field outcomes false/true/true produce action `correct=True` but retained `row_correct="False"` in Stage 2. This can make descriptive field accuracy disagree with significance inputs. No such numerical discrepancy was found in the current two cloud-cascade arms, so the saved field tables are not claimed affected.

Also, Stage 2 accepts whatever rows exist and marks them valid (`s2_metrics.py:230–253,330`); significance uses the intersection of command IDs (`s4_significance.py:121`) without requiring full matching coverage. Holm adjusts over the number of tested hypotheses (`:132–146`), which becomes a smaller family if declared tests are missing.

**Action:** share one collapse policy per metric; require complete, error-free key coverage before report publication; explicitly decide how unrun hypotheses remain in the declared family. Current primary output contains all eight tests, so no missing-test family reduction occurred there.

### F11 — P2: importing rule implementations rewrites older results

**Evidence:** `thesis2/scripts/p5_rules_firewalled.py` executes evaluation/output code at module scope. `thesis2/final/arms.py:194–216` snapshots `baseline_rules_firewall.csv`, imports the script, then restores it.

**Impact:** restoration is not in `finally`; an import failure or process interruption can leave an older measurement modified. If the file did not previously exist, the import-created output remains. Importing a predictor also unnecessarily evaluates held-out data. The issue is acknowledged in the wrapper, but the workaround retains the side effect.

**Action:** put evaluation under `main()` and expose side-effect-free rule construction. Preserve generated artifacts; remove the snapshot/restore workaround once importing the predictor no longer writes anything.

### F12 — P2: original provider path lacks explicit generation limits and loses usage

**Evidence:** `llm/clients.py` does not set an Ollama `num_predict` budget or explicit request timeout. `bench/run_cell.py:322` writes blank token/cost columns despite clients capturing authoritative `last_usage`. Token backfill scripts reconstruct counts afterward.

**Observed corroboration:** `thesis2/final/s6_report.py:57–80` already reports runaway LoRA CoT generation from saved outputs. A later final-run path has improved retries, but that does not bound the original provider generation behavior.

**Action:** make budgets/timeouts explicit for future conditions; persist provider usage immediately per call and distinguish model-call time from retrieval and total time. Freeze these settings in manifests. Any change to budgets is a new experimental condition, not a transparent repair of past measurements.

### F13 — P3: onboarding and reproducibility documentation lag the implementation

**Evidence:** `README.md` contains only the project title. `map.md` remains useful but describes V5 as demo-only and says measured numbers use the original common pipeline, while V5 measurements and a separate registry path now exist. `thesis2/out/final/CODEX_HANDOFF.md` describes an earlier unfinished Stage-0 state; later saved results demonstrate that it is historical.

`requirements.txt` explicitly describes the August audit environment. The environment used for this audit's final-run tests has torch `2.14.0+cu126` and transformers `5.17.0`, different from those root pins. `thesis2/out/env_thesis2gpu_pipfreeze.txt` exists as additional evidence, but there is no single clear bootstrap path for all environments. `ft/merge_and_export.py`, GRPO subprocess paths and `thesis2/final/autorun.ps1` include absolute machine paths. Checkpoints/exports are partly ignored by Git and require an external artifact source.

**Action:** turn README into the entry point: current run order, environment matrix, artifact retrieval/checksums, non-billed smoke checks, canonical final report, and explicit historical sections. Keep older reports for provenance rather than editing them to appear current. A fresh clone alone is not sufficient to reproduce model-backed runs.

## Quantization and measurement interpretation

`thesis2/scripts/p3_quantization_gate.py:79–143` computes integer codes but installs **dequantized floating-point weights** into ordinary `nn.Linear` layers. Its codes use int8 storage, and `storage_bytes()` estimates hypothetical packed 4-bit storage. It does not implement a packed INT4 matrix-multiplication kernel. The older final report already discloses the absence of an INT4 kernel.

This supports studying accuracy under weight quantization perturbation. It does not by itself demonstrate native INT4 throughput or actual 4-bit resident-memory savings. `assert_single_model()` only establishes that the module tree is nonempty; one module tree can still retain master/code/dequantized tensor copies. State measured GPU memory separately from theoretical storage. F01 is an additional dtype mismatch specific to the newer registry.

Interpret accuracy at the command level: the 12,152 V5 rows are repeats/cells over 217 commands, not 12,152 independent commands. V3/V4c are related vocabulary variants with shared label structure, not independent natural-language populations. Exact command disjointness is useful but is not proof that paraphrase/template leakage is absent.

The paired statistics code uses command-level outcomes, exact McNemar, a seeded bootstrap and Holm correction. It also discloses that H1/H2/H4/H5 were measured before the newer hypothesis family was declared. Preserve that disclosure. This audit checked implementation/data integrity, not the scientific validity of post hoc hypothesis selection or external novelty claims.

## Verification results

| Check | Result |
|---|---|
| Parse all tracked Python source | 184 files, no syntax errors; two invalid-escape warnings |
| Dataset schema and duplicate checks | Seven V2–V4c files, all checked truths valid, no normalized exact duplicates within each |
| Checked split overlaps | Train/dev, train/V3, train/V4c, dev/V3: zero stripped/lowercased exact overlaps |
| Dataset sizes | Seen V2 290; train 230; dev 57; held-out V2 132; V3/V4b/V4c 217 each. Three fixed exemplars explain 290 − 230 − 57 |
| Final registry predictions | 60 files / 16,492 rows; no duplicate `(command_id, repeat)` keys; complete command coverage checked separately |
| Stored predictions vs frozen scorer | Zero `correct` discrepancies; zero command/truth mismatches against source datasets |
| Final V5 outputs | Each split: 36 cells, 12,152 rows, no duplicate cell/command/repeat keys; no stored prompt/model-error stages |
| Full schema validity vs action-name shortcut | 28 overcounted predictions, affecting 14 aggregate rows |
| Stage-0 test suite | PASS in `thesis2gpu`; logs/output redirected to a temporary directory; intentional negative gate messages were expected |
| Retry test suite | PASS in `thesis2gpu`; provider stubs only; verifies backoff and targeted repair preserves unaffected rows |
| Factorized runtime dtype | 114 FP32 parameter tensors on CUDA under the arm named FP16; cache-only load |
| Demo integration probes | Grid mapping failure and acceptance of an invalid extra-field stop reproduced without Isaac startup |
| Edge/robustness probes | Extractor string-brace bug, permissive modifiers, occupied-cell fallback, incomplete census and field-collapse inconsistency reproduced |

Commands used for existing suites: `D:/condaenvs/thesis2gpu/python.exe thesis2/final/test_retries.py`; Stage-0 `main()` invoked after redirecting `common.OUT`, `common.LOGS` and `start_log` to temporary/no-op logging. No raw benchmark/report generator was run in place. Generated evidence JSON is the retained output of the new read-only audit checker; separate suite/dtype results above were observed during this audit.

## Smallest useful remediation order

1. Correct/document F01's precision condition; repair F02's derived validity metric from saved predictions. Preserve the current evidence and label revisions.
2. Repair the new demo's validation and coordinate adaptation together (F03/F04), then check the actual demo entry point.
3. Add exact output-key/provenance checks and atomic replacement (F05/F06); make metrics refuse incomplete/error-containing inputs.
4. Fix extraction and simulation failure reporting without silently changing the frozen historical scorer (F07/F08).
5. Tighten energy acquisition gates and metric collapse consistency (F09/F10).
6. Make reusable experiment imports side-effect-free, capture provider budgets/usage, and publish a current README/environment/artifact guide (F11–F13).

## Maintainability pass (Ponytail audit)

These are bounded simplifications; historical scripts and results should not be deleted merely because newer versions exist.

- `shrink:` remove the snapshot/restore machinery in `arms._rules` after guarding legacy evaluation with `main()`; replacement is a pure import of `make_predict`. [arms.py](../thesis2/final/arms.py)
- `shrink:` reuse one verified canonical-action validation/mapping path in the new demo; replacement is the established mapping plus shared validator, not another bespoke grammar. [run_isaac_demo.py](../thesis2/demo/run_isaac_demo.py)
- `shrink:` share command-level aggregation between metric and significance scripts; replacement is one small, pure collapse function with an explicit metric. [s2_metrics.py](../thesis2/final/s2_metrics.py)
- `delete:` remove `ultralytics` and `opencv-python` from the mandatory core environment if no optional external workflow needs them; the root requirements explicitly say repository vision is unimplemented. Replacement: an optional vision environment. [requirements.txt](../requirements.txt)

net: -0 source lines quantified, -2 mandatory dependencies possible. The line reduction is intentionally unquantified beyond zero guaranteed deletions: no cleanup patch was built, and preserving experiment provenance takes priority over an inflated savings estimate.
