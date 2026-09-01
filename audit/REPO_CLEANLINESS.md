# Repo cleanliness audit

Read-only pass over every tracked and untracked file, 2026-09-01, at `main` @ `dc876a6`.
Nothing was changed. Recommendations only.

199 tracked files. **Zero untracked files** — everything not tracked is covered by
`.gitignore` (weights, 219 MB of results CSVs, `out/`, `bridge/`, `__pycache__`).

---

## 0. Summary of recommendations

| # | Finding | Verdict |
|---|---------|---------|
| 1 | `sim/reachability.json` is a stale second sweep of `data/reachability.json` | **delete** |
| 2 | `sim/voice_pipeline.py` | **already gone** — no action |
| 3 | `audit/full_pass.{txt,err}` throwaway run logs, tracked | **delete** |
| 4 | `bench/llm_client.py` re-implements `llm/` + `schema/` for the demo | **partial merge** — see §3 |
| 5 | `sim/run_demo.py` crashes on its own default (`--constrain on`) | **fix** — see §3.2 |
| 6 | `bench/diag_runs.py` reads `bench/results_seen.csv`, which does not exist | **delete** |
| 7 | 7 V1/V2 diagnostics still run, but only against V1/V2 data | **keep, label** — see §4 |
| 8 | `audit/demo_stage2_check.py` is a strict subset of `audit/demo_full_pass.py` | **merge** |
| 9 | `data/seen_v2_backup_203.jsonl` gitignored but documented as provenance | **track** |
| 10 | `results_v4a_local.CONTAMINATED.csv` | **already quarantined correctly** — no action |
| 11 | `map.md` misses `audit/`, `bridge/`, the whole Isaac backend, V4/V5 prompts | **update** |
| 12 | `scripts/` empty directory | **delete** |
| 13 | 20 tracked `.py` files carry a UTF-8 BOM | **strip** (cosmetic) |

Nothing here is load-bearing for the thesis numbers. The only item that changes
behaviour is #5.

---

## 1. What every file is for

### Root
| File | Purpose | Verdict |
|---|---|---|
| `README.md` | one line: `# llm-6dof-arm`. `map.md` is the real README. | keep, or fold into `map.md` |
| `map.md` | project map / invariants. Stale — see §6. | keep, update |
| `.gitignore` | weights, results CSVs, `out/`, `bridge/`. Lists `.env` twice. | keep |
| `.gitattributes` | `* text=auto eol=lf`. Working correctly (index LF, worktree CRLF). | keep |
| `requirements.txt` | 56 direct pins, audit-time env, carries the F-ENV provenance warning | keep |
| `requirements.lock.txt` | 124-line transitive closure | keep |
| `validate.py` | dataset judge: schema-checks a `.jsonl`, enforces the seen/held-out seal | keep |
| `smoke_test.py` | env sanity check, imports only, no weight downloads | keep |
| `make_split.py` | carves `train_v2`/`dev_v2` out of `seen_v2.jsonl` | keep |
| `inspect_smoke.py` | prints rows of `bench/results_smoke_v3.csv` next to the RAG corpus | keep (V3-current) |
| `scratch_*.py` (16) | one-off investigations, kept for provenance | see §4 |

### `schema/` — action grammar, single source of truth
| File | Purpose | Verdict |
|---|---|---|
| `schema.json` | V1 grammar | keep (V1 runs are scored against it) |
| `schema_v2.json` | V2 grammar: 7 verbs, 3 targets, 14 locations. Object key is **`action`**. | keep |
| `grid_spec.json` | measured grid geometry, hand-maintained from the reachability sweep | keep |
| `gen_schema_v2.py` | `grid_spec.json` -> `schema_v2.json` | keep |
| `validate_object.py` | step 3 of the one measurement path | keep |
| `__init__.py` | package marker | keep |

### `llm/` — the machine
| File | Purpose | Verdict |
|---|---|---|
| `clients.py` | `LLMClient` ABC + OpenAI/Ollama impls, `all_clients()` roster | keep |
| `classifier_client.py` | rung-4 DistilBERT wrapped as a duck-typed client | keep |
| `prompts.py` | V3 legend, 6 strategies. Provenance of the 30,380 measured rows. | keep, freeze |
| `prompts_v4.py` | V4 legend only; rebinds `prompts._grid_legend`. Used by `run_cell` + 2 audit scripts. | keep |
| `prompts_v5.py` | V5 demo legend. Used by `run_cell`, `check_legend_v5`. | keep |
| `retrieve.py` | rung-2 embedding retrieval over `train_v2.jsonl` only | keep |
| `extract.py` | raw text -> dict. Takes the **last** balanced object (cot answers land last). | keep |
| `pipeline.py` | `run_once()`, never raises | keep |
| `score.py` | strict + normalized grading | keep |
| `schema_json.py` | JSON-Schema emitters for constrained decoding (Ollama `oneOf`, OpenAI flat) | keep |
| `repl.py` | hand-poking toy. **Zero importers**, but it is an entrypoint, 41 lines, labelled "NOT part of the measurement". | keep |
| `__init__.py` | package marker | keep |

### `bench/` — eval harness
| File | Purpose | Verdict |
|---|---|---|
| `dataset.py` | the one place the seen/exemplar split is defined | keep |
| `run_cell.py` / `run_grid.py` | one cell / full grid, resumable, append-only | keep |
| `rescore.py`, `backfill_tokens.py`, `backfill_tokens_v3.py` | post-hoc rescoring / token passes | keep |
| `analyze.py`, `summary.py`, `figures.py` | V2 figures/tables. Superseded for V3 by `audit/figures_v3*.py`, but they are the provenance of the committed V2 figures. | keep, label V2-only |
| `peek.py` | prints rows of a results CSV | keep |
| `build_seen_v2.py` | builds `seen_v2` **and** `held_out_v2` together — where the firewall is established | keep |
| `diag_*.py` (9) | targeted diagnostics | see §4 |
| `llm_client.py` | **demo** LLM client for the Isaac/MuJoCo bridge. Not the harness. | see §3 |
| `results_v4*.csv.manifest.json` (5) | argv + env + model digests + package versions per V4 run. The CSVs are gitignored (219 MB); these are the committed provenance. | keep |
| `__init__.py` | package marker | keep |

### `ft/` — rungs 3 and 4
`prep_data.py` (SFT data built through `llm.prompts.build_prompt`, so the fine-tune sees
the harness prompt), `train_lora.py`, `merge_and_export.py` (adapter -> GGUF -> Ollama),
`train_classifier.py`, `eval_classifier.py` (labelled "NOT a reported number"),
`data/*_sft_*.jsonl` (4), `runs/**/dev_metrics.json` + `history.json` + `label_maps.json`.
Weights gitignored, metrics tracked — invariant 5 holds. **All keep.**

### `sim/`
| File | Purpose | Verdict |
|---|---|---|
| `workspace.py` | the one place a location symbol becomes a coordinate | keep |
| `sim_backend.py` | V1 MuJoCo backend, 8 V1 locations, lifts over obstacles | keep |
| `full_grasp_backend.py` | V2 MuJoCo backend, real grasp/lift, V1 names + 9 grid cells | keep |
| `reachability.py` | 9x9x3 sweep -> `data/reachability.json` | keep |
| `run_demo.py` | current unified MuJoCo demo (keyboard/voice, model+strategy flags) | keep, **fix §3.2** |
| `llm_demo.py` | earlier typed-only demo. Not dead: `run_demo` imports `drive`, `describe`, `print_legend`, `draw_box_markers` from it. | keep |
| `watch_full.py` | 14-line visual loop over `FullGraspBackend` | keep |
| `test_coords.py`, `test_move_to.py`, `test_obstacles.py`, `test_verbs.py` | V1 `SimBackend` checks, 6-27 lines each, no framework | keep |
| `isaac_backend.py` | Isaac Sim 6.0 backend, same 7 verbs, real carried objects, live stage pose | keep |
| `isaac_bridge.py` | Isaac-side file-IPC watcher (`bridge/inbox` -> `bridge/outbox`), env `isaac6` | keep |
| `mujoco_bridge.py` | MuJoCo-side mirror of the same protocol, env `thesis`, no GPU | keep |
| `isaac_demo_scene.py` | Stage A: layout sanity render | keep (scaffold provenance) |
| `isaac_scene_headless.py` | headless scene + fixed camera | keep (scaffold provenance) |
| `isaac_scene_window.py` | windowed full-Kit scene (records the absolute-`.kit`-path fix) | keep (scaffold provenance) |
| `isaac_grasp_test.py` | Stage B gate: grasp + lift + place. Records the "open the gripper after `world.reset()`" fix. | keep |
| `reachability.json` | **duplicate** — see §2.1 | **delete** |

The four Isaac scene/scaffold files have zero importers and are superseded by
`isaac_backend.py`. Each is a standalone `python sim/x.py` entrypoint whose docstring
records a specific Isaac gotcha, so they read as documentation-by-runnable-example rather
than dead code. Keeping them is cheap. If you want them gone, `isaac_scene_headless.py` and
`isaac_demo_scene.py` are the two whose lessons are fully absorbed into `isaac_backend.py`.

### `asr/`
`record_test.py` (push-to-talk, mic selected by name), `transcribe_test.py` (offline
Whisper, records the 169s -> 0.4s fix). Both keep. `clip.wav` gitignored.

### `data/`
V1 `seen`/`held_out`; `seen_v2` -> `train_v2` + `dev_v2`; sealed `held_out_v2`,
`held_out_v3` (217 rows, current KPI), `held_out_v4b`, `held_out_v4c`;
`location_phrases_v2.json`; `reachability.json`. All tracked via `!` exceptions.
Plus `seen_v2_backup_203.jsonl` — untracked, see §5.

### `audit/`
Reports: `AUDIT_REPORT.md` (2051 L, hostile-reviewer pass), `DEFENCE_NUMBERS.md`,
`PREREGISTRATION_v4.md` (written before V4 ran), `V4_RESULTS.md`,
`MODEL_PROVENANCE.txt` (Ollama blob digests), `vocab_conflicts.md`, `confusion_summary.txt`.

Scripts: `b_checks.py`, `b1_leniency.py`, `build_v4b.py`, `build_v4c.py`,
`check_dispatch.py`, `check_legend_v5.py`, `cls_margins.py`, `confusion.py`,
`count_unique.py`, `escalation_sim.py`, `estimate_v4_runs.py`, `figures_v3.py`,
`figures_v3_style.py`, `figures_v4.py`, `lora_collapse.py`, `lora_vocab_test.py`,
`numctx.py`, `probe_tokenizers.py`, `reconstruct_prompts.py`, `unseen_exact.py`,
`v4_results.py`, `verify_legend_v4.py`, `vocab_conflicts.py`.

Derived data: `cls_margins.csv`, `escalation_sim.csv`, `v4_cells.csv`,
`lora_vocab_test.csv`, `confusion_*.csv` (6), `figures/` (2.6 MB: 8 V4 figures + 4 V3-style).
`discarded/` — see §5.

**All keep.** The tracked derived CSVs and figures look redundant but are not: their inputs
are the 219 MB of gitignored results CSVs, so these are the only committed trace of the
measured numbers. Same argument as the manifests.

### Empty
`vision/`, `rl/`, `real/`, and the `.gitkeep` files in `bench/`, `llm/`, `sim/`, `schema/`,
`asr/` — deliberate placeholders, keep.

`scripts/` — empty, not even a `.gitkeep`, so it does not exist in git at all. Delete the
local directory and the `map.md` paragraph about it.

---

## 2. Duplicates and dead files

### 2.1 `sim/reachability.json` — DELETE

Confirmed duplicate, but not a byte-identical one, which is why it survived:

```
data/reachability.json   25,760 bytes   summary.wall_time_s: 11.9
sim/reachability.json    24,431 bytes   summary.wall_time_s: 18.1
```

All 81 cells are **identical**, field for field (`x`, `y`, `reach_ok`, `reach_steps`,
`place_ok_rate`, `gap_cm_mean`, `gap_cm_max`, `grasp_tries_mean`, `reasons`, `legal`).
`params`, `table_extent` and every `summary` key match. The only differing value in the
whole file is `wall_time_s`. It is a second run of the same sweep, written before `--out`
defaulted to `data/reachability.json`.

Nothing reads it. Every reference in the repo — `sim/reachability.py:21`, `:99`, and four
places in `map.md` — names `data/reachability.json`.

Note: it is **tracked** (committed in `aea20ca`), so it is no longer the untracked file an
earlier session saw. It got committed since. Delete it; the sweep is reproducible and
`data/` holds the copy the pipeline actually consumes.

### 2.2 `sim/voice_pipeline.py` — ALREADY GONE

The file does not exist. `grep -r voice_pipeline` across every `.py` and `.md` in the repo
returns nothing: no import, no doc reference, no dangling `map.md` entry. Closed, no action.

### 2.3 `audit/full_pass.{txt,err}` — DELETE

`full_pass.txt` (4.4 KB) is captured stdout from `audit/demo_full_pass.py`: an
8-client x N-command table of parsed actions, stages and latencies. `full_pass.err` (643 B)
is captured stderr — a HuggingFace weight-loading progress bar and a DistilBERT
`UNEXPECTED key` report, with raw ANSI escape codes still embedded.

Neither is referenced by anything. The producing script is tracked and rerunnable. The
timings in `full_pass.txt` are demo-path timings the codebase explicitly forbids quoting.
Delete both; keep `demo_full_pass.py`.

### 2.4 `bench/diag_runs.py` — DELETE

Reads `bench/results_heldout.csv` and `bench/results_seen.csv`. The second does not exist
and is not produced by anything in the repo — the seen-grid output is
`bench/results_grid.csv`. This script cannot have run since that rename. Its stated question
("did the seen and held-out grids run under different conditions?") is answered by
`diag_coldstart.py` and `diag_variation.py`, which both read `results_grid.csv` and work.

---

## 3. The two client paths

`bench/llm_client.py` (621 L) vs `sim/run_demo.py` (450 L).

### 3.1 What they actually do

**They do not both call `run_once()`.** `sim/run_demo.py` imports `llm.pipeline.run_once`.
`bench/llm_client.py` imports nothing from `llm/`, `schema/` or `bench/` — it has its own
`run_one()` (note the different name) and a private reimplementation of the entire
measurement path. Side by side:

| Stage | `sim/run_demo.py` | `bench/llm_client.py` |
|---|---|---|
| Client roster | `llm.clients.all_clients()` — 8 clients, incl. LoRA + classifier | hardcoded `MODELS` dict — 5 + a `rules` fallback |
| Prompt | `llm.prompts.build_prompt`, legend v3/v4/v4lite/v5 via `bench.run_cell.apply_legend` | private `build_prompt()`, one inline `SCHEMA_TEXT` |
| Few-shot source | `bench.dataset.load_seen_split` over `seen_v2.jsonl` | 4 hardcoded pairs in `FEW_SHOT_COMMON` |
| RAG | `llm.retrieve.Retriever` — sentence-transformers, corpus `train_v2.jsonl` | token-overlap Jaccard over `seen.jsonl` |
| JSON extraction | `llm.extract.extract_json` — takes the **last** balanced object | private `_extract_json` — **refuses** when >1 object parses |
| Validation | `schema.validate_object` against `schema_v2.json` | private `validate()` against an inline `FIELD_MASK` |
| Object key | **`action`** | **`verb`** |
| Location vocabulary | 14: `box_1..3`, `grid_A1..C3`, `max_reach`, `nearest_free_cell` | isaac: `box_1..3`, `handover`, `cell_0_0..2_2`; mujoco: `box_1..3`, `table_center/left/right/front/back` |
| Constrained decoding | yes, via `llm.schema_json` | no |
| Execution | in-process `FullGraspBackend` | file IPC: `bridge/inbox` -> `bridge/outbox` |
| Extra | — | `--selftest` gate, `--script` batch, `rules` fallback |
| Voice | Whisper `base`, push-to-talk (Enter to start, Enter to stop) | Whisper `small.en`, fixed 5 s window |

### Can they merge?

**Two rows in that table are load-bearing and cannot merge. The rest can.**

**Blocker 1 — the location vocabulary.** Neither of `llm_client.py`'s `BACKENDS` entries is
`schema_v2.json`. The Isaac entry uses `cell_R_C` with 0-2 indices plus a `handover` point;
`schema_v2` uses `grid_A1..C3` and has no `handover`. The MuJoCo entry uses the V1 `table_*`
names, not the V2 grid at all. So `schema.validate_object` cannot validate an Isaac action,
and `llm/prompts.py` cannot prompt for one, without adding a third vocabulary to
`schema_v2.json` — invariant-4 territory, and it would touch the provenance of the 30,380
measured rows. **Do not do that.** The file's own docstring is right that prompting one
backend's vocabulary while the other executes is the silent-mismatch failure the V4 runs
were about; two vocabularies is the correct answer, the split is deliberate.

**Blocker 2 — the extraction policy, and this is the subtle one.** The two extractors
implement *opposite* rules for the same input, and both are deliberate:

- `llm/extract.py`: scan every balanced `{...}` and take the **last** one that parses.
  Documented reason: the cot prompt puts the answer on the final line, so earlier braces are
  reasoning fragments or schema echoes.
- `llm_client.py:_extract_json`: collect every candidate carrying a `verb` key and **raise
  if there is more than one**. Documented reason: multiple actions is ambiguity, not a menu;
  small models emit 3-5 step plans and any positional rule picks a wrong-but-legal action
  some of the time.

Both are correct for their context — a benchmark must score every row, a demo must not move
the arm on a guess. Merging them would change either the measured numbers or the demo's
safety property. **Keep both, and cross-reference them in each docstring**, because this is
precisely the divergence that will rot: someone patches one extractor for a new model's
output shape and the two silently classify the same reply differently.

What *can* merge, at zero risk to any measured number:

1. **Model roster.** Replace the hardcoded `MODELS` dict with `llm.clients.all_clients()`,
   which is what `run_demo.py` already does and for a documented reason (a bare
   `startswith("gpt-")` dispatch used to silently produce an `OllamaClient` for
   `classifier:distilbert`, failing at call time instead of selection time).
   `llm_client.py` currently cannot reach rungs 3 or 4 at all — the Isaac demo can only
   show rungs 1 and 2.
2. **`validate()`'s constant tables.** Keep the function (it must stay vocabulary-aware),
   but build `VERBS`, `TARGETS` and `FIELD_MASK` by reading `schema_v2.json` rather than
   retyping them. `FIELD_MASK` and `schema_v2.json:verbs.*.required` are the same table
   typed twice today, in two files, with no check that they agree. Only `locations` stays
   per-backend.
3. **Voice front-end.** Both files carry a Whisper block. `run_demo.py`'s push-to-talk is
   strictly better than `llm_client.py`'s fixed 5-second window, and `run_demo.py` also has
   the offline-cache handling. One shared `listen()`.
4. **Mic selection.** `_find_mic` and `resolve_input_device` are the same
   select-by-name-substring-with-default-fallback logic, written twice.

That removes roughly 150 of `llm_client.py`'s 621 lines and, more importantly, makes the
remaining divergence explicit and intentional rather than incidental.

**Recommendation: merge items 1-4; leave the vocabulary and the extraction policy split;
add a one-line pointer in each file's docstring naming the other and why they differ.**
Both files already carry the right warning ("this is the DEMO path, not the measurement
harness"), so the risk is not that a demo number gets quoted into the thesis. The risk is
the two extractors and the two field masks drifting until the demo reports a different
failure classification than the harness for the same model output.

### 3.2 Bug found while diffing: `sim/run_demo.py` crashes on its own default

[sim/run_demo.py:322-328](sim/run_demo.py#L322-L328) uses `schema`:

```python
if args.constrain == "on":                               # default IS "on"
    from llm.schema_json import to_json_schema, to_flat_json_schema
    if client.name.startswith("openai:"):
        client.json_schema = to_flat_json_schema(schema)     # line 326
    elif client.name.startswith("ollama:"):
        client.json_schema = to_json_schema(schema)          # line 328
```

`schema` is not assigned until line 363, in the same function:

```python
schema = load_schema(os.path.join(root, "schema", name))     # line 363
```

Same scope, so this is an `UnboundLocalError`, not a fallback to a module global.
`--constrain` defaults to `on`, so `python -m sim.run_demo` dies right after the menus —
before the viewer opens — for any OpenAI or Ollama client. That includes `--default`, which
resolves to `openai:gpt-4o-mini`. Only `classifier:distilbert` survives, because neither
branch fires for it.

Fix: move the constrained-decoding block below the `load_schema` call (after line 364). A
one-line move, no logic change.

Why nothing caught it: the three `audit/demo_*_check.py` scripts that bind to this file
import `build_roster` and `strategies_for` and then reimplement stage 2 themselves. Nothing
in the repo calls `run_demo.main()`.

---

## 4. Superseded scripts — which are stale

All nine `bench/diag_*.py` and all sixteen `scratch_*.py` were checked against their input
files on disk.

### Actually broken (input does not exist)
- **`bench/diag_runs.py`** — `bench/results_seen.csv`. Delete (§2.4).

That is the only one. Every other script's inputs are present.

### Runs, but only against V1/V2 data — scope-limited, not stale
These read files that still exist and still matter (V1/V2 numbers are cited), but none of
them will tell you anything about V3 or V4:

| Script | Reads | Note |
|---|---|---|
| `bench/diag_cells.py` | `results_grid_v2_scored.csv`, `results_heldout_v2_scored.csv` | the V2 phrase/legend collision — the finding that motivated V4 |
| `bench/diag_rowword.py` | same two | V2 row-vocabulary effect; superseded in substance by `audit/lora_vocab_test.py` on V3 |
| `bench/diag_score_edges.py` | `results_grid_v2.csv`, `results_heldout_v2.csv` | scoring edge cases; provenance of the frozen policy |
| `bench/diag_cold_accuracy.py` | `results_grid.csv`, `results_heldout.csv` | V1 |
| `bench/diag_coldstart.py` | `results_grid.csv` | V1 |
| `bench/diag_variation.py` | `results_grid.csv` | V1 |
| `bench/analyze.py`, `bench/figures.py` | V2 scored CSVs | provenance of the committed V2 figures. `audit/figures_v3.py` says explicitly it was written fresh rather than extending `figures.py`, and that `figures.py` is not broken. |

**Keep all seven.** Add one line to each docstring: *"V1/V2 only. V3 -> `audit/figures_v3.py`,
V4 -> `audit/figures_v4.py`."* They are the only record of how the V1/V2 numbers were
produced, and deleting them breaks the audit trail `AUDIT_REPORT.md` rests on.

`bench/diag_determinism.py` calls Ollama live and reads no file — still runs. Keep.
`bench/diag_legend.py` reads `seen_v2.jsonl` + `schema_v2.json` — current. Keep.

### `scratch_*.py` — all 16 still run
- **V3-current** (7): `scratch_gatecheck`, `scratch_labelspace`, `scratch_modifiers`,
  `scratch_phrasing_inventory`, `scratch_build_container_rows`, `scratch_matrix_integrity`,
  `scratch_rung1_integrity`. All read `held_out_v3.jsonl` / `train_v2.jsonl` /
  `results_heldout_v3_*.csv`, all present. Keep.
- **Environment probes** (5): `scratch_api`, `scratch_paths`, `scratch_tmpl`,
  `scratch_ckpt`, `scratch_merge_check`. Print `transformers`/`peft` signatures and
  checkpoint contents. They still work, but they answer questions
  `requirements.lock.txt` now pins. Keep — they are the provenance of the training-config
  choices.
- **Smoke inspectors** (4): `scratch_smoke`, `scratch_clscheck`, `scratch_counts`,
  `scratch_scored_table`. All inputs present. `scratch_scored_table.py` carries one of the
  most valuable docstrings in the repo — why invalid-output columns must key off `stage` and
  never `category` (9,536 rows with `stage != validated` vs 1,004 carrying `category
  invalid:*`; keying on `category` hides 8,532 structural failures). Keep.

### `audit/demo_*_check.py` — not throwaway, and one merge
The earlier session's note that these were throwaway is half right. Three of the five bind
to `sim/run_demo.py`'s public surface and are the only automated cover that file has.

| Script | What it guards | Verdict |
|---|---|---|
| `demo_examples_check.py` | that `few_shot`/`rag` are not silently dead in the demo — they were: `run_demo` hardcoded `examples = None` and `build_prompt` raises without them, so `run_once` recorded `prompt_error` and the arm never moved with no error shown | **keep** — real regression guard |
| `demo_firewall_check.py` | that the demo's retrieval matches the measurement's (`CORPUS_PATH`, `DEFAULT_K`, `load_seen_split`) — guards invariant 2 | **keep** |
| `demo_full_pass.py` | imports `sim.run_demo.build_roster`, walks **all 8** clients through stage 2 | **keep** |
| `demo_stage2_check.py` | same imports, same shape, but only "the three NEW demo clients" — builds `{c.name: c for c, _ in build_roster()}` and indexes three of them | **merge into `demo_full_pass.py`**, or delete: `demo_full_pass` already covers everything it does |
| `demo_smoke_v5.py` | 8 clients x 10 commands under V5 legend + constrained decoding — the only exercise of the constrained-decoding path anywhere in the repo | **keep** |

Worth noting alongside §3.2: none of these calls `run_demo.main()`, which is exactly why the
`UnboundLocalError` on the demo's own default path went unnoticed.

---

## 5. Tracking

### Tracked, should not be
| File | Why |
|---|---|
| `sim/reachability.json` | duplicate (§2.1) |
| `audit/full_pass.txt` | captured stdout of a tracked, rerunnable script (§2.3) |
| `audit/full_pass.err` | captured stderr — ANSI escapes and a HF progress bar (§2.3) |

Nothing else. In particular, the tracked derived CSVs and PNG/PDF figures under `audit/`
(2.6 MB) **should stay tracked**: their inputs are gitignored, so they are the only
committed evidence of the measured results. Same for the five
`bench/results_v4*.csv.manifest.json` files, which preserve argv, dataset, schema version,
exemplar source, env vars, Ollama model digests, package versions, Python and platform.

### Gitignored, should be tracked
| File | Why |
|---|---|
| `data/seen_v2_backup_203.jsonl` | 31 KB. `map.md` documents it as *"pre-gate backup of seen_v2 (203 rows, before the container-row expansion that took held_out_v3 to 217); kept for provenance, not read by any code."* A provenance artifact that exists on exactly one machine is not provenance. It is caught by the blanket `data/*`; add `!data/seen_v2_backup_203.jsonl` next to the other exceptions. |

Everything else gitignored is correctly gitignored: 219 MB of results CSVs, 2.2 GB
`ft/export/`, 308 MB `out/`, `bridge/` (runtime IPC scratch), `.env`, `__pycache__`, `*.wav`.

### `results_v4a_local.CONTAMINATED.csv` — already correct, no action

Current state:

```
audit/discarded/results_v4a_local.CONTAMINATED.csv            8.9 MB, gitignored
audit/discarded/results_v4a_local.CONTAMINATED.manifest.json  tracked
audit/discarded/README.md                                     tracked
```

This is quarantine, not deletion, and it is done properly:

- the file is out of `bench/`, so no analysis script's glob can reach it;
- the `.CONTAMINATED` infix makes it unmistakable in any listing;
- `README.md` records the mechanism (two concurrent `run_grid` processes appending to one
  output; 6,944 rows = exactly 2 x 3,472; **775 of 3,472 duplicated keys, 22.3%, disagree on
  `pred_json`**), why de-duplication cannot rescue it (no principled basis for choosing
  between two answers), the latency evidence for Ollama weight-thrashing, and the harness
  gap it exposed (`run_grid.py` has no lockfile on `--out` — append-only resume is safe
  against a crashed run, not a concurrent one);
- the manifest preserves the full provenance of the discarded run.

The CSV being gitignored is consistent with every other 8 MB results file in the repo.
**Leave it exactly as it is.** One optional improvement: `audit/discarded/*.csv` is
gitignored by pattern, so a future discarded run will land there silently. That is fine, but
`discarded/README.md` should become the index for the directory rather than the writeup of
one run.

---

## 6. `map.md` drift

### Referenced in `map.md`, missing from disk
**None.** Every path `map.md` names exists, including `data/seen_v2_backup_203.jsonl`,
`ft/export/`, and `scripts/` — which it correctly describes as an empty leftover with the
standing suggestion to delete it or give it a purpose.

### On disk, missing from `map.md`
Filtering out files covered by `map.md`'s glob descriptions (`scratch_*.py`, `diag_*.py`,
`test_*.py`, `results_*.csv`, `figures/`), the real gaps are:

1. **The entire `audit/` directory** — 23 scripts, 7 reports, 6 confusion CSVs, 12 figures,
   the `discarded/` quarantine. This is the largest body of work in the repo and `map.md`
   does not mention that it exists. The `AUDIT_REPORT.md` / `DEFENCE_NUMBERS.md` /
   `PREREGISTRATION_v4.md` trio in particular is what a viva would be run from.
2. **The whole Isaac Sim backend** — `isaac_backend.py`, `isaac_bridge.py`,
   `isaac_demo_scene.py`, `isaac_grasp_test.py`, `isaac_scene_headless.py`,
   `isaac_scene_window.py`. `map.md`'s header still says "a MuJoCo simulator as the current
   backend", singular. There are two, in two conda envs (`thesis` / `isaac6`).
3. **The bridge architecture** — `sim/mujoco_bridge.py`, `sim/isaac_bridge.py`,
   `bench/llm_client.py`, and the gitignored `bridge/inbox|outbox|done` protocol. A reader
   of `map.md` has no way to know the demo runs as two processes that never share an
   interpreter.
4. **`bench/llm_client.py`** — the second client path in §3, undocumented. Its existence is
   the single most surprising thing in the repo for a new reader.
5. **`llm/prompts_v4.py`, `llm/prompts_v5.py`, `llm/schema_json.py`** — `map.md` lists
   `llm/` file by file and stops at V3. Legend selection and constrained decoding are both
   invisible.
6. **V4 datasets** — `data/held_out_v4b.jsonl`, `data/held_out_v4c.jsonl`. `map.md` says
   data is "versioned v1 -> v3".
7. **`requirements.txt` / `requirements.lock.txt`** — including the F-ENV provenance
   warning, which a reader should meet before trusting any pin.
8. `README.md`, `bench/backfill_tokens_v3.py`, and the `__init__.py` package markers in
   `bench/`, `llm/`, `schema/`, plus `ft/data/*_sft_*.jsonl`.

`map.md`'s KEY INVARIANTS section is still accurate — all five hold, verified against the
code. The drift is entirely in the directory guide, which stopped being updated around the
V3 measurement.

Suggested additions: an `audit/` section; an Isaac subsection under `sim/`; a short
"two-process demo bridge" paragraph; a **sixth invariant** covering the demo-vs-measurement
firewall (both `llm_client.py` and `mujoco_bridge.py` already carry that warning in prose,
and §3 shows it is the one most at risk of erosion — it belongs in the invariant list); and
a "versioned v1 -> v4" correction.

---

## 7. Cosmetic

20 tracked `.py` files start with a UTF-8 BOM: `bench/analyze.py`,
`bench/backfill_tokens.py`, `inspect_smoke.py`, `llm/extract.py`, `llm/score.py`, and all 16
`scratch_*.py`. Python tolerates it; `git diff` and some editors render it as `ï»¿`. Strip
when you next touch each file — not worth a dedicated commit.

`.gitignore` lists `.env` twice (line 3 and the last line). Harmless.

Line endings are correct. `git ls-files --eol` shows `i/lf w/crlf attr/text=auto eol=lf`
throughout, which is `.gitattributes` doing its job on Windows, not a problem.
