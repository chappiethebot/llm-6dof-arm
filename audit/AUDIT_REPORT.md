# Repository audit, 13 September 2026

**Revision 2 — 17 September 2026.** Findings 24–32 added; verdict, fix list and
"what I could not verify" updated for the 17 September scope freeze. The
original 13 September read-only pass at `093c1ee` is unchanged below; new
material is marked **[R2]**.

Read-only pass at `093c1ee`. Nothing in the repo was modified; no model was
called; no file under `data/held_out_*` or `bench/results_*` was written.
Analysis scripts were written to a scratch directory outside the repo.

> **This file replaced the previous V3-era `audit/AUDIT_REPORT.md`** (110 KB,
> 2,051 lines, the hostile-reviewer pass whose `F-ENV` finding
> `requirements.txt` still cites). That document is committed and recoverable
> with `git show 093c1ee:audit/AUDIT_REPORT.md`. Recover it before writing, or
> the F-ENV reference dangles.

---

## Verdict

The measurement itself is sound: the `held_out_v3` firewall holds at zero exact
and zero near-duplicate overlap, the LoRA adapters and the classifier trained
only on `train_v2`/`dev_v2`, the V5 manifests match their CSVs exactly, and the
12,152 rows regroup losslessly with zero verdict drift. **But the repo is not
currently defensible as thesis evidence, for two reasons that are both
fixable in a day.** First, every one of the 12,152 measured rows is excluded
from version control by `.gitignore:13`, and the only copy of the V5 result set
outside `bench/` is an untracked 4.8 MB zip sitting in the same working tree —
one `git clean -xdf` destroys the thesis. Second, two of the five headline
numbers are computed on a denominator that silently includes a cold-start
warmup row, `audit/figures_v3.py` computes the same quantity on a different
denominator, and no committed document states any V5 number at all.

**[R2] A third reason, which the 13 September pass missed entirely.** The audit
checked *training* leakage exhaustively and found none. It never asked whether
the held-out set had been *reused*. It has — three times, and the V5 legend was
authored to fix failures observed on it. See finding 24. That is the single
most likely line of attack at the viva and no committed document discloses it.

**[R2] A fourth, structural.** The thesis's core contribution is a
cost/accuracy Pareto plot, and the winning cell has no latency or cost figure in
any committed document. See finding 25.

---

## Findings

| # | Severity | Area | Finding | Evidence | Confidence |
|---|---|---|---|---|---|
| 1 | BLOCKER | E Risk | All 12,152 measured rows are gitignored. Zero result CSVs are tracked; the sole backup is an untracked zip in the same tree | `.gitignore:13`; `git ls-files bench/*.csv` → empty; `v5_measurement_backup.zip` is `??` in `git status` | [Certain] |
| 2 | BLOCKER | C Scoring | `summary.py` includes the warmup row in accuracy but excludes it from latency. For 31 of 36 cells the warmup row is `command_id 0`, the *only* observation of that command | `bench/summary.py:176` (no filter) vs `:208` (filters); `bench/run_cell.py:275` | [Certain] |
| 3 | BLOCKER | C Scoring | Two definitions of strict accuracy coexist and disagree. Classifier = 74.7% in `summary.py`, 75.0% in `figures_v3.py` | `bench/summary.py:161-162` vs `audit/figures_v3.py:97-99`; computed 162/217 vs 162/216 | [Certain] |
| 4 | BLOCKER | F Docs | `DEFENCE_NUMBERS.md` is entirely V3-era. Its headline "best system 0.924" and "3b-lora 0.654" contradict V5's 0.938 and 0.959 | `audit/DEFENCE_NUMBERS.md:23-29` vs recomputed V5 | [Certain] |
| 5 | BLOCKER | A Repro | No committed file states any of the five published V5 numbers, and no committed script produces the four `results_v5_*_scored.csv` that everything downstream reads | `git grep "95.9\|93.8\|74.7\|12152"` → no hits; only `backfill_tokens_v5.py:70` mentions them, as *input* | [Certain] |
| 6 | HIGH | F Docs | `DEFENCE_NUMBERS.md` "spatial reasoning 0.970" matches neither grid-only definition in the data (0.9605 cells-only, 0.9137 non-container). 0.9722 is the *container* accuracy | `audit/DEFENCE_NUMBERS.md:25` vs `audit/figures/cell_stats.csv:2` and recomputation | [Certain] claim is wrong; [Likely] it is the container figure |
| 7 | HIGH | E Risk | The V3 measurement (30,380 rows, the basis of every number in `DEFENCE_NUMBERS.md`) has no manifest at all | `bench/results_heldout_v3_*.csv` have no `.manifest.json`; V4/V5 do | [Certain] |
| 8 | HIGH | B Data | `held_out_v4b.jsonl` shares 56 exact commands with `train_v2` and 72 with `seen_v2`. It is named "held_out" but is 26% seen | computed overlap matrix; by construction (`audit/build_v4b.py` rewrites into training vocabulary) | [Certain] |
| 9 | HIGH | A Repro | The four tracked `bench/figures_v5/*.png` were produced by `analyze.py`, which draws a SEEN-vs-HELD-OUT comparison — but V5 measured only `held_out_v3`. What was passed as `--seen` is unrecorded | `bench/analyze.py:177-180` names exactly those four files; `bench/figures_v5/` committed in `093c1ee`; all 12,152 rows are `dataset=held_out_v3` | [Certain] the gap; [Guessing] what was passed |
| 10 | HIGH | C Scoring | No confidence intervals exist for any V5 number. `clustered_ci` is correct but hardcoded to V3 files | `audit/figures_v3.py:92, 104-125`; no V5 equivalent in `git ls-files` | [Certain] |
| 11 | HIGH | D Version | `21 of 27` is still wrong in two files after being corrected in two others | `audit/DEFENCE_NUMBERS.md:78`, `audit/V4_RESULTS.md:121`; corrected in `audit/V3_VS_V4.md:316`; `v4_cells.csv` → 7 real / 20 noise of 27 | [Certain] |
| 12 | MEDIUM | Trap 1 | `analyze.py` never calls `csv.field_size_limit`; `results_v5_all.csv` has a 286,686-char field vs the 131,072 default | `bench/analyze.py:22, 49`; measured max field | [Certain] |
| 13 | MEDIUM | Trap 2 | `analyze.py` defaults to V1 files that still exist on disk, so a bare invocation silently analyses July data | `bench/analyze.py:168-169`; both files present, dated 2026-07-20/21 | [Certain] |
| 14 | MEDIUM | Trap 5 | `llm_client.py` restricts LoRA to `zero_shot`; LoRA was measured on all five strategies, and `role_based` scored 0.949 | `bench/llm_client.py:220, 249-250` vs `results_v5_qwen3.csv.manifest.json` cells | [Certain] |
| 15 | MEDIUM | Trap 4 | `run_cell.py` writes `tokens_in`/`tokens_out`/`cost_usd` as empty strings; only the backfill populates them | `bench/run_cell.py:322` | [Certain] |
| 16 | MEDIUM | G Quality | `run_grid.py` silently discards the entire prior manifest history if the JSON is unparseable | `bench/run_grid.py:166-169` | [Certain] |
| 17 | MEDIUM | Trap 3 | `backfill_tokens_v5.py` is a 263-line copy of `_v3.py` differing in 2 functional lines; its docstring still says "backfill_tokens_v3.py … held_out_v3" | `bench/backfill_tokens_v5.py:3, 60, 70-71` | [Certain] |
| 18 | MEDIUM | Trap 6 | Isaac has no reset/respawn verb. `dispatch()` handles the 7 schema verbs only | `sim/isaac_backend.py:627-644`; `sim/isaac_bridge.py:107-113` | [Certain] |
| 19 | MEDIUM | E Risk | `ft/merge_and_export.py` hardcodes `C:\Users\USER\llama.cpp` | `ft/merge_and_export.py:35` | [Certain] |
| 20 | LOW | Trap 7 | `latency_s` excludes retrieval, as documented. `summary.py` reports the sum correctly; `analyze.py` and `figures.py` do not | `bench/run_cell.py:44-47, 104, 118`; `bench/summary.py:341-353`; `bench/analyze.py:103` | [Certain] |
| 21 | LOW | B Data | 12 V1 `seen.jsonl` commands appear in `held_out_v3`; none is in `train_v2` | computed; all are "middle of the table" phrasings | [Certain] |
| 22 | LOW | D Version | 22 result CSVs on disk have no manifest; `results_v5_all.csv` / `_slim.csv` are undocumented intermediates | file listing | [Certain] |
| 23 | LOW | F Docs | `docs/SESSION_2026-09-07.md:152` cites `session_record_v3.md`, which is not in the repo | `find . -iname "*session_record*"` → nothing | [Certain] |
| **24** | **BLOCKER** | **B Data [R2]** | **`held_out_v3` has been measured three times (V3, V4c, V5) and the V5 legend was authored to fix failures seen on it. Zero training leakage, but the sealed-set guarantee is spent. No committed document discloses this** | `results_heldout_v3_*`, `results_v4c_lora`, all 8 V5 manifests all name `held_out_v3`; V5 legend design rationale in `session_record_v3.md §2.15`; plan v6 don't-do list: "❌ Tune on the held-out set" | [Certain] |
| **25** | **BLOCKER** | **F Docs [R2]** | **The V5 winning cell `qwen2.5-3b-lora/zero_shot` has no latency or cost figure in any committed document. The Pareto plot's top-right point cannot be placed** | `DEFENCE_NUMBERS.md` rev 1 Pareto table is V3-only; `git grep` finds no V5 latency for any LoRA cell | [Certain] |
| **26** | **HIGH** | **C Scoring [R2]** | **The V5 headline margin (0.9583 vs 0.9382 = 2.0 points) is below the thesis's own stated ~6-point claimability threshold, and finding 10 means it cannot currently be tested** | `DEFENCE_NUMBERS.md` rev 1 §6 states median CI half-width 5.8 pts; V5 cell values recomputed | [Certain] |
| **27** | **MEDIUM** | **D Version [R2]** | **V3 row count is stated as 30,352 in `DEFENCE_NUMBERS.md` §2 and 30,380 in this report §1 and §7. 28-row discrepancy, unreconciled** | both documents | [Certain] |
| **28** | **MEDIUM** | **E Risk [R2]** | **Git history is clean of secrets, but a `.env` containing a live `OPENAI_API_KEY` was placed on the demo USB drive. Git hygiene does not cover physical media** | `docs/DEMO_PORTABLE.md` describes the USB contents; `.gitignore:4, :47`; `git log --all -- .env` empty | [Likely] the copy happened; [Certain] git is clean |
| **29** | **MEDIUM** | **G Quality [R2]** | **Three open engineering decisions from `SESSION_2026-09-07.md §4` were never carried into this audit: `MOVE_TOL = 0.02` is unreachable (best observed 0.029) so `move_to` reports `ok: false` every time; `nearest_free_cell` occupancy handling may differ between MuJoCo and Isaac; `run_demo.py` and `asr/voice.py` may pass different Whisper models (`small.en` vs `base`)** | `SESSION_2026-09-07.md:180-193`; `sim/workspace.py`; `llm_client.py` vs `asr/voice.py` | [Certain] they are open; [Likely] the Whisper mismatch is live |
| **30** | **MEDIUM** | **F Docs [R2]** | **Field-level escalation (`session_record_v3.md §2.14`) appears in no not-reportable list in the repo, but rests on an untested assumption (cloud location accuracy unchanged when only the location is asked) and an unverified novelty claim** | `session_record_v3.md §2.14`; `audit/escalation_sim.csv` | [Certain] |
| **31** | **MEDIUM** | **Scope [R2]** | **Phase 6 (vision) and Phase 2.5 (RL backend) were formally cut in writing by the supervisor on 17 September 2026. Finding "vision zero lines" is now a scope decision, not a gap — but the cover page still promises vision-based manipulation** | supervisor's written instruction, 17 Sep 2026; thesis title | [Certain] the cut; [Likely] the title is unreconciled |
| **32** | **LOW** | **F Docs [R2]** | **The `_scored` suffix on the four V5 files is a misnomer — they carry `run_cell.py`'s original verdicts unchanged, so `rescore.py` either did not run or changed nothing** | 12,152 rows agree, 0 differ, raw vs scored | [Certain] the equality; [Guessing] which |

---

## Detail

### 1. BLOCKER — the entire measurement is one `git clean` from gone

`.gitignore:13` is `bench/results_*.csv`. `git ls-files` returns **zero**
tracked CSVs under `bench/`. That covers:

- the V5 set: 18 files, **72.6 MB**, the 12,152 rows behind every number;
- the V3 set: `results_heldout_v3_*.csv`, ~61 MB, 30,380 rows;
- everything else: **302.1 MB across 63 files**.

What *is* tracked is the eight `results_v5_*.manifest.json` files — argv, env,
Ollama digests, package versions. Those let you re-run the experiment. They do
not let you recover the answers, and re-running the cloud cells costs money and
will not reproduce byte-for-byte (gpt-4o-mini was 77.1% byte-identical across
repeats, per the previous session record).

The only other copy is `v5_measurement_backup.zip` — 4.8 MB compressed, 72.6 MB
uncompressed, 26 entries, complete and intact (verified: all 18 V5 CSVs plus 8
manifests). It is **untracked**, in the same working tree, on the same disk.
`git clean -xdf` — the normal way to tidy a repo — deletes both it and every
CSV, with no warning and no recovery.

Total untracked-or-ignored data in the tree is 10.94 GB, dominated by
`ft/runs` (7.9 GB) and `ft/export` (2.3 GB), which are legitimately excluded and
regenerable. The 302 MB of results are not regenerable.

**Fix:** commit the four `results_v5_*_scored.csv` (15.9 MB) plus the classifier
file, or all 18 (72.6 MB) — GitHub's warning threshold is 50 MB per file, and
the largest here is 15.86 MB. Add negation lines under `.gitignore:13` exactly
as `data/` already does. Push. 1 hour, and it retires the highest-severity
category in this report.

**If not fixed:** a disk failure or a routine `git clean` between now and 30
November ends the thesis. There is no second copy anywhere else.

### 2. BLOCKER — the warmup row is inside the accuracy denominator and outside the latency one

`bench/run_cell.py:275` sets `is_warmup = (cid == 0 and run_index == 0)`. For
the 31 local cells, which ran `--repeats 1`, that means **`command_id 0` is
measured once and that single observation is the warmup**. Verified: command 0
(`"put the bottle in the upper left corner of the table"`) is absent from 31 of
36 cells once warmups are dropped; it survives only for `gpt-4o-mini`, which ran
5 repeats.

`bench/summary.py:176` builds its cell with no `is_warmup` filter and
`:161-162` divides by `len(cell)`. `bench/summary.py:208` — the latency
section, forty lines below — *does* filter it out.

So for the classifier, the same row is counted as a legitimate observation when
computing 74.7% accuracy and thrown away as unrepresentative when computing 18
ms latency. That row's latency is **14,651 ms** — the cold model load — against
a warm median of 17.9 ms. It is 818× the median. If it is not representative of
latency it is not representative of accuracy either.

Reproducing the published numbers confirms `summary.py` is the source:

| cell | warmup in | warmup out | published |
|---|---|---|---|
| `qwen2.5-3b-lora` / zero_shot | 208/217 = **0.9585** | 207/216 = 0.9583 | **95.9%** |
| `classifier:distilbert` | 162/217 = **0.7465** | 162/216 = 0.7500 | **74.7%** |
| `gpt-4o-mini` / rag | 1018/1085 = 0.9382 | 1017/1084 = 0.9382 | **93.8%** |

The cloud number is immune because command 0 has four non-warmup repeats. The
two local numbers are not. Note the direction differs by luck: the LoRA warmup
row happened to be correct (inflating 95.8% → 95.9%), the classifier warmup row
happened to be wrong (deflating 75.0% → 74.7%).

**[R2] Second-order consequence, not noted on 13 September.** `command_id 0` is
a *grid* command. Under the warmup-excluded rule the 31 local cells have **131
grid commands, not 132**, so the LoRA's perfect grid score is 131/131 and the
classifier's grid denominator is 131. Any table mixing cloud (132) and local
(131) grid counts in one column is wrong. State the denominator per row.

**Fix:** decide one rule. Either exclude warmups everywhere (then 31 cells are
honestly n=216 and one command is unmeasured for them), or re-run
`command_id 0` for the 31 local cells — cheap, local, no cloud spend. Add the
`is_warmup` filter to `summary.py:176` either way. 2 hours, or 4 with the re-run.

**If not fixed:** an examiner who recomputes from the CSV gets 95.8% and 75.0%
and asks why the thesis says otherwise. The honest answer — "the denominator
includes a cold-start row that the latency figure discards" — is a bad answer to
give live.

### 3. BLOCKER — two live definitions of strict accuracy

The *scorer* is single-sourced and correct: `llm/score.py:134-138`, called once
at `bench/run_cell.py:291` and re-derivable at `bench/rescore.py:130`. The
`strict_match` column is written once and every other script reads it. Good.

The *aggregation* is not single-sourced:

| script | line | warmup | classifier result |
|---|---|---|---|
| `bench/summary.py` | 176 | **included** | 0.7465 |
| `bench/analyze.py` | 64-72 | **included** | 0.7465 |
| `audit/figures_v3.py` | 97-99 | **excluded** | 0.7500 |
| `bench/figures.py` | 273 | excluded (latency only) | — |

`audit/figures_v3.py` is the script that produced the committed thesis figures
and `cell_stats.csv`. `bench/summary.py` is the script that produced the
headline table. **They disagree by 0.35 points on the classifier and 0.02 on the
LoRA.** A figure and a table in the same chapter would print different numbers
for the same cell.

The dedupe key is consistent everywhere and is *better* than the brief states:
all four scripts key on the 5-tuple `(dataset, model, strategy, command_id,
run_index)`, not the 4-tuple — `bench/summary.py:101`, `bench/analyze.py:45-46`,
`bench/figures.py:68`, `bench/rescore.py:105`. `analyze.py:40-44` documents why.
On the V5 data the key is moot: zero duplicates across all 12,152 rows.

**Fix:** one `accuracy(rows)` helper in `bench/`, imported by all four. 3 hours.

### 4. BLOCKER — `DEFENCE_NUMBERS.md` describes a superseded experiment

The file that exists to say "what can be quoted" quotes V3 throughout. Every
row of its headline table is wrong against V5:

| `DEFENCE_NUMBERS.md` | says | V5 measured |
|---|---|---|
| Best system overall (`:23`) | gpt-4o-mini + RAG **0.924** | **0.9382** |
| Best local (`:25`) | classifier **0.747** | 0.7465 (unchanged — the classifier has no prompt, so the legend fix could not touch it) |
| Best fine-tuned (`:27`) | qwen2.5-3b-lora **0.654** | **0.9585** |
| Pareto (`:33-37`) | classifier 19 ms / role_based 0.855 / RAG 0.924 | 17.9 ms / 0.9133 / 0.9382 |
| Structural (`:48`) | "31.4% of generative rows never validated (9,530 of 30,352)" | V3 row count |
| Basis note (`:5`) | "Effective n = 217, not 1,085 rows" | correct for cloud; local cells are 216 or 217 depending on rule — see finding 2 |

The 3b-lora line is the dangerous one: **0.654 → 0.959 is a 30-point move**, and
the V5 number makes a 3B local LoRA the best system in the study, beating
gpt-4o-mini. That inverts the thesis's headline conclusion. `DEFENCE_NUMBERS.md`
still states the old ordering as "safe to quote [Certain]".

**[R2] Status: DONE, 17 September 2026.** `audit/DEFENCE_NUMBERS.md` revision 2
rebuilds the document on the V5 basis, adopts the three-way grid/container/
no-location split, states the warmup rule explicitly, removes the carried-over
V3 intervals, and records findings 24–26 as limitations. It does **not** invent
V5 intervals — §0.2 of that document states plainly that the headline ordering
is unconfirmed until finding 10 is fixed.

**Fix:** regenerate the whole file from the V5 set. It cannot be patched
number-by-number, because the *ranking* changed. 4 hours, and it depends on
findings 2, 3 and 10 being settled first.

### 5. BLOCKER — the chain from run files to reported numbers is not committed

`git grep` for any of the five published numbers across every `.md` and `.py`
returns **nothing**. They exist only in the conversation that produced them.

Worse, the analysis chain has a missing link. The measurement wrote six
per-model files (`results_v5_{cloud,cls,deepseek,llama,qwen05,qwen3}.csv`) —
that is what the manifests record. Everything downstream reads four per-rung
files (`results_v5_{rung1,rag,lora,cls}_scored.csv`). **No committed script
performs that regrouping.** The only tracked mention of those four filenames is
`bench/backfill_tokens_v5.py:70`, which consumes them.

The regrouping is verifiably lossless — I checked:

```
RAW    12,152 rows, 36 cells, 0 duplicate keys
SCORED 12,152 rows, 36 cells, 0 duplicate keys
in raw not in scored: 0     in scored not in raw: 0
strict_match verdicts: 12,152 agree, 0 differ
```

Two consequences. First, the `_scored` suffix is a misnomer: those files carry
`run_cell.py`'s original verdicts unchanged, so `rescore.py` either was not run
or changed nothing. Second, the partition rule is inferable (rung1 = the four
prompt strategies on the five base models; rag = rag on those five; lora = both
adapters × all five; cls = the classifier) but a stranger has to derive it.

**Fix:** commit a ten-line `bench/regroup_v5.py`, and a `RESULTS_V5.md` stating
each number with its file, its filter and its denominator. 3 hours.

### 6. HIGH — "spatial reasoning 0.970" is not any grid-only number in the data

`DEFENCE_NUMBERS.md:25` claims gpt-4o-mini + RAG scores **0.970 [0.937, 0.996]**
on "grid-only". Recomputed from `results_heldout_v3_rag_scored.csv`:

| basis | n rows | n commands | strict |
|---|---|---|---|
| all | 1,084 | 217 | 0.9234 |
| grid cells only | 659 | 132 | **0.9605** |
| non-container (`cell_stats.csv`'s `grid` column) | 904 | 181 | **0.9137** |
| **container** | 180 | 36 | **0.9722** |

0.970 matches neither grid definition. It is within rounding of the
**container** figure — the easy task that `DEFENCE_NUMBERS.md:11` itself warns
"names the answer in the command". The 0.914 in `cell_stats.csv` is the
non-container pool.

There is a definitional problem underneath the arithmetic. "Grid-only (n=181)"
pools 132 commands that name a grid cell with **49 that have no location at
all** (`stop`, `reject`, `fetch`, `hold`, `inspect`). Calling that basis
"spatial reasoning" is not defensible when 27% of it contains no spatial
content. The three-way split — 108 grid + 24 runtime / 36 container / 49 no-location — is
the honest decomposition and the data supports it cleanly.

For reference, the V5 three-way split of the headline cells:

| cell | all | grid (132/131) | container (36) | no-location (49) |
|---|---|---|---|---|
| gpt-4o-mini / rag | 0.9382 | 0.9848 | 0.9889 | 0.7755 |
| qwen2.5-3b-lora / zero_shot | 0.9583 | **1.0000** | 1.0000 | 0.8163 |
| classifier | 0.7500 | 0.6947 | 1.0000 | 0.7143 |

**The LoRA scores 131/131 on grid cells.** That is the thesis's most striking
result and it will be the first thing an examiner probes. It is not string
leakage — `train_v2 ∩ held_out_v3` is 0 exact and 0 near. But "perfect" invites
the question and deserves a paragraph of its own analysis before submission.

**[R2] Resolution, 17 September 2026.** Revision 2 of `DEFENCE_NUMBERS.md`
retires the `n=181` basis entirely and adopts the three-way split. The 0.970
figure is recorded as the container accuracy, misfiled. **[R2] The honest
mechanism behind 131/131 is the V5 legend rewrite, which removed the vocabulary
mismatch that suppressed this cell in V3 (0.407 → 1.000) — and that legend was
authored after seeing V3 fail on this same set. Finding 24 is the reason the
131/131 paragraph has to be written carefully rather than celebrated.**

### 7. HIGH — the V3 measurement has no manifest

Every V4 and V5 run wrote a `.manifest.json` with argv, dataset, schema version,
exemplar source, env vars, Ollama model digests, package versions, Python and
platform. All eight V5 manifests verify clean against their CSVs: cell sets
match exactly, row counts match `expected_rows_per_cell`, digests and package
blocks present.

`results_heldout_v3_{rung1,rag,lora,cls}.csv` — 30,380 rows, and the basis of
every number in `DEFENCE_NUMBERS.md` — have **no manifest**. The manifest
feature was added afterwards. This is the `F-ENV` finding that
`requirements.txt` already discloses, and it is the reason `requirements.txt`
pins the audit-time environment rather than the measurement-time one.

If V3 is retired in favour of V5 this becomes moot. If any V3 number is still
quoted — and `DEFENCE_NUMBERS.md` quotes many — it is quoted without provenance.

**[R2]** Revision 2 retires V3 as a headline source but still quotes it for the
CoT cost sentence, the row-vs-column structural result, the 31.4% invalid rate
and the 5.8-point CI half-width. Each is now labelled *V3* in place. Those four
remain provenance-free; see finding 27 and fix-list item 16.

### 8. HIGH — `held_out_v4b.jsonl` is not held out

`build_v4b.py` rewrites `held_out_v3` row words into the training vocabulary
(`upper→far`, `bottom→near`). That is the point of the condition. The
side-effect is that the rewritten commands become **verbatim training
commands**:

| pair | exact overlap | near-duplicate |
|---|---|---|
| `train_v2` ↔ `held_out_v4b` | **56** | 56 |
| `seen_v2` ↔ `held_out_v4b` | **72** | 72 |
| `dev_v2` ↔ `held_out_v4b` | **14** | 14 |
| `train_v2` ↔ `held_out_v3` | 0 | 0 |
| `train_v2` ↔ `held_out_v4c` | 0 | 0 |

26% of `held_out_v4b` was seen in training. The saving grace: **it was never
run.** No manifest references `held_out_v4b`; the only V4 dataset measured was
`held_out_v4c` (`results_v4c_lora`). So the contamination is latent. But the
file is tracked, sits in `data/` beside genuine sealed sets, and is named
`held_out_*` — which is exactly the naming that `run_grid.py`'s
`--i-understand-heldout-is-sealed` gate keys on. A future run would pass the
gate and produce a contaminated number that looks legitimate.

**Fix:** rename to `probe_v4b_vocab_aligned.jsonl` and add one line to its
header comment. 30 minutes.

### 9. HIGH — the four committed V5 figures have an unknown input

`bench/figures_v5/` contains `accuracy_heatmap.png`, `latency_tails.png`,
`generalization_gap.png`, `cot_strict_vs_norm.png`, committed in `093c1ee`.
Those are exactly the four filenames `bench/analyze.py:177-180` writes, and
`analyze.py:170` takes `--outdir`. So `analyze.py` produced them.

`analyze.py` takes `--seen` and `--held` and draws a *seen versus held-out*
comparison. **V5 measured only `held_out_v3`** — all 12,152 rows carry
`dataset=held_out_v3`. So `--seen` was given either the same file (making
`generalization_gap.png` a comparison of a set with itself) or a stale V1/V2
file (making it a cross-version comparison labelled as a generalization gap).
Both defaults still exist on disk, dated July. I could not determine which
without the shell history.

`results_v5_slim.csv` exists because of finding 12: the `raw` column in
`results_v5_all.csv` reaches **286,686 characters**, and `analyze.py` never
raises the 131,072 default limit, so it crashes. `slim` caps fields at 2,000.
That is a sensible workaround, but it is undocumented and uncommitted, and it
means the figures were drawn from a *truncated* copy of the data.

`analyze.py` also carries the docstring label `DATA VERSION: V1/V2 ONLY` at
lines 17-20 — a label that is now false, since it produced the V5 figures.

**[R2] Do not put these four figures in the thesis.** With the input unknown and
the source data truncated, a caption cannot honestly state what they show.
Either regenerate from a committed script with named inputs, or drop them.

**Fix:** record the exact command, or regenerate the figures from a committed
script with the inputs named in the caption. 2 hours.

### 10. HIGH — no confidence intervals for any V5 number

`audit/figures_v3.py:104-125` implements `clustered_ci` correctly: it bootstraps
over `command_id` clusters, takes cluster means so a deterministic cluster
contributes zero within-variance, and reports `n_clusters`. `cell_stats.csv`
records `n_clusters=217` against `n_rows=1084`. **The intervals are computed on
217 commands, not on the raw row count.** They are not too narrow. That part of
the brief's concern is unfounded — for V3.

There is no V5 equivalent. `audit/figures_v3.py:92` hardcodes
`results_heldout_v3_{tag}_scored.csv`. So the five published V5 numbers have no
intervals at all, and the intervals printed in `DEFENCE_NUMBERS.md` belong to
different numbers from a different run.

This matters most for the claim the thesis will want to make. V5's top two
cells are `qwen2.5-3b-lora/zero_shot` at 0.9583 and `gpt-4o-mini/rag` at 0.9382
— **two points apart**. `DEFENCE_NUMBERS.md:113` states the median CI
half-width is 5.8 points and that "any difference under ~6 points is not
claimable". Under its own stated rule, *the headline finding of the thesis is
not claimable* until V5 intervals exist to test it.

**[R2] This is now the highest-priority fix in the report**, ahead of everything
except committing the data. It is 2 hours of work and it decides whether the
thesis has a headline finding or an unconfirmed ordering. See finding 26.

**Fix:** point `figures_v3.py` at the V5 files via a parameter. The bootstrap
code is already correct and needs no change. 2 hours.

### 11. HIGH — a corrected number that was never propagated

`audit/v4_cells.csv` resolves it definitively: the V4a condition has **27**
cells, `verdict` = 7 `**real**` and 20 `noise`.

- `audit/V3_VS_V4.md:316` — states the correction: "`21 of 27` is wrong. The correct figure is 20 of 27."
- `docs/SESSION_2026-09-07.md:152` — states it again, with three independent counting methods agreeing on seven.
- `audit/DEFENCE_NUMBERS.md:78` — **still says 21 of 27.**
- `audit/V4_RESULTS.md:121` — **still says 21 of 27.**

The correction was written down twice and applied zero times. `SESSION_2026-09-07.md:157`
also records that the wrong count is baked into the caption of
`fig5_v3_vs_v4a.png`, and that `fig7_v4c_row_vs_column.png` has a label
collision where the `36` on the 3b-lora bar is overprinted.

**[R2] Partial resolution, 17 September 2026.** Fixed in
`audit/DEFENCE_NUMBERS.md` revision 2 §5, which also corrects the secondary
error: `session_record_v3.md §2.13` says "six real moves", and the correct
figure is **seven**. **Still outstanding: `audit/V4_RESULTS.md:121`, the
`fig5_v3_vs_v4a.png` caption, and the `fig7` label collision.** This correction
has now survived four sessions. Do all four locations in one pass or it will
survive a fifth.

### 24. [R2] BLOCKER — `held_out_v3` is not a sealed set any more

The 13 September pass verified, exhaustively and correctly, that no held-out
command influenced training or prompt-selection. That is a *leakage* check. It
is not the same question as whether the set has been **reused**, and the reuse
answer is worse.

`held_out_v3` has been measured three times:

| run | conditions | legend |
|---|---|---|
| V3 | 28 cells, 30,380 rows | v3 |
| V4 | `results_v4c_lora` (the LoRA pair) | v4 |
| V5 | 36 cells, 12,152 rows | **v5** |

The V5 legend was not an arbitrary rewrite. `session_record_v3.md §2.15` records
that it was designed against a mechanical checklist built from the specific
failures V3 and V4 exhibited **on this set** — the row/column label collision,
the contradictory framing, the cell-ID anchoring. All four checks pass for V5
and none passed for V3.

No file was modified and the prompt is not data, so nothing in the letter of the
protocol was broken. But the guarantee a held-out set provides expires the first
time you look at it, and the project's own plan v6 don't-do list reads
**"❌ Tune on the held-out set"**. The V5 numbers measure performance under a
legend fitted to this set; they are an optimistic estimate of performance on a
fresh one, by an amount the study cannot bound.

This was a knowing decision — recorded as "option C: accept the confound and
declare it." **The declaration has not happened.** It appears in no committed
document, and `DEFENCE_NUMBERS.md` revision 1 presented the V5 ordering without
qualification.

**Why this is the top viva risk:** it is discoverable from the tracked manifests
alone. All eight V5 manifests name `held_out_v3`; so do the V3 result files. An
examiner comparing them finds the reuse in under a minute, and then asks why it
is not in the methods chapter.

**Fix:** one paragraph in methods, in these terms, plus one line in
`DEFENCE_NUMBERS.md` (**done** — revision 2 §6.1). 1 hour, and it converts the
finding from a flaw into ordinary experimental honesty. The clean experiment —
a fourth held-out set authored after the legend was frozen — is future work, not
this thesis.

### 25. [R2] BLOCKER — the winning cell has no cost axis

The thesis's stated core contribution is a measured cost curve across four
adaptation methods. The V5 winner, `qwen2.5-3b-lora/zero_shot` at 0.9583, has
**no latency figure and no cost figure in any committed document**. Revision 1's
Pareto table is V3-only and does not contain the cell at all.

The data exists — `results_v5_lora_scored.csv` carries `latency_s` per row — but
no document states the p50, and `bench/summary.py`'s token sections
**silently vanish** when a raw `_scored` file is passed rather than a `_tokens`
file (finding 15, trap 4). So the number is one command away and nobody has run
it.

Without it the headline figure of the thesis cannot be drawn.

**Fix:** extract p50 `latency_s` for all LoRA cells; pass the `_tokens` file to
`summary.py` for the cost columns; add both to `RESULTS_V5.md`. 1 hour.

### 26. [R2] HIGH — the headline margin is inside the noise floor

0.9583 vs 0.9382 is **2.0 points**. The thesis's own stated resolution is ~6
points. Finding 10 means there is currently no V5 interval to test the gap
against.

Three outcomes are possible once the intervals exist, and the thesis should be
written so that all three are survivable:

1. **Intervals separate.** The headline stands: a 3B local LoRA beats a frontier
   cloud model on this task. Strongest result in the study.
2. **Intervals overlap.** The headline becomes "a 3B local LoRA *matches*
   gpt-4o-mini at a fraction of the cost" — which is still a good result, and
   arguably the more interesting one for an edge-AI thesis.
3. **The warmup rule flips the rounding** (0.9585 vs 0.9583). Immaterial to the
   comparison, material to every printed table.

**Do not write the results chapter around outcome 1 before the intervals
exist.** Write it around outcome 2, which holds either way, and strengthen it if
the intervals separate.

### 27. [R2] MEDIUM — two V3 row counts

`DEFENCE_NUMBERS.md` revision 1 §2 states the V3 generative pool as **30,352**
rows (9,530 invalid = 31.4%). This report §1 and §7 state the V3 measurement as
**30,380** rows. A 28-row gap, unexplained. Likely a warmup or a rung-boundary
difference; not established. Reconcile before either figure appears in the
thesis, and state which pool the 31.4% is a fraction of.

### 28. [R2] MEDIUM — git hygiene is not media hygiene

`.env` is gitignored at `.gitignore:4` and `:47` and has never been committed —
`git log --all -- .env` is empty. The 13 September pass concluded "Secrets:
clean", and for the repository that is correct.

It is not the whole surface. `docs/DEMO_PORTABLE.md` describes an ~18.6 GB USB
drive prepared for the supervisor visit, containing the working tree. If `.env`
was in the tree at copy time, a live `OPENAI_API_KEY` left the machine on
removable media.

**Fix:** rotate the key at `platform.openai.com` regardless of whether the copy
happened; delete `.env` from the USB drive; re-create it locally from the new
key. 10 minutes. There is no scenario where rotating costs more than not
rotating.

### 29. [R2] MEDIUM — three open decisions never carried into the audit

From `SESSION_2026-09-07.md §4`, none of which appear in the 13 September pass:

- **`MOVE_TOL = 0.02` is unreachable.** Best observed is 0.029, so `move_to`
  reports `ok: false` on every call. Either raise to 0.05 and document the
  change, or leave it honest and red — but decide, because a demo that returns
  failure on every `move_to` invites a question with no good live answer.
- **`nearest_free_cell` excludes the carried object from the occupancy test.**
  `sim/workspace.py` delegates this to the caller. Check what
  `sim/sim_backend.py` passes as `occupied_xy`. If MuJoCo includes the target
  and Isaac does not, one of the two backends is wrong and the same command
  means different places in the two demos.
- **Whisper model mismatch.** `llm_client.py` uses `small.en`; `asr/voice.py`
  defaults to `base`. If `run_demo.py` passes different models on the two paths,
  the two demos mishear different words — and voice is part of the demo.

All three are demo-path only; no measured number is affected. All three are
live at the viva.

### 30. [R2] MEDIUM — field-level escalation is in no not-reportable list

`session_record_v3.md §2.14` reports that escalating **only the location** field
to gpt-4o-mini when the classifier's location head has a low softmax margin
dominates `role_based` at t=0.30 and ties the RAG frontier point at t=0.50 for
half the latency. It is the most interesting system result the project produced.

It is also **a simulation from full-object prompts**, so it assumes cloud
location accuracy is unchanged when only the location is asked — untested,
and flagged `[Certain]` as load-bearing in the source record. The novelty claim
(per-field deferral within one structured output, versus query-level cascades
like FrugalGPT, RouteLLM, CP-Router, UCCI) rests on two searches and is tagged
`[Guessing]`.

No committed document lists it as not-reportable, so nothing stops it reaching a
slide as a measured result.

**Fix:** it is now in `DEFENCE_NUMBERS.md` revision 2 §7. Report it in the
thesis as a *simulated* extension with the assumption stated, in future work —
or run the location-only prompt condition and make it real. The latter is new
measurement work and is out of scope under the 17 September freeze.

### 31. [R2] Scope freeze, 17 September 2026

The supervisor instructed in writing to stop building: **no Phase 6 (vision) and
nothing beyond it.** Phase 2.5 (RL backend, SAC+HER via SB3) is likewise not
being built. Remaining work is correction and write-up only.

This changes the character of two entries elsewhere in this report. "Vision or
real-arm results — zero lines of implementation" was a gap; it is now a recorded
scope decision with supervisor agreement, which is a far better answer at a
viva. The fix list below is re-prioritised accordingly: nothing that constitutes
new capability is scheduled, and the Isaac `reset` verb (finding 18) survives
only because the demo *is* the viva.

**One thing the written instruction does not cover:** the cover page still
promises vision-based manipulation. A title that describes work the thesis does
not contain is a defect independent of whose decision the cut was. Confirm the
title change with the supervisor, in writing, and reconcile the abstract and
introduction to match.

### 32. [R2] LOW — the `_scored` suffix asserts something unverified

The four V5 `_scored.csv` files carry verdicts identical to the raw run files on
all 12,152 rows. That is consistent both with "`rescore.py` ran and changed
nothing" and with "`rescore.py` never ran". The suffix asserts the first. The
evidence cannot distinguish, and the command history is gone.

Harmless to the numbers — `llm/score.py` is single-sourced and the verdicts
agree — but it is an unsupported claim baked into a filename that the thesis
will cite. Either re-run `rescore.py` and record the invocation, or rename.

---

## Reproducibility trace

Every chain below starts at `bench/run_grid.py`, whose invocation is fully
recorded in the tracked manifests, and breaks in the same two places: the
regrouping step is uncommitted, and the final aggregation is undocumented.

**95.9% strict, `qwen2.5-3b-lora`, zero-shot**
`data/held_out_v3.jsonl` (tracked, 217 rows) → `bench/run_grid.py` with the argv
in `bench/results_v5_qwen3.csv.manifest.json` (tracked; legend v5, repeats 1,
num_ctx 4096, 10 cells) → `bench/results_v5_qwen3.csv` (**untracked**) →
*[missing step: regroup by rung]* → `bench/results_v5_lora_scored.csv`
(**untracked**) → `bench/summary.py:161-176`. **Recomputed: 208/217 = 0.9585,
warmup included.** Excluding the warmup gives 207/216 = 0.9583 → 95.8%.
**Chain is complete but two links are uncommitted, and the published rounding
depends on the warmup convention.** **[R2] No latency or cost figure exists for
this cell in any document — finding 25.**

**93.8% strict, `gpt-4o-mini`, retrieval**
Same path via `results_v5_cloud.csv.manifest.json` (repeats 5, 5 cells) →
`results_v5_rag_scored.csv` → `summary.py`. **Recomputed: 1017/1084 = 0.9382
excluding warmup, 1018/1085 = 0.9382 including it.** This number is robust to
the warmup rule because command 0 has four other repeats. **Chain holds; only
the uncommitted links are at issue.**

**74.7% strict, `classifier:distilbert`**
Via `results_v5_cls.csv.manifest.json` (1 cell, 217 rows) →
`results_v5_cls_scored.csv` → `summary.py`. **Recomputed: 162/217 = 0.7465
including the warmup row; 162/216 = 0.7500 excluding it.** `audit/figures_v3.py`
would print 0.7500 for the same cell. **Chain reaches a number that depends on
which script you ask.**

**18 ms median latency, classifier**
Same file → `bench/summary.py:206-212`, which *does* filter warmups.
**Recomputed: 17.87 ms median over 216 rows (min 14.06, max 24.44).** The
excluded warmup row is 14,650.9 ms. **Chain holds and is internally consistent
— but it uses the opposite warmup rule to the accuracy figure beside it.**

**12,152 total scored rows across 36 cells**
Sum of the four `results_v5_*_scored.csv`: 217 + 2,170 + 1,953 + 7,812 =
**12,152 exactly**, 36 distinct cells, 0 duplicate keys, all
`dataset=held_out_v3`. **This one verifies perfectly.** Note it includes 36
warmup rows (one per cell); the non-warmup count is 12,116. If the thesis
adopts the warmup-excluded convention for accuracy, the headline row count
should become 12,116 to match.

### Could a stranger with this repo regenerate them?

**No.** Required and not committed:

| missing input | size | replaceable? |
|---|---|---|
| all 18 `bench/results_v5_*.csv` | 72.6 MB | only by re-running; cloud cells cost money and are non-deterministic |
| the regroup script (6 per-model files → 4 per-rung files) | ~10 lines | inferable from the rung definition, not automated |
| the `summary.py` invocation that produced the five numbers | 1 line | unknown filters/flags |
| the `analyze.py` invocation behind `bench/figures_v5/` | 1 line | `--seen` argument unknown |
| `ft/runs/distilbert_cls/epoch_09/` | 265 MB | retrainable from `train_v2` (tracked) but not bit-identical |
| `ft/export/*.gguf` | 2.3 GB | regenerable via `merge_and_export.py`, which needs `C:\Users\USER\llama.cpp` |
| `.env` (`OPENAI_API_KEY`) | — | supply your own |
| Ollama model blobs at `OLLAMA_MODELS=D:\LLM` | ~10 GB | digests recorded in the manifests, so verifiable |

What *is* committed and sufficient: the datasets, the schema, all prompt
legends, the scorer, the harness, and the eight V5 manifests. The experiment is
re-runnable. The results are not recoverable.

---

## Dataset map

| file | rows | purpose | overlaps with | trains |
|---|---|---|---|---|
| `data/seen.jsonl` | 131 | V1 development set | `seen_v2` 71, `train_v2` 59, `held_out_v3` **12** | nothing (V1 only) |
| `data/held_out.jsonl` | 97 | V1 held-out | nothing (0 with every V2+ file) | — |
| `data/seen_v2.jsonl` | 290 | V2 authored pool | `train_v2` 230, `dev_v2` 57, `held_out_v4b` 72 | source of the splits |
| `data/train_v2.jsonl` | 230 | LoRA + classifier training; **the only RAG corpus** | `seen_v2` 230, `held_out_v4b` **56** | LoRA 0.5B/3B, DistilBERT |
| `data/dev_v2.jsonl` | 57 | checkpoint/epoch selection only | `seen_v2` 57, `held_out_v4b` 14 | selection signal only |
| `data/held_out_v2.jsonl` | 132 | V2 sealed KPI | ⊂ `held_out_v3` (all 132) | never |
| `data/held_out_v3.jsonl` | **217** | **the V5 measurement set** | `held_out_v2` 132, `seen.jsonl` 12, `v4b`/`v4c` 145 | **never — 0 exact, 0 near vs train/dev/seen_v2. [R2] But measured three times; see finding 24** |
| `data/held_out_v4b.jsonl` | 217 | vocabulary-aligned probe | **`train_v2` 56, `seen_v2` 72, `dev_v2` 14** | never — and never run |
| `data/held_out_v4c.jsonl` | 217 | neither-vocabulary probe | `held_out_v3` 145; **0 with train/dev/seen_v2** | never |
| `data/seen_v2_backup_203.jsonl` | 203 | pre-gate provenance snapshot | `seen_v2` 203 | nothing; read by no code |

Composition of `held_out_v3`: **108 grid + 24 runtime (12 `max_reach`, 12
`nearest_free_cell`), 36 container, 49 no-location.** No duplicate command
strings in any file. **[R2] Under the warmup-excluded rule the 31 local cells see
131 grid + runtime commands (107 + 24), because `command_id 0` is a grid
command.** _(Corrected 2026-09-28: previously "132 grid-cell", which counted the
24 runtime-symbol commands as grid; verified from `data/held_out_v3.jsonl`.)_

Three `seen_v2` commands are in neither `train_v2` nor `dev_v2`
(`"put the bottle in the far left corner of the table"`,
`"…far side middle edge…"`, `"…farthest spot you can reach"`). These are the
fixed few-shot exemplars from `bench/dataset.py`, deliberately withheld from
both splits. 230 + 57 = 287, + 3 = 290. The arithmetic is correct and
intentional, but it is not stated in `map.md`.

### Leakage verdict

**No sealed command influenced any training or prompt-selection decision.**
[Certain], on four independent checks:

1. `train_v2 ∩ held_out_v3` = **0 exact, 0 near-duplicate** (near = lowercase,
   punctuation stripped, whitespace collapsed). Same for `seen_v2` and `dev_v2`.
2. `ft/prep_data.py:27-28` reads only `train_v2` and `dev_v2`;
   `ft/train_lora.py:257-258` reads only the files that produces.
3. `ft/train_classifier.py:87-88, 251-252` reads only `train_v2` and `dev_v2`.
   No `held_out` string appears anywhere in `ft/`.
4. `llm/retrieve.py:56` pins `CORPUS_PATH = "data/train_v2.jsonl"`.

Leave-one-out **is** enforced: `llm/retrieve.py:141-163` defaults
`exclude_exact=True` and sets `sims[i] = -inf` for any corpus row whose command
equals the query after `strip().lower()`; `bench/run_cell.py:280` passes it
explicitly. One limitation: the match is exact-after-normalisation, so a
near-duplicate would survive. Moot for the measured run, where the overlap is
zero on both measures — but it would *not* protect a `held_out_v4b` run, where
56 commands match the corpus verbatim (those would be excluded) while their
near-variants would not.

**[R2] Reuse verdict — separate question, worse answer.** The leakage verdict
above is about *training*. It says nothing about how many times the set has been
looked at, and the answer is three, with the V5 prompt fitted to what the
earlier looks revealed. See finding 24. **Do not cite the leakage verdict as
though it answers the reuse question; an examiner who knows the difference will
notice that it does not.**

---

## What I could not verify

- **What `--seen` was passed to `analyze.py` when `bench/figures_v5/*.png` were
  generated.** V5 has no seen-set rows, so the figure is either self-comparing
  or cross-version. Needs the shell history, or a regeneration from a committed
  script.
- **Whether `bench/rescore.py` was run on the V5 files at all.** The `_scored`
  files carry verdicts identical to the raw run files on all 12,152 rows, which
  is consistent both with "rescore ran and changed nothing" and with "rescore
  never ran". The `_scored` suffix asserts the first; the evidence cannot
  distinguish. Needs the command history. **[R2] Recorded as finding 32.**
- **Whether `bench/analyze.py` actually crashes on the V5 data.** The defect is
  certain from the source (`:22`, `:49`, no `field_size_limit`) and the data
  (286,686-char field vs a 131,072 limit), but I did not execute it — running
  it writes PNGs. Needs one run in a scratch `--outdir`.
- **Which of the seven V4a cells are the "real" moves.** `v4_cells.csv` marks
  seven; `SESSION_2026-09-07.md:145-151` records that three counting methods
  agree on seven but disagree on *which* seven. Needs a decision, not an
  analysis. **[R2] Decision taken: report both disputed cells
  (`gpt-4o-mini/cot`, `qwen2.5:0.5b/few_shot`) as ambiguous rather than picking
  the rule that tells the tidier story.**
- **Whether the Isaac demo can complete a full run without a restart.** Finding
  18 is certain from the source — no reset verb exists — but the practical blast
  radius depends on how many placements a demo makes before the three boxes are
  full. Needs a live Isaac session, which this machine cannot run.
- **Whether `0.970` in `DEFENCE_NUMBERS.md:25` is the container figure or a
  third computation.** It matches 0.9722 to within rounding, but no script in
  the repo emits 0.970 for that cell. Needs whatever produced the original
  table. **[R2] Resolved by removal: revision 2 retires the basis and the
  figure. If the provenance is ever recovered, record it; otherwise the number
  simply does not appear anywhere again.**
- **`n=539` on slide 17 and the "0.1–0.5 cm placement" claim on slide 22**, both
  flagged in `SESSION_2026-09-07.md:196-199`. The slides are not in the repo.
  **[R2] Partially resolved: the verified Isaac figure is 0.2–0.4 cm for
  `pick_place`, so the slide's 0.1–0.5 cm range is wrong at both ends and
  conflates `pick_place` with `move_to`. Correct the slide. `n=539` is still
  unexplained and should be removed from the slide unless someone can say what
  it counts.**
- **[R2] Whether the thesis title has been changed to match the reduced scope.**
  Finding 31. Needs the supervisor, not an analysis.

---

## Fix list

**[R2] Re-prioritised 17 September 2026 for the scope freeze.** Nothing below
adds capability. Item ordering is now: protect the data, make the headline
defensible, then correct the documents.

### Before writing starts — 21 hours

| # | Task | Hours | Status |
|---|---|---|---|
| 1 | Commit the V5 result CSVs; add `!bench/results_v5_*.csv` negations to `.gitignore`; push; copy the zip off-machine | 1 | **do first, today** |
| 28 | **[R2]** Rotate the OpenAI key; delete `.env` from the USB drive | 0.2 | **do first, today** |
| 5 | **[R2]** Point `clustered_ci` at the V5 files; produce intervals for all 36 cells; check whether 0.9583 vs 0.9382 separates | 2 | decides the headline |
| 25 | **[R2]** Extract p50 latency and cost for `qwen2.5-3b-lora/zero_shot` and all LoRA cells; the Pareto plot cannot be drawn without it | 1 | |
| 2 | Decide the warmup rule; add the filter to `summary.py:176`; re-run `command_id 0` for the 31 local cells if you want n=217 honestly | 4 | |
| 3 | One shared `accuracy(rows)` helper; make `summary.py`, `analyze.py`, `figures_v3.py` import it | 3 | |
| 4 | Commit `bench/regroup_v5.py`; write `audit/RESULTS_V5.md` giving each number its file, filter and denominator | 3 | |
| 24 | **[R2]** Write the held-out-reuse paragraph into the methods chapter | 1 | |
| 6 | Rewrite `DEFENCE_NUMBERS.md` from V5 — the ranking changed, so rewrite rather than patch | 4 | **done 17 Sep (rev 2)** |
| 7 | **[R2] moved up from "can wait":** fix `21 of 27` in `V4_RESULTS.md:121`, the `fig5` caption and the `fig7` label collision — this correction has survived four sessions | 1 | `DEFENCE_NUMBERS.md` done |
| 31 | **[R2]** Confirm the title change with the supervisor in writing; reconcile abstract and introduction | 0.5 | |

Item 6 depended on 2, 3 and 5; revision 2 was written anyway, with the
unresolved dependencies stated in its §0 rather than papered over. Items 5 and 25
must still land before any figure is drawn.

### Can wait until after — 11 hours

| # | Task | Hours |
|---|---|---|
| 8 | Resolve `0.970`; adopt the three-way grid/container/no-location split as the reported basis | 2 (**split adopted 17 Sep**) |
| 27 | **[R2]** Reconcile 30,352 vs 30,380 for V3 | 0.5 |
| 9 | Rename `held_out_v4b.jsonl` to `probe_v4b_vocab_aligned.jsonl` | 0.5 |
| 10 | Add `csv.field_size_limit` to `analyze.py:23`; replace the stale defaults at `:168-169` with required arguments | 0.5 |
| 11 | Give `backfill_tokens_v5.py` a `--in`/`--out`; fix its docstring, which still says `_v3`; or merge the two files behind one flag | 1.5 |
| 12 | Align `llm_client.py:220` with what was measured, or change its comment to say the restriction is a demo choice, not a measurement fact | 0.5 |
| 29 | **[R2]** Decide `MOVE_TOL`; check `occupied_xy` parity between backends; align the Whisper model on both paths | 2 |
| 13 | Add a `reset` verb to `isaac_backend.dispatch` and the bridge protocol — kept only because the demo is the viva | 2 |
| 14 | Make `run_grid.py:166-169` refuse to overwrite an unparseable manifest instead of discarding it | 0.5 |
| 15 | Make `merge_and_export.py:35` read `LLAMA_CPP` from the environment | 0.5 |
| 16 | Write V3 manifests retrospectively from `MODEL_PROVENANCE.txt`, marked reconstructed — only if V3 numbers are still quoted (they are: CoT cost, row-vs-column, 31.4%, the 5.8-pt CI) | 1 |
| 30 | **[R2]** Move field-level escalation into future work with its assumption stated | 0.5 |
| 32 | **[R2]** Re-run `rescore.py` and record the invocation, or rename the `_scored` files | 0.5 |
| 23 | **[R2]** Either commit `session_record_v3.md` or fix the dangling citation at `SESSION_2026-09-07.md:152` | 0.2 |

---

## Trap confirmations

All seven confirmed in source. Blast radius stated.

**1. `analyze.py` has no `field_size_limit`** — `bench/analyze.py:22` imports
`csv`, `:49` calls `csv.DictReader`, and nothing between raises the limit. Every
other CSV reader in the repo does (`summary.py:67`, `rescore.py:59`,
`run_cell.py:87`, `run_grid.py:201`, `figures.py:45`, all nine `audit/`
scripts). `results_v5_all.csv` contains a 286,686-char `raw` field. **Radius:**
`analyze.py` is unrunnable on the full V5 data; `results_v5_slim.csv` is the
undocumented workaround; the four figures in `bench/figures_v5/` were therefore
drawn from truncated data. Those figures are committed and presumably headed for
the thesis.

**2. `analyze.py --seen` default loads stale data** — `bench/analyze.py:168-169`
default to `bench/results_grid_scored.csv` and `bench/results_heldout_scored.csv`.
Both exist, 77,545 and 56,765 lines, dated 2026-07-20 and 2026-07-21. A bare
`python -m bench.analyze` produces V1 figures with no warning. **Radius:** see
finding 9 — this is the likeliest explanation for whatever `--seen` received.

**3. `backfill_tokens_v3.py` hardcodes filenames** — `:69-70` set `IN_TMPL` and
`OUT_TMPL` as module constants with no CLI override; `main()` takes only
`--check-only`. `backfill_tokens_v5.py:69-70` is the same two lines with `v5`
substituted, in an otherwise byte-identical 263-line file whose docstring at
`:3` still reads "`bench/backfill_tokens_v3.py` — token/cost backfill for the
held_out_v3 files" and whose usage block at `:60` still says
`python -m bench.backfill_tokens_v3`. **Radius:** no wrong number today — the
token columns are only used for the cost sentence — but two 263-line files that
must stay in step is the clearest drift hazard in the repo.

**4. `run_grid.py` writes token columns it never fills** — `bench/run_cell.py:322`
writes `"tokens_in": "", "tokens_out": "", "cost_usd": ""` literally; the
columns are declared at `:122-124` with comments saying "recompute from prompt".
Only the backfill scripts populate them. **Radius:** `bench/summary.py:128`
defines `has_tokens()`, and `:199` / `:299` gate the token and throughput sections on it, so those sections
**silently vanish** when a raw `_scored` file is passed and appear when a
`_tokens` file is. Two runs of the same command on two filenames produce
different-shaped reports with no error. The V5 `_tokens` files exist, so the
cost numbers are recoverable — but only if you remember which file to pass.
**[R2] This is why finding 25 happened: the cost columns for the winning cell
were never printed because the wrong file was passed, and nothing errored.**

**5. `llm_client.py` allowlist contradicts what was measured** —
`bench/llm_client.py:220` sets `LORA_STRATEGIES = ("zero_shot",)` and `:249-250`
routes any `*-lora` model to it. The justification in the docstring at
`:240-243` says the adapters were "trained and measured on bare zero-shot
prompts". **That is false for V5.** `results_v5_qwen3.csv.manifest.json` and
`results_v5_qwen05.csv.manifest.json` each list 10 cells including
`qwen2.5-3b-lora/{few_shot,cot,role_based,rag}`. The measured results:
`role_based` **0.9491**, `cot` 0.8148, `few_shot` 0.7593, `rag` 0.7454 against
`zero_shot` 0.9583. **Radius:** no measured number is wrong. But the demo cannot
show the second-best LoRA cell, and the comment asserts a fact about the
measurement that the manifests contradict — exactly the kind of thing an
examiner finds by reading one file. **[R2] The measured spread is itself a
result worth reporting: prompting on top of a fine-tuned adapter costs up to 21
points.**

**6. Isaac has no reset** — `sim/isaac_backend.py:627-644`, `dispatch()`
dispatches `move_to`, `pick_place`, `fetch`, `hold`, `inspect`, `stop`,
`reject`, then `raise KeyError`. No `reset`, no `respawn`, no scene restore.
`sim/isaac_bridge.py:107-113` only ever calls `backend.dispatch(action)`; its
`except Exception` at `:118` converts any failure into
`{"ok": False, "error": "execution_failed"}`. Combined with the documented
one-object-per-box rule, a demo that fills the three boxes cannot continue.
**Radius:** demo only, zero measured numbers — but the demo is the viva, and the
recovery is killing and relaunching Isaac in front of the examiner.

**7. `latency_s` excludes retrieval** — documented at `bench/run_cell.py:44-47`
("End-to-end for a rag cell is the SUM of the two — never report `latency_s`
alone as rag's latency") and at the column definitions `:104` and `:118`.
`bench/summary.py:341-353` computes and prints the sum correctly.
**Radius: small.** Measured retrieval medians are 5.3–6.7 ms, 0.1–0.7% of
end-to-end:

| model | llm p50 | retrieval p50 | e2e p50 | retrieval share |
|---|---|---|---|---|
| gpt-4o-mini | 800.7 ms | 5.3 ms | 806.2 ms | 0.7% |
| qwen2.5:3b | 1560.6 ms | 6.5 ms | 1567.1 ms | 0.4% |
| deepseek-r1:7b | 8469.8 ms | 6.7 ms | 8476.5 ms | 0.1% |

The Pareto entry "804 ms" sits between the two. The residual risk is that
`analyze.py:103` and `figures.py:273` read `latency_s` directly with no
retrieval term, so any figure they produce understates RAG by ~5 ms — below
the resolution of any claim being made.

---

## Code quality — only what could produce a wrong number

- **Duplicated aggregation.** Finding 3. Four scripts compute accuracy four
  times; two disagree. The scorer itself (`llm/score.py`) is correctly
  single-sourced and called from one place.
- **Duplicated backfill.** Finding 17. 263 lines × 2, differing in two.
- **Manifest history silently discarded.** `bench/run_grid.py:166-169`: a
  `json.loads` failure on an existing manifest sets `log = []`, and the next
  line appends and writes — destroying every prior launch record for that output
  file. Provenance is the one thing that must fail loudly.
- **Deliberate broad catches, correctly used.** `llm/pipeline.py:35, 44, 64`
  convert every failure into a recorded `stage` value. This is the documented
  design — "failures are RECORDED, not raised" — and it is right: a model error
  must not kill a 12,152-row run. Not a defect.
- **Silent degradation in the demo path.** `sim/isaac_bridge.py:118` catches
  every exception into a generic `execution_failed` result. It does call
  `traceback.print_exc()`, so it is not fully silent. Demo only.
- **Long functions.** `run_cell.run_cell` 111 lines (`:225-336`),
  `run_grid.main` 131 lines (`:208-339`), `run_grid.write_run_manifest` 96
  lines (`:76-172`), `summary.sec_retrieval` 82 lines, `rescore.main` 90 lines.
  `run_cell` is the one that matters — it holds the warmup flag, the retrieval
  branch, the scoring call and the row write in one scope, which is how the
  warmup-denominator issue stayed invisible.
- **No mutable default arguments** anywhere in `bench/`, `llm/`, `schema/`,
  `ft/`. Checked.
- **Absolute paths.** Only one is functional: `ft/merge_and_export.py:35`,
  `LLAMA_CPP = Path(r"C:\Users\USER\llama.cpp")`. The rest are in docstrings,
  manifests (where recording the real argv is correct) and `docs/DEMO_PORTABLE.md`
  (a personal runbook).
- **Secrets: clean in git.** `.env` exists, is gitignored at `.gitignore:4` and
  `:47`, and has never been committed (`git log --all -- .env` is empty). No key
  patterns in any tracked file or anywhere in history. **[R2] Git is not the
  whole surface — see finding 28 for the USB drive.**

---

## [R2] Changelog

**Revision 2 — 17 September 2026.** Added findings 24–32. Promoted finding 10
(no V5 intervals) and added findings 24 (held-out reuse) and 25 (missing cost
axis for the winning cell) to BLOCKER. Recorded the 17 September scope freeze
and its consequences for the fix list and the expected viva questions. Moved the
`21 of 27` correction from "can wait" to "before writing". Updated the leakage
verdict to distinguish leakage from reuse. Recorded the resolution of findings 4,
6 and 11 in `audit/DEFENCE_NUMBERS.md` revision 2, and the two dependencies
(findings 5 and 25 on the fix list) that revision 2 states openly rather than
resolves.
