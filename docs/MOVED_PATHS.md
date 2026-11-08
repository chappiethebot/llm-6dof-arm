# Moved paths (repository cleanup, November 2026)

Every result was produced before this cleanup; the state before it is tag
`thesis-results-v1`. Tracked files were moved with `git mv`, so `git log --follow`
shows their history. Nothing was deleted.

## In the repository

| old path | new path | why |
|---|---|---|
| `smoke_test.py` | `scripts/smoke_test.py` | root scripts still in use go to `scripts/` |
| `validate.py` | `scripts/validate.py` | same; usage lines in the file updated |
| `make_split.py` | `scripts/make_split.py` | same; usage lines updated. It reads paths relative to the current directory, so it still runs from the repo root |
| `scratch_*.py` (16 files) | `archive/scratch/scratch_*.py` | one-off development checks, not needed to reproduce results |
| `inspect_smoke.py` | `archive/scratch/inspect_smoke.py` | same |
| `vision/` | `archive/unused_phases/vision/` | phase cut on 17 Sep 2026; placeholder only |
| `real/` | `archive/unused_phases/real/` | phase cut on 17 Sep 2026; placeholder only |
| `rl/` | `archive/unused_phases/rl/` | phase cut on 31 Aug 2026; placeholder only |
| `map.md` | `docs/REPO_MAP.md` | documentation goes to `docs/`; updated to the new layout |

References updated in the same step: the usage lines of `scripts/validate.py` and
`scripts/make_split.py`, the comment in `.gitignore` that names the map, and the 20
file links in `audit/REPO_AUDIT_INVENTORY_2026-09-29.md` that pointed at the moved
files. Plain-text mentions of the old names in the dated audit reports (`audit/*.md`)
are left as written: they describe the repository on their date.

## Isaac demo videos (gitignored; attached to GitHub release v1.0)

| old path | new path |
|---|---|
| `ISSAC SIM RECORDS/DistilBERT.mp4` | `media/isaac_demo/distilbert.mp4` |
| `ISSAC SIM RECORDS/GPT40MINI+COT.mp4` | `media/isaac_demo/gpt4omini_cot.mp4` |
| `ISSAC SIM RECORDS/GPT40MINI+FEWSHOT.mp4` | `media/isaac_demo/gpt4omini_fewshot.mp4` |
| `ISSAC SIM RECORDS/GPT40MINI+RAG.mp4` | `media/isaac_demo/gpt4omini_rag.mp4` |
| `ISSAC SIM RECORDS/GPT40MINI+ROLEBASED.mp4` | `media/isaac_demo/gpt4omini_rolebased.mp4` |
| `ISSAC SIM RECORDS/QWEN2.53B+ZEROSHOT.mp4` | `media/isaac_demo/qwen25_3b_zeroshot.mp4` |

## Moved out of the repository (never tracked)

| old path | new location | why |
|---|---|---|
| `records/` | `C:\Users\USER\llm-6dof-arm-private\records\` | private thesis-writing material, not code |
| `data.zip`, `schema.zip`, `v5_measurement_backup.zip` | `C:\Users\USER\llm-6dof-arm-backups\` | local backups |

## Not moved

| path | why |
|---|---|
| `audit/` | stays at the root because its scripts resolve the repository root from their own folder (21 of them take the parent of `audit/` as the root), so under `docs/audit/` they would read the wrong paths. `thesis2/scripts/p1_attribution.py` (frozen) also opens `audit/cls_margins.csv`, and the frozen packages and `thesis2/out/final/METHODS_NOTES.md` cite `audit/...` paths |
