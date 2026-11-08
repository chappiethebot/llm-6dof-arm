# archive/

Kept for provenance only. Nothing here is needed to reproduce any result in the thesis.

## scratch/

`scratch_*.py` and `inspect_smoke.py` were one-off checks run during development
(label space, modifiers, matrix integrity, gate checks, phrasing inventory, environment
probes, smoke-output inspection). They were moved here from the repository root
unchanged. Some import `llm` and read paths relative to the repository root, so they
only run from the root with the root on the path (`$env:PYTHONPATH = "."`).

## unused_phases/

`vision/`, `rl/` and `real/` were placeholders (`.gitkeep` only) for phases that were
cut and are not part of the thesis results:

- `rl/` (reinforcement-learning motion control): cut on 31 Aug 2026.
- `vision/` (object detection) and `real/` (a real-arm backend): cut on 17 Sep 2026.

Nothing in the repository imports them.
