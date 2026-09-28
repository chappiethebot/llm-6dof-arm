# Slide fixes to apply by hand (2026-09-28)

The slide deck is **not in the repository** (no `.pptx` anywhere in the working
tree or in git), and `python-pptx` is not installed in `thesis2gpu` or the system
Python. Per the instruction, nothing was installed: apply these two edits by hand.
[Certain]

## Slide 22 -- placement error

| | text |
|---|---|
| **old** | `0.1–0.5 cm` (the "placement" figure, as recorded in `audit/AUDIT_REPORT.md:772-776` and `audit/DEFENCE_NUMBERS.md:325`, section "7. NOT reportable — do not quote") |
| **new** | `0.2–0.4 cm` -- and label it **`pick_place`** placement error |

Source: `docs/SESSION_2026-09-07.md:106-108` -- live hardware selftest, "`pick_place`
at 0.2–0.4 cm. `[Certain]`". The old range was wrong at both ends and mixed
`pick_place` with `move_to`, whose error is a different and much larger number
(`audit/DEFENCE_NUMBERS.md`). [Certain that this is the verified figure; I have not
seen the slide itself, so the exact old wording on it is as the audit recorded it
-- Likely]

Two other places carried the old range. `docs/DEMO_PORTABLE.md:262` described the
same `pick_place` placement figure ("... holds 0.1–0.5 cm, which is the figure in
the thesis") and was corrected to 0.2–0.4 cm on 2026-09-28. The code comment in
`sim/isaac_backend.py:49` is untouched (`sim/` is read-only).

## Slide 17 -- what `n = 539` counts

No text change is required if the number stays, but the slide should say what it
counts. Suggested caption: **"n = 108 grid commands (539 rows: 5 repeats, warm-up
excluded)"**.

**What 539 is** [Certain]: the number of **rows**, not commands, in the V3-era
LoRA cells on the grid commands of `held_out_v3`:

- source file: `bench/results_heldout_v3_lora_scored.csv`, cells
  `ollama:qwen2.5-0.5b-lora / zero_shot` and `ollama:qwen2.5-3b-lora / zero_shot`;
- 108 grid commands x 5 repeats = 540 rows, minus the one warm-up row
  (`command_id 0` is a grid command, run 0) = **539**;
- row accuracy on those 539 rows: **0.306** (0.5B) and **0.408** (3B), the values in
  `audit/V4_RESULTS.md:75-76` and printed as the "sealed measurement" comparison by
  `audit/lora_vocab_test.py:101-103`;
- recomputed 2026-09-28 from the CSV: 539 rows, 108 distinct commands, repeats
  {0: 107, 1-4: 108 each}, row accuracy 0.306 / 0.408.

It is the only 539 in the repository's documents and scripts, so it is [Likely]
the slide's number. The independent sample size behind it is **108 commands**;
the 5 repeats are not independent observations, so quoting n = 539 overstates the
evidence by about 5x.
