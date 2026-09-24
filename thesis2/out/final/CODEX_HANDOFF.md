# Continuation from Claude Code, 2026-09-23

[Certain] Recovered the latest user rulings from Claude's project session
`108a013d-dd66-4b0f-a5f1-024d760c2c9d`, user message at
2026-09-23T09:08:35Z. Claude stopped after creating `thesis2rl` and writing the
adapter probe, before running it. The user reserved the legacy training runs
for themselves and instructed Claude to stop after preparing these changes.

## Completed

- [Certain] Saved an untrained 3B LoRA adapter in `thesis2rl`, loaded and merged
  it via PEFT in `thesis2gpu`, and decoded 5/5 train_v2 commands into objects.
  This proves adapter compatibility, not task accuracy. Evidence:
  `grpo_adapter_probe.json` and `logs/s0_grpo_adapter_probe_load_20260923_154405.log`.
- [Certain] Added `s0_grpo_signal_check.py`: eight temperature-1 samples per
  train_v2 prompt, exact-match reward from `llm/score.py`, skip SFT GRPO below
  10% mixed prompts. The same script performs greedy dev inference in thesis2gpu.
- [Certain] Added `s0_train_grpo.py`: separate training environment, LoRA-only
  epoch artifacts, dev inference in a thesis2gpu subprocess, earliest epoch
  within one dev hit of the best, and explicit 3B OOM reporting.
  Training settings follow the brief; microbatch 1, accumulation 8, and BF16
  are recorded implementation choices. TRL's original GRPO loss is explicitly
  selected rather than relying on its current default; see
  [TRL loss types](https://huggingface.co/docs/trl/grpo_trainer#loss-types).
- [Certain] Added `s0_retrain_legacy.py` and `legacy_recipes.py` for the six
  legacy arms. Training bodies preserve the original scripts and seeds.
  Selected checkpoints are persisted and hashed before v3 comparison.
  Exactly 217/217 identical predictions are required to freeze; mismatches
  retain all differing command IDs and the status "retrained, not reproduced".
- [Certain] Connected the persisted legacy and GRPO inference paths to the arm
  registry. Fixed frozen directory hashing and the registry's `--list` CLI.
  Updated stale GRPO status text and `RUN_ORDER.md`. R=1 is accepted; H6 stays
  in the eight-hypothesis family.

## Validation and limits

- [Certain] `test_stage0.py` passes: AST comparison of all six training bodies,
  exact reward, plateau selection, full v3 ID coverage, rejection of changed
  frozen directories, legacy freeze gate, 22/230 versus 23/230 signal boundary,
  and untrained BiLSTM checkpoint reload including OOV and truncation behavior.
  Evidence: `logs/test_stage0_20260923_155204.log`.
- [Certain] `s0_train_grpo.py --arm grpo_base --check-config` constructs the
  installed TRL configuration successfully; `s1_run_arm.py --list` lists 30 arms.
- [Certain] No full training, signal sampling, new held-out model inference,
  cloud calls or package changes were performed. The actual training loop,
  GPU memory use, epoch-to-dev subprocess transition and six-arm reproduction
  results remain unmeasured until the user runs them. Downstream Stages 1-6
  were not audited end to end during this Stage-0 continuation.
- [Certain] `git diff --stat` is empty because all this work remains in the
  previously untracked `thesis2/final`, `thesis2/demo` and `thesis2/out/final`
  directories. This is proof of unchanged tracked files, not absence of new work.
  No commit or push was made.
- [Certain] `git rev-parse HEAD:thesis2/out/baseline_rules_firewall.csv` and
  `git hash-object --no-filters thesis2/out/baseline_rules_firewall.csv` both return
  `2400f2d541d07943f50a967e66c8c1639eb0a13d`: the raw bytes equal the committed blob.

## SALSA disagreements: recovered report, no prompt changes

[Certain] These are Claude's existing diagnostic results from 2026-09-23,
recovered from its terminal output, not a new inference run. All JSON below
includes the original empty modifiers field. The model was
Qwen2.5-3B-Instruct plus `ft/runs/qwen3b_lora/checkpoint-87`.

1. **move to the far side middle edge of the table**

   True: `{"action":"move_to","location":"grid_A2","modifiers":[]}`

   SALSA: `{"action":"move_to","location":"grid_A1","modifiers":[]}`

   Different field: location. Chosen letter A = grid_A1 (p=0.7467);
   truth was B = grid_A2. Exact location options:
   `A=grid_A1 B=grid_A2 C=grid_A3 D=grid_B1 E=grid_B2 F=grid_B3 G=grid_C1 H=grid_C2 I=grid_C3 J=max_reach K=nearest_free_cell`.
   Verb D = move_to (p=0.9978); target not asked.

2. **put the bottle in anywhere you see free**

   True: `{"action":"pick_place","target":"bottle","location":"nearest_free_cell","modifiers":[]}`

   SALSA: `{"action":"inspect","target":"bottle","modifiers":[]}`

   Different fields: action and missing location. Chosen verb E = inspect
   (p=0.4982); truth was A = pick_place. Exact verb options:
   `A=pick_place B=fetch C=hold D=move_to E=inspect F=stop G=reject`.
   Location was not asked because the predicted verb inspect requires none.
   Target B = bottle (p=0.9987), options
   `A=rubiks_cube B=bottle C=toy_car`.

3. **unscrew the cap off the bottle**

   True: `{"action":"reject","modifiers":[]}`

   SALSA: `{"action":"fetch","target":"bottle","modifiers":[]}`

   Different fields: action and extra target. Chosen verb B = fetch
   (p=0.7877); truth was G = reject. Exact verb options:
   `A=pick_place B=fetch C=hold D=move_to E=inspect F=stop G=reject`.
   Chosen target B = bottle (p=0.9934), exact target options
   `A=rubiks_cube B=bottle C=toy_car`; absence of a target was not an option
   after the wrong fetch prediction. Location was not asked.

## Next execution

[Certain] Use `thesis2/final/RUN_ORDER.md`, Stage 0. S0-5 has the signal check
and GRPO commands; S0-5b has the six legacy commands. The compatibility probe
already passed. Review each failing gate, and review all Stage-0 results before
starting Stage 1. No result here asserts that an unrun arm has reproduced.
