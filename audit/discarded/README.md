# Discarded run — V4a local, 2026-08-28

`results_v4a_local.CONTAMINATED.csv` was produced by TWO CONCURRENT run_grid
processes writing to the same output file.

The first launch used `nohup ... &` from inside a tool call. It appeared dead —
`ps` reported no matching process — so a second launch was started. The first
was in fact still alive. Both ran all 16 cells to completion against the same
Ollama server and appended to the same CSV.

## Why the data cannot be salvaged

- 6,944 rows = exactly 2 x 3,472. Every key has 2 copies.
- **775 of 3,472 duplicated keys (22.3%) DISAGREE on `pred_json`.**
- Local models were measured 100% byte-identical across 5 repeats in V3, so two
  runs of the same command on the same model must agree. They do not.
- Disagreement is present in every cell, from 2.8% (llama3.2 role_based) to
  72.4% (qwen2.5:0.5b cot).
- The mechanism is visible in the latencies. Example, same command and model:
      1.83 s  validated      {"action":"move_to","location":"grid_A1"}
      0.43 s  schema_failed  {"action":"put","target":"toy_car",...}
  0.43 s is far below llama3.2's 1.8 s p50. With OLLAMA_MAX_LOADED_MODELS=1 and
  OLLAMA_KEEP_ALIVE=0, two processes requesting different models evict each
  other's weights continuously, and requests are served by a thrashing server.

De-duplication cannot rescue this: for 775 keys there are two different answers
and no principled basis for choosing between them. Keeping "last" would silently
pick whichever process happened to write second.

## Harness gap this exposes

`bench/run_grid.py` has no protection against two writers on one output path.
The append-only resume rule is safe against a crashed run but not against a
concurrent one. A lockfile on `--out` would have refused the second launch.
Recorded as a finding; not fixed here.

Kept as evidence. Not used for any reported number.
