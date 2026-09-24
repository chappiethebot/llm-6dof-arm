# Measuring Isaac's own VRAM, and the "fits on 16 GB" rule

The GPU is one RTX 4060 Ti, **16,380 MiB**. A demo parser and Isaac share it, so
"does this parser fit" is a question about the *sum*, and it has to be measured
with the demo scene actually loaded — not guessed from the model's weight size.

This file is the procedure and the decision rule. It is a procedure for *you* to
run: nothing here was measured by the author of these scripts, and no number in
it is filled in. [Certain — the file is deliberately unpopulated]

---

## 1. Measure Isaac's baseline with the scene loaded

Isaac's own footprint is the fixed cost. Measure it with the demo scene up and
**no parser loaded**.

```powershell
conda activate D:\condaenvs\isaac6 ; python -m sim.run_demo --model qwen2.5-3b-lora --strategy zero_shot
```

Let the viewer finish loading the scene, then in a second terminal:

```powershell
nvidia-smi --query-gpu=memory.used,memory.total --format=csv ; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

Record, in `thesis2/out/final/isaac_vram_baseline.txt`:

- `memory.used` with the scene loaded and idle — call this **ISAAC_MB**;
- the same after one `pick_place` has executed (physics allocates during motion)
  — call this **ISAAC_PEAK_MB**, and use this one, not the idle figure;
- the compute-apps list, so it is on record that nothing else held the GPU.

`ISAAC_PEAK_MB` is the number the rule below uses. Do not substitute a
remembered value: it changes with the scene, the Isaac version and the driver.

## 2. Measure each parser's peak VRAM alone

`thesis2/demo/smoke_parsers.py` already reports `peak_vram_mb` per parser, from
`torch.cuda.max_memory_allocated()` over the 10 seen demo commands, with Isaac
**not** running:

```powershell
conda activate D:\condaenvs\thesis2gpu ; python thesis2\demo\smoke_parsers.py
```

Two cautions about that number:

- `max_memory_allocated()` is the **allocator's** peak, not the process's
  reservation. `nvidia-smi` will read higher (CUDA context, cuDNN workspaces,
  fragmentation). For the fit decision, read `nvidia-smi` while the parser is
  resident, and treat the allocator figure as a lower bound. [Certain]
- a parser that holds two models (`cascade_local_tauA`,
  `pipeline_always_local`) peaks at the sum, and the smoke test showed
  ~6,860 MB for those against ~6,593 MB for `lora_greedy` alone.

Record each parser's `nvidia-smi` peak as **PARSER_MB**.

## 3. The rule

For each GPU parser:

```
HEADROOM_MB = 16380 - ISAAC_PEAK_MB - PARSER_MB
```

| condition | mark the parser |
|---|---|
| `HEADROOM_MB >= 1500` | **fits** |
| `0 <= HEADROOM_MB < 1500` | **fits, no margin** — usable for a demo, but state it |
| `HEADROOM_MB < 0` | **does not fit** |

The 1,500 MB margin is a judgement, fixed here in advance so it is not chosen
after seeing which parsers pass: it covers the allocator/`nvidia-smi` gap in §2
plus driver overhead. [Guessing on the exact figure; Certain that a margin is
needed]

## 4. No silent fallback

If a parser is marked **does not fit**, `run_isaac_demo.py` must fail with that
parser named. It must never quietly substitute a smaller model: a demo that
silently ran `flat_gpu` while the slide said `lora_greedy` is a false claim
about the system. If you want the smaller parser, ask for it by name on the
command line.

CPU-only parsers (`flat_cpu`, `rules_open`, `rules_firewalled`) and cloud
parsers (`cascade_cloud_*`) add no VRAM and are always marked **fits**; for them
the constraint is CPU time and the network, not memory.

## 5. What to write down

Fill this table into `thesis2/out/final/isaac_vram_fit.csv` once measured:

| parser | PARSER_MB (nvidia-smi) | ISAAC_PEAK_MB | HEADROOM_MB | verdict |
|---|---|---|---|---|
| flat_gpu | | | | |
| factorized_fp16 | | | | |
| factorized_int4 | | | | |
| textcnn | | | | |
| mlpbow | | | | |
| lora_greedy | | | | |
| salsa_full_qwen2.5-3b-lora | | | | |
| salsa_full_qwen2.5-0.5b-lora | | | | |
| cascade_local_tauA | | | | |
| pipeline_always_local | | | | |

Two expectations worth testing rather than assuming: a 3B FP16 parser at
~6.6 GB plus Isaac is likely to be the tight case, and the two-model cascades
are likely to be the first to fail. Neither is established here. [Likely]
