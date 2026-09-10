# Running the Isaac demo on another machine

Written 7 September 2026, for the visit to the supervisor's PC (Windows, RTX 5090).

**The one thing to understand first:** `git clone` gives you the code and nothing
else. Every model this demo runs is excluded from the repo by
`.gitignore:32 ft/runs/**/checkpoint-*/` and by the standing rule never to track
weights. Two of the models — the merged LoRA adapters — were built on the author's
machine and **exist nowhere on the internet**. `ollama pull qwen2.5-3b-lora`
will fail. Without the USB drive, rungs 3 and 4 do not exist and the demo is a
cloud model talking to a simulator.

---

## What must travel on the USB drive

Verified on the source machine, 7 September 2026.

| # | Source path | Size | Without it |
|---|---|---|---|
| 1 | `D:\LLM\blobs` | 10.58 GB | no local models at all |
| 2 | `D:\LLM\manifests` | tiny | Ollama cannot see any tag |
| 3 | `C:\Users\USER\llm-6dof-arm\ft\runs\distilbert_cls\epoch_09` | 265 MB | **no rung 4** — the 19 ms result |
| 4 | `C:\Users\USER\.cache\huggingface` | 7.71 GB | classifier base and Whisper both fail |
| 5 | `C:\Users\USER\llm-6dof-arm\.env` | 1 KB | no gpt-4o-mini |

Total ≈ 18.6 GB.

`manifests` holds exactly six files — `7b`, `3b`, `0.5b`, `3b`, `latest`,
`latest`. The two `latest` entries are the LoRA tags. If those two are missing
after the copy, stop and recopy; nothing else can produce them.

**Not needed:** `D:\LLM\ComfyUI` (26 GB), `D:\LLM\ai-models`, the standalone
`Qwen3.6-27B…gguf` (14 GB), the LoRA `.safetensors` checkpoints under
`ft/runs/*_lora/` (already merged into the Ollama tags), `Modelfile_*.txt`
(provenance, not runtime). Excluding these is the difference between 18.6 GB and
59 GB.

**`.env` carries a live OpenAI key.** Delete it from the USB drive afterwards, or
bring the drive home.

### Making the drive

```powershell
robocopy D:\LLM E:\thesis_demo\LLM blobs manifests /E /NFL /NDL
robocopy C:\Users\USER\llm-6dof-arm\ft\runs\distilbert_cls\epoch_09 E:\thesis_demo\distilbert_cls\epoch_09 /E /NFL /NDL
robocopy C:\Users\USER\.cache\huggingface E:\thesis_demo\hf_cache /E /NFL /NDL
copy C:\Users\USER\llm-6dof-arm\.env E:\thesis_demo\.env
```

Change `E:` to the actual drive letter. `robocopy` with `/NFL /NDL` prints
nothing until it finishes — that is not a hang. Check **FAILED : 0** in the
summary table.

Two traps that already cost an hour on the source machine:

- **`$usb` is per-window.** A variable set in one PowerShell session does not
  exist in another, and robocopy silently wrote 24 GB to `C:\` when it expanded
  to nothing. Use literal paths.
- **`/XD Qwen3` excludes a *directory* named Qwen3.** The 14 GB Qwen3 item in
  `D:\LLM` is a file, so it was never excluded. Naming `blobs manifests`
  positionally is safer than excluding things.

Verify before unplugging:

```powershell
Get-ChildItem E:\thesis_demo\LLM\manifests -Recurse -File | Select-Object -ExpandProperty Name
```

Six names. If you see fewer, the copy is incomplete.

---

## Setup on the target machine, in order

Each step has a check. Do not move on until the check passes — a failure at
step 3 is invisible until step 8, and by then the cause is three steps back.

### 1. Isaac Sim 6.0

Install it first; everything else is faster and this is the only step that can
fail for reasons outside your control.

**Do not install 5.1.** It crashes in `rtx.scenedb.plugin` on driver branch
R610/R616. `[Guessing]` on the 5090 specifically: it is Blackwell, and whether
Isaac 6.0's shipped driver requirement matches what is installed there is
unknown. If Isaac will not start, this is the most likely reason and it is not
something a runbook can fix on the day. **This is why the fallback video exists.**

Check: Isaac Sim launches and shows a viewport.

### 2. Conda environments

Two, and they must stay separate. `thesis` is Python 3.11; `isaac6` is Python
3.12 and holds Isaac. **Never install `isaacsim` into `thesis`** — the two
cannot share a process.

```powershell
conda create -n thesis python=3.11 -y
conda activate thesis
pip install -r requirements.txt
```

`isaac6` comes from the Isaac Sim installer, not from pip.

Check: `conda activate thesis; python -c "import mujoco"` prints nothing and exits 0.

### 3. Repo

```powershell
git clone https://github.com/chappiethebot/llm-6dof-arm
cd llm-6dof-arm
```

Check: `git log --oneline -1` shows `e5ac03d` or later.

### 4. Ollama store

Install Ollama, then point it at the copied store **before starting it**. Ollama
reads `OLLAMA_MODELS` at launch; setting it afterwards has no effect until the
service restarts.

```powershell
[Environment]::SetEnvironmentVariable("OLLAMA_MODELS", "D:\LLM", "User")
```

Copy `blobs` and `manifests` from the USB to whatever path you set. Then open a
**new** terminal so the variable is in scope, and restart the Ollama service.

Check — this is the critical one:

```powershell
ollama list
```

Six models, including `qwen2.5-3b-lora:latest` and `qwen2.5-0.5b-lora:latest`.
If those two are absent, rung 3 is gone and there is no way to recover it on
that machine.

### 5. Classifier weights

Copy `distilbert_cls\epoch_09` from the USB to
`<repo>\ft\runs\distilbert_cls\epoch_09`. All five files: `model.pt`,
`tokenizer.json`, `vocab.txt`, `tokenizer_config.json`,
`special_tokens_map.json`.

### 6. HuggingFace cache

Copy `hf_cache` from the USB to `C:\Users\<them>\.cache\huggingface`.

This supplies the `distilbert-base-uncased` architecture the classifier loads on
top of `model.pt`, and the `small.en` Whisper model for voice.
`asr/voice.py` sets `HF_HUB_OFFLINE=1`, so it reads the cache and does not
download. It relaxes that for exactly one retry if a model is missing — which
needs internet, and is the slow path you are trying to avoid.

### 7. API key

Copy `.env` to the repo root.

### 8. Smoke test — no Isaac, no arm

```powershell
conda activate thesis
python bench\llm_client.py --list
python bench\llm_client.py --backend isaac --model rules --selftest
python bench\llm_client.py --backend isaac --model classifier:distilbert --strategy classifier --selftest
python bench\llm_client.py --backend isaac --model qwen2.5-3b-lora --selftest
python bench\llm_client.py --backend isaac --model gpt-4o-mini --selftest
```

`--list` must show eight models plus `rules`. Each selftest must end with
`all strategies valid and correct`. This exercises every rung without needing
Isaac running, so a failure here isolates cleanly to models rather than to
simulation.

### 9. Full demo

Terminal 1:
```powershell
conda activate isaac6
python sim\isaac_bridge.py
```

Terminal 2:
```powershell
conda activate thesis
python bench\llm_client.py --backend isaac
```

---

## Running the demo

Interactive mode. Type a command, or `v` to speak (push-to-talk: Enter to start,
Enter to stop). Switch without restarting:

```
model gpt-4o-mini
model qwen2.5-3b-lora
model classifier:distilbert
strategy rag
```

`--list` prints which strategies each model accepts. The LoRA models are
zero_shot only and the classifier is `classifier` only — those restrictions are
not arbitrary. The adapters were fine-tuned and measured on bare zero-shot
prompts, so any other prompt is off-distribution for them, and the classifier is
an encoder over the command string with no prompt surface at all. The client
refuses an illegal combination rather than silently running it.

### The sequence that shows the thesis

Same command, three models, watching the `llm_s` field:

```
model gpt-4o-mini
put the rubiks cube in box 1
model qwen2.5-3b-lora
put the rubiks cube in box 1
model classifier:distilbert
put the rubiks cube in box 1
```

Roughly 700 ms → 1.8 s → 19 ms. That inversion — a 66M-parameter classifier
beating a fine-tuned 3B on both speed and accuracy — is the result the thesis is
built on, and this is it running live.

Then:

- `put the bottle in box 3` followed by `put the toy car in box 3` — the second
  is refused, `box_occupied`, because one box holds one object and occupancy is
  read from live poses rather than tracked in a variable.
- `put the bottle anywhere you see free` — resolves `nearest_free_cell` against
  the live scene and reports which cell it chose in a `resolved` field.

### Things that will bite on stage

- **Say "the rubiks cube", not "the red cube".** The canonical legend has no
  colour glossary. A colour word makes `reject` the *correct* answer.
- **Say the words, not the symbol.** "the far left corner", not "grid A one".
- **First RAG command pays a one-off ~3 s** to load embedding weights. It is
  outside the timed block so `llm_s` stays honest, but there is a visible pause.
  Fire one throwaway RAG command before anyone is watching.
- **`handover` is unreachable** in the default canonical mode. Needs
  `--vocab backend`.
- **`move_to` reports `ok: false`** at ~4 cm. See the known-issues list below.
- **ASR errors become confident wrong actions.** Constrained decoding forces a
  legal symbol, so the model cannot answer "I'm not sure" — it must pick from
  the enum. Whisper hearing "card" for "car" produced a confident `rubiks_cube`.
  That is worth saying out loud if it happens; it is a real property of the
  design, not an embarrassment.

---

## Known issues — say these before you are asked

**`move_to` stops ~4 cm short and reports `ok: false`.** `MOVE_TOL` is 0.02 and
the best observed is 0.029. The cause is understood: `PickPlaceController` is a
grasp state machine being used to hover, and its approach phase carries offsets
it never removes because it expects to descend and close. Fixing it needs
RMPflow or direct IK. `pick_place` uses a different path and holds 0.1–0.5 cm,
which is the figure in the thesis.

**The column half of the grid mapping is a convention, not a fact.** Verified
physically on 7 September: schema column 3 ("right") maps to the side nearest
the boxes, which is the robot's own left. So left/right are stated from the
**operator's** view, facing the robot across the table. Correct for an audience
standing opposite; mirrored for anyone standing behind the robot. This needs one
sentence in the thesis.

**Isaac produces no measured rows.** Every number in the results chapter comes
from MuJoCo. An Isaac result is not comparable to a MuJoCo row — different
simulator, different robot, different dynamics. This demo shows the system
working; it is not evidence.

---

## If Isaac will not start

Show the video. It was recorded on a machine where the whole pipeline works, it
shows all three rungs and the latency comparison, and it is the mitigation
already named in the plan's risk register. A driver incompatibility on an
unfamiliar GPU on the morning of a meeting is not a problem to debug live.
