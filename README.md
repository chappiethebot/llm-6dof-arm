# llm-6dof-arm

Code, data and results of an MSc thesis at TU Dresden. A user gives a robot arm a command in
plain English, for example "put the bottle in the back left corner of the table". The system
turns it into a checked JSON action, and the arm carries it out in a simulator. The thesis
compares 66 ways of doing this (rules, small classifiers, local and cloud language models) on
accuracy, speed and energy.

The registered thesis title includes a vision pipeline with real-time object detection. That part
was not carried out; Section 8.3 of the thesis (Scope and future work)
explains why.

Built with Qwen.

## What is in this repository

| folder | what it holds |
|---|---|
| `data/` | the command datasets: training, development and the two test sets (217 commands each) |
| `schema/` | the action format (`schema_v2.json`) and its validator |
| `llm/` | the model clients, prompts and scoring |
| `bench/` | the first evaluation harness and its results |
| `ft/` | training of the fine-tuned models |
| `sim/` | the MuJoCo and Isaac Sim robot simulations and the demo |
| `thesis2/` | the final run: every final number is in `thesis2/out/final/` |
| `thesis3/` | the unified benchmark of all 66 systems |
| `scripts/` | installation check and dataset check |
| `envs/` | the conda environment files |
| `docs/` | `RESULTS_MAP.md` (which file each thesis figure and table comes from), `REPO_MAP.md` (detailed folder map) |

## What you need

- Windows 11 with [Git](https://git-scm.com) and Miniconda or Anaconda. Everything was run on
  one PC: AMD Ryzen 5 7600X, 32 GB RAM, NVIDIA RTX 4060 Ti 16 GB.
- An NVIDIA GPU, only to re-run the classifier (step 5) and for the Isaac Sim demo (RTX).
- [Ollama](https://ollama.com), only for the local language models and the demo.
- An OpenAI API key, only for the gpt-4o-mini systems.

## Install

Open the "Anaconda PowerShell Prompt" from the Start menu (or a normal PowerShell window after
running `conda init powershell` once; otherwise `conda activate` does not work). Then, from the
folder you want the repository in:

```powershell
git clone https://github.com/chappiethebot/llm-6dof-arm.git
cd llm-6dof-arm
conda env create -n thesis -f envs\thesis.yml
conda env create -n thesis2gpu -f envs\thesis2gpu.yml
conda env create -n thesis3meter -f envs\thesis3meter.yml
```

These three environment files install every library the steps below need. (`requirements.txt`
is an older record and is not needed.)

If conda asks you to accept its Terms of Service, run `conda tos accept` and repeat the command.

If you want to use gpt-4o-mini, run `Copy-Item .env.example .env` and put your OpenAI key in
`.env`.

## Downloads

The trained models, the benchmark data and the demo videos are too large for git. Download
them from the [v1.0 release](https://github.com/chappiethebot/llm-6dof-arm/releases/tag/v1.0)
(only the ones you need). Put the downloaded zips in the repository folder and unpack each one
there; the files land in the right place:

```powershell
Expand-Archive thesis3_c1.zip -DestinationPath .
```

| file | what it is | needed for |
|---|---|---|
| `thesis3_c1.zip` | all raw data of the unified benchmark | step 4 |
| `distilbert_cls_epoch_09.zip` | the trained DistilBERT classifier | step 5 |
| `qwen3b-lora-Q4_K_M.gguf`, `qwen0.5b-lora-Q4_K_M.gguf`, `modelfiles.zip` | the two fine-tuned Qwen models for Ollama | local models and demo |
| `qwen3b_LICENSE_NOTICE.zip`, `qwen0.5b_LICENSE_NOTICE.zip` | the licence files for the two models | keep with the models |
| `qwen3b_lora_checkpoint-87.zip`, `qwen0.5b_lora_checkpoint-87.zip` | the fine-tuned adapters in Hugging Face format | using the adapters with transformers |
| `isaac_demo_videos.zip` | six recordings of the Isaac Sim demo | watching the demo |

`SHA256SUMS.txt` in the release lets you check every download
(`Get-FileHash <file> -Algorithm SHA256`).

## Check the results

Each step below prints a clear pass message. Steps 3 and 5 rewrite some result files with new
dates and timings; run `git checkout -- thesis2` afterwards to restore them.

### 1. Check the installation

```powershell
conda activate thesis
python scripts\smoke_test.py
python scripts\validate.py schema\schema_v2.json data\seen_v2.jsonl data\held_out_v3.jsonl
```

The smoke test ends with `10/10 components OK` (a warning about Adroit environments is harmless).
`validate.py` ends with `RESULT: PASS`.

### 2. Check the test sets

```powershell
Get-FileHash data\held_out_v3.jsonl, data\held_out_v4c.jsonl -Algorithm SHA256
```

```
5A767445A5BCBD717895B2D30D0725A1A3163AC9A0E7B9439FF7548131E8D687  ...\data\held_out_v3.jsonl
D507992EDE161ECE9A395C8746AD6790E57062ACC2A9A245A7703939EE8E95C8  ...\data\held_out_v4c.jsonl
```

These are the test sets the thesis used (its appendix gives the first 12 characters).

### 3. Rebuild the final-run tables and figures (about 90 s, no GPU)

```powershell
conda activate thesis2gpu
foreach ($s in "s2_metrics","s4_significance","s4_metrics_final","s5_extra_analysis","s4_figures","s6_report","make_defence_numbers","check_defence_numbers") { python thesis2\final\$s.py; if ($LASTEXITCODE -ne 0) { break } }
```

Each script ends with `GATE PASS`, except `make_defence_numbers`, which ends with
`641 cited numbers`. The tables and the six figures in `thesis2/figures/final/` come out
identical to the stored ones.

### 4. Rebuild the unified benchmark of 66 systems (about 2 min, no GPU)

Needs `thesis3_c1.zip` from the release, unpacked in the repository folder.

```powershell
conda activate thesis3meter
python thesis3\bench\verify_campaign.py thesis3\out\c1 --check-manifest
python thesis3\bench\analyze.py thesis3\out\c1
python thesis3\bench\verify_campaign.py thesis3\out\c1 --after-analysis
```

You see `MANIFEST PASS (754 files listed)`, then `analysed 132 runs`, then
`VERDICT PASS; 0 warnings`. The results (`summary.csv`, `compare.csv` and the others in
`thesis3\out\c1`) come out identical to the stored ones.

### 5. Re-run the DistilBERT classifier on both test sets (NVIDIA GPU, under 30 s)

Needs `distilbert_cls_epoch_09.zip` from the release, unpacked in the repository folder. The
first run downloads `distilbert-base-uncased` (about 260 MB).

```powershell
conda activate thesis2gpu
python thesis2\final\s1_run_arm.py --arm flat_gpu --split held_out_v3 --force-rerun-after-crash
python thesis2\final\s1_run_arm.py --arm flat_gpu --split held_out_v4c --force-rerun-after-crash
```

```
  action match 0.7465 (n=217); row accuracy 0.6204 (n=108)
  reproducibility vs saved held_out_v3: 217/217 identical
GATE PASS
  action match 0.7143 (n=217); row accuracy 0.5556 (n=108)
  reproducibility vs saved held_out_v4c: 217/217 identical
GATE PASS
```

## Local language models (Ollama)

The four stock models:

```powershell
ollama pull qwen2.5:3b
ollama pull qwen2.5:0.5b
ollama pull llama3.2:3b
ollama pull deepseek-r1:7b
```

The two fine-tuned models: put the two `.gguf` files from the release into `ft\export\`, unpack
`modelfiles.zip` in the repository folder, then:

```powershell
ollama create qwen2.5-3b-lora -f ft\export\Modelfile.qwen2.5-3b-lora
ollama create qwen2.5-0.5b-lora -f ft\export\Modelfile.qwen2.5-0.5b-lora
```

`audit/MODEL_PROVENANCE.txt` lists the exact model versions the thesis measured.

## Demo

Both demos need the Ollama models from the section above.

### MuJoCo

Type a command and watch the arm carry it out:

```powershell
conda activate thesis
python -m sim.run_demo --model qwen2.5-3b-lora --strategy zero_shot
```

Choose `1` (keyboard), then type a command, for example
`put the bottle in the far left corner of the table`. A simulator window opens and the arm
moves the block. Press Enter on an empty line to stop.

### Isaac Sim 6.0 (NVIDIA RTX GPU)

Create the Isaac environment once (a large download):

```powershell
conda env create -n isaac6 -f envs\isaac6.yml
```

Terminal 1, the simulator. Wait until it prints `[bridge] watching` (3 to 4 minutes; red
`[Error]` lines from Isaac Sim extensions during start-up are harmless):

```powershell
conda activate isaac6
python sim\isaac_bridge.py
```

Terminal 2, the language side:

```powershell
conda activate thesis
python bench\llm_client.py --backend isaac --model qwen2.5-3b-lora
```

Type a command, for example `put the rubiks cube in box 1`. The scene has a rubiks cube, a
bottle and a toy car, and boxes 1 to 3. More things to type:

- `model classifier:distilbert` switches to the DistilBERT classifier (needs step 5 run once)
- `reset` puts every object back
- `quit` stops the language side; press Ctrl+C in terminal 1 to close the simulator

The release file `isaac_demo_videos.zip` has six recordings of this demo.

## Where the thesis results are

- Final run (accuracy, speed, energy of every system): `thesis2/out/final/metrics_final.csv`
- Significance tests: `thesis2/out/final/final_significance.csv`
- Unified benchmark of all 66 systems: `thesis3/out/c1/summary.csv` (from `thesis3_c1.zip`)

`docs/RESULTS_MAP.md` lists the file behind every figure and table of the thesis.

## Licence

- Code: MIT, see [LICENSE](LICENSE).
- Data and results (`data/`, the result files and `thesis3_c1.zip`):
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- The fine-tuned 3B model is based on Qwen2.5-3B-Instruct and is for research and
  non-commercial use only (Qwen Research Licence). Qwen is licensed under the Qwen RESEARCH
  LICENSE AGREEMENT, Copyright (c) Alibaba Cloud. All Rights Reserved.
- The fine-tuned 0.5B model (Qwen2.5-0.5B-Instruct) and the DistilBERT classifier are under the
  Apache License 2.0.
- The MASSIVE dataset (Amazon Science, CC BY 4.0) is used through the MTEB copy
  `mteb/amazon_massive_intent`; the `thesis2/out/*massive*` files hold labels and predictions,
  not its sentences.

## How to cite

```bibtex
@mastersthesis{dhara2026llm6dof,
  author = {Ritayan Dhara},
  title  = {LLM-Driven Control and Vision-Based Manipulation for a 6-DoF Robotic Arm: Bridging Natural Language Commands with Real-Time Object Detection},
  school = {Technische Universität Dresden},
  year   = {2026}
}
```

## Contact

Ritayan Dhara, via GitHub issues on this repository.
