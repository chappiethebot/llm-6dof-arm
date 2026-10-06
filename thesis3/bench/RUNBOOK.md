# thesis3 unified bench: what the author does, step by step

Everything runs in PowerShell from `C:\Users\USER\llm-6dof-arm`.
Conventions: `;` not `&&`, one command per line.

## Step 1. Prepare the PC (once, about 20 min)
1. High-performance power plan:
   `powercfg /setactive SCHEME_MIN`
2. Pause Windows Update for 2 weeks (Settings > Windows Update > Pause).
3. Free disk space: at least 10 GB on C: and 10 GB on D: for the backup.
4. Install LibreHardwareMonitor (the CPU power sensor):
   - Download the latest release zip from https://github.com/LibreHardwareMonitor/LibreHardwareMonitor/releases
     and unzip it to `C:\Tools\LibreHardwareMonitor`.
   - Right-click `LibreHardwareMonitor.exe` > Run as administrator. If Kaspersky asks about
     its driver, allow it.
   - Options > Remote Web Server > Run (port 8085).
   - Options > Update Interval > 500 ms (or 1 s if 500 ms is not offered).
   - Check it works: open http://127.0.0.1:8085/data.json in a browser. You should see JSON
     text that contains "CPU Package" (Intel) or "Package" (AMD).
   - It must stay open, as administrator, for the whole campaign.

## Step 2. Create the measurement environment (once, 5 min)
The sampler and the analysis get their own small env, so the frozen envs (`thesis`,
`thesis2gpu`, `thesis2rl`) stay untouched.

    conda create -p D:\condaenvs\thesis3meter python=3.11 -y
    conda activate D:\condaenvs\thesis3meter
    pip install psutil nvidia-ml-py pandas numpy matplotlib

## Step 3. Put the files in place (once)

    cd C:\Users\USER\llm-6dof-arm
    git status
    New-Item -ItemType Directory -Force thesis3\bench | Out-Null
    Expand-Archive -Path "$HOME\Downloads\thesis3_bench.zip" -DestinationPath "$env:TEMP\t3" -Force
    Copy-Item "$env:TEMP\t3\thesis3_bench\*" thesis3\bench\ -Force
    Add-Content -Path .gitignore -Encoding ascii -Value "`nthesis3/out/"
    Get-ChildItem thesis3\bench

The list must show 9 files: telemetry.py, analyze.py, verify_campaign.py, make_synthetic.py,
corrupt_test.py, systems_expected.csv, SCHEMA.md, RUNBOOK.md, CLAUDE_CODE_PROMPT.md.
(`thesis3/out/` is git-ignored because the raw data is several GB; it is backed up in step 7.)

## Step 4. Check the sampler yourself (2 min)

    conda activate D:\condaenvs\thesis3meter
    python thesis3\bench\telemetry.py --selftest
    python thesis3\bench\telemetry.py --probe

- The self-test must end with `SELFTEST PASS`.
- The probe must end with `PROBE PASS` and a line `CPU sensor used: /intelcpu/0/power/0` (or
  `/amdcpu/...`). If it says FAIL: LibreHardwareMonitor is not running as administrator, its
  web server is off, or its update interval is slower than 1 s.

Stop here and paste me both outputs if anything fails.

## Step 5. Let Claude Code build the harness (about 1 to 2 h, it stops by itself)
1. Open Claude Code in `C:\Users\USER\llm-6dof-arm`.
2. Open `thesis3\bench\CLAUDE_CODE_PROMPT.md`, copy everything below the `---` line and
   paste it as the first message.
3. It writes `registry.py` and `run_bench.py`, runs five checks and stops.
4. Paste me the terminal output of the five checks and the file `thesis3\bench\PLAN.md`.
   **Do not start the campaign before I have checked them.** What I look for: the synthetic
   check shows exactly two FAIL lines (both r6a), `24 of 24 corruptions caught`, the smoke
   run gives sensible numbers, and the system -> env -> legend table is right.

## Step 6. Commit the code (after I have checked step 5)

    git add thesis3\bench .gitignore
    git commit -m "thesis3 unified bench: harness and checks"
    git push origin HEAD

## Step 7. Run the campaign, one block at a time
Order: A, B, C1, C2, D. PLAN.md gives the time of each block; C1 and C2 are the long ones
(C2, the five deepseek-r1 cells, is about 7.5 h on its own). Start a block in the evening.

Before every block: LibreHardwareMonitor open as administrator, all other programs closed,
nothing else running on the GPU (no Isaac, no browser video). Do not use the PC while a
block runs: anything you do adds to the measured power. The sampler keeps Windows and the
screen awake by itself.

    conda activate D:\condaenvs\thesis3meter
    cd C:\Users\USER\llm-6dof-arm
    python thesis3\bench\run_bench.py --campaign c1 --block A

After every block, before the next one:

    python thesis3\bench\verify_campaign.py thesis3\out\c1 --partial
    robocopy thesis3\out\c1 D:\thesis3_backup\c1 /MIR
    python thesis3\bench\verify_campaign.py D:\thesis3_backup\c1 --check-manifest

- The first command must end with `VERDICT PASS` (warnings about runs not yet done are expected).
- robocopy exit codes 0 to 7 mean success.
- The last command must print `MANIFEST PASS`.
- Also copy `D:\thesis3_backup\c1` off the machine (OneDrive or USB) after each block.

## Step 8. If something goes wrong
- **Crash, power cut, Windows restart:** run the same block command again. It skips every
  finished run and continues.
- **verify says "files without a runs.csv row":** that run was cut off. Run
  `python thesis3\bench\run_bench.py --campaign c1 --redo <run_id>` with the id it names,
  then verify again.
- **verify FAILs on a finished run:** paste me the FAIL line before re-running anything.
- Never edit or delete files in `thesis3\out` by hand.

## Step 9. After block D

    python thesis3\bench\analyze.py thesis3\out\c1
    python thesis3\bench\verify_campaign.py thesis3\out\c1 --after-analysis

The second command must end with `VERDICT PASS`. Only then is the data complete. Send me:
`summary.csv`, `report.md`, `quality.csv`, `compare.csv`, `per_loc_type.csv`, `per_verb.csv`,
`repro_check.csv` and `verify_report.txt` from `thesis3\out\c1`.

## While the blocks run
Write. The Methods chapter needs none of these numbers.
