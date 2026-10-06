# thesis3: fast path from here to the end (no more Claude Code needed)

Save these 5 files into `C:\Users\USER\llm-6dof-arm\thesis3\bench\`:
`block.ps1`, `finish.ps1`, `diag_energy.py`, `diag_load.py`, `RUNBOOK_FAST.md`.
The Claude Code session can be closed; it already built `registry.py`, `worker.py`, `run_bench.py`.

## Once, today (about 15 min)
1. Kaspersky: Settings > Network settings > Encrypted connections scanning > Manage exclusions >
   add `api.openai.com` (menu names may differ slightly by version).
2. Windows Update: Settings > Windows Update > Pause for 2 weeks.
3. NVIDIA App: overlay off, Instant Replay off.
4. Unblock the scripts (Windows marks downloaded files):

       cd C:\Users\USER\llm-6dof-arm
       Unblock-File thesis3\bench\block.ps1, thesis3\bench\finish.ps1

5. Energy check, about 4 min, do not touch the PC while it runs:

       D:\condaenvs\thesis3meter\python.exe thesis3\bench\diag_energy.py

   It does not block the campaign: every run keeps all three GPU power sources, so the choice
   of which one to quote can be made later from the data. Paste the two tables to Claude when
   you next have budget.
6. Commit the harness:

       git add .gitignore thesis3\bench
       git commit -m "thesis3 unified bench: harness, checks, run scripts"
       git push origin HEAD

## Each block, one per evening (order A, B, C1, C2, D)
Before: LibreHardwareMonitor open as administrator; close Claude, browsers, Isaac.

    powershell -ExecutionPolicy Bypass -File C:\Users\USER\llm-6dof-arm\thesis3\bench\block.ps1 -Block A

The script does everything: pre-checks, a 3-command dry run of every system in the block,
the block itself, verify, backup to `D:\thesis3_backup\c1`, backup check. When it says
"block 3/6", switch the monitor off and leave the PC alone.

| Block | Content | Time |
|---|---|---|
| A | CPU systems | about 2 h |
| B | transformers GPU systems | about 5.5 h |
| C1 | Ollama cells except deepseek-r1 | about 10.5 h |
| C2 | the 5 deepseek-r1 cells | about 9 h |
| D | gpt-4o-mini cells and systems that call it | about 4 h |

- Green `BLOCK X COMPLETE` = go to the next block. Also copy `D:\thesis3_backup\c1` to OneDrive or USB.
- Red `STOPPED` = do not edit anything. Re-run the same command once (it resumes and skips
  finished runs). If it stops again, paste the red line and the lines above it to Claude.
- If verify names "files without a runs.csv row", run
  `D:\condaenvs\thesis3meter\python.exe thesis3\bench\run_bench.py --campaign c1 --redo <run_id>`
  then the block command again.
- Power cut or crash: just run the block command again.

## After block D

    powershell -ExecutionPolicy Bypass -File C:\Users\USER\llm-6dof-arm\thesis3\bench\finish.ps1

Green `FINAL VERDICT PASS` = the data is complete. Send Claude `thesis3\out\thesis3_results.zip`.

## While blocks run
Write the Methods chapter. It needs none of these numbers.
