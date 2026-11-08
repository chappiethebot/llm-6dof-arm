"""Each of 24 corruptions of a clean synthetic campaign must turn into a FAIL that names it.

    python corrupt_test.py thesis3/out/synth      (after make_synthetic.py and analyze.py)"""
import json, os, shutil, subprocess, sys
import numpy as np, pandas as pd
import tempfile
B = os.path.dirname(os.path.abspath(__file__))
SRC = sys.argv[1]
TMP = tempfile.mkdtemp()
ARGS = ["--subset", "--partial", "--min-load-s", "20", "--min-idle-s", "29", "--cooldown-s", "4", "--loadedidle-s", "5"]
def calls(d, rid): return pd.read_csv(f"{d}/calls/{rid}.csv.gz", dtype=str)
def wcalls(d, rid, c): c.to_csv(f"{d}/calls/{rid}.csv.gz", index=False, compression="gzip")
def runs(d): return pd.read_csv(f"{d}/runs.csv", dtype=str)
def tel(d, s): return pd.read_csv(f"{d}/telemetry_{s}.csv", dtype=str)
def c1(d):  c = calls(d, "r1a"); wcalls(d, "r1a", c.drop(c[(c["pass"] == "1") & (c.cmd_id == "7")].index))
def c2(d):  c = calls(d, "r1a"); m = (c["pass"] == "1") & (c.cmd_id == "5"); c.loc[m, "cmd_id"] = "6"; wcalls(d, "r1a", c)
def c3(d):  c = calls(d, "r2a"); c.loc[c.index[5], "action_ok"] = ""; wcalls(d, "r2a", c)
def c4(d):  r = runs(d); r.loc[r.run_id == "r2a", "legend"] = "v4"; r.to_csv(f"{d}/runs.csv", index=False)
def c5(d):  r = runs(d); r.loc[r.run_id == "r1a", "dataset_sha256"] = ""; r.to_csv(f"{d}/runs.csv", index=False)
def c6(d):  m = json.load(open(f"{d}/machine.json")); m.pop("datasets"); json.dump(m, open(f"{d}/machine.json", "w"))
def c7(d):
    t = tel(d, "s1"); r = runs(d).set_index("run_id"); a = int(r.loc["r2a", "t_load_start_mono_ns"]) + 50 * 10**9
    t = t[~((t.run_id == "r2a") & (t.t_mono_ns.astype(float) > a) & (t.t_mono_ns.astype(float) < a + 6e9))]; t.to_csv(f"{d}/telemetry_s1.csv", index=False)
def c8(d):
    p = f"{d}/outputs/r1a.jsonl"; L = open(p).read().splitlines(); L = [l for l in L if not (json.loads(l)["pass"] == 1 and json.loads(l)["cmd_id"] == 9)]
    open(p, "w").write("\n".join(L) + "\n")
def c9(d):  t = tel(d, "s1"); t = t[~((t.run_id == "r2a") & (t.phase == "loadedidle"))]; t.to_csv(f"{d}/telemetry_s1.csv", index=False)
def c10(d): shutil.copy(f"{d}/calls/r1a.csv.gz", f"{d}/calls/r9z.csv.gz")
def c11(d): c = calls(d, "r5a"); m = (c["pass"] == "1") & (c.cmd_id == "3"); c.loc[m, "loc_true"] = "grid_C3"; wcalls(d, "r5a", c)
def c12(d):
    t = tel(d, "s2"); t.loc[t.run_id == "r3a", "cpu_pkg_power_w"] = "30.0"; t.to_csv(f"{d}/telemetry_s2.csv", index=False)
def c13(d): c = calls(d, "r1a"); c.loc[c["pass"] == "0", "cmd_id"] = ["10", "11", "12"]; wcalls(d, "r1a", c)
def c14(d): r = runs(d); r.loc[r.run_id == "r2a", "t_load_end_mono_ns"] = str(int(r.loc[r.run_id == "r2a", "t_load_start_mono_ns"].iloc[0]) + 100 * 10**9); r.to_csv(f"{d}/runs.csv", index=False)
def c15(d): r = runs(d); r.loc[r.run_id == "r3a", "idle_id"] = "idle_s9_0001"; r.to_csv(f"{d}/runs.csv", index=False)
def c16(d): i = pd.read_csv(f"{d}/idle.csv", dtype=str); i.loc[1, "idle_id"] = i.loc[0, "idle_id"]; i.to_csv(f"{d}/idle.csv", index=False)
def c17(d): x = np.load(f"{d}/latency/r1a.npy"); np.save(f"{d}/latency/r1a.npy", x[:-10])
def c18(d): r = runs(d); r.loc[r.run_id == "r1a", "system"] = "flat_gpu_v2"; r.to_csv(f"{d}/runs.csv", index=False)
def c19(d): r = runs(d); r.loc[r.run_id == "r4a", "repeats"] = "1"; r.to_csv(f"{d}/runs.csv", index=False)
def c20(d): r = runs(d); r.loc[r.run_id == "r2a", "gpu_temp_start_c"] = "68"; r.to_csv(f"{d}/runs.csv", index=False)
def c21(d): r = runs(d); r = pd.concat([r, r[r.run_id == "r1a"].assign(run_id="r1c")]); r.to_csv(f"{d}/runs.csv", index=False)
def c22(d): os.remove(f"{d}/passes/r3a.csv")
def c23(d): r = runs(d); r.loc[r.run_id == "r2a", "model_load_s"] = ""; r.to_csv(f"{d}/runs.csv", index=False)
def c24(d): p = pd.read_csv(f"{d}/passes/r3a.csv"); p.loc[p.index[-1], "calls"] = 100; p.to_csv(f"{d}/passes/r3a.csv", index=False)
names = {f"c{i}": d for i, d in enumerate([c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19,c20,c21,c22,c23,c24], 1)}
desc = ["216 commands", "cmd 5 replaced by 6", "blank action_ok", "legend v4", "empty dataset sha", "no datasets in machine.json",
        "6 s telemetry gap", "raw output missing", "no loadedidle", "orphan calls file", "truth changed in one run", "stuck CPU sensor",
        "warm-up on wrong commands", "load window cut", "idle window not found", "duplicate idle id", "latency file short",
        "unknown system", "cloud repeats 1", "hot start 68 C", "duplicated run", "passes file missing", "model_load_s empty", "incomplete pass"]
base = subprocess.run([sys.executable, f"{B}/verify_campaign.py", SRC] + ARGS, capture_output=True, text=True).stdout
base_fails = {l for l in base.splitlines() if l.startswith("FAIL")}
silent = 0
for (k, f), ds in zip(names.items(), desc):
    d = os.path.join(TMP, f"corrupt_{k}"); shutil.rmtree(d, ignore_errors=True); shutil.copytree(SRC, d); f(d)
    pr = subprocess.run([sys.executable, f"{B}/verify_campaign.py", d] + ARGS, capture_output=True, text=True)
    out = pr.stdout
    if "VERDICT" not in out:
        print(f"CRASH   {ds:28s} {pr.stderr.strip().splitlines()[-1][:100]}"); silent += 1; continue
    new = [l for l in out.splitlines() if l.startswith("FAIL") and l not in base_fails]
    print(f"{'CAUGHT' if new else 'SILENT'}  {ds:28s} {new[0][6:120] if new else ''}")
    silent += not new
    shutil.rmtree(d, ignore_errors=True)
print(f"\n{len(names) - silent} of {len(names)} corruptions caught")
