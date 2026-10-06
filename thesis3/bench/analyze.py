"""thesis3 analysis: every metric from the unified bench, in one pass, one run at a time.

    python analyze.py thesis3/out/<campaign>

reads the files in SCHEMA.md and writes into the campaign folder:
    summary.csv              one row per run (system x test set), ~180 columns
    per_verb.csv             accuracy and latency per verb, per run
    per_loc_type.csv         accuracy per location type (grid / runtime / container / none)
    confusion.csv            verb and location confusion counts, per run (pass 1)
    per_pass.csv             accuracy and latency of every load pass (stability)
    per_command_energy.csv   net Wh of every call (slow systems only, >= 5 samples per call)
    compare.csv              paired comparisons vs reference systems (exploratory, Holm per split)
    quality.csv              measurement quality per run
    report.md, figures/

Units: power in W, energy in Wh (per command, per 1,000 commands) and kWh (per million
commands), CO2 in g CO2e. No joules anywhere. "Command" = one decided command: for
gpt-4o-mini (5 votes per command) one command costs 5 calls, and every per-command,
per-1,000 and per-million figure counts the 5 calls together; latency columns are per call,
`dec_lat_*` columns are per decided command (5 calls one after another).

Scale-up constants (change here, everything recomputes):
  CI_G_PER_KWH = 344   German electricity mix 2025, g CO2 per kWh (Umweltbundesamt 2026)
  PUE_DC       = 1.56  data-centre overhead for the at-scale scenario (Uptime Institute 2024)
  PRICE_EUR    = 0.370 EUR per kWh, German household average 2026 (BDEW Strompreisanalyse)
  ACT_WH_PANDA = 0.087 Wh, the cheapest measured point-to-point move of a Franka Panda
                       (311.98 J in 3.22 s; Petrichenko et al. 2024, arXiv 2411.03194, Table 1)
  DUTY_S       = 60    deployment scenario: one command per minute, model kept resident

Baseline tiers (thesis-defined, derived from the cited sources; see report.md):
  accuracy  floor 0.70 (Wickens & Dixon 2007) | research parity 0.85 (SayCan 84 %, KnowNo
            1-eps = 0.85) | supervised use: point >= 0.95 and Wilson lower bound >= 0.90
  safety    unsafe accepts and missed stops with an exact one-sided 95 % upper bound
            (Clopper-Pearson; for zero events ~ rule of three, Hanley & Lippman-Hand 1983);
            the hardware e-stop stays the safety function (ISO 13850, ISO 13849 PL d)
  latency   p95 <= 100 ms instant | <= 700 ms fielded dialogue robots (Inoue et al. 2025) |
            <= 1 s preferred (Shiwa et al. 2009) | <= 2 s tolerable | > 2 s sluggish
  energy    net per command <= 1 mWh negligible (~1 % of one arm move) | <= 10 mWh small |
            <= 87 mWh below one arm move | above: comparable to actuation
"""
from __future__ import annotations

import gzip
import json
import math
import os
import sys

import numpy as np
import pandas as pd

CI_G_PER_KWH = 344.0
PUE_DC = 1.56
PRICE_EUR = 0.370
ACT_WH_PANDA = 0.087
DUTY_S = 60.0
THRESH_MS = [100, 200, 700, 1000, 2000]
BOOT_N, BOOT_SEED = 2000, 20260827
BOOT_LAT_MAX = 50_000            # latency percentile CIs bootstrap a seeded subsample of at most 50,000 calls
MIN_SAMPLES_PER_CALL = 5          # per-command energy only when a call spans >= 5 samples (0.5 s at 10 Hz)
BLOCK_S = 30.0                    # energy stability: the load window is cut into 30 s blocks
MIN_DRIFT_S = 120.0               # latency drift only over load windows of at least 2 min
REFERENCES = ["lora_greedy", "flat_gpu"]   # paired comparisons are made against these when present
EXPECTED_N = 217
HERE = os.path.dirname(os.path.abspath(__file__))

WH = 1.0 / 3600.0                 # W*s -> Wh
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
NAN = float("nan")


# ------------------------------------------------------------------ helpers
def trapz_wh(t_ns, p_w) -> float:
    p_w = np.asarray(p_w, dtype=float)
    ok = ~np.isnan(p_w)
    if ok.sum() < 2:
        return NAN
    return float(_trapz(p_w[ok], np.asarray(t_ns, dtype=float)[ok] / 1e9)) * WH


def trapz_between(df: pd.DataFrame, col: str, a: float, b: float) -> float:
    """Wh of a power column between two instants, with the trace interpolated at both ends
    (the samples bracketing a and b come from the neighbouring phases of the same run)."""
    d = df[["t_mono_ns", col]].dropna()
    if len(d) < 2:
        return NAN
    t, p = d.t_mono_ns.values.astype(float), d[col].values.astype(float)
    if t[0] > a or t[-1] < b:
        return NAN
    inside = (t > a) & (t < b)
    ts = np.concatenate(([a], t[inside], [b]))
    ps = np.concatenate(([np.interp(a, t, p)], p[inside], [np.interp(b, t, p)]))
    return float(_trapz(ps, ts / 1e9)) * WH


GPU_ENERGY_SOURCE = "board_power"   # "board_power": integral of NVML board power (gpu_power_w, = nvidia-smi
                                    # power.draw; the reading the card's 165 W power limit is enforced on,
                                    # and the source thesis2 used). "counter": NVML total-energy counter.
                                    # On the RTX 4060 Ti (driver 617.14) the counter reads a near-constant
                                    # 13-17 % lower than every power reading, at idle and at steady load
                                    # (diag_energy, 6 Oct 2026); both are kept in summary.csv.


def gpu_wh(df: pd.DataFrame, a: float, b: float) -> float:
    """GPU Wh between two instants from the chosen source."""
    if GPU_ENERGY_SOURCE == "counter":
        return gpu_energy_between(df, a, b)
    return trapz_between(df, "gpu_power_w", a, b)


def gpu_energy_between(df: pd.DataFrame, a: float, b: float) -> float:
    """GPU Wh between two instants: NVML counter interpolated at both ends when present,
    else the interpolated trapezoid of power."""
    ec = df[["t_mono_ns", "gpu_energy_mj"]].dropna()
    if len(ec) >= 2 and ec.t_mono_ns.iloc[0] <= a and ec.t_mono_ns.iloc[-1] >= b:
        t, e = ec.t_mono_ns.values.astype(float), ec.gpu_energy_mj.values.astype(float)
        return float(np.interp(b, t, e) - np.interp(a, t, e)) / 1000.0 * WH
    return trapz_between(df, "gpu_power_w", a, b)


def boot_ci(x, n=BOOT_N, seed=BOOT_SEED):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return (NAN, NAN)
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), size=(n, len(x)))].mean(axis=1)
    return (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


def boot_pct_ci(x, q, n=1000, seed=BOOT_SEED, chunk=50):
    """bootstrap CI of a percentile over calls; at most BOOT_LAT_MAX calls (seeded subsample)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 10:
        return (NAN, NAN)
    rng = np.random.default_rng(seed)
    if len(x) > BOOT_LAT_MAX:
        x = rng.choice(x, BOOT_LAT_MAX, replace=False)
    est = []
    for i in range(0, n, chunk):
        k = min(chunk, n - i)
        est.append(np.percentile(x[rng.integers(0, len(x), size=(k, len(x)))], q, axis=1))
    est = np.concatenate(est)
    return (float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5)))


def wilson(k, n, z=1.96):
    if n == 0:
        return (NAN, NAN)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def binom_cdf(k, n, p):
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(0, k + 1))


def cp_upper95(k, n):
    """exact one-sided 95 % upper bound of an event rate (Clopper-Pearson). For k = 0 it is
    1 - 0.05**(1/n), which the rule of three (3/n) approximates."""
    if n == 0:
        return NAN
    if k >= n:
        return 1.0
    lo, hi = k / n, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if binom_cdf(k, n, mid) > 0.05:
            lo = mid
        else:
            hi = mid
    return hi


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n)


def holm(p):
    p = np.asarray(p, dtype=float)
    adj, run = np.empty_like(p), 0.0
    for r, i in enumerate(np.argsort(p)):
        run = max(run, min(1.0, (len(p) - r) * p[i]))
        adj[i] = run
    return adj


def _lab(v):
    """canonical label string: 1, 1.0, '1' and ' 1 ' all become '1'; missing stays NaN."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return np.nan
    t = str(v).strip()
    if t in ("", "nan", "None", "<NA>"):
        return np.nan
    try:
        f = float(t)
        return str(int(f)) if f.is_integer() else t
    except ValueError:
        return t


def loc_type(sym) -> str:
    s = _lab(sym)
    if not isinstance(s, str):
        return "none"
    if s.startswith("grid_"):
        return "grid"
    if s.startswith("box_"):
        return "container"
    if s in ("max_reach", "nearest_free_cell"):
        return "runtime"
    return "other"


def pct(x, q):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return float(np.percentile(x, q)) if len(x) else NAN


def fnan(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return NAN


def refresh_interval_s(t_ns, v) -> float:
    """median time between value changes: reveals a sensor that updates slower than we sample."""
    v = np.asarray(v, dtype=float)
    ok = ~np.isnan(v)
    t, v = np.asarray(t_ns)[ok], v[ok]
    if len(v) < 3:
        return NAN
    ch = np.nonzero(np.diff(v) != 0)[0]
    if len(ch) < 2:
        return NAN
    return float(np.median(np.diff(t[ch + 1])) / 1e9)


def ece(conf, correct, bins=15):
    conf, correct = np.asarray(conf, dtype=float), np.asarray(correct, dtype=float)
    ok = ~np.isnan(conf) & ~np.isnan(correct)
    conf, correct = conf[ok], correct[ok]
    if len(conf) < 20:
        return NAN
    idx = np.clip(np.digitize(conf, np.linspace(0, 1, bins + 1)[1:-1]), 0, bins - 1)
    return float(sum((idx == b).mean() * abs(conf[idx == b].mean() - correct[idx == b].mean())
                     for b in range(bins) if (idx == b).any()))


# ------------------------------------------------------------------ accuracy
FIELDS_OK = ["action_ok", "verb_ok", "target_ok", "loc_ok", "valid_json"]
ALWAYS_SCORED = ["action_ok", "verb_ok", "valid_json"]      # never empty: an empty value is scored wrong


def accuracy_block(c1: pd.DataFrame, repeats: int = 1):
    """c1 = pass-1 calls. Repeated calls (gpt-4o-mini): majority vote over ALL planned
    repeats; a failed or missing repeat counts as wrong, a tie is wrong."""
    c1 = c1.copy()
    for col in ["loc_true", "loc_pred", "confidence", "row_word"]:
        if col not in c1:
            c1[col] = np.nan
    n_blank = int(c1[ALWAYS_SCORED].isna().sum().sum())
    c1[ALWAYS_SCORED] = c1[ALWAYS_SCORED].astype(float).fillna(0.0)
    R = max(int(repeats), int(c1["repeat"].max()) if len(c1) else 1)
    if R > 1:
        def vote(s):
            if s.isna().all() and s.name in ("target_ok", "loc_ok"):
                return np.nan                                  # field absent in the truth
            return float(s.astype(float).fillna(0).sum() > R / 2)

        def label(s):
            vc = s.dropna().astype(str).value_counts()
            return vc.index[0] if len(vc) and vc.iloc[0] > R / 2 else "__no_majority__"
        g = c1.groupby("cmd_id")
        per = pd.DataFrame({c: g[c].agg(vote) for c in FIELDS_OK})
        per = per.join(g.agg({"row_true": "first", "col_true": "first", "verb_true": "first",
                              "row_word": "first", "loc_true": "first"}))
        per = per.assign(row_pred=g["row_pred"].agg(label), col_pred=g["col_pred"].agg(label),
                         verb_pred=g["verb_pred"].agg(label), loc_pred=g["loc_pred"].agg(label), confidence=np.nan)
        agree = float(g["action_ok"].agg(lambda s: max(s.mean(), 1 - s.mean())).mean())
    else:
        per = c1.drop_duplicates("cmd_id").set_index("cmd_id")
        agree = NAN
    n = int(len(per))
    out = {"n_commands": n, "blank_scores_pass1": n_blank,
           "errors_pass1": int(c1["error"].notna().sum()) if "error" in c1 else 0}
    for k in FIELDS_OK:
        out[k.replace("_ok", "") + "_acc" if k != "valid_json" else "valid_json_rate"] = float(per[k].astype(float).mean())
    a = per["action_ok"].astype(float).values
    out["action_ci_lo"], out["action_ci_hi"] = boot_ci(a)
    k_ok = int(np.nansum(a))
    out["action_correct"] = k_ok
    out["action_wilson_lo"], out["action_wilson_hi"] = wilson(k_ok, n)
    for col in ["row_true", "row_pred", "col_true", "col_pred", "loc_true", "loc_pred", "verb_true", "verb_pred"]:
        per[col] = per[col].map(_lab)
    per["loc_type"] = per["loc_true"].map(loc_type)
    grid = per[per.row_true.notna()]
    out["n_grid"] = int(len(grid))
    if len(grid):
        rt, rp = grid.row_true.astype(str), grid.row_pred.astype(str)
        cok = grid.col_pred.astype(str) == grid.col_true.astype(str)
        out["row_acc"], out["col_acc"] = float((rp == rt).mean()), float(cok.mean())
        out["row_and_col_acc"] = float(((rp == rt) & cok).mean())
        wrong = rp != rt
        out["row_err_mirror"] = int((wrong & (((rt == "A") & (rp == "C")) | ((rt == "C") & (rp == "A")))).sum())
        out["row_err_default_B"] = int((wrong & rt.isin(["A", "C"]) & (rp == "B")).sum())
        out["row_err_other"] = int(wrong.sum()) - out["row_err_mirror"] - out["row_err_default_B"]
        for grp, words in {"seen": ["middle", "center"], "unseen": ["upper", "bottom", "back", "front"],
                           "mixed": ["mixed"]}.items():
            gg = grid[grid.row_word.isin(words)]
            out[f"row_acc_{grp}"] = float((gg.row_pred.astype(str) == gg.row_true.astype(str)).mean()) if len(gg) else NAN
            out[f"n_row_{grp}"] = int(len(gg))
    vt, vp = per.verb_true.astype(str), per.verb_pred.astype(str)
    rej, st = vt == "reject", vt == "stop"
    out["n_reject"], out["n_stop"] = int(rej.sum()), int(st.sum())
    out["reject_recall"] = float((vp[rej] == "reject").mean()) if rej.any() else NAN
    out["unsafe_accepts"] = int(((vp != "reject") & rej).sum())        # impossible/unsafe command acted on
    out["unsafe_accept_rate_ub95"] = cp_upper95(out["unsafe_accepts"], out["n_reject"])
    out["false_rejects"] = int(((vp == "reject") & ~rej).sum())
    out["stop_recall"] = float((vp[st] == "stop").mean()) if st.any() else NAN
    out["missed_stops"] = int(((vp != "stop") & st).sum())
    out["missed_stop_rate_ub95"] = cp_upper95(out["missed_stops"], out["n_stop"])
    out["false_stops"] = int(((vp == "stop") & ~st).sum())
    out["cloud_repeat_agreement"] = agree
    out["action_ece"] = ece(per["confidence"].values, a) if R == 1 else NAN
    out["acc_tier"] = acc_tier(out["action_acc"], out["action_wilson_lo"])
    return out, per


def acc_tier(p, lo):
    if math.isnan(p):
        return "n/a"
    if p >= 0.95 and lo >= 0.90:
        return "supervised-use"
    if p >= 0.85:
        return "research-parity"
    return "above-floor" if p >= 0.70 else "below-floor"


def lat_tier(p95):
    if math.isnan(p95):
        return "n/a"
    for lim, name in [(100, "instant"), (700, "fluent"), (1000, "preferred"), (2000, "tolerable")]:
        if p95 <= lim:
            return name
    return "sluggish"


def energy_tier(wh_cmd):
    if math.isnan(wh_cmd):
        return "n/a"
    if wh_cmd <= 0.001:
        return "negligible"
    if wh_cmd <= 0.010:
        return "small"
    return "below-one-arm-move" if wh_cmd <= ACT_WH_PANDA else "comparable-to-actuation"


# ------------------------------------------------------------------ one run
def analyse_run(run, c, lat_all, passes, tel, iw):
    """run   = runs.csv row
    c       = calls/<run_id>.csv.gz (pass 0 and passes 1..row_passes, one row per call)
    lat_all = latency/<run_id>.npy (latency of EVERY load call, all passes)
    passes  = passes/<run_id>.csv (one row per load pass)
    tel     = the telemetry of the run's session (all phases; used around window edges)
    iw      = the idle window's telemetry rows"""
    mine = tel[tel.run_id == run.run_id]
    tt_all = tel.t_mono_ns.values

    def seg(a, b, pad=3e9):
        """telemetry around [a, b], whatever the phase, so both edges can be interpolated"""
        i0, i1 = np.searchsorted(tt_all, a - pad), np.searchsorted(tt_all, b + pad, side="right")
        return tel.iloc[i0:i1]
    out = {k: run.get(k, NAN) for k in ["run_id", "system", "family", "split", "kind", "device", "runtime", "legend"]}
    R = int(fnan(run.get("repeats", 1)) or 1)
    out["repeats"] = R
    rows_load = c[c["pass"] >= 1]
    warm = c[c["pass"] == 0].sort_values("t_start_mono_ns")
    acc, per = accuracy_block(c[c["pass"] == 1], R)
    out.update(acc)
    out["uses_cloud"] = int(str(run.get("uses_cloud", "0")) in ("1", "1.0", "True"))
    t0, t1 = int(run.t_load_start_mono_ns), int(run.t_load_end_mono_ns)
    T = (t1 - t0) / 1e9
    out["load_window_s"] = T

    # ---- stability over passes
    out["passes"] = int(len(passes))
    out["row_passes"] = int(rows_load["pass"].nunique())
    if out["row_passes"] >= 2 and R == 1:
        key = "pred_hash" if "pred_hash" in rows_load and rows_load.pred_hash.notna().any() else None
        if key is None:
            rows_load = rows_load.assign(_ph=rows_load[["verb_pred", "row_pred", "col_pred", "loc_pred"]].astype(str).agg("|".join, axis=1))
            key = "_ph"
        nun = rows_load.groupby("cmd_id")[key].nunique()
        out["pass_flip_rate"], out["pass_flip_cmds"] = float((nun > 1).mean()), int((nun > 1).sum())
    else:
        out["pass_flip_rate"], out["pass_flip_cmds"] = NAN, 0
    full = passes[passes.calls == EXPECTED_N * R]
    pa = (full.action_ok_sum / full.calls).values.astype(float)
    out["acc_full_passes"] = int(len(pa))
    out["acc_pass_sd"] = float(np.std(pa, ddof=1)) if len(pa) > 1 else NAN
    out["acc_pass_min"] = float(np.min(pa)) if len(pa) else NAN

    # ---- latency per call (every load call)
    lat = np.asarray(lat_all, dtype=float)
    n_calls = int(len(lat))
    n_dec = n_calls / R                                   # decided commands
    out.update({"calls": n_calls, "decisions": n_dec,
                "lat_mean_ms": float(np.mean(lat)), "lat_std_ms": float(np.std(lat, ddof=1)) if n_calls > 1 else NAN,
                "lat_min_ms": float(np.min(lat)), "lat_p50_ms": pct(lat, 50), "lat_p90_ms": pct(lat, 90),
                "lat_p95_ms": pct(lat, 95), "lat_p99_ms": pct(lat, 99), "lat_p999_ms": pct(lat, 99.9),
                "lat_max_ms": float(np.max(lat)), "lat_iqr_ms": pct(lat, 75) - pct(lat, 25)})
    out["lat_tail_ratio_p99_p50"] = out["lat_p99_ms"] / out["lat_p50_ms"] if out["lat_p50_ms"] > 0 else NAN
    out["lat_p50_ci_lo"], out["lat_p50_ci_hi"] = boot_pct_ci(lat, 50)
    out["lat_p95_ci_lo"], out["lat_p95_ci_hi"] = boot_pct_ci(lat, 95)
    for th in THRESH_MS:
        out[f"share_le_{th}ms"] = float((lat <= th).mean())
    # per decided command (repeats > 1: the R calls of one command, one after another)
    if R > 1:
        dl = rows_load.groupby(["pass", "cmd_id"]).latency_ms.sum().values.astype(float)
        out["dec_lat_p50_ms"], out["dec_lat_p95_ms"] = pct(dl, 50), pct(dl, 95)
    else:
        out["dec_lat_p50_ms"], out["dec_lat_p95_ms"] = out["lat_p50_ms"], out["lat_p95_ms"]
    out["lat_tier"] = lat_tier(out["dec_lat_p95_ms"])
    out["first_call_ms"] = float(warm.latency_ms.iloc[0]) if len(warm) else NAN
    out["first_call_over_p50"] = out["first_call_ms"] / out["lat_p50_ms"] if out["lat_p50_ms"] > 0 else NAN
    # drift over the load window (thermal or memory effects), ms per minute
    if T >= MIN_DRIFT_S:
        if len(passes) >= 5:
            tm = ((passes.t_start_mono_ns.astype(float) + passes.t_end_mono_ns.astype(float)) / 2 - t0) / 6e10
            ym = (passes.t_end_mono_ns.astype(float) - passes.t_start_mono_ns.astype(float)) / 1e6 / passes.calls
        else:
            tm = (rows_load.t_start_mono_ns.astype(float) - t0) / 6e10
            ym = rows_load.latency_ms.astype(float)
        out["lat_drift_ms_per_min"] = float(np.polyfit(tm, ym, 1)[0]) if np.ptp(tm) > 0 else NAN
    else:
        out["lat_drift_ms_per_min"] = NAN
    stop_ids = per.index[per.verb_true.astype(str) == "stop"]
    sl = rows_load[rows_load.cmd_id.isin(stop_ids)].latency_ms.values.astype(float)
    out["stop_lat_p95_ms"] = pct(sl, 95) if len(sl) else NAN
    if "t_infer_start_mono_ns" in rows_load and rows_load.t_infer_start_mono_ns.notna().any():
        inf = (rows_load.t_infer_end_mono_ns.astype(float) - rows_load.t_infer_start_mono_ns.astype(float)) / 1e6
        out["infer_p50_ms"] = pct(inf, 50)
        out["overhead_p50_ms"] = pct(rows_load.latency_ms.astype(float).values - inf.values, 50)
    else:
        out["infer_p50_ms"] = out["overhead_p50_ms"] = NAN
    def colmean(col):
        return float(rows_load[col].astype(float).mean()) if col in rows_load and rows_load[col].notna().any() else NAN
    out["tokens_in_mean"], out["tokens_out_mean"], out["tokens_thinking_mean"] = colmean("tokens_in"), colmean("tokens_out"), colmean("tokens_thinking")
    tok = rows_load.tokens_out.astype(float) if "tokens_out" in rows_load else pd.Series(np.nan, index=rows_load.index)
    tps = (tok / (rows_load.latency_ms.astype(float) / 1000)).replace([np.inf, -np.inf], np.nan)
    out["tokens_per_s_p50"] = float(tps.median()) if tps.notna().any() else NAN
    ttft = rows_load.ttft_ms.astype(float) if "ttft_ms" in rows_load else pd.Series(np.nan, index=rows_load.index)
    out["ttft_p50_ms"] = float(ttft.median()) if ttft.notna().any() else NAN
    out["ttft_p95_ms"] = pct(ttft, 95) if ttft.notna().any() else NAN
    tpot = ((rows_load.latency_ms.astype(float) - ttft) / (tok - 1))[tok > 1].replace([np.inf, -np.inf], np.nan)
    out["tpot_p50_ms"] = float(tpot.median()) if tpot.notna().any() else NAN
    out["throughput_cmd_per_s"] = n_dec / T if T > 0 else NAN

    # ---- power and energy over the load window (load-phase samples only)
    w = mine[(mine.phase == "load") & (mine.t_mono_ns >= t0) & (mine.t_mono_ns <= t1)]
    q = {"run_id": run.run_id,
         "sample_hz": float((len(w) - 1) / T) if T > 0 and len(w) > 1 else NAN,
         "late_sample_share": float((w.late_ms > 50).mean()) if len(w) else NAN,
         "max_gap_s": float(np.max(np.diff(w.t_mono_ns.values)) / 1e9) if len(w) > 1 else NAN,
         "nan_gpu_power_share": float(w.gpu_power_w.isna().mean()) if len(w) else NAN}
    if out["uses_cloud"] or run.kind == "cloud":
        out["energy_note"] = "uses gpt-4o-mini: the cloud part runs on OpenAI servers and is not measured; local energy only"
    for dev, col in [("gpu", "gpu_power_w"), ("cpu", "cpu_pkg_power_w")]:
        p = w[col].values.astype(float)
        ip = iw[col].values.astype(float) if len(iw) else np.array([])
        idle_med = float(np.nanmedian(ip)) if len(ip) and not np.all(np.isnan(ip)) else NAN
        cov = float(np.mean(~np.isnan(p))) if len(p) else 0.0
        out[f"{dev}_power_coverage"] = cov
        good = cov >= 0.95
        out[f"{dev}_idle_w"] = idle_med
        out[f"{dev}_idle_iqr_w"] = pct(ip, 75) - pct(ip, 25) if len(ip) else NAN
        out[f"{dev}_mean_w"] = float(np.nanmean(p)) if good else NAN
        out[f"{dev}_std_w"] = float(np.nanstd(p)) if good else NAN
        out[f"{dev}_p95_w"] = pct(p, 95) if good else NAN
        out[f"{dev}_peak_w"] = float(np.nanmax(p)) if good else NAN
        out[f"{dev}_net_mean_w"] = out[f"{dev}_mean_w"] - idle_med
        if dev == "gpu":
            sg = seg(t0, t1)
            gross = gpu_wh(sg, t0, t1) if good else NAN
            out["gpu_energy_source"] = GPU_ENERGY_SOURCE
            ctr = gpu_energy_between(sg, t0, t1) if w.gpu_energy_mj.notna().sum() >= 2 else NAN
            pw = trapz_between(sg, "gpu_power_w", t0, t1)
            out["gpu_gross_wh_counter"], out["gpu_gross_wh_board_power"] = ctr, pw
            out["gpu_counter_to_power_ratio"] = ctr / pw if pw and pw > 0 else NAN
            q["gpu_counter_vs_integral_pct"] = 100 * (pw - ctr) / ctr if ctr and ctr > 0 else NAN
        else:
            gross = trapz_between(seg(t0, t1), col, t0, t1) if good else NAN   # do not pretend a gappy CPU trace is complete
            q["cpu_sensor_refresh_s"] = refresh_interval_s(w.t_mono_ns.values, p)
        out[f"{dev}_gross_wh"], out[f"{dev}_net_wh"] = gross, gross - idle_med * T * WH
    e_net, e_gross = out["gpu_net_wh"] + out["cpu_net_wh"], out["gpu_gross_wh"] + out["cpu_gross_wh"]   # NaN if either is missing
    for basis, e in [("net", e_net), ("gross", e_gross)]:
        out[f"{basis}_wh_per_cmd"] = e / n_dec
        out[f"{basis}_wh_per_1k"] = e / n_dec * 1000
        out[f"{basis}_kwh_per_million"] = e / n_dec * 1e6 / 1000
    out["energy_complete"] = bool(not math.isnan(e_net))
    out["gpu_net_wh_per_1k"] = out["gpu_net_wh"] / n_dec * 1000
    out["cpu_net_wh_per_1k"] = out["cpu_net_wh"] / n_dec * 1000
    out["static_share_of_gross"] = (out["gpu_idle_w"] + out["cpu_idle_w"]) * T * WH / e_gross if out["energy_complete"] and e_gross > 0 else NAN
    unc = out["gpu_idle_iqr_w"] + (out["cpu_idle_iqr_w"] if not math.isnan(out["cpu_idle_iqr_w"]) else 0.0)
    out["net_wh_per_1k_idle_unc"] = unc / 2 * T * WH / n_dec * 1000     # +- half the idle IQR carried through the window
    acc_a = out.get("action_acc", NAN)
    out["net_wh_per_1k_correct"] = out["net_wh_per_1k"] / acc_a if acc_a and acc_a > 0 else NAN
    out["cmds_per_wh_net"] = 1 / out["net_wh_per_cmd"] if out["net_wh_per_cmd"] and out["net_wh_per_cmd"] > 0 else NAN
    out["wh_per_1k_tokens_out"] = (out["net_wh_per_cmd"] / R / out["tokens_out_mean"] * 1000
                                   if out["tokens_out_mean"] and out["tokens_out_mean"] > 0 else NAN)
    out["co2_g_per_million"] = out["net_kwh_per_million"] * CI_G_PER_KWH
    out["co2_g_per_million_dc"] = out["net_kwh_per_million"] * PUE_DC * CI_G_PER_KWH
    out["co2_g_per_million_gross"] = out["gross_kwh_per_million"] * CI_G_PER_KWH
    out["eur_per_million"] = out["net_kwh_per_million"] * PRICE_EUR
    out["eur_per_million_dc"] = out["net_kwh_per_million"] * PUE_DC * PRICE_EUR
    out["device_hours_per_million"] = out["lat_mean_ms"] * R * 1e6 / 1000 / 3600
    out["sci_gco2_per_cmd"] = out["co2_g_per_million"] / 1e6       # SCI operational part (embodied M not included)
    out["pct_of_panda_move"] = 100 * out["net_wh_per_cmd"] / ACT_WH_PANDA
    out["energy_tier"] = energy_tier(out["net_wh_per_cmd"])

    # ---- system-specific counters from `extra`
    out["escalation_rate"], out["retries_total"], out["accepted_tokens_mean"] = NAN, 0, NAN
    if "extra" in rows_load:
        ex = rows_load["extra"].dropna().map(lambda x: json.loads(x) if str(x).startswith("{") else {})
        for key, name, how in [("escalated", "escalation_rate", "mean"), ("retries", "retries_total", "sum"),
                               ("accepted_tokens", "accepted_tokens_mean", "mean")]:
            v = ex.map(lambda d: d.get(key)).dropna().astype(float) if len(ex) else pd.Series(dtype=float)
            if len(v):
                out[name] = float(v.mean()) if how == "mean" else int(v.sum())

    # ---- cold start and standby (logged under the system's v3 run id; copied to v4c in main)
    ml = mine[mine.phase == "modelload"]
    out["model_load_s"] = fnan(run.get("model_load_s", NAN))        # the harness's own timer
    out["model_load_net_wh"] = NAN
    if len(ml) > 1:
        a, b = float(ml.t_mono_ns.iloc[0]), float(ml.t_mono_ns.iloc[-1])
        Tm = (b - a) / 1e9
        if math.isnan(out["model_load_s"]):
            out["model_load_s"] = Tm
        sm_ = seg(a, b)
        out["model_load_net_wh"] = ((gpu_wh(sm_, a, b) - out["gpu_idle_w"] * Tm * WH)
                                    + (trapz_between(sm_, "cpu_pkg_power_w", a, b) - out["cpu_idle_w"] * Tm * WH))
    li = mine[mine.phase == "loadedidle"]
    if len(li) > 5:
        out["loaded_idle_s"] = float(np.ptp(li.t_mono_ns) / 1e9)
        out["loaded_idle_gpu_w"] = float(np.nanmedian(li.gpu_power_w))
        out["loaded_idle_cpu_w"] = float(np.nanmedian(li.cpu_pkg_power_w)) if li.cpu_pkg_power_w.notna().any() else NAN
        out["loaded_idle_net_w"] = (out["loaded_idle_gpu_w"] - out["gpu_idle_w"]) + (out["loaded_idle_cpu_w"] - out["cpu_idle_w"])
        out["loaded_idle_vram_mb"] = float(np.nanmedian(li.vram_used_mb))
    else:
        for k in ["loaded_idle_s", "loaded_idle_gpu_w", "loaded_idle_cpu_w", "loaded_idle_net_w", "loaded_idle_vram_mb"]:
            out[k] = NAN

    # ---- memory, thermal, process
    iv = float(np.nanmedian(iw.vram_used_mb)) if len(iw) else NAN
    out["vram_peak_mb"] = float(np.nanmax(w.vram_used_mb)) if len(w) else NAN
    out["vram_above_idle_mb"] = out["vram_peak_mb"] - iv
    out["rss_peak_mb"] = float(np.nanmax(w.proc_rss_mb)) if len(w) and w.proc_rss_mb.notna().any() else NAN
    dproc = np.diff(w.proc_cpu_s.values.astype(float)) if len(w) > 1 else np.array([])
    out["proc_cpu_ms_per_cmd"] = float(np.nansum(dproc[dproc > 0])) * 1000 / n_dec if len(dproc) else NAN
    out["cpu_util_mean_pct"] = float(np.nanmean(w.cpu_util_pct)) if len(w) else NAN
    out["gpu_util_mean_pct"] = float(np.nanmean(w.gpu_util_pct)) if len(w) else NAN
    out["gpu_temp_start_c"] = fnan(run.get("gpu_temp_start_c", NAN))
    out["gpu_temp_max_c"] = float(np.nanmax(w.gpu_temp_c)) if len(w) else NAN
    sm = w.gpu_sm_clock_mhz.astype(float)
    out["gpu_sm_clock_p50_mhz"] = float(sm.median()) if len(sm) else NAN
    out["throttle_flag"] = bool(len(sm) > 20 and sm.quantile(0.05) < 0.8 * sm.median() and out["gpu_util_mean_pct"] > 50)

    # ---- per-pass accuracy and latency; energy per 30 s block
    pp = [{"run_id": run.run_id, "system": run.system, "split": run.split, "pass": int(r.pass_), "calls": int(r.calls),
           "complete": bool(r.calls == EXPECTED_N * R), "dur_s": (r.t_end_mono_ns - r.t_start_mono_ns) / 1e9,
           "action_acc": r.action_ok_sum / r.calls, "lat_mean_ms": (r.t_end_mono_ns - r.t_start_mono_ns) / 1e6 / r.calls}
          for r in passes.rename(columns={"pass": "pass_"}).itertuples()]
    if len(passes) >= 2:
        units = passes[["t_start_mono_ns", "t_end_mono_ns", "calls"]].astype(float)
    else:
        units = rows_load[["t_start_mono_ns", "t_end_mono_ns"]].astype(float).assign(calls=1.0)
    eb = []
    if out["energy_complete"] and T >= 2 * BLOCK_S:
        for _, ub in units.groupby(((units.t_start_mono_ns - t0) / 1e9 // BLOCK_S).astype(int)):
            a, b = float(ub.t_start_mono_ns.min()), float(ub.t_end_mono_ns.max())
            Tb = (b - a) / 1e9
            if Tb < 0.5 * BLOCK_S:
                continue                                   # the short tail block is not representative
            sb = seg(a, b)
            e = ((gpu_wh(sb, a, b) - out["gpu_idle_w"] * Tb * WH)
                 + (trapz_between(sb, "cpu_pkg_power_w", a, b) - out["cpu_idle_w"] * Tb * WH))
            eb.append(e / (ub.calls.sum() / R) * 1000)
    ep = np.array(eb, dtype=float)
    ep = ep[~np.isnan(ep)]
    out["energy_blocks"] = int(len(ep))
    out["net_wh_per_1k_block_sd"] = float(np.std(ep, ddof=1)) if len(ep) > 1 else NAN
    out["net_wh_per_1k_block_cv_pct"] = 100 * out["net_wh_per_1k_block_sd"] / float(np.mean(ep)) if len(ep) > 1 and np.mean(ep) > 0 else NAN

    # ---- per-command energy (slow systems only: a call must span >= 5 samples and >= 3 CPU refreshes)
    pce = []
    refresh = q.get("cpu_sensor_refresh_s", NAN)
    min_ms = max(MIN_SAMPLES_PER_CALL * 100, 3 * refresh * 1000 if not math.isnan(refresh) else 0)
    out["per_cmd_energy_min_ms"] = min_ms
    if len(w) and out["lat_p50_ms"] >= min_ms and out["energy_complete"]:
        gp = w.gpu_power_instant_w.values.astype(float) if w.gpu_power_instant_w.notna().mean() > 0.95 else w.gpu_power_w.values.astype(float)
        net_p = (gp - out["gpu_idle_w"]) + (w.cpu_pkg_power_w.values.astype(float) - out["cpu_idle_w"])
        ok = ~np.isnan(net_p)
        tt, net_p = w.t_mono_ns.values[ok].astype(float), net_p[ok]
        for r in rows_load[["t_start_mono_ns", "t_end_mono_ns", "cmd_id", "pass", "latency_ms"]].itertuples(index=False):
            a, b = float(r.t_start_mono_ns), float(r.t_end_mono_ns)
            inside = (tt > a) & (tt < b)
            if inside.sum() + 2 < MIN_SAMPLES_PER_CALL or len(tt) < 2:
                continue
            ts = np.concatenate(([a], tt[inside], [b]))
            ps = np.concatenate(([np.interp(a, tt, net_p)], net_p[inside], [np.interp(b, tt, net_p)]))
            pce.append({"run_id": run.run_id, "cmd_id": r.cmd_id, "pass": r[3],
                        "latency_ms": r.latency_ms, "net_wh": float(_trapz(ps, ts / 1e9)) * WH})
    out["per_cmd_net_wh_p50"] = out["per_cmd_net_wh_p95"] = out["per_cmd_net_wh_max"] = NAN
    if pce:
        e = np.array([x["net_wh"] for x in pce])
        out["per_cmd_net_wh_p50"], out["per_cmd_net_wh_p95"], out["per_cmd_net_wh_max"] = pct(e, 50), pct(e, 95), float(e.max())

    q.update({"idle_samples": int(len(iw)), "idle_window_s": float(np.ptp(iw.t_mono_ns) / 1e9) if len(iw) > 1 else NAN,
              "load_samples": int(len(w))})

    # ---- per verb, per location type, confusion (pass 1, voted)
    per = per.assign(_lat=rows_load.groupby("cmd_id").latency_ms.median().reindex(per.index).values)
    per_verb = [{"run_id": run.run_id, "system": run.system, "split": run.split, "verb": v, "n": int(len(g)),
                 "action_acc": float(g.action_ok.astype(float).mean()), "lat_p50_ms": pct(g._lat, 50)}
                for v, g in per.groupby(per.verb_true.fillna("none").astype(str))]
    per_loc = []
    for lt, g in per.groupby("loc_type"):
        k = int(g.action_ok.astype(float).sum())
        lo, hi = wilson(k, len(g))
        per_loc.append({"run_id": run.run_id, "system": run.system, "split": run.split, "loc_type": lt, "n": int(len(g)),
                        "action_acc": k / len(g), "wilson_lo": lo, "wilson_hi": hi,
                        "loc_acc": float(g.loc_ok.astype(float).mean()) if g.loc_ok.notna().any() else NAN})
        out[f"action_acc_{lt}"] = k / len(g)
        out[f"n_{lt}"] = int(len(g))
    conf = []
    for field in ["verb", "loc"]:
        tr, pr = per[f"{field}_true"].fillna("none").astype(str), per[f"{field}_pred"].fillna("none").astype(str)
        for (t, p), k in pd.crosstab(tr, pr).stack().items():
            if k:
                conf.append({"run_id": run.run_id, "system": run.system, "split": run.split, "field": field,
                             "true": t, "pred": p, "count": int(k)})
    return out, pd.DataFrame(pce), q, per_verb, per_loc, conf, pp, per["action_ok"].astype(float).rename(run.run_id), w


# ------------------------------------------------------------------ comparisons
def comparisons(summary: pd.DataFrame, correct: dict) -> pd.DataFrame:
    """paired, per split: every ranked system vs each reference present. Exploratory: Holm
    over each (split, reference) family. Not the preregistered H1-H8."""
    rows = []
    for split, d in summary[summary.ranked].groupby("split"):
        for ref in REFERENCES:
            rr = d[d.system == ref]
            if rr.empty:
                continue
            a = correct[rr.run_id.iloc[0]]
            fam = []
            for r in d.itertuples():
                if r.system == ref:
                    continue
                j = pd.concat([a, correct[r.run_id]], axis=1, join="inner").dropna()
                if j.empty:
                    continue
                x, y = j.iloc[:, 0].values, j.iloc[:, 1].values
                lo, hi = boot_ci(y - x)
                nb, nc = int(((x == 1) & (y == 0)).sum()), int(((x == 0) & (y == 1)).sum())
                fam.append({"split": split, "reference": ref, "system": r.system, "n": int(len(j)),
                            "diff_pts": 100 * (y - x).mean(), "ci_lo_pts": 100 * lo, "ci_hi_pts": 100 * hi,
                            "ref_only_correct": nb, "sys_only_correct": nc, "p_mcnemar": mcnemar_exact(nb, nc)})
            if fam:
                for f, p in zip(fam, holm([f["p_mcnemar"] for f in fam])):
                    f["p_holm"], f["separable"] = float(p), bool(p < 0.05)
                rows += fam
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ report
def fmt(v, nd=2):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, float):
        if abs(v) >= 1000:
            return f"{v:,.0f}"
        if abs(v) >= 1:
            return f"{v:.{nd}f}"
        return f"{v:.{max(nd, 3)}g}"
    return str(v)


def md_table(df: pd.DataFrame, cols: list[str], heads: list[str]) -> str:
    lines = ["| " + " | ".join(heads) + " |", "|" + "---|" * len(heads)]
    for _, r in df.iterrows():
        vals = [r.get(c, NAN) for c in cols]
        lines.append("| " + " | ".join(v if isinstance(v, str) else fmt(v) for v in vals) + " |")
    return "\n".join(lines)


def figures(summary: pd.DataFrame, traces: dict, figdir: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(figdir, exist_ok=True)
    for rid, w in traces.items():
        if not len(w):
            continue
        t = (w.t_mono_ns - w.t_mono_ns.iloc[0]) / 1e9
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.plot(t, w.gpu_power_w, lw=0.8, label="GPU W")
        if w.cpu_pkg_power_w.notna().any():
            ax.plot(t, w.cpu_pkg_power_w, lw=0.8, label="CPU package W")
        ax.set_xlabel("time in load window (s)"); ax.set_ylabel("power (W)"); ax.legend(loc="upper right")
        ax.set_title(rid); fig.tight_layout(); fig.savefig(os.path.join(figdir, f"trace_{rid}.png"), dpi=120); plt.close(fig)
    r = summary[summary.ranked]
    for split, d in r[r.energy_complete.astype(bool)].groupby("split"):
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(d.net_wh_per_1k, d.action_acc * 100)
        for _, x in d.iterrows():
            ax.annotate(x.system, (x.net_wh_per_1k, x.action_acc * 100), fontsize=6)
        ax.axvline(ACT_WH_PANDA * 1000, ls=":", lw=0.8, color="grey")
        ax.axhline(70, ls=":", lw=0.8, color="grey"); ax.axhline(95, ls=":", lw=0.8, color="grey")
        ax.set_xscale("log"); ax.set_xlabel("net energy, Wh per 1,000 commands (log)"); ax.set_ylabel("action match (%)")
        ax.set_title(f"Accuracy vs energy, {split}"); fig.tight_layout()
        fig.savefig(os.path.join(figdir, f"pareto_energy_{split}.png"), dpi=130); plt.close(fig)
    for split, d in r.groupby("split"):
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(d.dec_lat_p95_ms, d.action_acc * 100)
        for _, x in d.iterrows():
            ax.annotate(x.system, (x.dec_lat_p95_ms, x.action_acc * 100), fontsize=6)
        for v in (100, 1000, 2000):
            ax.axvline(v, ls=":", lw=0.8, color="grey")
        ax.set_xscale("log"); ax.set_xlabel("p95 latency per command (ms, log)"); ax.set_ylabel("action match (%)")
        ax.set_title(f"Accuracy vs latency, {split}"); fig.tight_layout()
        fig.savefig(os.path.join(figdir, f"pareto_latency_{split}.png"), dpi=130); plt.close(fig)


STR_COLS = ["session", "run_id", "idle_id", "system", "family", "split", "kind", "device", "runtime", "legend",
            "row_true", "row_pred", "col_true", "col_pred", "verb_true", "verb_pred", "loc_true", "loc_pred",
            "row_word", "error", "extra", "phase", "notes", "model_sha256", "dataset_sha256", "uses_cloud", "pred_hash"]
TEL_COLS = ["t_mono_ns", "run_id", "phase", "gpu_power_w", "gpu_power_instant_w", "gpu_energy_mj",
            "gpu_util_pct", "vram_used_mb", "gpu_temp_c", "gpu_sm_clock_mhz", "cpu_pkg_power_w",
            "cpu_util_pct", "proc_cpu_s", "proc_rss_mb", "late_ms"]


def read(path: str, usecols=None) -> pd.DataFrame:
    head = pd.read_csv(path, nrows=0).columns
    if usecols is not None:
        usecols = [c for c in usecols if c in head]
    return pd.read_csv(path, usecols=usecols, low_memory=False,
                       dtype={c: str for c in STR_COLS if c in head and (usecols is None or c in usecols)})


def system_classes() -> dict:
    p = os.path.join(HERE, "systems_expected.csv")
    if not os.path.exists(p):
        return {}
    se = pd.read_csv(p, dtype=str)
    return dict(zip(se.system, se.get("system_class", pd.Series("system", index=se.index)).fillna("system")))


def main(root: str):
    runs = read(os.path.join(root, "runs.csv"))
    idle_tab = read(os.path.join(root, "idle.csv"))
    classes = system_classes()
    tel_cache: dict[str, pd.DataFrame] = {}

    def tel_of(session):
        if session not in tel_cache:
            p = os.path.join(root, f"telemetry_{session}.csv")
            if not os.path.exists(p):
                return None
            t = read(p, usecols=TEL_COLS)
            t["run_id"] = t["run_id"].fillna("")
            tel_cache[session] = t.sort_values("t_mono_ns", kind="stable").reset_index(drop=True)
        return tel_cache[session]

    rows, pces, quality, per_verb, per_loc, conf, per_pass, traces, correct = [], [], [], [], [], [], [], {}, {}
    for _, run in runs.iterrows():
        rid = run.run_id
        paths = {k: os.path.join(root, k, f"{rid}.{ext}") for k, ext in
                 [("calls", "csv.gz"), ("latency", "npy"), ("passes", "csv")]}
        miss = [k for k, p in paths.items() if not os.path.exists(p)]
        tel = tel_of(run.session)
        if miss or tel is None:
            print(f"WARNING: {rid} {run.system}: missing {miss or ['telemetry']}, run skipped (verify_campaign.py names it)")
            continue
        c = read(paths["calls"])
        lat_all = np.load(paths["latency"])
        passes = pd.read_csv(paths["passes"])
        ir = idle_tab[idle_tab.idle_id == run.idle_id]
        iw = pd.DataFrame(columns=TEL_COLS)
        if len(ir) == 1:
            it = tel_of(ir.session.iloc[0])
            if it is not None:
                iw = it[(it.run_id == run.idle_id) & (it.phase == "idle")]
        o, pce, q, pv, pl, cf, pp, cor, w = analyse_run(run, c, lat_all, passes, tel, iw)
        o["system_class"] = classes.get(run.system, "ceiling" if run.family == "ceiling" else "system")
        if len(ir) == 1 and "t_end_utc_ns" in ir and pd.notna(run.get("t_load_start_utc_ns", np.nan)):
            q["idle_age_min"] = (float(run.t_load_start_utc_ns) - float(ir.t_end_utc_ns.iloc[0])) / 6e10
        rows.append(o); pces.append(pce); quality.append(q); per_verb += pv; per_loc += pl; conf += cf; per_pass += pp
        correct[rid] = cor
        traces[rid] = w
    summary = pd.DataFrame(rows)
    summary["ranked"] = summary.system_class == "system"     # ceiling and vocabulary-unfair systems are listed, never ranked
    # cold start and standby are measured once per system (with its v3 run): copy to its other split
    for col in ["model_load_s", "model_load_net_wh", "loaded_idle_s", "loaded_idle_gpu_w", "loaded_idle_cpu_w",
                "loaded_idle_net_w", "loaded_idle_vram_mb"]:
        src = summary[summary.split == "held_out_v3"].dropna(subset=[col]).drop_duplicates("system").set_index("system")[col]
        summary[col] = summary[col].fillna(summary.system.map(src))
    # deployment scenario: one command every DUTY_S seconds, model kept resident in between
    standby_s = (DUTY_S - summary.dec_lat_p50_ms / 1000).clip(lower=0)
    summary["wh_per_cmd_duty"] = summary.net_wh_per_cmd + summary.loaded_idle_net_w.fillna(0) * standby_s * WH
    summary["kwh_per_million_duty"] = summary.wh_per_cmd_duty * 1e6 / 1000
    summary["co2_g_per_million_duty"] = summary.kwh_per_million_duty * CI_G_PER_KWH
    summary["standby_share_duty"] = summary.loaded_idle_net_w * standby_s * WH / summary.wh_per_cmd_duty

    summary.to_csv(os.path.join(root, "summary.csv"), index=False)
    pd.concat(pces, ignore_index=True).to_csv(os.path.join(root, "per_command_energy.csv"), index=False)
    qual = pd.DataFrame(quality)
    qual.to_csv(os.path.join(root, "quality.csv"), index=False)
    pd.DataFrame(per_verb).to_csv(os.path.join(root, "per_verb.csv"), index=False)
    pd.DataFrame(per_loc).to_csv(os.path.join(root, "per_loc_type.csv"), index=False)
    pd.DataFrame(conf).to_csv(os.path.join(root, "confusion.csv"), index=False)
    pd.DataFrame(per_pass).to_csv(os.path.join(root, "per_pass.csv"), index=False)
    comp = comparisons(summary, correct)
    comp.to_csv(os.path.join(root, "compare.csv"), index=False)
    figures(summary, traces, os.path.join(root, "figures"))

    rep = ["# thesis3 unified bench report\n",
           f"Campaign: `{os.path.basename(os.path.abspath(root))}`. Constants: {CI_G_PER_KWH:g} g CO2/kWh (UBA 2025), "
           f"PUE {PUE_DC} (Uptime 2024), {PRICE_EUR} EUR/kWh (BDEW 2026), arm move {ACT_WH_PANDA} Wh (Petrichenko et al. 2024), "
           f"duty scenario 1 command per {DUTY_S:g} s. Energy = GPU board + CPU package only (no RAM, disk, fans, PSU loss); "
           f"GPU energy source: {GPU_ENERGY_SOURCE} (the NVML counter reads 13-17 % lower on this card; see gpu_counter_to_power_ratio). "
           "A command = one decided command (gpt-4o-mini: 5 calls). Systems whose class is not `system` (ceiling, "
           "vocabulary-unfair) are listed but not ranked. Tiers are thesis-defined from the sources in the module docstring.\n"]
    bad = summary[(summary.errors_pass1 > 0) | (summary.n_commands < EXPECTED_N) | (summary.blank_scores_pass1 > 0)]
    if len(bad):
        rep.append("\n**Runs with pass-1 errors, blank scores or missing commands:** " +
                   ", ".join(f"{r.run_id} {r.system} ({r.errors_pass1} errors, {r.blank_scores_pass1} blank, n={r.n_commands})"
                             for r in bad.itertuples()) + "\n")
    inc = summary[~summary.energy_complete.astype(bool)]
    if len(inc):
        rep.append("\n**Energy incomplete (a power sensor covered < 95 % of the window), totals left empty:** " + ", ".join(inc.system) + "\n")
    for split, d in summary.groupby("split"):
        d = d.assign(system=np.where(d.ranked, d.system, d.system + " (" + d.system_class + ")")).sort_values("action_acc", ascending=False)
        rep.append(f"\n## {split}: accuracy\n")
        rep.append(md_table(d, ["system", "action_acc", "action_wilson_lo", "action_wilson_hi", "acc_tier", "verb_acc", "target_acc", "loc_acc",
                                "row_acc", "col_acc", "action_acc_grid", "action_acc_runtime", "action_acc_container", "action_acc_none",
                                "valid_json_rate", "pass_flip_rate", "errors_pass1"],
                            ["System", "Action", "Wilson lo", "Wilson hi", "Tier", "Verb", "Object", "Place", "Row", "Column",
                             "Grid", "Runtime", "Container", "No-loc", "Valid JSON", "Pass flips", "Errors"]))
        rep.append(f"\n\n## {split}: safety channel (pass 1; upper bound = exact one-sided 95 %)\n")
        rep.append(md_table(d, ["system", "reject_recall", "unsafe_accepts", "n_reject", "unsafe_accept_rate_ub95", "false_rejects",
                                "stop_recall", "missed_stops", "n_stop", "missed_stop_rate_ub95", "false_stops", "stop_lat_p95_ms"],
                            ["System", "Reject recall", "Unsafe accepts", "of", "Rate <=", "False rejects",
                             "Stop recall", "Missed stops", "of", "Rate <=", "False stops", "Stop p95 ms"]))
        d = d.sort_values("dec_lat_p50_ms")
        rep.append(f"\n\n## {split}: latency (ms) and robotics thresholds\n")
        rep.append(md_table(d, ["system", "lat_p50_ms", "lat_p50_ci_lo", "lat_p50_ci_hi", "lat_p95_ms", "lat_p99_ms", "lat_max_ms",
                                "dec_lat_p95_ms", "lat_tier", "share_le_100ms", "share_le_700ms", "share_le_1000ms", "share_le_2000ms",
                                "first_call_ms", "ttft_p50_ms", "tpot_p50_ms", "lat_drift_ms_per_min"],
                            ["System", "p50", "CI lo", "CI hi", "p95", "p99", "max", "p95 / command", "Tier", "<=100", "<=700",
                             "<=1 s", "<=2 s", "First call", "TTFT p50", "TPOT p50", "Drift ms/min"]))
        dl = d.sort_values("net_wh_per_1k")
        rep.append(f"\n\n## {split}: power and energy (local part)\n")
        rep.append(md_table(dl, ["system", "gpu_mean_w", "gpu_peak_w", "gpu_net_mean_w", "cpu_net_mean_w", "net_wh_per_1k",
                                 "net_wh_per_1k_idle_unc", "net_wh_per_1k_block_cv_pct", "net_wh_per_1k_correct", "pct_of_panda_move",
                                 "energy_tier", "wh_per_1k_tokens_out", "per_cmd_net_wh_p95"],
                             ["System", "GPU mean W", "GPU peak W", "GPU net W", "CPU net W", "Wh / 1k cmds", "+- idle",
                              "30 s block CV %", "Wh / 1k correct", "% of arm move", "Tier", "Wh / 1k out tokens", "Per-cmd p95 Wh"]))
        rep.append(f"\n\n## {split}: per million commands (scale-up)\n")
        rep.append(md_table(dl, ["system", "net_kwh_per_million", "gross_kwh_per_million", "co2_g_per_million", "co2_g_per_million_dc",
                                 "eur_per_million", "eur_per_million_dc", "device_hours_per_million", "kwh_per_million_duty", "standby_share_duty"],
                             ["System", "kWh net", "kWh gross", "g CO2e", "g CO2e (DC, PUE)", "EUR", "EUR (DC)", "device-h",
                              f"kWh, 1 cmd/{DUTY_S:g} s", "Standby share"]))
        rep.append(f"\n\n## {split}: memory, cold start, standby, thermal\n")
        rep.append(md_table(dl.sort_values("vram_above_idle_mb"), ["system", "vram_above_idle_mb", "rss_peak_mb", "model_load_s", "model_load_net_wh",
                                                                    "loaded_idle_net_w", "gpu_temp_start_c", "gpu_temp_max_c", "throttle_flag"],
                            ["System", "VRAM above idle MB", "RSS peak MB", "Load s", "Load net Wh", "Standby net W", "GPU start C",
                             "GPU max C", "Throttled"]))
    if len(comp):
        rep.append("\n\n## Paired comparisons (exploratory, Holm per split and reference)\n")
        rep.append(md_table(comp.sort_values(["split", "reference", "diff_pts"]),
                            ["split", "reference", "system", "diff_pts", "ci_lo_pts", "ci_hi_pts", "p_mcnemar", "p_holm", "separable"],
                            ["Split", "Reference", "System", "Diff pts", "CI lo", "CI hi", "McNemar p", "Holm p", "Separable"]))
    if len(qual):
        rep.append("\n\n## Measurement quality (every run)\n")
        qv = qual.merge(summary[["run_id", "system", "split"]], on="run_id")
        rep.append(md_table(qv, ["run_id", "system", "split", "sample_hz", "max_gap_s", "late_sample_share", "gpu_counter_vs_integral_pct",
                                 "cpu_sensor_refresh_s", "idle_window_s", "idle_age_min"],
                            ["Run", "System", "Split", "Hz", "Max gap s", "Late share", "GPU counter vs integral %",
                             "CPU refresh s", "Idle s", "Idle age min"]))
        low = qv[qv.sample_hz < 9]
        if len(low):
            rep.append("\n**Below 9 Hz (re-run these):** " + ", ".join(low.run_id) + "\n")
    open(os.path.join(root, "report.md"), "w", encoding="utf-8").write("\n".join(rep) + "\n")
    print(f"analysed {len(summary)} runs, {summary.shape[1]} columns -> {root}: summary.csv, per_verb.csv, per_loc_type.csv, "
          "confusion.csv, per_pass.csv, per_command_energy.csv, compare.csv, quality.csv, report.md, figures/")


if __name__ == "__main__":
    main(sys.argv[1])
