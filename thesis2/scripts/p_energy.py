"""thesis2/scripts/p_energy.py -- GPU energy per command, per the stated protocol.

  1. idle baseline: nvidia-smi power.draw at 1 Hz for IDLE_S seconds, no workload
  2. per arm: start the logger, run the arm over the full held-out set in a tight
     loop until at least LOAD_S seconds of wall clock have elapsed
  3. energy per command = (mean load power - median idle power) * wall / N

LIMITATIONS, stated in the output as required:
  * GPU power ONLY. CPU and system power are excluded.
  * Cloud model energy is NOT measurable from this machine, at all.
  * nvidia-smi samples at ~1 Hz, far coarser than a sub-20 ms inference. That is
    why each arm runs for >= 120 s -- a single inference is invisible to the
    sampler and only the aggregate is meaningful.
  * CPU-only arms report a GPU delta of ~0, which does NOT mean they are free.
"""
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "thesis2" / "scripts"))

OUT = REPO / "thesis2" / "out"
IDLE_S = 300
LOAD_S = 120


class PowerLogger(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.samples = []
        self._halt = threading.Event()

    def run(self):
        while not self._halt.is_set():
            try:
                o = subprocess.run(
                    ["nvidia-smi", "--query-gpu=power.draw,utilization.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5).stdout.strip()
                p, u = [x.strip() for x in o.split(",")[:2]]
                self.samples.append((time.time(), float(p), float(u)))
            except Exception:
                pass
            time.sleep(1.0)

    def stop(self):
        self._halt.set()
        self.join(timeout=5)


def measure(name, step_fn, n_items, idle_w):
    """Run step_fn() over n_items repeatedly for >= LOAD_S seconds."""
    lg = PowerLogger()
    lg.start()
    time.sleep(3)
    t0 = time.time()
    n = 0
    while time.time() - t0 < LOAD_S:
        step_fn()
        n += n_items
    wall = time.time() - t0
    time.sleep(2)
    lg.stop()
    live = [p for t, p, u in lg.samples if t >= t0]
    mean_w = statistics.mean(live) if live else float("nan")
    delta = mean_w - idle_w
    j_per_cmd = delta * wall / n if n else float("nan")
    print("  %-26s N=%-7d wall %6.1fs  meanP %6.1f W  delta %6.1f W  "
          "%8.4f J/cmd  (%d samples)"
          % (name, n, wall, mean_w, delta, j_per_cmd, len(live)))
    return dict(arm=name, n_commands=n, wall_s=round(wall, 1),
                mean_power_w=round(mean_w, 2), idle_power_w=round(idle_w, 2),
                delta_power_w=round(delta, 2),
                joules_per_command=round(j_per_cmd, 5),
                power_samples=len(live))


def main():
    print("=" * 84)
    print("ENERGY PROTOCOL -- GPU power only")
    print("=" * 84)
    print("  EXCLUDES CPU and system power. Cloud model energy is NOT measurable")
    print("  from this machine. nvidia-smi samples at ~1 Hz, far coarser than a")
    print("  sub-20 ms inference, which is why every arm runs >= %d s." % LOAD_S)
    print()
    print("  idle baseline: %d s with no workload ..." % IDLE_S)
    lg = PowerLogger()
    lg.start()
    time.sleep(IDLE_S)
    lg.stop()
    idle = [p for _t, p, _u in lg.samples]
    idle_w = statistics.median(idle)
    with open(OUT / "power_idle.csv", "w", encoding="utf-8", newline="") as fh:
        fh.write("timestamp,power_w,util_pct\n")
        for t, p, u in lg.samples:
            fh.write("%.3f,%.2f,%.1f\n" % (t, p, u))
    print("  idle median %.1f W  (n=%d samples, min %.1f, max %.1f)"
          % (idle_w, len(idle), min(idle), max(idle)))
    print("  wrote thesis2/out/power_idle.csv")
    print()

    import json as _json
    from transformers import AutoTokenizer
    from p2_train_factorized import (FactorizedClassifier, build_label_space,
                                     MAX_LEN)
    from ft.train_classifier import MultiHeadClassifier, build_label_space as fsp
    from salsa_row import SalsaRowScorer

    rows = [_json.loads(l) for l in open(REPO / "data" / "held_out_v3.jsonl",
                                         encoding="utf-8") if l.strip()]
    cmds = [r["command"] for r in rows]
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    space = build_label_space()
    results = []

    # --- factorized DistilBERT on GPU ---
    m = FactorizedClassifier(space)
    m.load_state_dict(torch.load(
        REPO / "thesis2" / "models" / "distilbert_factorized" / "epoch_11" / "model.pt",
        map_location="cpu"))
    m.eval().cuda().half()
    enc = [tok(c, truncation=True, max_length=MAX_LEN, padding="max_length",
               return_tensors="pt").to("cuda") for c in cmds]

    def step_fact():
        with torch.no_grad():
            for e in enc:
                m(e["input_ids"], e["attention_mask"])
    results.append(measure("factorized_distilbert_gpu", step_fact, len(cmds), idle_w))
    del m
    torch.cuda.empty_cache()

    # --- flat DistilBERT on GPU ---
    f = MultiHeadClassifier(fsp())
    f.load_state_dict(torch.load(
        REPO / "ft" / "runs" / "distilbert_cls" / "epoch_09" / "model.pt",
        map_location="cpu"))
    f.eval().cuda().half()

    def step_flat():
        with torch.no_grad():
            for e in enc:
                f(e["input_ids"], e["attention_mask"])
    results.append(measure("flat_distilbert_gpu", step_flat, len(cmds), idle_w))
    del f
    torch.cuda.empty_cache()

    # --- SALSA row, qwen2.5-3b base and +lora ---
    for tag, adapter in (("salsa_qwen3b", None),
                         ("salsa_qwen3blora",
                          str(REPO / "ft" / "runs" / "qwen3b_lora" / "checkpoint-87"))):
        s = SalsaRowScorer(adapter=adapter)
        sub = cmds[:60]          # 60 x ~36 ms = ~2.2 s per pass, many passes in 120 s

        def step_salsa():
            for c in sub:
                s.score(c, "letters")
        results.append(measure(tag, step_salsa, len(sub), idle_w))
        del s
        torch.cuda.empty_cache()

    import csv
    p = OUT / "energy.csv"
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(results[0]) + ["note"])
        w.writeheader()
        for r in results:
            r["note"] = "GPU power only; excludes CPU/system"
            w.writerow(r)
        w.writerow(dict(arm="gpt-4o-mini (cloud)", n_commands="", wall_s="",
                        mean_power_w="", idle_power_w="", delta_power_w="",
                        joules_per_command="",
                        power_samples="", note="NOT MEASURABLE from this machine"))
        w.writerow(dict(arm="rules / tfidf / knn (CPU)", n_commands="", wall_s="",
                        mean_power_w="", idle_power_w="", delta_power_w="",
                        joules_per_command="~0",
                        power_samples="", note="GPU delta ~0; this does NOT capture "
                        "their true CPU cost"))
    print()
    print("wrote %s" % p.relative_to(REPO))
    print("=" * 84)
    return 0


if __name__ == "__main__":
    sys.exit(main())
