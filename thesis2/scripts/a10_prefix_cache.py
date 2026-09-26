"""thesis2/scripts/a10_prefix_cache.py -- A10: prefix KV caching on top of A8.

Every one of the 217 prompts is the same V5 legend with a different command
glued near the end. A8 re-ran the whole 460-474 token prefill for each command.
Here the longest common token prefix is processed ONCE, its KV cache is kept
resident, and each command pays only for its own tokens plus the draft.

Mechanically: the cache is a DynamicCache held at `prefix_len`; each command
appends (suffix + draft), decodes, then the cache is cropped back to
`prefix_len`. Nothing is copied.

WHY THIS IS NOT FREE, AND WHAT THE CHECK IS FOR
-----------------------------------------------
Splitting one forward pass over P tokens into (prefix) then (suffix) is
mathematically identical but NOT numerically identical -- different batch
shapes take different kernel paths, so logits can move in the last bits and a
near-tied argmax can flip. That is exactly what the 217/217 byte-identity
check detects. It is re-verified here against plain greedy computed in the same
process, before any speed number is reported.

Arms (all on held_out_v3, n=217; 3 commands are pre-run as GPU warm-up and
then measured with the rest -- nothing is excluded, Part B S-5):
  plain greedy                    -- the A8 baseline, no cache reuse
  prefix-cached plain greedy      -- isolates the caching effect alone
  speculative (A8)                -- classifier draft, full prefill each time
  prefix-cached speculative       -- both, the arm this task asks for

Run (conda env thesis2gpu):
    python thesis2/scripts/a10_prefix_cache.py            # correctness + latency
    python thesis2/scripts/a10_prefix_cache.py --energy   # adds the power runs
"""
import argparse
import csv
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
csv.field_size_limit(2 ** 31 - 1)

BASE = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER = REPO / "ft" / "runs" / "qwen3b_lora" / "checkpoint-87"
SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
MAX_NEW = 64
IDLE_W = 12.12
LOAD_S = 120
N_WARMUP = 3

ap = argparse.ArgumentParser()
ap.add_argument("--energy", action="store_true")
# --- added 2026-09-25 -------------------------------------------------------
# --split lets this arm run on either held-out family (and on train_v2 for a
# smoke test). --limit truncates the command list for a smoke run. --outdir
# defaults to thesis2/out/final/ so a re-run NEVER overwrites the committed
# files in thesis2/out/ from the earlier run.
ap.add_argument("--split", default="held_out_v3",
                choices=["held_out_v3", "held_out_v4c", "train_v2"])
ap.add_argument("--limit", type=int, default=0,
                help="use only the first N commands (smoke test)")
ap.add_argument("--outdir", default=str(REPO / "thesis2" / "out" / "final"))
# --- added 2026-09-26 -------------------------------------------------------
# --emit-preds-only converts this arm's EXISTING per-command CSV into the
# standard pred_<arm>_<split>.csv and exits. No model is loaded, nothing is
# re-run: it is how the held_out_v3 file is produced from the measurement
# that already exists.
ap.add_argument("--emit-preds-only", dest="emit_preds_only", action="store_true",
                help="write pred_<arm>_<split>.csv from the existing "
                     "per-command CSV and exit; no model is loaded")
args = ap.parse_args()
OUT = Path(args.outdir)
OUT.mkdir(parents=True, exist_ok=True)
DATA = REPO / "data" / ("%s.jsonl" % args.split)
SPLIT = args.split
TAG = "" if SPLIT == "held_out_v3" else ("_" + SPLIT)
print("  split=%s  data=%s  outdir=%s" % (SPLIT, DATA.name, OUT))

# --- added 2026-09-26: the standard pred_<arm>_<split>.csv -------------------
# s1_repro_check.py and Stage 2 read pred_<arm>_<split>.csv. This script used to
# write only its own per-command CSV, so the arm showed as NOT RUN YET. It now
# writes both, through the ONE emitter every arm shares (common.py), with this
# arm's own verified output and its own per-command latency.
sys.path.insert(0, str(REPO / "thesis2" / "final"))
import common as C                                                # noqa: E402
PRED_ARM, PRED_TEXT, PRED_MS = "a10_speculative", "prefix_spec_text", "prefix_spec_ms"
PERCOMMAND = OUT / ("a10_prefix_cache_percommand%s.csv" % TAG)
if args.emit_preds_only:
    if not PERCOMMAND.exists():
        C.gate(False, ["%s does not exist -- nothing to convert; run this script "
                       "without --emit-preds-only first" % PERCOMMAND])
    _recs = list(csv.DictReader(open(PERCOMMAND, encoding="utf-8", newline="")))
    _p, _n = C.preds_from_percommand(PRED_ARM, SPLIT, _recs, PRED_TEXT, PRED_MS,
                                     note="from %s; no re-run" % PERCOMMAND.name,
                                     outdir=OUT)
    _n_cmd = len(C.load_split(SPLIT))
    C.gate(_n == len(_recs) == (args.limit or _n_cmd), [
        "--emit-preds-only: NO model loaded, nothing re-run",
        "read %s (%d rows)" % (PERCOMMAND.name, len(_recs)),
        "wrote %s (%d rows; text=%s, latency_ms=%s)" % (_p.name, _n, PRED_TEXT, PRED_MS),
    ])


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))] if xs else float("nan")


print("=" * 96)
print("A10 -- PREFIX KV CACHING FOR A8")
print("=" * 96)

from bench.run_cell import apply_legend                             # noqa: E402
from llm import prompts as P                                        # noqa: E402

apply_legend("v5")
schema = P.load_schema(str(REPO / "schema" / "schema_v2.json"))


def chat_wrap(user):
    return ("<|im_start|>system\n%s<|im_end|>\n"
            "<|im_start|>user\n%s<|im_end|>\n"
            "<|im_start|>assistant\n" % (SYSTEM, user))


rows = [json.loads(l) for l in DATA.read_text(encoding="utf-8").splitlines() if l.strip()]
if args.limit:
    rows = rows[:args.limit]
    print("  SMOKE: first %d commands of %s" % (len(rows), SPLIT))

print("  loading %s + %s (fp16, merged)" % (BASE, ADAPTER.name))
from transformers import AutoTokenizer, AutoModelForCausalLM          # noqa: E402
from peft import PeftModel                                            # noqa: E402

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16).to("cuda")
model = PeftModel.from_pretrained(model, str(ADAPTER)).merge_and_unload()
model.eval()
IM_END = tok.convert_tokens_to_ids("<|im_end|>")
STOP = {IM_END, tok.eos_token_id}

from llm.classifier_client import DistilBertClassifierClient          # noqa: E402

clf = DistilBertClassifierClient()
_ = clf.complete("warm up the classifier")


def render(o):
    d = {"action": o.get("action")}
    if o.get("target"):
        d["target"] = o["target"]
    if o.get("location"):
        d["location"] = o["location"]
    d["modifiers"] = o.get("modifiers") or []
    return json.dumps(d)


items = []
for i, r in enumerate(rows):
    cmd = r["command"]
    ptxt = chat_wrap(P.build_prompt(cmd, "zero_shot", schema))
    t0 = time.perf_counter()
    raw = clf.complete(cmd)
    dms = (time.perf_counter() - t0) * 1000
    try:
        dtxt = render(json.loads(raw))
    except Exception:
        dtxt = ""
    pids = tok(ptxt, add_special_tokens=False)["input_ids"]
    if dtxt:
        fids = tok(ptxt + dtxt, add_special_tokens=False)["input_ids"]
        dids = (fids[len(pids):] if fids[:len(pids)] == pids
                else tok(dtxt, add_special_tokens=False)["input_ids"]) + [IM_END]
    else:
        dids = []
    items.append(dict(command_id=str(i), command=cmd, prompt_ids=pids,
                      draft_ids=dids, draft_ms=dms))

# ---------------------------------------------------------------- the prefix
PL = min(len(x["prompt_ids"]) for x in items)
k = 0
while k < PL and len({tuple(x["prompt_ids"][:k + 1]) for x in items}) == 1:
    k += 1
PREFIX = items[0]["prompt_ids"][:k]
print()
print("  longest common token prefix: %d tokens of %d-%d  (%.1f%% of the median prompt)"
      % (k, min(len(x["prompt_ids"]) for x in items),
         max(len(x["prompt_ids"]) for x in items),
         100.0 * k / statistics.median([len(x["prompt_ids"]) for x in items])))
print("  per-command suffix: min %d  median %d  max %d tokens"
      % (min(len(x["prompt_ids"]) - k for x in items),
         int(statistics.median([len(x["prompt_ids"]) - k for x in items])),
         max(len(x["prompt_ids"]) - k for x in items)))
print("  prefix ends: ...%s" % repr(tok.decode(PREFIX[-16:]))[:70])
if k == 0:
    sys.exit("ABORT: no common prefix; nothing to cache.")

PREFIX_CACHE = None


@torch.no_grad()
def build_prefix_cache():
    global PREFIX_CACHE
    out = model(input_ids=torch.tensor([PREFIX], device="cuda"), use_cache=True)
    PREFIX_CACHE = out.past_key_values
    return out.logits[0, -1]


LAST_PREFIX_LOGIT = build_prefix_cache()


def reset_cache():
    n = PREFIX_CACHE.get_seq_length() - len(PREFIX)
    if n > 0:
        PREFIX_CACHE.crop(-n)


# ---------------------------------------------------------------- decoders
@torch.no_grad()
def greedy_plain(pids):
    out = model(input_ids=torch.tensor([pids], device="cuda"), use_cache=True)
    past = out.past_key_values
    nxt = int(out.logits[0, -1].argmax())
    gen, steps = [], 1
    while len(gen) < MAX_NEW and nxt not in STOP:
        gen.append(nxt)
        out = model(input_ids=torch.tensor([[nxt]], device="cuda"),
                    past_key_values=past, use_cache=True)
        past = out.past_key_values
        nxt = int(out.logits[0, -1].argmax())
        steps += 1
    return gen, steps


@torch.no_grad()
def greedy_prefix(pids):
    reset_cache()
    suf = pids[len(PREFIX):]
    out = model(input_ids=torch.tensor([suf], device="cuda"),
                past_key_values=PREFIX_CACHE, use_cache=True)
    nxt = int(out.logits[0, -1].argmax())
    gen, steps = [], 1
    while len(gen) < MAX_NEW and nxt not in STOP:
        gen.append(nxt)
        out = model(input_ids=torch.tensor([[nxt]], device="cuda"),
                    past_key_values=PREFIX_CACHE, use_cache=True)
        nxt = int(out.logits[0, -1].argmax())
        steps += 1
    return gen, steps


@torch.no_grad()
def spec_plain(pids, dids):
    p_, kk = len(pids), len(dids)
    out = model(input_ids=torch.tensor([pids + dids], device="cuda"), use_cache=True)
    past = out.past_key_values
    preds = out.logits[0, p_ - 1:p_ + kk].argmax(dim=-1).tolist()
    n = 0
    while n < kk and preds[n] == dids[n]:
        n += 1
    gen = []
    for t in dids[:n]:
        if t in STOP:
            return gen, 1, n, kk
        gen.append(t)
    nxt = preds[n]
    if kk > n:
        past.crop(-(kk - n))
    steps = 1
    while len(gen) < MAX_NEW and nxt not in STOP:
        gen.append(nxt)
        out = model(input_ids=torch.tensor([[nxt]], device="cuda"),
                    past_key_values=past, use_cache=True)
        past = out.past_key_values
        nxt = int(out.logits[0, -1].argmax())
        steps += 1
    return gen, steps, n, kk


@torch.no_grad()
def spec_prefix(pids, dids):
    """The arm this task asks for: prefix cached AND classifier-drafted."""
    reset_cache()
    suf = pids[len(PREFIX):]
    s_, kk = len(suf), len(dids)
    out = model(input_ids=torch.tensor([suf + dids], device="cuda"),
                past_key_values=PREFIX_CACHE, use_cache=True)
    preds = out.logits[0, s_ - 1:s_ + kk].argmax(dim=-1).tolist()
    n = 0
    while n < kk and preds[n] == dids[n]:
        n += 1
    gen = []
    for t in dids[:n]:
        if t in STOP:
            return gen, 1, n, kk
        gen.append(t)
    nxt = preds[n]
    if kk > n:
        PREFIX_CACHE.crop(-(kk - n))
    steps = 1
    while len(gen) < MAX_NEW and nxt not in STOP:
        gen.append(nxt)
        out = model(input_ids=torch.tensor([[nxt]], device="cuda"),
                    past_key_values=PREFIX_CACHE, use_cache=True)
        nxt = int(out.logits[0, -1].argmax())
        steps += 1
    return gen, steps, n, kk


# ---------------------------------------------------------------- warmup
print()
print("  warming up (%d commands pre-run; all %d are measured below)" % (N_WARMUP, len(items)))
for x in items[:N_WARMUP]:
    greedy_plain(x["prompt_ids"]); greedy_prefix(x["prompt_ids"])
    spec_plain(x["prompt_ids"], x["draft_ids"])
    spec_prefix(x["prompt_ids"], x["draft_ids"])
torch.cuda.synchronize()

# ---------------------------------------------------------------- measure
print("  decoding all %d commands four ways" % len(items))
recs = []
t_all = time.time()
for n_i, x in enumerate(items):
    pids, dids = x["prompt_ids"], x["draft_ids"]

    torch.cuda.synchronize(); t0 = time.perf_counter()
    g_ids, g_steps = greedy_plain(pids)
    torch.cuda.synchronize(); g_ms = (time.perf_counter() - t0) * 1000

    torch.cuda.synchronize(); t0 = time.perf_counter()
    gp_ids, gp_steps = greedy_prefix(pids)
    torch.cuda.synchronize(); gp_ms = (time.perf_counter() - t0) * 1000

    torch.cuda.synchronize(); t0 = time.perf_counter()
    s_ids, s_steps, s_n, s_k = spec_plain(pids, dids)
    torch.cuda.synchronize(); s_ms = (time.perf_counter() - t0) * 1000 + x["draft_ms"]

    torch.cuda.synchronize(); t0 = time.perf_counter()
    sp_ids, sp_steps, sp_n, sp_k = spec_prefix(pids, dids)
    torch.cuda.synchronize(); sp_ms = (time.perf_counter() - t0) * 1000 + x["draft_ms"]

    gt = tok.decode(g_ids, skip_special_tokens=True)
    recs.append(dict(command_id=x["command_id"], command=x["command"],
                     greedy_text=gt,
                     prefix_greedy_identical=(gp_ids == g_ids),
                     spec_identical=(s_ids == g_ids),
                     prefix_spec_identical=(sp_ids == g_ids),
                     prefix_spec_text=tok.decode(sp_ids, skip_special_tokens=True),
                     draft_tokens=sp_k, draft_accepted=sp_n,
                     full_draft_accepted=(sp_k > 0 and sp_n == sp_k),
                     spec_accepted_plain=s_n,
                     greedy_forward_passes=g_steps, spec_forward_passes=s_steps,
                     prefix_spec_forward_passes=sp_steps,
                     greedy_ms=round(g_ms, 3), prefix_greedy_ms=round(gp_ms, 3),
                     spec_ms=round(s_ms, 3), prefix_spec_ms=round(sp_ms, 3),
                     draft_ms=round(x["draft_ms"], 3)))
    if (n_i + 1) % 50 == 0:
        print("    %3d / %d  (%.0fs)" % (n_i + 1, len(items), time.time() - t_all))
print("    %3d / %d  DONE (%.0fs)" % (len(items), len(items), time.time() - t_all))

# ---------------------------------------------------------------- 1. identity
m = recs
print()
print("=" * 96)
print("1. BYTE-IDENTICAL OUTPUT vs PLAIN GREEDY  (must be %d/%d before any speed claim)" % (len(m), len(m)))
print("=" * 96)
checks = [("speculative, no prefix cache (A8)", "spec_identical"),
          ("prefix-cached plain greedy", "prefix_greedy_identical"),
          ("PREFIX-CACHED SPECULATIVE (A10)", "prefix_spec_identical")]
allpass = True
for lbl, key in checks:
    c = sum(r[key] for r in m)
    ok = c == len(m)
    allpass &= ok
    print("    %-38s %d / %d   %s" % (lbl, c, len(m), "PASS" if ok else "*** FAIL ***"))
    if not ok:
        for r in m:
            if not r[key]:
                print("        MISMATCH cmd %s" % r["command_id"])
                print("          greedy: %r" % r["greedy_text"])
                print("          a10   : %r" % r["prefix_spec_text"])
if not allpass:
    print()
    print("    Speed numbers below are reported but MUST NOT be used: the arm is wrong.")

# ---------------------------------------------------------------- 2. acceptance
tot_d = sum(r["draft_tokens"] for r in m)
tot_a = sum(r["draft_accepted"] for r in m)
same_accept = sum(1 for r in m if r["draft_accepted"] == r["spec_accepted_plain"])
print()
print("=" * 96)
print("2. ACCEPTANCE (unchanged by caching, as it must be)")
print("=" * 96)
print("    token acceptance rate        %.4f  (%d / %d)" % (tot_a / tot_d, tot_a, tot_d))
print("    full draft accepted          %d / %d = %.4f"
      % (sum(r["full_draft_accepted"] for r in m), len(m),
         sum(r["full_draft_accepted"] for r in m) / len(m)))
print("    same accepted count as A8    %d / %d" % (same_accept, len(m)))
print("    forward passes  plain %d   spec %d   prefix-spec %d"
      % (sum(r["greedy_forward_passes"] for r in m),
         sum(r["spec_forward_passes"] for r in m),
         sum(r["prefix_spec_forward_passes"] for r in m)))

# ---------------------------------------------------------------- 3. latency
print()
print("=" * 96)
print("3. LATENCY  (n=%d, all commands included; %d were pre-run as GPU warm-up)" % (len(m), N_WARMUP))
print("=" * 96)
lat = {"plain greedy": [r["greedy_ms"] for r in m],
       "prefix-cached plain greedy": [r["prefix_greedy_ms"] for r in m],
       "speculative (A8)": [r["spec_ms"] for r in m],
       "PREFIX-CACHED SPECULATIVE (A10)": [r["prefix_spec_ms"] for r in m]}
print("    %-36s %9s %9s %9s %9s" % ("arm", "p50", "p95", "p99", "mean"))
print("    " + "-" * 76)
for kk2, v in lat.items():
    print("    %-36s %8.2f %8.2f %8.2f %8.2f"
          % (kk2, pct(v, .50), pct(v, .95), pct(v, .99), statistics.mean(v)))
base = lat["speculative (A8)"]
new = lat["PREFIX-CACHED SPECULATIVE (A10)"]
print()
print("    A10 vs A8   p50 %.2fx   p95 %.2fx   p99 %.2fx   mean %.2fx"
      % (pct(base, .50) / pct(new, .50), pct(base, .95) / pct(new, .95),
         pct(base, .99) / pct(new, .99),
         statistics.mean(base) / statistics.mean(new)))
print("    A10 vs plain greedy  p50 %.2fx"
      % (pct(lat["plain greedy"], .50) / pct(new, .50)))
print("    classifier draft floor: p50 %.2f ms of the A10 p50 %.2f ms (%.0f%%)"
      % (pct([r["draft_ms"] for r in m], .50), pct(new, .50),
         100 * pct([r["draft_ms"] for r in m], .50) / pct(new, .50)))

# ---------------------------------------------------------------- 4. energy
energy_rows = []
if args.energy:
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
                    p_, u_ = [y.strip() for y in o.split(",")[:2]]
                    self.samples.append((time.time(), float(p_), float(u_)))
                except Exception:
                    pass
                time.sleep(1.0)

        def stop(self):
            self._halt.set(); self.join(timeout=5)

    def measure(name, fn, n_items):
        lg = PowerLogger(); lg.start(); time.sleep(3)
        t0 = time.time(); n = 0
        while time.time() - t0 < LOAD_S:
            fn(); n += n_items
        wall = time.time() - t0
        time.sleep(2); lg.stop()
        live = [p_ for t_, p_, u_ in lg.samples if t_ >= t0]
        mw = statistics.mean(live) if live else float("nan")
        d = mw - IDLE_W
        j = d * wall / n if n else float("nan")
        print("    %-34s N=%-6d wall %6.1fs  meanP %6.1f W  delta %6.1f W  %8.4f J/cmd"
              % (name, n, wall, mw, d, j))
        return dict(arm=name, n_commands=n, wall_s=round(wall, 1),
                    mean_power_w=round(mw, 2), idle_power_w=IDLE_W,
                    delta_power_w=round(d, 2), joules_per_command=round(j, 5),
                    power_samples=len(live))

    print()
    print("=" * 96)
    print("4. ENERGY (GPU only, idle %.2f W, same protocol as energy.csv)" % IDLE_W)
    print("=" * 96)
    energy_rows.append(measure("a10_speculative_A8_no_prefix_cache",
                               lambda: [spec_plain(x["prompt_ids"], x["draft_ids"])
                                        for x in items], len(items)))
    energy_rows.append(measure("a10_prefix_cached_speculative",
                               lambda: [spec_prefix(x["prompt_ids"], x["draft_ids"])
                                        for x in items], len(items)))
    e = {r["arm"]: r["joules_per_command"] for r in energy_rows}
    print()
    print("    energy reduction vs A8 speculative: %.4f  (%.2fx)"
          % (1 - e["a10_prefix_cached_speculative"] / e["a10_speculative_A8_no_prefix_cache"],
             e["a10_speculative_A8_no_prefix_cache"] / e["a10_prefix_cached_speculative"]))
else:
    print()
    print("  energy NOT measured this run (pass --energy)")

# ---------------------------------------------------------------- write
p = OUT / ("a10_prefix_cache_percommand%s.csv" % TAG)
with open(p, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(recs[0].keys()))
    w.writeheader(); w.writerows(recs)
_pp, _pn = C.preds_from_percommand(PRED_ARM, SPLIT, recs, PRED_TEXT, PRED_MS,
                                   outdir=OUT)
print("wrote %s (%d rows, standard pred format, latency from %s)"
      % (_pp, _pn, PRED_MS))
(OUT / ("a10_prefix_cache_summary%s.json" % TAG)).write_text(json.dumps(dict(
    prefix_tokens=k, n_commands=len(m), warmup_prerun_not_excluded=N_WARMUP,
    byte_identical={lbl: sum(r[key] for r in m) for lbl, key in checks},
    all_identity_checks_pass=bool(allpass),
    token_acceptance_rate=round(tot_a / tot_d, 4),
    forward_passes=dict(plain=sum(r["greedy_forward_passes"] for r in m),
                        spec=sum(r["spec_forward_passes"] for r in m),
                        prefix_spec=sum(r["prefix_spec_forward_passes"] for r in m)),
    latency_ms={kk2: dict(p50=round(pct(v, .50), 2), p95=round(pct(v, .95), 2),
                          p99=round(pct(v, .99), 2), mean=round(statistics.mean(v), 2))
                for kk2, v in lat.items()},
    energy=energy_rows), indent=2), encoding="utf-8")
if energy_rows:
    pe = OUT / ("a10_energy%s.csv" % TAG)
    with open(pe, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(energy_rows[0].keys()))
        w.writeheader(); w.writerows(energy_rows)
    print("wrote %s" % pe.relative_to(REPO))
print()
print("wrote %s" % p.relative_to(REPO))
print("wrote %s" % (OUT / ("a10_prefix_cache_summary%s.json" % TAG)))
print("=" * 96)

# ---------------------------------------------------------------- byte-identity
# GATE: on any split, this arm must reproduce lora_greedy's saved output exactly.
# Speculative decoding is output-exact by construction, so a mismatch means the
# implementation is wrong and no speed number may be quoted.
_lg = OUT / ("pred_lora_greedy_%s.csv" % SPLIT)
if _lg.exists():
    _base = {}
    for _r in csv.DictReader(open(_lg, encoding="utf-8", newline="")):
        if str(_r.get("repeat", "0")) == "0":
            try:
                _base[_r["command_id"]] = json.dumps(json.loads(_r["pred_json"]), sort_keys=True)
            except Exception:
                _base[_r["command_id"]] = ""
    _diff = []
    for _r in recs:
        _cid = str(_r["command_id"])
        if _cid not in _base:
            continue
        try:
            _mine = json.dumps(json.loads(_r["prefix_spec_text"]), sort_keys=True)
        except Exception:
            _mine = ""
        if _mine != _base[_cid]:
            _diff.append(_cid)
    print()
    print("=" * 88)
    print("BYTE-IDENTITY vs lora_greedy on %s" % SPLIT)
    print("=" * 88)
    if _diff:
        print("    *** GATE FAIL: differs on %d command(s): %s ***"
              % (len(_diff), _diff[:20]))
        sys.exit(1)
    print("    %d / %d identical -- PASS" % (len(_base), len(_base)))
else:
    print()
    print("  byte-identity vs lora_greedy NOT CHECKED: %s absent. Run "
          "s1_run_arm.py --arm lora_greedy --split %s first; on held_out_v4c this "
          "check is a required gate." % (_lg.name, SPLIT))
