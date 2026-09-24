"""thesis2/scripts/a12_smalllm_drafter.py -- A12: a CONVENTIONAL small-LM drafter
for the same verifier, as the comparison arm for A8/A10.

WHAT THIS ARM IS. A8/A10 draft with a discriminative classifier that emits the
whole output in one shot. The standard alternative -- Leviathan et al. 2023,
Chen et al. 2023 -- is an autoregressive DRAFT MODEL from the same family. Here
that is Qwen2.5-0.5B-Instruct drafting for the same Qwen2.5-3B-Instruct +
qwen3b_lora verifier, with:

  * the SAME prompt (V5 legend, same chat wrapper) for drafter and verifier,
  * the SAME prefix KV cache treatment as A10: each model processes the 449-token
    common prefix ONCE and crops back to it before every command,
  * the SAME acceptance rule, stop handling and byte-identity requirement.

PARAMETERS FIXED BEFORE THE RUN, NOT SWEPT:
  gamma = 4 draft tokens per iteration. This is the conventional default in the
  speculative-decoding literature. No gamma sweep is run: sweeping it on
  held_out_v3 and quoting the best value would be threshold selection on the
  test set, which this project has ruled out everywhere else.

The 0.5B is the OFF-THE-SHELF instruct model. It is not fine-tuned on this task
-- that is the point of the comparison: A8's drafter is free because the
pipeline already ships it, while a conventional drafter must either be trained
or be accepted as-is.

ONE CACHE PER MODEL, CROPPED NOT MUTATED. Both prefix caches are cropped back
to prefix length at the start of every command, so no state crosses commands.

Run (conda env thesis2gpu):
    python thesis2/scripts/a12_smalllm_drafter.py            # correctness + latency
    python thesis2/scripts/a12_smalllm_drafter.py --energy   # adds the power runs
"""
import argparse
import collections
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
DRAFTER = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER = REPO / "ft" / "runs" / "qwen3b_lora" / "checkpoint-87"
SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
MAX_NEW = 64
GAMMA = 4                # fixed in advance, not swept -- see docstring
IDLE_W = 12.12           # median of power_idle.csv, same baseline as energy.csv
LOAD_S = 120
N_WARMUP = 3             # pre-run to warm the GPU; all 217 are measured

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
args = ap.parse_args()
OUT = Path(args.outdir)
OUT.mkdir(parents=True, exist_ok=True)
DATA = REPO / "data" / ("%s.jsonl" % args.split)
SPLIT = args.split
TAG = "" if SPLIT == "held_out_v3" else ("_" + SPLIT)
print("  split=%s  data=%s  outdir=%s" % (SPLIT, DATA.name, OUT))


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))] if xs else float("nan")


print("=" * 96)
print("A12 -- CONVENTIONAL SMALL-LM DRAFTER (qwen2.5-0.5b) FOR THE SAME VERIFIER")
print("=" * 96)

from bench.run_cell import apply_legend                              # noqa: E402
from llm import prompts as P                                         # noqa: E402

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

from transformers import AutoTokenizer, AutoModelForCausalLM           # noqa: E402
from peft import PeftModel                                             # noqa: E402

tok = AutoTokenizer.from_pretrained(BASE)
dtok = AutoTokenizer.from_pretrained(DRAFTER)
IM_END = tok.convert_tokens_to_ids("<|im_end|>")
STOP = {IM_END, tok.eos_token_id}

# the two models must share a vocabulary or the draft ids are meaningless
probe = chat_wrap("put the bottle in the upper left corner of the table") + '{"action": "pick_place"}'
same_vocab = (tok(probe, add_special_tokens=False)["input_ids"]
              == dtok(probe, add_special_tokens=False)["input_ids"])
print("  tokenizer shared by verifier and drafter: %s (vocab %d / %d)"
      % (same_vocab, len(tok), len(dtok)))
if not same_vocab:
    sys.exit("ABORT: drafter and verifier do not share a tokenizer.")

print("  loading verifier %s + %s (fp16, merged)" % (BASE, ADAPTER.name))
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16).to("cuda")
model = PeftModel.from_pretrained(model, str(ADAPTER)).merge_and_unload()
model.eval()
print("  loading drafter  %s (fp16, off-the-shelf, NOT fine-tuned)" % DRAFTER)
drafter = AutoModelForCausalLM.from_pretrained(DRAFTER, dtype=torch.float16).to("cuda")
drafter.eval()
print("  gamma = %d draft tokens per iteration (fixed in advance, not swept)" % GAMMA)

items = []
for i, r in enumerate(rows):
    ptxt = chat_wrap(P.build_prompt(r["command"], "zero_shot", schema))
    items.append(dict(command_id=str(i), command=r["command"],
                      prompt_ids=tok(ptxt, add_special_tokens=False)["input_ids"]))

PL = min(len(x["prompt_ids"]) for x in items)
k = 0
while k < PL and len({tuple(x["prompt_ids"][:k + 1]) for x in items}) == 1:
    k += 1
PREFIX = items[0]["prompt_ids"][:k]
print("  common token prefix: %d tokens of %d-%d"
      % (k, min(len(x["prompt_ids"]) for x in items),
         max(len(x["prompt_ids"]) for x in items)))


@torch.no_grad()
def build_cache(m):
    return m(input_ids=torch.tensor([PREFIX], device="cuda"), use_cache=True).past_key_values


V_CACHE = build_cache(model)
D_CACHE = build_cache(drafter)


def reset(cache):
    n = cache.get_seq_length() - len(PREFIX)
    if n > 0:
        cache.crop(len(PREFIX))
    assert cache.get_seq_length() == len(PREFIX)


@torch.no_grad()
def greedy_prefix(pids):
    """Plain greedy with the verifier's prefix cache -- the control."""
    reset(V_CACHE)
    out = model(input_ids=torch.tensor([pids[len(PREFIX):]], device="cuda"),
                past_key_values=V_CACHE, use_cache=True)
    nxt = int(out.logits[0, -1].argmax())
    gen, steps = [], 1
    while len(gen) < MAX_NEW and nxt not in STOP:
        gen.append(nxt)
        out = model(input_ids=torch.tensor([[nxt]], device="cuda"),
                    past_key_values=V_CACHE, use_cache=True)
        nxt = int(out.logits[0, -1].argmax())
        steps += 1
    return gen, steps


@torch.no_grad()
def spec_smalllm(pids):
    """Conventional speculative decoding: the 0.5B drafts GAMMA tokens
    autoregressively, the 3B verifies them in one pass, repeat.

    Returns (tokens, verifier_passes, drafter_passes, offered, accepted,
             drafter_ms). Both caches are prefix-cropped per command.
    """
    reset(V_CACHE)
    reset(D_CACHE)
    suf = pids[len(PREFIX):]
    ctx = list(suf)                 # everything after the shared prefix
    v_len = len(PREFIX)             # tokens already in the verifier cache
    d_len = len(PREFIX)             # tokens already in the drafter cache
    gen = []
    v_passes = d_passes = 0
    offered = accepted = 0
    d_ms = 0.0
    while len(gen) < MAX_NEW:
        # ---- drafter: GAMMA greedy tokens, its own KV cache ----------------
        torch.cuda.synchronize(); t0 = time.perf_counter()
        feed = ctx[d_len - len(PREFIX):]
        out = drafter(input_ids=torch.tensor([feed], device="cuda"),
                      past_key_values=D_CACHE, use_cache=True)
        d_passes += 1
        d_len = len(PREFIX) + len(ctx)
        nxt = int(out.logits[0, -1].argmax())
        draft = [nxt]
        while len(draft) < GAMMA:
            out = drafter(input_ids=torch.tensor([[draft[-1]]], device="cuda"),
                          past_key_values=D_CACHE, use_cache=True)
            d_passes += 1
            d_len += 1
            draft.append(int(out.logits[0, -1].argmax()))
        torch.cuda.synchronize(); d_ms += (time.perf_counter() - t0) * 1000
        # the drafter cache now holds PREFIX + ctx + draft[:GAMMA-1]

        # ---- verifier: one pass over the new context + the draft ----------
        new = ctx[v_len - len(PREFIX):]
        out = model(input_ids=torch.tensor([new + draft], device="cuda"),
                    past_key_values=V_CACHE, use_cache=True)
        v_passes += 1
        s_ = len(new)
        preds = out.logits[0, s_ - 1:s_ + GAMMA].argmax(dim=-1).tolist()
        n = 0
        while n < GAMMA and preds[n] == draft[n]:
            n += 1
        offered += GAMMA
        accepted += n
        # emit accepted tokens, stopping before any stop token (as plain greedy does)
        stopped = False
        for t in draft[:n]:
            if t in STOP:
                stopped = True
                break
            gen.append(t)
        if stopped or len(gen) >= MAX_NEW:
            break
        bonus = preds[n]
        # verifier cache is valid for PREFIX + ctx + draft[:n]
        v_len = len(PREFIX) + len(ctx) + n
        if GAMMA > n:
            V_CACHE.crop(v_len)
        # drafter cache: keep the accepted draft entries it already holds
        d_keep = len(PREFIX) + len(ctx) + min(n, GAMMA - 1)
        D_CACHE.crop(d_keep)
        d_len = d_keep
        ctx = ctx + draft[:n] + [bonus]
        if bonus in STOP:
            break
        gen.append(bonus)
    return gen, v_passes, d_passes, offered, accepted, d_ms


# ------------------------------------------------------------------ warmup
print()
print("  warming up (%d commands pre-run; all %d are measured below)" % (N_WARMUP, len(items)))
for x in items[:N_WARMUP]:
    greedy_prefix(x["prompt_ids"])
    spec_smalllm(x["prompt_ids"])
torch.cuda.synchronize()

print("  decoding all %d commands two ways" % len(items))
recs = []
t_all = time.time()
for n_i, x in enumerate(items):
    pids = x["prompt_ids"]
    torch.cuda.synchronize(); t0 = time.perf_counter()
    g_ids, g_steps = greedy_prefix(pids)
    torch.cuda.synchronize(); g_ms = (time.perf_counter() - t0) * 1000

    torch.cuda.synchronize(); t0 = time.perf_counter()
    s_ids, v_p, d_p, off, acc, d_ms = spec_smalllm(pids)
    torch.cuda.synchronize(); s_ms = (time.perf_counter() - t0) * 1000

    recs.append(dict(command_id=x["command_id"], command=x["command"],
                     greedy_text=tok.decode(g_ids, skip_special_tokens=True),
                     spec_text=tok.decode(s_ids, skip_special_tokens=True),
                     identical=(s_ids == g_ids),
                     greedy_tokens=len(g_ids),
                     greedy_forward_passes=g_steps,
                     verifier_passes=v_p, drafter_passes=d_p,
                     draft_tokens_offered=off, draft_tokens_accepted=acc,
                     acceptance=round(acc / off, 4) if off else "",
                     prefix_greedy_ms=round(g_ms, 3), spec_ms=round(s_ms, 3),
                     drafter_ms=round(d_ms, 3)))
    if (n_i + 1) % 50 == 0:
        print("    %3d / %d  (%.0fs)" % (n_i + 1, len(items), time.time() - t_all))
print("    %3d / %d  DONE (%.0fs)" % (len(items), len(items), time.time() - t_all))

m = recs
n_id = sum(r["identical"] for r in m)
print()
print("=" * 96)
print("1. BYTE-IDENTICAL OUTPUT vs PLAIN GREEDY (must be %d/%d before any speed claim)" % (len(m), len(m)))
print("=" * 96)
print("    0.5b-drafted speculative == plain greedy : %d / %d   %s"
      % (n_id, len(m), "PASS" if n_id == len(m) else "*** FAIL ***"))
for r in m:
    if not r["identical"]:
        print("      MISMATCH cmd %s\n        greedy: %r\n        spec  : %r"
              % (r["command_id"], r["greedy_text"], r["spec_text"]))

# cross-check against the A10 run's saved greedy text (same runtime, same prompts)
try:
    a10 = {r["command_id"]: r["greedy_text"] for r in
           csv.DictReader(open(OUT / ("a10_prefix_cache_percommand%s.csv" % TAG), encoding="utf-8", newline=""))}
    agree = sum(1 for r in m if a10.get(r["command_id"]) == r["greedy_text"])
    print("    this run's plain greedy == A10's saved greedy : %d / %d" % (agree, len(m)))
except Exception as e:                                             # noqa: BLE001
    print("    (could not cross-check against A10: %s)" % e)

tot_off = sum(r["draft_tokens_offered"] for r in m)
tot_acc = sum(r["draft_tokens_accepted"] for r in m)
print()
print("=" * 96)
print("2. ACCEPTANCE")
print("=" * 96)
print("    draft tokens offered / accepted   %d / %d" % (tot_off, tot_acc))
print("    TOKEN ACCEPTANCE RATE             %.4f" % (tot_acc / tot_off))
print("    verifier forward passes           %d" % sum(r["verifier_passes"] for r in m))
print("    drafter  forward passes           %d" % sum(r["drafter_passes"] for r in m))
print("    plain-greedy forward passes       %d" % sum(r["greedy_forward_passes"] for r in m))
print("    commands finished in ONE verifier pass %d / %d"
      % (sum(1 for r in m if r["verifier_passes"] == 1), len(m)))
pa = [r["draft_tokens_accepted"] / r["draft_tokens_offered"] for r in m]
print("    per-command acceptance p50 %.4f  p95 %.4f  min %.4f  max %.4f"
      % (pct(pa, .5), pct(pa, .95), min(pa), max(pa)))
hist = collections.Counter(r["verifier_passes"] for r in m)
print("    verifier passes per command: %s" % sorted(hist.items()))

print()
print("=" * 96)
print("3. LATENCY (n=%d, all commands included; %d were pre-run as GPU warm-up)" % (len(m), N_WARMUP))
print("=" * 96)
lat = {"prefix-cached plain greedy": [r["prefix_greedy_ms"] for r in m],
       "0.5b-drafted speculative (A12)": [r["spec_ms"] for r in m],
       "   of which the drafter": [r["drafter_ms"] for r in m]}
print("    %-34s %9s %9s %9s %9s" % ("arm", "p50", "p95", "p99", "mean"))
print("    " + "-" * 74)
for kk, v in lat.items():
    print("    %-34s %8.2f %8.2f %8.2f %8.2f"
          % (kk, pct(v, .50), pct(v, .95), pct(v, .99), statistics.mean(v)))

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
        print("    %-36s N=%-6d wall %6.1fs  meanP %6.1f W  delta %6.1f W  %8.4f J/cmd"
              % (name, n, wall, mw, d, j))
        return dict(arm=name, n_commands=n, wall_s=round(wall, 1),
                    mean_power_w=round(mw, 2), idle_power_w=IDLE_W,
                    delta_power_w=round(d, 2), joules_per_command=round(j, 5),
                    power_samples=len(live))

    print()
    print("=" * 96)
    print("4. ENERGY (GPU only, idle %.2f W, same protocol as energy.csv / a8 / a10)" % IDLE_W)
    print("=" * 96)
    energy_rows.append(measure("a12_prefix_cached_plain_greedy",
                               lambda: [greedy_prefix(x["prompt_ids"]) for x in items], len(items)))
    energy_rows.append(measure("a12_prefix_cached_spec_qwen05b_drafter",
                               lambda: [spec_smalllm(x["prompt_ids"]) for x in items], len(items)))
else:
    print()
    print("  energy NOT measured this run (pass --energy)")

p = OUT / ("a12_smalllm_drafter_percommand%s.csv" % TAG)
with open(p, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(recs[0].keys()))
    w.writeheader(); w.writerows(recs)
(OUT / ("a12_smalllm_drafter_summary%s.json" % TAG)).write_text(json.dumps(dict(
    drafter=DRAFTER, verifier=BASE, adapter=str(ADAPTER.relative_to(REPO)),
    gamma=GAMMA, gamma_swept=False, prefix_tokens=k, n_commands=len(m),
    warmup_prerun_not_excluded=N_WARMUP,
    byte_identical_vs_plain_greedy=n_id,
    draft_tokens_offered=tot_off, draft_tokens_accepted=tot_acc,
    token_acceptance_rate=round(tot_acc / tot_off, 4),
    verifier_passes=sum(r["verifier_passes"] for r in m),
    drafter_passes=sum(r["drafter_passes"] for r in m),
    plain_greedy_passes=sum(r["greedy_forward_passes"] for r in m),
    one_verifier_pass_commands=sum(1 for r in m if r["verifier_passes"] == 1),
    latency_ms={kk: dict(p50=round(pct(v, .50), 2), p95=round(pct(v, .95), 2),
                         p99=round(pct(v, .99), 2), mean=round(statistics.mean(v), 2))
                for kk, v in lat.items()},
    energy=energy_rows), indent=2), encoding="utf-8")
if energy_rows:
    pe = OUT / ("a12_energy%s.csv" % TAG)
    with open(pe, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(energy_rows[0].keys()))
        w.writeheader(); w.writerows(energy_rows)
    print("wrote %s" % pe.relative_to(REPO))
print()
print("wrote %s" % p.relative_to(REPO))
print("wrote %s" % (OUT / ("a12_smalllm_drafter_summary%s.json" % TAG)))
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
            _mine = json.dumps(json.loads(_r["spec_text"]), sort_keys=True)
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
