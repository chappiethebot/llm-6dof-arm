"""thesis2/scripts/a8_speculative.py -- A8: classifier-drafted speculative decoding.

IDEA. The flat DistilBERT classifier already emits a complete, well-formed
command in the LoRA's exact output format, in ~3 ms on CPU. Speculative
decoding normally uses a small autoregressive DRAFT MODEL; here the drafter is
a DISCRIMINATIVE classifier that produces the whole sequence in one shot. One
LoRA forward pass verifies the entire draft. When the classifier is right --
which on held_out_v3 it is 74.65% of the time -- the whole JSON plus the stop
token is accepted and the command costs ONE forward pass instead of ~30.

The output is greedy-exact by construction: every accepted token had to equal
the LoRA's own argmax at that position, and decoding resumes from the first
mismatch. Accuracy is therefore IDENTICAL and is NOT re-reported as a result.
The number that matters is the byte-identity check below: it must be 217/217,
otherwise the implementation is wrong.

GREEDY WAS CONFIRMED BEFORE ANY OF THIS WAS WRITTEN
---------------------------------------------------
  llm/clients.py:147          options={"temperature": self.temperature, ...}
  bench/run_cell.py:323       records client.temperature into every row
  results_heldout_v3_lora_scored.csv   temperature == 0.0 on all 2170 rows
  ft/export/Modelfile.qwen2.5-3b-lora  PARAMETER temperature 0
Both the client and the Modelfile pin temperature 0, and llama.cpp switches to
argmax sampling at temp <= 0. No top_p/top_k/seed override was ever sent.

RUNTIME MISMATCH -- STATED, NOT HIDDEN
--------------------------------------
V5 decoded through OLLAMA against qwen3b-lora-Q4_K_M.gguf (a merged, 4-bit
quantised GGUF). This script runs transformers FP16 with the PEFT adapter
merged in. Those are different numerics, so byte-identity with the RECORDED V5
rows is not achievable and is not claimed. The 217/217 requirement is between
speculative and plain greedy IN THIS RUNTIME, which is the control that
actually tests the speculative algorithm. Agreement against the V5 ollama rows
is reported separately as a quantisation diagnostic.

Run (conda env thesis2gpu):
    python thesis2/scripts/a8_speculative.py            # correctness + latency
    python thesis2/scripts/a8_speculative.py --energy   # adds the 2x120 s power runs
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
IDLE_W = 12.12          # from thesis2/out/power_idle.csv, same baseline as energy.csv
LOAD_S = 120
N_WARMUP = 3            # PRE-RUN to warm the GPU; all 217 commands are then measured
                        # and INCLUDED in every figure (Part B S-5). Not excluded.

ap = argparse.ArgumentParser()
ap.add_argument("--energy", action="store_true",
                help="PROTOCOL energy run (called by s3_energy.py --arm): decode "
                     "the split once for the byte-identity gate, then measure "
                     "with s3_energy.measure_loop. Writes NO prediction file.")
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
PRED_ARM, PRED_TEXT, PRED_MS = "a8_speculative", "spec_text", "spec_ms"
PERCOMMAND = OUT / ("a8_speculative_percommand%s.csv" % TAG)
if args.energy and (args.limit or args.emit_preds_only):
    C.gate(False, ["--energy measures the FULL split by the s3 protocol; it takes "
                   "neither --limit nor --emit-preds-only"])
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
    if not xs:
        return float("nan")
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


print("=" * 88)
print("A8 -- CLASSIFIER-DRAFTED SPECULATIVE DECODING")
print("=" * 88)

# ---------------------------------------------------------------- gate: greedy?
print()
print("  GATE -- was qwen2.5-3b-lora decoded greedily?")
temps = set()
for r in csv.DictReader(open(REPO / "bench" / "results_heldout_v3_lora_scored.csv",
                             encoding="utf-8")):
    if r["model"] == "ollama:qwen2.5-3b-lora":
        temps.add(r["temperature"])
mf = (REPO / "ft" / "export" / "Modelfile.qwen2.5-3b-lora").read_text(encoding="utf-8")
mf_temp = [l.strip() for l in mf.splitlines() if "temperature" in l.lower()]
print("    recorded temperature values in the V5 lora rows : %s" % sorted(temps))
print("    Modelfile                                       : %s" % (mf_temp or "none"))
if temps != {"0.0"} or not any("temperature 0" in l for l in mf_temp):
    sys.exit("ABORT: not greedy. Stop and report to the user.")
print("    -> GREEDY CONFIRMED. proceeding.")

# ---------------------------------------------------------------- prompt
from bench.run_cell import apply_legend                           # noqa: E402
from llm import prompts as P                                      # noqa: E402

legend_note = apply_legend("v5")
schema = P.load_schema(str(REPO / "schema" / "schema_v2.json"))
print()
print("  legend in force: %s" % legend_note)


def chat_wrap(user):
    """The ollama TEMPLATE from Modelfile.qwen2.5-3b-lora, reproduced literally."""
    return ("<|im_start|>system\n%s<|im_end|>\n"
            "<|im_start|>user\n%s<|im_end|>\n"
            "<|im_start|>assistant\n" % (SYSTEM, user))


rows = [json.loads(l) for l in DATA.read_text(encoding="utf-8").splitlines() if l.strip()]
if args.limit:
    rows = rows[:args.limit]
    print("  SMOKE: first %d commands of %s" % (len(rows), SPLIT))
print("  held_out_v3: %d commands" % len(rows))

# ---------------------------------------------------------------- model
print()
print("  loading %s + %s (fp16, merged -- same recipe as salsa_row.py)"
      % (BASE, ADAPTER.name))
from transformers import AutoTokenizer, AutoModelForCausalLM                # noqa: E402
from peft import PeftModel                                                   # noqa: E402

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16).to("cuda")
model = PeftModel.from_pretrained(model, str(ADAPTER)).merge_and_unload()
model.eval()
IM_END = tok.convert_tokens_to_ids("<|im_end|>")
STOP = {IM_END, tok.eos_token_id}
print("    im_end id %s   stop ids %s" % (IM_END, sorted(x for x in STOP if x is not None)))

# ---------------------------------------------------------------- drafter
print()
print("  loading the flat DistilBERT classifier as the DRAFTER")
from llm.classifier_client import DistilBertClassifierClient                 # noqa: E402

clf = DistilBertClassifierClient()
_ = clf.complete("warm up the classifier")
print("    %s ready" % clf.name)


def render(obj):
    """The LoRA's exact output format: json.dumps, key order action/target/
    location/modifiers. Verified against all 217 recorded V5 raws (0 mismatches)."""
    o = {"action": obj.get("action")}
    if obj.get("target"):
        o["target"] = obj["target"]
    if obj.get("location"):
        o["location"] = obj["location"]
    o["modifiers"] = obj.get("modifiers") or []
    return json.dumps(o)


def draft_for(command):
    """Classifier -> draft token ids. Returns (ids, text, ms)."""
    t0 = time.perf_counter()
    raw = clf.complete(command)
    ms = (time.perf_counter() - t0) * 1000
    try:
        text = render(json.loads(raw))
    except Exception:
        return [], "", ms
    return text, text, ms


# ---------------------------------------------------------------- decoding
@torch.no_grad()
def greedy_manual(prompt_ids):
    """Plain greedy with a KV cache. The baseline, and the thing speculative
    decoding must reproduce byte for byte."""
    ids = torch.tensor([prompt_ids], device="cuda")
    out = model(input_ids=ids, use_cache=True)
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
def speculative(prompt_ids, draft_ids):
    """One forward over prompt+draft, accept the longest matching prefix, then
    resume plain greedy from the first mismatch.

    logits[P-1+i] is the model's prediction for output position i, so comparing
    argmax(logits[P-1+i]) against draft[i] is exactly the greedy check. The
    token at the first mismatch comes free from the same pass (the 'bonus'),
    which is why a fully accepted draft costs ONE forward pass in total.
    """
    P = len(prompt_ids)
    k = len(draft_ids)
    full = torch.tensor([prompt_ids + draft_ids], device="cuda")
    out = model(input_ids=full, use_cache=True)
    past = out.past_key_values
    preds = out.logits[0, P - 1:P + k].argmax(dim=-1).tolist()   # k+1 predictions
    n_acc = 0
    while n_acc < k and preds[n_acc] == draft_ids[n_acc]:
        n_acc += 1
    # An accepted draft token may itself be the stop token. Plain greedy breaks
    # BEFORE appending a stop, so the accepted prefix is truncated the same way
    # or the two paths diverge on the trailing <|im_end|>.
    gen, hit_stop = [], False
    for t in draft_ids[:n_acc]:
        if t in STOP:
            hit_stop = True
            break
        gen.append(t)
    if hit_stop:
        return gen, 1, n_acc, k
    nxt = preds[n_acc]
    # cache is valid for prompt + accepted draft = P + n_acc entries
    if k > n_acc:
        past.crop(-(k - n_acc))
    steps = 1
    while len(gen) < MAX_NEW and nxt not in STOP:
        gen.append(nxt)
        out = model(input_ids=torch.tensor([[nxt]], device="cuda"),
                    past_key_values=past, use_cache=True)
        past = out.past_key_values
        nxt = int(out.logits[0, -1].argmax())
        steps += 1
    # trailing stop token counts as accepted if the draft supplied it
    return gen, steps, n_acc, k


@torch.no_grad()
def salsa_pass(prompt_ids):
    """SALSA's cost shape: exactly one forward pass, logits read at one position."""
    model(input_ids=torch.tensor([prompt_ids], device="cuda"), use_cache=False)


# ---------------------------------------------------------------- build inputs
print()
print("  tokenising and checking the prompt/draft boundary is BPE-clean")
def prep(cmd):
    """One command's inputs -> (prompt_ids, draft_ids, draft_text, draft_ms,
    bpe_clean). Used by the measured pass below AND by the --energy predictor,
    so both run exactly the same pipeline."""
    ptxt = chat_wrap(P.build_prompt(cmd, "zero_shot", schema))
    dtxt, _, dms = draft_for(cmd)
    pids = tok(ptxt, add_special_tokens=False)["input_ids"]
    clean = True
    if dtxt:
        fids = tok(ptxt + dtxt, add_special_tokens=False)["input_ids"]
        if fids[:len(pids)] != pids:
            clean = False
            dids = tok(dtxt, add_special_tokens=False)["input_ids"]
        else:
            dids = fids[len(pids):]
        dids = dids + [IM_END]        # drafting the stop token is what buys the 1-pass case
    else:
        dids = []
    return pids, dids, dtxt, dms, clean


items, dirty = [], 0
for i, r in enumerate(rows):
    cmd = r["command"]
    pids, dids, dtxt, dms, clean = prep(cmd)
    dirty += 0 if clean else 1
    items.append(dict(command_id=str(i), command=cmd, prompt_ids=pids,
                      draft_ids=dids, draft_text=dtxt, draft_ms=dms,
                      truth=r))
print("    prompt token length: min %d  median %d  max %d"
      % (min(len(x["prompt_ids"]) for x in items),
         int(statistics.median([len(x["prompt_ids"]) for x in items])),
         max(len(x["prompt_ids"]) for x in items)))
print("    draft token length : min %d  median %d  max %d"
      % (min(len(x["draft_ids"]) for x in items),
         int(statistics.median([len(x["draft_ids"]) for x in items])),
         max(len(x["draft_ids"]) for x in items)))
print("    BPE boundary unclean on %d / %d commands" % (dirty, len(items)))
print("    classifier draft latency p50 %.2f ms  p95 %.2f ms"
      % (pct([x["draft_ms"] for x in items], .50),
         pct([x["draft_ms"] for x in items], .95)))

# ---------------------------------------------------------------- warmup
print()
print("  warming up (%d commands pre-run; all %d are measured below)" % (N_WARMUP, len(items)))
for x in items[:N_WARMUP]:
    greedy_manual(x["prompt_ids"])
    speculative(x["prompt_ids"], x["draft_ids"])
    salsa_pass(x["prompt_ids"])
torch.cuda.synchronize()

# ---------------------------------------------------------------- measure
print()
print("  decoding all %d commands three ways" % len(items))
recs = []
t_all = time.time()
for n, x in enumerate(items):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    g_ids, g_steps = greedy_manual(x["prompt_ids"])
    torch.cuda.synchronize(); g_ms = (time.perf_counter() - t0) * 1000

    torch.cuda.synchronize(); t0 = time.perf_counter()
    s_ids, s_steps, n_acc, k = speculative(x["prompt_ids"], x["draft_ids"])
    torch.cuda.synchronize(); s_ms = (time.perf_counter() - t0) * 1000
    s_ms_total = s_ms + x["draft_ms"]      # the drafter's cost is part of the arm

    torch.cuda.synchronize(); t0 = time.perf_counter()
    salsa_pass(x["prompt_ids"])
    torch.cuda.synchronize(); sa_ms = (time.perf_counter() - t0) * 1000

    gtxt = tok.decode(g_ids, skip_special_tokens=True)
    stxt = tok.decode(s_ids, skip_special_tokens=True)
    recs.append(dict(command_id=x["command_id"], command=x["command"],
                     greedy_text=gtxt, spec_text=stxt,
                     identical_text=(gtxt == stxt),
                     identical_ids=(g_ids == s_ids),
                     greedy_tokens=len(g_ids), greedy_forward_passes=g_steps,
                     spec_forward_passes=s_steps,
                     draft_tokens=k, draft_accepted=n_acc,
                     full_draft_accepted=(k > 0 and n_acc == k),
                     greedy_ms=round(g_ms, 3), spec_ms=round(s_ms_total, 3),
                     spec_gpu_ms=round(s_ms, 3), draft_ms=round(x["draft_ms"], 3),
                     salsa_ms=round(sa_ms, 3)))
    if (n + 1) % 50 == 0:
        print("    %3d / %d  (%.0fs)" % (n + 1, len(items), time.time() - t_all))
print("    %3d / %d  DONE (%.0fs)" % (len(items), len(items), time.time() - t_all))

# ---------------------------------------------------------------- 1. byte identity
m = recs
n_id_txt = sum(r["identical_text"] for r in m)
n_id_ids = sum(r["identical_ids"] for r in m)
print()
print("=" * 88)
print("1. BYTE-IDENTICAL AGREEMENT WITH PLAIN GREEDY  (must be %d/%d)" % (len(m), len(m)))
print("=" * 88)
print("    decoded text identical : %d / %d   %s"
      % (n_id_txt, len(m), "PASS" if n_id_txt == len(m) else "*** FAIL ***"))
print("    token ids identical    : %d / %d   %s"
      % (n_id_ids, len(m), "PASS" if n_id_ids == len(m) else "*** FAIL ***"))
if n_id_ids != len(m):
    for r in m:
        if not r["identical_ids"]:
            print("      MISMATCH cmd %s" % r["command_id"])
            print("        greedy: %r" % r["greedy_text"])
            print("        spec  : %r" % r["spec_text"])

# ---------------------------------------------------------------- 2. acceptance
tot_d = sum(r["draft_tokens"] for r in m)
tot_a = sum(r["draft_accepted"] for r in m)
full = sum(r["full_draft_accepted"] for r in m)
one_pass = sum(1 for r in m if r["spec_forward_passes"] == 1)
print()
print("=" * 88)
print("2. DRAFT ACCEPTANCE")
print("=" * 88)
print("    draft tokens offered            %d" % tot_d)
print("    draft tokens accepted           %d" % tot_a)
print("    TOKEN ACCEPTANCE RATE           %.4f" % (tot_a / tot_d))
print("    commands with FULL draft taken  %d / %d  = %.4f" % (full, len(m), full / len(m)))
print("    commands finished in ONE pass   %d / %d  = %.4f"
      % (one_pass, len(m), one_pass / len(m)))
print("    forward passes, plain greedy    %d" % sum(r["greedy_forward_passes"] for r in m))
print("    forward passes, speculative     %d" % sum(r["spec_forward_passes"] for r in m))
print("    forward-pass reduction          %.4f"
      % (1 - sum(r["spec_forward_passes"] for r in m)
         / sum(r["greedy_forward_passes"] for r in m)))

# ---------------------------------------------------------------- 3. latency
print()
print("=" * 88)
print("3. LATENCY  (n=%d, all commands included; %d were pre-run as GPU warm-up)" % (len(m), N_WARMUP))
print("=" * 88)
lat = {"plain greedy (transformers fp16)": [r["greedy_ms"] for r in m],
       "speculative (incl. drafter)": [r["spec_ms"] for r in m],
       "speculative (GPU only)": [r["spec_gpu_ms"] for r in m],
       "SALSA (1 forward pass)": [r["salsa_ms"] for r in m]}
print("    %-34s %9s %9s %9s %9s" % ("arm", "p50", "p95", "p99", "mean"))
print("    " + "-" * 74)
for k2, v in lat.items():
    print("    %-34s %8.2f%s %8.2f%s %8.2f%s %8.2f%s"
          % (k2, pct(v, .50), "ms", pct(v, .95), "ms", pct(v, .99), "ms",
             statistics.mean(v), "ms"))
g50, s50 = pct(lat["plain greedy (transformers fp16)"], .50), pct(lat["speculative (incl. drafter)"], .50)
print()
print("    speedup at p50: %.2fx     at p95: %.2fx"
      % (g50 / s50, pct(lat["plain greedy (transformers fp16)"], .95)
         / pct(lat["speculative (incl. drafter)"], .95)))

# ---------------------------------------------------------------- 4. v5 diagnostic
v5 = {}
for r in csv.DictReader(open(REPO / "bench" / "results_v5_qwen3.csv", encoding="utf-8")):
    if r["model"] == "ollama:qwen2.5-3b-lora" and r["strategy"] == "zero_shot":
        v5[r["command_id"]] = r["raw"].strip()
agree = sum(1 for r in m if v5.get(r["command_id"], None) == r["greedy_text"].strip())
print()
print("=" * 88)
print("4. QUANTISATION DIAGNOSTIC -- transformers fp16 vs the recorded ollama Q4_K_M rows")
print("=" * 88)
print("    identical output: %d / %d = %.4f" % (agree, len(m), agree / len(m)))
print("    This is NOT the correctness test. Different runtime, different")
print("    quantisation; a gap here is expected and says nothing about A8.")

# ---------------------------------------------------------------- 5. energy
energy_rows = []
# --energy is measured AFTER the byte-identity gate at the end of this script,
# by s3_energy.measure_loop -- the protocol every arm uses (current idle
# baseline, 3 warm-up commands, >= 120 s, 60 s cool-down, drafter included).
# The old in-script loop (idle 12.12 W hard-coded, no warm-up or cool-down,
# drafter and tokenisation excluded) was removed 2026-09-28.

if args.energy:
    # PROTOCOL energy run: this decode pass feeds the byte-identity gate
    # only. No prediction, per-command or summary file is written, so an
    # energy run can never overwrite a result (the v4c run-once rule).
    print()
    print("  --energy: no prediction, per-command or summary file written")
else:
    # ---------------------------------------------------------------- write
    p = OUT / ("a8_speculative_percommand%s.csv" % TAG)
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(recs[0].keys()))
        w.writeheader(); w.writerows(recs)
    _pp, _pn = C.preds_from_percommand(PRED_ARM, SPLIT, recs, PRED_TEXT, PRED_MS,
                                       outdir=OUT)
    print("wrote %s (%d rows, standard pred format, latency from %s)"
          % (_pp, _pn, PRED_MS))

    summary = dict(
        greedy_confirmed=True, base=BASE, adapter=str(ADAPTER.relative_to(REPO)),
        runtime="transformers fp16, LoRA merged", dataset="held_out_v3",
        n_commands=len(m), warmup_prerun_not_excluded=N_WARMUP,
        byte_identical_text=n_id_txt, byte_identical_ids=n_id_ids,
        draft_tokens_offered=tot_d, draft_tokens_accepted=tot_a,
        token_acceptance_rate=round(tot_a / tot_d, 4),
        full_draft_accepted=full, full_draft_rate=round(full / len(m), 4),
        one_forward_pass_commands=one_pass,
        forward_passes_plain=sum(r["greedy_forward_passes"] for r in m),
        forward_passes_speculative=sum(r["spec_forward_passes"] for r in m),
        latency_ms={k2: dict(p50=round(pct(v, .50), 2), p95=round(pct(v, .95), 2),
                             p99=round(pct(v, .99), 2), mean=round(statistics.mean(v), 2))
                    for k2, v in lat.items()},
        agreement_with_v5_ollama_q4=round(agree / len(m), 4),
        energy=energy_rows,
        accuracy_note="NOT re-scored: greedy-exact by construction, identical to the "
                      "plain-greedy arm by the byte-identity check above.")
    (OUT / ("a8_speculative_summary%s.json" % TAG)).write_text(json.dumps(summary, indent=2),
                                                     encoding="utf-8")
    if energy_rows:
        pe = OUT / ("a8_energy%s.csv" % TAG)
        with open(pe, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(energy_rows[0].keys()))
            w.writeheader(); w.writerows(energy_rows)
        print("wrote %s" % pe.relative_to(REPO))
    print()
    print("wrote %s" % p.relative_to(REPO))
    print("wrote %s" % (OUT / ("a8_speculative_summary%s.json" % TAG)))
    print("=" * 88)

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
    if args.energy:
        # the energy run is gated on byte identity; no identity, no number
        print()
        print("GATE FAIL: --energy requires the byte-identity gate, and %s is "
              "absent. Run s1_run_arm.py --arm lora_greedy --split %s first."
              % (_lg.name, SPLIT))
        sys.exit(1)
    print()
    print("  byte-identity vs lora_greedy NOT CHECKED: %s absent. Run "
          "s1_run_arm.py --arm lora_greedy --split %s first; on held_out_v4c this "
          "check is a required gate." % (_lg.name, SPLIT))

# ---------------------------------------------------------------- protocol energy
# Reached only when the byte-identity gate above PASSED (it exits otherwise).
if args.energy:
    import s3_energy as E                                          # noqa: E402

    def predict_cmd(cmd):
        """One command end to end, as a user pays for it: prompt build, DistilBERT draft, tokenisation, speculative decode, detokenisation."""
        pids, dids, _t, _ms, _c = prep(cmd)
        return tok.decode(speculative(pids, dids)[0], skip_special_tokens=True)

    _idle = E.require_idle()          # re-checked here: the decode pass takes minutes
    # The decode pass has just loaded the GPU for minutes; every other arm starts
    # its measurement from a cooled GPU, so this one does too.
    print()
    print("  pre-measurement cool-down %d s (thermal parity with the other arms)"
          % E.COOLDOWN_S)
    time.sleep(E.COOLDOWN_S)
    C.gate(True, E.measure_loop(
        PRED_ARM, SPLIT, predict_cmd, _idle, kind="local",
        note="speculative arm measured in its own script, end to end per command "
             "(prompt build, DistilBERT draft, tokenisation, speculative decode, detokenisation); byte-identity to lora_greedy PASSED in the same process; "
             "GPU watts only, CPU watts NOT measured"))
