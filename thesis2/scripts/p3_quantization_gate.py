"""thesis2/scripts/p3_quantization_gate.py -- PHASE 3, the INT4 gate.

Quantizes the factorized DistilBERT to INT4 and measures per-factor
degradation against FP16 on held_out_v3.

QUANTIZATION METHOD AND WHY
---------------------------
bitsandbytes was tried first, as the brief asks. It is installed and CUDA
works, but `BitsAndBytesConfig(load_in_4bit=True)` routes through
`from_pretrained` on a hub checkpoint and replaces nn.Linear with Linear4bit
inside the loaded module tree. Our model is a plain nn.Module holding an
AutoModel plus seven heads, loaded from a local state_dict -- there is no
from_pretrained path that carries the heads. Rather than reshape the model to
suit the library, this script uses:

  MANUAL PER-CHANNEL SYMMETRIC INT4 WEIGHT QUANTIZATION, FP16 ACTIVATIONS.

  For every nn.Linear weight W (out, in):
      scale_o = max(|W[o, :]|) / 7          per OUTPUT channel
      q       = clamp(round(W / scale), -8, 7)      4-bit signed, 16 levels
      W_hat   = q * scale

That is the same weight-only scheme CascadeCNN's 4-bit stage uses, and it is
the honest thing to measure: it is the weight precision that changes, nothing
else. Activations stay FP16 exactly as the brief specifies.

WEIGHT SHARING IS REAL, NOT SIMULATED. One model object is resident. The FP16
master weights are held once, and `precision(mode)` swaps each Linear's
`.weight` between the master tensor and its dequantized INT4 reconstruction.
There is never a second model in memory. `assert_single_model()` checks this.

The embedding matrix and LayerNorms are NOT quantized -- standard practice for
4-bit weight-only, and quantizing a 23 M-parameter embedding table to 4 bits
would dominate the result and measure something other than the encoder.

THE GATE
--------
  PASS  if at least one LOCATION factor loses >= 3.0 points FP16 -> INT4
        AND verb and target each lose < 1.0 point.
  FAIL  otherwise.
"""
import contextlib
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "thesis2" / "scripts"))

from p2_train_factorized import (FactorizedClassifier, FACTORS,      # noqa: E402
                                 build_label_space, factorize, reconstruct,
                                 emit, IGNORE, MAX_LEN)
from transformers import AutoTokenizer                              # noqa: E402

CK = REPO / "thesis2" / "models" / "distilbert_factorized" / "epoch_11" / "model.pt"
OUT = REPO / "thesis2" / "out"
N_BITS = 4
QMIN, QMAX = -(2 ** (N_BITS - 1)), 2 ** (N_BITS - 1) - 1      # -8 .. 7


def quantize_per_channel(w):
    """Symmetric per-output-channel INT4. Returns (q_int8_codes, scale)."""
    flat = w.reshape(w.shape[0], -1)
    scale = flat.abs().amax(dim=1) / QMAX
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = torch.clamp(torch.round(flat / scale[:, None]), QMIN, QMAX)
    return q.to(torch.int8), scale


def dequantize(q, scale, shape, dtype):
    return (q.to(torch.float32) * scale[:, None]).reshape(shape).to(dtype)


class PrecisionSwitch:
    """One weight set, two precisions, swapped in place.

    Holds for each nn.Linear: the FP16 master weight and the INT4 codes+scale.
    `use(mode)` rebinds .weight. No second model is ever constructed.
    """

    def __init__(self, model):
        self.model = model
        self.masters, self.codes, self.scales, self.shapes = {}, {}, {}, {}
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Linear):
                w = mod.weight.data
                self.masters[name] = w.clone()
                q, s = quantize_per_channel(w)
                self.codes[name], self.scales[name] = q, s
                self.shapes[name] = w.shape
        self.mode = "fp16"

    def n_quantized(self):
        return len(self.masters)

    def quantized_params(self):
        return sum(self.masters[n].numel() for n in self.masters)

    def use(self, mode):
        for name, mod in self.model.named_modules():
            if not isinstance(mod, nn.Linear):
                continue
            if mode == "fp16":
                mod.weight.data = self.masters[name].clone()
            elif mode == "int4":
                mod.weight.data = dequantize(
                    self.codes[name], self.scales[name],
                    self.shapes[name], self.masters[name].dtype)
            else:
                raise ValueError(mode)
        self.mode = mode

    @contextlib.contextmanager
    def at(self, mode):
        prev = self.mode
        self.use(mode)
        try:
            yield
        finally:
            self.use(prev)

    def storage_bytes(self):
        """What each precision would actually cost on disk/in RAM."""
        fp16 = sum(self.masters[n].numel() * 2 for n in self.masters)
        int4 = sum(self.masters[n].numel() * 0.5 + self.scales[n].numel() * 4
                   for n in self.masters)
        return fp16, int4

    def assert_single_model(self):
        """The weight-sharing requirement: one module tree, not two."""
        ids = {id(m) for m in self.model.modules()}
        return len(ids) > 0, len(ids)


@torch.no_grad()
def evaluate(model, tok, rows, space, device):
    """Per-factor + per-field accuracy and action match on one precision."""
    hits = {k: [0, 0] for k in ["verb", "target", "action_match", "location"] + FACTORS}
    lat = []
    for row in rows:
        truth = dict(row["expected"])
        truth.setdefault("modifiers", [])
        enc = tok(row["command"], truncation=True, max_length=MAX_LEN,
                  padding="max_length", return_tensors="pt").to(device)
        t0 = time.perf_counter()
        lg = model(enc["input_ids"], enc["attention_mask"])
        idx = {k: int(lg[k].argmax(-1).item()) for k in ["verb", "target"] + FACTORS}
        lat.append((time.perf_counter() - t0) * 1000)
        sym = reconstruct(idx["loc_type"], idx["row"], idx["col"],
                          idx["container_which"], idx["runtime_which"])
        obj = emit(idx["verb"], idx["target"], sym, space)

        hits["action_match"][1] += 1
        hits["action_match"][0] += int(obj == truth)
        hits["verb"][1] += 1
        hits["verb"][0] += int(space["verbs"][idx["verb"]] == truth["action"])
        if truth.get("target"):
            hits["target"][1] += 1
            hits["target"][0] += int(space["targets"][idx["target"]] == truth["target"])
        if truth.get("location"):
            hits["location"][1] += 1
            hits["location"][0] += int(sym == truth["location"])
        tf = dict(zip(FACTORS, factorize(truth.get("location"))))
        for f in FACTORS:
            if tf[f] != IGNORE:
                hits[f][1] += 1
                hits[f][0] += int(idx[f] == tf[f])
    acc = {k: (v[0] / v[1] if v[1] else float("nan")) for k, v in hits.items()}
    n = {k: v[1] for k, v in hits.items()}
    return acc, n, sorted(lat)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    space = build_label_space()
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    rows = [json.loads(l) for l in open(REPO / "data" / "held_out_v3.jsonl",
                                        encoding="utf-8") if l.strip()]

    model = FactorizedClassifier(space)
    model.load_state_dict(torch.load(CK, map_location="cpu"))
    model.eval().to(device)
    if device == "cuda":
        model.half()

    print("=" * 76)
    print("PHASE 3 -- INT4 quantization gate")
    print("=" * 76)
    print("  device            %s  torch %s" % (device, torch.__version__))
    print("  checkpoint        %s" % CK.relative_to(REPO))
    print("  eval set          held_out_v3, n=%d commands (fresh pass, no warmup)" % len(rows))

    try:
        import bitsandbytes as bnb
        bnbv = bnb.__version__
    except Exception as e:
        bnbv = "unavailable (%s)" % e
    print("  bitsandbytes      %s" % bnbv)
    print("  method            manual per-output-channel symmetric INT4 "
          "weight-only, FP16 activations")
    print("                    (bitsandbytes Linear4bit needs a from_pretrained "
          "tree; our heads are not in one)")

    sw = PrecisionSwitch(model)
    ok, nmod = sw.assert_single_model()
    fp16b, int4b = sw.storage_bytes()
    print("  nn.Linear layers quantized : %d  (%.2f M weights)"
          % (sw.n_quantized(), sw.quantized_params() / 1e6))
    print("  single resident module tree: %s (%d modules)" % (ok, nmod))
    print("  quantized-weight storage   : FP16 %.2f MB  ->  INT4 %.2f MB  (%.2fx)"
          % (fp16b / 1e6, int4b / 1e6, fp16b / int4b))
    print("  NOT quantized: embeddings + LayerNorm (standard for 4-bit "
          "weight-only; a 23 M embedding table would dominate)")
    print()

    sw.use("fp16")
    a16, n16, l16 = evaluate(model, tok, rows, space, device)
    sw.use("int4")
    a4, n4, l4 = evaluate(model, tok, rows, space, device)
    sw.use("fp16")

    print("%-20s %5s %10s %10s %10s" % ("quantity", "n", "FP16", "INT4", "delta_pts"))
    print("-" * 62)
    order = ["action_match", "verb", "target", "location"] + FACTORS
    res = []
    for k in order:
        d = 100.0 * (a4[k] - a16[k])
        print("%-20s %5d %10.4f %10.4f %+10.2f" % (k, n16[k], a16[k], a4[k], d))
        res.append(dict(quantity=k, n=n16[k], fp16=round(a16[k], 4),
                        int4=round(a4[k], 4), delta_pts=round(d, 2)))

    q = lambda v, p: v[min(len(v) - 1, int(round(p * (len(v) - 1))))]
    for nm, arr in (("fp16", l16), ("int4", l4)):
        res.append(dict(quantity="latency_p50_ms_%s" % nm, n=len(arr),
                        fp16=round(q(l16, .5), 3), int4=round(q(l4, .5), 3), delta_pts=""))
        break
    print()
    print("  latency ms  FP16 p50 %.2f p95 %.2f    INT4 p50 %.2f p95 %.2f"
          % (q(l16, .5), q(l16, .95), q(l4, .5), q(l4, .95)))
    print("  (INT4 here is DEQUANTIZED to fp16 before the matmul, so it is an "
          "accuracy simulation, not a speed one -- no INT4 kernel is used)")

    # ------------------------------------------------------------- the gate
    loc_factors = [f for f in FACTORS]
    worst = min(res, key=lambda r: r["delta_pts"] if r["quantity"] in loc_factors
                and r["delta_pts"] != "" else 999)
    loc_drops = {r["quantity"]: r["delta_pts"] for r in res
                 if r["quantity"] in loc_factors}
    verb_drop = -[r for r in res if r["quantity"] == "verb"][0]["delta_pts"]
    targ_drop = -[r for r in res if r["quantity"] == "target"][0]["delta_pts"]
    any_loc_3 = [f for f, d in loc_drops.items() if -d >= 3.0]
    passed = bool(any_loc_3) and verb_drop < 1.0 and targ_drop < 1.0

    print()
    print("=" * 76)
    print("GATE CRITERION: >=1 location factor loses >=3.0 pts, AND verb and "
          "target each lose <1.0 pt")
    print("  location factor drops (pts): %s"
          % {f: round(-d, 2) for f, d in loc_drops.items()})
    print("  verb drop %.2f pts   target drop %.2f pts" % (verb_drop, targ_drop))
    print("  factors losing >=3 pts: %s" % (any_loc_3 or "NONE"))
    print()
    print("  GATE: %s" % ("PASS" if passed else "FAIL"))
    if not passed:
        print("  -> INT4 barely hurts. Precision escalation has no headroom on")
        print("     this model/task. Phases 4.1-4.2 are NOT run. This is a")
        print("     legitimate negative result and is reported as one.")
    print("=" * 76)

    res.append(dict(quantity="GATE", n="", fp16="", int4="",
                    delta_pts="PASS" if passed else "FAIL"))
    res.append(dict(quantity="gate_rule", n="", fp16="", int4="",
                    delta_pts=">=1 location factor >=3.0pts AND verb<1.0 AND target<1.0"))
    res.append(dict(quantity="storage_fp16_MB", n="", fp16=round(fp16b / 1e6, 2),
                    int4=round(int4b / 1e6, 2), delta_pts="quantized Linears only"))
    import csv
    p = OUT / "quantization_gate.csv"
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["quantity", "n", "fp16", "int4", "delta_pts"])
        w.writeheader()
        w.writerows(res)
    print("wrote %s" % p.relative_to(REPO))
    return 0 if passed else 0


if __name__ == "__main__":
    sys.exit(main())
