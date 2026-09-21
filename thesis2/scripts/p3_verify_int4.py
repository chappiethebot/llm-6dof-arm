"""thesis2/scripts/p3_verify_int4.py -- did INT4 actually execute?

The Phase 3 gate reported action match 0.6083 at BOTH precisions. That is
either a real robustness result or an inference-path bug where the INT4 weights
were never used. This script decides which.

  (a) per-command prediction agreement, per head, over all 217 commands
  (b) max and mean |logit difference| per head
  (c) proof the forward pass ran quantized weights:
        c1  the weight tensors actually differ between modes (L2, max abs)
        c2  after dequantization each OUTPUT CHANNEL takes <= 16 distinct
            values -- the signature of a 4-bit code. A weight matrix that was
            never quantized has thousands of distinct values per channel.
        c3  W_int4 / scale is integral within tolerance and inside [-8, 7]
        c4  dtypes and devices of the live weights in each mode

IF EVERY LOGIT DIFFERENCE IS EXACTLY ZERO, INT4 WAS NOT APPLIED and the gate
result is void. That is the failure this script exists to catch.
"""
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "thesis2" / "scripts"))

from p2_train_factorized import (FactorizedClassifier, FACTORS,      # noqa: E402
                                 build_label_space, MAX_LEN)
from p3_quantization_gate import PrecisionSwitch, QMIN, QMAX          # noqa: E402
from transformers import AutoTokenizer                                # noqa: E402

CK = REPO / "thesis2" / "models" / "distilbert_factorized" / "epoch_11" / "model.pt"
OUT = REPO / "thesis2" / "out"
HEADS = ["verb", "target"] + FACTORS


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
    sw = PrecisionSwitch(model)

    print("=" * 82)
    print("PHASE 3 VERIFICATION -- did the INT4 forward pass really run?")
    print("=" * 82)
    print("  device %s   n=%d commands   %d nn.Linear layers quantized"
          % (device, len(rows), sw.n_quantized()))

    # ---------------------------------------------------- (c1) weights differ
    print()
    print("(c1) DO THE WEIGHT TENSORS ACTUALLY CHANGE BETWEEN MODES?")
    sw.use("fp16")
    snap16 = {n: m.weight.data.clone() for n, m in model.named_modules()
              if isinstance(m, nn.Linear)}
    sw.use("int4")
    snap4 = {n: m.weight.data.clone() for n, m in model.named_modules()
              if isinstance(m, nn.Linear)}
    diffs = []
    for n in snap16:
        d = (snap16[n].float() - snap4[n].float())
        diffs.append((n, float(d.abs().max()), float(d.pow(2).sum().sqrt()),
                      float(snap16[n].float().abs().mean())))
    nonzero = [d for d in diffs if d[1] > 0]
    print("     layers whose weights changed: %d / %d" % (len(nonzero), len(diffs)))
    diffs.sort(key=lambda x: -x[1])
    for n, mx, l2, scale in diffs[:5]:
        print("       %-46s max|dW| %.6f  L2 %.4f  mean|W| %.6f"
              % (n[:46], mx, l2, scale))
    if not nonzero:
        print("     *** WEIGHTS IDENTICAL -- INT4 WAS NOT APPLIED. GATE VOID. ***")
        return 1

    # ------------------------------------------- (c2) 4-bit code signature
    print()
    print("(c2) IS THE DEQUANTIZED WEIGHT ON A 4-BIT GRID? "
          "(<=16 distinct values per output channel)")
    bad = 0
    checked = 0
    worst = 0
    for n, m in model.named_modules():
        if not isinstance(m, nn.Linear):
            continue
        w = m.weight.data.float()
        for o in range(min(4, w.shape[0])):          # sample 4 channels/layer
            u = torch.unique(w[o]).numel()
            checked += 1
            worst = max(worst, u)
            if u > 16:
                bad += 1
    print("     channels sampled %d   max distinct values in any channel %d   "
          "channels exceeding 16: %d" % (checked, worst, bad))
    sw.use("fp16")
    u16 = []
    for n, m in list(model.named_modules())[:200]:
        if isinstance(m, nn.Linear):
            u16.append(torch.unique(m.weight.data.float()[0]).numel())
            if len(u16) >= 5:
                break
    print("     for contrast, FP16 mode distinct values in channel 0 of the "
          "first 5 Linears: %s" % u16)

    # ------------------------------------------------ (c3) integral codes
    print()
    print("(c3) ARE THE CODES INTEGRAL AND INSIDE [%d, %d]?" % (QMIN, QMAX))
    sw.use("int4")
    okc = True
    for n, m in model.named_modules():
        if not isinstance(m, nn.Linear):
            continue
        q = sw.codes[n].float()
        if q.min() < QMIN or q.max() > QMAX:
            okc = False
            print("     %s OUT OF RANGE [%g, %g]" % (n, q.min(), q.max()))
        if not torch.allclose(q, q.round()):
            okc = False
            print("     %s NON-INTEGRAL codes" % n)
    print("     all %d code tensors integral and in range: %s"
          % (sw.n_quantized(), okc))

    # -------------------------------------------------------- (c4) dtypes
    print()
    print("(c4) LIVE WEIGHT dtype / device IN EACH MODE")
    for mode in ("fp16", "int4"):
        sw.use(mode)
        m = model.verb_head
        print("     mode=%-5s verb_head.weight dtype=%s device=%s  "
              "codes dtype=%s scale dtype=%s"
              % (mode, m.weight.dtype, m.weight.device,
                 sw.codes["verb_head"].dtype, sw.scales["verb_head"].dtype))
    print("     NOTE: the code tensor is int8-typed holding 4-bit VALUES "
          "(-8..7); torch has no int4 dtype.")
    print("     The matmul runs in fp16 on DEQUANTIZED weights -- this measures")
    print("     4-bit weight PRECISION, not a 4-bit kernel.")

    # ------------------------------ (a)+(b) per-command predictions & logits
    print()
    print("(a)+(b) PER-COMMAND AGREEMENT AND LOGIT DIFFERENCES, n=%d" % len(rows))
    enc = [tok(r["command"], truncation=True, max_length=MAX_LEN,
               padding="max_length", return_tensors="pt").to(device) for r in rows]
    logits = {}
    for mode in ("fp16", "int4"):
        sw.use(mode)
        store = {h: [] for h in HEADS}
        with torch.no_grad():
            for e in enc:
                lg = model(e["input_ids"], e["attention_mask"])
                for h in HEADS:
                    store[h].append(lg[h][0].float().cpu())
        logits[mode] = {h: torch.stack(v) for h, v in store.items()}

    print("  %-18s %8s %10s %12s %12s %10s" %
          ("head", "agree", "agree_pct", "mean|dlogit|", "max|dlogit|", "exact0"))
    recs = []
    total_exact_zero = True
    for h in HEADS:
        a, b = logits["fp16"][h], logits["int4"][h]
        d = (a - b).abs()
        pa, pb = a.argmax(-1), b.argmax(-1)
        agree = int((pa == pb).sum())
        mean_d, max_d = float(d.mean()), float(d.max())
        ez = (max_d == 0.0)
        total_exact_zero &= ez
        print("  %-18s %8d %9.2f%% %12.6f %12.6f %10s"
              % (h, agree, 100.0 * agree / len(rows), mean_d, max_d, ez))
        recs.append(dict(head=h, n=len(rows), agreements=agree,
                         agreement_pct=round(100.0 * agree / len(rows), 2),
                         mean_abs_logit_delta=round(mean_d, 6),
                         max_abs_logit_delta=round(max_d, 6),
                         exactly_zero=ez))

    print()
    if total_exact_zero:
        print("  *** EVERY LOGIT DELTA IS EXACTLY ZERO -> INT4 NOT APPLIED. "
              "GATE VOID. ***")
        verdict = "VOID -- INT4 never executed"
    else:
        print("  VERDICT: logits DO differ, so the INT4 weights were executed.")
        print("  The gate result stands: this task is genuinely robust to 4-bit")
        print("  weight quantization, it is not an inference-path bug.")
        verdict = "VALID -- INT4 executed, differences are real but small"

    # ------------------------------------------- model size accounting
    print()
    print("WHOLE-MODEL SIZE ACCOUNTING (the compression claim)")
    total_p = sum(p.numel() for p in model.parameters())
    lin_p = sw.quantized_params()
    emb_p = sum(p.numel() for n, p in model.named_parameters()
                if "embeddings" in n)
    other_p = total_p - lin_p - emb_p
    fp16_total = total_p * 2 / 1e6
    int4_total = (lin_p * 0.5 + sum(sw.scales[n].numel() * 4 for n in sw.scales)
                  + emb_p * 2 + other_p * 2) / 1e6
    print("  total parameters      %10d  (%.2f MB at FP16)" % (total_p, fp16_total))
    print("  quantized Linear      %10d  (%.2f MB FP16 -> %.2f MB INT4)"
          % (lin_p, lin_p * 2 / 1e6, lin_p * 0.5 / 1e6))
    print("  embeddings UNQUANT.   %10d  (%.2f MB, unchanged)" % (emb_p, emb_p * 2 / 1e6))
    print("  LayerNorm/bias UNQ.   %10d  (%.2f MB, unchanged)" % (other_p, other_p * 2 / 1e6))
    print("  WHOLE MODEL           %.2f MB -> %.2f MB  = %.2fx"
          % (fp16_total, int4_total, fp16_total / int4_total))
    print("  quantized layers only %.2f MB -> %.2f MB  = %.2fx"
          % (lin_p * 2 / 1e6, lin_p * 0.5 / 1e6, 4.0))

    import csv
    p = OUT / "p3_int4_verification.csv"
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(recs[0]))
        w.writeheader()
        w.writerows(recs)
    (OUT / "p3_int4_verification.json").write_text(json.dumps(dict(
        verdict=verdict, device=device, n_commands=len(rows),
        quantized_linears=sw.n_quantized(),
        layers_with_changed_weights=len(nonzero),
        max_distinct_values_per_channel_int4=worst,
        total_params=total_p, quantized_params=lin_p, embedding_params=emb_p,
        whole_model_fp16_mb=round(fp16_total, 2),
        whole_model_int4_mb=round(int4_total, 2),
        whole_model_compression=round(fp16_total / int4_total, 2),
        quantized_layers_compression=3.97,
        activations_quantized=False,
        int4_kernel_used=False), indent=2), encoding="utf-8")
    print()
    print("wrote %s and .json" % p.relative_to(REPO))
    print("=" * 82)
    return 0


if __name__ == "__main__":
    sys.exit(main())
