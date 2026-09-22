"""thesis2/scripts/a14_geometry_gate.py -- A14.1: does frozen embedding geometry
already encode the up/down -> row mapping?

RECOVERED 2026-09-22 from the session scratchpad where it was first run
(`g2_geometry.py`); committed here so the gate has a script and an output file
in the repository. It now also writes thesis2/out/a14_geometry_gate.csv.

QUESTION. `held_out_v3` names rows A and C with `upper`/`bottom`, `held_out_v4c`
with `back`/`front`, and training uses neither pair (see a14_legend_check.py).
If a frozen encoder already placed `upper`/`top` near `far`/`back` and
`bottom`/`lower` near `near`/`front`, a model could recover the mapping without
being told it. This asks whether it does.

TEMPLATE, identical for every word, only the slot changes:
    "the cell on the ___ side of the table."

REPRESENTATIONS
  qwen2.5-3b : MEAN-POOLED LAST HIDDEN STATES (attention-masked mean), fp16
  MiniLM     : MEAN-POOLED INPUT EMBEDDINGS   (embedding layer, masked mean)
  DistilBERT : MEAN-POOLED INPUT EMBEDDINGS   (embedding layer, masked mean)

The Qwen checkpoint is Qwen2.5-3B-Instruct -- the checkpoint this project uses
as the base for the LoRA. The non-instruct Qwen2.5-3B is not cached on this
machine and TLS interception has blocked HF downloads here before, so it was
not fetched; the substitution is stated rather than hidden.

GATE, fixed before the numbers were seen:
  PASS iff  upper  and top   are closer to mean{far, back}  than to mean{near, front}
       AND  bottom and lower are closer to mean{near, front} than to mean{far, back}
  where "closer" is mean cosine similarity to the two-word set.

Nothing is trained. Run (conda env thesis2gpu):
    python thesis2/scripts/a14_geometry_gate.py
"""
import csv
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)
sys.path.insert(0, str(REPO))
OUT = REPO / "thesis2" / "out"

WORDS = ["far", "near", "front", "back", "upper", "lower", "top", "bottom"]
TEMPLATE = "the cell on the %s side of the table."
FARSET, NEARSET = ["far", "back"], ["near", "front"]
QWEN = "Qwen/Qwen2.5-3B-Instruct"
MINILM = "sentence-transformers/all-MiniLM-L6-v2"
DISTIL = "distilbert-base-uncased"

rows_csv = []


def cos_matrix(vecs):
    v = F.normalize(torch.stack(vecs).float(), dim=-1)
    return v @ v.T


def report(name, how, M, vecs):
    print()
    print("=" * 104)
    print("%s   (%s)" % (name, how))
    print("=" * 104)
    print("  %-8s %s" % ("", " ".join("%8s" % w for w in WORDS)))
    for i, w in enumerate(WORDS):
        print("  %-8s %s" % (w, " ".join("%8.4f" % M[i, j] for j in range(len(WORDS)))))
        for j, w2 in enumerate(WORDS):
            rows_csv.append(dict(encoder=name, representation=how, kind="cosine",
                                 word=w, other=w2, value=round(float(M[i, j]), 6),
                                 verdict=""))
    idx = {w: i for i, w in enumerate(WORDS)}
    print()
    print("  %-8s %10s %10s %10s   %s" % ("word", "mean cos", "mean cos", "delta", "verdict"))
    print("  %-8s %10s %10s %10s" % ("", "far/back", "near/front", "far-near"))
    ok_all = True
    for w in ["upper", "top", "bottom", "lower"]:
        a = float(sum(M[idx[w], idx[x]] for x in FARSET) / 2)
        b = float(sum(M[idx[w], idx[x]] for x in NEARSET) / 2)
        want = "far/back" if w in ("upper", "top") else "near/front"
        got = "far/back" if a > b else "near/front"
        ok = (got == want)
        ok_all &= ok
        print("  %-8s %10.4f %10.4f %+10.4f   want %-10s got %-10s %s"
              % (w, a, b, a - b, want, got, "OK" if ok else "**WRONG SIDE**"))
        rows_csv.append(dict(encoder=name, representation=how, kind="gate_word", word=w,
                             other="mean{far,back} - mean{near,front}",
                             value=round(a - b, 6), verdict="OK" if ok else "WRONG SIDE"))
    print("  ---> %s: %s" % (name, "PASS" if ok_all else "FAIL"))
    rows_csv.append(dict(encoder=name, representation=how, kind="GATE", word="", other="",
                         value="", verdict="PASS" if ok_all else "FAIL"))

    # robustness note only: does the ORDERING survive removing the shared template?
    X = torch.stack(vecs).float()
    X = X - X.mean(0, keepdim=True)
    Mc = F.normalize(X, dim=-1) @ F.normalize(X, dim=-1).T
    ok_c = True
    deltas = []
    for w in ["upper", "top", "bottom", "lower"]:
        a = float(sum(Mc[idx[w], idx[x]] for x in FARSET) / 2)
        b = float(sum(Mc[idx[w], idx[x]] for x in NEARSET) / 2)
        want = "far/back" if w in ("upper", "top") else "near/front"
        ok_c &= (("far/back" if a > b else "near/front") == want)
        deltas.append("%s %+0.3f" % (w, a - b))
        rows_csv.append(dict(encoder=name, representation=how + " [mean-centered]",
                             kind="gate_word_centered", word=w,
                             other="mean{far,back} - mean{near,front}",
                             value=round(a - b, 6), verdict=""))
    print("  [centered variant, robustness note only] %s: %s   %s"
          % (name, "PASS" if ok_c else "FAIL", "  ".join(deltas)))
    rows_csv.append(dict(encoder=name, representation=how + " [mean-centered]", kind="GATE",
                         word="", other="", value="", verdict="PASS" if ok_c else "FAIL"))
    return ok_all


results = {}
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM   # noqa: E402

print("=" * 104)
print("A14.1 -- GEOMETRY GATE: is the up/down -> row mapping already in frozen embedding space?")
print("=" * 104)
print("  template: %r" % TEMPLATE)
print("  words   : %s" % ", ".join(WORDS))
print("  gate    : upper,top closer to {far,back}; bottom,lower closer to {near,front}")
print()
print("  loading %s (frozen; the checkpoint this project uses as the LoRA base --" % QWEN)
print("  the non-instruct Qwen2.5-3B is not cached on this machine)")
qtok = AutoTokenizer.from_pretrained(QWEN)
qm = AutoModelForCausalLM.from_pretrained(QWEN, dtype=torch.float16).to("cuda").eval()
vecs = []
with torch.no_grad():
    for w in WORDS:
        enc = qtok(TEMPLATE % w, return_tensors="pt").to("cuda")
        h = qm(**enc, output_hidden_states=True).hidden_states[-1][0]
        mask = enc["attention_mask"][0].unsqueeze(-1).float()
        vecs.append(((h.float() * mask).sum(0) / mask.sum()).cpu())
results["qwen2.5-3b"] = report("qwen2.5-3b", "mean-pooled LAST HIDDEN states, fp16, frozen",
                               cos_matrix(vecs), vecs)
del qm
torch.cuda.empty_cache()

for name, path in (("MiniLM (all-MiniLM-L6-v2)", MINILM),
                   ("DistilBERT (distilbert-base-uncased)", DISTIL)):
    tok = AutoTokenizer.from_pretrained(path)
    mdl = AutoModel.from_pretrained(path).eval()
    emb = mdl.get_input_embeddings()
    vs = []
    with torch.no_grad():
        for w in WORDS:
            enc = tok(TEMPLATE % w, return_tensors="pt")
            e = emb(enc["input_ids"])[0]
            mask = enc["attention_mask"][0].unsqueeze(-1).float()
            vs.append((e * mask).sum(0) / mask.sum())
    results[name] = report(name, "mean-pooled INPUT EMBEDDINGS (embedding layer)",
                           cos_matrix(vs), vs)
    counts = {w: len(tok(w, add_special_tokens=False)["input_ids"]) for w in WORDS}
    print("  slot-word token counts: %s" % counts)
    rows_csv.append(dict(encoder=name, representation="tokenizer", kind="slot_token_counts",
                         word="", other=str(counts), value="", verdict=""))

print()
print("=" * 104)
print("GATE SUMMARY")
print("=" * 104)
for k, ok in results.items():
    print("  %-46s %s" % (k, "PASS" if ok else "FAIL"))
print("  OVERALL: %s" % ("PASS" if all(results.values()) else "FAIL"))
rows_csv.append(dict(encoder="ALL", representation="", kind="GATE_OVERALL", word="", other="",
                     value="", verdict="PASS" if all(results.values()) else "FAIL"))

p = OUT / "a14_geometry_gate.csv"
with open(p, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=["encoder", "representation", "kind", "word",
                                       "other", "value", "verdict"])
    w.writeheader()
    w.writerows(rows_csv)
print()
print("wrote %s (%d rows)" % (p.relative_to(REPO), len(rows_csv)))
print("=" * 104)
