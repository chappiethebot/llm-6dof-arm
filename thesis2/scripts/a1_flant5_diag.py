"""thesis2/scripts/a1_flant5_diag.py -- why did Flan-T5 score 0.0000?

Zero action match AND zero schema validity is not a model result, it is a
harness failure. The prime suspect is T5's SentencePiece vocabulary, which is
known not to contain the curly-brace characters JSON needs.

Checks:
  1. does the tokenizer round-trip a target JSON string at all?
  2. which characters are lost?
  3. what does the fine-tuned model actually emit?
"""
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)
from transformers import AutoTokenizer                       # noqa: E402

tok = AutoTokenizer.from_pretrained("google/flan-t5-base")
target = json.dumps({"action": "pick_place", "target": "bottle",
                     "location": "grid_A1"}, separators=(",", ":"))

print("=" * 78)
print("1. TOKENIZER ROUND-TRIP OF THE TRAINING TARGET")
print("=" * 78)
print("   original : %r" % target)
ids = tok(target, add_special_tokens=False)["input_ids"]
back = tok.decode(ids, skip_special_tokens=True)
print("   decoded  : %r" % back)
print("   IDENTICAL: %s" % (target == back))

print()
print("2. WHICH CHARACTERS SURVIVE?")
lost = []
for ch in sorted(set(target)):
    enc = tok(ch, add_special_tokens=False)["input_ids"]
    dec = tok.decode(enc, skip_special_tokens=True)
    ok = dec.strip() == ch.strip()
    if not ok:
        lost.append(ch)
    print("   %-4r -> ids %-18s decode %-6r %s"
          % (ch, enc, dec, "OK" if ok else "*** LOST ***"))
print()
print("   characters the vocabulary cannot represent: %r" % lost)

print()
print("3. WHAT DOES THE FINE-TUNED MODEL EMIT?")
try:
    import torch
    from transformers import AutoModelForSeq2SeqLM
    m = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m.to(dev).eval()
    probe = "translate command to action json: put the bottle in box 1"
    e = tok(probe, return_tensors="pt").to(dev)
    with torch.no_grad():
        o = m.generate(**e, max_new_tokens=48, do_sample=False)
    print("   (base model, not the fine-tuned checkpoint -- that was not saved)")
    print("   emits: %r" % tok.decode(o[0], skip_special_tokens=True))
except Exception as exc:
    print("   skipped: %s: %s" % (type(exc).__name__, exc))

print()
print("=" * 78)
print("CONCLUSION")
if lost:
    print("   T5's SentencePiece vocabulary cannot represent %r." % lost)
    print("   The training targets were therefore unlearnable AS WRITTEN: the")
    print("   model was asked to emit characters it has no tokens for, so")
    print("   0.0000 schema validity measures MY TARGET FORMAT, not Flan-T5.")
    print("   A brace-free target format is required for a fair test.")
else:
    print("   The tokenizer round-trips fine; the failure is elsewhere.")
print("=" * 78)
