"""thesis2/scripts/salsa_row.py -- single-pass logit classification for the ROW factor.

Per SALSA (arXiv 2510.22691). Map each class to a distinct SINGLE token, prompt
for a one-token answer, and read the logits at that position projected onto
ONLY the valid class tokens. One forward pass, no autoregressive generation, no
sampling.

WHY THIS IS THE RIGHT MECHANISM FOR A FACTOR SUB-QUERY
------------------------------------------------------
The full location question needs 14 classes and `grid_A1` is not one token. The
ROW sub-question needs THREE, and A/B/C are single tokens in Qwen's vocabulary.
Factorization is what makes the sub-query cheap: 3 logits read from one forward
pass instead of generating a JSON object.

LABEL-TOKEN CHOICE IS A CONFOUND AND IS TESTED. Two mappings are implemented,
letters (A/B/C) and digits (1/2/3), because which tokens carry the answer
affects the logits. Both are reported.

Ollama is not used: it does not expose arbitrary token logits. HuggingFace
transformers with the locally cached weights instead. This module does not
import or modify bench/llm_client.py.
"""
import os
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]

ROWS = ["A", "B", "C"]

# The row legend, stated in the SAME terms the v5 schema legend uses: row A is
# the far side of the table, row C the near side. This wording is fixed here
# and is NOT tuned against any evaluation set.
ROW_LEGEND = (
    "A table in front of a robot is divided into three rows.\n"
    "Row A is the row FURTHEST from the robot (the far side, the upper side).\n"
    "Row B is the MIDDLE row.\n"
    "Row C is the row NEAREST the robot (the near side, the bottom side).\n"
)

MAPPINGS = {
    "letters": {"tokens": ["A", "B", "C"], "legend": "A, B or C"},
    "digits": {"tokens": ["1", "2", "3"], "legend": "1, 2 or 3"},
}


def build_prompt(command, mapping):
    m = MAPPINGS[mapping]
    if mapping == "letters":
        opts = "A = far row, B = middle row, C = near row"
    else:
        opts = "1 = far row, 2 = middle row, 3 = near row"
    return (
        ROW_LEGEND
        + "\nWhich row does this command refer to?\n"
        + "Answer with one character only (%s).\n" % m["legend"]
        + opts + "\n\n"
        + "Command: %s\nRow:" % command
    )


class SalsaRowScorer:
    """One model, one forward pass per command, logits over 3 class tokens."""

    def __init__(self, base="Qwen/Qwen2.5-3B-Instruct", adapter=None,
                 device=None, dtype=torch.float16):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(base)
        self.model = AutoModelForCausalLM.from_pretrained(
            base, dtype=dtype).to(self.device)
        self.adapter = adapter
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
            self.model = self.model.merge_and_unload()
        self.model.eval()
        self.name = base + ("+lora" if adapter else "")
        self._ids = {}
        for mp, spec in MAPPINGS.items():
            ids = []
            for t in spec["tokens"]:
                # the answer follows "Row:" so the class token is preceded by a
                # space; encode it that way or the id is the wrong variant
                enc = self.tok.encode(" " + t, add_special_tokens=False)
                if len(enc) != 1:
                    enc = self.tok.encode(t, add_special_tokens=False)
                if len(enc) != 1:
                    raise ValueError("%r is not a single token for %s" % (t, base))
                ids.append(enc[0])
            if len(set(ids)) != len(ids):
                raise ValueError("class tokens collide for mapping %s" % mp)
            self._ids[mp] = ids

    def token_ids(self, mapping):
        return list(self._ids[mapping])

    @torch.no_grad()
    def score(self, command, mapping="letters"):
        """Return (pred_row_letter, probs over A/B/C, latency_ms).

        ONE forward pass. The returned distribution is a softmax over ONLY the
        three class-token logits -- every other token in the vocabulary is
        excluded, which is what makes this a classifier rather than a generator.
        """
        prompt = build_prompt(command, mapping)
        enc = self.tok(prompt, return_tensors="pt").to(self.device)
        t0 = time.perf_counter()
        out = self.model(**enc)
        last = out.logits[0, -1, :]
        sel = last[self._ids[mapping]].float()
        probs = torch.softmax(sel, dim=-1)
        ms = (time.perf_counter() - t0) * 1000
        return ROWS[int(probs.argmax())], probs.tolist(), ms

    @torch.no_grad()
    def generate_baseline(self, command, mapping="letters", max_new_tokens=8):
        """Same prompt, but decoded autoregressively -- the P6.3 comparison."""
        prompt = build_prompt(command, mapping)
        enc = self.tok(prompt, return_tensors="pt").to(self.device)
        t0 = time.perf_counter()
        out = self.model.generate(**enc, max_new_tokens=max_new_tokens,
                                  do_sample=False,
                                  pad_token_id=self.tok.eos_token_id)
        ms = (time.perf_counter() - t0) * 1000
        txt = self.tok.decode(out[0][enc["input_ids"].shape[1]:],
                              skip_special_tokens=True).strip()
        pred = None
        for i, t in enumerate(MAPPINGS[mapping]["tokens"]):
            if txt[:1].upper() == t.upper():
                pred = ROWS[i]
                break
        return pred, txt, ms

    def memory_mb(self):
        if self.device == "cuda":
            return torch.cuda.max_memory_allocated() / 1e6
        return float("nan")


def selftest():
    """Prove the mechanism before any evaluation uses it."""
    print("=" * 74)
    print("SALSA row scorer selftest")
    print("=" * 74)
    s = SalsaRowScorer()
    print("  model   %s on %s" % (s.name, s.device))
    for mp in MAPPINGS:
        print("  mapping %-8s class token ids %s" % (mp, s.token_ids(mp)))
    probes = [
        ("put the bottle in the upper left corner of the table", "A"),
        ("put the bottle in the bottom left corner of the table", "C"),
        ("put the bottle in the far left corner of the table", "A"),
        ("put the bottle in the near left corner of the table", "C"),
        ("put the bottle in the middle of the table", "B"),
    ]
    for mp in MAPPINGS:
        print("\n  --- mapping=%s ---" % mp)
        for cmd, want in probes:
            pred, probs, ms = s.score(cmd, mp)
            print("    %-58s want %s got %s  p=%s  %.0f ms"
                  % (cmd[:58], want, pred,
                     "[" + " ".join("%.2f" % p for p in probs) + "]", ms))
    print("=" * 74)


if __name__ == "__main__":
    selftest()
