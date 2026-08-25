"""llm/retrieve.py -- Rung 2: retrieve few-shot examples from the training split.

WHAT THIS IS
------------
Few-shot (rung 1) uses 3 FIXED exemplars, the same three for every command.
RAG (rung 2) uses 3 exemplars chosen PER COMMAND by semantic similarity.

Same k. Same prompt builder. The only thing that changes is how the 3 examples
are selected -- so any accuracy difference is attributable to retrieval alone.

CORPUS FIREWALL
---------------
The corpus is data/train_v2.jsonl and nothing else. Not seen_v2 (which contains
the dev rows), not held_out_v3 (which is the KPI). This is enforced here, in one
constant, so it cannot drift.

LEAVE-ONE-OUT
-------------
When a query is itself a corpus row -- which happens whenever you retrieve
against seen_v2 -- returning that row hands the model the answer. `exclude_exact`
drops any corpus entry whose command matches the query. It is ON by default.

NOTE: this only removes the EXACT self-match. The dataset is templated across
three objects, so "retrieve the bottle" will still retrieve "retrieve the toy
car", which gives away nearly everything. Leave-one-out makes seen numbers less
absurd; it does not make them meaningful. Report RAG on held-out.

DEVICE
------
Pinned to CPU. The env's torch is CPU-only anyway, but the pin is explicit so
that retrieval can never silently start competing with Ollama for VRAM, and so
that the retrieval stage in the latency decomposition is honestly a CPU stage.

FIRST RUN DOWNLOADS THE MODEL
-----------------------------
~90 MB from HuggingFace. If HF_HUB_OFFLINE is set (the Whisper setup sets it in
code), the download fails. This module unsets it for its own process on the
first load and tells you what it did.

USAGE
-----
    python -m llm.retrieve --smoke
    python -m llm.retrieve --query "put the bottle in the far left corner"
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ---- frozen constants: change these and the thesis claim changes -------------
CORPUS_PATH = "data/train_v2.jsonl"     # firewalled. never seen_v2, never held_out.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_K = 3                            # matches few-shot's k. do not diverge.
DEVICE = "cpu"


@dataclass
class Retrieved:
    """One retrieval result set, plus the timing the harness records."""
    examples: list[dict]                 # seen.jsonl-shaped rows, ready for build_prompt
    ids: list[int] = field(default_factory=list)      # corpus indices, for the CSV
    scores: list[float] = field(default_factory=list) # cosine similarity
    latency_s: float = 0.0               # encode + search, excludes model load


class Retriever:
    """Encodes the corpus once, then answers queries by cosine similarity.

    Build it ONCE per run and reuse it. Constructing it per command would put a
    2-3 second model load inside every measured retrieval and destroy the
    latency numbers.
    """

    def __init__(self, corpus_path: str = CORPUS_PATH, model_name: str = MODEL_NAME,
                 device: str = DEVICE, quiet: bool = False):
        self.corpus_path = Path(corpus_path)
        self.model_name = model_name
        self.device = device

        if not self.corpus_path.exists():
            raise FileNotFoundError(
                f"corpus not found: {self.corpus_path}\n"
                f"Run make_split.py first -- RAG's corpus is the TRAIN split, "
                f"never seen_v2.jsonl."
            )

        self.rows = [json.loads(l) for l in self.corpus_path.open(encoding="utf-8") if l.strip()]
        if not self.rows:
            raise ValueError(f"{self.corpus_path} is empty")
        self.commands = [r["command"] for r in self.rows]

        self.model = self._load_model(quiet=quiet)

        t0 = time.perf_counter()
        self.corpus_emb = self.model.encode(
            self.commands,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,   # unit vectors -> dot product IS cosine
            show_progress_bar=False,
            device=self.device,
        )
        self.encode_corpus_s = time.perf_counter() - t0

        if not quiet:
            print(f"[retriever] corpus   : {self.corpus_path}  ({len(self.rows)} rows)")
            print(f"[retriever] model    : {self.model_name}  on {self.device}")
            print(f"[retriever] dim      : {self.corpus_emb.shape[1]}")
            print(f"[retriever] encoded  : {self.encode_corpus_s:.2f}s (one-time, "
                  f"NOT part of per-command latency)")

    # -- model loading, with the offline-flag trap handled loudly ----------
    def _load_model(self, quiet: bool = False):
        from sentence_transformers import SentenceTransformer

        offline = os.environ.get("HF_HUB_OFFLINE")
        try:
            return SentenceTransformer(self.model_name, device=self.device)
        except Exception as e:
            if offline:
                if not quiet:
                    print(f"[retriever] HF_HUB_OFFLINE={offline} blocked the download; "
                          f"retrying online for this process only.")
                os.environ.pop("HF_HUB_OFFLINE", None)
                try:
                    m = SentenceTransformer(self.model_name, device=self.device)
                finally:
                    os.environ["HF_HUB_OFFLINE"] = offline
                return m
            raise RuntimeError(
                f"could not load {self.model_name}: {e}\n"
                f"First run needs network access to download ~90 MB."
            ) from e

    # -- the actual retrieval ----------------------------------------------
    def retrieve(self, query: str, k: int = DEFAULT_K,
                 exclude_exact: bool = True) -> Retrieved:
        """Return the k most similar corpus rows to `query`.

        exclude_exact drops any corpus row whose command string equals the query.
        Keep it True. See the leave-one-out note at the top of this file.
        """
        t0 = time.perf_counter()

        q = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            device=self.device,
        )[0]

        sims = self.corpus_emb @ q          # both unit-norm -> cosine similarity

        if exclude_exact:
            qn = query.strip().lower()
            for i, c in enumerate(self.commands):
                if c.strip().lower() == qn:
                    sims[i] = -np.inf

        k = min(k, int(np.isfinite(sims).sum()))
        if k <= 0:
            return Retrieved(examples=[], ids=[], scores=[],
                             latency_s=time.perf_counter() - t0)

        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]   # highest similarity first

        return Retrieved(
            examples=[self.rows[i] for i in top],
            ids=[int(i) for i in top],
            scores=[float(sims[i]) for i in top],
            latency_s=time.perf_counter() - t0,
        )


# ---------------------------------------------------------------- CLI --------

def _smoke():
    """Ten queries chosen to expose the failure modes, not to flatter the model."""
    r = Retriever()
    probes = [
        # short stop commands -- the case I expect to break
        ("cease",                                   "stop"),
        ("wait, hold on a second",                  "stop"),
        # paraphrase, the reason we chose embeddings over BM25
        ("escort the toy car over to me",           "fetch"),
        ("would you mind bringing the bottle over", "fetch"),
        # hold vs fetch, semantically adjacent
        ("latch onto the bottle",                   "hold"),
        # reject -- can retrieval surface a reject example at all?
        ("make me a coffee",                        "reject"),
        ("put the bottle in a spot past the right edge", "reject"),
        # location commands, incl. the collision phrasing
        ("put the bottle in the left side middle edge", "pick_place"),
        ("move to the far left corner",             "move_to"),
        # a template row -- leave-one-out will not save this one
        ("retrieve the bottle",                     "fetch"),
    ]
    print()
    n_right = 0
    for q, expect in probes:
        res = r.retrieve(q)
        verbs = [e["expected"]["action"] for e in res.examples]
        hit = expect in verbs
        n_right += hit
        print(f"{'OK ' if hit else 'MISS'}  {q!r}   want={expect}")
        for e, s, i in zip(res.examples, res.scores, res.ids):
            print(f"        {s:.3f}  [{i:>3}] {e['expected']['action']:<10} {e['command']}")
        print(f"        retrieval {res.latency_s*1000:.1f} ms")
        print()
    print(f"correct verb present in top-{DEFAULT_K}: {n_right}/{len(probes)}")
    print("\nThis is a sanity check, not a result. Read the retrieved rows, not the score.")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="run the 10 probe queries")
    ap.add_argument("--query", help="retrieve for one command")
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--corpus", default=CORPUS_PATH)
    args = ap.parse_args()

    if args.smoke:
        _smoke()
        return
    if not args.query:
        ap.error("give --smoke or --query")

    r = Retriever(corpus_path=args.corpus)
    res = r.retrieve(args.query, k=args.k)
    print()
    for e, s, i in zip(res.examples, res.scores, res.ids):
        print(f"  {s:.3f}  [{i:>3}] {e['expected']['action']:<10} {e['command']}")
        print(f"         -> {json.dumps(e['expected'])}")
    print(f"\n  retrieval {res.latency_s*1000:.1f} ms")


if __name__ == "__main__":
    main()
