"""Original legacy training bodies, with evaluation removed and artifacts returned.

Copied from a1_arms.py, a1_rest.py and p5_baselines.py without changing
training, seeds, dev selection or tokenization. See test_stage0.py for an AST
comparison of every training body against its original source.
"""
import random
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import common as C
sys.path.insert(0, str(C.REPO / "thesis2/scripts"))
from a1_arms import (TinyFactorized, BiLSTMJoint, set_seed, SEED, PLATEAU_TOL,
                     V, T, L, SPACE, IGNORE, FACTORS, factorize, reconstruct, emit)


def load(name):
    if name not in (C.TRAIN, C.DEV):
        raise RuntimeError("Legacy training may read train/dev only")
    return C.load_split(name)


def run_frozen_probe():
    """MiniLM FROZEN, three linear heads on the pooled embedding."""
    from sentence_transformers import SentenceTransformer
    set_seed(SEED)
    enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    for p in enc.parameters():
        p.requires_grad = False
    tr, dv = load("train_v2"), load("dev_v2")

    def embed(rows):
        return torch.tensor(enc.encode([r["command"] for r in rows],
                                       normalize_embeddings=True,
                                       show_progress_bar=False))
    Xtr, Xdv = embed(tr), embed(dv)
    h = Xtr.shape[1]
    heads = nn.ModuleDict(dict(verb=nn.Linear(h, len(V)),
                               target=nn.Linear(h, len(T)),
                               location=nn.Linear(h, len(L))))
    ytr = {"verb": torch.tensor([V.index(r["expected"]["action"]) for r in tr]),
           "target": torch.tensor([T.index(r["expected"]["target"])
                                   if r["expected"].get("target") else IGNORE for r in tr]),
           "location": torch.tensor([L.index(r["expected"]["location"])
                                     if r["expected"].get("location") else IGNORE for r in tr])}
    opt = torch.optim.AdamW(heads.parameters(), lr=1e-2, weight_decay=0.01)
    ce = nn.CrossEntropyLoss(ignore_index=IGNORE)
    best, hist = None, []
    for ep in range(1, 201):
        heads.train(); opt.zero_grad()
        loss = sum(ce(heads[k](Xtr), ytr[k]) for k in heads)
        loss.backward(); opt.step()
        if ep % 10:
            continue
        heads.eval()
        with torch.no_grad():
            hit = 0
            for i, r in enumerate(dv):
                o = emit(int(heads["verb"](Xdv[i:i+1]).argmax()),
                         int(heads["target"](Xdv[i:i+1]).argmax()),
                         L[int(heads["location"](Xdv[i:i+1]).argmax())], SPACE)
                t = dict(r["expected"]); t.setdefault("modifiers", [])
                hit += int(o == t)
        hist.append((ep, hit))
        if best is None or hit > best[1]:
            best = (ep, hit, {k: {n: p.clone() for n, p in heads[k].state_dict().items()}
                              for k in heads})
    bh = max(h2 for _e, h2 in hist)
    chosen = min([(e, h2) for e, h2 in hist if h2 >= bh - PLATEAU_TOL])
    print("  dev plateau %d/%d, plateau-earliest epoch %d; LOADED argmax epoch %d"
          % (bh, len(dv), chosen[0], best[0]))
    for k in heads:
        heads[k].load_state_dict(best[2][k])
    heads.eval()
    return dict(encoder=enc, heads=heads, selected_epoch=best[0], dev_hits=best[1], history=hist)


def run_tinybert():
    from transformers import AutoTokenizer
    set_seed(SEED)
    base = "huawei-noah/TinyBERT_General_4L_312D"
    tok = AutoTokenizer.from_pretrained(base)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr, dv = load("train_v2"), load("dev_v2")

    def enc_rows(rows):
        e = tok([r["command"] for r in rows], truncation=True, max_length=64,
                padding="max_length", return_tensors="pt")
        lab = {"verb": torch.tensor([V.index(r["expected"]["action"]) for r in rows]),
               "target": torch.tensor([T.index(r["expected"]["target"])
                                       if r["expected"].get("target") else IGNORE
                                       for r in rows])}
        fs = [factorize(r["expected"].get("location")) for r in rows]
        for i, f in enumerate(FACTORS):
            lab[f] = torch.tensor([x[i] for x in fs])
        return e, lab
    etr, ltr = enc_rows(tr)
    m = TinyFactorized(base).to(dev)
    print("  TinyBERT parameters: %.2f M" % (sum(p.numel() for p in m.parameters()) / 1e6))
    opt = torch.optim.AdamW(m.parameters(), lr=5e-5, weight_decay=0.01)
    ce = nn.CrossEntropyLoss(ignore_index=IGNORE)
    n, bs, EP = len(tr), 16, 30
    hist, best = [], None
    for ep in range(1, EP + 1):
        m.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            lg = m(etr["input_ids"][idx].to(dev), etr["attention_mask"][idx].to(dev))
            loss = 0
            for k in lg:
                lab = ltr[k][idx].to(dev)
                if (lab != IGNORE).any():
                    loss = loss + ce(lg[k], lab)
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval(); hit = 0
        with torch.no_grad():
            for r in dv:
                e = tok(r["command"], truncation=True, max_length=64,
                        padding="max_length", return_tensors="pt").to(dev)
                lg = m(e["input_ids"], e["attention_mask"])
                ix = {k: int(lg[k].argmax(-1)) for k in lg}
                o = emit(ix["verb"], ix["target"],
                         reconstruct(ix["loc_type"], ix["row"], ix["col"],
                                     ix["container_which"], ix["runtime_which"]), SPACE)
                t = dict(r["expected"]); t.setdefault("modifiers", [])
                hit += int(o == t)
        hist.append((ep, hit))
        if best is None or hit > best[1]:
            best = (ep, hit, {k: v.clone() for k, v in m.state_dict().items()})
    bh = max(h for _e, h in hist)
    chosen = min([(e, h) for e, h in hist if h >= bh - PLATEAU_TOL])
    print("  dev plateau %d/%d, plateau-earliest epoch %d; LOADED argmax epoch %d"
          % (bh, len(dv), chosen[0], best[0]))
    m.load_state_dict(best[2]); m.eval()
    return dict(model=m, tokenizer=tok, selected_epoch=best[0], dev_hits=best[1], history=hist)


def run_bilstm():
    set_seed(SEED)
    tr, dv = load("train_v2"), load("dev_v2")
    # vocabulary from TRAIN ONLY -- an OOV word at eval maps to <unk>
    vocab = {"<pad>": 0, "<unk>": 1}
    for r in tr:
        for w in r["command"].lower().split():
            vocab.setdefault(w, len(vocab))
    print("  vocabulary built from train_v2 only: %d types" % len(vocab))
    MAXL = 24

    def ids_of(cmd):
        t = [vocab.get(w, 1) for w in cmd.lower().split()][:MAXL]
        return t + [0] * (MAXL - len(t))
    Xtr = torch.tensor([ids_of(r["command"]) for r in tr])
    ytr = {"verb": torch.tensor([V.index(r["expected"]["action"]) for r in tr]),
           "target": torch.tensor([T.index(r["expected"]["target"])
                                   if r["expected"].get("target") else IGNORE for r in tr]),
           "location": torch.tensor([L.index(r["expected"]["location"])
                                     if r["expected"].get("location") else IGNORE for r in tr])}
    m = BiLSTMJoint(len(vocab))
    print("  BiLSTM parameters: %.2f M" % (sum(p.numel() for p in m.parameters()) / 1e6))
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss(ignore_index=IGNORE)
    hist, best, n, bs = [], None, len(tr), 16
    for ep in range(1, 61):
        m.train(); perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            lg = m(Xtr[idx])
            loss = sum(ce(lg[k], ytr[k][idx]) for k in lg
                       if (ytr[k][idx] != IGNORE).any())
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval(); hit = 0
        with torch.no_grad():
            for r in dv:
                lg = m(torch.tensor([ids_of(r["command"])]))
                o = emit(int(lg["verb"].argmax()), int(lg["target"].argmax()),
                         L[int(lg["location"].argmax())], SPACE)
                t = dict(r["expected"]); t.setdefault("modifiers", [])
                hit += int(o == t)
        hist.append((ep, hit))
        if best is None or hit > best[1]:
            best = (ep, hit, {k: v.clone() for k, v in m.state_dict().items()})
    bh = max(h for _e, h in hist)
    chosen = min([(e, h) for e, h in hist if h >= bh - PLATEAU_TOL])
    print("  dev plateau %d/%d, plateau-earliest epoch %d; LOADED argmax epoch %d"
          % (bh, len(dv), chosen[0], best[0]))
    m.load_state_dict(best[2]); m.eval()
    return dict(model=m, vocab=vocab, selected_epoch=best[0], dev_hits=best[1], history=hist)


def run_flant5():
    from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    base = "google/flan-t5-base"
    tok = AutoTokenizer.from_pretrained(base)
    m = AutoModelForSeq2SeqLM.from_pretrained(base).to(dev)
    print("  flan-t5-base parameters: %.2f M"
          % (sum(p.numel() for p in m.parameters()) / 1e6))
    tr, dv = load("train_v2"), load("dev_v2")

    # BRACE-FREE TARGET FORMAT. T5's SentencePiece vocabulary cannot represent
    # "{" or "}" -- both encode to [3, 2] and decode to the empty string
    # (verified in a1_flant5_diag.py). Training it to emit JSON braces asks for
    # characters it has no tokens for, which is what produced 0.0000 schema
    # validity on the first attempt. The object is emitted as key=value pairs
    # and reconstructed afterwards; the schema check is applied to the
    # reconstructed object exactly as for every other arm.
    def tgt(exp):
        parts = ["action=%s" % exp["action"]]
        if exp.get("target"):
            parts.append("target=%s" % exp["target"])
        if exp.get("location"):
            parts.append("location=%s" % exp["location"])
        return " | ".join(parts)

    def parse_kv(txt):
        o = {}
        for piece in txt.split("|"):
            piece = piece.strip()
            if "=" not in piece:
                continue
            k, v = piece.split("=", 1)
            k, v = k.strip(), v.strip()
            if k in ("action", "target", "location") and v:
                o[k] = v
        return o

    PRE = "translate command to action json: "
    Xtr = tok([PRE + r["command"] for r in tr], padding=True, truncation=True,
              max_length=64, return_tensors="pt")
    Ytr = tok([tgt(r["expected"]) for r in tr], padding=True, truncation=True,
              max_length=48, return_tensors="pt")
    labels = Ytr["input_ids"].clone()
    labels[labels == tok.pad_token_id] = -100
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.01)
    n, bs, EP = len(tr), 8, 12
    best, hist = None, []
    for ep in range(1, EP + 1):
        m.train(); perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            loss = m(input_ids=Xtr["input_ids"][idx].to(dev),
                     attention_mask=Xtr["attention_mask"][idx].to(dev),
                     labels=labels[idx].to(dev)).loss
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval(); hit = 0
        with torch.no_grad():
            for r in dv:
                e = tok(PRE + r["command"], return_tensors="pt").to(dev)
                o = m.generate(**e, max_new_tokens=48, do_sample=False)
                txt = tok.decode(o[0], skip_special_tokens=True)
                p = parse_kv(txt)
                t = dict(r["expected"]); t.setdefault("modifiers", [])
                p.setdefault("modifiers", [])
                hit += int(p == t)
        hist.append((ep, hit))
        if best is None or hit > best[1]:
            best = (ep, hit, {k: v.detach().clone() for k, v in m.state_dict().items()})
        print("    epoch %2d dev %2d/%d" % (ep, hit, len(dv)))
    bh = max(h for _e, h in hist)
    chosen = min([(e, h) for e, h in hist if h >= bh - 1])
    print("  dev plateau %d/%d, plateau-earliest epoch %d; LOADED argmax epoch %d"
          % (bh, len(dv), chosen[0], best[0]))   # Part B S-7: argmax is what runs
    m.load_state_dict(best[2]); m.eval()
    return dict(model=m, tokenizer=tok, selected_epoch=best[0], dev_hits=best[1], history=hist)


def build_knn():
    TRAIN = load("train_v2")
    from sentence_transformers import SentenceTransformer
    import numpy as np
    m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    corpus = [r["command"] for r in TRAIN]
    E = m.encode(corpus, normalize_embeddings=True, show_progress_bar=False)
    return dict(encoder=m, embeddings=E, train=TRAIN)


def build_tfidf():
    TRAIN = load("train_v2")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
    X = vec.fit_transform([r["command"] for r in TRAIN])
    heads = {}
    for name, key, space in (("verb", "action", V), ("target", "target", T),
                             ("location", "location", L)):
        idx = [i for i, r in enumerate(TRAIN) if r["expected"].get(key)]
        y = [space.index(TRAIN[i]["expected"][key]) for i in idx]
        clf = LogisticRegression(max_iter=2000, C=4.0)
        clf.fit(X[idx], y)
        heads[name] = clf
    return dict(vectorizer=vec, heads=heads)


