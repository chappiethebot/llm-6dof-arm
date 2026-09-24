"""Persist six original legacy recipes; reproduce v3 once, freeze only at 217/217.

Run by the user, one --arm at a time in thesis2gpu. Training and dev selection
finish and the checkpoint hash is recorded BEFORE held_out_v3 is opened.
held_out_v4c is never read. A mismatch is retained as 'retrained, not reproduced';
it never triggers another training attempt or checkpoint selection.
"""
import argparse
import csv
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
from s0_freeze import dir_sha

RECIPES = dict(tinybert="run_tinybert", minilm_probe="run_frozen_probe",
               bilstm="run_bilstm", flant5="run_flant5", knn_minilm="build_knn",
               tfidf_logreg="build_tfidf")


def checkpoint(arm):
    return C.MODELS / ("legacy_" + arm)


def record_path(arm):
    return C.OUT / ("selection_%s.json" % arm)


def persist(arm, artifacts):
    import torch
    dest = checkpoint(arm)
    dest.mkdir(parents=True, exist_ok=False)
    if "encoder" in artifacts:
        artifacts["encoder"].save(str(dest / "encoder"))
    if arm == "flant5":
        artifacts["model"].save_pretrained(str(dest / "model"))
    elif "model" in artifacts:
        torch.save(artifacts["model"].state_dict(), dest / "model.pt")
    if "tokenizer" in artifacts:
        artifacts["tokenizer"].save_pretrained(str(dest / "tokenizer"))
    if arm == "minilm_probe":
        torch.save(artifacts["heads"].state_dict(), dest / "heads.pt")
    if arm == "bilstm":
        (dest / "vocab.json").write_text(json.dumps(artifacts["vocab"]), encoding="utf-8")
    if arm == "knn_minilm":
        import numpy as np
        np.save(dest / "embeddings.npy", artifacts["embeddings"])
        (dest / "train.json").write_text(json.dumps(artifacts["train"]), encoding="utf-8")
    if arm == "tfidf_logreg":
        with (dest / "classifier.pkl").open("wb") as fh:
            pickle.dump((artifacts["vectorizer"], artifacts["heads"]), fh)
    return dest


def load_predict(arm):
    """Inference only: rebuild the persisted selected state, never train here."""
    import numpy as np
    import torch
    from torch import nn
    from legacy_recipes import (TinyFactorized, BiLSTMJoint, V, T, L, SPACE,
                                emit, reconstruct, FACTORS)
    dest = checkpoint(arm)
    if not dest.exists():
        raise SystemExit("Run s0_retrain_legacy.py --arm %s first." % arm)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = tok = encoder = heads = vocab = embeddings = train = vec = None
    if arm in ("minilm_probe", "knn_minilm"):
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(str(dest / "encoder"), device="cpu")
        if arm == "minilm_probe":
            heads = nn.ModuleDict(dict(verb=nn.Linear(encoder.get_sentence_embedding_dimension(), len(V)),
                                       target=nn.Linear(encoder.get_sentence_embedding_dimension(), len(T)),
                                       location=nn.Linear(encoder.get_sentence_embedding_dimension(), len(L))))
            heads.load_state_dict(torch.load(dest / "heads.pt", map_location="cpu", weights_only=True))
            heads.eval()
        else:
            embeddings = np.load(dest / "embeddings.npy")
            train = json.loads((dest / "train.json").read_text(encoding="utf-8"))
    elif arm == "tfidf_logreg":
        # Trusted local artifact produced above, never load an external pickle.
        with (dest / "classifier.pkl").open("rb") as fh:
            vec, heads = pickle.load(fh)
    elif arm == "bilstm":
        vocab = json.loads((dest / "vocab.json").read_text(encoding="utf-8"))
        model = BiLSTMJoint(len(vocab))
        model.load_state_dict(torch.load(dest / "model.pt", map_location="cpu", weights_only=True))
        model.eval()
    else:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        tok = AutoTokenizer.from_pretrained(str(dest / "tokenizer"))
        if arm == "tinybert":
            model = TinyFactorized()
            model.load_state_dict(torch.load(dest / "model.pt", map_location="cpu", weights_only=True))
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(str(dest / "model"))
        model.to(device).eval()

    def predict(command):
        start = time.perf_counter()
        info, probs = {}, None
        with torch.no_grad():
            if arm == "knn_minilm":
                q = encoder.encode([command], normalize_embeddings=True, show_progress_bar=False)[0]
                exp = train[int(np.argmax(embeddings @ q))]["expected"]
                obj = emit(V.index(exp["action"]), T.index(exp["target"]) if exp.get("target") else 0,
                           exp.get("location") or L[0], SPACE)
            elif arm == "tfidf_logreg":
                x = vec.transform([command])
                obj = emit(int(heads["verb"].predict(x)[0]), int(heads["target"].predict(x)[0]),
                           L[int(heads["location"].predict(x)[0])], SPACE)
                probs = {k: h.predict_proba(x)[0] for k, h in heads.items()}
            elif arm == "flant5":
                x = tok("translate command to action json: " + command, return_tensors="pt").to(device)
                ids = model.generate(**x, max_new_tokens=48, do_sample=False)
                info["raw"] = tok.decode(ids[0], skip_special_tokens=True)
                obj = {}
                for piece in info["raw"].split("|"):
                    if "=" in piece:
                        k, v = (s.strip() for s in piece.split("=", 1))
                        if k in ("action", "target", "location") and v:
                            obj[k] = v
                obj.setdefault("modifiers", [])
            else:
                if arm == "tinybert":
                    x = tok(command, truncation=True, max_length=64, padding="max_length",
                            return_tensors="pt").to(device)
                    logits = model(x["input_ids"], x["attention_mask"])
                elif arm == "bilstm":
                    ids = [vocab.get(w, 1) for w in command.lower().split()][:24]
                    logits = model(torch.tensor([ids + [0] * (24 - len(ids))]))
                else:
                    x = torch.tensor(encoder.encode([command], normalize_embeddings=True,
                                                    show_progress_bar=False))
                    logits = {k: h(x) for k, h in heads.items()}
                probs = {k: torch.softmax(v, -1)[0] for k, v in logits.items()}
                ix = {k: int(p.argmax()) for k, p in probs.items()}
                loc = (reconstruct(*(ix[k] for k in FACTORS)) if arm == "tinybert"
                       else L[ix["location"]])
                obj = emit(ix["verb"], ix["target"], loc, SPACE)
        if probs:
            info.update({"p_" + k: float(p.max()) for k, p in probs.items()})
            if arm == "tinybert":
                info["p_location"] = min(info["p_" + k] for k in FACTORS)
        info["latency_ms"] = (time.perf_counter() - start) * 1000
        return obj, info
    return predict


def compare(saved, predictions):
    """Missing or duplicated command coverage must not pass an intersection check."""
    expected = {str(i) for i in range(217)}
    if set(saved or {}) != expected or set(predictions) != expected:
        raise ValueError("Reproducibility requires exactly command_ids 0..216 in both sets")
    return [str(i) for i in range(217) if saved[str(i)] != predictions[str(i)]]


def freeze_reproduced(arm, rec):
    if rec["status"] != "reproduced" or rec["identical"] != 217:
        raise ValueError("Only 217/217 reproduced legacy arms may freeze")
    dest = checkpoint(arm)
    digest = dir_sha(dest)
    if digest != rec["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed after dev selection")
    rows = C.read_frozen()
    rows[arm] = dict(arm=arm, kind="retrained legacy, reproduced 217/217", path=str(dest),
                     sha256=digest, selected_epoch=rec["selected_epoch"],
                     dev_action_match=rec["dev_action_match"],
                     frozen_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    with C.FROZEN.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=C.FROZEN_COLS)
        writer.writeheader()
        writer.writerows(rows[k] for k in sorted(rows))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True, choices=sorted(RECIPES))
    args = ap.parse_args()
    from s0_grpo_signal_check import require_env, GPU_PYTHON
    require_env(GPU_PYTHON)
    C.start_log("s0_retrain_legacy_%s" % args.arm)
    # Rule 3, at startup: the legacy recipes build every lexicon from train_v2
    # (the BiLSTM's vocabulary, the TF-IDF vocabulary, the kNN index). Nothing
    # may be added from a held-out split. `top` occurs once in train_v2, so the
    # guard distinguishes inherited from added -- see common.assert_vocab_from_train_only.
    C.assert_vocab_from_train_only(C.train_vocab_words(),
                                   where="the train_v2 vocabulary the legacy recipes build from")
    arm = args.arm
    dest, record = checkpoint(arm), record_path(arm)
    if record.exists():
        rec = json.loads(record.read_text(encoding="utf-8"))
        if dir_sha(dest) != rec["checkpoint_sha256"]:
            raise SystemExit("STOP: persisted legacy checkpoint has changed")
        if rec.get("status") == "reproduced":
            freeze_reproduced(arm, rec)
            C.gate(True, ["[Certain] Existing 217/217 comparison preserved; no retraining or inference."])
        if rec.get("status") == "retrained, not reproduced":
            C.gate(False, ["[Certain] Previous mismatch preserved: %d/217 differ; no retraining."
                           % len(rec["differing_command_ids"])])
        raise SystemExit("STOP: partial legacy run exists; review before any repeat.")
    if dest.exists():
        raise SystemExit("STOP: checkpoint without a completed selection record; review partial run.")
    import gc
    import torch
    import legacy_recipes as recipes
    artifacts = getattr(recipes, RECIPES[arm])()
    persist(arm, artifacts)
    rec = dict(arm=arm, selected_epoch=artifacts.get("selected_epoch", "n/a: no epochs"),
               seed=recipes.SEED if arm not in ("knn_minilm", "tfidf_logreg") else "original recipe: no seed",
               history=artifacts.get("history", []), checkpoint=str(dest), smoke=False,
               rule="original argmax dev selection" if "selected_epoch" in artifacts else "fixed original recipe",
               checkpoint_sha256=dir_sha(dest), status="selected before v3 comparison")
    del artifacts
    gc.collect()
    torch.cuda.empty_cache()
    predict = load_predict(arm)
    dev = C.load_split(C.DEV)
    rec["dev_action_match"] = sum(C.score(predict(r["command"])[0], C.truth_of(r)).strict_match
                                   for r in dev) / len(dev)
    record.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    # First held-out read: the chosen weights and their hash are already fixed.
    from s1_run_arm import saved_predictions, canon
    saved = saved_predictions(arm, "held_out_v3")
    rows, predictions = [], {}
    for i, row in enumerate(C.load_split("held_out_v3")):
        obj, info = predict(row["command"])
        predictions[str(i)] = canon(obj)
        rows.append(C.score_row(arm, "held_out_v3", i, 0, row["command"],
                                info.get("raw", json.dumps(obj)), obj, C.truth_of(row),
                                info["latency_ms"], note=json.dumps(info)))
    differing = compare(saved, predictions)
    rec.update(identical=217 - len(differing), differing_command_ids=differing,
               status="retrained, not reproduced" if differing else "reproduced")
    record.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    if differing:
        path = C.OUT / ("retrained_%s_held_out_v3.csv" % arm)
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=C.PRED_COLS)
            writer.writeheader()
            writer.writerows(rows)
    else:
        freeze_reproduced(arm, rec)
        C.write_preds(arm, "held_out_v3", rows)
    C.gate(not differing, ["[Certain] %s: %d/217 identical; %s"
                           % (arm, rec["identical"], rec["status"]),
                           "[Certain] Differing command_ids: %s" % differing,
                           "[Certain] No held_out_v4c data was read."])


if __name__ == "__main__":
    main()
