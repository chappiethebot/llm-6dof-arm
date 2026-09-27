"""thesis2/final/arms.py -- the ONE arm registry.

Every arm in the brief's registry has an entry here. An entry is:

    name -> Arm(kind, determinism, builder, note)

  kind          "local" or "cloud" -- decides R (read from the V5 census) and
                whether energy is measurable.
  determinism   "deterministic"  : all R repeats must be byte-identical
                "sampling"       : repeats collapse by majority vote (A9's rule)
  builder       callable() -> predictor. A predictor is
                    predict(command) -> (obj, info)
                where obj is built by the SHARED emit() and info carries
                latency_ms plus whatever the arm can expose (class probabilities,
                acceptance, first-rejection field).
  note          the label the report must carry for this arm.

Builders are lazy: importing this module loads no model. s1_run_arm.py builds
exactly the arm it was asked for.

Withdrawn arms (A1.7 field-level FLMD, every SALSA digits mapping) are NOT
registered, by rule 7.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C                                                      # noqa: E402


class Arm:
    def __init__(self, name, kind, determinism, builder, note="", probs=False,
                 speculative=False, blocked=None):
        self.name = name
        self.kind = kind
        self.determinism = determinism
        self.builder = builder
        self.note = note
        self.probs = probs                  # exposes class probabilities -> ECE
        self.speculative = speculative      # exposes token acceptance
        self.blocked = blocked              # reason string, or None

    def build(self):
        if self.blocked:
            print("GATE FAIL: arm %r is blocked: %s" % (self.name, self.blocked))
            raise SystemExit(2)
        return self.builder()


REGISTRY: dict[str, Arm] = {}


def register(name, kind, determinism, builder, note="", probs=False,
             speculative=False, blocked=None):
    REGISTRY[name] = Arm(name, kind, determinism, builder, note, probs, speculative, blocked)


# ======================================================================= A.
# Encoder classifiers. All reuse the shared emit() and label space.
def _flat(device):
    import torch
    from ft.train_classifier import MultiHeadClassifier
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    m = MultiHeadClassifier(C.SPACE)
    m.load_state_dict(torch.load(
        C.REPO / "ft" / "runs" / "distilbert_cls" / "epoch_09" / "model.pt",
        map_location="cpu"))
    m.eval().to(device)

    def predict(cmd):
        import time
        t0 = time.perf_counter()
        enc = tok(cmd, truncation=True, max_length=C.MAX_LEN, return_tensors="pt").to(device)
        with torch.no_grad():
            lv, lt, ll = m(enc["input_ids"], enc["attention_mask"])
        obj = C.emit(int(lv.argmax()), int(lt.argmax()), int(ll.argmax()), C.SPACE)
        p = torch.softmax(ll, -1)[0]
        return obj, dict(latency_ms=(time.perf_counter() - t0) * 1000,
                         p_location=float(p.max()),
                         p_verb=float(torch.softmax(lv, -1).max()),
                         probs_location=[round(float(x), 6) for x in p])
    return predict


def _factorized(int4=False):
    import torch
    sys.path.insert(0, str(C.REPO / "thesis2" / "scripts"))
    from p2_train_factorized import (FactorizedClassifier, FACTORS, build_label_space,
                                     reconstruct, emit as fact_emit)
    from transformers import AutoTokenizer
    space = build_label_space()
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    m = FactorizedClassifier(space)
    m.load_state_dict(torch.load(
        C.REPO / "thesis2" / "models" / "distilbert_factorized" / "epoch_11" / "model.pt",
        map_location="cpu"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m.eval().to(dev)
    if int4:
        if dev == "cuda":
            m.half()
        from p3_quantization_gate import PrecisionSwitch
        sw = PrecisionSwitch(m)
        sw.use("int4")

    def predict(cmd):
        import time
        t0 = time.perf_counter()
        enc = tok(cmd, truncation=True, max_length=C.MAX_LEN, padding="max_length",
                  return_tensors="pt").to(dev)
        with torch.no_grad():
            lg = m(enc["input_ids"], enc["attention_mask"])
        idx = {k: int(lg[k].argmax(-1)) for k in ["verb", "target"] + FACTORS}
        sym = reconstruct(idx["loc_type"], idx["row"], idx["col"],
                          idx["container_which"], idx["runtime_which"])
        obj = fact_emit(idx["verb"], idx["target"], sym, space)
        probs = {k: float(torch.softmax(lg[k], -1).max()) for k in FACTORS}
        info = {"p_" + k: v for k, v in probs.items()}       # includes p_row
        info["latency_ms"] = (time.perf_counter() - t0) * 1000
        # the cascade's gate signal is the ROW confidence; p_location is the
        # weakest factor, which is what an ECE over "the location" should use
        info["p_location"] = min(probs.values())
        return obj, info
    return predict


def _local_pt_arm(ckpt_name, kind):
    """TextCNN / MLP-BoW: rebuild from the frozen .pt (vocabulary travels with it)."""
    import torch
    blob = torch.load(C.MODELS / ckpt_name, map_location="cpu", weights_only=False)
    vocab = blob["vocab"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if kind == "textcnn":
        from s0_train_textcnn import TextCNN, MAXL
        m = TextCNN(len(vocab), len(C.VERBS), len(C.TARGETS), len(C.LOCATIONS))
    else:
        from s0_train_mlpbow import MLPBoW
        m = MLPBoW(len(vocab), len(C.VERBS), len(C.TARGETS), len(C.LOCATIONS))
        MAXL = None
    m.load_state_dict(blob["state_dict"])
    m.eval().to(dev)

    def featurise(cmd):
        if kind == "textcnn":
            t = [vocab.get(w, 1) for w in cmd.lower().split()][:MAXL]
            return torch.tensor([t + [0] * (MAXL - len(t))])
        v = torch.zeros(1, len(vocab))
        for w in cmd.lower().split():
            v[0, vocab.get(w, 1)] = 1.0
        return v

    def predict(cmd):
        import time
        t0 = time.perf_counter()
        with torch.no_grad():
            lv, lt, ll = m(featurise(cmd).to(dev))
        obj = C.emit(int(lv.argmax()), int(lt.argmax()), int(ll.argmax()), C.SPACE)
        p = torch.softmax(ll, -1)[0]
        return obj, dict(latency_ms=(time.perf_counter() - t0) * 1000,
                         p_location=float(p.max()),
                         probs_location=[round(float(x), 6) for x in p])
    return predict


register("flat_gpu", "local", "deterministic", lambda: _flat("cuda"),
         "V5 rung-4 flat DistilBERT head, GPU", probs=True)
register("flat_cpu", "local", "deterministic", lambda: _flat("cpu"),
         "same weights on CPU; CPU ms reported instead of J/command", probs=True)
register("factorized_fp16", "local", "deterministic", lambda: _factorized(False),
         "factorized location head, FP16", probs=True)
register("factorized_int4", "local", "deterministic", lambda: _factorized(True),
         "factorized head, per-channel symmetric INT4 weights", probs=True)


def _hf_arm(path, kind):
    """Load only a persisted legacy arm that reproduced all 217 v3 predictions."""
    C.require_frozen(kind)
    from s0_retrain_legacy import load_predict
    return load_predict(kind)


register("tinybert", "local", "deterministic", lambda: _hf_arm(None, "tinybert"),
         "TinyBERT-4; persisted by s0_retrain_legacy.py", probs=True)
register("minilm_probe", "local", "deterministic", lambda: _hf_arm(None, "minilm_probe"),
         "frozen MiniLM + linear probe; persisted legacy checkpoint", probs=True)
register("knn_minilm", "local", "deterministic", lambda: _hf_arm(None, "knn_minilm"),
         "cosine 1-NN over train_v2 MiniLM embeddings", probs=False)

# ======================================================================= B.
def _rules(firewalled):
    """The two hand-written grammars, imported from thesis2/scripts.

    p5_rules_firewalled.py runs its whole evaluation at MODULE level and rewrites
    thesis2/out/baseline_rules_firewall.csv as a side effect of being imported
    (FINAL_REPORT section 14 records this). That file belongs to the earlier run,
    so it is snapshotted and restored around the import: this runner writes only
    under thesis2/out/final/.
    """
    sys.path.insert(0, str(C.REPO / "thesis2" / "scripts"))
    import io, contextlib
    victim = C.REPO / "thesis2" / "out" / "baseline_rules_firewall.csv"
    before = victim.read_bytes() if victim.exists() else None
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        import p5_rules_firewalled as RF
    if before is not None and victim.read_bytes() != before:
        victim.write_bytes(before)
        print("  restored %s (rewritten by the import; earlier run's file preserved)"
              % victim.name)
    fn = RF.make_predict(RF.ROW_FW if firewalled else RF.ROW_OPEN)

    def predict(cmd):
        import time
        t0 = time.perf_counter()
        obj = fn(cmd)
        return obj, dict(latency_ms=(time.perf_counter() - t0) * 1000)
    return predict


register("rules_open", "local", "deterministic", lambda: _rules(False),
         "VOCABULARY-UNFAIR: the hand-written grammar lists back/front alongside "
         "upper/bottom, so it was given the v4c vocabulary by hand")
register("rules_firewalled", "local", "deterministic", lambda: _rules(True),
         "honest floor: only row words present in train_v2")
register("tfidf_logreg", "local", "deterministic", lambda: _hf_arm(None, "tfidf_logreg"),
         "TF-IDF 1-2gram + logistic regression; persisted legacy checkpoint", probs=True)
register("bilstm", "local", "deterministic", lambda: _hf_arm(None, "bilstm"),
         "BiLSTM from scratch, train_v2 vocabulary; persisted legacy checkpoint", probs=True)
register("textcnn", "local", "deterministic", lambda: _local_pt_arm("textcnn.pt", "textcnn"),
         "NEW: TextCNN, train_v2 vocabulary only", probs=True)
register("mlpbow", "local", "deterministic", lambda: _local_pt_arm("mlpbow.pt", "mlpbow"),
         "NEW: binary bag-of-words MLP, train_v2 vocabulary only", probs=True)

# ======================================================================= C.
register("flant5", "local", "deterministic", lambda: _hf_arm(None, "flant5"),
         "Flan-T5-base seq2seq, brace-free target format; persisted legacy checkpoint")

# ======================================================================= D.
def _lora_greedy():
    import time, torch, json as _json
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    from bench.run_cell import apply_legend
    from llm import prompts as P
    apply_legend("v5")
    schema = P.load_schema(str(C.SCHEMA_FILE))
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct",
                                             dtype=torch.float16).to("cuda")
    m = PeftModel.from_pretrained(
        m, str(C.REPO / "ft" / "runs" / "qwen3b_lora" / "checkpoint-87")).merge_and_unload()
    m.eval()
    IM_END = tok.convert_tokens_to_ids("<|im_end|>")
    STOP = {IM_END, tok.eos_token_id}
    SYS = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

    def predict(cmd):
        t0 = time.perf_counter()
        txt = ("<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n"
               "<|im_start|>assistant\n" % (SYS, P.build_prompt(cmd, "zero_shot", schema)))
        ids = tok(txt, add_special_tokens=False)["input_ids"]
        with torch.no_grad():
            out = m(input_ids=torch.tensor([ids], device="cuda"), use_cache=True)
            past = out.past_key_values
            nxt = int(out.logits[0, -1].argmax())
            gen = []
            while len(gen) < 64 and nxt not in STOP:
                gen.append(nxt)
                out = m(input_ids=torch.tensor([[nxt]], device="cuda"),
                        past_key_values=past, use_cache=True)
                past = out.past_key_values
                nxt = int(out.logits[0, -1].argmax())
        raw = tok.decode(gen, skip_special_tokens=True)
        try:
            obj = _json.loads(raw)
        except Exception:
            obj = {}
        return obj, dict(latency_ms=(time.perf_counter() - t0) * 1000, raw=raw,
                         forward_passes=len(gen) + 1)
    return predict


def _spec(kind):
    """A8 / A10 / A12 -- delegated to the committed Stage-A scripts' logic.

    These arms exist and are verified; the runner re-executes them through their
    own modules rather than re-implementing the decoders here.
    """
    raise SystemExit(
        "GATE FAIL: arm %r must be run through its own script "
        "(thesis2/scripts/a8_speculative.py, a10_prefix_cache.py, "
        "a12_smalllm_drafter.py), which writes its own per-command file. "
        "s1_run_arm.py imports those outputs rather than re-implementing three "
        "verified decoders. See RUN_ORDER.md step S1-spec." % kind)


register("a8_speculative", "local", "deterministic", lambda: _spec("a8_speculative"),
         "classifier-drafted speculative decoding; byte-identical to lora_greedy",
         speculative=True)
register("a10_speculative", "local", "deterministic", lambda: _spec("a10_speculative"),
         "prefix-cached classifier-drafted speculative decoding", speculative=True)
register("a12_smalllm", "local", "deterministic", lambda: _spec("a12_smalllm"),
         "qwen2.5-0.5b drafter, gamma = 4", speculative=True)
register("lora_greedy", "local", "deterministic", _lora_greedy,
         "qwen2.5-3b-lora zero-shot, transformers FP16 merged, greedy")


def _salsa_full(key):
    def build():
        from s0_salsa_full import SalsaFull
        sf = SalsaFull(key)
        return lambda cmd: sf.predict(cmd, want_probs=True)
    return build


for _k in ("qwen2.5-3b-lora", "Qwen2.5-3B-Instruct",
           "qwen2.5-0.5b-lora", "Qwen2.5-0.5B-Instruct"):
    register("salsa_full_%s" % _k, "local", "deterministic", _salsa_full(_k),
             "NEW: SALSA-full, every field by letter logits, letters mapping only",
             probs=True)

def _grpo(name):
    C.require_frozen(name)
    from s0_grpo_signal_check import build_predict
    return build_predict(C.MODELS / name)


register("grpo_base", "local", "deterministic", lambda: _grpo("grpo_base"),
         "NEW: train in thesis2rl; frozen adapter inference in thesis2gpu")
register("grpo_sft", "local", "deterministic", lambda: _grpo("grpo_sft"),
         "NEW: checkpoint-87 GRPO; requires passing the 10% signal gate")

# ======================================================================= F.
def _cascade(target, tau, oracle=False):
    """Per-factor cascade: local factorized head, row escalated when its
    confidence < tau. Reconstruction uses the ESCALATED row and the LOCAL column,
    exactly as P4.5 defines it."""
    import time
    sys.path.insert(0, str(C.REPO / "thesis2" / "scripts"))
    from p2_train_factorized import reconstruct, emit as fact_emit, build_label_space
    space = build_label_space()
    local = _factorized(False)
    ROWS = ["A", "B", "C"]
    esc = None
    if target == "qwen3blora":
        from s0_salsa_full import SalsaFull  # reuse the loader, row question only
        sys.path.insert(0, str(C.REPO / "thesis2" / "scripts"))
        from salsa_row import SalsaRowScorer
        esc = SalsaRowScorer(base="Qwen/Qwen2.5-3B-Instruct",
                             adapter=str(C.REPO / "ft/runs/qwen3b_lora/checkpoint-87"))
    elif target == "cloud":
        from dotenv import load_dotenv
        load_dotenv(C.REPO / ".env")
        from openai import OpenAI
        from salsa_row import build_prompt as row_prompt
        client = C.sdk_no_retries(OpenAI())   # our retries only; see common.py

    def predict(cmd, truth_row=None):
        t0 = time.perf_counter()
        obj, info = local(cmd)
        loc = obj.get("location") or ""
        escalated = False
        cloud = {}          # stage + retries, recorded for the cloud target only
        if target == "cloud":
            cloud = dict(stage="ok", retries=0)
        if loc.startswith("grid_") and info.get("p_row", 1.0) < tau:
            if oracle:
                row = truth_row
            elif target == "qwen3blora":
                row, _p, _ms = esc.score(cmd, "letters")
            else:
                # Same prompt, model and temperature on every attempt.
                msgs = [{"role": "user", "content": row_prompt(cmd, "letters")}]
                try:
                    r, n_retry = C.with_retries(lambda: client.chat.completions.create(
                        model="gpt-4o-mini", temperature=0.0, max_tokens=2,
                        messages=msgs))
                except Exception as e:                    # noqa: BLE001
                    # Retries exhausted, or a non-retryable error. This is NOT a
                    # measured outcome: no prediction is emitted, the row is
                    # marked model_error, and --retry-errors re-runs it.
                    return {}, dict(latency_ms=(time.perf_counter() - t0) * 1000,
                                    stage="model_error",
                                    retries=getattr(e, "retries", 0),
                                    error="%s: %s" % (type(e).__name__, str(e)[:120]),
                                    escalated=False, p_row=info.get("p_row"))
                cloud["retries"] = n_retry
                txt = (r.choices[0].message.content or "").strip().upper()
                row = txt[:1] if txt[:1] in ROWS else None
            if row in ROWS:
                obj = dict(obj, location="grid_%s%s" % (row, loc[6]))
                escalated = True
        return obj, dict(latency_ms=(time.perf_counter() - t0) * 1000, **cloud,
                         escalated=escalated, p_row=info.get("p_row"),
                         p_location=info.get("p_location"))
    return predict


register("cascade_local_tauA", "local", "deterministic",
         lambda: _cascade("qwen3blora", 0.50),
         "pre-registered tau_A = 0.50; row -> qwen2.5-3b-lora SALSA letters", probs=True)
register("pipeline_always_local", "local", "deterministic",
         lambda: _cascade("qwen3blora", 1.01),
         "PIPELINE, NOT ROUTING: tau = 1.0, every grid-typed row delegated", probs=True)
register("cascade_cloud_tauA", "cloud", "sampling", lambda: _cascade("cloud", 0.50),
         "pre-registered tau_A = 0.50; row -> gpt-4o-mini", probs=True)
register("cascade_cloud_always", "cloud", "sampling", lambda: _cascade("cloud", 1.01),
         "PIPELINE, NOT ROUTING: tau = 1.0; row -> gpt-4o-mini", probs=True)
register("cascade_oracle", "local", "deterministic", lambda: _cascade("oracle", 1.01, True),
         "CEILING, accuracy only: the true row is supplied", probs=True)


ENSEMBLE_CELLS = (("classifier:distilbert", "classifier"),
                  ("ollama:qwen2.5-3b-lora", "zero_shot"),
                  ("openai:gpt-4o-mini", "rag"))


def _ensemble():
    """The disagreement ensemble, rebuilt OFFLINE from results_v5cells_<split>.csv.

    No model and no API call: every input is a row s1_run_v5_cells.py already
    measured. The rule is the original one (thesis2/scripts/a1_rest.py
    run_ensemble): if the classifier and qwen2.5-3b-lora/zero_shot agree on
    (action, target, location), emit the CLASSIFIER's object; otherwise emit
    gpt-4o-mini/rag's.

    gpt-4o-mini/rag has R = 5 repeats. They are collapsed by MAJORITY VOTE over
    the canonical prediction (json, sorted keys); a tie goes to the earliest
    run_index among the tied predictions. The original kept the LAST repeat
    instead (its loader overwrote a dict); on V5's own files the two rules give
    the same 216/216 predictions, with no ties.

    Warm-up rows are used as predictions: a warm-up row is a real, validated
    answer, excluded by V5 from timing and denominators only. That is what gives
    command_id 0 a prediction (the classifier and LoRA cells have only their
    warm-up row there), so this arm has all n commands like every other arm.
    The original file has 216 -- command_id 0 was dropped with the warm-up.

    The runner passes command_id and split; the command text is checked
    against the V5 row so a misaligned pairing fails instead of scoring.
    """
    import collections
    import csv as _csv
    _csv.field_size_limit(2 ** 31 - 1)
    cache = {}

    def load(split):
        p = C.OUT / ("results_v5cells_%s.csv" % split)
        if not p.exists():
            C.gate(False, ["ensemble: %s does not exist -- run s1_run_v5_cells.py "
                           "--split %s first" % (p.relative_to(C.REPO), split)])
        by = {k: {} for k in ENSEMBLE_CELLS}
        with p.open(encoding="utf-8", newline="") as fh:
            for r in _csv.DictReader(fh):
                k = (r["model"], r["strategy"])
                if k in by:
                    by[k].setdefault(r["command_id"], []).append(r)
        n, census, bad = len(C.load_split(split)), C.v5_census(), []
        for k in ENSEMBLE_CELLS:
            R = census[k]["repeats"]
            short = [c for c in map(str, range(n)) if len(by[k].get(c, [])) != R]
            errs = sum(1 for rs in by[k].values() for r in rs
                       if r["stage"] in ("model_error", "prompt_error"))
            if short or errs:
                bad.append("%s/%s: %d command(s) without exactly R=%d rows %s; "
                           "%d model_error/prompt_error row(s)"
                           % (k[0], k[1], len(short), R, short[:8], errs))
        if bad:
            C.gate(False, ["ensemble: an input cell in %s is incomplete or has "
                           "errors -- fix it with s1_run_v5_cells.py first" % p.name]
                   + bad)
        print("  ensemble inputs: %s -- 3 cells complete, no errors" % p.name)
        return by

    def pred(r):
        try:
            o = json.loads(r["pred_json"] or "{}")
        except ValueError:
            o = {}
        return o if isinstance(o, dict) else {}

    def vote(reps):
        reps = sorted(reps, key=lambda r: int(r["run_index"]))
        keys = [json.dumps(pred(r), sort_keys=True) for r in reps]
        cnt = collections.Counter(keys)
        top = max(cnt.values())
        win = next(k for k in keys if cnt[k] == top)     # tie -> earliest run
        tied = sum(1 for v in cnt.values() if v == top) > 1
        return json.loads(win), "%d/%d%s" % (top, len(keys), " TIE" if tied else "")

    field3 = lambda o: (o.get("action"), o.get("target"), o.get("location"))

    def predict(cmd, command_id=None, split=None):
        if command_id is None or split is None:
            C.gate(False, ["ensemble pairs rows by command_id: call it through "
                           "s1_run_arm.py, which passes command_id and split"])
        if split not in cache:
            cache[split] = load(split)
        cid = str(command_id)
        cls, lora, rag = (cache[split][k][cid] for k in ENSEMBLE_CELLS)
        if any(r["command"] != cmd for r in cls + lora + rag):
            C.gate(False, ["ensemble: command_id %s in results_v5cells_%s.csv is "
                           "not %r -- misaligned pairing" % (cid, split, cmd)])
        a, b = pred(cls[0]), pred(lora[0])               # both R = 1
        if field3(a) == field3(b):
            return a, dict(latency_ms=None, route="agree")
        c, votes = vote(rag)
        return c, dict(latency_ms=None, route="escalate_cloud", cloud_votes=votes)
    return predict


register("ensemble", "local", "deterministic", _ensemble,
         "offline pairing of three measured V5 cells (classifier, qwen2.5-3b-lora/"
         "zero_shot, gpt-4o-mini/rag by majority vote); claim withdrawn (p = 1.0)")


def get(name) -> Arm:
    if name not in REGISTRY:
        print("GATE FAIL: unknown arm %r. Registered: %s"
              % (name, ", ".join(sorted(REGISTRY))))
        raise SystemExit(2)
    return REGISTRY[name]


def summary():
    print("  %-34s %-6s %-14s %s" % ("arm", "kind", "determinism", "note"))
    for n in sorted(REGISTRY):
        a = REGISTRY[n]
        tag = "BLOCKED" if a.blocked else a.determinism
        print("  %-34s %-6s %-14s %s" % (n, a.kind, tag, a.note[:60]))


if __name__ == "__main__":
    print("registry: %d arms" % len(REGISTRY))
    summary()
