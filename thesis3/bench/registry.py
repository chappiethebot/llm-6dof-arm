"""thesis3 registry: the 66 systems, as metadata (read by the parent, run_bench.py) and as
builders (run inside a worker, worker.py, in the system's own conda env).

Importing this module loads no model and imports nothing from thesis2: every builder
imports lazily, inside the worker. Every system is built from the existing code:
  - the 30 final-run systems from thesis2/final/arms.py (A8/A10/A12 from their own
    scripts in thesis2/scripts, which arms.py delegates to),
  - the 36 V5 cells through llm/pipeline.run_once with llm/clients, exactly as
    thesis2/final/s1_run_v5_cells.py and s3_energy.v5_cell_predictor run them,
  - ensemble online, by the rule of arms._ensemble (METHODS_NOTES item 8).

A worker-side system is a Built:  predict(command, truth) -> (raw, obj, info), timed by the
worker; finish(...) fills confidence, tokens, extra and error AFTER the call's timing ends.
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PY = {"thesis2gpu": r"D:\condaenvs\thesis2gpu\python.exe",
      "thesis": r"C:\Users\USER\.conda\envs\thesis\python.exe"}
SPLITS = ("held_out_v3", "held_out_v4c")
BLOCKS = ("A", "B", "C1", "C2", "D")


# ====================================================================== metadata
@dataclass
class System:
    name: str
    family: str
    kind: str          # local / cloud
    device: str        # gpu / cpu / cloud
    runtime: str
    env: str
    legend: str        # what this system records: v5 / none / salsa_row
    uses_cloud: int
    repeats: int
    block: str
    builder: str       # arm / spec / v5 / ensemble
    arm: str = ""      # thesis2 arm name (arm / spec builders)
    model: str = ""    # V5 client name (v5 builder)
    strategy: str = ""
    note: str = ""

    @property
    def slug(self):
        return re.sub(r"[^A-Za-z0-9._-]+", "-", self.name).strip("-")


SYSTEMS: dict[str, System] = {}


def _add(name, family, kind, device, runtime, env, legend, uses_cloud, repeats, block, builder, **kw):
    SYSTEMS[name] = System(name, family, kind, device, runtime, env, legend, uses_cloud, repeats, block,
                           builder, **kw)


def _arm(name, family, device, runtime, legend, block, kind="local", uses_cloud=0, repeats=1, **kw):
    _add(name, family, kind, device, runtime, "thesis2gpu", legend, uses_cloud, repeats, block, "arm",
         arm=name, **kw)


# ---- block A: CPU systems
_arm("flat_cpu", "classifier", "cpu", "torch", "none", "A")
_arm("rules_open", "baseline", "cpu", "python", "none", "A")
_arm("rules_firewalled", "baseline", "cpu", "python", "none", "A")
_arm("tfidf_logreg", "baseline", "cpu", "sklearn", "none", "A")
_arm("bilstm", "baseline", "cpu", "torch", "none", "A")
_arm("knn_minilm", "classifier", "cpu", "sentence-transformers", "none", "A")
_arm("minilm_probe", "classifier", "cpu", "sentence-transformers", "none", "A")
_add("v5:classifier:distilbert/classifier", "v5_cell", "local", "cpu", "torch", "thesis", "none", 0, 1, "A",
     "v5", model="classifier:distilbert", strategy="classifier")
# ---- block B: transformers / torch GPU systems (lora_greedy before the speculative arms)
_arm("flat_gpu", "classifier", "gpu", "torch", "none", "B")
_arm("factorized_fp16", "classifier", "gpu", "torch", "none", "B")
_arm("factorized_int4", "classifier", "gpu", "torch", "none", "B")
_arm("tinybert", "classifier", "gpu", "transformers", "none", "B")
_arm("textcnn", "baseline", "gpu", "torch", "none", "B")
_arm("mlpbow", "baseline", "gpu", "torch", "none", "B")
_arm("flant5", "llm_local", "gpu", "transformers", "none", "B")
_arm("lora_greedy", "llm_local", "gpu", "transformers", "v5", "B")
for _n in ("a8_speculative", "a10_speculative", "a12_smalllm"):
    _add(_n, "speculative", "local", "gpu", "transformers", "thesis2gpu", "v5", 0, 1, "B", "spec", arm=_n,
         note="model_load_s includes a 3-command decode: the thesis2 script can only be loaded by running "
              "its pipeline (imported with --limit 3, all its decoders)")
for _k in ("qwen2.5-3b-lora", "Qwen2.5-3B-Instruct", "qwen2.5-0.5b-lora", "Qwen2.5-0.5B-Instruct"):
    _arm("salsa_full_" + _k, "salsa", "gpu", "transformers", "v5", "B")
_arm("grpo_base", "grpo", "gpu", "transformers", "v5", "B")
_arm("grpo_sft", "grpo", "gpu", "transformers", "v5", "B")
_arm("cascade_local_tauA", "cascade", "gpu", "torch+transformers", "salsa_row", "B")
_arm("pipeline_always_local", "cascade", "gpu", "torch+transformers", "salsa_row", "B")
_arm("cascade_oracle", "ceiling", "gpu", "torch", "none", "B")
# ---- blocks C1 / C2: Ollama cells
for _m in ("qwen2.5-3b-lora", "qwen2.5:3b", "qwen2.5-0.5b-lora", "qwen2.5:0.5b", "llama3.2:3b", "deepseek-r1:7b"):
    for _s in ("zero_shot", "few_shot", "cot", "role_based", "rag"):
        _add("v5:ollama:%s/%s" % (_m, _s), "v5_cell", "local", "gpu", "ollama", "thesis", "v5", 0, 1,
             "C2" if _m.startswith("deepseek") else "C1", "v5", model="ollama:" + _m, strategy=_s)
# ---- block D: cloud cells and systems that call gpt-4o-mini
for _s in ("zero_shot", "few_shot", "cot", "role_based", "rag"):
    _add("v5:openai:gpt-4o-mini/%s" % _s, "v5_cell", "cloud", "cloud", "openai", "thesis", "v5", 1, 5, "D", "v5",
         model="openai:gpt-4o-mini", strategy=_s)
_arm("cascade_cloud_tauA", "cascade", "gpu", "torch+openai", "salsa_row", "D", kind="cloud", uses_cloud=1, repeats=5)
_arm("cascade_cloud_always", "cascade", "gpu", "torch+openai", "salsa_row", "D", kind="cloud", uses_cloud=1, repeats=5)
_add("ensemble", "ensemble", "local", "gpu", "torch+ollama+openai", "thesis", "v5", 1, 1, "D", "ensemble",
     note="online: classifier:distilbert/classifier + ollama:qwen2.5-3b-lora/zero_shot; on disagreement "
          "5 x openai:gpt-4o-mini/rag, majority vote")

ORDER = [n for b in BLOCKS for n, s in SYSTEMS.items() if s.block == b]


def check_against_expected(path=HERE / "systems_expected.csv"):
    """-> list of mismatches between this registry and systems_expected.csv (empty = exact match)."""
    exp = {r["system"]: r for r in csv.DictReader(open(path, encoding="utf-8", newline=""))}
    bad = [f"{n}: not in systems_expected.csv" for n in SYSTEMS if n not in exp]
    bad += [f"{n}: in systems_expected.csv, not in the registry" for n in exp if n not in SYSTEMS]
    for n, e in exp.items():
        s = SYSTEMS.get(n)
        if s is None:
            continue
        for col in ("family", "kind", "env", "uses_cloud", "repeats"):
            if str(getattr(s, col)) != e[col]:
                bad.append(f"{n}: {col} {getattr(s, col)} != expected {e[col]}")
        le = e["legend_expected"]
        if (le in ("v5", "none") and s.legend != le) or (le == "recorded" and s.legend not in ("v5", "salsa_row")):
            bad.append(f"{n}: legend {s.legend} vs expected {le}")
    return bad


# ====================================================================== worker side
def _setup_paths():
    for p in (REPO, REPO / "thesis2" / "final", REPO / "thesis2" / "scripts"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def _resolve(spec):
    """'module:Class.attr' -> (owner, attr)."""
    import importlib
    mod, rest = spec.split(":")
    owner = importlib.import_module(mod)
    parts = rest.split(".")
    for p in parts[:-1]:
        owner = getattr(owner, p)
    return owner, parts[-1]


class Meter:
    """Stamps the model call(s) of ONE system call: t0 = first wrapped call starts, t1 = last
    wrapped call ends. Also keeps what finish() needs (classifier logits, token counts),
    so nothing is recomputed inside the timed window."""

    def __init__(self):
        self.active = False
        self.reset()

    def reset(self):
        self.t0 = self.t1 = None
        self.starts = {}            # wrapped target -> its first two start stamps
        self.n_in = None
        self.out = None
        self.gen = None
        self.resp = None
        self.n = {}

    def wrap(self, spec, capture=False, count_in=False, gen=False, resp=False, count=False):
        import functools
        owner, attr = _resolve(spec)
        f = getattr(owner, attr)
        m, key = self, spec

        @functools.wraps(f)
        def w(*a, **k):
            t = time.perf_counter_ns()
            if m.t0 is None:
                m.t0 = t
            st = m.starts.setdefault(key, [])
            if len(st) < 2:
                st.append(t)
            if count:
                m.n[key] = m.n.get(key, 0) + 1
            if count_in and m.n_in is None:
                ids = k.get("input_ids", a[1] if len(a) > 1 else None)
                if ids is not None:
                    m.n_in = int(ids.shape[-1])
            r = f(*a, **k)
            m.t1 = time.perf_counter_ns()
            if capture:
                m.out = r
            if gen:
                m.gen = (int(k["input_ids"].shape[-1]), int(r.shape[-1]))
            if resp:
                m.resp = r
            return r
        setattr(owner, attr, w)
        self.active = True


@dataclass
class Built:
    predict: object                     # (command, truth) -> (raw or None, obj, info)
    finish: object                      # (raw, obj, info, meter, t_start_ns, t_end_ns) -> dict
    prepare: object = None              # (split, rows) -> None, before a run's warm-up
    unload: object = None
    notes: list = field(default_factory=list)


MHC = "ft.train_classifier:MultiHeadClassifier.forward"
FAC = "p2_train_factorized:FactorizedClassifier.forward"
QWEN = "transformers.models.qwen2.modeling_qwen2:Qwen2ForCausalLM.forward"
T5F = "transformers.models.t5.modeling_t5:T5ForConditionalGeneration.forward"
GEN = "transformers.generation.utils:GenerationMixin.generate"
ST = "sentence_transformers:SentenceTransformer.encode"
OAI = "openai.resources.chat.completions:Completions.create"
ARM_CFG = {   # system -> (wraps [(spec, flags)], confidence source, token source)
    "flat_gpu": ([(MHC, "capture")], "cap3", None),
    "flat_cpu": ([(MHC, "capture")], "cap3", None),
    "factorized_fp16": ([(FAC, "capture")], "fac", None),
    "factorized_int4": ([(FAC, "capture")], "fac", None),
    "tinybert": ([("legacy_recipes:TinyFactorized.forward", "")], "info", None),
    "bilstm": ([("legacy_recipes:BiLSTMJoint.forward", "")], "info", None),
    "minilm_probe": ([(ST, "")], "info", None),
    "knn_minilm": ([(ST, "")], None, None),
    "tfidf_logreg": ([("sklearn.feature_extraction.text:TfidfVectorizer.transform", ""),
                      ("sklearn.linear_model:LogisticRegression.predict", ""),
                      ("sklearn.linear_model:LogisticRegression.predict_proba", "")], "info", None),
    "rules_open": ([], None, None),
    "rules_firewalled": ([], None, None),
    "textcnn": ([("s0_train_textcnn:TextCNN.forward", "capture")], "cap3", None),
    "mlpbow": ([("s0_train_mlpbow:MLPBoW.forward", "capture")], "cap3", None),
    "flant5": ([(GEN, "gen"), (T5F, "")], None, "s2s"),
    "lora_greedy": ([(QWEN, "count_in")], None, "lora"),
    "grpo_base": ([(GEN, "gen"), (QWEN, "")], None, "dec"),
    "grpo_sft": ([(GEN, "gen"), (QWEN, "")], None, "dec"),
    "cascade_local_tauA": ([(FAC, "capture"), (QWEN, "")], "fac", None),
    "pipeline_always_local": ([(FAC, "capture"), (QWEN, "")], "fac", None),
    "cascade_oracle": ([(FAC, "capture")], "fac", None),
    "cascade_cloud_tauA": ([(FAC, "capture"), (OAI, "count")], "fac", None),
    "cascade_cloud_always": ([(FAC, "capture"), (OAI, "count")], "fac", None),
}
for _k in ("qwen2.5-3b-lora", "Qwen2.5-3B-Instruct", "qwen2.5-0.5b-lora", "Qwen2.5-0.5B-Instruct"):
    ARM_CFG["salsa_full_" + _k] = ([(QWEN, "")], "info", None)


def _apply(meter, wraps):
    for spec, flags in wraps:
        meter.wrap(spec, **{f: True for f in flags.split(",") if f})


def _maxp(t):
    import torch
    return float(torch.softmax(t.float(), -1).max())


def _confidence(mode, meter, info):
    """classifier: product of the verb, target and location heads' max softmax."""
    if mode == "cap3" and meter.out is not None:
        lv, lt, ll = meter.out
        return _maxp(lv) * _maxp(lt) * _maxp(ll)
    if mode == "fac" and meter.out is not None:
        from p2_train_factorized import FACTORS
        lg = meter.out
        return _maxp(lg["verb"]) * _maxp(lg["target"]) * min(_maxp(lg[f]) for f in FACTORS)
    if mode in ("info", "fac", "cap3"):
        ps = [info.get(k) for k in ("p_verb", "p_target", "p_location")]
        ps = [float(p) for p in ps if isinstance(p, (int, float))]
        if ps:
            out = 1.0
            for p in ps:
                out *= p
            return out
    return ""


def _ttft(meter, t_start, spec=QWEN):
    """call start -> start of the LLM's second forward pass (the first token is decided by then)"""
    st = meter.starts.get(spec, [])
    if len(st) >= 2:
        return (st[1] - t_start) / 1e6
    return ""


def hf_rev(repo_id):
    p = Path.home() / ".cache" / "huggingface" / "hub" / ("models--" + repo_id.replace("/", "--")) / "refs" / "main"
    return "hf:%s@%s" % (repo_id, p.read_text().strip() if p.exists() else "unknown")


def _sha_file(p):
    import common as C
    return C.sha256_of(Path(p))


def _sha_dir(p):
    from s0_freeze import dir_sha
    return dir_sha(Path(p))


FLAT_PT = REPO / "ft" / "runs" / "distilbert_cls" / "epoch_09" / "model.pt"
FACT_PT = REPO / "thesis2" / "models" / "distilbert_factorized" / "epoch_11" / "model.pt"
LORA3B = REPO / "ft" / "runs" / "qwen3b_lora" / "checkpoint-87"


def model_sha(s: System) -> str:
    """Weights hash, computed BEFORE the timed model load. Frozen arms are re-verified
    against thesis2/out/final/frozen_checkpoints.csv first (common.require_frozen exits on
    a mismatch, which stops the campaign)."""
    _setup_paths()
    import common as C
    frozen = C.read_frozen()
    n = s.arm or s.name
    if n in frozen:
        rec = C.require_frozen(n)
        sha = rec["sha256"]
        if sha.startswith("n/a"):
            return "%s; %s" % (sha, hf_rev(rec["path"]))
        return sha
    if n in ("flat_gpu", "flat_cpu") or s.model == "classifier:distilbert":
        return "sha256:" + _sha_file(FLAT_PT)
    if n.startswith("factorized") or n == "cascade_oracle" or n.startswith("cascade_cloud"):
        return "sha256:" + _sha_file(FACT_PT)
    if n in ("cascade_local_tauA", "pipeline_always_local"):
        return "factorized:%s; lora:%s" % (_sha_file(FACT_PT), _sha_dir(LORA3B))
    if n in ("lora_greedy", "a8_speculative", "a10_speculative"):
        return "lora:%s; %s" % (_sha_dir(LORA3B), hf_rev("Qwen/Qwen2.5-3B-Instruct"))
    if n == "a12_smalllm":
        return "lora:%s; %s; drafter %s" % (_sha_dir(LORA3B), hf_rev("Qwen/Qwen2.5-3B-Instruct"),
                                            hf_rev("Qwen/Qwen2.5-0.5B-Instruct"))
    if n.startswith("rules_"):
        return "code:" + _sha_file(REPO / "thesis2" / "scripts" / "p5_rules_firewalled.py")
    if s.model.startswith("ollama:"):
        return "ollama:" + ollama_digest(s.model[len("ollama:"):])
    if s.model.startswith("openai:"):
        return "n/a"
    if s.builder == "ensemble":
        return "classifier:%s; ollama:%s; gpt-4o-mini: n/a" % (_sha_file(FLAT_PT), ollama_digest("qwen2.5-3b-lora"))
    return "n/a"


def ollama_digest(name):
    import ollama
    full = name if ":" in name else name + ":latest"
    for m in ollama.list().models:
        if m.model == full:
            return m.digest
    raise SystemExit("ollama model %s not found on the server (OLLAMA_MODELS=D:\\LLM?)" % full)


def _json_raw(raw, obj):
    return raw if raw is not None else json.dumps(obj)


# ---------------------------------------------------------------- arms.py systems
def _build_arm(s: System, meter: Meter) -> Built:
    import arms as A
    import common as C
    arm = A.get(s.arm)
    if arm.kind != s.kind:
        raise SystemExit("REGISTRY MISMATCH: arms.py says %s is %s, the registry says %s" % (s.arm, arm.kind, s.kind))
    predict = arm.build()
    wraps, conf_mode, tok_mode = ARM_CFG[s.name]
    _apply(meter, wraps)
    oracle = s.name == "cascade_oracle"

    def pred(cmd, truth):
        if oracle:
            obj, info = predict(cmd, truth_row=C.rowof(truth.get("location")))
        else:
            obj, info = predict(cmd)
        return info.get("raw"), obj, info

    def finish(raw, obj, info, m, t0, t1):
        d = {"confidence": _confidence(conf_mode, m, info)}
        if info.get("stage") == "model_error":
            d["error"] = info.get("error") or "model_error"
        ex = {}
        if s.family == "cascade":
            ex["escalated"] = int(bool(info.get("escalated")))
        if s.uses_cloud:
            ex["retries"] = int(info.get("retries") or 0)
            ex["cloud_calls"] = m.n.get(OAI, 0) - ex["retries"]
        d["extra"] = ex
        if tok_mode == "lora":
            d.update(tokens_in=m.n_in, tokens_out=int(info.get("forward_passes", 1)) - 1, tokens_thinking=0,
                     ttft_ms=_ttft(m, t0))
        elif tok_mode == "dec" and m.gen:
            d.update(tokens_in=m.gen[0], tokens_out=m.gen[1] - m.gen[0], tokens_thinking=0, ttft_ms=_ttft(m, t0))
        elif tok_mode == "s2s" and m.gen:
            d.update(tokens_in=m.gen[0], tokens_out=m.gen[1] - 1, tokens_thinking=0, ttft_ms=_ttft(m, t0, T5F))
        return d
    return Built(pred, finish)


# ---------------------------------------------------------------- A8 / A10 / A12
SPEC_SCRIPT = {"a8_speculative": "a8_speculative.py", "a10_speculative": "a10_prefix_cache.py",
               "a12_smalllm": "a12_smalllm_drafter.py"}


def _a11_regions():
    """regions() from thesis2/scripts/a11_rejection_positions.py, taken from the file
    without running it (the script does its whole analysis at import)."""
    import ast
    p = REPO / "thesis2" / "scripts" / "a11_rejection_positions.py"
    tree = ast.parse(p.read_text(encoding="utf-8"))
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "regions"]
    ns = {}
    exec(compile(ast.Module(body=fn, type_ignores=[]), str(p), "exec"), ns)
    return ns["regions"]


def _load_script(fname, tmp):
    import importlib.util
    path = REPO / "thesis2" / "scripts" / fname
    argv = sys.argv
    sys.argv = [str(path), "--split", "held_out_v3", "--limit", "3", "--outdir", tmp]
    try:
        spec = importlib.util.spec_from_file_location(fname[:-3], path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[fname[:-3]] = mod
        spec.loader.exec_module(mod)
    finally:
        sys.argv = argv
    return mod


def _build_spec(s: System, meter: Meter) -> Built:
    tmp = REPO / "thesis3" / "out" / "_spec_import" / s.name     # the scripts print paths relative to the repo
    tmp.mkdir(parents=True, exist_ok=True)
    mod = _load_script(SPEC_SCRIPT[s.name], str(tmp))
    a12 = s.name == "a12_smalllm"
    _apply(meter, [(QWEN, "")] + ([] if a12 else [(MHC, "")]))
    regions = _a11_regions()
    tok = mod.tok

    def prepare(split, rows):
        """A10/A12 cache the token prefix common to the split's prompts, as their thesis2
        runs did per split (the --limit 3 import computed it over 3 commands only)."""
        if s.name == "a8_speculative":
            return
        ids = [mod.prep(r["command"]) if a12 else mod.prep(r["command"])[0] for r in rows]
        k, L = 0, min(len(x) for x in ids)
        while k < L and len({tuple(x[:k + 1]) for x in ids}) == 1:
            k += 1
        mod.PREFIX = ids[0][:k]
        if a12:
            mod.V_CACHE = mod.build_cache(mod.model)
            mod.D_CACHE = mod.build_cache(mod.drafter)
        else:
            mod.LAST_PREFIX_LOGIT = mod.build_prefix_cache()

    def pred(cmd, truth):
        if a12:
            pids = mod.prep(cmd)
            gen, vp, _dp, off, acc, _dms = mod.spec_smalllm(pids)
            info = dict(pids=pids, gen=gen, verifier_passes=vp, draft_tokens=off, accepted_tokens=acc)
        else:
            if s.name == "a8_speculative":
                pids, dids, _t, _ms, _c = mod.prep(cmd)
                gen, steps, n_acc, k = mod.speculative(pids, dids)
            else:
                pids, dids, _ms = mod.prep(cmd)
                gen, steps, n_acc, k = mod.spec_prefix(pids, dids)
            info = dict(pids=pids, dids=dids, gen=gen, verifier_passes=steps, draft_tokens=k, accepted_tokens=n_acc)
        raw = tok.decode(info["gen"], skip_special_tokens=True)
        try:
            obj = json.loads(raw)
        except Exception:
            obj = {}
        return raw, obj if isinstance(obj, dict) else {}, info

    def first_rejection(info, raw):
        dids, n = info["dids"], info["accepted_tokens"]
        if not dids:
            return "no_draft"
        if n == len(dids):
            return "none"
        if n == len(dids) - 1:
            return "stop_token"
        dtxt = tok.decode(dids[:-1])
        d = 0
        while d < min(len(dtxt), len(raw)) and dtxt[d] == raw[d]:
            d += 1
        try:
            reg = regions(dtxt, json.loads(dtxt))
        except Exception:
            return "json_syntax"
        return reg[d] if d < len(reg) else "json_syntax"

    def finish(raw, obj, info, m, t0, t1):
        ex = {k: int(info[k]) for k in ("accepted_tokens", "draft_tokens", "verifier_passes")}
        if not a12:
            ex["first_rejection_field"] = first_rejection(info, raw)
        return dict(tokens_in=len(info["pids"]), tokens_out=len(info["gen"]), tokens_thinking=0, extra=ex)
    return Built(pred, finish, prepare, notes=[s.note])


# ---------------------------------------------------------------- V5 cells and ensemble
def _v5_common():
    import common as C
    from bench.run_cell import apply_legend
    from llm import prompts as P
    from llm.clients import all_clients
    apply_legend("v5")
    schema = P.load_schema(str(C.SCHEMA_FILE))
    return C, schema, {c.name: c for c in all_clients()}


def _ollama_load(client):
    """Ollama model load: the first request, empty prompt, the cell's own options (a
    different num_ctx would make the server reload the model at the first real call)."""
    client._ollama.generate(model=client.model, prompt="",
                            options={"temperature": client.temperature, "num_ctx": client.num_ctx})


_THINK_TOK = {}


def _thinking_tokens(model, text):
    if not text:
        return 0
    if model not in _THINK_TOK:
        from bench.backfill_tokens_v5 import Tokenizers
        _THINK_TOK[model] = Tokenizers().for_model(model)[0]
    return _THINK_TOK[model](text)


def _v5_finish_fields(model, info, m):
    d = {}
    u = info.get("usage") or {}
    if model.startswith(("ollama:", "openai:")):
        d["tokens_in"] = u.get("tokens_in", "")
        d["tokens_out"] = u.get("tokens_out_total", "")
        d["tokens_thinking"] = _thinking_tokens(model, info.get("thinking", "")) if "deepseek" in model else 0
    r = m.resp
    if model.startswith("ollama:") and r is not None:
        tot, ev = getattr(r, "total_duration", None), getattr(r, "eval_duration", None)
        if tot and ev is not None:
            d["ttft_ms"] = (tot - ev) / 1e6        # server side: everything before the first generated token
    return d


def _stage_error(st, reasons):
    if st in ("model_error", "prompt_error", "pipeline_error"):
        return "%s: %s" % (st, "; ".join(reasons or [])[:500])
    return ""


def _build_v5(s: System, meter: Meter) -> Built:
    C, schema, roster = _v5_common()
    from llm.pipeline import run_once
    client = roster[s.model]
    cloud = s.model.startswith("openai:")
    if s.model.startswith("ollama:"):
        import ollama
        meter.wrap("llm.clients:OllamaClient.complete")
        meter.wrap("ollama:generate", resp=True)
        _ollama_load(client)
    elif cloud:
        client = C.RetryingClient(client)
        meter.wrap("llm.clients:OpenAIClient.complete")
    else:
        client._ensure_loaded()
        meter.wrap("llm.classifier_client:DistilBertClassifierClient.complete")
        meter.wrap(MHC, capture=True)
    retriever = C.v5_retriever(quiet=True) if s.strategy == "rag" else None
    state = {}

    def prepare(split, rows):
        v5rows, fewshot, _r = C.v5_pool(split, s.strategy)
        if [r["command"] for r in v5rows] != [r["command"] for r in rows]:
            raise SystemExit("bench/run_cell._load_rows(%r) disagrees with data/%s.jsonl" % (split, split))
        state["fewshot"] = fewshot

    def pred(cmd, truth):
        ex = C.v5_examples_for(cmd, s.strategy, state["fewshot"], retriever)[0]
        r = run_once(cmd, client, s.strategy, schema, examples=ex)
        info = dict(stage=r.stage, reasons=r.reasons, thinking=r.thinking,
                    usage=dict(getattr(client, "last_usage", {}) or {}))
        if cloud:
            info["retries"] = client.last_retries
        return r.raw, (r.obj if isinstance(r.obj, dict) else {}), info

    def finish(raw, obj, info, m, t0, t1):
        d = _v5_finish_fields(s.model, info, m)
        d["error"] = _stage_error(info["stage"], info["reasons"])
        if s.model.startswith("classifier:"):
            d["confidence"] = _confidence("cap3", m, info)
        if cloud:
            d["extra"] = {"retries": int(info.get("retries") or 0), "cloud_calls": 1}
        return d
    return Built(pred, finish, prepare)


def _build_ensemble(s: System, meter: Meter) -> Built:
    """arms._ensemble's rule, online: classifier and qwen2.5-3b-lora/zero_shot agree on
    (action, target, location) -> the classifier's object; otherwise gpt-4o-mini/rag x 5,
    majority vote over the canonical prediction, a tie to the earliest repeat, a failed
    repeat votes {} (arms._ensemble: pred() of a failed row is {})."""
    import collections
    C, schema, roster = _v5_common()
    from llm.pipeline import run_once
    cls, lora = roster["classifier:distilbert"], roster["ollama:qwen2.5-3b-lora"]
    gpt = C.RetryingClient(roster["openai:gpt-4o-mini"])
    import ollama  # noqa: F401
    meter.wrap("llm.clients:OllamaClient.complete")
    meter.wrap("llm.clients:OpenAIClient.complete")
    meter.wrap("llm.classifier_client:DistilBertClassifierClient.complete")
    cls._ensure_loaded()
    _ollama_load(lora)
    retriever = C.v5_retriever(quiet=True)
    f3 = lambda o: (o.get("action"), o.get("target"), o.get("location"))   # noqa: E731

    def pred(cmd, truth):
        a = run_once(cmd, cls, "classifier", schema)
        b = run_once(cmd, lora, "zero_shot", schema)
        oa = a.obj if isinstance(a.obj, dict) else {}
        ob = b.obj if isinstance(b.obj, dict) else {}
        info = dict(escalated=0, cloud_calls=0, retries=0, errors=[], stages=[a.stage, b.stage])
        raw = {"classifier": a.raw, "qwen2.5-3b-lora/zero_shot": b.raw}
        for st, r in (("classifier", a), ("lora", b)):
            e = _stage_error(r.stage, r.reasons)
            if e:
                info["errors"].append("%s %s" % (st, e))
        if f3(oa) == f3(ob):
            return json.dumps(raw), oa, info
        ex = C.v5_examples_for(cmd, "rag", None, retriever)[0]
        reps = []
        for _ in range(5):
            r = run_once(cmd, gpt, "rag", schema, examples=ex)
            info["retries"] += int(gpt.last_retries or 0)
            reps.append(r)
        keys = [json.dumps(r.obj if isinstance(r.obj, dict) else {}, sort_keys=True) for r in reps]
        cnt = collections.Counter(keys)
        top = max(cnt.values())
        win = next(k for k in keys if cnt[k] == top)
        info.update(escalated=1, cloud_calls=5, votes="%d/5" % top)
        cloud_err = [_stage_error(r.stage, r.reasons) for r in reps]
        info["cloud_errors"] = sum(1 for e in cloud_err if e)
        if win == "{}" and info["cloud_errors"]:
            info["errors"].append(next(e for e in cloud_err if e))
        raw["gpt-4o-mini/rag"] = [r.raw for r in reps]
        return json.dumps(raw), json.loads(win), info

    def finish(raw, obj, info, m, t0, t1):
        ex = {k: info[k] for k in ("escalated", "cloud_calls", "retries")}
        if info.get("cloud_errors"):
            ex["cloud_errors"] = info["cloud_errors"]
        return dict(error="; ".join(info["errors"]), extra=ex)
    return Built(pred, finish)


def build(s: System, meter: Meter) -> Built:
    _setup_paths()
    return {"arm": _build_arm, "spec": _build_spec, "v5": _build_v5, "ensemble": _build_ensemble}[s.builder](s, meter)


if __name__ == "__main__":
    bad = check_against_expected()
    print("%d systems; registry vs systems_expected.csv: %s" % (len(SYSTEMS), "EXACT MATCH" if not bad else bad))
