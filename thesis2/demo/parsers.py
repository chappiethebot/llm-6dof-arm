"""thesis2/demo/parsers.py -- text -> action JSON, one entry per arm.

Every parser implements

    parse(text) -> action_json          (a dict built by the SHARED emit())

and is built from the SAME Stage 1 registry (thesis2/final/arms.py), so a parser
cannot drift from the arm that was measured.

HOW THIS REACHES ISAAC WITHOUT TOUCHING sim/ OR llm/
----------------------------------------------------
sim/run_demo.py calls llm/pipeline.py's run_once(command, client, strategy,
schema), and `client` is anything with `.name` and `.complete(prompt) -> str`.
So a parser is exposed here as a CLIENT SHIM: .complete() ignores the prompt
text (the arm gets the bare command) and returns the arm's JSON as a string.
sim/, llm/ and asr/ are imported unchanged and never modified.

REGISTERED: every arm that produces a FULL command.
NOT REGISTERED, by rule 7 and by definition:
  * cascade_oracle      -- needs the ground-truth row; a ceiling, not a system
  * SALSA row-only      -- produces a row, not a command
  * A1.7 field-level FLMD and every SALSA digits mapping -- withdrawn
  * grpo_base / grpo_sft -- unavailable until their selected adapters are frozen

v5_ollama reproduces the CURRENT demo behaviour exactly: it is the unmodified
ollama client the demo already uses, with the V5 legend and zero_shot strategy.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for p in (str(REPO), str(REPO / "thesis2" / "final")):
    if p not in sys.path:
        sys.path.insert(0, p)

import common as C                                                      # noqa: E402
import arms as A                                                        # noqa: E402


class Parser:
    """One arm, wrapped so it can be used either directly or as a client shim."""

    def __init__(self, name, build, strategy="classifier", note="", gpu=False):
        self.name = name
        self._build = build
        self.strategy = strategy
        self.note = note
        self.gpu = gpu
        self._fn = None
        self.last_thinking = ""
        self.last_latency_ms = None

    # -- direct use -------------------------------------------------------
    def parse(self, text):
        if self._fn is None:
            self._fn = self._build()
        t0 = time.perf_counter()
        obj, _info = self._fn(text)
        self.last_latency_ms = (time.perf_counter() - t0) * 1000
        return obj

    # -- client-shim use, for llm/pipeline.py run_once() -------------------
    def complete(self, prompt):
        """`prompt` is ignored: these arms consume the bare command. The demo
        passes the command through prompts.py's 'classifier' branch, which
        returns it verbatim."""
        return json.dumps(self.parse(prompt))


def _registry_parser(arm_name, **kw):
    def build():
        return A.get(arm_name).build()
    return lambda: Parser(arm_name, build, **kw)


def _v5_ollama():
    """The CURRENT demo behaviour, unchanged: the existing ollama client with the
    V5 legend and zero_shot. Not a re-implementation -- the real client."""
    from llm.clients import all_clients
    from bench.run_cell import apply_legend
    apply_legend("v5")
    client = {c.name: c for c in all_clients()}.get("ollama:qwen2.5-3b-lora")
    if client is None:
        raise SystemExit("v5_ollama: ollama:qwen2.5-3b-lora is not in "
                         "llm.clients.all_clients() -- is Ollama running?")
    p = Parser("v5_ollama", lambda: None, strategy="zero_shot",
               note="the current demo behaviour, reproduced exactly")
    p._client = client

    def parse(text):
        from llm.pipeline import run_once
        from llm import prompts as P
        schema = P.load_schema(str(C.SCHEMA_FILE))
        t0 = time.perf_counter()
        res = run_once(text, client, "zero_shot", schema)
        p.last_latency_ms = (time.perf_counter() - t0) * 1000
        return getattr(res, "obj", None) or {}

    p.parse = parse
    p.complete = lambda prompt: json.dumps(parse(prompt))
    return p


PARSERS = {
    # the reference: what the demo does today
    "v5_ollama": _v5_ollama,
    # encoder classifiers
    "flat_gpu": _registry_parser("flat_gpu", note="V5 rung-4 flat head", gpu=True),
    "flat_cpu": _registry_parser("flat_cpu", note="same weights, CPU"),
    "factorized_fp16": _registry_parser("factorized_fp16", note="factorized head", gpu=True),
    "factorized_int4": _registry_parser("factorized_int4", note="INT4 weights", gpu=True),
    # classical / small
    "rules_open": _registry_parser("rules_open", note="VOCABULARY-UNFAIR grammar"),
    "rules_firewalled": _registry_parser("rules_firewalled", note="honest grammar floor"),
    "textcnn": _registry_parser("textcnn", note="NEW: TextCNN", gpu=True),
    "mlpbow": _registry_parser("mlpbow", note="NEW: MLP bag-of-words", gpu=True),
    # generative / logit
    "lora_greedy": _registry_parser("lora_greedy", strategy="zero_shot",
                                    note="transformers FP16 merged, greedy", gpu=True),
    "salsa_full_qwen2.5-3b-lora": _registry_parser(
        "salsa_full_qwen2.5-3b-lora", note="NEW: SALSA-full, letters", gpu=True),
    "salsa_full_Qwen2.5-3B-Instruct": _registry_parser(
        "salsa_full_Qwen2.5-3B-Instruct", note="NEW: SALSA-full, letters", gpu=True),
    "salsa_full_qwen2.5-0.5b-lora": _registry_parser(
        "salsa_full_qwen2.5-0.5b-lora", note="NEW: SALSA-full, letters", gpu=True),
    "salsa_full_Qwen2.5-0.5B-Instruct": _registry_parser(
        "salsa_full_Qwen2.5-0.5B-Instruct", note="NEW: SALSA-full, letters", gpu=True),
    # composite
    "cascade_local_tauA": _registry_parser("cascade_local_tauA",
                                           note="tau_A = 0.50, row -> 3B LoRA", gpu=True),
    "pipeline_always_local": _registry_parser("pipeline_always_local",
                                              note="PIPELINE, NOT ROUTING", gpu=True),
    "cascade_cloud_tauA": _registry_parser("cascade_cloud_tauA",
                                           note="tau_A = 0.50, row -> gpt-4o-mini"),
    "cascade_cloud_always": _registry_parser("cascade_cloud_always",
                                             note="PIPELINE, NOT ROUTING; cloud"),
    # Inference always runs in thesis2gpu, using the frozen selected adapters.
    "grpo_base": _registry_parser("grpo_base", note="requires frozen GRPO adapter", gpu=True),
    "grpo_sft": _registry_parser("grpo_sft", note="requires eligible, frozen GRPO adapter", gpu=True),
}

NOT_REGISTERED = {
    "cascade_oracle": "ceiling: needs the ground-truth row, not a deployable system",
    "salsa_row_only": "produces a row, not a full command",
    "a1_7_field_flmd": "WITHDRAWN (FINAL_REPORT B-3)",
    "salsa_digits_any": "WITHDRAWN (FINAL_REPORT B-4)",
}

CLOUD = {"cascade_cloud_tauA", "cascade_cloud_always"}
BLOCKED = {"grpo_base", "grpo_sft"} - set(C.read_frozen())
# Which parsers put weights on the GPU. Declared here rather than probed, so
# reading this set never constructs a parser (construction loads a model).
GPU_PARSERS = {
    "flat_gpu", "factorized_fp16", "factorized_int4", "textcnn", "mlpbow",
    "lora_greedy", "cascade_local_tauA", "pipeline_always_local",
    "salsa_full_qwen2.5-3b-lora", "salsa_full_Qwen2.5-3B-Instruct",
    "salsa_full_qwen2.5-0.5b-lora", "salsa_full_Qwen2.5-0.5B-Instruct",
    # registered with gpu=True above but missing here until 2026-09-28, so the
    # smoke test never measured their VRAM
    "grpo_base", "grpo_sft",
}


def get(name) -> Parser:
    if name not in PARSERS:
        raise SystemExit("unknown parser %r. Registered: %s. Not registered: %s"
                         % (name, ", ".join(sorted(PARSERS)),
                            json.dumps(NOT_REGISTERED)))
    return PARSERS[name]()


def names():
    return sorted(PARSERS)


if __name__ == "__main__":
    print("registered parsers (%d):" % len(PARSERS))
    for n in names():
        print("  %s" % n)
    print()
    print("deliberately NOT registered:")
    for n, why in NOT_REGISTERED.items():
        print("  %-22s %s" % (n, why))
