"""
bench/llm_client.py

LLM-side half of the bridge: model selection, prompt strategies, text or
voice input, per-stage latency timing. Runs in conda env `thesis`.
Never imports isaacsim.

Works with EITHER simulator via --backend. The two have different location
vocabularies; prompting one while the other executes reproduces exactly the
silent symbol mismatch the V4 runs were about, so it is selected explicitly.

WARNING: this is the DEMO path, not the measurement harness. Do not quote
timings or accuracy from this script in the thesis.

    python bench/llm_client.py --list
    python bench/llm_client.py --backend isaac  --model gpt-4o-mini --strategy rag
    python bench/llm_client.py --backend isaac  --model classifier:distilbert
    python bench/llm_client.py --backend mujoco --model gpt-4o-mini --selftest
    python bench/llm_client.py --once "put the red cube in box 1"
    python bench/llm_client.py --script demo_script.txt


PARITY WITH sim/run_demo.py  (READ THIS BEFORE CHANGING EITHER FILE)
=====================================================================
sim/run_demo.py is the OTHER demo client. It drives MuJoCo in-process through
llm/pipeline.py's run_once(); this file drives EITHER simulator out-of-process
over the bridge/ file protocol, so a model crash cannot kill the simulator.

They SHARE, and must keep sharing:
  * the client roster           llm.clients.all_clients()   (rungs 1-4)
  * the legend                  bench.run_cell.apply_legend(), v5 by default
  * constrained decoding        llm.schema_json, on by default
  * the verb/target/field-mask grammar, read from schema/schema_v2.json
  * the few-shot exemplars      bench.dataset.load_seen_split(data/seen_v2.jsonl)
  * the RAG retriever           llm.retrieve.Retriever
  * the voice front-end         asr/voice.py

VOCABULARY -- the one place they still differ, and it is now a FLAG.
--------------------------------------------------------------------
Earlier versions of this file prompted rungs 1 and 2 in the ATTACHED BACKEND's
location vocabulary (Isaac's cell_R_C, or the V1 table_* names behind
mujoco_bridge.py) while rungs 3 and 4 ran canonically over schema_v2. That made
three things impossible at once:

  * the legend could not be applied to rungs 1/2 (it is written in schema_v2
    grid terms, not cell_R_C terms);
  * constrained decoding could not be attached to rungs 1/2 on Isaac, because
    the schema_v2 JSON Schema forbids every legal Isaac location;
  * a rung-1 answer and a rung-4 answer to the same command were not produced
    under the same prompt, so the demo's headline comparison -- 19 ms classifier
    beside a cloud model -- was not like for like.

So `--vocab canonical` now prompts EVERY rung in schema_v2 (grid_A1..grid_C3)
and translates to the backend's names at the edge, through the same
to_backend_action() that rungs 3 and 4 already used. Nothing edits
schema_v2.json; nothing touches the 30,380 measured rows. It is the default on
Isaac, where every grid cell has an exact equivalent.

`--vocab backend` keeps the old behaviour and is the default on MuJoCo, whose
table_center / table_left / table_right / table_front / table_back have NO
schema_v2 equivalent and cannot be reached canonically.

Known cost of canonical mode on Isaac: `handover` has no schema_v2 name, so it
is unreachable from rungs 1 and 2. Use --vocab backend to demo it.

JSON EXTRACTION POLICY -- still deliberately different from llm/extract.py:

  llm/extract.py     takes the LAST balanced {...} that parses, because the cot
                     prompt puts the answer on the final line and earlier braces
                     are reasoning fragments. A benchmark must produce a verdict
                     for every row.

  _extract_json here REFUSES when more than one candidate parses, because
                     multiple actions is ambiguity, not a menu. Small models
                     emit three-to-five step plans, and any positional rule
                     picks a wrong-but-legal action some of the time. A demo
                     must not move an arm on a guess.

Do NOT "unify" these. Changing either one changes either the measured numbers
or the demo's safety property. If you touch one, read the other first.
"""

import argparse, contextlib, json, os, re, sys, time, uuid

# ---------------------------------------------------------------------------
# OFFLINE BY DEFAULT -- set BEFORE any HuggingFace import can read it.
#
# A demo machine may have no internet, and a supervisor should never watch a
# progress bar. Any HF asset that is not already cached must fail loudly and
# name itself, not reach out.
#
# This is an os.environ default, not a hard override: HF_HUB_OFFLINE=0 in the
# environment still wins, so a deliberate first-time download on a connected
# machine is one env var away. Nothing here touches the measurement path --
# bench/run_cell.py and bench/run_grid.py do not import this module.
# ---------------------------------------------------------------------------
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)   # so llm/, schema/, bench/ and asr/ resolve

BRIDGE_DIR = os.path.join(_REPO_ROOT, "bridge")
INBOX = os.path.join(BRIDGE_DIR, "inbox")
OUTBOX = os.path.join(BRIDGE_DIR, "outbox")
SEEN_JSONL = os.path.join(_REPO_ROOT, "data", "seen.jsonl")
SEEN_V2_JSONL = os.path.join(_REPO_ROOT, "data", "seen_v2.jsonl")
SCHEMA_V2 = os.path.join(_REPO_ROOT, "schema", "schema_v2.json")

# BRIDGE WAIT BOUND.
#
# Was 300 s. Technically bounded, but five minutes of dead air on camera is
# indistinguishable from a hang. 60 s instead:
#
#   ~17 s  slowest motion observed in the dry run
#    25 s  the backend's own hard ceiling -- _transport caps at STEP_LIMIT=1500
#          steps and move_to at MOVE_STEP_CAP=900 (sim/isaac_backend.py), so a
#          step-capped run cannot legitimately take much longer than this
#    60 s  3.5x the slowest observation, past the step cap with room for a
#          reset's world.reset() plus settle, and short enough that a dead
#          bridge hands the prompt back inside a minute
#
# This bounds MOTION only. The model call happens client-side before send() and
# is bounded separately by CLOUD_ATTEMPTS / CLOUD_BACKOFF. Override per run with
# --bridge-timeout.
RESULT_TIMEOUT = 60.0
POLL_SECONDS = 0.25


# ---------------------------------------------------------------------------
# GRAMMAR -- read from schema_v2.json, never retyped.
#
# VERBS, TARGETS and FIELD_MASK used to be three hand-maintained literals in
# this file, duplicating schema/schema_v2.json with nothing checking that they
# agreed. Only `locations` is genuinely per-backend (see BACKENDS below); the
# verb set, the target set and the required-field mask are the same everywhere
# and now come from the one file that defines them.
# ---------------------------------------------------------------------------
def _load_grammar(path=SCHEMA_V2):
    with open(path, "r", encoding="utf-8") as fh:
        s = json.load(fh)
    verbs = set(s["verbs"])
    targets = set(s["targets"])
    mask = {v: set(spec.get("required", [])) for v, spec in s["verbs"].items()}
    return verbs, targets, mask, set(s["locations"])


VERBS, TARGETS, FIELD_MASK, CANONICAL_LOCATIONS = _load_grammar()

STRATEGIES = ("zero_shot", "few_shot", "cot", "role_based", "rag")

BACKENDS = {
    "isaac": {
        # max_reach and nearest_free_cell are RUNTIME symbols: the model emits
        # the symbol and sim/isaac_backend.py resolves it against the live
        # scene, by the same rules as sim/workspace.py. They are legal Isaac
        # locations, so they belong in this set; the backend reports which cell
        # each became in a "resolved" field.
        "locations": ({"box_1", "box_2", "box_3", "handover",
                       "max_reach", "nearest_free_cell"}
                      | set("cell_%d_%d" % (r, c)
                            for r in range(3) for c in range(3))),
        "location_text": ("box_1, box_2, box_3, handover, max_reach, "
                          "nearest_free_cell, "
                          "cell_R_C where R and C are 0-2"),
        # Every symbol named above MUST be defined here. Listing a symbol with
        # no definition is the defect audit/V3_VS_V4.md section 4 prices at a
        # 55.1% failure rate for nearest_free_cell: the model is told a word
        # exists and never told what it means. Adding the two runtime symbols
        # to this backend re-created that defect until these lines were added.
        "location_notes": (
            "cell_R_C is a square on the table. R is the row: 0 is the row "
            "nearest the robot, 2 is the row furthest away. C is the column, "
            "0 to 2 across the table.\n"
            "handover is the point where the robot hands an object to a "
            "person. Use it for \"give it to me\" or \"hand it over\".\n"
            "max_reach is the square the arm can still touch at its farthest. "
            "Use it for \"as far as you can reach\".\n"
            "nearest_free_cell is any square that is free; the robot picks "
            "which one. Use it for \"anywhere you want\" or \"anywhere free\"."),
        # Isaac's grid maps 1:1 onto schema_v2's, so canonical prompting works
        # and is what makes legend + constrained decoding available.
        "default_vocab": "canonical",
    },
    "mujoco": {
        "locations": {"table_center", "table_left", "table_right",
                      "table_front", "table_back",
                      "box_1", "box_2", "box_3"},
        "location_text": ("box_1, box_2, box_3, table_center, table_left, "
                          "table_right, table_front, table_back"),
        "location_notes": (
            "table_left, table_right, table_front and table_back are fixed "
            "spots at the edges of the table; table_center is the middle."),
        # The V1 table_* names exist in no schema_v2 vocabulary. Canonical
        # prompting cannot reach them, so this backend stays on its own names.
        "default_vocab": "backend",
    },
}

BACKEND = "isaac"
VOCAB = "canonical"
LOCATIONS = set()
SCHEMA_TEXT = ""


def _set_backend(name, vocab="auto"):
    """Point the validator, the prompt and the vocabulary mode at one backend."""
    global BACKEND, VOCAB, LOCATIONS, SCHEMA_TEXT
    BACKEND = name
    VOCAB = BACKENDS[name]["default_vocab"] if vocab == "auto" else vocab
    LOCATIONS = BACKENDS[name]["locations"]
    required_lines = "\n".join(
        "  %-10s -> %s" % (v, ", ".join(sorted(FIELD_MASK[v])) or "(none)")
        for v in ("move_to", "pick_place", "fetch", "hold", "inspect", "stop", "reject"))
    SCHEMA_TEXT = """Verbs: %s
Targets (objects): %s
Locations: %s

What the location symbols mean:
%s

Required fields per verb (emit EXACTLY these keys plus "verb", no others):
%s

Object colours: rubiks_cube is red, bottle is blue, toy_car is yellow.

Emit EXACTLY ONE JSON object. Do not emit a plan or a sequence of steps.""" % (
        ", ".join(sorted(VERBS)), ", ".join(sorted(TARGETS)),
        BACKENDS[name]["location_text"],
        BACKENDS[name]["location_notes"], required_lines)


_set_backend(BACKEND)


# ---------------------------------------------------------------------------
# CLIENT ROSTER -- llm.clients.all_clients(), not a hardcoded list.
#
# This file used to carry its own MODELS dict and dispatch on {"kind": ...},
# which meant rungs 3 and 4 were unreachable from the bridge demo: the two
# merged LoRA models and the DistilBERT classifier simply were not in it. The
# roster is now the same one the harness and sim/run_demo.py use, so the demo
# cannot drift out of sync with what was measured.
#
# `rules` is not a model and is kept outside the roster: it takes the command
# string directly, has no prompt, and exists so the bridge can be exercised with
# no model server at all.
# ---------------------------------------------------------------------------
BASE_STRATEGIES = STRATEGIES
# Rung 3 runs the FULL set. This was ("zero_shot",) until the V5 measurement, on
# the belief that the adapters had only ever been measured zero-shot. That is
# false: results_v5_qwen3.csv.manifest.json and results_v5_qwen05.csv.manifest.json
# each record 10 cells -- both adapters x all five strategies -- and those results
# are published. The client is not entitled to forbid a combination the
# measurement reports.
#
# Best cell is still zero_shot (0.958); rag is the worst thing in that row
# (0.745). Unlocked for consistency, not because every option is worth demoing.
LORA_STRATEGIES = STRATEGIES
CLASSIFIER_STRATEGIES = ("classifier",)

RULES = "rules"

_ROSTER = None


def roster():
    """[client] from llm.clients.all_clients(), constructed once."""
    global _ROSTER
    if _ROSTER is None:
        from llm.clients import all_clients
        _ROSTER = list(all_clients())
    return _ROSTER


def strategies_for(name):
    """Allowed strategies for a client, keyed on its .name.

    THE ONLY per-model strategy table in the repo. sim/run_demo.py imports this
    function rather than keeping a second copy -- two copies is how the rung-3
    entry went stale for a whole measurement round.

    bench/run_grid.py has no equivalent table and never did: it validates
    strategy NAMES against llm.prompts.ALL_STRATEGIES (run_grid.py:238) and then
    takes the full model x strategy cross-product (run_grid.py:282). So this is
    not a mirror of the harness. It is a DEMO-SIDE GUARD, and it may only forbid
    combinations that cannot produce a meaningful answer at all:

      classifier  llm/prompts.py:145 returns the BARE COMMAND for the
                  "classifier" strategy. DistilBERT is an encoder with three
                  classification heads and no prompt surface; handing it a schema
                  prompt yields a confident WRONG INDEX, not an error. Silent
                  failure is precisely what a guard is for.
      rules       a keyword baseline. There is no prompt to vary.

    Rung 3 is NOT such a case -- see LORA_STRATEGIES above.
    """
    if name == RULES:
        return ("zero_shot",)
    if name.startswith("classifier:"):
        return CLASSIFIER_STRATEGIES
    if name.endswith("-lora"):
        return LORA_STRATEGIES
    return BASE_STRATEGIES


def rung_of(name):
    if name == RULES:
        return "rules"
    if name.startswith("classifier:"):
        return "rung 4"
    if name.endswith("-lora"):
        return "rung 3"
    return "rung 1/2"


def model_names():
    return [RULES] + [c.name for c in roster()]


def resolve_model(arg):
    """Full name ('ollama:qwen2.5:3b') or the bare name ('qwen2.5:3b')."""
    if arg == RULES:
        return RULES
    for client in roster():
        if client.name == arg:
            return client
    for client in roster():
        bare = client.name.split(":", 1)[1] if ":" in client.name else client.name
        if bare == arg:
            return client
    return None


def uses_canonical(name):
    """True when this client is prompted in schema_v2 vocabulary.

    Rungs 3 and 4 are ALWAYS canonical and this is not a choice: the LoRA
    adapters were fine-tuned on llm/prompts.py zero-shot prompts over schema_v2,
    and the classifier's label space IS schema_v2 -- it has no prompt to change.
    Rungs 1 and 2 follow --vocab.
    """
    if name == RULES:
        return False        # the rules fallback emits backend names directly
    if rung_of(name) in ("rung 3", "rung 4"):
        return True
    return VOCAB == "canonical"


# ---------------------------------------------------------------------------
# CANONICAL -> BACKEND TRANSLATION
#
# Canonically-prompted output (schema_v2, "action" key, grid_A1..grid_C3) is
# translated here, at the edge, into the attached backend's names.
#
# GRID MAPPING, AND ITS ONE UNVERIFIED HALF -- read before demoing Isaac:
#   rows    schema_v2 A/B/C run FAR -> NEAR (sim/workspace.py: "far -> near").
#           isaac_backend.cell_centre says "Row 0 is nearest the robot".
#           So A->2, B->1, C->0. This half is certain, from both docstrings.
#   columns schema_v2 1/2/3 run LEFT -> RIGHT. Isaac's col index runs -y -> +y.
#           Whether Isaac's +y is the same "right" the legend means has NOT been
#           confirmed against a running scene.
#
#   A MIRRORED COLUMN MAP PRODUCES PLAUSIBLE MOTION AND WRONG SEMANTICS WITH NO
#   ERROR. Nothing downstream can catch it. Verify physically:
#       --backend isaac --model rules --selftest
#   drives grid_A1 / grid_A3 / grid_C1 / grid_C3. A1 and A3 are the SAME ROW and
#   OPPOSITE COLUMNS, so they isolate the column half: if the arm reaches the
#   wrong side for those two while the rows are right, flip _GRID_COL below and
#   nothing else changes.
#
#   Also settle WHOSE left "left" is -- the legend says left/right without ever
#   naming the viewpoint. Write the answer into the thesis, not just the code.
# ---------------------------------------------------------------------------
_GRID_ROW = {"A": 2, "B": 1, "C": 0}      # far -> near  ==  isaac row 2 -> 0
_GRID_COL = {"1": 0, "2": 1, "3": 2}      # left -> right ==  isaac col 0 -> 2

GRID_TO_ISAAC = {
    "grid_%s%s" % (r, c): "cell_%d_%d" % (_GRID_ROW[r], _GRID_COL[c])
    for r in "ABC" for c in "123"
}


def _to_backend_location(loc):
    # box_1..3 exist in both vocabularies, and on Isaac so do max_reach and
    # nearest_free_cell -- the backend resolves those against the live scene.
    if loc in LOCATIONS:
        return loc
    if BACKEND == "isaac" and loc in GRID_TO_ISAAC:
        return GRID_TO_ISAAC[loc]
    if loc in CANONICAL_LOCATIONS:
        raise ValueError(
            "%r is a schema_v2 location with no %s equivalent. The grid cells "
            "and the runtime symbols exist in Isaac but not behind "
            "mujoco_bridge.py, which runs the V1 SimBackend."
            % (loc, BACKEND))
    raise ValueError("unknown location for backend %s: %r" % (BACKEND, loc))


def to_backend_action(obj):
    """Canonical schema_v2 object -> action in the attached backend's vocabulary.

    Renames "action" to "verb" and maps the location.

    ON EXTRA FIELDS. `modifiers` is DROPPED: schema_v2 carries it legitimately,
    the scorer ignores it when falsy, and the bridge protocol has no slot for
    it. Any OTHER field outside the verb's mask is REFUSED, not dropped.

    That distinction matters and was wrong here until it was tested. Dropping
    silently made {"action": "stop", "target": "bottle"} succeed as
    {"verb": "stop"} -- a field-mask violation quietly repaired. The
    measurement path does not repair it: schema/validate_object.py rejects it
    and the scorer counts the row wrong. A demo that is MORE FORGIVING than the
    harness will appear to work on exactly the outputs the results chapter
    counted as failures, which overstates how well the system runs live. The
    demo must fail where the measurement failed.
    """
    if not isinstance(obj, dict):
        raise ValueError("action must be a JSON object")
    verb = obj.get("verb", obj.get("action"))
    if verb not in FIELD_MASK:
        raise ValueError("illegal verb: %r" % (verb,))

    allowed = FIELD_MASK[verb] | {"verb", "action", "modifiers"}
    extra = sorted(set(obj) - allowed)
    if extra:
        raise ValueError("field mask violation for %s: %s not allowed (mask is %s)"
                         % (verb, extra, sorted(FIELD_MASK[verb]) or "no fields"))

    out = {"verb": verb}
    for key in FIELD_MASK[verb]:
        if obj.get(key) in (None, ""):
            raise ValueError("%s is missing required field %r" % (verb, key))
        out[key] = obj[key]
    if "location" in out:
        out["location"] = _to_backend_location(out["location"])
    return out


def validate(action):
    if not isinstance(action, dict):
        raise ValueError("action must be a JSON object")
    verb = action.get("verb")
    if verb not in VERBS:
        raise ValueError("illegal verb: %r" % (verb,))
    required = FIELD_MASK[verb]
    present = set(k for k in action if k != "verb")
    if present != required:
        raise ValueError("field mask violation for %s: expected %s, got %s"
                         % (verb, sorted(required), sorted(present)))
    if "target" in action and action["target"] not in TARGETS:
        raise ValueError("unknown target: %r" % (action["target"],))
    if "location" in action and action["location"] not in LOCATIONS:
        raise ValueError("unknown location for backend %s: %r"
                         % (BACKEND, action["location"]))
    return action


# ---------------------------------------------------------------------------
# CANONICAL PROMPTING -- the same code path sim/run_demo.py uses.
#
# The legend is applied once at startup by bench.run_cell.apply_legend(), which
# is what llm/prompts.py's build_prompt() reads. Applying it is the whole reason
# rungs 1 and 2 moved onto this path: the legend is written in grid_A1..grid_C3
# terms and cannot be expressed in cell_R_C terms at all.
# ---------------------------------------------------------------------------
_CANON_SCHEMA = None
_CANON_FEWSHOT = None
_RETRIEVER = None


def canonical_schema():
    global _CANON_SCHEMA
    if _CANON_SCHEMA is None:
        from llm.prompts import load_schema
        _CANON_SCHEMA = load_schema(SCHEMA_V2)
    return _CANON_SCHEMA


def warm_retriever():
    """Construct the Retriever without retrieving anything.

    Separate from canonical_examples() so callers can pay the one-off weight
    load outside a timed block without also firing a retrieval and printing a
    second `retrieved:` line for the same command.
    """
    global _RETRIEVER
    if _RETRIEVER is None:
        from llm.retrieve import Retriever
        print("  building retriever (once; ~2-3 s model load)...")
        _RETRIEVER = Retriever(quiet=True)
        print("  retrieval corpus: %s (%d rows)"
              % (_RETRIEVER.corpus_path, len(_RETRIEVER.rows)))
    return _RETRIEVER


def canonical_examples(strategy, command):
    """In-context examples for the canonical path, or None.

    few_shot: the three fixed exemplars from data/seen_v2.jsonl -- the same
    split bench/dataset.py hands the harness, and the same call sim/run_demo.py
    makes. NEVER held_out.

    rag: llm.retrieve.Retriever, selected per command. This replaces the
    token-overlap approximation this file used to carry, which was the reason
    the old docstring said RAG numbers here were not comparable.
    """
    global _CANON_FEWSHOT, _RETRIEVER
    if strategy == "few_shot":
        if _CANON_FEWSHOT is None:
            from bench.dataset import load_seen_split
            _CANON_FEWSHOT, _ = load_seen_split(k=3, seen_path=SEEN_V2_JSONL)
            print("  few-shot exemplars: %d from data/seen_v2.jsonl"
                  % len(_CANON_FEWSHOT))
        return _CANON_FEWSHOT
    if strategy == "rag":
        res = warm_retriever().retrieve(command, k=3)
        print("  retrieved:     %s  (%.0f ms)" % (res.ids, res.latency_s * 1000))
        return res.examples
    return None


def canonical_prompt(command, client, strategy):
    """The prompt the canonical path builds, legend already applied."""
    if client.name.startswith("classifier:"):
        return command      # classifier_client.complete(): the prompt IS the command
    from llm.prompts import build_prompt as canon_build
    if rung_of(client.name) == "rung 3":
        strategy = "zero_shot"      # the only prompt the adapters ever saw
    examples = canonical_examples(strategy, command)
    schema = canonical_schema()
    if examples is None:
        return canon_build(command, strategy, schema)
    return canon_build(command, strategy, schema, examples=examples)


# ---------------------------------------------------------------------------
# BACKEND-VOCABULARY PROMPTING -- --vocab backend only (MuJoCo by default).
# ---------------------------------------------------------------------------
FEW_SHOT_COMMON = [
    ("put the red cube in box 1",
     {"verb": "pick_place", "target": "rubiks_cube", "location": "box_1"}),
    ("bring me the bottle", {"verb": "fetch", "target": "bottle"}),
    ("pick up the yellow car and hold it",
     {"verb": "hold", "target": "toy_car"}),
    ("make me a sandwich", {"verb": "reject"}),
]


def _few_shot():
    """Examples must use the ACTIVE backend's locations."""
    extra = ("move to box 3", {"verb": "move_to", "location": "box_3"})
    return FEW_SHOT_COMMON + [extra]


_SEEN_CACHE = None


def _load_seen():
    rows = []
    if not os.path.exists(SEEN_JSONL):
        return rows
    with open(SEEN_JSONL, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "command" in obj:
                    rows.append(obj)
            except json.JSONDecodeError:
                continue
    return rows


def _normalise_label(row):
    """seen.jsonl label -> active field-mask action, or None.

    Stored rows look like:
      {"command": ..., "expected": {"action": "pick_place", "target": ...,
                                    "location": ..., "modifiers": []}}
    The mask uses "verb" not "action" and has no "modifiers" field, so raw
    rows injected as RAG examples teach the model the wrong shape.
    """
    label = row.get("expected") or row.get("action") or row
    if not isinstance(label, dict):
        return None
    if "verb" not in label and "action" in label:
        label = dict(label)
        label["verb"] = label.pop("action")
    if "verb" not in label:
        return None
    verb = label["verb"]
    if verb not in FIELD_MASK:
        return None
    clean = {"verb": verb}
    for key in FIELD_MASK[verb]:
        if key in label:
            clean[key] = label[key]
    try:
        return validate(clean)
    except ValueError:
        return None


def _retrieve(command, k=5):
    """Retrieve similar seen.jsonl examples, normalised to the active mask.

    NOTE: token-overlap retrieval, NOT the embedding store used in the measured
    RAG runs. Backend-vocabulary mode only -- the canonical path uses the real
    llm.retrieve.Retriever. Demo approximation; do not quote.
    """
    global _SEEN_CACHE
    if _SEEN_CACHE is None:
        _SEEN_CACHE = _load_seen()
    if not _SEEN_CACHE:
        return _few_shot()

    q = set(re.findall(r"[a-z0-9]+", command.lower()))
    scored = []
    for row in _SEEN_CACHE:
        toks = set(re.findall(r"[a-z0-9]+", row["command"].lower()))
        if not toks:
            continue
        scored.append((len(q & toks) / (len(q | toks) ** 0.5), row))
    scored.sort(key=lambda t: -t[0])

    out = []
    for _, row in scored:
        clean = _normalise_label(row)
        if clean is None:
            continue
        out.append((row["command"], clean))
        if len(out) >= k:
            break
    return out or _few_shot()


def build_prompt(command, strategy):
    base = ("You convert robot commands into a JSON action object.\n"
            "Reply with JSON only. No prose, no markdown, no code fences.\n\n"
            + SCHEMA_TEXT)
    if strategy == "zero_shot":
        return base, command
    if strategy in ("few_shot", "rag"):
        examples = _few_shot() if strategy == "few_shot" else _retrieve(command)
        lines = ["%s -> %s" % (c, json.dumps(a)) for c, a in examples]
        return base + "\n\nExamples:\n" + "\n".join(lines), command
    if strategy == "cot":
        return (base + "\n\nReason briefly about which single verb applies. "
                "Then, on the FINAL line, output only that one JSON object."), command
    if strategy == "role_based":
        return ("You are a strict robot command parser. You do not chat, "
                "explain, or apologise. You emit exactly one JSON action "
                "object and nothing else.\n\n" + SCHEMA_TEXT), command
    raise ValueError("unknown strategy: %r" % (strategy,))


# ---------------------------------------------------------------------------
# CONSTRAINED DECODING
#
# Same converters sim/run_demo.py uses, from llm/schema_json.py. Ollama takes
# the full per-verb oneOf form; OpenAI rejects oneOf at the root, so it gets the
# flat form and the field mask is enforced by validate() downstream instead.
# Constrained decoding is the belt; the validator is the braces.
#
# TWO DIFFERENCES from run_demo.py, both forced by this file's shape:
#
#   1. The schema is attached to EVERY eligible client in the roster, not just
#      the one chosen at startup, because this file lets you switch models mid
#      session with `model <name>`. Attaching to one client would leave the
#      others silently unconstrained after a switch.
#
#   2. In --vocab backend mode the location enum must come from the BACKEND's
#      vocabulary, and the verb key is "verb" rather than schema_v2's "action".
#      _to_backend_schema() re-points both. It also drops "modifiers", which
#      schema_v2 allows but validate() rejects for bridge actions.
# ---------------------------------------------------------------------------
def _rename_action_key(branch):
    props = {("verb" if k == "action" else k): v
             for k, v in branch["properties"].items()}
    props.pop("modifiers", None)
    req = ["verb" if k == "action" else k for k in branch["required"]]
    out = dict(branch)
    out["properties"] = props
    out["required"] = req
    return out


def _to_backend_form(js):
    """schema_v2-shaped JSON Schema -> the shape backend-vocabulary prompts ask for."""
    if "oneOf" in js:
        return {"type": "object", "oneOf": [_rename_action_key(b) for b in js["oneOf"]]}
    return _rename_action_key(js)


def apply_constraints(mode):
    """Attach or clear client.json_schema across the whole roster."""
    from llm.schema_json import to_json_schema, to_flat_json_schema
    canon = canonical_schema()
    backend_src = {"verbs": canon["verbs"],
                   "targets": sorted(TARGETS),
                   "locations": sorted(LOCATIONS)}
    attached = []
    for client in roster():
        if mode == "off":
            client.json_schema = None
            continue
        canonical = uses_canonical(client.name)
        src = canon if canonical else backend_src
        if client.name.startswith("openai:"):
            js = to_flat_json_schema(src)
        elif client.name.startswith("ollama:"):
            js = to_json_schema(src)
        else:
            client.json_schema = None    # classifier has no decoder to constrain
            continue
        client.json_schema = js if canonical else _to_backend_form(js)
        attached.append(client.name)
    return attached


def _candidate_objects(text):
    """Every balanced {...} block, skipping braces inside string literals."""
    out, depth, start, in_str, esc = [], 0, -1, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    out.append(text[start:i + 1])
                    start = -1
    return out


def _extract_json(text, key="verb"):
    """Pull the action object out of a model reply.

    If the model emits MULTIPLE candidate actions, that is ambiguity, not a
    menu. Selecting by position, or by 'first that validates', silently picks
    a wrong-but-legal action: small models emit three-to-five step plans and
    any positional rule gets it wrong some of the time.

    This is deliberately NOT llm/extract.py's rule -- see the module docstring.
    `key` is "action" for canonical schema_v2 replies and "verb" for
    backend-vocabulary replies.
    """
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    cleaned = re.sub(r"<\|.*?\|>", "", cleaned)
    cleaned = cleaned.replace("```json", "").replace("```", "")
    blocks = _candidate_objects(cleaned)
    if not blocks:
        raise ValueError("no JSON object in model reply: %r" % (text[:300],))

    parsed = []
    for blk in blocks:
        try:
            obj = json.loads(blk)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and key in obj:
            parsed.append(obj)

    if not parsed:
        raise ValueError("no action object with a %r key: %r" % (key, text[:300],))
    if len(parsed) > 1:
        verbs = [str(p.get(key)) for p in parsed]
        raise ValueError("model emitted %d actions (%s) -- ambiguous, "
                         "expected exactly one" % (len(parsed), ", ".join(verbs)))
    return parsed[0]


def call_rules(command):
    """Deterministic fallback. Colour words checked BEFORE nouns: 'blue cube'
    must resolve to the blue object, not to whichever rule mentions 'cube'."""
    t = command.lower().strip()
    if t in ("stop", "halt"):
        return {"verb": "stop"}

    target = None
    if "blue" in t:
        target = "bottle"
    elif "yellow" in t:
        target = "toy_car"
    elif "red" in t:
        target = "rubiks_cube"
    elif "bottle" in t:
        target = "bottle"
    elif "car" in t:
        target = "toy_car"
    elif "cube" in t or "rubik" in t:
        target = "rubiks_cube"

    location = None
    for n in (1, 2, 3):
        if ("box %d" % n) in t or ("box_%d" % n) in t:
            location = "box_%d" % n
            break

    if location is None and BACKEND == "isaac":
        for r in range(3):
            for c in range(3):
                if ("cell %d %d" % (r, c)) in t or ("cell_%d_%d" % (r, c)) in t:
                    location = "cell_%d_%d" % (r, c)
        if location is None and ("hand" in t or "handover" in t):
            location = "handover"

    if location is None and BACKEND == "mujoco":
        for word, loc in (("center", "table_center"),
                          ("centre", "table_center"),
                          ("left", "table_left"),
                          ("right", "table_right"),
                          ("front", "table_front"),
                          ("back", "table_back")):
            if word in t:
                location = loc
                break

    if any(w in t for w in ("bring", "fetch", "give")) and target:
        return {"verb": "fetch", "target": target}
    if "hold" in t and target:
        return {"verb": "hold", "target": target}
    if ("inspect" in t or "look at" in t) and target:
        return {"verb": "inspect", "target": target}
    if target and location:
        return {"verb": "pick_place", "target": target, "location": location}
    if location and not target:
        return {"verb": "move_to", "location": location}
    return {"verb": "reject"}


def query(command, model, strategy):
    """Return (action_dict, raw_reply_or_None). Result is in BACKEND vocabulary."""
    client = resolve_model(model)
    if client is None:
        raise ValueError("unknown model: %r" % (model,))
    if client is RULES:
        return call_rules(command), None

    # complete_with_retry() is transparent when the first attempt succeeds and
    # re-raises non-transport errors immediately, so a local model is unaffected
    # in practice -- but an Ollama server that went down mid-demo gets the same
    # three tries as the cloud, which is what you want at 10pm the night before.
    if uses_canonical(client.name):
        raw = complete_with_retry(client, canonical_prompt(command, client, strategy))
        return to_backend_action(_extract_json(raw, key="action")), raw

    system, user = build_prompt(command, strategy)
    raw = complete_with_retry(client, system + "\n\n" + user)
    return _extract_json(raw, key="verb"), raw


def to_action(command, model, strategy):
    return query(command, model, strategy)[0]


def listen(seconds=None, mic_substring="Seiren", model="small.en"):
    """Record and transcribe. Thin wrapper over the shared asr/voice.py.

    seconds=None -> push-to-talk: Enter to start, Enter to stop. This is the
    DEFAULT and the right mode for a live demo, because the speaker decides
    when the sentence is finished. The previous fixed 5-second window cut
    people off mid-command and forced them to pad short ones with silence.

    seconds=N -> fixed N-second window, kept for scripted or unattended runs
    where there is nobody to press Enter.

    `model` is Whisper's size. asr/voice.py defaults to "base"; this client
    asks for "small.en", which is more accurate on short English commands.
    CHECK sim/run_demo.py PASSES THE SAME ONE -- a different ASR model on the
    two demo paths is a parity gap, and it is the kind that shows up as one
    demo mishearing a word the other gets right.
    """
    from asr import voice
    text, asr_s = voice.listen(model=model, mic=mic_substring,
                               seconds=seconds, quiet=True)
    return (text or ""), asr_s


def send(action, timeout=None):
    # Resolved at CALL time, not def time. --bridge-timeout rebinds the module
    # global, and a default argument would have captured the old value at import
    # and silently ignored the flag.
    if timeout is None:
        timeout = RESULT_TIMEOUT
    os.makedirs(INBOX, exist_ok=True)
    os.makedirs(OUTBOX, exist_ok=True)
    name = "%s_%s.json" % (time.strftime("%H%M%S"), uuid.uuid4().hex[:6])
    tmp = os.path.join(INBOX, name + ".tmp")
    final = os.path.join(INBOX, name)
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(action, fh, indent=2)
    os.replace(tmp, final)
    result_path = os.path.join(OUTBOX, name)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(result_path):
            with open(result_path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        time.sleep(POLL_SECONDS)
    raise TimeoutError("no result after %ss -- is a bridge running, "
                       "and is it watching %s?" % (timeout, INBOX))


# ---------------------------------------------------------------------------
# DEMO HYGIENE: quiet loads, offline assets, bounded cloud calls
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def quiet_transformers():
    """Silence the transformers LOAD REPORT for the duration of one load.

    DistilBERT prints a five-row "UNEXPECTED key" table on first use --
    vocab_transform, vocab_layer_norm, vocab_projector. They are the masked-LM
    head of distilbert-base-uncased, which a 3-head classifier legitimately does
    not use, so the report is noise. It lands in screen recordings.

    Scoped to the `with` block and restored afterwards. This is NOT a global
    warning filter and NOT warnings.simplefilter("ignore"): it moves the
    transformers logger's own verbosity and puts it back. Anything logged
    outside the block, by any library, is unaffected.
    """
    try:
        from transformers.utils import logging as hf_logging
    except Exception:
        yield                      # transformers absent; nothing to quieten
        return
    previous = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()
    try:
        yield
    finally:
        hf_logging.set_verbosity(previous)


def warm_classifier(client, verbose=True):
    """Force the rung-4 weight load now, quietly, instead of mid-demo.

    The load is lazy by design (llm/classifier_client.py explains why), so
    without this the 268 MB load and its LOAD REPORT happen on the first
    classifier command -- on camera, inside the visible latency.
    """
    if client is None or client is RULES or not client.name.startswith("classifier:"):
        return True, "not a classifier"
    ensure = getattr(client, "_ensure_loaded", None)
    if ensure is None:
        return True, "no loader"
    t0 = time.perf_counter()
    try:
        with quiet_transformers():
            ensure()
    except FileNotFoundError as exc:
        return False, str(exc).splitlines()[0]
    except Exception as exc:
        return False, _offline_hint(exc)
    if verbose:
        print("  classifier weights loaded in %.1fs" % (time.perf_counter() - t0))
    return True, "loaded in %.1fs" % (time.perf_counter() - t0)


def _offline_hint(exc):
    """Turn an offline-cache miss into a message that names the missing asset.

    HF_HUB_OFFLINE is on by default here, so a missing asset raises instead of
    downloading. The raw exception buries the asset name in a paragraph about
    connection errors; this pulls it out.
    """
    text = "%s: %s" % (type(exc).__name__, exc)
    offline_markers = ("offline", "couldn't connect", "could not connect",
                       "local_files_only", "cached", "connection error")
    if not any(m in text.lower() for m in offline_markers):
        return text.splitlines()[0][:200]
    asset = None
    for pat in (r"'([\w\-./]+/[\w\-.]+)'", r'"([\w\-./]+/[\w\-.]+)"'):
        m = re.search(pat, text)
        if m:
            asset = m.group(1)
            break
    return ("asset not in the local cache: %s -- the demo runs with "
            "HF_HUB_OFFLINE=1 and will not download. Pre-cache it, or rerun "
            "with HF_HUB_OFFLINE=0 on a connected machine."
            % (asset or "<name not in the error text>"))


CLOUD_ATTEMPTS = 3
CLOUD_BACKOFF = (1.0, 3.0)          # seconds before attempts 2 and 3


def complete_with_retry(client, prompt):
    """client.complete() with a bounded retry on transport failure.

    The live dry run produced a bare `model error: Connection error.` and the
    demo stopped. Three attempts with a short backoff covers the transient case
    without making the audience wait: worst case 4 s of added delay.

    ONLY transport-shaped failures are retried. A refusal, a bad request or a
    schema rejection is deterministic -- retrying it wastes the audience's time
    and the money. NEVER swaps models: the printed message names a local
    alternative and the operator decides.
    """
    transport = ("connection", "timeout", "timed out", "temporarily",
                 "unavailable", "502", "503", "504", "rate limit", "overloaded")
    last = None
    for attempt in range(1, CLOUD_ATTEMPTS + 1):
        try:
            return client.complete(prompt)
        except Exception as exc:
            last = exc
            text = ("%s %s" % (type(exc).__name__, exc)).lower()
            if not any(t in text for t in transport):
                raise                      # deterministic; do not burn retries
            if attempt == CLOUD_ATTEMPTS:
                break
            wait = CLOUD_BACKOFF[min(attempt - 1, len(CLOUD_BACKOFF) - 1)]
            print("  cloud call failed (attempt %d/%d): %s -- retrying in %.0fs"
                  % (attempt, CLOUD_ATTEMPTS, type(last).__name__, wait))
            time.sleep(wait)
    raise ConnectionError(
        "%s is unreachable after %d attempts (%s: %s).\n"
        "  The demo does NOT switch models on your behalf. To continue locally, "
        "type:  model qwen2.5-3b-lora   (0.958 zero-shot, no network)"
        % (getattr(client, "name", "the cloud model"), CLOUD_ATTEMPTS,
           type(last).__name__, last))


# ---------------------------------------------------------------------------
# VALIDATOR-FEEDBACK RETRY (task 10) -- DEMO ONLY
#
# When the first answer looks wrong, the model is handed its OWN output plus a
# statement of what is inconsistent, and asked again. CODE NEVER EDITS THE JSON.
# Nothing here repairs, maps or patches a model's output.
#
# UNREACHABLE FROM THE MEASUREMENT PATH. bench/run_grid.py and bench/run_cell.py
# do not import this module. The only import of bench.llm_client anywhere in the
# repo is sim/run_demo.py, which takes four strategy constants and nothing else.
# query_with_retry() is called only from run_one() and preflight(), both in this
# file. A measured row cannot acquire a second attempt.
#
# WHY THIS IS NOT "REPAIR": the correction names the INCONSISTENCY, never the
# answer. It says move_to carries no target while the command named an object;
# it does not say "use pick_place with target=toy_car". If the model cannot act
# on that, the demo shows it failing, which is the honest outcome.
# ---------------------------------------------------------------------------
RETRY_ENABLED = True

# Nouns that name a schema target in ordinary speech. Written out rather than
# derived from schema_v2.json because the schema stores identifiers
# ("rubiks_cube"), not the words people say.
#
# COLOUR WORDS ARE DELIBERATELY ABSENT. The canonical v5 legend carries no
# colour glossary, so "the red cube" is genuinely unanswerable canonically and
# `reject` is the correct response. Treating a colour as a target noun would
# fire this detector on a right answer. (The backend-vocab SCHEMA_TEXT in this
# file does define colours; the detector ignores them in both modes so its
# behaviour does not depend on --vocab.)
TARGET_NOUNS = {
    "rubiks_cube": ("rubiks cube", "rubik's cube", "rubiks_cube", "rubik cube",
                    "the cube", "cube"),
    "bottle": ("bottle",),
    "toy_car": ("toy car", "toy_car", "the car", "car"),
}

# Verbs whose field mask has no "target", derived from the schema so it cannot
# drift.
_TARGETLESS = {v for v, mask in FIELD_MASK.items() if "target" not in mask}

# `reject` is EXCLUDED even though it takes no target. An impossible command
# usually still names an object ("put the bottle on the ceiling", "paint the
# bottle green"), and `reject` is the CORRECT answer to those. Firing a retry
# there asks the model to talk itself out of its safest answer, and refusal is
# the behaviour that most needs to be stable. move_to and stop still fire.
SEMANTIC_TRIGGER_VERBS = _TARGETLESS - {"reject"}


def names_target(command):
    """Which schema target this command names in plain words, or None.

    Longest phrase wins, so "toy car" is not matched as the bare "car" of a
    different entry and "rubiks cube" beats "cube".
    """
    low = " %s " % " ".join(re.sub(r"[^a-z0-9 ]", " ", command.lower()).split())
    best = (0, None)
    for target, phrases in TARGET_NOUNS.items():
        for phrase in phrases:
            if (" %s " % phrase) in low and len(phrase) > best[0]:
                best = (len(phrase), target)
    return best[1]


def diagnose(command, action, exc):
    """Why this first answer should be re-asked, as one factual sentence.

    None when the answer is fine. The string goes verbatim into the correction
    prompt, so it must state the INCONSISTENCY and never the fix.
    """
    if exc is not None:
        # ONLY CONTENT FAILURES ARE RETRYABLE.
        #
        # Everything in the parse/validate family raises ValueError:
        # _extract_json, to_backend_action and validate. A transport failure
        # arrives as ConnectionError from complete_with_retry, and anything else
        # (auth, a missing checkpoint) is its own type. None of those are things
        # the model can fix by being asked again.
        #
        # This was wrong in the first version and the Task 12 run caught it: an
        # unreachable cloud model was reported to the model as "your answer was
        # rejected by the schema validator", it burned a second set of three
        # cloud attempts, and on one command the re-ask produced a confidently
        # WRONG action where the honest outcome was "the model is unreachable".
        # A retry must never manufacture an answer out of a network failure.
        if not isinstance(exc, ValueError):
            return None
        text = str(exc)
        if "no JSON object" in text or "no action object" in text:
            return ("Your reply did not contain a JSON object. "
                    "The parser reported: %s" % text.split(":")[0].strip())
        if "ambiguous" in text:
            return ("Your reply contained more than one action object, so it is "
                    "ambiguous which one to execute. %s" % text)
        # Trigger 2: the validator's own words, not a paraphrase of them.
        return "Your answer was rejected by the schema validator: %s" % text

    # Trigger 3: valid JSON, legal verb, wrong action.
    if isinstance(action, dict):
        verb = action.get("verb")
        if verb in SEMANTIC_TRIGGER_VERBS and names_target(command) is not None:
            return ("Your answer used %s, which carries no target, but the "
                    "command names an object. If the object is to be relocated, "
                    "the verb is pick_place." % verb)
    return None


def correction_prompt(command, first_raw, reason):
    """Original command + the model's own JSON + what is inconsistent.

    Short and factual. No examples, no schema restatement, no correct answer --
    any of those would make this a different prompt strategy rather than a
    validator feeding its error back.
    """
    shown = (first_raw or "").strip()
    if len(shown) > 600:
        shown = shown[:600] + " ...[truncated]"
    return ("Command: %s\n\n"
            "Your previous answer:\n%s\n\n"
            "%s\n\n"
            "Answer again with exactly one JSON object."
            % (command, shown, reason))


def _retry_prompt_for(client, strategy, correction):
    """Wrap the correction in the SAME prompt shape attempt one used.

    Legend and constrained decoding are untouched between attempts: changing
    either would make attempt two a different experimental condition and the
    before/after comparison meaningless.
    """
    if uses_canonical(client.name):
        return canonical_prompt(correction, client, strategy)
    system, _ = build_prompt(correction, strategy)
    return system + "\n\n" + correction


def query_with_retry(command, model, strategy, allow_retry=True):
    """query(), plus at most ONE re-ask when the first answer looks wrong.

    Returns (action, raw, info). info is None when no retry happened, else a
    dict describing both attempts. Raises only if the SECOND attempt fails --
    the caller must never fall back to the first answer.
    """
    first_action = first_raw = None
    first_exc = None
    try:
        first_action, first_raw = query(command, model, strategy)
        validate(first_action)
    except Exception as exc:
        first_exc = exc

    reason = None if not allow_retry else diagnose(command, first_action, first_exc)
    if reason is None:
        if first_exc is not None:
            raise first_exc
        return first_action, first_raw, None

    client = resolve_model(model)

    # `rules` is a keyword baseline: re-asking returns the same string.
    if client is RULES:
        print("  note: no retry possible, `rules` has no prompt surface")
        if first_exc is not None:
            raise first_exc
        return first_action, first_raw, None

    # The classifier is an encoder over the command string with three argmax
    # heads. There is no prompt on which a correction could be expressed, so
    # "no retry is possible here" is the true state of the world, not an error.
    # Say so and let the first answer stand.
    if client.name.startswith("classifier:"):
        print("  note: no retry possible, the classifier has no prompt surface")
        if first_exc is not None:
            raise first_exc
        return first_action, first_raw, None

    print("  RETRY  first answer: %s"
          % (json.dumps(first_action) if first_action is not None
             else "<%s: %s>" % (type(first_exc).__name__, first_exc)))
    print("  RETRY  reason: %s" % reason)

    info = {"retried": True, "reason": reason, "first": first_action,
            "first_error": None if first_exc is None else str(first_exc)}

    try:
        prompt = _retry_prompt_for(client, strategy,
                                   correction_prompt(command, first_raw, reason))
        raw2 = complete_with_retry(client, prompt)
        action2 = (to_backend_action(_extract_json(raw2, key="action"))
                   if uses_canonical(client.name)
                   else _extract_json(raw2, key="verb"))
        validate(action2)
    except Exception as exc2:
        # Execute NOTHING. Falling back to the first answer would run output the
        # validator already refused, or an action the command contradicts.
        print("  RETRY  second answer ALSO failed: %s" % exc2)
        raise RuntimeError(
            "both attempts failed, nothing executed. first: %s | second: %s"
            % (first_exc or "semantic mismatch", exc2))

    print("  RETRY  second answer: %s" % json.dumps(action2))
    info["second"] = action2
    return action2, raw2, info


# ---------------------------------------------------------------------------
# STARTUP PRECONDITIONS (task 8) -- verify, do not assume.
#
# A demo that dies three commands in because Ollama was not running is
# avoidable. Each check is one pass/fail line in the banner. None of them is
# fatal: they report and let the operator decide, because a MuJoCo-less or
# bridge-less session is still useful for showing the language layer with
# --no-exec.
# ---------------------------------------------------------------------------
OK, BAD, WARN = "ok  ", "FAIL", "warn"


def check_ollama(required):
    """Is the server up, and are the tagged models actually pulled?"""
    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        r.raise_for_status()
        have = set(m["name"] for m in r.json().get("models", []))
    except Exception as exc:
        return BAD, ("ollama unreachable at localhost:11434 (%s). Start it with "
                     "`ollama serve`." % type(exc).__name__)
    # Ollama resolves a bare name to :latest; compare both ways.
    def present(tag):
        return tag in have or ("%s:latest" % tag) in have or any(
            h.split(":", 1)[0] == tag for h in have)
    missing = sorted(t for t in required if not present(t))
    if missing:
        return BAD, ("ollama up, %d models, MISSING: %s (`ollama pull` them)"
                     % (len(have), ", ".join(missing)))
    return OK, "ollama up, %d models, all %d required present" % (len(have), len(required))


def check_bridge():
    """Is anything consuming bridge/inbox?

    There is no handshake in the protocol, so this is inferred: drop a
    no-op control message and see whether it is picked up. A live bridge
    answers in well under a second; a dead one leaves the file sitting there,
    and we clean it up so the inbox does not accumulate probes.
    """
    probe = {"control": "ping"}
    os.makedirs(INBOX, exist_ok=True)
    os.makedirs(OUTBOX, exist_ok=True)
    name = "ping_%s.json" % uuid.uuid4().hex[:6]
    path = os.path.join(INBOX, name)
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(probe, fh)
    except Exception as exc:
        return BAD, "cannot write %s (%s)" % (INBOX, exc)

    deadline = time.time() + 2.0
    result_path = os.path.join(OUTBOX, name)
    while time.time() < deadline:
        if os.path.exists(result_path):
            # The bridge answers "unknown_control" for ping, which is a perfectly
            # good liveness signal: something read the file and replied.
            try:
                os.remove(result_path)
            except OSError:
                pass
            return OK, "bridge alive, consuming %s" % INBOX
        if not os.path.exists(path):
            return OK, "bridge alive (probe consumed)"
        time.sleep(0.1)
    try:
        os.remove(path)                      # do not leave probes in the inbox
    except OSError:
        pass
    return BAD, ("nothing is consuming %s -- start `python sim/isaac_bridge.py` "
                 "in the isaac6 env, or run with --no-exec" % INBOX)


def check_openai(model_name):
    if not model_name.startswith("openai:"):
        return None, None
    if os.environ.get("OPENAI_API_KEY"):
        return OK, "OPENAI_API_KEY set"
    return BAD, ("OPENAI_API_KEY not set and %s is selected -- put it in .env "
                 "or pick a local model" % model_name)


def check_offline():
    off = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
    return (OK if off else WARN), (
        "HF_HUB_OFFLINE=1, no downloads mid-demo" if off else
        "HF_HUB_OFFLINE is OFF -- a cache miss will try to download on camera")


def preconditions(model_name, execute=True):
    """Print the banner. Returns True when nothing failed."""
    print("\nstartup checks")
    rows = []

    needed = [c.name.split(":", 1)[1] for c in roster()
              if c.name.startswith("ollama:")]
    if model_name.startswith("ollama:"):
        needed = [model_name.split(":", 1)[1]]
    rows.append(("ollama", ) + check_ollama(needed))

    if execute:
        rows.append(("bridge", ) + check_bridge())
    else:
        rows.append(("bridge", WARN, "--no-exec: nothing will be sent"))

    state, msg = check_openai(model_name)
    if state:
        rows.append(("openai", state, msg))

    rows.append(("hf cache", ) + check_offline())

    for label, state, msg in rows:
        print("  [%s] %-9s %s" % (state, label, msg))
    failed = [r for r in rows if r[1] == BAD]
    if failed:
        print("  -> %d check(s) failed. The client still starts; fix them or "
              "use --no-exec." % len(failed))
    return not failed


def send_control(name, timeout=None):
    """Scene control over the bridge. NOT a schema verb and NOT a model action.

    Carries a top-level "control" key instead of "verb". sim/isaac_bridge.py
    checks for it BEFORE calling backend.dispatch(), so a control message never
    enters the verb path at all.

    A model cannot reach this. to_backend_action() builds every model-derived
    message from scratch as {"verb": ...} plus that verb's field-mask keys, and
    refuses any field outside the mask rather than dropping it -- so no model
    output can acquire a "control" key on the way out.
    """
    return send({"control": name}, timeout=timeout)


def do_reset():
    """Restore the scene to its start state. Returns True on success.

    Objects home, boxes empty, arm home -- one command, one meaning. Every clip
    starts from an identical state or the recordings are not comparable, and the
    arm's pose is part of that state.
    """
    print("  resetting scene (objects home, boxes empty, arm home)...")
    t0 = time.perf_counter()
    try:
        result = send_control("reset")
    except TimeoutError as exc:
        print("  reset TIMED OUT: %s" % exc)
        return False
    except Exception as exc:
        print("  reset failed to send: %s: %s" % (type(exc).__name__, exc))
        return False

    if not result.get("ok"):
        print("  reset REFUSED by the bridge: %s" % json.dumps(result))
        return False

    filled = [b for b, o in (result.get("occupancy") or {}).items() if o]
    print("  scene reset in %.1fs -- objects: %s | boxes: %s"
          % (time.perf_counter() - t0,
             ", ".join(result.get("objects", [])) or "none",
             ("STILL OCCUPIED " + ", ".join(filled)) if filled else "empty"))
    return True


def run_one(text, model, strategy, asr_s=0.0, execute=True, show_raw=False):
    print("\n> %s" % text)
    # Warm the retriever OUTSIDE the timer. Its first call loads embedding
    # weights (~2-3 s); leaving that inside llm_s inflates the first rag command
    # of every session by an order of magnitude and makes the printed latency a
    # lie. Retrieval per command is ~5 ms and stays inside the measurement,
    # which is correct -- it is part of the strategy's real cost.
    client = resolve_model(model)
    if (client is not None and client is not RULES
            and strategy == "rag" and uses_canonical(client.name)):
        warm_retriever()

    timings = {"asr_s": round(asr_s, 3)}
    raw = None
    retry_info = None
    t0 = time.perf_counter()
    try:
        action, raw, retry_info = query_with_retry(
            text, model, strategy, allow_retry=RETRY_ENABLED)
    except Exception as exc:
        print("  model error: %s" % exc)
        return None
    timings["llm_s"] = round(time.perf_counter() - t0, 3)
    if show_raw and raw is not None:
        print("  raw: %r" % (raw[:400],))
    t0 = time.perf_counter()
    try:
        action = validate(action)
    except ValueError as exc:
        print("  INVALID: %s" % exc)
        print("  parsed: %s" % json.dumps(action))
        return None
    timings["validate_s"] = round(time.perf_counter() - t0, 4)
    print("  action: %s" % json.dumps(action))
    if retry_info:
        print("  note: executing the SECOND answer; the first was %s"
              % (json.dumps(retry_info.get("first"))
                 if retry_info.get("first") is not None
                 else retry_info.get("first_error")))
    if not execute:
        print("  timing: %s" % json.dumps(timings))
        return action
    t0 = time.perf_counter()
    try:
        result = send(action)
    except TimeoutError as exc:
        print("  timeout: %s" % exc)
        return None
    timings["motion_s"] = round(time.perf_counter() - t0, 3)
    timings["total_s"] = round(sum(timings.values()), 3)
    if retry_info:
        # Recorded so a retry is inspectable after the fact. ADDITIVE: no
        # existing key changes name, type or meaning, and the demo path writes
        # no CSV, so no column anywhere is affected.
        result["retry"] = retry_info
    print("  result: %s" % json.dumps(result))
    print("  timing: %s" % json.dumps(timings))
    return result


# The pairs a demo actually shows. Deliberately short: the point is to catch
# breakage before an audience does, not to re-run the grid. Each is one model
# and one strategy, chosen as the cell that model is worth demoing on.
#
#   qwen2.5-3b-lora / zero_shot   0.958 -- the headline local result
#   gpt-4o-mini / rag             0.938 -- the headline cloud result
#   gpt-4o-mini / role_based      0.913 -- the cheap cloud point on the frontier
#   classifier:distilbert         0.747 at ~18 ms -- the latency argument
#   rules                         no model server; proves the bridge alone
PREFLIGHT_PAIRS = (
    ("ollama:qwen2.5-3b-lora", "zero_shot"),
    ("openai:gpt-4o-mini", "rag"),
    ("openai:gpt-4o-mini", "role_based"),
    ("classifier:distilbert", "classifier"),
    ("rules", "zero_shot"),
)

PREFLIGHT_PROBE = "put the rubiks cube in box 1"
PREFLIGHT_EXPECT = {"verb": "pick_place", "target": "rubiks_cube",
                    "location": "box_1"}


def preflight(pairs=PREFLIGHT_PAIRS, show_raw=False):
    """Exercise everything the demo will touch, without moving the arm.

    WHAT --selftest AND --no-exec ALREADY DO, and why this is still needed:

      --selftest  takes ONE model and probes every strategy it allows, never
                  executing, and already warms the retriever. It is the right
                  shape -- this reuses its probe and its never-send property --
                  but it covers one model per invocation, so five runs and five
                  cold starts are still five chances to fail on camera.
      --no-exec   only suppresses send(). It loads nothing ahead of time and
                  reports nothing.

    Neither loads DistilBERT's 268 MB, and neither prints a table. This does
    both, in one pass, across the pairs the demo will actually use.

    Cold starts eliminated here, so none of them lands on camera:
      * every Ollama model is called once, so its weights are resident
      * the retriever is built once (~2-3 s of HF weight load)
      * DistilBERT is loaded once, quietly (see quiet_transformers)
      * the cloud path is exercised, so a dead key or dead network shows now
    """
    print("\npreflight -- loading everything the demo touches, arm stays still")
    print("probe: %r" % PREFLIGHT_PROBE)

    # Warm the shared machinery first so its cost is not charged to whichever
    # pair happens to come first in the table.
    if any(s == "rag" for _, s in pairs):
        warm_retriever()
    for name, _ in pairs:
        client = resolve_model(name)
        if client is not None and client is not RULES:
            ok, msg = warm_classifier(client, verbose=False)
            if not ok:
                print("  classifier preload FAILED: %s" % msg)

    rows = []
    for name, strategy in pairs:
        client = resolve_model(name)
        if client is None:
            rows.append((name, strategy, "FAIL", 0.0, "not in the roster"))
            continue
        resolved = RULES if client is RULES else client.name
        if strategy not in strategies_for(resolved):
            rows.append((resolved, strategy, "FAIL", 0.0,
                         "strategy not legal for this model"))
            continue

        t0 = time.perf_counter()
        try:
            action, raw = query(PREFLIGHT_PROBE, resolved, strategy)
            action = validate(action)
            ms = (time.perf_counter() - t0) * 1000
            if action == PREFLIGHT_EXPECT:
                rows.append((resolved, strategy, "PASS", ms, json.dumps(action)))
            else:
                # A wrong-but-legal answer is not a broken demo. The model is
                # allowed to be wrong; that is what the results chapter measures.
                rows.append((resolved, strategy, "WRONG", ms, json.dumps(action)))
            if show_raw and raw:
                print("    raw[%s/%s]: %r" % (resolved, strategy, raw[:200]))
        except Exception as exc:
            ms = (time.perf_counter() - t0) * 1000
            rows.append((resolved, strategy, "FAIL", ms,
                         "%s: %s" % (type(exc).__name__, str(exc)[:120])))

    print("\n%-26s %-11s %-6s %9s  %s"
          % ("model", "strategy", "state", "ms", "answer / reason"))
    print("-" * 100)
    for name, strategy, state, ms, detail in rows:
        print("%-26s %-11s %-6s %9.0f  %s" % (name, strategy, state, ms, detail))

    n_fail = sum(1 for r in rows if r[2] == "FAIL")
    n_wrong = sum(1 for r in rows if r[2] == "WRONG")
    print("-" * 100)
    print("%d pairs: %d pass, %d wrong-but-working, %d BROKEN"
          % (len(rows), len(rows) - n_fail - n_wrong, n_wrong, n_fail))
    if n_fail:
        print("BROKEN means the pair raised. Fix those before demoing.")
    elif n_wrong:
        print("Nothing is broken. WRONG rows are the model being wrong, which "
              "is a measured property, not a fault.")
    return n_fail == 0


def selftest(model, show_raw=False):
    """Check every legal strategy emits ONE valid, correct action.

    few_shot and rag were silently broken in the MuJoCo demo while working in
    the measurement harness. This gate exists so that cannot recur unnoticed.

    On Isaac the grid corners are printed too: that is where the column half of
    GRID_TO_ISAAC would show up as mirrored if it is wrong.
    """
    client = resolve_model(model)
    if client is None:
        print("[selftest] unknown model %r" % (model,))
        return False
    name = RULES if client is RULES else client.name
    strats = strategies_for(name)

    print("[selftest] backend=%s vocab=%s model=%s rung=%s"
          % (BACKEND, "canonical" if uses_canonical(name) else "backend",
             name, rung_of(name)))
    if name != RULES:
        print("[selftest] constrained=%s"
              % ("on" if getattr(client, "json_schema", None) else "off"))
    all_ok = True

    # THE PROBE MUST MATCH THE ACTIVE VOCABULARY.
    #
    # "the red cube" is answerable only in backend mode, whose SCHEMA_TEXT
    # carries a colour glossary ("rubiks_cube is red"). The canonical legend in
    # llm/prompts.py has no colour mapping at all -- it is the measurement
    # legend, and the measured datasets name objects, not colours. Probing
    # canonically with a colour word makes `reject` the CORRECT answer and the
    # selftest reports a failure that is really a badly chosen probe.
    #
    # Do not "fix" this by adding colours to llm/prompts.py. That file is the
    # provenance of the 30,380 measured rows.
    if uses_canonical(name):
        probe = "put the rubiks cube in box 1"
    else:
        probe = "put the red cube in box 1"
    expect = {"verb": "pick_place", "target": "rubiks_cube", "location": "box_1"}
    print("  probe: %r" % probe)

    # Build the retriever BEFORE the timed calls. canonical_examples()
    # constructs it lazily inside query(), so on a rag run the one-off HF weight
    # load lands inside llm_s and reports gpt-4o-mini at ~10 s against a
    # measured p50 of 699 ms. The demo's whole point is a latency comparison;
    # a warm-up cost inside the visible number destroys it.
    if "rag" in strats and uses_canonical(name):
        warm_retriever()
    for strat in strats:
        raw = None
        try:
            action, raw = query(probe, name, strat)
            action = validate(action)
            if action == expect:
                print("  %-12s VALID   %s" % (strat, json.dumps(action)))
            else:
                all_ok = False
                print("  %-12s WRONG   %s  (expected %s)"
                      % (strat, json.dumps(action), json.dumps(expect)))
        except Exception as exc:
            all_ok = False
            print("  %-12s BROKEN  %s" % (strat, exc))
        if show_raw and raw is not None:
            print("       raw: %r" % (raw[:400],))

    if BACKEND == "isaac":
        print("\n  grid corner mapping (schema_v2 -> isaac):")
        for cell in ("grid_A1", "grid_A3", "grid_C1", "grid_C3"):
            print("    %-8s -> %-10s  (row: %s)"
                  % (cell, GRID_TO_ISAAC[cell],
                     "far" if cell[5] == "A" else "near"))
        print("    ROW half is verified from two docstrings. COLUMN half is not.")
        print("    Drive grid_A1 and grid_A3 -- same row, opposite columns, so")
        print("    they isolate the column axis. If the arm reaches the wrong")
        print("    side for those two, flip _GRID_COL in this file.")
        print("    Decide and record WHOSE left the legend means.")

    print("\n[selftest] %s" % ("all strategies valid and correct" if all_ok
                               else "FAILURES ABOVE"))
    return all_ok


def main():
    global RESULT_TIMEOUT, RETRY_ENABLED
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="isaac", choices=sorted(BACKENDS),
                   help="which simulator vocabulary to validate and execute against")
    p.add_argument("--vocab", default="auto", choices=["auto", "canonical", "backend"],
                   help="which vocabulary rungs 1/2 are PROMPTED in. auto = "
                        "canonical on isaac, backend on mujoco. canonical is "
                        "what makes the legend and constrained decoding "
                        "available; it cannot reach isaac's `handover`.")
    p.add_argument("--model", default=RULES,
                   help="client name, full ('ollama:qwen2.5:3b') or bare "
                        "('qwen2.5:3b'), or 'rules'. See --list.")
    p.add_argument("--strategy", default="zero_shot",
                   help="must be legal for the chosen model; see --list")
    p.add_argument("--list", action="store_true",
                   help="print the model/strategy table and exit")
    # ---- demo defaults, matched to sim/run_demo.py. NOT measurement settings.
    # Every measured row in this repo was produced with legend v3 and no
    # constraint; both remain reachable by flag.
    p.add_argument("--legend", default="v5", choices=["v3", "v4", "v4lite", "v5"],
                   help="prompt legend, applied to the canonical path. v5 "
                        "(default) is the demo legend: the only one passing all "
                        "four checks in audit/check_legend_v5.py. v3 is what "
                        "produced the 30,380 measured rows.")
    p.add_argument("--constrain", default="on", choices=["on", "off"],
                   help="constrained decoding via Ollama `format` / OpenAI "
                        "structured outputs. on by default for the demo; "
                        "measurements were all run off.")
    p.add_argument("--voice", action="store_true")
    p.add_argument("--seconds", type=float, default=None,
                   help="fixed recording window in seconds. Omitted (the "
                        "default) means PUSH-TO-TALK: Enter to start, Enter to "
                        "stop. Pass a number only for unattended runs.")
    p.add_argument("--whisper", default="small.en",
                   help="faster-whisper model size for voice input")
    p.add_argument("--mic", default="Seiren")
    p.add_argument("--once", default=None)
    p.add_argument("--script", default=None)
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--preflight", action="store_true",
                   help="load every model the demo uses, build the retriever, "
                        "load DistilBERT, probe each demo pair with execution "
                        "suppressed, print a pass/fail table, and exit")
    p.add_argument("--no-exec", action="store_true")
    p.add_argument("--raw", action="store_true")
    p.add_argument("--bridge-timeout", type=float, default=None,
                   help="seconds to wait for the bridge to answer before giving "
                        "the prompt back (default 60; see the RESULT_TIMEOUT "
                        "note in this file for how that was chosen)")
    p.add_argument("--no-retry", action="store_true",
                   help="disable the validator-feedback retry (on by default "
                        "for the demo; the measurement path cannot reach it)")
    p.add_argument("--skip-checks", action="store_true",
                   help="skip the startup precondition banner")
    args = p.parse_args()

    # Legend first: llm/prompts.py's build_prompt() reads whatever this sets, so
    # it must run before any prompt is built. Same ordering as sim/run_demo.py.
    from bench.run_cell import apply_legend
    legend_note = apply_legend(args.legend)

    _set_backend(args.backend, args.vocab)

    if args.list:
        print("\nmodel                      rung       vocab      strategies")
        for name in model_names():
            print("%-26s %-10s %-10s %s"
                  % (name, rung_of(name),
                     "canonical" if uses_canonical(name) else "backend",
                     ", ".join(strategies_for(name))))
        return

    # ---- GUARDS. Found by enumerating all 16 (backend, vocab, rung) paths and
    # asking which of the model's LEGAL outputs can actually be executed. Isaac
    # is reachable on all 8 of its paths; MuJoCo has two dead ones. Both failed
    # mid-command rather than at startup, which on stage is the worst possible
    # time to find out.
    if BACKEND == "mujoco" and VOCAB == "canonical":
        sys.exit(
            "--vocab canonical is not usable with --backend mujoco.\n"
            "  Canonical prompting emits schema_v2 locations (grid_A1..grid_C3,\n"
            "  max_reach, nearest_free_cell). Behind mujoco_bridge.py sits the V1\n"
            "  SimBackend, whose vocabulary is table_center / table_left /\n"
            "  table_right / table_front / table_back -- 11 of the 14 canonical\n"
            "  symbols have no equivalent, so only the three box_N commands would\n"
            "  run. Use --vocab backend, or --backend isaac.")

    client = resolve_model(args.model)
    if client is None:
        sys.exit("unknown --model %r. Available:\n  %s"
                 % (args.model, "\n  ".join(model_names())))
    name = RULES if client is RULES else client.name
    allowed = strategies_for(name)

    if BACKEND == "mujoco" and rung_of(name) in ("rung 3", "rung 4"):
        sys.exit(
            "%s cannot drive --backend mujoco from this client.\n"
            "  Rungs 3 and 4 are ALWAYS canonical -- the LoRA adapters were\n"
            "  fine-tuned on schema_v2 prompts and the classifier's label space\n"
            "  IS schema_v2, so neither has a prompt that could be switched to\n"
            "  table_* names. 11 of their 14 possible locations are unreachable\n"
            "  here. Use --backend isaac, or sim/run_demo.py, which drives the\n"
            "  MuJoCo backend in-process over the schema_v2 vocabulary." % name)

    if args.strategy not in allowed:
        sys.exit("--strategy %r is not valid for %s.\n  %s supports: %s\n"
                 "  (rung-3 LoRA models were trained and measured on zero_shot "
                 "only; the rung-4 classifier has no prompt at all.)"
                 % (args.strategy, name, name, ", ".join(allowed)))

    attached = apply_constraints(args.constrain)

    print("[client] backend  %s" % args.backend)
    print("[client] vocab    %s  (rungs 1/2; rungs 3/4 are always canonical)" % VOCAB)
    print("[client] legend   %s  -- %s" % (args.legend, legend_note))
    print("[client] constrain %s  (%d client%s)"
          % (args.constrain, len(attached), "" if len(attached) == 1 else "s"))
    print("[client] model    %s  (%s)" % (name, rung_of(name)))
    print("[client] strategy %s" % args.strategy)
    print("[client] voice    %s (whisper %s, mic ~%r)"
          % ("push-to-talk" if args.seconds is None
             else "fixed %.0fs window" % args.seconds, args.whisper, args.mic))
    print("[client] inbox    %s" % INBOX)
    if BACKEND == "isaac" and VOCAB == "canonical":
        print("[client] NOTE     `handover` is unreachable in canonical mode; "
              "use --vocab backend to demo it")

    execute = not args.no_exec

    if args.bridge_timeout is not None:
        RESULT_TIMEOUT = args.bridge_timeout
        print("[client] bridge wait %.0fs (overridden)" % RESULT_TIMEOUT)
    else:
        print("[client] bridge wait %.0fs" % RESULT_TIMEOUT)

    RETRY_ENABLED = not args.no_retry
    print("[client] retry    %s  (one re-ask on extract / schema / semantic "
          "failure)" % ("on" if RETRY_ENABLED else "off"))

    if not args.skip_checks:
        preconditions(name, execute=execute)

    # Pay the rung-4 weight load NOW, quietly, if the classifier is selected.
    # It is lazy by design, so otherwise it lands on the first command: 4 s of
    # dead air plus a five-row LOAD REPORT, both on camera, and both charged to
    # the visible latency of a model whose entire selling point is 18 ms.
    if client is not RULES:
        ok, msg = warm_classifier(client)
        if not ok:
            print("  [FAIL] classifier weights: %s" % msg)

    if args.preflight:
        return 0 if preflight(show_raw=args.raw) else 1

    if args.selftest:
        selftest(name, show_raw=args.raw)
        return

    if args.once:
        run_one(args.once, name, args.strategy, execute=execute,
                show_raw=args.raw)
        return

    if args.script:
        with open(args.script, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#"):
                    run_one(line, name, args.strategy, execute=execute,
                            show_raw=args.raw)
        return

    print("\nCommands: type text, 'v' to speak, 'reset' to restore the scene, "
          "'quit' to exit.")
    print("Switch on the fly: 'model classifier:distilbert' / 'strategy rag'")
    model, strategy = name, args.strategy
    while True:
        try:
            text = input("\n[%s/%s/%s]> " % (BACKEND, model, strategy)).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text.lower() in ("quit", "exit"):
            break
        if text.lower() == "reset":
            # Handled here, before anything reaches the language layer -- the
            # same place `model` and `strategy` are handled. The word "reset" is
            # never prompted, never scored and never sent as a verb.
            do_reset()
            continue
        if text.lower().startswith("model "):
            cand = text.split(None, 1)[1].strip()
            picked = resolve_model(cand)
            if picked is not None:
                cand_name = RULES if picked is RULES else picked.name
                if BACKEND == "mujoco" and rung_of(cand_name) in ("rung 3", "rung 4"):
                    # Same guard as startup. Without it, switching to the
                    # classifier mid-session on the MuJoCo backend leaves 11 of
                    # its 14 locations unexecutable and the next grid command
                    # dies in translation.
                    print("  %s is canonical-only and cannot drive the mujoco "
                          "backend; staying on %s" % (cand_name, model))
                    continue
                model = cand_name
                allowed = strategies_for(model)
                if strategy not in allowed:
                    strategy = allowed[0]
                    print("  strategy -> %s (only option for %s)" % (strategy, model))
                print("  model -> %s  (%s, %s vocab)"
                      % (model, rung_of(model),
                         "canonical" if uses_canonical(model) else "backend"))
                # Same reason as startup: never let the 268 MB load happen on
                # the first command after a switch.
                if picked is not RULES:
                    warmed, why = warm_classifier(picked)
                    if not warmed:
                        print("  [FAIL] classifier weights: %s" % why)
            else:
                print("  unknown model. options: %s" % model_names())
            continue
        if text.lower().startswith("strategy "):
            cand = text.split(None, 1)[1].strip()
            if cand in strategies_for(model):
                strategy = cand
                print("  strategy -> %s" % strategy)
            else:
                print("  %s supports: %s" % (model, list(strategies_for(model))))
            continue
        if text.lower() in ("v", "voice"):
            try:
                spoken, asr_s = listen(args.seconds, args.mic, args.whisper)
            except Exception as exc:
                print("  voice error: %s" % exc)
                continue
            if not spoken:
                print("  heard nothing")
                continue
            print("  heard (%.2fs): %s" % (asr_s, spoken))
            run_one(spoken, model, strategy, asr_s=asr_s, execute=execute,
                    show_raw=args.raw)
            continue
        run_one(text, model, strategy, execute=execute, show_raw=args.raw)


if __name__ == "__main__":
    main()
