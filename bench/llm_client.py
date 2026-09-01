"""
bench/llm_client.py

LLM-side half of the bridge: model selection, prompt strategies, text or
voice input, per-stage latency timing. Runs in conda env `thesis`.
Never imports isaacsim.

Works with EITHER simulator via --backend. The two have different location
vocabularies; prompting one while the other executes reproduces exactly the
silent symbol mismatch the V4 runs were about, so it is selected explicitly.

WARNING: this is the DEMO path, not the measurement harness. RAG here uses
token-overlap retrieval, not the embedding store used for the measured runs.
Do not quote timings or accuracy from this script in the thesis.

    python bench/llm_client.py --list
    python bench/llm_client.py --backend isaac  --model gpt-4o-mini --strategy rag
    python bench/llm_client.py --backend mujoco --model gpt-4o-mini --strategy rag
    python bench/llm_client.py --backend mujoco --model gpt-4o-mini --selftest
    python bench/llm_client.py --once "put the red cube in box 1"
    python bench/llm_client.py --script demo_script.txt


RELATIONSHIP TO sim/run_demo.py  (READ THIS BEFORE CHANGING EITHER FILE)
========================================================================
sim/run_demo.py is the OTHER demo client. It drives MuJoCo in-process through
llm/pipeline.py's run_once(); this file drives EITHER simulator out-of-process
over the bridge/ file protocol, so a model crash cannot kill the simulator.

They now SHARE:
  * the client roster          llm.clients.all_clients()   (rungs 1-4)
  * the verb/target/field-mask grammar, read from schema/schema_v2.json
  * the voice front-end        asr/voice.py

They deliberately DIFFER in two places, and both differences are load-bearing:

1. LOCATION VOCABULARY. run_demo speaks schema_v2 (grid_A1..grid_C3, plus the
   runtime symbols). This file speaks whichever simulator is attached: Isaac's
   cell_R_C + handover, or the V1 table_* names that sim/sim_backend.py behind
   mujoco_bridge.py actually implements. Neither is schema_v2, and unifying them
   would mean editing schema_v2.json -- which is the provenance of 30,380
   measured rows. See CANONICAL -> BACKEND TRANSLATION below for how rung-3/4
   output crosses that line.

2. JSON EXTRACTION POLICY -- the subtle one. The two extractors implement
   OPPOSITE rules for the same input, on purpose:

     llm/extract.py     takes the LAST balanced {...} that parses, because the
                        cot prompt puts the answer on the final line and earlier
                        braces are reasoning fragments or schema echoes. A
                        benchmark must produce a verdict for every row.

     _extract_json here REFUSES when more than one candidate parses, because
                        multiple actions is ambiguity, not a menu. Small models
                        emit three-to-five step plans, and any positional rule
                        picks a wrong-but-legal action some of the time. A demo
                        must not move a real arm on a guess.

   Do NOT "unify" these. Changing either one changes either the measured numbers
   or the demo's safety property. If you touch one, read the other first.
"""

import argparse, json, os, re, sys, time, uuid

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)   # so llm/, schema/ and asr/ resolve when run as a script

BRIDGE_DIR = os.path.join(_REPO_ROOT, "bridge")
INBOX = os.path.join(BRIDGE_DIR, "inbox")
OUTBOX = os.path.join(BRIDGE_DIR, "outbox")
SEEN_JSONL = os.path.join(_REPO_ROOT, "data", "seen.jsonl")
SCHEMA_V2 = os.path.join(_REPO_ROOT, "schema", "schema_v2.json")

RESULT_TIMEOUT = 300.0
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
        "locations": {"box_1", "box_2", "box_3", "handover"} | set(
            "cell_%d_%d" % (r, c) for r in range(3) for c in range(3)),
        "location_text": ("box_1, box_2, box_3, handover, "
                          "cell_R_C where R and C are 0-2"),
    },
    "mujoco": {
        "locations": {"table_center", "table_left", "table_right",
                      "table_front", "table_back",
                      "box_1", "box_2", "box_3"},
        "location_text": ("box_1, box_2, box_3, table_center, table_left, "
                          "table_right, table_front, table_back"),
    },
}

BACKEND = "isaac"
LOCATIONS = set()
SCHEMA_TEXT = ""


def _set_backend(name):
    """Point the validator and the prompt at one backend's vocabulary."""
    global BACKEND, LOCATIONS, SCHEMA_TEXT
    BACKEND = name
    LOCATIONS = BACKENDS[name]["locations"]
    required_lines = "\n".join(
        "  %-10s -> %s" % (v, ", ".join(sorted(FIELD_MASK[v])) or "(none)")
        for v in ("move_to", "pick_place", "fetch", "hold", "inspect", "stop", "reject"))
    SCHEMA_TEXT = """Verbs: %s
Targets (objects): %s
Locations: %s

Required fields per verb (emit EXACTLY these keys plus "verb", no others):
%s

Object colours: rubiks_cube is red, bottle is blue, toy_car is yellow.

Emit EXACTLY ONE JSON object. Do not emit a plan or a sequence of steps.""" % (
        ", ".join(sorted(VERBS)), ", ".join(sorted(TARGETS)),
        BACKENDS[name]["location_text"], required_lines)


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
LORA_STRATEGIES = ("zero_shot",)
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

    Mirrors sim/run_demo.py's table, for the same reason: the rung-3 adapters
    were trained and measured on bare zero-shot prompts, so a cot or few_shot
    prompt is off-distribution for them, and the rung-4 classifier is an encoder
    over the command string with no prompt surface at all.
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


# ---------------------------------------------------------------------------
# CANONICAL -> BACKEND TRANSLATION
#
# Rungs 3 and 4 cannot be prompted in a backend vocabulary: the LoRA adapters
# were fine-tuned on llm/prompts.py zero-shot prompts over schema_v2, and the
# classifier's label space IS schema_v2 -- it has no prompt to change. So they
# are run canonically (schema_v2, "action" key, grid_A1..grid_C3) and their
# output is translated here, at the edge, into the attached backend's names.
# Rungs 1 and 2 are prompted in the backend vocabulary directly, as before.
#
# GRID MAPPING, AND ITS ONE UNVERIFIED HALF -- read before demoing Isaac:
#   rows    schema_v2 A/B/C run FAR -> NEAR (sim/workspace.py: "far -> near").
#           isaac_backend.cell_centre says "Row 0 is nearest the robot".
#           So A->2, B->1, C->0. This half is certain, from both docstrings.
#   columns schema_v2 1/2/3 run LEFT -> RIGHT. Isaac's col index runs -y -> +y.
#           Whether Isaac's +y is the same "right" the legend means has NOT been
#           confirmed against a running scene -- Isaac is not installed on the
#           machine this was written on.
#   VERIFY BEFORE USE: `--backend isaac --model rules --selftest` drives all
#   four grid corners. If left/right come out mirrored, flip _COL below and
#   nothing else changes.
# ---------------------------------------------------------------------------
_GRID_ROW = {"A": 2, "B": 1, "C": 0}      # far -> near  ==  isaac row 2 -> 0
_GRID_COL = {"1": 0, "2": 1, "3": 2}      # left -> right ==  isaac col 0 -> 2

GRID_TO_ISAAC = {
    "grid_%s%s" % (r, c): "cell_%d_%d" % (_GRID_ROW[r], _GRID_COL[c])
    for r in "ABC" for c in "123"
}


def _to_backend_location(loc):
    if loc in LOCATIONS:
        return loc                      # box_1..3 exist in both vocabularies
    if BACKEND == "isaac" and loc in GRID_TO_ISAAC:
        return GRID_TO_ISAAC[loc]
    if loc in CANONICAL_LOCATIONS:
        raise ValueError(
            "%r is a schema_v2 location with no %s equivalent. Runtime symbols "
            "(max_reach, nearest_free_cell) are resolved by the in-process "
            "backend in sim/workspace.py and have no bridge protocol; the grid "
            "cells exist in Isaac but not behind mujoco_bridge.py, which runs "
            "the V1 SimBackend." % (loc, BACKEND))
    raise ValueError("unknown location for backend %s: %r" % (BACKEND, loc))


def to_backend_action(obj):
    """Canonical schema_v2 object -> action in the attached backend's vocabulary.

    Renames "action" to "verb", maps the location, and drops every field outside
    the mask (schema_v2 carries "modifiers"; the bridge protocol does not).
    """
    if not isinstance(obj, dict):
        raise ValueError("action must be a JSON object")
    verb = obj.get("verb", obj.get("action"))
    if verb not in FIELD_MASK:
        raise ValueError("illegal verb: %r" % (verb,))
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

    NOTE: token-overlap retrieval, NOT the embedding store used in the
    measured RAG runs. Demo approximation only.
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


def canonical_prompt(command, client):
    """The prompt rungs 3 and 4 were trained and measured on."""
    if client.name.startswith("classifier:"):
        return command      # classifier_client.complete(): the prompt IS the command
    from llm.prompts import build_prompt as canon_build, load_schema
    return canon_build(command, "zero_shot", load_schema(SCHEMA_V2))


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
    `key` is "verb" for backend-vocabulary replies and "action" for canonical
    schema_v2 replies from rungs 3 and 4.
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
    """Return (action_dict, raw_reply_or_None)."""
    client = resolve_model(model)
    if client is None:
        raise ValueError("unknown model: %r" % (model,))
    if client is RULES:
        return call_rules(command), None

    if rung_of(client.name) in ("rung 3", "rung 4"):
        # Canonical prompt, canonical output, translated at the edge.
        raw = client.complete(canonical_prompt(command, client))
        return to_backend_action(_extract_json(raw, key="action")), raw

    system, user = build_prompt(command, strategy)
    raw = client.complete(system + "\n\n" + user)
    return _extract_json(raw), raw


def to_action(command, model, strategy):
    return query(command, model, strategy)[0]


def listen(seconds=5.0, mic_substring="Seiren"):
    """Record and transcribe. Thin wrapper over the shared asr/voice.py.

    Kept as a function here so --voice keeps its fixed-window behaviour (a
    scripted bridge run has nobody to press Enter); sim/run_demo.py calls the
    same module with seconds=None for push-to-talk.
    """
    from asr import voice
    text, asr_s = voice.listen(model="small.en", mic=mic_substring,
                               seconds=seconds, quiet=True)
    return (text or ""), asr_s


def send(action, timeout=RESULT_TIMEOUT):
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


def run_one(text, model, strategy, asr_s=0.0, execute=True, show_raw=False):
    print("\n> %s" % text)
    timings = {"asr_s": round(asr_s, 3)}
    raw = None
    t0 = time.perf_counter()
    try:
        action, raw = query(text, model, strategy)
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
    print("  result: %s" % json.dumps(result))
    print("  timing: %s" % json.dumps(timings))
    return result


def selftest(model, show_raw=False):
    """Check every legal strategy emits ONE valid, correct action.

    few_shot and rag were silently broken in the MuJoCo demo while working in
    the measurement harness. This gate exists so that cannot recur unnoticed.

    On Isaac the grid corners are probed too: that is where the row/column
    mapping in GRID_TO_ISAAC would show up as mirrored if it is wrong.
    """
    client = resolve_model(model)
    if client is None:
        print("[selftest] unknown model %r" % (model,))
        return False
    name = RULES if client is RULES else client.name
    strats = strategies_for(name)

    probes = [("put the red cube in box 1",
               {"verb": "pick_place", "target": "rubiks_cube",
                "location": "box_1"})]
    if BACKEND == "isaac":
        for cell in ("grid_A1", "grid_A3", "grid_C1", "grid_C3"):
            probes.append((None, {"verb": "move_to",
                                  "location": GRID_TO_ISAAC[cell]}))

    print("[selftest] backend=%s model=%s rung=%s" % (BACKEND, name, rung_of(name)))
    all_ok = True

    probe, expect = probes[0]
    print("  probe: %r" % probe)
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

    if len(probes) > 1:
        print("\n  grid corner mapping (schema_v2 -> %s):" % BACKEND)
        for cell in ("grid_A1", "grid_A3", "grid_C1", "grid_C3"):
            mapped = GRID_TO_ISAAC[cell]
            print("    %-8s -> %-10s  (%s)" % (
                cell, mapped,
                "far" if cell[5] == "A" else "near"))
        print("    drive these four and confirm left/right are not mirrored;")
        print("    if they are, flip _GRID_COL in this file.")

    print("\n[selftest] %s" % ("all strategies valid and correct" if all_ok
                               else "FAILURES ABOVE"))
    return all_ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="isaac", choices=sorted(BACKENDS),
                   help="which simulator vocabulary to prompt and validate against")
    p.add_argument("--model", default=RULES,
                   help="client name, full ('ollama:qwen2.5:3b') or bare "
                        "('qwen2.5:3b'), or 'rules'. See --list.")
    p.add_argument("--strategy", default="zero_shot",
                   help="must be legal for the chosen model; see --list")
    p.add_argument("--list", action="store_true",
                   help="print the model/strategy table and exit")
    p.add_argument("--voice", action="store_true")
    p.add_argument("--seconds", type=float, default=5.0)
    p.add_argument("--mic", default="Seiren")
    p.add_argument("--once", default=None)
    p.add_argument("--script", default=None)
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--no-exec", action="store_true")
    p.add_argument("--raw", action="store_true")
    args = p.parse_args()

    _set_backend(args.backend)

    if args.list:
        print("\nmodel                      rung       strategies")
        for name in model_names():
            print("%-26s %-10s %s" % (name, rung_of(name),
                                      ", ".join(strategies_for(name))))
        return

    client = resolve_model(args.model)
    if client is None:
        sys.exit("unknown --model %r. Available:\n  %s"
                 % (args.model, "\n  ".join(model_names())))
    name = RULES if client is RULES else client.name
    allowed = strategies_for(name)
    if args.strategy not in allowed:
        sys.exit("--strategy %r is not valid for %s.\n  %s supports: %s\n"
                 "  (rung-3 LoRA models were trained and measured on zero_shot "
                 "only; the rung-4 classifier has no prompt at all.)"
                 % (args.strategy, name, name, ", ".join(allowed)))

    print("[client] backend  %s" % args.backend)
    print("[client] model    %s  (%s)" % (name, rung_of(name)))
    print("[client] strategy %s" % args.strategy)
    print("[client] inbox    %s" % INBOX)

    if args.selftest:
        selftest(name, show_raw=args.raw)
        return

    execute = not args.no_exec

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

    print("\nCommands: type text, 'v' to speak, 'quit' to exit.")
    print("Switch on the fly: 'model qwen2.5:3b' / 'strategy rag'")
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
        if text.lower().startswith("model "):
            cand = text.split(None, 1)[1].strip()
            picked = resolve_model(cand)
            if picked is not None:
                model = RULES if picked is RULES else picked.name
                allowed = strategies_for(model)
                if strategy not in allowed:
                    strategy = allowed[0]
                    print("  strategy -> %s (only option for %s)" % (strategy, model))
                print("  model -> %s" % model)
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
                spoken, asr_s = listen(args.seconds, args.mic)
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
