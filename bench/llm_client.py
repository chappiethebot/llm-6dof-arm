import argparse, json, os, re, time, uuid

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BRIDGE_DIR = os.path.join(_REPO_ROOT, "bridge")
INBOX = os.path.join(BRIDGE_DIR, "inbox")
OUTBOX = os.path.join(BRIDGE_DIR, "outbox")
SEEN_JSONL = os.path.join(_REPO_ROOT, "data", "seen.jsonl")

RESULT_TIMEOUT = 300.0
POLL_SECONDS = 0.25

MODELS = {
    "gpt-4o-mini": {"kind": "cloud"},
    "llama3.2:3b": {"kind": "ollama"},
    "qwen2.5:3b": {"kind": "ollama"},
    "qwen2.5:0.5b": {"kind": "ollama"},
    "deepseek-r1:7b": {"kind": "ollama"},
    "rules": {"kind": "rules"},
}
STRATEGIES = ("zero_shot", "few_shot", "cot", "role_based", "rag")

VERBS = {"move_to", "pick_place", "fetch", "hold", "inspect", "stop", "reject"}
TARGETS = {"rubiks_cube", "bottle", "toy_car"}
LOCATIONS = {"box_1", "box_2", "box_3", "handover"} | {
    "cell_%d_%d" % (r, c) for r in range(3) for c in range(3)
}
FIELD_MASK = {
    "move_to": {"location"},
    "pick_place": {"target", "location"},
    "fetch": {"target"},
    "hold": {"target"},
    "inspect": {"target"},
    "stop": set(),
    "reject": set(),
}

SCHEMA_TEXT = """Verbs: fetch, hold, inspect, move_to, pick_place, reject, stop
Targets (objects): bottle, rubiks_cube, toy_car
Locations: box_1, box_2, box_3, handover, cell_R_C where R and C are 0-2

Required fields per verb (emit EXACTLY these keys plus "verb", no others):
  move_to    -> location
  pick_place -> target, location
  fetch      -> target
  hold       -> target
  inspect    -> target
  stop       -> (none)
  reject     -> (none)

Object colours: rubiks_cube is red, bottle is blue, toy_car is yellow.

Emit EXACTLY ONE JSON object. Do not emit a plan or a sequence of steps."""

FEW_SHOT = [
    ("put the red cube in box 1",
     {"verb": "pick_place", "target": "rubiks_cube", "location": "box_1"}),
    ("bring me the bottle", {"verb": "fetch", "target": "bottle"}),
    ("move to box 3", {"verb": "move_to", "location": "box_3"}),
    ("pick up the yellow car and hold it",
     {"verb": "hold", "target": "toy_car"}),
    ("make me a sandwich", {"verb": "reject"}),
]


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
        raise ValueError("unknown location: %r" % (action["location"],))
    return action


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
    """seen.jsonl label -> Isaac field-mask action, or None.

    Stored rows look like:
      {"command": ..., "expected": {"action": "pick_place", "target": ...,
                                    "location": ..., "modifiers": []}}
    The Isaac mask uses "verb" not "action" and has no "modifiers" field, so
    raw rows injected as RAG examples teach the model the wrong shape.
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
    """Retrieve similar seen.jsonl examples, normalised to the Isaac mask.

    NOTE: token-overlap retrieval, NOT the embedding store used in the
    measured RAG runs. Demo approximation only.
    """
    global _SEEN_CACHE
    if _SEEN_CACHE is None:
        _SEEN_CACHE = _load_seen()
    if not _SEEN_CACHE:
        return FEW_SHOT

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
    return out or FEW_SHOT


def build_prompt(command, strategy):
    base = ("You convert robot commands into a JSON action object.\n"
            "Reply with JSON only. No prose, no markdown, no code fences.\n\n"
            + SCHEMA_TEXT)
    if strategy == "zero_shot":
        return base, command
    if strategy in ("few_shot", "rag"):
        examples = FEW_SHOT if strategy == "few_shot" else _retrieve(command)
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


def _candidate_objects(text):
    """Every balanced {...} block, skipping braces inside string literals."""
    out = []
    depth = 0
    start = -1
    in_str = False
    esc = False
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


def _extract_json(text):
    """Pull the action object out of a model reply.

    If the model emits MULTIPLE candidate actions, that is ambiguity, not a
    menu. Selecting by position, or by 'first that validates', silently picks
    a wrong-but-legal action: qwen2.5:3b emits three-to-five step plans, and
    any positional rule gets it wrong some of the time. Ambiguity is reported
    as a failure rather than resolved by guesswork.
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
        if isinstance(obj, dict) and "verb" in obj:
            parsed.append(obj)

    if not parsed:
        raise ValueError("no action object with a 'verb' key: %r"
                         % (text[:300],))
    if len(parsed) > 1:
        verbs = [str(p.get("verb")) for p in parsed]
        raise ValueError("model emitted %d actions (%s) -- ambiguous, "
                         "expected exactly one" % (len(parsed), ", ".join(verbs)))
    return parsed[0]


def call_cloud(system, user, model="gpt-4o-mini"):
    from openai import OpenAI
    client = OpenAI()
    resp = client.chat.completions.create(
        model=model, temperature=0.0,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}])
    return resp.choices[0].message.content


def call_ollama(system, user, model):
    import requests
    resp = requests.post(
        "http://localhost:11434/api/chat",
        json={"model": model, "stream": False,
              "options": {"temperature": 0.0},
              "messages": [{"role": "system", "content": system},
                           {"role": "user", "content": user}]},
        timeout=300)
    resp.raise_for_status()
    return resp.json()["message"]["content"]


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
    if location is None:
        for r in range(3):
            for c in range(3):
                if ("cell %d %d" % (r, c)) in t or ("cell_%d_%d" % (r, c)) in t:
                    location = "cell_%d_%d" % (r, c)
    if location is None and ("hand" in t or "handover" in t):
        location = "handover"
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
    kind = MODELS[model]["kind"]
    if kind == "rules":
        return call_rules(command), None
    system, user = build_prompt(command, strategy)
    raw = (call_cloud(system, user, model) if kind == "cloud"
           else call_ollama(system, user, model))
    return _extract_json(raw), raw


def to_action(command, model, strategy):
    return query(command, model, strategy)[0]


_WHISPER = None


def _find_mic(substring="Seiren"):
    """Select by name substring, not device index -- indices shift."""
    import sounddevice as sd
    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0 and substring.lower() in dev["name"].lower():
            return idx, dev["name"]
    default = sd.default.device[0]
    return default, sd.query_devices(default)["name"]


def listen(seconds=5.0, mic_substring="Seiren"):
    global _WHISPER
    import sounddevice as sd
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if _WHISPER is None:
        from faster_whisper import WhisperModel
        try:
            _WHISPER = WhisperModel("small.en", device="cpu",
                                    compute_type="int8")
        except Exception:
            print("  model not cached -- downloading once (HF_HUB_OFFLINE=0)")
            os.environ["HF_HUB_OFFLINE"] = "0"
            _WHISPER = WhisperModel("small.en", device="cpu",
                                    compute_type="int8")
            os.environ["HF_HUB_OFFLINE"] = "1"
    idx, name = _find_mic(mic_substring)
    print("  mic: %s" % name)
    print("  recording %.0fs -- speak now..." % seconds)
    rate = 16000
    audio = sd.rec(int(seconds * rate), samplerate=rate, channels=1,
                   dtype="float32", device=idx)
    sd.wait()
    t0 = time.perf_counter()
    segments, _ = _WHISPER.transcribe(audio.flatten(), language="en")
    text = " ".join(s.text for s in segments).strip()
    return text, time.perf_counter() - t0


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
    raise TimeoutError("no result after %ss -- is sim/isaac_bridge.py running, "
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
    """Check every strategy emits ONE valid, correct action.

    few_shot and rag were silently broken in the MuJoCo demo while working in
    the measurement harness. This gate exists so that cannot recur unnoticed.
    """
    probe = "put the red cube in box 1"
    expect = {"verb": "pick_place", "target": "rubiks_cube",
              "location": "box_1"}
    print("[selftest] model=%s probe=%r\n" % (model, probe))
    all_ok = True
    for strat in STRATEGIES:
        raw = None
        try:
            action, raw = query(probe, model, strat)
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
    print("\n[selftest] %s" % ("all strategies valid and correct" if all_ok
                               else "FAILURES ABOVE"))
    return all_ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="rules", choices=sorted(MODELS))
    p.add_argument("--strategy", default="zero_shot", choices=STRATEGIES)
    p.add_argument("--voice", action="store_true")
    p.add_argument("--seconds", type=float, default=5.0)
    p.add_argument("--mic", default="Seiren")
    p.add_argument("--once", default=None)
    p.add_argument("--script", default=None)
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--no-exec", action="store_true")
    p.add_argument("--raw", action="store_true")
    args = p.parse_args()

    print("[client] model    %s" % args.model)
    print("[client] strategy %s" % args.strategy)
    print("[client] inbox    %s" % INBOX)

    if args.selftest:
        selftest(args.model, show_raw=args.raw)
        return

    execute = not args.no_exec

    if args.once:
        run_one(args.once, args.model, args.strategy, execute=execute,
                show_raw=args.raw)
        return

    if args.script:
        with open(args.script, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#"):
                    run_one(line, args.model, args.strategy, execute=execute,
                            show_raw=args.raw)
        return

    print("\nCommands: type text, 'v' to speak, 'quit' to exit.")
    print("Switch on the fly: 'model qwen2.5:3b' / 'strategy rag'")
    model, strategy = args.model, args.strategy
    while True:
        try:
            text = input("\n[%s/%s]> " % (model, strategy)).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text.lower() in ("quit", "exit"):
            break
        if text.lower().startswith("model "):
            cand = text.split(None, 1)[1].strip()
            if cand in MODELS:
                model = cand
                print("  model -> %s" % model)
            else:
                print("  unknown model. options: %s" % sorted(MODELS))
            continue
        if text.lower().startswith("strategy "):
            cand = text.split(None, 1)[1].strip()
            if cand in STRATEGIES:
                strategy = cand
                print("  strategy -> %s" % strategy)
            else:
                print("  unknown strategy. options: %s" % list(STRATEGIES))
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
