"""sim/run_demo.py — unified demo: pick keyboard or voice input at startup.

Menu picks the INPUT front-end; everything downstream (run_once -> drive -> arm)
is shared, so both modes run identical motion code. llm_demo.py is left untouched
as the stable typed reference; this is additive.

Model + strategy are flags (voice adds no menu for them):
  Windows:  python -m sim.run_demo --model gpt-4o-mini --strategy zero_shot
  Mac:      python -m sim.run_demo --model qwen2.5:3b --strategy cot

V2: run from the REPO ROOT with `python -m sim.run_demo`, NOT from inside sim/.
The backend now imports sim.workspace for the nine grid cells, and that import
only resolves when the repo root is on sys.path. Schema selection follows the
backend: if the grid loaded, the V2 schema is used, so spoken commands may use
the same phrasings as the V2 datasets ("the far left corner of the table").
V1 location names still work in both modes.

macOS note: run under plain `python`, NOT `mjpython`. mjpython runs user code on
a non-main thread, which makes gymnasium's render_mode="human" window creation
crash ('NSWindow should only be instantiated on the main thread'). Plain python
keeps the viewer on the main thread. (The backend is also built before Whisper as
a belt-and-braces main-thread measure; harmless on Windows.)

ASR AND THE GRID: Whisper transcribes speech to text, and the grid phrasings are
ordinary English words, so no ASR change is needed. But spoken "A1" or "grid A1"
will NOT work -- the schema has no such phrasing and the model was never shown
one. Say the words: "the far left corner of the table".

RELATIONSHIP TO bench/llm_client.py  (READ BEFORE CHANGING EITHER FILE)
=======================================================================
bench/llm_client.py is the OTHER demo client. It drives EITHER simulator
out-of-process over the bridge/ file protocol; this file drives MuJoCo
in-process through llm/pipeline.py's run_once().

Shared: the client roster (llm.clients.all_clients()), the verb/target/field-mask
grammar from schema/schema_v2.json, and the voice front-end (asr/voice.py).

Deliberately different, and both differences are load-bearing:

  * LOCATION VOCABULARY. This file speaks schema_v2 (grid_A1..grid_C3 plus the
    runtime symbols). llm_client speaks whichever simulator is attached --
    Isaac's cell_R_C, or the V1 table_* names behind mujoco_bridge.py. Unifying
    them would mean editing schema_v2.json, which is the provenance of the
    30,380 measured rows.

  * JSON EXTRACTION. llm/extract.py, used here, takes the LAST balanced object
    that parses, because the cot prompt puts the answer on the final line and a
    benchmark must score every row. llm_client REFUSES when more than one
    parses, because a demo must not move an arm on a guess. Opposite rules, both
    correct for their context. Do not unify them.
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import sys, time, argparse
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
# Both are needed. _ROOT so `sim.workspace` resolves (that is how the backend
# finds the grid); _HERE so the bare `from full_grasp_backend import ...` below
# resolves. Adding both means the demo runs BOTH ways:
#     python -m sim.llm_demo      (from the repo root)
#     python llm_demo.py          (from inside sim/)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from llm.prompts import load_schema
from llm.pipeline import run_once
from llm.clients import all_clients
from bench.dataset import load_seen_split

# The MuJoCo-dependent imports are DEFERRED to load_sim(), not taken at module
# level. full_grasp_backend and llm_demo both pull in gymnasium_robotics ->
# mujoco, so importing them here made every code path require a working physics
# engine -- including `--list` and the model/strategy menu, which need no
# simulator at all. Deferring also means a blocked or missing MuJoCo produces a
# clear message at the point of use instead of an import traceback at startup.


def load_sim():
    """Import the MuJoCo-backed pieces. Returns (FullGraspBackend, HAS_GRID, helpers)."""
    try:
        from full_grasp_backend import FullGraspBackend, HAS_GRID
        from llm_demo import drive, describe, print_legend, draw_box_markers
    except OSError as e:
        sys.exit(f"cannot load the MuJoCo backend: {e}\n"
                 f"  The simulator is required to drive the arm. The model and "
                 f"strategy menu (--list) works without it.")
    return FullGraspBackend, HAS_GRID, (drive, describe, print_legend, draw_box_markers)

# --- audio config (proven Phase 4 steps 1-3) ---
# Channel count and sample-rate fallbacks live in asr/voice.py now; only the
# choices that are this demo's own stay here.
MIC_NAME_HINT = "Seiren"
WHISPER_MODEL = "base"
LANGUAGE      = "en"
CLIP_PATH     = Path(os.path.join(os.path.dirname(__file__), "voice_clip.wav"))


# ---------- client roster + per-client strategy constraints ----------
#
# The roster is built from llm.clients.all_clients() rather than a hardcoded
# list, so the demo cannot drift out of sync with the harness again. Previously
# this file had no model menu at all: `make_client` dispatched on whether the
# name started with "gpt-", which silently produced an OllamaClient for
# anything else -- including "classifier:distilbert", which would then fail at
# call time rather than at selection time.
#
# NOT EVERY CLIENT SUPPORTS EVERY STRATEGY. This file used to carry its own
# copy of that table. It no longer does: the copy went stale for a whole
# measurement round -- it forbade every rung-3 strategy except zero_shot while
# bench/run_grid.py was measuring all five, and the published V5 numbers include
# cells this demo refused to run. Two copies is how that happens.
#
# bench/llm_client.py is now the ONE table. Read its strategies_for() docstring
# for why the classifier and rules entries are load-bearing and rung 3 is not.
# The import is stdlib-only and never touches isaacsim or MuJoCo.
from bench.llm_client import (BASE_STRATEGIES, LORA_STRATEGIES,
                              CLASSIFIER_STRATEGIES, strategies_for)

# Strategies that need in-context examples supplied by the caller. build_prompt
# RAISES for these when `examples` is empty, which run_once records as
# prompt_error -- the arm silently never moves. See wire_examples().
NEEDS_EXAMPLES = {"few_shot", "rag"}


def build_roster():
    """[(client, allowed_strategies)] in all_clients() order."""
    return [(c, strategies_for(c.name)) for c in all_clients()]


def resolve_model(roster, arg):
    """Map a --model value to a client. Accepts the full name ('ollama:qwen2.5:3b')
    or the bare name the old flag used ('qwen2.5:3b', 'gpt-4o-mini')."""
    for client, _ in roster:
        if client.name == arg:
            return client
    for client, _ in roster:
        bare = client.name.split(":", 1)[1] if ":" in client.name else client.name
        if bare == arg:
            return client
    return None


def _fmt_rung(name):
    if name.startswith("classifier:"):
        return "rung 4"
    if name.endswith("-lora"):
        return "rung 3"
    return "rung 1/2"


def choose_client(roster):
    print("\nSelect model:")
    for i, (client, strats) in enumerate(roster, 1):
        only = f"  [{strats[0]} only]" if len(strats) == 1 else ""
        print(f"  {i}. {client.name:26} {_fmt_rung(client.name):9}{only}")
    while True:
        raw = input(f"Enter 1-{len(roster)}: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(roster):
            return roster[int(raw) - 1]
        print(f"  Please type a number from 1 to {len(roster)}.")


def choose_strategy(client, strats):
    # One option means there is nothing to ask. Skip the prompt entirely.
    if len(strats) == 1:
        print(f"\nStrategy: {strats[0]}  (only strategy {client.name} supports)")
        return strats[0]
    print(f"\nSelect strategy for {client.name}:")
    for i, s in enumerate(strats, 1):
        note = "  (needs examples; loaded automatically)" if s in NEEDS_EXAMPLES else ""
        print(f"  {i}. {s}{note}")
    while True:
        raw = input(f"Enter 1-{len(strats)}: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(strats):
            return strats[int(raw) - 1]
        print(f"  Please type a number from 1 to {len(strats)}.")


# ---------- input mode selection ----------
def choose_mode():
    print("\nSelect input mode:")
    print("  1. Keyboard (type commands)")
    print("  2. Voice (speak commands)")
    while True:
        choice = input("Enter 1 or 2: ").strip()
        if choice == "1":
            return "keyboard"
        if choice == "2":
            return "voice"
        print("  Please type 1 or 2.")


# ---------- voice front-end (only used if mode 2 chosen) ----------
# The mic resolver, the push-to-talk recorder and the Whisper block used to live
# here AND in bench/llm_client.py, and the two copies had already drifted: this
# one had push-to-talk but no offline-cache retry, that one had the retry but a
# fixed 5-second window. Both now call asr/voice.py, which is the union.
def get_voice_command():
    """(command_text, asr_ms), or (None, None) on near-silence."""
    from asr import voice
    text, asr_s = voice.listen(model=WHISPER_MODEL, mic=MIC_NAME_HINT,
                               seconds=None, save_to=CLIP_PATH,
                               language=LANGUAGE)
    if text is None:
        return None, None
    return text, asr_s * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                    help="client name, full ('ollama:qwen2.5:3b') or bare "
                         "('qwen2.5:3b'). Omit to pick from the menu; pass "
                         "--default to take the demo default without a menu.")
    ap.add_argument("--default", action="store_true",
                    help="use the demo default client (gpt-4o-mini / rag: the "
                         "top of the measured Pareto frontier at 0.924) and "
                         "skip both menus")
    ap.add_argument("--strategy", default=None,
                    help="must be legal for the chosen model; see --list")
    ap.add_argument("--list", action="store_true",
                    help="print the model/strategy table and exit")
    ap.add_argument("--schema", default="auto", choices=["auto", "v1", "v2"],
                    help="auto = v2 when the grid loaded, else v1")
    # ---- demo defaults. NOT measurement settings. -------------------------
    # `python -m sim.run_demo` with no flags gives the best demo: V5 legend,
    # constrained decoding on, gpt-4o-mini. Every measured row in this repo was
    # produced with legend v3 and no constraint; those remain reachable by flag.
    ap.add_argument("--legend", default="v5", choices=["v3", "v4", "v4lite", "v5"],
                    help="prompt legend. v5 (default) is the demo legend: it is "
                         "the only one passing all four checks in "
                         "audit/check_legend_v5.py. v3 is what produced the "
                         "30,380 measured rows.")
    ap.add_argument("--constrain", default="on", choices=["on", "off"],
                    help="constrained decoding via Ollama `format` / OpenAI "
                         "structured outputs. on by default for the demo; "
                         "measurements were all run off.")
    args = ap.parse_args()

    from bench.run_cell import apply_legend
    legend_note = apply_legend(args.legend)

    roster = build_roster()

    if args.list:
        print("\nmodel                      rung       strategies")
        for client, strats in roster:
            print(f"{client.name:26} {_fmt_rung(client.name):10} {', '.join(strats)}")
        return

    # --- flags: resolve and VALIDATE against the same table the menu uses ---
    client = strategy = None
    if args.model is not None:
        client = resolve_model(roster, args.model)
        if client is None:
            names = "\n  ".join(c.name for c, _ in roster)
            sys.exit(f"unknown --model {args.model!r}. Available:\n  {names}")
    if args.strategy is not None:
        if client is None:
            sys.exit("--strategy requires --model (a strategy is only legal "
                     "relative to a model). Use --list to see the pairs.")
        allowed = strategies_for(client.name)
        if args.strategy not in allowed:
            sys.exit(
                f"--strategy {args.strategy!r} is not valid for {client.name}.\n"
                f"  {client.name} supports: {', '.join(allowed)}\n"
                f"  (rung-3 LoRA models were trained and measured on zero_shot "
                f"only; the rung-4 classifier has no prompt at all.)")
        strategy = args.strategy

    mode = choose_mode()
    if client is None and args.default:
        client = resolve_model(roster, "openai:gpt-4o-mini")
        strategy = strategy or "rag"
        allowed = strategies_for(client.name)
        print(f"\nModel: {client.name} / {strategy}  (demo default)")
    elif client is None:
        client, allowed = choose_client(roster)
    else:
        allowed = strategies_for(client.name)
        print(f"\nModel: {client.name}  (from --model)")
    if strategy is None:
        strategy = choose_strategy(client, allowed)

    print(f"legend: {legend_note}")

    # few_shot/rag need in-context examples. build_prompt RAISES without them,
    # which run_once turns into a prompt_error and the arm never moves -- this
    # file previously hardcoded `examples = None` with a "wired later" note, so
    # both strategies were silently dead. Exemplars come from seen_v2 ONLY
    # (never held_out): the same split bench/dataset.py gives the harness.
    examples = None
    retriever = None
    if strategy == "few_shot":
        examples, _ = load_seen_split(k=3, seen_path=os.path.join(_ROOT, "data", "seen_v2.jsonl"))
        print(f"few-shot exemplars: {len(examples)} from data/seen_v2.jsonl")
    elif strategy == "rag":
        from llm.retrieve import Retriever
        print("building retriever (once; ~2-3 s model load)...")
        retriever = Retriever(quiet=True)
        print(f"retrieval corpus: {retriever.corpus_path} ({len(retriever.rows)} rows)")

    # --- BUILD SIM VIEWER FIRST (belt-and-braces main-thread measure) ---
    # Primary macOS fix is running under plain `python` (see module docstring).
    # Building the backend before Whisper/sounddevice is an extra safeguard and
    # is order-independent on Windows.
    FullGraspBackend, HAS_GRID, helpers = load_sim()
    drive, describe, print_legend, draw_box_markers = helpers
    root = _ROOT
    if args.schema == "auto":
        name = "schema_v2.json" if HAS_GRID else "schema.json"
    else:
        name = "schema_v2.json" if args.schema == "v2" else "schema.json"
    if name.endswith("_v2.json") and not HAS_GRID:
        sys.exit("REFUSING: v2 schema requested but the backend has no grid cells "
                 "(sim/workspace.py did not import). Run from the repo root: "
                 "python -m sim.run_demo")
    schema = load_schema(os.path.join(root, "schema", name))
    print(f"schema: {name}  ({len(schema['locations'])} locations)")

    # Constrained decoding: attach the JSON Schema to the chosen client. Ollama
    # takes the full per-verb oneOf form; OpenAI rejects oneOf at the root, so it
    # gets the flat form and the field mask is enforced by validate_object
    # downstream instead. See llm/schema_json.py.
    #
    # MUST stay below load_schema(): this block used to sit above it and read
    # `schema` before it was assigned, so the default `--constrain on` raised
    # UnboundLocalError for every openai:/ollama: client before the viewer opened.
    if args.constrain == "on":
        from llm.schema_json import to_json_schema, to_flat_json_schema
        if client.name.startswith("openai:"):
            client.json_schema = to_flat_json_schema(schema)
        elif client.name.startswith("ollama:"):
            client.json_schema = to_json_schema(schema)
    print(f"constrained decoding: {args.constrain}")

    backend = FullGraspBackend(render=True, gain=3.0)   # opens viewer on main thread
    backend.set_object([1.30, 0.72, 0.425])

    # --- THEN set up voice (mode 2 only) ---
    # Loaded up front, not on the first spoken command, so the model-load cost
    # never lands inside a measured-looking ASR latency. asr/voice.py caches it.
    if mode == "voice":
        from asr import voice
        print("Loading Whisper (offline)...")
        t = time.perf_counter()
        voice.load_whisper(WHISPER_MODEL)
        print(f"Whisper loaded in {time.perf_counter()-t:.2f}s")

    print(f"\nModel: {client.name} | Strategy: {strategy} | Input: {mode.upper()}")
    print_legend()
    if mode == "voice" and HAS_GRID:
        print('  VOICE + GRID: say the WORDS, not the cell code. "the far left')
        print('  corner of the table" works; "grid A one" does not.\n')

    markers_ok = None
    try:
        while True:
            # --- STAGE 1: get command (mode-specific) ---
            asr_ms = None
            if mode == "keyboard":
                command = input("command> ").strip()
                if command.lower() in {"quit", "exit", ""}:
                    break
            else:
                command, asr_ms = get_voice_command()
                if command is None:
                    continue

            # --- STAGE 2: LLM -> schema (raw text; identical both modes) ---
            # rag selects its examples PER COMMAND; few_shot reuses the fixed
            # exemplars loaded once above. Everything downstream is unchanged.
            turn_examples = examples
            if retriever is not None:
                res = retriever.retrieve(command, k=3)
                turn_examples = res.examples
                print(f"  retrieved:     {res.ids}  ({res.latency_s*1000:.0f} ms)")
            t0 = time.perf_counter()
            r = run_once(command, client, strategy, schema, examples=turn_examples)
            llm_ms = (time.perf_counter() - t0) * 1000
            print(f"  understood as: {r.obj}  ({'valid' if r.ok else 'FAILED at ' + r.stage})")

            # --- STAGE 3: motion (shared) ---
            if r.ok:
                describe(r.obj)
                result = drive(backend, r.obj)
                if result.get("resolved") and result["resolved"] != r.obj.get("location"):
                    print(f"  resolved to:   {result['resolved']}")
                print(f"  arm result:    {result}")
            else:
                print(f"  arm did not move (reasons={r.reasons})")

            # --- latency (adapts to mode: no fake ASR figure in keyboard) ---
            if asr_ms is not None:
                print(f"  LATENCY: ASR {asr_ms:.0f} ms | LLM {llm_ms:.0f} ms "
                      f"| total-to-parse {asr_ms+llm_ms:.0f} ms")
            else:
                print(f"  LATENCY: LLM {llm_ms:.0f} ms")

            if markers_ok is None:
                markers_ok = draw_box_markers(backend)
                print(f"  [floor markers: {'ON' if markers_ok else 'unavailable'}]")
            elif markers_ok:
                draw_box_markers(backend)

            if mode == "voice":
                if input("\n  Another? [Enter=yes, q=quit]: ").strip().lower() == "q":
                    break
            print()
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        backend.close()
        print("\nclosed.")


if __name__ == "__main__":
    main()
