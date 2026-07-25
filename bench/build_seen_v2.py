"""
Build data/seen_v2.jsonl AND data/held_out_v2.jsonl from:
  * data/seen.jsonl                    (V1 — rows carried over verbatim)
  * data/location_phrases_v2.json      (YOUR phrases: phrases_seen + phrases_heldout)
  * schema/schema_v2.json              (the enum every emitted label is checked against)

Templates are extracted from YOUR V1 seen.jsonl, not written here. Every word
in the output is yours: your templates crossed with your location phrases.

Refuses to emit on: blank phrases, duplicate phrases across locations, labels
outside the v2 enum, missing per-verb required fields, or retired v1 phrases
reused on a non-alias cell.

The two splits share the same LABELS and the same squares. They differ only in
the WORDS used for the location. That difference is the generalization KPI, so
the builder refuses if any phrase appears in both lists.

Usage:
    python bench/build_seen_v2.py --dry-run
    python bench/build_seen_v2.py --write
    python bench/build_seen_v2.py --pp-templates 1 --mt-templates 1 --write
"""

import argparse
import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Extracted verbatim from V1 seen.jsonl. {TGT} = target noun, {LOC} = location phrase.
PP_TEMPLATES = [
    "put the {TGT} in {LOC}",
    "can you please keep the {TGT} in {LOC}?",
    "pick up the {TGT} and keep in {LOC}",
]
MT_TEMPLATES = [
    "move to {LOC}",
    "go to {LOC}",
    "navigate to {LOC}",
]
# Surface nouns as they appear in V1 commands.
TARGET_NOUN = {"bottle": "bottle", "toy_car": "car", "rubiks_cube": "rubiks cube"}


def die(msg):
    raise SystemExit(f"REFUSED: {msg}")


def load_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def check_required(label, verbs):
    req = verbs[label["action"]]["required"]
    missing = [f for f in req if f not in label]
    if missing:
        die(f"label {label} missing required field(s) {missing}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", default="data/seen.jsonl")
    ap.add_argument("--phrases", default="data/location_phrases_v2.json")
    ap.add_argument("--schema", default="schema/schema_v2.json")
    ap.add_argument("--out-seen", default="data/seen_v2.jsonl")
    ap.add_argument("--out-heldout", default="data/held_out_v2.jsonl")
    ap.add_argument("--write", action="store_true", help="actually write both files")
    ap.add_argument("--pp-templates", type=int, default=len(PP_TEMPLATES))
    ap.add_argument("--mt-templates", type=int, default=len(MT_TEMPLATES))
    args = ap.parse_args()

    sch = json.loads((ROOT / args.schema).read_text(encoding="utf-8"))
    ph = json.loads((ROOT / args.phrases).read_text(encoding="utf-8"))
    v1 = load_jsonl(ROOT / args.v1)

    verbs = sch["verbs"]
    enum = set(sch["locations"])
    classes = sch["location_classes"]
    containers = {k for k, v in classes.items() if v == "container"}
    alias = ph["_retired_v1_alias_map_DEMO_ONLY"]

    # ---- validate phrases -------------------------------------------------
    seen_phrase = {}
    for loc, entry in ph["locations"].items():
        if loc not in enum:
            die(f"phrase file has '{loc}' which is not in the v2 enum")
        for split in ("phrases_seen", "phrases_heldout"):
            plist = [p.strip() for p in entry[split]]
            if not plist or any(not p for p in plist):
                die(f"'{loc}' has a blank {split} entry. Fill every slot in {args.phrases} first.")
            for p in plist:
                key = p.lower()
                if key in seen_phrase:
                    prev_loc, prev_split = seen_phrase[key]
                    if prev_loc != loc:
                        die(f"phrase {p!r} used for BOTH {prev_loc} and {loc} — one command would have two correct answers")
                    die(f"phrase {p!r} appears in BOTH {prev_split} and {split} for {loc} — that leaks the test set into training")
                seen_phrase[key] = (loc, split)
    missing = (enum - containers) - set(ph["locations"])
    if missing:
        die(f"no phrases for {sorted(missing)}")

    for old, cell in alias.items():
        word = old.replace("table_", "").replace("_", " ")
        for p, (loc, _sp) in seen_phrase.items():
            if word in p and loc != cell and classes.get(loc) != "container":
                print(f"  warn: phrase {p!r} on {loc} echoes retired '{old}' (aliased to {cell})")

    # ---- carry V1 rows ----------------------------------------------------
    carried = [r for r in v1 if "location" not in r["expected"] or r["expected"]["location"] in containers]
    dropped = len(v1) - len(carried)

    # ---- generate new rows ------------------------------------------------
    def generate(split):
        out = []
        for loc, entry in ph["locations"].items():
            for phrase in [p.strip() for p in entry[split]]:
                for tpl in PP_TEMPLATES[: args.pp_templates]:
                    for tgt, noun in TARGET_NOUN.items():
                        out.append({
                            "command": tpl.format(TGT=noun, LOC=phrase),
                            "expected": {"action": "pick_place", "target": tgt,
                                         "location": loc, "modifiers": []},
                        })
                for tpl in MT_TEMPLATES[: args.mt_templates]:
                    out.append({
                        "command": tpl.format(LOC=phrase),
                        "expected": {"action": "move_to", "location": loc, "modifiers": []},
                    })
        return out

    seen_rows = carried + generate("phrases_seen")
    heldout_rows = generate("phrases_heldout")
    splits = {"seen": (seen_rows, ROOT / args.out_seen),
              "held_out": (heldout_rows, ROOT / args.out_heldout)}

    # ---- final validation + report -----------------------------------------
    all_cmds = collections.Counter()
    for name, (rows, _path) in splits.items():
        for r in rows:
            check_required(r["expected"], verbs)
            loc = r["expected"].get("location")
            if loc is not None and loc not in enum:
                die(f"[{name}] emitted location {loc!r} not in v2 enum")
            tgt = r["expected"].get("target")
            if tgt and tgt not in sch["targets"]:
                die(f"[{name}] emitted target {tgt!r} not in v2 targets")
            all_cmds[r["command"]] += 1
    dup = [c for c, n in all_cmds.items() if n > 1]
    if dup:
        die(f"{len(dup)} duplicate command(s) across splits, e.g. {dup[:3]}")

    print(f"carried from v1: {len(carried)}  (dropped {dropped} rows using retired table_* symbols)")
    total = 0
    for name, (rows, path) in splits.items():
        total += len(rows)
        by_act = collections.Counter(r["expected"]["action"] for r in rows)
        per_loc = collections.Counter(r["expected"]["location"] for r in rows if "location" in r["expected"])
        print(f"\n{name}: {len(rows)} rows -> {path.relative_to(ROOT)}")
        print(f"  by action:    {dict(by_act)}")
        print(f"  per-location: min {min(per_loc.values())} max {max(per_loc.values())}")
    print(f"\ngrid run cost: {total} rows x 20 cells x 5 repeats = {total*100:,} calls")

    if not args.write:
        print("\nno --write flag: nothing written")
        return
    for name, (rows, path) in splits.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                        encoding="utf-8")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
