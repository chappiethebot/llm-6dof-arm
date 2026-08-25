import json, collections
ld = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
tr = ld("data/train_v2.jsonl"); ho = ld("data/held_out_v3.jsonl")
print(f"held_out_v3: {len(ho)} rows")
for key in ("action", "target", "location"):
    t = {r["expected"].get(key) for r in tr} - {None}
    h = {r["expected"].get(key) for r in ho} - {None}
    print(f"  {key:9} unseen-in-train: {sorted(h - t) or 'NONE'}   train-only: {sorted(t - h) or 'none'}")
print("  verbs:", dict(collections.Counter(r["expected"]["action"] for r in ho)))
print("  locations:", dict(collections.Counter(r["expected"].get("location","<none>") for r in ho)))
bad = [r["command"] for r in ho if any(w in r["command"].lower() for w in ("middle","center","centre","grid_"))]
print(f"  collision-rule violations: {len(bad)}", bad[:5])
dupes = {r["command"] for r in ho} & {r["command"] for r in tr}
print(f"  commands shared with train_v2: {len(dupes)}", list(dupes)[:5])
nonempty = [r for r in ho if r["expected"].get("modifiers")]
print(f"  non-empty modifiers: {len(nonempty)}")
