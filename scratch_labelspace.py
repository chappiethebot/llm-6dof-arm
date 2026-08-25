import json
ld = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
tr = ld("data/train_v2.jsonl"); ho = ld("data/held_out_v3.jsonl")
sch = json.load(open("schema/schema_v2.json", encoding="utf-8"))
print("schema keys:", list(sch.keys()))
for k in ("actions", "targets", "locations"):
    if k in sch: print(f"  {k} ({len(sch[k])}):", sch[k])
print(f"\nheld_out_v3: {len(ho)} rows")
for key in ("action", "target", "location"):
    t = {r["expected"].get(key) for r in tr} - {None}
    h = {r["expected"].get(key) for r in ho} - {None}
    print(f"  {key:9} unseen-in-train: {sorted(h - t) or 'NONE'}   train-only: {sorted(t - h) or 'none'}")
import collections
print("  held_out_v3 verbs:", dict(collections.Counter(r["expected"]["action"] for r in ho)))
