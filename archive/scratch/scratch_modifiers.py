import json, collections
ld = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
for name in ("train_v2", "dev_v2", "held_out_v3"):
    rows = ld(f"data/{name}.jsonl")
    nonempty = [r for r in rows if r["expected"].get("modifiers")]
    keys = collections.Counter(k for r in rows for k in r["expected"])
    print(f"{name:12} n={len(rows):4}  non-empty modifiers: {len(nonempty)}")
    print(f"             keys: {dict(keys)}")
    for r in nonempty[:3]:
        print(f"                {r['command'][:50]!r} -> {r['expected']}")
