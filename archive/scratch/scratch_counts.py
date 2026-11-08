import json, collections
for split in ["train_v2", "dev_v2"]:
    rows = [json.loads(l) for l in open(f"data/{split}.jsonl", encoding="utf-8") if l.strip()]
    print(f"\n=== {split}  ({len(rows)} rows) ===")
    for head, key in [("verb","action"), ("target","target"), ("location","location")]:
        c = collections.Counter(r["expected"].get(key, "<none>") for r in rows)
        print(f"  {head} ({len(c)} classes):")
        for cls, n in sorted(c.items(), key=lambda kv: -kv[1]):
            print(f"      {cls:22} {n}")
    print("  verb x location-present:")
    cx = collections.Counter((r["expected"]["action"], "location" in r["expected"]) for r in rows)
    for (v, has), n in sorted(cx.items()):
        print(f"      {v:12} loc={str(has):5} {n}")
