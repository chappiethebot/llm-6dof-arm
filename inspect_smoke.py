import csv, json
csv.field_size_limit(2**31 - 1)
rows = list(csv.DictReader(open("bench/results_smoke_v3.csv", encoding="utf-8")))
corpus = [json.loads(l) for l in open("data/train_v2.jsonl", encoding="utf-8") if l.strip()]
for r in rows:
    print(f"cid={r['command_id']}  strict={r['strict_match']}")
    print(f"  cmd   : {r['command']}")
    print(f"  truth : {r['truth_json']}")
    print(f"  pred  : {r['pred_json']}")
    ids = json.loads(r["retrieved_ids"]); sc = json.loads(r["retrieved_scores"])
    for i, s in zip(ids, sc):
        c = corpus[i]
        print(f"    {s:.3f} [{i:>3}] {c['expected']['action']:<10} {c['command']}")
        print(f"           -> {json.dumps(c['expected'])}")
    print()
