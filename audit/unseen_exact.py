import csv, collections
csv.field_size_limit(2**31 - 1)
T = lambda v: v in ("True", "true", "1")
d = {}
for f in ["rung1", "rag", "lora", "cls"]:
    for r in csv.DictReader(open(f"bench/results_heldout_v3_{f}_tokens.csv",
                                 encoding="utf-8", newline="")):
        d[(r["model"], r["strategy"], r["command_id"], r["run_index"])] = r

per = collections.defaultdict(lambda: [0, 0])
src = collections.Counter()
for r in d.values():
    if T(r["is_warmup"]):
        continue
    src[r["token_source"]] += 1
    a = per[r["model"]]
    a[0] += int(r["tokens_out_visible"])
    a[1] += int(r["tokens_thinking"])

print("token_source stamps:", dict(src))
print()
print(f"{'model':26}{'visible':>12}{'thinking':>13}{'total':>13}{'unseen':>9}")
for m, (v, t) in sorted(per.items(), key=lambda x: -x[1][1]):
    tot = v + t
    print(f"{m:26}{v:>12,}{t:>13,}{tot:>13,}{(t/tot*100 if tot else 0):>8.1f}%")

ds = per["ollama:deepseek-r1:7b"]
print(f"\ndeepseek hidden:visible = {ds[1]/ds[0]:.1f}:1   "
      f"unseen share = {ds[1]/(ds[0]+ds[1])*100:.1f}%")

# cloud cost detail
cloud = [r for r in d.values() if r["model"] == "openai:gpt-4o-mini"
         and not T(r["is_warmup"])]
by = collections.defaultdict(float)
for r in cloud:
    by[r["strategy"]] += float(r["cost_usd"] or 0)
print("\ngpt-4o-mini cost by strategy (billed on tokens_out_total):")
for s, c in sorted(by.items(), key=lambda x: -x[1]):
    print(f"   {s:12} ${c:.4f}")
print(f"   {'TOTAL':12} ${sum(by.values()):.4f}")
