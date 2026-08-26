"""How many DISTINCT strings must actually be tokenised per model?"""
import csv, collections
csv.field_size_limit(2**31 - 1)
T = lambda v: v in ("True", "true", "1")

d = {}
for f in ["rung1", "rag", "lora", "cls"]:
    for r in csv.DictReader(open(f"bench/results_heldout_v3_{f}_tokens.csv",
                                 encoding="utf-8", newline="")):
        d[(r["model"], r["strategy"], r["command_id"], r["run_index"])] = r

per = collections.defaultdict(set)
rows = collections.Counter()
chars = collections.Counter()
for r in d.values():
    m = r["model"]
    if m.startswith("openai") or m.startswith("classifier"):
        continue
    rows[m] += 1
    for col in ("raw", "thinking"):
        s = r.get(col) or ""
        if s:
            per[m].add(s)
            chars[m] += len(s)

print(f"{'model':26}{'rows':>8}{'unique strings':>16}{'total chars':>14}")
tot_u = 0
for m in sorted(per):
    print(f"{m:26}{rows[m]:>8}{len(per[m]):>16,}{chars[m]:>14,}")
    tot_u += len(per[m])
print(f"\ntotal unique strings to tokenise across local models: {tot_u:,}")
