"""num_ctx confound check. READ-ONLY over bench/.

Q1: is tokens_in/tokens_out populated at all, and for which cells?
Q2: per cell, max prompt tokens and max (prompt+gen); how many rows exceed
    2048 (Ollama server default) and 4096?
Q3: for any cell with rows over a ceiling, does strict accuracy differ
    above vs below it? If accuracy collapses above 2048, the finding is a
    config artifact, not a model property.
"""
import csv, collections
csv.field_size_limit(2**31 - 1)

T = lambda v: v in ("True", "true", "1")
FILES = ["rung1", "rag", "lora", "cls"]
CEILINGS = [2048, 4096, 8192]

d = {}
for f in FILES:
    for r in csv.DictReader(open(f"bench/results_heldout_v3_{f}_scored.csv",
                                 encoding="utf-8", newline="")):
        d[(r["model"], r["strategy"], r["command_id"], r["run_index"])] = r

cells = collections.defaultdict(list)
for r in d.values():
    if not T(r["is_warmup"]):
        cells[(r["model"], r["strategy"])].append(r)


def i(r, k):
    v = r.get(k, "")
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


print("=" * 118)
print("Q1 -- TOKEN COLUMN COVERAGE (can this question be answered from the CSVs at all?)")
print("=" * 118)
print(f"{'model':26}{'strategy':11}{'n':>6}{'tok_in set':>12}{'tok_out set':>13}{'coverage':>10}")
uncovered = []
for (m, s), c in sorted(cells.items()):
    ti = sum(1 for r in c if i(r, "tokens_in") is not None)
    to = sum(1 for r in c if i(r, "tokens_out") is not None)
    cov = ti / len(c) * 100
    if cov < 100:
        uncovered.append((m, s, cov))
    print(f"{m:26}{s:11}{len(c):>6}{ti:>12}{to:>13}{cov:>9.1f}%")

print()
print("=" * 118)
print("Q2 -- CONTEXT PRESSURE PER CELL  (prompt tokens, and prompt+generated)")
print("=" * 118)
hdr = (f"{'model':26}{'strategy':11}{'max_in':>8}{'p99_in':>8}"
       f"{'max_tot':>9}{'p99_tot':>9}")
for c_ in CEILINGS:
    hdr += f"{'>'+str(c_):>8}"
print(hdr)


def pctl(vals, q):
    if not vals:
        return 0
    v = sorted(vals)
    return v[min(int(q * len(v)), len(v) - 1)]


over_rows = {}
for (m, s), c in sorted(cells.items()):
    ins = [i(r, "tokens_in") for r in c if i(r, "tokens_in") is not None]
    tots = [(i(r, "tokens_in") or 0) + (i(r, "tokens_out") or 0)
            for r in c if i(r, "tokens_in") is not None]
    if not ins:
        print(f"{m:26}{s:11}{'-- no token data --':>40}")
        continue
    line = (f"{m:26}{s:11}{max(ins):>8}{pctl(ins,.99):>8}"
            f"{max(tots):>9}{pctl(tots,.99):>9}")
    for c_ in CEILINGS:
        n_over = sum(1 for t in tots if t > c_)
        line += f"{n_over:>8}"
        if c_ == 2048 and n_over:
            over_rows[(m, s)] = c_
    print(line)

print()
print("=" * 118)
print("Q3 -- ACCURACY ABOVE vs BELOW EACH CEILING  (only cells with rows on both sides)")
print("=" * 118)
any_split = False
for c_ in CEILINGS:
    printed_hdr = False
    for (m, s), c in sorted(cells.items()):
        rows = [r for r in c if i(r, "tokens_in") is not None]
        if not rows:
            continue
        lo = [r for r in rows if (i(r, "tokens_in") or 0) + (i(r, "tokens_out") or 0) <= c_]
        hi = [r for r in rows if (i(r, "tokens_in") or 0) + (i(r, "tokens_out") or 0) > c_]
        if not lo or not hi:
            continue
        any_split = True
        if not printed_hdr:
            print(f"\n--- ceiling {c_} ---")
            print(f"{'model':26}{'strategy':11}{'n<=':>7}{'acc<=':>8}{'n>':>7}"
                  f"{'acc>':>8}{'delta':>8}{'inv<=':>8}{'inv>':>8}")
            printed_hdr = True
        a_lo = sum(T(r["strict_match"]) for r in lo) / len(lo)
        a_hi = sum(T(r["strict_match"]) for r in hi) / len(hi)
        iv = lambda g: sum(1 for r in g if r.get("stage") not in (None, "", "validated")) / len(g)
        print(f"{m:26}{s:11}{len(lo):>7}{a_lo:>8.3f}{len(hi):>7}{a_hi:>8.3f}"
              f"{a_hi-a_lo:>+8.3f}{iv(lo):>8.3f}{iv(hi):>8.3f}")
if not any_split:
    print("\nNo cell has rows on both sides of any ceiling tested.")
