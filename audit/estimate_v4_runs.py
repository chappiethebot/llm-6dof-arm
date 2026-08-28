"""Wall-clock estimate for the V4a / V4b runs, from MEASURED p50 latencies."""
import csv, os, collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATS = os.path.join(ROOT, "audit", "figures", "cell_stats.csv")

rows = list(csv.DictReader(open(STATS, encoding="utf-8")))
p50 = {(r["model"], r["strategy"]): float(r["p50_s"]) for r in rows}
p99 = {(r["model"], r["strategy"]): float(r["p99_s"]) for r in rows}

BASE = ["openai:gpt-4o-mini", "ollama:llama3.2:3b", "ollama:qwen2.5:3b",
        "ollama:qwen2.5:0.5b", "ollama:deepseek-r1:7b"]
PROMPT_STRATS = ["zero_shot", "few_shot", "cot", "role_based"]
LORA = ["ollama:qwen2.5-0.5b-lora", "ollama:qwen2.5-3b-lora"]
N_CMD = 217


def fmt(sec):
    h, m = divmod(int(sec + 0.5), 3600)
    m, s = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def plan(name, cells, repeats_for):
    print(f"\n{'='*94}")
    print(f"{name}")
    print(f"{'='*94}")
    print(f"{'model':26}{'strategy':11}{'reps':>5}{'calls':>8}"
          f"{'p50 s':>8}{'est p50':>10}{'est p99':>10}")
    tot50 = tot99 = 0.0
    per_model = collections.defaultdict(float)
    ncalls = 0
    for m, s in cells:
        reps = repeats_for(m)
        calls = N_CMD * reps
        a = p50.get((m, s))
        b = p99.get((m, s))
        if a is None:
            print(f"{m:26}{s:11}{reps:>5}{calls:>8}   no V3 measurement")
            continue
        e50, e99 = calls * a, calls * b
        tot50 += e50; tot99 += e99; ncalls += calls
        per_model[m] += e50
        print(f"{m:26}{s:11}{reps:>5}{calls:>8}{a:>8.3f}{fmt(e50):>10}{fmt(e99):>10}")
    print("-" * 94)
    print(f"{'TOTAL':37}{ncalls:>8}{'':8}{fmt(tot50):>10}{fmt(tot99):>10}")
    print()
    print("  per model (at p50):")
    for m, v in sorted(per_model.items(), key=lambda x: -x[1]):
        print(f"    {m:26} {fmt(v):>9}   ({100*v/tot50:4.1f}% of the run)")
    return tot50, tot99, ncalls


# ---- V4a: 25 cells on held_out_v3 with the v4 legend ----------------------
cells_a = [(m, s) for m in BASE for s in PROMPT_STRATS]          # 20
cells_a += [(m, "rag") for m in BASE]                            # 5
cells_a += [(m, "zero_shot") for m in LORA]                      # 2
reps = lambda m: 5 if m.startswith("openai:") else 1
a50, a99, an = plan("V4a  —  25 cells, held_out_v3, --legend v4 "
                    "(classifier skipped: it never reads the legend)",
                    cells_a, reps)

# ---- V4b: 2 LoRA cells on held_out_v4b ------------------------------------
cells_b = [(m, "zero_shot") for m in LORA]
b50, b99, bn = plan("V4b  —  2 LoRA cells, held_out_v4b, --legend v4", cells_b, reps)

print(f"\n{'='*94}")
print("GRAND TOTAL")
print(f"{'='*94}")
print(f"  calls       : {an + bn:,}")
print(f"  at p50      : {fmt(a50 + b50)}")
print(f"  at p99      : {fmt(a99 + b99)}   (worst case; the tail is real — "
      f"deepseek max was ~380 s)")
print()
print("  NOTE ON THE p99 COLUMN: p99 x every call is a deliberate over-estimate. "
      "The\n  realistic figure is close to p50 because the tail is a small share "
      "of rows.")
print()
print(f"  If 5 repeats were used for ALL models instead of 1:")
all5_50 = sum(N_CMD * 5 * p50[(m, s)] for m, s in cells_a + cells_b if (m, s) in p50)
print(f"    {fmt(all5_50)}  — the single-run decision saves {fmt(all5_50 - a50 - b50)}")
