"""Diagnostic A part 2: does GPU memory pressure cause the instability?"""
import ollama, time

PROMPT = ("Describe in three sentences how a robot arm picks up a bottle "
          "from a table, then reply with a JSON object summarising it.")

def trial(label):
    print("=" * 60)
    print(label)
    print("  resident:", [m.model for m in (ollama.ps().models or [])])
    outs, times = [], []
    for _ in range(6):
        t0 = time.perf_counter()
        r = ollama.generate(model="qwen2.5:3b", prompt=PROMPT,
                            options={"temperature": 0.0}, keep_alive="5m")
        times.append(time.perf_counter() - t0)
        outs.append(r["response"])
    print(f"  median {sorted(times)[len(times)//2]:.2f}s   "
          f"distinct {len(set(outs))}/6   "
          f"{'DETERMINISTIC' if len(set(outs))==1 else 'NOT DETERMINISTIC'}")

trial("ALONE")

print("\nloading deepseek-r1:7b to fill the GPU...")
ollama.generate(model="deepseek-r1:7b", prompt="hi", keep_alive="5m")
ollama.generate(model="llama3.2:3b", prompt="hi", keep_alive="5m")

trial("UNDER MEMORY PRESSURE")