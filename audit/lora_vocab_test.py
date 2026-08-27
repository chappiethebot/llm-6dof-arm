"""Does the row collapse disappear when the ROW VOCABULARY matches training?

FIGV2-4 argued [Likely] that the rung-3 adapters failed the row axis because
seen_v2 (their training data) says far/near while held_out_v3 says
upper/bottom, whereas left/right is shared. That was an inference from an
alignment, not a demonstration. This demonstrates it.

THE NATURAL EXPERIMENT, already present in the data:
  train_v2      85 grid rows, 56 far/near, 0 upper/bottom   TRAINED ON
  dev_v2        21 grid rows, 14 far/near, 0 upper/bottom   NOT trained on
  held_out_v3  108 grid rows,  0 far/near, 72 upper/bottom  NOT trained on

dev_v2 is the decisive split: same row vocabulary as training, never trained
on, so a jump there is a VOCABULARY effect and not memorisation. train_v2 is
included only as the memorisation ceiling.

NOT A REPORTED NUMBER. This is a diagnostic run on non-sealed splits. It calls
llm.pipeline.run_once directly -- the same function bench/run_cell.py calls --
and writes only to audit/.
"""
import csv, json, os, re, sys, time, collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("OLLAMA_MODELS", r"D:\LLM")

from llm.prompts import load_schema
from llm.pipeline import run_once
from llm.clients import OllamaClient

CELLS = [f"grid_{r}{c}" for r in "ABC" for c in "123"]
MODELS = ["qwen2.5-0.5b-lora", "qwen2.5-3b-lora"]
SPLITS = ["train_v2", "dev_v2"]
schema = load_schema(os.path.join(ROOT, "schema", "schema_v2.json"))


def grid_rows(name):
    p = os.path.join(ROOT, "data", f"{name}.jsonl")
    rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    return [r for r in rows if (r.get("expected") or {}).get("location") in CELLS]


def pred_loc(obj):
    if not isinstance(obj, dict):
        return None
    v = obj.get("location")
    if v is None and isinstance(obj.get("action"), dict):
        v = obj["action"].get("location")
    return v if isinstance(v, str) else None


out = []
for mname in MODELS:
    client = OllamaClient(mname)
    for split in SPLITS:
        rows = grid_rows(split)
        t0 = time.perf_counter()
        for i, row in enumerate(rows):
            cmd = row["command"]
            truth = row["expected"]["location"]
            r = run_once(cmd, client, "zero_shot", schema, examples=None)
            p = pred_loc(r.obj)
            out.append(dict(
                model=f"ollama:{mname}", split=split, command=cmd,
                truth=truth, pred=p or "",
                vocab=("far/near" if re.search(r"\b(far|near)\b", cmd.lower())
                       else ("upper/bottom" if re.search(r"\b(upper|bottom)\b", cmd.lower())
                             else "other")),
                row_ok=int(bool(p and p in CELLS and p[5] == truth[5])),
                col_ok=int(bool(p and p in CELLS and p[6] == truth[6])),
                cell_ok=int(p == truth),
                stage=r.stage))
        print(f"  {mname:20} {split:10} {len(rows):3} rows "
              f"in {time.perf_counter()-t0:5.1f}s")

p = os.path.join(ROOT, "audit", "lora_vocab_test.csv")
with open(p, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
    w.writeheader(); w.writerows(out)
print(f"\nwrote audit/lora_vocab_test.csv ({len(out)} rows)\n")

print("=" * 96)
print("ROW vs COLUMN ACCURACY BY SPLIT   (chance on either axis = 33.3%)")
print("=" * 96)
print(f"{'model':24}{'split':12}{'row vocab':14}{'n':>5}{'row':>8}{'col':>8}{'cell':>8}")
for m in [f"ollama:{x}" for x in MODELS]:
    for split in SPLITS:
        g = [r for r in out if r["model"] == m and r["split"] == split]
        for vocab in ("far/near", "other", "ALL"):
            s = g if vocab == "ALL" else [r for r in g if r["vocab"] == vocab]
            if not s:
                continue
            n = len(s)
            print(f"{m:24}{split:12}{vocab:14}{n:>5}"
                  f"{100*sum(r['row_ok'] for r in s)/n:>7.1f}%"
                  f"{100*sum(r['col_ok'] for r in s)/n:>7.1f}%"
                  f"{100*sum(r['cell_ok'] for r in s)/n:>7.1f}%")
    print()

print("COMPARISON -- held_out_v3 (upper/bottom, from the sealed measurement):")
print(f"{'ollama:qwen2.5-0.5b-lora':24}{'held_out_v3':12}{'upper/bottom':14}{539:>5}"
      f"{30.6:>7.1f}%{90.0:>7.1f}%{30.6:>7.1f}%")
print(f"{'ollama:qwen2.5-3b-lora':24}{'held_out_v3':12}{'upper/bottom':14}{539:>5}"
      f"{40.8:>7.1f}%{86.1:>7.1f}%{40.8:>7.1f}%")

print()
print("PREDICTED-ROW MARGINAL per split (collapse onto B shows up here):")
for m in [f"ollama:{x}" for x in MODELS]:
    for split in SPLITS:
        g = [r for r in out if r["model"] == m and r["split"] == split]
        c = collections.Counter(r["pred"][5] if r["pred"] in CELLS else "?" for r in g)
        n = len(g)
        print(f"  {m:24}{split:12} A={100*c['A']/n:5.1f}%  B={100*c['B']/n:5.1f}%  "
              f"C={100*c['C']/n:5.1f}%  none={100*c['?']/n:5.1f}%")
