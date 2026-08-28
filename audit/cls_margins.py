"""Step 9a: log the classifier's per-head softmax margins on held_out_v3.

One re-run of the rung-4 cell, reading logits instead of only argmax.
Writes audit/cls_margins.csv. Does not touch any results file.
"""
import csv, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from llm.classifier_client import DistilBertClassifierClient

c = DistilBertClassifierClient()
c._ensure_loaded()
torch = c._torch

rows = [json.loads(l) for l in
        open(os.path.join(ROOT, "data", "held_out_v3.jsonl"), encoding="utf-8")
        if l.strip()]

out = []
for cid, r in enumerate(rows):
    kw = dict(truncation=True, max_length=c._max_len, return_tensors="pt")
    kw["padding"] = "max_length" if c.pad_to_max else False
    with torch.no_grad():
        enc = c._tok(r["command"], **kw)
        vl, tl, ll = c._model(enc["input_ids"], enc["attention_mask"])
    rec = dict(command_id=cid, command=r["command"])
    for name, lg in (("verb", vl), ("target", tl), ("location", ll)):
        p = torch.softmax(lg, -1)[0]
        top2 = torch.topk(p, 2)
        rec[f"{name}_top1"] = float(top2.values[0])
        rec[f"{name}_margin"] = float(top2.values[0] - top2.values[1])
        rec[f"{name}_idx"] = int(top2.indices[0])
    obj = c._emit(rec["verb_idx"], rec["target_idx"], rec["location_idx"])
    rec["pred_json"] = json.dumps(obj)
    rec["truth_json"] = json.dumps(r["expected"])
    out.append(rec)

p = os.path.join(ROOT, "audit", "cls_margins.csv")
with open(p, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
    w.writeheader(); w.writerows(out)
print(f"wrote audit/cls_margins.csv ({len(out)} rows)")
m = sorted(r["location_margin"] for r in out)
print(f"location margin: min {m[0]:.3f}  p10 {m[len(m)//10]:.3f}  "
      f"median {m[len(m)//2]:.3f}  max {m[-1]:.3f}")
