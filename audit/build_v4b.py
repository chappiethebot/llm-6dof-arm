"""Build data/held_out_v4b.jsonl -- held_out_v3 with row vocabulary aligned to training.

Same 217 commands, ground truth UNCHANGED, only the row word rewritten:
    upper -> far      bottom -> near
held_out_v3.jsonl is opened read-only and is not modified.

READ THE COLLISION WARNING BELOW BEFORE INTERPRETING ANY RESULT FROM THIS FILE.
"""
import json, os, re, hashlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "data", "held_out_v3.jsonl")
DST = os.path.join(ROOT, "data", "held_out_v4b.jsonl")


def reword(c):
    c = re.sub(r"\bupper\b", "far", c)
    c = re.sub(r"\bUpper\b", "Far", c)
    c = re.sub(r"\bbottom\b", "near", c)
    c = re.sub(r"\bBottom\b", "Near", c)
    return c


src_bytes = open(SRC, "rb").read()
rows = [json.loads(l) for l in src_bytes.decode("utf-8").splitlines() if l.strip()]
print(f"source: data/held_out_v3.jsonl  {len(rows)} rows  "
      f"sha256={hashlib.sha256(src_bytes).hexdigest()[:16]}")

out, changed = [], 0
for r in rows:
    new = dict(r)
    c2 = reword(r["command"])
    if c2 != r["command"]:
        changed += 1
    new["command"] = c2
    # ground truth must be byte-identical
    assert new.get("expected") == r.get("expected"), "ground truth changed!"
    out.append(new)

assert len(out) == len(rows) == 217
with open(DST, "w", encoding="utf-8", newline="\n") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"wrote data/held_out_v4b.jsonl  {len(out)} rows, {changed} commands reworded")

# integrity: source untouched
assert open(SRC, "rb").read() == src_bytes, "SOURCE FILE WAS MODIFIED"
print("held_out_v3.jsonl verified byte-identical after write")

# ground-truth parity
gt_a = [r.get("expected") for r in rows]
gt_b = [r.get("expected") for r in out]
print(f"ground truth identical on all {len(rows)} rows: {gt_a == gt_b}")

# ---- the collision warning -------------------------------------------------
seen = {r["command"] for r in
        (json.loads(l) for l in open(os.path.join(ROOT, "data", "seen_v2.jsonl"),
                                     encoding="utf-8") if l.strip())}
train = {r["command"] for r in
         (json.loads(l) for l in open(os.path.join(ROOT, "data", "train_v2.jsonl"),
                                      encoding="utf-8") if l.strip())}
dev = {r["command"] for r in
       (json.loads(l) for l in open(os.path.join(ROOT, "data", "dev_v2.jsonl"),
                                    encoding="utf-8") if l.strip())}
coll = [r["command"] for r in out if r["command"] in seen]
print()
print("!" * 78)
print(f"COLLISION: {len(coll)} of {len(out)} v4b commands are VERBATIM in seen_v2")
print(f"           {sum(1 for c in coll if c in train)} in train_v2 (LoRA training data)")
print(f"           {sum(1 for c in coll if c in dev)} in dev_v2 (checkpoint selection)")
print("!" * 78)
