"""Build data/held_out_v4c.jsonl -- row vocabulary in NEITHER split.

held_out_v3 with the row words replaced by front/back:
    upper  -> back    (row A = far from the robot = back of the table)
    bottom -> front   (row C = near the robot     = front of the table)

Ground truth UNCHANGED. held_out_v3.jsonl opened read-only.

WHY THIS EXISTS. held_out_v4b reworded upper->far, bottom->near to match
training vocabulary. That reverted the exact substitution used to BUILD
held_out_v3 from the training phrasings, so all 72 reworded commands landed
verbatim on training strings (56 in train_v2, 14 in dev_v2, 0 novel). v4b can
therefore only measure memorisation. v4c uses a THIRD vocabulary present in
neither split, which tests whether the adapters learned a row CONCEPT or two
specific strings.
"""
import json, os, re, hashlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "data", "held_out_v3.jsonl")
DST = os.path.join(ROOT, "data", "held_out_v4c.jsonl")


def reword(c):
    c = re.sub(r"\bupper\b", "back", c)
    c = re.sub(r"\bUpper\b", "Back", c)
    c = re.sub(r"\bbottom\b", "front", c)
    c = re.sub(r"\bBottom\b", "Front", c)
    return c


src = open(SRC, "rb").read()
rows = [json.loads(l) for l in src.decode("utf-8").splitlines() if l.strip()]
print(f"source: data/held_out_v3.jsonl  {len(rows)} rows  "
      f"sha256={hashlib.sha256(src).hexdigest()[:16]}")

out, changed = [], 0
for r in rows:
    new = dict(r)
    c2 = reword(r["command"])
    if c2 != r["command"]:
        changed += 1
    new["command"] = c2
    assert new.get("expected") == r.get("expected"), "ground truth changed!"
    out.append(new)

assert len(out) == 217
with open(DST, "w", encoding="utf-8", newline="\n") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"wrote data/held_out_v4c.jsonl  {len(out)} rows, {changed} reworded")

assert open(SRC, "rb").read() == src, "SOURCE MODIFIED"
print("held_out_v3.jsonl verified byte-identical after write")
print(f"ground truth identical on all 217 rows: "
      f"{[r.get('expected') for r in rows] == [r.get('expected') for r in out]}")


def cmds(name):
    p = os.path.join(ROOT, "data", f"{name}.jsonl")
    return {json.loads(l)["command"] for l in open(p, encoding="utf-8") if l.strip()}


print()
print("=" * 78)
print("COLLISION CHECK -- none of the 217 may appear verbatim in any training split")
print("=" * 78)
allbad = 0
for split in ("seen_v2", "train_v2", "dev_v2"):
    s = cmds(split)
    hit = [r["command"] for r in out if r["command"] in s]
    allbad += len(hit)
    print(f"  verbatim in {split:10}: {len(hit)}"
          + ("   <-- FAIL" if hit else "   PASS"))
    for h in hit[:5]:
        print(f"      {h!r}")
print(f"\n  TOTAL collisions: {allbad}  -> {'FAIL' if allbad else 'PASS'}")

# also confirm v4c differs from v3 and from v4b
v3c = {r["command"] for r in rows}
same_as_v3 = sum(1 for r in out if r["command"] in v3c)
print(f"\n  v4c commands also present verbatim in held_out_v3: {same_as_v3} "
      f"(expected {217-changed} — the untouched rows)")

# vocabulary sanity
print()
print("=" * 78)
print("ROW VOCABULARY BY SPLIT after the change")
print("=" * 78)
SPL = ["seen_v2", "train_v2", "dev_v2", "held_out_v3"]
D = {s: [json.loads(l) for l in open(os.path.join(ROOT, "data", f"{s}.jsonl"),
                                     encoding="utf-8") if l.strip()] for s in SPL}
D["held_out_v4c"] = out
for w_ in ("far", "near", "upper", "bottom", "front", "back"):
    line = "  " + f"{w_:8}"
    for s in list(SPL) + ["held_out_v4c"]:
        n = sum(1 for r in D[s] if re.search(rf"\b{w_}\b", r["command"].lower()))
        line += f"  {s}={n:<4}"
    print(line)
print()
print("  NOTE: 'back' also occurs once non-locatively in held_out_v3 —")
print("  'secure the bottle and bring it back' (no location in ground truth).")
print("  v4c therefore contains one command where a row word is used")
print("  non-spatially. That is the §6b conflict class. Swapping back->rear")
print("  (0 occurrences anywhere) removes it if preferred; it is a one-word change.")
