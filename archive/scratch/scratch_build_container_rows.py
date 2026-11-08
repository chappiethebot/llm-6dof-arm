import json, re, sys, shutil

TR = "data/train_v2.jsonl"; HO = "data/held_out_v3.jsonl"
APPLY = "--apply" in sys.argv

def ld(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

OBJ = ["rubiks cube", "rubik s cube", "rubiks_cube", "rubik cube", "toy car", "toy_car", "bottle"]
LOC = ["box 1", "box 2", "box 3", "box_1", "box_2", "box_3"]
def skel(s):
    s = s.lower()
    for o in OBJ: s = s.replace(o, "<OBJ>")
    for l in LOC: s = s.replace(l, "<LOC>")
    return re.sub(r"\s+", " ", s).strip()

tr = ld(TR); ho = ld(HO)
fails = []

# GATE 0 - INFORMATIONAL. Carrier-word hits are reported, not blocking.
# Judgement: a hit is only disqualifying if the SAME LEADING VERB maps to a
# different action with a LEGAL target. Destination-driven rejects are fine.
for w in ("drop", "set ", "stick", "head over", "shift", "make your way"):
    hits = [(r["expected"]["action"], r["command"]) for r in tr + ho if w in r["command"].lower()]
    if hits:
        print(f"GATE 0  [info] {w!r}: {len(hits)} existing rows")
        for a, c in hits[:6]: print(f"          {a} | {c}")

# BUILD
P = ["drop the {o} in {b}", "set the {o} in {b}", "stick the {o} in {b}"]
M = ["head over to {b}", "shift over to {b}", "make your way to {b}"]
OBJS = [("bottle", "bottle"), ("rubiks cube", "rubiks_cube"), ("toy car", "toy_car")]
BOXES = [("box 1", "box_1"), ("box 2", "box_2"), ("box 3", "box_3")]

new = []
for bs, bsym in BOXES:
    for t in P:
        for os_, osym in OBJS:
            new.append({"command": t.format(o=os_, b=bs),
                        "expected": {"action": "pick_place", "target": osym,
                                     "location": bsym, "modifiers": []}})
    for t in M:
        new.append({"command": t.format(b=bs),
                    "expected": {"action": "move_to", "location": bsym, "modifiers": []}})

print(f"\nBUILT {len(new)} rows")
if len(new) != 36: fails.append(f"expected 36 rows, built {len(new)}")

bad = [r["command"] for r in new if any(w in r["command"].lower()
       for w in ("middle", "center", "centre", "grid_"))]
print(f"GATE 1  forbidden tokens: {len(bad)}")
if bad: fails.append(f"forbidden tokens in {len(bad)} rows")

ex = {r["command"] for r in new} & ({r["command"] for r in tr} | {r["command"] for r in ho})
print(f"GATE 2  exact command overlap: {len(ex)}", sorted(ex)[:3])
if ex: fails.append(f"{len(ex)} exact command duplicates")

old_sk = {skel(r["command"]) for r in tr + ho}
ov = sorted({skel(r["command"]) for r in new} & old_sk)
print(f"GATE 3  skeleton overlap: {len(ov)}", ov[:3])
if ov: fails.append(f"{len(ov)} skeleton collisions")

for r in new:
    want = ["action","target","location","modifiers"] if r["expected"]["action"]=="pick_place" \
           else ["action","location","modifiers"]
    if list(r["expected"].keys()) != want: fails.append(f"key order wrong: {r}")
print(f"GATE 4  key order/shape: {'OK' if not any('key order' in f for f in fails) else 'FAIL'}")

if fails:
    print("\n*** BLOCKED ***")
    for f in fails: print("  -", f)
    sys.exit(1)
print("\nALL GATES PASS.")

if not APPLY:
    print("Dry run. Re-run with --apply to append.")
    sys.exit(0)

raw = open(HO, "rb").read()
if not raw.endswith(b"\n"): print("ABORT: file lacks trailing newline"); sys.exit(1)
nl = "\r\n" if b"\r\n" in raw else "\n"
shutil.copyfile(HO, HO + ".bak_pregate")
with open(HO, "a", encoding="utf-8", newline="") as f:
    for r in new: f.write(json.dumps(r, ensure_ascii=False) + nl)
print(f"APPENDED 36 rows. Backup at {HO}.bak_pregate")
print("held_out_v3 now:", len(ld(HO)), "rows")
