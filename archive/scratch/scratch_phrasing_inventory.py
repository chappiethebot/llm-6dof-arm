import json, re, collections

def ld(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

TR = "data/train_v2.jsonl"; HO = "data/held_out_v3.jsonl"
tr = ld(TR); ho = ld(HO)

OBJ = ["rubiks cube", "rubik s cube", "rubiks_cube", "rubik cube", "toy car", "toy_car", "bottle"]
LOC = ["box 1", "box 2", "box 3", "box_1", "box_2", "box_3"]

def skel(s):
    s = s.lower()
    for o in OBJ: s = s.replace(o, "<OBJ>")
    for l in LOC: s = s.replace(l, "<LOC>")
    return re.sub(r"\s+", " ", s).strip()

print("=== 1. FILE SHAPE (must match exactly when appending) ===")
raw = open(HO, "rb").read()
print("ends with newline:", raw.endswith(b"\n"))
print("BOM at head:", raw[:3] == b"\xef\xbb\xbf")
print("any BOM inside:", raw.count(b"\xef\xbb\xbf"))
print("last raw line:", raw.splitlines()[-1].decode("utf-8"))
print("key order:", list(ho[-1].keys()), list(ho[-1]["expected"].keys()))

print()
print("=== 2. TRAIN_V2 CONTAINER ROWS — THE FORBIDDEN LIST ===")
cont = [r for r in tr if str(r["expected"].get("location") or "").startswith("box")]
print(f"{len(cont)} rows")
for r in cont:
    print(f'  {r["expected"]["action"]:10} {r["expected"]["location"]}  |  {r["command"]}')

print()
print("=== 3. FORBIDDEN SKELETONS (train_v2, pick_place + move_to) ===")
tr_sk = sorted({skel(r["command"]) for r in tr
                if r["expected"]["action"] in ("pick_place", "move_to")})
for s in tr_sk: print("  ", s)

print()
print("=== 4. SKELETONS ALREADY IN HELD_OUT_V3 (also avoid repeating) ===")
ho_sk = sorted({skel(r["command"]) for r in ho
                if r["expected"]["action"] in ("pick_place", "move_to")})
for s in ho_sk: print("  ", s)

print()
print("=== 5. PRE-EXISTING OVERLAP — is constraint 5 already violated? ===")
overlap = sorted(set(tr_sk) & set(ho_sk))
print(f"skeletons shared train_v2 <-> held_out_v3: {len(overlap)}")
for s in overlap: print("  !", s)
n_rows = sum(1 for r in ho if r["expected"]["action"] in ("pick_place","move_to")
             and skel(r["command"]) in set(tr_sk))
print(f"held_out_v3 rows sitting on a train skeleton: {n_rows} / {len(ho)}")

print()
print("=== 6. CURRENT DISTRIBUTIONS ===")
print("verbs:", dict(collections.Counter(r["expected"]["action"] for r in ho)))
print("locs :", dict(collections.Counter(r["expected"].get("location","<none>") for r in ho)))
