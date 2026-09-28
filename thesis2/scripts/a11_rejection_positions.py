"""thesis2/scripts/a11_rejection_positions.py -- A11: where does the classifier
draft first diverge from the verifier?

RECOVERED 2026-09-22 from the session scratchpad where it was first run
(`a11_reject.py`); committed here so `a11_rejection_positions.csv` has a
generating script in the repository. Re-running it reproduces that file byte
for byte.

NO NEW INFERENCE. The A8/A10 drafter is the flat DistilBERT classifier, whose
217 predictions are already saved in p45_predictions_held_out_v3.csv; the draft
is a deterministic json.dumps of them (verified: the reconstructed token length
matches a8_speculative_percommand.csv's draft_tokens on 217/217). The
verifier's output is the saved greedy_text. Only the Qwen tokenizer is loaded.

A BPE token can straddle two schema regions -- '_C' covers the 'grid_' prefix
AND the row letter -- so a rejection is attributed to the region of the FIRST
CHARACTER at which draft and verifier text differ, not to the token's majority
region.

Run (conda env thesis2gpu):  python thesis2/scripts/a11_rejection_positions.py

    --split held_out_v4c   (added 2026-09-28) the same analysis on held_out_v4c:
                           drafts from p45_predictions_held_out_v4c.csv, verifier
                           text from the final run's
                           thesis2/out/final/a8_speculative_percommand_held_out_v4c.csv,
                           output to thesis2/out/final/a11_rejection_positions_held_out_v4c.csv.
                           The default (held_out_v3) is unchanged and still
                           reproduces thesis2/out/a11_rejection_positions.csv.
"""
import csv, json, sys, collections

R = r"C:\Users\USER\llm-6dof-arm"
OUT = R + r"\thesis2\out"
csv.field_size_limit(2 ** 31 - 1)
rd = lambda p: list(csv.DictReader(open(p, encoding="utf-8", newline="")))
sch = json.load(open(R + r"\schema\schema_v2.json"))
V = list(sch["verbs"]); T = sch["targets"]
REQ = {v: sch["verbs"][v]["required"] for v in V}
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
IM_END = tok.convert_tokens_to_ids("<|im_end|>")
SPLIT = sys.argv[sys.argv.index("--split") + 1] if "--split" in sys.argv else "held_out_v3"
assert SPLIT in ("held_out_v3", "held_out_v4c"), SPLIT
ho = [json.loads(l) for l in open(R + "\\data\\%s.jsonl" % SPLIT, encoding="utf-8") if l.strip()]
p45 = rd(OUT + "\\p45_predictions_%s.csv" % SPLIT)
a8 = rd(OUT + r"\a8_speculative_percommand.csv" if SPLIT == "held_out_v3"
        else OUT + r"\final\a8_speculative_percommand_held_out_v4c.csv")
CSV_OUT = (OUT + r"\a11_rejection_positions.csv" if SPLIT == "held_out_v3"
           else OUT + r"\final\a11_rejection_positions_held_out_v4c.csv")


def render(v, t, loc):
    o = {"action": v}
    if "target" in REQ[v]:
        o["target"] = t
    if "location" in REQ[v]:
        o["location"] = loc
    o["modifiers"] = []
    return json.dumps(o)


def regions(txt, obj):
    """char index -> schema region, for the rendered draft."""
    reg = ["json_syntax"] * len(txt)
    v = obj["action"]
    s = txt.index('"action": "') + len('"action": "')
    for j in range(s, s + len(v)):
        reg[j] = "verb"
    if "target" in obj:
        s = txt.index('"target": "') + len('"target": "')
        for j in range(s, s + len(obj["target"])):
            reg[j] = "target"
    if "location" in obj:
        loc = obj["location"]
        s = txt.index('"location": "') + len('"location": "')
        if loc.startswith("grid_"):
            for j in range(s, s + 5):
                reg[j] = "loc_type"
            reg[s + 5] = "row"
            reg[s + 6] = "column"
        elif loc.startswith("box_"):
            for j in range(s, s + 4):
                reg[j] = "loc_type"
            reg[s + 4] = "container_index"
        else:
            for j in range(s, s + len(loc)):
                reg[j] = "runtime_symbol"
    return reg


rows = []
for i, (r8, r4) in enumerate(zip(a8, p45)):
    assert r8["command"] == r4["command"] == ho[i]["command"]
    v = V[int(r4["flat_verb_idx"])]
    t = T[int(r4["flat_target_idx"])]
    dtxt = render(v, t, r4["flat_location"])
    draft_obj = json.loads(dtxt)
    enc = tok(dtxt, add_special_tokens=False, return_offsets_mapping=True)
    dids = enc["input_ids"]; offs = enc["offset_mapping"]
    assert len(dids) + 1 == int(r8["draft_tokens"])
    reg = regions(dtxt, draft_obj)
    n_acc = int(r8["draft_accepted"]); k = int(r8["draft_tokens"])
    gids = tok(r8["greedy_text"], add_special_tokens=False)["input_ids"] + [IM_END]
    full = (n_acc == k)
    vtok = tok.decode([gids[n_acc]]) if n_acc < len(gids) else ""
    gtxt = r8["greedy_text"]
    if full:
        r = dict(command_id=r8["command_id"], region="(none: full draft accepted)",
                 draft_token="", verifier_token="", diff_char="")
    elif n_acc == len(dids):
        r = dict(command_id=r8["command_id"], region="stop_token",
                 draft_token="<|im_end|>", verifier_token=vtok, diff_char="")
    else:
        a, b = offs[n_acc]
        # A BPE token can straddle two schema regions (e.g. '_C' covers the
        # 'grid_' prefix AND the row letter), so attribute the rejection to the
        # region of the FIRST CHARACTER at which draft and verifier text differ.
        d = 0
        while d < min(len(dtxt), len(gtxt)) and dtxt[d] == gtxt[d]:
            d += 1
        assert d >= a, (r8["command_id"], d, a, b)
        rs = reg[d] if d < len(reg) else "json_syntax"
        r = dict(command_id=r8["command_id"], region=rs, draft_token=dtxt[a:b],
                 verifier_token=vtok,
                 diff_char="%s->%s" % (dtxt[d] if d < len(dtxt) else "",
                                       gtxt[d] if d < len(gtxt) else ""))
    truth = dict(ho[i]["expected"])
    frame_d = (draft_obj["action"], draft_obj.get("target"), "location" in draft_obj)
    frame_t = (truth["action"], truth.get("target"), truth.get("location") is not None)
    try:
        gobj = json.loads(r8["greedy_text"])
    except Exception:
        gobj = {}
    frame_g = (gobj.get("action"), gobj.get("target"), "location" in gobj)
    r.update(n_accepted=n_acc, n_draft=k, full_draft=full,
             frame_matches_truth=(frame_d == frame_t),
             frame_matches_verifier=(frame_d == frame_g),
             loc_draft=draft_obj.get("location", ""),
             loc_verifier=gobj.get("location", ""),
             loc_truth=truth.get("location") or "",
             spec_passes=int(r8["spec_forward_passes"]), command=r8["command"])
    rows.append(r)

print("=" * 96)
print("A11.1 -- FIRST-REJECTION POSITION BY SCHEMA REGION (%s, n=217, no new inference)"
      % SPLIT)
print("=" * 96)
c = collections.Counter(r["region"] for r in rows)
tot_rej = sum(v for k2, v in c.items() if not k2.startswith("(none"))
print("  %-22s %6s %8s %14s" % ("region", "n", "of 217", "of rejected"))
for k2, v in sorted(c.items(), key=lambda x: -x[1]):
    share = "-" if k2.startswith("(none") else "%.1f%%" % (100 * v / tot_rej)
    print("  %-22s %6d %7.1f%% %14s" % (k2, v, 100 * v / 217, share))
print("  rejected total %d / 217" % tot_rej)
print()
print("  what the two sides put at the first rejected position:")
ex = collections.Counter((r["region"], r["draft_token"], r["verifier_token"])
                         for r in rows if not r["region"].startswith("(none"))
for (rg, d, g), n in sorted(ex.items(), key=lambda x: -x[1]):
    print("    %-16s draft %-14r verifier %-14r  x%d" % (rg, d, g, n))

print()
print("=" * 96)
print("A11.2 -- FRAME CORRECT vs DRAFT ACCEPTED  (frame = verb + target + field set; location VALUE excluded)")
print("=" * 96)
for lbl, key in (("frame vs GROUND TRUTH", "frame_matches_truth"),
                 ("frame vs VERIFIER output", "frame_matches_verifier")):
    m = collections.Counter((r[key], r["full_draft"]) for r in rows)
    print("  %s" % lbl)
    print("    %-22s %16s %16s" % ("", "draft accepted", "draft rejected"))
    print("    %-22s %16d %16d" % ("frame correct", m[(True, True)], m[(True, False)]))
    print("    %-22s %16d %16d" % ("frame wrong", m[(False, True)], m[(False, False)]))
print()
sel = [r for r in rows if r["frame_matches_truth"] and not r["full_draft"]]
print("  frame correct but draft REJECTED (%d), by region: %s"
      % (len(sel), dict(collections.Counter(r["region"] for r in sel))))
sel2 = [r for r in rows if not r["frame_matches_truth"] and r["full_draft"]]
print("  frame WRONG but draft fully ACCEPTED (%d):" % len(sel2))
for r in sel2:
    print("    cmd %-4s %-44s draft=%-16s truth=%s" % (r["command_id"], r["command"][:44],
          r["loc_draft"] or "-", r["loc_truth"] or "-"))
sel3 = [r for r in rows if not r["frame_matches_truth"] and not r["full_draft"]]
print("  frame wrong AND draft rejected (%d), by region: %s"
      % (len(sel3), dict(collections.Counter(r["region"] for r in sel3))))
print()
print("  forward passes by rejection region:")
byr = collections.defaultdict(list)
for r in rows:
    if not r["region"].startswith("(none"):
        byr[r["region"]].append(r["spec_passes"])
for k2, v in sorted(byr.items(), key=lambda x: -len(x[1])):
    print("    %-16s n=%-4d mean %5.1f  min %2d  max %2d" % (k2, len(v), sum(v) / len(v), min(v), max(v)))
acc_by = collections.defaultdict(list)
for r in rows:
    if not r["region"].startswith("(none"):
        acc_by[r["region"]].append(r["n_accepted"] / r["n_draft"])
print()
print("  acceptance fraction at rejection, by region:")
for k2, v in sorted(acc_by.items(), key=lambda x: -len(x[1])):
    print("    %-16s n=%-4d mean %.4f  median %.4f" % (k2, len(v), sum(v) / len(v), sorted(v)[len(v) // 2]))
with open(CSV_OUT, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print()
print("wrote %s" % CSV_OUT)
