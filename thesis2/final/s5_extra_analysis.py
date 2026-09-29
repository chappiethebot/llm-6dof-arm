"""thesis2/final/s5_extra_analysis.py -- two descriptive analyses. NO MODEL RUNS.

    python thesis2/final/s5_extra_analysis.py

Reads only the final run's prediction files (thesis2/out/final/pred_<arm>_<split>.csv)
and the split files; writes two NEW files and changes nothing else. No significance
test is run: both outputs are counts.

1. row_confusion_by_word.csv
   For flat_gpu, lora_greedy, grpo_base, grpo_sft, textcnn and cascade_cloud_always
   on both splits: over the 108 grid commands, TRUE row word x PREDICTED row
   (A / B / C / none). Row words: far, near, middle, center, upper, bottom, back,
   front (whole word, case-insensitive). "pure" = the command contains exactly one
   of them; "mixed" = more than one (the row is then the combination, e.g.
   middle+upper). "none" = the prediction has no grid location.
   cascade_cloud_always has 5 repeats per command: its predicted row is the MODAL
   row over the 5 (a tie goes to the earliest repeat) -- one prediction per
   command, like every other arm.

2. draft_verifier_gh.csv
   Per split: draft = flat_gpu (the classifier that drafts for A8/A10), verifier
   output = lora_greedy (A8/A10's output is byte-identical to it). Per command:
   G = draft wrong -> verifier right, H = draft right -> verifier wrong,
   both right, both wrong. Checks (G - H) / n == acc(verifier) - acc(draft).
   G is broken down by the field(s) the DRAFT got wrong, H by the field(s) the
   VERIFIER got wrong: verb, target, location row, location column, location
   non-grid (box / runtime symbol / missing), other (all four fields right but the
   action still wrong: an extra field or a modifier). A command can count in
   several fields.
"""
import collections
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C                                                      # noqa: E402

csv.field_size_limit(2 ** 31 - 1)
ROW_WORDS = ["far", "near", "middle", "center", "upper", "bottom", "back", "front"]
ARMS = ["flat_gpu", "lora_greedy", "grpo_base", "grpo_sft", "textcnn",
        "cascade_cloud_always"]
PRED_ROWS = ["A", "B", "C", "none"]


def words_in(cmd):
    low = cmd.lower()
    return [w for w in ROW_WORDS if re.search(r"\b%s\b" % w, low)]


def by_command(arm, split):
    """{command_id: [rows sorted by repeat]} from the final run's pred file."""
    out = collections.defaultdict(list)
    for r in C.read_preds(arm, split):
        out[str(r["command_id"])].append(r)
    for rs in out.values():
        rs.sort(key=lambda r: int(r["repeat"]))
    if not out:
        C.gate(False, ["missing pred_%s_%s.csv" % (arm, split)])
    return out


def pred_row(rows):
    """Modal predicted row over the repeats; a tie goes to the earliest repeat."""
    rws = []
    for r in rows:
        try:
            p = json.loads(r["pred_json"] or "{}")
        except ValueError:
            p = {}
        rws.append(C.rowof(p.get("location") if isinstance(p, dict) else None) or "none")
    cnt = collections.Counter(rws)
    top = max(cnt.values())
    return next(x for x in rws if cnt[x] == top)


def confusion():
    out = []
    for split in C.SPLITS:
        cmds = C.load_split(split)
        for arm in ARMS:
            preds = by_command(arm, split)
            cells = collections.defaultdict(collections.Counter)
            truth_rows = collections.defaultdict(collections.Counter)
            for cid, row in enumerate(cmds):
                t = C.rowof(row["expected"].get("location"))
                if not t:
                    continue                                  # grid commands only
                ws = words_in(row["command"])
                kind = "pure" if len(ws) == 1 else ("mixed" if ws else "no row word")
                key = (kind, "+".join(ws) or "-")
                cells[key][pred_row(preds[str(cid)])] += 1
                truth_rows[key][t] += 1
            for (kind, word), c in sorted(cells.items()):
                n = sum(c.values())
                tr = truth_rows[(kind, word)]
                correct = sum(v for p_, v in c.items() if tr.get(p_) and len(tr) == 1)
                out.append(dict(arm=arm, split=split, kind=kind, row_word=word, n=n,
                                true_row=";".join("%s:%d" % kv for kv in sorted(tr.items())),
                                pred_A=c["A"], pred_B=c["B"], pred_C=c["C"],
                                pred_none=c["none"],
                                correct=correct if len(tr) == 1 else "mixed truth"))
    return out


def fields_wrong(r):
    """The fields this prediction got wrong, from the pred file's own columns
    (written by common.score_row with llm/score.py)."""
    bad = []
    f = lambda k: r[k] in (False, "False")
    if f("verb_correct"):
        bad.append("verb")
    if f("target_correct"):
        bad.append("target")
    if f("row_correct"):
        bad.append("row")
    if f("col_correct"):
        bad.append("column")
    if f("location_correct") and r["row_correct"] in ("", None):
        bad.append("location_nongrid")
    if not bad:
        bad.append("other")
    return bad


def gain_harm():
    out = []
    fields = ["verb", "target", "row", "column", "location_nongrid", "other"]
    for split in C.SPLITS:
        d, v = by_command("flat_gpu", split), by_command("lora_greedy", split)
        n = len(d)
        ok = lambda rs: rs[0]["correct"] in (True, "True")
        g, h, br, bw = [], [], 0, 0
        for cid in d:
            dd, vv = ok(d[cid]), ok(v[cid])
            if not dd and vv:
                g.append(cid)
            elif dd and not vv:
                h.append(cid)
            elif dd and vv:
                br += 1
            else:
                bw += 1
        acc_d = sum(ok(rs) for rs in d.values()) / n
        acc_v = sum(ok(rs) for rs in v.values()) / n
        gf = collections.Counter(x for c in g for x in fields_wrong(d[c][0]))
        hf = collections.Counter(x for c in h for x in fields_wrong(v[c][0]))
        rec = dict(split=split, n=n, draft="flat_gpu", verifier="lora_greedy",
                   acc_draft=round(acc_d, 4), acc_verifier=round(acc_v, 4),
                   acc_diff=round(acc_v - acc_d, 4), G=len(g), H=len(h),
                   both_right=br, both_wrong=bw,
                   G_minus_H_over_n=round((len(g) - len(h)) / n, 4),
                   check_equal=abs((len(g) - len(h)) / n - (acc_v - acc_d)) < 1e-12)
        for fld in fields:
            rec["G_draft_wrong_%s" % fld] = gf[fld]
        for fld in fields:
            rec["H_verifier_wrong_%s" % fld] = hf[fld]
        out.append(rec)
    return out


def write(name, rows):
    p = C.OUT / name
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return p


def main():
    C.start_log("s5_extra_analysis")
    conf = confusion()
    gh = gain_harm()
    p1, p2 = write("row_confusion_by_word.csv", conf), write("draft_verifier_gh.csv", gh)

    print()
    print("ROW CONFUSION BY WORD -- grid commands; counts of predicted row")
    print("  %-22s %-13s %-6s %-14s %4s %-9s %4s %4s %4s %5s %s"
          % ("arm", "split", "kind", "row word", "n", "true", "A", "B", "C", "none", "correct"))
    for r in conf:
        print("  %-22s %-13s %-6s %-14s %4d %-9s %4d %4d %4d %5d %s"
              % (r["arm"], r["split"], r["kind"], r["row_word"], r["n"], r["true_row"],
                 r["pred_A"], r["pred_B"], r["pred_C"], r["pred_none"], r["correct"]))
    print()
    print("BACK / FRONT (held_out_v4c, pure commands) -- all six arms")
    for r in conf:
        if r["kind"] == "pure" and r["row_word"] in ("back", "front"):
            print("  %-22s %-6s n=%d true %s -> A %d  B %d  C %d  none %d"
                  % (r["arm"], r["row_word"], r["n"], r["true_row"], r["pred_A"],
                     r["pred_B"], r["pred_C"], r["pred_none"]))
    print()
    print("DRAFT (flat_gpu) vs VERIFIER (lora_greedy) -- gain / harm")
    for r in gh:
        print("  %s  n=%d  acc draft %.4f  verifier %.4f  diff %+.4f | G %d  H %d  "
              "both right %d  both wrong %d | (G-H)/n %+.4f  equal: %s"
              % (r["split"], r["n"], r["acc_draft"], r["acc_verifier"], r["acc_diff"],
                 r["G"], r["H"], r["both_right"], r["both_wrong"],
                 r["G_minus_H_over_n"], r["check_equal"]))
        print("     G by field the DRAFT got wrong:    %s"
              % {k[len("G_draft_wrong_"):]: v for k, v in r.items() if k.startswith("G_draft")})
        print("     H by field the VERIFIER got wrong: %s"
              % {k[len("H_verifier_wrong_"):]: v for k, v in r.items() if k.startswith("H_verif")})
    C.gate(all(r["check_equal"] for r in gh), [
        "NO MODEL RUNS: read pred_<arm>_<split>.csv and the split files only",
        "wrote %s (%d rows) and %s (%d rows)" % (p1.name, len(conf), p2.name, len(gh)),
        "(G - H) / n equals the accuracy difference on both splits: %s"
        % all(r["check_equal"] for r in gh),
        "descriptive counts only; no significance test",
    ])


if __name__ == "__main__":
    main()
