"""thesis2/final/s0_salsa_diag.py -- diagnose SALSA-full. REPORT ONLY.

    python thesis2/final/s0_salsa_diag.py --demo10      # the 10 seen demo commands
    python thesis2/final/s0_salsa_diag.py --train230    # all 230 train_v2 commands

THE PROMPT IS NOT CHANGED. This script imports SalsaFull and repeats exactly the
three asks the arm makes (`_ask` with the same question text and the same option
lists), purely to expose what was chosen and why. Nothing is trained, no
threshold is selected, and no held-out file is opened.

Why per-field accuracy needs two denominators. The field masks make pass 2 and
pass 3 CONDITIONAL on pass 1: if the predicted verb requires no target, the
target question is never asked, and the same for location. So a field can be
wrong in two different ways -- asked and answered wrongly, or never asked because
the verb was wrong. Both are reported:

  unconditional   over commands whose TRUE verb requires the field; a field that
                  was never asked counts as WRONG. This is the number that
                  matters for accuracy.
  conditional     over commands where the field WAS asked. This isolates the
                  logit read from the verb cascade.
"""
import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C                                                      # noqa: E402
from s0_salsa_full import (SalsaFull, Q_VERB, Q_TARGET, Q_LOCATION,     # noqa: E402
                           options_block, legal_locations, letters_for, LETTERS,
                           LOC_LETTERS, LOC_LETTER_OF, MODELS)

import torch                                                            # noqa: E402


def predict_traced(sf, cmd):
    """The arm's own three passes, with the letter and option list recorded."""
    pids = sf.tok(sf.prefix_text(cmd), add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        out = sf.model(input_ids=torch.tensor([pids], device="cuda"), use_cache=True)
    cache = out.past_key_values
    plen = cache.get_seq_length()
    trace = {}

    v_letters = letters_for(C.VERBS, "verb")
    verb, p_v, _, _ = sf._ask(cache, plen, Q_VERB % options_block(C.VERBS, v_letters),
                              C.VERBS, v_letters)
    trace["verb"] = dict(symbol=verb, letter=v_letters[C.VERBS.index(verb)],
                         options=list(C.VERBS), letters=v_letters, p=p_v)
    need = C.VERB_FIELDS[verb]

    t_idx = 0
    if "target" in need:
        t_letters = letters_for(C.TARGETS, "target")
        tgt, p_t, _, _ = sf._ask(cache, plen, Q_TARGET % options_block(C.TARGETS, t_letters),
                                 C.TARGETS, t_letters)
        t_idx = C.TARGETS.index(tgt)
        trace["target"] = dict(symbol=tgt, letter=t_letters[t_idx],
                               options=list(C.TARGETS), letters=t_letters, p=p_t)

    l_idx = 0
    legal = legal_locations(verb)
    if legal:
        l_letters = letters_for(legal, "location")
        loc, p_l, _, _ = sf._ask(cache, plen, Q_LOCATION % options_block(legal, l_letters),
                                 legal, l_letters)
        l_idx = C.LOCATIONS.index(loc)
        trace["location"] = dict(symbol=loc, letter=LOC_LETTER_OF[loc],
                                 options=legal, letters=l_letters, p=p_l)

    return C.emit(C.VERBS.index(verb), t_idx, l_idx, C.SPACE), trace


def show_disagreements(sf, rows, label):
    print()
    print("=" * 100)
    print("DISAGREEMENTS on %s" % label)
    print("=" * 100)
    bad = []
    for cmd, truth in rows:
        obj, trace = predict_traced(sf, cmd)
        if not bool(C.score(obj, truth).strict_match):
            bad.append((cmd, truth, obj, trace))
    print("  %d of %d commands disagree with the dataset row" % (len(bad), len(rows)))
    for i, (cmd, truth, obj, trace) in enumerate(bad, 1):
        diff = [f for f in ("action", "target", "location") if truth.get(f) != obj.get(f)]
        print()
        print("--- %d. %s" % (i, cmd))
        print("    true JSON  : %s" % json.dumps(truth))
        print("    SALSA JSON : %s" % json.dumps(obj))
        print("    field(s) differing: %s" % ", ".join(diff))
        for field, key in (("verb", "action"), ("target", "target"), ("location", "location")):
            if field not in trace:
                print("    %-9s NOT ASKED -- the predicted verb %r requires no %s"
                      % (field, obj.get("action"), field))
                continue
            t = trace[field]
            mark = "  <-- DIFFERS" if key in diff else ""
            print("    %-9s chose %r = %-18s p=%.4f%s"
                  % (field, t["letter"], t["symbol"], t["p"], mark))
            print("              exact option list in this field's prompt (%d options):"
                  % len(t["options"]))
            print("              %s" % "  ".join("%s=%s" % (t["letters"][j], s)
                                                 for j, s in enumerate(t["options"])))
            if key in diff:
                want = truth.get(key)
                if want in t["options"]:
                    print("              the TRUE symbol %r was option %r"
                          % (want, t["letters"][t["options"].index(want)]))
                else:
                    print("              the TRUE symbol %r was NOT in the option list" % want)
    return bad


def per_field(sf, rows, label):
    print()
    print("=" * 100)
    print("PER-FIELD ACCURACY on %s (n=%d)" % (label, len(rows)))
    print("=" * 100)
    n = len(rows)
    hit = collections.Counter()
    den_true = collections.Counter()      # truth requires the field
    den_asked = collections.Counter()     # the field was actually asked
    hit_asked = collections.Counter()
    action_ok = 0
    verb_conf = collections.Counter()
    row_hit = row_den = col_hit = col_den = 0
    recs = []
    for i, (cmd, truth) in enumerate(rows):
        obj, trace = predict_traced(sf, cmd)
        ok = bool(C.score(obj, truth).strict_match)
        action_ok += ok
        # verb
        den_true["verb"] += 1
        v_ok = obj.get("action") == truth.get("action")
        hit["verb"] += v_ok
        if not v_ok:
            verb_conf[(truth.get("action"), obj.get("action"))] += 1
        # target / location
        for field, key in (("target", "target"), ("location", "location")):
            if truth.get(key) is not None:
                den_true[field] += 1
                hit[field] += (obj.get(key) == truth.get(key))
            if field in trace:
                den_asked[field] += 1
                hit_asked[field] += (obj.get(key) == truth.get(key))
        # row / col on grid truths
        tl = truth.get("location")
        if C.rowof(tl):
            row_den += 1
            col_den += 1
            row_hit += (C.rowof(obj.get("location")) == C.rowof(tl))
            col_hit += (C.colof(obj.get("location")) == C.colof(tl))
        recs.append(dict(command=cmd, truth=json.dumps(truth), pred=json.dumps(obj),
                         correct=ok,
                         verb_letter=trace["verb"]["letter"], verb_p=trace["verb"]["p"],
                         target_letter=trace.get("target", {}).get("letter", ""),
                         location_letter=trace.get("location", {}).get("letter", ""),
                         location_n_options=len(trace.get("location", {}).get("options", []))))
        if (i + 1) % 50 == 0:
            print("     %3d/%d" % (i + 1, n))

    print()
    print("  action match (full object, llm/score.py strict) : %d/%d = %.4f"
          % (action_ok, n, action_ok / n))
    print()
    print("  %-10s %-34s %-34s" % ("field", "UNCONDITIONAL (truth needs it)",
                                   "CONDITIONAL (field was asked)"))
    for field in ("verb", "target", "location"):
        u = ("%d/%d = %.4f" % (hit[field], den_true[field], hit[field] / den_true[field])
             if den_true[field] else "n/a")
        if field == "verb":
            c = "same as unconditional (always asked)"
        else:
            c = ("%d/%d = %.4f" % (hit_asked[field], den_asked[field],
                                   hit_asked[field] / den_asked[field])
                 if den_asked[field] else "never asked")
        print("  %-10s %-34s %-34s" % (field, u, c))
    print()
    print("  location sub-factors on grid truths: row %d/%d = %.4f   column %d/%d = %.4f"
          % (row_hit, row_den, row_hit / row_den if row_den else float("nan"),
             col_hit, col_den, col_hit / col_den if col_den else float("nan")))
    print()
    print("  field-mask cascade: %d command(s) where truth needs a target but none was "
          "asked; %d where truth needs a location but none was asked"
          % (den_true["target"] - den_asked["target"] if den_true["target"] > den_asked["target"] else 0,
             den_true["location"] - den_asked["location"] if den_true["location"] > den_asked["location"] else 0))
    print("  (a field never asked counts as WRONG in the unconditional column)")
    if verb_conf:
        print()
        print("  verb confusion (truth -> predicted), %d errors:" % sum(verb_conf.values()))
        for (t, p), k in verb_conf.most_common():
            print("     %-12s -> %-12s x%d" % (t, p, k))
    return recs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--demo10", action="store_true", help="the 10 seen demo commands")
    ap.add_argument("--train230", action="store_true", help="all train_v2 commands")
    ap.add_argument("--model", default="qwen2.5-3b-lora", choices=sorted(MODELS))
    a = ap.parse_args()
    if not (a.demo10 or a.train230):
        ap.error("pick --demo10, --train230, or both")
    C.start_log("s0_salsa_diag_%s" % a.model)
    print("  model: %s" % a.model)
    print("  THE PROMPT IS NOT MODIFIED: this replays the arm's own three asks.")
    sf = SalsaFull(a.model)

    if a.demo10:
        rows = []
        for line in (C.REPO / "thesis2" / "demo" / "demo_commands.txt").read_text(
                encoding="utf-8").splitlines():
            if line.strip() and not line.startswith("#"):
                cmd, exp = line.split("\t")
                e = json.loads(exp)
                rows.append((cmd, dict(e, modifiers=e.get("modifiers", []))))
        show_disagreements(sf, rows, "the 10 seen demo commands (data/seen_v2.jsonl)")

    if a.train230:
        rows = [(r["command"], C.truth_of(r)) for r in C.load_split(C.TRAIN)]
        recs = per_field(sf, rows, "all %s commands" % C.TRAIN)
        import csv
        p = C.OUT / ("salsa_diag_train_v2_%s.csv" % a.model)
        with p.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(recs[0].keys()))
            w.writeheader(); w.writerows(recs)
        print()
        print("  per-command detail: %s" % p.relative_to(C.REPO))

    C.gate(True, [
        "REPORT ONLY: no prompt change, no training, no selection, no held-out file",
        "train_v2 is training data and is explicitly permitted for this diagnostic",
    ])


if __name__ == "__main__":
    main()
