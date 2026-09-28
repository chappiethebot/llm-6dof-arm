"""thesis2/final/check_defence_numbers.py -- verify DEFENCE_NUMBERS_rev3.md.

    python thesis2/final/check_defence_numbers.py

Re-reads EVERY cited number in thesis2/out/final/DEFENCE_NUMBERS_rev3.md from the
cell it cites and asserts string equality. Citation grammar:

    `VALUE`⟦F;KEY;SPLIT;COLUMN⟧

    F      m = metrics_final.csv (key = arm)      s = final_significance.csv (key = id)
           e = unified_efficiency.csv (key = arm) v = v5cells_both_splits.csv (key = model/strategy)
    SPLIT  v3 | v4c
    COLUMN a column of that file; row_by_word_group#<group> selects one group's
           "hits/total" fraction from that column.

It also fails if a table data cell contains a digit that is not inside a citation,
so no number can sit in a table unsourced. Shares no code with the generator.
"""
import csv
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "thesis2" / "out" / "final"
DOC = OUT / "DEFENCE_NUMBERS_rev3.md"
FILES = {"m": "metrics_final.csv", "s": "final_significance.csv",
         "e": "unified_efficiency.csv", "v": "v5cells_both_splits.csv"}
SPLIT = {"v3": "held_out_v3", "v4c": "held_out_v4c"}
CITE = re.compile(r"`([^`]*)`⟦([^⟧]*)⟧")


def load(f):
    with (OUT / FILES[f]).open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def lookup(tables, f, key, split, col):
    rows = tables[f]
    if f in ("m", "e"):
        hit = [r for r in rows if r["arm"] == key and r["split"] == SPLIT[split]]
    elif f == "s":
        hit = [r for r in rows if r["id"] == key and r["family"] == "primary"]
        if hit and hit[0]["split"] != SPLIT[split]:
            raise ValueError("hypothesis %s is on %s, cited as %s"
                             % (key, hit[0]["split"], split))
    else:
        hit = [r for r in rows if "%s/%s" % (r["model"], r["strategy"]) == key]
        if not col.startswith(split + "_"):
            raise ValueError("v-file column %r does not belong to split %s" % (col, split))
    if len(hit) != 1:
        raise ValueError("%d rows match %s;%s;%s" % (len(hit), f, key, split))
    row = hit[0]
    if "#" in col:
        base, group = col.split("#", 1)
        parts = dict(p.strip().rsplit(" ", 1) for p in row[base].split(";") if p.strip())
        return parts[group]
    return row[col]


def main():
    if not DOC.exists():
        print("GATE FAIL: %s does not exist" % DOC)
        return 1
    tables = {f: load(f) for f in FILES}
    text = DOC.read_text(encoding="utf-8")
    checked, bad = 0, []
    for m in CITE.finditer(text):
        value, cite = m.group(1), m.group(2)
        parts = cite.split(";")
        if len(parts) != 4:
            bad.append("malformed citation %r" % cite)
            continue
        try:
            src = lookup(tables, *parts)
        except (KeyError, ValueError) as ex:
            bad.append("unresolvable %r: %s" % (cite, ex))
            continue
        checked += 1
        if src.strip() != value.strip():
            bad.append("MISMATCH %r: document %r, source %r" % (cite, value, src))

    # no unsourced number in any table data cell
    # names are not numbers: arm names and V5 cell keys may contain digits
    names = sorted({r["arm"] for r in tables["m"]} | {r["arm"] for r in tables["e"]}
                   | {"%s/%s" % (r["model"], r["strategy"]) for r in tables["v"]}
                   | {r["arm_a"] for r in tables["s"]} | {r["arm_b"] for r in tables["s"]},
                   key=len, reverse=True)
    unsourced = []
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if not line.startswith("|") or set(line) <= set("|- "):
            continue
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if nxt.startswith("|") and set(nxt) <= set("|- "):
            continue                                   # a header row: the next line is |---|
        cells = [c.strip() for c in line.strip("|").split("|")]
        for c in cells[1:]:
            stripped = CITE.sub("", c)
            for nm in names:
                stripped = stripped.replace(nm, "")
            if re.search(r"\d", stripped) and not re.fullmatch(r"(v3|v4c|H\d)", stripped.strip()):
                unsourced.append("line %d: %r" % (i, c[:60]))

    print("  document : %s" % DOC.relative_to(REPO))
    print("  sources  : %s" % ", ".join(FILES.values()))
    print("  citations checked: %d; mismatches or unresolvable: %d; unsourced table "
          "numbers: %d" % (checked, len(bad), len(unsourced)))
    for b in (bad + unsourced)[:25]:
        print("     " + b)
    ok = checked > 0 and not bad and not unsourced
    print("GATE PASS" if ok else "GATE FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
