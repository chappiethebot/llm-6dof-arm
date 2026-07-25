"""
Generate schema/schema_v2.json from schema/grid_spec.json.

Output format MIRRORS the real v1 schema.json (bespoke dict with per-verb
`required` lists), NOT JSON Schema draft-07. v2 adds `schema_version`,
`location_classes`, and generation provenance. Nothing else about the
contract changes.

Why a generator and not a hand-written enum:
    The legal location set is an OUTPUT of the Step 1 reachability sweep.
    When the sweep lands you edit grid_spec.json and re-run this.

Usage (PowerShell, from repo root):
    python schema/gen_schema_v2.py
    python schema/gen_schema_v2.py --legal-only     # only after the sweep

--legal-only refuses to run while reachability.source == "unmeasured".
The pitch guard refuses to run once geometry is known and the grid is too
fine to be distinguishable at the backend's success tolerance.
"""

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC_PATH = HERE / "grid_spec.json"
OUT_PATH = HERE / "schema_v2.json"


def cell_id(row, col):
    return f"grid_{row}{col}"


def classify(row, col, rows, cols):
    """corner | edge | interior — drives the Step 7 layout-generalization filter."""
    on_row_edge = row in (rows[0], rows[-1])
    on_col_edge = col in (cols[0], cols[-1])
    if on_row_edge and on_col_edge:
        return "corner"
    if on_row_edge or on_col_edge:
        return "edge"
    return "interior"


def check_pitch(spec):
    """Refuse a grid finer than the backend can resolve. Silent otherwise."""
    g = spec["geometry"]
    rows, cols = spec["grid"]["rows"], spec["grid"]["cols"]
    sx, sy, min_pitch = g.get("usable_span_x_m"), g.get("usable_span_y_m"), g.get("min_pitch_m")
    if sx is None or sy is None or min_pitch is None:
        return  # geometry unmeasured; nothing to check yet
    px = sx / (len(rows) - 1) if len(rows) > 1 else float("inf")
    py = sy / (len(cols) - 1) if len(cols) > 1 else float("inf")
    if min(px, py) <= min_pitch:
        raise SystemExit(
            f"REFUSED: cell pitch {min(px, py):.3f} m <= min_pitch {min_pitch:.3f} m. "
            f"Adjacent cells are not distinguishable at the backend success tolerance. "
            f"Shrink the grid or enlarge the usable span in grid_spec.json."
        )


def build(spec, legal_only):
    rows, cols = spec["grid"]["rows"], spec["grid"]["cols"]
    reach = spec["reachability"]

    if legal_only:
        if reach.get("source") == "unmeasured" or not reach.get("legal_cells"):
            raise SystemExit(
                "REFUSED: --legal-only requested but reachability is unmeasured. "
                "Run Step 1 (sim/reachability.py) first."
            )
        allowed = set(reach["legal_cells"])
    else:
        allowed = None

    classes = {}
    for c in spec["carried_from_v1"]["containers"]:
        classes[c] = "container"

    for r in rows:
        for c in cols:
            cid = cell_id(r, c)
            if allowed is not None and cid not in allowed:
                continue
            classes[cid] = classify(r, c, rows, cols)

    for s in spec["runtime_symbols"]:
        classes[s["symbol"]] = "runtime_symbol"

    retired = set(spec["retired_v1_locations"]) - {"_note"}
    leaked = retired & set(classes)
    if leaked:
        raise SystemExit(f"REFUSED: retired v1 location(s) leaked into the v2 enum: {sorted(leaked)}")

    order = {"container": 0, "corner": 1, "edge": 1, "interior": 1, "runtime_symbol": 2}
    locations = sorted(classes, key=lambda k: (order[classes[k]], k))
    return classes, locations


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--legal-only", action="store_true")
    args = ap.parse_args()

    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    check_pitch(spec)
    classes, locations = build(spec, args.legal_only)
    v1 = spec["carried_from_v1"]

    schema = {
        "schema_version": 2,
        "generated_by": "schema/gen_schema_v2.py",
        "generated_from": {
            "file": "grid_spec.json",
            "spec_version": spec["spec_version"],
            "reachability_source": spec["reachability"]["source"],
            "enum_restricted_to_legal_cells": bool(args.legal_only),
        },
        "verbs": v1["verbs"],
        "targets": v1["targets"],
        "locations": locations,
        "location_classes": classes,
        "modifiers_allowed": v1["modifiers_allowed"],
    }

    OUT_PATH.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")

    import collections
    tally = collections.Counter(classes.values())
    print(f"wrote {OUT_PATH}")
    print(f"locations: {len(locations)}  (v1 had 8)")
    print(f"  by class: {dict(tally)}")
    print(f"reachability source: {spec['reachability']['source']}")


if __name__ == "__main__":
    main()
