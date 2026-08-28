r"""llm/prompts_v4.py -- V4 legend. LEGEND TEXT ONLY; no other prompt change.

llm/prompts.py is NOT modified: it is the provenance of the 30,380 measured
rows. This module reuses every one of its functions and swaps ONLY the legend
block, so "identical apart from the legend" is guaranteed by construction
rather than by a test -- there is literally no second code path. build_prompt
below calls prompts.build_prompt with prompts._grid_legend temporarily bound to
the V4 text.

WHAT CHANGED, AND WHY
=====================

FIX 1 -- row B and column 2 no longer share a label.
  V3 legend:  Rows:    A = far from the robot,  B = middle,  C = near the robot.
              Columns: 1 = left,  2 = middle,  3 = right.
  `middle` labelled BOTH axes. grid_B2 is the only cell where the two readings
  agree, and 871 predictions collapsed onto it (audit/vocab_conflicts.md §1).
  V4 labels row B `centre` and column 2 `middle`, which are now disjoint.

  BUT A PURE RELABEL WOULD BE WRONG, and this is the part worth reading. In the
  actual commands `middle` genuinely names both axes, and which one it names is
  fully determined by context. Measured over all 120 grid commands in seen_v2 +
  held_out_v3 that contain middle/centre/center, with ZERO exceptions:

      middle + a row word (far, near, upper, bottom)  -> names the COLUMN -> 2   48/48
      middle + a column word (left, right)            -> names the ROW    -> B   48/48
      middle alone                                    -> means grid_B2          24/24

  A legend that simply declared "middle = column 2" would therefore contradict
  the 48 commands where it means row B. V4 states the disambiguation RULE
  instead. The rule is not invented; it is read off the data and holds
  everywhere it can be tested.

FIX 2 -- the runtime symbols are defined in words commands actually use.
  V3 legend:  max_reach = the furthest point the arm can reach; resolved by the robot.
              nearest_free_cell = whichever square is currently empty; resolved by the robot.
  `furthest`, `empty` and `nearest` appear in ZERO commands across all four
  splits. Both definitions were written in vocabulary no command ever uses.
  What commands actually say:

      max_reach          train: "the farthest spot you can reach"
                         held:  "the maximum distance you can reach"
                         shared: "you can reach"
      nearest_free_cell  train: "anywhere you see free"
                         held:  "anywhere you want"
                         shared: "anywhere you"

  Note this is the SAME train/held vocabulary split that broke the row axis
  (farthest/maximum, free/want are disjoint across splits), which is a better
  explanation of the 55% nearest_free_cell failure than ambiguity alone.
  V4 anchors each definition on the SHARED wording and lists both variants.

  V4 also fixes a name/definition contradiction: nearest_free_cell is named for
  nearness but its commands mean ANY free square ("anywhere you want"), not the
  closest one. V4 says so explicitly, because a model reasoning from the symbol
  name will otherwise look for a nearest-something that the task never asks for.

NOT CHANGED: the schema, the verb/target/location lists, the output rule, the
example format, every strategy branch, and all four datasets. held_out_v3.jsonl
is untouched and remains sealed.

COMPARABILITY WARNING: any run under the V4 legend is NOT comparable with the
30,380 rows measured under V3. Report them as separate conditions.
"""
import contextlib

from llm import prompts as _v3

# Re-export the V3 surface unchanged so callers can swap module and nothing else.
load_schema = _v3.load_schema
EXAMPLE_STRATEGIES = getattr(_v3, "EXAMPLE_STRATEGIES", None)
STRATEGIES = getattr(_v3, "STRATEGIES", None)

LEGEND_VERSION = "v4"


def _grid_legend_v4(schema: dict) -> str:
    """V4 legend. Same guard conditions as V3: empty for V1 schemas."""
    if schema.get("schema_version", 1) < 2:
        return ""
    if not any(loc.startswith("grid_") for loc in schema.get("locations", [])):
        return ""
    return (
        "\nGrid locations are squares on the table surface, named grid_<ROW><COL>.\n"
        "Rows run front-to-back:   A = far from the robot,  B = centre,  C = near the robot.\n"
        "Columns run side-to-side: 1 = left,  2 = middle,  3 = right.\n"
        "\n"
        'The words "middle", "centre" and "center" can refer to either axis.\n'
        "Decide which from the rest of the command:\n"
        "  - alongside a row word (far, near, upper, bottom) it names the COLUMN -> 2\n"
        "  - alongside a column word (left, right) it names the ROW -> B\n"
        "  - on its own it means the centre square -> grid_B2\n"
        "\n"
        "max_reach = the square farthest from the robot that the arm can still\n"
        '  reach. Commands say "the farthest spot you can reach" or "the maximum\n'
        '  distance you can reach".\n'
        "nearest_free_cell = ANY square that is currently unoccupied; the robot\n"
        '  chooses which one. Commands say "anywhere you want" or "anywhere you\n'
        '  see free". Despite the name it does NOT mean the closest square, and\n'
        "  it is not a fixed position.\n"
    )


def _grid_legend_v4lite(schema: dict) -> str:
    """V4a-lite: same rule as V4, stated in prose only.

    V4 measured a regression on llama3.2 (few_shot -0.178, rag -0.155) while
    deepseek gained (+0.214 zero_shot). Diagnosis: llama3.2 anchored on the two
    concrete hooks the V4 rule block offers a small model -- the literal token
    `grid_B2` (the only cell id written anywhere in the prompt) and the `->`
    arrow notation. Its rag failures are systematically truth grid_A1 predicted
    grid_B2, i.e. the collapse V4 was meant to prevent, made worse.

    This variant removes both hooks and changes nothing else: no cell id spelled
    out, no arrows, rule expressed as sentences.
    """
    if schema.get("schema_version", 1) < 2:
        return ""
    if not any(loc.startswith("grid_") for loc in schema.get("locations", [])):
        return ""
    return (
        "\nGrid locations are squares on the table surface, named grid_<ROW><COL>.\n"
        "Rows run front-to-back:   A = far from the robot,  B = centre,  C = near the robot.\n"
        "Columns run side-to-side: 1 = left,  2 = middle,  3 = right.\n"
        "\n"
        'The words "middle", "centre" and "center" can refer to either axis. If the\n'
        "command also gives a row word, then middle describes the column. If the\n"
        "command also gives a column word, then middle describes the row. If the\n"
        "command gives neither, it means the square at the centre of the table.\n"
        "\n"
        "max_reach = the square farthest from the robot that the arm can still\n"
        '  reach. Commands say "the farthest spot you can reach" or "the maximum\n'
        '  distance you can reach".\n'
        "nearest_free_cell = any square that is currently unoccupied; the robot\n"
        '  chooses which one. Commands say "anywhere you want" or "anywhere you\n'
        '  see free". Despite the name it does not mean the closest square.\n'
    )


@contextlib.contextmanager
def _v4_legend():
    original = _v3._grid_legend
    _v3._grid_legend = _grid_legend_v4
    try:
        yield
    finally:
        _v3._grid_legend = original


def build_prompt(command: str, strategy: str, schema: dict,
                 examples: list[dict] | None = None) -> str:
    """Identical to prompts.build_prompt except for the legend block."""
    with _v4_legend():
        return _v3.build_prompt(command, strategy, schema, examples=examples)


def legend_text(schema: dict) -> str:
    return _grid_legend_v4(schema)


def legend_text_v3(schema: dict) -> str:
    return _v3._grid_legend(schema)


if __name__ == "__main__":
    import difflib, sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    schema = load_schema("schema/schema_v2.json")
    a, b = legend_text_v3(schema), legend_text(schema)
    print("=" * 78)
    print("V3 LEGEND")
    print("=" * 78)
    print(a)
    print("=" * 78)
    print("V4 LEGEND")
    print("=" * 78)
    print(b)
    print("=" * 78)
    print("UNIFIED DIFF")
    print("=" * 78)
    for line in difflib.unified_diff(a.splitlines(), b.splitlines(),
                                     "legend_v3", "legend_v4", lineterm="", n=1):
        print(line)
