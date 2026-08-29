r"""llm/prompts_v5.py -- V5 DEMO legend. Not a measurement condition.

prompts.py (V3) and prompts_v4.py are untouched: they are the provenance of the
30,380 V3 rows and the V4 conditions respectively. This module reuses every
function in prompts.py and swaps only the legend, so "identical apart from the
legend" holds by construction.

THIS IS A DEMO CONFIGURATION. No number produced under it is comparable with
V3 or V4, and none is reported.

WHAT V5 FIXES

1. The "middle" collision (present in V3).
   V3 labelled row B "middle" AND column 2 "middle". grid_B2 was the only cell
   where both readings agreed and 871 predictions collapsed onto it.
   V5 gives row B NO label word at all -- it is defined positionally, "the row
   between A and C". `middle` and `center` therefore name exactly one axis.

2. The front-to-back contradiction (introduced by me in V4).
   V4 wrote "Rows run front-to-back: A = far from the robot". "front-to-back"
   implies A is the front; "far from the robot" says it is the back. The two
   halves of one line disagreed, and gpt-4o-mini's V4 few_shot regression was
   dominated by A->C inversion (grid_A1->grid_C1 x49).
   V5 states the row order once, in one direction, with no second framing.

3. Both row vocabularies are named.
   Training says far/near; the evaluation set says upper/bottom. V3 and V4 named
   only one pair, so the other was unmapped. V5 names both as synonyms for the
   same row, which is what the data actually shows.

KEPT FROM V4a-lite (which measurably worked)
   No literal cell id anywhere -- llama3.2 anchored on `grid_B2` when V4 spelled
   it out, and removing it took the anchoring rate from 40.3% back to 11.1%.
   No arrow notation. Rule in prose.

VERIFIED MECHANICALLY by audit/check_legend_v5.py before first use.
"""
import contextlib

from llm import prompts as _v3

load_schema = _v3.load_schema
LEGEND_VERSION = "v5"


def _grid_legend_v5(schema: dict) -> str:
    if schema.get("schema_version", 1) < 2:
        return ""
    if not any(loc.startswith("grid_") for loc in schema.get("locations", [])):
        return ""
    return (
        "\nGrid locations are squares on the table surface, named grid_<ROW><COL>.\n"
        "\n"
        "Rows, ordered from the far side of the table to the near side:\n"
        "  A is the far row. A command may also call it the upper row.\n"
        "  B is the row between A and C.\n"
        "  C is the near row. A command may also call it the bottom row.\n"
        "\n"
        "Columns, ordered from one side of the table to the other:\n"
        "  1 is the left column.\n"
        "  2 is the middle column. A command may also call it the center column.\n"
        "  3 is the right column.\n"
        "\n"
        '"middle" and "center" name a column. If the command also names a\n'
        "column, they name row B instead. If the command names neither a row nor\n"
        "a column, they mean the square where row B meets column 2.\n"
        "\n"
        "max_reach is the square the arm can still touch at its farthest. A\n"
        'command asks for it with "the farthest spot you can reach" or "the\n'
        'maximum distance you can reach".\n'
        "\n"
        "nearest_free_cell is any square that is free; the robot picks which one.\n"
        'A command asks for it with "anywhere you want" or "anywhere you see\n'
        'free". It is not a fixed square.\n'
    )


@contextlib.contextmanager
def _v5_legend():
    original = _v3._grid_legend
    _v3._grid_legend = _grid_legend_v5
    try:
        yield
    finally:
        _v3._grid_legend = original


def build_prompt(command, strategy, schema, examples=None):
    with _v5_legend():
        return _v3.build_prompt(command, strategy, schema, examples=examples)


def legend_text(schema):
    return _grid_legend_v5(schema)
