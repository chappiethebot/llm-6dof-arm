# Vocabulary conflicts in the command datasets

Read-only scan. Nothing was modified.

Sources: `schema/schema_v2.json`, the legend emitted by `llm/prompts.py:_grid_legend()`, and every command in `seen_v2` (290), `train_v2` (230), `dev_v2` (57), `held_out_v3` (217).

**Pricing convention.** Each held-out command appears 5 times per cell (5 repeats) in each of 28 measured cells, so *1 command = 5 rows per cell = 140 rows across the whole measurement*. Costs below are given in commands and in rows-per-cell, which is the number that moves a reported accuracy.

## The legend, verbatim

```
Grid locations are squares on the table surface, named grid_<ROW><COL>.
Rows:    A = far from the robot,  B = middle,  C = near the robot.
Columns: 1 = left,  2 = middle,  3 = right.
max_reach = the furthest point the arm can reach; resolved by the robot.
nearest_free_cell = whichever square is currently empty; resolved by the robot.
```

---

## 1. CONFIRMED — `middle` is used for BOTH a row and a column

This is the headline conflict and it is visible in the legend text itself:

| legend line | says `middle` means |
|---|---|
| `Rows:    A = far from the robot,  B = middle,  C = near the robot.` | row **B** |
| `Columns: 1 = left,  2 = middle,  3 = right.` | column **2** |

A command containing `middle` is therefore under-determined by the legend alone. `grid_B2` is the only cell where both readings agree, which is exactly where predictions pile up.

**RECONCILIATION OF THE COUNT — the reported 871 is confirmed, and the difference from other numbers below is a definition, not a discrepancy.**

| population | truth symbols counted | rows onto `grid_B2` | column-2 share |
|---|---|---|---|
| 20 rung-1 cells, validated rows *(the reported figure)* | all 14 | **871** | **91.4%** |
| 20 rung-1 cells, validated rows | 9 grid cells only | 861 | — |
| all 28 cells, all rows | all 14 | 1,417 | 89.8% |

**The reported 871 rows at 91.4% reproduces exactly.** The 861 figure is the same measurement restricted to grid-cell truths; the 10 row difference is commands whose ground truth is `nearest_free_cell` that were answered `grid_B2` — a runtime symbol collapsing onto the middle cell, which is section 3's problem, not section 1's. Quoting one number without its truth-set definition is how these drift.

Per true cell, all 28 cells, all rows:

| true cell | rows predicted `grid_B2` | % of that cell's rows |
|---|---|---|
| `grid_C2` | 654 | 38.9% |
| `grid_A2` | 618 | 36.8% |
| `grid_B3` | 80 | 4.8% |
| `grid_B1` | 50 | 3.0% |
| `nearest_free_cell` | 10 | 0.6% |
| `grid_A3` | 5 | 0.3% |
| **total** | **1,417** | — |


### Every held-out command containing a `middle`-class word

60 of 217 held-out commands = **300 rows per cell**.

| # | command | ground truth |
|---|---|---|
| 1 | `can you please keep the bottle in the upper side middle edge of the table?` | `grid_A2` |
| 2 | `can you please keep the car in the upper side middle edge of the table?` | `grid_A2` |
| 3 | `can you please keep the rubiks cube in the upper side middle edge of the table?` | `grid_A2` |
| 4 | `go to the upper side middle edge of the table` | `grid_A2` |
| 5 | `move to the upper side middle edge of the table` | `grid_A2` |
| 6 | `navigate to the upper side middle edge of the table` | `grid_A2` |
| 7 | `pick up the bottle and keep in the upper side middle edge of the table` | `grid_A2` |
| 8 | `pick up the car and keep in the upper side middle edge of the table` | `grid_A2` |
| 9 | `pick up the rubiks cube and keep in the upper side middle edge of the table` | `grid_A2` |
| 10 | `put the bottle in the upper side middle edge of the table` | `grid_A2` |
| 11 | `put the car in the upper side middle edge of the table` | `grid_A2` |
| 12 | `put the rubiks cube in the upper side middle edge of the table` | `grid_A2` |
| 13 | `can you please keep the bottle in the left side edge from the table center?` | `grid_B1` |
| 14 | `can you please keep the car in the left side edge from the table center?` | `grid_B1` |
| 15 | `can you please keep the rubiks cube in the left side edge from the table center?` | `grid_B1` |
| 16 | `go to the left side edge from the table center` | `grid_B1` |
| 17 | `move to the left side edge from the table center` | `grid_B1` |
| 18 | `navigate to the left side edge from the table center` | `grid_B1` |
| 19 | `pick up the bottle and keep in the left side edge from the table center` | `grid_B1` |
| 20 | `pick up the car and keep in the left side edge from the table center` | `grid_B1` |
| 21 | `pick up the rubiks cube and keep in the left side edge from the table center` | `grid_B1` |
| 22 | `put the bottle in the left side edge from the table center` | `grid_B1` |
| 23 | `put the car in the left side edge from the table center` | `grid_B1` |
| 24 | `put the rubiks cube in the left side edge from the table center` | `grid_B1` |
| 25 | `can you please keep the bottle in the middle of the table?` | `grid_B2` |
| 26 | `can you please keep the car in the middle of the table?` | `grid_B2` |
| 27 | `can you please keep the rubiks cube in the middle of the table?` | `grid_B2` |
| 28 | `go to the middle of the table` | `grid_B2` |
| 29 | `move to the middle of the table` | `grid_B2` |
| 30 | `navigate to the middle of the table` | `grid_B2` |
| 31 | `pick up the bottle and keep in the middle of the table` | `grid_B2` |
| 32 | `pick up the car and keep in the middle of the table` | `grid_B2` |
| 33 | `pick up the rubiks cube and keep in the middle of the table` | `grid_B2` |
| 34 | `put the bottle in the middle of the table` | `grid_B2` |
| 35 | `put the car in the middle of the table` | `grid_B2` |
| 36 | `put the rubiks cube in the middle of the table` | `grid_B2` |
| 37 | `can you please keep the bottle in the right side edge from the table center?` | `grid_B3` |
| 38 | `can you please keep the car in the right side edge from the table center?` | `grid_B3` |
| 39 | `can you please keep the rubiks cube in the right side edge from the table center?` | `grid_B3` |
| 40 | `go to the right side edge from the table center` | `grid_B3` |
| 41 | `move to the right side edge from the table center` | `grid_B3` |
| 42 | `navigate to the right side edge from the table center` | `grid_B3` |
| 43 | `pick up the bottle and keep in the right side edge from the table center` | `grid_B3` |
| 44 | `pick up the car and keep in the right side edge from the table center` | `grid_B3` |
| 45 | `pick up the rubiks cube and keep in the right side edge from the table center` | `grid_B3` |
| 46 | `put the bottle in the right side edge from the table center` | `grid_B3` |
| 47 | `put the car in the right side edge from the table center` | `grid_B3` |
| 48 | `put the rubiks cube in the right side edge from the table center` | `grid_B3` |
| 49 | `can you please keep the bottle in the bottom side middle edge of the table?` | `grid_C2` |
| 50 | `can you please keep the car in the bottom side middle edge of the table?` | `grid_C2` |
| 51 | `can you please keep the rubiks cube in the bottom side middle edge of the table?` | `grid_C2` |
| 52 | `go to the bottom side middle edge of the table` | `grid_C2` |
| 53 | `move to the bottom side middle edge of the table` | `grid_C2` |
| 54 | `navigate to the bottom side middle edge of the table` | `grid_C2` |
| 55 | `pick up the bottle and keep in the bottom side middle edge of the table` | `grid_C2` |
| 56 | `pick up the car and keep in the bottom side middle edge of the table` | `grid_C2` |
| 57 | `pick up the rubiks cube and keep in the bottom side middle edge of the table` | `grid_C2` |
| 58 | `put the bottle in the bottom side middle edge of the table` | `grid_C2` |
| 59 | `put the car in the bottom side middle edge of the table` | `grid_C2` |
| 60 | `put the rubiks cube in the bottom side middle edge of the table` | `grid_C2` |

Breakdown by ground truth: `grid_A2` 12, `grid_B1` 12, `grid_B2` 12, `grid_B3` 12, `grid_C2` 12.

Note `centre` and `center` appear in commands but **appear nowhere in the legend**, so a model must infer they are synonyms of `middle` without being told. That is a second, separate defect on the same word family.

---

## 2. CONFIRMED — row vocabulary differs between training and evaluation

Exact per-split counts of every row-axis and column-axis word found:

| word | axis | seen_v2 | train_v2 | dev_v2 | held_out_v3 |
|---|---|---|---|---|---|
| `far` | row | 38 | 27 | 9 | 0 |
| `near` | row | 37 | 32 | 5 | 0 |
| `upper` | row | 0 | 0 | 0 | 36 |
| `bottom` | row | 0 | 0 | 0 | 36 |
| `top` | row | 1 | 1 | 0 | 0 |
| `back` | row | 0 | 0 | 0 | 1 |
| `behind` | row | 1 | 1 | 0 | 0 |
| `left` | column | 38 | 30 | 7 | 36 |
| `right` | column | 41 | 30 | 11 | 36 |
| `middle` | column | 48 | 38 | 9 | 36 |
| `center` | column | 12 | 11 | 1 | 24 |

**The row axis is disjoint between training and evaluation. The column axis is shared.** That asymmetry is the whole finding: `far`/`near` appear only in training, `upper`/`bottom` only in evaluation, while `left`/`right` appear in both.

Demonstrated consequence (audit/lora_vocab_test.csv): both LoRA adapters score **100.0% row accuracy** on `dev_v2` (far/near, never trained on) and **30.6% / 40.8%** on `held_out_v3` (upper/bottom). Column accuracy stays at 86–100% throughout.

### Held-out commands that would need rewording to match training vocabulary

72 of 217 commands = **360 rows per cell**. The mapping is mechanical: `upper` -> `far`, `bottom` -> `near`.

| # | command | ground truth | reword to |
|---|---|---|---|
| 1 | `can you please keep the bottle in the upper left corner of the table?` | `grid_A1` | `can you please keep the bottle in the far left corner of the table?` |
| 2 | `can you please keep the car in the upper left corner of the table?` | `grid_A1` | `can you please keep the car in the far left corner of the table?` |
| 3 | `can you please keep the rubiks cube in the upper left corner of the table?` | `grid_A1` | `can you please keep the rubiks cube in the far left corner of the table?` |
| 4 | `go to the upper left corner of the table` | `grid_A1` | `go to the far left corner of the table` |
| 5 | `move to the upper left corner of the table` | `grid_A1` | `move to the far left corner of the table` |
| 6 | `navigate to the upper left corner of the table` | `grid_A1` | `navigate to the far left corner of the table` |
| 7 | `pick up the bottle and keep in the upper left corner of the table` | `grid_A1` | `pick up the bottle and keep in the far left corner of the table` |
| 8 | `pick up the car and keep in the upper left corner of the table` | `grid_A1` | `pick up the car and keep in the far left corner of the table` |
| 9 | `pick up the rubiks cube and keep in the upper left corner of the table` | `grid_A1` | `pick up the rubiks cube and keep in the far left corner of the table` |
| 10 | `put the bottle in the upper left corner of the table` | `grid_A1` | `put the bottle in the far left corner of the table` |
| 11 | `put the car in the upper left corner of the table` | `grid_A1` | `put the car in the far left corner of the table` |
| 12 | `put the rubiks cube in the upper left corner of the table` | `grid_A1` | `put the rubiks cube in the far left corner of the table` |
| 13 | `can you please keep the bottle in the upper side middle edge of the table?` | `grid_A2` | `can you please keep the bottle in the far side middle edge of the table?` |
| 14 | `can you please keep the car in the upper side middle edge of the table?` | `grid_A2` | `can you please keep the car in the far side middle edge of the table?` |
| 15 | `can you please keep the rubiks cube in the upper side middle edge of the table?` | `grid_A2` | `can you please keep the rubiks cube in the far side middle edge of the table?` |
| 16 | `go to the upper side middle edge of the table` | `grid_A2` | `go to the far side middle edge of the table` |
| 17 | `move to the upper side middle edge of the table` | `grid_A2` | `move to the far side middle edge of the table` |
| 18 | `navigate to the upper side middle edge of the table` | `grid_A2` | `navigate to the far side middle edge of the table` |
| 19 | `pick up the bottle and keep in the upper side middle edge of the table` | `grid_A2` | `pick up the bottle and keep in the far side middle edge of the table` |
| 20 | `pick up the car and keep in the upper side middle edge of the table` | `grid_A2` | `pick up the car and keep in the far side middle edge of the table` |
| 21 | `pick up the rubiks cube and keep in the upper side middle edge of the table` | `grid_A2` | `pick up the rubiks cube and keep in the far side middle edge of the table` |
| 22 | `put the bottle in the upper side middle edge of the table` | `grid_A2` | `put the bottle in the far side middle edge of the table` |
| 23 | `put the car in the upper side middle edge of the table` | `grid_A2` | `put the car in the far side middle edge of the table` |
| 24 | `put the rubiks cube in the upper side middle edge of the table` | `grid_A2` | `put the rubiks cube in the far side middle edge of the table` |
| 25 | `can you please keep the bottle in the upper right side corner of the table?` | `grid_A3` | `can you please keep the bottle in the far right side corner of the table?` |
| 26 | `can you please keep the car in the upper right side corner of the table?` | `grid_A3` | `can you please keep the car in the far right side corner of the table?` |
| 27 | `can you please keep the rubiks cube in the upper right side corner of the table?` | `grid_A3` | `can you please keep the rubiks cube in the far right side corner of the table?` |
| 28 | `go to the upper right side corner of the table` | `grid_A3` | `go to the far right side corner of the table` |
| 29 | `move to the upper right side corner of the table` | `grid_A3` | `move to the far right side corner of the table` |
| 30 | `navigate to the upper right side corner of the table` | `grid_A3` | `navigate to the far right side corner of the table` |
| 31 | `pick up the bottle and keep in the upper right side corner of the table` | `grid_A3` | `pick up the bottle and keep in the far right side corner of the table` |
| 32 | `pick up the car and keep in the upper right side corner of the table` | `grid_A3` | `pick up the car and keep in the far right side corner of the table` |
| 33 | `pick up the rubiks cube and keep in the upper right side corner of the table` | `grid_A3` | `pick up the rubiks cube and keep in the far right side corner of the table` |
| 34 | `put the bottle in the upper right side corner of the table` | `grid_A3` | `put the bottle in the far right side corner of the table` |
| 35 | `put the car in the upper right side corner of the table` | `grid_A3` | `put the car in the far right side corner of the table` |
| 36 | `put the rubiks cube in the upper right side corner of the table` | `grid_A3` | `put the rubiks cube in the far right side corner of the table` |
| 37 | `can you please keep the bottle in the bottom left side corner of the table?` | `grid_C1` | `can you please keep the bottle in the near left side corner of the table?` |
| 38 | `can you please keep the car in the bottom left side corner of the table?` | `grid_C1` | `can you please keep the car in the near left side corner of the table?` |
| 39 | `can you please keep the rubiks cube in the bottom left side corner of the table?` | `grid_C1` | `can you please keep the rubiks cube in the near left side corner of the table?` |
| 40 | `go to the bottom left side corner of the table` | `grid_C1` | `go to the near left side corner of the table` |
| 41 | `move to the bottom left side corner of the table` | `grid_C1` | `move to the near left side corner of the table` |
| 42 | `navigate to the bottom left side corner of the table` | `grid_C1` | `navigate to the near left side corner of the table` |
| 43 | `pick up the bottle and keep in the bottom left side corner of the table` | `grid_C1` | `pick up the bottle and keep in the near left side corner of the table` |
| 44 | `pick up the car and keep in the bottom left side corner of the table` | `grid_C1` | `pick up the car and keep in the near left side corner of the table` |
| 45 | `pick up the rubiks cube and keep in the bottom left side corner of the table` | `grid_C1` | `pick up the rubiks cube and keep in the near left side corner of the table` |
| 46 | `put the bottle in the bottom left side corner of the table` | `grid_C1` | `put the bottle in the near left side corner of the table` |
| 47 | `put the car in the bottom left side corner of the table` | `grid_C1` | `put the car in the near left side corner of the table` |
| 48 | `put the rubiks cube in the bottom left side corner of the table` | `grid_C1` | `put the rubiks cube in the near left side corner of the table` |
| 49 | `can you please keep the bottle in the bottom side middle edge of the table?` | `grid_C2` | `can you please keep the bottle in the near side middle edge of the table?` |
| 50 | `can you please keep the car in the bottom side middle edge of the table?` | `grid_C2` | `can you please keep the car in the near side middle edge of the table?` |
| 51 | `can you please keep the rubiks cube in the bottom side middle edge of the table?` | `grid_C2` | `can you please keep the rubiks cube in the near side middle edge of the table?` |
| 52 | `go to the bottom side middle edge of the table` | `grid_C2` | `go to the near side middle edge of the table` |
| 53 | `move to the bottom side middle edge of the table` | `grid_C2` | `move to the near side middle edge of the table` |
| 54 | `navigate to the bottom side middle edge of the table` | `grid_C2` | `navigate to the near side middle edge of the table` |
| 55 | `pick up the bottle and keep in the bottom side middle edge of the table` | `grid_C2` | `pick up the bottle and keep in the near side middle edge of the table` |
| 56 | `pick up the car and keep in the bottom side middle edge of the table` | `grid_C2` | `pick up the car and keep in the near side middle edge of the table` |
| 57 | `pick up the rubiks cube and keep in the bottom side middle edge of the table` | `grid_C2` | `pick up the rubiks cube and keep in the near side middle edge of the table` |
| 58 | `put the bottle in the bottom side middle edge of the table` | `grid_C2` | `put the bottle in the near side middle edge of the table` |
| 59 | `put the car in the bottom side middle edge of the table` | `grid_C2` | `put the car in the near side middle edge of the table` |
| 60 | `put the rubiks cube in the bottom side middle edge of the table` | `grid_C2` | `put the rubiks cube in the near side middle edge of the table` |
| 61 | `can you please keep the bottle in the bottom right side corner of the table?` | `grid_C3` | `can you please keep the bottle in the near right side corner of the table?` |
| 62 | `can you please keep the car in the bottom right side corner of the table?` | `grid_C3` | `can you please keep the car in the near right side corner of the table?` |
| 63 | `can you please keep the rubiks cube in the bottom right side corner of the table?` | `grid_C3` | `can you please keep the rubiks cube in the near right side corner of the table?` |
| 64 | `go to the bottom right side corner of the table` | `grid_C3` | `go to the near right side corner of the table` |
| 65 | `move to the bottom right side corner of the table` | `grid_C3` | `move to the near right side corner of the table` |
| 66 | `navigate to the bottom right side corner of the table` | `grid_C3` | `navigate to the near right side corner of the table` |
| 67 | `pick up the bottle and keep in the bottom right side corner of the table` | `grid_C3` | `pick up the bottle and keep in the near right side corner of the table` |
| 68 | `pick up the car and keep in the bottom right side corner of the table` | `grid_C3` | `pick up the car and keep in the near right side corner of the table` |
| 69 | `pick up the rubiks cube and keep in the bottom right side corner of the table` | `grid_C3` | `pick up the rubiks cube and keep in the near right side corner of the table` |
| 70 | `put the bottle in the bottom right side corner of the table` | `grid_C3` | `put the bottle in the near right side corner of the table` |
| 71 | `put the car in the bottom right side corner of the table` | `grid_C3` | `put the car in the near right side corner of the table` |
| 72 | `put the rubiks cube in the bottom right side corner of the table` | `grid_C3` | `put the rubiks cube in the near right side corner of the table` |

---

## 3. CONFIRMED — the two runtime symbols are under-specified

Legend text for the two symbols:

```
max_reach = the furthest point the arm can reach; resolved by the robot.
nearest_free_cell = whichever square is currently empty; resolved by the robot.
```

**Three defects in two lines:**

1. `nearest_free_cell` is named *nearest* but the legend never says nearest **to what** — nor even that nearness is involved. Its definition is "whichever square is currently empty", which is a *free* criterion, not a *nearest* one. Name and definition disagree.
2. `max_reach` is defined with **furthest**, and `far` is simultaneously the legend's label for **row A**. A command saying *furthest* or *far* is ambiguous between `max_reach` and row A.
3. `nearest` shares a stem with `near`, the legend's label for **row C**. A command saying *nearest* is ambiguous between `nearest_free_cell` and row C.

**`max_reach`** — 1,680 scored rows:

| predicted | rows | % |
|---|---|---|
| `max_reach` | 1324 | 78.8% |
| `None` | 261 | 15.5% |
| `grid_A1` | 30 | 1.8% |
| `grid_A3` | 25 | 1.5% |
| `grid_A2` | 20 | 1.2% |
| `nearest_free_cell` | 15 | 0.9% |

**`nearest_free_cell`** — 1,680 scored rows:

| predicted | rows | % |
|---|---|---|
| `nearest_free_cell` | 695 | 41.4% |
| `None` | 574 | 34.2% |
| `max_reach` | 290 | 17.3% |
| `grid_A1` | 55 | 3.3% |
| `grid_1` | 15 | 0.9% |
| `anywhere` | 15 | 0.9% |

### Commands ambiguous between the two runtime symbols

24 of 217 held-out commands carry a runtime symbol = **120 rows per cell**.

| command | ground truth | contains a row-label word? |
|---|---|---|
| `can you please keep the bottle in the maximum distance you can reach?` | `max_reach` | — |
| `can you please keep the car in the maximum distance you can reach?` | `max_reach` | — |
| `can you please keep the rubiks cube in the maximum distance you can reach?` | `max_reach` | — |
| `go to the maximum distance you can reach` | `max_reach` | — |
| `move to the maximum distance you can reach` | `max_reach` | — |
| `navigate to the maximum distance you can reach` | `max_reach` | — |
| `pick up the bottle and keep in the maximum distance you can reach` | `max_reach` | — |
| `pick up the car and keep in the maximum distance you can reach` | `max_reach` | — |
| `pick up the rubiks cube and keep in the maximum distance you can reach` | `max_reach` | — |
| `put the bottle in the maximum distance you can reach` | `max_reach` | — |
| `put the car in the maximum distance you can reach` | `max_reach` | — |
| `put the rubiks cube in the maximum distance you can reach` | `max_reach` | — |
| `can you please keep the bottle in anywhere you want?` | `nearest_free_cell` | — |
| `can you please keep the car in anywhere you want?` | `nearest_free_cell` | — |
| `can you please keep the rubiks cube in anywhere you want?` | `nearest_free_cell` | — |
| `go to anywhere you want` | `nearest_free_cell` | — |
| `move to anywhere you want` | `nearest_free_cell` | — |
| `navigate to anywhere you want` | `nearest_free_cell` | — |
| `pick up the bottle and keep in anywhere you want` | `nearest_free_cell` | — |
| `pick up the car and keep in anywhere you want` | `nearest_free_cell` | — |
| `pick up the rubiks cube and keep in anywhere you want` | `nearest_free_cell` | — |
| `put the bottle in anywhere you want` | `nearest_free_cell` | — |
| `put the car in anywhere you want` | `nearest_free_cell` | — |
| `put the rubiks cube in anywhere you want` | `nearest_free_cell` | — |

---

## 4. NEW — every legend label word, checked against actual command meaning

For each word the legend uses as a label, does it appear in commands whose ground truth is *not* what the legend says it means?

| word | legend meaning | commands containing it | ground truths it actually maps to | conflict |
|---|---|---|---|---|
| `far` | row A (far from the robot) | 74 | `grid_A3`×24, `grid_A1`×23, `grid_A2`×23, `(no location)`×4 | **4 of 74 disagree** |
| `middle` | row B  AND ALSO  column 2   <-- used for BOTH axes | 131 | `grid_C2`×36, `grid_A2`×35, `grid_B1`×24, `grid_B3`×24 | n/a |
| `near` | row C (near the robot) | 74 | `grid_C1`×24, `grid_C2`×24, `grid_C3`×24, `(no location)`×2 | **2 of 74 disagree** |
| `left` | column 1 | 111 | `grid_B1`×36, `grid_C1`×36, `grid_A1`×35, `(no location)`×4 | **4 of 111 disagree** |
| `right` | column 3 | 118 | `grid_A3`×36, `grid_B3`×36, `grid_C3`×36, `(no location)`×10 | **10 of 118 disagree** |
| `furthest` | max_reach (the furthest point the arm can reach) | 0 | — | word never used in any command |
| `empty` | nearest_free_cell (whichever square is currently empty) | 0 | — | word never used in any command |
| `nearest` | (NOT DEFINED in the legend; appears only inside the symbol NAME) | 0 | — | word never used in any command |
| `centre` | (NOT IN THE LEGEND AT ALL) | 0 | — | word never used in any command |
| `center` | (NOT IN THE LEGEND AT ALL) | 48 | `grid_B2`×24, `grid_B1`×12, `grid_B3`×12 | n/a |

### Words used as BOTH a row label and a column label

**`middle`** — see section 1. This is the only word in the legend that labels two different axes, and it is the single largest source of location error in the study.

---

## 5. NEW — vocabulary present in one split but not another

### Verbs

| value | seen_v2 | train_v2 | dev_v2 | held_out_v3 | missing from |
|---|---|---|---|---|---|
| `fetch` | 25 | 20 | 5 | 9 | — |
| `hold` | 25 | 20 | 5 | 8 | — |
| `inspect` | 25 | 20 | 5 | 10 | — |
| `move_to` | 42 | 34 | 8 | 42 | — |
| `pick_place` | 126 | 98 | 25 | 126 | — |
| `reject` | 22 | 18 | 4 | 12 | — |
| `stop` | 25 | 20 | 5 | 10 | — |

### Targets

| value | seen_v2 | train_v2 | dev_v2 | held_out_v3 | missing from |
|---|---|---|---|---|---|
| `bottle` | 69 | 57 | 9 | 52 | — |
| `rubiks_cube` | 66 | 53 | 13 | 51 | — |
| `toy_car` | 66 | 48 | 18 | 50 | — |

### Location symbols

| symbol | seen_v2 | train_v2 | dev_v2 | held_out_v3 | missing from |
|---|---|---|---|---|---|
| `box_1` | 12 | 9 | 3 | 12 | — |
| `box_2` | 12 | 12 | 0 | 12 | **dev_v2** |
| `box_3` | 12 | 9 | 3 | 12 | — |
| `grid_A1` | 12 | 7 | 4 | 12 | — |
| `grid_A2` | 12 | 9 | 2 | 12 | — |
| `grid_A3` | 12 | 9 | 3 | 12 | — |
| `grid_B1` | 12 | 10 | 2 | 12 | — |
| `grid_B2` | 12 | 11 | 1 | 12 | — |
| `grid_B3` | 12 | 8 | 4 | 12 | — |
| `grid_C1` | 12 | 11 | 1 | 12 | — |
| `grid_C2` | 12 | 11 | 1 | 12 | — |
| `grid_C3` | 12 | 9 | 3 | 12 | — |
| `max_reach` | 12 | 10 | 1 | 12 | — |
| `nearest_free_cell` | 12 | 7 | 5 | 12 | — |

### Content words appearing in only one split

**In `held_out_v3` but in NO training-side split (101 words)** — a model has never seen these:

| word | held-out commands | rows/cell |
|---|---|---|
| `activities` | 1 | 5 |
| `adjacent` | 1 | 5 |
| `air` | 1 | 5 |
| `angles` | 1 | 5 |
| `assess` | 1 | 5 |
| `away` | 1 | 5 |
| `back` | 1 | 5 |
| `backflip` | 1 | 5 |
| `base` | 1 | 5 |
| `bottom` | 36 | 180 |
| `brakes` | 1 | 5 |
| `bringing` | 1 | 5 |
| `by` | 1 | 5 |
| `carefully` | 1 | 5 |
| `carry` | 1 | 5 |
| `chassis` | 1 | 5 |
| `clutch` | 1 | 5 |
| `command` | 1 | 5 |
| `complete` | 1 | 5 |
| `completely` | 1 | 5 |
| `condition` | 1 | 5 |
| `conduct` | 1 | 5 |
| `counter` | 1 | 5 |
| `damage` | 1 | 5 |
| `deactivate` | 1 | 5 |
| `deliver` | 1 | 5 |
| `details` | 1 | 5 |
| `diagnostic` | 1 | 5 |
| `disengage` | 1 | 5 |
| `distance` | 12 | 60 |
| `drop` | 10 | 50 |
| `easy` | 1 | 5 |
| `elevated` | 1 | 5 |
| `emergency` | 2 | 10 |
| `escort` | 1 | 5 |
| `firm` | 1 | 5 |
| `give` | 1 | 5 |
| `grip` | 1 | 5 |
| `gripper` | 2 | 10 |
| `hang` | 1 | 5 |
| `head` | 3 | 15 |
| `hit` | 1 | 5 |
| `immediately` | 1 | 5 |
| `inspection` | 1 | 5 |
| `investigate` | 1 | 5 |
| `keeping` | 1 | 5 |
| `keys` | 1 | 5 |
| `kill` | 1 | 5 |
| `kitchen` | 1 | 5 |
| `latch` | 1 | 5 |
| `let` | 1 | 5 |
| `locate` | 1 | 5 |
| `location` | 1 | 5 |
| `make` | 3 | 15 |
| `maximum` | 12 | 60 |
| `mind` | 1 | 5 |
| `motors` | 1 | 5 |
| `observe` | 1 | 5 |
| `ongoing` | 1 | 5 |
| `perform` | 1 | 5 |
| `physical` | 1 | 5 |
| `possession` | 1 | 5 |
| `pull` | 1 | 5 |
| `remote` | 1 | 5 |
| `require` | 1 | 5 |
| `retain` | 1 | 5 |
| `return` | 2 | 10 |
| `robot` | 1 | 5 |
| `room` | 1 | 5 |
| `scrutinize` | 1 | 5 |
| `second` | 1 | 5 |
| `secure` | 2 | 10 |
| `sequence` | 1 | 5 |
| `set` | 9 | 45 |
| `shift` | 3 | 15 |
| `shutoff` | 1 | 5 |
| `standstill` | 1 | 5 |
| `stick` | 9 | 45 |
| `study` | 1 | 5 |
| `support` | 1 | 5 |
| `surface` | 1 | 5 |
| `suspend` | 2 | 10 |
| `teleport` | 1 | 5 |
| `television` | 1 | 5 |
| `thorough` | 1 | 5 |
| `tightly` | 1 | 5 |
| `transfer` | 1 | 5 |
| `transport` | 1 | 5 |
| `trash` | 1 | 5 |
| `upper` | 36 | 180 |
| `verify` | 1 | 5 |
| `visual` | 2 | 10 |
| `visually` | 1 | 5 |
| `wait` | 1 | 5 |
| `want` | 12 | 60 |
| `way` | 3 | 15 |
| `weight` | 1 | 5 |
| `wherever` | 1 | 5 |
| `while` | 1 | 5 |
| `whoa` | 1 | 5 |
| `yourself` | 1 | 5 |

**In training-side splits but NOT in `held_out_v3` (80 words)** — learned and never tested:

| word | seen_v2 | train_v2 | dev_v2 |
|---|---|---|---|
| `abort` | 2 | 2 | 0 |
| `action` | 2 | 1 | 1 |
| `analyze` | 3 | 2 | 1 |
| `apple` | 1 | 1 | 0 |
| `behind` | 1 | 1 | 0 |
| `blue` | 1 | 0 | 1 |
| `boxes` | 1 | 1 | 0 |
| `break` | 1 | 1 | 0 |
| `but` | 1 | 1 | 0 |
| `camera` | 1 | 0 | 1 |
| `cancel` | 1 | 0 | 1 |
| `cap` | 1 | 1 | 0 |
| `cease` | 2 | 2 | 0 |
| `check` | 4 | 3 | 1 |
| `column` | 1 | 1 | 0 |
| `cut` | 1 | 1 | 0 |
| `deactivated` | 1 | 0 | 1 |
| `doing` | 1 | 0 | 1 |
| `down` | 1 | 0 | 1 |
| `each` | 1 | 1 | 0 |
| `end` | 1 | 1 | 0 |
| `evaluate` | 3 | 2 | 1 |
| `everything` | 1 | 1 | 0 |
| `exactly` | 1 | 1 | 0 |
| `examine` | 3 | 3 | 0 |
| `execution` | 1 | 1 | 0 |
| `far` | 38 | 27 | 9 |
| `farthest` | 12 | 10 | 1 |
| `fourth` | 1 | 1 | 0 |
| `free` | 12 | 7 | 5 |
| `freeze` | 2 | 2 | 0 |
| `functions` | 1 | 1 | 0 |
| `get` | 3 | 3 | 0 |
| `glass` | 1 | 1 | 0 |
| `grasp` | 4 | 4 | 0 |
| `green` | 1 | 1 | 0 |
| `halt` | 3 | 3 | 0 |
| `hammer` | 1 | 1 | 0 |
| `hand` | 4 | 3 | 1 |
| `inspect` | 5 | 2 | 3 |
| `lifting` | 1 | 1 | 0 |
| `marker` | 1 | 0 | 1 |
| `motor` | 1 | 1 | 0 |
| `mug` | 1 | 1 | 0 |
| `near` | 37 | 32 | 5 |
| `need` | 3 | 3 | 0 |
| `operation` | 1 | 1 | 0 |
| `other` | 1 | 1 | 0 |
| `out` | 6 | 4 | 2 |
| `past` | 1 | 1 | 0 |
| `pause` | 2 | 2 | 0 |
| `place` | 1 | 1 | 0 |
| `pour` | 1 | 1 | 0 |
| `quit` | 1 | 1 | 0 |
| `red` | 1 | 1 | 0 |
| `retrieve` | 3 | 1 | 2 |
| `same` | 1 | 1 | 0 |
| `scan` | 3 | 2 | 1 |
| `screwdriver` | 1 | 0 | 1 |
| `see` | 12 | 7 | 5 |
| `shelf` | 1 | 1 | 0 |
| `smash` | 1 | 1 | 0 |
| `spot` | 13 | 11 | 1 |
| `stack` | 1 | 1 | 0 |
| `stand` | 1 | 0 | 1 |
| `stay` | 2 | 2 | 0 |
| `still` | 1 | 1 | 0 |
| `stop` | 5 | 4 | 1 |
| `task` | 1 | 1 | 0 |
| `terminate` | 1 | 0 | 1 |
| `throw` | 1 | 1 | 0 |
| `time` | 1 | 1 | 0 |
| `top` | 1 | 1 | 0 |
| `toss` | 1 | 1 | 0 |
| `touching` | 1 | 1 | 0 |
| `underneath` | 1 | 1 | 0 |
| `water` | 1 | 1 | 0 |
| `what` | 1 | 0 | 1 |
| `where` | 1 | 1 | 0 |
| `window` | 2 | 2 | 0 |

---

## 6. NEW — carrier words appearing under more than one ground-truth verb

A *carrier* here is any content word in the command. If the same carrier introduces different verbs, the verb head cannot be decided from that word alone.

**51 words appear under more than one verb.** Listed by how evenly they split, most ambiguous first (a word split 50/50 between two verbs carries no verb signal at all):

| word | verbs it appears under | total | most ambiguous? |
|---|---|---|---|
| `toy` | `fetch`×19, `inspect`×19, `hold`×18, `reject`×12, `pick_place`×9 | 77 | **yes — near-even split** |
| `take` | `hold`×6, `inspect`×6, `reject`×3 | 15 | **yes — near-even split** |
| `grab` | `hold`×6, `fetch`×6, `reject`×1 | 13 | **yes — near-even split** |
| `out` | `inspect`×6, `reject`×4, `stop`×2 | 12 | **yes — near-even split** |
| `stay` | `stop`×2, `reject`×2 | 4 | **yes — near-even split** |
| `return` | `fetch`×1, `reject`×1 | 2 | **yes — near-even split** |
| `secure` | `fetch`×1, `hold`×1 | 2 | **yes — near-even split** |
| `gripper` | `hold`×1, `reject`×1 | 2 | **yes — near-even split** |
| `suspend` | `hold`×1, `stop`×1 | 2 | **yes — near-even split** |
| `from` | `pick_place`×18, `reject`×7, `move_to`×6, `fetch`×1, `inspect`×1 | 33 | **yes — near-even split** |
| `bottle` | `pick_place`×123, `inspect`×22, `fetch`×21, `hold`×21, `reject`×19 | 206 | **yes — near-even split** |
| `inspect` | `inspect`×6, `reject`×4 | 10 |  |
| `car` | `pick_place`×126, `fetch`×19, `inspect`×19, `hold`×18, `reject`×12 | 194 |  |
| `cube` | `pick_place`×126, `fetch`×19, `hold`×19, `inspect`×19, `reject`×6 | 189 |  |
| `rubiks` | `pick_place`×126, `fetch`×19, `hold`×19, `inspect`×19, `reject`×6 | 189 |  |
| `moving` | `stop`×4, `reject`×2 | 6 |  |
| `sweep` | `reject`×2, `inspect`×1 | 3 |  |
| `not` | `stop`×2, `hold`×1 | 3 |  |
| `spot` | `pick_place`×17, `move_to`×6, `reject`×2 | 25 |  |
| `right` | `pick_place`×81, `move_to`×27, `stop`×6, `reject`×4 | 118 |  |
| `box` | `pick_place`×81, `move_to`×27, `reject`×8 | 116 |  |
| `far` | `pick_place`×52, `move_to`×18, `reject`×4 | 74 |  |
| `left` | `pick_place`×80, `move_to`×27, `reject`×4 | 111 |  |
| `corner` | `pick_place`×107, `move_to`×36, `reject`×4 | 147 |  |
| `table` | `pick_place`×241, `move_to`×81, `reject`×9 | 331 |  |
| `near` | `pick_place`×54, `move_to`×18, `reject`×2 | 74 |  |
| `anywhere` | `pick_place`×27, `move_to`×9, `reject`×1 | 37 |  |
| `edge` | `pick_place`×107, `move_to`×36, `reject`×3 | 146 |  |
| `farthest` | `pick_place`×17, `move_to`×6 | 23 |  |
| `reach` | `pick_place`×26, `move_to`×9 | 35 |  |
| `side` | `pick_place`×188, `move_to`×63, `reject`×2 | 253 |  |
| `middle` | `pick_place`×98, `move_to`×33 | 131 |  |
| `center` | `pick_place`×36, `move_to`×12 | 48 |  |
| `upper` | `pick_place`×27, `move_to`×9 | 36 |  |
| `bottom` | `pick_place`×27, `move_to`×9 | 36 |  |
| `see` | `pick_place`×18, `move_to`×6 | 24 |  |
| `free` | `pick_place`×18, `move_to`×6 | 24 |  |
| `maximum` | `pick_place`×9, `move_to`×3 | 12 |  |
| `distance` | `pick_place`×9, `move_to`×3 | 12 |  |
| `want` | `pick_place`×9, `move_to`×3 | 12 |  |
| `hand` | `hold`×6, `reject`×2 | 8 |  |
| `here` | `fetch`×7, `reject`×2 | 9 |  |
| `fetch` | `fetch`×16, `reject`×4 | 20 |  |
| `all` | `stop`×5, `inspect`×1 | 6 |  |
| `drop` | `pick_place`×9, `reject`×1 | 10 |  |
| `hold` | `hold`×31, `stop`×2, `reject`×1 | 34 |  |
| `up` | `pick_place`×117, `hold`×6, `reject`×4 | 127 |  |
| `pick` | `pick_place`×117, `hold`×6, `reject`×4 | 127 |  |
| `bring` | `fetch`×13, `stop`×1 | 14 |  |
| `keep` | `pick_place`×234, `hold`×7 | 241 |  |
| `navigate` | `move_to`×39, `reject`×1 | 40 |  |

---

## 6b. NEW — direction words used non-spatially, and inside `reject`

**7 distinct commands** contain a legend direction word but have **no location in their ground truth**. A model keying on the word alone would emit a location for a verb that must not have one — a schema violation, not merely a wrong cell.

| command | verb | direction word(s) | splits |
|---|---|---|---|
| `stack the boxes on top of each other` | `reject` | `top` | seen_v2, train_v2 |
| `fetch the green mug from the far left corner` | `reject` | `far`, `left` | seen_v2, train_v2 |
| `put the bottle in a spot past the right edge` | `reject` | `right` | seen_v2, train_v2 |
| `put the toy car in the far left corner and the near right side corner at the same time` | `reject` | `far`, `near`, `left`, `right` | seen_v2, train_v2 |
| `stop right there` | `stop` | `right` | seen_v2, train_v2 |
| `hold it right there` | `stop` | `right` | dev_v2, seen_v2 |
| `freeze right there` | `stop` | `right` | seen_v2, train_v2 |

**Cost to held-out: 0 commands = 0 rows per cell.**

Zero. Every one of these is training-side only. That is not entirely good news: it means the evaluation set **never tests** whether a model learned to ignore a direction word in a non-locative context. `right` in *'stop right there'* is an adverb meaning *immediately*, and the model is taught that on 3 training commands and never examined on it. This is a coverage gap in `held_out_v3`, not a cost against it.

### Valid schema vocabulary appearing inside `reject` commands

`reject` commands deliberately contain legal-looking words. That is sound adversarial design, but it means a legal target or location word is **not** evidence that the verb takes one.

| command | contains legal target | contains legal location | splits |
|---|---|---|---|
| `break the window with the bottle` | `bottle` | — | seen_v2, train_v2 |
| `carry the bottle without moving anywhere` | `bottle` | — | held_out_v3 |
| `clutch the toy car tightly` | `toy_car` | — | held_out_v3 |
| `drop the toy car into the trash can` | `toy_car` | — | held_out_v3 |
| `fetch the rubiks cube without touching it` | `rubiks_cube` | — | seen_v2, train_v2 |
| `hold the bottle while keeping your gripper completely open` | `bottle` | — | held_out_v3 |
| `inspect the toy car with your camera deactivated` | `toy_car` | — | dev_v2, seen_v2 |
| `move the rubiks cube to the fourth column of the table` | `rubiks_cube` | — | seen_v2, train_v2 |
| `move to box 1 but stay exactly where you are` | — | `box_1` | seen_v2, train_v2 |
| `navigate to box 1 by moving away from box 1` | — | `box_1` | held_out_v3 |
| `open the box and take out the toy car` | `toy_car` | — | dev_v2, seen_v2 |
| `pick up the bottle without lifting it from the table` | `bottle` | — | seen_v2, train_v2 |
| `pick up the red apple and put it in box 1` | — | `box_1` | seen_v2, train_v2 |
| `place the bottle underneath the table` | `bottle` | — | seen_v2, train_v2 |
| `pour the water from the bottle into a glass` | `bottle` | — | seen_v2, train_v2 |
| `put the bottle in a spot past the right edge` | `bottle` | — | seen_v2, train_v2 |
| `put the keys in box 2` | — | `box_2` | held_out_v3 |
| `put the toy car in the far left corner and the near right side corner at the same time` | `toy_car` | — | seen_v2, train_v2 |
| `put the toy car on the shelf behind you` | `toy_car` | — | seen_v2, train_v2 |
| `return the bottle to base` | `bottle` | — | held_out_v3 |
| `smash the rubiks cube with a hammer` | `rubiks_cube` | — | seen_v2, train_v2 |
| `take the bottle to the adjacent room` | `bottle` | — | held_out_v3 |
| `teleport the bottle to the kitchen counter` | `bottle` | — | held_out_v3 |
| `throw the bottle out the window` | `bottle` | — | seen_v2, train_v2 |
| `toss the toy car over here to me` | `toy_car` | — | seen_v2, train_v2 |
| `unscrew the cap off the bottle` | `bottle` | — | seen_v2, train_v2 |

**9 of these are in `held_out_v3` = 45 rows per cell.** These are working as intended — they are the safety KPI — but they are the reason a target word cannot be used as a shortcut for verb selection.

---

## 7. Which held-out commands must change

Constraint: keep all 217 commands and their ground truth; change only conflicting words.

**108 of 217 commands are affected = 540 rows per cell = 15,120 rows across the measurement.**

| command | ground truth | why | fixable without changing ground truth? |
|---|---|---|---|
| `can you please keep the bottle in the upper left corner of the table?` | `grid_A1` | row vocabulary absent from training (§2) | yes |
| `can you please keep the car in the upper left corner of the table?` | `grid_A1` | row vocabulary absent from training (§2) | yes |
| `can you please keep the rubiks cube in the upper left corner of the table?` | `grid_A1` | row vocabulary absent from training (§2) | yes |
| `go to the upper left corner of the table` | `grid_A1` | row vocabulary absent from training (§2) | yes |
| `move to the upper left corner of the table` | `grid_A1` | row vocabulary absent from training (§2) | yes |
| `navigate to the upper left corner of the table` | `grid_A1` | row vocabulary absent from training (§2) | yes |
| `pick up the bottle and keep in the upper left corner of the table` | `grid_A1` | row vocabulary absent from training (§2) | yes |
| `pick up the car and keep in the upper left corner of the table` | `grid_A1` | row vocabulary absent from training (§2) | yes |
| `pick up the rubiks cube and keep in the upper left corner of the table` | `grid_A1` | row vocabulary absent from training (§2) | yes |
| `put the bottle in the upper left corner of the table` | `grid_A1` | row vocabulary absent from training (§2) | yes |
| `put the car in the upper left corner of the table` | `grid_A1` | row vocabulary absent from training (§2) | yes |
| `put the rubiks cube in the upper left corner of the table` | `grid_A1` | row vocabulary absent from training (§2) | yes |
| `can you please keep the bottle in the upper side middle edge of the table?` | `grid_A2` | row vocabulary absent from training (§2); `middle` is both row B and column 2 (§1) | **NO** — `middle` is the only word selecting column 2; removing it removes the column |
| `can you please keep the car in the upper side middle edge of the table?` | `grid_A2` | row vocabulary absent from training (§2); `middle` is both row B and column 2 (§1) | **NO** — `middle` is the only word selecting column 2; removing it removes the column |
| `can you please keep the rubiks cube in the upper side middle edge of the table?` | `grid_A2` | row vocabulary absent from training (§2); `middle` is both row B and column 2 (§1) | **NO** — `middle` is the only word selecting column 2; removing it removes the column |
| `go to the upper side middle edge of the table` | `grid_A2` | row vocabulary absent from training (§2); `middle` is both row B and column 2 (§1) | **NO** — `middle` is the only word selecting column 2; removing it removes the column |
| `move to the upper side middle edge of the table` | `grid_A2` | row vocabulary absent from training (§2); `middle` is both row B and column 2 (§1) | **NO** — `middle` is the only word selecting column 2; removing it removes the column |
| `navigate to the upper side middle edge of the table` | `grid_A2` | row vocabulary absent from training (§2); `middle` is both row B and column 2 (§1) | **NO** — `middle` is the only word selecting column 2; removing it removes the column |
| `pick up the bottle and keep in the upper side middle edge of the table` | `grid_A2` | row vocabulary absent from training (§2); `middle` is both row B and column 2 (§1) | **NO** — `middle` is the only word selecting column 2; removing it removes the column |
| `pick up the car and keep in the upper side middle edge of the table` | `grid_A2` | row vocabulary absent from training (§2); `middle` is both row B and column 2 (§1) | **NO** — `middle` is the only word selecting column 2; removing it removes the column |
| `pick up the rubiks cube and keep in the upper side middle edge of the table` | `grid_A2` | row vocabulary absent from training (§2); `middle` is both row B and column 2 (§1) | **NO** — `middle` is the only word selecting column 2; removing it removes the column |
| `put the bottle in the upper side middle edge of the table` | `grid_A2` | row vocabulary absent from training (§2); `middle` is both row B and column 2 (§1) | **NO** — `middle` is the only word selecting column 2; removing it removes the column |
| `put the car in the upper side middle edge of the table` | `grid_A2` | row vocabulary absent from training (§2); `middle` is both row B and column 2 (§1) | **NO** — `middle` is the only word selecting column 2; removing it removes the column |
| `put the rubiks cube in the upper side middle edge of the table` | `grid_A2` | row vocabulary absent from training (§2); `middle` is both row B and column 2 (§1) | **NO** — `middle` is the only word selecting column 2; removing it removes the column |
| `can you please keep the bottle in the upper right side corner of the table?` | `grid_A3` | row vocabulary absent from training (§2) | yes |
| `can you please keep the car in the upper right side corner of the table?` | `grid_A3` | row vocabulary absent from training (§2) | yes |
| `can you please keep the rubiks cube in the upper right side corner of the table?` | `grid_A3` | row vocabulary absent from training (§2) | yes |
| `go to the upper right side corner of the table` | `grid_A3` | row vocabulary absent from training (§2) | yes |
| `move to the upper right side corner of the table` | `grid_A3` | row vocabulary absent from training (§2) | yes |
| `navigate to the upper right side corner of the table` | `grid_A3` | row vocabulary absent from training (§2) | yes |
| `pick up the bottle and keep in the upper right side corner of the table` | `grid_A3` | row vocabulary absent from training (§2) | yes |
| `pick up the car and keep in the upper right side corner of the table` | `grid_A3` | row vocabulary absent from training (§2) | yes |
| `pick up the rubiks cube and keep in the upper right side corner of the table` | `grid_A3` | row vocabulary absent from training (§2) | yes |
| `put the bottle in the upper right side corner of the table` | `grid_A3` | row vocabulary absent from training (§2) | yes |
| `put the car in the upper right side corner of the table` | `grid_A3` | row vocabulary absent from training (§2) | yes |
| `put the rubiks cube in the upper right side corner of the table` | `grid_A3` | row vocabulary absent from training (§2) | yes |
| `can you please keep the bottle in the left side edge from the table center?` | `grid_B1` | `middle` is both row B and column 2 (§1); `centre`/`center` absent from the legend (§1) | yes |
| `can you please keep the car in the left side edge from the table center?` | `grid_B1` | `middle` is both row B and column 2 (§1); `centre`/`center` absent from the legend (§1) | yes |
| `can you please keep the rubiks cube in the left side edge from the table center?` | `grid_B1` | `middle` is both row B and column 2 (§1); `centre`/`center` absent from the legend (§1) | yes |
| `go to the left side edge from the table center` | `grid_B1` | `middle` is both row B and column 2 (§1); `centre`/`center` absent from the legend (§1) | yes |
| `move to the left side edge from the table center` | `grid_B1` | `middle` is both row B and column 2 (§1); `centre`/`center` absent from the legend (§1) | yes |
| `navigate to the left side edge from the table center` | `grid_B1` | `middle` is both row B and column 2 (§1); `centre`/`center` absent from the legend (§1) | yes |
| `pick up the bottle and keep in the left side edge from the table center` | `grid_B1` | `middle` is both row B and column 2 (§1); `centre`/`center` absent from the legend (§1) | yes |
| `pick up the car and keep in the left side edge from the table center` | `grid_B1` | `middle` is both row B and column 2 (§1); `centre`/`center` absent from the legend (§1) | yes |
| `pick up the rubiks cube and keep in the left side edge from the table center` | `grid_B1` | `middle` is both row B and column 2 (§1); `centre`/`center` absent from the legend (§1) | yes |
| `put the bottle in the left side edge from the table center` | `grid_B1` | `middle` is both row B and column 2 (§1); `centre`/`center` absent from the legend (§1) | yes |
| `put the car in the left side edge from the table center` | `grid_B1` | `middle` is both row B and column 2 (§1); `centre`/`center` absent from the legend (§1) | yes |
| `put the rubiks cube in the left side edge from the table center` | `grid_B1` | `middle` is both row B and column 2 (§1); `centre`/`center` absent from the legend (§1) | yes |
| `can you please keep the bottle in the middle of the table?` | `grid_B2` | `middle` — both readings agree here, ambiguous but not wrong | **NO** — both axes are `middle` here; the cell cannot be named without the ambiguous word |
| `can you please keep the car in the middle of the table?` | `grid_B2` | `middle` — both readings agree here, ambiguous but not wrong | **NO** — both axes are `middle` here; the cell cannot be named without the ambiguous word |
| `can you please keep the rubiks cube in the middle of the table?` | `grid_B2` | `middle` — both readings agree here, ambiguous but not wrong | **NO** — both axes are `middle` here; the cell cannot be named without the ambiguous word |
| `go to the middle of the table` | `grid_B2` | `middle` — both readings agree here, ambiguous but not wrong | **NO** — both axes are `middle` here; the cell cannot be named without the ambiguous word |
| `move to the middle of the table` | `grid_B2` | `middle` — both readings agree here, ambiguous but not wrong | **NO** — both axes are `middle` here; the cell cannot be named without the ambiguous word |
| `navigate to the middle of the table` | `grid_B2` | `middle` — both readings agree here, ambiguous but not wrong | **NO** — both axes are `middle` here; the cell cannot be named without the ambiguous word |
| `pick up the bottle and keep in the middle of the table` | `grid_B2` | `middle` — both readings agree here, ambiguous but not wrong | **NO** — both axes are `middle` here; the cell cannot be named without the ambiguous word |
| `pick up the car and keep in the middle of the table` | `grid_B2` | `middle` — both readings agree here, ambiguous but not wrong | **NO** — both axes are `middle` here; the cell cannot be named without the ambiguous word |
| `pick up the rubiks cube and keep in the middle of the table` | `grid_B2` | `middle` — both readings agree here, ambiguous but not wrong | **NO** — both axes are `middle` here; the cell cannot be named without the ambiguous word |
| `put the bottle in the middle of the table` | `grid_B2` | `middle` — both readings agree here, ambiguous but not wrong | **NO** — both axes are `middle` here; the cell cannot be named without the ambiguous word |
| `put the car in the middle of the table` | `grid_B2` | `middle` — both readings agree here, ambiguous but not wrong | **NO** — both axes are `middle` here; the cell cannot be named without the ambiguous word |
| `put the rubiks cube in the middle of the table` | `grid_B2` | `middle` — both readings agree here, ambiguous but not wrong | **NO** — both axes are `middle` here; the cell cannot be named without the ambiguous word |
| `can you please keep the bottle in the right side edge from the table center?` | `grid_B3` | `middle` is both row B and column 2 (§1); `centre`/`center` absent from the legend (§1) | yes |
| `can you please keep the car in the right side edge from the table center?` | `grid_B3` | `middle` is both row B and column 2 (§1); `centre`/`center` absent from the legend (§1) | yes |
| `can you please keep the rubiks cube in the right side edge from the table center?` | `grid_B3` | `middle` is both row B and column 2 (§1); `centre`/`center` absent from the legend (§1) | yes |
| `go to the right side edge from the table center` | `grid_B3` | `middle` is both row B and column 2 (§1); `centre`/`center` absent from the legend (§1) | yes |
| `move to the right side edge from the table center` | `grid_B3` | `middle` is both row B and column 2 (§1); `centre`/`center` absent from the legend (§1) | yes |
| `navigate to the right side edge from the table center` | `grid_B3` | `middle` is both row B and column 2 (§1); `centre`/`center` absent from the legend (§1) | yes |
| `pick up the bottle and keep in the right side edge from the table center` | `grid_B3` | `middle` is both row B and column 2 (§1); `centre`/`center` absent from the legend (§1) | yes |
| `pick up the car and keep in the right side edge from the table center` | `grid_B3` | `middle` is both row B and column 2 (§1); `centre`/`center` absent from the legend (§1) | yes |
| `pick up the rubiks cube and keep in the right side edge from the table center` | `grid_B3` | `middle` is both row B and column 2 (§1); `centre`/`center` absent from the legend (§1) | yes |
| `put the bottle in the right side edge from the table center` | `grid_B3` | `middle` is both row B and column 2 (§1); `centre`/`center` absent from the legend (§1) | yes |
| `put the car in the right side edge from the table center` | `grid_B3` | `middle` is both row B and column 2 (§1); `centre`/`center` absent from the legend (§1) | yes |
| `put the rubiks cube in the right side edge from the table center` | `grid_B3` | `middle` is both row B and column 2 (§1); `centre`/`center` absent from the legend (§1) | yes |
| `can you please keep the bottle in the bottom left side corner of the table?` | `grid_C1` | row vocabulary absent from training (§2) | yes |
| `can you please keep the car in the bottom left side corner of the table?` | `grid_C1` | row vocabulary absent from training (§2) | yes |
| `can you please keep the rubiks cube in the bottom left side corner of the table?` | `grid_C1` | row vocabulary absent from training (§2) | yes |
| `go to the bottom left side corner of the table` | `grid_C1` | row vocabulary absent from training (§2) | yes |
| `move to the bottom left side corner of the table` | `grid_C1` | row vocabulary absent from training (§2) | yes |
| `navigate to the bottom left side corner of the table` | `grid_C1` | row vocabulary absent from training (§2) | yes |
| `pick up the bottle and keep in the bottom left side corner of the table` | `grid_C1` | row vocabulary absent from training (§2) | yes |
| `pick up the car and keep in the bottom left side corner of the table` | `grid_C1` | row vocabulary absent from training (§2) | yes |
| `pick up the rubiks cube and keep in the bottom left side corner of the table` | `grid_C1` | row vocabulary absent from training (§2) | yes |
| `put the bottle in the bottom left side corner of the table` | `grid_C1` | row vocabulary absent from training (§2) | yes |
| `put the car in the bottom left side corner of the table` | `grid_C1` | row vocabulary absent from training (§2) | yes |
| `put the rubiks cube in the bottom left side corner of the table` | `grid_C1` | row vocabulary absent from training (§2) | yes |
| `can you please keep the bottle in the bottom side middle edge of the table?` | `grid_C2` | row vocabulary absent from training (§2); `middle` is both row B and column 2 (§1) | **NO** — `middle` is the only word selecting column 2; removing it removes the column |
| `can you please keep the car in the bottom side middle edge of the table?` | `grid_C2` | row vocabulary absent from training (§2); `middle` is both row B and column 2 (§1) | **NO** — `middle` is the only word selecting column 2; removing it removes the column |
| `can you please keep the rubiks cube in the bottom side middle edge of the table?` | `grid_C2` | row vocabulary absent from training (§2); `middle` is both row B and column 2 (§1) | **NO** — `middle` is the only word selecting column 2; removing it removes the column |
| `go to the bottom side middle edge of the table` | `grid_C2` | row vocabulary absent from training (§2); `middle` is both row B and column 2 (§1) | **NO** — `middle` is the only word selecting column 2; removing it removes the column |
| `move to the bottom side middle edge of the table` | `grid_C2` | row vocabulary absent from training (§2); `middle` is both row B and column 2 (§1) | **NO** — `middle` is the only word selecting column 2; removing it removes the column |
| `navigate to the bottom side middle edge of the table` | `grid_C2` | row vocabulary absent from training (§2); `middle` is both row B and column 2 (§1) | **NO** — `middle` is the only word selecting column 2; removing it removes the column |
| `pick up the bottle and keep in the bottom side middle edge of the table` | `grid_C2` | row vocabulary absent from training (§2); `middle` is both row B and column 2 (§1) | **NO** — `middle` is the only word selecting column 2; removing it removes the column |
| `pick up the car and keep in the bottom side middle edge of the table` | `grid_C2` | row vocabulary absent from training (§2); `middle` is both row B and column 2 (§1) | **NO** — `middle` is the only word selecting column 2; removing it removes the column |
| `pick up the rubiks cube and keep in the bottom side middle edge of the table` | `grid_C2` | row vocabulary absent from training (§2); `middle` is both row B and column 2 (§1) | **NO** — `middle` is the only word selecting column 2; removing it removes the column |
| `put the bottle in the bottom side middle edge of the table` | `grid_C2` | row vocabulary absent from training (§2); `middle` is both row B and column 2 (§1) | **NO** — `middle` is the only word selecting column 2; removing it removes the column |
| `put the car in the bottom side middle edge of the table` | `grid_C2` | row vocabulary absent from training (§2); `middle` is both row B and column 2 (§1) | **NO** — `middle` is the only word selecting column 2; removing it removes the column |
| `put the rubiks cube in the bottom side middle edge of the table` | `grid_C2` | row vocabulary absent from training (§2); `middle` is both row B and column 2 (§1) | **NO** — `middle` is the only word selecting column 2; removing it removes the column |
| `can you please keep the bottle in the bottom right side corner of the table?` | `grid_C3` | row vocabulary absent from training (§2) | yes |
| `can you please keep the car in the bottom right side corner of the table?` | `grid_C3` | row vocabulary absent from training (§2) | yes |
| `can you please keep the rubiks cube in the bottom right side corner of the table?` | `grid_C3` | row vocabulary absent from training (§2) | yes |
| `go to the bottom right side corner of the table` | `grid_C3` | row vocabulary absent from training (§2) | yes |
| `move to the bottom right side corner of the table` | `grid_C3` | row vocabulary absent from training (§2) | yes |
| `navigate to the bottom right side corner of the table` | `grid_C3` | row vocabulary absent from training (§2) | yes |
| `pick up the bottle and keep in the bottom right side corner of the table` | `grid_C3` | row vocabulary absent from training (§2) | yes |
| `pick up the car and keep in the bottom right side corner of the table` | `grid_C3` | row vocabulary absent from training (§2) | yes |
| `pick up the rubiks cube and keep in the bottom right side corner of the table` | `grid_C3` | row vocabulary absent from training (§2) | yes |
| `put the bottle in the bottom right side corner of the table` | `grid_C3` | row vocabulary absent from training (§2) | yes |
| `put the car in the bottom right side corner of the table` | `grid_C3` | row vocabulary absent from training (§2) | yes |
| `put the rubiks cube in the bottom right side corner of the table` | `grid_C3` | row vocabulary absent from training (§2) | yes |

### Cannot be fixed without changing ground truth: 36 commands

| command | ground truth | why not |
|---|---|---|
| `can you please keep the bottle in the upper side middle edge of the table?` | `grid_A2` | the ambiguous word is the *only* word selecting that cell's column; any replacement changes which cell is meant |
| `can you please keep the car in the upper side middle edge of the table?` | `grid_A2` | the ambiguous word is the *only* word selecting that cell's column; any replacement changes which cell is meant |
| `can you please keep the rubiks cube in the upper side middle edge of the table?` | `grid_A2` | the ambiguous word is the *only* word selecting that cell's column; any replacement changes which cell is meant |
| `go to the upper side middle edge of the table` | `grid_A2` | the ambiguous word is the *only* word selecting that cell's column; any replacement changes which cell is meant |
| `move to the upper side middle edge of the table` | `grid_A2` | the ambiguous word is the *only* word selecting that cell's column; any replacement changes which cell is meant |
| `navigate to the upper side middle edge of the table` | `grid_A2` | the ambiguous word is the *only* word selecting that cell's column; any replacement changes which cell is meant |
| `pick up the bottle and keep in the upper side middle edge of the table` | `grid_A2` | the ambiguous word is the *only* word selecting that cell's column; any replacement changes which cell is meant |
| `pick up the car and keep in the upper side middle edge of the table` | `grid_A2` | the ambiguous word is the *only* word selecting that cell's column; any replacement changes which cell is meant |
| `pick up the rubiks cube and keep in the upper side middle edge of the table` | `grid_A2` | the ambiguous word is the *only* word selecting that cell's column; any replacement changes which cell is meant |
| `put the bottle in the upper side middle edge of the table` | `grid_A2` | the ambiguous word is the *only* word selecting that cell's column; any replacement changes which cell is meant |
| `put the car in the upper side middle edge of the table` | `grid_A2` | the ambiguous word is the *only* word selecting that cell's column; any replacement changes which cell is meant |
| `put the rubiks cube in the upper side middle edge of the table` | `grid_A2` | the ambiguous word is the *only* word selecting that cell's column; any replacement changes which cell is meant |
| `can you please keep the bottle in the middle of the table?` | `grid_B2` | the ambiguous word is the *only* word selecting that cell's column; any replacement changes which cell is meant |
| `can you please keep the car in the middle of the table?` | `grid_B2` | the ambiguous word is the *only* word selecting that cell's column; any replacement changes which cell is meant |
| `can you please keep the rubiks cube in the middle of the table?` | `grid_B2` | the ambiguous word is the *only* word selecting that cell's column; any replacement changes which cell is meant |
| `go to the middle of the table` | `grid_B2` | the ambiguous word is the *only* word selecting that cell's column; any replacement changes which cell is meant |
| `move to the middle of the table` | `grid_B2` | the ambiguous word is the *only* word selecting that cell's column; any replacement changes which cell is meant |
| `navigate to the middle of the table` | `grid_B2` | the ambiguous word is the *only* word selecting that cell's column; any replacement changes which cell is meant |
| `pick up the bottle and keep in the middle of the table` | `grid_B2` | the ambiguous word is the *only* word selecting that cell's column; any replacement changes which cell is meant |
| `pick up the car and keep in the middle of the table` | `grid_B2` | the ambiguous word is the *only* word selecting that cell's column; any replacement changes which cell is meant |
| `pick up the rubiks cube and keep in the middle of the table` | `grid_B2` | the ambiguous word is the *only* word selecting that cell's column; any replacement changes which cell is meant |
| `put the bottle in the middle of the table` | `grid_B2` | the ambiguous word is the *only* word selecting that cell's column; any replacement changes which cell is meant |
| `put the car in the middle of the table` | `grid_B2` | the ambiguous word is the *only* word selecting that cell's column; any replacement changes which cell is meant |
| `put the rubiks cube in the middle of the table` | `grid_B2` | the ambiguous word is the *only* word selecting that cell's column; any replacement changes which cell is meant |
| `can you please keep the bottle in the bottom side middle edge of the table?` | `grid_C2` | the ambiguous word is the *only* word selecting that cell's column; any replacement changes which cell is meant |
| `can you please keep the car in the bottom side middle edge of the table?` | `grid_C2` | the ambiguous word is the *only* word selecting that cell's column; any replacement changes which cell is meant |
| `can you please keep the rubiks cube in the bottom side middle edge of the table?` | `grid_C2` | the ambiguous word is the *only* word selecting that cell's column; any replacement changes which cell is meant |
| `go to the bottom side middle edge of the table` | `grid_C2` | the ambiguous word is the *only* word selecting that cell's column; any replacement changes which cell is meant |
| `move to the bottom side middle edge of the table` | `grid_C2` | the ambiguous word is the *only* word selecting that cell's column; any replacement changes which cell is meant |
| `navigate to the bottom side middle edge of the table` | `grid_C2` | the ambiguous word is the *only* word selecting that cell's column; any replacement changes which cell is meant |
| `pick up the bottle and keep in the bottom side middle edge of the table` | `grid_C2` | the ambiguous word is the *only* word selecting that cell's column; any replacement changes which cell is meant |
| `pick up the car and keep in the bottom side middle edge of the table` | `grid_C2` | the ambiguous word is the *only* word selecting that cell's column; any replacement changes which cell is meant |
| `pick up the rubiks cube and keep in the bottom side middle edge of the table` | `grid_C2` | the ambiguous word is the *only* word selecting that cell's column; any replacement changes which cell is meant |
| `put the bottle in the bottom side middle edge of the table` | `grid_C2` | the ambiguous word is the *only* word selecting that cell's column; any replacement changes which cell is meant |
| `put the car in the bottom side middle edge of the table` | `grid_C2` | the ambiguous word is the *only* word selecting that cell's column; any replacement changes which cell is meant |
| `put the rubiks cube in the bottom side middle edge of the table` | `grid_C2` | the ambiguous word is the *only* word selecting that cell's column; any replacement changes which cell is meant |

**The fix for these is not in the data, it is in the legend.** Giving row B and column 2 distinct labels — for example `B = centre row`, `2 = middle column` — resolves every one of them without touching a single command or any ground truth.

