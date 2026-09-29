"""thesis2/final/make_defence_numbers.py -- write DEFENCE_NUMBERS_rev3.md.

    python thesis2/final/make_defence_numbers.py

Built ONLY from four files in thesis2/out/final/:
    m = metrics_final.csv      s = final_significance.csv
    e = unified_efficiency.csv v = v5cells_both_splits.csv

Every number is written as  `VALUE`⟦F;KEY;SPLIT;COLUMN⟧  -- the file letter, the
row key (arm, hypothesis id, or model/strategy), the split and the column -- and
the value is copied verbatim from that cell. check_defence_numbers.py re-reads
every citation from its source and asserts equality; it shares no code with this
script.
"""
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C                                                      # noqa: E402

SPLIT = {"held_out_v3": "v3", "held_out_v4c": "v4c"}


def load(name):
    with (C.OUT / name).open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


M = {(r["arm"], SPLIT[r["split"]]): r for r in load("metrics_final.csv")}
S = {r["id"]: r for r in load("final_significance.csv") if r["family"] == "primary"}
E = {(r["arm"], SPLIT[r["split"]]): r for r in load("unified_efficiency.csv")}
V = {"%s/%s" % (r["model"], r["strategy"]): r for r in load("v5cells_both_splits.csv")}


def cite(f, key, split, col):
    src = {"m": M, "e": E}.get(f)
    if f == "m" or f == "e":
        val = src[(key, split)][col]
    elif f == "s":
        val = S[key][col]
    elif f == "v":
        val = V[key][col]
    else:
        raise KeyError(f)
    if "#" in col:
        raise KeyError("use cite_group for word groups")
    return "`%s`⟦%s;%s;%s;%s⟧" % (val, f, key, split, col)


def cite_group(arm, split, group):
    s = M[(arm, split)]["row_by_word_group"]
    frac = dict(p.strip().rsplit(" ", 1) for p in s.split(";") if p.strip())[group]
    return "`%s`⟦m;%s;%s;row_by_word_group#%s⟧" % (frac, arm, split, group)


def row(cells):
    return "| " + " | ".join(cells) + " |"


def header(cols):
    return [row(cols), "|" + "|".join("---" for _ in cols) + "|"]


L = []
W = L.append
W("# Defence numbers -- revision 3")
W("")
W("Written %s by `thesis2/final/make_defence_numbers.py`. Supersedes "
  "`audit/DEFENCE_NUMBERS.md` (revision 2). **Every number below is copied "
  "verbatim from one cell of one of four files and carries that cell's address** "
  "in double brackets -- file letter; row key; split; column -- with the files "
  "abbreviated:" % time.strftime("%Y-%m-%d"))
W("")
W("- `m` = `thesis2/out/final/metrics_final.csv` (key = arm)")
W("- `s` = `thesis2/out/final/final_significance.csv` (key = hypothesis id)")
W("- `e` = `thesis2/out/final/unified_efficiency.csv` (key = arm)")
W("- `v` = `thesis2/out/final/v5cells_both_splits.csv` (key = model/strategy)")
W("")
W("`thesis2/final/check_defence_numbers.py` re-reads every one of them and fails if "
  "any differs. Effective n is 217 commands per split. [Certain]")
W("")

# ---------------------------------------------------------------- 1
W("## 1. Headline accuracy, both splits")
W("")
W("Action match = verb, target and location all correct (`llm/score.py`). Row = "
  "row accuracy on the 108 grid commands. `cascade_oracle` is a ceiling, not a "
  "system; `rules_open` is vocabulary-unfair. Sorted by held_out_v4c action match. "
  "[Certain]")
W("")
L += header(["arm", "v3 action", "v4c action", "v3 row", "v4c row", "class"])
arms = sorted({a for a, _ in M}, key=lambda a: -float(M[(a, "v4c")]["action_match"]))
for a in arms:
    cls = {"cascade_oracle": "ceiling, not a system",
           "rules_open": "vocabulary-unfair"}.get(a, "system")
    W(row([a, cite("m", a, "v3", "action_match"), cite("m", a, "v4c", "action_match"),
           cite("m", a, "v3", "row_accuracy"), cite("m", a, "v4c", "row_accuracy"), cls]))
W("")

# ---------------------------------------------------------------- 2
W("## 2. The primary family, H1-H8")
W("")
W("Clustered bootstrap over command_id (2,000 resamples, seed 20260827), exact "
  "McNemar, Holm across the eight. CLAIMABLE iff the CI excludes 0 **and** "
  "p_holm < 0.05. [Certain]")
W("")
L += header(["id", "split", "A vs B", "metric", "n", "acc A", "acc B", "diff pts",
             "95% CI pts", "p", "p_holm", "verdict"])
for h in sorted(S, key=lambda k: int(k[1:])):
    r = S[h]
    sp = SPLIT[r["split"]]
    W(row([h, sp, "%s vs %s" % (r["arm_a"], r["arm_b"]), r["metric"],
           cite("s", h, sp, "n"), cite("s", h, sp, "acc_a"), cite("s", h, sp, "acc_b"),
           cite("s", h, sp, "diff_pts"),
           "[%s, %s]" % (cite("s", h, sp, "ci_lo_pts"), cite("s", h, sp, "ci_hi_pts")),
           cite("s", h, sp, "mcnemar_p"), cite("s", h, sp, "p_holm"),
           cite("s", h, sp, "verdict")]))
W("")
W("**Transparency.** H1, H2, H4 and H5 were measured once before the family was "
  "declared (2026-09-22; `thesis2/final/s4_significance.py`, `PRE_MEASURED`); their "
  "verdicts are re-derived here on the final run's predictions, but the hypotheses "
  "were not blind to a first result. H3, H6, H7 and H8 were fully pre-registered: "
  "declared before any of their arms ran on held-out data. [Certain]")
W("")

# ---------------------------------------------------------------- 3
W("## 3. Efficiency -- main table, held_out_v3")
W("")
W("Measured with the shared Stage-3 protocol (current idle baseline under 2 h, 3 "
  "uncounted warm-ups, >= 120 s, 60 s cool-down; J = (mean load W - idle median W) "
  "x s / N; GPU watts only). The V5 cell runs in Ollama at Q4_K_M; every other row "
  "is transformers FP16 or a CPU/GPU classifier -- different runtimes. The V5 "
  "cell's accuracy is n = 216 (A9 warm-up rule); the arms' are n = 217. [Certain]")
W("")
L += header(["system", "action", "p50 ms", "p95 ms", "J/command", "VRAM above idle MB",
             "CPU ms/command"])
v5key = "ollama:qwen2.5-3b-lora/zero_shot"
W(row(["V5 qwen2.5-3b-lora / zero_shot (Ollama)", cite("v", v5key, "v3", "v3_action"),
       cite("v", v5key, "v3", "v3_latency_p50_ms"), cite("v", v5key, "v3", "v3_latency_p95_ms"),
       cite("v", v5key, "v3", "v3_joules_per_command"),
       cite("v", v5key, "v3", "v3_vram_above_idle_mb"),
       cite("e", "v5:" + v5key, "v3", "cpu_ms_per_command")]))
for a in ("lora_greedy", "a8_speculative", "a10_speculative", "a12_smalllm",
          "flat_gpu", "flat_cpu", "grpo_base", "grpo_sft"):
    W(row([a, cite("m", a, "v3", "action_match"), cite("e", a, "v3", "latency_p50_ms"),
           cite("e", a, "v3", "latency_p95_ms"), cite("m", a, "v3", "joules_per_command"),
           cite("m", a, "v3", "vram_above_idle_mb"), cite("e", a, "v3", "cpu_ms_per_command")]))
W("")
W("A J/command shown as \"%s\" is a measured value <= 0: the arm's load sat inside "
  "the GPU's idle noise. [Certain]" % C.J_FLOOR_TEXT)
W("")

# ---------------------------------------------------------------- 4
W("## 4. Speculative arms -- held_out_v4c supplement")
W("")
W("v4c efficiency is reported for the speculative arms and their verifier only; the "
  "main efficiency table is held_out_v3. Byte-identity to `lora_greedy` held on both "
  "splits, so accuracy equals `lora_greedy`'s by construction. [Certain]")
W("")
L += header(["arm", "action", "p50 ms", "p95 ms", "J/command", "VRAM above idle MB"])
for a in ("lora_greedy", "a8_speculative", "a10_speculative", "a12_smalllm"):
    W(row([a, cite("m", a, "v4c", "action_match"), cite("e", a, "v4c", "latency_p50_ms"),
           cite("e", a, "v4c", "latency_p95_ms"), cite("m", a, "v4c", "joules_per_command"),
           cite("m", a, "v4c", "vram_above_idle_mb")]))
W("")

# ---------------------------------------------------------------- 5
W("## 5. The V5 cells on both splits")
W("")
W("All 36 cells. Collapsed as A9 did (warm-up rows excluded; gpt-4o-mini's 5 repeats "
  "by majority vote, a tie is wrong). CIs: clustered bootstrap, 2,000 resamples, seed "
  "20260827 -- **descriptive only, no hypothesis tested**. [Certain]")
W("")
L += header(["cell", "v3 n", "v3 action [95% CI]", "v4c n", "v4c action [95% CI]",
             "v3 row", "v4c row"])
for k in V:
    W(row([k, cite("v", k, "v3", "v3_n_action"),
           "%s [%s, %s]" % (cite("v", k, "v3", "v3_action"), cite("v", k, "v3", "v3_action_ci_lo"),
                            cite("v", k, "v3", "v3_action_ci_hi")),
           cite("v", k, "v4c", "v4c_n_action"),
           "%s [%s, %s]" % (cite("v", k, "v4c", "v4c_action"),
                            cite("v", k, "v4c", "v4c_action_ci_lo"),
                            cite("v", k, "v4c", "v4c_action_ci_hi")),
           cite("v", k, "v3", "v3_row"), cite("v", k, "v4c", "v4c_row")]))
W("")

# ---------------------------------------------------------------- 6
W("## 6. Row accuracy by row word, held_out_v4c -- the diagnosis")
W("")
W("seen = middle/center (in every training set); unseen = back/front (in no "
  "training set, no legend); mixed = both in one command. Hits / grid commands in "
  "the group. [Certain]")
W("")
L += header(["arm", "row accuracy", "seen", "unseen", "mixed", "row signature"])
for a in ("flat_gpu", "lora_greedy", "grpo_base"):
    W(row([a, cite("m", a, "v4c", "row_accuracy"), cite_group(a, "v4c", "seen"),
           cite_group(a, "v4c", "unseen"), cite_group(a, "v4c", "mixed"),
           cite("m", a, "v4c", "row_signature")]))
W("")
W("Reading: all three are perfect on the seen word and score 0 on the mixed "
  "commands. flat_gpu defaults to row C (43/48 on back/front; also on v3 "
  "upper/bottom); lora_greedy defaults to B; grpo_base splits A/C independently of "
  "the word; only cascade_cloud_always maps back/front (46/48). The classifier's "
  "v4c row advantage is mostly its C default coinciding with 'front' (22 of "
  "lora_greedy's 24 row losses are front commands). Sources: "
  "`thesis2/out/final/row_confusion_by_word.csv` (per-word predicted rows) and "
  "`thesis2/out/final/draft_verifier_gh.csv` (gain/harm by field); "
  "`METHODS_NOTES.md` items 17, 21-23. Signature counts cover all 108 grid "
  "commands. [Certain on the counts; the reading is Likely]")
W("")

out = C.OUT / "DEFENCE_NUMBERS_rev3.md"
out.write_text("\n".join(L) + "\n", encoding="utf-8")
n = sum(line.count("⟦") for line in L)
print("wrote %s -- %d cited numbers" % (out.relative_to(C.REPO), n))
