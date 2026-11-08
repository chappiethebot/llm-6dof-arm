"""Read-only audit checks. Run from the repository root; no model calls."""
import ast
import collections
import csv
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm.extract import extract_json
from llm.score import score
from schema.validate_object import validate_object
from sim.workspace import grid_cells, resolve

csv.field_size_limit(2**31 - 1)


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def isolated_function(path, name, namespace):
    """Execute an unchanged function AST without importing script side effects."""
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8-sig"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), namespace)
    return namespace[name]


def main():
    files = subprocess.check_output(
        ["git", "ls-files", "*.py"], cwd=ROOT, text=True).splitlines()
    syntax_errors, counts = [], collections.defaultdict(lambda: [0, 0])
    for name in files:
        text = (ROOT / name).read_text(encoding="utf-8-sig")
        group = name.split("/")[0] if "/" in name else "(root)"
        counts[group][0] += 1
        counts[group][1] += len(text.splitlines())
        try:
            ast.parse(text, filename=name)
        except SyntaxError as e:
            syntax_errors.append(str(e))
    schema = json.loads((ROOT / "schema/schema_v2.json").read_text())
    splits, datasets = {}, {}
    for name in ("seen_v2", "train_v2", "dev_v2", "held_out_v2", "held_out_v3", "held_out_v4b", "held_out_v4c"):
        rows = [json.loads(x) for x in (ROOT / f"data/{name}.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        datasets[name] = rows
        cmds = [r["command"].strip().lower() for r in rows]
        splits[name] = dict(rows=len(rows), duplicate_commands=len(cmds)-len(set(cmds)),
                            invalid_truth=sum(not validate_object(r["expected"], schema).ok for r in rows))
    overlap = {}
    for a, b in (("train_v2", "dev_v2"), ("train_v2", "held_out_v3"), ("train_v2", "held_out_v4c"), ("dev_v2", "held_out_v3")):
        overlap[a + "/" + b] = len({r["command"].strip().lower() for r in datasets[a]} & {r["command"].strip().lower() for r in datasets[b]})
    pred_files, pred_rows, bad_scores, bad_truth, duplicates, invalid, validity_gaps = 0, 0, [], [], [], collections.Counter(), []
    final = ROOT / "thesis2/out/final"
    for path in sorted(final.glob("pred_*.csv")):
        rows = read_csv(path)
        pred_files += 1
        pred_rows += len(rows)
        seen = set()
        for r in rows:
            key = (r["command_id"], r["repeat"])
            if key in seen:
                duplicates.append([path.name, key])
            seen.add(key)
            obj, truth = json.loads(r["pred_json"]), json.loads(r["truth_json"])
            valid = validate_object(obj, schema).ok
            if not valid:
                invalid[path.name] += 1
            if isinstance(obj, dict) and obj.get("action") in schema["verbs"] and not valid:
                validity_gaps.append([path.name, r["command_id"], obj])
            if score(obj, truth).strict_match != (r["correct"].lower() == "true"):
                bad_scores.append([path.name, key])
            split = "held_out_v4c" if "held_out_v4c" in path.name else "held_out_v3"
            source = datasets[split][int(r["command_id"])]
            if r["command"] != source["command"] or not score(truth, source["expected"]).strict_match:
                bad_truth.append([path.name, key])
    cells = {}
    for path in sorted(final.glob("results_v5cells_*.csv")):
        rows = read_csv(path)
        keys = [(r["model"], r["strategy"], r["command_id"], r["run_index"]) for r in rows]
        cells[path.name] = dict(rows=len(rows), duplicates=len(keys)-len(set(keys)),
                                cells=len({k[:2] for k in keys}), stages=dict(collections.Counter(r["stage"] for r in rows)))
    probes = {
        "prose_json_with_quoted_closing_brace": extract_json('Answer: {"action":"stop","modifiers":["}"]}').__dict__,
        "invalid_modifiers_accepted": validate_object({"action":"stop", "modifiers":42}, schema).__dict__,
        "strict_normalized_false_modifier": score({"action":"stop", "modifiers":False}, {"action":"stop"}).__dict__,
        "full_grid_nearest_free": resolve("nearest_free_cell", occupied_xy=list(grid_cells().values())),
    }
    # Exercise the new demo's dispatch with the real resolver, without starting Isaac.
    from sim.isaac_backend import IsaacBackend
    backend = object.__new__(IsaacBackend)
    backend.np = None  # grid_A1 is rejected before any NumPy or world access.
    execute = isolated_function("thesis2/demo/run_isaac_demo.py", "execute", {"json": json})
    probes["isaac_demo_grid_mapping"] = execute(
        SimpleNamespace(move_to=lambda loc: backend.resolve(loc)),
        {"action": "move_to", "location": "grid_A1"})
    probes["isaac_demo_invalid_stop_extra_field"] = execute(
        SimpleNamespace(), {"action": "stop", "target": "bottle"})
    # Correct totals and distinct-ID counts can still hide missing repeats.
    fake_rows = [{"command_id": cid, "repeat": "0"} for cid in ("0", "0", "1", "1")]
    census = isolated_function("thesis2/final/common.py", "pred_census", {
        "pred_rows_of": lambda _: fake_rows, "pred_path": lambda *a: None})
    probes["census_accepts_duplicate_repeat_keys"] = census("stub", "stub", 4, 2)
    def majority(xs):
        vals = list(xs)
        return sum(vals) * 2 > len(vals)
    collapse = isolated_function("thesis2/final/s2_metrics.py", "collapse", {
        "defaultdict": collections.defaultdict, "C": SimpleNamespace(majority=majority)})
    sampling_rows = [dict(command_id="0", repeat=str(i), correct=str(i > 0), row_correct=str(i > 0)) for i in range(3)]
    probes["sampling_field_collapse"] = collapse(sampling_rows, "sampling")
    result = dict(commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,text=True).strip(),
                  source_counts=dict(counts), syntax_errors=syntax_errors, datasets=splits, exact_overlap=overlap,
                  prediction_files=pred_files, prediction_rows=pred_rows, duplicate_prediction_keys=duplicates,
                  stored_score_mismatches=bad_scores, stored_truth_mismatches=bad_truth,
                  invalid_predictions=dict(invalid), schema_validity_overcount_rows=len(validity_gaps),
                  schema_validity_examples=validity_gaps[:8], v5_cells=cells, probes=probes)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
