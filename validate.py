#!/usr/bin/env python3
"""
validate.py — check a command dataset (.jsonl) against schema.json.

This is the "judge": it decides, for every row, whether the expected answer
follows the schema rules. It is used two ways:
  1. While authoring the dataset, to catch typos and illegal values.
  2. Later, by the eval harness, to score whether an LLM's output is valid.

Usage:
    python validate.py schema/schema.json data/seen.jsonl
    python validate.py schema/schema.json data/seen.jsonl data/held_out.jsonl

If you pass two data files, it also checks that no command string appears in
both — the "sealed" discipline for the held-out set.

Exit code is 0 if everything passes, 1 if any error is found (so it can be
used in scripts / CI).
"""

import json
import sys


def load_schema(path):
    with open(path, encoding="utf-8") as f:
        schema = json.load(f)
    return (
        schema["verbs"],
        set(schema["targets"]),
        set(schema["locations"]),
    )


def validate_file(path, verbs, targets, locations):
    """Return (errors, verb_counts, command_list) for one .jsonl file."""
    errors = []
    verb_counts = {}
    commands = {}  # command string -> [line numbers]

    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            # 1. Must be valid JSON.
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append((i, f"not valid JSON: {e}"))
                continue

            # Catch a leftover template/README line early.
            if "_README" in row:
                errors.append((i, "README/template line still present — delete it"))
                continue

            cmd = row.get("command")
            exp = row.get("expected", {})
            commands.setdefault(cmd, []).append(i)

            act = exp.get("action")
            verb_counts[act] = verb_counts.get(act, 0) + 1

            # 2. action must be a known verb.
            if act not in verbs:
                errors.append((i, f"unknown verb '{act}'"))
                continue

            # 3. required fields must be present and non-empty.
            for req in verbs[act]["required"]:
                if req not in exp or exp[req] in (None, "", []):
                    errors.append((i, f"'{act}' missing required field '{req}'"))

            # 4. target, if present, must be legal.
            if "target" in exp and exp["target"] not in targets:
                errors.append((i, f"target '{exp['target']}' not in schema"))

            # 5. location, if present, must be legal.
            if "location" in exp and exp["location"] not in locations:
                errors.append((i, f"location '{exp['location']}' not in schema"))

            # 6. no stray fields this verb shouldn't carry.
            allowed = set(verbs[act]["required"]) | {"modifiers"}
            for key in exp:
                if key not in allowed and key != "action":
                    errors.append((i, f"'{act}' carries unexpected field '{key}'"))

    return errors, verb_counts, commands


def report(path, errors, verb_counts, commands):
    total = sum(verb_counts.values())
    print(f"\n=== {path} ===")
    print(f"Rows: {total}    Errors: {len(errors)}")
    for i, msg in errors:
        print(f"  line {i}: {msg}")

    print("Per verb: " + ", ".join(f"{v}={verb_counts[v]}" for v in sorted(verb_counts)))

    dups = {c: ls for c, ls in commands.items() if len(ls) > 1}
    print(f"Duplicate commands within file: {len(dups)}")
    for c, ls in list(dups.items())[:10]:
        print(f"  lines {ls}: {c!r}")


def main():
    if len(sys.argv) < 3:
        print("Usage: python validate.py schema.json data1.jsonl [data2.jsonl]")
        sys.exit(2)

    schema_path = sys.argv[1]
    data_paths = sys.argv[2:]
    verbs, targets, locations = load_schema(schema_path)

    any_error = False
    all_commands = {}  # file path -> set of command strings

    for path in data_paths:
        errors, verb_counts, commands = validate_file(path, verbs, targets, locations)
        report(path, errors, verb_counts, commands)
        if errors:
            any_error = True
        all_commands[path] = set(commands.keys())

    # Cross-file overlap check (only meaningful for two files, e.g. seen vs held_out).
    if len(data_paths) == 2:
        a, b = data_paths
        overlap = all_commands[a] & all_commands[b]
        print(f"\nExact command overlap between the two files: {len(overlap)}")
        for c in list(overlap)[:10]:
            print(f"  {c!r}")
        if overlap:
            any_error = True
            print("  ^ these appear in BOTH files — held-out must not repeat seen.")

    print("\nRESULT:", "FAIL" if any_error else "PASS")
    sys.exit(1 if any_error else 0)


if __name__ == "__main__":
    main()
