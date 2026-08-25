"""ft/prep_data.py — build LoRA SFT data for rung 3.

INTEGRITY: the prompt is produced by llm.prompts.build_prompt with
schema/schema_v2.json and strategy "zero_shot" — the SAME call bench/run_cell.py
makes. It is not transcribed here. If build_prompt changes, this changes with it.

The chat wrapper is tokenizer.apply_chat_template, verified byte-identical to
Ollama's generate-path template rendering for Qwen2.5-*-Instruct.

Outputs prompt_text and target_text SEPARATELY. Loss masking happens in the
trainer, not here, so the boundary stays inspectable.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoTokenizer
from llm.prompts import build_prompt, load_schema

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = ROOT / "schema" / "schema_v2.json"
OUT_DIR = ROOT / "ft" / "data"

SPLITS = {
    "train": ROOT / "data" / "train_v2.jsonl",
    "dev":   ROOT / "data" / "dev_v2.jsonl",
}

MODELS = {
    "3b":   "Qwen/Qwen2.5-3B-Instruct",
    "0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
}


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    schema = load_schema(str(SCHEMA_PATH))
    print(f"schema      : {SCHEMA_PATH}")
    print(f"schema_ver  : {schema.get('schema_version')}")
    print(f"locations   : {len(schema.get('locations', []))}")
    print(f"verbs       : {list(schema.get('verbs', {}).keys())}")
    print("=" * 72)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for tag, model_id in MODELS.items():
        tok = AutoTokenizer.from_pretrained(model_id)
        print(f"\n### tokenizer: {model_id}")

        for split, src in SPLITS.items():
            rows = read_jsonl(src)
            out_path = OUT_DIR / f"{split}_sft_{tag}.jsonl"

            prompt_lens, target_lens, total_lens = [], [], []
            written = 0

            with open(out_path, "w", encoding="utf-8") as out:
                for row in rows:
                    raw_prompt = build_prompt(
                        row["command"], "zero_shot", schema, examples=None
                    )
                    prompt_text = tok.apply_chat_template(
                        [{"role": "user", "content": raw_prompt}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    target_text = json.dumps(row["expected"]) + "<|im_end|>"

                    p_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
                    t_ids = tok(target_text, add_special_tokens=False)["input_ids"]
                    prompt_lens.append(len(p_ids))
                    target_lens.append(len(t_ids))
                    total_lens.append(len(p_ids) + len(t_ids))

                    out.write(json.dumps({
                        "command":     row["command"],
                        "prompt_text": prompt_text,
                        "target_text": target_text,
                    }, ensure_ascii=False) + "\n")
                    written += 1

            print(f"  {split:5s} -> {out_path.name}  rows={written}")
            print(f"    prompt tokens  min={min(prompt_lens)} max={max(prompt_lens)}")
            print(f"    target tokens  min={min(target_lens)} max={max(target_lens)}")
            print(f"    TOTAL tokens   min={min(total_lens)} max={max(total_lens)}")

    print("=" * 72)
    print("SAMPLE — first train row, 3b tokenizer, full repr:")
    first = read_jsonl(OUT_DIR / "train_sft_3b.jsonl")[0]
    print()
    print("PROMPT_TEXT:")
    print(repr(first["prompt_text"]))
    print()
    print("TARGET_TEXT:")
    print(repr(first["target_text"]))
    print("=" * 72)
    print("HUMAN-READABLE:")
    print(first["prompt_text"] + first["target_text"])


if __name__ == "__main__":
    main()