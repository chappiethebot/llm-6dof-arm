import json, sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from llm.prompts import build_prompt, load_schema
from llm.clients import OllamaClient

schema = load_schema("schema/schema_v2.json")
client = OllamaClient("qwen2.5-0.5b-lora")

PROMPTS = [
    "can you grab the toy car for me",
    "put the bottle in the left side middle edge",
    "stop right now",
    "put the bottle in a spot past the right edge",
]
for cmd in PROMPTS:
    p = build_prompt(cmd, "zero_shot", schema, examples=None)
    try:
        out = client.complete(p)
        try:
            json.loads(out.strip()); tag = "OK "
        except Exception:
            tag = "BAD"
    except Exception as e:
        out, tag = f"{type(e).__name__}: {e}", "ERR"
    print(f"{tag} {cmd!r}\n     -> {out.strip()!r}\n")
