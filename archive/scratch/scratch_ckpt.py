import json
from pathlib import Path

for tag in ["0.5b", "3b"]:
    ck = Path("ft/runs") / f"qwen{tag}_lora" / "checkpoint-87"
    print("=" * 72)
    print(ck, "EXISTS:", ck.exists())
    if not ck.exists():
        continue
    for f in sorted(ck.iterdir()):
        print(f"   {f.name:34s} {f.stat().st_size:>12,} bytes")
    cfg = ck / "adapter_config.json"
    if cfg.exists():
        print("   --- adapter_config.json ---")
        print(json.dumps(json.loads(cfg.read_text(encoding="utf-8")), indent=2))
print("=" * 72)

import subprocess
print("llama.cpp available?")
print(subprocess.run(["where", "llama-quantize"], capture_output=True, text=True).stdout or "  not on PATH")
