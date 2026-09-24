"""thesis2/final/s0_freeze.py -- STAGE 0: freeze every NEW arm's checkpoint.

Writes the SHA-256 of each new frozen checkpoint to
thesis2/out/final/frozen_checkpoints.csv together with the selected epoch and
the dev_v2 action match that selected it.

This file is the gate for rule 2: s1_run_arm.py refuses to touch a held-out
split for a NEW arm whose hash is not on record, and re-checks the hash before
every held-out run, so a checkpoint that changed after freezing cannot be
evaluated silently.

Pre-existing arms (the flat/factorized classifiers, the LoRA, the V5 cells, ...)
are NOT registered here: their provenance is the earlier run, and the gate
exempts them by name in common.PREEXISTING_ARMS.

Run AFTER every Stage 0 trainer and the SALSA dev passes, BEFORE any Stage 1
held-out run:
    python thesis2/final/s0_freeze.py
"""
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C                                                      # noqa: E402

# arm -> (checkpoint path, selection record). A missing pair is reported, not
# invented: an arm that was never trained stays out of the file and therefore
# stays blocked from held-out inference.
CANDIDATES = [
    ("textcnn", C.MODELS / "textcnn.pt", C.OUT / "selection_textcnn.json"),
    ("mlpbow", C.MODELS / "mlpbow.pt", C.OUT / "selection_mlpbow.json"),
    ("grpo_base", C.MODELS / "grpo_base", C.OUT / "selection_grpo_base.json"),
    ("grpo_sft", C.MODELS / "grpo_sft", C.OUT / "selection_grpo_sft.json"),
]
for key in ("qwen2.5-3b-lora", "Qwen2.5-3B-Instruct",
            "qwen2.5-0.5b-lora", "Qwen2.5-0.5B-Instruct"):
    CANDIDATES.append(("salsa_full_%s" % key, None,
                       C.OUT / ("selection_salsa_full_%s.json" % key)))


def dir_sha(p: Path) -> str:
    """Hash a checkpoint directory deterministically (sorted relative paths)."""
    import hashlib
    h = hashlib.sha256()
    for f in sorted(p.rglob("*")):
        if f.is_file():
            h.update(str(f.relative_to(p)).replace("\\", "/").encode())
            h.update(C.sha256_of(f).encode())
    return h.hexdigest()


def main():
    C.start_log("s0_freeze")
    rows, missing, skipped = [], [], []
    for arm, ckpt, sel in CANDIDATES:
        if not sel.exists():
            missing.append("%s: no selection record (%s)" % (arm, sel.name))
            continue
        rec = json.loads(sel.read_text(encoding="utf-8"))
        if rec.get("smoke"):
            skipped.append("%s: selection record is from a SMOKE run -- not frozen" % arm)
            continue
        if arm.startswith("salsa_full_"):
            # SALSA-full trains nothing: the frozen object is the model+adapter it
            # reads, which is pre-existing and already immutable upstream. The row
            # records WHICH weights and the dev score, with no new checkpoint.
            m = rec.get("model", {})
            path = C.REPO / (m.get("adapter") or "")
            sha = dir_sha(path) if m.get("adapter") and path.exists() else "n/a: base weights only"
            rows.append(dict(arm=arm, kind="logit-classifier over frozen weights",
                             path=str(path if m.get("adapter") else m.get("base", "")),
                             sha256=sha, selected_epoch="n/a: no training",
                             dev_action_match=rec.get("dev_action_match", ""),
                             frozen_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
            continue
        if ckpt is None or not Path(ckpt).exists():
            missing.append("%s: selection record exists but checkpoint %s does not"
                           % (arm, ckpt))
            continue
        p = Path(ckpt)
        sha = dir_sha(p) if p.is_dir() else C.sha256_of(p)
        rows.append(dict(arm=arm, kind="trained checkpoint",
                         path=str(p), sha256=sha,
                         selected_epoch=rec.get("selected_epoch", ""),
                         dev_action_match=rec.get("dev_action_match", ""),
                         frozen_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))

    # Merge with anything already frozen; a re-freeze of the same arm replaces it.
    existing = C.read_frozen()
    for r in rows:
        existing[r["arm"]] = r
    with C.FROZEN.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=C.FROZEN_COLS)
        w.writeheader()
        for arm in sorted(existing):
            w.writerow({k: existing[arm].get(k, "") for k in C.FROZEN_COLS})

    print()
    print("  %-34s %-12s %-18s %s" % ("arm", "selected", "dev action match", "sha256[:16]"))
    for arm in sorted(existing):
        r = existing[arm]
        print("  %-34s %-12s %-18s %s"
              % (arm, r.get("selected_epoch", ""), r.get("dev_action_match", ""),
                 str(r.get("sha256", ""))[:16]))
    for m in missing:
        print("  NOT FROZEN -- %s" % m)
    for s in skipped:
        print("  NOT FROZEN -- %s" % s)

    blocked = sorted(C.NEW_ARMS - set(existing))
    C.gate(len(existing) > 0, [
        "%d arm(s) frozen in %s" % (len(existing), C.FROZEN.relative_to(C.REPO)),
        "new arms still BLOCKED from held-out inference: %s" % (blocked or "none"),
        "pre-existing arms are exempt by name (common.PREEXISTING_ARMS)",
        "next: Stage 1, s1_run_arm.py --split held_out_v3",
    ])


if __name__ == "__main__":
    main()
