"""Focused Stage-0 checks. No training, downloads or held-out reads.

Uses one randomly initialized BiLSTM to check checkpoint serialization on CPU.
"""
import ast
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
from s0_grpo_signal_check import exact_reward, rewards, prompt
from s0_train_grpo import select_epoch, signal_allows_sft
from s0_retrain_legacy import compare
from s0_freeze import dir_sha


def raises(exception, fn):
    try:
        fn()
    except exception:
        return
    raise AssertionError("Expected %s" % exception)


def recipe_check():
    copied = ast.parse(Path(__file__).with_name("legacy_recipes.py").read_text(encoding="utf-8"))
    recipes = [("a1_arms.py", "run_frozen_probe", "predict"),
               ("a1_arms.py", "run_tinybert", "predict"),
               ("a1_arms.py", "run_bilstm", "predict"),
               ("a1_rest.py", "run_flant5", "out"),
               ("p5_baselines.py", "build_knn", "predict"),
               ("p5_baselines.py", "build_tfidf", "predict")]
    for filename, name, boundary in recipes:
        source = ast.parse((C.REPO / "thesis2/scripts" / filename).read_text(encoding="utf-8"))
        original = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == name)
        actual = next(n for n in copied.body if isinstance(n, ast.FunctionDef) and n.name == name)
        stop = next(i for i, n in enumerate(original.body)
                    if (isinstance(n, ast.FunctionDef) and n.name == boundary)
                    or (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == boundary
                                                         for t in n.targets)))
        # p5 formerly loaded TRAIN/DEV/HELD at import; only TRAIN is now read locally.
        actual_body = actual.body[1:-1] if filename == "p5_baselines.py" else actual.body[:-1]
        assert [ast.dump(n) for n in original.body[:stop]] == [ast.dump(n) for n in actual_body], name
    print("[Certain] Six training bodies match the original AST, including seeds and dev selection.")


def main():
    C.start_log("test_stage0")
    recipe_check()
    for path in Path(__file__).parent.glob("*.py"):
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    truth = dict(action="fetch", target="bottle", modifiers=[])
    assert exact_reward('{"action":"fetch","target":"bottle"}', truth) == 1.0
    for bad in ("broken", "[]", "null", '"fetch"', '{"action":"hold","target":"bottle"}'):
        assert exact_reward(bad, truth) == 0.0
    assert rewards([json.dumps(truth), "invalid"], [json.dumps(truth)] * 2) == [1.0, 0.0]
    assert select_epoch([dict(epoch=1, dev_hits=52), dict(epoch=2, dev_hits=54),
                         dict(epoch=3, dev_hits=55), dict(epoch=4, dev_hits=55)])["epoch"] == 2
    saved = {str(i): "{}" for i in range(217)}
    assert compare(saved, saved) == []
    assert compare(saved, dict(saved, **{"216": "changed"})) == ["216"]
    raises(ValueError, lambda: compare(saved, {k: v for k, v in saved.items() if k != "216"}))
    print("[Certain] Exact-match reward, earliest plateau and complete 217-command coverage checks pass.")
    with tempfile.TemporaryDirectory(dir=C.OUT, prefix="stage0_check_") as temp:
        root = Path(temp)
        adapter = root / "adapter"
        adapter.mkdir()
        (adapter / "weights").write_bytes(b"original")
        frozen = dict(arm="grpo_base", path=str(adapter), sha256=dir_sha(adapter),
                      dev_action_match="0.5")
        with patch.object(C, "read_frozen", return_value={"grpo_base": frozen}):
            C.require_frozen("grpo_base")
            (adapter / "weights").write_bytes(b"changed")
            raises(SystemExit, lambda: C.require_frozen("grpo_base"))
        with patch.object(C, "read_frozen", return_value={}):
            raises(SystemExit, lambda: C.require_frozen("tinybert"))
        signal = root / "signal.json"
        with patch("s0_train_grpo.SIGNAL", signal), patch.object(C, "sha256_of", return_value="hash"):
            base = dict(prompts=230, group_size=8, temperature=1.0,
                        sft_sha256="hash", train_sha256="hash")
            signal.write_text(json.dumps(dict(base, mixed_prompts=22)), encoding="utf-8")
            assert signal_allows_sft() is False
            signal.write_text(json.dumps(dict(base, mixed_prompts=23)), encoding="utf-8")
            assert signal_allows_sft() is True
    # Importing recipes must not read any dataset or launch a trainer.
    with patch.object(C, "load_split", side_effect=AssertionError("Unexpected dataset read")):
        import legacy_recipes
        assert legacy_recipes.SEED == 20260820
    import torch
    from s0_retrain_legacy import persist, load_predict
    with tempfile.TemporaryDirectory(dir=C.OUT, prefix="legacy_reload_") as temp:
        with patch.object(C, "MODELS", Path(temp)):
            vocab = {"<pad>": 0, "<unk>": 1, "fetch": 2, "bottle": 3}
            model = legacy_recipes.BiLSTMJoint(len(vocab)).eval()
            persist("bilstm", dict(model=model, vocab=vocab))
            loaded = load_predict("bilstm")
            for command in ("fetch bottle", "unknown bottle", " ".join(["fetch"] * 30)):
                ids = [vocab.get(w, 1) for w in command.lower().split()][:24]
                with torch.no_grad():
                    lg = model(torch.tensor([ids + [0] * (24 - len(ids))]))
                expected = legacy_recipes.emit(int(lg["verb"].argmax()), int(lg["target"].argmax()),
                                               legacy_recipes.L[int(lg["location"].argmax())],
                                               legacy_recipes.SPACE)
                obj, info = loaded(command)
                assert obj == expected
                assert info["p_location"] == float(torch.softmax(lg["location"], -1).max())
    print("[Certain] Untrained BiLSTM save/reload preserves outputs, OOV/truncation and confidence.")
    assert prompt("fetch the bottle").endswith("<|im_start|>assistant\n")
    print("[Certain] Directory tampering is rejected; legacy freeze and exact 10% signal boundary pass.")
    C.gate(True, ["[Certain] Stage-0 checks passed; no training, downloads or held-out inference."])


if __name__ == "__main__":
    main()
