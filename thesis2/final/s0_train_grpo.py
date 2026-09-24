"""Train GRPO in thesis2rl; save adapters only; dev inference in thesis2gpu.

The user runs this script. No held-out data is read. Each epoch's adapter is
evaluated in a separate thesis2gpu process after moving training tensors to CPU.
Selection is the earliest epoch within one dev hit of the best, over 10 epochs.
Partial runs are retained and refused on restart (no silent optimizer reset).
"""
import argparse
import gc
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
from s0_grpo_adapter_probe import BASE, LORA
from s0_grpo_signal_check import (GPU_PYTHON, SFT, SIGNAL, SYSTEM, prompt, rewards,
                                  require_env, require_probe)

RL_PYTHON = Path("D:/condaenvs/thesis2rl/python.exe")


def select_epoch(history):
    best = max(r["dev_hits"] for r in history)
    return min((r for r in history if r["dev_hits"] >= best - 1), key=lambda r: r["epoch"])


def config(arm):
    from trl import GRPOConfig
    return GRPOConfig(
        output_dir=str(C.MODELS / (arm + "_epochs")), num_train_epochs=10,
        learning_rate=2e-5, lr_scheduler_type="cosine", beta=0.04,
        num_generations=8, temperature=1.0, top_p=1.0, top_k=0,
        max_completion_length=64, loss_type="grpo", scale_rewards="group",
        per_device_train_batch_size=1, gradient_accumulation_steps=8,
        generation_batch_size=8, gradient_checkpointing=True,
        bf16=True, fp16=False, seed=C.SEED, data_seed=C.SEED,
        save_strategy="no", eval_strategy="no", report_to="none",
        logging_steps=1, remove_unused_columns=False,
        dataloader_num_workers=0, optim="adamw_torch", disable_tqdm=True,
    )


def signal_allows_sft():
    if not SIGNAL.exists():
        raise SystemExit("Run s0_grpo_signal_check.py in thesis2gpu first.")
    r = json.loads(SIGNAL.read_text(encoding="utf-8"))
    if (r.get("prompts") != 230 or r.get("group_size") != 8 or r.get("temperature") != 1.0
            or r.get("sft_sha256") != C.sha256_of(SFT / "adapter_model.safetensors")
            or r.get("train_sha256") != C.sha256_of(C.DATA / "train_v2.jsonl")):
        raise SystemExit("STOP: signal check provenance is incomplete or changed.")
    return r["mixed_prompts"] * 10 >= r["prompts"]


def train(arm):
    require_env(RL_PYTHON)
    require_probe()
    # Rule 3, at startup: see the note in s0_grpo_signal_check.main(). The V5
    # legend is pre-existing and train_v2 is used as it is; what this script adds
    # is the wrapper, which must be free of held-out row vocabulary.
    C.assert_no_heldout_vocab(SYSTEM, where="the GRPO training prompt wrapper")
    if arm == "grpo_sft" and not signal_allows_sft():
        C.gate(True, ["[Certain] SKIPPED grpo_sft: fewer than 10% mixed rewards; threshold unchanged."])
    final = C.MODELS / arm
    selection = C.OUT / ("selection_%s.json" % arm)
    if selection.exists():
        C.gate(True, ["[Certain] Existing selection preserved: %s" % selection])
    run = C.MODELS / (arm + "_epochs")
    if final.exists() or (run.exists() and any(run.iterdir())):
        raise SystemExit("STOP: partial GRPO run exists; retain and review it before restarting.")
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed
    from peft import LoraConfig, PeftModel
    from trl import GRPOTrainer

    set_seed(C.SEED)
    torch.cuda.reset_peak_memory_stats()
    rows = C.load_split(C.TRAIN)
    dataset = Dataset.from_list([dict(prompt=prompt(r["command"]),
                                     expected=json.dumps(C.truth_of(r))) for r in rows])
    tok = AutoTokenizer.from_pretrained(BASE, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16)
    model.config.use_cache = False
    lora = LoraConfig(task_type="CAUSAL_LM", **LORA)
    if arm == "grpo_sft":
        model = PeftModel.from_pretrained(model, str(SFT), is_trainable=True)
        # TRL copies this pretrained adapter as its frozen KL reference.
        lora = None
    history = []

    class DevSelection(TrainerCallback):
        def on_epoch_end(self, args, state, control, model=None, optimizer=None, **kwargs):
            epoch = int(round(state.epoch))
            checkpoint = run / ("epoch_%02d" % epoch)
            model.save_pretrained(str(checkpoint), selected_adapters=["default"])
            # Keep the optimizer and scheduler alive; only release their GPU storage.
            model.zero_grad(set_to_none=True)
            model.to("cpu")
            moved = []
            for values in optimizer.state.values():
                for key, value in values.items():
                    if torch.is_tensor(value) and value.is_cuda:
                        moved.append((values, key, value.device))
                        values[key] = value.cpu()
            gc.collect()
            torch.cuda.empty_cache()
            output = C.OUT / ("%s_dev_epoch_%02d.json" % (arm, epoch))
            subprocess.run([str(GPU_PYTHON), str(Path(__file__).with_name("s0_grpo_signal_check.py")),
                            "--evaluate-adapter", str(checkpoint), "--output", str(output)],
                           cwd=C.REPO, check=True)
            result = json.loads(output.read_text(encoding="utf-8"))
            result.update(epoch=epoch, checkpoint=str(checkpoint))
            history.append(result)
            (C.OUT / ("history_%s.json" % arm)).write_text(json.dumps(dict(
                dev=history, training=state.log_history), indent=2), encoding="utf-8")
            model.to("cuda")
            for values, key, device in moved:
                values[key] = values[key].to(device)
            model.train()
            return control

    trainer = GRPOTrainer(model=model, args=config(arm), reward_funcs=rewards,
                          train_dataset=dataset, processing_class=tok, peft_config=lora,
                          callbacks=[DevSelection()])
    trainer.train()
    chosen = select_epoch(history)
    shutil.copytree(chosen["checkpoint"], final)
    selection.write_text(json.dumps(dict(
        arm=arm, selected_epoch=chosen["epoch"], dev_hits=chosen["dev_hits"],
        dev_action_match=chosen["dev_action_match"], dev_total=chosen["dev_total"],
        rule="earliest epoch within one dev hit of best", seed=C.SEED,
        checkpoint=str(final.relative_to(C.REPO)), smoke=False, history=history,
        training_python=sys.executable, inference_python=str(GPU_PYTHON),
        peak_vram_mb=torch.cuda.max_memory_allocated() / 1e6,
    ), indent=2), encoding="utf-8")
    C.gate(True, ["[Certain] Selected epoch %d: %d/57 dev hits; adapter only."
                  % (chosen["epoch"], chosen["dev_hits"]),
                  "[Certain] Run s0_freeze.py before any held-out inference."])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True, choices=["grpo_base", "grpo_sft"])
    ap.add_argument("--check-config", action="store_true", help="validate config without training")
    args = ap.parse_args()
    require_env(RL_PYTHON)
    C.start_log("s0_train_%s" % args.arm)
    if args.check_config:
        cfg = config(args.arm)
        print(cfg.to_json_string())
        C.gate(True, ["[Certain] GRPOConfig constructed; no model loaded or training run."])
    try:
        train(args.arm)
    except __import__("torch").cuda.OutOfMemoryError:
        import torch
        peak = dict(arm=args.arm, peak_allocated_mb=torch.cuda.max_memory_allocated() / 1e6,
                    peak_reserved_mb=torch.cuda.max_memory_reserved() / 1e6)
        (C.OUT / (args.arm + "_oom.json")).write_text(json.dumps(peak, indent=2), encoding="utf-8")
        C.gate(False, ["[Certain] 3B GRPO out of memory: %s" % peak,
                       "STOP: no fallback to a smaller model."])


if __name__ == "__main__":
    main()
