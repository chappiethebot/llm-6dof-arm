"""llm/classifier_client.py — Rung 4. The DistilBERT classifier behind the same
interface every other model uses.

Duck-types llm.clients.LLMClient rather than subclassing it, so that
llm/clients.py can import THIS module inside all_clients() without a circular
import. The harness only ever uses `.name`, `.complete()`, `.last_thinking` and
`.temperature`, all via getattr — subclassing buys nothing here.

WHAT run_cell.py READS OFF A CLIENT, and what this provides:
    .name           -> "classifier:distilbert"
    .complete(p)    -> raw string. Here: a JSON object, so extract_json and
                       validate_object run UNCHANGED. Same scoring path as
                       every other rung — that is the point.
    .last_thinking  -> "" always. A classifier has no reasoning trace, so
                       `thinking_chars` is 0 for this cell. That is a true
                       measurement, not a missing value.
    .temperature    -> DELIBERATELY NOT DEFINED. run_cell does
                       getattr(client, "temperature", "") and writes a blank
                       cell. There is no sampling here; argmax is deterministic.
                       Writing 0.0 would imply a sampling parameter that does
                       not exist.

LAZY LOADING IS NOT AN OPTIMISATION, IT IS REQUIRED. bench/run_cell.py's
resolve_clients() calls all_clients() on EVERY invocation, including rung-1 and
rung-2 runs. Constructing this eagerly would import torch and load 268 MB of
weights into every single grid run that does not use it. Nothing is imported or
read until the first complete() call.

PADDING — a reported-number decision, recorded here
---------------------------------------------------
Training used padding="max_length" (64 tokens) because batching requires it.
Inference does not: attention_mask already masks padding, so unpadded inference
is numerically equivalent and much faster — commands are ~8 tokens, not 64.
PAD_TO_MAX = False measures the deployment you would actually ship. Set it True
only if you want the padded number, and if you change it, change it BEFORE the
harness run, never after.

SUBSTRATE — state this in the results chapter, every time
----------------------------------------------------------
This runs in-process on CPU, single-threaded. There is no Ollama server, no HTTP
round-trip, no model-load-per-call. Its latency is NOT comparable to the Ollama
numbers without that caveat attached. Part of any advantage is architectural and
part is deployment.
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# The checkpoint chosen by the rung-3 selection rule: EARLIEST epoch reaching
# the dev plateau, not argmax. Epoch 9 (53/57); argmax would have said 10
# (54/57), a one-row difference on a set with +-13-point precision.
SELECTED_EPOCH = 9
CHECKPOINT_DIR = REPO / "ft" / "runs" / "distilbert_cls" / f"epoch_{SELECTED_EPOCH:02d}"

PAD_TO_MAX = False   # see PADDING note above
NUM_THREADS = 1      # single-threaded, matching the stated deployment


class DistilBertClassifierClient:
    """Rung 4 client. Input: a bare command string (strategy="classifier").
    Output: a JSON action object as text."""

    def __init__(self, checkpoint_dir=CHECKPOINT_DIR, pad_to_max=PAD_TO_MAX,
                 num_threads=NUM_THREADS):
        self.name = "classifier:distilbert"
        self.last_thinking = ""          # never populated; a classifier has no trace
        self.checkpoint_dir = Path(checkpoint_dir)
        self.pad_to_max = pad_to_max
        self.num_threads = num_threads
        self._model = None               # loaded on first complete()
        self._tok = None
        self._space = None
        self._max_len = None
        # NOTE: self.temperature is intentionally absent. See module docstring.

    # -- loading -----------------------------------------------------------
    def _ensure_loaded(self):
        if self._model is not None:
            return
        if not self.checkpoint_dir.exists():
            raise FileNotFoundError(
                f"rung-4 checkpoint not found: {self.checkpoint_dir}\n"
                f"Train it first: python -m ft.train_classifier")

        import torch
        from transformers import AutoTokenizer
        # Imported here, not at module top: ft.train_classifier imports torch,
        # and this whole module must stay cheap for runs that never use it.
        from ft.train_classifier import (MultiHeadClassifier, build_label_space,
                                         MAX_LEN)

        torch.set_num_threads(self.num_threads)
        self._torch = torch
        self._max_len = MAX_LEN
        self._space = build_label_space()
        self._tok = AutoTokenizer.from_pretrained(str(self.checkpoint_dir))
        model = MultiHeadClassifier(self._space)
        model.load_state_dict(
            torch.load(self.checkpoint_dir / "model.pt", map_location="cpu"))
        model.eval()
        self._model = model

    # -- inference ---------------------------------------------------------
    def complete(self, prompt: str) -> str:
        """`prompt` IS the bare command: llm/prompts.py's "classifier" branch
        returns it verbatim. If you see schema text arriving here, the cell was
        run with the wrong strategy."""
        self._ensure_loaded()
        torch = self._torch

        kw = dict(truncation=True, max_length=self._max_len, return_tensors="pt")
        kw["padding"] = "max_length" if self.pad_to_max else False

        with torch.no_grad():
            enc = self._tok(prompt, **kw)
            vl, tl, ll = self._model(enc["input_ids"], enc["attention_mask"])
            obj = self._emit(vl.argmax(-1).item(), tl.argmax(-1).item(),
                             ll.argmax(-1).item())

        self.last_thinking = ""
        return json.dumps(obj)

    def _emit(self, verb_idx, target_idx, location_idx):
        """Reuses ft.train_classifier.emit so training, diagnostics and the
        harness cannot drift apart. The field mask is read from schema_v2.json,
        the same file llm/pipeline.py's validator reads."""
        from ft.train_classifier import emit
        return emit(verb_idx, target_idx, location_idx, self._space)


if __name__ == "__main__":
    c = DistilBertClassifierClient()
    print(f"name         : {c.name}")
    print(f"temperature  : {getattr(c, 'temperature', '<absent — correct>')}")
    print(f"checkpoint   : {c.checkpoint_dir}")
    print(f"pad_to_max   : {c.pad_to_max}   threads: {c.num_threads}")
    print(f"loaded before first call: {c._model is not None}  (must be False)\n")
    for cmd in ["put the bottle in box 1",
                "stop",
                "move to the far left corner",
                "inspect the screwdriver"]:
        print(f"  {cmd!r}\n    -> {c.complete(cmd)}")
    print(f"\nloaded after calls: {c._model is not None}   "
          f"last_thinking: {c.last_thinking!r}")
