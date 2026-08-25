"""llm/clients.py — Phase 3, step 1: one interface to all models.
Returns RAW model text. No JSON extraction, no validation, no scoring — those are later steps.

RUNG 3: the two merged LoRA models are served by Ollama exactly like the base
models — same client class, same `ollama.generate` path, same quantisation as
their base (verified: 397 MB / 1.9 GB, matching stock qwen2.5:0.5b / qwen2.5:3b).
Identical cold-call conditions is the whole point; if they were served
differently, the rung-1 vs rung-3 comparison would be dead.
Ollama tags confirmed via `ollama list`:
    qwen2.5-0.5b-lora:latest   397 MB
    qwen2.5-3b-lora:latest     1.9 GB
The ":latest" suffix is implicit — Ollama resolves the bare name.

RUNG 4: the DistilBERT classifier is registered here alongside the LLMs, but
imported INSIDE the function. Two reasons:
  1. No circular import — llm/classifier_client.py does not import this module.
  2. resolve_clients() calls all_clients() on every bench run. A top-level
     import would pull torch into rung-1 and rung-2 runs that never touch it.
The client itself loads no weights until its first complete() call, so merely
appearing in this list costs nothing.

It is NOT served by Ollama and never will be: it is a PyTorch encoder running
in-process on CPU. That different substrate IS the rung. Its latency is not
comparable to the Ollama numbers without that caveat stated.

NONE of the three additions are in bench/run_grid.py's ALL_MODELS, so the frozen
20-cell base grid is untouched. Ask for them explicitly:
    --models "ollama:qwen2.5-0.5b-lora,ollama:qwen2.5-3b-lora" --strategies zero_shot
    --models "classifier:distilbert" --strategies classifier
"""
import os
from abc import ABC, abstractmethod

from dotenv import load_dotenv

load_dotenv()  # reads OPENAI_API_KEY from .env


class LLMClient(ABC):
    """Every model, cloud or local, answers the same call: .complete(prompt) -> raw string."""
    name: str
    last_thinking: str = ""   # V2: reasoning trace from the last complete() call. "" for non-reasoning models.

    @abstractmethod
    def complete(self, prompt: str) -> str:
        ...


class OpenAIClient(LLMClient):
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0):
        from openai import OpenAI
        self._client = OpenAI()
        self.model = model
        self.temperature = temperature
        self.name = f"openai:{model}"

    def complete(self, prompt: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        self.last_thinking = ""   # OpenAI chat completions expose no reasoning trace
        return resp.choices[0].message.content


class OllamaClient(LLMClient):
    def __init__(self, model: str, temperature: float = 0.0):
        import ollama
        self._ollama = ollama
        self.model = model
        self.temperature = temperature
        self.name = f"ollama:{model}"

    def complete(self, prompt: str) -> str:
        resp = self._ollama.generate(
            model=self.model,
            prompt=prompt,
            options={"temperature": self.temperature},
        )
        # V2 fix: Ollama returns the reasoning trace in a SEPARATE `thinking` field.
        # V1 read only `response`, so every deepseek-r1 trace was generated and discarded.
        # getattr with a fallback keeps non-reasoning models (llama, qwen) working unchanged.
        self.last_thinking = getattr(resp, "thinking", None) or ""
        return resp["response"]


def all_clients():
    # Rung 4. Imported here, not at module level — see the module docstring.
    from llm.classifier_client import DistilBertClassifierClient
    return [
        # ---- rung 1 / rung 2: base models, 5 of them, the frozen grid ----
        OpenAIClient("gpt-4o-mini"),
        OllamaClient("llama3.2:3b"),
        OllamaClient("qwen2.5:3b"),
        OllamaClient("qwen2.5:0.5b"),
        OllamaClient("deepseek-r1:7b"),   # reasoning model — 5th model, makes grid 5x4=20
        # ---- rung 3: merged LoRA, same client, same quantisation as base ----
        OllamaClient("qwen2.5-0.5b-lora"),
        OllamaClient("qwen2.5-3b-lora"),
        # ---- rung 4: different substrate. lazy; loads weights on first call ----
        DistilBertClassifierClient(),
    ]


if __name__ == "__main__":
    test_prompt = 'Reply with a JSON object: {"status": "ok"}'
    for client in all_clients():
        print("=" * 60)
        print(f"MODEL: {client.name}")
        print("-" * 60)
        try:
            raw = client.complete(test_prompt)
            print(repr(raw))  # repr shows hidden newlines, fences, whitespace
            if client.last_thinking:
                print(f"  [thinking: {len(client.last_thinking)} chars] {client.last_thinking[:120]!r}...")
            else:
                print("  [thinking: none]")
        except Exception as e:
            print(f"ERROR: {type(e).__name__}: {e}")
    print("=" * 60)
