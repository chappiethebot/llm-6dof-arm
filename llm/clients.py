"""llm/clients.py — Phase 3, step 1: one interface to all three models.
Returns RAW model text. No JSON extraction, no validation, no scoring — those are later steps.
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
    return [
        OpenAIClient("gpt-4o-mini"),
        OllamaClient("llama3.2:3b"),
        OllamaClient("qwen2.5:3b"),
        OllamaClient("qwen2.5:0.5b"),
        OllamaClient("deepseek-r1:7b"),   # reasoning model — 5th model, makes grid 5x4=20
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
