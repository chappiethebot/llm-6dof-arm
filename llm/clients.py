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
        return resp["response"]


def all_clients():
    return [
        OpenAIClient("gpt-4o-mini"),
        OllamaClient("llama3.2:3b"),
        OllamaClient("qwen2.5:3b"),
        OllamaClient("qwen2.5:0.5b"),
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
        except Exception as e:
            print(f"ERROR: {type(e).__name__}: {e}")
    print("=" * 60)