"""Streaming chat with a local Ollama model."""

import json
from collections.abc import AsyncIterator

import httpx


class OllamaLLM:
    def __init__(self, base_url: str, model: str, max_tokens: int, keep_alive: str, num_ctx: int = 4096):
        self.model = model
        self.num_ctx = num_ctx
        self.max_tokens = max_tokens
        self.keep_alive = keep_alive
        self.client = httpx.AsyncClient(base_url=base_url, timeout=httpx.Timeout(120, connect=5))

    async def stream(self, messages: list[dict]) -> AsyncIterator[str]:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "think": False,  # qwen3: skip the hidden reasoning pass, answer directly
            "keep_alive": self.keep_alive,
            "options": {"num_predict": self.max_tokens, "num_ctx": self.num_ctx, "temperature": 0.7},
        }
        async with self.client.stream("POST", "/api/chat", json=payload) as resp:
            if resp.status_code != 200:
                body = (await resp.aread()).decode(errors="replace")
                raise RuntimeError(f"Ollama error {resp.status_code}: {body[:300]}")
            async for line in resp.aiter_lines():
                if not line:
                    continue
                data = json.loads(line)
                if "error" in data:
                    raise RuntimeError(f"Ollama error: {data['error']}")
                token = data.get("message", {}).get("content", "")
                if token:
                    yield token
                if data.get("done"):
                    break

    async def warmup(self):
        """Load the model into memory so the first real reply is fast."""
        async for _ in self.stream([{"role": "user", "content": "Say ok."}]):
            pass

    async def close(self):
        await self.client.aclose()
