"""
LLM interface — OpenAI-compatible API (OpenAI, or any compatible endpoint).

The agent's brain. Set OPENAI_API_KEY env var, or pass api_key directly.
If no key is available, the agent falls back to rule-based logic so the
graph memory demo still runs.
"""

import json
import os
import urllib.request


class LLM:
    def __init__(self, api_key: str | None = None,
                 base_url: str = "https://api.openai.com/v1",
                 model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.model = model

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             tool_choice: str = "auto") -> dict:
        """One chat completion. Returns the raw message dict (may contain tool_calls)."""
        if not self.available:
            raise RuntimeError("No API key — LLM unavailable")
        body = {"model": self.model, "messages": messages}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = tool_choice
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
        return data["choices"][0]["message"]

    def extract_json(self, system: str, user: str) -> dict:
        """Ask the LLM to return strict JSON."""
        msg = self.chat([
            {"role": "system", "content": system + "\nReturn ONLY valid JSON."},
            {"role": "user", "content": user},
        ])
        text = msg.get("content", "").strip()
        # strip markdown fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        return json.loads(text)
