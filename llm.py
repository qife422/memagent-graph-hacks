"""
LLM interface — one client, three backends, all OpenAI-compatible.

Backends (choose with the LLM_BACKEND env var, or let it auto-detect):

  openai      OpenAI cloud. Needs an API key. This is the original default
              and its behaviour is unchanged.
  compatible  Any OpenAI-compatible endpoint (OpenRouter, DeepSeek, a
              company gateway, ...). Set LLM_BASE_URL + LLM_API_KEY.
  ollama      A local Ollama server — free, private, no key needed.
              Defaults to http://localhost:11434/v1 and a tool-capable
              model (qwen2.5:7b). Install from https://ollama.com, then
              `ollama pull qwen2.5:7b`.

Auto-detect (no LLM_BACKEND set), in order:
  1. An API key (LLM_API_KEY / OPENAI_API_KEY) is present
     -> "compatible" if LLM_BASE_URL is also set, otherwise "openai".
  2. LLM_BASE_URL points at localhost -> "ollama".
  3. A local Ollama answers on http://localhost:11434 -> "ollama".
  4. Otherwise "openai" with no key: `available` is False and callers
     fall back to rule-based logic, exactly as before.

Configuration env vars:
  LLM_BACKEND   openai | compatible | ollama   (optional, see above)
  LLM_MODEL     model name override for any backend
  LLM_BASE_URL  endpoint override for any backend
  LLM_API_KEY   key for openai / compatible (OPENAI_API_KEY also works)

Explicit constructor arguments always win over env vars, and the old
positional call LLM(api_key, base_url, model) keeps working.
"""

import json
import os
import urllib.error
import urllib.request

OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENAI_DEFAULT_MODEL = "gpt-4o-mini"
OLLAMA_BASE_URL = "http://localhost:11434/v1"
# Local models must support tool calling for the Scout/Advisor tool loop;
# qwen2.5:7b and llama3.1:8b both do. Smaller/older models may not.
OLLAMA_DEFAULT_MODEL = "qwen2.5:7b"


def _is_localhost(url: str) -> bool:
    return "://localhost" in url or "://127.0.0.1" in url


def _ollama_reachable(base_url: str, timeout: float = 1.0) -> bool:
    """True if an OpenAI-compatible server answers GET /models there."""
    try:
        req = urllib.request.Request(f"{base_url.rstrip('/')}/models")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


class LLM:
    def __init__(self, api_key: str | None = None,
                 base_url: str | None = None,
                 model: str | None = None,
                 backend: str | None = None):
        env = os.environ.get
        self.api_key = (api_key or env("LLM_API_KEY")
                        or env("OPENAI_API_KEY", ""))
        base_url = base_url or env("LLM_BASE_URL") or None
        model = model or env("LLM_MODEL") or None
        backend = (backend or env("LLM_BACKEND") or "").strip().lower() or None

        if backend is None:
            backend = self._detect_backend(base_url)
        self.backend = backend

        if backend == "ollama":
            self.base_url = (base_url or OLLAMA_BASE_URL).rstrip("/")
            self.model = model or OLLAMA_DEFAULT_MODEL
            # Ollama ignores the key, but the Authorization header is
            # always sent, so use a placeholder when none was given.
            self.api_key = self.api_key or "ollama"
        elif backend == "compatible":
            self.base_url = (base_url or OPENAI_BASE_URL).rstrip("/")
            self.model = model or OPENAI_DEFAULT_MODEL
        else:  # openai
            self.backend = "openai"
            self.base_url = (base_url or OPENAI_BASE_URL).rstrip("/")
            self.model = model or OPENAI_DEFAULT_MODEL

        # Local inference can be slow to first token; give it more room.
        self.timeout = 180 if self.backend == "ollama" else 60
        self._reachable: bool | None = None  # cached Ollama probe result

    def _detect_backend(self, base_url: str | None) -> str:
        if self.api_key:
            return "compatible" if base_url else "openai"
        if base_url and _is_localhost(base_url):
            return "ollama"
        if _ollama_reachable(OLLAMA_BASE_URL):
            return "ollama"
        return "openai"

    @property
    def available(self) -> bool:
        if self.backend == "ollama":
            if self._reachable is None:
                self._reachable = _ollama_reachable(self.base_url)
            return self._reachable
        return bool(self.api_key)

    def describe(self) -> str:
        """One-line summary for startup logs, e.g. 'ollama:qwen2.5:7b@http://localhost:11434/v1'."""
        return f"{self.backend}:{self.model}@{self.base_url}"

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             tool_choice: str = "auto") -> dict:
        """One chat completion. Returns the raw message dict (may contain tool_calls)."""
        if not self.available:
            if self.backend == "ollama":
                raise RuntimeError(
                    f"Ollama is not reachable at {self.base_url} — "
                    "start it with `ollama serve` and pull the model with "
                    f"`ollama pull {self.model}`")
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
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read())
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Cannot reach the {self.backend} backend at "
                f"{self.base_url}: {e}") from e
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
