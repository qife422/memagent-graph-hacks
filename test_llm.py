"""Offline tests for the multi-backend llm.py.

Spins up a stub OpenAI-compatible HTTP server (no real network calls)
and checks config resolution, availability, chat and extract_json for
each backend, plus backward compatibility with the old client.
"""
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FAILURES = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


class Stub(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.endswith("/models"):
            self._send({"data": [{"id": "qwen2.5:7b"}]})
        else:
            self._send({}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        msgs = body.get("messages", [])
        if msgs and "Return ONLY valid JSON" in msgs[0].get("content", ""):
            content = '{"ok": true}'
        elif body.get("tools"):
            content = "tool-call path ok"
        else:
            content = "hello from stub"
        self._send({"choices": [{"message": {"role": "assistant", "content": content}}]})

    def _send(self, obj, status=200):
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def serve(port):
    srv = HTTPServer(("127.0.0.1", port), Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


ENV_KEYS = ["OPENAI_API_KEY", "LLM_API_KEY", "LLM_BACKEND", "LLM_MODEL", "LLM_BASE_URL"]


def clean_env(**overrides):
    for k in ENV_KEYS:
        os.environ.pop(k, None)
    os.environ.update(overrides)


def main():
    stub = serve(18099)
    stub_url = "http://127.0.0.1:18099/v1"
    from llm import LLM

    # 1. OpenAI default is unchanged when a key exists (original behaviour)
    clean_env(OPENAI_API_KEY="sk-test")
    llm = LLM()
    check("openai: backend", llm.backend == "openai", llm.backend)
    check("openai: base_url", llm.base_url == "https://api.openai.com/v1", llm.base_url)
    check("openai: model", llm.model == "gpt-4o-mini", llm.model)
    check("openai: available with key", llm.available is True)

    # 2. No key and no Ollama anywhere -> unavailable, chat raises (old fallback preserved)
    clean_env()
    llm = LLM(base_url="http://127.0.0.1:9/v1")  # nothing listening there
    check("nokey: available False", llm.available is False)
    try:
        llm.chat([{"role": "user", "content": "hi"}])
        check("nokey: chat raises", False)
    except RuntimeError:
        check("nokey: chat raises", True)

    # 3. Explicit Ollama backend against the stub: no key needed at all
    clean_env(LLM_BACKEND="ollama", LLM_BASE_URL=stub_url)
    llm = LLM()
    check("ollama: backend", llm.backend == "ollama", llm.backend)
    check("ollama: default model", llm.model == "qwen2.5:7b", llm.model)
    check("ollama: available without key", llm.available is True)
    msg = llm.chat([{"role": "user", "content": "hi"}])
    check("ollama: chat works", msg.get("content") == "hello from stub", str(msg))
    msg = llm.chat([{"role": "user", "content": "hi"}], tools=[{"type": "function"}])
    check("ollama: chat with tools works", msg.get("content") == "tool-call path ok", str(msg))
    out = llm.extract_json("sys", "usr")
    check("ollama: extract_json works", out == {"ok": True}, str(out))

    # 4. Auto-detect: no key, LLM_BASE_URL points at localhost -> ollama
    clean_env(LLM_BASE_URL=stub_url)
    llm = LLM()
    check("autodetect localhost: backend ollama", llm.backend == "ollama", llm.backend)
    check("autodetect localhost: available", llm.available is True)

    # 5. Auto-detect: no config at all, Ollama on the default port -> ollama
    stub2 = serve(11434)
    clean_env()
    llm = LLM()
    check("autodetect default port: backend ollama", llm.backend == "ollama", llm.backend)
    check("autodetect default port: base", llm.base_url == "http://localhost:11434/v1", llm.base_url)
    stub2.shutdown()

    # 6. Compatible backend: key + custom base URL
    clean_env(LLM_API_KEY="sk-other", LLM_BASE_URL=stub_url, LLM_MODEL="some-model")
    llm = LLM()
    check("compatible: backend", llm.backend == "compatible", llm.backend)
    check("compatible: model override", llm.model == "some-model", llm.model)
    check("compatible: available", llm.available is True)
    msg = llm.chat([{"role": "user", "content": "hi"}])
    check("compatible: chat works", msg.get("content") == "hello from stub", str(msg))

    # 7. Backward compat: old positional constructor
    clean_env()
    llm = LLM("sk-x", "https://example.com/v1", "m1")
    check("legacy ctor: fields", (llm.api_key, llm.base_url, llm.model) == ("sk-x", "https://example.com/v1", "m1"),
          llm.describe())
    check("legacy ctor: available", llm.available is True)
    # a custom base_url is by definition a "compatible" endpoint, not api.openai.com
    check("describe() format", llm.describe() == "compatible:m1@https://example.com/v1", llm.describe())
    llm = LLM("sk-x")
    check("legacy ctor key-only: openai", llm.backend == "openai" and llm.model == "gpt-4o-mini", llm.describe())

    # 8. Ollama configured but server down -> available False, helpful error
    clean_env(LLM_BACKEND="ollama", LLM_BASE_URL="http://127.0.0.1:9/v1")
    llm = LLM()
    check("ollama down: available False", llm.available is False)
    try:
        llm.chat([{"role": "user", "content": "hi"}])
        check("ollama down: helpful error", False)
    except RuntimeError as e:
        check("ollama down: helpful error", "ollama pull" in str(e), str(e))

    # 9. Compatible backend on localhost without a key (llama.cpp & co):
    #    placeholder key, long timeout, chat works against the stub
    clean_env(LLM_BACKEND="compatible", LLM_BASE_URL=stub_url, LLM_MODEL="qwen4b")
    llm = LLM()
    check("compatible-local: backend", llm.backend == "compatible", llm.backend)
    check("compatible-local: model", llm.model == "qwen4b", llm.model)
    check("compatible-local: available without key", llm.available is True)
    check("compatible-local: placeholder key", llm.api_key == "local", llm.api_key)
    check("compatible-local: timeout 180", llm.timeout == 180, str(llm.timeout))
    msg = llm.chat([{"role": "user", "content": "hi"}])
    check("compatible-local: chat works", msg.get("content") == "hello from stub", str(msg))

    # 10. Compatible on a remote host still requires a key, keeps 60s timeout
    clean_env(LLM_BACKEND="compatible", LLM_BASE_URL="https://example.com/v1")
    llm = LLM()
    check("compatible-remote: available False without key", llm.available is False)
    check("compatible-remote: timeout 60", llm.timeout == 60, str(llm.timeout))

    # 11. Compatible on localhost WITH a real key keeps that key
    clean_env(LLM_BACKEND="compatible", LLM_BASE_URL=stub_url, LLM_API_KEY="sk-real")
    llm = LLM()
    check("compatible-local: real key kept", llm.api_key == "sk-real", llm.api_key)

    stub.shutdown()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURES: {FAILURES}")
        sys.exit(1)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
