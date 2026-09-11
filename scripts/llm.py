#!/usr/bin/env python3
"""Minimal OpenAI-compatible chat client — standard library only.

Talks to anything that speaks the OpenAI ``/v1/chat/completions`` shape:
api.openai.com, Azure-style gateways, OpenRouter, Together, Groq, Fireworks,
LiteLLM, vLLM, llama.cpp, LM Studio, Ollama's /v1 shim, ...

Configuration comes from the environment; the dashboard may override any of it
per request (the basic, server-less build has no environment to read):

    OPENAI_BASE_URL   default https://api.openai.com/v1
    OPENAI_API_KEY    bearer token — leave blank for local servers
    OPENAI_MODEL      default gpt-4o-mini

Nothing here imports numpy or the sentence-transformers stack, so the analysis
path stays usable even when the vector index is absent.
"""

import json
import os
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_BASE_URL = (os.environ.get("OPENAI_BASE_URL")
                    or os.environ.get("OPENAI_API_BASE")
                    or "https://api.openai.com/v1")
DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_API_KEY = os.environ.get("OPENAI_API_KEY", "")
TIMEOUT = int(os.environ.get("OPENAI_TIMEOUT", "300"))


class LLMError(RuntimeError):
    """An endpoint refused the request; the message is safe to show a user."""


def defaults():
    """What the server would use if the browser sends no overrides."""
    return {"base_url": DEFAULT_BASE_URL, "model": DEFAULT_MODEL,
            "has_key": bool(DEFAULT_API_KEY)}


def endpoint(base_url=None):
    b = (base_url or DEFAULT_BASE_URL).strip().rstrip("/")
    if b.endswith("/chat/completions"):
        return b
    return b + "/chat/completions"


_LOOPBACK = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}
_direct_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _opener(url):
    """A local model server must not be sent through an http_proxy set for the
    outside world — the usual reason a working LM Studio/Ollama endpoint 502s."""
    host = (urllib.parse.urlparse(url).hostname or "").lower()
    return _direct_opener if host in _LOOPBACK else urllib.request.build_opener()


def _request(url, payload, api_key, timeout):
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        return _opener(url).open(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8", "replace")[:600]
        except Exception:
            pass
        raise LLMError(f"{e.code} {e.reason} from {url}"
                       + (f" — {detail}" if detail else "")) from None
    except urllib.error.URLError as e:
        raise LLMError(f"could not reach {url} — {e.reason}") from None


def _payload(messages, model, temperature, max_tokens, stream, extra):
    p = {"model": model or DEFAULT_MODEL, "messages": messages, "stream": stream}
    if temperature is not None:
        p["temperature"] = temperature
    if max_tokens:
        p["max_tokens"] = max_tokens
    if extra:
        p.update(extra)
    return p


# Endpoints disagree on two parameters: newer OpenAI reasoning models reject
# max_tokens and a non-default temperature, while older and self-hosted ones
# require max_tokens. Send the widely-understood spelling, then adapt on 400.
def _adapt(payload, message):
    """Rewrite a payload a 400 complained about. Returns True if worth retrying."""
    m = message.lower()
    if "max_tokens" in m and "max_completion_tokens" in m and "max_tokens" in payload:
        payload["max_completion_tokens"] = payload.pop("max_tokens")
        return True
    if "temperature" in m and "temperature" in payload:
        payload.pop("temperature")
        return True
    if "response_format" in m and "response_format" in payload:
        payload.pop("response_format")
        return True
    return False


def complete(messages, *, base_url=None, api_key=None, model=None,
             temperature=0.1, max_tokens=2600, timeout=TIMEOUT, extra=None):
    """Blocking call. Returns {"text", "model", "usage"}."""
    url = endpoint(base_url)
    key = DEFAULT_API_KEY if api_key is None else api_key
    payload = _payload(messages, model, temperature, max_tokens, False, extra)
    for attempt in range(4):
        try:
            with _request(url, payload, key, timeout) as resp:
                data = json.loads(resp.read().decode("utf-8", "replace"))
            break
        except LLMError as e:
            if attempt < 3 and str(e).startswith("400") and _adapt(payload, str(e)):
                continue
            raise
    choice = (data.get("choices") or [{}])[0]
    text = (choice.get("message") or {}).get("content") or ""
    return {"text": text, "model": data.get("model", payload["model"]),
            "usage": data.get("usage") or {},
            "finish_reason": choice.get("finish_reason")}


def stream(messages, *, base_url=None, api_key=None, model=None,
           temperature=0.1, max_tokens=2600, timeout=TIMEOUT, extra=None):
    """Generator of content deltas (str). Falls back to one blocking call if the
    endpoint ignores stream=true."""
    url = endpoint(base_url)
    key = DEFAULT_API_KEY if api_key is None else api_key
    payload = _payload(messages, model, temperature, max_tokens, True, extra)
    resp = None
    for attempt in range(4):
        try:
            resp = _request(url, payload, key, timeout)
            break
        except LLMError as e:
            if attempt < 3 and str(e).startswith("400") and _adapt(payload, str(e)):
                continue
            raise
    saw_any = False
    with resp:
        ctype = resp.headers.get("Content-Type", "")
        if "text/event-stream" not in ctype:
            data = json.loads(resp.read().decode("utf-8", "replace"))
            txt = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
            if txt:
                yield txt
            return
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            chunk = line[5:].strip()
            if chunk == "[DONE]":
                break
            try:
                obj = json.loads(chunk)
            except json.JSONDecodeError:
                continue
            for ch in obj.get("choices") or []:
                piece = (ch.get("delta") or {}).get("content")
                if piece:
                    saw_any = True
                    yield piece
    if not saw_any:
        # Some gateways accept stream=true and answer with an empty stream.
        out = complete(messages, base_url=base_url, api_key=api_key, model=model,
                       temperature=temperature, max_tokens=max_tokens,
                       timeout=timeout, extra=extra)
        if out["text"]:
            yield out["text"]
