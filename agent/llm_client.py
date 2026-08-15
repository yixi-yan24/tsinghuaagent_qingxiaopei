import json, os, re, time
from collections.abc import Generator
from typing import Any

# Conda 有时会设置指向不存在的 SSL_CERT_FILE，导致 httpx 启动失败
if "SSL_CERT_FILE" in os.environ and not os.path.isfile(os.environ["SSL_CERT_FILE"]):
    del os.environ["SSL_CERT_FILE"]

import httpx

# ── Errors that are worth retrying (transient) vs. those that aren't ────
_RETRYABLE = (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError)
_FATAL = (KeyError, IndexError, TypeError, ValueError)


def chat_completion(
    api_key: str,
    base_url: str,
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.3,
    max_tokens: int = 4096,
    timeout: float = 90,
    retries: int = 1,
    model: str = "deepseek-chat",
) -> str:
    """Call an OpenAI-compatible chat endpoint with one transient-failure retry."""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.post(
                    f"{base_url.rstrip('/')}/chat/completions",
                    headers=headers, json=payload,
                )
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise RuntimeError("上游模型返回了空回答")
            return content
        except _RETRYABLE:
            if attempt == retries:
                raise
            time.sleep(0.5 * (attempt + 1))
        except _FATAL as exc:
            # Structural / logic errors — retrying won't help.
            last_error = exc
            break

    raise RuntimeError("模型调用失败") from last_error


def chat_completion_stream(
    api_key: str,
    base_url: str,
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.3,
    max_tokens: int = 4096,
    timeout: float = 90,
    model: str = "deepseek-chat",
) -> Generator[str, None, None]:
    """Stream tokens from an OpenAI-compatible chat endpoint.

    Yields content strings as they arrive over SSE.  Does NOT retry —
    streaming requests are handled by the caller's own retry logic.
    """
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }

    with httpx.Client(timeout=timeout) as client:
        with client.stream(
            "POST",
            f"{base_url.rstrip('/')}/chat/completions",
            headers=headers,
            json=payload,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    delta = chunk["choices"][0].get("delta", {})
                    token = delta.get("content", "")
                    if token:
                        yield token
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue


def chat_completion_with_tools(
    api_key: str,
    base_url: str,
    messages: list[dict],
    tools: list[dict],
    *,
    temperature: float = 0.3,
    max_tokens: int = 4096,
    timeout: float = 90,
    retries: int = 1,
    model: str = "deepseek-chat",
) -> tuple[str, list[dict]]:
    """Call an OpenAI-compatible chat endpoint with native function calling.

    Returns ``(content, tool_calls)`` where ``tool_calls`` is a list of
    ``{"id", "name", "arguments"(dict)}``.  Empty list means the model gave a
    direct answer (in *content*).
    """
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "tools": tools,
    }

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.post(
                    f"{base_url.rstrip('/')}/chat/completions",
                    headers=headers, json=payload,
                )
            response.raise_for_status()
            data = response.json()
            msg = data["choices"][0]["message"]
            content = msg.get("content") or ""
            tool_calls: list[dict] = []
            for tc in (msg.get("tool_calls") or []):
                fn = tc.get("function", {})
                args_raw = fn.get("arguments") or "{}"
                try:
                    args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append({
                    "id": tc.get("id", ""),
                    "name": fn.get("name", ""),
                    "arguments": args if isinstance(args, dict) else {},
                })
            return content, tool_calls
        except _RETRYABLE:
            if attempt == retries:
                raise
            time.sleep(0.5 * (attempt + 1))
        except _FATAL as exc:
            last_error = exc
            break

    raise RuntimeError("模型调用失败") from last_error


def chat_completion_stream_with_tools(
    api_key: str,
    base_url: str,
    messages: list[dict],
    tools: list[dict],
    *,
    temperature: float = 0.3,
    max_tokens: int = 4096,
    timeout: float = 90,
    model: str = "deepseek-chat",
) -> Generator[dict, None, None]:
    """Stream an OpenAI-compatible chat endpoint with native function calling.

    Yields event dicts:
      {"type": "content", "text": str}              — answer token
      {"type": "tool_call", "index", "id", "name", "arguments"} — incremental
        tool-call fragment (``arguments`` is a partial string; concatenate
        fragments per ``index`` to rebuild the JSON).
    """
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "tools": tools,
        "stream": True,
    }

    with httpx.Client(timeout=timeout) as client:
        with client.stream(
            "POST",
            f"{base_url.rstrip('/')}/chat/completions",
            headers=headers,
            json=payload,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    delta = chunk["choices"][0].get("delta", {})
                    token = delta.get("content", "")
                    if token:
                        yield {"type": "content", "text": token}
                    for tc in (delta.get("tool_calls") or []):
                        fn = tc.get("function") or {}
                        yield {
                            "type": "tool_call",
                            "index": tc.get("index", 0),
                            "id": tc.get("id") or "",
                            "name": fn.get("name") or "",
                            "arguments": fn.get("arguments") or "",
                        }
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
