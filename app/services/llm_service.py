from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_DEFAULT_MODEL = "openai/gpt-4o-mini"
_TIMEOUT_SECONDS = 120


class LLMServiceError(Exception):
    pass


def chat(
    messages: list[dict[str, str]],
    model: str | None = None,
    temperature: float = 0.3,
    response_format: dict | None = None,
) -> str:
    t0 = time.perf_counter()
    payload: dict[str, Any] = {
        "model": model or _DEFAULT_MODEL,
        "messages": messages,
        "temperature": temperature,
    }
    if response_format:
        payload["response_format"] = response_format

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("[LLM] OPENROUTER_API_KEY is not set")
        raise LLMServiceError("OPENROUTER_API_KEY is not set")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    logger.info("[LLM] request model=%s", payload["model"])
    try:
        resp = requests.post(_OPENROUTER_URL, headers=headers, json=payload, timeout=_TIMEOUT_SECONDS)
    except requests.exceptions.ReadTimeout as exc:
        logger.error("[LLM] read timeout after %ss: %s", _TIMEOUT_SECONDS, exc)
        raise LLMServiceError("OpenRouter read timeout") from exc
    except requests.exceptions.RequestException as exc:
        logger.error("[LLM] request error: %s", exc)
        raise LLMServiceError("OpenRouter request error") from exc

    if resp.status_code != 200:
        logger.error("[LLM] error status=%s body=%s", resp.status_code, resp.text)
        raise LLMServiceError(f"OpenRouter API error: status={resp.status_code}")

    data = resp.json()
    content = data["choices"][0]["message"]["content"] or ""
    duration_ms = (time.perf_counter() - t0) * 1000
    logger.info("[LLM] response model=%s | %.0f ms", payload["model"], duration_ms)
    usage = data.get("usage")
    if usage:
        logger.info("[LLM] usage=%s", usage)
    return content


def chat_json(messages: list[dict[str, str]], model: str | None = None) -> dict:
    raw = chat(messages, model=model, response_format={"type": "json_object"})
    return json.loads(raw)