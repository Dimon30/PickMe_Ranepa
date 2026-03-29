from __future__ import annotations

import json
import logging
import time
from typing import Any

from groq import Groq

import app.config as cfg

logger = logging.getLogger(__name__)
_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=cfg.GROQ_API_KEY)
    return _client


def chat(
    messages: list[dict[str, str]],
    model: str | None = None,
    temperature: float = 0.3,
    response_format: dict | None = None,
) -> str:
    t0 = time.perf_counter()
    kwargs: dict[str, Any] = {
        "model": model or cfg.GROQ_MODEL,
        "messages": messages,
        "temperature": temperature,
    }
    if response_format:
        kwargs["response_format"] = response_format

    logger.info("[LLM] запрос model=%s", kwargs["model"])
    resp = _get_client().chat.completions.create(**kwargs)
    content = resp.choices[0].message.content or ""
    logger.info("[LLM] ответ model=%s | %.0f ms", kwargs["model"], (time.perf_counter() - t0) * 1000)
    return content


def chat_json(messages: list[dict[str, str]], model: str | None = None) -> dict:
    raw = chat(messages, model=model, response_format={"type": "json_object"})
    return json.loads(raw)
