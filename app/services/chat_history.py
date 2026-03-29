from __future__ import annotations

import json
import logging

import app.config as cfg
from app.dependencies import get_redis

logger = logging.getLogger(__name__)

_KEY_PREFIX = "chat:"


def _key(user_id: int) -> str:
    return f"{_KEY_PREFIX}{user_id}"


def get_history(user_id: int | None) -> list[dict[str, str]]:
    if user_id is None:
        return []
    try:
        r = get_redis()
        raw_items = r.lrange(_key(user_id), 0, -1)
        return [json.loads(item) for item in raw_items]
    except Exception:
        logger.exception("[HISTORY] ошибка чтения истории user_id=%s", user_id)
        return []


def save_exchange(user_id: int | None, question: str, answer: str) -> None:
    if user_id is None:
        return
    try:
        r = get_redis()
        key = _key(user_id)
        r.rpush(key, json.dumps({"role": "user", "content": question}, ensure_ascii=False))
        r.rpush(key, json.dumps({"role": "assistant", "content": answer}, ensure_ascii=False))
        max_items = cfg.HISTORY_MAX_MESSAGES * 2
        r.ltrim(key, -max_items, -1)
        r.expire(key, cfg.HISTORY_TTL)
    except Exception:
        logger.exception("[HISTORY] ошибка сохранения истории user_id=%s", user_id)
