from __future__ import annotations

import logging
import time

import redis
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

import app.config as cfg

logger = logging.getLogger(__name__)

_embed_model: SentenceTransformer | None = None
_milvus_client: MilvusClient | None = None
_redis_client: redis.Redis | None = None


def get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        t0 = time.perf_counter()
        logger.info("embed.model_load_start name=%s", cfg.EMBED_MODEL)
        _embed_model = SentenceTransformer(cfg.EMBED_MODEL)
        logger.info("embed.model_load_ok elapsed_ms=%.1f", (time.perf_counter() - t0) * 1000)
    return _embed_model


def get_milvus() -> MilvusClient:
    global _milvus_client
    if _milvus_client is None:
        t0 = time.perf_counter()
        logger.info("milvus.client_init_start uri=%s", cfg.MILVUS_URI)
        _milvus_client = MilvusClient(uri=cfg.MILVUS_URI)
        logger.info("milvus.client_init_ok elapsed_ms=%.1f", (time.perf_counter() - t0) * 1000)
    return _milvus_client


def get_redis() -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        t0 = time.perf_counter()
        logger.info("redis.client_init_start url=%s", cfg.REDIS_URL)
        _redis_client = redis.Redis.from_url(cfg.REDIS_URL, decode_responses=True)
        _redis_client.ping()
        logger.info("redis.client_init_ok elapsed_ms=%.1f", (time.perf_counter() - t0) * 1000)
    return _redis_client


def warmup() -> None:
    """Загрузить модель и подключиться к Milvus/Redis при старте приложения."""
    get_embed_model()
    get_milvus()
    get_redis()
