from __future__ import annotations

import logging
import time
from typing import Any

import app.config as cfg
from app.dependencies import get_embed_model, get_milvus

logger = logging.getLogger(__name__)
FAQ_COLLECTION = "faq"
TERMS_COLLECTION = "terms"


def _embed(text: str) -> list[float]:
    t0 = time.perf_counter()
    model = get_embed_model()
    vec = model.encode(text, normalize_embeddings=True).tolist()
    logger.info("[RAG] эмбеддинг готов | %.0f ms", (time.perf_counter() - t0) * 1000)
    return vec


def _search(
    collection: str,
    query: str,
    top_k: int,
    output_fields: list[str],
    anns_field: str,
) -> list[dict[str, Any]]:
    t0 = time.perf_counter()
    logger.info("[RAG] поиск в коллекции %s (top_k=%d)...", collection, top_k)
    client = get_milvus()
    try:
        if not client.has_collection(collection):
            logger.info("[RAG] коллекция %s не найдена | %.0f ms", collection, (time.perf_counter() - t0) * 1000)
            return []
    except Exception:
        logger.exception("[RAG] ошибка проверки коллекции %s", collection)
        return []

    vector = _embed(query)
    try:
        results = client.search(
            collection_name=collection,
            data=[vector],
            anns_field=anns_field,
            limit=top_k,
            output_fields=output_fields,
            search_params={"metric_type": "COSINE", "params": {}},
        )
    except Exception:
        logger.exception("[RAG] ошибка поиска в %s", collection)
        return []

    hits = results[0] if results else []
    out = [{"score": h["distance"], **h["entity"]} for h in hits]
    logger.info("[RAG] коллекция %s: %d результатов | %.0f ms", collection, len(out), (time.perf_counter() - t0) * 1000)
    return out


def search_faq(query: str, top_k: int = cfg.FAQ_TOP_K) -> list[dict[str, Any]]:
    """Поиск по FAQ (Database.xlsx): Question + Answer."""
    return _search(
        collection=FAQ_COLLECTION,
        query=query,
        top_k=top_k,
        output_fields=["question", "answer", "question_type"],
        anns_field="question_vector",
    )


def search_terms(query: str, top_k: int = cfg.FAQ_TOP_K) -> list[dict[str, Any]]:
    """Поиск по терминам (Database-2.xlsx): header + text."""
    return _search(
        collection=TERMS_COLLECTION,
        query=query,
        top_k=top_k,
        output_fields=["header", "text"],
        anns_field="header_vector",
    )


def search_all(query: str, top_k: int = cfg.FAQ_TOP_K) -> list[dict[str, Any]]:
    """Объединённый поиск по обеим коллекциям, сортировка по score."""
    faq = search_faq(query, top_k)
    terms = search_terms(query, top_k)
    combined = faq + terms
    combined.sort(key=lambda x: x["score"], reverse=True)
    return combined[:top_k]
