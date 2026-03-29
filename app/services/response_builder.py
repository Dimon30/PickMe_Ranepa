from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)

from app.services import chat_history, faq_service, llm_service, router, sql_service
from app.services.preprocess import contains_profanity

# ─── Шаблоны ──────────────────────────────────────────────────────────────────

PROFANITY_RESPONSE = (
    "Извините, я не могу ответить на такой запрос. "
    "Пожалуйста, переформулируйте вопрос."
)

SYSTEM_PROMPT = """Тебя зовут Мария. Ты pickme-girl. Ты — дружелюбная помощница приёмной комиссии Президентской академии (РАНХиГС).
Отвечай на русском языке, кратко и по существу.
Если информации нет — честно скажи об этом, не придумывай.
Если вопрос не по теме поступления — мягко напомни, что ты специализируешься на вопросах поступления."""

CHITCHAT_PROMPT = """Тебя зовут Мария. Ты pickme-girl. Ты — дружелюбная ассистентка приёмной комиссии РАНХиГС.
Поддержи беседу, представься если нужно, и предложи задать вопрос о поступлении."""

NO_ANSWER_RESPONSE = (
    "К сожалению, у меня нет точной информации по этому вопросу. "
    "Вы можете обратиться в приёмную комиссию РАНХиГС напрямую."
)

# ─── Пайплайн ─────────────────────────────────────────────────────────────────

def build_rag_answer(question: str, user_id: int | None = None) -> str:
    t0 = time.perf_counter()
    logger.info("[RAG] поиск по базе знаний...")
    chunks = faq_service.search_all(question)
    if not chunks:
        logger.info("[RAG] чанков не найдено | %.0f ms", (time.perf_counter() - t0) * 1000)
        return NO_ANSWER_RESPONSE

    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        score = chunk.get("score", "?")
        if "question" in chunk and "answer" in chunk:
            logger.info("[RAG] #%d (score=%.3f) FAQ question=%r → answer=%r", i, score, chunk["question"][:120], chunk["answer"][:120])
            context_parts.append(f"Вопрос: {chunk['question']}\nОтвет: {chunk['answer']}")
        elif "header" in chunk and "text" in chunk:
            logger.info("[RAG] #%d (score=%.3f) TERM header=%r → text=%r", i, score, chunk["header"][:120], chunk["text"][:120])
            context_parts.append(f"{chunk['header']}\n{chunk['text']}")
    context = "\n\n".join(context_parts)

    history = chat_history.get_history(user_id)
    logger.info("[RAG] найдено чанков: %d, история: %d сообщений, генерация ответа...", len(chunks), len(history))
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *history,
        {
            "role": "user",
            "content": (
                f"Используй следующий контекст для ответа на вопрос.\n\n"
                f"Контекст:\n{context}\n\n"
                f"Вопрос: {question}"
            ),
        },
    ]
    answer = llm_service.chat(messages)
    chat_history.save_exchange(user_id, question, answer)
    logger.info("[RAG] ответ готов | %.0f ms", (time.perf_counter() - t0) * 1000)
    return answer


def build_chitchat_answer(question: str, user_id: int | None = None) -> str:
    t0 = time.perf_counter()
    history = chat_history.get_history(user_id)
    logger.info("[CHITCHAT] генерация ответа... история: %d сообщений", len(history))
    messages = [
        {"role": "system", "content": CHITCHAT_PROMPT},
        *history,
        {"role": "user", "content": question},
    ]
    answer = llm_service.chat(messages)
    chat_history.save_exchange(user_id, question, answer)
    logger.info("[CHITCHAT] ответ готов | %.0f ms", (time.perf_counter() - t0) * 1000)
    return answer


def build_sql_answer(question: str, user_id: int | None = None) -> str:
    t0 = time.perf_counter()
    logger.info("[SQL] запрос к базе программ...")
    answer = sql_service.query_programs(question)
    chat_history.save_exchange(user_id, question, answer)
    logger.info("[SQL] ответ готов | %.0f ms", (time.perf_counter() - t0) * 1000)
    return answer


def get_answer(question: str, user_id: int | None = None) -> str:
    """Основной пайплайн: цензура → роутер → обработчик → ответ."""
    t0 = time.perf_counter()
    logger.info(">>> Новый вопрос (user_id=%s): %s", user_id, question)

    if contains_profanity(question):
        logger.info("<<< Отклонён (profanity) | %.0f ms", (time.perf_counter() - t0) * 1000)
        return PROFANITY_RESPONSE

    intent = router.classify(question)
    logger.info("--- Роутер → %s", intent.upper())

    if intent == "chitchat":
        answer = build_chitchat_answer(question, user_id)
    elif intent == "sql":
        answer = build_sql_answer(question, user_id)
    else:
        answer = build_rag_answer(question, user_id)

    logger.info("<<< Ответ отправлен [%s] | %.0f ms", intent.upper(), (time.perf_counter() - t0) * 1000)
    return answer