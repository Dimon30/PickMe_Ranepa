from __future__ import annotations

import logging
import time

from sqlalchemy import text

from app.db.session import SessionLocal
from app.services import llm_service

logger = logging.getLogger(__name__)

# Описание полей для промпта (LLM не пишет SQL — только выбирает значение фильтра)
PROGRAMS_COLUMNS = {
    "program": "название программы (строка)",
    "megacluster": "мегакластер, напр. 'информационные технологии'",
    "institute": "институт, напр. 'иэмит'",
    "major": "направление подготовки",
    "qual": "квалификация: 'бакалавриат' или 'магистратура'",
    "edu_form": "форма обучения: 'очная', 'заочная', 'очная дистанционная'",
    "edu_years": "срок обучения (число)",
    "pass_2024": "проходной балл 2024 (число)",
    "budget_2025": "бюджетных мест 2025 (число)",
    "contract_2025": "платных мест 2025 (число)",
    "cost": "стоимость обучения руб/год (число)",
    "eges_contract": "ЕГЭ для платного поступления (текст)",
    "eges_budget": "ЕГЭ для бюджетного поступления (текст)",
}

COLUMNS_HINT = "\n".join(f"- {k}: {v}" for k, v in PROGRAMS_COLUMNS.items())

FILTER_SYSTEM_PROMPT = f"""Ты — помощник, который извлекает параметры фильтрации из вопроса пользователя.
Таблица programs содержит информацию об образовательных программах РАНХиГС.
Поля таблицы:
{COLUMNS_HINT}

Твоя задача: из вопроса пользователя извлечь:
1. "filter_column" — имя колонки для поиска (одна из перечисленных выше)
2. "filter_value"  — значение для фильтрации (строка или null)

Если вопрос касается конкретной программы — используй filter_column = "program".
Если конкретная программа не упомянута — верни filter_column = null, filter_value = null (вернём все программы).

Отвечай ТОЛЬКО JSON: {{"filter_column": "...", "filter_value": "..."}}
"""

# Базовые шаблоны запросов — LLM заполняет только значение фильтра
_BASE_QUERY = "SELECT * FROM programs LIMIT 10"
_FILTER_QUERY = "SELECT * FROM programs WHERE {col} ILIKE :val"


def _extract_filter(question: str) -> tuple[str | None, str | None]:
    t0 = time.perf_counter()
    messages = [
        {"role": "system", "content": FILTER_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    data = llm_service.chat_json(messages)
    col = data.get("filter_column") or None
    val = data.get("filter_value") or None
    # Допускаем только известные колонки (защита от инъекций)
    if col and col not in PROGRAMS_COLUMNS:
        col = None
        val = None
    logger.info(
        "[SQL] фильтр: col=%s val=%s | %.0f ms",
        col, val, (time.perf_counter() - t0) * 1000,
    )
    return col, val


def _rows_to_text(rows: list[dict]) -> str:
    if not rows:
        return ""
    lines = []
    for row in rows:
        parts = []
        for k, v in row.items():
            if v is not None and str(v).strip():
                parts.append(f"{k}: {v}")
        lines.append("; ".join(parts))
    return "\n".join(lines)


def query_programs(question: str) -> str:
    t0 = time.perf_counter()
    col, val = _extract_filter(question)
    question_l = question.lower()

    numeric_columns = {"pass_2024", "cost", "budget_places", "paid_places"}

    with SessionLocal() as db:
        # --- если есть фильтр ---
        if col and val:

            # --- числовые поля ---
            if col in numeric_columns:
                try:
                    num_val = float(str(val).replace(",", "."))
                except ValueError:
                    logger.warning(
                        "[SQL] не удалось преобразовать число: col=%s val=%s",
                        col, val
                    )
                    return "Не удалось корректно распознать числовое значение."

                # спец логика для проходных баллов
                if col == "pass_2024" and (
                    "набрал" in question_l
                    or "на что могу рассчитывать" in question_l
                    or "с таким баллом" in question_l
                    or "куда могу пройти" in question_l
                ):
                    query = """
                        SELECT * FROM programs
                        WHERE pass_2024 <= :val
                        ORDER BY pass_2024 DESC
                    """
                    params = {"val": num_val}

                else:
                    ALLOWED_COLUMNS = {
                        "program",
                        "faculty",
                        "pass_2024",
                        "cost",
                        "budget_places",
                        "paid_places",
                    }

                    if col not in ALLOWED_COLUMNS:
                        return "Некорректный параметр запроса."
                    query = f"SELECT * FROM programs WHERE {col} = :val"
                    params = {"val": num_val}

            # --- текстовые поля ---
            else:
                query = _FILTER_QUERY.format(col=col)
                params = {"val": f"%{val}%"}

        # --- без фильтра ---
        else:
            query = _BASE_QUERY
            params = {}

        logger.info("[SQL] запрос: %s | params=%s", query, params)

        stmt = text(query)
        t_db = time.perf_counter()
        rows = db.execute(stmt, params).mappings().all()

        logger.info(
            "[SQL] результат: %d строк | %.0f ms",
            len(rows),
            (time.perf_counter() - t_db) * 1000,
        )

    rows_text = _rows_to_text([dict(r) for r in rows])

    if not rows_text:
        return "К сожалению, в базе нет данных по вашему запросу."

    system = (
        "Ты — помощник приёмной комиссии РАНХиГС. "
        "Отвечай кратко, точно и по существу на основе данных из таблицы."
    )

    user_msg = f"Вопрос: {question}\n\nДанные из базы:\n{rows_text}"

    answer = llm_service.chat([
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg}
    ])

    logger.info("[SQL] LLM-ответ сформирован | %.0f ms", (time.perf_counter() - t0) * 1000)

    return answer