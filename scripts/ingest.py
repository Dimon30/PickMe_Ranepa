"""
Скрипт загрузки данных:
  - all_program.xlsx  → PostgreSQL (таблица programs)
  - Database.xlsx     → Milvus (коллекция faq)
  - Database-2.xlsx   → Milvus (коллекция terms)

Запуск:
  python -m scripts.ingest
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
import re

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import openpyxl
from pymilvus import MilvusClient, DataType
from sentence_transformers import SentenceTransformer
from sqlalchemy import text

import app.config as cfg
from app.db.models import Base, Program
from app.db.session import engine, SessionLocal

DATA_DIR = ROOT / "data"

# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_embed_model() -> SentenceTransformer:
    print(f"Loading embedding model: {cfg.EMBED_MODEL}")
    return SentenceTransformer(cfg.EMBED_MODEL)


def embed_batch(model: SentenceTransformer, texts: list[str]) -> list[list[float]]:
    vecs = model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
    return vecs.tolist()


def read_xlsx(path: Path) -> tuple[list[str], list[list]]:
    wb = openpyxl.load_workbook(path)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    headers = [str(h).lower().strip() if h else f"col_{i}" for i, h in enumerate(rows[0])]
    data = [list(row) for row in rows[1:] if any(v is not None for v in row)]
    return headers, data


def safe_str(val) -> str:
    if val is None:
        return ""
    return str(val).strip()


def _to_number(val, *, is_int: bool):
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return int(val) if is_int else float(val)
    if isinstance(val, str):
        s = val.strip().lower()
        if not s:
            return None
        m = re.search(r"-?\d+(?:[.,]\d+)?", s)
        if not m:
            return None
        try:
            num = float(m.group(0).replace(",", "."))
        except ValueError:
            return None
        return int(num) if is_int else num
    return None


# ─── PostgreSQL: programs ──────────────────────────────────────────────────────

def ingest_programs():
    print("\n=== Загрузка all_program.xlsx → PostgreSQL ===")
    Base.metadata.create_all(engine)

    headers, rows = read_xlsx(DATA_DIR / "all_program.xlsx")
    print(f"  Найдено строк: {len(rows)}")

    program_fields = [c.key for c in Program.__table__.columns if c.key != "id"]
    columns = {c.key: c for c in Program.__table__.columns}

    with SessionLocal() as db:
        db.execute(text("TRUNCATE TABLE programs RESTART IDENTITY"))
        db.commit()

        objects = []
        for row in rows:
            row_dict = {headers[i]: row[i] for i in range(min(len(headers), len(row)))}
            cleaned = {}
            for f in program_fields:
                val = row_dict.get(f)
                col = columns.get(f)
                if col is not None and col.type is not None:
                    if col.type.__class__.__name__ == "Integer":
                        val = _to_number(val, is_int=True)
                    elif col.type.__class__.__name__ == "Float":
                        val = _to_number(val, is_int=False)
                cleaned[f] = val
            obj = Program(**cleaned)
            objects.append(obj)

        db.add_all(objects)
        db.commit()

    print(f"  Загружено: {len(objects)} записей")


# ─── Milvus helpers ────────────────────────────────────────────────────────────

def ensure_collection(
    client: MilvusClient,
    name: str,
    fields_def: list[dict],
    vector_field: str = "question_vector",
) -> None:
    if client.has_collection(name):
        client.drop_collection(name)

    schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
    for fd in fields_def:
        if fd["type"] == "varchar":
            schema.add_field(fd["name"], DataType.VARCHAR, max_length=fd.get("max_length", 65535))
        elif fd["type"] == "vector":
            schema.add_field(fd["name"], DataType.FLOAT_VECTOR, dim=cfg.EMBED_DIM)

    index_params = client.prepare_index_params()
    # индексируем все векторные поля
    for fd in fields_def:
        if fd["type"] == "vector":
            index_params.add_index(
                field_name=fd["name"],
                index_type="IVF_FLAT",
                metric_type="COSINE",
                params={"nlist": 128},
            )

    client.create_collection(collection_name=name, schema=schema, index_params=index_params)
    print(f"  Коллекция '{name}' создана")


# ─── Milvus: FAQ ──────────────────────────────────────────────────────────────

def ingest_faq(model: SentenceTransformer, client: MilvusClient):
    print("\n=== Загрузка Database.xlsx → Milvus (faq) ===")
    headers, rows = read_xlsx(DATA_DIR / "Database.xlsx")
    print(f"  Найдено строк: {len(rows)}")

    # Колонки: №, Question, Answer, Question type
    q_idx = next((i for i, h in enumerate(headers) if "question" in h and "type" not in h), 1)
    a_idx = next((i for i, h in enumerate(headers) if "answer" in h), 2)
    t_idx = next((i for i, h in enumerate(headers) if "type" in h), 3)

    questions = [safe_str(r[q_idx]) for r in rows]
    answers = [safe_str(r[a_idx]) for r in rows]
    types = [safe_str(r[t_idx]) for r in rows]

    fields_def = [
        {"name": "question", "type": "varchar"},
        {"name": "answer", "type": "varchar"},
        {"name": "question_type", "type": "varchar"},
        {"name": "question_vector", "type": "vector"},
        {"name": "answer_vector", "type": "vector"},
    ]
    ensure_collection(client, "faq", fields_def, vector_field="question_vector")

    print("  Векторизация вопросов...")
    q_vecs = embed_batch(model, questions)
    print("  Векторизация ответов...")
    a_vecs = embed_batch(model, answers)

    data = [
        {
            "question": q,
            "answer": a,
            "question_type": t,
            "question_vector": qv,
            "answer_vector": av,
        }
        for q, a, t, qv, av in zip(questions, answers, types, q_vecs, a_vecs)
    ]

    client.insert(collection_name="faq", data=data)
    print(f"  Вставлено: {len(data)} записей")


# ─── Milvus: Terms ────────────────────────────────────────────────────────────

def ingest_terms(model: SentenceTransformer, client: MilvusClient):
    print("\n=== Загрузка Database-2.xlsx → Milvus (terms) ===")
    headers, rows = read_xlsx(DATA_DIR / "Database-2.xlsx")
    print(f"  Найдено строк: {len(rows)}")

    h_idx = next((i for i, h in enumerate(headers) if "header" in h), 0)
    t_idx = next((i for i, h in enumerate(headers) if "text" in h), 1)

    header_vals = [safe_str(r[h_idx]) for r in rows]
    text_vals = [safe_str(r[t_idx]) for r in rows]

    fields_def = [
        {"name": "header", "type": "varchar"},
        {"name": "text", "type": "varchar"},
        {"name": "header_vector", "type": "vector"},
        {"name": "text_vector", "type": "vector"},
    ]
    ensure_collection(client, "terms", fields_def, vector_field="header_vector")

    print("  Векторизация заголовков...")
    h_vecs = embed_batch(model, header_vals)
    print("  Векторизация текстов...")
    t_vecs = embed_batch(model, text_vals)

    data = [
        {
            "header": h,
            "text": t,
            "header_vector": hv,
            "text_vector": tv,
        }
        for h, t, hv, tv in zip(header_vals, text_vals, h_vecs, t_vecs)
    ]

    client.insert(collection_name="terms", data=data)
    print(f"  Вставлено: {len(data)} записей")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ingest_programs()

    print(f"\nПодключение к Milvus: {cfg.MILVUS_URI}")
    milvus_client = MilvusClient(uri=cfg.MILVUS_URI)
    embed_model = load_embed_model()

    ingest_faq(embed_model, milvus_client)
    ingest_terms(embed_model, milvus_client)

    print("\n✓ Загрузка завершена")


if __name__ == "__main__":
    main()
