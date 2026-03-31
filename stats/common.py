
from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
except Exception:  # pragma: no cover
    TfidfVectorizer = None
    sklearn_cosine_similarity = None

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / 'stats' / 'results'
PLOTS_DIR = ROOT / 'stats' / 'plots'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = os.getenv('STATS_EMBED_MODEL', 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
_EMBED_MODEL = None


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def normalize_text(text: str | float | int | None) -> str:
    if text is None:
        return ''
    s = str(text)
    s = s.replace('\xa0', ' ')
    s = s.replace('ё', 'е').replace('Ё', 'Е')
    s = s.lower().strip()
    s = re.sub(r'\s+', ' ', s)
    return s


def tokenize(text: str | None) -> list[str]:
    s = normalize_text(text)
    return re.findall(r'[a-zа-я0-9]+', s)


def exact_match(a: str, b: str) -> float:
    return float(normalize_text(a) == normalize_text(b))


def token_f1(pred: str, gold: str) -> float:
    pred_tokens = tokenize(pred)
    gold_tokens = tokenize(gold)
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    pred_counts = {}
    gold_counts = {}
    for t in pred_tokens:
        pred_counts[t] = pred_counts.get(t, 0) + 1
    for t in gold_tokens:
        gold_counts[t] = gold_counts.get(t, 0) + 1
    common = sum(min(pred_counts.get(t, 0), gold_counts.get(t, 0)) for t in set(pred_counts) | set(gold_counts))
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def get_embed_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None and SentenceTransformer is not None:
        _EMBED_MODEL = SentenceTransformer(MODEL_NAME)
    return _EMBED_MODEL


def semantic_cosine(pred: str, gold: str) -> float:
    pred = pred or ''
    gold = gold or ''
    model = get_embed_model()
    if model is not None:
        vecs = model.encode([pred, gold], normalize_embeddings=True)
        return float(vecs[0] @ vecs[1])
    if TfidfVectorizer is None or sklearn_cosine_similarity is None:
        return 0.0
    vec = TfidfVectorizer().fit_transform([pred, gold])
    return float(sklearn_cosine_similarity(vec[0], vec[1])[0, 0])


def percentile(values: Iterable[float], q: float) -> float:
    xs = sorted(float(v) for v in values)
    if not xs:
        return float('nan')
    k = (len(xs) - 1) * q
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs[int(k)]
    return xs[f] * (c - k) + xs[c] * (k - f)


@dataclass
class DatasetSpec:
    name: str
    path: Path
    question_col: str
    answer_col: str
    paraphrase_col: str | None = None
    source_type: str = 'api'


def detect_dataset(path: str | Path) -> DatasetSpec:
    path = Path(path)
    if not path.exists():
        path = DATA_DIR / path
    df = pd.read_excel(path)
    cols = set(df.columns)
    if {'Вопрос', 'Ответ'}.issubset(cols):
        return DatasetSpec(name='all_program', path=path, question_col='Вопрос', answer_col='Ответ', source_type='sql')
    if {'Оригинальный вопрос', 'Перефразировка', 'Ответ'}.issubset(cols):
        return DatasetSpec(name='database', path=path, question_col='Оригинальный вопрос', answer_col='Ответ', paraphrase_col='Перефразировка', source_type='rag')
    if {'Вопрос абитуриента', 'Ответ (text)'}.issubset(cols):
        return DatasetSpec(name='database2', path=path, question_col='Вопрос абитуриента', answer_col='Ответ (text)', source_type='rag')
    raise ValueError(f'Неизвестный формат датасета: {path}')


def expand_eval_rows(spec: DatasetSpec, include_paraphrases: bool = True) -> pd.DataFrame:
    df = pd.read_excel(spec.path).copy()
    rows = []
    for idx, row in df.iterrows():
        base = {
            'dataset': spec.name,
            'row_id': int(idx),
            'gold_answer': row[spec.answer_col],
        }
        rows.append({**base, 'question_variant': 'main', 'question': row[spec.question_col]})
        if include_paraphrases and spec.paraphrase_col and pd.notna(row[spec.paraphrase_col]):
            rows.append({**base, 'question_variant': 'paraphrase', 'question': row[spec.paraphrase_col]})
    return pd.DataFrame(rows)


def save_json(path: str | Path, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def now_ts() -> str:
    return time.strftime('%Y-%m-%d %H:%M:%S')
