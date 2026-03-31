
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from stats.common import RESULTS_DIR, detect_dataset, ensure_dirs, normalize_text, save_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.services import faq_service  # noqa: E402


def build_gold_maps(project_root: Path) -> tuple[dict[str, str], dict[str, str]]:
    db1 = pd.read_excel(project_root / 'data' / 'Database.xlsx')
    db2 = pd.read_excel(project_root / 'data' / 'Database-2.xlsx')
    faq_map = {normalize_text(ans): f'faq:{i}' for i, ans in enumerate(db1['Answer'])}
    term_map = {normalize_text(txt): f'terms:{i}' for i, txt in enumerate(db2['text'])}
    return faq_map, term_map


def result_id(hit: dict, faq_map: dict[str, str], term_map: dict[str, str]) -> str | None:
    if 'answer' in hit:
        return faq_map.get(normalize_text(hit.get('answer')))
    if 'text' in hit:
        return term_map.get(normalize_text(hit.get('text')))
    return None


def evaluate_dataset(ds_path: Path, top_k: int) -> tuple[pd.DataFrame, dict]:
    spec = detect_dataset(ds_path)
    faq_map, term_map = build_gold_maps(PROJECT_ROOT)
    raw = pd.read_excel(ds_path)
    rows = []
    for i, row in raw.iterrows():
        question_variants = [('main', row[spec.question_col])]
        if spec.paraphrase_col and pd.notna(row[spec.paraphrase_col]):
            question_variants.append(('paraphrase', row[spec.paraphrase_col]))

        gold_answer = row[spec.answer_col]
        gold_id = faq_map.get(normalize_text(gold_answer)) or term_map.get(normalize_text(gold_answer))
        if gold_id is None:
            raise ValueError(f'Не найден gold source для строки {i} в {ds_path}')

        for variant, question in question_variants:
            hits = faq_service.search_all(str(question), top_k=top_k)
            retrieved_ids = [result_id(hit, faq_map, term_map) for hit in hits]
            rank = None
            for pos, rid in enumerate(retrieved_ids, start=1):
                if rid == gold_id:
                    rank = pos
                    break
            rows.append({
                'dataset': spec.name,
                'row_id': int(i),
                'question_variant': variant,
                'question': question,
                'gold_id': gold_id,
                'rank': rank,
                'hit_at_k': int(rank is not None),
                'rr': 0.0 if rank is None else 1.0 / rank,
                'retrieved_ids': ' | '.join(rid or 'None' for rid in retrieved_ids),
            })
    df = pd.DataFrame(rows)
    summary = {
        'dataset': spec.name,
        'rows': int(len(df)),
        'top_k': int(top_k),
        'recall_at_k': float(df['hit_at_k'].mean()),
        'mrr': float(df['rr'].mean()),
    }
    return df, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--datasets', nargs='+', default=['test_database.xlsx', 'test_database2.xlsx'])
    args = parser.parse_args()

    ensure_dirs()
    all_frames = []
    summaries = []
    for ds in args.datasets:
        df, summary = evaluate_dataset(Path(ds), top_k=args.top_k)
        df.to_csv(RESULTS_DIR / f'retrieval_eval_{summary["dataset"]}.csv', index=False)
        all_frames.append(df)
        summaries.append(summary)

    pd.DataFrame(summaries).to_csv(RESULTS_DIR / 'retrieval_eval_summary.csv', index=False)
    save_json(RESULTS_DIR / 'retrieval_eval_summary.json', {'datasets': summaries})
    pd.concat(all_frames, ignore_index=True).to_csv(RESULTS_DIR / 'retrieval_eval_all.csv', index=False)
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == '__main__':
    main()
