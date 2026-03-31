
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz

from stats.common import RESULTS_DIR, ensure_dirs, normalize_text, save_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.services import sql_service  # noqa: E402


def extract_gold_program(answer: str, program_names: list[str]) -> str | None:
    m = re.search(r'[«\"]([^»\"]+)[»\"]', str(answer))
    if m:
        quoted = normalize_text(m.group(1))
        scored = [(name, fuzz.partial_ratio(quoted, normalize_text(name))) for name in program_names]
        best_name, best_score = max(scored, key=lambda x: x[1])
        if best_score >= 70:
            return best_name
    ans_norm = normalize_text(answer)
    scored = [(name, fuzz.partial_ratio(ans_norm, normalize_text(name))) for name in program_names]
    best_name, best_score = max(scored, key=lambda x: x[1])
    return best_name if best_score >= 75 else None


def build_candidate_ranking(df_programs: pd.DataFrame, filter_column: str | None, filter_value: str | None, question: str) -> list[str]:
    if filter_column and filter_value and filter_column in df_programs.columns:
        work = df_programs[['program', filter_column]].copy()
        target = normalize_text(filter_value)
        work['score'] = work[filter_column].fillna('').astype(str).map(lambda x: fuzz.partial_ratio(normalize_text(x), target))
        work = work.sort_values(['score', 'program'], ascending=[False, True])
        return work['program'].astype(str).tolist()
    work = df_programs[['program']].copy()
    q = normalize_text(question)
    work['score'] = work['program'].astype(str).map(lambda x: fuzz.partial_ratio(normalize_text(x), q))
    work = work.sort_values(['score', 'program'], ascending=[False, True])
    return work['program'].astype(str).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='test_all_program.xlsx')
    args = parser.parse_args()
    ensure_dirs()

    test_df = pd.read_excel(args.dataset)
    programs = pd.read_excel(PROJECT_ROOT / 'data' / 'all_program.xlsx')
    program_names = programs['program'].dropna().astype(str).tolist()

    rows = []
    for i, row in test_df.iterrows():
        question = str(row['Вопрос'])
        answer = str(row['Ответ'])
        gold_program = extract_gold_program(answer, program_names)
        pred_col, pred_val = sql_service._extract_filter(question)  # noqa: SLF001
        ranking = build_candidate_ranking(programs, pred_col, pred_val, question)
        top5 = ranking[:5]
        rank = ranking.index(gold_program) + 1 if gold_program in ranking else None
        rows.append({
            'row_id': int(i),
            'question': question,
            'gold_program': gold_program,
            'pred_filter_column': pred_col,
            'pred_filter_value': pred_val,
            'top5_programs': ' | '.join(top5),
            'hit_at_5': int(gold_program in top5) if gold_program else 0,
            'rr': 0.0 if rank is None else 1.0 / rank,
            'rank': rank,
            'filter_column_is_program': int(pred_col == 'program'),
        })

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / 'sql_diagnostics.csv', index=False)
    summary = {
        'dataset': 'all_program',
        'rows': int(len(out)),
        'filter_column_program_rate': float(out['filter_column_is_program'].mean()),
        'recall_at_5_proxy': float(out['hit_at_5'].mean()),
        'mrr_proxy': float(out['rr'].mean()),
    }
    pd.DataFrame([summary]).to_csv(RESULTS_DIR / 'sql_diagnostics_summary.csv', index=False)
    save_json(RESULTS_DIR / 'sql_diagnostics_summary.json', summary)
    print(pd.DataFrame([summary]).to_string(index=False))


if __name__ == '__main__':
    main()
