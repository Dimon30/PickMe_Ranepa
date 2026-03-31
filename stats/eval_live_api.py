
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests

from stats.common import (
    RESULTS_DIR,
    DatasetSpec,
    detect_dataset,
    ensure_dirs,
    exact_match,
    expand_eval_rows,
    percentile,
    save_json,
    semantic_cosine,
    token_f1,
)


def evaluate_dataset(spec: DatasetSpec, api_url: str, user_id_base: int, timeout_s: int = 120) -> tuple[pd.DataFrame, dict]:
    rows = expand_eval_rows(spec, include_paraphrases=True)
    out = []
    for i, row in rows.iterrows():
        payload = {'text': str(row['question']), 'user_id': user_id_base + i}
        t0 = time.perf_counter()
        resp = requests.post(api_url, json=payload, timeout=timeout_s)
        latency_ms = (time.perf_counter() - t0) * 1000
        resp.raise_for_status()
        pred = resp.json().get('answer', '')
        gold = str(row['gold_answer'])
        out.append({
            **row.to_dict(),
            'pred_answer': pred,
            'latency_ms': round(latency_ms, 1),
            'exact_match': exact_match(pred, gold),
            'f1': token_f1(pred, gold),
            'cosine_similarity': semantic_cosine(pred, gold),
            'pred_len': len(str(pred)),
            'gold_len': len(gold),
        })
    df = pd.DataFrame(out)
    summary = {
        'dataset': spec.name,
        'rows': int(len(df)),
        'exact_match_mean': float(df['exact_match'].mean()),
        'f1_mean': float(df['f1'].mean()),
        'cosine_similarity_mean': float(df['cosine_similarity'].mean()),
        'latency_ms_mean': float(df['latency_ms'].mean()),
        'latency_ms_p50': float(percentile(df['latency_ms'], 0.50)),
        'latency_ms_p95': float(percentile(df['latency_ms'], 0.95)),
    }
    return df, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--api-url', default='http://127.0.0.1:8000/ask')
    parser.add_argument('--user-id-base', type=int, default=9_000_000_000)
    parser.add_argument('--datasets', nargs='+', default=[
        'test_all_program.xlsx',
        'test_database.xlsx',
        'test_database2.xlsx',
    ])
    args = parser.parse_args()

    ensure_dirs()
    summaries = []
    all_frames = []
    for ds_idx, ds_path in enumerate(args.datasets):
        spec = detect_dataset(Path(ds_path))
        df, summary = evaluate_dataset(spec, args.api_url, args.user_id_base + ds_idx * 100_000)
        all_frames.append(df)
        summaries.append(summary)
        df.to_csv(RESULTS_DIR / f'api_eval_{spec.name}.csv', index=False)

    summary_df = pd.DataFrame(summaries).sort_values('dataset')
    summary_df.to_csv(RESULTS_DIR / 'api_eval_summary.csv', index=False)
    save_json(RESULTS_DIR / 'api_eval_summary.json', {'datasets': summaries})
    pd.concat(all_frames, ignore_index=True).to_csv(RESULTS_DIR / 'api_eval_all.csv', index=False)
    print(summary_df.to_string(index=False))


if __name__ == '__main__':
    main()
