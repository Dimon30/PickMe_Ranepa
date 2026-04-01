
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


def evaluate_dataset(
    spec: DatasetSpec,
    api_url: str,
    user_id_base: int,
    timeout_s: int = 120
) -> tuple[pd.DataFrame, dict]:
    rows = expand_eval_rows(spec, include_paraphrases=True)
    out = []

    for i, row in rows.iterrows():
        question = str(row["question"])
        gold = str(row["gold_answer"])
        payload = {
            "text": question,
            "user_id": user_id_base + i,
        }

        t0 = time.perf_counter()

        pred = ""
        error = ""
        status_code = None

        try:
            resp = requests.post(api_url, json=payload, timeout=timeout_s)
            latency_ms = (time.perf_counter() - t0) * 1000
            status_code = resp.status_code
            resp.raise_for_status()

            data = resp.json()
            pred = str(data.get("answer", "") or "")

        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000
            error = str(e)

        ok = int(error == "")

        out.append({
            **row.to_dict(),
            "pred_answer": pred,
            "error": error,
            "ok": ok,
            "status_code": status_code,
            "latency_ms": round(latency_ms, 1),
            "exact_match": exact_match(pred, gold) if pred else 0.0,
            "f1": token_f1(pred, gold) if pred else 0.0,
            "cosine_similarity": semantic_cosine(pred, gold) if pred else 0.0,
            "pred_len": len(pred),
            "gold_len": len(gold),
        })

    df = pd.DataFrame(out)

    success_df = df[df["ok"] == 1].copy()

    if len(success_df) > 0:
        exact_match_mean = float(success_df["exact_match"].mean())
        f1_mean = float(success_df["f1"].mean())
        cosine_similarity_mean = float(success_df["cosine_similarity"].mean())
        latency_ms_mean = float(success_df["latency_ms"].mean())
        latency_ms_p50 = float(percentile(success_df["latency_ms"], 0.50))
        latency_ms_p95 = float(percentile(success_df["latency_ms"], 0.95))
    else:
        exact_match_mean = 0.0
        f1_mean = 0.0
        cosine_similarity_mean = 0.0
        latency_ms_mean = 0.0
        latency_ms_p50 = 0.0
        latency_ms_p95 = 0.0

    summary = {
        "dataset": spec.name,
        "rows_total": int(len(df)),
        "rows_success": int(len(success_df)),
        "rows_failed": int(len(df) - len(success_df)),
        "success_rate": float(len(success_df) / len(df)) if len(df) else 0.0,
        "exact_match_mean": exact_match_mean,
        "f1_mean": f1_mean,
        "cosine_similarity_mean": cosine_similarity_mean,
        "latency_ms_mean": latency_ms_mean,
        "latency_ms_p50": latency_ms_p50,
        "latency_ms_p95": latency_ms_p95,
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
