
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from stats.common import PLOTS_DIR, RESULTS_DIR, ensure_dirs


def maybe_read(name: str) -> pd.DataFrame | None:
    path = RESULTS_DIR / name
    if not path.exists():
        return None
    return pd.read_csv(path)


def bar_plot(df: pd.DataFrame, x: str, y: str, title: str, path: Path, ylabel: str) -> None:
    plt.figure(figsize=(9, 5))
    plt.bar(df[x].astype(str), df[y])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def histogram(series: pd.Series, title: str, path: Path, xlabel: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(series.dropna(), bins=20)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main():
    ensure_dirs()
    api_summary = maybe_read('api_eval_summary.csv')
    api_all = maybe_read('api_eval_all.csv')
    retrieval = maybe_read('retrieval_eval_summary.csv')
    sql_diag = maybe_read('sql_diagnostics_summary.csv')

    if api_summary is not None:
        bar_plot(api_summary, 'dataset', 'cosine_similarity_mean', 'Средняя cosine similarity по датасетам', PLOTS_DIR / 'api_cosine_by_dataset.png', 'cosine similarity')
        bar_plot(api_summary, 'dataset', 'f1_mean', 'Средний F1 по датасетам', PLOTS_DIR / 'api_f1_by_dataset.png', 'F1')
        bar_plot(api_summary, 'dataset', 'latency_ms_mean', 'Средняя latency по датасетам', PLOTS_DIR / 'api_latency_by_dataset.png', 'Latency, ms')
    if api_all is not None:
        histogram(api_all['latency_ms'], 'Распределение latency по всем API-запросам', PLOTS_DIR / 'api_latency_hist.png', 'Latency, ms')
        histogram(api_all['cosine_similarity'], 'Распределение cosine similarity', PLOTS_DIR / 'api_cosine_hist.png', 'cosine similarity')
    if retrieval is not None:
        bar_plot(retrieval, 'dataset', 'recall_at_k', 'Recall@5 для RAG-ретривала', PLOTS_DIR / 'retrieval_recall_at5.png', 'Recall@5')
        bar_plot(retrieval, 'dataset', 'mrr', 'MRR для RAG-ретривала', PLOTS_DIR / 'retrieval_mrr.png', 'MRR')
    if sql_diag is not None:
        bar_plot(sql_diag, 'dataset', 'recall_at_5_proxy', 'Proxy Recall@5 для SQL-ветки', PLOTS_DIR / 'sql_recall_at5_proxy.png', 'Recall@5 proxy')
        bar_plot(sql_diag, 'dataset', 'mrr_proxy', 'Proxy MRR для SQL-ветки', PLOTS_DIR / 'sql_mrr_proxy.png', 'MRR proxy')


if __name__ == '__main__':
    main()
