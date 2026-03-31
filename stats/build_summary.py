
from __future__ import annotations

from pathlib import Path

import pandas as pd

from stats.common import RESULTS_DIR, ensure_dirs


def maybe_read(name: str):
    path = RESULTS_DIR / name
    if path.exists():
        return pd.read_csv(path)
    return None


def main():
    ensure_dirs()
    api = maybe_read('api_eval_summary.csv')
    retrieval = maybe_read('retrieval_eval_summary.csv')
    sql_diag = maybe_read('sql_diagnostics_summary.csv')

    lines = ['# PickMe stats summary', '']
    if api is not None:
        lines += ['## End-to-end API metrics', '', api.to_markdown(index=False), '']
    if retrieval is not None:
        lines += ['## RAG retrieval metrics', '', retrieval.to_markdown(index=False), '']
    if sql_diag is not None:
        lines += ['## SQL diagnostics', '', sql_diag.to_markdown(index=False), '']

    out = RESULTS_DIR / 'SUMMARY.md'
    out.write_text('\n'.join(lines), encoding='utf-8')
    print(out)


if __name__ == '__main__':
    main()
