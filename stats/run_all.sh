#!/usr/bin/env bash
set -euo pipefail

API_URL="${API_URL:-http://127.0.0.1:8000/ask}"
USER_ID_BASE="${USER_ID_BASE:-9000000000}"

python -m stats.eval_live_api --api-url "$API_URL" --user-id-base "$USER_ID_BASE"
python -m stats.eval_retrieval_rag --top-k 5
python -m stats.eval_sql_diagnostics
python -m stats.plot_metrics
python -m stats.build_summary

echo "Done. See stats/results and stats/plots"
