#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${CAMPUS_PYTHON:-.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
  echo 'Create .venv and install requirements-dev.txt first. See README.md.'
  exit 1
fi
if [ ! -f reports/latest/model.joblib ]; then "$PYTHON" run_pipeline.py --sample; fi
(cd frontend && npm run build)
echo 'Open http://127.0.0.1:8031'
exec "$PYTHON" -m uvicorn src.api:app --host 127.0.0.1 --port 8031
