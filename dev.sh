#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"
lab_python=python3
if [[ -x .venv/bin/python ]]; then
    lab_python=.venv/bin/python
fi

"$lab_python" -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --timeout-graceful-shutdown 5 &
lab_api_pid=$!
trap 'kill "$lab_api_pid" 2>/dev/null || true; wait "$lab_api_pid" 2>/dev/null || true' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

npm --prefix frontend run dev
