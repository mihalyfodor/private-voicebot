#!/usr/bin/env bash
# Check everything: JS syntax, the node DOM fixture, then the unit suite.
# Extra arguments go to pytest: `scripts/check.sh -m e2e tests/e2e`, `scripts/check.sh -m live`.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-python3}
if [ -x .venv/bin/python ]; then PY=.venv/bin/python; fi

echo "== node --check static/app.js"
node --check static/app.js
echo "== node tests/app_fixture.mjs"
node tests/app_fixture.mjs > /dev/null
if [ $# -eq 0 ]; then
  echo "== pytest tests/"
  "$PY" -m pytest tests/ -q
else
  echo "== pytest $*"
  "$PY" -m pytest -q "$@"
fi
