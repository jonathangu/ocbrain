#!/usr/bin/env bash
# Convenience wrapper: symlink tracked git hooks (ops/hooks) into .git/hooks.
# It runs the source-tree ops CLI directly, so a contributor does not need an
# already-installed console entry point.
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
if [[ -x "${ROOT}/.venv/bin/python" ]]; then
  PY="${ROOT}/.venv/bin/python"
else
  PY="$(command -v python3)"
fi
exec env PYTHONPATH="${ROOT}/src:${ROOT}/packages/ops/src" \
  "${PY}" -m ocbrain_ops.cli install-hooks --root "${ROOT}"
