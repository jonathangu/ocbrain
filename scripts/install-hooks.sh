#!/usr/bin/env bash
# Convenience wrapper: symlink tracked git hooks (ops/hooks) into .git/hooks.
# Equivalent to `ocbrain install-hooks`. Pure stdlib, no deps.
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
PY="$(command -v python3)"
exec env PYTHONPATH="${ROOT}/src" "${PY}" -m ocbrain.cli install-hooks --root "${ROOT}"
