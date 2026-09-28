#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "usage: $0 OUTPUT_DIRECTORY" >&2
  exit 2
fi

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
environment_dir=${POINCARE_EXEC_ENV_DIR:-$repository_root/.venv}
cd "$repository_root"

"$environment_dir/bin/poincare-exec" numerical-repair \
  --protocol protocol.yaml \
  --config configs/numerics/e02_repair.yaml \
  --cases configs/numerics/numerical_cases.json \
  --output-dir "$1"
