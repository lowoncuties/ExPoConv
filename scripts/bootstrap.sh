#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repository_root=$(cd -- "$script_dir/.." && pwd -P)
python_executable=${POINCARE_EXEC_PYTHON:-python3.11}
environment_dir=${POINCARE_EXEC_ENV_DIR:-$repository_root/.venv}

command -v uv >/dev/null 2>&1 || {
  echo "uv is required and was not found on PATH" >&2
  exit 69
}

uv lock --check --offline --project "$repository_root"
UV_PROJECT_ENVIRONMENT="$environment_dir" uv sync \
  --project "$repository_root" \
  --python "$python_executable" \
  --frozen \
  --extra dev
echo "Created CPU/reference environment: $environment_dir"
