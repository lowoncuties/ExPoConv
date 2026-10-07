#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repository_root=$(cd -- "$script_dir/.." && pwd -P)
environment_dir=${POINCARE_EXEC_ENV_DIR:-$repository_root/.venv-cuda12}
python_executable=${POINCARE_EXEC_PYTHON:-python3.11}

POINCARE_EXEC_ENV_DIR="$environment_dir" \
  POINCARE_EXEC_PYTHON="$python_executable" \
  "$script_dir/bootstrap.sh"
uv pip install \
  --python "$environment_dir/bin/python" \
  --requirement "$script_dir/cuda12_a6000_requirements.txt"
echo "Installed pinned A6000/CUDA 12 environment: $environment_dir"
