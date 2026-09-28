#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repository_root=$(cd -- "$script_dir/.." && pwd -P)
environment_dir=${POINCARE_EXEC_ENV_DIR:-$repository_root/.venv}

POINCARE_EXEC_ENV_DIR="$environment_dir" "$script_dir/bootstrap.sh"
uv pip install \
  --python "$environment_dir/bin/python" \
  --reinstall \
  --no-deps \
  'jaxlib==0.4.25+cuda11.cudnn86' \
  --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
uv pip install \
  --python "$environment_dir/bin/python" \
  --requirement "$script_dir/cuda11_v100_requirements.txt"
echo "Installed pinned V100/CUDA 11.8 environment: $environment_dir"
