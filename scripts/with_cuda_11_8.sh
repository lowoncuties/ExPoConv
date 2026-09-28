#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repository_root=$(cd -- "$script_dir/.." && pwd -P)
environment_dir=${POINCARE_EXEC_ENV_DIR:-$repository_root/.venv}
site_packages=$(
  "$environment_dir/bin/python" -c \
    'import sysconfig; print(sysconfig.get_paths()["purelib"])'
)
cuda_library_path=$(
  find "$site_packages/nvidia" -mindepth 2 -maxdepth 2 -type d -name lib \
    -printf '%p:' | sort
)
export LD_LIBRARY_PATH="${cuda_library_path}/usr/local/cuda-11.8/targets/x86_64-linux/lib"
export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false}
exec "$@"
