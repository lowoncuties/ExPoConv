#!/usr/bin/env bash
set -euo pipefail

repository_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
environment_dir=${POINCARE_EXEC_ENV_DIR:-$repository_root/.venv-jax0112-cuda12}
artifact_dir=${POINCARE_EXEC_ARTIFACT_DIR:-$repository_root/artifacts}
visible_devices=${CUDA_VISIBLE_DEVICES:-}

if [[ ! "$visible_devices" =~ ^[0-9]+$ ]]; then
  echo "CUDA_VISIBLE_DEVICES must name exactly one physical GPU index" >&2
  exit 2
fi
if [[ ! -x "$environment_dir/bin/python" ]]; then
  echo "Missing current-JAX Python environment: $environment_dir" >&2
  exit 66
fi

# The current stable JAX wheels bundle their CUDA-12 user-space libraries.
# A host LD_LIBRARY_PATH can override those libraries, so keep it out of this
# separately reported portability cohort.
unset LD_LIBRARY_PATH
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$repository_root/src${PYTHONPATH:+:$PYTHONPATH}"

cd "$repository_root"
mkdir -p "$artifact_dir"
"$environment_dir/bin/python" -m poincare_exec.cli benchmark \
  --protocol protocol.yaml \
  --config configs/benchmarks/e27_current_jax_unet_update.yaml \
  --output "$artifact_dir/e27_current_jax_unet_update.csv"
