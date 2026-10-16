#!/usr/bin/env bash
set -euo pipefail

repository_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
environment_dir=${POINCARE_EXEC_ENV_DIR:-$repository_root/.venv}
artifact_dir=${POINCARE_EXEC_ARTIFACT_DIR:-$repository_root/artifacts}
output=${POINCARE_EXEC_E24_OUTPUT:-$artifact_dir/e24_unet_mechanism_ablation.csv}
visible_devices=${CUDA_VISIBLE_DEVICES:-}

if [[ ! "$visible_devices" =~ ^[0-9]+$ ]]; then
  echo "CUDA_VISIBLE_DEVICES must name exactly one physical GPU index" >&2
  exit 2
fi

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_PYTHON_CLIENT_PREALLOCATE=false

cd "$repository_root"
mkdir -p "$artifact_dir"
"$environment_dir/bin/poincare-exec" benchmark \
  --protocol protocol.yaml \
  --config configs/benchmarks/e24_unet_mechanism_ablation.yaml \
  --output "$output"
