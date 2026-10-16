#!/usr/bin/env bash
set -euo pipefail

repository_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
environment_dir=${POINCARE_EXEC_ENV_DIR:-$repository_root/.venv}
artifact_dir=${POINCARE_EXEC_ARTIFACT_DIR:-$repository_root/artifacts}
config=configs/benchmarks/e22_unet_allocator_fraction.yaml
output=${POINCARE_EXEC_E22_OUTPUT:-$artifact_dir/e22_unet_allocator_fraction.csv}
visible_devices=${CUDA_VISIBLE_DEVICES:-}

if [[ ! "$visible_devices" =~ ^[0-9]+$ ]]; then
  echo "CUDA_VISIBLE_DEVICES must name exactly one physical GPU index" >&2
  exit 2
fi

# E22 differs from E13 only in the JAX allocator cap (default 0.75 of device memory).
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_PYTHON_CLIENT_PREALLOCATE=false

cd "$repository_root"
mkdir -p "$artifact_dir"
"$environment_dir/bin/poincare-exec" benchmark \
  --protocol configs/protocols/protocol_1_11.yaml \
  --config "$config" \
  --output "$output"
