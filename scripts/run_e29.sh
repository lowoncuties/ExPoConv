#!/usr/bin/env bash
set -euo pipefail

repository_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
environment_dir=${POINCARE_EXEC_ENV_DIR:-$repository_root/.venv}
artifact_dir=${POINCARE_EXEC_ARTIFACT_DIR:-$repository_root/artifacts}
config=configs/benchmarks/e29_v100_euclidean_reference.yaml
output=${POINCARE_EXEC_E29_OUTPUT:-$artifact_dir/e29_v100_euclidean_reference.csv}
visible_devices=${CUDA_VISIBLE_DEVICES:-}

if [[ ! "$visible_devices" =~ ^[0-9]+$ ]]; then
  echo "CUDA_VISIBLE_DEVICES must name exactly one physical GPU index" >&2
  exit 2
fi

# E29 times only the Euclidean path E on the E08/E13 workloads.
# It keeps the E08/E13 allocator conditions: the default JAX fraction (0.75 of
# device memory, XLA_PYTHON_CLIENT_MEM_FRACTION unset) without preallocation.
unset XLA_PYTHON_CLIENT_MEM_FRACTION
unset XLA_PYTHON_CLIENT_ALLOCATOR
unset XLA_FLAGS
export XLA_PYTHON_CLIENT_PREALLOCATE=false

cd "$repository_root"
mkdir -p "$artifact_dir"
"$environment_dir/bin/poincare-exec" benchmark \
  --protocol protocol.yaml \
  --config "$config" \
  --output "$output"
