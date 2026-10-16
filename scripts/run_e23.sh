#!/usr/bin/env bash
set -euo pipefail

repository_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
environment_dir=${POINCARE_EXEC_ENV_DIR:-$repository_root/.venv}
artifact_dir=${POINCARE_EXEC_ARTIFACT_DIR:-$repository_root/artifacts}
config=configs/benchmarks/e23_unet_b1_forward_memory.yaml
mode=${1:-}
visible_devices=${CUDA_VISIBLE_DEVICES:-}

if [[ ! "$visible_devices" =~ ^[0-9]+$ ]]; then
  echo "CUDA_VISIBLE_DEVICES must name exactly one physical GPU index" >&2
  exit 2
fi

case "$mode" in
  default)
    unset XLA_FLAGS
    ;;
  autotune0)
    # Disables cuDNN/cuBLAS algorithm search during compilation.
    export XLA_FLAGS=--xla_gpu_autotune_level=0
    ;;
  *)
    echo "usage: $0 default|autotune0" >&2
    exit 2
    ;;
esac
export XLA_PYTHON_CLIENT_PREALLOCATE=false

cd "$repository_root"
mkdir -p "$artifact_dir"
"$environment_dir/bin/poincare-exec" benchmark \
  --protocol configs/protocols/protocol_1_11.yaml \
  --config "$config" \
  --output "$artifact_dir/e23_unet_b1_forward_memory_${mode}.csv"
