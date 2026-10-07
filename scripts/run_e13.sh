#!/usr/bin/env bash
set -euo pipefail

repository_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
environment_dir=${POINCARE_EXEC_ENV_DIR:-$repository_root/.venv}
artifact_dir=${POINCARE_EXEC_ARTIFACT_DIR:-$repository_root/artifacts}
config=${POINCARE_EXEC_E13_CONFIG:-configs/benchmarks/e13_unet_scaling.yaml}
output=${POINCARE_EXEC_E13_OUTPUT:-$artifact_dir/e13_unet_scaling.csv}
visible_devices=${CUDA_VISIBLE_DEVICES:-}

if [[ ! "$visible_devices" =~ ^[0-9]+$ ]]; then
  echo "CUDA_VISIBLE_DEVICES must name exactly one physical GPU index" >&2
  exit 2
fi

cd "$repository_root"
"$environment_dir/bin/poincare-exec" benchmark \
  --protocol configs/protocols/protocol_1_11.yaml \
  --config "$config" \
  --output "$output"
