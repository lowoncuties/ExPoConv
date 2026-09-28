#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
environment_dir=${POINCARE_EXEC_ENV_DIR:-$repository_root/.venv}
artifact_dir=${POINCARE_EXEC_ARTIFACT_DIR:-$repository_root/artifacts}
config=${POINCARE_EXEC_E11_CONFIG:-configs/profiles/e11_unet_bounded.yaml}
output_dir=${POINCARE_EXEC_E11_OUTPUT_DIR:-$artifact_dir/profiles/e11_unet_bounded}
visible_devices=${CUDA_VISIBLE_DEVICES:-}

if [[ ! "$visible_devices" =~ ^[0-9]+$ ]]; then
  echo "CUDA_VISIBLE_DEVICES must name exactly one physical GPU index" >&2
  exit 2
fi

cd "${repository_root}"

"$environment_dir/bin/poincare-exec" profile \
  --protocol protocol.yaml \
  --config "$config" \
  --output-dir "$output_dir"
