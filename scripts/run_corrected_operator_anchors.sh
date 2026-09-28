#!/usr/bin/env bash
set -euo pipefail

repository_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
environment_dir=${POINCARE_EXEC_ENV_DIR:-$repository_root/.venv}
artifact_dir=${POINCARE_EXEC_ARTIFACT_DIR:-$repository_root/artifacts}
output=${POINCARE_EXEC_CORRECTED_ANCHOR_OUTPUT:-$artifact_dir/corrected_operator_anchors.csv}
visible_devices=${CUDA_VISIBLE_DEVICES:-}

if [[ ! "$visible_devices" =~ ^[0-9]+$ ]]; then
  echo "CUDA_VISIBLE_DEVICES must name exactly one physical GPU index" >&2
  exit 2
fi

cd "$repository_root"
"$environment_dir/bin/poincare-exec" benchmark \
  --protocol protocol.yaml \
  --config configs/benchmarks/corrected_operator_anchors.yaml \
  --output "$output"
