#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "usage: $0 CONFIG VARIANT SEED" >&2
  exit 2
fi

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
environment_dir=${POINCARE_EXEC_ENV_DIR:-$repository_root/.venv}
artifact_dir=${POINCARE_EXEC_ARTIFACT_DIR:-$repository_root/artifacts}
config_path="$1"
variant="$2"
seed="$3"
experiment_id="$(basename "${config_path}" .yaml | cut -d_ -f1)"
visible_devices=${CUDA_VISIBLE_DEVICES:-}

if [[ ! "$visible_devices" =~ ^[0-9]+$ ]]; then
  echo "CUDA_VISIBLE_DEVICES must name exactly one physical GPU index" >&2
  exit 2
fi

cd "${repository_root}"

"$environment_dir/bin/poincare-exec" training-worker \
  --protocol protocol.yaml \
  --config "${config_path}" \
  --variant "${variant}" \
  --seed "${seed}" \
  --data-root data/raw \
  --manifest-dir splits \
  --output-dir "$artifact_dir/training/${experiment_id}/${variant}/seed_${seed}"
