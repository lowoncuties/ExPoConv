#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "usage: $0 TRAINING_CONFIG" >&2
  exit 2
fi

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
environment_dir=${POINCARE_EXEC_ENV_DIR:-$repository_root/.venv}
artifact_dir=${POINCARE_EXEC_ARTIFACT_DIR:-$repository_root/artifacts}
training_config="$1"
experiment_id="$(basename "${training_config}" .yaml | cut -d_ -f1)"
cd "${repository_root}"

"$environment_dir/bin/poincare-exec" checkpoint-campaign \
  --protocol protocol.yaml \
  --config configs/checkpoints/e06_e07.yaml \
  --training-config "${training_config}" \
  --data-root data/raw \
  --manifest-dir splits \
  --training-dir "$artifact_dir/training/${experiment_id}" \
  --output-dir "$artifact_dir/checkpoint_experiments/trained/${experiment_id}"
