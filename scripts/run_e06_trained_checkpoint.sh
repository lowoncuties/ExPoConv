#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "usage: $0 TRAINING_CONFIG VARIANT SEED" >&2
  exit 2
fi

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
environment_dir=${POINCARE_EXEC_ENV_DIR:-$repository_root/.venv}
artifact_dir=${POINCARE_EXEC_ARTIFACT_DIR:-$repository_root/artifacts}
training_config="$1"
variant="$2"
seed="$3"
experiment_id="$(basename "${training_config}" .yaml | cut -d_ -f1)"
checkpoint_dir="$artifact_dir/training/$experiment_id/$variant/seed_$seed/checkpoints"
checkpoint_path="$checkpoint_dir/final.pkl"
output_dir="$artifact_dir/checkpoint_experiments/trained/$experiment_id/$variant/seed_$seed"
cd "${repository_root}"

"$environment_dir/bin/poincare-exec" checkpoint-evaluation \
  --protocol protocol.yaml \
  --config configs/checkpoints/e06_e07.yaml \
  --training-config "${training_config}" \
  --checkpoint "${checkpoint_path}" \
  --data-root data/raw \
  --manifest-dir splits \
  --output-dir "${output_dir}"
