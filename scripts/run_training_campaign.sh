#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 || "$#" -gt 2 ]]; then
  echo "usage: $0 TRAINING_CONFIG [ACCEPT_COMPLETED_COMMIT]" >&2
  exit 2
fi

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
environment_dir=${POINCARE_EXEC_ENV_DIR:-$repository_root/.venv}
artifact_dir=${POINCARE_EXEC_ARTIFACT_DIR:-$repository_root/artifacts}
data_root=${POINCARE_EXEC_DATA_ROOT:-$repository_root/data/raw}
manifest_dir=${POINCARE_EXEC_MANIFEST_DIR:-$repository_root/splits}
max_attempts=${POINCARE_EXEC_MAX_ATTEMPTS:-}
config_path="$1"
accepted_commit=${2:-}
experiment_id="$(basename "${config_path}" .yaml | cut -d_ -f1)"
visible_devices=${CUDA_VISIBLE_DEVICES:-}

if [[ ! "$visible_devices" =~ ^[0-9]+$ ]]; then
  echo "CUDA_VISIBLE_DEVICES must name exactly one physical GPU index" >&2
  exit 2
fi

cd "${repository_root}"

campaign_args=(
  training-campaign
  --protocol protocol.yaml
  --config "${config_path}"
  --data-root "$data_root"
  --manifest-dir "$manifest_dir"
  --output-dir "$artifact_dir/training/${experiment_id}"
)
if [[ -n "$accepted_commit" ]]; then
  campaign_args+=(--accept-completed-commit "$accepted_commit")
fi
if [[ -n "$max_attempts" ]]; then
  campaign_args+=(--max-attempts "$max_attempts")
fi

"$environment_dir/bin/poincare-exec" "${campaign_args[@]}"
