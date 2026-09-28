#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 0 ]]; then
  echo "usage: $0" >&2
  exit 2
fi

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
environment_dir=${POINCARE_EXEC_ENV_DIR:-$repository_root/.venv}
artifact_dir=${POINCARE_EXEC_ARTIFACT_DIR:-$repository_root/artifacts}
diagnostic_config=configs/checkpoints/e07_journal_diagnostics_v2.yaml
visible_devices=${CUDA_VISIBLE_DEVICES:-}

if [[ ! "$visible_devices" =~ ^[0-9]+$ ]]; then
  echo "CUDA_VISIBLE_DEVICES must name exactly one physical GPU index" >&2
  exit 2
fi

cd "${repository_root}"

for training_config in \
  configs/training/e09_isic16_unet.yaml \
  configs/training/e10_cifar100_resnet20.yaml
do
  experiment_id="$(basename "${training_config}" .yaml | cut -d_ -f1)"
  "$environment_dir/bin/poincare-exec" retained-diagnostic-campaign \
    --protocol protocol.yaml \
    --config "${diagnostic_config}" \
    --training-config "${training_config}" \
    --data-root data/raw \
    --manifest-dir splits \
    --training-dir "$artifact_dir/training/${experiment_id}" \
    --output-dir "$artifact_dir/checkpoint_experiments/journal_diagnostics_v2/${experiment_id}"
done
