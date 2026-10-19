#!/usr/bin/env bash
# E35: FP32 R/F/T gradients and the FP64 oracle at the E07-J2 ResNet-20 states.
# Run on the E07-J2 device (physical V100 GPU 1) with the CUDA 11.8 stack:
#   CUDA_VISIBLE_DEVICES=1 ./scripts/with_cuda_11_8.sh ./scripts/run_e35.sh
# Set E35_ORACLE_DEVICE=cpu only if the FP64 oracle cannot run on the GPU.
# Export afterwards with scripts/export_e35_retained_oracle.py --artifact-dir <output>.
set -euo pipefail

repository_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
environment_dir=${POINCARE_EXEC_ENV_DIR:-$repository_root/.venv}
artifact_dir=${POINCARE_EXEC_ARTIFACT_DIR:-$repository_root/artifacts}
output_dir=${POINCARE_EXEC_E35_OUTPUT:-$artifact_dir/e35_retained_oracle}
oracle_device=${E35_ORACLE_DEVICE:-gpu}
visible_devices=${CUDA_VISIBLE_DEVICES:-}

if [[ ! "$visible_devices" =~ ^[0-9]+$ ]]; then
  echo "CUDA_VISIBLE_DEVICES must name exactly one physical GPU index" >&2
  exit 2
fi

export XLA_PYTHON_CLIENT_PREALLOCATE=false

cd "$repository_root"
"$environment_dir/bin/python" scripts/run_e35_retained_oracle.py \
  --stage float32 --device gpu --output-dir "$output_dir"
"$environment_dir/bin/python" scripts/run_e35_retained_oracle.py \
  --stage float64 --device "$oracle_device" --output-dir "$output_dir"
