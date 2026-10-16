#!/usr/bin/env bash
set -euo pipefail

repository_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
environment_dir=${POINCARE_EXEC_ENV_DIR:-$repository_root/.venv}
artifact_dir=${POINCARE_EXEC_ARTIFACT_DIR:-$repository_root/artifacts}
visible_devices=${CUDA_VISIBLE_DEVICES:-}

if [[ ! "$visible_devices" =~ ^[0-9]+$ ]]; then
  echo "CUDA_VISIBLE_DEVICES must name exactly one physical GPU index" >&2
  exit 2
fi

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$repository_root/src"

cd "$repository_root"
mkdir -p "$artifact_dir"
for replay in 0 1 2; do
  output="$artifact_dir/e28_elementwise_replay_${replay}.jsonl"
  if [[ -e "$output" ]]; then
    echo "Refusing to reuse E28 output: $output" >&2
    exit 64
  fi
  "$environment_dir/bin/python" -m poincare_exec.cli structured-numerics \
    --protocol protocol.yaml \
    --config configs/numerics/e28_elementwise_replay_stability.yaml \
    --output "$output"
done
