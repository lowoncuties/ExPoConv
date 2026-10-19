#!/usr/bin/env bash
set -euo pipefail

repository_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
environment_dir=${POINCARE_EXEC_ENV_DIR:-$repository_root/.venv}
"$environment_dir/bin/python" - <<'PY'
import jax
import jaxlib

devices = jax.devices()
print(f"jax={jax.__version__} jaxlib={jaxlib.__version__} devices={devices}")
if not any(device.platform == "gpu" for device in devices):
    raise SystemExit("No JAX GPU device is available")
value = (jax.numpy.ones((1024, 1024)) @ jax.numpy.ones((1024, 1024))).block_until_ready()
if float(value[0, 0]) != 1024.0:
    raise SystemExit("Compiled CUDA smoke result was incorrect")
PY
