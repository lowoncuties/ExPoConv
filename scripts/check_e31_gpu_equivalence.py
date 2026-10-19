#!/usr/bin/env python3
"""Compare R with Rg and M with Mg on the benchmark GPU for the E31 workloads.

The CPU test suite proves the gather-built patch equal to R's one-hot
convolution patch. On a GPU, XLA may lower that convolution to a cuDNN
algorithm of its choosing, so this check repeats the comparison on the device
used for timing, with the benchmark worker's JAX settings:

* every distinct convolution patch extraction that R executes in each E31 model
  workload, on deterministic random tangent inputs, compared elementwise; and
* one compiled call of each workload's benchmark function from identical state,
  compared leaf by leaf (forward logits, or loss and updated parameters and
  optimizer state for complete updates).

Each comparison reports exact equality, the maximum absolute difference, and
the number of differing elements. The check is descriptive and does not time.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "benchmarks" / "e31_gather_reference.yaml"
PAIRS = (("R", "Rg"), ("M", "Mg"))
ENVIRONMENT_FIELDS = (
    "CUDA_VISIBLE_DEVICES",
    "JAX_PLATFORMS",
    "LD_LIBRARY_PATH",
    "NVIDIA_TF32_OVERRIDE",
    "XLA_FLAGS",
    "XLA_PYTHON_CLIENT_ALLOCATOR",
    "XLA_PYTHON_CLIENT_MEM_FRACTION",
    "XLA_PYTHON_CLIENT_PREALLOCATE",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=REPOSITORY_ROOT / "protocol.yaml")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--session", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _compare(reference: Any, candidate: Any) -> dict[str, Any]:
    import jax
    import numpy as np

    reference_leaves = jax.tree_util.tree_leaves(reference)
    candidate_leaves = jax.tree_util.tree_leaves(candidate)
    if len(reference_leaves) != len(candidate_leaves):
        raise ValueError("compared trees have different leaf counts")
    differing = 0
    total = 0
    maximum = 0.0
    maximum_relative = 0.0
    for reference_leaf, candidate_leaf in zip(
        reference_leaves, candidate_leaves, strict=True
    ):
        left = np.asarray(reference_leaf)
        right = np.asarray(candidate_leaf)
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError("compared leaves differ in shape or dtype")
        total += left.size
        bits = np.dtype(f"u{left.dtype.itemsize}")
        differing += int(np.count_nonzero(left.view(bits) != right.view(bits)))
        if left.size:
            difference = np.abs(left.astype(np.float64) - right.astype(np.float64))
            maximum = max(maximum, float(np.max(difference)))
            scale = float(np.max(np.abs(left.astype(np.float64))))
            if scale > 0.0:
                maximum_relative = max(maximum_relative, float(np.max(difference)) / scale)
    return {
        "bitwise_equal": differing == 0,
        "differing_elements": differing,
        "elements": total,
        "max_abs_difference": maximum,
        "max_abs_difference_over_reference_max_abs": maximum_relative,
    }


def _patch_calls(function: Any, state: Any, inputs: Any, curvature: Any) -> list[dict]:
    """Trace R abstractly and record every patch extraction it performs."""

    import jax

    from poincare_exec.core import hnnpp

    calls: list[dict] = []
    original = hnnpp._extract_patches

    def recording(tangent, kernel_shape, *, stride, padding, dilation):
        call = {
            "input_shape": list(tangent.shape),
            "dtype": str(tangent.dtype),
            "kernel_shape": list(kernel_shape),
            "stride": list(stride),
            "padding": padding,
            "dilation": list(dilation),
        }
        if call not in calls:
            calls.append(call)
        return original(
            tangent, kernel_shape, stride=stride, padding=padding, dilation=dilation
        )

    hnnpp._extract_patches = recording
    try:
        jax.eval_shape(function, state, inputs, curvature)
    finally:
        hnnpp._extract_patches = original
    return calls


def main() -> None:
    arguments = _parser().parse_args()

    import jax
    import jax.numpy as jnp
    import jaxlib

    from poincare_exec.artifacts.provenance import git_provenance, timestamp_utc
    from poincare_exec.config.loader import load_yaml
    from poincare_exec.core import hnnpp
    from poincare_exec.experiments.benchmark import _operator_state

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    protocol = load_yaml(arguments.protocol)
    config = load_yaml(arguments.config)
    seed = int(protocol["seeds"]["measurement_order"]) + arguments.session
    device = jax.devices()[0]

    patch_results = []
    model_results = []
    for workload in config["workloads"]:
        function, state, inputs, curvature = _operator_state(workload, "R", seed, protocol)
        for index, call in enumerate(_patch_calls(function, state, inputs, curvature)):
            key = jax.random.fold_in(jax.random.PRNGKey(seed), index)
            tangent = jax.random.normal(
                key, tuple(call["input_shape"]), dtype=jnp.dtype(call["dtype"])
            )
            options = {
                "stride": tuple(call["stride"]),
                "padding": call["padding"],
                "dilation": tuple(call["dilation"]),
            }
            kernel_shape = tuple(call["kernel_shape"])
            reference = jax.jit(
                lambda value, kernel_shape=kernel_shape, options=options: (
                    hnnpp._extract_patches(value, kernel_shape, **options)
                )
            )(tangent)
            candidate = jax.jit(
                lambda value, kernel_shape=kernel_shape, options=options: (
                    hnnpp._gather_patches(value, kernel_shape, **options)
                )
            )(tangent)
            patch_results.append(
                {"workload_id": workload["id"], **call, **_compare(reference, candidate)}
            )
            del reference, candidate, tangent
        del function, state, inputs, curvature

        for reference_variant, candidate_variant in PAIRS:
            outputs = {}
            for variant in (reference_variant, candidate_variant):
                function, state, inputs, curvature = _operator_state(
                    workload, variant, seed, protocol
                )
                compiled = jax.jit(function).lower(state, inputs, curvature).compile()
                outputs[variant] = jax.device_get(compiled(state, inputs, curvature))
                del function, state, inputs, curvature, compiled
            model_results.append(
                {
                    "workload_id": workload["id"],
                    "mode": workload["mode"],
                    "reference_variant": reference_variant,
                    "candidate_variant": candidate_variant,
                    **_compare(outputs[reference_variant], outputs[candidate_variant]),
                }
            )

    payload = {
        "analysis_id": "e31_gpu_gather_equivalence",
        "captured_at_utc": timestamp_utc(),
        "git": git_provenance(REPOSITORY_ROOT),
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "device_kind": device.device_kind,
        "environment": {name: os.environ.get(name) for name in ENVIRONMENT_FIELDS},
        "session_seed": seed,
        "patch_comparisons": patch_results,
        "model_comparisons": model_results,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
