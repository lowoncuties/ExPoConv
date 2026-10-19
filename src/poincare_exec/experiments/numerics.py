"""Frozen E02 case-manifest generation and shared numerical-comparison helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from poincare_exec.artifacts import (
    write_json,
)
from poincare_exec.config.loader import load_yaml
from poincare_exec.core.hnnpp import (
    HNNPPParams,
    _extract_patches,
    beta_ratio,
    conv2d,
    init_conv2d,
)
from poincare_exec.core.numerics import safe_norm

Array = jax.Array

CASE_GENERATOR_VERSION = "1.0"
FACTORS: dict[str, tuple[Any, ...]] = {
    "precision": ("float32", "float64"),
    "curvature": (0.001, 0.1, 1.0, 10.0),
    "normalized_radius": (0.0, 1.0e-6, 0.1, 0.5, 0.9, 0.99, 0.9999, 0.999999),
    "batch": (1, 2),
    "height": (1, 3, 5),
    "width": (1, 3, 5),
    "input_channels": (1, 3, 8),
    "output_channels": (1, 3, 8),
    "kernel_size": (1, 3),
    "offset_case": ("zero", "nonzero"),
    "orientation_case": ("source", "near_zero", "zero_column"),
    "direction_case": ("random", "aligned", "anti_aligned"),
}


def _uncovered_pairs() -> set[tuple[int, int, int, int]]:
    sizes = [len(values) for values in FACTORS.values()]
    return {
        (left, left_value, right, right_value)
        for left in range(len(sizes))
        for right in range(left + 1, len(sizes))
        for left_value in range(sizes[left])
        for right_value in range(sizes[right])
    }


def _covered_pairs(candidate: tuple[int, ...]) -> set[tuple[int, int, int, int]]:
    return {
        (left, candidate[left], right, candidate[right])
        for left in range(len(candidate))
        for right in range(left + 1, len(candidate))
    }


def _pairwise_rows(count: int, seed: int) -> list[tuple[int, ...]]:
    """Construct a deterministic greedy pairwise cover and pad to ``count``."""

    rng = np.random.default_rng(seed)
    sizes = np.asarray([len(values) for values in FACTORS.values()], dtype=np.int64)
    uncovered = _uncovered_pairs()
    selected: list[tuple[int, ...]] = []
    selected_set: set[tuple[int, ...]] = set()
    while uncovered:
        candidates = rng.integers(0, sizes, size=(4096, len(sizes)))
        best: tuple[int, ...] | None = None
        best_score = -1
        for raw_candidate in candidates:
            candidate = tuple(int(value) for value in raw_candidate)
            score = len(_covered_pairs(candidate) & uncovered)
            if score > best_score and candidate not in selected_set:
                best, best_score = candidate, score
        if best is None or best_score <= 0:
            left, left_value, right, right_value = min(uncovered)
            raw = rng.integers(0, sizes)
            raw[left], raw[right] = left_value, right_value
            best = tuple(int(value) for value in raw)
        selected.append(best)
        selected_set.add(best)
        uncovered.difference_update(_covered_pairs(best))
        if len(selected) > count:
            raise RuntimeError("requested case count is insufficient for pairwise coverage")
    while len(selected) < count:
        candidate = tuple(int(value) for value in rng.integers(0, sizes))
        if candidate not in selected_set:
            selected.append(candidate)
            selected_set.add(candidate)
    return selected


def generate_case_manifest(protocol: dict[str, Any]) -> dict[str, Any]:
    """Generate exactly the preregistered 100 pairwise-covering cases."""

    numerics = protocol["numerics"]
    count = int(numerics["configurations"])
    seed = int(protocol["seeds"]["numerical"])
    names = tuple(FACTORS)
    rows = _pairwise_rows(count, seed)
    cases = []
    for index, row in enumerate(rows):
        case = {
            name: FACTORS[name][value_index]
            for name, value_index in zip(names, row, strict=True)
        }
        case.update(
            {
                "case_id": f"N{index:03d}",
                "padding": "SAME",
                "stride": 1,
                "dilation": 1,
                "realizations": int(numerics["realizations_per_configuration"]),
            }
        )
        cases.append(case)
    return {
        "schema_version": "1.0",
        "generator_version": CASE_GENERATOR_VERSION,
        "protocol_version": protocol["protocol_version"],
        "seed": seed,
        "coverage": "all value pairs across every named factor",
        "factor_values": {name: list(values) for name, values in FACTORS.items()},
        "cases": cases,
    }


def _assert_manifest(manifest: dict[str, Any], protocol: dict[str, Any]) -> None:
    cases = manifest.get("cases")
    if not isinstance(cases, list) or len(cases) != protocol["numerics"]["configurations"]:
        raise ValueError("case manifest has the wrong number of configurations")
    uncovered = _uncovered_pairs()
    names = tuple(FACTORS)
    for case in cases:
        encoded = tuple(FACTORS[name].index(case[name]) for name in names)
        uncovered.difference_update(_covered_pairs(encoded))
    if uncovered:
        raise ValueError(f"case manifest misses {len(uncovered)} factor-value pairs")


def write_case_manifest(
    protocol_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Create and validate the immutable numerical case manifest."""

    protocol = load_yaml(protocol_path)
    manifest = generate_case_manifest(protocol)
    _assert_manifest(manifest, protocol)
    write_json(output_path, manifest)
    return manifest


def _case_parameters(case: dict[str, Any], key: Array, dtype: Any) -> HNNPPParams:
    offset_scale = 0.15 if case["offset_case"] == "nonzero" else 0.0
    params = init_conv2d(
        key,
        case["kernel_size"],
        case["input_channels"],
        case["output_channels"],
        dtype=dtype,
        offset_scale=offset_scale,
        initialization="source",
    )
    if case["orientation_case"] == "near_zero":
        params = HNNPPParams(weight=1.0e-12 * params.weight, offset=params.offset)
    elif case["orientation_case"] == "zero_column":
        params = HNNPPParams(weight=params.weight.at[..., 0].set(0.0), offset=params.offset)
    return params


def _case_input(
    case: dict[str, Any], params: HNNPPParams, key: Array, dtype: Any
) -> tuple[Array, float]:
    shape = (
        case["batch"],
        case["height"],
        case["width"],
        case["input_channels"],
    )
    tangent = jax.random.normal(key, shape, dtype=dtype)
    if case["direction_case"] in {"aligned", "anti_aligned"}:
        channel_direction = jnp.sum(params.weight[..., 0], axis=(0, 1))
        direction_norm = safe_norm(channel_direction, axis=-1, keepdims=True)
        fallback = jnp.ones_like(channel_direction)
        channel_direction = jnp.where(direction_norm > 0.0, channel_direction, fallback)
        sign = 1.0 if case["direction_case"] == "aligned" else -1.0
        tangent = sign * jnp.broadcast_to(channel_direction, shape)
    kernel = (case["kernel_size"], case["kernel_size"])
    patches = _extract_patches(
        tangent,
        kernel,
        stride=(1, 1),
        padding="SAME",
        dilation=(1, 1),
    )
    patch_dimension = case["kernel_size"] ** 2 * case["input_channels"]
    ratio = beta_ratio(patch_dimension, case["input_channels"], dtype=dtype)
    anchor = ratio * patches[0, case["height"] // 2, case["width"] // 2]
    anchor_norm = safe_norm(anchor, axis=-1)
    target_rho = jnp.asarray(case["normalized_radius"], dtype=dtype)
    curvature = jnp.asarray(case["curvature"], dtype=dtype)
    target_norm = jnp.arctanh(target_rho) / jnp.sqrt(curvature)
    scale = jnp.where(anchor_norm > 0.0, target_norm / anchor_norm, 0.0)
    tangent = tangent * scale
    realized_anchor = (
        ratio
        * _extract_patches(
            tangent,
            kernel,
            stride=(1, 1),
            padding="SAME",
            dilation=(1, 1),
        )[0, case["height"] // 2, case["width"] // 2]
    )
    realized_rho = jnp.tanh(jnp.sqrt(curvature) * safe_norm(realized_anchor, axis=-1))
    return tangent, float(np.asarray(realized_rho))


def _cast_tree(tree: Any, dtype: Any) -> Any:
    return jax.tree_util.tree_map(lambda value: jnp.asarray(value, dtype=dtype), tree)


def _evaluate(
    variant: str,
    params: HNNPPParams,
    tangent: Array,
    curvature: Array,
    cotangent: Array,
    *,
    margin: float | Array | None = None,
) -> tuple[Array, tuple[Any, Array, Array]]:
    def scalarized(selected_params, selected_tangent, selected_curvature):
        output = conv2d(
            selected_params,
            selected_tangent,
            selected_curvature,
            variant=variant,
            padding="SAME",
            margin=margin,
        )
        return jnp.sum(output * cotangent), output

    (_, output), gradients = jax.value_and_grad(
        scalarized, argnums=(0, 1, 2), has_aux=True
    )(params, tangent, curvature)
    jax.block_until_ready((output, gradients))
    return output, gradients


def _finite_metrics(reference: Array, candidate: Array, floor: float) -> dict[str, Any]:
    reference_np = np.asarray(reference, dtype=np.float64)
    candidate_np = np.asarray(candidate, dtype=np.float64)
    finite = bool(np.isfinite(reference_np).all() and np.isfinite(candidate_np).all())
    if not finite:
        return {
            "max_absolute_error": None,
            "relative_l2_error": None,
            "reference_l2_norm": None,
            "finite": False,
        }
    difference = candidate_np - reference_np
    reference_norm = float(np.linalg.norm(reference_np.ravel()))
    return {
        "max_absolute_error": float(np.max(np.abs(difference), initial=0.0)),
        "relative_l2_error": float(
            np.linalg.norm(difference.ravel()) / max(reference_norm, floor)
        ),
        "reference_l2_norm": reference_norm,
        "finite": True,
    }


def _groups(output: Array, gradients: tuple[HNNPPParams, Array, Array]) -> dict[str, Array]:
    parameter_gradients, input_gradient, curvature_gradient = gradients
    return {
        "output": output,
        "input_gradient": input_gradient,
        "weight_gradient": parameter_gradients.weight,
        "offset_gradient": parameter_gradients.offset,
        "curvature_gradient": curvature_gradient,
    }
