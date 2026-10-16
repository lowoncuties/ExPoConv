"""Shared model initialization and application helpers."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import jax
import jax.numpy as jnp

from poincare_exec.core.hnnpp import HNNPPParams, conv2d, init_conv2d
from poincare_exec.core.operations import (
    MidpointBatchNormParams,
    euclidean_batch_norm,
    init_midpoint_batch_norm,
    midpoint_batch_norm,
)

Array = jax.Array
ParameterTree = dict[str, Any]


def key_stream(key: Array) -> Iterator[Array]:
    """Yield deterministic subkeys without maintaining global RNG state."""

    while True:
        key, selected = jax.random.split(key)
        yield selected


def init_double_conv(
    keys: Iterator[Array],
    input_channels: int,
    output_channels: int,
    *,
    middle_channels: int | None = None,
    dtype: Any = jnp.float32,
) -> ParameterTree:
    """Initialize a source-shaped Conv-BN-ReLU-Conv-BN-ReLU block."""

    middle = output_channels if middle_channels is None else middle_channels
    return {
        "conv1": init_conv2d(
            next(keys), 3, input_channels, middle, dtype=dtype, initialization="source"
        ),
        "bn1": init_midpoint_batch_norm(middle, dtype=dtype),
        "conv2": init_conv2d(
            next(keys), 3, middle, output_channels, dtype=dtype, initialization="source"
        ),
        "bn2": init_midpoint_batch_norm(output_channels, dtype=dtype),
    }


def _zero_offset(params: HNNPPParams) -> HNNPPParams:
    """Represent the source U-Net's bias-free convolution explicitly."""

    return HNNPPParams(
        weight=params.weight,
        offset=jax.lax.stop_gradient(jnp.zeros_like(params.offset)),
    )


def apply_double_conv(
    params: ParameterTree,
    tangent: Array,
    curvature: Array,
    *,
    variant: str,
    epsilon: float,
    convolution_bias: bool,
) -> Array:
    """Apply a complete double-convolution block in BHWC layout."""

    conv1 = params["conv1"] if convolution_bias else _zero_offset(params["conv1"])
    conv2 = params["conv2"] if convolution_bias else _zero_offset(params["conv2"])

    def normalize(normalization_params: MidpointBatchNormParams, value: Array) -> Array:
        if variant == "E":
            return euclidean_batch_norm(normalization_params, value, epsilon=epsilon)
        return midpoint_batch_norm(normalization_params, value, curvature, epsilon=epsilon)

    tangent = conv2d(conv1, tangent, curvature, variant=variant, padding="SAME")
    tangent = normalize(params["bn1"], tangent)
    tangent = jax.nn.relu(tangent)
    tangent = conv2d(conv2, tangent, curvature, variant=variant, padding="SAME")
    tangent = normalize(params["bn2"], tangent)
    return jax.nn.relu(tangent)


def count_parameters(params: Any) -> int:
    """Count scalar array leaves in a model tree."""

    return sum(int(leaf.size) for leaf in jax.tree_util.tree_leaves(params))


def normalization_parameters(params: MidpointBatchNormParams) -> tuple[Array, Array]:
    """Expose normalization groups for numerical result labeling."""

    return params.bias, params.weight
