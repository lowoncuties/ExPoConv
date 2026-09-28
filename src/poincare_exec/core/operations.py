"""Operator-preserving building blocks used by the complete models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from poincare_exec.core.geometry import (
    distance,
    expmap,
    expmap0,
    logmap,
    logmap0,
    mobius_add,
    parallel_transport,
    project_tangent,
)
from poincare_exec.core.hnnpp import HNNPPParams, beta_ratio, dense, spatial_pair
from poincare_exec.core.numerics import positive_denominator

Array = jax.Array


class MidpointBatchNormParams(NamedTuple):
    """Source-compatible learnable midpoint-normalization parameters."""

    bias: Array
    weight: Array


def init_midpoint_batch_norm(
    channels: int, *, dtype=jnp.float32
) -> MidpointBatchNormParams:
    """Initialize the vector bias and shared scalar variance target."""

    if channels <= 0:
        raise ValueError("channels must be positive")
    return MidpointBatchNormParams(
        bias=jnp.zeros((channels,), dtype=dtype),
        weight=jnp.ones((), dtype=dtype),
    )


def poincare_midpoint(point: Array, curvature: Array, *, axis: int = 0) -> Array:
    """Compute the closed-form midpoint used by both inspected source projects."""

    curvature = jnp.asarray(curvature, dtype=point.dtype)
    radius_sq = curvature * jnp.sum(jnp.square(point), axis=-1, keepdims=True)
    gamma_sq = 1.0 / positive_denominator(1.0 - radius_sq)
    numerator = jnp.sum(gamma_sq * point, axis=axis, keepdims=True)
    count = point.shape[axis]
    denominator = jnp.sum(gamma_sq, axis=axis, keepdims=True) - 0.5 * count
    fraction = numerator / positive_denominator(denominator)
    fraction_radius_sq = curvature * jnp.sum(
        jnp.square(fraction), axis=-1, keepdims=True
    )
    gamma = 1.0 / jnp.sqrt(positive_denominator(1.0 - fraction_radius_sq))
    return gamma * fraction / (1.0 + gamma)


def midpoint_batch_norm(
    params: MidpointBatchNormParams,
    tangent: Array,
    curvature: Array,
    *,
    epsilon: float = 1.0e-6,
) -> Array:
    """Apply current-batch Poincare BN at a tangent-coordinate boundary.

    Spatial axes are folded into the sample axis, matching the inspected
    ``HBatchNorm2d`` and Poincare-ResNet behavior. There are deliberately no
    running statistics because neither source path uses them for evaluation.
    """

    if tangent.shape[-1] != params.bias.shape[0]:
        raise ValueError("normalization bias and channel dimensions do not agree")
    original_shape = tangent.shape
    flattened = jnp.reshape(tangent, (-1, original_shape[-1]))
    point = expmap0(flattened, curvature)
    mean = poincare_midpoint(point, curvature, axis=0)
    variance = jnp.mean(jnp.square(distance(mean, point, curvature)))
    bias_point = expmap0(params.bias, curvature)
    centered = logmap(mean, point, curvature)
    transported = parallel_transport(mean, bias_point, centered, curvature)
    scale = jnp.sqrt(params.weight / (variance + epsilon))
    output_point = expmap(bias_point, scale * transported, curvature)
    return jnp.reshape(logmap0(output_point, curvature), original_shape)


def max_pool2d(
    tangent: Array,
    kernel_size: int | Sequence[int],
    *,
    stride: int | Sequence[int] | None = None,
    padding: str = "VALID",
) -> Array:
    """Apply source HMaxPool2d directly in origin-tangent coordinates."""

    kernel = spatial_pair(kernel_size, name="kernel_size")
    strides = kernel if stride is None else spatial_pair(stride, name="stride")
    return lax.reduce_window(
        tangent,
        -jnp.inf,
        lax.max,
        window_dimensions=(1, kernel[0], kernel[1], 1),
        window_strides=(1, strides[0], strides[1], 1),
        padding=padding.upper(),
    )


def average_pool2d(
    tangent: Array,
    kernel_size: int | Sequence[int],
    *,
    stride: int | Sequence[int] | None = None,
    padding: str = "VALID",
) -> Array:
    """Apply the Poincare-ResNet's Euclidean pooling to tangent features."""

    kernel = spatial_pair(kernel_size, name="kernel_size")
    strides = kernel if stride is None else spatial_pair(stride, name="stride")
    summed = lax.reduce_window(
        tangent,
        # A scalar identity selects JAX's differentiable reduce-window-sum
        # primitive. An array initializer takes the generic reducer path in
        # JAX 0.4.25, whose reverse-mode rule is intentionally unavailable.
        0.0,
        lax.add,
        window_dimensions=(1, kernel[0], kernel[1], 1),
        window_strides=(1, strides[0], strides[1], 1),
        padding=padding.upper(),
    )
    return summed / (kernel[0] * kernel[1])


def mobius_residual_tangent(
    transformed: Array,
    residual: Array,
    curvature: Array,
    *,
    order: str = "fr",
) -> Array:
    """Preserve the ordered Mobius residual at a tangent boundary."""

    transformed_point = expmap0(transformed, curvature)
    residual_point = expmap0(residual, curvature)
    if order == "fr":
        output = mobius_add(transformed_point, residual_point, curvature)
    elif order == "rf":
        output = mobius_add(residual_point, transformed_point, curvature)
    else:
        raise ValueError("residual order must be 'fr' or 'rf'")
    return logmap0(output, curvature)


def _fold_expanded_tangent(
    expanded: Array,
    *,
    kernel_shape: tuple[int, int],
    output_channels: int,
    stride: tuple[int, int],
    padding: tuple[int, int],
) -> Array:
    """Fold per-input-pixel kernel contributions with overlap summation."""

    batch, input_height, input_width, _ = expanded.shape
    kernel_height, kernel_width = kernel_shape
    full_height = (input_height - 1) * stride[0] + kernel_height
    full_width = (input_width - 1) * stride[1] + kernel_width
    expanded = jnp.reshape(
        expanded,
        (
            batch,
            input_height,
            input_width,
            kernel_height,
            kernel_width,
            output_channels,
        ),
    )
    output = jnp.zeros(
        (batch, full_height, full_width, output_channels), dtype=expanded.dtype
    )
    for row in range(kernel_height):
        for column in range(kernel_width):
            output = output.at[
                :,
                row : row + input_height * stride[0] : stride[0],
                column : column + input_width * stride[1] : stride[1],
                :,
            ].add(expanded[:, :, :, row, column, :])
    pad_height, pad_width = padding
    if pad_height or pad_width:
        output = output[
            :,
            pad_height : full_height - pad_height,
            pad_width : full_width - pad_width,
            :,
        ]
    return output


def _nonnegative_pair(
    value: int | Sequence[int], *, name: str
) -> tuple[int, int]:
    values = (value, value) if isinstance(value, int) else tuple(value)
    if len(values) != 2 or any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0
        for item in values
    ):
        raise ValueError(f"{name} must contain two nonnegative Python integers")
    return values


def transpose_conv2d(
    params: HNNPPParams,
    tangent: Array,
    curvature: Array,
    *,
    output_channels: int,
    kernel_size: int | Sequence[int] = 2,
    stride: int | Sequence[int] = 2,
    padding: int | Sequence[int] = 0,
    variant: str,
) -> Array:
    """Apply the source FC-plus-Poincare-fold transpose convolution."""

    kernel = spatial_pair(kernel_size, name="kernel_size")
    strides = spatial_pair(stride, name="stride")
    paddings = _nonnegative_pair(padding, name="padding")
    expected = kernel[0] * kernel[1] * output_channels
    if params.weight.ndim != 2 or params.weight.shape[1] != expected:
        raise ValueError("transpose parameters do not match kernel/output channels")
    expanded = dense(params, tangent, curvature, variant=variant)
    stacked_dimension = expected
    fold_scale = 1.0 / beta_ratio(
        stacked_dimension, output_channels, dtype=tangent.dtype
    )
    folded = _fold_expanded_tangent(
        fold_scale * expanded,
        kernel_shape=kernel,
        output_channels=output_channels,
        stride=strides,
        padding=paddings,
    )
    if variant == "T":
        return project_tangent(folded, curvature)
    return logmap0(expmap0(folded, curvature), curvature)


