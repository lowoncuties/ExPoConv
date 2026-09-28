"""Stable scalar/vector primitives shared by the geometric kernels."""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp

from poincare_exec.config.constants import (
    MIN_POSITIVE_DENOMINATOR,
    SERIES_CUTOFF_FLOAT32,
    SERIES_CUTOFF_FLOAT64,
)

Array = jax.Array


def _stable_norm_primal(
    value: Array,
    axes: tuple[int, ...],
    keepdims: bool,
) -> Array:
    scale = jnp.max(jnp.abs(value), axis=axes, keepdims=True)
    safe_scale = jnp.where(scale > 0.0, scale, jnp.ones_like(scale))
    scaled = value / safe_scale
    scaled_square_sum = jnp.sum(jnp.square(scaled), axis=axes, keepdims=True)
    safe_square_sum = jnp.where(
        scaled_square_sum > 0.0,
        scaled_square_sum,
        jnp.ones_like(scaled_square_sum),
    )
    result = safe_scale * jnp.sqrt(safe_square_sum)
    result = jnp.where(scale > 0.0, result, jnp.zeros_like(result))
    if keepdims:
        return result
    return jnp.squeeze(result, axis=axes)


@partial(jax.custom_jvp, nondiff_argnums=(1, 2))
def _stable_norm(
    value: Array,
    axes: tuple[int, ...],
    keepdims: bool,
) -> Array:
    return _stable_norm_primal(value, axes, keepdims)


@_stable_norm.defjvp
def _stable_norm_jvp(
    axes: tuple[int, ...],
    keepdims: bool,
    primals: tuple[Array],
    tangents: tuple[Array],
) -> tuple[Array, Array]:
    """Use ``x / ||x||`` directly, avoiding reciprocal-square overflow."""

    (value,), (tangent,) = primals, tangents
    result = _stable_norm_primal(value, axes, keepdims)
    result_with_axes = result
    if not keepdims:
        for axis in axes:
            result_with_axes = jnp.expand_dims(result_with_axes, axis=axis)
    denominator = jnp.where(
        result_with_axes > 0.0,
        result_with_axes,
        jnp.ones_like(result_with_axes),
    )
    unit = value / denominator
    derivative = jnp.sum(unit * tangent, axis=axes, keepdims=keepdims)
    return result, derivative


def as_real_array(value: Any, *, name: str) -> Array:
    """Return a floating JAX array while rejecting bool/complex operands."""

    array = jnp.asarray(value)
    if jnp.issubdtype(array.dtype, jnp.bool_) or jnp.issubdtype(
        array.dtype, jnp.complexfloating
    ):
        raise TypeError(f"{name} must be real-valued")
    if not jnp.issubdtype(array.dtype, jnp.floating):
        array = array.astype(jnp.float32)
    return array


def series_cutoff(dtype: jnp.dtype) -> float:
    """Return the fixed analytic-series threshold for one dtype."""

    return (
        SERIES_CUTOFF_FLOAT64
        if jnp.dtype(dtype) == jnp.dtype(jnp.float64)
        else SERIES_CUTOFF_FLOAT32
    )


def safe_norm(
    value: Array,
    *,
    axis: int | Sequence[int] = -1,
    keepdims: bool = False,
) -> Array:
    """Compute a stable norm with a zero derivative convention at exact zero.

    The custom JVP evaluates the nonzero derivative as ``x / ||x||`` without
    differentiating through reciprocal scaling. This avoids FP32 overflow for
    tiny, nonzero vectors while retaining the source-compatible zero-gradient
    convention for the norm itself at the nondifferentiable origin.
    """

    raw_axes = (axis,) if isinstance(axis, int) else tuple(axis)
    axes = tuple(sorted(item % value.ndim for item in raw_axes))
    if len(set(axes)) != len(axes):
        raise ValueError("norm axes must be unique")
    return _stable_norm(value, axes, keepdims)


def _series_or_direct(
    value: Array,
    direct,
    series,
) -> Array:
    cutoff = jnp.asarray(series_cutoff(value.dtype), dtype=value.dtype)
    use_direct = jnp.abs(value) > cutoff
    direct_input = jnp.where(use_direct, value, jnp.ones_like(value))
    series_input = jnp.where(use_direct, jnp.zeros_like(value), value)
    return jnp.where(use_direct, direct(direct_input), series(series_input))


def sinhc(value: Array) -> Array:
    """Evaluate ``sinh(x) / x`` with its analytic continuation at zero."""

    return _series_or_direct(
        value,
        lambda x: jnp.sinh(x) / x,
        lambda x: 1.0
        + jnp.square(x)
        * (1.0 / 6.0 + jnp.square(x) * (1.0 / 120.0 + jnp.square(x) / 5040.0)),
    )


def asinhc(value: Array) -> Array:
    """Evaluate ``asinh(x) / x`` with its analytic continuation at zero."""

    return _series_or_direct(
        value,
        lambda x: jnp.arcsinh(x) / x,
        lambda x: 1.0
        + jnp.square(x)
        * (-1.0 / 6.0 + jnp.square(x) * (3.0 / 40.0 - 5.0 * jnp.square(x) / 112.0)),
    )


def asinhc_from_square(square: Array) -> Array:
    """Evaluate ``asinh(sqrt(q)) / sqrt(q)`` from a squared norm.

    Accepting the squared norm avoids differentiating a vector norm at the
    origin and prevents reciprocal-square overflow for tiny FP32 vectors.
    """

    cutoff_sq = jnp.asarray(series_cutoff(square.dtype) ** 2, dtype=square.dtype)
    use_direct = square > cutoff_sq
    direct_square = jnp.where(use_direct, square, jnp.ones_like(square))
    direct_root = jnp.sqrt(direct_square)
    direct = jnp.arcsinh(direct_root) / direct_root
    series_square = jnp.where(use_direct, jnp.zeros_like(square), square)
    series = 1.0 + series_square * (
        -1.0 / 6.0
        + series_square * (3.0 / 40.0 - 5.0 * series_square / 112.0)
    )
    return jnp.where(use_direct, direct, series)


def atanhc(value: Array) -> Array:
    """Evaluate ``atanh(x) / x`` with its analytic continuation at zero."""

    return _series_or_direct(
        value,
        lambda x: jnp.arctanh(x) / x,
        lambda x: 1.0
        + jnp.square(x)
        * (1.0 / 3.0 + jnp.square(x) * (1.0 / 5.0 + jnp.square(x) / 7.0)),
    )


def atanhc_from_square(square: Array) -> Array:
    """Evaluate ``atanh(sqrt(q)) / sqrt(q)`` from a squared norm."""

    cutoff_sq = jnp.asarray(series_cutoff(square.dtype) ** 2, dtype=square.dtype)
    use_direct = square > cutoff_sq
    direct_square = jnp.where(use_direct, square, jnp.ones_like(square))
    direct_root = jnp.sqrt(direct_square)
    direct = jnp.arctanh(direct_root) / direct_root
    series_square = jnp.where(use_direct, jnp.zeros_like(square), square)
    series = 1.0 + series_square * (
        1.0 / 3.0 + series_square * (1.0 / 5.0 + series_square / 7.0)
    )
    return jnp.where(use_direct, direct, series)


def tanhc_from_square(square: Array) -> Array:
    """Evaluate ``tanh(sqrt(q))/sqrt(q)`` without a zero sqrt derivative."""

    cutoff_sq = jnp.asarray(series_cutoff(square.dtype) ** 2, dtype=square.dtype)
    use_direct = square > cutoff_sq
    direct_square = jnp.where(use_direct, square, jnp.ones_like(square))
    direct_root = jnp.sqrt(direct_square)
    direct = jnp.tanh(direct_root) / direct_root
    series_square = jnp.where(use_direct, jnp.zeros_like(square), square)
    series = 1.0 + series_square * (
        -1.0 / 3.0 + series_square * (2.0 / 15.0 - 17.0 * series_square / 315.0)
    )
    return jnp.where(use_direct, direct, series)


def tanh_square_from_square(square: Array) -> Array:
    """Evaluate ``tanh(sqrt(q))**2`` with a smooth zero continuation."""

    cutoff_sq = jnp.asarray(series_cutoff(square.dtype) ** 2, dtype=square.dtype)
    use_direct = square > cutoff_sq
    direct_square = jnp.where(use_direct, square, jnp.ones_like(square))
    direct = jnp.square(jnp.tanh(jnp.sqrt(direct_square)))
    series_square = jnp.where(use_direct, jnp.zeros_like(square), square)
    series = series_square * (
        1.0 + series_square * (-2.0 / 3.0 + 17.0 * series_square / 45.0)
    )
    return jnp.where(use_direct, direct, series)


def positive_denominator(value: Array) -> Array:
    """Apply the source libraries' positive denominator floor."""

    floor = jnp.asarray(MIN_POSITIVE_DENOMINATOR, dtype=value.dtype)
    return jnp.maximum(value, floor)


def asinh_exp(log_value: Array) -> Array:
    """Evaluate ``asinh(exp(log_value))`` without materializing overflow."""

    positive = log_value >= 0.0
    positive_input = jnp.where(positive, log_value, jnp.zeros_like(log_value))
    negative_input = jnp.where(positive, jnp.zeros_like(log_value), log_value)
    positive_result = positive_input + jnp.log1p(
        jnp.sqrt(1.0 + jnp.exp(-2.0 * positive_input))
    )
    negative_result = jnp.arcsinh(jnp.exp(negative_input))
    return jnp.where(positive, positive_result, negative_result)
