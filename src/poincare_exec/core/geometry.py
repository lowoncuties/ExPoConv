"""Poincare-ball operations under positive curvature magnitude ``c``."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from poincare_exec.core.numerics import (
    atanhc,
    atanhc_from_square,
    positive_denominator,
    safe_norm,
    tanhc_from_square,
)

Array = jax.Array


def projection_margin(dtype: Any) -> Array:
    """Return the inspected source projection margin for one floating dtype."""

    resolved = jnp.dtype(dtype)
    value = 1.0e-5 if resolved == jnp.dtype(jnp.float64) else 4.0e-3
    return jnp.asarray(value, dtype=resolved)


def resolve_projection_margin(dtype: Any, margin: Any | None = None) -> Array:
    """Return an explicit scalar projection margin in the computation dtype."""

    resolved = jnp.dtype(dtype)
    result = (
        projection_margin(resolved)
        if margin is None
        else jnp.asarray(margin, dtype=resolved)
    )
    if result.ndim != 0:
        raise ValueError("projection margin must be scalar")
    return result


def validate_curvature(curvature: Array, dtype: Any) -> Array:
    """Cast a positive scalar curvature magnitude for transformation-safe use."""

    result = jnp.asarray(curvature, dtype=dtype)
    if result.ndim != 0:
        raise ValueError("curvature magnitude must be scalar")
    return result


def project_ball(point: Array, curvature: Array, *, margin: Array | None = None) -> Array:
    """Project to the dtype-specific source radius without hiding invalid curvature."""

    curvature = validate_curvature(curvature, point.dtype)
    resolved_margin = resolve_projection_margin(point.dtype, margin)
    maximum = (1.0 - resolved_margin) / jnp.sqrt(curvature)
    norm = safe_norm(point, axis=-1, keepdims=True)
    outside = norm > maximum
    outside_norm = jnp.where(outside, norm, jnp.ones_like(norm))
    scale = jnp.where(outside, maximum / outside_norm, jnp.ones_like(norm))
    projected = point * scale
    valid = jnp.isfinite(curvature) & (curvature > 0.0)
    return jnp.where(valid, projected, jnp.full_like(projected, jnp.nan))


def expmap0(
    tangent: Array,
    curvature: Array,
    *,
    project: bool = True,
    margin: Array | None = None,
) -> Array:
    """Map origin-tangent coordinates to the Poincare ball."""

    curvature = validate_curvature(curvature, tangent.dtype)
    norm_sq = jnp.sum(jnp.square(tangent), axis=-1, keepdims=True)
    radial_sq = curvature * norm_sq
    point = tangent * tanhc_from_square(radial_sq)
    return project_ball(point, curvature, margin=margin) if project else point


def logmap0(point: Array, curvature: Array) -> Array:
    """Map a valid Poincare point to origin-tangent coordinates."""

    curvature = validate_curvature(curvature, point.dtype)
    radius_sq = curvature * jnp.sum(jnp.square(point), axis=-1, keepdims=True)
    return point * atanhc_from_square(radius_sq)


def project_tangent(
    tangent: Array,
    curvature: Array,
    *,
    margin: Array | None = None,
) -> Array:
    """Evaluate ``log0(project(exp0(t)))`` with an identity origin derivative."""

    curvature = validate_curvature(curvature, tangent.dtype)
    resolved_margin = resolve_projection_margin(tangent.dtype, margin)
    maximum = jnp.arctanh(1.0 - resolved_margin) / jnp.sqrt(curvature)
    norm = safe_norm(tangent, axis=-1, keepdims=True)
    outside = norm > maximum
    outside_norm = jnp.where(outside, norm, jnp.ones_like(norm))
    scale = jnp.where(outside, maximum / outside_norm, jnp.ones_like(norm))
    projected = tangent * scale
    valid = jnp.isfinite(curvature) & (curvature > 0.0)
    return jnp.where(valid, projected, jnp.full_like(projected, jnp.nan))


def conformal_factor(point: Array, curvature: Array) -> Array:
    """Return ``2 / (1-c||x||^2)`` along the coordinate axis."""

    curvature = validate_curvature(curvature, point.dtype)
    radius_sq = curvature * jnp.sum(jnp.square(point), axis=-1, keepdims=True)
    return 2.0 / positive_denominator(1.0 - radius_sq)


def mobius_add(left: Array, right: Array, curvature: Array) -> Array:
    """Apply ordered Mobius addition; the operand order is semantically significant."""

    curvature = validate_curvature(curvature, jnp.result_type(left, right))
    left_sq = jnp.sum(jnp.square(left), axis=-1, keepdims=True)
    right_sq = jnp.sum(jnp.square(right), axis=-1, keepdims=True)
    inner = jnp.sum(left * right, axis=-1, keepdims=True)
    numerator = (1.0 + 2.0 * curvature * inner + curvature * right_sq) * left
    numerator += (1.0 - curvature * left_sq) * right
    denominator = 1.0 + 2.0 * curvature * inner
    denominator += jnp.square(curvature) * left_sq * right_sq
    return numerator / positive_denominator(denominator)


def distance(
    left: Array,
    right: Array,
    curvature: Array,
    *,
    keepdims: bool = False,
) -> Array:
    """Return Poincare geodesic distance along the final coordinate axis."""

    curvature = validate_curvature(curvature, jnp.result_type(left, right))
    difference = mobius_add(-left, right, curvature)
    radius = jnp.sqrt(curvature) * safe_norm(
        difference, axis=-1, keepdims=keepdims
    )
    return 2.0 * jnp.arctanh(radius) / jnp.sqrt(curvature)


def logmap(base: Array, point: Array, curvature: Array) -> Array:
    """Return the tangent vector at ``base`` pointing to ``point``."""

    curvature = validate_curvature(curvature, jnp.result_type(base, point))
    difference = mobius_add(-base, point, curvature)
    difference_norm = safe_norm(difference, axis=-1, keepdims=True)
    scaled_norm = jnp.sqrt(curvature) * difference_norm
    return (
        2.0
        * difference
        * atanhc(scaled_norm)
        / conformal_factor(base, curvature)
    )


def expmap(base: Array, tangent: Array, curvature: Array) -> Array:
    """Map a tangent vector at ``base`` to the Poincare ball."""

    curvature = validate_curvature(curvature, jnp.result_type(base, tangent))
    tangent_norm = safe_norm(tangent, axis=-1, keepdims=True)
    scaled_sq = jnp.square(
        0.5 * jnp.sqrt(curvature) * conformal_factor(base, curvature)
    ) * jnp.square(tangent_norm)
    second = tangent * 0.5 * conformal_factor(base, curvature) * tanhc_from_square(
        scaled_sq
    )
    return project_ball(mobius_add(base, second, curvature), curvature)


def gyration(first: Array, second: Array, value: Array, curvature: Array) -> Array:
    """Apply the gyrovector gyration used by source parallel transport."""

    curvature = validate_curvature(curvature, jnp.result_type(first, second, value))
    first_sq = jnp.sum(jnp.square(first), axis=-1, keepdims=True)
    second_sq = jnp.sum(jnp.square(second), axis=-1, keepdims=True)
    first_second = jnp.sum(first * second, axis=-1, keepdims=True)
    first_value = jnp.sum(first * value, axis=-1, keepdims=True)
    second_value = jnp.sum(second * value, axis=-1, keepdims=True)
    curvature_sq = jnp.square(curvature)
    coefficient_first = (
        -curvature_sq * first_value * second_sq
        + curvature * second_value
        + 2.0 * curvature_sq * first_second * second_value
    )
    coefficient_second = -curvature_sq * second_value * first_sq - curvature * first_value
    denominator = 1.0 + 2.0 * curvature * first_second
    denominator += curvature_sq * first_sq * second_sq
    return value + 2.0 * (
        coefficient_first * first + coefficient_second * second
    ) / positive_denominator(denominator)


def parallel_transport(
    source: Array,
    target: Array,
    tangent: Array,
    curvature: Array,
) -> Array:
    """Parallel transport using the exact source gyrovector expression."""

    ratio = conformal_factor(source, curvature) / conformal_factor(target, curvature)
    return ratio * gyration(target, -source, tangent, curvature)
