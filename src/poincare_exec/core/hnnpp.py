"""Independent HNN++ reference, factorized, and tangent execution paths."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import betaln

from poincare_exec.config.constants import CONV_DIMENSION_NUMBERS, EXECUTION_VARIANTS
from poincare_exec.core.geometry import (
    expmap0,
    logmap0,
    project_ball,
    project_tangent,
    resolve_projection_margin,
)
from poincare_exec.core.numerics import (
    asinh_exp,
    asinhc_from_square,
    positive_denominator,
    safe_norm,
    sinhc,
    tanh_square_from_square,
    tanhc_from_square,
)

Array = jax.Array


class HNNPPParams(NamedTuple):
    """HNN++ orientation weights and signed hyperplane offsets."""

    weight: Array
    offset: Array


def _positive_size(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a Python integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def spatial_pair(value: int | Sequence[int], *, name: str) -> tuple[int, int]:
    """Normalize a positive scalar or length-two spatial argument."""

    if isinstance(value, int) and not isinstance(value, bool):
        size = _positive_size(value, name=name)
        return size, size
    values = tuple(value)
    if len(values) != 2:
        raise ValueError(f"{name} must have exactly two entries")
    return (
        _positive_size(values[0], name=f"{name}[0]"),
        _positive_size(values[1], name=f"{name}[1]"),
    )


def dimension_beta(dimension: int, *, dtype: Any) -> Array:
    """Return ``B(d/2, 1/2)`` in the requested floating dtype."""

    dimension = _positive_size(dimension, name="dimension")
    resolved = jnp.dtype(dtype)
    half = jnp.asarray(0.5, dtype=resolved)
    return jnp.exp(betaln(half * jnp.asarray(dimension, resolved), half))


def beta_ratio(total_dimension: int, block_dimension: int, *, dtype: Any) -> Array:
    """Return a dimension-beta ratio in log space."""

    total_dimension = _positive_size(total_dimension, name="total_dimension")
    block_dimension = _positive_size(block_dimension, name="block_dimension")
    resolved = jnp.dtype(dtype)
    half = jnp.asarray(0.5, dtype=resolved)
    total = betaln(half * jnp.asarray(total_dimension, resolved), half)
    block = betaln(half * jnp.asarray(block_dimension, resolved), half)
    return jnp.exp(total - block)


def init_dense(
    key: Array,
    input_dim: int,
    output_dim: int,
    *,
    dtype: Any = jnp.float32,
    weight_scale: float = 1.0,
    offset_scale: float = 0.0,
    initialization: str = "normal",
) -> HNNPPParams:
    """Initialize HNN++ orientations under a named reproducible convention.

    ``source`` reproduces HyPLL/Poincare-ResNet: half an identity when the
    input width does not exceed the output width, otherwise a zero-mean normal
    with standard deviation ``(2 * input_dim * output_dim)**-0.5``.
    """

    input_dim = _positive_size(input_dim, name="input_dim")
    output_dim = _positive_size(output_dim, name="output_dim")
    weight_key, offset_key = jax.random.split(key)
    if initialization == "normal":
        weight = jax.random.normal(weight_key, (input_dim, output_dim), dtype=dtype)
        weight *= jnp.asarray(weight_scale / math.sqrt(input_dim), dtype=dtype)
        column_norm = safe_norm(weight, axis=0, keepdims=True)
        fallback = jnp.zeros_like(weight).at[0, :].set(1.0)
        weight = jnp.where(column_norm > 0.0, weight, fallback)
    elif initialization == "source":
        if input_dim <= output_dim:
            weight = 0.5 * jnp.eye(input_dim, output_dim, dtype=dtype)
        else:
            source_scale = (2.0 * input_dim * output_dim) ** -0.5
            weight = source_scale * jax.random.normal(
                weight_key, (input_dim, output_dim), dtype=dtype
            )
    else:
        raise ValueError("initialization must be 'normal' or 'source'")
    offset = jnp.asarray(offset_scale, dtype=dtype) * jax.random.normal(
        offset_key, (output_dim,), dtype=dtype
    )
    return HNNPPParams(weight=weight, offset=offset)


def init_conv2d(
    key: Array,
    kernel_size: int | Sequence[int],
    input_channels: int,
    output_channels: int,
    *,
    dtype: Any = jnp.float32,
    weight_scale: float = 1.0,
    offset_scale: float = 0.0,
    initialization: str = "normal",
) -> HNNPPParams:
    """Initialize a literal HWIO kernel and HNN++ offset vector."""

    kernel_height, kernel_width = spatial_pair(kernel_size, name="kernel_size")
    input_channels = _positive_size(input_channels, name="input_channels")
    output_channels = _positive_size(output_channels, name="output_channels")
    dense_params = init_dense(
        key,
        kernel_height * kernel_width * input_channels,
        output_channels,
        dtype=dtype,
        weight_scale=weight_scale,
        offset_scale=offset_scale,
        initialization=initialization,
    )
    kernel = jnp.reshape(
        dense_params.weight,
        (kernel_height, kernel_width, input_channels, output_channels),
    )
    return HNNPPParams(weight=kernel, offset=dense_params.offset)


def hyperplane_scores_ball(
    point: Array,
    weight: Array,
    offset: Array,
    curvature: Array,
) -> Array:
    """Evaluate the published HNN++ hyperplane scores from ball coordinates."""

    dtype = jnp.result_type(point, weight, offset, curvature)
    point = jnp.asarray(point, dtype=dtype)
    weight = jnp.asarray(weight, dtype=dtype)
    offset = jnp.asarray(offset, dtype=dtype)
    curvature = jnp.asarray(curvature, dtype=dtype)
    sqrt_curvature = jnp.sqrt(curvature)
    weight_norm = safe_norm(weight, axis=0)
    safe_weight_norm = jnp.maximum(
        weight_norm, jnp.asarray(1.0e-15, dtype=weight_norm.dtype)
    )
    radius_sq = curvature * jnp.sum(jnp.square(point), axis=-1, keepdims=True)
    conformal = 2.0 / positive_denominator(1.0 - radius_sq)
    oriented = jnp.einsum("...i,io->...o", point, weight) / safe_weight_norm
    doubled_offset = 2.0 * sqrt_curvature * offset
    argument = sqrt_curvature * conformal * oriented * jnp.cosh(doubled_offset)
    argument -= (conformal - 1.0) * jnp.sinh(doubled_offset)
    scores = 2.0 * safe_weight_norm * jnp.arcsinh(argument) / sqrt_curvature
    valid = jnp.isfinite(curvature) & (curvature > 0.0)
    return jnp.where(valid, scores, jnp.full_like(scores, jnp.nan))


def _reconstruct_scores(scores: Array, curvature: Array) -> tuple[Array, Array]:
    """Return stable ball and origin-tangent reconstructions of HNN++ scores."""

    sqrt_curvature = jnp.sqrt(curvature)
    dimensionless = sqrt_curvature * scores
    maximum = jax.lax.stop_gradient(jnp.max(jnp.abs(dimensionless), axis=-1, keepdims=True))
    use_scaled = maximum > jnp.asarray(20.0, dtype=scores.dtype)

    direct_scores = jnp.where(use_scaled, jnp.zeros_like(scores), scores)
    direct_dimensionless = sqrt_curvature * direct_scores
    direct_sinh = jnp.sinh(direct_dimensionless)
    direct_norm_sq = jnp.sum(jnp.square(direct_sinh), axis=-1, keepdims=True)
    direct_point = direct_sinh / (sqrt_curvature * (1.0 + jnp.sqrt(1.0 + direct_norm_sq)))
    direct_tangent = (
        0.5
        * direct_scores
        * sinhc(direct_dimensionless)
        * asinhc_from_square(direct_norm_sq)
    )

    scaled_dimensionless = jnp.where(
        use_scaled, dimensionless, jnp.zeros_like(dimensionless)
    )
    scaled_maximum = jax.lax.stop_gradient(
        jnp.max(jnp.abs(scaled_dimensionless), axis=-1, keepdims=True)
    )
    scaled_sinh = 0.5 * (
        jnp.exp(scaled_dimensionless - scaled_maximum)
        - jnp.exp(-scaled_dimensionless - scaled_maximum)
    )
    scaled_norm = safe_norm(scaled_sinh, axis=-1, keepdims=True)
    scaled_safe_norm = jnp.where(scaled_norm > 0.0, scaled_norm, jnp.ones_like(scaled_norm))
    scaled_direction = scaled_sinh / scaled_safe_norm
    log_norm = scaled_maximum + jnp.log(scaled_safe_norm)
    scaled_radial = 0.5 * asinh_exp(log_norm) / sqrt_curvature
    scaled_tangent = scaled_radial * scaled_direction
    scaled_normalized_radius = jnp.tanh(sqrt_curvature * scaled_radial)
    scaled_point = scaled_direction * scaled_normalized_radius / sqrt_curvature

    point = jnp.where(use_scaled, scaled_point, direct_point)
    tangent = jnp.where(use_scaled, scaled_tangent, direct_tangent)
    return point, tangent


def scores_to_ball(
    scores: Array,
    curvature: Array,
    *,
    margin: Array | None = None,
) -> Array:
    """Reconstruct and source-project the HNN++ output point."""

    curvature = jnp.asarray(curvature, dtype=scores.dtype)
    point, _ = _reconstruct_scores(scores, curvature)
    return project_ball(point, curvature, margin=margin)


def scores_to_projected_tangent(
    scores: Array,
    curvature: Array,
    *,
    margin: Array | None = None,
) -> Array:
    """Return ``log0(project(HNN++ output))`` without materializing that point."""

    curvature = jnp.asarray(curvature, dtype=scores.dtype)
    _, tangent = _reconstruct_scores(scores, curvature)
    return project_tangent(tangent, curvature, margin=margin)


def _scores_from_radius_products(
    radius_sq: Array,
    directional_products: Array,
    weight_norm: Array,
    offset: Array,
    curvature: Array,
    *,
    margin: Array | None = None,
) -> Array:
    """Evaluate HNN++ scores from sufficient tangent-patch statistics."""

    dtype = jnp.result_type(radius_sq, directional_products, weight_norm, offset, curvature)
    curvature = jnp.asarray(curvature, dtype=dtype)
    radius_sq = jnp.asarray(radius_sq, dtype=dtype)
    sqrt_curvature = jnp.sqrt(curvature)
    safe_weight_norm = jnp.maximum(
        weight_norm, jnp.asarray(1.0e-15, dtype=weight_norm.dtype)
    )

    dimensionless_radius_sq = curvature * radius_sq
    raw_normalized_sq = tanh_square_from_square(dimensionless_radius_sq)
    maximum = 1.0 - resolve_projection_margin(dtype, margin)
    projected_normalized_sq = jnp.minimum(raw_normalized_sq, jnp.square(maximum))

    raw_radial_over_radius = sqrt_curvature * tanhc_from_square(dimensionless_radius_sq)
    radius = jnp.sqrt(jnp.where(radius_sq > 0.0, radius_sq, jnp.ones_like(radius_sq)))
    clipped_radial_over_radius = maximum / radius
    clipped = raw_normalized_sq > jnp.square(maximum)
    radial_over_radius = jnp.where(
        clipped, clipped_radial_over_radius, raw_radial_over_radius
    )

    conformal = 2.0 / positive_denominator(1.0 - projected_normalized_sq)
    oriented_dimensionless = radial_over_radius * directional_products / safe_weight_norm
    doubled_offset = 2.0 * sqrt_curvature * offset
    argument = conformal * oriented_dimensionless * jnp.cosh(doubled_offset)
    argument -= (conformal - 1.0) * jnp.sinh(doubled_offset)
    scores = 2.0 * safe_weight_norm * jnp.arcsinh(argument) / sqrt_curvature
    valid = jnp.isfinite(curvature) & (curvature > 0.0)
    return jnp.where(valid, scores, jnp.full_like(scores, jnp.nan))


def _variant_output(
    scores: Array,
    curvature: Array,
    variant: str,
    *,
    margin: Array | None = None,
) -> Array:
    if variant == "T":
        return scores_to_projected_tangent(scores, curvature, margin=margin)
    if variant in {"R", "F", "M", "Rg", "Mg"}:
        return logmap0(scores_to_ball(scores, curvature, margin=margin), curvature)
    raise ValueError(f"unsupported hyperbolic variant: {variant}")


def _radius_products(vectors: Array, weight: Array) -> tuple[Array, Array]:
    """Return ``||v||^2`` and ``v . w`` from explicit tangent vectors."""

    radius_sq = jnp.sum(jnp.square(vectors), axis=-1, keepdims=True)
    products = jnp.einsum("...i,io->...o", vectors, weight)
    return radius_sq, products


def dense(
    params: HNNPPParams,
    tangent: Array,
    curvature: Array,
    *,
    variant: str,
    margin: Array | None = None,
) -> Array:
    """Apply one R/F/T/M/E/Rg/Mg dense layer at a common coordinate boundary.

    A dense layer has no patch to materialize, so M and Mg execute as F and Rg
    executes as R. E applies the parameter-shape-matched Euclidean affine map and
    does not use curvature.
    """

    if variant not in EXECUTION_VARIANTS:
        raise ValueError(f"variant must be one of {EXECUTION_VARIANTS}")
    weight = params.weight
    if weight.ndim != 2 or tangent.shape[-1] != weight.shape[0]:
        raise ValueError("dense weight/input dimensions do not agree")
    if params.offset.shape != (weight.shape[1],):
        raise ValueError("dense offset has the wrong shape")
    if variant == "E":
        return jnp.einsum("...i,io->...o", tangent, weight) + params.offset
    if variant in {"R", "Rg"}:
        point = expmap0(tangent, curvature, margin=margin)
        scores = hyperplane_scores_ball(point, weight, params.offset, curvature)
    else:
        radius_sq, products = _radius_products(tangent, weight)
        scores = _scores_from_radius_products(
            radius_sq,
            products,
            safe_norm(weight, axis=0),
            params.offset,
            curvature,
            margin=margin,
        )
    return _variant_output(scores, curvature, variant, margin=margin)


def _extract_patches(
    tangent: Array,
    kernel_shape: tuple[int, int],
    *,
    stride: tuple[int, int],
    padding: str,
    dilation: tuple[int, int],
) -> Array:
    """Materialize literal HWC-ordered patches for variant R."""

    kernel_height, kernel_width = kernel_shape
    input_channels = tangent.shape[-1]
    channel_major = lax.conv_general_dilated_patches(
        tangent,
        filter_shape=kernel_shape,
        window_strides=stride,
        padding=padding,
        rhs_dilation=dilation,
        dimension_numbers=CONV_DIMENSION_NUMBERS,
    )
    batch, output_height, output_width, _ = channel_major.shape
    channel_height_width = jnp.reshape(
        channel_major,
        (batch, output_height, output_width, input_channels, kernel_height, kernel_width),
    )
    height_width_channel = jnp.transpose(channel_height_width, (0, 1, 2, 4, 5, 3))
    return jnp.reshape(
        height_width_channel,
        (
            batch,
            output_height,
            output_width,
            kernel_height * kernel_width * input_channels,
        ),
    )


def _gather_patches(
    tangent: Array,
    kernel_shape: tuple[int, int],
    *,
    stride: tuple[int, int],
    padding: str,
    dilation: tuple[int, int],
) -> Array:
    """Gather the patches of :func:`_extract_patches` from shifted slices (Rg/Mg).

    The input is zero-padded by the convolution's own SAME/VALID rule. The
    ``kernel_height * kernel_width`` strided slices, one per kernel offset in
    row-major order, are concatenated on the channel axis. That yields R's
    ``(kernel_row, kernel_column, channel)`` flattening directly, with no
    one-hot convolution and no patch transpose, as in the HNN++ reference's
    concatenation of shifted ``narrow`` slices.
    """

    kernel_height, kernel_width = kernel_shape
    batch, height, width, channels = tangent.shape
    window = (
        (kernel_height - 1) * dilation[0] + 1,
        (kernel_width - 1) * dilation[1] + 1,
    )
    (top, bottom), (left, right) = lax.padtype_to_pads(
        (height, width), window, stride, padding
    )
    padded = lax.pad(
        tangent,
        jnp.zeros((), dtype=tangent.dtype),
        ((0, 0, 0), (top, bottom, 0), (left, right, 0), (0, 0, 0)),
    )
    output_height = (height + top + bottom - window[0]) // stride[0] + 1
    output_width = (width + left + right - window[1]) // stride[1] + 1
    slices = []
    for row in range(kernel_height):
        for column in range(kernel_width):
            start_row = row * dilation[0]
            start_column = column * dilation[1]
            slices.append(
                lax.slice(
                    padded,
                    (0, start_row, start_column, 0),
                    (
                        batch,
                        start_row + (output_height - 1) * stride[0] + 1,
                        start_column + (output_width - 1) * stride[1] + 1,
                        channels,
                    ),
                    (1, stride[0], stride[1], 1),
                )
            )
    return jnp.concatenate(slices, axis=-1)


def _factorized_conv_scores(
    params: HNNPPParams,
    tangent: Array,
    curvature: Array,
    *,
    stride: tuple[int, int],
    padding: str,
    dilation: tuple[int, int],
    margin: Array | None = None,
) -> Array:
    kernel = params.weight
    kernel_height, kernel_width, input_channels, _ = kernel.shape
    patch_dimension = kernel_height * kernel_width * input_channels
    ratio = beta_ratio(patch_dimension, input_channels, dtype=tangent.dtype)
    products = ratio * lax.conv_general_dilated(
        tangent,
        kernel,
        window_strides=stride,
        padding=padding,
        rhs_dilation=dilation,
        dimension_numbers=CONV_DIMENSION_NUMBERS,
    )
    scalar_input = jnp.sum(jnp.square(tangent), axis=-1, keepdims=True)
    scalar_kernel = jnp.ones((kernel_height, kernel_width, 1, 1), dtype=tangent.dtype)
    radius_sq = jnp.square(ratio) * lax.conv_general_dilated(
        scalar_input,
        scalar_kernel,
        window_strides=stride,
        padding=padding,
        rhs_dilation=dilation,
        dimension_numbers=CONV_DIMENSION_NUMBERS,
    )
    weight_norm = safe_norm(kernel, axis=(0, 1, 2))
    return _scores_from_radius_products(
        radius_sq,
        products,
        weight_norm,
        params.offset,
        curvature,
        margin=margin,
    )


def _materialized_patch_scores(
    params: HNNPPParams,
    patches: Array,
    curvature: Array,
    *,
    margin: Array | None = None,
) -> Array:
    """Score R's explicit beta-scaled tangent patches through F (ablation M).

    M contracts R's materialized patch ``P`` with the flattened kernel by R's
    einsum to form ``D = z . P``, takes ``R^2 = ||P||^2`` from the same patch,
    and evaluates F's clipped score; ``conv2d`` then applies F's output map.
    M versus R isolates the pointwise ball geometry R applies to the n*K*C_in
    patch elements; F versus M isolates F's native convolutions, which never
    materialize the patch.
    """

    kernel = params.weight
    radius_sq, products = _radius_products(
        patches, jnp.reshape(kernel, (patches.shape[-1], kernel.shape[-1]))
    )
    return _scores_from_radius_products(
        radius_sq,
        products,
        safe_norm(kernel, axis=(0, 1, 2)),
        params.offset,
        curvature,
        margin=margin,
    )


def conv2d(
    params: HNNPPParams,
    tangent: Array,
    curvature: Array,
    *,
    variant: str,
    stride: int | Sequence[int] = 1,
    padding: str = "SAME",
    dilation: int | Sequence[int] = 1,
    margin: Array | None = None,
) -> Array:
    """Apply R/F/T/M/Rg/Mg HNN++ or E Euclidean Conv2D in BHWC layout.

    Rg and Mg are R and M with the patch built by :func:`_gather_patches`
    instead of :func:`_extract_patches`; everything after the patch is shared.
    """

    if variant not in EXECUTION_VARIANTS:
        raise ValueError(f"variant must be one of {EXECUTION_VARIANTS}")
    if tangent.ndim != 4:
        raise ValueError("convolution input must have BHWC rank four")
    kernel = params.weight
    if kernel.ndim != 4 or tangent.shape[-1] != kernel.shape[-2]:
        raise ValueError("convolution requires a compatible HWIO kernel")
    if params.offset.shape != (kernel.shape[-1],):
        raise ValueError("convolution offset has the wrong shape")
    normalized_padding = padding.upper()
    if normalized_padding not in {"SAME", "VALID"}:
        raise ValueError("padding must be SAME or VALID")
    strides = spatial_pair(stride, name="stride")
    dilations = spatial_pair(dilation, name="dilation")

    if variant == "E":
        products = lax.conv_general_dilated(
            tangent,
            kernel,
            window_strides=strides,
            padding=normalized_padding,
            rhs_dilation=dilations,
            dimension_numbers=CONV_DIMENSION_NUMBERS,
        )
        return products + params.offset

    if variant in {"R", "M", "Rg", "Mg"}:
        kernel_height, kernel_width, input_channels, output_channels = kernel.shape
        patch_builder = _gather_patches if variant in {"Rg", "Mg"} else _extract_patches
        patches = patch_builder(
            tangent,
            (kernel_height, kernel_width),
            stride=strides,
            padding=normalized_padding,
            dilation=dilations,
        )
        patch_dimension = kernel_height * kernel_width * input_channels
        patches *= beta_ratio(patch_dimension, input_channels, dtype=tangent.dtype)
        if variant in {"R", "Rg"}:
            patch_points = expmap0(patches, curvature, margin=margin)
            scores = hyperplane_scores_ball(
                patch_points,
                jnp.reshape(kernel, (patch_dimension, output_channels)),
                params.offset,
                curvature,
            )
        else:
            scores = _materialized_patch_scores(params, patches, curvature, margin=margin)
    else:
        scores = _factorized_conv_scores(
            params,
            tangent,
            curvature,
            stride=strides,
            padding=normalized_padding,
            dilation=dilations,
            margin=margin,
        )
    return _variant_output(scores, curvature, variant, margin=margin)


def beta_concat_tangent(*blocks: Array) -> Array:
    """Apply HNN++ beta concatenation directly to tangent-coordinate blocks."""

    if not blocks:
        raise ValueError("at least one block is required")
    leading = blocks[0].shape[:-1]
    if any(block.shape[:-1] != leading for block in blocks):
        raise ValueError("beta-concatenated blocks must have identical leading shapes")
    dimensions = tuple(block.shape[-1] for block in blocks)
    total = sum(dimensions)
    dtype = jnp.result_type(*blocks)
    scaled = [
        jnp.asarray(block, dtype=dtype) * beta_ratio(total, dimension, dtype=dtype)
        for block, dimension in zip(blocks, dimensions, strict=True)
    ]
    return jnp.concatenate(scaled, axis=-1)
