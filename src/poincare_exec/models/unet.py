"""Controlled source-shaped HNN++ Poincare U-Net."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from poincare_exec.core.hnnpp import (
    beta_concat_tangent,
    conv2d,
    init_conv2d,
    init_dense,
)
from poincare_exec.core.operations import max_pool2d, transpose_conv2d
from poincare_exec.models.common import (
    ParameterTree,
    apply_double_conv,
    init_double_conv,
    key_stream,
)

Array = jax.Array


def inverse_softplus(value: float, *, dtype: Any) -> Array:
    """Return the unconstrained scalar producing ``value`` under softplus."""

    selected = jnp.asarray(value, dtype=dtype)
    return selected + jnp.log(-jnp.expm1(-selected))


def init_unet(
    key: Array,
    *,
    input_channels: int = 3,
    classes: int = 2,
    initial_width: int = 8,
    depth: int = 4,
    initial_curvature: float = 0.1,
    dtype: Any = jnp.float32,
) -> ParameterTree:
    """Initialize the frozen four-level transpose-decoder architecture."""

    if depth <= 0:
        raise ValueError("depth must be positive")
    keys = key_stream(key)
    widths = [initial_width * 2**index for index in range(depth + 1)]
    params: ParameterTree = {
        "curvature_raw": inverse_softplus(initial_curvature, dtype=dtype),
        "input_block": init_double_conv(keys, input_channels, widths[0], dtype=dtype),
        "down_blocks": [],
        "up_blocks": [],
    }
    for level in range(depth):
        params["down_blocks"].append(
            init_double_conv(keys, widths[level], widths[level + 1], dtype=dtype)
        )
    for level in range(depth, 0, -1):
        output_width = widths[level - 1]
        transpose = init_dense(
            next(keys),
            widths[level],
            4 * output_width,
            dtype=dtype,
            initialization="source",
        )
        params["up_blocks"].append(
            {
                "transpose": transpose,
                "conv": init_double_conv(keys, 2 * output_width, output_width, dtype=dtype),
            }
        )
    params["output_conv"] = init_conv2d(
        next(keys), 1, widths[0], classes, dtype=dtype, initialization="source"
    )
    return params


def _crop_or_pad_like(tangent: Array, reference: Array) -> Array:
    """Center crop/pad a decoder tensor to the skip tensor's static shape."""

    target_height, target_width = reference.shape[1:3]
    height, width = tangent.shape[1:3]
    crop_top = max((height - target_height) // 2, 0)
    crop_left = max((width - target_width) // 2, 0)
    tangent = tangent[
        :,
        crop_top : crop_top + min(height, target_height),
        crop_left : crop_left + min(width, target_width),
        :,
    ]
    pad_height = target_height - tangent.shape[1]
    pad_width = target_width - tangent.shape[2]
    return jnp.pad(
        tangent,
        (
            (0, 0),
            (pad_height // 2, pad_height - pad_height // 2),
            (pad_width // 2, pad_width - pad_width // 2),
            (0, 0),
        ),
    )


def decoder_block_forward(
    params: ParameterTree,
    tangent: Array,
    skip: Array,
    curvature: Array,
    *,
    variant: str,
    epsilon: float = 1.0e-6,
) -> Array:
    """Apply one source transpose-convolution, beta-skip, double-conv block."""

    output_channels = params["transpose"].weight.shape[1] // 4
    tangent = transpose_conv2d(
        params["transpose"],
        tangent,
        curvature,
        output_channels=output_channels,
        variant=variant,
    )
    tangent = _crop_or_pad_like(tangent, skip)
    tangent = (
        jnp.concatenate((skip, tangent), axis=-1)
        if variant == "E"
        else beta_concat_tangent(skip, tangent)
    )
    return apply_double_conv(
        params["conv"],
        tangent,
        curvature,
        variant=variant,
        epsilon=epsilon,
        convolution_bias=False,
    )


def _unet_forward(
    params: ParameterTree,
    inputs: Array,
    *,
    variant: str,
    epsilon: float = 1.0e-6,
    retain_intermediates: bool,
) -> tuple[Array, dict[str, Array]]:
    """Apply the U-Net and optionally retain registered stage boundaries."""

    curvature = (
        jnp.asarray(0.0, dtype=inputs.dtype)
        if variant == "E"
        else jax.nn.softplus(params["curvature_raw"])
    )
    intermediates: dict[str, Array] = {}
    tangent = apply_double_conv(
        params["input_block"],
        inputs,
        curvature,
        variant=variant,
        epsilon=epsilon,
        convolution_bias=False,
    )
    if retain_intermediates:
        intermediates["encoder_level_0"] = tangent
    skips = [tangent]
    for level, block in enumerate(params["down_blocks"], start=1):
        tangent = max_pool2d(tangent, 2)
        tangent = apply_double_conv(
            block,
            tangent,
            curvature,
            variant=variant,
            epsilon=epsilon,
            convolution_bias=False,
        )
        if retain_intermediates:
            intermediates[f"encoder_level_{level}"] = tangent
        skips.append(tangent)
    for level, (block, skip) in enumerate(
        zip(params["up_blocks"], reversed(skips[:-1]), strict=True), start=1
    ):
        tangent = decoder_block_forward(
            block,
            tangent,
            skip,
            curvature,
            variant=variant,
            epsilon=epsilon,
        )
        if retain_intermediates:
            intermediates[f"decoder_level_{level}"] = tangent
    logits = conv2d(params["output_conv"], tangent, curvature, variant=variant)
    return logits, intermediates


def unet_forward(
    params: ParameterTree,
    inputs: Array,
    *,
    variant: str,
    epsilon: float = 1.0e-6,
) -> Array:
    """Return per-pixel two-class tangent logits in BHWC layout."""

    logits, _ = _unet_forward(
        params,
        inputs,
        variant=variant,
        epsilon=epsilon,
        retain_intermediates=False,
    )
    return logits


def unet_forward_with_intermediates(
    params: ParameterTree,
    inputs: Array,
    *,
    variant: str,
    epsilon: float = 1.0e-6,
) -> tuple[Array, dict[str, Array]]:
    """Return logits and frozen encoder/decoder activation boundaries."""

    return _unet_forward(
        params,
        inputs,
        variant=variant,
        epsilon=epsilon,
        retain_intermediates=True,
    )
