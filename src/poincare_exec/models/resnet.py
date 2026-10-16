"""Controlled Poincare ResNet-20 in external tangent coordinates."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import jax
import jax.numpy as jnp

from poincare_exec.core.hnnpp import conv2d, dense, init_conv2d, init_dense
from poincare_exec.core.operations import (
    average_pool2d,
    euclidean_batch_norm,
    init_midpoint_batch_norm,
    midpoint_batch_norm,
    mobius_residual_tangent,
)
from poincare_exec.models.common import ParameterTree, key_stream

Array = jax.Array


def init_residual_block(
    keys: Iterator[Array],
    input_channels: int,
    output_channels: int,
    *,
    stride: int,
    dtype: Any,
) -> ParameterTree:
    """Initialize one source-compatible Poincare residual block."""

    block = {
        "conv1": init_conv2d(
            next(keys),
            3,
            input_channels,
            output_channels,
            dtype=dtype,
            initialization="source",
        ),
        "bn1": init_midpoint_batch_norm(output_channels, dtype=dtype),
        "conv2": init_conv2d(
            next(keys),
            3,
            output_channels,
            output_channels,
            dtype=dtype,
            initialization="source",
        ),
        "bn2": init_midpoint_batch_norm(output_channels, dtype=dtype),
    }
    if stride != 1 or input_channels != output_channels:
        block["downsample"] = init_conv2d(
            next(keys),
            1,
            input_channels,
            output_channels,
            dtype=dtype,
            initialization="source",
        )
    return block


def init_resnet20(
    key: Array,
    *,
    channels: Sequence[int] = (8, 16, 32),
    blocks_per_group: Sequence[int] = (3, 3, 3),
    classes: int = 100,
    dtype: Any = jnp.float32,
) -> ParameterTree:
    """Initialize the fixed controlled ResNet topology."""

    if len(channels) != 3 or len(blocks_per_group) != 3:
        raise ValueError("ResNet requires exactly three channel/depth groups")
    keys = key_stream(key)
    params: ParameterTree = {
        "stem_conv": init_conv2d(
            next(keys), 3, 3, channels[0], dtype=dtype, initialization="source"
        ),
        "stem_bn": init_midpoint_batch_norm(channels[0], dtype=dtype),
        "groups": [],
    }
    input_channels = channels[0]
    for group_index, (output_channels, depth) in enumerate(
        zip(channels, blocks_per_group, strict=True)
    ):
        blocks = []
        for block_index in range(depth):
            stride = 2 if group_index > 0 and block_index == 0 else 1
            blocks.append(
                init_residual_block(
                    keys,
                    input_channels,
                    output_channels,
                    stride=stride,
                    dtype=dtype,
                )
            )
            input_channels = output_channels
        params["groups"].append(blocks)
    params["classifier"] = init_dense(
        next(keys), channels[-1], classes, dtype=dtype, initialization="source"
    )
    return params


def residual_block_forward(
    params: ParameterTree,
    tangent: Array,
    curvature: Array,
    *,
    variant: str,
    epsilon: float,
    residual_order: str,
) -> Array:
    """Apply one ordered Mobius residual block at a tangent boundary."""

    residual = tangent
    stride = 2 if "downsample" in params else 1
    tangent = conv2d(params["conv1"], tangent, curvature, variant=variant, stride=stride)
    if variant == "E":
        tangent = euclidean_batch_norm(params["bn1"], tangent, epsilon=epsilon)
    else:
        tangent = midpoint_batch_norm(params["bn1"], tangent, curvature, epsilon=epsilon)
    tangent = jax.nn.relu(tangent)
    tangent = conv2d(params["conv2"], tangent, curvature, variant=variant)
    if variant == "E":
        tangent = euclidean_batch_norm(params["bn2"], tangent, epsilon=epsilon)
    else:
        tangent = midpoint_batch_norm(params["bn2"], tangent, curvature, epsilon=epsilon)
    if "downsample" in params:
        residual = conv2d(
            params["downsample"],
            residual,
            curvature,
            variant=variant,
            stride=stride,
            padding="SAME",
        )
    tangent = (
        tangent + residual
        if variant == "E"
        else mobius_residual_tangent(tangent, residual, curvature, order=residual_order)
    )
    return jax.nn.relu(tangent)


def _resnet20_forward(
    params: ParameterTree,
    inputs: Array,
    curvature: Array,
    *,
    variant: str,
    epsilon: float = 1.0e-6,
    residual_order: str = "fr",
    retain_intermediates: bool,
) -> tuple[Array, dict[str, Array]]:
    """Apply ResNet-20 and optionally retain registered stage boundaries."""

    intermediates: dict[str, Array] = {}
    tangent = conv2d(params["stem_conv"], inputs, curvature, variant=variant)
    tangent = (
        euclidean_batch_norm(params["stem_bn"], tangent, epsilon=epsilon)
        if variant == "E"
        else midpoint_batch_norm(params["stem_bn"], tangent, curvature, epsilon=epsilon)
    )
    tangent = jax.nn.relu(tangent)
    if retain_intermediates:
        intermediates["stem"] = tangent
    for group_index, group in enumerate(params["groups"], start=1):
        for block in group:
            tangent = residual_block_forward(
                block,
                tangent,
                curvature,
                variant=variant,
                epsilon=epsilon,
                residual_order=residual_order,
            )
        if retain_intermediates:
            intermediates[f"residual_group_{group_index}"] = tangent
    tangent = average_pool2d(tangent, 8)
    tangent = jnp.reshape(tangent, (tangent.shape[0], -1))
    if retain_intermediates:
        intermediates["pooled"] = tangent
    logits = dense(params["classifier"], tangent, curvature, variant=variant)
    return logits, intermediates


def resnet20_forward(
    params: ParameterTree,
    inputs: Array,
    curvature: Array,
    *,
    variant: str,
    epsilon: float = 1.0e-6,
    residual_order: str = "fr",
) -> Array:
    """Return classification logits for BHWC inputs."""

    logits, _ = _resnet20_forward(
        params,
        inputs,
        curvature,
        variant=variant,
        epsilon=epsilon,
        residual_order=residual_order,
        retain_intermediates=False,
    )
    return logits


def resnet20_forward_with_intermediates(
    params: ParameterTree,
    inputs: Array,
    curvature: Array,
    *,
    variant: str,
    epsilon: float = 1.0e-6,
    residual_order: str = "fr",
) -> tuple[Array, dict[str, Array]]:
    """Return logits and frozen stem/group/pooling activation boundaries."""

    return _resnet20_forward(
        params,
        inputs,
        curvature,
        variant=variant,
        epsilon=epsilon,
        residual_order=residual_order,
        retain_intermediates=True,
    )
