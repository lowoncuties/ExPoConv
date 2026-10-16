from __future__ import annotations

import jax
import jax.numpy as jnp

from poincare_exec.config.constants import CONV_DIMENSION_NUMBERS
from poincare_exec.core.hnnpp import HNNPPParams, conv2d, dense
from poincare_exec.core.operations import (
    euclidean_batch_norm,
    init_midpoint_batch_norm,
)
from poincare_exec.models import init_resnet20, init_unet, resnet20_forward, unet_forward


def test_euclidean_dense_is_affine_and_curvature_independent() -> None:
    params = HNNPPParams(
        weight=jnp.asarray([[1.0, -2.0], [0.5, 3.0]]),
        offset=jnp.asarray([0.25, -0.5]),
    )
    inputs = jnp.asarray([[2.0, -1.0], [0.0, 4.0]])
    expected = inputs @ params.weight + params.offset

    actual = dense(params, inputs, jnp.asarray(jnp.nan), variant="E")

    assert jnp.array_equal(actual, expected)


def test_euclidean_convolution_uses_fused_affine_contraction() -> None:
    inputs = jnp.arange(2 * 4 * 5 * 2, dtype=jnp.float32).reshape((2, 4, 5, 2))
    kernel = jnp.arange(3 * 2 * 2 * 3, dtype=jnp.float32).reshape((3, 2, 2, 3))
    params = HNNPPParams(weight=kernel / 100.0, offset=jnp.asarray([0.1, 0.2, 0.3]))
    expected = (
        jax.lax.conv_general_dilated(
            inputs,
            params.weight,
            window_strides=(2, 1),
            padding="VALID",
            rhs_dilation=(1, 2),
            dimension_numbers=CONV_DIMENSION_NUMBERS,
        )
        + params.offset
    )

    actual = conv2d(
        params,
        inputs,
        jnp.asarray(jnp.nan),
        variant="E",
        stride=(2, 1),
        padding="VALID",
        dilation=(1, 2),
    )

    assert jnp.array_equal(actual, expected)


def test_euclidean_batch_norm_uses_batch_and_spatial_axes() -> None:
    params = init_midpoint_batch_norm(2)
    value = jnp.asarray([[[[1.0, 8.0], [3.0, 4.0]]], [[[5.0, 0.0], [7.0, -4.0]]]])
    mean = jnp.mean(value, axis=(0, 1, 2), keepdims=True)
    variance = jnp.mean(jnp.square(value - mean), axis=(0, 1, 2), keepdims=True)
    expected = (value - mean) / jnp.sqrt(variance + 1.0e-6)

    actual = euclidean_batch_norm(params, value)

    assert jnp.allclose(actual, expected)


def test_euclidean_unet_has_no_curvature_dependency() -> None:
    params = init_unet(jax.random.PRNGKey(3), initial_width=2, depth=1, dtype=jnp.float32)
    params.pop("curvature_raw")
    inputs = jax.random.normal(jax.random.PRNGKey(5), (2, 8, 8, 3), dtype=jnp.float32)

    logits = unet_forward(params, inputs, variant="E")
    parameter_gradients = jax.grad(
        lambda selected: jnp.sum(unet_forward(selected, inputs, variant="E"))
    )(params)

    assert logits.shape == (2, 8, 8, 2)
    assert all(
        jnp.all(jnp.isfinite(leaf))
        for leaf in jax.tree_util.tree_leaves((logits, parameter_gradients))
    )


def test_euclidean_resnet_is_curvature_independent() -> None:
    params = init_resnet20(
        jax.random.PRNGKey(7),
        channels=(2, 4, 8),
        blocks_per_group=(1, 1, 1),
        classes=5,
    )
    inputs = jax.random.normal(jax.random.PRNGKey(11), (2, 32, 32, 3), dtype=jnp.float32)

    finite = resnet20_forward(params, inputs, jnp.asarray(0.1), variant="E")
    nonfinite = resnet20_forward(params, inputs, jnp.asarray(jnp.nan), variant="E")

    assert finite.shape == (2, 5)
    assert jnp.array_equal(finite, nonfinite)
