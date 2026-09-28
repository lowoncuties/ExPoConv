from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from poincare_exec.core.hnnpp import (
    HNNPPParams,
    conv2d,
    dense,
    init_conv2d,
    init_dense,
    scores_to_ball,
    scores_to_projected_tangent,
)

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("curvature", [0.001, 0.1, 1.0, 10.0])
def test_dense_variants_agree_in_float64(curvature: float) -> None:
    key_params, key_input = jax.random.split(jax.random.PRNGKey(11))
    params = init_dense(
        key_params,
        7,
        5,
        dtype=jnp.float64,
        weight_scale=0.4,
        offset_scale=0.1,
    )
    tangent = 0.2 * jax.random.normal(key_input, (3, 7), dtype=jnp.float64)
    outputs = {
        variant: dense(params, tangent, jnp.asarray(curvature), variant=variant)
        for variant in ("R", "F", "T")
    }
    assert jnp.allclose(outputs["R"], outputs["F"], atol=1.0e-11, rtol=1.0e-11)
    assert jnp.allclose(outputs["R"], outputs["T"], atol=1.0e-11, rtol=1.0e-11)


@pytest.mark.parametrize("kernel_size", [1, 3])
@pytest.mark.parametrize("padding", ["SAME", "VALID"])
def test_convolution_variants_and_gradients_agree(kernel_size: int, padding: str) -> None:
    key_params, key_input = jax.random.split(jax.random.PRNGKey(29))
    params = init_conv2d(
        key_params,
        kernel_size,
        3,
        4,
        dtype=jnp.float64,
        weight_scale=0.25,
        offset_scale=0.05,
    )
    tangent = 0.1 * jax.random.normal(key_input, (2, 5, 5, 3), dtype=jnp.float64)
    curvature = jnp.asarray(0.7, dtype=jnp.float64)

    def loss(variant, selected_params, selected_tangent, selected_curvature):
        output = conv2d(
            selected_params,
            selected_tangent,
            selected_curvature,
            variant=variant,
            padding=padding,
        )
        return jnp.sum(jnp.square(output))

    values = {}
    gradients = {}
    for variant in ("R", "F", "T"):
        values[variant] = conv2d(
            params, tangent, curvature, variant=variant, padding=padding
        )
        gradients[variant] = jax.grad(loss, argnums=(1, 2, 3))(
            variant, params, tangent, curvature
        )

    assert jnp.allclose(values["R"], values["F"], atol=2.0e-10, rtol=2.0e-10)
    assert jnp.allclose(values["R"], values["T"], atol=2.0e-10, rtol=2.0e-10)
    for reference_leaf, factored_leaf, tangent_leaf in zip(
        jax.tree_util.tree_leaves(gradients["R"]),
        jax.tree_util.tree_leaves(gradients["F"]),
        jax.tree_util.tree_leaves(gradients["T"]),
        strict=True,
    ):
        assert jnp.allclose(reference_leaf, factored_leaf, atol=5.0e-9, rtol=5.0e-9)
        assert jnp.allclose(reference_leaf, tangent_leaf, atol=5.0e-9, rtol=5.0e-9)


@pytest.mark.parametrize(
    ("input_shape", "kernel_size", "stride", "dilation", "output_shape"),
    [
        ((1, 5, 6, 2), (2, 3), (2, 1), 1, (1, 2, 4, 2)),
        ((1, 5, 6, 2), (2, 2), 1, (2, 1), (1, 3, 5, 2)),
    ],
    ids=("nondefault-stride", "nondefault-dilation"),
)
def test_reference_and_factorized_convolution_agree_for_nondefault_geometry(
    input_shape: tuple[int, ...],
    kernel_size: tuple[int, int],
    stride: int | tuple[int, int],
    dilation: int | tuple[int, int],
    output_shape: tuple[int, ...],
) -> None:
    key_params, key_input = jax.random.split(jax.random.PRNGKey(113))
    params = init_conv2d(
        key_params,
        kernel_size,
        input_shape[-1],
        output_shape[-1],
        dtype=jnp.float64,
        weight_scale=0.25,
        offset_scale=0.05,
    )
    tangent = 0.1 * jax.random.normal(key_input, input_shape, dtype=jnp.float64)
    curvature = jnp.asarray(0.7, dtype=jnp.float64)

    def loss(selected_params, selected_tangent, selected_curvature, *, variant):
        output = conv2d(
            selected_params,
            selected_tangent,
            selected_curvature,
            variant=variant,
            stride=stride,
            padding="VALID",
            dilation=dilation,
        )
        return jnp.sum(jnp.square(output))

    values = {
        variant: conv2d(
            params,
            tangent,
            curvature,
            variant=variant,
            stride=stride,
            padding="VALID",
            dilation=dilation,
        )
        for variant in ("R", "F")
    }
    gradients = {
        variant: jax.grad(loss, argnums=(0, 1, 2))(
            params, tangent, curvature, variant=variant
        )
        for variant in ("R", "F")
    }

    assert values["R"].shape == output_shape
    assert jnp.allclose(values["R"], values["F"], atol=2.0e-10, rtol=2.0e-10)
    for reference_leaf, factored_leaf in zip(
        jax.tree_util.tree_leaves(gradients["R"]),
        jax.tree_util.tree_leaves(gradients["F"]),
        strict=True,
    ):
        assert jnp.allclose(reference_leaf, factored_leaf, atol=5.0e-9, rtol=5.0e-9)


def test_projection_regime_is_preserved() -> None:
    params = init_dense(
        jax.random.PRNGKey(47),
        3,
        3,
        dtype=jnp.float64,
        weight_scale=1.0,
        offset_scale=0.2,
    )
    tangent = jnp.asarray([[12.0, -8.0, 5.0]], dtype=jnp.float64)
    curvature = jnp.asarray(10.0, dtype=jnp.float64)
    reference = dense(params, tangent, curvature, variant="R")
    proposed = dense(params, tangent, curvature, variant="T")
    assert jnp.all(jnp.isfinite(reference))
    assert jnp.allclose(reference, proposed, atol=1.0e-9, rtol=1.0e-9)


def test_source_zero_orientation_convention_is_finite() -> None:
    params = init_dense(
        jax.random.PRNGKey(71),
        2,
        4,
        dtype=jnp.float64,
        offset_scale=0.1,
        initialization="source",
    )
    tangent = jnp.asarray([[0.2, -0.1]], dtype=jnp.float64)
    curvature = jnp.asarray(0.7, dtype=jnp.float64)
    outputs = [dense(params, tangent, curvature, variant=item) for item in "RFT"]
    assert all(jnp.all(jnp.isfinite(item)) for item in outputs)
    assert jnp.allclose(outputs[0], outputs[1], atol=1e-12, rtol=1e-12)
    assert jnp.allclose(outputs[0], outputs[2], atol=1e-12, rtol=1e-12)

    zero_offset_params = HNNPPParams(
        weight=params.weight,
        offset=params.offset.at[2:].set(0.0),
    )
    input_direction = jnp.asarray([[0.3, 0.4]], dtype=jnp.float64)
    for variant in "RFT":
        output, derivative = jax.jvp(
            lambda value, selected_variant=variant: dense(
                zero_offset_params, value, curvature, variant=selected_variant
            ),
            (tangent,),
            (input_direction,),
        )
        assert jnp.array_equal(output[..., 2:], jnp.zeros_like(output[..., 2:]))
        assert jnp.array_equal(
            derivative[..., 2:], jnp.zeros_like(derivative[..., 2:])
        )


def test_scaled_single_score_reconstruction_matches_closed_form() -> None:
    scores = jnp.asarray([[25.0], [-23.0]], dtype=jnp.float64)
    direction = jnp.asarray([[0.3], [-0.4]], dtype=jnp.float64)
    curvature = jnp.asarray(1.0, dtype=jnp.float64)
    margin = jnp.asarray(1.0e-14, dtype=jnp.float64)

    ball = scores_to_ball(scores, curvature, margin=margin)
    tangent, tangent_jvp = jax.jvp(
        lambda value: scores_to_projected_tangent(
            value, curvature, margin=margin
        ),
        (scores,),
        (direction,),
    )

    # In one output dimension, HNN++ reconstruction reduces analytically to
    # tanh(v / 2) in the ball and v / 2 in the origin tangent space.
    assert jnp.all(jnp.isfinite(ball))
    assert jnp.all(jnp.isfinite(tangent))
    assert jnp.allclose(ball, jnp.tanh(scores / 2.0), atol=2.0e-14, rtol=2.0e-14)
    assert jnp.allclose(tangent, scores / 2.0, atol=2.0e-13, rtol=2.0e-13)
    assert jnp.allclose(tangent_jvp, direction / 2.0, atol=2.0e-13, rtol=2.0e-13)
