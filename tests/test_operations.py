from __future__ import annotations

import jax
import jax.numpy as jnp

from poincare_exec.core.hnnpp import init_dense
from poincare_exec.core.operations import (
    _fold_expanded_tangent,
    init_midpoint_batch_norm,
    midpoint_batch_norm,
    project_tangent,
    transpose_conv2d,
)

jax.config.update("jax_enable_x64", True)


def test_midpoint_batch_norm_has_finite_values_and_gradients() -> None:
    tangent = 0.1 * jax.random.normal(
        jax.random.PRNGKey(3), (2, 3, 3, 4), dtype=jnp.float64
    )
    params = init_midpoint_batch_norm(4, dtype=jnp.float64)
    curvature = jnp.asarray(0.7, dtype=jnp.float64)

    def loss(selected_params, selected_input, selected_curvature):
        output = midpoint_batch_norm(selected_params, selected_input, selected_curvature)
        return jnp.sum(jnp.square(output))

    value, gradients = jax.value_and_grad(loss, argnums=(0, 1, 2))(
        params, tangent, curvature
    )
    assert jnp.isfinite(value)
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree_util.tree_leaves(gradients))


def test_direct_tangent_projection_matches_round_trip() -> None:
    tangent = jnp.asarray([[12.0, -8.0, 5.0], [0.1, 0.2, -0.3]], dtype=jnp.float64)
    curvature = jnp.asarray(10.0, dtype=jnp.float64)
    from poincare_exec.core.geometry import expmap0, logmap0

    reference = logmap0(expmap0(tangent, curvature), curvature)
    assert jnp.allclose(project_tangent(tangent, curvature), reference, atol=1e-12)


def test_transpose_convolution_variants_agree() -> None:
    params = init_dense(
        jax.random.PRNGKey(5),
        3,
        2 * 2 * 4,
        dtype=jnp.float64,
        weight_scale=0.2,
        offset_scale=0.05,
    )
    tangent = 0.1 * jax.random.normal(
        jax.random.PRNGKey(7), (2, 3, 3, 3), dtype=jnp.float64
    )
    curvature = jnp.asarray(0.7, dtype=jnp.float64)
    outputs = {
        variant: transpose_conv2d(
            params,
            tangent,
            curvature,
            output_channels=4,
            variant=variant,
        )
        for variant in ("R", "F", "T")
    }
    assert outputs["R"].shape == (2, 6, 6, 4)
    assert jnp.allclose(outputs["R"], outputs["F"], atol=2e-10, rtol=2e-10)
    assert jnp.allclose(outputs["R"], outputs["T"], atol=2e-10, rtol=2e-10)


def test_disjoint_transpose_fold_matches_scatter_values_and_gradients() -> None:
    expanded = jax.random.normal(
        jax.random.PRNGKey(19), (2, 3, 4, 2 * 2 * 5), dtype=jnp.float32
    )
    cotangent = jax.random.normal(jax.random.PRNGKey(23), (2, 6, 8, 5), dtype=jnp.float32)

    def scatter_reference(value):
        blocks = jnp.reshape(value, (2, 3, 4, 2, 2, 5))
        output = jnp.zeros((2, 6, 8, 5), dtype=value.dtype)
        for row in range(2):
            for column in range(2):
                output = output.at[:, row::2, column::2, :].add(
                    blocks[:, :, :, row, column, :]
                )
        return output

    def disjoint_fold(value):
        return _fold_expanded_tangent(
            value,
            kernel_shape=(2, 2),
            output_channels=5,
            stride=(2, 2),
            padding=(0, 0),
        )

    expected = scatter_reference(expanded)
    actual = disjoint_fold(expanded)
    assert jnp.array_equal(actual, expected)

    expected_gradient = jax.grad(
        lambda value: jnp.sum(scatter_reference(value) * cotangent)
    )(expanded)
    actual_gradient = jax.grad(lambda value: jnp.sum(disjoint_fold(value) * cotangent))(
        expanded
    )
    assert jnp.array_equal(actual_gradient, expected_gradient)
