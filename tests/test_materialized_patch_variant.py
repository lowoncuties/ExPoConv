from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from poincare_exec.core.hnnpp import conv2d, dense, init_conv2d, init_dense
from poincare_exec.core.operations import transpose_conv2d
from poincare_exec.models import init_resnet20, init_unet, resnet20_forward, unet_forward

jax.config.update("jax_enable_x64", True)

FLOAT64_TOLERANCE = 1.0e-10


def _assert_paper_agreement(reference, candidate) -> None:
    """Apply max|y_cand - y_ref| <= eps_abs + eps_rel * ||y_ref||_2 per tensor."""

    for reference_leaf, candidate_leaf in zip(
        jax.tree_util.tree_leaves(reference),
        jax.tree_util.tree_leaves(candidate),
        strict=True,
    ):
        reference_array = np.asarray(reference_leaf)
        error = np.max(np.abs(np.asarray(candidate_leaf) - reference_array), initial=0.0)
        assert error <= FLOAT64_TOLERANCE * (1.0 + np.linalg.norm(reference_array))


@pytest.mark.parametrize(
    ("stride", "padding", "scale", "curvature"),
    [(1, "SAME", 0.1, 0.7), (2, "VALID", 0.5, 1.0)],
)
def test_materialized_patch_variant_agrees_with_reference_and_factorized(
    stride: int, padding: str, scale: float, curvature: float
) -> None:
    key_params, key_input, key_cotangent = jax.random.split(jax.random.PRNGKey(31), 3)
    params = init_conv2d(
        key_params, 3, 3, 4, dtype=jnp.float64, weight_scale=1.0, offset_scale=0.05
    )
    tangent = scale * jax.random.normal(key_input, (2, 7, 7, 3), dtype=jnp.float64)
    curvature_value = jnp.asarray(curvature, dtype=jnp.float64)

    def output_and_gradients(variant: str):
        def forward(selected_params, selected_tangent, selected_curvature):
            return conv2d(
                selected_params,
                selected_tangent,
                selected_curvature,
                variant=variant,
                stride=stride,
                padding=padding,
            )

        output, pullback = jax.vjp(forward, params, tangent, curvature_value)
        cotangent = jax.random.normal(key_cotangent, output.shape, dtype=output.dtype)
        return output, pullback(cotangent)

    results = {variant: output_and_gradients(variant) for variant in ("R", "F", "M")}
    _assert_paper_agreement(results["R"], results["M"])
    _assert_paper_agreement(results["F"], results["M"])


def test_materialized_patch_variant_executes_dense_parts_as_factorized() -> None:
    params = init_dense(
        jax.random.PRNGKey(5), 3, 2 * 2 * 4, dtype=jnp.float64, offset_scale=0.05
    )
    tangent = 0.1 * jax.random.normal(jax.random.PRNGKey(7), (2, 3, 3, 3), jnp.float64)
    curvature = jnp.asarray(0.7, dtype=jnp.float64)

    def apply_dense(variant: str):
        return dense(params, tangent, curvature, variant=variant)

    def apply_transpose(variant: str):
        return transpose_conv2d(
            params, tangent, curvature, output_channels=4, variant=variant
        )

    assert jnp.array_equal(apply_dense("M"), apply_dense("F"))
    assert jnp.array_equal(apply_transpose("M"), apply_transpose("F"))


def test_materialized_patch_variant_agrees_with_reference_in_complete_models() -> None:
    unet = init_unet(
        jax.random.PRNGKey(13), classes=2, initial_width=2, depth=2, dtype=jnp.float64
    )
    images = jax.random.uniform(jax.random.PRNGKey(17), (2, 8, 8, 3), dtype=jnp.float64)
    _assert_paper_agreement(
        unet_forward(unet, images, variant="R"), unet_forward(unet, images, variant="M")
    )

    resnet = init_resnet20(
        jax.random.PRNGKey(7),
        channels=(2, 4, 8),
        blocks_per_group=(1, 1, 1),
        classes=5,
        dtype=jnp.float64,
    )
    inputs = 0.02 * jax.random.normal(jax.random.PRNGKey(11), (2, 32, 32, 3), jnp.float64)
    curvature = jnp.asarray(0.1, dtype=jnp.float64)
    _assert_paper_agreement(
        resnet20_forward(resnet, inputs, curvature, variant="R"),
        resnet20_forward(resnet, inputs, curvature, variant="M"),
    )
