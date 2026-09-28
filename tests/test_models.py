from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from poincare_exec.models import (
    init_resnet20,
    init_unet,
    resnet20_forward,
    unet_forward,
)


@pytest.fixture(autouse=True)
def _float32_only():
    # The geometry, HNN++, and operations test modules enable x64 at import time.
    # These model tests exercise the FP32 training configuration, so pin x64 off
    # for their duration and restore the previous setting afterwards.
    previous = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", False)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


def test_small_resnet_reference_and_tangent_paths_agree() -> None:
    params = init_resnet20(
        jax.random.PRNGKey(7), channels=(2, 4, 8), blocks_per_group=(1, 1, 1), classes=5
    )
    inputs = 0.02 * jax.random.normal(jax.random.PRNGKey(11), (2, 32, 32, 3))
    curvature = jnp.asarray(0.1, dtype=jnp.float32)
    reference = resnet20_forward(params, inputs, curvature, variant="R")
    candidate = resnet20_forward(params, inputs, curvature, variant="T")
    assert jnp.allclose(reference, candidate, atol=2e-4, rtol=2e-4)


def test_small_unet_reference_and_tangent_paths_agree() -> None:
    params = init_unet(
        jax.random.PRNGKey(13), classes=2, initial_width=2, depth=2
    )
    inputs = 0.02 * jax.random.normal(jax.random.PRNGKey(17), (2, 8, 8, 3))
    reference = unet_forward(params, inputs, variant="R")
    candidate = unet_forward(params, inputs, variant="T")
    assert reference.shape == (2, 8, 8, 2)
    assert jnp.allclose(reference, candidate, atol=5e-4, rtol=5e-4)
