from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from poincare_exec.core.geometry import (
    expmap,
    expmap0,
    logmap,
    logmap0,
    mobius_add,
    project_ball,
    project_tangent,
)

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_origin_maps_round_trip_inside_projection(dtype) -> None:
    tangent = jnp.asarray([[0.0, 0.0], [0.1, -0.2], [0.3, 0.05]], dtype=dtype)
    curvature = jnp.asarray(0.7, dtype=dtype)
    recovered = logmap0(expmap0(tangent, curvature), curvature)
    tolerance = 2.0e-6 if dtype == jnp.float32 else 1.0e-12
    assert jnp.allclose(recovered, tangent, atol=tolerance, rtol=tolerance)


def test_mobius_addition_is_order_sensitive() -> None:
    left = jnp.asarray([0.2, 0.1], dtype=jnp.float64)
    right = jnp.asarray([-0.1, 0.3], dtype=jnp.float64)
    curvature = jnp.asarray(1.0, dtype=jnp.float64)
    assert not jnp.allclose(
        mobius_add(left, right, curvature),
        mobius_add(right, left, curvature),
    )


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_general_maps_round_trip(dtype) -> None:
    base = jnp.asarray([[0.1, -0.05], [-0.2, 0.04]], dtype=dtype)
    tangent = jnp.asarray([[0.03, 0.02], [-0.01, 0.04]], dtype=dtype)
    curvature = jnp.asarray(0.7, dtype=dtype)
    recovered = logmap(base, expmap(base, tangent, curvature), curvature)
    tolerance = 2.0e-6 if dtype == jnp.float32 else 1.0e-12
    assert jnp.allclose(recovered, tangent, atol=tolerance, rtol=tolerance)


def test_tangent_projection_derivatives_match_ball_round_trip_away_from_cap() -> None:
    curvature = jnp.asarray(0.7, dtype=jnp.float64)
    margin = jnp.asarray(1.0e-5, dtype=jnp.float64)
    direction = jnp.asarray([0.4, 0.2, -0.5], dtype=jnp.float64)
    cotangent = jnp.asarray([0.3, -0.2, 0.7], dtype=jnp.float64)

    def direct(value):
        return project_tangent(value, curvature, margin=margin)

    def ball_round_trip(value):
        point = expmap0(value, curvature, project=False)
        return logmap0(project_ball(point, curvature, margin=margin), curvature)

    # The first point is inside the cap and the second is strictly outside it.
    for tangent in (
        jnp.asarray([0.2, -0.3, 0.1], dtype=jnp.float64),
        jnp.asarray([8.0, -3.0, 2.0], dtype=jnp.float64),
    ):
        direct_value, direct_jvp = jax.jvp(direct, (tangent,), (direction,))
        round_trip_value, round_trip_jvp = jax.jvp(
            ball_round_trip, (tangent,), (direction,)
        )
        direct_gradient = jax.grad(lambda value: jnp.vdot(direct(value), cotangent))(
            tangent
        )
        round_trip_gradient = jax.grad(
            lambda value: jnp.vdot(ball_round_trip(value), cotangent)
        )(tangent)

        assert jnp.allclose(
            direct_value, round_trip_value, atol=2.0e-10, rtol=2.0e-10
        )
        assert jnp.allclose(
            direct_jvp, round_trip_jvp, atol=2.0e-10, rtol=2.0e-10
        )
        assert jnp.allclose(
            direct_gradient, round_trip_gradient, atol=2.0e-10, rtol=2.0e-10
        )


def test_tangent_projection_has_identity_derivative_at_origin() -> None:
    curvature = jnp.asarray(0.7, dtype=jnp.float64)
    origin = jnp.zeros((3,), dtype=jnp.float64)
    direction = jnp.asarray([0.2, -0.4, 0.7], dtype=jnp.float64)

    value, derivative = jax.jvp(
        lambda tangent: project_tangent(tangent, curvature),
        (origin,),
        (direction,),
    )
    jacobian = jax.jacrev(lambda tangent: project_tangent(tangent, curvature))(
        origin
    )

    assert jnp.array_equal(value, origin)
    assert jnp.array_equal(derivative, direction)
    assert jnp.array_equal(jacobian, jnp.eye(3, dtype=jnp.float64))
