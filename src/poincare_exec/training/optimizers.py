"""Small explicit optimizers for auditable experiment state transitions."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

Array = jax.Array


class AdamState(NamedTuple):
    """Complete Adam state in the same pytree structure as the parameters."""

    step: Array
    first_moment: Any
    second_moment: Any


class UpdateDiagnostics(NamedTuple):
    """Scalar diagnostics synchronized at each complete update boundary."""

    gradient_norm: Array
    update_norm: Array


def tree_l2_norm(tree: Any) -> Array:
    """Return the global Euclidean L2 norm of an array pytree."""

    squared = [jnp.sum(jnp.square(leaf)) for leaf in jax.tree_util.tree_leaves(tree)]
    if not squared:
        return jnp.asarray(0.0, dtype=jnp.float32)
    return jnp.sqrt(jnp.sum(jnp.stack(squared)))


def init_adam(params: Any) -> AdamState:
    """Initialize moments without any hidden optimizer defaults."""

    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return AdamState(
        step=jnp.asarray(0, dtype=jnp.int32),
        first_moment=zeros,
        second_moment=zeros,
    )


def _scalar_like(value: float, reference: Array) -> Array:
    """Materialize an optimizer scalar without promoting the reference dtype."""

    return jnp.asarray(value, dtype=reference.dtype)


def adam_update(
    params: Any,
    gradients: Any,
    state: AdamState,
    *,
    learning_rate: float,
    weight_decay: float,
    gradient_clip_global_norm: float | None,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1.0e-8,
) -> tuple[Any, AdamState, UpdateDiagnostics]:
    """Apply Adam with coupled L2 decay and optional global-norm clipping."""

    gradient_norm = tree_l2_norm(gradients)
    if gradient_clip_global_norm is not None:
        limit = jnp.asarray(gradient_clip_global_norm, dtype=gradient_norm.dtype)
        one = _scalar_like(1.0, gradient_norm)
        norm_floor = _scalar_like(1.0e-15, gradient_norm)
        scale = jnp.minimum(one, limit / jnp.maximum(gradient_norm, norm_floor))
        gradients = jax.tree_util.tree_map(lambda value: scale * value, gradients)
    if weight_decay:
        gradients = jax.tree_util.tree_map(
            lambda gradient, parameter: gradient
            + _scalar_like(weight_decay, parameter) * parameter,
            gradients,
            params,
        )
    step = state.step + 1
    first = jax.tree_util.tree_map(
        lambda moment, gradient: _scalar_like(beta1, moment) * moment
        + (_scalar_like(1.0, moment) - _scalar_like(beta1, moment)) * gradient,
        state.first_moment,
        gradients,
    )
    second = jax.tree_util.tree_map(
        lambda moment, gradient: _scalar_like(beta2, moment) * moment
        + (_scalar_like(1.0, moment) - _scalar_like(beta2, moment)) * jnp.square(gradient),
        state.second_moment,
        gradients,
    )

    def parameter_update(first_value: Array, second_value: Array) -> Array:
        one = _scalar_like(1.0, first_value)
        first_scale = one - jnp.power(_scalar_like(beta1, first_value), step)
        second_scale = one - jnp.power(_scalar_like(beta2, second_value), step)
        return (
            _scalar_like(learning_rate, first_value)
            * (first_value / first_scale)
            / (jnp.sqrt(second_value / second_scale) + _scalar_like(epsilon, second_value))
        )

    updates = jax.tree_util.tree_map(
        parameter_update,
        first,
        second,
    )
    updated = jax.tree_util.tree_map(
        lambda parameter, update: parameter - update, params, updates
    )
    return (
        updated,
        AdamState(step=step, first_moment=first, second_moment=second),
        UpdateDiagnostics(
            gradient_norm=gradient_norm,
            update_norm=tree_l2_norm(updates),
        ),
    )
