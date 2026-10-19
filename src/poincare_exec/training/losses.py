"""Frozen task objectives for the matched training campaigns."""

from __future__ import annotations

import jax
import jax.numpy as jnp

Array = jax.Array


def cross_entropy(logits: Array, targets: Array) -> Array:
    """Mean multiclass cross-entropy without label smoothing."""

    log_probabilities = jax.nn.log_softmax(logits, axis=-1)
    selected = jnp.take_along_axis(log_probabilities, targets[..., None], axis=-1)[..., 0]
    return -jnp.mean(selected)


def dice_focal_loss(
    logits: Array,
    targets: Array,
    *,
    smoothing: float,
    denominator_epsilon: float,
    alpha: float,
    gamma: float,
) -> Array:
    """Explicit two-class soft Dice plus summed one-vs-all focal loss."""

    classes = logits.shape[-1]
    one_hot = jax.nn.one_hot(targets, classes, dtype=logits.dtype)
    probabilities = jax.nn.softmax(logits, axis=-1)
    reduction_axes = tuple(range(logits.ndim - 1))
    intersection = jnp.sum(probabilities * one_hot, axis=reduction_axes)
    denominator = jnp.sum(probabilities + one_hot, axis=reduction_axes)
    score = (2.0 * intersection + smoothing) / jnp.maximum(
        denominator + smoothing, denominator_epsilon
    )
    present = jnp.sum(one_hot, axis=reduction_axes) > 0
    dice = jnp.mean(jnp.where(present, 1.0 - score, 0.0))

    positive_logits = jnp.maximum(logits, 0.0)
    binary_cross_entropy = (
        positive_logits - logits * one_hot + jnp.log1p(jnp.exp(-jnp.abs(logits)))
    )
    probability_correct = jnp.exp(-binary_cross_entropy)
    alpha_factor = alpha * one_hot + (1.0 - alpha) * (1.0 - one_hot)
    focal_by_element = (
        alpha_factor * jnp.power(1.0 - probability_correct, gamma) * binary_cross_entropy
    )
    focal = jnp.sum(jnp.mean(focal_by_element, axis=reduction_axes))
    return dice + focal
