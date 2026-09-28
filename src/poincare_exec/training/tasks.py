"""Task construction, forward maps, and objectives shared by experiments."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from poincare_exec.models.resnet import (
    init_resnet20,
    resnet20_forward,
    resnet20_forward_with_intermediates,
)
from poincare_exec.models.unet import (
    init_unet,
    unet_forward,
    unet_forward_with_intermediates,
)
from poincare_exec.training.losses import cross_entropy, dice_focal_loss
from poincare_exec.training.optimizers import adam_update

Array = jax.Array


def initialize_task(task_name: str, seed: int, task: dict[str, Any]) -> Any:
    """Create the exact source-compatible parameter tree for one task."""

    key = jax.random.PRNGKey(seed)
    if task_name == "isic16_poincare_unet":
        model = task["model"]
        return init_unet(
            key,
            input_channels=int(task["input"]["channels"]),
            classes=int(task["classes"]),
            initial_width=int(model["initial_width"]),
            depth=int(model["levels"]),
            initial_curvature=float(task["curvature"]["initial_magnitude"]),
        )
    if task_name == "cifar100_poincare_resnet20":
        model = task["model"]
        return init_resnet20(
            key,
            channels=tuple(model["channels"]),
            blocks_per_group=tuple(model["blocks_per_group"]),
            classes=int(task["classes"]),
        )
    raise ValueError(f"unsupported training task: {task_name}")


def forward_task(
    task_name: str,
    task: dict[str, Any],
    params: Any,
    inputs: Array,
    variant: str,
) -> Array:
    """Apply one frozen task topology in a named execution variant."""

    if task_name == "isic16_poincare_unet":
        return unet_forward(
            params,
            inputs,
            variant=variant,
            epsilon=float(task["normalization"]["epsilon"]),
        )
    curvature = jnp.asarray(task["curvature"]["initial_magnitude"], dtype=inputs.dtype)
    if task_name == "cifar100_poincare_resnet20":
        return resnet20_forward(
            params,
            inputs,
            curvature,
            variant=variant,
            epsilon=float(task["normalization"]["epsilon"]),
            residual_order=task["model"]["skip_order"],
        )
    raise ValueError(f"unsupported training task: {task_name}")


def forward_task_with_intermediates(
    task_name: str,
    task: dict[str, Any],
    params: Any,
    inputs: Array,
    variant: str,
) -> tuple[Array, dict[str, Array]]:
    """Apply a task and retain the E06 stage-boundary activations."""

    if task_name == "isic16_poincare_unet":
        return unet_forward_with_intermediates(
            params,
            inputs,
            variant=variant,
            epsilon=float(task["normalization"]["epsilon"]),
        )
    curvature = jnp.asarray(task["curvature"]["initial_magnitude"], dtype=inputs.dtype)
    if task_name == "cifar100_poincare_resnet20":
        return resnet20_forward_with_intermediates(
            params,
            inputs,
            curvature,
            variant=variant,
            epsilon=float(task["normalization"]["epsilon"]),
            residual_order=task["model"]["skip_order"],
        )
    raise ValueError(f"unsupported training task: {task_name}")


def task_loss(
    task_name: str,
    task: dict[str, Any],
    logits: Array,
    targets: Array,
) -> Array:
    """Evaluate the explicit objective declared for one task."""

    if task_name != "isic16_poincare_unet":
        return cross_entropy(logits, targets)
    loss = task["loss"]
    return dice_focal_loss(
        logits,
        targets,
        smoothing=float(loss["dice_smoothing"]),
        denominator_epsilon=float(loss["dice_denominator_epsilon"]),
        alpha=float(loss["focal_alpha"]),
        gamma=float(loss["focal_gamma"]),
    )


def make_train_step(
    task_name: str,
    task: dict[str, Any],
    variant: str,
):
    """Build one complete loss-gradient-Adam update with frozen semantics."""

    optimizer = task["optimizer"]

    def objective(params, inputs, targets):
        logits = forward_task(task_name, task, params, inputs, variant)
        return task_loss(task_name, task, logits, targets)

    def train_step(params, state, inputs, targets):
        loss, gradients = jax.value_and_grad(objective)(params, inputs, targets)
        updated, updated_state, diagnostics = adam_update(
            params,
            gradients,
            state,
            learning_rate=float(optimizer["learning_rate"]),
            weight_decay=float(optimizer["weight_decay"]),
            gradient_clip_global_norm=optimizer["gradient_clip_global_norm"],
        )
        return updated, updated_state, loss, diagnostics

    return jax.jit(train_step)
