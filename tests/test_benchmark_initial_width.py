"""The optional U-Net ``initial_width`` workload key leaves existing workloads unchanged."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import pytest

from poincare_exec.config.loader import load_yaml
from poincare_exec.experiments.benchmark import _model_state

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = load_yaml(REPOSITORY_ROOT / "protocol.yaml")
SESSION_SEED = int(PROTOCOL["seeds"]["measurement_order"])
WORKLOAD = {
    "id": "unet_update_small",
    "kind": "unet",
    "mode": "update",
    "shape": {"batch": 1, "height": 16, "width": 16},
    "curvature": 0.1,
    "dtype": "float32",
}


def _state(workload: dict[str, Any], variant: str) -> tuple[Any, Any, Any]:
    _, state, inputs, curvature = _model_state(workload, variant, SESSION_SEED, PROTOCOL)
    return state, inputs, curvature


def _assert_identical(left: Any, right: Any) -> None:
    left_leaves, left_structure = jax.tree_util.tree_flatten(left)
    right_leaves, right_structure = jax.tree_util.tree_flatten(right)
    assert left_structure == right_structure
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        assert left_leaf.shape == right_leaf.shape
        assert left_leaf.dtype == right_leaf.dtype
        assert bool(jnp.array_equal(left_leaf, right_leaf))


@pytest.mark.parametrize("variant", ["R", "F", "E"])
@pytest.mark.parametrize("mode", ["forward", "update"])
def test_absent_initial_width_equals_registered_width_eight(
    variant: str, mode: str
) -> None:
    assert PROTOCOL["tasks"]["isic16_poincare_unet"]["model"]["initial_width"] == 8
    workload = {**WORKLOAD, "mode": mode}

    absent = _state(workload, variant)
    explicit = _state({**workload, "initial_width": 8}, variant)

    _assert_identical(absent, explicit)


def test_initial_width_override_scales_every_unet_level() -> None:
    (default_params, _), _, _ = _state(WORKLOAD, "F")
    (wide_params, _), _, _ = _state({**WORKLOAD, "initial_width": 16}, "F")

    default_shapes = [leaf.shape for leaf in jax.tree_util.tree_leaves(default_params)]
    wide_shapes = [leaf.shape for leaf in jax.tree_util.tree_leaves(wide_params)]
    assert len(default_shapes) == len(wide_shapes)
    assert default_params["input_block"]["conv1"].weight.shape[-1] == 8
    assert wide_params["input_block"]["conv1"].weight.shape[-1] == 16
    assert default_params["down_blocks"][-1]["conv2"].weight.shape[-1] == 128
    assert wide_params["down_blocks"][-1]["conv2"].weight.shape[-1] == 256
    assert default_params["output_conv"].weight.shape == (1, 1, 8, 2)
    assert wide_params["output_conv"].weight.shape == (1, 1, 16, 2)


@pytest.mark.parametrize("value", [0, -8, 8.0, True, "16"])
def test_invalid_initial_width_fails_closed(value: Any) -> None:
    with pytest.raises(ValueError, match="initial_width"):
        _state({**WORKLOAD, "initial_width": value}, "F")


def test_initial_width_is_rejected_outside_unet_workloads() -> None:
    workload = {
        "id": "resnet20_small",
        "kind": "resnet20",
        "mode": "forward",
        "shape": {"batch": 1, "height": 32, "width": 32},
        "curvature": 0.1,
        "dtype": "float32",
        "initial_width": 8,
    }
    with pytest.raises(ValueError, match="initial_width"):
        _state(workload, "F")
