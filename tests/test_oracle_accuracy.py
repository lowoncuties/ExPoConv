from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from poincare_exec.core import geometry
from poincare_exec.core.hnnpp import HNNPPParams, conv2d
from poincare_exec.experiments.numerical_structured import StructuredCase
from scripts.run_oracle_accuracy import (
    evaluate_paths,
    fixed_projection_margin,
    paired_summary,
    tensor_rows,
)

CURVATURE = 0.0625  # exactly representable, like WEIGHT, so FP32 inputs are exact
WEIGHT = 0.25
TANGENT = 20.0  # sqrt(c) * 20 = 5, far beyond the FP32 source cap


def _hand_output(margin: float) -> float:
    """One-pixel, one-channel R output with an active input cap.

    The capped ball point has normalized radius m and conformal factor
    2 / (1 - m^2), so the score is 2|w| asinh(2m / (1 - m^2)) / sqrt(c). Its
    single-channel reconstruction returns log0 = |w| asinh(2m / (1 - m^2)) / sqrt(c),
    whose normalized ball radius stays below m (no output projection).
    """

    cap = 1.0 - margin
    return WEIGHT * math.asinh(2.0 * cap / (1.0 - cap * cap)) / math.sqrt(CURVATURE)


def _case() -> StructuredCase:
    params = HNNPPParams(
        weight=jnp.full((1, 1, 1, 1), WEIGHT, dtype=jnp.float32),
        offset=jnp.zeros((1,), dtype=jnp.float32),
    )
    inputs = jnp.full((1, 1, 1, 1), TANGENT, dtype=jnp.float32)

    def function(selected, tangent, curvature, variant):
        return conv2d(selected, tangent, curvature, variant=variant)

    return StructuredCase(params, inputs, function)


@pytest.fixture(autouse=True)
def _x64():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


def test_oracle_reproduces_hand_checked_capped_layer() -> None:
    case = _case()
    curvature = jnp.asarray(CURVATURE, dtype=jnp.float32)
    cotangent = jnp.ones((1, 1, 1, 1), dtype=jnp.float32)
    evaluated = evaluate_paths(case, curvature, cotangent, 0.004)

    oracle = float(evaluated[("R", "float64")]["output"].ravel()[0])
    assert oracle == pytest.approx(_hand_output(0.004), rel=1.0e-12)
    # The dtype-default FP64 cap 1 - 1e-5 lies above rho = tanh(5): unclipped, the
    # layer returns 2|w| artanh(rho) / sqrt(c) = 2|w| t, a different function.
    default = conv2d(
        _cast(case.parameters), _cast(case.inputs), jnp.float64(CURVATURE), variant="R"
    )
    assert float(default.ravel()[0]) == pytest.approx(2.0 * WEIGHT * TANGENT, rel=1.0e-12)
    assert abs(float(default.ravel()[0]) - oracle) > 1.0

    rows = tensor_rows(
        evaluated, norm_floor=1.0e-12, quantiles=(0.5, 0.9, 0.95, 0.99, 0.999), identity={}
    )
    assert len(rows) == 5 * 5  # five candidates; output, curvature, weight, offset, input
    by_key = {(row["variant"], row["dtype"], row["variable_group"]): row for row in rows}
    r32 = float(evaluated[("R", "float32")]["output"].ravel()[0])
    expected = abs(r32 - oracle) / abs(oracle)
    output = by_key[("R", "float32", "output")]
    assert output["relative_l2_error"] == pytest.approx(expected, rel=1.0e-12)
    assert output["p50"] == pytest.approx(expected, rel=1.0e-12)
    assert 0.0 < expected < 1.0e-6
    for variant in ("F", "T"):
        assert by_key[(variant, "float64", "output")]["relative_l2_error"] < 1.0e-14


def test_fixed_margin_is_restored() -> None:
    with fixed_projection_margin(0.004):
        assert float(geometry.projection_margin(jnp.float64)) == 0.004
    assert float(geometry.projection_margin(jnp.float64)) == 1.0e-5


def test_paired_summary_excludes_exact_zero_ties() -> None:
    def row(variant: str, group: str, error: float) -> dict[str, object]:
        return {
            "curvature": 0.1,
            "case_id": "case",
            "realization": 0,
            "variant": variant,
            "dtype": "float32",
            "scope": "gradient",
            "variable_group": group,
            "relative_l2_error": error,
            "max_absolute_error": error,
        }

    rows = [
        row("R", "a", 2.0),
        row("F", "a", 1.0),
        row("R", "b", 1.0),
        row("F", "b", 3.0),
        row("R", "zero", 0.0),
        row("F", "zero", 0.0),
    ]
    (summary,) = [
        item for item in paired_summary(rows) if item["case_scope"] == "all_cases"
    ]
    assert summary["tensor_count"] == 3
    assert summary["both_exact_tensors"] == 1
    assert summary["fraction_relative_l2_le_R"] == 0.5
    assert summary["fraction_relative_l2_le_2x_R"] == 0.5
    assert summary["median_relative_l2_ratio_to_R"] == pytest.approx(1.75)


def _cast(tree):
    return jax.tree_util.tree_map(lambda value: jnp.asarray(value, np.float64), tree)
