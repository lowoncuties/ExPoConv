"""Regression tests for reviewer-requested numerical and equivalence summaries."""

from __future__ import annotations

import json

import numpy as np
import pytest

from poincare_exec.analysis.reviewer_requested import (
    elementwise_relative_error_summary,
    paired_mean_interval,
    paired_tost,
)
from poincare_exec.config.loader import load_yaml
from poincare_exec.experiments.numerical_structured import _elementwise_settings
from scripts.export_reviewer_requested_analysis import _elementwise_status


def test_elementwise_relative_error_reports_tail_and_floor_exposure() -> None:
    summary = elementwise_relative_error_summary(
        np.asarray([0.0, 1.0, 2.0, -4.0]),
        np.asarray([0.0, 1.1, 1.8, -4.0]),
        1.0e-6,
    )

    assert summary["elementwise_relative_error_elements"] == 4
    assert summary["elementwise_relative_error_denominator_floored_elements"] == 1
    assert summary["elementwise_relative_error_denominator_floored_fraction"] == 0.25
    assert summary["elementwise_relative_error_p50"] == pytest.approx(0.05)
    assert summary["elementwise_relative_error_p99_9"] < 0.101
    assert summary["elementwise_relative_error_max"] == pytest.approx(0.1)


def test_elementwise_relative_error_rejects_shape_or_quantile_ambiguity() -> None:
    with pytest.raises(ValueError, match="shapes differ"):
        elementwise_relative_error_summary([1.0], [1.0, 2.0], 1.0e-12)
    with pytest.raises(ValueError, match="strictly increasing"):
        elementwise_relative_error_summary([1.0], [1.0], 1.0e-12, quantiles=(0.95, 0.5))


def test_paired_tost_uses_both_one_sided_tests() -> None:
    equivalent = paired_tost([-0.001, 0.0, 0.001, 0.0, 0.0], margin=0.01)
    outside = paired_tost([0.02, 0.021, 0.019, 0.02, 0.02], margin=0.01)

    assert equivalent["equivalent_at_alpha"] is True
    assert equivalent["equivalence_ci_lower"] > -0.01
    assert equivalent["equivalence_ci_upper"] < 0.01
    assert outside["equivalent_at_alpha"] is False
    assert outside["tost_upper_pvalue"] > 0.05


def test_paired_tost_handles_zero_variance_without_nan() -> None:
    inside = paired_tost([0.0, 0.0, 0.0], margin=0.005)
    boundary = paired_tost([0.005, 0.005, 0.005], margin=0.005)

    assert inside["equivalent_at_alpha"] is True
    assert inside["tost_pvalue"] == 0.0
    assert boundary["equivalent_at_alpha"] is False
    assert boundary["tost_upper_pvalue"] == 1.0


def test_paired_mean_interval_preserves_paired_sample_size() -> None:
    summary = paired_mean_interval([0.1, -0.1, 0.0, 0.05])

    assert summary["pairs"] == 4
    assert summary["paired_t_degrees_of_freedom"] == 3
    assert summary["paired_t_interval_lower"] < summary["mean_difference"]
    assert summary["paired_t_interval_upper"] > summary["mean_difference"]


def test_accepted_aggregate_rows_are_not_misrepresented_as_elementwise(tmp_path) -> None:
    source = tmp_path / "structured.jsonl"
    source.write_text(
        json.dumps(
            {
                "record_type": "agreement",
                "relative_l2_error": 1.0e-6,
                "reference_l2_norm": 2.0,
                "max_absolute_error": 1.0e-7,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    status = _elementwise_status(source)

    assert status["status"] == "rerun_required"
    assert status["accepted_agreement_rows"] == 1
    assert "elementwise_relative_error_p99" in status["missing_elementwise_fields"]
    assert "cannot be reconstructed" in status["reason"]


def test_e26_freezes_elementwise_definition_without_changing_e02() -> None:
    protocol = load_yaml("protocol.yaml")
    e02 = load_yaml("configs/numerics/e02_repair_structured.yaml")
    e26 = load_yaml("configs/numerics/e26_elementwise_agreement.yaml")

    assert _elementwise_settings(e02, protocol) is None
    assert _elementwise_settings(e26, protocol) == (
        1.0e-12,
        (0.5, 0.9, 0.95, 0.99, 0.999),
    )
    assert e26["experiment_id"] == "E26"
    assert e26["finite_difference_relative_steps"] == []
