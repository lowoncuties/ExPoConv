"""Regression tests for order-independent matched-seed inference."""

from __future__ import annotations

import random
from typing import Any

import pytest

from poincare_exec.analysis.build_all import (
    paired_quality_summary,
    paired_training_efficiency_summary,
)


def _protocol() -> dict[str, Any]:
    return {
        "seeds": {"training": [11, 29, 47, 71, 101], "bootstrap": 20260907},
        "quality": {"evaluation_bootstrap_replicates": 256},
        "variants": {
            "core_training": ["R", "F", "T"],
            "training_roles": {
                "baseline": "R",
                "primary_method": "F",
                "ablation": "T",
            },
        },
        "tasks": {
            "task_a": {"primary_metric": "accuracy"},
            "task_b": {"primary_metric": "accuracy"},
        },
    }


def _rows(task: str, offset: float) -> list[dict[str, Any]]:
    rows = []
    for seed_index, seed in enumerate((11, 29, 47, 71, 101)):
        reference = 0.50 + offset + seed_index * 0.01
        for variant, quality_delta, time_scale in (
            ("R", 0.0, 1.0),
            ("F", 0.01 + seed_index * 0.001, 0.7),
            ("T", -0.005 + seed_index * 0.0005, 0.8),
        ):
            rows.append(
                {
                    "status": "ok",
                    "task": task,
                    "endpoint": "fixed_final_update",
                    "seed": seed,
                    "variant": variant,
                    "official_metric": reference + quality_delta,
                    "deduplicated_metric": reference + quality_delta / 2.0,
                    "compute_seconds": (100.0 + seed_index) * time_scale,
                    "elapsed_seconds": (120.0 + seed_index) * time_scale,
                }
            )
    return rows


def _record(rows: list[dict[str, Any]], **identity: str) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if all(row.get(field) == value for field, value in identity.items())
    ]
    assert len(selected) == 1
    return selected[0]


@pytest.mark.parametrize(
    ("summarizer", "identity", "interval_fields"),
    [
        (
            paired_quality_summary,
            {
                "task": "task_a",
                "endpoint": "fixed_final_update",
                "test_partition": "official_test",
                "comparison": "primary_vs_baseline",
            },
            (
                "paired_t_95_lower",
                "paired_t_95_upper",
                "bootstrap_2.5_percentile",
                "bootstrap_97.5_percentile",
                "bootstrap_stream_sha256",
            ),
        ),
        (
            paired_training_efficiency_summary,
            {
                "task": "task_a",
                "comparison": "primary_vs_baseline",
                "measurement": "compute_seconds",
            },
            (
                "log_t_95_lower",
                "log_t_95_upper",
                "bootstrap_2.5_percentile",
                "bootstrap_97.5_percentile",
                "bootstrap_stream_sha256",
            ),
        ),
    ],
)
def test_inference_is_independent_of_rows_and_unrelated_tasks(
    summarizer: Any, identity: dict[str, str], interval_fields: tuple[str, ...]
) -> None:
    protocol = _protocol()
    task_a = _rows("task_a", 0.0)
    baseline = _record(summarizer(task_a, protocol), **identity)

    reordered = list(reversed(task_a))
    with_unrelated_task = _rows("task_b", 0.2) + reordered
    random.Random(7).shuffle(with_unrelated_task)
    actual = _record(summarizer(with_unrelated_task, protocol), **identity)

    for field in interval_fields:
        assert actual[field] == baseline[field]


def test_quality_summary_reports_small_sample_sensitivity() -> None:
    record = _record(
        paired_quality_summary(_rows("task_a", 0.0), _protocol()),
        task="task_a",
        endpoint="fixed_final_update",
        test_partition="official_test",
        comparison="primary_vs_baseline",
    )

    assert record["difference_sample_sd"] > 0
    assert record["paired_t_95_lower"] < record["mean_difference"]
    assert record["paired_t_95_upper"] > record["mean_difference"]
    assert record["paired_t_degrees_of_freedom"] == 4
    assert record["exact_two_sided_sign_pvalue"] == pytest.approx(0.0625)
