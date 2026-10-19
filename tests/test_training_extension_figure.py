"""Focused tests for compact E20/E21 quality-strata figure regeneration."""

from __future__ import annotations

import csv
import hashlib
import os
import subprocess
import sys
from itertools import pairwise
from pathlib import Path

import pytest
from PIL import Image

from poincare_exec.analysis.training_extension_figure import (
    REQUIRED_COLUMNS,
    build_quality_strata_figure,
    write_quality_strata_figure_from_csv,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = REPOSITORY_ROOT / "results" / "e20_e21_training_replication"
HISTORICAL_HASHES = {
    "e09_e10_e20_e21_quality_strata.pdf": (
        "bc1898eda8cf0ca69046a133de8cc3d03c936c591f8efc4011526ed0815018a6"
    ),
    "e09_e10_e20_e21_quality_strata.png": (
        "e330b9f7bba6052cc2994c26238c05fb959d5ce57a77751f9f5e77ff44e2eda3"
    ),
}
DERIVATIVE_HASHES = {
    "e09_e10_e20_e21_quality_strata_manuscript.pdf": (
        "00088b7e9717fb51be687edc531359505ffd168cb947ff6873c9e78a61ed2ec2"
    ),
    "e09_e10_e20_e21_quality_strata_manuscript.png": (
        "5678b7161cc1f89a5211fc9cda84a13266df3031b63a37ae4b9e153add577d1f"
    ),
}
ROWS = [
    {
        "experiment_id": "E09",
        "task": "isic16_poincare_unet",
        "physical_gpu_name": "Tesla V100-PCIE-32GB",
        "pairs": "5",
        "estimate": "-0.010016497575623885",
        "interval_95_lower": "-0.033349421561881185",
        "interval_95_upper": "0.013316426410633419",
        "endpoint": "fixed_final_update",
        "test_partition": "official_test",
        "comparison": "F_minus_R",
        "interval_method": "paired_t",
        "analysis_scope": "within_platform_no_grand_pool",
        "grand_pool_performed": "False",
        "result_family": "quality",
        "outcome": "mean_per_image_foreground_dice",
    },
    {
        "experiment_id": "E10",
        "task": "cifar100_poincare_resnet20",
        "physical_gpu_name": "Tesla V100-PCIE-32GB",
        "pairs": "5",
        "estimate": "0.022140000000000014",
        "interval_95_lower": "-0.008101414286713945",
        "interval_95_upper": "0.05238141428671397",
        "endpoint": "fixed_final_update",
        "test_partition": "official_test",
        "comparison": "F_minus_R",
        "interval_method": "paired_t",
        "analysis_scope": "within_platform_no_grand_pool",
        "grand_pool_performed": "False",
        "result_family": "quality",
        "outcome": "top1_accuracy",
    },
    {
        "experiment_id": "E20",
        "task": "isic16_poincare_unet",
        "physical_gpu_name": "NVIDIA RTX A6000",
        "pairs": "15",
        "estimate": "0.0026278039957840617",
        "interval_95_lower": "-0.004784650512074686",
        "interval_95_upper": "0.01004025850364281",
        "endpoint": "fixed_final_update",
        "test_partition": "official_test",
        "comparison": "F_minus_R",
        "interval_method": "paired_t",
        "analysis_scope": "within_platform_no_grand_pool",
        "grand_pool_performed": "False",
        "result_family": "quality",
        "outcome": "mean_per_image_foreground_dice",
    },
    {
        "experiment_id": "E21",
        "task": "cifar100_poincare_resnet20",
        "physical_gpu_name": "NVIDIA RTX A6000",
        "pairs": "15",
        "estimate": "-0.0017133333333333224",
        "interval_95_lower": "-0.005529197967388261",
        "interval_95_upper": "0.002102531300721616",
        "endpoint": "fixed_final_update",
        "test_partition": "official_test",
        "comparison": "F_minus_R",
        "interval_method": "paired_t",
        "analysis_scope": "within_platform_no_grand_pool",
        "grand_pool_performed": "False",
        "result_family": "quality",
        "outcome": "top1_accuracy",
    },
]


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_compact_csv(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted(REQUIRED_COLUMNS))
        writer.writeheader()
        writer.writerows(ROWS)


def test_figure_has_compact_labels_and_nonoverlapping_x_ticks() -> None:
    figure = build_quality_strata_figure(ROWS)
    try:
        figure.canvas.draw()
        assert figure.texts[0].get_text() == (
            "Matched-seed quality differences by cohort/platform"
        )
        assert [label.get_text() for label in figure.axes[0].get_yticklabels()] == [
            "E09 | V100 | n=5",
            "E20 | A6000 | n=15",
        ]
        assert [label.get_text() for label in figure.axes[1].get_yticklabels()] == [
            "E10 | V100 | n=5",
            "E21 | A6000 | n=15",
        ]

        renderer = figure.canvas.get_renderer()
        for axis in figure.axes:
            tick_boxes = sorted(
                (
                    label.get_window_extent(renderer)
                    for label in axis.get_xticklabels()
                    if label.get_visible() and label.get_text()
                ),
                key=lambda box: box.x0,
            )
            assert all(left.x1 <= right.x0 for left, right in pairwise(tick_boxes))
    finally:
        from matplotlib import pyplot as plt

        plt.close(figure)


def test_figure_rejects_pooled_rows() -> None:
    rows = [dict(row) for row in ROWS]
    rows[0]["grand_pool_performed"] = "True"
    with pytest.raises(ValueError, match="pooled rows"):
        build_quality_strata_figure(rows)


def test_compact_csv_cli_is_jax_free_and_byte_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "platform_strata.csv"
    _write_compact_csv(source)
    source_digest = _digest(source)
    import_guard = tmp_path / "import_guard"
    import_guard.mkdir()
    (import_guard / "jax.py").write_text(
        "raise AssertionError('compact figure CLI imported JAX')\n", encoding="utf-8"
    )
    environment = os.environ.copy()
    environment["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")
    environment["PYTHONPATH"] = os.pathsep.join(
        [
            str(import_guard),
            str(REPOSITORY_ROOT / "src"),
            environment.get("PYTHONPATH", ""),
        ]
    )
    script = REPOSITORY_ROOT / "scripts" / "plot_e20_e21_quality_strata.py"
    stems = [tmp_path / "first" / "quality", tmp_path / "second" / "quality"]
    for stem in stems:
        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "--input-csv",
                str(source),
                "--output-stem",
                str(stem),
            ],
            cwd=REPOSITORY_ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert stem.with_suffix(".pdf").is_file()
        assert stem.with_suffix(".png").is_file()

    assert _digest(stems[0].with_suffix(".pdf")) == _digest(stems[1].with_suffix(".pdf"))
    assert _digest(stems[0].with_suffix(".png")) == _digest(stems[1].with_suffix(".png"))
    pdf_bytes = stems[0].with_suffix(".pdf").read_bytes()
    assert b"/CreationDate" not in pdf_bytes
    assert b"/ModDate" not in pdf_bytes
    with Image.open(stems[0].with_suffix(".png")) as image:
        assert image.info["Software"] == "poincare-execution-experiments"
        assert "Creation Time" not in image.info
    assert _digest(source) == source_digest

    tracked_before = {name: _digest(RESULT_DIR / name) for name in DERIVATIVE_HASHES}
    refused = subprocess.run(
        [sys.executable, str(script), "--input-csv", str(source)],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused.returncode != 0
    assert "explicit --output-stem" in refused.stderr
    assert {
        name: _digest(RESULT_DIR / name) for name in DERIVATIVE_HASHES
    } == tracked_before


def test_tracked_historical_and_derivative_hashes(tmp_path: Path) -> None:
    assert {
        name: _digest(RESULT_DIR / name) for name in HISTORICAL_HASHES
    } == HISTORICAL_HASHES
    assert {
        name: _digest(RESULT_DIR / name) for name in DERIVATIVE_HASHES
    } == DERIVATIVE_HASHES

    regenerated = tmp_path / "e09_e10_e20_e21_quality_strata_manuscript"
    pdf, png = write_quality_strata_figure_from_csv(
        RESULT_DIR / "e09_e10_e20_e21_platform_strata.csv", regenerated
    )
    assert (
        _digest(pdf) == DERIVATIVE_HASHES["e09_e10_e20_e21_quality_strata_manuscript.pdf"]
    )
    assert (
        _digest(png) == DERIVATIVE_HASHES["e09_e10_e20_e21_quality_strata_manuscript.png"]
    )
