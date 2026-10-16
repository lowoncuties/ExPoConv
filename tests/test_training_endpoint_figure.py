"""Focused tests for exact E09/E10 endpoint-figure regeneration."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from poincare_exec.analysis.training_endpoint_figure import (
    ENDPOINTS,
    SEEDS,
    TASK_ORDER,
    VARIANTS,
    build_training_endpoint_figure,
    read_training_runs_csv,
    validate_training_endpoint_rows,
    write_training_endpoint_figure_from_csv,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = REPOSITORY_ROOT / "results" / "training"
SOURCE = RESULT_DIR / "training_runs.csv"
SOURCE_SHA256 = "208ddcca42ef3348d7d10df2435951d84e634ed4dffa1f4ddb1d6b0240523ef8"
FIGURE_HASHES = {
    "training_all_endpoints.pdf": (
        "768ecf9cfa35197046d414caf65bc28dfdf47f4e59bb295a9b12d334120a1c6d"
    ),
    "training_all_endpoints.png": (
        "fcf4fab1adde69c3ae7d994267420c68eea1db300aac17fa7b81a636a98f35f2"
    ),
}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row_lookup(rows: list[dict[str, str]]) -> dict[tuple[str, str, int, str], float]:
    return {
        (row["task"], row["variant"], int(row["seed"]), row["endpoint"]): float(
            row["official_metric"]
        )
        for row in rows
    }


def test_figure_retains_every_matched_seed_and_endpoint() -> None:
    rows = read_training_runs_csv(SOURCE)
    lookup = _row_lookup(rows)
    figure = build_training_endpoint_figure(rows)
    try:
        figure.canvas.draw()
        assert len(figure.axes) == 4
        for task_index, task in enumerate(TASK_ORDER):
            task_limits = None
            for endpoint_index, (endpoint, endpoint_label) in enumerate(ENDPOINTS):
                axis = figure.axes[task_index * 2 + endpoint_index]
                assert endpoint_label in axis.get_title()
                assert [label.get_text() for label in axis.get_xticklabels()] == list(
                    VARIANTS
                )
                assert len(axis.lines) == len(SEEDS)
                for line, seed in zip(axis.lines, SEEDS, strict=True):
                    assert list(line.get_xdata()) == list(range(len(VARIANTS)))
                    assert list(line.get_ydata()) == [
                        lookup[(task, variant, seed, endpoint)]
                        for variant in VARIANTS
                    ]
                mean_collections = axis.collections[-len(VARIANTS) :]
                for variant, collection in zip(
                    VARIANTS, mean_collections, strict=True
                ):
                    offsets = collection.get_offsets()
                    expected_mean = sum(
                        lookup[(task, variant, seed, endpoint)] for seed in SEEDS
                    ) / len(SEEDS)
                    assert len(offsets) == 1
                    assert float(offsets[0, 1]) == pytest.approx(expected_mean)
                if task_limits is None:
                    task_limits = axis.get_ylim()
                else:
                    assert axis.get_ylim() == task_limits
    finally:
        from matplotlib import pyplot as plt

        plt.close(figure)


def test_validation_rejects_incomplete_or_mislabeled_cohorts() -> None:
    rows = read_training_runs_csv(SOURCE)
    with pytest.raises(ValueError, match="expected 60"):
        validate_training_endpoint_rows(rows[:-1])

    mislabeled = [dict(row) for row in rows]
    mislabeled[0]["config_hash"] = "0" * 64
    with pytest.raises(ValueError, match="unexpected config_hash"):
        validate_training_endpoint_rows(mislabeled)


def test_compact_csv_cli_is_jax_free_and_byte_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "training_runs.csv"
    shutil.copyfile(SOURCE, source)
    import_guard = tmp_path / "import_guard"
    import_guard.mkdir()
    (import_guard / "jax.py").write_text(
        "raise AssertionError('endpoint figure CLI imported JAX')\n", encoding="utf-8"
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
    script = REPOSITORY_ROOT / "scripts" / "plot_training_all_endpoints.py"
    stems = [tmp_path / "first" / "endpoints", tmp_path / "second" / "endpoints"]
    for stem in stems:
        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "--input-csv",
                str(source),
                "--output-stem",
                str(stem),
                "--expected-source-sha256",
                SOURCE_SHA256,
            ],
            cwd=REPOSITORY_ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr

    for extension in (".pdf", ".png"):
        assert _digest(stems[0].with_suffix(extension)) == _digest(
            stems[1].with_suffix(extension)
        )
    pdf_bytes = stems[0].with_suffix(".pdf").read_bytes()
    assert b"/CreationDate" not in pdf_bytes
    assert b"/ModDate" not in pdf_bytes
    assert SOURCE_SHA256.encode() in pdf_bytes
    with Image.open(stems[0].with_suffix(".png")) as image:
        assert image.info["Software"] == "poincare-execution-experiments"
        assert SOURCE_SHA256 in image.info["Source"]
        assert "Creation Time" not in image.info

    bad_hash = subprocess.run(
        [
            sys.executable,
            str(script),
            "--input-csv",
            str(source),
            "--output-stem",
            str(tmp_path / "bad"),
            "--expected-source-sha256",
            "0" * 64,
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert bad_hash.returncode != 0
    assert "SHA-256 mismatch" in bad_hash.stderr


def test_tracked_source_and_figure_hashes(tmp_path: Path) -> None:
    assert _digest(SOURCE) == SOURCE_SHA256
    assert {
        name: _digest(RESULT_DIR / name) for name in FIGURE_HASHES
    } == FIGURE_HASHES

    regenerated = tmp_path / "training_all_endpoints"
    pdf, png = write_training_endpoint_figure_from_csv(SOURCE, regenerated)
    assert _digest(pdf) == FIGURE_HASHES["training_all_endpoints.pdf"]
    assert _digest(png) == FIGURE_HASHES["training_all_endpoints.png"]
