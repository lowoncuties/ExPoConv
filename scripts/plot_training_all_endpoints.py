#!/usr/bin/env python3
"""Regenerate the accepted E09/E10 all-endpoints figure on CPU."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from poincare_exec.analysis.training_endpoint_figure import (
    source_sha256,
    write_training_endpoint_figure_from_csv,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = REPOSITORY_ROOT / "results" / "training"
DEFAULT_SOURCE = RESULT_DIR / "training_runs.csv"
DEFAULT_DESTINATION = RESULT_DIR / "training_all_endpoints"
DEFAULT_SOURCE_SHA256 = "208ddcca42ef3348d7d10df2435951d84e634ed4dffa1f4ddb1d6b0240523ef8"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Compact E09/E10 training table (default: tracked result table).",
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=DEFAULT_DESTINATION,
        help="Output path without an extension (writes both PDF and PNG).",
    )
    parser.add_argument(
        "--expected-source-sha256",
        help="Optional SHA-256 gate for a custom compact input table.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    source = arguments.input_csv.expanduser().resolve()
    destination = arguments.output_stem.expanduser().resolve()
    if destination == DEFAULT_DESTINATION.resolve() and source != DEFAULT_SOURCE.resolve():
        raise RuntimeError(
            "refusing to overwrite the tracked endpoint figure from a custom CSV; "
            "pass an explicit --output-stem"
        )
    expected_digest = arguments.expected_source_sha256
    if source == DEFAULT_SOURCE.resolve():
        expected_digest = expected_digest or DEFAULT_SOURCE_SHA256
    digest = source_sha256(source)
    if expected_digest is not None and digest != expected_digest.lower():
        raise RuntimeError(
            f"source CSV SHA-256 mismatch: expected {expected_digest.lower()}, "
            f"observed {digest}"
        )
    pdf, png = write_training_endpoint_figure_from_csv(source, destination)
    print(pdf)
    print(png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
