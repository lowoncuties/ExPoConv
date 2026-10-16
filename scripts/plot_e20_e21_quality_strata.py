#!/usr/bin/env python3
"""Regenerate the presentation-only quality-strata derivative on CPU from its CSV."""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Sequence
from pathlib import Path

from poincare_exec.analysis.training_extension_figure import (
    write_quality_strata_figure_from_csv,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = REPOSITORY_ROOT / "results" / "e20_e21_training_replication"
DEFAULT_SOURCE = RESULT_DIR / "e09_e10_e20_e21_platform_strata.csv"
DEFAULT_DESTINATION = RESULT_DIR / "e09_e10_e20_e21_quality_strata_manuscript"
DEFAULT_SOURCE_SHA256 = "6a587c4781817c3bc5a6b3840f92ede0fc64b4eb6b9e073c950bb6a53693995d"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Compact platform-strata CSV (default: tracked result table).",
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=DEFAULT_DESTINATION,
        help="Output path without an extension (writes both PDF and PNG).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    source = arguments.input_csv.expanduser().resolve()
    destination = arguments.output_stem.expanduser().resolve()
    if destination == DEFAULT_DESTINATION.resolve() and source != DEFAULT_SOURCE.resolve():
        raise RuntimeError(
            "refusing to overwrite the tracked manuscript derivative from a custom CSV; "
            "pass an explicit --output-stem"
        )
    if source == DEFAULT_SOURCE.resolve():
        source_sha256 = hashlib.sha256(DEFAULT_SOURCE.read_bytes()).hexdigest()
        if source_sha256 != DEFAULT_SOURCE_SHA256:
            raise RuntimeError(
                "tracked platform-strata CSV SHA-256 mismatch; refusing to regenerate "
                "the manuscript derivative"
            )
    pdf, png = write_quality_strata_figure_from_csv(source, destination)
    print(pdf)
    print(png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
