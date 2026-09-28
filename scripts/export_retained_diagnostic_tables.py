#!/usr/bin/env python3
"""Validate and export the complete E07-J2 retained-state diagnostic matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from poincare_exec.analysis.retained_diagnostics import (
    export_retained_diagnostic_tables,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts/checkpoint_experiments/journal_diagnostics_v2"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/checkpoints/e07_journal_diagnostics_v2.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    summary = export_retained_diagnostic_tables(
        arguments.artifact_dir, arguments.config, arguments.output_dir
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
