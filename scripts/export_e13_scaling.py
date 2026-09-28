#!/usr/bin/env python3
"""Export tables and a four-panel figure from completed E13 raw evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

from poincare_exec.analysis.scaling import export_scaling_campaign


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CPU-only audit/export for the paired R/F U-Net scaling campaign."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--raw",
        type=Path,
        action="append",
        required=True,
        help="Raw CSV, JSON, or JSONL evidence. Repeat for failure sidecars.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    export_scaling_campaign(arguments.raw, arguments.manifest, arguments.output_dir)


if __name__ == "__main__":
    main()
