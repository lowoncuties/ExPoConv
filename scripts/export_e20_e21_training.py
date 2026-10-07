#!/usr/bin/env python3
"""Export the frozen E20/E21 replication analysis after its joint gate passes."""

from __future__ import annotations

import argparse
from pathlib import Path

from poincare_exec.analysis.training_extensions import (
    export_extension_campaign,
    parse_cohort_roots,
    parse_runtime_evidence_roots,
    parse_supplemental_evidence_roots,
)
from poincare_exec.config.constants import REPOSITORY_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPOSITORY_ROOT
        / "configs"
        / "analysis"
        / "e20_e21_training_replication.yaml",
    )
    parser.add_argument(
        "--cohort-root",
        action="append",
        required=True,
        metavar="EXPERIMENT_ID=PATH",
        help=(
            "Root containing summary.json files for one cohort. Repeat for "
            "E09, E10, E20, and E21."
        ),
    )
    parser.add_argument(
        "--runtime-evidence",
        action="append",
        default=[],
        metavar="EVIDENCE_ID=PATH",
        help="Registered accepted-runtime evidence root. Repeat when registered.",
    )
    parser.add_argument(
        "--supplemental-attempt-evidence",
        action="append",
        default=[],
        metavar="EVIDENCE_ID=PATH",
        help="Registered rejected-attempt evidence root. Repeat when registered.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    export_extension_campaign(
        arguments.manifest,
        parse_cohort_roots(arguments.cohort_root),
        arguments.output_dir,
        repository_root=REPOSITORY_ROOT,
        runtime_evidence_roots=parse_runtime_evidence_roots(arguments.runtime_evidence),
        supplemental_evidence_roots=parse_supplemental_evidence_roots(
            arguments.supplemental_attempt_evidence
        ),
    )


if __name__ == "__main__":
    main()
