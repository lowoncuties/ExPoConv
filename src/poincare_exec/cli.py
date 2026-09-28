"""Command-line entry point for every reproducible workflow."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from poincare_exec.artifacts import capture_environment, write_json
from poincare_exec.config.constants import DEFAULT_PROTOCOL_PATH, HYPERBOLIC_VARIANTS
from poincare_exec.config.loader import load_yaml
from poincare_exec.data.splits import build_all_manifests
from poincare_exec.experiments.benchmark import benchmark_worker, run_benchmark_suite
from poincare_exec.experiments.checkpoint import run_trained_checkpoint_evaluation
from poincare_exec.experiments.checkpoint_campaign import (
    run_trained_checkpoint_campaign,
)
from poincare_exec.experiments.numerical_repair import run_numerical_repair
from poincare_exec.experiments.numerical_structured import run_structured_numerics
from poincare_exec.experiments.numerics import write_case_manifest
from poincare_exec.experiments.profile import profile_worker, run_profile_suite
from poincare_exec.experiments.retained_diagnostic_campaign import (
    run_retained_diagnostic_campaign,
)
from poincare_exec.experiments.retained_diagnostics import run_retained_diagnostic
from poincare_exec.training.campaign import run_training_campaign
from poincare_exec.training.runner import run_training_worker


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="poincare-exec")
    subparsers = parser.add_subparsers(dest="command", required=True)

    environment = subparsers.add_parser(
        "capture-environment", help="capture hardware and software provenance"
    )
    environment.add_argument("--output", type=Path, required=True)

    cases = subparsers.add_parser(
        "generate-cases", help="freeze the E02 pairwise numerical manifest"
    )
    cases.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    cases.add_argument("--output", type=Path, required=True)

    numerical_repair = subparsers.add_parser(
        "numerical-repair",
        help="run the bounded audit-driven E02 derivative-repair matrix",
    )
    numerical_repair.add_argument(
        "--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH
    )
    numerical_repair.add_argument("--config", type=Path, required=True)
    numerical_repair.add_argument("--cases", type=Path, required=True)
    numerical_repair.add_argument("--output-dir", type=Path, required=True)

    benchmark = subparsers.add_parser(
        "benchmark", help="run paired fresh-process benchmark sessions"
    )
    benchmark.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    benchmark.add_argument("--config", type=Path, required=True)
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--max-workloads", type=int)

    worker = subparsers.add_parser("benchmark-worker", help=argparse.SUPPRESS)
    worker.add_argument("--protocol", type=Path, required=True)
    worker.add_argument("--config", type=Path, required=True)
    worker.add_argument("--workload-id", required=True)
    worker.add_argument("--variant", required=True)
    worker.add_argument("--session", type=int, required=True)
    worker.add_argument("--output", type=Path, required=True)
    worker.add_argument("--phase-status", type=Path)

    splits = subparsers.add_parser(
        "make-splits", help="freeze the E09/E10 train/development/test manifests"
    )
    splits.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    splits.add_argument("--data-root", type=Path, required=True)
    splits.add_argument("--output-dir", type=Path, required=True)

    training = subparsers.add_parser(
        "training-worker", help="run or resume one E09/E10 variant-seed training run"
    )
    training.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    training.add_argument("--config", type=Path, required=True)
    training.add_argument("--variant", choices=HYPERBOLIC_VARIANTS, required=True)
    training.add_argument("--seed", type=int, required=True)
    training.add_argument("--data-root", type=Path, required=True)
    training.add_argument("--manifest-dir", type=Path, required=True)
    training.add_argument("--output-dir", type=Path, required=True)

    training_campaign = subparsers.add_parser(
        "training-campaign",
        help="run or resume every frozen variant-seed pair in fresh processes",
    )
    training_campaign.add_argument(
        "--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH
    )
    training_campaign.add_argument("--config", type=Path, required=True)
    training_campaign.add_argument("--data-root", type=Path, required=True)
    training_campaign.add_argument("--manifest-dir", type=Path, required=True)
    training_campaign.add_argument("--output-dir", type=Path, required=True)
    training_campaign.add_argument(
        "--accept-completed-commit",
        action="append",
        default=[],
        help=(
            "allow a clean completed run from this exact commit to be skipped; "
            "repeat once per explicitly reviewed compatible commit"
        ),
    )

    retained_checkpoint = subparsers.add_parser(
        "checkpoint-evaluation",
        help="run E06 R/F/T evaluation for one retained final R/T checkpoint",
    )
    retained_checkpoint.add_argument(
        "--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH
    )
    retained_checkpoint.add_argument("--config", type=Path, required=True)
    retained_checkpoint.add_argument("--training-config", type=Path, required=True)
    retained_checkpoint.add_argument("--checkpoint", type=Path, required=True)
    retained_checkpoint.add_argument("--data-root", type=Path, required=True)
    retained_checkpoint.add_argument("--manifest-dir", type=Path, required=True)
    retained_checkpoint.add_argument("--output-dir", type=Path, required=True)

    checkpoint_campaign = subparsers.add_parser(
        "checkpoint-campaign",
        help="evaluate every retained final checkpoint through R/F/T",
    )
    checkpoint_campaign.add_argument(
        "--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH
    )
    checkpoint_campaign.add_argument("--config", type=Path, required=True)
    checkpoint_campaign.add_argument(
        "--training-config", type=Path, required=True
    )
    checkpoint_campaign.add_argument("--data-root", type=Path, required=True)
    checkpoint_campaign.add_argument("--manifest-dir", type=Path, required=True)
    checkpoint_campaign.add_argument("--training-dir", type=Path, required=True)
    checkpoint_campaign.add_argument("--output-dir", type=Path, required=True)

    retained_diagnostic = subparsers.add_parser(
        "retained-diagnostic",
        help="compare R/F/T gradients and updates from one retained training state",
    )
    retained_diagnostic.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    retained_diagnostic.add_argument("--config", type=Path, required=True)
    retained_diagnostic.add_argument("--training-config", type=Path, required=True)
    retained_diagnostic.add_argument("--checkpoint", type=Path, required=True)
    retained_diagnostic.add_argument("--source-summary", type=Path, required=True)
    retained_diagnostic.add_argument("--data-root", type=Path, required=True)
    retained_diagnostic.add_argument("--manifest-dir", type=Path, required=True)
    retained_diagnostic.add_argument("--output-dir", type=Path, required=True)

    retained_diagnostic_campaign = subparsers.add_parser(
        "retained-diagnostic-campaign",
        help="run the frozen retained-state diagnostic sources in fresh processes",
    )
    retained_diagnostic_campaign.add_argument(
        "--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH
    )
    retained_diagnostic_campaign.add_argument("--config", type=Path, required=True)
    retained_diagnostic_campaign.add_argument(
        "--training-config", type=Path, required=True
    )
    retained_diagnostic_campaign.add_argument("--data-root", type=Path, required=True)
    retained_diagnostic_campaign.add_argument(
        "--manifest-dir", type=Path, required=True
    )
    retained_diagnostic_campaign.add_argument("--training-dir", type=Path, required=True)
    retained_diagnostic_campaign.add_argument("--output-dir", type=Path, required=True)

    profile = subparsers.add_parser(
        "profile", help="run fresh-process E11 compiler/device attribution"
    )
    profile.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    profile.add_argument("--config", type=Path, required=True)
    profile.add_argument("--output-dir", type=Path, required=True)

    profile_worker_parser = subparsers.add_parser(
        "profile-worker", help=argparse.SUPPRESS
    )
    profile_worker_parser.add_argument("--protocol", type=Path, required=True)
    profile_worker_parser.add_argument("--config", type=Path, required=True)
    profile_worker_parser.add_argument("--workload-id", required=True)
    profile_worker_parser.add_argument("--variant", required=True)
    profile_worker_parser.add_argument("--output-dir", type=Path, required=True)

    structured = subparsers.add_parser(
        "structured-numerics", help="run structured and finite-difference E02"
    )
    structured.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    structured.add_argument("--config", type=Path, required=True)
    structured.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch one requested experiment workflow."""

    arguments = _parser().parse_args(argv)
    if arguments.command == "capture-environment":
        write_json(arguments.output, capture_environment())
        return 0
    if arguments.command == "generate-cases":
        write_case_manifest(arguments.protocol, arguments.output)
        return 0
    if arguments.command == "numerical-repair":
        summary = run_numerical_repair(
            arguments.protocol,
            arguments.config,
            arguments.cases,
            arguments.output_dir,
        )
        print(
            f"wrote {summary['result_rows']} numerical repair rows; "
            f"required_contract_pass={summary['required_contract_pass']}"
        )
        return 0
    if arguments.command == "benchmark":
        written = run_benchmark_suite(
            arguments.protocol,
            arguments.config,
            arguments.output,
            max_workloads=arguments.max_workloads,
        )
        print(f"appended {written} benchmark rows")
        return 0
    if arguments.command == "benchmark-worker":
        benchmark_worker(
            arguments.protocol,
            arguments.config,
            workload_id=arguments.workload_id,
            variant=arguments.variant,
            session=arguments.session,
            output_path=arguments.output,
            phase_status_path=arguments.phase_status,
        )
        return 0
    if arguments.command == "make-splits":
        protocol = load_yaml(arguments.protocol)
        build_all_manifests(
            arguments.data_root,
            arguments.output_dir,
            seed=int(protocol["seeds"]["split"]),
        )
        return 0
    if arguments.command == "training-worker":
        run_training_worker(
            arguments.protocol,
            arguments.config,
            variant=arguments.variant,
            seed=arguments.seed,
            data_root=arguments.data_root,
            manifest_dir=arguments.manifest_dir,
            output_dir=arguments.output_dir,
        )
        return 0
    if arguments.command == "training-campaign":
        attempts = run_training_campaign(
            arguments.protocol,
            arguments.config,
            data_root=arguments.data_root,
            manifest_dir=arguments.manifest_dir,
            output_dir=arguments.output_dir,
            accepted_completed_commits=arguments.accept_completed_commit,
        )
        print(f"completed {attempts} training process attempts")
        return 0
    if arguments.command == "checkpoint-evaluation":
        run_trained_checkpoint_evaluation(
            arguments.protocol,
            arguments.config,
            arguments.training_config,
            checkpoint_path=arguments.checkpoint,
            data_root=arguments.data_root,
            manifest_dir=arguments.manifest_dir,
            output_dir=arguments.output_dir,
        )
        return 0
    if arguments.command == "checkpoint-campaign":
        attempts = run_trained_checkpoint_campaign(
            arguments.protocol,
            arguments.config,
            arguments.training_config,
            data_root=arguments.data_root,
            manifest_dir=arguments.manifest_dir,
            training_dir=arguments.training_dir,
            output_dir=arguments.output_dir,
        )
        print(f"completed {attempts} retained-checkpoint attempts")
        return 0
    if arguments.command == "retained-diagnostic":
        run_retained_diagnostic(
            arguments.protocol,
            arguments.config,
            arguments.training_config,
            checkpoint_path=arguments.checkpoint,
            source_summary_path=arguments.source_summary,
            data_root=arguments.data_root,
            manifest_dir=arguments.manifest_dir,
            output_dir=arguments.output_dir,
        )
        return 0
    if arguments.command == "retained-diagnostic-campaign":
        attempts = run_retained_diagnostic_campaign(
            arguments.protocol,
            arguments.config,
            arguments.training_config,
            data_root=arguments.data_root,
            manifest_dir=arguments.manifest_dir,
            training_dir=arguments.training_dir,
            output_dir=arguments.output_dir,
        )
        print(f"completed {attempts} retained-diagnostic attempts")
        return 0
    if arguments.command == "profile":
        written = run_profile_suite(
            arguments.protocol, arguments.config, arguments.output_dir
        )
        print(f"completed {written} profile groups")
        return 0
    if arguments.command == "profile-worker":
        profile_worker(
            arguments.protocol,
            arguments.config,
            workload_id=arguments.workload_id,
            variant=arguments.variant,
            output_dir=arguments.output_dir,
        )
        return 0
    if arguments.command == "structured-numerics":
        written = run_structured_numerics(
            arguments.protocol, arguments.config, arguments.output
        )
        print(f"appended {written} structured numerical rows")
        return 0
    raise AssertionError(f"unhandled command: {arguments.command}")


if __name__ == "__main__":
    raise SystemExit(main())
