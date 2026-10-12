"""Synthetic tests for cohort isolation and the E20/E21 endpoint embargo."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

import poincare_exec.analysis.training_extension_statistics as extension_statistics
import poincare_exec.analysis.training_extensions as extension_analysis
from poincare_exec.analysis.training_extensions import (
    ExtensionAnalysisError,
    export_extension_campaign,
)
from poincare_exec.analysis.training_extensions import (
    _analyze_extension_campaign_registered as analyze_extension_campaign,
)
from poincare_exec.training.optimizers import AdamState

SYNTHETIC_EPOCHS = 2
SYNTHETIC_UPDATES = 10

IDENTITY_FIELDS = [
    "analysis_id",
    "experiment_id",
    "task",
    "endpoint",
    "test_partition",
    "metric_name",
    "comparison",
    "reference_variant",
    "candidate_variant",
    "measurement",
]
OUTPUTS = {
    "gate_audit": "e20_e21_gate_audit.json",
    "run_audit": "e20_e21_run_audit.csv",
    "endpoint_rows": "e20_e21_training_runs.csv",
    "variant_summary": "e20_e21_training_variant_summary.csv",
    "paired_quality_by_seed": "e20_e21_paired_quality_by_seed.csv",
    "paired_quality": "e20_e21_paired_quality_summary.csv",
    "paired_iou": "e20_paired_iou_summary.csv",
    "paired_efficiency_by_seed": ("e20_e21_paired_training_efficiency_by_seed.csv"),
    "paired_efficiency": "e20_e21_paired_training_efficiency_summary.csv",
    "platform_strata": "e09_e10_e20_e21_platform_strata.csv",
    "forest_figure_stem": "e09_e10_e20_e21_quality_strata",
    "audit_summary": "e20_e21_analysis_summary.json",
}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        return pickle.load(stream)


def _write_pickle(path: Path, value: Any) -> None:
    with path.open("wb") as stream:
        pickle.dump(value, stream, protocol=5)


def _snapshot(device: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "ok",
        "index": device["physical_index"],
        "uuid": device["uuid"],
        "name": device["name"],
        "memory.used": 2,
        "memory.total": 49140,
        "utilization.gpu": 0,
        "compute_processes": [],
        "process_query_status": "ok",
    }


def _canonical_line_digest(value: dict[str, Any]) -> str:
    payload = (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _metrics(task: str, value: float, iou: float | None) -> dict[str, Any]:
    metric_name = (
        "mean_foreground_dice" if task == "isic16_poincare_unet" else "top1_accuracy"
    )
    output = {
        metric_name: value,
        "loss": 1.0 - value / 2.0,
        "examples": 4,
        "output_nonfinite_count": 0,
    }
    if iou is not None:
        output["mean_foreground_iou"] = iou
    return output


def _write_checkpoint(
    path: Path,
    cohort: dict[str, Any],
    *,
    variant: str,
    seed: int,
    commit: str,
    epoch: int,
    update: int,
    compute_seconds: float,
    elapsed_seconds: float,
    best_metric: float,
    best_update: int,
    nonfinite: bool = False,
) -> None:
    value = np.asarray([np.nan if nonfinite else 0.1], dtype=np.float32)
    state = AdamState(
        step=np.asarray(update, dtype=np.int32),
        first_moment={"weight": np.zeros_like(value)},
        second_moment={"weight": np.zeros_like(value)},
    )
    payload = {
        "format": "poincare-exec-checkpoint-v1",
        "config_hash": cohort["run_config_hash"],
        "protocol_version": cohort["protocol_version"],
        "code_commit": commit,
        "code_dirty": False,
        "task": cohort["task"],
        "variant": variant,
        "seed": seed,
        "epoch": epoch,
        "update": update,
        "elapsed_seconds": elapsed_seconds,
        "compute_seconds": compute_seconds,
        "best_metric": best_metric,
        "best_update": best_update,
        "params": {"weight": value},
        "optimizer_state": state,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        pickle.dump(payload, stream, protocol=5)


def _write_run(
    root: Path,
    cohort: dict[str, Any],
    *,
    variant: str,
    seed: int,
    seed_index: int,
) -> tuple[Path, dict[str, Any]]:
    run_dir = root / variant / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    task = cohort["task"]
    candidate = 1.0 if variant == "F" else 0.0
    if task == "isic16_poincare_unet":
        primary = 0.70 + seed_index * 0.01 + candidate * 0.02
        iou = 0.60 + seed_index * 0.01 + candidate * 0.01
    else:
        primary = 0.40 + seed_index * 0.01 + candidate * 0.03
        iou = None
    official = _metrics(task, primary, iou)
    deduplicated = _metrics(
        task,
        primary - 0.01 + candidate * 0.002,
        None if iou is None else iou - 0.01 + candidate * 0.001,
    )
    best_official = _metrics(
        task,
        primary + 0.005,
        None if iou is None else iou + 0.004,
    )
    best_deduplicated = _metrics(
        task,
        primary - 0.004 + candidate * 0.003,
        None if iou is None else iou - 0.005 + candidate * 0.002,
    )
    assignment = next(
        assignment
        for assignment in cohort["accepted_code_assignments"]
        if variant in assignment["variants"] and seed in assignment["seeds"]
    )
    device_id = cohort.get(
        "logical_device_id",
        f"{cohort['physical_device']['host']}::gpu:0::{cohort['physical_device']['name']}",
    )
    compute_seconds = 100.0 + seed_index if variant == "R" else 52.0 + 2.0 * seed_index
    elapsed_seconds = 150.0 + seed_index if variant == "R" else 103.0 + 3.0 * seed_index
    summary = {
        "experiment_id": cohort["experiment_id"],
        "protocol_version": cohort["protocol_version"],
        "config_hash": cohort["run_config_hash"],
        "code_commit": assignment["commit"],
        "code_dirty": False,
        "device_id": device_id,
        "task": task,
        "variant": variant,
        "seed_or_session": seed,
        "status": "ok",
        "epochs": SYNTHETIC_EPOCHS,
        "updates": SYNTHETIC_UPDATES,
        "expected_updates": SYNTHETIC_UPDATES,
        "compute_seconds": compute_seconds,
        "elapsed_seconds": elapsed_seconds,
        "best_development_metric": primary + 0.01,
        "best_update": SYNTHETIC_UPDATES,
        "official_test": official,
        "deduplicated_test": deduplicated,
        "validation_best": {
            "update": SYNTHETIC_UPDATES,
            "official_test": best_official,
            "deduplicated_test": best_deduplicated,
        },
    }
    summary_path = run_dir / "summary.json"
    _write_json(summary_path, summary)
    checkpoint_specs = (
        ("checkpoints/initial.pkl", 0, 0, -np.inf, -1),
        ("checkpoints/10_percent.pkl", 1, 1, -np.inf, -1),
        ("checkpoints/50_percent.pkl", 1, 5, -np.inf, -1),
        (
            "checkpoints/validation_best.pkl",
            SYNTHETIC_EPOCHS,
            SYNTHETIC_UPDATES,
            primary + 0.01,
            SYNTHETIC_UPDATES,
        ),
        (
            "checkpoints/final.pkl",
            SYNTHETIC_EPOCHS,
            SYNTHETIC_UPDATES,
            primary + 0.01,
            SYNTHETIC_UPDATES,
        ),
        (
            "resume.pkl",
            SYNTHETIC_EPOCHS,
            SYNTHETIC_UPDATES,
            primary + 0.01,
            SYNTHETIC_UPDATES,
        ),
    )
    for (
        relative,
        epoch,
        update,
        checkpoint_best,
        checkpoint_best_update,
    ) in checkpoint_specs:
        checkpoint_compute = compute_seconds * update / SYNTHETIC_UPDATES
        _write_checkpoint(
            run_dir / relative,
            cohort,
            variant=variant,
            seed=seed,
            commit=assignment["commit"],
            epoch=epoch,
            update=update,
            compute_seconds=checkpoint_compute,
            elapsed_seconds=checkpoint_compute + update * 0.1 + 0.1,
            best_metric=float(checkpoint_best),
            best_update=checkpoint_best_update,
        )
    sample_ids = np.asarray([f"sample-{index}" for index in range(4)])
    predictions = (
        np.zeros((4, 2, 2), dtype=np.uint8)
        if task == "isic16_poincare_unet"
        else np.zeros((4,), dtype=np.int16)
    )
    for relative in (
        "predictions_official_test.npz",
        "predictions_deduplicated_test.npz",
        "predictions_validation_best_official_test.npz",
        "predictions_validation_best_deduplicated_test.npz",
    ):
        np.savez_compressed(
            run_dir / relative, sample_id=sample_ids, prediction=predictions
        )
    primary_name = (
        "mean_foreground_dice" if task == "isic16_poincare_unet" else "top1_accuracy"
    )
    metric_identity = {
        "experiment_id": cohort["experiment_id"],
        "task": task,
        "protocol_version": cohort["protocol_version"],
        "config_hash": cohort["run_config_hash"],
        "code_commit": assignment["commit"],
        "code_dirty": False,
        "device_id": device_id,
        "variant": variant,
        "seed_or_session": seed,
    }
    training_rows = []
    for update in range(1, SYNTHETIC_UPDATES + 1):
        epoch = 1 if update <= SYNTHETIC_UPDATES // 2 else 2
        cumulative = compute_seconds * update / SYNTHETIC_UPDATES
        training_rows.append(
            {
                **metric_identity,
                "status": "ok",
                "split": "train",
                "epoch": epoch,
                "update": update,
                "examples": 2 if update % (SYNTHETIC_UPDATES // 2) == 0 else 4,
                "loss": 1.0,
                "gradient_norm": 0.5,
                "parameter_update_norm": 0.1,
                "curvature": 0.1,
                "parameter_nonfinite_count": 0,
                "optimizer_nonfinite_count": 0,
                "step_seconds": compute_seconds / SYNTHETIC_UPDATES,
                "compute_seconds": cumulative,
                "elapsed_seconds": cumulative + update * 0.1,
            }
        )
        if update in {SYNTHETIC_UPDATES // 2, SYNTHETIC_UPDATES}:
            development_value = primary + (0.01 if epoch == 2 else 0.0)
            training_rows.append(
                {
                    **metric_identity,
                    "status": "ok",
                    "split": "development",
                    "epoch": epoch,
                    "update": update,
                    "examples": 4,
                    "loss": 0.8,
                    "curvature": 0.1,
                    primary_name: development_value,
                    "output_nonfinite_count": 0,
                    "step_seconds": None,
                    "compute_seconds": cumulative,
                    "elapsed_seconds": cumulative + update * 0.1 + 0.05,
                }
            )
    (run_dir / "training_metrics.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in training_rows),
        encoding="utf-8",
    )
    return summary_path, summary


def _fixture(
    tmp_path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Path],
    dict[str, Path],
    dict[str, Path],
    Path,
    Path,
]:
    repository = tmp_path / "repository"
    repository.mkdir()
    splits_dir = repository / "splits"
    splits_dir.mkdir()
    split_digests = {}
    for dataset in ("isic16", "cifar100"):
        split_path = splits_dir / f"{dataset}.csv"
        split_path.write_text(
            "sample_id,split,exclusion_reason\n"
            + "".join(f"sample-{index},test,\n" for index in range(4)),
            encoding="utf-8",
        )
        split_digests[dataset] = _digest(split_path)
    protocol_path = repository / "protocol.yaml"
    protocol = {
        "data_independence": {"split_manifests": split_digests},
        "tasks": {
            "isic16_poincare_unet": {
                "epochs": SYNTHETIC_EPOCHS,
                "max_updates": SYNTHETIC_UPDATES,
                "input": {"height": 2, "width": 2},
            },
            "cifar100_poincare_resnet20": {
                "epochs": SYNTHETIC_EPOCHS,
                "max_updates": SYNTHETIC_UPDATES,
            },
        }
    }
    protocol_path.write_text(yaml.safe_dump(protocol, sort_keys=True), encoding="utf-8")
    contract_path = repository / "contract.md"
    contract_path.write_text("frozen synthetic contract\n", encoding="utf-8")

    extension_device = {
        "host": "aibox2a6000",
        "physical_index": 1,
        "uuid": "GPU-extension",
        "name": "NVIDIA RTX A6000",
        "driver": "test-driver",
        "advertised_cuda": "test-cuda",
    }
    original_device = {
        "host": "argexpr3.vsb.cz",
        "uuid": "GPU-original",
        "name": "Tesla V100-PCIE-32GB",
    }
    definitions = {
        "E09": (
            "isic16_poincare_unet",
            "isic16",
            ["R", "F", "T"],
            [11, 29],
            "a" * 40,
            original_device,
            "completed_original_v100",
        ),
        "E10": (
            "cifar100_poincare_resnet20",
            "cifar100",
            ["R", "F", "T"],
            [11, 29],
            "b" * 40,
            original_device,
            "completed_original_v100",
        ),
        "E20": (
            "isic16_poincare_unet",
            "isic16",
            ["R", "F"],
            [101, 103],
            "c" * 40,
            extension_device,
            "prospective_independent_platform_replication",
        ),
        "E21": (
            "cifar100_poincare_resnet20",
            "cifar100",
            ["R", "F"],
            [101, 103],
            "d" * 40,
            extension_device,
            "prospective_independent_platform_replication",
        ),
    }
    cohorts: dict[str, Any] = {}
    roots = {}
    for experiment_id, definition in definitions.items():
        task, dataset, variants, seeds, commit, device, role = definition
        config_path = repository / f"{experiment_id}.yaml"
        config = {
            "experiment_id": experiment_id,
            "task": task,
            "dataset": dataset,
            "variants": variants,
            "seeds": seeds,
        }
        config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
        cohort = {
            "experiment_id": experiment_id,
            "role": role,
            "task": task,
            "dataset": dataset,
            "config_document": {
                "path": config_path.relative_to(repository).as_posix(),
                "sha256": _digest(config_path),
            },
            "run_config_hash": (
                extension_analysis.config_hash(protocol, config)
                if experiment_id in {"E20", "E21"}
                else f"hash-{experiment_id}"
            ),
            "protocol_version": "test-1",
            "variants": variants,
            "analysis_variants": ["R", "F"],
            "seeds": seeds,
            "accepted_code_assignments": [
                {"commit": commit, "seeds": seeds, "variants": variants}
            ],
            "endpoint_example_counts": {
                endpoint: {"official_test": 4, "deduplicated_test": 4}
                for endpoint in ("fixed_final_update", "validation_best")
            },
            "training_health": {
                "full_train_rows_per_epoch": 4,
                "examples_per_full_train_row": 4,
                "final_train_rows_per_epoch": 1,
                "examples_per_final_train_row": 2,
                "total_train_rows_per_epoch": 5,
                "development_examples": 4,
                "epoch_index_origin": 1,
                "epoch_boundary_update_formula": "epoch_times_5",
            },
            "physical_device": device,
            "runtime": "synthetic runtime",
        }
        if experiment_id in {"E09", "E10"}:
            cohort["logical_device_id"] = "argexpr3.vsb.cz::gpu:0::Tesla V100-PCIE-32GB"
        cohorts[experiment_id] = cohort
        root = tmp_path / "artifacts" / experiment_id.lower()
        roots[experiment_id] = root
        attempts = []
        for seed_index, seed in enumerate(seeds):
            for variant in variants:
                summary_path, summary = _write_run(
                    root,
                    cohort,
                    variant=variant,
                    seed=seed,
                    seed_index=seed_index,
                )
                if experiment_id in {"E20", "E21"}:
                    attempts.append(
                        {
                            "experiment_id": experiment_id,
                            "protocol_version": cohort["protocol_version"],
                            "config_hash": cohort["run_config_hash"],
                            "code_commit": summary["code_commit"],
                            "code_dirty": False,
                            "task": task,
                            "variant": variant,
                            "seed": seed,
                            "status": "ok",
                            "returncode": 0,
                            "summary_path": str(summary_path),
                            "gpu_snapshot_before": _snapshot(extension_device),
                            "gpu_snapshot_after": _snapshot(extension_device),
                        }
                    )
        if attempts:
            (root / "campaign_attempts.jsonl").write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in attempts),
                encoding="utf-8",
            )

    manifest = {
        "schema_version": "1.0",
        "analysis_id": "synthetic_extension_analysis",
        "analysis_status": "frozen_before_official_test_unblinding",
        "protocol_document": {
            "path": "protocol.yaml",
            "sha256": _digest(protocol_path),
        },
        "contract_document": {
            "path": "contract.md",
            "sha256": _digest(contract_path),
        },
        "bootstrap": {
            "base_seed": 20260907,
            "replicates": 128,
            "quantile_method": "linear",
            "stream_identity_fields": IDENTITY_FIELDS,
        },
        "global_embargo": {
            "experiments": ["E20", "E21"],
            "required_successful_runs": 8,
            "required_complete_pairs": 4,
            "gate_is_joint_across_experiments": True,
        },
        "comparison": {
            "reference_variant": "R",
            "candidate_variant": "F",
            "t_ablation_extended": False,
        },
        "cohorts": cohorts,
        "cross_platform_reporting": {
            "task_strata": {
                "isic16_poincare_unet": ["E09", "E20"],
                "cifar100_poincare_resnet20": ["E10", "E21"],
            },
            "grand_pooled_estimate": "prohibited",
            "pooled_hypothesis_test": "prohibited",
            "inverse_variance_fixed_effect": "prohibited",
            "random_effects": "prohibited",
        },
        "failure_policy": {
            "replace_failed_seeds": False,
            "complete_case_primary_analysis": "prohibited",
            "pre_update_infrastructure_failure": {
                "retry_same_seed_and_variant": "allowed",
                "retain_every_attempt": True,
                "accepted_retry_must_match_frozen_identity": True,
                "failed_attempt_must_be_registered_before_endpoint_unblinding": True,
                "registration_section": "supplemental_attempt_evidence",
                "immutable_evidence_archive_required": True,
                "file_sha256_registration_required": True,
                "complete_identity_registration_required": True,
                "unregistered_attempt_disposition": "keep_global_embargo_closed",
            },
            "external_interruption_after_updates": {
                "exact_checkpoint_resume": "allowed",
                "retain_every_attempt": True,
                "provenance_match_required": True,
                "immutable_contemporaneous_checkpoint_required": True,
                "interruption_evidence_required": True,
                "archive_and_register_before_endpoint_unblinding": True,
                "registration_section": "supplemental_attempt_evidence",
                "file_sha256_registration_required": True,
                "complete_identity_registration_required": True,
                "unregistered_attempt_disposition": "keep_global_embargo_closed",
                "currently_registered_attempts": 0,
            },
            "scientific_or_nonfinite_failure": {
                "retry_to_obtain_endpoint": "prohibited",
                "disposition": "terminal_registered_failure",
            },
            "unresolved_or_incompatible_run": {
                "disposition": "keep_global_embargo_closed_and_report_campaign_incomplete"
            },
        },
        "required_outputs": dict(OUTPUTS),
    }
    runtime_roots, _ = _register_runtime_evidence(manifest, tmp_path)
    supplemental_roots = _register_supplemental_failure(manifest, tmp_path)
    manifest_path = repository / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return (
        manifest,
        roots,
        runtime_roots,
        supplemental_roots,
        repository,
        manifest_path,
    )


def _register_post_worker_disposition(
    manifest: dict[str, Any],
    roots: dict[str, Path],
    tmp_path: Path,
    *,
    experiment_id: str = "E20",
    variant: str = "R",
    seed: int = 101,
) -> dict[str, Path]:
    cohort = manifest["cohorts"][experiment_id]
    attempts_path = roots[experiment_id] / "campaign_attempts.jsonl"
    attempts = [
        json.loads(line)
        for line in attempts_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = [
        row
        for row in attempts
        if row["variant"] == variant and int(row["seed"]) == seed
    ]
    assert len(selected) == 1
    attempt = selected[0]
    attempt["timestamp_utc"] = "2026-10-07T20:13:52.671Z"
    attempt["gpu_snapshot_after"] = {
        **_snapshot(cohort["physical_device"]),
        "memory.used": 18496,
        "utilization.gpu": 57,
        "compute_processes": [
            {
                "gpu_uuid": cohort["physical_device"]["uuid"],
                "pid": 3723741,
                "process_name": "/usr/local/lib/ollama/llama-server",
                "used_memory_mib": 18482,
            }
        ],
    }
    attempts_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in attempts),
        encoding="utf-8",
    )

    evidence_id = "synthetic_post_worker_occupancy"
    evidence_root = tmp_path / "post-worker-evidence"
    evidence_root.mkdir(parents=True)
    campaign_snapshot = evidence_root / "campaign_attempts_pre_handoff.jsonl"
    campaign_snapshot.write_text(
        attempts_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    last_worker_line = "2026/10/07 20:13:51.779, 1, 59, 27119, 49140, SETUP"
    first_foreign_line = "2026/10/07 20:13:52.817, 1, 90, 18496, 49140, SETUP"
    file_contents = {
        "wrapper_telemetry": last_worker_line + "\n" + first_foreign_line + "\n",
        "wrapper_status": (
            "status\tFAILED_COMMAND\n"
            "command_exit_status\t1\n"
            "end_utc\t2026-10-07T20:14:25Z\n"
        ),
        "wrapper_stderr": (
            "RuntimeError: selected GPU compute processes did not clear within "
            "30 seconds; refusing final measurement\n"
        ),
        "wrapper_command": "synthetic registered campaign command\n",
        "wrapper_gpu_phase": "SETUP\n",
    }
    evidence_paths = {"campaign_attempts_snapshot": campaign_snapshot}
    for role, contents in file_contents.items():
        path = evidence_root / f"{role}.txt"
        path.write_text(contents, encoding="utf-8")
        evidence_paths[role] = path

    summary = roots[experiment_id] / variant / f"seed_{seed}" / "summary.json"
    summary_mtime_ns = 1_791_404_030_720_105_567
    os.utime(summary, ns=(summary_mtime_ns, summary_mtime_ns))
    manifest["post_worker_snapshot_policy"] = {
        "before_snapshot": "require_idle",
        "after_snapshot": "require_identity_and_exact_registration_if_nonidle",
    }
    manifest["post_worker_snapshot_evidence"] = [
        {
            "evidence_id": evidence_id,
            "input_kind": "post_worker_evidence_root",
            "identity": {
                "experiment_id": experiment_id,
                "variant": variant,
                "seed": seed,
            },
            "required_files": {
                role: {
                    "relative_path": path.relative_to(evidence_root).as_posix(),
                    "sha256": _digest(path),
                }
                for role, path in evidence_paths.items()
            },
            "required_text_lines": {
                "wrapper_telemetry": [last_worker_line, first_foreign_line],
                "wrapper_status": [
                    "status\tFAILED_COMMAND",
                    "command_exit_status\t1",
                ],
                "wrapper_stderr": [
                    "RuntimeError: selected GPU compute processes did not clear "
                    "within 30 seconds; refusing final measurement"
                ],
            },
            "expected_attempt": {
                "canonical_row_sha256": _canonical_line_digest(attempt),
                "after_snapshot_sha256": _canonical_line_digest(
                    attempt["gpu_snapshot_after"]
                ),
                "timestamp_utc": attempt["timestamp_utc"],
            },
            "summary_stat": {
                "relative_path": f"{variant}/seed_{seed}/summary.json",
                "size": summary.stat().st_size,
                "mtime_ns": summary_mtime_ns,
            },
            "timing_evidence": {
                "last_worker_allocation_line": last_worker_line,
                "first_foreign_only_line": first_foreign_line,
            },
            "classification": "foreign_process_started_after_completed_worker",
        }
    ]
    return {evidence_id: evidence_root}


def _record(rows: list[dict[str, Any]], **identity: Any) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if all(row.get(field) == value for field, value in identity.items())
    ]
    assert len(selected) == 1
    return selected[0]


def _register_supplemental_failure(
    manifest: dict[str, Any], tmp_path: Path
) -> dict[str, Path]:
    evidence_id = "synthetic_preupdate_failure"
    root = tmp_path / "rejected-attempt"
    cohort = manifest["cohorts"]["E20"]
    identity = {
        "experiment_id": "E20",
        "task": cohort["task"],
        "variant": "F",
        "seed": cohort["seeds"][0],
        "protocol_version": "test-old",
        "config_hash": "rejected-config",
        "code_commit": "e" * 40,
        "code_dirty": False,
    }
    outcome = {
        "status": "failed",
        "returncode": 1,
        "failure_class": "infrastructure_compiler_failure",
        "failure_stage": "before_first_training_update",
        "completed_updates": 0,
        "error_type": "XlaRuntimeError",
        "error": "synthetic compiler failure",
    }
    run_dir = root / "training" / "e20" / "F" / f"seed_{identity['seed']}"
    failure_path = run_dir / "synthetic.failure.json"
    _write_json(
        failure_path,
        {
            **{key: value for key, value in identity.items() if key != "seed"},
            "seed_or_session": identity["seed"],
            "status": outcome["status"],
            "error_type": outcome["error_type"],
            "error": outcome["error"],
        },
    )
    checkpoint_path = run_dir / "checkpoints" / "initial.pkl"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_value = np.asarray([0.1], dtype=np.float32)
    checkpoint_payload = {
        "format": "poincare-exec-checkpoint-v1",
        "config_hash": identity["config_hash"],
        "protocol_version": identity["protocol_version"],
        "code_commit": identity["code_commit"],
        "code_dirty": identity["code_dirty"],
        "task": identity["task"],
        "variant": identity["variant"],
        "seed": identity["seed"],
        "epoch": 0,
        "update": 0,
        "elapsed_seconds": 0.25,
        "compute_seconds": 0.0,
        "best_metric": -np.inf,
        "best_update": -1,
        "params": {"weight": checkpoint_value},
        "optimizer_state": AdamState(
            step=np.asarray(0, dtype=np.int32),
            first_moment={"weight": np.zeros_like(checkpoint_value)},
            second_moment={"weight": np.zeros_like(checkpoint_value)},
        ),
    }
    with checkpoint_path.open("wb") as stream:
        pickle.dump(checkpoint_payload, stream, protocol=5)
    environment_path = root / "provenance" / "environment.json"
    environment = {
        "captured_at_utc": "2026-10-07T00:00:00.000Z",
        "hostname": cohort["physical_device"]["host"],
        "platform": "synthetic-linux",
        "git": {
            "commit": identity["code_commit"],
            "dirty": identity["code_dirty"],
            "status_porcelain": [],
        },
        "python": {"version": "3.11.synthetic", "executable": "/old-env/python"},
        "jax": {
            "version": "0.5.3",
            "jaxlib_version": "0.5.3",
            "backend": "gpu",
            "x64_enabled": True,
            "devices": [
                {
                    "id": 0,
                    "platform": "gpu",
                    "device_kind": cohort["physical_device"]["name"],
                }
            ],
        },
        "environment": {
            "CUDA_VISIBLE_DEVICES": "1",
            "JAX_ENABLE_X64": "true",
            "JAX_PLATFORMS": None,
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "NVIDIA_TF32_OVERRIDE": None,
            "XLA_FLAGS": None,
        },
        "packages": {
            "jax": "0.5.3",
            "jaxlib": "0.5.3",
            "jax-cuda12-pjrt": "0.5.3",
            "jax-cuda12-plugin": "0.5.3",
            "numpy": "1.26.4",
            "scipy": "1.13.1",
            "nvidia-cuda-runtime-cu12": "12.9.79",
            "nvidia-cudnn-cu12": "9.25.1.1",
            "nvidia-cublas-cu12": "12.9.2.10",
        },
        "nvidia_smi": {"stdout": "synthetic"},
        "nvcc": {"stdout": "Cuda compilation tools, release 12.0, V12.0.140"},
    }
    _write_json(
        environment_path,
        environment,
    )
    attempt_path = root / "training" / "e20" / "campaign_attempts.jsonl"
    attempt_path.parent.mkdir(parents=True, exist_ok=True)
    attempt_path.write_text(
        json.dumps(
            {
                **identity,
                "status": outcome["status"],
                "returncode": outcome["returncode"],
                "summary_path": None,
                "stderr": f"traceback: {outcome['error']}",
                "gpu_snapshot_before": _snapshot(cohort["physical_device"]),
                "gpu_snapshot_after": _snapshot(cohort["physical_device"]),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    files = {
        "failure_manifest": failure_path,
        "initial_checkpoint": checkpoint_path,
        "environment": environment_path,
        "campaign_attempts": attempt_path,
    }
    manifest["supplemental_attempt_evidence"] = [
        {
            "evidence_id": evidence_id,
            "input_kind": "evidence_root",
            "required_files": {
                role: {
                    "relative_path": path.relative_to(root).as_posix(),
                    "sha256": _digest(path),
                }
                for role, path in files.items()
            },
            "expected_identity": identity,
            "expected_outcome": outcome,
            "expected_environment": {
                "captured_at_utc": environment["captured_at_utc"],
                "hostname": environment["hostname"],
                "platform": environment["platform"],
                "git": environment["git"],
                "python": environment["python"],
                "jax": {
                    "version": "0.5.3",
                    "jaxlib_version": "0.5.3",
                    "backend": "gpu",
                    "x64_enabled": True,
                    "device_count": 1,
                    "device_id": 0,
                    "device_platform": "gpu",
                    "device_kind": cohort["physical_device"]["name"],
                },
                "environment": environment["environment"],
                "packages": environment["packages"],
                "selected_physical_gpu": {
                    "physical_index": cohort["physical_device"]["physical_index"],
                    "uuid": cohort["physical_device"]["uuid"],
                    "name": cohort["physical_device"]["name"],
                    "bind_with_campaign_attempt_snapshots": True,
                },
                "contextual_system_toolkit": {
                    "nvcc_release": "12.0",
                    "nvcc_version": "12.0.140",
                    "used_for_jax_runtime": False,
                },
            },
            "disposition": "rejected_pre_update_infrastructure_attempt_retained_for_audit",
            "accepted_training_run": False,
            "counts_toward_embargo_gate": False,
            "endpoint_values_expected": False,
        }
    ]
    return {evidence_id: root}


def _register_runtime_evidence(
    manifest: dict[str, Any], tmp_path: Path
) -> tuple[dict[str, Path], Path]:
    evidence_id = "synthetic_accepted_runtime"
    root = tmp_path / "accepted-runtime"
    environment_path = root / "environment.json"
    environment = {
        "captured_at_utc": "2026-10-07T00:00:00.000Z",
        "hostname": "aibox2a6000",
        "platform": "synthetic-linux",
        "git": {"commit": "c" * 40, "dirty": False, "status_porcelain": []},
        "python": {"version": "3.11.synthetic", "executable": "/env/python"},
        "jax": {
            "version": "0.4.25",
            "jaxlib_version": "0.4.25",
            "backend": "gpu",
            "x64_enabled": True,
            "devices": [
                {
                    "id": 0,
                    "platform": "gpu",
                    "device_kind": "NVIDIA RTX A6000",
                }
            ],
        },
        "environment": {
            "CUDA_VISIBLE_DEVICES": "1",
            "JAX_ENABLE_X64": "true",
            "JAX_PLATFORMS": None,
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "NVIDIA_TF32_OVERRIDE": None,
            "XLA_FLAGS": None,
        },
        "packages": {"jax": "0.4.25", "jaxlib": "0.4.25+cuda11.cudnn86"},
        "nvidia_smi": {"stdout": "test-driver test-cuda"},
        "nvcc": {"stdout": "Cuda compilation tools, release 12.0, V12.0.140"},
    }
    _write_json(environment_path, environment)
    physical = manifest["cohorts"]["E20"]["physical_device"]
    manifest["accepted_runtime_evidence"] = [
        {
            "evidence_id": evidence_id,
            "input_kind": "accepted_evidence_root",
            "cohorts": ["E20", "E21"],
            "required_files": {
                "environment_record": {
                    "relative_path": environment_path.relative_to(root).as_posix(),
                    "sha256": _digest(environment_path),
                }
            },
            "expected_environment": {
                "captured_at_utc": environment["captured_at_utc"],
                "hostname": environment["hostname"],
                "platform": environment["platform"],
                "git": environment["git"],
                "python": environment["python"],
                "jax": {
                    "version": "0.4.25",
                    "jaxlib_version": "0.4.25",
                    "backend": "gpu",
                    "x64_enabled": True,
                    "device_count": 1,
                    "device_id": 0,
                    "device_platform": "gpu",
                    "device_kind": "NVIDIA RTX A6000",
                },
                "environment": environment["environment"],
                "packages": environment["packages"],
                "selected_physical_gpu": {
                    "physical_index": physical["physical_index"],
                    "uuid": physical["uuid"],
                    "name": physical["name"],
                    "driver": physical["driver"],
                    "driver_advertised_cuda": physical["advertised_cuda"],
                    "bind_with_campaign_attempt_snapshots": True,
                },
                "contextual_system_toolkit": {
                    "nvcc_release": "12.0",
                    "nvcc_version": "12.0.140",
                    "used_for_jax_runtime": False,
                },
            },
            "validation_order": "before_opening_any_e20_or_e21_summary",
            "accepted_training_run": False,
            "counts_toward_embargo_gate": False,
            "endpoint_values_expected": False,
        }
    ]
    return {evidence_id: root}, environment_path


def test_cohorts_are_isolated_and_outputs_are_deterministic(tmp_path: Path) -> None:
    manifest, roots, runtime_roots, supplemental_roots, repository, _ = _fixture(tmp_path)

    first = analyze_extension_campaign(
        manifest,
        roots,
        repository_root=repository,
        runtime_evidence_roots=runtime_roots,
        supplemental_evidence_roots=supplemental_roots,
    )
    second = analyze_extension_campaign(
        manifest,
        dict(reversed(list(roots.items()))),
        repository_root=repository,
        runtime_evidence_roots=runtime_roots,
        supplemental_evidence_roots=supplemental_roots,
    )

    assert first == second
    quality = first[OUTPUTS["paired_quality"]]
    assert {row["experiment_id"] for row in quality} == {"E20", "E21"}
    assert all(row["reference_variant"] == "R" for row in quality)
    assert all(row["candidate_variant"] == "F" for row in quality)
    assert all(row["pairs"] == 2 for row in quality)
    assert all(row["paired_t_degrees_of_freedom"] == 1 for row in quality)
    assert len(quality) == 8
    assert len(first[OUTPUTS["paired_iou"]]) == 4
    assert len(first[OUTPUTS["paired_quality_by_seed"]]) == 24
    assert all(row["bootstrap_quantile_method"] == "linear" for row in quality)

    official = _record(
        quality,
        experiment_id="E20",
        endpoint="fixed_final_update",
        test_partition="official_test",
    )
    deduplicated = _record(
        quality,
        experiment_id="E20",
        endpoint="fixed_final_update",
        test_partition="official_test_excluding_exact_training_duplicates",
    )
    assert official["mean_difference"] == pytest.approx(0.02)
    assert deduplicated["mean_difference"] == pytest.approx(0.022)
    assert official["bootstrap_stream_sha256"] != deduplicated["bootstrap_stream_sha256"]

    efficiency = first[OUTPUTS["paired_efficiency"]]
    compute = _record(efficiency, experiment_id="E20", measurement="compute_seconds")
    elapsed = _record(efficiency, experiment_id="E20", measurement="elapsed_seconds")
    assert compute["geometric_mean_speedup"] != elapsed["geometric_mean_speedup"]
    assert compute["bootstrap_stream_sha256"] != elapsed["bootstrap_stream_sha256"]

    strata = first[OUTPUTS["platform_strata"]]
    assert len(strata) == 4
    assert {row["experiment_id"] for row in strata} == {"E09", "E10", "E20", "E21"}
    assert all(row["grand_pool_performed"] is False for row in strata)
    assert all(row["analysis_scope"] == "within_platform_no_grand_pool" for row in strata)


def test_joint_gate_blocks_all_endpoint_extraction(tmp_path: Path) -> None:
    manifest, roots, runtime_roots, supplemental_roots, repository, _ = _fixture(tmp_path)
    missing = roots["E21"] / "F" / "seed_103" / "summary.json"
    missing.unlink()
    blinded = roots["E20"] / "F" / "seed_101" / "summary.json"
    value = json.loads(blinded.read_text(encoding="utf-8"))
    value["official_test"]["mean_foreground_dice"] = "embargoed-not-numeric"
    _write_json(blinded, value)

    with pytest.raises(ExtensionAnalysisError, match="incomplete run matrix"):
        analyze_extension_campaign(
            manifest,
            roots,
            repository_root=repository,
            runtime_evidence_roots=runtime_roots,
            supplemental_evidence_roots=supplemental_roots,
        )


def test_foreign_process_before_worker_always_closes_gate(tmp_path: Path) -> None:
    manifest, roots, runtime_roots, supplemental_roots, repository, _ = _fixture(tmp_path)
    attempts_path = roots["E20"] / "campaign_attempts.jsonl"
    attempts = [
        json.loads(line)
        for line in attempts_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    attempts[0]["gpu_snapshot_before"]["compute_processes"] = [
        {"process_name": "foreign", "pid": 7}
    ]
    attempts_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in attempts),
        encoding="utf-8",
    )

    with pytest.raises(ExtensionAnalysisError, match="competing compute processes"):
        analyze_extension_campaign(
            manifest,
            roots,
            repository_root=repository,
            runtime_evidence_roots=runtime_roots,
            supplemental_evidence_roots=supplemental_roots,
        )


def test_registered_post_worker_process_is_accepted_exactly(tmp_path: Path) -> None:
    manifest, roots, runtime_roots, supplemental_roots, repository, _ = _fixture(tmp_path)
    post_worker_roots = _register_post_worker_disposition(manifest, roots, tmp_path)

    outputs = analyze_extension_campaign(
        manifest,
        roots,
        repository_root=repository,
        runtime_evidence_roots=runtime_roots,
        supplemental_evidence_roots=supplemental_roots,
        post_worker_evidence_roots=post_worker_roots,
    )

    row = _record(
        outputs[OUTPUTS["run_audit"]],
        experiment_id="E20",
        variant="R",
        seed=101,
        record_type="accepted_registered_training_run",
    )
    assert row["post_worker_snapshot_disposition"] == (
        "registered_post_worker_occupancy"
    )
    assert row["post_worker_evidence_id"] == "synthetic_post_worker_occupancy"

    attempts_path = roots["E20"] / "campaign_attempts.jsonl"
    attempts = [
        json.loads(line)
        for line in attempts_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = [
        attempt
        for attempt in attempts
        if attempt["variant"] == "R" and int(attempt["seed"]) == 101
    ]
    assert len(selected) == 1
    selected[0]["gpu_snapshot_after"]["compute_processes"][0]["pid"] = 8
    attempts_path.write_text(
        "".join(json.dumps(attempt, sort_keys=True) + "\n" for attempt in attempts),
        encoding="utf-8",
    )
    with pytest.raises(ExtensionAnalysisError, match="canonical_row_sha256"):
        analyze_extension_campaign(
            manifest,
            roots,
            repository_root=repository,
            runtime_evidence_roots=runtime_roots,
            supplemental_evidence_roots=supplemental_roots,
            post_worker_evidence_roots=post_worker_roots,
        )


def test_unregistered_post_worker_process_closes_gate(tmp_path: Path) -> None:
    manifest, roots, runtime_roots, supplemental_roots, repository, _ = _fixture(tmp_path)
    attempts_path = roots["E20"] / "campaign_attempts.jsonl"
    attempts = [
        json.loads(line)
        for line in attempts_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    attempts[0]["gpu_snapshot_after"]["compute_processes"] = [
        {"process_name": "foreign", "pid": 7}
    ]
    attempts_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in attempts),
        encoding="utf-8",
    )

    with pytest.raises(
        ExtensionAnalysisError, match="no exact pre-unblinding disposition"
    ):
        analyze_extension_campaign(
            manifest,
            roots,
            repository_root=repository,
            runtime_evidence_roots=runtime_roots,
            supplemental_evidence_roots=supplemental_roots,
        )


def test_empty_post_worker_process_list_allows_lagging_utilization(
    tmp_path: Path,
) -> None:
    manifest, roots, runtime_roots, supplemental_roots, repository, _ = _fixture(tmp_path)
    attempts_path = roots["E20"] / "campaign_attempts.jsonl"
    attempts = [
        json.loads(line)
        for line in attempts_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    attempts[0]["gpu_snapshot_after"]["memory.used"] = 2
    attempts[0]["gpu_snapshot_after"]["utilization.gpu"] = 99
    attempts[0]["gpu_snapshot_after"]["compute_processes"] = []
    attempts_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in attempts),
        encoding="utf-8",
    )

    outputs = analyze_extension_campaign(
        manifest,
        roots,
        repository_root=repository,
        runtime_evidence_roots=runtime_roots,
        supplemental_evidence_roots=supplemental_roots,
    )
    assert outputs[OUTPUTS["run_audit"]]


@pytest.mark.parametrize(
    ("filename", "amendment_id", "frozen", "evidence_count"),
    [
        (
            "e20_e21_training_replication_v2.yaml",
            "e20_e21_training_replication_v2",
            "2026-10-07",
            1,
        ),
        (
            "e20_e21_training_replication_v3.yaml",
            "e20_e21_training_replication_v3",
            "2026-10-12",
            3,
        ),
    ],
)
def test_amendment_overlay_preserves_the_frozen_base(
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
    amendment_id: str,
    frozen: str,
    evidence_count: int,
) -> None:
    repository = extension_analysis.SOURCE_REPOSITORY_ROOT
    manifest_path = (
        repository
        / "configs"
        / "analysis"
        / filename
    )
    monkeypatch.setattr(
        extension_analysis,
        "_validate_analysis_checkout",
        lambda *_args, **_kwargs: {"commit": "test"},
    )

    manifest, selected_path, selected_repository, _ = (
        extension_analysis._load_checked_manifest(manifest_path, repository)
    )

    assert selected_path == manifest_path
    assert selected_repository == repository
    assert manifest["analysis_id"] == "e20_e21_training_replication_v1"
    assert manifest["amendment_id"] == amendment_id
    assert manifest["frozen"] == frozen
    assert manifest["base_analysis_id"] == "e20_e21_training_replication_v1"
    assert manifest["cohorts"]["E20"]["run_config_hash"].startswith("b226335a")
    assert len(manifest["post_worker_snapshot_evidence"]) == evidence_count
    if amendment_id.endswith("_v3"):
        assert manifest["prior_amendment_manifest_document"]["path"].endswith(
            "e20_e21_training_replication_v2.yaml"
        )
        assert manifest["prior_amendment_contract_document"]["path"].endswith(
            "e20_e21_analysis_contract_amendment_v2.md"
        )
        registrations = manifest["post_worker_snapshot_evidence"]
        assert {
            (
                item["evidence_id"],
                item["identity"]["experiment_id"],
                item["identity"]["variant"],
                item["identity"]["seed"],
            )
            for item in registrations
        } == {
            ("e20_r1338995025_post_worker_ollama", "E20", "R", 1338995025),
            ("e21_r1619140427_post_worker_ollama", "E21", "R", 1619140427),
            ("e21_f646228129_post_worker_ollama", "E21", "F", 646228129),
        }
        e21 = [
            item for item in registrations if item["identity"]["experiment_id"] == "E21"
        ]
        assert len({item["archived_remote_root"] for item in e21}) == 1
        assert {
            item["expected_attempt"]["timestamp_utc"] for item in e21
        } == {
            "2026-10-08T08:03:43.159Z",
            "2026-10-08T09:23:01.021Z",
        }

    base = yaml.safe_load(
        (repository / "configs/analysis/e20_e21_training_replication.yaml").read_text(
            encoding="utf-8"
        )
    )
    identities = (
        (
            "E20",
            "isic16_poincare_unet",
            "fixed_final_update",
            "official_test",
            "mean_foreground_dice",
            "F_minus_R",
            "R",
            "F",
            "not_applicable",
        ),
        (
            "E21",
            "cifar100_poincare_resnet20",
            "fixed_final_update",
            "not_applicable",
            "not_applicable",
            "R_over_F",
            "R",
            "F",
            "compute_seconds",
        ),
    )
    for identity in identities:
        _, base_stream = extension_statistics._bootstrap_generator(
            int(base["bootstrap"]["base_seed"]),
            str(base["analysis_id"]),
            *identity,
        )
        _, amended_stream = extension_statistics._bootstrap_generator(
            int(manifest["bootstrap"]["base_seed"]),
            str(manifest["analysis_id"]),
            *identity,
        )
        assert amended_stream == base_stream


def test_gate_does_not_deserialize_any_summary_before_health_is_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, roots, runtime_roots, supplemental_roots, repository, _ = _fixture(tmp_path)
    summary_path = roots["E20"] / "F" / "seed_101" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["official_test"]["mean_foreground_dice"] = "embargoed-not-numeric"
    _write_json(summary_path, summary)
    (roots["E21"] / "F" / "seed_103" / "checkpoints" / "50_percent.pkl").unlink()
    original_loads = json.loads

    def guarded_loads(value: str, *args: Any, **kwargs: Any) -> Any:
        if "embargoed-not-numeric" in value:
            raise AssertionError("endpoint-bearing JSON was deserialized before the gate")
        return original_loads(value, *args, **kwargs)

    monkeypatch.setattr(extension_analysis.json, "loads", guarded_loads)
    with pytest.raises(
        ExtensionAnalysisError, match="missing or empty registered checkpoints"
    ):
        analyze_extension_campaign(
            manifest,
            roots,
            repository_root=repository,
            runtime_evidence_roots=runtime_roots,
            supplemental_evidence_roots=supplemental_roots,
        )


def test_registered_rejected_attempt_is_audited_but_not_counted(tmp_path: Path) -> None:
    manifest, roots, runtime_roots, supplemental_roots, repository, _ = _fixture(tmp_path)

    outputs = analyze_extension_campaign(
        manifest,
        roots,
        repository_root=repository,
        runtime_evidence_roots=runtime_roots,
        supplemental_evidence_roots=supplemental_roots,
    )

    audit = outputs[OUTPUTS["run_audit"]]
    rejected = _record(audit, record_type="rejected_pre_update_infrastructure_attempt")
    assert rejected["accepted_training_run"] is False
    assert rejected["counts_toward_embargo_gate"] is False
    same_pair = [
        row
        for row in audit
        if row["experiment_id"] == "E20"
        and row["variant"] == "F"
        and row["seed"] == manifest["cohorts"]["E20"]["seeds"][0]
    ]
    assert {row["record_type"] for row in same_pair} == {
        "accepted_registered_training_run",
        "rejected_pre_update_infrastructure_attempt",
    }


def test_runtime_evidence_is_required_and_checked_before_endpoints(
    tmp_path: Path,
) -> None:
    manifest, roots, runtime_roots, supplemental_roots, repository, _ = _fixture(tmp_path)
    environment_path = next(iter(runtime_roots.values())) / "environment.json"

    with pytest.raises(ExtensionAnalysisError, match="runtime evidence roots"):
        analyze_extension_campaign(manifest, roots, repository_root=repository)

    outputs = analyze_extension_campaign(
        manifest,
        roots,
        repository_root=repository,
        runtime_evidence_roots=runtime_roots,
        supplemental_evidence_roots=supplemental_roots,
    )
    assert outputs["_runtime_audit"][0]["runtime_identity_validated"] is True
    assert outputs["_runtime_audit"][0]["evidence_root"]
    assert outputs["_runtime_audit"][0]["registered_relative_file"]
    assert outputs["_runtime_audit"][0]["matched_software"]["jax"]["version"]
    assert outputs["_runtime_audit"][0]["matched_device"]["uuid"]

    blinded = roots["E20"] / "F" / "seed_101" / "summary.json"
    value = json.loads(blinded.read_text(encoding="utf-8"))
    value["official_test"]["mean_foreground_dice"] = "embargoed-not-numeric"
    _write_json(blinded, value)
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    environment["jax"]["version"] = "unexpected"
    _write_json(environment_path, environment)
    registration = manifest["accepted_runtime_evidence"][0]
    registration["required_files"]["environment_record"]["sha256"] = _digest(
        environment_path
    )

    with pytest.raises(ExtensionAnalysisError, match="jax.*version"):
        analyze_extension_campaign(
            manifest,
            roots,
            repository_root=repository,
            runtime_evidence_roots=runtime_roots,
            supplemental_evidence_roots=supplemental_roots,
        )


def test_legacy_attempt_history_is_not_reinterpreted_as_extension_gate(
    tmp_path: Path,
) -> None:
    manifest, roots, runtime_roots, supplemental_roots, repository, _ = _fixture(tmp_path)
    legacy_attempts = roots["E09"] / "campaign_attempts.jsonl"
    legacy_attempts.write_text(
        json.dumps({"status": "failed", "variant": "F", "seed": 11}) + "\n",
        encoding="utf-8",
    )

    outputs = analyze_extension_campaign(
        manifest,
        roots,
        repository_root=repository,
        runtime_evidence_roots=runtime_roots,
        supplemental_evidence_roots=supplemental_roots,
    )

    assert len(outputs[OUTPUTS["platform_strata"]]) == 4


def test_registered_failed_attempt_can_precede_accepted_retry(tmp_path: Path) -> None:
    manifest, roots, runtime_roots, supplemental_roots, repository, _ = _fixture(tmp_path)
    rejected_root = next(iter(supplemental_roots.values()))
    rejected_path = rejected_root / "training" / "e20" / "campaign_attempts.jsonl"
    accepted_path = roots["E20"] / "campaign_attempts.jsonl"
    accepted_path.write_text(
        rejected_path.read_text(encoding="utf-8")
        + accepted_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    outputs = analyze_extension_campaign(
        manifest,
        roots,
        repository_root=repository,
        runtime_evidence_roots=runtime_roots,
        supplemental_evidence_roots=supplemental_roots,
    )

    accepted = [
        row
        for row in outputs[OUTPUTS["run_audit"]]
        if row["record_type"] == "accepted_registered_training_run"
    ]
    assert len(accepted) == 8
    assert all(row["attempt_path"] and row["attempt_row"] for row in accepted)
    rejected = _record(
        outputs[OUTPUTS["run_audit"]],
        record_type="rejected_pre_update_infrastructure_attempt",
    )
    assert rejected["accepted_root_attempt_path"] == str(accepted_path)
    assert rejected["accepted_root_attempt_row"] == 1


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("mixed_provenance", "config_hash"),
        ("missing_update", "training update history"),
        ("compute_recurrence", "compute_seconds recurrence from step_seconds"),
        ("development_compute", "development compute_seconds"),
    ],
)
def test_metric_history_is_exact_and_provenance_complete(
    tmp_path: Path, mutation: str, message: str
) -> None:
    manifest, roots, runtime_roots, supplemental_roots, repository, _ = _fixture(tmp_path)
    path = roots["E20"] / "R" / "seed_101" / "training_metrics.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if mutation == "mixed_provenance":
        rows[0]["config_hash"] = "wrong"
    elif mutation == "missing_update":
        rows = rows[1:]
    elif mutation == "compute_recurrence":
        next(row for row in rows if row["split"] == "train")["compute_seconds"] += 0.01
    elif mutation == "development_compute":
        next(row for row in rows if row["split"] == "development")[
            "compute_seconds"
        ] += 0.01
    else:  # pragma: no cover - exhaustive parameter
        raise AssertionError(mutation)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(ExtensionAnalysisError, match=message):
        analyze_extension_campaign(
            manifest,
            roots,
            repository_root=repository,
            runtime_evidence_roots=runtime_roots,
            supplemental_evidence_roots=supplemental_roots,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("corrupt", "checkpoint cannot be loaded"),
        ("wrong_identity", "config_hash"),
        ("swapped", "variant"),
        ("milestone", "update"),
        ("nonfinite", "params"),
        ("moment_structure", "tree structure does not match params"),
        ("moment_shape", "shape.*does not match params shape"),
        ("moment_dtype", "dtype.*does not match params dtype"),
        ("parameter_tree_drift", "parameter tree signature across checkpoints"),
    ],
)
def test_checkpoint_payloads_are_identity_complete_and_finite(
    tmp_path: Path, mutation: str, message: str
) -> None:
    manifest, roots, runtime_roots, supplemental_roots, repository, _ = _fixture(tmp_path)
    target = roots["E20"] / "R" / "seed_101" / "checkpoints" / "50_percent.pkl"
    if mutation == "corrupt":
        target.write_bytes(b"not-a-checkpoint")
    elif mutation == "swapped":
        source = roots["E20"] / "F" / "seed_101" / "checkpoints" / "50_percent.pkl"
        target.write_bytes(source.read_bytes())
    else:
        payload = _read_pickle(target)
        if mutation == "wrong_identity":
            payload["config_hash"] = "wrong"
        elif mutation == "milestone":
            payload["update"] = 4
        elif mutation == "nonfinite":
            payload["params"]["weight"][0] = np.nan
        elif mutation == "moment_structure":
            payload["optimizer_state"] = payload["optimizer_state"]._replace(
                first_moment={}
            )
        elif mutation == "moment_shape":
            payload["optimizer_state"] = payload["optimizer_state"]._replace(
                second_moment={"weight": np.zeros((2,), dtype=np.float32)}
            )
        elif mutation == "moment_dtype":
            payload["optimizer_state"] = payload["optimizer_state"]._replace(
                first_moment={"weight": np.zeros((1,), dtype=np.float64)}
            )
        elif mutation == "parameter_tree_drift":
            value = payload["params"].pop("weight")
            payload["params"]["replacement"] = value
            payload["optimizer_state"] = payload["optimizer_state"]._replace(
                first_moment={"replacement": np.zeros_like(value)},
                second_moment={"replacement": np.zeros_like(value)},
            )
        else:  # pragma: no cover - exhaustive parameter
            raise AssertionError(mutation)
        _write_pickle(target, payload)

    with pytest.raises(ExtensionAnalysisError, match=message):
        analyze_extension_campaign(
            manifest,
            roots,
            repository_root=repository,
            runtime_evidence_roots=runtime_roots,
            supplemental_evidence_roots=supplemental_roots,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("full_batch", "batch cardinalities"),
        ("remainder", "batch cardinalities"),
        ("development_count", "examples"),
        ("development_boundary", "row schedule"),
    ],
)
def test_training_health_enforces_exact_batching_and_epoch_boundaries(
    tmp_path: Path, mutation: str, message: str
) -> None:
    manifest, roots, runtime_roots, supplemental_roots, repository, _ = _fixture(tmp_path)
    path = roots["E20"] / "R" / "seed_101" / "training_metrics.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    train_rows = [row for row in rows if row["split"] == "train"]
    development_rows = [row for row in rows if row["split"] == "development"]
    if mutation == "full_batch":
        train_rows[0]["examples"] = 2
    elif mutation == "remainder":
        train_rows[4]["examples"] = 4
    elif mutation == "development_count":
        development_rows[0]["examples"] = 3
    elif mutation == "development_boundary":
        development_rows[0]["update"] = 4
    else:  # pragma: no cover - exhaustive parameter
        raise AssertionError(mutation)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(ExtensionAnalysisError, match=message):
        analyze_extension_campaign(
            manifest,
            roots,
            repository_root=repository,
            runtime_evidence_roots=runtime_roots,
            supplemental_evidence_roots=supplemental_roots,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("best_summary", "best_update"),
        ("compute_summary", "compute_seconds"),
        ("elapsed_summary", "elapsed_seconds predates"),
        ("earliest_strict_best", "epoch"),
    ],
)
def test_summary_is_bound_to_health_history_and_checkpoint_metadata(
    tmp_path: Path, mutation: str, message: str
) -> None:
    manifest, roots, runtime_roots, supplemental_roots, repository, _ = _fixture(tmp_path)
    run_dir = roots["E20"] / "R" / "seed_101"
    if mutation == "earliest_strict_best":
        metrics = run_dir / "training_metrics.jsonl"
        rows = [
            json.loads(line) for line in metrics.read_text(encoding="utf-8").splitlines()
        ]
        development = [row for row in rows if row["split"] == "development"]
        metric_name = "mean_foreground_dice"
        development[0][metric_name] = development[1][metric_name]
        metrics.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
    else:
        summary_path = run_dir / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if mutation == "best_summary":
            summary["best_update"] = 5
            summary["validation_best"]["update"] = 5
        elif mutation == "compute_summary":
            summary["compute_seconds"] += 1.0
        elif mutation == "elapsed_summary":
            summary["elapsed_seconds"] = summary["compute_seconds"] + 0.5
        else:  # pragma: no cover - exhaustive parameter
            raise AssertionError(mutation)
        _write_json(summary_path, summary)

    with pytest.raises(ExtensionAnalysisError, match=message):
        analyze_extension_campaign(
            manifest,
            roots,
            repository_root=repository,
            runtime_evidence_roots=runtime_roots,
            supplemental_evidence_roots=supplemental_roots,
        )


def test_supplemental_preupdate_claim_requires_absent_progress(tmp_path: Path) -> None:
    manifest, roots, runtime_roots, supplemental_roots, repository, _ = _fixture(tmp_path)
    registration = manifest["supplemental_attempt_evidence"][0]
    failure_relative = Path(
        registration["required_files"]["failure_manifest"]["relative_path"]
    )
    rejected_root = next(iter(supplemental_roots.values()))
    _write_json(rejected_root / failure_relative.parent / "summary.json", {"status": "ok"})

    with pytest.raises(ExtensionAnalysisError, match="progress artifacts"):
        analyze_extension_campaign(
            manifest,
            roots,
            repository_root=repository,
            runtime_evidence_roots=runtime_roots,
            supplemental_evidence_roots=supplemental_roots,
        )


@pytest.mark.parametrize(
    ("artifact", "message"),
    [
        ("checkpoint", "best_update"),
        ("environment", "jax.*version"),
    ],
)
def test_supplemental_failure_requires_registered_checkpoint_and_runtime(
    tmp_path: Path, artifact: str, message: str
) -> None:
    manifest, roots, runtime_roots, supplemental_roots, repository, _ = _fixture(tmp_path)
    registration = manifest["supplemental_attempt_evidence"][0]
    rejected_root = next(iter(supplemental_roots.values()))
    if artifact == "checkpoint":
        metadata = registration["required_files"]["initial_checkpoint"]
        path = rejected_root / metadata["relative_path"]
        payload = _read_pickle(path)
        payload["best_update"] = 0
        _write_pickle(path, payload)
    else:
        metadata = registration["required_files"]["environment"]
        path = rejected_root / metadata["relative_path"]
        environment = json.loads(path.read_text(encoding="utf-8"))
        environment["jax"]["version"] = "unexpected"
        _write_json(path, environment)
    metadata["sha256"] = _digest(path)

    with pytest.raises(ExtensionAnalysisError, match=message):
        analyze_extension_campaign(
            manifest,
            roots,
            repository_root=repository,
            runtime_evidence_roots=runtime_roots,
            supplemental_evidence_roots=supplemental_roots,
        )


def test_endpoint_schema_is_checked_after_gate(tmp_path: Path) -> None:
    manifest, roots, runtime_roots, supplemental_roots, repository, _ = _fixture(tmp_path)
    path = roots["E20"] / "F" / "seed_101" / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["official_test"]["mean_foreground_dice"] = 1.01
    _write_json(path, summary)

    with pytest.raises(ExtensionAnalysisError, match="outside.*0, 1"):
        analyze_extension_campaign(
            manifest,
            roots,
            repository_root=repository,
            runtime_evidence_roots=runtime_roots,
            supplemental_evidence_roots=supplemental_roots,
        )


def test_registered_split_manifest_hash_is_enforced_before_endpoints(
    tmp_path: Path,
) -> None:
    manifest, roots, runtime_roots, supplemental_roots, repository, _ = _fixture(tmp_path)
    split_path = repository / "splits" / "isic16.csv"
    split_path.write_text(
        split_path.read_text(encoding="utf-8") + "wrong,test,\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="split manifest SHA-256 mismatch"):
        analyze_extension_campaign(
            manifest,
            roots,
            repository_root=repository,
            runtime_evidence_roots=runtime_roots,
            supplemental_evidence_roots=supplemental_roots,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("summary_count", "examples"),
        ("prediction_count", "sample_id count"),
        ("sample_order", "values or order disagree"),
        ("sample_identity", "values or order disagree"),
        ("prediction_shape", "prediction shape"),
        ("missing_array", "archive schema"),
        ("nonfinite_prediction", "noninteger or nonfinite"),
        ("out_of_range", "out of range"),
    ],
)
def test_endpoint_cardinality_and_prediction_archives_are_checked_post_gate(
    tmp_path: Path, mutation: str, message: str
) -> None:
    manifest, roots, runtime_roots, supplemental_roots, repository, _ = _fixture(tmp_path)
    run_dir = roots["E20"] / "F" / "seed_101"
    if mutation == "summary_count":
        summary_path = run_dir / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["official_test"]["examples"] = 3
        _write_json(summary_path, summary)
    else:
        archive_path = run_dir / "predictions_official_test.npz"
        sample_ids = np.asarray(["sample-0", "sample-1", "sample-2", "sample-3"])
        predictions: np.ndarray = np.zeros((4, 2, 2), dtype=np.uint8)
        if mutation == "prediction_count":
            sample_ids = sample_ids[:3]
            predictions = predictions[:3]
        elif mutation == "sample_order":
            sample_ids = sample_ids[::-1]
        elif mutation == "sample_identity":
            sample_ids[0] = "wrong-sample"
        elif mutation == "prediction_shape":
            predictions = np.zeros((4, 2), dtype=np.uint8)
        elif mutation == "missing_array":
            np.savez_compressed(archive_path, sample_id=sample_ids)
            predictions = np.asarray([])
        elif mutation == "nonfinite_prediction":
            predictions = predictions.astype(np.float32)
            predictions[0, 0, 0] = np.nan
        elif mutation == "out_of_range":
            predictions[0, 0, 0] = 2
        else:  # pragma: no cover - exhaustive parameter
            raise AssertionError(mutation)
        if mutation != "missing_array":
            np.savez_compressed(archive_path, sample_id=sample_ids, prediction=predictions)

    with pytest.raises(ExtensionAnalysisError, match=message):
        analyze_extension_campaign(
            manifest,
            roots,
            repository_root=repository,
            runtime_evidence_roots=runtime_roots,
            supplemental_evidence_roots=supplemental_roots,
        )


def test_output_names_must_be_unique_safe_basenames(tmp_path: Path) -> None:
    manifest, roots, runtime_roots, supplemental_roots, repository, _ = _fixture(tmp_path)
    manifest["required_outputs"]["paired_quality"] = "../escaped.csv"

    with pytest.raises(ValueError, match="safe relative basename"):
        analyze_extension_campaign(
            manifest,
            roots,
            repository_root=repository,
            runtime_evidence_roots=runtime_roots,
            supplemental_evidence_roots=supplemental_roots,
        )


@pytest.mark.parametrize("document", ["protocol_document", "contract_document"])
def test_frozen_documents_are_checked_by_direct_api_before_summaries(
    tmp_path: Path, document: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, roots, runtime_roots, supplemental_roots, repository, _ = _fixture(tmp_path)
    path = repository / manifest[document]["path"]
    path.write_text(path.read_text(encoding="utf-8") + "mutation\n", encoding="utf-8")
    endpoint_summary = roots["E20"] / "F" / "seed_101" / "summary.json"
    endpoint_summary.write_text("summary-must-not-be-read", encoding="utf-8")
    original_read_text = Path.read_text

    def guarded_read_text(self: Path, *args: Any, **kwargs: Any) -> str:
        if self == endpoint_summary:
            raise AssertionError("endpoint-bearing summary was read before document checks")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    with pytest.raises(ValueError, match=f"{document} SHA-256 mismatch"):
        analyze_extension_campaign(
            manifest,
            roots,
            repository_root=repository,
            runtime_evidence_roots=runtime_roots,
            supplemental_evidence_roots=supplemental_roots,
        )


def test_config_document_is_semantically_bound_to_registered_cohort(
    tmp_path: Path,
) -> None:
    manifest, roots, runtime_roots, supplemental_roots, repository, _ = _fixture(tmp_path)
    registration = manifest["cohorts"]["E20"]["config_document"]
    path = repository / registration["path"]
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["task"] = "cifar100_poincare_resnet20"
    path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    registration["sha256"] = _digest(path)

    with pytest.raises(ValueError, match="task disagrees"):
        analyze_extension_campaign(
            manifest,
            roots,
            repository_root=repository,
            runtime_evidence_roots=runtime_roots,
            supplemental_evidence_roots=supplemental_roots,
        )


def test_unregistered_scientific_retry_is_rejected(tmp_path: Path) -> None:
    manifest, roots, runtime_roots, supplemental_roots, repository, _ = _fixture(tmp_path)
    path = roots["E20"] / "campaign_attempts.jsonl"
    attempts = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rejected = dict(attempts[0])
    rejected.update(
        {
            "status": "failed",
            "returncode": 1,
            "summary_path": None,
            "stderr": "nonfinite training loss after update 4",
        }
    )
    path.write_text(
        json.dumps(rejected, sort_keys=True) + "\n" + path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    with pytest.raises(ExtensionAnalysisError, match="not exactly registered"):
        analyze_extension_campaign(
            manifest,
            roots,
            repository_root=repository,
            runtime_evidence_roots=runtime_roots,
            supplemental_evidence_roots=supplemental_roots,
        )


def test_manifest_cannot_claim_unregistered_postupdate_resume(tmp_path: Path) -> None:
    manifest, roots, runtime_roots, supplemental_roots, repository, _ = _fixture(tmp_path)
    manifest["failure_policy"]["external_interruption_after_updates"][
        "currently_registered_attempts"
    ] = 1

    with pytest.raises(ExtensionAnalysisError, match="currently_registered_attempts"):
        analyze_extension_campaign(
            manifest,
            roots,
            repository_root=repository,
            runtime_evidence_roots=runtime_roots,
            supplemental_evidence_roots=supplemental_roots,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("failed", "status"),
        ("failed_attempt", "status"),
        ("mixed_config", "config_hash"),
        ("duplicate", "incomplete run matrix"),
        ("extra_seed", "incomplete run matrix"),
    ],
)
def test_invalid_run_matrices_fail_closed(
    tmp_path: Path, mutation: str, message: str
) -> None:
    manifest, roots, runtime_roots, supplemental_roots, repository, _ = _fixture(tmp_path)
    source = roots["E20"] / "R" / "seed_101" / "summary.json"
    value = json.loads(source.read_text(encoding="utf-8"))
    if mutation == "failed":
        value["status"] = "nonfinite"
        _write_json(source, value)
    elif mutation == "failed_attempt":
        attempts_path = roots["E20"] / "campaign_attempts.jsonl"
        attempts = [
            json.loads(line)
            for line in attempts_path.read_text(encoding="utf-8").splitlines()
        ]
        attempts[0]["status"] = "failed"
        attempts[0]["returncode"] = 1
        attempts_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in attempts),
            encoding="utf-8",
        )
    elif mutation == "mixed_config":
        value["config_hash"] = "different"
        _write_json(source, value)
    elif mutation == "duplicate":
        _write_json(roots["E20"] / "duplicate" / "summary.json", value)
    elif mutation == "extra_seed":
        value["seed_or_session"] = 999
        _write_json(roots["E20"] / "extra" / "summary.json", value)
    else:  # pragma: no cover - exhaustive test parameter
        raise AssertionError(mutation)

    with pytest.raises(ExtensionAnalysisError, match=message):
        analyze_extension_campaign(
            manifest,
            roots,
            repository_root=repository,
            runtime_evidence_roots=runtime_roots,
            supplemental_evidence_roots=supplemental_roots,
        )


def test_export_is_atomic_and_refuses_stale_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (
        _,
        roots,
        runtime_roots,
        supplemental_roots,
        repository,
        manifest_path,
    ) = _fixture(tmp_path)
    output = tmp_path / "export"
    monkeypatch.setattr(
        extension_analysis,
        "_validate_analysis_checkout",
        lambda *args: {
            "commit": "f" * 40,
            "dirty": False,
            "status_porcelain": [],
            "tracked_sources": list(extension_analysis.EXPORTER_SOURCE_FILES),
        },
    )

    export_extension_campaign(
        manifest_path,
        roots,
        output,
        repository_root=repository,
        runtime_evidence_roots=runtime_roots,
        supplemental_evidence_roots=supplemental_roots,
    )

    assert (output / OUTPUTS["gate_audit"]).is_file()
    assert (output / OUTPUTS["audit_summary"]).is_file()
    assert (output / f"{OUTPUTS['forest_figure_stem']}.pdf").is_file()
    assert (output / f"{OUTPUTS['forest_figure_stem']}.png").is_file()
    with (output / OUTPUTS["platform_strata"]).open(newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 4
    audit_summary = json.loads(
        (output / OUTPUTS["audit_summary"]).read_text(encoding="utf-8")
    )
    assert audit_summary["bootstrap_quantile_method"] == "linear"
    assert set(audit_summary["analysis_runtime"]["packages"]) == {
        "numpy",
        "scipy",
        "matplotlib",
    }
    assert audit_summary["exporter_git"]["commit"]
    assert len(audit_summary["exporter_source_sha256"]) == 4
    with pytest.raises(FileExistsError, match="stale"):
        export_extension_campaign(
            manifest_path,
            roots,
            output,
            repository_root=repository,
            runtime_evidence_roots=runtime_roots,
            supplemental_evidence_roots=supplemental_roots,
        )


@pytest.mark.parametrize("status", [" M tracked.py", "?? untracked.py"])
def test_analysis_checkout_rejects_dirty_or_untracked_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    root = tmp_path / "checkout"
    root.mkdir()
    manifest_path = root / "manifest.yaml"
    protocol_path = root / "protocol.yaml"
    contract_path = root / "contract.md"
    for path in (manifest_path, protocol_path, contract_path):
        path.write_text("tracked\n", encoding="utf-8")
    for relative in extension_analysis.EXPORTER_SOURCE_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("tracked\n", encoding="utf-8")
    manifest = {
        "protocol_document": {"path": "protocol.yaml"},
        "contract_document": {"path": "contract.md"},
    }
    monkeypatch.setattr(extension_analysis, "SOURCE_REPOSITORY_ROOT", root)

    def fake_git(repository: Path, *arguments: str) -> str:
        assert repository == root
        if arguments == ("rev-parse", "--show-toplevel"):
            return str(root)
        if arguments == ("status", "--porcelain", "--untracked-files=all"):
            return status
        raise AssertionError(arguments)

    monkeypatch.setattr(extension_analysis, "_git_command", fake_git)
    with pytest.raises(ExtensionAnalysisError, match="must be clean"):
        extension_analysis._validate_analysis_checkout(manifest_path, manifest, root)


def test_analysis_checkout_requires_all_sources_tracked_at_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "checkout"
    root.mkdir()
    manifest_path = root / "manifest.yaml"
    protocol_path = root / "protocol.yaml"
    contract_path = root / "contract.md"
    for path in (manifest_path, protocol_path, contract_path):
        path.write_text("tracked\n", encoding="utf-8")
    for relative in extension_analysis.EXPORTER_SOURCE_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("tracked\n", encoding="utf-8")
    manifest = {
        "protocol_document": {"path": "protocol.yaml"},
        "contract_document": {"path": "contract.md"},
    }
    monkeypatch.setattr(extension_analysis, "SOURCE_REPOSITORY_ROOT", root)

    def fake_git(repository: Path, *arguments: str) -> str:
        assert repository == root
        if arguments == ("rev-parse", "--show-toplevel"):
            return str(root)
        if arguments == ("status", "--porcelain", "--untracked-files=all"):
            return ""
        if arguments[:2] == ("cat-file", "-e"):
            if arguments[-1].endswith("training_extensions.py"):
                raise ExtensionAnalysisError("missing tracked source")
            return ""
        if arguments == ("rev-parse", "HEAD"):
            return "f" * 40
        raise AssertionError(arguments)

    monkeypatch.setattr(extension_analysis, "_git_command", fake_git)
    with pytest.raises(ExtensionAnalysisError, match="missing tracked source"):
        extension_analysis._validate_analysis_checkout(manifest_path, manifest, root)


def test_analysis_checkout_tracks_prior_amendment_documents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "checkout"
    root.mkdir()
    manifest_path = root / "manifest.yaml"
    protocol_path = root / "protocol.yaml"
    contract_path = root / "contract.md"
    prior_manifest_path = root / "prior-manifest.yaml"
    prior_contract_path = root / "prior-contract.md"
    required_paths = (
        manifest_path,
        protocol_path,
        contract_path,
        prior_manifest_path,
        prior_contract_path,
    )
    for path in required_paths:
        path.write_text("tracked\n", encoding="utf-8")
    for relative in extension_analysis.EXPORTER_SOURCE_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("tracked\n", encoding="utf-8")
    manifest = {
        "protocol_document": {"path": protocol_path.name},
        "contract_document": {"path": contract_path.name},
        "prior_amendment_manifest_document": {"path": prior_manifest_path.name},
        "prior_amendment_contract_document": {"path": prior_contract_path.name},
    }
    monkeypatch.setattr(extension_analysis, "SOURCE_REPOSITORY_ROOT", root)
    tracked_at_head: list[str] = []

    def fake_git(repository: Path, *arguments: str) -> str:
        assert repository == root
        if arguments == ("rev-parse", "--show-toplevel"):
            return str(root)
        if arguments == ("status", "--porcelain", "--untracked-files=all"):
            return ""
        if arguments[:2] == ("cat-file", "-e"):
            tracked_at_head.append(arguments[-1].removeprefix("HEAD:"))
            return ""
        if arguments == ("rev-parse", "HEAD"):
            return "f" * 40
        raise AssertionError(arguments)

    monkeypatch.setattr(extension_analysis, "_git_command", fake_git)
    checkout = extension_analysis._validate_analysis_checkout(
        manifest_path, manifest, root
    )

    for path in (prior_manifest_path, prior_contract_path):
        relative = path.relative_to(root).as_posix()
        assert relative in checkout["tracked_sources"]
        assert relative in tracked_at_head


@pytest.mark.parametrize(
    ("api_name", "internal_name"),
    [
        ("analyze_extension_campaign", "_analyze_extension_campaign_registered"),
        ("validate_global_embargo", "_validate_global_embargo_registered"),
    ],
)
def test_public_analysis_apis_reject_checkout_before_registered_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    api_name: str,
    internal_name: str,
) -> None:
    (
        _,
        roots,
        runtime_roots,
        supplemental_roots,
        repository,
        manifest_path,
    ) = _fixture(tmp_path)

    def reject_checkout(*args: Any, **kwargs: Any) -> Any:
        raise ExtensionAnalysisError("analysis checkout must be clean")

    def reject_evidence_access(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("registered evidence was accessed before checkout validation")

    monkeypatch.setattr(extension_analysis, "_validate_analysis_checkout", reject_checkout)
    monkeypatch.setattr(extension_analysis, internal_name, reject_evidence_access)
    with pytest.raises(ExtensionAnalysisError, match="must be clean"):
        getattr(extension_analysis, api_name)(
            manifest_path,
            roots,
            repository_root=repository,
            runtime_evidence_roots=runtime_roots,
            supplemental_evidence_roots=supplemental_roots,
        )


def test_frozen_repository_manifest_validates_without_artifacts() -> None:
    repository = Path(__file__).resolve().parents[1]
    manifest = yaml.safe_load(
        (repository / "configs/analysis/e20_e21_training_replication.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert manifest["global_embargo"]["required_successful_runs"] == 60
    assert manifest["cohorts"]["E20"]["variants"] == ["R", "F"]
    assert manifest["cohorts"]["E21"]["variants"] == ["R", "F"]
    assert manifest["cross_platform_reporting"]["grand_pooled_estimate"] == "prohibited"
    extension_analysis._validate_document(
        manifest["protocol_document"],
        name="protocol_document",
        repository_root=repository,
    )
    extension_analysis._validate_document(
        manifest["contract_document"],
        name="contract_document",
        repository_root=repository,
    )
    extension_analysis._validate_manifest_contract(manifest)
    specs = extension_analysis._cohort_specs(manifest, repository_root=repository)
    assert specs["E20"].training_health["total_train_rows_per_epoch"] == 102
    assert specs["E21"].training_health["total_train_rows_per_epoch"] == 352
