"""Fresh-process coordinator for the bounded retained-state diagnostics."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from poincare_exec.artifacts import append_jsonl
from poincare_exec.artifacts.provenance import git_provenance, timestamp_utc
from poincare_exec.config.loader import (
    config_hash,
    load_yaml,
    require_keys,
    validate_hashed_document,
)
from poincare_exec.experiments.benchmark import (
    _gpu_snapshot,
    _require_idle_gpu,
    _selected_physical_gpu,
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_completed_summary(
    summary: dict[str, Any],
    *,
    protocol: dict[str, Any],
    diagnostic_config: dict[str, Any],
    training_config: dict[str, Any],
    task_entry: dict[str, Any],
    source_variant: str,
    source_checkpoint_sha256: str,
    source_summary_sha256: str,
    code: dict[str, Any],
) -> None:
    design_path, design_sha256 = validate_hashed_document(
        diagnostic_config, field="design_document"
    )
    repeat_policy = str(
        diagnostic_config["controls"].get("unperturbed_repeat_policy", "bitwise")
    )
    expected = {
        "experiment_id": diagnostic_config["experiment_id"],
        "protocol_version": protocol["protocol_version"],
        "variant": "shared",
        "config_hash": config_hash(
            protocol, diagnostic_config, task_entry, training_config
        ),
        "code_commit": code["commit"],
        "code_dirty": False,
        "seed_or_session": int(task_entry["source_seed"]),
        "task": training_config["task"],
        "source_experiment_id": training_config["experiment_id"],
        "source_variant": source_variant,
        "source_seed": int(task_entry["source_seed"]),
        "source_config_hash": task_entry["source_config_hash"],
        "source_protocol_version": task_entry["source_protocol_version"],
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "source_summary_sha256": source_summary_sha256,
        "status": "ok",
        "unperturbed_repeat_policy": repeat_policy,
        "design_document_path": design_path,
        "design_document_sha256": design_sha256,
    }
    mismatches = {
        field: (summary.get(field), value)
        for field, value in expected.items()
        if summary.get(field) != value
    }
    if mismatches:
        detail = ", ".join(
            f"{field}={actual!r} (expected {wanted!r})"
            for field, (actual, wanted) in mismatches.items()
        )
        raise RuntimeError(f"completed diagnostic identity mismatch: {detail}")
    repeat_exact = summary.get("unperturbed_repeat_bitwise_exact")
    if not isinstance(repeat_exact, bool):
        raise RuntimeError("completed diagnostic lacks the repeated-R exactness result")
    if repeat_policy == "bitwise" and repeat_exact is not True:
        raise RuntimeError("completed diagnostic failed the bitwise repeated-R policy")


def run_retained_diagnostic_campaign(
    protocol_path: str | Path,
    diagnostic_config_path: str | Path,
    training_config_path: str | Path,
    *,
    data_root: str | Path,
    manifest_dir: str | Path,
    training_dir: str | Path,
    output_dir: str | Path,
) -> int:
    """Run the three frozen source variants for one task in isolated processes."""

    protocol_file = Path(protocol_path).expanduser().resolve()
    diagnostic_config_file = Path(diagnostic_config_path).expanduser().resolve()
    training_config_file = Path(training_config_path).expanduser().resolve()
    data = Path(data_root).expanduser().resolve()
    manifests = Path(manifest_dir).expanduser().resolve()
    training_root = Path(training_dir).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    protocol = load_yaml(protocol_file)
    diagnostic_config = load_yaml(diagnostic_config_file)
    training_config = load_yaml(training_config_file)
    design_path, design_sha256 = validate_hashed_document(
        diagnostic_config, field="design_document"
    )
    require_keys(
        diagnostic_config,
        ("experiment_id", "tasks"),
        name="retained diagnostic config",
    )
    require_keys(
        training_config,
        ("experiment_id", "task", "dataset", "variants", "seeds"),
        name="training config",
    )
    task_entry = next(
        (
            entry
            for entry in diagnostic_config["tasks"]
            if entry["task"] == training_config["task"]
        ),
        None,
    )
    if task_entry is None:
        raise ValueError(
            f"diagnostic config has no task entry for {training_config['task']}"
        )
    source_seed = int(task_entry["source_seed"])
    if source_seed not in training_config["seeds"]:
        raise RuntimeError("diagnostic source seed is outside the training matrix")
    source_variants = tuple(str(value) for value in task_entry["source_variants"])
    if source_variants != tuple(protocol["variants"]["core_training"]):
        raise RuntimeError("diagnostic source variants disagree with the protocol")
    if training_config["experiment_id"] != task_entry["training_experiment_id"]:
        raise RuntimeError("diagnostic task selects the wrong training experiment")

    destination.mkdir(parents=True, exist_ok=True)
    attempts_path = destination / "campaign_attempts.jsonl"
    code = git_provenance()
    if code["dirty"] or not code["commit"]:
        raise RuntimeError("retained diagnostic campaign requires a clean commit")
    diagnostic_digest = config_hash(
        protocol, diagnostic_config, task_entry, training_config
    )
    attempt_identity = {
        "experiment_id": diagnostic_config["experiment_id"],
        "protocol_version": protocol["protocol_version"],
        "config_hash": diagnostic_digest,
        "code_commit": code["commit"],
        "code_dirty": code["dirty"],
        "design_document_path": design_path,
        "design_document_sha256": design_sha256,
        "training_experiment_id": training_config["experiment_id"],
        "task": training_config["task"],
    }
    attempts = 0
    for source_variant in source_variants:
        source_run = training_root / source_variant / f"seed_{source_seed}"
        checkpoint = source_run / "checkpoints" / "final.pkl"
        source_summary = source_run / "summary.json"
        selected = destination / source_variant / f"seed_{source_seed}"
        completed_summary = selected / "summary.json"
        if not checkpoint.is_file() or not source_summary.is_file():
            append_jsonl(
                attempts_path,
                [
                    {
                        **attempt_identity,
                        "source_variant": source_variant,
                        "source_seed": source_seed,
                        "status": "not_applicable",
                        "reason": "accepted_final_training_state_missing",
                        "source_checkpoint": str(checkpoint),
                        "source_summary": str(source_summary),
                        "source_checkpoint_sha256": (
                            _file_sha256(checkpoint) if checkpoint.is_file() else None
                        ),
                        "source_summary_sha256": (
                            _file_sha256(source_summary)
                            if source_summary.is_file()
                            else None
                        ),
                        "timestamp_utc": timestamp_utc(),
                    }
                ],
            )
            attempts += 1
            raise RuntimeError(
                f"required final training state is missing for "
                f"{source_variant}/seed {source_seed}"
            )
        source_checkpoint_sha256 = _file_sha256(checkpoint)
        source_summary_sha256 = _file_sha256(source_summary)
        if completed_summary.exists():
            with completed_summary.open("r", encoding="utf-8") as stream:
                _validate_completed_summary(
                    json.load(stream),
                    protocol=protocol,
                    diagnostic_config=diagnostic_config,
                    training_config=training_config,
                    task_entry=task_entry,
                    source_variant=source_variant,
                    source_checkpoint_sha256=source_checkpoint_sha256,
                    source_summary_sha256=source_summary_sha256,
                    code=code,
                )
            continue
        if selected.exists():
            raise FileExistsError(
                f"incomplete diagnostic output requires audit before rerun: {selected}"
            )
        snapshot_before = _require_idle_gpu(protocol["timing"])
        selected.parent.mkdir(parents=True, exist_ok=True)
        failure_output = None
        validation_error = None
        attempt_timestamp = timestamp_utc()
        with tempfile.TemporaryDirectory(
            prefix=f".{selected.name}.", dir=selected.parent
        ) as directory:
            temporary_output = Path(directory) / "result"
            process = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "poincare_exec.cli",
                    "retained-diagnostic",
                    "--protocol",
                    str(protocol_file),
                    "--config",
                    str(diagnostic_config_file),
                    "--training-config",
                    str(training_config_file),
                    "--checkpoint",
                    str(checkpoint),
                    "--source-summary",
                    str(source_summary),
                    "--data-root",
                    str(data),
                    "--manifest-dir",
                    str(manifests),
                    "--output-dir",
                    str(temporary_output),
                ],
                cwd=directory,
                check=False,
                capture_output=True,
                text=True,
            )
            summary_file = temporary_output / "summary.json"
            succeeded = process.returncode == 0 and summary_file.exists()
            if succeeded:
                try:
                    with summary_file.open("r", encoding="utf-8") as stream:
                        summary = json.load(stream)
                    _validate_completed_summary(
                        summary,
                        protocol=protocol,
                        diagnostic_config=diagnostic_config,
                        training_config=training_config,
                        task_entry=task_entry,
                        source_variant=source_variant,
                        source_checkpoint_sha256=source_checkpoint_sha256,
                        source_summary_sha256=source_summary_sha256,
                        code=code,
                    )
                except (KeyError, TypeError, ValueError, RuntimeError) as error:
                    succeeded = False
                    validation_error = f"{type(error).__name__}: {error}"
            if succeeded:
                temporary_output.replace(selected)
            elif temporary_output.exists():
                safe_timestamp = attempt_timestamp.replace(":", "-")
                failure_output = (
                    destination
                    / "failed"
                    / source_variant
                    / f"seed_{source_seed}"
                    / safe_timestamp
                )
                failure_output.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(temporary_output), failure_output)
        snapshot_after = _gpu_snapshot(_selected_physical_gpu())
        append_jsonl(
            attempts_path,
            [
                {
                    **attempt_identity,
                    "source_variant": source_variant,
                    "source_seed": source_seed,
                    "status": "ok" if succeeded else "failed",
                    "returncode": process.returncode,
                    "source_checkpoint": str(checkpoint),
                    "source_summary": str(source_summary),
                    "source_checkpoint_sha256": source_checkpoint_sha256,
                    "source_summary_sha256": source_summary_sha256,
                    "output_dir": (
                        str(selected)
                        if succeeded
                        else str(failure_output)
                        if failure_output
                        else None
                    ),
                    "validation_error": validation_error,
                    "gpu_snapshot_before": snapshot_before,
                    "gpu_snapshot_after": snapshot_after,
                    "stdout": process.stdout,
                    "stderr": process.stderr,
                    "timestamp_utc": attempt_timestamp,
                }
            ],
        )
        attempts += 1
        if not succeeded:
            raise RuntimeError(
                f"retained diagnostic failed for {source_variant}/seed {source_seed}"
            )
    return attempts
