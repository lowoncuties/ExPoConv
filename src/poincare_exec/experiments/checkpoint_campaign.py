"""Coordinator for E06 evaluation of every retained final checkpoint."""

from __future__ import annotations

import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from poincare_exec.artifacts import append_jsonl
from poincare_exec.artifacts.provenance import git_provenance, timestamp_utc
from poincare_exec.config.loader import config_hash, load_yaml, require_keys
from poincare_exec.experiments.benchmark import (
    _gpu_snapshot,
    _require_idle_gpu,
    _selected_physical_gpu,
)


def _validate_summary(
    summary: dict[str, Any],
    *,
    protocol: dict[str, Any],
    checkpoint_config: dict[str, Any],
    training_config: dict[str, Any],
    task_entry: dict[str, Any],
    source_variant: str,
    seed: int,
    code: dict[str, Any],
) -> None:
    expected = {
        "experiment_id": "E06",
        "protocol_version": protocol["protocol_version"],
        "variant": "shared",
        "config_hash": config_hash(
            protocol, checkpoint_config, task_entry, training_config
        ),
        "code_commit": code["commit"],
        "code_dirty": code["dirty"],
        "seed_or_session": seed,
        "task": training_config["task"],
        "source_checkpoint_experiment": training_config["experiment_id"],
        "source_checkpoint_variant": source_variant,
        "source_checkpoint_seed": seed,
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
        raise RuntimeError(f"completed E06 evaluation identity mismatch: {detail}")


def run_trained_checkpoint_campaign(
    protocol_path: str | Path,
    checkpoint_config_path: str | Path,
    training_config_path: str | Path,
    *,
    data_root: str | Path,
    manifest_dir: str | Path,
    training_dir: str | Path,
    output_dir: str | Path,
) -> int:
    """Evaluate every frozen final R/T checkpoint in an isolated process."""

    protocol_file = Path(protocol_path).expanduser().resolve()
    checkpoint_config_file = Path(checkpoint_config_path).expanduser().resolve()
    training_config_file = Path(training_config_path).expanduser().resolve()
    data = Path(data_root).expanduser().resolve()
    manifests = Path(manifest_dir).expanduser().resolve()
    training_root = Path(training_dir).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    protocol = load_yaml(protocol_file)
    checkpoint_config = load_yaml(checkpoint_config_file)
    training_config = load_yaml(training_config_file)
    require_keys(
        training_config,
        ("experiment_id", "task", "dataset", "variants", "seeds"),
        name="training config",
    )
    require_keys(
        checkpoint_config,
        ("update_variants", "tasks"),
        name="checkpoint config",
    )
    if list(training_config["variants"]) != list(protocol["variants"]["core_training"]):
        raise ValueError("checkpoint campaign variants disagree with the protocol")
    if list(training_config["seeds"]) != list(protocol["seeds"]["training"]):
        raise ValueError("checkpoint campaign seeds disagree with the protocol")
    source_variants = [str(value) for value in checkpoint_config["update_variants"]]
    if not source_variants or any(
        value not in training_config["variants"] for value in source_variants
    ):
        raise ValueError(
            "checkpoint source variants must be a nonempty subset of training variants"
        )
    task_entry = next(
        entry
        for entry in checkpoint_config["tasks"]
        if entry["task"] == training_config["task"]
    )

    destination.mkdir(parents=True, exist_ok=True)
    attempts_path = destination / "campaign_attempts.jsonl"
    code = git_provenance()
    randomizer = random.Random(int(protocol["seeds"]["measurement_order"]))
    attempts = 0
    for seed_value in training_config["seeds"]:
        seed = int(seed_value)
        order = list(source_variants)
        randomizer.shuffle(order)
        for source_variant in order:
            source = (
                training_root
                / source_variant
                / f"seed_{seed}"
                / "checkpoints"
                / "final.pkl"
            )
            selected = destination / source_variant / f"seed_{seed}"
            summary_path = selected / "summary.json"
            if summary_path.exists():
                _validate_summary(
                    json.loads(summary_path.read_text(encoding="utf-8")),
                    protocol=protocol,
                    checkpoint_config=checkpoint_config,
                    training_config=training_config,
                    task_entry=task_entry,
                    source_variant=source_variant,
                    seed=seed,
                    code=code,
                )
                continue
            if selected.exists():
                raise FileExistsError(
                    f"incomplete E06 output requires audit before rerun: {selected}"
                )
            if not source.exists():
                append_jsonl(
                    attempts_path,
                    [
                        {
                            "experiment_id": "E06",
                            "protocol_version": protocol["protocol_version"],
                            "training_experiment_id": training_config["experiment_id"],
                            "task": training_config["task"],
                            "source_variant": source_variant,
                            "seed": seed,
                            "status": "not_applicable",
                            "reason": "final_training_checkpoint_missing",
                            "source_checkpoint": str(source),
                            "timestamp_utc": timestamp_utc(),
                        }
                    ],
                )
                attempts += 1
                continue

            snapshot_before = _require_idle_gpu(protocol["timing"])
            selected.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(
                prefix=f".{selected.name}.", dir=selected.parent
            ) as directory:
                temporary_output = Path(directory) / "result"
                process = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "poincare_exec.cli",
                        "checkpoint-evaluation",
                        "--protocol",
                        str(protocol_file),
                        "--config",
                        str(checkpoint_config_file),
                        "--training-config",
                        str(training_config_file),
                        "--checkpoint",
                        str(source),
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
                completed_summary = temporary_output / "summary.json"
                succeeded = process.returncode == 0 and completed_summary.exists()
                if succeeded:
                    temporary_output.replace(selected)
            snapshot_after = _gpu_snapshot(_selected_physical_gpu())
            append_jsonl(
                attempts_path,
                [
                    {
                        "experiment_id": "E06",
                        "protocol_version": protocol["protocol_version"],
                        "training_experiment_id": training_config["experiment_id"],
                        "task": training_config["task"],
                        "source_variant": source_variant,
                        "seed": seed,
                        "status": "ok" if succeeded else "failed",
                        "returncode": process.returncode,
                        "source_checkpoint": str(source),
                        "output_dir": str(selected) if succeeded else None,
                        "gpu_snapshot_before": snapshot_before,
                        "gpu_snapshot_after": snapshot_after,
                        "stdout": process.stdout,
                        "stderr": process.stderr,
                        "timestamp_utc": timestamp_utc(),
                    }
                ],
            )
            attempts += 1
    return attempts
