"""Fresh-process coordinator for the fixed E09 and E10 campaigns."""

from __future__ import annotations

import json
import random
import subprocess
import sys
import tempfile
from collections.abc import Collection
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


def _validate_completed_run(
    summary: dict[str, Any],
    *,
    protocol: dict[str, Any],
    config: dict[str, Any],
    variant: str,
    seed: int,
    code: dict[str, Any],
    accepted_code_commits: Collection[str],
) -> None:
    expected = {
        "experiment_id": config["experiment_id"],
        "protocol_version": protocol["protocol_version"],
        "variant": variant,
        "config_hash": config_hash(protocol, config),
        "seed_or_session": seed,
        "task": config["task"],
    }
    mismatches = {
        field: (summary.get(field), value)
        for field, value in expected.items()
        if summary.get(field) != value
    }
    result_commit = summary.get("code_commit")
    allowed_commits = {code["commit"], *accepted_code_commits}
    if result_commit not in allowed_commits:
        mismatches["code_commit"] = (result_commit, sorted(allowed_commits))
    expected_dirty = code["dirty"] if result_commit == code["commit"] else False
    if summary.get("code_dirty") != expected_dirty:
        mismatches["code_dirty"] = (summary.get("code_dirty"), expected_dirty)
    if mismatches:
        detail = ", ".join(
            f"{field}={actual!r} (expected {wanted!r})"
            for field, (actual, wanted) in mismatches.items()
        )
        raise RuntimeError(f"completed training-run identity mismatch: {detail}")
    if summary.get("status") not in {"ok", "nonfinite"}:
        raise RuntimeError("completed training run has no terminal status")


def run_training_campaign(
    protocol_path: str | Path,
    config_path: str | Path,
    *,
    data_root: str | Path,
    manifest_dir: str | Path,
    output_dir: str | Path,
    accepted_completed_commits: Collection[str] = (),
) -> int:
    """Run every frozen variant-seed pair and retain each process outcome."""

    protocol_file = Path(protocol_path).expanduser().resolve()
    config_file = Path(config_path).expanduser().resolve()
    data = Path(data_root).expanduser().resolve()
    manifests = Path(manifest_dir).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    protocol = load_yaml(protocol_file)
    config = load_yaml(config_file)
    accepted_commits = frozenset(str(value) for value in accepted_completed_commits)
    invalid_commits = sorted(
        value
        for value in accepted_commits
        if len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    )
    if invalid_commits:
        raise ValueError(
            "accepted completed commits must be full lowercase Git object names: "
            + ", ".join(invalid_commits)
        )
    require_keys(
        config,
        ("experiment_id", "task", "dataset", "variants", "seeds"),
        name="training campaign config",
    )
    if list(config["variants"]) != list(protocol["variants"]["core_training"]):
        raise ValueError("campaign variants disagree with the frozen protocol")
    if list(config["seeds"]) != list(protocol["seeds"]["training"]):
        raise ValueError("campaign seeds disagree with the frozen protocol")

    destination.mkdir(parents=True, exist_ok=True)
    attempts_path = destination / "campaign_attempts.jsonl"
    code = git_provenance()
    randomizer = random.Random(int(protocol["seeds"]["measurement_order"]))
    attempts = 0
    for seed_value in config["seeds"]:
        seed = int(seed_value)
        order = list(config["variants"])
        randomizer.shuffle(order)
        for variant in order:
            run_dir = destination / variant / f"seed_{seed}"
            summary_path = run_dir / "summary.json"
            if summary_path.exists():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                _validate_completed_run(
                    summary,
                    protocol=protocol,
                    config=config,
                    variant=variant,
                    seed=seed,
                    code=code,
                    accepted_code_commits=accepted_commits,
                )
                continue

            snapshot_before = _require_idle_gpu(protocol["timing"])
            run_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="poincare-training-") as directory:
                process = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "poincare_exec.cli",
                        "training-worker",
                        "--protocol",
                        str(protocol_file),
                        "--config",
                        str(config_file),
                        "--variant",
                        variant,
                        "--seed",
                        str(seed),
                        "--data-root",
                        str(data),
                        "--manifest-dir",
                        str(manifests),
                        "--output-dir",
                        str(run_dir),
                    ],
                    cwd=directory,
                    check=False,
                    capture_output=True,
                    text=True,
                )
            snapshot_after = _gpu_snapshot(_selected_physical_gpu())
            summary = (
                json.loads(summary_path.read_text(encoding="utf-8"))
                if summary_path.exists()
                else None
            )
            status = summary["status"] if summary is not None else "failed"
            append_jsonl(
                attempts_path,
                [
                    {
                        "experiment_id": config["experiment_id"],
                        "protocol_version": protocol["protocol_version"],
                        "config_hash": config_hash(protocol, config),
                        "code_commit": code["commit"],
                        "code_dirty": code["dirty"],
                        "task": config["task"],
                        "variant": variant,
                        "seed": seed,
                        "status": status,
                        "returncode": process.returncode,
                        "summary_path": (
                            str(summary_path) if summary is not None else None
                        ),
                        "gpu_snapshot_before": snapshot_before,
                        "gpu_snapshot_after": snapshot_after,
                        "stdout": process.stdout,
                        "stderr": process.stderr,
                        "accepted_completed_commits": sorted(accepted_commits),
                        "timestamp_utc": timestamp_utc(),
                    }
                ],
            )
            attempts += 1
    return attempts
