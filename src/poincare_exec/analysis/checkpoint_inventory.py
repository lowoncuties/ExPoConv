"""Deterministic post-gate inventory of registered training checkpoints.

The exporter deliberately has no code path for reading endpoint summaries or
predictions.  An accepted joint-gate audit must be validated before any cohort
root is resolved or traversed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from poincare_exec.config.constants import REPOSITORY_ROOT
from poincare_exec.config.loader import (
    config_hash,
    load_yaml,
    validate_hashed_document,
)
from poincare_exec.training.checkpoints import load_checkpoint

INVENTORY_FIELDS = (
    "experiment_id",
    "task",
    "variant",
    "seed",
    "role",
    "path",
    "bytes",
    "sha256",
    "code_commit",
    "config_hash",
    "protocol_version",
)
EXPERIMENT_IDS = ("E09", "E10", "E20", "E21")
EXTENSION_IDS = ("E20", "E21")
HISTORICAL_IDS = frozenset({"E09", "E10"})
HISTORICAL_ROLES = frozenset(
    {"initial", "10_percent", "50_percent", "validation_best", "final"}
)
# The checked-in E09/E10 inventory is role-lexicographic.  Inserting ``resume``
# in that same order preserves its five-role projection byte for byte.
ROLE_PATHS = {
    "10_percent": Path("checkpoints/10_percent.pkl"),
    "50_percent": Path("checkpoints/50_percent.pkl"),
    "final": Path("checkpoints/final.pkl"),
    "initial": Path("checkpoints/initial.pkl"),
    "resume": Path("resume.pkl"),
    "validation_best": Path("checkpoints/validation_best.pkl"),
}
EXPECTED_RUN_COUNTS = {"E09": 15, "E10": 15, "E20": 30, "E21": 30}


class CheckpointInventoryError(ValueError):
    """Raised when the gate, registration, or checkpoint matrix is invalid."""


@dataclass(frozen=True)
class CohortRegistration:
    """Inventory-relevant subset of one frozen cohort registration."""

    experiment_id: str
    task: str
    protocol_version: str
    config_hash: str
    variants: tuple[str, ...]
    seeds: tuple[int, ...]
    accepted_commits: Mapping[tuple[str, int], str]
    expected_epochs: int
    expected_updates: int

    @property
    def run_count(self) -> int:
        return len(self.variants) * len(self.seeds)


def _require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CheckpointInventoryError(f"{name} must be a mapping")
    return value


def _require_sequence(value: Any, *, name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise CheckpointInventoryError(f"{name} must be a sequence")
    return value


def _valid_digest(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _load_manifest(manifest_path: str | Path, repository_root: Path) -> dict[str, Any]:
    manifest_file = Path(manifest_path).expanduser().resolve()
    registration = load_yaml(manifest_file)
    if registration.get("base_manifest_document") is None:
        return registration

    try:
        relative, _ = validate_hashed_document(
            registration,
            field="base_manifest_document",
            repository_root=repository_root,
        )
    except ValueError as error:
        raise CheckpointInventoryError(str(error)) from error
    manifest = load_yaml(repository_root / relative)
    if manifest.get("analysis_id") != registration.get("analysis_id"):
        raise CheckpointInventoryError(
            "analysis amendment and base manifest analysis_id disagree"
        )
    if manifest.get("analysis_id") != registration.get("base_analysis_id"):
        raise CheckpointInventoryError(
            "analysis amendment base_analysis_id does not identify the base manifest"
        )
    return manifest


def _accepted_commits(
    experiment_id: str,
    raw: Any,
    expected_keys: frozenset[tuple[str, int]],
) -> dict[tuple[str, int], str]:
    commits: dict[tuple[str, int], str] = {}
    assignments = _require_sequence(
        raw, name=f"cohorts.{experiment_id}.accepted_code_assignments"
    )
    for index, raw_assignment in enumerate(assignments):
        assignment = _require_mapping(
            raw_assignment,
            name=f"cohorts.{experiment_id}.accepted_code_assignments[{index}]",
        )
        commit = str(assignment.get("commit", ""))
        if len(commit) != 40 or any(
            character not in "0123456789abcdef" for character in commit
        ):
            raise CheckpointInventoryError(
                f"{experiment_id} accepted commit must be a full lowercase Git object name"
            )
        variants = tuple(
            str(value)
            for value in _require_sequence(
                assignment.get("variants"), name=f"{experiment_id} assignment variants"
            )
        )
        seeds = tuple(
            int(value)
            for value in _require_sequence(
                assignment.get("seeds"), name=f"{experiment_id} assignment seeds"
            )
        )
        for key in ((variant, seed) for variant in variants for seed in seeds):
            if key not in expected_keys:
                raise CheckpointInventoryError(
                    f"{experiment_id} has an unregistered accepted-code assignment {key}"
                )
            if key in commits:
                raise CheckpointInventoryError(
                    f"{experiment_id} has duplicate accepted-code assignments for {key}"
                )
            commits[key] = commit
    if set(commits) != expected_keys:
        missing = sorted(expected_keys - set(commits))
        raise CheckpointInventoryError(
            f"{experiment_id} is missing accepted-code assignments for {missing}"
        )
    return commits


def _cohort_registrations(
    manifest: Mapping[str, Any], repository_root: Path
) -> dict[str, CohortRegistration]:
    try:
        protocol_relative, _ = validate_hashed_document(
            manifest, field="protocol_document", repository_root=repository_root
        )
    except ValueError as error:
        raise CheckpointInventoryError(str(error)) from error
    protocol = load_yaml(repository_root / protocol_relative)
    tasks = _require_mapping(protocol.get("tasks"), name="protocol.tasks")
    raw_cohorts = _require_mapping(manifest.get("cohorts"), name="cohorts")
    if set(raw_cohorts) != set(EXPERIMENT_IDS):
        raise CheckpointInventoryError(
            f"inventory requires exactly the registered cohorts {list(EXPERIMENT_IDS)}"
        )

    registrations: dict[str, CohortRegistration] = {}
    for experiment_id in EXPERIMENT_IDS:
        cohort = _require_mapping(
            raw_cohorts[experiment_id], name=f"cohorts.{experiment_id}"
        )
        variants = tuple(
            str(value)
            for value in _require_sequence(
                cohort.get("variants"), name=f"cohorts.{experiment_id}.variants"
            )
        )
        seeds = tuple(
            int(value)
            for value in _require_sequence(
                cohort.get("seeds"), name=f"cohorts.{experiment_id}.seeds"
            )
        )
        expected_variants = (
            {"R", "F", "T"} if experiment_id in HISTORICAL_IDS else {"R", "F"}
        )
        if set(variants) != expected_variants or len(variants) != len(expected_variants):
            raise CheckpointInventoryError(
                f"{experiment_id} variants do not match the frozen inventory scope"
            )
        if not seeds or len(seeds) != len(set(seeds)):
            raise CheckpointInventoryError(
                f"{experiment_id} seeds must be unique and nonempty"
            )
        expected_keys = frozenset((variant, seed) for variant in variants for seed in seeds)
        if len(expected_keys) != EXPECTED_RUN_COUNTS[experiment_id]:
            raise CheckpointInventoryError(
                f"{experiment_id} must register {EXPECTED_RUN_COUNTS[experiment_id]} runs"
            )

        task_name = str(cohort.get("task", ""))
        task = _require_mapping(tasks.get(task_name), name=f"protocol.tasks.{task_name}")
        expected_epochs = int(task.get("epochs", -1))
        expected_updates = int(task.get("max_updates", -1))
        health = _require_mapping(
            cohort.get("training_health"),
            name=f"cohorts.{experiment_id}.training_health",
        )
        updates_per_epoch = int(health.get("total_train_rows_per_epoch", -1))
        if (
            min(expected_epochs, expected_updates, updates_per_epoch) < 1
            or expected_epochs * updates_per_epoch != expected_updates
        ):
            raise CheckpointInventoryError(
                f"{experiment_id} has an inconsistent registered update budget"
            )

        try:
            config_relative, _ = validate_hashed_document(
                cohort, field="config_document", repository_root=repository_root
            )
        except ValueError as error:
            raise CheckpointInventoryError(str(error)) from error
        config = load_yaml(repository_root / config_relative)
        for field, expected in (
            ("experiment_id", experiment_id),
            ("task", task_name),
            ("variants", list(variants)),
            ("seeds", list(seeds)),
        ):
            if config.get(field) != expected:
                raise CheckpointInventoryError(
                    f"{experiment_id} config {field} disagrees with its registration"
                )

        digest = str(cohort.get("run_config_hash", ""))
        if not _valid_digest(digest):
            raise CheckpointInventoryError(
                f"{experiment_id} run_config_hash must be a lowercase SHA-256 digest"
            )
        protocol_version = str(cohort.get("protocol_version", ""))
        if not protocol_version:
            raise CheckpointInventoryError(f"{experiment_id} protocol_version is empty")
        if experiment_id in EXTENSION_IDS:
            if protocol_version != str(protocol.get("protocol_version", "")):
                raise CheckpointInventoryError(
                    f"{experiment_id} protocol version disagrees with protocol.yaml"
                )
            if config_hash(protocol, config) != digest:
                raise CheckpointInventoryError(
                    f"{experiment_id} run_config_hash disagrees with protocol and config"
                )

        registrations[experiment_id] = CohortRegistration(
            experiment_id=experiment_id,
            task=task_name,
            protocol_version=protocol_version,
            config_hash=digest,
            variants=variants,
            seeds=seeds,
            accepted_commits=_accepted_commits(
                experiment_id,
                cohort.get("accepted_code_assignments"),
                expected_keys,
            ),
            expected_epochs=expected_epochs,
            expected_updates=expected_updates,
        )
    if sum(cohort.run_count for cohort in registrations.values()) != 90:
        raise CheckpointInventoryError(
            "inventory registration must contain exactly 90 runs"
        )
    return registrations


def _validate_gate_audit(
    manifest: Mapping[str, Any],
    registrations: Mapping[str, CohortRegistration],
    analysis_output_dir: str | Path,
) -> None:
    required_outputs = _require_mapping(
        manifest.get("required_outputs"), name="required_outputs"
    )
    gate_name = Path(str(required_outputs.get("gate_audit", "")))
    if not gate_name.parts or gate_name.is_absolute() or ".." in gate_name.parts:
        raise CheckpointInventoryError("required_outputs.gate_audit must be relative")
    analysis_root = Path(analysis_output_dir).expanduser().resolve()
    gate_path = analysis_root / gate_name
    try:
        audit = _require_mapping(
            json.loads(gate_path.read_text(encoding="utf-8")), name=str(gate_path)
        )
    except (OSError, json.JSONDecodeError) as error:
        raise CheckpointInventoryError(
            f"accepted analysis gate audit is unavailable: {gate_path}"
        ) from error

    embargo = _require_mapping(manifest.get("global_embargo"), name="global_embargo")
    expected_runs = sum(registrations[value].run_count for value in EXTENSION_IDS)
    expected_pairs = sum(len(registrations[value].seeds) for value in EXTENSION_IDS)
    required_runs = int(embargo.get("required_successful_runs", -1))
    required_pairs = int(embargo.get("required_complete_pairs", -1))
    for field, expected in (
        ("analysis_id", manifest.get("analysis_id")),
        ("status", "open"),
        ("opened_only_after_joint_validation", True),
        ("required_successful_runs", required_runs),
        ("required_complete_pairs", required_pairs),
    ):
        if audit.get(field) != expected:
            raise CheckpointInventoryError(
                f"accepted analysis gate audit has invalid {field}"
            )
    if required_runs != expected_runs or required_pairs != expected_pairs:
        raise CheckpointInventoryError(
            "manifest joint-gate counts disagree with the registered extension matrix"
        )

    raw_cohorts = _require_sequence(audit.get("cohorts"), name="gate audit cohorts")
    cohort_audits: dict[str, Mapping[str, Any]] = {}
    for value in raw_cohorts:
        entry = _require_mapping(value, name="gate audit cohort")
        experiment_id = str(entry.get("experiment_id", ""))
        if experiment_id in cohort_audits:
            raise CheckpointInventoryError(
                f"accepted analysis gate audit duplicates {experiment_id}"
            )
        cohort_audits[experiment_id] = entry
    if set(cohort_audits) != set(EXTENSION_IDS):
        raise CheckpointInventoryError(
            "accepted analysis gate audit must contain exactly E20 and E21"
        )
    for experiment_id in EXTENSION_IDS:
        entry = cohort_audits[experiment_id]
        cohort = registrations[experiment_id]
        for field, expected in (
            ("status", "complete"),
            ("expected_runs", cohort.run_count),
            ("validated_runs", cohort.run_count),
            ("expected_pairs", len(cohort.seeds)),
            ("validated_pairs", len(cohort.seeds)),
            ("official_endpoint_embargo", "opened_after_global_validation"),
        ):
            if entry.get(field) != expected:
                raise CheckpointInventoryError(
                    f"accepted analysis gate audit has invalid {experiment_id}.{field}"
                )


def parse_cohort_roots(values: Sequence[str]) -> dict[str, Path]:
    """Parse repeatable ``EXPERIMENT_ID=PATH`` CLI values without touching paths."""

    roots: dict[str, Path] = {}
    for value in values:
        experiment_id, separator, raw_path = value.partition("=")
        if not separator or not experiment_id or not raw_path:
            raise CheckpointInventoryError(
                "cohort roots must use EXPERIMENT_ID=PATH syntax"
            )
        if experiment_id in roots:
            raise CheckpointInventoryError(f"duplicate cohort root: {experiment_id}")
        roots[experiment_id] = Path(raw_path)
    return roots


def _resolve_roots(
    roots: Mapping[str, str | Path],
) -> dict[str, Path]:
    if set(roots) != set(EXPERIMENT_IDS):
        missing = sorted(set(EXPERIMENT_IDS) - set(roots))
        extra = sorted(set(roots) - set(EXPERIMENT_IDS))
        raise CheckpointInventoryError(
            f"cohort roots must match the registration; missing={missing}, extra={extra}"
        )
    resolved: dict[str, Path] = {}
    for experiment_id in EXPERIMENT_IDS:
        root = Path(roots[experiment_id]).expanduser().resolve()
        if not root.is_dir():
            raise CheckpointInventoryError(
                f"{experiment_id} checkpoint root is missing: {root}"
            )
        resolved[experiment_id] = root
    return resolved


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stat_signature(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _checkpoint_files(run_dir: Path) -> dict[str, Path]:
    checkpoint_dir = run_dir / "checkpoints"
    if not checkpoint_dir.is_dir():
        raise CheckpointInventoryError(f"missing checkpoint directory: {checkpoint_dir}")
    expected_checkpoint_names = {
        relative.name
        for relative in ROLE_PATHS.values()
        if relative.parent.name == "checkpoints"
    }
    actual_checkpoint_names = {
        path.name for path in checkpoint_dir.iterdir() if path.suffix == ".pkl"
    }
    missing = sorted(expected_checkpoint_names - actual_checkpoint_names)
    extra = sorted(actual_checkpoint_names - expected_checkpoint_names)
    if missing or extra:
        raise CheckpointInventoryError(
            f"{run_dir}: checkpoint roles disagree with registration; "
            f"missing={missing}, extra={extra}"
        )
    root_pickle_names = {path.name for path in run_dir.iterdir() if path.suffix == ".pkl"}
    if root_pickle_names != {"resume.pkl"}:
        missing_root = sorted({"resume.pkl"} - root_pickle_names)
        extra_root = sorted(root_pickle_names - {"resume.pkl"})
        raise CheckpointInventoryError(
            f"{run_dir}: root checkpoint roles disagree with registration; "
            f"missing={missing_root}, extra={extra_root}"
        )
    return {role: run_dir / relative for role, relative in ROLE_PATHS.items()}


def _optimizer_step(payload: Mapping[str, Any], *, context: str) -> int:
    state = payload.get("optimizer_state")
    step = state.get("step") if isinstance(state, Mapping) else getattr(state, "step", None)
    if step is None:
        raise CheckpointInventoryError(f"{context}: optimizer state has no step")
    array = np.asarray(step)
    if array.size != 1:
        raise CheckpointInventoryError(f"{context}: optimizer step is not scalar")
    return int(array.reshape(()))


def _expected_milestone(
    cohort: CohortRegistration, role: str, payload: Mapping[str, Any]
) -> tuple[int, int]:
    updates_per_epoch = cohort.expected_updates // cohort.expected_epochs
    if role == "initial":
        return 0, 0
    if role == "10_percent":
        update = math.ceil(cohort.expected_updates * 0.10)
        return update, math.ceil(update / updates_per_epoch)
    if role == "50_percent":
        update = math.ceil(cohort.expected_updates * 0.50)
        return update, math.ceil(update / updates_per_epoch)
    if role in {"final", "resume"}:
        return cohort.expected_updates, cohort.expected_epochs
    if role != "validation_best":
        raise CheckpointInventoryError(f"unsupported checkpoint role: {role}")
    update = int(payload.get("update", -1))
    if not 1 <= update <= cohort.expected_updates or update % updates_per_epoch:
        raise CheckpointInventoryError(
            f"{cohort.experiment_id} validation_best is not at an epoch boundary"
        )
    return update, update // updates_per_epoch


def _validate_payload(
    payload: Mapping[str, Any],
    cohort: CohortRegistration,
    *,
    variant: str,
    seed: int,
    role: str,
    context: str,
) -> None:
    expected_commit = cohort.accepted_commits[(variant, seed)]
    for field, expected in (
        ("format", "poincare-exec-checkpoint-v1"),
        ("config_hash", cohort.config_hash),
        ("protocol_version", cohort.protocol_version),
        ("code_commit", expected_commit),
        ("task", cohort.task),
        ("variant", variant),
        ("seed", seed),
    ):
        if payload.get(field) != expected:
            raise CheckpointInventoryError(
                f"{context}: checkpoint identity field {field} disagrees with registration"
            )
    if payload.get("code_dirty") is not False:
        raise CheckpointInventoryError(f"{context}: checkpoint code_dirty must be false")

    expected_update, expected_epoch = _expected_milestone(cohort, role, payload)
    try:
        update = int(payload.get("update", -1))
        epoch = int(payload.get("epoch", -1))
    except (TypeError, ValueError, OverflowError) as error:
        raise CheckpointInventoryError(
            f"{context}: invalid checkpoint milestone"
        ) from error
    if (update, epoch) != (expected_update, expected_epoch):
        raise CheckpointInventoryError(
            f"{context}: checkpoint milestone {(update, epoch)} != "
            f"{(expected_update, expected_epoch)}"
        )
    if _optimizer_step(payload, context=context) != expected_update:
        raise CheckpointInventoryError(
            f"{context}: optimizer step disagrees with checkpoint update"
        )

    try:
        best_update = int(payload.get("best_update", -2))
        best_metric = float(payload.get("best_metric", math.nan))
    except (TypeError, ValueError, OverflowError) as error:
        raise CheckpointInventoryError(
            f"{context}: invalid best-checkpoint metadata"
        ) from error
    updates_per_epoch = cohort.expected_updates // cohort.expected_epochs
    if role == "initial":
        valid_best = best_update == -1 and best_metric == -math.inf
    elif role == "validation_best":
        valid_best = best_update == expected_update and math.isfinite(best_metric)
    elif best_update == -1:
        valid_best = best_metric == -math.inf
    else:
        valid_best = (
            1 <= best_update <= expected_update
            and best_update % updates_per_epoch == 0
            and math.isfinite(best_metric)
        )
    if not valid_best or (math.isfinite(best_metric) and not 0.0 <= best_metric <= 1.0):
        raise CheckpointInventoryError(f"{context}: invalid best-checkpoint metadata")


def _read_checkpoint_row(
    path: Path,
    cohort: CohortRegistration,
    *,
    variant: str,
    seed: int,
    role: str,
) -> dict[str, Any]:
    context = str(path)
    if path.is_symlink() or not path.is_file():
        raise CheckpointInventoryError(f"{context}: checkpoint must be a regular file")
    before = path.stat()
    digest = _sha256_file(path)
    try:
        payload = load_checkpoint(path)
    except Exception as error:
        raise CheckpointInventoryError(f"{context}: checkpoint cannot be loaded") from error
    after = path.stat()
    if _stat_signature(before) != _stat_signature(after):
        raise CheckpointInventoryError(
            f"{context}: checkpoint changed while being inventoried"
        )
    _validate_payload(
        payload,
        cohort,
        variant=variant,
        seed=seed,
        role=role,
        context=context,
    )
    canonical_path = (
        Path("artifacts")
        / "training"
        / cohort.experiment_id.lower()
        / variant
        / f"seed_{seed}"
        / ROLE_PATHS[role]
    )
    return {
        "experiment_id": cohort.experiment_id,
        "task": cohort.task,
        "variant": variant,
        "seed": seed,
        "role": role,
        "path": canonical_path.as_posix(),
        "bytes": before.st_size,
        "sha256": digest,
        "code_commit": cohort.accepted_commits[(variant, seed)],
        "config_hash": cohort.config_hash,
        "protocol_version": cohort.protocol_version,
    }


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=INVENTORY_FIELDS, lineterminator="\n", extrasaction="raise"
    )
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _validate_historical_projection(
    rows: Sequence[Mapping[str, Any]], existing_inventory: str | Path
) -> None:
    existing_path = Path(existing_inventory).expanduser().resolve()
    try:
        existing = existing_path.read_bytes()
    except OSError as error:
        raise CheckpointInventoryError(
            f"existing checkpoint inventory is unavailable: {existing_path}"
        ) from error
    try:
        reader = csv.DictReader(io.StringIO(existing.decode("utf-8"), newline=""))
        if tuple(reader.fieldnames or ()) != INVENTORY_FIELDS:
            raise CheckpointInventoryError(
                "existing checkpoint inventory does not use the frozen 11-column schema"
            )
        existing_rows = list(reader)
    except UnicodeDecodeError as error:
        raise CheckpointInventoryError(
            "existing checkpoint inventory is not valid UTF-8"
        ) from error
    existing_projection = [
        row
        for row in existing_rows
        if row["experiment_id"] in HISTORICAL_IDS and row["role"] in HISTORICAL_ROLES
    ]
    if len(existing_projection) != 150 or len(existing_rows) not in {150, 540}:
        raise CheckpointInventoryError(
            "existing checkpoint inventory must be either the 150-row historical "
            "inventory or a prior 540-row registered inventory"
        )
    expected_projection = (
        existing if len(existing_rows) == 150 else _csv_bytes(existing_projection)
    )
    projection = [
        row
        for row in rows
        if row["experiment_id"] in HISTORICAL_IDS and row["role"] in HISTORICAL_ROLES
    ]
    if len(projection) != 150:
        raise CheckpointInventoryError(
            f"historical five-role projection has {len(projection)} rows, expected 150"
        )
    if _csv_bytes(projection) != expected_projection:
        raise CheckpointInventoryError(
            "generated E09/E10 five-role projection is not byte-identical to the "
            "existing checkpoint inventory"
        )


def _atomic_write(path: str | Path, payload: bytes) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def export_checkpoint_inventory(
    manifest_path: str | Path,
    cohort_roots: Mapping[str, str | Path],
    analysis_output_dir: str | Path,
    existing_inventory: str | Path,
    output_path: str | Path,
    *,
    repository_root: str | Path = REPOSITORY_ROOT,
) -> tuple[dict[str, Any], ...]:
    """Validate and atomically export the registered 90-run checkpoint matrix."""

    repository = Path(repository_root).expanduser().resolve()
    manifest = _load_manifest(manifest_path, repository)
    registrations = _cohort_registrations(manifest, repository)

    # This is intentionally the last operation before any cohort-root resolution,
    # stat, listing, hashing, or checkpoint deserialization.
    _validate_gate_audit(manifest, registrations, analysis_output_dir)
    roots = _resolve_roots(cohort_roots)

    rows: list[dict[str, Any]] = []
    for experiment_id in EXPERIMENT_IDS:
        cohort = registrations[experiment_id]
        for variant in sorted(cohort.variants):
            for seed in cohort.seeds:
                run_dir = roots[experiment_id] / variant / f"seed_{seed}"
                files = _checkpoint_files(run_dir)
                for role in ROLE_PATHS:
                    rows.append(
                        _read_checkpoint_row(
                            files[role],
                            cohort,
                            variant=variant,
                            seed=seed,
                            role=role,
                        )
                    )
    if len(rows) != 540:
        raise CheckpointInventoryError(
            f"checkpoint inventory has {len(rows)} rows, expected 540"
        )
    _validate_historical_projection(rows, existing_inventory)
    _atomic_write(output_path, _csv_bytes(rows))
    return tuple(rows)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPOSITORY_ROOT
        / "configs"
        / "analysis"
        / "e20_e21_training_replication_v3.yaml",
    )
    parser.add_argument(
        "--analysis-output-dir",
        type=Path,
        required=True,
        help="Completed E20/E21 analysis directory containing the accepted gate audit.",
    )
    parser.add_argument(
        "--cohort-root",
        action="append",
        required=True,
        metavar="EXPERIMENT_ID=PATH",
        help="Registered accepted-run root; repeat for E09, E10, E20, and E21.",
    )
    parser.add_argument(
        "--existing-inventory",
        type=Path,
        default=REPOSITORY_ROOT / "results" / "checkpoints" / "checkpoint_inventory.csv",
        help="Frozen E09/E10 five-role inventory used for byte-equivalence validation.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=REPOSITORY_ROOT,
        help=argparse.SUPPRESS,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the deterministic checkpoint-inventory CLI."""

    arguments = _parser().parse_args(argv)
    rows = export_checkpoint_inventory(
        arguments.manifest,
        parse_cohort_roots(arguments.cohort_root),
        arguments.analysis_output_dir,
        arguments.existing_inventory,
        arguments.output,
        repository_root=arguments.repository_root,
    )
    print(f"wrote {len(rows)} checkpoint rows to {arguments.output}")
    return 0


__all__ = [
    "INVENTORY_FIELDS",
    "ROLE_PATHS",
    "CheckpointInventoryError",
    "export_checkpoint_inventory",
    "main",
    "parse_cohort_roots",
]
