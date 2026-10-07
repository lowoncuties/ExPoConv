"""Fail-closed analysis for the registered E20/E21 training replication.

This module is deliberately separate from :mod:`poincare_exec.analysis.build_all`.
The legacy exporter assumes one training cohort per task, whereas E20/E21 repeat
the E09/E10 tasks with different seeds and on a different platform.  All
selection and pairing here is therefore keyed by ``experiment_id``.

Endpoint values are not extracted until the global embargo gate has established
that every registered E20/E21 run is present, successful, and provenance-valid.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jax
import numpy as np

from poincare_exec.analysis.training_extension_statistics import (
    common_metadata,
    efficiency_rows,
    endpoint_rows,
    platform_rows,
    quality_rows,
    variant_rows,
)
from poincare_exec.analysis.training_extension_types import (
    CANDIDATE_VARIANT,
    PRIMARY_METRIC_FIELDS,
    REFERENCE_VARIANT,
    RUN_AUDIT_FIELDS,
    CohortSpec,
    ExtensionAnalysisError,
    ValidatedRun,
)
from poincare_exec.config.loader import canonical_json, config_hash, load_yaml
from poincare_exec.training.checkpoints import load_checkpoint

REQUIRED_OUTPUT_KEYS = frozenset(
    {
        "gate_audit",
        "run_audit",
        "endpoint_rows",
        "variant_summary",
        "paired_quality_by_seed",
        "paired_quality",
        "paired_iou",
        "paired_efficiency_by_seed",
        "paired_efficiency",
        "platform_strata",
        "forest_figure_stem",
        "audit_summary",
    }
)
SOURCE_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
EXPORTER_SOURCE_FILES = (
    "scripts/export_e20_e21_training.py",
    "src/poincare_exec/analysis/training_extension_statistics.py",
    "src/poincare_exec/analysis/training_extension_types.py",
    "src/poincare_exec/analysis/training_extensions.py",
)


def _require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _require_sequence(value: Any, *, name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise ValueError(f"{name} must be a sequence")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_line_sha256(value: Mapping[str, Any]) -> str:
    payload = (canonical_json(dict(value)) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_document(
    metadata: Any,
    *,
    name: str,
    repository_root: Path,
) -> tuple[str, str]:
    declared = _require_mapping(metadata, name=name)
    relative = Path(str(declared.get("path", "")))
    expected = str(declared.get("sha256", ""))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError(f"{name}.path must be repository relative")
    if len(expected) != 64 or any(
        character not in "0123456789abcdef" for character in expected
    ):
        raise ValueError(f"{name}.sha256 must be a lowercase SHA-256 digest")
    resolved = (repository_root / relative).resolve()
    if not resolved.is_relative_to(repository_root) or not resolved.is_file():
        raise ValueError(f"{name}.path is missing or outside the repository")
    actual = _sha256_file(resolved)
    if actual != expected:
        raise ValueError(f"{name} SHA-256 mismatch")
    return relative.as_posix(), actual


def _git_command(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ExtensionAnalysisError(
            f"git {' '.join(arguments)} failed: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def _validate_analysis_checkout(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    repository_root: Path,
) -> dict[str, Any]:
    """Require the exact clean, committed exporter before any result access."""

    source_root = SOURCE_REPOSITORY_ROOT.resolve()
    if repository_root != source_root:
        raise ExtensionAnalysisError(
            "analysis documents and exporter must come from the same checkout"
        )
    top_level = Path(_git_command(source_root, "rev-parse", "--show-toplevel")).resolve()
    if top_level != source_root:
        raise ExtensionAnalysisError("analysis source is not at the Git checkout root")
    status = _git_command(source_root, "status", "--porcelain", "--untracked-files=all")
    if status:
        raise ExtensionAnalysisError(
            "analysis checkout must be clean before endpoint unblinding"
        )
    protocol = Path(
        str(_require_mapping(manifest["protocol_document"], name="protocol")["path"])
    )
    contract = Path(
        str(_require_mapping(manifest["contract_document"], name="contract")["path"])
    )
    required = [
        manifest_path,
        repository_root / protocol,
        repository_root / contract,
        *(source_root / relative for relative in EXPORTER_SOURCE_FILES),
    ]
    for field in ("base_manifest_document", "base_contract_document"):
        if manifest.get(field) is not None:
            relative = Path(
                str(_require_mapping(manifest[field], name=field).get("path", ""))
            )
            required.append(repository_root / relative)
    relative_paths = []
    for path in required:
        resolved = path.resolve()
        if not resolved.is_relative_to(source_root) or not resolved.is_file():
            raise ExtensionAnalysisError(
                f"required analysis source is missing/outside checkout: {resolved}"
            )
        relative = resolved.relative_to(source_root).as_posix()
        _git_command(source_root, "cat-file", "-e", f"HEAD:{relative}")
        relative_paths.append(relative)
    commit = _git_command(source_root, "rev-parse", "HEAD")
    return {
        "commit": commit,
        "dirty": False,
        "status_porcelain": [],
        "tracked_sources": relative_paths,
    }


def _validate_registered_file(root: Path, metadata: Any, *, name: str) -> Path:
    registered = _require_mapping(metadata, name=name)
    relative = Path(str(registered.get("relative_path", "")))
    expected = str(registered.get("sha256", ""))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ExtensionAnalysisError(f"{name}.relative_path must be root relative")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ExtensionAnalysisError(f"{name} is missing or outside its evidence root")
    if _sha256_file(path) != expected:
        raise ExtensionAnalysisError(f"{name} SHA-256 mismatch")
    return path


def _matches_identity(row: Mapping[str, Any], identity: Mapping[str, Any]) -> bool:
    for field, expected in identity.items():
        actual = (
            row.get("seed", row.get("seed_or_session"))
            if field == "seed"
            else row.get(field)
        )
        if actual != expected:
            return False
    return True


def _require_expected_subset(
    actual: Mapping[str, Any], expected: Mapping[str, Any], *, context: str
) -> None:
    for field, value in expected.items():
        _require_equal(actual.get(field), value, context=context, field=field)


def _validate_environment_registration(
    record: Mapping[str, Any], expected: Mapping[str, Any], *, context: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate a SHA-pinned environment record against its frozen subset."""

    for field in ("captured_at_utc", "hostname", "platform"):
        _require_equal(record.get(field), expected.get(field), context=context, field=field)
    for section in ("git", "python", "environment", "packages"):
        _require_expected_subset(
            _require_mapping(record.get(section), name=f"{context}.{section}"),
            _require_mapping(expected.get(section), name=f"expected.{section}"),
            context=f"{context}.{section}",
        )
    actual_jax = _require_mapping(record.get("jax"), name=f"{context}.jax")
    expected_jax = _require_mapping(expected.get("jax"), name="expected.jax")
    jax_fields = ("version", "jaxlib_version", "backend", "x64_enabled")
    _require_expected_subset(
        actual_jax,
        {field: expected_jax[field] for field in jax_fields},
        context=f"{context}.jax",
    )
    devices = _require_sequence(actual_jax.get("devices"), name=f"{context}.jax.devices")
    _require_equal(
        len(devices),
        expected_jax["device_count"],
        context=context,
        field="jax.device_count",
    )
    if len(devices) != 1:
        raise ExtensionAnalysisError(
            f"{context}: exactly one visible JAX device is required"
        )
    device = _require_mapping(devices[0], name=f"{context}.jax.devices[0]")
    for actual_field, expected_field in (
        ("id", "device_id"),
        ("platform", "device_platform"),
        ("device_kind", "device_kind"),
    ):
        _require_equal(
            device.get(actual_field),
            expected_jax[expected_field],
            context=context,
            field=f"jax.{expected_field}",
        )

    physical = _require_mapping(
        expected.get("selected_physical_gpu"), name="expected.selected_physical_gpu"
    )
    _require_equal(
        physical.get("bind_with_campaign_attempt_snapshots"),
        True,
        context=context,
        field="bind_with_campaign_attempt_snapshots",
    )
    driver_tokens = [
        str(physical[field])
        for field in ("driver", "driver_advertised_cuda")
        if physical.get(field) is not None
    ]
    if driver_tokens:
        nvidia_smi = _require_mapping(
            record.get("nvidia_smi"), name=f"{context}.nvidia_smi"
        )
        nvidia_text = str(nvidia_smi.get("stdout", ""))
        for token in driver_tokens:
            if token not in nvidia_text:
                raise ExtensionAnalysisError(
                    f"{context}: nvidia-smi output lacks {token!r}"
                )
    toolkit = _require_mapping(
        expected.get("contextual_system_toolkit"),
        name="expected.contextual_system_toolkit",
    )
    _require_equal(
        toolkit.get("used_for_jax_runtime"),
        False,
        context=context,
        field="used_for_jax_runtime",
    )
    nvcc = _require_mapping(record.get("nvcc"), name=f"{context}.nvcc")
    nvcc_text = str(nvcc.get("stdout", ""))
    release = re.escape(str(toolkit["nvcc_release"]))
    version = re.escape(str(toolkit["nvcc_version"]))
    if not re.search(rf"release\s+{release}\b", nvcc_text) or not re.search(
        rf"V{version}\b", nvcc_text
    ):
        raise ExtensionAnalysisError(f"{context}: nvcc identity mismatch")
    software = {
        "git": dict(_require_mapping(expected["git"], name="expected.git")),
        "python": dict(_require_mapping(expected["python"], name="expected.python")),
        "jax": {field: expected_jax[field] for field in jax_fields},
        "packages": dict(_require_mapping(expected["packages"], name="expected.packages")),
        "environment": dict(
            _require_mapping(expected["environment"], name="expected.environment")
        ),
    }
    matched_device = {
        "jax_device_id": expected_jax["device_id"],
        "jax_device_kind": expected_jax["device_kind"],
        **dict(physical),
    }
    return software, matched_device


def _validate_runtime_evidence(
    manifest: Mapping[str, Any],
    evidence_roots: Mapping[str, Path],
    specs: Mapping[str, CohortSpec],
) -> list[dict[str, Any]]:
    """Verify the accepted runtime capture before opening endpoint summaries."""

    raw_specs = _require_sequence(
        manifest.get("accepted_runtime_evidence", ()),
        name="accepted_runtime_evidence",
    )
    registered = {
        str(
            _require_mapping(value, name="runtime evidence").get("evidence_id", "")
        ): _require_mapping(value, name="runtime evidence")
        for value in raw_specs
    }
    if not all(registered) or len(registered) != len(raw_specs):
        raise ValueError("runtime evidence IDs must be unique and nonempty")
    missing = sorted(set(registered) - set(evidence_roots))
    extra = sorted(set(evidence_roots) - set(registered))
    if missing or extra:
        raise ExtensionAnalysisError(
            f"runtime evidence roots disagree with registration; "
            f"missing={missing}, extra={extra}"
        )

    audit = []
    covered_cohorts: set[str] = set()
    for evidence_id, registration in sorted(registered.items()):
        _require_equal(
            registration.get("input_kind"),
            "accepted_evidence_root",
            context=evidence_id,
            field="input_kind",
        )
        _require_equal(
            registration.get("validation_order"),
            "before_opening_any_e20_or_e21_summary",
            context=evidence_id,
            field="validation_order",
        )
        for field in (
            "accepted_training_run",
            "counts_toward_embargo_gate",
            "endpoint_values_expected",
        ):
            _require_equal(registration.get(field), False, context=evidence_id, field=field)
        cohorts = tuple(
            str(value)
            for value in _require_sequence(
                registration.get("cohorts"), name=f"{evidence_id}.cohorts"
            )
        )
        if not cohorts or len(cohorts) != len(set(cohorts)):
            raise ExtensionAnalysisError(f"{evidence_id}: cohorts must be unique")
        if any(cohort not in specs for cohort in cohorts):
            raise ExtensionAnalysisError(f"{evidence_id}: references unknown cohorts")
        overlap = covered_cohorts.intersection(cohorts)
        if overlap:
            raise ExtensionAnalysisError(
                f"{evidence_id}: duplicate runtime evidence for {sorted(overlap)}"
            )
        covered_cohorts.update(cohorts)
        root = Path(evidence_roots[evidence_id]).expanduser().resolve()
        if not root.is_dir():
            raise ExtensionAnalysisError(f"{evidence_id}: evidence root is missing")
        files = _require_mapping(
            registration.get("required_files"), name=f"{evidence_id}.required_files"
        )
        if set(files) != {"environment_record"}:
            raise ExtensionAnalysisError(
                f"{evidence_id}: exactly one environment_record is required"
            )
        path = _validate_registered_file(
            root, files["environment_record"], name=f"{evidence_id}.environment_record"
        )
        record = _require_mapping(
            json.loads(path.read_text(encoding="utf-8")),
            name=f"{evidence_id}.environment_record",
        )
        expected = _require_mapping(
            registration.get("expected_environment"),
            name=f"{evidence_id}.expected_environment",
        )
        matched_software, matched_device = _validate_environment_registration(
            record, expected, context=evidence_id
        )
        physical = _require_mapping(
            expected.get("selected_physical_gpu"), name="expected.selected_physical_gpu"
        )
        for cohort_id in cohorts:
            cohort_device = specs[cohort_id].physical_device
            for manifest_field, expected_field in (
                ("physical_index", "physical_index"),
                ("uuid", "uuid"),
                ("name", "name"),
                ("driver", "driver"),
                ("advertised_cuda", "driver_advertised_cuda"),
            ):
                _require_equal(
                    cohort_device.get(manifest_field),
                    physical.get(expected_field),
                    context=f"{evidence_id}/{cohort_id}",
                    field=expected_field,
                )
        audit.append(
            {
                "evidence_id": evidence_id,
                "cohorts": list(cohorts),
                "evidence_root": str(root),
                "registered_relative_file": path.relative_to(root).as_posix(),
                "environment_path": str(path),
                "environment_sha256": _sha256_file(path),
                "matched_software": matched_software,
                "matched_device": matched_device,
                "runtime_identity_validated": True,
            }
        )
    expected_cohorts = set(_registered_extension_ids(manifest))
    if raw_specs and covered_cohorts != expected_cohorts:
        raise ExtensionAnalysisError(
            "accepted runtime evidence does not cover exactly the extension cohorts"
        )
    return audit


def _validate_supplemental_evidence(
    manifest: Mapping[str, Any],
    evidence_roots: Mapping[str, Path],
    specs: Mapping[str, CohortSpec],
) -> list[dict[str, Any]]:
    """Verify registered rejected attempts without treating them as gate runs."""

    raw_specs = _require_sequence(
        manifest.get("supplemental_attempt_evidence", ()),
        name="supplemental_attempt_evidence",
    )
    registered = {
        str(
            _require_mapping(value, name="supplemental attempt").get("evidence_id", "")
        ): _require_mapping(value, name="supplemental attempt")
        for value in raw_specs
    }
    if not all(registered) or len(registered) != len(raw_specs):
        raise ValueError("supplemental evidence IDs must be unique and nonempty")
    missing = sorted(set(registered) - set(evidence_roots))
    extra = sorted(set(evidence_roots) - set(registered))
    if missing or extra:
        raise ExtensionAnalysisError(
            f"supplemental evidence roots disagree with registration; "
            f"missing={missing}, extra={extra}"
        )

    audit_rows = []
    for evidence_id, registered_attempt in sorted(registered.items()):
        _require_equal(
            registered_attempt.get("input_kind"),
            "evidence_root",
            context=evidence_id,
            field="input_kind",
        )
        for field in (
            "accepted_training_run",
            "counts_toward_embargo_gate",
            "endpoint_values_expected",
        ):
            _require_equal(
                registered_attempt.get(field),
                False,
                context=evidence_id,
                field=field,
            )
        root = Path(evidence_roots[evidence_id]).expanduser().resolve()
        if not root.is_dir():
            raise ExtensionAnalysisError(f"{evidence_id}: evidence root is missing")
        files = _require_mapping(
            registered_attempt.get("required_files"),
            name=f"{evidence_id}.required_files",
        )
        required_names = {
            "failure_manifest",
            "initial_checkpoint",
            "environment",
            "campaign_attempts",
        }
        if set(files) != required_names:
            raise ExtensionAnalysisError(
                f"{evidence_id}: required file roles must be {sorted(required_names)}"
            )
        paths = {
            role: _validate_registered_file(root, metadata, name=f"{evidence_id}.{role}")
            for role, metadata in files.items()
        }
        identity = _require_mapping(
            registered_attempt.get("expected_identity"),
            name=f"{evidence_id}.expected_identity",
        )
        outcome = _require_mapping(
            registered_attempt.get("expected_outcome"),
            name=f"{evidence_id}.expected_outcome",
        )
        experiment_id = str(identity.get("experiment_id", ""))
        if experiment_id not in specs:
            raise ExtensionAnalysisError(
                f"{evidence_id}: unregistered experiment {experiment_id!r}"
            )
        cohort = specs[experiment_id]
        _require_equal(outcome.get("status"), "failed", context=evidence_id, field="status")
        if int(outcome.get("returncode", 0)) == 0:
            raise ExtensionAnalysisError(
                f"{evidence_id}: failure returncode must be nonzero"
            )
        _require_equal(
            outcome.get("failure_class"),
            "infrastructure_compiler_failure",
            context=evidence_id,
            field="failure_class",
        )
        _require_equal(
            outcome.get("failure_stage"),
            "before_first_training_update",
            context=evidence_id,
            field="failure_stage",
        )
        _require_equal(
            int(outcome.get("completed_updates", -1)),
            0,
            context=evidence_id,
            field="completed_updates",
        )
        failure = _require_mapping(
            json.loads(paths["failure_manifest"].read_text(encoding="utf-8")),
            name=f"{evidence_id}.failure_manifest",
        )
        if not _matches_identity(failure, identity):
            raise ExtensionAnalysisError(
                f"{evidence_id}: failure manifest identity disagrees with registration"
            )
        for field in ("status", "error_type", "error"):
            _require_equal(
                failure.get(field), outcome.get(field), context=evidence_id, field=field
            )
        if any(
            field in failure
            for field in ("official_test", "deduplicated_test", "validation_best")
        ):
            raise ExtensionAnalysisError(
                f"{evidence_id}: rejected attempt unexpectedly contains endpoints"
            )
        all_attempts = _jsonl(paths["campaign_attempts"])
        attempts = [row for row in all_attempts if _matches_identity(row, identity)]
        if len(attempts) != 1:
            raise ExtensionAnalysisError(
                f"{evidence_id}: expected one matching campaign attempt; "
                f"found {len(attempts)}"
            )
        attempt = attempts[0]
        for field in ("status", "returncode"):
            _require_equal(
                attempt.get(field), outcome.get(field), context=evidence_id, field=field
            )
        _require_equal(
            attempt.get("summary_path"), None, context=evidence_id, field="summary_path"
        )
        if str(outcome["error"]) not in str(attempt.get("stderr", "")):
            raise ExtensionAnalysisError(
                f"{evidence_id}: campaign stderr does not contain registered error"
            )
        _validate_snapshot(
            attempt.get("gpu_snapshot_before"),
            cohort=cohort,
            context=f"{evidence_id} before",
            require_idle=True,
        )
        _validate_snapshot(
            attempt.get("gpu_snapshot_after"),
            cohort=cohort,
            context=f"{evidence_id} after",
            require_idle=True,
        )

        environment = _require_mapping(
            json.loads(paths["environment"].read_text(encoding="utf-8")),
            name=f"{evidence_id}.environment",
        )
        expected_environment = _require_mapping(
            registered_attempt.get("expected_environment"),
            name=f"{evidence_id}.expected_environment",
        )
        matched_software, matched_device = _validate_environment_registration(
            environment, expected_environment, context=evidence_id
        )
        environment_git = _require_mapping(environment.get("git"), name="environment.git")
        _require_equal(
            environment_git.get("commit"),
            identity.get("code_commit"),
            context=evidence_id,
            field="environment.git.commit",
        )
        _require_equal(
            environment_git.get("dirty"),
            identity.get("code_dirty"),
            context=evidence_id,
            field="environment.git.dirty",
        )
        _require_equal(
            environment.get("hostname"),
            cohort.physical_device.get("host"),
            context=evidence_id,
            field="environment.hostname",
        )
        for field in ("physical_index", "uuid", "name"):
            _require_equal(
                matched_device.get(field),
                cohort.physical_device.get(field),
                context=evidence_id,
                field=f"environment.physical_gpu.{field}",
            )

        expected_run = (
            root
            / "training"
            / experiment_id.lower()
            / str(identity["variant"])
            / f"seed_{int(identity['seed'])}"
        ).resolve()
        _require_equal(
            paths["failure_manifest"].parent,
            expected_run,
            context=evidence_id,
            field="failure manifest run directory",
        )
        _require_equal(
            paths["initial_checkpoint"],
            expected_run / "checkpoints" / "initial.pkl",
            context=evidence_id,
            field="initial checkpoint path",
        )
        _validate_rejected_initial_checkpoint(
            paths["initial_checkpoint"], identity=identity
        )
        progress_artifacts = (
            "summary.json",
            "training_metrics.jsonl",
            "resume.pkl",
            "checkpoints/10_percent.pkl",
            "checkpoints/50_percent.pkl",
            "checkpoints/validation_best.pkl",
            "checkpoints/final.pkl",
            "predictions_official_test.npz",
            "predictions_deduplicated_test.npz",
            "predictions_validation_best_official_test.npz",
            "predictions_validation_best_deduplicated_test.npz",
        )
        unexpected = [
            relative
            for relative in progress_artifacts
            if (expected_run / relative).exists()
        ]
        if unexpected:
            raise ExtensionAnalysisError(
                f"{evidence_id}: pre-update failure has progress artifacts {unexpected}"
            )
        attempt_row = all_attempts.index(attempt) + 1
        audit_rows.append(
            {
                "analysis_id": manifest["analysis_id"],
                "evidence_id": evidence_id,
                "record_type": "rejected_pre_update_infrastructure_attempt",
                "dataset": cohort.dataset,
                **identity,
                **outcome,
                "disposition": registered_attempt["disposition"],
                "accepted_training_run": False,
                "counts_toward_embargo_gate": False,
                "endpoint_values_expected": False,
                "runtime_identity": canonical_json(matched_software),
                "physical_gpu_host": cohort.physical_device.get("host"),
                "physical_gpu_index": cohort.physical_device.get("physical_index"),
                "physical_gpu_uuid": cohort.physical_device.get("uuid"),
                "physical_gpu_name": cohort.physical_device.get("name"),
                "attempt_path": str(paths["campaign_attempts"]),
                "attempt_row": attempt_row,
                "evidence_root": str(root),
                "supporting_files": canonical_json(
                    {role: str(path) for role, path in sorted(paths.items())}
                ),
                "supporting_sha256": canonical_json(
                    {
                        role: str(_require_mapping(metadata, name=role)["sha256"])
                        for role, metadata in sorted(files.items())
                    }
                ),
            }
        )
    return audit_rows


def _parse_telemetry_timestamp(value: str, *, context: str) -> datetime:
    try:
        return datetime.strptime(value, "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=UTC)
    except ValueError as error:
        raise ExtensionAnalysisError(
            f"{context}: invalid telemetry timestamp {value!r}"
        ) from error


def _validate_post_worker_evidence(
    manifest: Mapping[str, Any],
    evidence_roots: Mapping[str, Path],
    specs: Mapping[str, CohortSpec],
) -> dict[tuple[str, str, int], dict[str, Any]]:
    """Validate exact registrations for non-idle snapshots taken after a worker.

    The before-worker snapshot remains the exclusivity boundary.  A non-idle
    after-worker snapshot is admissible only when infrastructure evidence fixed
    before endpoint unblinding establishes that the terminal summary preceded
    the first foreign-only telemetry sample.
    """

    raw_specs = _require_sequence(
        manifest.get("post_worker_snapshot_evidence", ()),
        name="post_worker_snapshot_evidence",
    )
    if not raw_specs:
        if evidence_roots:
            raise ExtensionAnalysisError(
                "post-worker evidence roots were supplied without registrations"
            )
        return {}

    policy = _require_mapping(
        manifest.get("post_worker_snapshot_policy"),
        name="post_worker_snapshot_policy",
    )
    _require_equal(
        policy.get("before_snapshot"),
        "require_idle",
        context="post-worker snapshot policy",
        field="before_snapshot",
    )
    _require_equal(
        policy.get("after_snapshot"),
        "require_identity_and_exact_registration_if_nonidle",
        context="post-worker snapshot policy",
        field="after_snapshot",
    )

    registrations: dict[str, Mapping[str, Any]] = {}
    for raw in raw_specs:
        registration = _require_mapping(raw, name="post-worker evidence")
        evidence_id = str(registration.get("evidence_id", ""))
        if not evidence_id or evidence_id in registrations:
            raise ValueError("post-worker evidence IDs must be unique and nonempty")
        registrations[evidence_id] = registration
    missing = sorted(set(registrations) - set(evidence_roots))
    extra = sorted(set(evidence_roots) - set(registrations))
    if missing or extra:
        raise ExtensionAnalysisError(
            "post-worker evidence roots disagree with registration; "
            f"missing={missing}, extra={extra}"
        )

    dispositions: dict[tuple[str, str, int], dict[str, Any]] = {}
    required_roles = {
        "campaign_attempts_snapshot",
        "wrapper_telemetry",
        "wrapper_status",
        "wrapper_stderr",
        "wrapper_command",
        "wrapper_gpu_phase",
    }
    for evidence_id, registration in sorted(registrations.items()):
        _require_equal(
            registration.get("input_kind"),
            "post_worker_evidence_root",
            context=evidence_id,
            field="input_kind",
        )
        identity = _require_mapping(
            registration.get("identity"), name=f"{evidence_id}.identity"
        )
        experiment_id = str(identity.get("experiment_id", ""))
        variant = str(identity.get("variant", ""))
        seed = int(identity.get("seed", -1))
        if experiment_id not in specs:
            raise ExtensionAnalysisError(
                f"{evidence_id}: unregistered experiment {experiment_id!r}"
            )
        cohort = specs[experiment_id]
        if (variant, seed) not in cohort.expected_run_keys:
            raise ExtensionAnalysisError(f"{evidence_id}: unregistered run identity")
        key = (experiment_id, variant, seed)
        if key in dispositions:
            raise ExtensionAnalysisError(
                f"{evidence_id}: duplicate post-worker disposition for {key}"
            )

        root = Path(evidence_roots[evidence_id]).expanduser().resolve()
        if not root.is_dir():
            raise ExtensionAnalysisError(f"{evidence_id}: evidence root is missing")
        files = _require_mapping(
            registration.get("required_files"), name=f"{evidence_id}.required_files"
        )
        if set(files) != required_roles:
            raise ExtensionAnalysisError(
                f"{evidence_id}: required file roles must be {sorted(required_roles)}"
            )
        paths = {
            role: _validate_registered_file(root, metadata, name=f"{evidence_id}.{role}")
            for role, metadata in files.items()
        }
        required_lines = _require_mapping(
            registration.get("required_text_lines"),
            name=f"{evidence_id}.required_text_lines",
        )
        if not set(required_lines).issubset(
            required_roles - {"campaign_attempts_snapshot"}
        ):
            raise ExtensionAnalysisError(
                f"{evidence_id}: required_text_lines references an unknown text role"
            )
        for role, raw_lines in required_lines.items():
            lines = paths[str(role)].read_text(encoding="utf-8").splitlines()
            for expected_line in _require_sequence(
                raw_lines, name=f"{evidence_id}.required_text_lines.{role}"
            ):
                if str(expected_line) not in lines:
                    raise ExtensionAnalysisError(
                        f"{evidence_id}: {role} lacks registered line {expected_line!r}"
                    )

        archived_attempts = _jsonl(paths["campaign_attempts_snapshot"])
        matching_attempts = [
            row
            for row in archived_attempts
            if str(row.get("experiment_id", "")) == experiment_id
            and str(row.get("variant", "")) == variant
            and int(row.get("seed", -1)) == seed
        ]
        if len(matching_attempts) != 1:
            raise ExtensionAnalysisError(
                f"{evidence_id}: expected one archived attempt; "
                f"found {len(matching_attempts)}"
            )
        archived_attempt = matching_attempts[0]
        expected_attempt = _require_mapping(
            registration.get("expected_attempt"),
            name=f"{evidence_id}.expected_attempt",
        )
        _require_equal(
            _canonical_line_sha256(archived_attempt),
            expected_attempt.get("canonical_row_sha256"),
            context=evidence_id,
            field="canonical_row_sha256",
        )
        archived_after = _require_mapping(
            archived_attempt.get("gpu_snapshot_after"),
            name=f"{evidence_id}.gpu_snapshot_after",
        )
        _require_equal(
            _canonical_line_sha256(archived_after),
            expected_attempt.get("after_snapshot_sha256"),
            context=evidence_id,
            field="after_snapshot_sha256",
        )
        _require_equal(
            archived_attempt.get("timestamp_utc"),
            expected_attempt.get("timestamp_utc"),
            context=evidence_id,
            field="timestamp_utc",
        )
        _validate_snapshot(
            archived_attempt.get("gpu_snapshot_before"),
            cohort=cohort,
            context=f"{evidence_id} archived before",
            require_idle=True,
        )
        _validate_snapshot(
            archived_after,
            cohort=cohort,
            context=f"{evidence_id} archived after",
            require_idle=False,
        )
        if not archived_after.get("compute_processes"):
            raise ExtensionAnalysisError(
                f"{evidence_id}: registered after snapshot is already idle"
            )

        timing = _require_mapping(
            registration.get("timing_evidence"),
            name=f"{evidence_id}.timing_evidence",
        )
        last_worker_line = str(timing.get("last_worker_allocation_line", ""))
        first_foreign_line = str(timing.get("first_foreign_only_line", ""))
        telemetry_lines = paths["wrapper_telemetry"].read_text(
            encoding="utf-8"
        ).splitlines()
        try:
            last_worker_index = telemetry_lines.index(last_worker_line)
            first_foreign_index = telemetry_lines.index(first_foreign_line)
        except ValueError as error:
            raise ExtensionAnalysisError(
                f"{evidence_id}: registered telemetry transition is missing"
            ) from error
        if first_foreign_index != last_worker_index + 1:
            raise ExtensionAnalysisError(
                f"{evidence_id}: telemetry transition is not consecutive"
            )
        last_worker_time = _parse_telemetry_timestamp(
            last_worker_line.split(",", 1)[0], context=evidence_id
        )
        first_foreign_time = _parse_telemetry_timestamp(
            first_foreign_line.split(",", 1)[0], context=evidence_id
        )
        if first_foreign_time <= last_worker_time:
            raise ExtensionAnalysisError(
                f"{evidence_id}: foreign-only telemetry does not follow worker telemetry"
            )

        dispositions[key] = {
            "evidence_id": evidence_id,
            "identity": dict(identity),
            "expected_attempt": dict(expected_attempt),
            "summary_stat": dict(
                _require_mapping(
                    registration.get("summary_stat"),
                    name=f"{evidence_id}.summary_stat",
                )
            ),
            "first_foreign_time": first_foreign_time,
            "evidence_root": str(root),
            "supporting_files": {role: str(path) for role, path in sorted(paths.items())},
            "supporting_sha256": {
                role: str(_require_mapping(metadata, name=role)["sha256"])
                for role, metadata in sorted(files.items())
            },
            "used": False,
        }
    return dispositions


def _accepted_commits(
    experiment_id: str,
    raw: Any,
    expected_run_keys: frozenset[tuple[str, int]],
) -> dict[tuple[str, int], str]:
    commits: dict[tuple[str, int], str] = {}
    for index, entry_value in enumerate(
        _require_sequence(raw, name=f"cohorts.{experiment_id}.accepted_code_assignments")
    ):
        entry = _require_mapping(
            entry_value,
            name=f"cohorts.{experiment_id}.accepted_code_assignments[{index}]",
        )
        commit = str(entry.get("commit", ""))
        if len(commit) != 40 or any(
            character not in "0123456789abcdef" for character in commit
        ):
            raise ValueError(
                f"{experiment_id} accepted commit must be a full Git object name"
            )
        variants = tuple(
            str(value)
            for value in _require_sequence(
                entry.get("variants"), name=f"{experiment_id} assignment variants"
            )
        )
        seeds = tuple(
            int(value)
            for value in _require_sequence(
                entry.get("seeds"), name=f"{experiment_id} assignment seeds"
            )
        )
        for key in ((variant, seed) for variant in variants for seed in seeds):
            if key not in expected_run_keys:
                raise ValueError(
                    f"{experiment_id} has an unregistered code assignment {key}"
                )
            if key in commits:
                raise ValueError(
                    f"{experiment_id} has duplicate code assignments for {key}"
                )
            commits[key] = commit
    if set(commits) != expected_run_keys:
        missing = sorted(expected_run_keys - set(commits))
        raise ValueError(f"{experiment_id} is missing code assignments for {missing}")
    return commits


def _registered_endpoint_sample_ids(
    protocol: Mapping[str, Any],
    *,
    dataset: str,
    repository_root: Path,
    endpoint_counts: Mapping[str, Mapping[str, int]],
) -> Mapping[str, tuple[str, ...]]:
    independence = _require_mapping(
        protocol.get("data_independence"), name="protocol.data_independence"
    )
    digests = _require_mapping(
        independence.get("split_manifests"),
        name="protocol.data_independence.split_manifests",
    )
    expected_digest = str(digests.get(dataset, ""))
    if len(expected_digest) != 64 or any(
        character not in "0123456789abcdef" for character in expected_digest
    ):
        raise ValueError(f"protocol has no valid split-manifest SHA-256 for {dataset}")
    path = (repository_root / "splits" / f"{dataset}.csv").resolve()
    if not path.is_relative_to(repository_root) or not path.is_file():
        raise ValueError(f"registered split manifest is missing for {dataset}")
    if _sha256_file(path) != expected_digest:
        raise ValueError(f"registered split manifest SHA-256 mismatch for {dataset}")
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required_fields = {"sample_id", "split", "exclusion_reason"}
    if not rows or not required_fields.issubset(rows[0]):
        raise ValueError(f"registered split manifest schema is invalid for {dataset}")
    official = tuple(row["sample_id"] for row in rows if row["split"] == "test")
    deduplicated = tuple(
        row["sample_id"]
        for row in rows
        if row["split"] == "test" and not row["exclusion_reason"]
    )
    sample_ids = {
        "official_test": official,
        "deduplicated_test": deduplicated,
    }
    for endpoint, counts in endpoint_counts.items():
        for partition, identifiers in sample_ids.items():
            expected = int(counts[partition])
            if len(identifiers) != expected or len(set(identifiers)) != expected:
                raise ValueError(
                    f"{dataset} {endpoint}/{partition} split identity disagrees with "
                    "the registered endpoint count"
                )
    return sample_ids


def _cohort_specs(
    manifest: Mapping[str, Any],
    *,
    repository_root: Path,
) -> dict[str, CohortSpec]:
    protocol_metadata = _require_mapping(
        manifest.get("protocol_document"), name="protocol_document"
    )
    protocol_relative = Path(str(protocol_metadata.get("path", "")))
    protocol = load_yaml(repository_root / protocol_relative)
    protocol_tasks = _require_mapping(protocol.get("tasks"), name="protocol tasks")
    raw_cohorts = _require_mapping(manifest.get("cohorts"), name="cohorts")
    extension_ids = set(_registered_extension_ids(manifest))
    output: dict[str, CohortSpec] = {}
    for experiment_id, value in raw_cohorts.items():
        cohort = _require_mapping(value, name=f"cohorts.{experiment_id}")
        config_relative, _ = _validate_document(
            cohort.get("config_document"),
            name=f"cohorts.{experiment_id}.config_document",
            repository_root=repository_root,
        )
        config = load_yaml(repository_root / config_relative)
        variants = tuple(
            str(item)
            for item in _require_sequence(
                cohort.get("variants"), name=f"cohorts.{experiment_id}.variants"
            )
        )
        analysis_variants = tuple(
            str(item)
            for item in _require_sequence(
                cohort.get("analysis_variants"),
                name=f"cohorts.{experiment_id}.analysis_variants",
            )
        )
        if len(analysis_variants) != 2:
            raise ValueError(f"{experiment_id} must declare exactly two analysis variants")
        if analysis_variants != (REFERENCE_VARIANT, CANDIDATE_VARIANT):
            raise ValueError(f"{experiment_id} extension analysis must compare R then F")
        if not set(analysis_variants).issubset(variants):
            raise ValueError(
                f"{experiment_id} analysis variants are not registered variants"
            )
        seeds = tuple(
            int(item)
            for item in _require_sequence(
                cohort.get("seeds"), name=f"cohorts.{experiment_id}.seeds"
            )
        )
        if len(seeds) != len(set(seeds)) or not seeds:
            raise ValueError(f"{experiment_id} seeds must be unique and nonempty")
        for field, expected in (
            ("experiment_id", str(experiment_id)),
            ("task", str(cohort.get("task", ""))),
            ("dataset", str(cohort.get("dataset", ""))),
            ("variants", list(variants)),
            ("seeds", list(seeds)),
        ):
            if config.get(field) != expected:
                raise ValueError(
                    f"cohorts.{experiment_id}.config_document {field} "
                    "disagrees with the analysis registration"
                )
        if str(experiment_id) in extension_ids:
            _require_equal(
                config_hash(protocol, config),
                cohort.get("run_config_hash"),
                context=f"cohorts.{experiment_id}",
                field="run_config_hash",
            )
        expected_run_keys = frozenset(
            (variant, seed) for variant in variants for seed in seeds
        )
        device = _require_mapping(
            cohort.get("physical_device"), name=f"cohorts.{experiment_id}.physical_device"
        )
        task_name = str(cohort.get("task", ""))
        task = _require_mapping(
            protocol_tasks.get(task_name), name=f"protocol task {task_name}"
        )
        runtime = str(cohort.get("runtime", ""))
        platform_stratum = f"{device.get('name', 'unknown')} | {runtime}"
        raw_counts = _require_mapping(
            cohort.get("endpoint_example_counts"),
            name=f"cohorts.{experiment_id}.endpoint_example_counts",
        )
        endpoint_counts: dict[str, dict[str, int]] = {}
        for endpoint in ("fixed_final_update", "validation_best"):
            partitions = _require_mapping(
                raw_counts.get(endpoint),
                name=f"cohorts.{experiment_id}.endpoint_example_counts.{endpoint}",
            )
            if set(partitions) != {"official_test", "deduplicated_test"}:
                raise ValueError(
                    f"{experiment_id} {endpoint} example-count partitions are invalid"
                )
            endpoint_counts[endpoint] = {
                partition: int(partitions[partition])
                for partition in ("official_test", "deduplicated_test")
            }
            if min(endpoint_counts[endpoint].values()) < 1:
                raise ValueError(
                    f"{experiment_id} endpoint example counts must be positive"
                )
        raw_health = _require_mapping(
            cohort.get("training_health"),
            name=f"cohorts.{experiment_id}.training_health",
        )
        expected_health_keys = {
            "full_train_rows_per_epoch",
            "examples_per_full_train_row",
            "final_train_rows_per_epoch",
            "examples_per_final_train_row",
            "total_train_rows_per_epoch",
            "development_examples",
            "epoch_index_origin",
            "epoch_boundary_update_formula",
        }
        if set(raw_health) != expected_health_keys:
            raise ValueError(f"{experiment_id} training-health schema is invalid")
        _require_equal(
            raw_health["epoch_index_origin"],
            1,
            context=f"cohorts.{experiment_id}.training_health",
            field="epoch_index_origin",
        )
        total_rows = int(raw_health["total_train_rows_per_epoch"])
        _require_equal(
            raw_health["epoch_boundary_update_formula"],
            f"epoch_times_{total_rows}",
            context=f"cohorts.{experiment_id}.training_health",
            field="epoch_boundary_update_formula",
        )
        integer_health = {
            field: int(raw_health[field])
            for field in expected_health_keys
            if field not in {"epoch_boundary_update_formula"}
        }
        if min(integer_health.values()) < 1:
            raise ValueError(f"{experiment_id} training-health counts must be positive")
        _require_equal(
            integer_health["full_train_rows_per_epoch"]
            + integer_health["final_train_rows_per_epoch"],
            total_rows,
            context=f"cohorts.{experiment_id}.training_health",
            field="total_train_rows_per_epoch",
        )
        _require_equal(
            int(task["epochs"]) * total_rows,
            int(task["max_updates"]),
            context=f"cohorts.{experiment_id}.training_health",
            field="registered update budget",
        )
        dataset = str(cohort.get("dataset", ""))
        endpoint_sample_ids = _registered_endpoint_sample_ids(
            protocol,
            dataset=dataset,
            repository_root=repository_root,
            endpoint_counts=endpoint_counts,
        )
        if task_name == "isic16_poincare_unet":
            task_input = _require_mapping(task.get("input"), name=f"{task_name}.input")
            prediction_item_shape = (
                int(task_input.get("height", -1)),
                int(task_input.get("width", -1)),
            )
            if min(prediction_item_shape) < 1:
                raise ValueError(f"{task_name} prediction dimensions must be positive")
        else:
            prediction_item_shape = ()
        output[str(experiment_id)] = CohortSpec(
            analysis_id=str(manifest.get("analysis_id", "")),
            experiment_id=str(experiment_id),
            role=str(cohort.get("role", "")),
            task=task_name,
            dataset=dataset,
            platform_stratum=platform_stratum,
            protocol_version=str(cohort.get("protocol_version", "")),
            config_hash=str(cohort.get("run_config_hash", "")),
            variants=variants,
            analysis_variants=(analysis_variants[0], analysis_variants[1]),
            seeds=seeds,
            accepted_commits=_accepted_commits(
                str(experiment_id),
                cohort.get("accepted_code_assignments"),
                expected_run_keys,
            ),
            physical_device=device,
            logical_device_id=(
                str(cohort["logical_device_id"])
                if cohort.get("logical_device_id") is not None
                else None
            ),
            runtime=runtime,
            expected_epochs=int(task["epochs"]),
            expected_updates=int(task["max_updates"]),
            endpoint_example_counts=endpoint_counts,
            endpoint_sample_ids=endpoint_sample_ids,
            prediction_item_shape=prediction_item_shape,
            training_health=integer_health,
        )
    return output


def _registered_extension_ids(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    embargo = _require_mapping(manifest.get("global_embargo"), name="global_embargo")
    return tuple(
        str(value)
        for value in _require_sequence(
            embargo.get("experiments"), name="global_embargo.experiments"
        )
    )


def _validate_manifest_contract(manifest: Mapping[str, Any]) -> None:
    if not str(manifest.get("analysis_id", "")):
        raise ValueError("analysis_id is required")
    _require_equal(
        manifest.get("analysis_status"),
        "frozen_before_official_test_unblinding",
        context="analysis manifest",
        field="analysis_status",
    )
    for field in ("accepted_runtime_evidence", "supplemental_attempt_evidence"):
        registrations = _require_sequence(manifest.get(field), name=field)
        if not registrations:
            raise ValueError(f"{field} must contain the frozen evidence registration")
    embargo = _require_mapping(manifest.get("global_embargo"), name="global_embargo")
    _require_equal(
        embargo.get("gate_is_joint_across_experiments"),
        True,
        context="analysis manifest",
        field="global_embargo.gate_is_joint_across_experiments",
    )
    comparison = _require_mapping(manifest.get("comparison"), name="comparison")
    _require_equal(
        comparison.get("reference_variant"),
        REFERENCE_VARIANT,
        context="analysis manifest",
        field="comparison.reference_variant",
    )
    _require_equal(
        comparison.get("candidate_variant"),
        CANDIDATE_VARIANT,
        context="analysis manifest",
        field="comparison.candidate_variant",
    )
    _require_equal(
        comparison.get("t_ablation_extended"),
        False,
        context="analysis manifest",
        field="comparison.t_ablation_extended",
    )
    cross_platform = _require_mapping(
        manifest.get("cross_platform_reporting"), name="cross_platform_reporting"
    )
    for field in (
        "grand_pooled_estimate",
        "pooled_hypothesis_test",
        "inverse_variance_fixed_effect",
        "random_effects",
    ):
        _require_equal(
            cross_platform.get(field),
            "prohibited",
            context="analysis manifest",
            field=f"cross_platform_reporting.{field}",
        )
    bootstrap = _require_mapping(manifest.get("bootstrap"), name="bootstrap")
    identity_fields = set(
        str(value)
        for value in _require_sequence(
            bootstrap.get("stream_identity_fields"),
            name="bootstrap.stream_identity_fields",
        )
    )
    required_identity = {
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
    }
    if identity_fields != required_identity:
        raise ValueError(
            "bootstrap stream identity fields disagree with the frozen contract"
        )
    failure_policy = _require_mapping(manifest.get("failure_policy"), name="failure_policy")
    for field, expected in (
        ("replace_failed_seeds", False),
        ("complete_case_primary_analysis", "prohibited"),
    ):
        _require_equal(
            failure_policy.get(field),
            expected,
            context="analysis manifest",
            field=f"failure_policy.{field}",
        )
    pre_update = _require_mapping(
        failure_policy.get("pre_update_infrastructure_failure"),
        name="failure_policy.pre_update_infrastructure_failure",
    )
    for field, expected in (
        ("retry_same_seed_and_variant", "allowed"),
        ("retain_every_attempt", True),
        ("accepted_retry_must_match_frozen_identity", True),
        ("failed_attempt_must_be_registered_before_endpoint_unblinding", True),
        ("registration_section", "supplemental_attempt_evidence"),
        ("immutable_evidence_archive_required", True),
        ("file_sha256_registration_required", True),
        ("complete_identity_registration_required", True),
        ("unregistered_attempt_disposition", "keep_global_embargo_closed"),
    ):
        _require_equal(
            pre_update.get(field),
            expected,
            context="analysis manifest",
            field=f"failure_policy.pre_update_infrastructure_failure.{field}",
        )
    interruption = _require_mapping(
        failure_policy.get("external_interruption_after_updates"),
        name="failure_policy.external_interruption_after_updates",
    )
    for field, expected in (
        ("exact_checkpoint_resume", "allowed"),
        ("retain_every_attempt", True),
        ("provenance_match_required", True),
        ("immutable_contemporaneous_checkpoint_required", True),
        ("interruption_evidence_required", True),
        ("archive_and_register_before_endpoint_unblinding", True),
        ("registration_section", "supplemental_attempt_evidence"),
        ("file_sha256_registration_required", True),
        ("complete_identity_registration_required", True),
        ("unregistered_attempt_disposition", "keep_global_embargo_closed"),
        ("currently_registered_attempts", 0),
    ):
        _require_equal(
            interruption.get(field),
            expected,
            context="analysis manifest",
            field=f"failure_policy.external_interruption_after_updates.{field}",
        )
    scientific = _require_mapping(
        failure_policy.get("scientific_or_nonfinite_failure"),
        name="failure_policy.scientific_or_nonfinite_failure",
    )
    for field, expected in (
        ("retry_to_obtain_endpoint", "prohibited"),
        ("disposition", "terminal_registered_failure"),
    ):
        _require_equal(
            scientific.get(field),
            expected,
            context="analysis manifest",
            field=f"failure_policy.scientific_or_nonfinite_failure.{field}",
        )
    unresolved = _require_mapping(
        failure_policy.get("unresolved_or_incompatible_run"),
        name="failure_policy.unresolved_or_incompatible_run",
    )
    _require_equal(
        unresolved.get("disposition"),
        "keep_global_embargo_closed_and_report_campaign_incomplete",
        context="analysis manifest",
        field="failure_policy.unresolved_or_incompatible_run.disposition",
    )
    _required_output_names(manifest)


def _required_output_names(manifest: Mapping[str, Any]) -> Mapping[str, str]:
    raw = _require_mapping(manifest.get("required_outputs"), name="required_outputs")
    if set(raw) != REQUIRED_OUTPUT_KEYS:
        raise ValueError("required_outputs keys disagree with the frozen output contract")
    names = {str(key): str(value) for key, value in raw.items()}
    for key, value in names.items():
        path = Path(value)
        if not value or value in {".", ".."} or path.is_absolute() or path.name != value:
            raise ValueError(f"required_outputs.{key} must be a safe relative basename")
    produced = [
        name
        for key, value in names.items()
        for name in (
            (f"{value}.pdf", f"{value}.png") if key == "forest_figure_stem" else (value,)
        )
    ]
    if len(produced) != len(set(produced)):
        raise ValueError("required output filenames must be unique")
    return names


def _jsonl(path: Path) -> list[Mapping[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ExtensionAnalysisError(f"{path}:{line_number} is not a JSON mapping")
            rows.append(value)
    return rows


def _summary_files(cohort_root: Path) -> list[Path]:
    return [
        path
        for path in sorted(cohort_root.glob("**/summary.json"))
        if "rejected" not in path.relative_to(cohort_root).parts
    ]


def _require_equal(
    actual: Any,
    expected: Any,
    *,
    context: str,
    field: str,
) -> None:
    if actual != expected:
        raise ExtensionAnalysisError(
            f"{context}: {field}={actual!r}; expected {expected!r}"
        )


def _require_float_close(
    actual: float,
    expected: float,
    *,
    context: str,
    field: str,
) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ExtensionAnalysisError(
            f"{context}: {field}={actual!r}; expected {expected!r}"
        )


def _validate_device_id(value: Any, cohort: CohortSpec, *, context: str) -> None:
    if cohort.logical_device_id is not None:
        _require_equal(value, cohort.logical_device_id, context=context, field="device_id")
        return
    device_id = str(value or "")
    device_name = str(cohort.physical_device.get("name", ""))
    device_host = str(cohort.physical_device.get("host", ""))
    if device_name not in device_id or (device_host and device_host not in device_id):
        raise ExtensionAnalysisError(
            f"{context}: device_id {device_id!r} is inconsistent with "
            f"registered {device_host}/{device_name}"
        )


def _validate_summary_metadata(
    summary: Mapping[str, Any],
    path: Path,
    cohort: CohortSpec,
) -> tuple[str, int]:
    context = str(path)
    variant = str(summary.get("variant", ""))
    seed = int(summary.get("seed_or_session", -1))
    _require_equal(
        summary.get("experiment_id"),
        cohort.experiment_id,
        context=context,
        field="experiment_id",
    )
    _require_equal(summary.get("task"), cohort.task, context=context, field="task")
    _require_equal(
        str(summary.get("protocol_version", "")),
        cohort.protocol_version,
        context=context,
        field="protocol_version",
    )
    _require_equal(
        summary.get("config_hash"), cohort.config_hash, context=context, field="config_hash"
    )
    _require_equal(summary.get("code_dirty"), False, context=context, field="code_dirty")
    if (variant, seed) not in cohort.expected_run_keys:
        raise ExtensionAnalysisError(
            f"{context}: unregistered variant/seed pair {variant}/seed {seed}"
        )
    _require_equal(
        summary.get("code_commit"),
        cohort.expected_commit(variant, seed),
        context=context,
        field="code_commit",
    )
    _validate_device_id(summary.get("device_id"), cohort, context=context)
    _require_equal(summary.get("status"), "ok", context=context, field="status")
    _require_equal(
        int(summary.get("epochs", -1)),
        cohort.expected_epochs,
        context=context,
        field="epochs",
    )
    _require_equal(
        int(summary.get("updates", -1)),
        cohort.expected_updates,
        context=context,
        field="updates",
    )
    _require_equal(
        int(summary.get("expected_updates", -1)),
        cohort.expected_updates,
        context=context,
        field="expected_updates",
    )
    return variant, seed


def _validate_summary_identity(
    summary: Mapping[str, Any], path: Path, cohort: CohortSpec
) -> tuple[str, int]:
    variant, seed = _validate_summary_metadata(summary, path, cohort)
    for field in ("official_test", "deduplicated_test", "validation_best"):
        if not isinstance(summary.get(field), Mapping):
            raise ExtensionAnalysisError(f"{path}: missing registered endpoint {field}")
    _validate_endpoint_schema(summary, path, cohort)
    return variant, seed


def _finite_float(value: Any, *, context: str) -> float:
    if isinstance(value, bool):
        raise ExtensionAnalysisError(f"{context}: boolean is not numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ExtensionAnalysisError(f"{context}: expected a numeric value") from error
    if not math.isfinite(result):
        raise ExtensionAnalysisError(f"{context}: nonfinite value")
    return result


def _bounded_score(value: Any, *, context: str) -> float:
    result = _finite_float(value, context=context)
    if not 0.0 <= result <= 1.0:
        raise ExtensionAnalysisError(f"{context}: score is outside [0, 1]")
    return result


def _validate_endpoint_partition(
    value: Any,
    *,
    context: str,
    cohort: CohortSpec,
    expected_examples: int,
) -> None:
    partition = _require_mapping(value, name=context)
    examples = partition.get("examples")
    _require_equal(
        examples,
        expected_examples,
        context=context,
        field="examples",
    )
    _finite_float(partition.get("loss"), context=f"{context}.loss")
    _require_equal(
        partition.get("output_nonfinite_count"),
        0,
        context=context,
        field="output_nonfinite_count",
    )
    primary = PRIMARY_METRIC_FIELDS[cohort.task]
    _bounded_score(partition.get(primary), context=f"{context}.{primary}")
    if cohort.task == "isic16_poincare_unet":
        _bounded_score(
            partition.get("mean_foreground_iou"),
            context=f"{context}.mean_foreground_iou",
        )


def _validate_endpoint_schema(
    summary: Mapping[str, Any], path: Path, cohort: CohortSpec
) -> None:
    """Validate endpoint structure only after the joint gate has opened."""

    compute = _finite_float(
        summary.get("compute_seconds"), context=f"{path}.compute_seconds"
    )
    elapsed = _finite_float(
        summary.get("elapsed_seconds"), context=f"{path}.elapsed_seconds"
    )
    if compute <= 0.0 or elapsed <= 0.0 or elapsed < compute:
        raise ExtensionAnalysisError(
            f"{path}: timing values must satisfy 0 < compute_seconds <= elapsed_seconds"
        )
    _bounded_score(
        summary.get("best_development_metric"),
        context=f"{path}.best_development_metric",
    )
    best_update = summary.get("best_update")
    if (
        isinstance(best_update, bool)
        or not isinstance(best_update, int)
        or not 1 <= best_update <= cohort.expected_updates
    ):
        raise ExtensionAnalysisError(f"{path}: best_update is outside the run budget")
    for field in ("official_test", "deduplicated_test"):
        _validate_endpoint_partition(
            summary[field],
            context=f"{path}.{field}",
            cohort=cohort,
            expected_examples=cohort.endpoint_example_counts["fixed_final_update"][field],
        )
    selected = _require_mapping(summary["validation_best"], name=f"{path}.validation_best")
    _require_equal(
        selected.get("update"),
        best_update,
        context=f"{path}.validation_best",
        field="update",
    )
    for field in ("official_test", "deduplicated_test"):
        _validate_endpoint_partition(
            selected.get(field),
            context=f"{path}.validation_best.{field}",
            cohort=cohort,
            expected_examples=cohort.endpoint_example_counts["validation_best"][field],
        )


def _validate_summary_health(
    summary: Mapping[str, Any], path: Path, health: Mapping[str, Any]
) -> None:
    _require_equal(
        summary.get("best_development_metric"),
        health["best_development_metric"],
        context=str(path),
        field="best_development_metric",
    )
    _require_equal(
        summary.get("best_update"),
        health["best_update"],
        context=str(path),
        field="best_update",
    )
    _require_equal(
        summary.get("compute_seconds"),
        health["compute_seconds"],
        context=str(path),
        field="compute_seconds",
    )
    elapsed = _finite_float(
        summary.get("elapsed_seconds"), context=f"{path} elapsed_seconds"
    )
    minimum_elapsed = max(
        float(health["elapsed_seconds"]),
        float(health["final_checkpoint_elapsed_seconds"]),
    )
    if elapsed < minimum_elapsed:
        raise ExtensionAnalysisError(
            f"{path}: elapsed_seconds predates final health/checkpoint evidence"
        )


def _validate_prediction_artifacts(cohort: CohortSpec, health: Mapping[str, Any]) -> None:
    registrations = zip(
        health["prediction_paths"],
        (
            ("fixed_final_update", "official_test"),
            ("fixed_final_update", "deduplicated_test"),
            ("validation_best", "official_test"),
            ("validation_best", "deduplicated_test"),
        ),
        strict=True,
    )
    for path, (endpoint, partition) in registrations:
        expected = cohort.endpoint_example_counts[endpoint][partition]
        try:
            with np.load(path, allow_pickle=False) as archive:
                if set(archive.files) != {"sample_id", "prediction"}:
                    raise ExtensionAnalysisError(
                        f"{path}: prediction archive schema is invalid"
                    )
                sample_ids = archive["sample_id"]
                predictions = archive["prediction"]
        except ExtensionAnalysisError:
            raise
        except BaseException as error:
            raise ExtensionAnalysisError(
                f"{path}: prediction archive cannot be loaded safely"
            ) from error
        if sample_ids.ndim != 1 or len(sample_ids) != expected:
            raise ExtensionAnalysisError(f"{path}: sample_id count is invalid")
        if sample_ids.dtype.kind not in "US":
            raise ExtensionAnalysisError(f"{path}: sample_id values are not strings")
        actual_sample_ids = tuple(str(value) for value in sample_ids.tolist())
        expected_sample_ids = cohort.endpoint_sample_ids[partition]
        if actual_sample_ids != expected_sample_ids:
            raise ExtensionAnalysisError(
                f"{path}: sample_id values or order disagree with the registered split"
            )
        expected_shape = (expected, *cohort.prediction_item_shape)
        if predictions.shape != expected_shape:
            raise ExtensionAnalysisError(
                f"{path}: prediction shape {predictions.shape} does not match "
                f"registered shape {expected_shape}"
            )
        if predictions.dtype.kind not in "iu" or not np.all(np.isfinite(predictions)):
            raise ExtensionAnalysisError(f"{path}: predictions are noninteger or nonfinite")
        maximum = 1 if cohort.task == "isic16_poincare_unet" else 99
        if predictions.size and (
            int(np.min(predictions)) < 0 or int(np.max(predictions)) > maximum
        ):
            raise ExtensionAnalysisError(f"{path}: prediction labels are out of range")


def _tree_array_leaves(value: Any) -> Iterable[np.ndarray]:
    if isinstance(value, Mapping):
        for child in value.values():
            yield from _tree_array_leaves(child)
    elif isinstance(value, tuple | list):
        for child in value:
            yield from _tree_array_leaves(child)
    else:
        yield np.asarray(value)


def _validate_finite_tree(value: Any, *, context: str) -> None:
    leaves = list(_tree_array_leaves(value))
    if not leaves:
        raise ExtensionAnalysisError(f"{context}: tree has no leaves")
    for index, leaf in enumerate(leaves):
        if leaf.dtype.kind not in "biufc" or not np.all(np.isfinite(leaf)):
            raise ExtensionAnalysisError(
                f"{context}: leaf {index} is nonnumeric or nonfinite"
            )


def _validate_matching_tree(
    reference: Any,
    candidate: Any,
    *,
    context: str,
) -> None:
    """Require a finite candidate tree with the reference paths and array specs."""

    reference_leaves, reference_definition = jax.tree_util.tree_flatten_with_path(
        reference
    )
    candidate_leaves, candidate_definition = jax.tree_util.tree_flatten_with_path(
        candidate
    )
    if reference_definition != candidate_definition:
        raise ExtensionAnalysisError(f"{context}: tree structure does not match params")
    for (reference_path, reference_leaf), (candidate_path, candidate_leaf) in zip(
        reference_leaves, candidate_leaves, strict=True
    ):
        reference_name = jax.tree_util.keystr(reference_path)
        candidate_name = jax.tree_util.keystr(candidate_path)
        if reference_name != candidate_name:
            raise ExtensionAnalysisError(f"{context}: tree paths do not match params")
        reference_array = np.asarray(reference_leaf)
        candidate_array = np.asarray(candidate_leaf)
        if candidate_array.shape != reference_array.shape:
            raise ExtensionAnalysisError(
                f"{context}{candidate_name}: shape {candidate_array.shape} does not "
                f"match params shape {reference_array.shape}"
            )
        if candidate_array.dtype != reference_array.dtype:
            raise ExtensionAnalysisError(
                f"{context}{candidate_name}: dtype {candidate_array.dtype} does not "
                f"match params dtype {reference_array.dtype}"
            )
        if candidate_array.dtype.kind not in "biufc" or not np.all(
            np.isfinite(candidate_array)
        ):
            raise ExtensionAnalysisError(
                f"{context}{candidate_name}: leaf is nonnumeric or nonfinite"
            )


def _tree_signature(value: Any) -> tuple[Any, tuple[tuple[str, tuple[int, ...], str], ...]]:
    """Return exact pytree and array-spec metadata for cross-checkpoint identity."""

    leaves, definition = jax.tree_util.tree_flatten_with_path(value)
    return (
        definition,
        tuple(
            (
                jax.tree_util.keystr(path),
                tuple(int(dimension) for dimension in np.asarray(leaf).shape),
                np.asarray(leaf).dtype.str,
            )
            for path, leaf in leaves
        ),
    )


def _optimizer_components(state: Any, *, context: str) -> tuple[Any, Any, Any]:
    if isinstance(state, Mapping):
        fields = tuple(
            state.get(field) for field in ("step", "first_moment", "second_moment")
        )
    else:
        fields = tuple(
            getattr(state, field, None)
            for field in ("step", "first_moment", "second_moment")
        )
    if any(value is None for value in fields):
        raise ExtensionAnalysisError(f"{context}: incomplete Adam state")
    return fields


def _validate_checkpoint(
    path: Path,
    cohort: CohortSpec,
    *,
    variant: str,
    seed: int,
    expected_epoch: int,
    expected_update: int,
    expected_best: tuple[float, int] | None = None,
) -> dict[str, Any]:
    context = str(path)
    try:
        payload = load_checkpoint(path)
    except Exception as error:
        raise ExtensionAnalysisError(f"{context}: checkpoint cannot be loaded") from error
    for field, expected in (
        ("format", "poincare-exec-checkpoint-v1"),
        ("config_hash", cohort.config_hash),
        ("protocol_version", cohort.protocol_version),
        ("code_commit", cohort.expected_commit(variant, seed)),
        ("code_dirty", False),
        ("task", cohort.task),
        ("variant", variant),
        ("seed", seed),
        ("epoch", expected_epoch),
        ("update", expected_update),
    ):
        _require_equal(payload.get(field), expected, context=context, field=field)
    compute = _finite_float(
        payload.get("compute_seconds"), context=f"{context} compute_seconds"
    )
    elapsed = _finite_float(
        payload.get("elapsed_seconds"), context=f"{context} elapsed_seconds"
    )
    if min(compute, elapsed) < 0.0 or elapsed < compute:
        raise ExtensionAnalysisError(f"{context}: invalid checkpoint timing")
    optimizer = payload.get("optimizer_state")
    step, first, second = _optimizer_components(optimizer, context=context)
    step_array = np.asarray(step)
    if step_array.size != 1 or int(step_array.reshape(())) != expected_update:
        raise ExtensionAnalysisError(f"{context}: Adam step disagrees with update")
    params = payload.get("params")
    _validate_finite_tree(params, context=f"{context} params")
    _validate_matching_tree(params, first, context=f"{context} Adam first moment")
    _validate_matching_tree(params, second, context=f"{context} Adam second moment")
    try:
        best_update = int(payload.get("best_update", -2))
        best_metric = float(payload.get("best_metric", math.nan))
    except (TypeError, ValueError, OverflowError) as error:
        raise ExtensionAnalysisError(
            f"{context}: invalid best metric/update metadata"
        ) from error
    if expected_best is not None:
        _require_equal(
            (best_metric, best_update),
            expected_best,
            context=context,
            field="best metric/update",
        )
    elif best_update == -1:
        if best_metric != -math.inf:
            raise ExtensionAnalysisError(
                f"{context}: unset best update must use -infinity metric"
            )
    elif not (
        1 <= best_update <= expected_update
        and best_update % (cohort.expected_updates // cohort.expected_epochs) == 0
        and math.isfinite(best_metric)
        and 0.0 <= best_metric <= 1.0
    ):
        raise ExtensionAnalysisError(f"{context}: invalid best metric/update metadata")
    return {
        "compute_seconds": compute,
        "elapsed_seconds": elapsed,
        "best_metric": best_metric,
        "best_update": best_update,
        "params_signature": _tree_signature(params),
    }


def _validate_rejected_initial_checkpoint(
    path: Path, *, identity: Mapping[str, Any]
) -> None:
    context = str(path)
    try:
        payload = load_checkpoint(path)
    except Exception as error:
        raise ExtensionAnalysisError(f"{context}: checkpoint cannot be loaded") from error
    for field, expected in (
        ("format", "poincare-exec-checkpoint-v1"),
        ("config_hash", identity["config_hash"]),
        ("protocol_version", identity["protocol_version"]),
        ("code_commit", identity["code_commit"]),
        ("code_dirty", identity["code_dirty"]),
        ("task", identity["task"]),
        ("variant", identity["variant"]),
        ("seed", identity["seed"]),
        ("epoch", 0),
        ("update", 0),
        ("compute_seconds", 0.0),
        ("best_metric", -math.inf),
        ("best_update", -1),
    ):
        _require_equal(payload.get(field), expected, context=context, field=field)
    elapsed = _finite_float(
        payload.get("elapsed_seconds"), context=f"{context} elapsed_seconds"
    )
    if elapsed < 0.0:
        raise ExtensionAnalysisError(f"{context}: initial elapsed time is negative")
    step, first, second = _optimizer_components(
        payload.get("optimizer_state"), context=context
    )
    if np.asarray(step).size != 1 or int(np.asarray(step).reshape(())) != 0:
        raise ExtensionAnalysisError(f"{context}: initial Adam step must be zero")
    params = payload.get("params")
    _validate_finite_tree(params, context=f"{context} params")
    _validate_matching_tree(params, first, context=f"{context} Adam first moment")
    _validate_matching_tree(params, second, context=f"{context} Adam second moment")


def _validate_run_health(
    run_dir: Path, cohort: CohortSpec, *, variant: str, seed: int
) -> dict[str, Any]:
    """Check permitted training/development health evidence without test endpoints."""

    required_checkpoints = (
        "checkpoints/initial.pkl",
        "checkpoints/10_percent.pkl",
        "checkpoints/50_percent.pkl",
        "checkpoints/validation_best.pkl",
        "checkpoints/final.pkl",
        "resume.pkl",
    )
    missing = [
        relative
        for relative in required_checkpoints
        if not (run_dir / relative).is_file() or (run_dir / relative).stat().st_size == 0
    ]
    if missing:
        raise ExtensionAnalysisError(
            f"{run_dir}: missing or empty registered checkpoints: {missing}"
        )
    prediction_paths = tuple(
        run_dir / relative
        for relative in (
            "predictions_official_test.npz",
            "predictions_deduplicated_test.npz",
            "predictions_validation_best_official_test.npz",
            "predictions_validation_best_deduplicated_test.npz",
        )
    )
    for path in prediction_paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise ExtensionAnalysisError(
                f"{run_dir}: missing endpoint artifact {path.name}"
            )

    metrics_path = run_dir / "training_metrics.jsonl"
    rows = _jsonl(metrics_path)
    if not rows:
        raise ExtensionAnalysisError(
            f"{run_dir}: training_metrics.jsonl is missing or empty"
        )
    training_updates: list[int] = []
    training_epochs: list[int] = []
    development_updates: list[int] = []
    development_epochs: list[int] = []
    compute_history: list[float] = []
    elapsed_history: list[float] = []
    final_training_update_by_epoch: dict[int, int] = {}
    compute_by_update: dict[int, float] = {}
    step_by_update: dict[int, float] = {}
    elapsed_by_update: dict[int, float] = {}
    development_scores: list[tuple[float, int]] = []
    development_compute_records: list[tuple[str, int, float]] = []
    observed_schedule: list[tuple[str, int, int]] = []
    training_examples_by_epoch: dict[int, list[int]] = {
        epoch: [] for epoch in range(1, cohort.expected_epochs + 1)
    }
    for index, row in enumerate(rows, start=1):
        context = f"{metrics_path}:{index}"
        for field, expected in (
            ("experiment_id", cohort.experiment_id),
            ("task", cohort.task),
            ("protocol_version", cohort.protocol_version),
            ("config_hash", cohort.config_hash),
            ("code_commit", cohort.expected_commit(variant, seed)),
            ("code_dirty", False),
            ("variant", variant),
            ("seed_or_session", seed),
        ):
            _require_equal(row.get(field), expected, context=context, field=field)
        _validate_device_id(row.get("device_id"), cohort, context=context)
        _require_equal(row.get("status"), "ok", context=context, field="status")
        split = row.get("split")
        update = int(row.get("update", -1))
        epoch = int(row.get("epoch", -1))
        if not 1 <= epoch <= cohort.expected_epochs:
            raise ExtensionAnalysisError(f"{context}: epoch {epoch} is out of range")
        if not 1 <= update <= cohort.expected_updates:
            raise ExtensionAnalysisError(f"{context}: update {update} is out of range")
        examples = row.get("examples")
        if isinstance(examples, bool) or not isinstance(examples, int) or examples < 1:
            raise ExtensionAnalysisError(f"{context}: examples must be a positive integer")
        _finite_float(row.get("loss"), context=f"{context} loss")
        _finite_float(row.get("curvature"), context=f"{context} curvature")
        compute = _finite_float(
            row.get("compute_seconds"), context=f"{context} compute_seconds"
        )
        elapsed = _finite_float(
            row.get("elapsed_seconds"), context=f"{context} elapsed_seconds"
        )
        if compute < 0.0 or elapsed < 0.0 or elapsed < compute:
            raise ExtensionAnalysisError(
                f"{context}: invalid cumulative compute/elapsed timing"
            )
        compute_history.append(compute)
        elapsed_history.append(elapsed)
        if split == "train":
            _finite_float(row.get("gradient_norm"), context=f"{context} gradient_norm")
            _finite_float(
                row.get("parameter_update_norm"),
                context=f"{context} parameter_update_norm",
            )
            step = _finite_float(row.get("step_seconds"), context=f"{context} step_seconds")
            if step <= 0.0:
                raise ExtensionAnalysisError(
                    f"{context}: training step_seconds must be positive"
                )
            _require_equal(
                int(row.get("parameter_nonfinite_count", -1)),
                0,
                context=context,
                field="parameter_nonfinite_count",
            )
            _require_equal(
                int(row.get("optimizer_nonfinite_count", -1)),
                0,
                context=context,
                field="optimizer_nonfinite_count",
            )
            training_updates.append(update)
            training_epochs.append(epoch)
            training_examples_by_epoch[epoch].append(examples)
            observed_schedule.append(("train", epoch, update))
            final_training_update_by_epoch[epoch] = update
            compute_by_update[update] = compute
            step_by_update[update] = step
            elapsed_by_update[update] = elapsed
        elif split == "development":
            metric_field = PRIMARY_METRIC_FIELDS[cohort.task]
            _bounded_score(row.get(metric_field), context=f"{context} {metric_field}")
            _require_equal(
                row.get("step_seconds"),
                None,
                context=context,
                field="step_seconds",
            )
            for field in ("gradient_norm", "parameter_update_norm"):
                _require_equal(
                    row.get(field),
                    None,
                    context=context,
                    field=field,
                )
            _require_equal(
                examples,
                cohort.training_health["development_examples"],
                context=context,
                field="examples",
            )
            _require_equal(
                int(row.get("output_nonfinite_count", -1)),
                0,
                context=context,
                field="output_nonfinite_count",
            )
            development_epochs.append(epoch)
            development_updates.append(update)
            development_scores.append((float(row[metric_field]), update))
            development_compute_records.append((context, update, compute))
            observed_schedule.append(("development", epoch, update))
        else:
            raise ExtensionAnalysisError(f"{context}: unregistered split {split!r}")
    _require_equal(
        training_updates,
        list(range(1, cohort.expected_updates + 1)),
        context=str(metrics_path),
        field="training update history",
    )
    rows_per_epoch = cohort.training_health["total_train_rows_per_epoch"]
    expected_schedule: list[tuple[str, int, int]] = []
    for epoch in range(1, cohort.expected_epochs + 1):
        first_update = (epoch - 1) * rows_per_epoch + 1
        final_update = epoch * rows_per_epoch
        expected_schedule.extend(
            ("train", epoch, update) for update in range(first_update, final_update + 1)
        )
        expected_schedule.append(("development", epoch, final_update))
        expected_examples = [
            cohort.training_health["examples_per_full_train_row"]
        ] * cohort.training_health["full_train_rows_per_epoch"] + [
            cohort.training_health["examples_per_final_train_row"]
        ] * cohort.training_health["final_train_rows_per_epoch"]
        _require_equal(
            training_examples_by_epoch[epoch],
            expected_examples,
            context=str(metrics_path),
            field=f"epoch {epoch} training batch cardinalities",
        )
    _require_equal(
        observed_schedule,
        expected_schedule,
        context=str(metrics_path),
        field="training/development row schedule",
    )
    _require_equal(
        development_epochs,
        list(range(1, cohort.expected_epochs + 1)),
        context=str(metrics_path),
        field="development epoch history",
    )
    if training_epochs != sorted(training_epochs):
        raise ExtensionAnalysisError(f"{metrics_path}: training epochs are not ordered")
    _require_equal(
        sorted(set(training_epochs)),
        list(range(1, cohort.expected_epochs + 1)),
        context=str(metrics_path),
        field="training epoch coverage",
    )
    if development_updates != sorted(set(development_updates)):
        raise ExtensionAnalysisError(
            f"{metrics_path}: development updates are duplicated or unordered"
        )
    _require_equal(
        development_updates[-1],
        cohort.expected_updates,
        context=str(metrics_path),
        field="final development update",
    )
    _require_equal(
        development_updates,
        [final_training_update_by_epoch[epoch] for epoch in development_epochs],
        context=str(metrics_path),
        field="development update schedule",
    )
    accumulated_step_seconds = 0.0
    for update in range(1, cohort.expected_updates + 1):
        accumulated_step_seconds += step_by_update[update]
        _require_float_close(
            compute_by_update[update],
            accumulated_step_seconds,
            context=f"{metrics_path}: training update {update}",
            field="compute_seconds recurrence from step_seconds",
        )
    for context, update, compute in development_compute_records:
        _require_float_close(
            compute,
            compute_by_update[update],
            context=context,
            field="development compute_seconds",
        )
    if compute_history != sorted(compute_history):
        raise ExtensionAnalysisError(
            f"{metrics_path}: compute_seconds history is not monotone"
        )
    if elapsed_history != sorted(elapsed_history):
        raise ExtensionAnalysisError(
            f"{metrics_path}: elapsed_seconds history is not monotone"
        )
    best_metric = -math.inf
    best_update = -1
    for metric, update in development_scores:
        if metric > best_metric:
            best_metric = metric
            best_update = update

    def best_before(update_limit: int) -> tuple[float, int]:
        selected_metric = -math.inf
        selected_update = -1
        for metric, update in development_scores:
            if update < update_limit and metric > selected_metric:
                selected_metric = metric
                selected_update = update
        return selected_metric, selected_update

    updates_per_epoch = cohort.expected_updates // cohort.expected_epochs
    if updates_per_epoch * cohort.expected_epochs != cohort.expected_updates:
        raise ExtensionAnalysisError(
            f"{cohort.experiment_id}: update budget is not epoch-divisible"
        )
    checkpoint_plan = (
        ("checkpoints/initial.pkl", 0, 0, (-math.inf, -1)),
        (
            "checkpoints/10_percent.pkl",
            math.ceil(cohort.expected_updates * 0.10),
            math.ceil(math.ceil(cohort.expected_updates * 0.10) / updates_per_epoch),
            best_before(math.ceil(cohort.expected_updates * 0.10)),
        ),
        (
            "checkpoints/50_percent.pkl",
            math.ceil(cohort.expected_updates * 0.50),
            math.ceil(math.ceil(cohort.expected_updates * 0.50) / updates_per_epoch),
            best_before(math.ceil(cohort.expected_updates * 0.50)),
        ),
        (
            "checkpoints/validation_best.pkl",
            best_update,
            best_update // updates_per_epoch,
            (best_metric, best_update),
        ),
        (
            "checkpoints/final.pkl",
            cohort.expected_updates,
            cohort.expected_epochs,
            (best_metric, best_update),
        ),
        (
            "resume.pkl",
            cohort.expected_updates,
            cohort.expected_epochs,
            (best_metric, best_update),
        ),
    )
    checkpoint_health: dict[str, dict[str, Any]] = {}
    params_signature: Any = None
    for relative, update, epoch, expected_best in checkpoint_plan:
        metadata = _validate_checkpoint(
            run_dir / relative,
            cohort,
            variant=variant,
            seed=seed,
            expected_epoch=epoch,
            expected_update=update,
            expected_best=expected_best,
        )
        expected_compute = 0.0 if update == 0 else compute_by_update[update]
        _require_equal(
            metadata["compute_seconds"],
            expected_compute,
            context=str(run_dir / relative),
            field="compute_seconds",
        )
        if update > 0:
            if metadata["elapsed_seconds"] < elapsed_by_update[update]:
                raise ExtensionAnalysisError(
                    f"{run_dir / relative}: elapsed time predates its training row"
                )
        if params_signature is None:
            params_signature = metadata["params_signature"]
        else:
            _require_equal(
                metadata["params_signature"],
                params_signature,
                context=str(run_dir / relative),
                field="parameter tree signature across checkpoints",
            )
        checkpoint_health[relative] = metadata
    return {
        "best_development_metric": best_metric,
        "best_update": best_update,
        "compute_seconds": compute_by_update[cohort.expected_updates],
        "elapsed_seconds": elapsed_history[-1],
        "final_checkpoint_elapsed_seconds": max(
            checkpoint_health["checkpoints/final.pkl"]["elapsed_seconds"],
            checkpoint_health["resume.pkl"]["elapsed_seconds"],
        ),
        "prediction_paths": prediction_paths,
    }


def _validate_snapshot(
    snapshot: Any,
    *,
    cohort: CohortSpec,
    context: str,
    require_idle: bool,
) -> Mapping[str, Any]:
    value = _require_mapping(snapshot, name=f"{context} snapshot")
    _require_equal(value.get("status"), "ok", context=context, field="snapshot.status")
    _require_equal(
        int(value.get("index", -1)),
        int(cohort.physical_device["physical_index"]),
        context=context,
        field="snapshot.index",
    )
    _require_equal(
        value.get("uuid"),
        cohort.physical_device["uuid"],
        context=context,
        field="snapshot.uuid",
    )
    _require_equal(
        value.get("name"),
        cohort.physical_device["name"],
        context=context,
        field="snapshot.name",
    )
    _require_equal(
        value.get("process_query_status"),
        "ok",
        context=context,
        field="snapshot.process_query_status",
    )
    processes = value.get("compute_processes")
    if not isinstance(processes, list):
        raise ExtensionAnalysisError(f"{context}: compute_processes must be a list")
    if require_idle and processes:
        raise ExtensionAnalysisError(
            f"{context}: registered run has competing compute processes: {processes!r}"
        )
    if require_idle:
        memory = int(value.get("memory.used", -1))
        utilization = int(value.get("utilization.gpu", -1))
        if memory < 0 or memory > 512:
            raise ExtensionAnalysisError(
                f"{context}: pre-worker memory is not idle: {memory} MiB"
            )
        if utilization < 0 or utilization > 5:
            raise ExtensionAnalysisError(
                f"{context}: pre-worker utilization is not idle: {utilization}%"
            )
    return value


def _validate_campaign_attempts(
    cohort_root: Path,
    cohort: CohortSpec,
    registered_rejections: Sequence[Mapping[str, Any]],
    post_worker_dispositions: Mapping[tuple[str, str, int], dict[str, Any]],
) -> dict[tuple[str, int], dict[str, Any]]:
    """Audit every accepted-root attempt and return the accepted rows.

    A failed row is permitted only when it is also represented by an exactly
    registered supplemental-evidence row.  This supports a registered retry
    without silently accepting an unexplained duplicate attempt.
    """

    paths = sorted(cohort_root.glob("**/campaign_attempts.jsonl"))
    if not paths:
        raise ExtensionAnalysisError(
            f"{cohort.experiment_id}: campaign_attempts.jsonl is required"
        )
    accepted: dict[tuple[str, int], dict[str, Any]] = {}
    matched_rejections: set[str] = set()
    for path in paths:
        path_sha256 = _sha256_file(path)
        for row_number, attempt in enumerate(_jsonl(path), start=1):
            variant = str(attempt.get("variant", ""))
            seed = int(attempt.get("seed", -1))
            key = (variant, seed)
            context = (
                f"{cohort.experiment_id} campaign attempt {variant}/seed {seed} "
                f"at {path}:{row_number}"
            )
            if attempt.get("status") == "failed":
                if key in accepted:
                    raise ExtensionAnalysisError(
                        f"{context}: rejected attempt occurs after accepted success"
                    )
                matches = [
                    row
                    for row in registered_rejections
                    if all(
                        attempt.get(field) == row.get(field)
                        for field in (
                            "experiment_id",
                            "task",
                            "variant",
                            "protocol_version",
                            "config_hash",
                            "code_commit",
                            "code_dirty",
                            "status",
                            "returncode",
                        )
                    )
                    and int(attempt.get("seed", -1)) == int(row.get("seed", -2))
                ]
                if len(matches) != 1:
                    raise ExtensionAnalysisError(
                        f"{context}: status 'failed' attempt is not exactly registered as "
                        "supplemental evidence"
                    )
                evidence_id = str(matches[0]["evidence_id"])
                if evidence_id in matched_rejections:
                    raise ExtensionAnalysisError(
                        f"{context}: duplicate occurrence of supplemental {evidence_id}"
                    )
                matched_rejections.add(evidence_id)
                registered = matches[0]
                registered["accepted_root_attempt_path"] = str(path)
                registered["accepted_root_attempt_row"] = row_number
                supporting_files = json.loads(str(registered["supporting_files"]))
                supporting_sha256 = json.loads(str(registered["supporting_sha256"]))
                supporting_files["accepted_root_campaign_attempts"] = str(path)
                supporting_sha256["accepted_root_campaign_attempts"] = path_sha256
                registered["supporting_files"] = canonical_json(supporting_files)
                registered["supporting_sha256"] = canonical_json(supporting_sha256)
                _require_equal(
                    attempt.get("summary_path"),
                    None,
                    context=context,
                    field="summary_path",
                )
                if str(registered.get("error", "")) not in str(attempt.get("stderr", "")):
                    raise ExtensionAnalysisError(
                        f"{context}: stderr does not contain the registered failure"
                    )
                _validate_snapshot(
                    attempt.get("gpu_snapshot_before"),
                    cohort=cohort,
                    context=f"{context} before",
                    require_idle=True,
                )
                _validate_snapshot(
                    attempt.get("gpu_snapshot_after"),
                    cohort=cohort,
                    context=f"{context} after",
                    require_idle=True,
                )
                continue

            if attempt.get("status") != "ok":
                raise ExtensionAnalysisError(
                    f"{context}: unregistered terminal status {attempt.get('status')!r}"
                )
            if key not in cohort.expected_run_keys:
                raise ExtensionAnalysisError(f"{context}: unregistered attempt")
            if key in accepted:
                raise ExtensionAnalysisError(f"{context}: duplicate successful attempt")
            accepted[key] = {
                "attempt_path": str(path),
                "attempt_row": row_number,
                "attempt_sha256": path_sha256,
            }
            _require_equal(
                attempt.get("experiment_id"),
                cohort.experiment_id,
                context=context,
                field="experiment_id",
            )
            _require_equal(attempt.get("task"), cohort.task, context=context, field="task")
            _require_equal(
                str(attempt.get("protocol_version", "")),
                cohort.protocol_version,
                context=context,
                field="protocol_version",
            )
            _require_equal(
                attempt.get("config_hash"),
                cohort.config_hash,
                context=context,
                field="config_hash",
            )
            _require_equal(
                attempt.get("code_commit"),
                cohort.expected_commit(variant, seed),
                context=context,
                field="code_commit",
            )
            _require_equal(
                attempt.get("code_dirty"), False, context=context, field="code_dirty"
            )
            _require_equal(
                int(attempt.get("returncode", -1)),
                0,
                context=context,
                field="returncode",
            )
            summary_path = Path(str(attempt.get("summary_path", "")))
            expected_tail = (variant, f"seed_{seed}", "summary.json")
            _require_equal(
                summary_path.parts[-3:],
                expected_tail,
                context=context,
                field="summary_path suffix",
            )
            _validate_snapshot(
                attempt.get("gpu_snapshot_before"),
                cohort=cohort,
                context=f"{context} before",
                require_idle=True,
            )
            after_snapshot = _validate_snapshot(
                attempt.get("gpu_snapshot_after"),
                cohort=cohort,
                context=f"{context} after",
                require_idle=False,
            )
            disposition_key = (cohort.experiment_id, variant, seed)
            disposition = post_worker_dispositions.get(disposition_key)
            if after_snapshot["compute_processes"]:
                if disposition is None:
                    raise ExtensionAnalysisError(
                        f"{context}: non-idle post-worker snapshot has no exact "
                        "pre-unblinding disposition"
                    )
                expected_attempt = _require_mapping(
                    disposition["expected_attempt"],
                    name=f"{context}.expected_attempt",
                )
                _require_equal(
                    _canonical_line_sha256(attempt),
                    expected_attempt.get("canonical_row_sha256"),
                    context=context,
                    field="canonical_row_sha256",
                )
                _require_equal(
                    _canonical_line_sha256(after_snapshot),
                    expected_attempt.get("after_snapshot_sha256"),
                    context=context,
                    field="after_snapshot_sha256",
                )
                _require_equal(
                    attempt.get("timestamp_utc"),
                    expected_attempt.get("timestamp_utc"),
                    context=context,
                    field="timestamp_utc",
                )
                summary_stat = _require_mapping(
                    disposition["summary_stat"], name=f"{context}.summary_stat"
                )
                registered_relative = Path(str(summary_stat.get("relative_path", "")))
                expected_relative = Path(variant) / f"seed_{seed}" / "summary.json"
                _require_equal(
                    registered_relative,
                    expected_relative,
                    context=context,
                    field="summary_stat.relative_path",
                )
                registered_summary = (cohort_root / registered_relative).resolve()
                if not registered_summary.is_relative_to(cohort_root.resolve()):
                    raise ExtensionAnalysisError(
                        f"{context}: summary stat path escapes the cohort root"
                    )
                stat = registered_summary.stat()
                _require_equal(
                    stat.st_size,
                    int(summary_stat.get("size", -1)),
                    context=context,
                    field="summary_stat.size",
                )
                _require_equal(
                    stat.st_mtime_ns,
                    int(summary_stat.get("mtime_ns", -1)),
                    context=context,
                    field="summary_stat.mtime_ns",
                )
                summary_time = datetime.fromtimestamp(stat.st_mtime_ns / 1e9, tz=UTC)
                if summary_time >= disposition["first_foreign_time"]:
                    raise ExtensionAnalysisError(
                        f"{context}: summary was not finalized before foreign-only "
                        "post-worker telemetry"
                    )
                disposition["used"] = True
                accepted[key]["post_worker_snapshot_disposition"] = (
                    "registered_post_worker_occupancy"
                )
                accepted[key]["post_worker_evidence_id"] = disposition["evidence_id"]
                accepted[key]["post_worker_supporting_files"] = disposition[
                    "supporting_files"
                ]
                accepted[key]["post_worker_supporting_sha256"] = disposition[
                    "supporting_sha256"
                ]
            else:
                accepted[key]["post_worker_snapshot_disposition"] = (
                    "idle_no_compute_process"
                )
                accepted[key]["post_worker_evidence_id"] = None
    if set(accepted) != cohort.expected_run_keys:
        missing = sorted(cohort.expected_run_keys - set(accepted))
        raise ExtensionAnalysisError(
            f"{cohort.experiment_id}: missing campaign attempts: {missing}"
        )
    return accepted


def _expected_summary_paths(
    cohort_root: Path, cohort: CohortSpec
) -> dict[tuple[str, int], Path]:
    """Resolve registered paths without opening endpoint-bearing summaries."""

    expected = {
        (variant, seed): cohort_root / variant / f"seed_{seed}" / "summary.json"
        for variant, seed in cohort.expected_run_keys
    }
    expected_paths = set(expected.values())
    actual_paths = set(_summary_files(cohort_root))
    missing = sorted(str(path) for path in expected_paths - actual_paths)
    extra = sorted(str(path) for path in actual_paths - expected_paths)
    if missing or extra:
        raise ExtensionAnalysisError(
            f"{cohort.experiment_id}: incomplete run matrix; "
            f"missing={missing}, extra={extra}"
        )
    return expected


def _validate_global_embargo_registered(
    manifest: Mapping[str, Any],
    cohort_roots: Mapping[str, Path],
    *,
    repository_root: Path,
    runtime_evidence_roots: Mapping[str, Path] | None = None,
    supplemental_evidence_roots: Mapping[str, Path] | None = None,
    post_worker_evidence_roots: Mapping[str, Path] | None = None,
) -> tuple[
    dict[str, CohortSpec],
    dict[str, dict[tuple[str, int], ValidatedRun]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, dict[tuple[str, int], dict[str, Any]]],
]:
    """Validate all registered runs before any endpoint value is extracted."""

    _validate_document(
        manifest.get("protocol_document"),
        name="protocol_document",
        repository_root=repository_root,
    )
    _validate_document(
        manifest.get("contract_document"),
        name="contract_document",
        repository_root=repository_root,
    )
    _validate_manifest_contract(manifest)
    specs = _cohort_specs(manifest, repository_root=repository_root)
    runtime_audit = _validate_runtime_evidence(
        manifest, runtime_evidence_roots or {}, specs
    )
    supplemental_audit = _validate_supplemental_evidence(
        manifest, supplemental_evidence_roots or {}, specs
    )
    post_worker_dispositions = _validate_post_worker_evidence(
        manifest, post_worker_evidence_roots or {}, specs
    )
    extension_ids = _registered_extension_ids(manifest)
    missing_specs = sorted(set(extension_ids) - set(specs))
    if missing_specs:
        raise ValueError(f"global embargo references undefined cohorts: {missing_specs}")
    extra_roots = sorted(set(cohort_roots) - set(specs))
    if extra_roots:
        raise ExtensionAnalysisError(f"unregistered cohort roots: {extra_roots}")
    missing_roots = sorted(set(specs) - set(cohort_roots))
    if missing_roots:
        raise ExtensionAnalysisError(f"missing registered cohort roots: {missing_roots}")

    summary_paths: dict[str, dict[tuple[str, int], Path]] = {}
    health_records: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    accepted_attempts: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    for experiment_id in sorted(specs):
        cohort = specs[experiment_id]
        root = Path(cohort_roots[experiment_id]).expanduser().resolve()
        if not root.is_dir():
            raise ExtensionAnalysisError(f"{experiment_id}: cohort root is missing: {root}")
        expected_paths = _expected_summary_paths(root, cohort)
        if experiment_id in extension_ids:
            attempts = _validate_campaign_attempts(
                root, cohort, supplemental_audit, post_worker_dispositions
            )
            if set(attempts) != cohort.expected_run_keys:
                missing = sorted(cohort.expected_run_keys - set(attempts))
                raise ExtensionAnalysisError(
                    f"{experiment_id}: missing accepted campaign attempts: {missing}"
                )
            accepted_attempts[experiment_id] = attempts
        cohort_health: dict[tuple[str, int], dict[str, Any]] = {}
        for expected_key, path in expected_paths.items():
            variant, seed = expected_key
            cohort_health[expected_key] = _validate_run_health(
                path.parent, cohort, variant=variant, seed=seed
            )
        summary_paths[experiment_id] = expected_paths
        health_records[experiment_id] = cohort_health

    unused_dispositions = sorted(
        key
        for key, disposition in post_worker_dispositions.items()
        if not disposition["used"]
    )
    if unused_dispositions:
        raise ExtensionAnalysisError(
            "registered post-worker dispositions were not matched exactly: "
            f"{unused_dispositions}"
        )

    embargo = _require_mapping(manifest.get("global_embargo"), name="global_embargo")
    expected_runs = int(embargo.get("required_successful_runs", -1))
    actual_runs = sum(len(summary_paths[experiment_id]) for experiment_id in extension_ids)
    if actual_runs != expected_runs:
        raise ExtensionAnalysisError(
            f"global endpoint embargo remains closed: {actual_runs} validated runs; "
            f"expected {expected_runs}"
        )
    expected_pairs = int(embargo.get("required_complete_pairs", -1))
    actual_pairs = sum(len(specs[experiment_id].seeds) for experiment_id in extension_ids)
    if actual_pairs != expected_pairs:
        raise ExtensionAnalysisError(
            "global endpoint embargo pair count mismatch: "
            f"{actual_pairs}; expected {expected_pairs}"
        )

    # The joint gate is now open. Only from this point may the endpoint-bearing
    # E20/E21 summaries be opened and their official-test values extracted.
    validated: dict[str, dict[tuple[str, int], ValidatedRun]] = {}
    for experiment_id in sorted(specs):
        cohort = specs[experiment_id]
        root = Path(cohort_roots[experiment_id]).expanduser().resolve()
        if not root.is_dir():
            raise ExtensionAnalysisError(f"{experiment_id}: cohort root is missing: {root}")
        selected: dict[tuple[str, int], ValidatedRun] = {}
        paths = list(summary_paths[experiment_id].values())
        for path in sorted(paths):
            value = json.loads(path.read_text(encoding="utf-8"))
            summary = _require_mapping(value, name=str(path))
            variant, seed = _validate_summary_identity(summary, path, cohort)
            key = (variant, seed)
            if key in selected:
                raise ExtensionAnalysisError(
                    f"{experiment_id}: duplicate summary for {variant}/seed {seed}"
                )
            health = health_records[experiment_id][key]
            _validate_summary_health(summary, path, health)
            _validate_prediction_artifacts(cohort, health)
            selected[key] = ValidatedRun(cohort, variant, seed, path, summary)
        if set(selected) != cohort.expected_run_keys:
            missing = sorted(cohort.expected_run_keys - set(selected))
            extra = sorted(set(selected) - cohort.expected_run_keys)
            raise ExtensionAnalysisError(
                f"{experiment_id}: incomplete run matrix; missing={missing}, extra={extra}"
            )
        validated[experiment_id] = selected
    return specs, validated, runtime_audit, supplemental_audit, accepted_attempts


def _analyze_extension_campaign_registered(
    manifest: Mapping[str, Any],
    cohort_roots: Mapping[str, Path],
    *,
    repository_root: Path,
    runtime_evidence_roots: Mapping[str, Path] | None = None,
    supplemental_evidence_roots: Mapping[str, Path] | None = None,
    post_worker_evidence_roots: Mapping[str, Path] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Return all E20/E21 outputs after opening the global endpoint embargo."""

    (
        _,
        validated,
        runtime_audit,
        supplemental_audit,
        accepted_attempts,
    ) = _validate_global_embargo_registered(
        manifest,
        cohort_roots,
        repository_root=repository_root,
        runtime_evidence_roots=runtime_evidence_roots,
        supplemental_evidence_roots=supplemental_evidence_roots,
        post_worker_evidence_roots=post_worker_evidence_roots,
    )
    extension_ids = frozenset(_registered_extension_ids(manifest))
    quality_by_seed, quality_summary = quality_rows(manifest, validated)
    efficiency_by_seed, efficiency_summary = efficiency_rows(manifest, validated)
    endpoint_table = endpoint_rows(validated, extension_ids)
    variant_summary = variant_rows(endpoint_table)
    audit = []
    inventory = []
    for experiment_id, run_map in sorted(validated.items()):
        if experiment_id not in extension_ids:
            continue
        cohort = next(iter(run_map.values())).cohort
        audit.append(
            {
                **common_metadata(cohort),
                "status": "complete",
                "expected_runs": len(cohort.expected_run_keys),
                "validated_runs": len(run_map),
                "expected_pairs": len(cohort.seeds),
                "validated_pairs": len(cohort.seeds),
                "official_endpoint_embargo": "opened_after_global_validation",
                "grand_pool_performed": False,
            }
        )
        for (variant, seed), run in sorted(run_map.items()):
            attempt = accepted_attempts[experiment_id][(variant, seed)]
            metrics_path = run.summary_path.parent / "training_metrics.jsonl"
            supporting_files = {
                "campaign_attempts": attempt["attempt_path"],
                "summary": str(run.summary_path),
                "training_metrics": str(metrics_path),
                **{
                    f"post_worker_{role}": path
                    for role, path in attempt.get(
                        "post_worker_supporting_files", {}
                    ).items()
                },
            }
            supporting_sha256 = {
                "campaign_attempts": attempt["attempt_sha256"],
                "summary": _sha256_file(run.summary_path),
                "training_metrics": _sha256_file(metrics_path),
                **{
                    f"post_worker_{role}": digest
                    for role, digest in attempt.get(
                        "post_worker_supporting_sha256", {}
                    ).items()
                },
            }
            inventory.append(
                {
                    "analysis_id": cohort.analysis_id,
                    "evidence_id": None,
                    "record_type": "accepted_registered_training_run",
                    "experiment_id": experiment_id,
                    "task": cohort.task,
                    "dataset": cohort.dataset,
                    "variant": variant,
                    "seed": seed,
                    "status": run.summary["status"],
                    "protocol_version": cohort.protocol_version,
                    "config_hash": cohort.config_hash,
                    "code_commit": run.summary["code_commit"],
                    "code_dirty": run.summary["code_dirty"],
                    "runtime_identity": cohort.runtime,
                    "accepted_training_run": True,
                    "counts_toward_embargo_gate": True,
                    "endpoint_values_expected": True,
                    "disposition": "accepted_registered_training_run",
                    "returncode": 0,
                    "completed_updates": run.summary["updates"],
                    "epochs": run.summary.get("epochs"),
                    "updates": run.summary.get("updates"),
                    "expected_updates": run.summary.get("expected_updates"),
                    "device_id": run.summary["device_id"],
                    "physical_gpu_host": cohort.physical_device.get("host"),
                    "physical_gpu_index": cohort.physical_device.get("physical_index"),
                    "physical_gpu_uuid": cohort.physical_device.get("uuid"),
                    "physical_gpu_name": cohort.physical_device.get("name"),
                    "attempt_path": attempt["attempt_path"],
                    "attempt_row": attempt["attempt_row"],
                    "post_worker_snapshot_disposition": attempt[
                        "post_worker_snapshot_disposition"
                    ],
                    "post_worker_evidence_id": attempt["post_worker_evidence_id"],
                    "summary_path": str(run.summary_path),
                    "supporting_files": canonical_json(supporting_files),
                    "supporting_sha256": canonical_json(supporting_sha256),
                }
            )
    extension_quality_by_seed = [
        row for row in quality_by_seed if row["experiment_id"] in extension_ids
    ]
    extension_quality = [
        row
        for row in quality_summary
        if row["experiment_id"] in extension_ids
        and row["metric_name"] != "mean_per_image_foreground_iou"
    ]
    extension_iou = [
        row
        for row in quality_summary
        if row["experiment_id"] == "E20"
        and row["metric_name"] == "mean_per_image_foreground_iou"
    ]
    extension_efficiency_by_seed = [
        row for row in efficiency_by_seed if row["experiment_id"] in extension_ids
    ]
    extension_efficiency = [
        row for row in efficiency_summary if row["experiment_id"] in extension_ids
    ]
    filenames = _required_output_names(manifest)
    run_audit = [
        {field: row.get(field) for field in RUN_AUDIT_FIELDS}
        for row in [*inventory, *supplemental_audit]
    ]
    run_audit.sort(
        key=lambda row: (
            str(row["experiment_id"]),
            str(row["variant"]),
            int(row["seed"]),
            str(row["record_type"]),
        )
    )
    return {
        str(filenames["run_audit"]): run_audit,
        str(filenames["endpoint_rows"]): endpoint_table,
        str(filenames["variant_summary"]): variant_summary,
        str(filenames["paired_quality_by_seed"]): extension_quality_by_seed,
        str(filenames["paired_quality"]): extension_quality,
        str(filenames["paired_iou"]): extension_iou,
        str(filenames["paired_efficiency_by_seed"]): extension_efficiency_by_seed,
        str(filenames["paired_efficiency"]): extension_efficiency,
        str(filenames["platform_strata"]): platform_rows(quality_summary, manifest),
        "_cohort_audit": audit,
        "_runtime_audit": runtime_audit,
    }


def _load_checked_manifest(
    manifest_path: str | Path, repository_root: str | Path
) -> tuple[dict[str, Any], Path, Path, dict[str, Any]]:
    manifest_file = Path(manifest_path).expanduser().resolve()
    repository = Path(repository_root).expanduser().resolve()
    registration = load_yaml(manifest_file)
    if registration.get("base_manifest_document") is None:
        manifest = registration
    else:
        allowed_fields = {
            "schema_version",
            "analysis_id",
            "amendment_id",
            "analysis_status",
            "frozen",
            "base_analysis_id",
            "base_manifest_document",
            "base_contract_document",
            "contract_document",
            "post_worker_snapshot_policy",
            "post_worker_snapshot_evidence",
            "immutability",
        }
        extra = sorted(set(registration) - allowed_fields)
        if extra:
            raise ValueError(f"analysis amendment has unsupported fields: {extra}")
        base_relative, _ = _validate_document(
            registration["base_manifest_document"],
            name="base_manifest_document",
            repository_root=repository,
        )
        _validate_document(
            registration["base_contract_document"],
            name="base_contract_document",
            repository_root=repository,
        )
        _validate_document(
            registration["contract_document"],
            name="contract_document",
            repository_root=repository,
        )
        base = load_yaml(repository / base_relative)
        _require_equal(
            base.get("analysis_id"),
            registration.get("base_analysis_id"),
            context="analysis amendment",
            field="base_analysis_id",
        )
        _require_equal(
            registration.get("analysis_id"),
            base.get("analysis_id"),
            context="analysis amendment",
            field="analysis_id",
        )
        manifest = copy.deepcopy(base)
        for field in (
            "analysis_id",
            "amendment_id",
            "analysis_status",
            "base_analysis_id",
            "contract_document",
            "base_manifest_document",
            "base_contract_document",
            "post_worker_snapshot_policy",
            "post_worker_snapshot_evidence",
            "immutability",
        ):
            manifest[field] = copy.deepcopy(registration[field])
    checkout = _validate_analysis_checkout(manifest_file, manifest, repository)
    return manifest, manifest_file, repository, checkout


def validate_global_embargo(
    manifest_path: str | Path,
    cohort_roots: Mapping[str, str | Path],
    *,
    repository_root: str | Path,
    runtime_evidence_roots: Mapping[str, str | Path] | None = None,
    supplemental_evidence_roots: Mapping[str, str | Path] | None = None,
    post_worker_evidence_roots: Mapping[str, str | Path] | None = None,
) -> tuple[
    dict[str, CohortSpec],
    dict[str, dict[tuple[str, int], ValidatedRun]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, dict[tuple[str, int], dict[str, Any]]],
]:
    """Open the gate only from a clean, committed analysis checkout."""

    manifest, _, repository, _ = _load_checked_manifest(manifest_path, repository_root)
    return _validate_global_embargo_registered(
        manifest,
        {
            str(experiment_id): Path(path).expanduser().resolve()
            for experiment_id, path in cohort_roots.items()
        },
        repository_root=repository,
        runtime_evidence_roots={
            str(evidence_id): Path(path).expanduser().resolve()
            for evidence_id, path in (runtime_evidence_roots or {}).items()
        },
        supplemental_evidence_roots={
            str(evidence_id): Path(path).expanduser().resolve()
            for evidence_id, path in (supplemental_evidence_roots or {}).items()
        },
        post_worker_evidence_roots={
            str(evidence_id): Path(path).expanduser().resolve()
            for evidence_id, path in (post_worker_evidence_roots or {}).items()
        },
    )


def analyze_extension_campaign(
    manifest_path: str | Path,
    cohort_roots: Mapping[str, str | Path],
    *,
    repository_root: str | Path,
    runtime_evidence_roots: Mapping[str, str | Path] | None = None,
    supplemental_evidence_roots: Mapping[str, str | Path] | None = None,
    post_worker_evidence_roots: Mapping[str, str | Path] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Analyze the campaign only from a clean, committed analysis checkout."""

    manifest, _, repository, _ = _load_checked_manifest(manifest_path, repository_root)
    return _analyze_extension_campaign_registered(
        manifest,
        {
            str(experiment_id): Path(path).expanduser().resolve()
            for experiment_id, path in cohort_roots.items()
        },
        repository_root=repository,
        runtime_evidence_roots={
            str(evidence_id): Path(path).expanduser().resolve()
            for evidence_id, path in (runtime_evidence_roots or {}).items()
        },
        supplemental_evidence_roots={
            str(evidence_id): Path(path).expanduser().resolve()
            for evidence_id, path in (supplemental_evidence_roots or {}).items()
        },
        post_worker_evidence_roots={
            str(evidence_id): Path(path).expanduser().resolve()
            for evidence_id, path in (post_worker_evidence_roots or {}).items()
        },
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ExtensionAnalysisError(
            f"refusing to create an empty registered output: {path.name}"
        )
    fieldnames = list(rows[0])
    if any(list(row) != fieldnames for row in rows):
        raise ExtensionAnalysisError(f"inconsistent output schema for {path.name}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())


def _write_strata_figure(
    rows: Sequence[Mapping[str, Any]], destination: Path
) -> tuple[Path, Path]:
    """Render four stratum estimates without a pooled marker or estimate."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    task_order = ("isic16_poincare_unet", "cifar100_poincare_resnet20")
    labels = {
        "isic16_poincare_unet": "ISIC16 Poincare U-Net",
        "cifar100_poincare_resnet20": "CIFAR-100 Poincare ResNet-20",
    }
    figure, axes = plt.subplots(1, 2, figsize=(7.1, 2.7), squeeze=False)
    for axis, task in zip(axes[0], task_order, strict=True):
        selected = sorted(
            (row for row in rows if row["task"] == task),
            key=lambda row: str(row["experiment_id"]),
        )
        if len(selected) != 2:
            raise ExtensionAnalysisError(f"{task}: expected exactly two platform strata")
        y_positions = [1, 0]
        for y_position, row in zip(y_positions, selected, strict=True):
            estimate = float(row["estimate"])
            lower = float(row["interval_95_lower"])
            upper = float(row["interval_95_upper"])
            axis.errorbar(
                estimate,
                y_position,
                xerr=[[estimate - lower], [upper - estimate]],
                fmt="D",
                color="#842C46",
                capsize=3,
                linewidth=1.3,
                markersize=4.5,
            )
        axis.axvline(0.0, color="#8C8C8C", linestyle="--", linewidth=0.9)
        axis.set_yticks(
            y_positions,
            [
                f"{row['experiment_id']} | {row['physical_gpu_name']} | n={row['pairs']}"
                for row in selected
            ],
        )
        axis.set_title(labels[task])
        axis.set_xlabel("F minus R (native score scale)")
        axis.grid(axis="x", alpha=0.2)
    figure.suptitle("Matched-seed quality differences by registered cohort/platform")
    figure.tight_layout()
    pdf = destination.with_suffix(".pdf")
    png = destination.with_suffix(".png")
    figure.savefig(
        pdf,
        bbox_inches="tight",
        metadata={
            "Creator": "poincare-execution-experiments",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    figure.savefig(
        png,
        dpi=300,
        bbox_inches="tight",
        metadata={"Software": "poincare-execution-experiments"},
    )
    plt.close(figure)
    return pdf, png


def export_extension_campaign(
    manifest_path: str | Path,
    cohort_roots: Mapping[str, str | Path],
    output_dir: str | Path,
    *,
    repository_root: str | Path,
    runtime_evidence_roots: Mapping[str, str | Path] | None = None,
    supplemental_evidence_roots: Mapping[str, str | Path] | None = None,
    post_worker_evidence_roots: Mapping[str, str | Path] | None = None,
) -> Path:
    """Serialize one exporter per destination and publish it atomically."""

    destination = Path(output_dir).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = destination.parent / f".{destination.name}.lock"
    try:
        descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise FileExistsError(
            f"another exporter holds the destination lock: {lock_path}"
        ) from error
    try:
        return _export_extension_campaign_locked(
            manifest_path,
            cohort_roots,
            destination,
            repository_root=repository_root,
            runtime_evidence_roots=runtime_evidence_roots,
            supplemental_evidence_roots=supplemental_evidence_roots,
            post_worker_evidence_roots=post_worker_evidence_roots,
        )
    finally:
        os.close(descriptor)
        lock_path.unlink(missing_ok=True)


def _export_extension_campaign_locked(
    manifest_path: str | Path,
    cohort_roots: Mapping[str, str | Path],
    output_dir: str | Path,
    *,
    repository_root: str | Path,
    runtime_evidence_roots: Mapping[str, str | Path] | None = None,
    supplemental_evidence_roots: Mapping[str, str | Path] | None = None,
    post_worker_evidence_roots: Mapping[str, str | Path] | None = None,
) -> Path:
    """Validate, analyze, and atomically publish a fresh output directory."""

    manifest_file = Path(manifest_path).expanduser().resolve()
    repository = Path(repository_root).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(
            "output directory already exists; refusing to retain or overwrite "
            f"stale files: {destination}"
        )
    manifest, _, _, exporter_git = _load_checked_manifest(manifest_file, repository)
    resolved_roots = {
        str(experiment_id): Path(path).expanduser().resolve()
        for experiment_id, path in cohort_roots.items()
    }
    resolved_evidence = {
        str(evidence_id): Path(path).expanduser().resolve()
        for evidence_id, path in (supplemental_evidence_roots or {}).items()
    }
    resolved_runtime = {
        str(evidence_id): Path(path).expanduser().resolve()
        for evidence_id, path in (runtime_evidence_roots or {}).items()
    }
    resolved_post_worker = {
        str(evidence_id): Path(path).expanduser().resolve()
        for evidence_id, path in (post_worker_evidence_roots or {}).items()
    }
    outputs = _analyze_extension_campaign_registered(
        manifest,
        resolved_roots,
        repository_root=repository,
        runtime_evidence_roots=resolved_runtime,
        supplemental_evidence_roots=resolved_evidence,
        post_worker_evidence_roots=resolved_post_worker,
    )
    cohort_audit = outputs.pop("_cohort_audit")
    runtime_audit = outputs.pop("_runtime_audit")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        for filename, rows in outputs.items():
            _write_csv(staging / filename, rows)
        required_outputs = _required_output_names(manifest)
        gate_path = staging / str(required_outputs["gate_audit"])
        gate_path.write_text(
            json.dumps(
                {
                    "analysis_id": manifest.get("analysis_id"),
                    "status": "open",
                    "opened_only_after_joint_validation": True,
                    "required_successful_runs": int(
                        _require_mapping(manifest["global_embargo"], name="global_embargo")[
                            "required_successful_runs"
                        ]
                    ),
                    "required_complete_pairs": int(
                        _require_mapping(manifest["global_embargo"], name="global_embargo")[
                            "required_complete_pairs"
                        ]
                    ),
                    "cohorts": cohort_audit,
                    "accepted_runtime_evidence": runtime_audit,
                },
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        strata_rows = outputs[str(required_outputs["platform_strata"])]
        _write_strata_figure(
            strata_rows, staging / str(required_outputs["forest_figure_stem"])
        )
        summary = {
            "analysis_id": manifest.get("analysis_id"),
            "analysis_status": manifest.get("analysis_status"),
            "manifest_path": str(manifest_file),
            "manifest_sha256": _sha256_file(manifest_file),
            "endpoint_embargo": "opened_after_global_validation",
            "grand_pool_performed": False,
            "bootstrap_quantile_method": _require_mapping(
                manifest.get("bootstrap"), name="bootstrap"
            )["quantile_method"],
            "analysis_runtime": {
                "python": sys.version,
                "packages": {
                    name: importlib.metadata.version(name)
                    for name in ("numpy", "scipy", "matplotlib")
                },
            },
            "exporter_git": exporter_git,
            "exporter_source_sha256": {
                relative: _sha256_file(SOURCE_REPOSITORY_ROOT / relative)
                for relative in EXPORTER_SOURCE_FILES
            },
            "outputs": {
                path.name: _sha256_file(path)
                for path in sorted(staging.iterdir())
                if path.is_file()
            },
        }
        (staging / str(required_outputs["audit_summary"])).write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if destination.exists():
            raise FileExistsError(
                "output directory appeared during export; refusing to overwrite it: "
                f"{destination}"
            )
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


def _parse_named_paths(values: Iterable[str], *, kind: str) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path:
            raise ValueError(f"{kind} must use ID=PATH")
        if name in output:
            raise ValueError(f"duplicate {kind}: {name}")
        output[name] = Path(raw_path)
    return output


def parse_cohort_roots(values: Iterable[str]) -> dict[str, Path]:
    """Parse repeatable ``EXPERIMENT_ID=PATH`` command-line arguments."""

    return _parse_named_paths(values, kind="cohort root")


def parse_supplemental_evidence_roots(values: Iterable[str]) -> dict[str, Path]:
    """Parse repeatable ``EVIDENCE_ID=PATH`` command-line arguments."""

    return _parse_named_paths(values, kind="supplemental evidence root")


def parse_runtime_evidence_roots(values: Iterable[str]) -> dict[str, Path]:
    """Parse repeatable ``EVIDENCE_ID=PATH`` runtime-evidence arguments."""

    return _parse_named_paths(values, kind="runtime evidence root")


def parse_post_worker_evidence_roots(values: Iterable[str]) -> dict[str, Path]:
    """Parse repeatable ``EVIDENCE_ID=PATH`` post-worker evidence arguments."""

    return _parse_named_paths(values, kind="post-worker evidence root")
