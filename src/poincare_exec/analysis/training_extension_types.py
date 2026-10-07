"""Shared contract types for the E20/E21 extension analysis."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REGISTERED_ENDPOINTS = ("fixed_final_update", "validation_best")
TEST_PARTITIONS = {
    "official_test": "official_test",
    "official_test_excluding_exact_training_duplicates": "deduplicated_test",
}
PRIMARY_METRIC_FIELDS = {
    "isic16_poincare_unet": "mean_foreground_dice",
    "cifar100_poincare_resnet20": "top1_accuracy",
}
IOU_METRIC_FIELD = "mean_foreground_iou"
METRIC_NAMES = {
    "mean_foreground_dice": "mean_per_image_foreground_dice",
    "mean_foreground_iou": "mean_per_image_foreground_iou",
    "top1_accuracy": "top1_accuracy",
}
REFERENCE_VARIANT = "R"
CANDIDATE_VARIANT = "F"
COMMON_METADATA_FIELDS = (
    "analysis_id",
    "experiment_id",
    "cohort_role",
    "platform_stratum",
    "task",
    "dataset",
    "protocol_version",
    "config_hash",
    "runtime",
    "physical_gpu_host",
    "physical_gpu_index",
    "physical_gpu_uuid",
    "physical_gpu_name",
)
RUN_AUDIT_FIELDS = (
    "analysis_id",
    "evidence_id",
    "record_type",
    "experiment_id",
    "task",
    "dataset",
    "variant",
    "seed",
    "status",
    "protocol_version",
    "config_hash",
    "code_commit",
    "code_dirty",
    "runtime_identity",
    "accepted_training_run",
    "counts_toward_embargo_gate",
    "endpoint_values_expected",
    "disposition",
    "returncode",
    "failure_class",
    "failure_stage",
    "completed_updates",
    "error_type",
    "error",
    "epochs",
    "updates",
    "expected_updates",
    "device_id",
    "physical_gpu_host",
    "physical_gpu_index",
    "physical_gpu_uuid",
    "physical_gpu_name",
    "attempt_path",
    "attempt_row",
    "accepted_root_attempt_path",
    "accepted_root_attempt_row",
    "summary_path",
    "evidence_root",
    "supporting_files",
    "supporting_sha256",
)


class ExtensionAnalysisError(RuntimeError):
    """Raised when registered evidence is missing, ambiguous, or inconsistent."""


@dataclass(frozen=True)
class CohortSpec:
    """Normalized, immutable analysis contract for one training experiment."""

    analysis_id: str
    experiment_id: str
    role: str
    task: str
    dataset: str
    platform_stratum: str
    protocol_version: str
    config_hash: str
    variants: tuple[str, ...]
    analysis_variants: tuple[str, str]
    seeds: tuple[int, ...]
    accepted_commits: Mapping[tuple[str, int], str]
    physical_device: Mapping[str, Any]
    logical_device_id: str | None
    runtime: str
    expected_epochs: int
    expected_updates: int
    endpoint_example_counts: Mapping[str, Mapping[str, int]]
    endpoint_sample_ids: Mapping[str, tuple[str, ...]]
    prediction_item_shape: tuple[int, ...]
    training_health: Mapping[str, int]

    @property
    def expected_run_keys(self) -> frozenset[tuple[str, int]]:
        return frozenset(
            (variant, seed) for seed in self.seeds for variant in self.variants
        )

    def expected_commit(self, variant: str, seed: int) -> str:
        return self.accepted_commits[(variant, seed)]


@dataclass(frozen=True)
class ValidatedRun:
    """One provenance-checked summary retained behind the embargo gate."""

    cohort: CohortSpec
    variant: str
    seed: int
    summary_path: Path
    summary: Mapping[str, Any]
