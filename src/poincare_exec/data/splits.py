"""Frozen, auditable split-manifest generation for E09/E10."""

from __future__ import annotations

import csv
import hashlib
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from poincare_exec.artifacts.writers import append_csv_rows, write_json

MANIFEST_FIELDS = (
    "sample_id",
    "source_name",
    "source_index",
    "parent_id",
    "label",
    "source_partition",
    "split",
    "input_sha256",
    "target_sha256",
    "exclusion_reason",
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _grouped_selection(group_ids: list[str], count: int, seed: int) -> set[int]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, group_id in enumerate(group_ids):
        groups[group_id].append(index)
    ordered_groups = sorted(groups)
    rng = np.random.default_rng(seed)
    selected: set[int] = set()
    for group_position in rng.permutation(len(ordered_groups)):
        members = groups[ordered_groups[int(group_position)]]
        if len(selected) + len(members) <= count:
            selected.update(members)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(
            f"cannot select exactly {count} examples without splitting a group"
        )
    return selected


def _stratified_development(
    labels: np.ndarray, group_ids: list[str], count: int, seed: int
) -> set[int]:
    classes = np.unique(labels)
    if count % len(classes):
        raise ValueError("development count must be divisible by the class count")
    per_class = count // len(classes)
    labels_by_group: dict[str, set[int]] = defaultdict(set)
    for label, group_id in zip(labels, group_ids, strict=True):
        labels_by_group[group_id].add(int(label))
    selected: set[int] = set()
    for label in classes:
        indices = np.asarray(
            [
                index
                for index in np.flatnonzero(labels == label)
                if labels_by_group[group_ids[int(index)]] == {int(label)}
            ],
            dtype=np.int64,
        )
        if len(indices) < per_class:
            raise ValueError(f"class {label} has too few examples")
        class_groups = [group_ids[int(index)] for index in indices]
        local = _grouped_selection(class_groups, per_class, seed + int(label))
        selected.update(int(indices[index]) for index in local)
    return selected


def _linkage_fields(
    input_hash: str,
    hash_counts: Counter[str],
    official_training_hashes: set[str],
    partition: str,
) -> tuple[str, str]:
    parent_id = f"exact-input-sha256:{input_hash}" if hash_counts[input_hash] > 1 else ""
    exclusion_reason = ""
    if partition == "test" and input_hash in official_training_hashes:
        exclusion_reason = "exact_input_duplicate_of_official_training"
    return parent_id, exclusion_reason


def _cifar_payload(path: Path) -> dict[bytes, Any]:
    with path.open("rb") as stream:
        return pickle.load(stream, encoding="bytes")  # Official CIFAR artifact.


def build_cifar100_manifest(data_root: Path, output_path: Path, seed: int) -> None:
    """Build the frozen 45k/5k stratified split plus untouched test split."""

    source = data_root / "cifar-100-python"
    train = _cifar_payload(source / "train")
    test = _cifar_payload(source / "test")
    train_labels = np.asarray(train[b"fine_labels"], dtype=np.uint8)
    train_images = np.asarray(train[b"data"], dtype=np.uint8)
    test_images = np.asarray(test[b"data"], dtype=np.uint8)
    train_hashes = [_sha256(image.tobytes()) for image in train_images]
    test_hashes = [_sha256(image.tobytes()) for image in test_images]
    all_hash_counts = Counter(train_hashes + test_hashes)
    official_training_hashes = set(train_hashes)
    development = _stratified_development(train_labels, train_hashes, 5_000, seed)
    rows = []
    for partition, payload, input_hashes in (
        ("train", train, train_hashes),
        ("test", test, test_hashes),
    ):
        labels = np.asarray(payload[b"fine_labels"], dtype=np.uint8)
        images = np.asarray(payload[b"data"], dtype=np.uint8)
        filenames = payload[b"filenames"]
        for index, (_image, label, filename) in enumerate(
            zip(images, labels, filenames, strict=True)
        ):
            input_hash = input_hashes[index]
            parent_id, exclusion_reason = _linkage_fields(
                input_hash, all_hash_counts, official_training_hashes, partition
            )
            split = "test"
            if partition == "train":
                split = "development" if index in development else "train"
            rows.append(
                dict(
                    zip(
                        MANIFEST_FIELDS,
                        (
                            f"{partition}:{index:05d}",
                            filename.decode("utf-8"),
                            index,
                            parent_id,
                            int(label),
                            partition,
                            split,
                            input_hash,
                            _sha256(bytes([int(label)])),
                            exclusion_reason,
                        ),
                        strict=True,
                    )
                )
            )
    append_csv_rows(output_path, rows)


def _isic_pairs(data_root: Path, partition: str) -> list[tuple[str, Path, Path]]:
    source = data_root / "ISIC"
    prefix = "Training" if partition == "train" else "Test"
    image_dir = source / f"ISBI2016_ISIC_Part1_{prefix}_Data"
    mask_dir = source / f"ISBI2016_ISIC_Part1_{prefix}_GroundTruth"
    pairs = []
    for image_path in sorted(image_dir.glob("ISIC_*.jpg")):
        sample_id = image_path.stem
        mask_path = mask_dir / f"{sample_id}_Segmentation.png"
        if not mask_path.is_file():
            raise FileNotFoundError(f"missing mask for {sample_id}")
        pairs.append((sample_id, image_path, mask_path))
    return pairs


def build_isic16_manifest(data_root: Path, output_path: Path, seed: int) -> None:
    """Build the fixed image-level 810/90 split and untouched 379-image test."""

    training = _isic_pairs(data_root, "train")
    testing = _isic_pairs(data_root, "test")
    if len(training) != 900 or len(testing) != 379:
        raise ValueError("ISIC16 counts do not match the official release")
    training_hashes = [_file_sha256(image_path) for _, image_path, _ in training]
    testing_hashes = [_file_sha256(image_path) for _, image_path, _ in testing]
    all_hash_counts = Counter(training_hashes + testing_hashes)
    official_training_hashes = set(training_hashes)
    development = _grouped_selection(training_hashes, 90, seed)
    rows = []
    for partition, pairs, input_hashes in (
        ("train", training, training_hashes),
        ("test", testing, testing_hashes),
    ):
        for index, (sample_id, image_path, mask_path) in enumerate(pairs):
            input_hash = input_hashes[index]
            parent_id, exclusion_reason = _linkage_fields(
                input_hash, all_hash_counts, official_training_hashes, partition
            )
            split = "test"
            if partition == "train":
                split = "development" if index in development else "train"
            rows.append(
                dict(
                    zip(
                        MANIFEST_FIELDS,
                        (
                            sample_id,
                            image_path.name,
                            index,
                            parent_id,
                            "",
                            partition,
                            split,
                            input_hash,
                            _file_sha256(mask_path),
                            exclusion_reason,
                        ),
                        strict=True,
                    )
                )
            )
    append_csv_rows(output_path, rows)


def _validate_counts(path: Path, expected: dict[str, int]) -> None:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    observed = {
        split: sum(row["split"] == split for row in rows)
        for split in ("train", "development", "test")
    }
    if observed != expected:
        raise RuntimeError(f"unexpected split counts in {path}: {observed}")


def _manifest_audit(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["input_sha256"]].append(row)
    duplicates = [members for members in grouped.values() if len(members) > 1]
    train_development_overlap = [
        members
        for members in duplicates
        if {row["split"] for row in members} >= {"train", "development"}
    ]
    official_training_test_overlap = [
        members
        for members in duplicates
        if {row["source_partition"] for row in members} >= {"train", "test"}
    ]
    conflicting_label_groups = [
        members
        for members in duplicates
        if len({row["label"] for row in members if row["label"]}) > 1
    ]
    if train_development_overlap:
        raise RuntimeError(f"linked sources cross train/development in {path}")
    return {
        "manifest": path.name,
        "manifest_sha256": _file_sha256(path),
        "rows": len(rows),
        "exact_duplicate_conflicting_label_groups": len(conflicting_label_groups),
        "exact_duplicate_groups": len(duplicates),
        "official_training_test_overlap_groups": len(official_training_test_overlap),
        "official_test_examples_excluded_from_deduplicated_secondary": sum(
            row["source_partition"] == "test"
            for members in official_training_test_overlap
            for row in members
        ),
        "train_development_overlap_groups": 0,
    }


def build_all_manifests(
    data_root: str | Path,
    output_dir: str | Path,
    *,
    seed: int,
) -> None:
    """Build both paper manifests, refusing to overwrite prior split decisions."""

    data_root = Path(data_root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "isic16": output_dir / "isic16.csv",
        "cifar100": output_dir / "cifar100.csv",
    }
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to replace split manifests: {existing}")
    build_isic16_manifest(data_root, outputs["isic16"], seed)
    build_cifar100_manifest(data_root, outputs["cifar100"], seed)
    _validate_counts(
        outputs["isic16"], {"train": 810, "development": 90, "test": 379}
    )
    _validate_counts(
        outputs["cifar100"],
        {"train": 45_000, "development": 5_000, "test": 10_000},
    )
    write_json(
        output_dir / "manifest_audit.json",
        {
            "split_seed": seed,
            "linkage_definition": "byte-identical raw input SHA-256",
            "manifests": [_manifest_audit(path) for path in outputs.values()],
        },
    )
