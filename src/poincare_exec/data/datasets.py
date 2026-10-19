"""Minimal dependency-free readers for the two frozen datasets."""

from __future__ import annotations

import csv
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class DatasetBundle:
    """A fixed split with deterministic indexed batch materialization."""

    name: str
    sample_ids: tuple[str, ...]
    labels: np.ndarray | None
    images: np.ndarray | None
    records: tuple[tuple[Path, Path], ...]

    def __len__(self) -> int:
        return len(self.sample_ids)

    def batch(self, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Load one batch in the model's declared input scaling."""

        if self.images is not None and self.labels is not None:
            return self.images[indices], self.labels[indices]
        loaded = [self._load_isic(int(index)) for index in indices]
        images, masks = zip(*loaded, strict=True)
        return np.stack(images), np.stack(masks)

    def _load_isic(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        image_path, mask_path = self.records[index]
        with Image.open(image_path) as image_file:
            image = np.asarray(image_file.convert("RGB"), dtype=np.uint8)
        with Image.open(mask_path) as mask_file:
            mask = np.asarray(mask_file.convert("L"), dtype=np.uint8)
        size = max(image.shape[:2])
        padded = np.full((size, size, 3), image.min(), dtype=np.uint8)
        padded[: image.shape[0], : image.shape[1]] = image
        padded_mask = np.zeros((size, size), dtype=np.uint8)
        padded_mask[: mask.shape[0], : mask.shape[1]] = mask
        image = np.asarray(
            Image.fromarray(padded).resize((256, 256), Image.Resampling.BICUBIC),
            dtype=np.float32,
        )
        mask = np.asarray(
            Image.fromarray(padded_mask).resize((256, 256), Image.Resampling.NEAREST),
            dtype=np.uint8,
        )
        return image / 255.0, (mask > 127).astype(np.int32)


def _manifest_rows(path: Path, split: str) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if split == "test_deduplicated":
        return [
            row for row in rows if row["split"] == "test" and not row["exclusion_reason"]
        ]
    return [row for row in rows if row["split"] == split]


def _cifar_payload(path: Path) -> dict[bytes, Any]:
    with path.open("rb") as stream:
        return pickle.load(stream, encoding="bytes")  # Official CIFAR artifact.


def _load_cifar(data_root: Path, manifest: Path, split: str) -> DatasetBundle:
    source = data_root / "cifar-100-python"
    payloads = {
        "train": _cifar_payload(source / "train"),
        "test": _cifar_payload(source / "test"),
    }
    arrays = {
        partition: (
            np.asarray(payload[b"data"], dtype=np.uint8),
            np.asarray(payload[b"fine_labels"], dtype=np.int32),
        )
        for partition, payload in payloads.items()
    }
    rows = _manifest_rows(manifest, split)
    selected_images = []
    selected_labels = []
    for row in rows:
        images, labels = arrays[row["source_partition"]]
        index = int(row["source_index"])
        selected_images.append(images[index])
        selected_labels.append(labels[index])
    image_array = np.stack(selected_images).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    image_array = image_array.astype(np.float32) / 127.5 - 1.0
    return DatasetBundle(
        name="cifar100",
        sample_ids=tuple(row["sample_id"] for row in rows),
        labels=np.asarray(selected_labels, dtype=np.int32),
        images=image_array,
        records=(),
    )


def _load_isic(data_root: Path, manifest: Path, split: str) -> DatasetBundle:
    source = data_root / "ISIC"
    rows = _manifest_rows(manifest, split)
    records = []
    for row in rows:
        prefix = "Training" if row["source_partition"] == "train" else "Test"
        image = source / f"ISBI2016_ISIC_Part1_{prefix}_Data" / f"{row['sample_id']}.jpg"
        mask = (
            source
            / f"ISBI2016_ISIC_Part1_{prefix}_GroundTruth"
            / (f"{row['sample_id']}_Segmentation.png")
        )
        records.append((image, mask))
    return DatasetBundle(
        name="isic16",
        sample_ids=tuple(row["sample_id"] for row in rows),
        labels=None,
        images=None,
        records=tuple(records),
    )


def load_dataset(
    name: str,
    data_root: str | Path,
    manifest_dir: str | Path,
    split: str,
) -> DatasetBundle:
    """Load one named frozen split."""

    data_root = Path(data_root).expanduser().resolve()
    manifest_dir = Path(manifest_dir).expanduser().resolve()
    if name == "cifar100":
        return _load_cifar(data_root, manifest_dir / "cifar100.csv", split)
    if name == "isic16":
        return _load_isic(data_root, manifest_dir / "isic16.csv", split)
    raise ValueError(f"unsupported dataset: {name}")


def augment_cifar100(images: np.ndarray, sample_seeds: np.ndarray) -> np.ndarray:
    """Apply indexed source-order zero-pad crop and horizontal flip."""

    augmented = np.empty_like(images)
    for position, (image, seed) in enumerate(zip(images, sample_seeds, strict=True)):
        rng = np.random.default_rng(int(seed))
        padded = np.pad(image, ((4, 4), (4, 4), (0, 0)), constant_values=0.0)
        top = int(rng.integers(0, 9))
        left = int(rng.integers(0, 9))
        selected = padded[top : top + 32, left : left + 32]
        augmented[position] = selected[:, ::-1] if rng.random() < 0.5 else selected
    return augmented
