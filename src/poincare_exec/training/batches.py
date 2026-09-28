"""Deterministic dataset materialization and indexed batch schedules."""

from __future__ import annotations

import hashlib

import numpy as np

from poincare_exec.data.datasets import DatasetBundle, augment_cifar100


def materialize_dataset(
    bundle: DatasetBundle, *, batch_size: int = 16
) -> DatasetBundle:
    """Materialize a lazy image dataset once using its frozen preprocessing."""

    if bundle.images is not None:
        return bundle
    images = []
    labels = []
    for start in range(0, len(bundle), batch_size):
        indices = np.arange(start, min(start + batch_size, len(bundle)))
        batch_images, batch_labels = bundle.batch(indices)
        images.append(batch_images)
        labels.append(batch_labels)
    return DatasetBundle(
        name=bundle.name,
        sample_ids=bundle.sample_ids,
        labels=np.concatenate(labels),
        images=np.concatenate(images),
        records=(),
    )


def ordered_batches(
    count: int, batch_size: int, seed: int, epoch: int
) -> list[np.ndarray]:
    """Return a seed/epoch-indexed permutation with a retained remainder batch."""

    random = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
    order = random.permutation(count)
    return [order[start : start + batch_size] for start in range(0, count, batch_size)]


def augmentation_seeds(
    sample_ids: tuple[str, ...], indices: np.ndarray, seed: int, epoch: int
) -> np.ndarray:
    """Derive an order-independent augmentation seed for every source example."""

    values = []
    for index in indices:
        identity = f"{seed}:{epoch}:{sample_ids[int(index)]}".encode()
        digest = hashlib.sha256(identity).digest()
        values.append(int.from_bytes(digest[:8], "little", signed=False))
    return np.asarray(values, dtype=np.uint64)


def load_training_batch(
    dataset: DatasetBundle,
    indices: np.ndarray,
    *,
    task_name: str,
    seed: int,
    epoch: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Materialize one batch and apply the task's deterministic augmentation."""

    images, targets = dataset.batch(indices)
    if task_name == "cifar100_poincare_resnet20":
        seeds = augmentation_seeds(dataset.sample_ids, indices, seed, epoch)
        images = augment_cifar100(images, seeds)
    return images, targets
