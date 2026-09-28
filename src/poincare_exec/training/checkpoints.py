"""Atomic complete-state checkpoints for matched continuation and evaluation."""

from __future__ import annotations

import os
import pickle
import tempfile
from pathlib import Path
from typing import Any

import jax


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    """Atomically store a host-resident protocol-5 checkpoint."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            pickle.dump(jax.device_get(payload), stream, protocol=5)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load a trusted checkpoint produced by this repository."""

    with Path(path).expanduser().resolve().open("rb") as stream:
        payload = pickle.load(stream)  # Repository-owned artifact.
    if payload.get("format") != "poincare-exec-checkpoint-v1":
        raise ValueError("unsupported checkpoint format")
    return payload
