"""Strict configuration and provenance helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from poincare_exec.config.constants import REPOSITORY_ROOT


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load one YAML mapping, rejecting ambiguous empty/non-mapping documents."""

    resolved = Path(path).expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"configuration must be a YAML mapping: {resolved}")
    return value


def canonical_json(value: Any) -> str:
    """Serialize a JSON-compatible value in a platform-independent form."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def config_hash(*configs: Mapping[str, Any]) -> str:
    """Return a SHA-256 binding for an ordered collection of mappings."""

    payload = canonical_json([dict(config) for config in configs]).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def require_keys(config: Mapping[str, Any], keys: tuple[str, ...], *, name: str) -> None:
    """Reject a configuration missing any declared keys."""

    missing = [key for key in keys if key not in config]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"{name} is missing required keys: {joined}")


def validate_hashed_document(
    config: Mapping[str, Any],
    *,
    field: str,
    repository_root: str | Path = REPOSITORY_ROOT,
) -> tuple[str, str]:
    """Validate one repository-relative document path and declared SHA-256."""

    metadata = config.get(field)
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{field} must be a mapping")
    relative = Path(str(metadata.get("path", "")))
    expected = str(metadata.get("sha256", ""))
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{field}.path must be repository relative")
    if len(expected) != 64 or any(value not in "0123456789abcdef" for value in expected):
        raise ValueError(f"{field}.sha256 must be a lowercase SHA-256 digest")
    root = Path(repository_root).expanduser().resolve()
    document = (root / relative).resolve()
    if not document.is_relative_to(root) or not document.is_file():
        raise ValueError(f"{field}.path is missing or outside the repository")
    digest = hashlib.sha256()
    with document.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    actual = digest.hexdigest()
    if actual != expected:
        raise ValueError(f"{field} SHA-256 mismatch")
    return relative.as_posix(), actual
