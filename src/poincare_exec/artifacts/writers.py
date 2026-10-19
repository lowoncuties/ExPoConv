"""Atomic and append-only result writers."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from poincare_exec.artifacts.provenance import dumps_pretty


def validate_jsonl_identity(
    path: str | Path,
    expected: Mapping[str, Any],
    *,
    required_fields: Sequence[str] | None = None,
) -> None:
    """Reject resumption when existing rows belong to another run identity."""

    source = Path(path).expanduser().resolve()
    if not source.exists():
        return
    fields = tuple(expected) if required_fields is None else tuple(required_fields)
    missing_expected = [field for field in fields if field not in expected]
    if missing_expected:
        raise ValueError(
            "expected identity is missing fields: " + ", ".join(missing_expected)
        )
    with source.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            mismatches = {
                field: (row.get(field), expected[field])
                for field in fields
                if row.get(field) != expected[field]
            }
            if mismatches:
                detail = ", ".join(
                    f"{field}={actual!r} (expected {wanted!r})"
                    for field, (actual, wanted) in mismatches.items()
                )
                raise ValueError(f"cannot resume {source}: row {line_number} has {detail}")


def write_json(path: str | Path, value: Any, *, overwrite: bool = False) -> None:
    """Atomically create a JSON document, refusing accidental replacement."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing artifact: {destination}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(dumps_pretty(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def append_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Append deterministic JSON Lines records and fsync once."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(dict(row), sort_keys=True, allow_nan=False))
            stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def append_csv_rows(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Append rows while enforcing one stable schema for the whole CSV."""

    materialized = [dict(row) for row in rows]
    if not materialized:
        return
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(materialized[0])
    if any(list(row) != fieldnames for row in materialized):
        raise ValueError("all appended CSV rows must have the same ordered fields")
    exists = destination.exists() and destination.stat().st_size > 0
    if exists:
        with destination.open("r", encoding="utf-8", newline="") as stream:
            existing = next(csv.reader(stream))
        if existing != fieldnames:
            raise ValueError("new CSV schema does not match the existing header")
    with destination.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        if not exists:
            writer.writeheader()
        writer.writerows(materialized)
        stream.flush()
        os.fsync(stream.fileno())
