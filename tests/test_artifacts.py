from __future__ import annotations

import csv
import json

import pytest

from poincare_exec.artifacts.writers import append_csv_rows, append_jsonl, write_json


def test_atomic_json_refuses_overwrite(tmp_path) -> None:
    path = tmp_path / "value.json"
    write_json(path, {"b": 2, "a": 1})
    assert json.loads(path.read_text()) == {"a": 1, "b": 2}
    with pytest.raises(FileExistsError):
        write_json(path, {"a": 2})


def test_append_writers_preserve_rows_and_schema(tmp_path) -> None:
    jsonl_path = tmp_path / "rows.jsonl"
    csv_path = tmp_path / "rows.csv"
    rows = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
    append_jsonl(jsonl_path, rows)
    append_csv_rows(csv_path, rows)
    assert len(jsonl_path.read_text().splitlines()) == 2
    with csv_path.open(newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 2
    with pytest.raises(ValueError):
        append_csv_rows(csv_path, [{"b": "z", "a": 3}])
