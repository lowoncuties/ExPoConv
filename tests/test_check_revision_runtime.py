from __future__ import annotations

import pytest

from scripts.check_revision_runtime import RuntimeValidationError, _validate_runtime


def _valid() -> dict[str, object]:
    return {
        "jax_version": "0.11.2",
        "jaxlib_version": "0.11.2",
        "backend": "gpu",
        "devices": [{"id": 0, "platform": "gpu", "device_kind": "NVIDIA RTX A6000"}],
        "expected_jax": "0.11.2",
        "expected_jaxlib": "0.11.2",
        "expected_device_substring": "A6000",
    }


def test_validate_runtime_accepts_registered_gpu() -> None:
    _validate_runtime(**_valid())


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("jax_version", "0.4.25", "expected JAX"),
        ("jaxlib_version", "0.4.25", "expected jaxlib"),
        ("backend", "cpu", "GPU backend"),
        ("devices", [], "exactly one"),
        (
            "devices",
            [{"id": 0, "platform": "gpu", "device_kind": "Tesla V100"}],
            "expected device",
        ),
    ],
)
def test_validate_runtime_fails_closed(field: str, value: object, message: str) -> None:
    arguments = _valid()
    arguments[field] = value
    with pytest.raises(RuntimeValidationError, match=message):
        _validate_runtime(**arguments)
