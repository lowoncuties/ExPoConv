"""Standalone exact-execution experiments for HNN++ Poincare networks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from poincare_exec.core.hnnpp import (
        HNNPPParams,
        conv2d,
        dense,
        init_conv2d,
        init_dense,
    )

__all__ = [
    "HNNPPParams",
    "conv2d",
    "dense",
    "init_conv2d",
    "init_dense",
]

__version__ = "0.1.0"


def __getattr__(name: str) -> Any:
    """Load numerical kernels only when the public kernel API is requested."""

    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from poincare_exec.core import hnnpp

    value = getattr(hnnpp, name)
    globals()[name] = value
    return value
