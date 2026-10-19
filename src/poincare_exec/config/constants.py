"""Repository-wide immutable constants.

Scientific choices belong in ``protocol.yaml`` or a named experiment config.
This module contains only schema, naming, and implementation constants.
"""

from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parents[1]
DEFAULT_PROTOCOL_PATH = REPOSITORY_ROOT / "protocol.yaml"
GPU_IDLE_SETTLE_TIMEOUT_SECONDS = 30.0
GPU_IDLE_POLL_INTERVAL_SECONDS = 1.0

HYPERBOLIC_VARIANTS = ("R", "F", "T")
# Layers also accept benchmark-only controls. M retains an explicit tangent patch
# while avoiding its elementwise ball map; E is the Euclidean topology reference.
# Rg and Mg are R and M with patches gathered from shifted strided slices instead
# of a one-hot patch convolution. All four stay out of the registered numerical
# and training matrices that iterate R/F/T.
EXECUTION_VARIANTS = (*HYPERBOLIC_VARIANTS, "M", "E", "Rg", "Mg")
CONV_DIMENSION_NUMBERS = ("NHWC", "HWIO", "NHWC")
MIN_POSITIVE_DENOMINATOR = 1.0e-15
SERIES_CUTOFF_FLOAT32 = 1.0e-3
SERIES_CUTOFF_FLOAT64 = 1.0e-6

REQUIRED_RESULT_FIELDS = (
    "experiment_id",
    "protocol_version",
    "variant",
    "config_hash",
    "code_commit",
    "reference_commit",
    "device_id",
    "precision_policy",
    "seed_or_session",
    "status",
    "timestamp_utc",
)

PLOT_COLORS = {
    "R": "#8C8C8C",
    "F": "#0072B2",
    "T": "#D55E00",
}
PLOT_MARKERS = {"R": "o", "F": "s", "T": "^"}
PLOT_LINESTYLES = {"R": "--", "F": "-.", "T": "-"}
DIAGNOSTIC_PLOT_COLORS = {
    "F": PLOT_COLORS["F"],
    "T": PLOT_COLORS["T"],
    "R_repeat": "#6A3D9A",
    "R_ulp": "#009E73",
}
DIAGNOSTIC_PLOT_MARKERS = {
    "F": PLOT_MARKERS["F"],
    "T": PLOT_MARKERS["T"],
    "R_repeat": "X",
    "R_ulp": "o",
}
DIAGNOSTIC_SOURCE_LINESTYLES = {"R": "-", "F": "--", "T": ":"}
DIAGNOSTIC_SYMLOG_LINEAR_THRESHOLD = 1.0e-8
TASK_DISPLAY_NAMES = {
    "isic16_poincare_unet": "ISIC 2016 Poincare U-Net",
    "cifar100_poincare_resnet20": "CIFAR-100 Poincare ResNet-20",
}
FIGURE_PDF_METADATA = (
    ("Creator", "poincare-execution-experiments"),
    ("CreationDate", None),
    ("ModDate", None),
)
