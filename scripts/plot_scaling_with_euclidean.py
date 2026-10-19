#!/usr/bin/env python3
"""Render the E13 U-Net update scaling figure with the E29 Euclidean reference.

R and F come from the committed E13 workload summary; E comes from the
validated E29 variant summary. Both inputs are SHA-256 checked before
plotting, and the plotted values are written to a figure-data CSV. R's
512-by-512 OOM is reported as an off-axis note, not as a plotted value.
Run from the repository root with the project environment:

    ./.venv/bin/python scripts/plot_scaling_with_euclidean.py \\
        --output-stem results/reviewer_requested/e29_unet_update_scaling
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from poincare_exec.config.constants import FIGURE_PDF_METADATA, PLOT_COLORS, PLOT_MARKERS
from poincare_exec.config.loader import load_yaml, validate_hashed_document
from scripts.export_e24_e25_e27_tables import _read_csv, _read_json, _sha256

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPOSITORY_ROOT / "results"
E13_MANIFEST = REPOSITORY_ROOT / "configs" / "analysis" / "e13_unet_scaling.yaml"
E13_SUMMARY = "e13_scaling/workload_summary.csv"
E29_SUMMARY = "reviewer_requested/e29_variant_summary.csv"
E29_AUDIT = "reviewer_requested/e29_audit.json"
VARIANTS = ("R", "F", "E")
LABELS = {"R": "R", "F": "F", "E": "Euclidean (E)"}
STYLES: dict[str, dict[str, Any]] = {
    "R": {
        "color": PLOT_COLORS["R"],
        "marker": PLOT_MARKERS["R"],
        "linestyle": "-",
        "linewidth": 1.5,
        "markersize": 5,
        "markerfacecolor": PLOT_COLORS["R"],
        "zorder": 3,
    },
    "F": {
        "color": PLOT_COLORS["F"],
        "marker": PLOT_MARKERS["F"],
        "linestyle": "-",
        "linewidth": 1.5,
        "markersize": 5,
        "markerfacecolor": PLOT_COLORS["F"],
        "zorder": 3,
    },
    # Recessive reference: lighter gray, dashed, open diamond, drawn beneath R/F.
    "E": {
        "color": "#A3A3A3",
        "marker": "D",
        "linestyle": (0, (5, 2)),
        "linewidth": 1.2,
        "markersize": 4.2,
        "markerfacecolor": "white",
        "zorder": 2,
    },
}
ALLOCATOR_STYLE = {"color": "#555555", "linestyle": (0, (3, 3)), "linewidth": 1}
FIGURE_SIZE_INCHES = (7.15, 5.2)
DATA_COLUMNS = (
    "variant",
    "source_experiment",
    "workload_id",
    "panels",
    "batch",
    "resolution",
    "planned_sessions",
    "successful_sessions",
    "plotted_sessions",
    "stall_excluded_sessions",
    "oom_sessions",
    "median_latency_ms",
    "min_session_latency_ms",
    "max_session_latency_ms",
    "median_peak_live_gib",
    "physical_gpu_uuid",
    "source_file",
    "source_sha256",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-stem", type=Path, required=True)
    parser.add_argument(
        "--manuscript-pdf",
        type=Path,
        help="optional second copy of the identical PDF, e.g. for the manuscript",
    )
    return parser


def _optional_float(value: str) -> float | None:
    return None if value == "" else float(value)


def _checksummed(relative: str) -> str:
    declared = {
        line[66:].removeprefix("./"): line[:64]
        for line in (RESULTS_DIR / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    digest = _sha256(RESULTS_DIR / relative)
    if declared.get(relative) != digest:
        raise ValueError(f"results/{relative} does not match results/SHA256SUMS")
    return digest


def _e29_summary_digest() -> str:
    """Require the E29 summary to be the exporter output named in its audit."""

    audit = _read_json(RESULTS_DIR / E29_AUDIT)
    declared = audit["output_evidence"][Path(E29_SUMMARY).name]["sha256"]
    digest = _checksummed(E29_SUMMARY)
    if digest != declared or audit.get("all_planned_groups_accounted_for") is not True:
        raise ValueError("E29 variant summary does not match its validated export audit")
    return digest


def build_figure_data() -> tuple[list[dict[str, Any]], float]:
    """Return one plotted row per E13 workload and variant, plus the allocator limit."""

    manifest = load_yaml(E13_MANIFEST)
    validate_hashed_document(manifest, field="allocator_limit_evidence")
    environment = _read_json(REPOSITORY_ROOT / manifest["allocator_limit_evidence"]["path"])
    allocator_limit_gib = (
        int(environment["hardware"]["jax_allocator_bytes_limit"]) / 1024**3
    )
    sources = {
        "E13": (E13_SUMMARY, _checksummed(E13_SUMMARY)),
        "E29": (E29_SUMMARY, _e29_summary_digest()),
    }
    e13 = {
        (row["workload_id"], row["variant"]): row
        for row in _read_csv(RESULTS_DIR / E13_SUMMARY)
    }
    e29 = {
        (row["workload_id"], row["variant"]): row
        for row in _read_csv(RESULTS_DIR / E29_SUMMARY)
    }
    rows = []
    for workload in manifest["workloads"]:
        for variant in VARIANTS:
            key = (workload["id"], variant)
            experiment = "E29" if variant == "E" else "E13"
            source = e29[key] if variant == "E" else e13[key]
            if variant == "E":
                if json.loads(source["shape"]) != {
                    "batch": workload["batch"],
                    "height": workload["resolution"],
                    "width": workload["resolution"],
                }:
                    raise ValueError(f"{key}: E29 shape mismatch")
                values = {
                    "planned_sessions": int(source["planned_sessions"]),
                    "successful_sessions": int(source["finite_sessions"]),
                    "plotted_sessions": int(source["accepted_finite_sessions"]),
                    "stall_excluded_sessions": int(source["driver_stall_flagged_sessions"]),
                    "oom_sessions": int(source["oom_sessions"]),
                    "median_latency_ms": _optional_float(source["median_latency_ms"]),
                    "min_session_latency_ms": _optional_float(
                        source["min_session_latency_ms"]
                    ),
                    "max_session_latency_ms": _optional_float(
                        source["max_session_latency_ms"]
                    ),
                    "median_peak_live_gib": _optional_float(source["median_peak_live_gib"]),
                }
            else:
                if (int(source["batch"]), int(source["resolution"])) != (
                    int(workload["batch"]),
                    int(workload["resolution"]),
                ):
                    raise ValueError(f"{key}: E13 shape mismatch")
                values = {
                    "planned_sessions": int(source["expected_sessions"]),
                    "successful_sessions": int(source["successful_sessions"]),
                    "plotted_sessions": int(source["successful_sessions"]),
                    "stall_excluded_sessions": 0,
                    "oom_sessions": int(source["oom_sessions"]),
                    "median_latency_ms": _optional_float(
                        source["median_latency_ms_successful_sessions"]
                    ),
                    "min_session_latency_ms": _optional_float(
                        source["min_latency_ms_successful_sessions"]
                    ),
                    "max_session_latency_ms": _optional_float(
                        source["max_latency_ms_successful_sessions"]
                    ),
                    "median_peak_live_gib": _optional_float(
                        source["median_peak_live_gib_successful_sessions"]
                    ),
                }
            complete = values["successful_sessions"] == values["planned_sessions"]
            if not complete and values["median_latency_ms"] is not None:
                raise ValueError(f"{key}: incomplete workload would be plotted")
            rows.append(
                {
                    "variant": variant,
                    "source_experiment": experiment,
                    "workload_id": workload["id"],
                    "panels": "|".join(workload["panels"]),
                    "batch": int(workload["batch"]),
                    "resolution": int(workload["resolution"]),
                    **values,
                    "physical_gpu_uuid": source["physical_gpu_uuid"],
                    "source_file": f"results/{sources[experiment][0]}",
                    "source_sha256": sources[experiment][1],
                }
            )
    return rows, allocator_limit_gib


def _plot_panel(
    axis: Any,
    *,
    rows: Sequence[Mapping[str, Any]],
    panel: str,
    metric: str,
    allocator_limit_gib: float,
) -> None:
    field = "batch" if panel == "batch" else "resolution"
    selected = sorted(
        {row[field] for row in rows if panel in str(row["panels"]).split("|")}
    )
    position = {value: index for index, value in enumerate(selected)}
    offset = 0.08
    plotted: list[float] = []
    for variant_index, variant in enumerate(VARIANTS):
        series = sorted(
            (
                row
                for row in rows
                if row["variant"] == variant and panel in str(row["panels"]).split("|")
            ),
            key=lambda row: row[field],
        )
        x_values, y_values, lower, upper = [], [], [], []
        for row in series:
            value = (
                row["median_latency_ms"]
                if metric == "latency"
                else row["median_peak_live_gib"]
            )
            if value is None:
                continue
            x_values.append(position[row[field]] + (variant_index - 1) * offset)
            y_values.append(value)
            if metric == "latency":
                lower.append(value - row["min_session_latency_ms"])
                upper.append(row["max_session_latency_ms"] - value)
            plotted.append(value)
        style = STYLES[variant]
        axis.errorbar(
            x_values,
            y_values,
            yerr=[lower, upper] if metric == "latency" else None,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            markersize=style["markersize"],
            markerfacecolor=style["markerfacecolor"],
            markeredgecolor=style["color"],
            elinewidth=style["linewidth"],
            capsize=2.5,
            label=LABELS[variant],
            zorder=style["zorder"],
        )
    axis.set_xticks(range(len(selected)), [str(value) for value in selected])
    axis.set_xlim(-0.4, len(selected) - 0.6)
    axis.set_xlabel(
        "Batch size" if panel == "batch" else "Input resolution (pixels per side)"
    )
    axis.grid(axis="y", color="#E5E5E5", linewidth=0.7)
    axis.set_axisbelow(True)
    axis.set_yscale("log")
    if metric == "latency":
        axis.set_ylabel("Complete-update latency (ms)")
        return
    axis.set_ylabel("Process-lifetime peak live allocation (GiB)")
    axis.axhline(allocator_limit_gib, **ALLOCATOR_STYLE)
    bottom = min(plotted) / 1.5
    top = max(allocator_limit_gib * 1.12, max(plotted) * 1.4)
    axis.set_ylim(bottom=bottom, top=top)
    ticks = [
        2.0**exponent
        for exponent in range(-6, 6)
        if bottom <= 2.0**exponent <= top and 2.0**exponent < allocator_limit_gib / 1.4
    ]
    axis.set_yticks(
        [*ticks, allocator_limit_gib],
        [*(f"{value:g}" for value in ticks), f"{allocator_limit_gib:.2f}"],
    )
    axis.minorticks_off()


def _oom_note(rows: Sequence[Mapping[str, Any]]) -> str | None:
    notes = [
        f"At {row['resolution']} \u00d7 {row['resolution']} (batch {row['batch']}), "
        f"{row['variant']}: OOM in {row['oom_sessions']}/{row['planned_sessions']} sessions"
        for row in rows
        if row["oom_sessions"]
    ]
    for row in rows:
        if (
            row["successful_sessions"] != row["planned_sessions"]
            and not row["oom_sessions"]
        ):
            raise ValueError(f"{row['workload_id']}/{row['variant']}: unannotated failure")
    return "; ".join(notes) if notes else None


def build_figure(rows: Sequence[Mapping[str, Any]], allocator_limit_gib: float) -> Any:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE_INCHES)
    figure.subplots_adjust(
        left=0.11, right=0.98, bottom=0.10, top=0.81, hspace=0.48, wspace=0.30
    )
    for axis, panel, metric, title in (
        (axes[0, 0], "batch", "latency", "A  Update latency by batch size"),
        (axes[0, 1], "resolution", "latency", "B  Update latency by resolution"),
        (axes[1, 0], "batch", "memory", "C  Peak allocation by batch size"),
        (axes[1, 1], "resolution", "memory", "D  Peak allocation by resolution"),
    ):
        _plot_panel(
            axis,
            rows=rows,
            panel=panel,
            metric=metric,
            allocator_limit_gib=allocator_limit_gib,
        )
        axis.set_title(title, loc="left", fontweight="bold")
    handles = [
        Line2D(
            [],
            [],
            color=STYLES[variant]["color"],
            marker=STYLES[variant]["marker"],
            linestyle=STYLES[variant]["linestyle"],
            linewidth=STYLES[variant]["linewidth"],
            markersize=STYLES[variant]["markersize"],
            markerfacecolor=STYLES[variant]["markerfacecolor"],
            label=LABELS[variant],
        )
        for variant in VARIANTS
    ]
    handles.append(
        Line2D(
            [],
            [],
            **ALLOCATOR_STYLE,
            label=f"JAX allocator limit ({allocator_limit_gib:.2f} GiB)",
        )
    )
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=len(handles),
        frameon=False,
        handlelength=2.6,
        columnspacing=1.6,
    )
    note = _oom_note(rows)
    if note:
        figure.text(
            0.5, 0.866, note, ha="center", va="center", fontsize=8.0, color="#333333"
        )
    figure.suptitle("Poincaré U-Net complete-update scaling", fontsize=11, y=0.985)
    return figure


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=DATA_COLUMNS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def render(output_stem: Path, manuscript_pdf: Path | None = None) -> list[Path]:
    rows, allocator_limit_gib = build_figure_data()
    output_stem = output_stem.expanduser().resolve()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    data_path = output_stem.with_name(f"{output_stem.name}_data.csv")
    pdf_paths = [output_stem.with_suffix(".pdf")]
    if manuscript_pdf is not None:
        pdf_paths.append(manuscript_pdf.expanduser().resolve())
    png_path = output_stem.with_suffix(".png")
    _write_csv(data_path, rows)
    figure = build_figure(rows, allocator_limit_gib)
    try:
        for pdf_path in pdf_paths:
            figure.savefig(pdf_path, metadata=dict(FIGURE_PDF_METADATA), facecolor="white")
        figure.savefig(
            png_path,
            dpi=220,
            metadata={"Software": "poincare-execution-experiments"},
            facecolor="white",
        )
    finally:
        plt.close(figure)
    if len({_sha256(path) for path in pdf_paths}) != 1:
        raise RuntimeError("PDF outputs are not byte-identical")
    return [data_path, *pdf_paths, png_path]


def main() -> None:
    arguments = _parser().parse_args()
    for path in render(arguments.output_stem, arguments.manuscript_pdf):
        print(f"{_sha256(path)}  {path}")


if __name__ == "__main__":
    main()
