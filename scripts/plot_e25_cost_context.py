#!/usr/bin/env python3
"""Render the validated E25 U-Net absolute-cost context figure."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts import export_e24_e25_e27_tables as exporter

ANALYSIS_ID = "e24_e25_e27_revision_benchmark_export"
FIGURE_DATA_SCHEMA_VERSION = "1.0"
EXPECTED_OUTPUT_ROWS = {
    exporter.OUTPUT_FILES["sessions"]: 145,
    exporter.OUTPUT_FILES["variants"]: 29,
    exporter.OUTPUT_FILES["contrasts"]: 25,
}
SESSION_COLUMNS = (
    "experiment_id",
    "runtime_role",
    "jax_version",
    "jaxlib_version",
    "workload_id",
    "workload_kind",
    "mode",
    "shape",
    "session",
    "variant",
    "variant_role",
    "outcome",
    "steady_state_observations",
    "session_median_latency_ms",
    "compilation_ms",
    "first_compiled_call_ms",
    "peak_live_gib",
    "peak_pool_gib",
    "process_reservation_mib",
    "failure_kind",
    "failure_phase",
    "failure_returncode",
    "failure_stdout",
    "failure_stderr",
    "protocol_version",
    "code_commit",
    "config_hash",
    "device_id",
    "physical_gpu_index",
    "physical_gpu_uuid",
    "physical_gpu_name",
)
WORKLOADS = (
    ("unet_b1_forward", "Batch 1\nforward", "forward"),
    ("unet_b8_forward", "Batch 8\nforward", "forward"),
    ("unet_b8_update", "Batch 8\nupdate", "update"),
)
VARIANTS = (
    ("E", "Euclidean", "euclidean_same_topology_context"),
    ("R", "Reference", "reference_poincare_execution"),
    ("F", "Factorized", "factorized_poincare_execution"),
)
SESSIONS = tuple(range(5))
RUNTIME_ROLE = "pinned_jax_0_4_25_cuda_11_8"
PLOT_COLORS = {"E": "#7A7A7A", "R": "#D55E00", "F": "#0072B2"}
PLOT_MARKERS = {"E": "o", "R": "s", "F": "^"}
FIGURE_RC = {
    "font.family": "DejaVu Sans",
    "font.size": 9.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.transparent": False,
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validated-dir", type=Path, required=True)
    parser.add_argument("--output-stem", type=Path, required=True)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read validated audit: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON mapping: {path}")
    return value


def _read_sessions(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != SESSION_COLUMNS:
                raise ValueError(
                    f"{path.name}: session-table schema differs from the validated "
                    "exporter schema"
                )
            return [dict(row) for row in reader]
    except OSError as error:
        raise ValueError(f"could not read validated session table: {path}") from error


def _require_mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context}: expected a mapping")
    return value


def _require_sha256(value: Any, *, context: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{context}: expected a lowercase SHA-256 digest")
    return text


def _require_commit(value: Any, *, context: str) -> str:
    text = str(value)
    if len(text) != 40 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{context}: expected a lowercase Git commit")
    return text


def _finite_positive(value: Any, *, context: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context}: expected a numeric value") from error
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{context}: expected a finite positive value")
    return number


def _validate_evidence(path: Path, evidence: Any) -> None:
    record = _require_mapping(evidence, context=f"{path.name} evidence")
    expected = {
        "name": path.name,
        "exists": True,
        "bytes": path.stat().st_size if path.is_file() else None,
        "sha256": _sha256(path) if path.is_file() else None,
    }
    for field, value in expected.items():
        if record.get(field) != value:
            raise ValueError(
                f"{path.name}: audit evidence mismatch for {field}; "
                f"observed {record.get(field)!r}, expected {value!r}"
            )


def _cohort_audit(audit: Mapping[str, Any], experiment_id: str) -> Mapping[str, Any]:
    cohorts = audit.get("cohorts")
    if not isinstance(cohorts, list):
        raise ValueError("audit cohorts must be a list")
    selected = [
        cohort
        for cohort in cohorts
        if isinstance(cohort, dict) and cohort.get("experiment_id") == experiment_id
    ]
    if len(selected) != 1:
        raise ValueError(f"audit must contain exactly one {experiment_id} cohort")
    return selected[0]


def validate_bundle(
    validated_dir: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate the exporter audit and return the exact E25 U-Net figure matrix."""

    source_dir = Path(validated_dir).expanduser().resolve()
    paths = {
        name: source_dir / filename for name, filename in exporter.OUTPUT_FILES.items()
    }
    audit = _read_json(paths["audit"])
    expected_identity = {
        "schema_version": "1.0",
        "analysis_id": ANALYSIS_ID,
        "protocol_version": exporter.EXPECTED_PROTOCOL_VERSION,
        "expected_run_commit": exporter.EXPECTED_COMMIT,
        "physical_device": exporter.EXPECTED_DEVICE,
        "all_planned_groups_accounted_for": True,
    }
    for field, expected in expected_identity.items():
        if audit.get(field) != expected:
            raise ValueError(
                f"audit {field} is {audit.get(field)!r}, expected {expected!r}"
            )

    analysis_code = _require_mapping(audit.get("analysis_code"), context="analysis_code")
    analysis_commit = _require_commit(
        analysis_code.get("commit"), context="analysis_code.commit"
    )
    if (
        analysis_code.get("dirty") is not False
        or analysis_code.get("status_porcelain") != []
    ):
        raise ValueError("validated export was not produced from a clean analysis checkout")

    output_rows = _require_mapping(audit.get("output_rows"), context="output_rows")
    if dict(output_rows) != EXPECTED_OUTPUT_ROWS:
        raise ValueError(
            f"audit output row matrix is {dict(output_rows)!r}, "
            f"expected {EXPECTED_OUTPUT_ROWS!r}"
        )
    output_evidence = _require_mapping(
        audit.get("output_evidence"), context="output_evidence"
    )
    if set(output_evidence) != set(EXPECTED_OUTPUT_ROWS):
        raise ValueError("audit output evidence does not cover the exact exporter tables")
    for filename in EXPECTED_OUTPUT_ROWS:
        _validate_evidence(source_dir / filename, output_evidence[filename])

    runtime_captures = _require_mapping(
        audit.get("runtime_captures"), context="runtime_captures"
    )
    runtime = _require_mapping(
        runtime_captures.get(RUNTIME_ROLE), context=f"runtime_captures.{RUNTIME_ROLE}"
    )
    for field, expected in {
        "runtime_role": RUNTIME_ROLE,
        "jax_version": "0.4.25",
        "jaxlib_version": "0.4.25",
    }.items():
        if runtime.get(field) != expected:
            raise ValueError(f"E25 runtime capture has unexpected {field}")

    e25_audit = _cohort_audit(audit, "E25")
    for field, expected in {
        "planned_groups": 90,
        "successful_worker_groups": 90,
        "failed_worker_groups": 0,
        "outcomes": {"ok": 90},
    }.items():
        if e25_audit.get(field) != expected:
            raise ValueError(
                f"E25 audit {field} is {e25_audit.get(field)!r}, expected {expected!r}"
            )
    config_hashes = _require_mapping(
        e25_audit.get("expected_config_hashes"),
        context="E25 expected_config_hashes",
    )

    all_sessions = _read_sessions(paths["sessions"])
    if len(all_sessions) != EXPECTED_OUTPUT_ROWS[exporter.OUTPUT_FILES["sessions"]]:
        raise ValueError("session table row count differs from its validated audit")
    selected = [
        row
        for row in all_sessions
        if row["experiment_id"] == "E25"
        and row["workload_id"] in {workload[0] for workload in WORKLOADS}
    ]
    expected_keys = {
        (workload_id, session, variant)
        for workload_id, _label, _mode in WORKLOADS
        for session in SESSIONS
        for variant, _label, _role in VARIANTS
    }
    indexed: dict[tuple[str, int, str], dict[str, str]] = {}
    for row in selected:
        try:
            key = (row["workload_id"], int(row["session"]), row["variant"])
        except (KeyError, ValueError) as error:
            raise ValueError("E25 U-Net row has an invalid matrix key") from error
        if key in indexed:
            raise ValueError(f"duplicate E25 U-Net session row: {key!r}")
        indexed[key] = row
    if set(indexed) != expected_keys:
        raise ValueError(
            "E25 U-Net session matrix is incomplete or contains unexpected rows; "
            f"missing={sorted(expected_keys - set(indexed))}, "
            f"extra={sorted(set(indexed) - expected_keys)}"
        )

    variant_roles = {variant: role for variant, _label, role in VARIANTS}
    workload_modes = {workload: mode for workload, _label, mode in WORKLOADS}
    expected_device_id = (
        f"{exporter.EXPECTED_DEVICE['hostname']}::gpu:0::{exporter.EXPECTED_DEVICE['name']}"
    )
    validated: list[dict[str, Any]] = []
    for workload_id, _label, _mode in WORKLOADS:
        config_hash = _require_sha256(
            config_hashes.get(workload_id), context=f"{workload_id} config hash"
        )
        for session in SESSIONS:
            for variant, _variant_label, _role in VARIANTS:
                row = indexed[(workload_id, session, variant)]
                expected_fields = {
                    "runtime_role": RUNTIME_ROLE,
                    "jax_version": "0.4.25",
                    "jaxlib_version": "0.4.25",
                    "workload_kind": "unet",
                    "mode": workload_modes[workload_id],
                    "variant_role": variant_roles[variant],
                    "outcome": "ok",
                    "protocol_version": exporter.EXPECTED_PROTOCOL_VERSION,
                    "code_commit": exporter.EXPECTED_COMMIT,
                    "config_hash": config_hash,
                    "device_id": expected_device_id,
                    "physical_gpu_index": str(exporter.EXPECTED_DEVICE["physical_index"]),
                    "physical_gpu_uuid": exporter.EXPECTED_DEVICE["uuid"],
                    "physical_gpu_name": exporter.EXPECTED_DEVICE["name"],
                }
                for field, expected in expected_fields.items():
                    if row.get(field) != expected:
                        raise ValueError(
                            f"E25/{workload_id}/session-{session}/{variant}: "
                            f"unexpected {field} {row.get(field)!r}; expected {expected!r}"
                        )
                if int(row["steady_state_observations"]) != 200:
                    raise ValueError("E25 figure rows must contain 200 steady-state calls")
                validated.append(
                    {
                        **row,
                        "session": session,
                        "session_median_latency_ms": _finite_positive(
                            row["session_median_latency_ms"],
                            context=f"{workload_id}/{variant} latency",
                        ),
                        "peak_live_gib": _finite_positive(
                            row["peak_live_gib"],
                            context=f"{workload_id}/{variant} peak live allocation",
                        ),
                    }
                )

    provenance = {
        "source_audit_sha256": _sha256(paths["audit"]),
        "source_session_table_sha256": _sha256(paths["sessions"]),
        "exporter_analysis_commit": analysis_commit,
    }
    return validated, provenance


def build_figure_data(
    rows: Sequence[Mapping[str, Any]], provenance: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Build the nine directly plotted E25 workload/variant summaries."""

    indexed = {
        (str(row["workload_id"]), int(row["session"]), str(row["variant"])): row
        for row in rows
    }
    output = []
    for workload_id, workload_label, mode in WORKLOADS:
        for variant, variant_label, variant_role in VARIANTS:
            selected = [indexed[(workload_id, session, variant)] for session in SESSIONS]
            latencies = [float(row["session_median_latency_ms"]) for row in selected]
            peaks = [float(row["peak_live_gib"]) for row in selected]
            exemplar = selected[0]
            output.append(
                {
                    "schema_version": FIGURE_DATA_SCHEMA_VERSION,
                    "figure_scope": (
                        "E25 U-Net absolute cost context; no cross-hardware pooling"
                    ),
                    "experiment_id": "E25",
                    "runtime_role": RUNTIME_ROLE,
                    "jax_version": "0.4.25",
                    "jaxlib_version": "0.4.25",
                    "workload_id": workload_id,
                    "workload_label": workload_label.replace("\n", " "),
                    "workload_kind": "unet",
                    "mode": mode,
                    "shape": exemplar["shape"],
                    "variant": variant,
                    "variant_label": variant_label,
                    "variant_role": variant_role,
                    "planned_sessions": 5,
                    "accepted_sessions": 5,
                    "session_ids": json.dumps(SESSIONS, separators=(",", ":")),
                    "session_median_latency_ms": json.dumps(
                        latencies, separators=(",", ":")
                    ),
                    "median_latency_ms": statistics.median(latencies),
                    "min_latency_ms": min(latencies),
                    "max_latency_ms": max(latencies),
                    "peak_live_gib_by_session": json.dumps(peaks, separators=(",", ":")),
                    "median_peak_live_gib": statistics.median(peaks),
                    "min_peak_live_gib": min(peaks),
                    "max_peak_live_gib": max(peaks),
                    "latency_definition": (
                        "median of five session-level medians; each session median uses "
                        "200 synchronized steady-state calls"
                    ),
                    "memory_definition": (
                        "process-lifetime JAX allocator peak through the final "
                        "synchronized measured call; range spans five sessions"
                    ),
                    "protocol_version": exporter.EXPECTED_PROTOCOL_VERSION,
                    "code_commit": exporter.EXPECTED_COMMIT,
                    "config_hash": exemplar["config_hash"],
                    "device_id": exemplar["device_id"],
                    "physical_gpu_uuid": exporter.EXPECTED_DEVICE["uuid"],
                    "source_session_table_sha256": provenance[
                        "source_session_table_sha256"
                    ],
                    "source_audit_sha256": provenance["source_audit_sha256"],
                    "exporter_analysis_commit": provenance["exporter_analysis_commit"],
                }
            )
    return output


def build_figure(figure_data: Sequence[Mapping[str, Any]]) -> Any:
    """Build a two-panel absolute-unit figure with all sessions and their range."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    lookup = {(str(row["workload_id"]), str(row["variant"])): row for row in figure_data}
    if set(lookup) != {
        (workload_id, variant)
        for workload_id, _label, _mode in WORKLOADS
        for variant, _label, _role in VARIANTS
    }:
        raise ValueError("figure data does not contain the exact E25 U-Net matrix")

    with matplotlib.rc_context():
        matplotlib.rcParams.update(matplotlib.rcParamsDefault)
        matplotlib.rcParams.update(FIGURE_RC)
        figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), squeeze=False)
        axes_row = axes[0]
        variant_offsets = {"E": -0.23, "R": 0.0, "F": 0.23}
        session_offsets = (-0.046, -0.023, 0.0, 0.023, 0.046)
        panels = (
            (
                "session_median_latency_ms",
                "median_latency_ms",
                "min_latency_ms",
                "max_latency_ms",
                "a  Steady-state latency",
                "Session-summary latency (ms)",
            ),
            (
                "peak_live_gib_by_session",
                "median_peak_live_gib",
                "min_peak_live_gib",
                "max_peak_live_gib",
                "b  Process-lifetime peak",
                "Peak live allocation (GiB)",
            ),
        )
        for axis, (values_field, median_field, min_field, max_field, title, ylabel) in zip(
            axes_row, panels, strict=True
        ):
            for workload_index, (workload_id, _label, _mode) in enumerate(WORKLOADS):
                for variant, _variant_label, _role in VARIANTS:
                    row = lookup[(workload_id, variant)]
                    center = workload_index + variant_offsets[variant]
                    values = [float(value) for value in json.loads(row[values_field])]
                    axis.scatter(
                        [center + offset for offset in session_offsets],
                        values,
                        color=PLOT_COLORS[variant],
                        marker=PLOT_MARKERS[variant],
                        edgecolor="white",
                        linewidth=0.25,
                        s=20,
                        alpha=0.58,
                        zorder=2,
                    )
                    axis.vlines(
                        center,
                        float(row[min_field]),
                        float(row[max_field]),
                        color=PLOT_COLORS[variant],
                        linewidth=1.05,
                        zorder=1,
                    )
                    axis.scatter(
                        center,
                        float(row[median_field]),
                        color=PLOT_COLORS[variant],
                        marker=PLOT_MARKERS[variant],
                        edgecolor="black",
                        linewidth=0.45,
                        s=42,
                        zorder=3,
                    )
            axis.set_xticks(range(len(WORKLOADS)), [workload[1] for workload in WORKLOADS])
            axis.set_xlim(-0.48, len(WORKLOADS) - 0.52)
            axis.set_ylim(bottom=0.0)
            axis.set_title(title, loc="left", fontsize=9.5)
            axis.set_ylabel(ylabel)
            axis.grid(axis="y", color="#D0D0D0", linewidth=0.55, alpha=0.62)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
        handles = [
            Line2D(
                [0],
                [0],
                marker=PLOT_MARKERS[variant],
                color="none",
                markerfacecolor=PLOT_COLORS[variant],
                markeredgecolor="black",
                markeredgewidth=0.4,
                markersize=6,
                label=f"{variant}: {label}",
            )
            for variant, label, _role in VARIANTS
        ]
        figure.legend(
            handles=handles,
            loc="lower center",
            ncol=3,
            frameon=False,
            bbox_to_anchor=(0.5, -0.005),
            handletextpad=0.35,
            columnspacing=1.35,
        )
        figure.subplots_adjust(left=0.105, right=0.99, top=0.90, bottom=0.27, wspace=0.30)
    return figure


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    materialized = [dict(row) for row in rows]
    if not materialized:
        raise ValueError("refusing to write empty figure data")
    fields = list(materialized[0])
    if any(list(row) != fields for row in materialized):
        raise ValueError("figure-data rows have inconsistent fields")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(materialized)
        stream.flush()
        os.fsync(stream.fileno())


def render(validated_dir: str | Path, output_stem: str | Path) -> tuple[Path, Path, Path]:
    """Validate E25 outputs, then atomically write PDF, PNG, and figure-data CSV."""

    rows, provenance = validate_bundle(validated_dir)
    figure_data = build_figure_data(rows, provenance)
    stem = Path(output_stem).expanduser().resolve()
    if stem.suffix:
        raise ValueError("output_stem must not have a filename suffix")
    destinations = (
        stem.with_suffix(".pdf"),
        stem.with_suffix(".png"),
        Path(f"{stem}_data.csv"),
    )
    existing = [str(path) for path in destinations if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite: " + ", ".join(existing))
    stem.parent.mkdir(parents=True, exist_ok=True)

    source_identity = (
        f"audit sha256:{provenance['source_audit_sha256']}; "
        f"sessions sha256:{provenance['source_session_table_sha256']}"
    )
    with tempfile.TemporaryDirectory(prefix=f".{stem.name}.", dir=stem.parent) as temporary:
        temporary_dir = Path(temporary)
        staged_pdf = temporary_dir / destinations[0].name
        staged_png = temporary_dir / destinations[1].name
        staged_data = temporary_dir / destinations[2].name
        figure = build_figure(figure_data)
        import matplotlib.pyplot as plt

        try:
            figure.savefig(
                staged_pdf,
                format="pdf",
                metadata={
                    "Creator": "poincare-execution-experiments",
                    "Producer": "poincare-execution-experiments",
                    "Subject": source_identity,
                    "CreationDate": None,
                    "ModDate": None,
                },
            )
            figure.savefig(
                staged_png,
                format="png",
                dpi=300,
                metadata={
                    "Software": "poincare-execution-experiments",
                    "Source": source_identity,
                },
            )
        finally:
            plt.close(figure)
        _write_csv(staged_data, figure_data)
        for staged, destination in zip(
            (staged_pdf, staged_png, staged_data), destinations, strict=True
        ):
            os.replace(staged, destination)
    return destinations


def main() -> None:
    arguments = _parser().parse_args()
    pdf, png, data = render(arguments.validated_dir, arguments.output_stem)
    print(json.dumps({"pdf": str(pdf), "png": str(png), "data": str(data)}))


if __name__ == "__main__":
    main()
