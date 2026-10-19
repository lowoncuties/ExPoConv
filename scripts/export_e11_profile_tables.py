#!/usr/bin/env python3
"""Audit accepted E11 traces and export descriptive profile tables."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from poincare_exec.artifacts import append_csv_rows, write_json
from poincare_exec.artifacts.provenance import git_provenance
from poincare_exec.config.constants import REPOSITORY_ROOT
from poincare_exec.config.loader import load_yaml


def _resolve(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return (
        (REPOSITORY_ROOT / candidate).resolve()
        if not candidate.is_absolute()
        else candidate.resolve()
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _evidence(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    evidence: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if path.exists():
        evidence.update({"sha256": _sha256(path), "byte_size": path.stat().st_size})
    return evidence


def _read_mapping(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON mapping: {path}")
    return value


def _only_file(root: Path, pattern: str, *, label: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {label} under {root}, found {len(matches)}")
    return matches[0]


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot calculate a quantile of an empty collection")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _expect(actual: Any, expected: Any, *, field: str, source: Path) -> None:
    if actual != expected:
        raise ValueError(
            f"unexpected {field} in {source}: expected {expected!r}, got {actual!r}"
        )


def _audit_snapshot(
    snapshot: dict[str, Any], expected_gpu: dict[str, Any], *, source: Path
) -> None:
    for field, expected in (
        ("status", "ok"),
        ("process_query_status", "ok"),
        ("index", int(expected_gpu["index"])),
        ("uuid", str(expected_gpu["uuid"])),
        ("name", str(expected_gpu["name"])),
        ("compute_processes", []),
    ):
        _expect(snapshot.get(field), expected, field=f"GPU snapshot {field}", source=source)


def _audit_rejected_attempts(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    audits = []
    for attempt in manifest.get("rejected_attempts", []):
        environment_path = _resolve(attempt["environment_path"])
        log_path = _resolve(attempt["log_path"])
        failure_path = _resolve(attempt["failure_path"])
        for path in (environment_path, log_path, failure_path):
            if not path.exists():
                raise FileNotFoundError(path)
        environment = _read_mapping(environment_path)
        _expect(
            environment["git"]["commit"],
            attempt["code_commit"],
            field="environment commit",
            source=environment_path,
        )
        _expect(
            environment["git"]["dirty"],
            False,
            field="environment dirty flag",
            source=environment_path,
        )
        failure = _read_mapping(failure_path)
        _expect(
            failure.get("code_commit"),
            attempt["code_commit"],
            field="failure commit",
            source=failure_path,
        )
        _expect(
            failure.get("code_dirty"),
            False,
            field="failure dirty flag",
            source=failure_path,
        )
        for field, expected in attempt["expected_failure"].items():
            _expect(failure.get(field), expected, field=field, source=failure_path)
        status_path = (
            _resolve(attempt["status_path"])
            if attempt.get("status_path") is not None
            else None
        )
        if status_path is not None:
            if not status_path.exists():
                raise FileNotFoundError(status_path)
            status = status_path.read_text(encoding="utf-8").strip()
            _expect(
                status,
                str(attempt["expected_status"]),
                field="wrapper status",
                source=status_path,
            )
        else:
            status = None
        retained_hlo_path = (
            _resolve(attempt["retained_hlo_path"])
            if attempt.get("retained_hlo_path") is not None
            else None
        )
        if retained_hlo_path is not None and not retained_hlo_path.exists():
            raise FileNotFoundError(retained_hlo_path)
        audits.append(
            {
                "id": attempt["id"],
                "disposition": attempt["disposition"],
                "code_commit": attempt["code_commit"],
                "environment": _evidence(environment_path),
                "log": _evidence(log_path),
                "status": _evidence(status_path),
                "status_value": status,
                "failure": _evidence(failure_path),
                "retained_hlo": _evidence(retained_hlo_path),
                "failure_identity": {
                    field: failure.get(field)
                    for field in (
                        "experiment_id",
                        "workload_id",
                        "variant",
                        "status",
                        "returncode",
                        "protocol_version",
                        "code_commit",
                        "code_dirty",
                        "config_hash",
                    )
                },
                "all_checks_pass": True,
            }
        )
    return audits


def _device_process_id(
    events: list[dict[str, Any]], expected_process_name: str, *, source: Path
) -> int:
    identifiers = {
        int(event["pid"])
        for event in events
        if event.get("ph") == "M"
        and event.get("name") == "process_name"
        and (event.get("args") or {}).get("name") == expected_process_name
    }
    if len(identifiers) != 1:
        raise ValueError(
            f"expected one {expected_process_name!r} process in {source}, "
            f"found {sorted(identifiers)}"
        )
    return identifiers.pop()


def _classify_kernel(event: dict[str, Any], categories: list[dict[str, Any]]) -> str:
    arguments = event.get("args") or {}
    searchable = " ".join(
        str(value)
        for value in (
            event.get("name", ""),
            arguments.get("hlo_op", ""),
            arguments.get("name", ""),
        )
    ).lower()
    fallback = None
    for category in categories:
        if category.get("fallback", False):
            fallback = str(category["id"])
            continue
        if any(str(needle).lower() in searchable for needle in category["needles"]):
            return str(category["id"])
    if fallback is None:
        raise ValueError("kernel category configuration has no fallback")
    return fallback


def _audit_accepted_source(
    manifest: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    expected = manifest["expected"]
    source = manifest["source"]
    contract = manifest["trace_contract"]
    profile_root = _resolve(source["profile_root"])
    config_path = _resolve(source["config_path"])
    environment_path = _resolve(source["environment_path"])
    log_path = _resolve(source["log_path"])
    status_path = _resolve(source["status_path"])
    for path in (profile_root, config_path, environment_path, log_path, status_path):
        if not path.exists():
            raise FileNotFoundError(path)
    status = status_path.read_text(encoding="utf-8").strip()
    _expect(
        status,
        str(source["expected_status"]),
        field="wrapper status",
        source=status_path,
    )
    failures = sorted(profile_root.rglob("failure.json")) + sorted(
        profile_root.rglob("failures.jsonl")
    )
    if failures:
        raise ValueError(f"accepted profile root contains failure artifacts: {failures}")
    environment = _read_mapping(environment_path)
    _expect(
        environment["git"]["commit"],
        expected["code_commit"],
        field="environment commit",
        source=environment_path,
    )
    _expect(
        environment["git"]["dirty"],
        False,
        field="environment dirty flag",
        source=environment_path,
    )
    _expect(
        environment["hostname"],
        expected["device_id"].split("::", maxsplit=1)[0],
        field="environment hostname",
        source=environment_path,
    )
    _expect(
        environment["environment"]["CUDA_VISIBLE_DEVICES"],
        str(expected["physical_gpu"]["index"]),
        field="CUDA_VISIBLE_DEVICES",
        source=environment_path,
    )

    summary_rows = []
    category_rows = []
    artifact_evidence: dict[str, Any] = {}
    variants = [str(value) for value in manifest["variants"]]
    duration_scale = float(contract["duration_units_per_millisecond"])
    expected_calls = int(expected["profiled_calls"])
    for variant in variants:
        variant_root = profile_root / expected["workload_id"] / variant
        summary_path = variant_root / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(summary_path)
        summary = _read_mapping(summary_path)
        expected_fields = {
            "experiment_id": manifest["experiment_id"],
            "workload_id": expected["workload_id"],
            "variant": variant,
            "code_commit": expected["code_commit"],
            "code_dirty": False,
            "protocol_version": expected["protocol_version"],
            "config_hash": expected["config_hash"],
            "device_id": expected["device_id"],
            "device": expected["jax_device"],
            "warmup_calls": int(expected["warmup_calls"]),
            "profiled_calls": expected_calls,
            "status": "ok",
        }
        for field, wanted in expected_fields.items():
            _expect(summary.get(field), wanted, field=field, source=summary_path)
        precision_policy = json.loads(summary["precision_policy"])
        _expect(
            precision_policy.get("storage_dtype"),
            expected["dtype"],
            field="precision storage dtype",
            source=summary_path,
        )
        _audit_snapshot(
            summary["gpu_snapshot_before"],
            expected["physical_gpu"],
            source=summary_path,
        )
        _audit_snapshot(
            summary["gpu_snapshot_after"],
            expected["physical_gpu"],
            source=summary_path,
        )

        hlo_path = variant_root / "compiled_hlo.txt"
        trace_path = _only_file(
            variant_root / "device_trace", contract["source_glob"], label="trace"
        )
        perfetto_path = _only_file(
            variant_root / "device_trace",
            contract["perfetto_glob"],
            label="Perfetto trace",
        )
        xplane_path = _only_file(
            variant_root / "device_trace",
            contract["xplane_glob"],
            label="XPlane trace",
        )
        if not hlo_path.exists():
            raise FileNotFoundError(hlo_path)
        if int(summary["hlo_text_bytes"]) != hlo_path.stat().st_size:
            raise ValueError(f"HLO byte count mismatch for {variant}: {hlo_path}")
        with gzip.open(trace_path, "rt", encoding="utf-8") as stream:
            trace = json.load(stream)
        with gzip.open(perfetto_path, "rt", encoding="utf-8") as stream:
            perfetto = json.load(stream)
        events = trace.get("traceEvents")
        perfetto_events = perfetto.get("traceEvents")
        if not isinstance(events, list) or not events:
            raise ValueError(f"missing trace events: {trace_path}")
        if not isinstance(perfetto_events, list) or len(perfetto_events) != len(events):
            raise ValueError(f"Perfetto/source event-count mismatch: {perfetto_path}")
        device_pid = _device_process_id(
            events, str(contract["device_process_name"]), source=trace_path
        )
        module_durations_ms = [
            float(event["dur"]) / duration_scale
            for event in events
            if event.get("ph") == "X" and event.get("name") == contract["module_event_name"]
        ]
        _expect(
            len(module_durations_ms),
            expected_calls,
            field="profiled module event count",
            source=trace_path,
        )
        device_events = [
            event
            for event in events
            if event.get("ph") == "X" and event.get("pid") == device_pid
        ]
        kernel_marker = str(contract["kernel_marker_argument"])
        memset_marker = str(contract["memset_marker_argument"])
        memcpy_marker = str(contract["memcpy_marker_argument"])
        kernels = [
            event for event in device_events if kernel_marker in (event.get("args") or {})
        ]
        memsets = [
            event for event in device_events if memset_marker in (event.get("args") or {})
        ]
        memcpys = [
            event for event in device_events if memcpy_marker in (event.get("args") or {})
        ]
        if not kernels:
            raise ValueError(f"no device kernel events found in {trace_path}")
        category_counts: Counter[str] = Counter()
        category_durations: Counter[str] = Counter()
        for event in kernels:
            category = _classify_kernel(event, manifest["kernel_categories"])
            category_counts[category] += 1
            category_durations[category] += float(event["dur"])
        cumulative_duration = sum(category_durations.values())
        if cumulative_duration <= 0.0:
            raise ValueError(f"nonpositive cumulative kernel duration: {trace_path}")
        memory = {
            str(field): int(value)
            for field, value in summary["compiler_memory_analysis"].items()
        }
        memory_status = (
            "unavailable_all_zero"
            if memory and all(value == 0 for value in memory.values())
            else "reported"
        )
        row = {
            "experiment_id": manifest["experiment_id"],
            "workload_id": expected["workload_id"],
            "variant": variant,
            "profiled_calls": expected_calls,
            "module_span_median_ms": statistics.median(module_durations_ms),
            "module_span_q1_ms": _quantile(module_durations_ms, 0.25),
            "module_span_q3_ms": _quantile(module_durations_ms, 0.75),
            "module_span_min_ms": min(module_durations_ms),
            "module_span_max_ms": max(module_durations_ms),
            "module_span_mean_ms": statistics.fmean(module_durations_ms),
            "device_kernel_launches_total": len(kernels),
            "device_kernel_launches_per_call": len(kernels) / expected_calls,
            "device_memsets_total": len(memsets),
            "device_memsets_per_call": len(memsets) / expected_calls,
            "device_memcpys_total": len(memcpys),
            "device_memcpys_per_call": len(memcpys) / expected_calls,
            "cumulative_device_kernel_time_ms_total": cumulative_duration / duration_scale,
            "cumulative_device_kernel_time_ms_per_call": cumulative_duration
            / duration_scale
            / expected_calls,
            "trace_event_count": len(events),
            "hlo_text_bytes": int(summary["hlo_text_bytes"]),
            "compiler_memory_status": memory_status,
            "protocol_version": summary["protocol_version"],
            "code_commit": summary["code_commit"],
            "config_hash": summary["config_hash"],
            "device_id": summary["device_id"],
            "trace_duration_scope": contract["duration_scope"],
        }
        for field, value in sorted(summary["hlo_operation_token_counts"].items()):
            row[f"hlo_token_count_{field.replace('-', '_')}"] = int(value)
        for field, value in sorted(memory.items()):
            row[field] = value
        summary_rows.append(row)

        category_ids = [str(item["id"]) for item in manifest["kernel_categories"]]
        for category in category_ids:
            duration = float(category_durations[category])
            category_rows.append(
                {
                    "experiment_id": manifest["experiment_id"],
                    "workload_id": expected["workload_id"],
                    "variant": variant,
                    "category": category,
                    "kernel_launches_total": int(category_counts[category]),
                    "kernel_launches_per_call": category_counts[category] / expected_calls,
                    "cumulative_kernel_time_ms_total": duration / duration_scale,
                    "cumulative_kernel_time_ms_per_call": duration
                    / duration_scale
                    / expected_calls,
                    "share_of_cumulative_kernel_time_percent": 100.0
                    * duration
                    / cumulative_duration,
                    "classification_policy": manifest["classification_policy"],
                    "protocol_version": summary["protocol_version"],
                    "code_commit": summary["code_commit"],
                    "config_hash": summary["config_hash"],
                }
            )
        artifact_evidence[variant] = {
            "summary": _evidence(summary_path),
            "compiled_hlo": _evidence(hlo_path),
            "source_trace": _evidence(trace_path),
            "perfetto_trace": _evidence(perfetto_path),
            "xplane_trace": _evidence(xplane_path),
            "trace_event_count": len(events),
            "perfetto_event_count": len(perfetto_events),
            "device_process_id": device_pid,
        }

    audit = {
        "source_config": _evidence(config_path),
        "environment": _evidence(environment_path),
        "log": _evidence(log_path),
        "status": _evidence(status_path),
        "status_value": status,
        "profile_root": str(profile_root),
        "failure_artifacts": [],
        "variant_artifacts": artifact_evidence,
        "variant_count": len(summary_rows),
        "all_checks_pass": True,
    }
    return summary_rows, category_rows, audit


def _comparison_rows(
    summaries: list[dict[str, Any]], manifest: dict[str, Any]
) -> list[dict[str, Any]]:
    by_variant = {str(row["variant"]): row for row in summaries}
    output = []
    for pair in manifest["comparison_pairs"]:
        baseline_name = str(pair["baseline"])
        candidate_name = str(pair["candidate"])
        baseline = by_variant[baseline_name]
        candidate = by_variant[candidate_name]
        output.append(
            {
                "experiment_id": manifest["experiment_id"],
                "workload_id": baseline["workload_id"],
                "baseline": baseline_name,
                "candidate": candidate_name,
                "baseline_over_candidate_module_span_median": baseline[
                    "module_span_median_ms"
                ]
                / candidate["module_span_median_ms"],
                "baseline_over_candidate_cumulative_kernel_time": baseline[
                    "cumulative_device_kernel_time_ms_per_call"
                ]
                / candidate["cumulative_device_kernel_time_ms_per_call"],
                "baseline_over_candidate_kernel_launches": baseline[
                    "device_kernel_launches_per_call"
                ]
                / candidate["device_kernel_launches_per_call"],
                "baseline_over_candidate_hlo_bytes": baseline["hlo_text_bytes"]
                / candidate["hlo_text_bytes"],
                "interpretation": (
                    "Descriptive single-profile ratio without a confidence interval; "
                    "E08 matched-session timing is the inferential performance evidence."
                ),
                "protocol_version": baseline["protocol_version"],
                "code_commit": baseline["code_commit"],
                "config_hash": baseline["config_hash"],
            }
        )
    return output


def _write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty table: {path}")
    if path.exists():
        raise FileExistsError(path)
    append_csv_rows(path, rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()

    manifest_path = _resolve(arguments.manifest)
    manifest = load_yaml(manifest_path)
    rejected = _audit_rejected_attempts(manifest)
    summaries, categories, source_audit = _audit_accepted_source(manifest)
    comparisons = _comparison_rows(summaries, manifest)

    output_dir = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = manifest["outputs"]
    paths = {
        "profile_summary": output_dir / outputs["profile_summary"],
        "category_attribution": output_dir / outputs["category_attribution"],
        "descriptive_comparison": output_dir / outputs["descriptive_comparison"],
        "audit_summary": output_dir / outputs["audit_summary"],
    }
    if any(path.exists() for path in paths.values()):
        existing = [str(path) for path in paths.values() if path.exists()]
        raise FileExistsError("refusing to overwrite: " + ", ".join(existing))
    _write_table(paths["profile_summary"], summaries)
    _write_table(paths["category_attribution"], categories)
    _write_table(paths["descriptive_comparison"], comparisons)

    summary = {
        "schema_version": manifest["schema_version"],
        "analysis_id": manifest["analysis_id"],
        "experiment_id": manifest["experiment_id"],
        "description": manifest["description"],
        "derivation": (
            "CPU-only audit and aggregation of existing E11 HLO and device-trace "
            "artifacts; no experiment was run by this exporter."
        ),
        "analysis_code": git_provenance(),
        "analysis_manifest": _evidence(manifest_path),
        "accepted_source_audit": source_audit,
        "rejected_attempt_audits": rejected,
        "classification_policy": manifest["classification_policy"],
        "trace_duration_scope": manifest["trace_contract"]["duration_scope"],
        "compiler_memory_policy": manifest["compiler_memory_policy"],
        "table_evidence": {
            name: _evidence(path) for name, path in paths.items() if name != "audit_summary"
        },
        "profile_rows": len(summaries),
        "category_rows": len(categories),
        "comparison_rows": len(comparisons),
        "all_checks_pass": True,
    }
    write_json(paths["audit_summary"], summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
