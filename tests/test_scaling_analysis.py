import csv
import json
import math
from pathlib import Path

import pytest
import yaml

from poincare_exec.analysis.scaling import export_scaling_campaign, load_raw_records
from poincare_exec.config.loader import config_hash

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPOSITORY_ROOT / "configs" / "analysis" / "e13_unet_scaling.yaml"
MANIFEST = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
GPU_UUID = "GPU-00000000-1111-2222-3333-444444444444"
PROTOCOL = yaml.safe_load(
    (REPOSITORY_ROOT / MANIFEST["protocol_document"]["path"]).read_text(encoding="utf-8")
)
BENCHMARK = yaml.safe_load(
    (REPOSITORY_ROOT / "configs" / "benchmarks" / "e13_unet_scaling.yaml").read_text(
        encoding="utf-8"
    )
)
BENCHMARK_WORKLOADS = {workload["id"]: workload for workload in BENCHMARK["workloads"]}


def _snapshot() -> str:
    return json.dumps(
        {
            "status": "ok",
            "index": 1,
            "uuid": GPU_UUID,
            "name": "Tesla V100-PCIE-32GB",
            "compute_processes": [],
            "process_query_status": "ok",
        },
        sort_keys=True,
    )


def _identity(workload: dict, variant: str, session: int) -> dict:
    return {
        "experiment_id": "E13",
        "protocol_version": PROTOCOL["protocol_version"],
        "variant": variant,
        "config_hash": config_hash(
            PROTOCOL, BENCHMARK, BENCHMARK_WORKLOADS[workload["id"]]
        ),
        "code_commit": "a" * 40,
        "code_dirty": False,
        "device_id": "test-host::gpu:0::Tesla V100-PCIE-32GB",
        "precision_policy": json.dumps(
            {
                "storage_dtype": "float32",
                "matmul_precision": "highest",
                "reduced_precision_reduction": False,
                "tf32": False,
            },
            sort_keys=True,
        ),
        "seed_or_session": session,
        "status": "ok",
        "workload_id": workload["id"],
        "shape": json.dumps(
            {
                "batch": workload["batch"],
                "height": workload["resolution"],
                "width": workload["resolution"],
            },
            sort_keys=True,
        ),
        "operation_boundary": "unet:update:depth=1",
        "gpu_snapshot_before": _snapshot(),
        "gpu_snapshot_after": _snapshot(),
    }


def _success_rows(
    workload: dict, variant: str, session: int, timed_calls: int
) -> list[dict]:
    identity = _identity(workload, variant, session)
    candidate_scale = 1.0 if variant == "R" else 0.5
    latency = candidate_scale * (20.0 + workload["batch"] + session)
    peak = (4.0 if variant == "R" else 2.0) * 1024**3
    shared = {
        **identity,
        "measured_peak_bytes_in_use": peak,
        "measured_peak_pool_bytes": peak * 1.5,
    }
    rows = [
        {**shared, "phase": "compilation", "observation": -2, "latency_ms": 1000.0},
        {
            **shared,
            "phase": "first_compiled_call",
            "observation": -1,
            "latency_ms": 100.0,
        },
    ]
    rows.extend(
        {
            **shared,
            "phase": "steady_state",
            "observation": observation,
            "latency_ms": latency,
        }
        for observation in range(timed_calls)
    )
    return rows


def _oom_row(workload: dict, session: int) -> dict:
    row = _identity(workload, "F", session)
    row.update(
        {
            "device_id": "test-host::CUDA_VISIBLE_DEVICES=1",
            "precision_policy": json.dumps(
                {
                    "oracle": "float64",
                    "primary": "float32",
                    "reduced_precision_enabled": False,
                    "matmul_precision": "highest",
                    "tf32": False,
                },
                sort_keys=True,
            ),
            "status": "failed",
            "failure_kind": "oom",
            "failure_phase": "compilation",
            "returncode": 1,
            "stderr": "RESOURCE_EXHAUSTED: explicit allocation failure",
        }
    )
    return row


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_export_preserves_oom_and_pairs_only_successful_sessions(tmp_path):
    manifest_path = MANIFEST_PATH
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    rows = []
    failed_workload = next(
        workload
        for workload in manifest["workloads"]
        if workload["id"] == "unet_update_b16_r256"
    )
    for workload in manifest["workloads"]:
        for session in range(manifest["sessions"]):
            for variant in ("R", "F"):
                if (
                    workload == failed_workload
                    and variant == "F"
                    and session == manifest["sessions"] - 1
                ):
                    continue
                rows.extend(
                    _success_rows(
                        workload,
                        variant,
                        session,
                        manifest["timed_calls_per_session"],
                    )
                )
    raw_csv = tmp_path / "raw.csv"
    _write_csv(raw_csv, rows)
    failures = tmp_path / "raw.csv.failures.jsonl"
    failures.write_text(
        json.dumps(_oom_row(failed_workload, manifest["sessions"] - 1)) + "\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "analysis"
    summary = export_scaling_campaign([raw_csv, failures], manifest_path, output_dir)

    assert summary["session_rows"] == 7 * 5 * 2
    assert summary["outcome_counts"] == {"ok": 69, "oom": 1}
    assert summary["provenance_contract"]["protocol_path"] == (
        "configs/protocols/protocol_1_11.yaml"
    )
    assert summary["provenance_contract"]["protocol_sha256"] == (
        "5dbd77ce1d6fa7e9a33114093a730208048bcf295012b66c0df00f6e3e8711fe"
    )
    assert summary["provenance_contract"]["protocol_version"] == "1.11"
    assert summary["validated_identity"]["physical_gpu_uuid"] == GPU_UUID
    assert len(summary["validated_identity"]["device_id_encodings"]) == 2
    assert len(summary["validated_identity"]["precision_policy_encodings"]) == 2
    with (output_dir / "session_outcomes.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        outcomes = list(csv.DictReader(stream))
    failure = next(row for row in outcomes if row["outcome"] == "oom")
    assert failure["failure_kind"] == "oom"
    assert failure["failure_phase"] == "compilation"
    assert failure["median_latency_ms"] == ""
    assert failure["peak_live_gib"] == ""

    with (output_dir / "paired_ratio_summary.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        ratios = list(csv.DictReader(stream))
    affected = next(row for row in ratios if row["workload_id"] == failed_workload["id"])
    assert affected["paired_latency_sessions"] == "4"
    assert json.loads(affected["paired_session_ids"]) == [0, 1, 2, 3]
    assert math.isclose(float(affected["geometric_mean_latency_ratio"]), 2.0)
    assert (output_dir / "unet_scaling.pdf").is_file()
    assert (output_dir / "unet_scaling.png").is_file()
    assert (output_dir / "summary.json").is_file()


def test_manifest_rejects_a_grid_other_than_the_seven_frozen_shapes(tmp_path):
    manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["sweeps"]["batch"]["values"][-1] = 32
    manifest["workloads"][-1]["batch"] = 32
    manifest["workloads"][-1]["id"] = "unet_update_b32_r256"
    manifest_path = tmp_path / "changed_grid.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    raw = tmp_path / "one.json"
    raw.write_text(json.dumps({"rows": []}), encoding="utf-8")

    with pytest.raises(ValueError, match="frozen E13 R/F grid"):
        export_scaling_campaign([raw], manifest_path, tmp_path / "analysis")


def test_json_loader_accepts_wrapped_rows(tmp_path):
    path = tmp_path / "rows.json"
    path.write_text(json.dumps({"rows": [{"value": 3}]}), encoding="utf-8")

    rows, sources = load_raw_records([path])

    assert rows[0]["value"] == 3
    assert rows[0]["_source_path"] == str(path.resolve())
    assert sources[0]["rows"] == 1
