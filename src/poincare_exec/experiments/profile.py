"""E11 compiler and device-trace attribution for representative workloads."""

from __future__ import annotations

import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import jax

from poincare_exec.artifacts import append_jsonl, base_result_row, write_json
from poincare_exec.config.loader import config_hash, load_yaml, require_keys
from poincare_exec.experiments.benchmark import (
    _gpu_snapshot,
    _memory_analysis,
    _operator_state,
    _require_idle_gpu,
    _selected_physical_gpu,
    _state_after_call,
    _synchronize,
)


def _hlo_counts(text: str) -> dict[str, int]:
    names = (
        "convolution",
        "custom-call",
        "exponential",
        "log",
        "tanh",
        "sinh",
        "cosh",
        "reduce",
        "gather",
        "scatter",
        "fusion",
    )
    lowered = text.lower()
    return {name: lowered.count(name) for name in names}


def profile_worker(
    protocol_path: str | Path,
    config_path: str | Path,
    *,
    workload_id: str,
    variant: str,
    output_dir: str | Path,
) -> None:
    """Compile, archive HLO, and capture one Perfetto-compatible device trace."""

    protocol = load_yaml(protocol_path)
    config = load_yaml(config_path)
    workload = next(item for item in config["workloads"] if item["id"] == workload_id)
    seed = int(protocol["seeds"]["measurement_order"])
    function, state, inputs, curvature = _operator_state(workload, variant, seed, protocol)
    compiled = jax.jit(function).lower(state, inputs, curvature).compile()
    first = compiled(state, inputs, curvature)
    _synchronize(first)
    del first
    for _ in range(int(config["warmup_calls"])):
        output = compiled(state, inputs, curvature)
        _synchronize(output)
        state = _state_after_call(workload["mode"], state, output)
        del output
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    hlo = compiled.as_text()
    (destination / "compiled_hlo.txt").write_text(hlo, encoding="utf-8")
    trace_dir = destination / "device_trace"
    with jax.profiler.trace(str(trace_dir), create_perfetto_trace=True):
        for _ in range(int(config["profiled_calls"])):
            output = compiled(state, inputs, curvature)
            _synchronize(output)
            state = _state_after_call(workload["mode"], state, output)
            del output
    cost = compiled.cost_analysis()
    write_json(
        destination / "summary.json",
        {
            **base_result_row(
                protocol,
                experiment_id=config["experiment_id"],
                variant=variant,
                configuration_hash=config_hash(protocol, config, workload),
                dtype=workload["dtype"],
                seed_or_session=seed,
            ),
            "workload_id": workload_id,
            "variant": variant,
            "device": str(jax.devices()[0]),
            "warmup_calls": int(config["warmup_calls"]),
            "profiled_calls": int(config["profiled_calls"]),
            "hlo_text_bytes": len(hlo.encode("utf-8")),
            "hlo_operation_token_counts": _hlo_counts(hlo),
            "compiler_cost_analysis": cost,
            "compiler_memory_analysis": _memory_analysis(compiled),
        },
    )


def run_profile_suite(
    protocol_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
) -> int:
    """Coordinate fresh-process profiles, preserving failed/OOM cases."""

    resolved_protocol_path = Path(protocol_path).expanduser().resolve()
    resolved_config_path = Path(config_path).expanduser().resolve()
    protocol = load_yaml(resolved_protocol_path)
    config = load_yaml(resolved_config_path)
    require_keys(
        config,
        ("experiment_id", "variants", "warmup_calls", "profiled_calls", "workloads"),
        name="profile config",
    )
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    randomizer = random.Random(int(protocol["seeds"]["measurement_order"]))
    written = 0
    for workload in config["workloads"]:
        order = list(config["variants"])
        randomizer.shuffle(order)
        for variant in order:
            selected = destination / workload["id"] / variant
            if (selected / "summary.json").exists() or (selected / "failure.json").exists():
                continue
            snapshot_before = _require_idle_gpu(config)
            selected.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="poincare-profile-") as directory:
                command = [
                    sys.executable,
                    "-m",
                    "poincare_exec.cli",
                    "profile-worker",
                    "--protocol",
                    str(resolved_protocol_path),
                    "--config",
                    str(resolved_config_path),
                    "--workload-id",
                    workload["id"],
                    "--variant",
                    variant,
                    "--output-dir",
                    str(selected),
                ]
                result = subprocess.run(
                    command,
                    cwd=directory,
                    check=False,
                    capture_output=True,
                    text=True,
                )
            snapshot_after = _gpu_snapshot(_selected_physical_gpu())
            if result.returncode != 0:
                failure = {
                    **base_result_row(
                        protocol,
                        experiment_id=config["experiment_id"],
                        variant=variant,
                        configuration_hash=config_hash(protocol, config, workload),
                        dtype=workload["dtype"],
                        seed_or_session=int(protocol["seeds"]["measurement_order"]),
                        status="failed",
                    ),
                    "workload_id": workload["id"],
                    "returncode": result.returncode,
                    "gpu_snapshot_before": snapshot_before,
                    "gpu_snapshot_after": snapshot_after,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
                write_json(
                    selected / "failure.json",
                    failure,
                )
                append_jsonl(
                    destination / "failures.jsonl",
                    [failure],
                )
            else:
                summary_path = selected / "summary.json"
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                summary["gpu_snapshot_before"] = snapshot_before
                summary["gpu_snapshot_after"] = snapshot_after
                write_json(summary_path, summary, overwrite=True)
            written += 1
    return written
