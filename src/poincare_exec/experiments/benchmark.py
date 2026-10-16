"""Fresh-process JAX operator benchmarking with explicit synchronization."""

from __future__ import annotations

import csv
import json
import os
import platform
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from poincare_exec.artifacts import (
    append_csv_rows,
    append_jsonl,
    base_result_row,
    write_json,
)
from poincare_exec.artifacts.provenance import (
    git_provenance,
    precision_policy,
    timestamp_utc,
)
from poincare_exec.config.constants import (
    GPU_IDLE_POLL_INTERVAL_SECONDS,
    GPU_IDLE_SETTLE_TIMEOUT_SECONDS,
)
from poincare_exec.config.loader import canonical_json, config_hash, load_yaml, require_keys
from poincare_exec.core.geometry import expmap0, logmap0
from poincare_exec.core.hnnpp import conv2d, dense, init_conv2d, init_dense
from poincare_exec.core.operations import (
    euclidean_batch_norm,
    init_midpoint_batch_norm,
    midpoint_batch_norm,
)
from poincare_exec.models.resnet import init_resnet20, resnet20_forward
from poincare_exec.models.unet import init_unet, unet_forward
from poincare_exec.training.losses import cross_entropy, dice_focal_loss
from poincare_exec.training.optimizers import adam_update, init_adam

Array = jax.Array


def _workload_experiment_id(config: dict[str, Any], workload: dict[str, Any]) -> str:
    groups = config.get("workload_experiment_prefixes")
    if groups is None:
        return str(config["experiment_id"])
    matches = [
        experiment_id
        for experiment_id, prefixes in groups.items()
        if any(workload["id"].startswith(prefix) for prefix in prefixes)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"workload {workload['id']} must match exactly one experiment prefix group"
        )
    return str(matches[0])


def _dtype(name: str):
    if name == "float32":
        return jnp.float32
    if name == "float64":
        return jnp.float64
    raise ValueError(f"unsupported benchmark dtype: {name}")


def _synchronize(tree: Any) -> None:
    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def _state_after_call(mode: str, state: Any, output: Any) -> Any:
    """Advance complete-update state while keeping other boundaries fixed."""

    if mode != "update":
        return state
    _, updated_params, updated_optimizer, _ = output
    return updated_params, updated_optimizer


def _state_summary(state: Any) -> dict[str, int | float | None]:
    """Return a compact synchronized summary of every state leaf."""

    leaves = jax.tree_util.tree_leaves(state)
    squares = [jnp.sum(jnp.square(jnp.asarray(leaf, dtype=jnp.float64))) for leaf in leaves]
    nonfinite = [jnp.sum(~jnp.isfinite(leaf)) for leaf in leaves]
    l2_norm = jnp.sqrt(sum(squares))
    nonfinite_count = sum(nonfinite)
    _synchronize((l2_norm, nonfinite_count))
    nonfinite_value = int(nonfinite_count)
    return {
        "leaf_count": len(leaves),
        "scalar_count": sum(int(leaf.size) for leaf in leaves),
        "l2_norm": float(l2_norm) if nonfinite_value == 0 else None,
        "nonfinite_count": nonfinite_value,
    }


def _tree_nonfinite_count(tree: Any) -> int:
    counts = [jnp.sum(~jnp.isfinite(leaf)) for leaf in jax.tree_util.tree_leaves(tree)]
    total = sum(counts)
    _synchronize(total)
    return int(total)


def _memory_analysis(compiled: Any) -> dict[str, int | None]:
    try:
        analysis = compiled.memory_analysis()
    except (AttributeError, RuntimeError):
        return {
            "compiler_argument_bytes": None,
            "compiler_output_bytes": None,
            "compiler_temporary_bytes": None,
            "compiler_alias_bytes": None,
        }
    return {
        "compiler_argument_bytes": getattr(analysis, "argument_size_in_bytes", None),
        "compiler_output_bytes": getattr(analysis, "output_size_in_bytes", None),
        "compiler_temporary_bytes": getattr(analysis, "temp_size_in_bytes", None),
        "compiler_alias_bytes": getattr(analysis, "alias_size_in_bytes", None),
    }


def _process_gpu_memory_mib() -> int | None:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 2 and fields[0] == str(os.getpid()):
            return int(fields[1])
    return 0


def _write_worker_phase(path: str | Path | None, phase: str) -> None:
    """Persist the last entered worker phase outside the measured boundary."""

    if path is None:
        return
    write_json(path, {"phase": phase}, overwrite=True)


def _read_worker_phase(path: str | Path) -> str:
    """Read a worker phase marker, retaining an explicit unknown state."""

    source = Path(path)
    if not source.is_file():
        return "unknown"
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "unknown"
    phase = value.get("phase") if isinstance(value, dict) else None
    return str(phase) if phase else "unknown"


def _classify_worker_failure(*, stdout: str, stderr: str, returncode: int) -> str:
    """Classify only explicit OOM diagnostics; do not infer OOM from a signal."""

    diagnostic = f"{stdout}\n{stderr}".casefold()
    oom_markers = (
        "out of memory",
        "cuda_error_out_of_memory",
        "failed to allocate",
        "ran out of memory",
    )
    if any(marker in diagnostic for marker in oom_markers):
        return "oom"
    if returncode < 0:
        return "terminated_by_signal"
    return "failed"


def _logical_gpu_device_identity(snapshot: dict[str, Any]) -> str:
    """Match the single-visible-GPU identity emitted by a benchmark worker."""

    if snapshot.get("status") != "ok" or not snapshot.get("name"):
        return (
            f"{platform.node()}::CUDA_VISIBLE_DEVICES="
            f"{os.environ.get('CUDA_VISIBLE_DEVICES')}"
        )
    return f"{platform.node()}::gpu:0::{snapshot['name']}"


def _model_state(
    workload: dict[str, Any],
    variant: str,
    seed: int,
    protocol: dict[str, Any],
):
    kind = workload["kind"]
    task_names = {
        "unet": "isic16_poincare_unet",
        "resnet20": "cifar100_poincare_resnet20",
    }
    task_name = task_names[kind]
    task = protocol["tasks"][task_name]
    shape = workload["shape"]
    key = jax.random.PRNGKey(seed)
    input_key, target_key, parameter_key = jax.random.split(key, 3)
    dtype = _dtype(workload["dtype"])
    batch = int(shape["batch"])
    if kind == "unet":
        inputs = jax.random.uniform(
            input_key,
            (batch, int(shape["height"]), int(shape["width"]), 3),
            dtype=dtype,
        )
        targets = jax.random.bernoulli(
            target_key,
            shape=(batch, int(shape["height"]), int(shape["width"])),
        ).astype(jnp.int32)
        params = init_unet(
            parameter_key,
            input_channels=3,
            classes=int(task["classes"]),
            initial_width=int(task["model"]["initial_width"]),
            depth=int(task["model"]["levels"]),
            initial_curvature=float(task["curvature"]["initial_magnitude"]),
            dtype=dtype,
        )
        if variant == "E":
            # The Euclidean topology has no learned curvature. All remaining
            # parameter arrays and their deterministic initialization are shared.
            params.pop("curvature_raw")

        def forward(selected_params, selected_inputs, _selected_curvature):
            return unet_forward(
                selected_params,
                selected_inputs,
                variant=variant,
                epsilon=float(task["normalization"]["epsilon"]),
            )

        def loss(selected_params, selected_inputs, selected_curvature):
            logits = forward(selected_params, selected_inputs, selected_curvature)
            loss_config = task["loss"]
            return dice_focal_loss(
                logits,
                targets,
                smoothing=float(loss_config["dice_smoothing"]),
                denominator_epsilon=float(loss_config["dice_denominator_epsilon"]),
                alpha=float(loss_config["focal_alpha"]),
                gamma=float(loss_config["focal_gamma"]),
            )

    elif kind == "resnet20":
        inputs = 2.0 * jax.random.uniform(input_key, (batch, 32, 32, 3), dtype=dtype) - 1.0
        targets = jax.random.randint(
            target_key, (batch,), 0, int(task["classes"]), dtype=jnp.int32
        )
        params = init_resnet20(
            parameter_key,
            channels=tuple(task["model"]["channels"]),
            blocks_per_group=tuple(task["model"]["blocks_per_group"]),
            classes=int(task["classes"]),
            dtype=dtype,
        )

        def forward(selected_params, selected_inputs, selected_curvature):
            return resnet20_forward(
                selected_params,
                selected_inputs,
                selected_curvature,
                variant=variant,
                epsilon=float(task["normalization"]["epsilon"]),
                residual_order=task["model"]["skip_order"],
            )

        def loss(selected_params, selected_inputs, selected_curvature):
            return cross_entropy(
                forward(selected_params, selected_inputs, selected_curvature), targets
            )

    else:
        raise ValueError(f"unsupported model workload kind: {kind}")

    curvature = jnp.asarray(task["curvature"]["initial_magnitude"], dtype=dtype)
    mode = workload["mode"]
    if mode == "forward":
        function = forward
        state = params
    elif mode == "backward":
        function = jax.value_and_grad(loss, argnums=(0, 1, 2))
        state = params
    elif mode == "update":
        optimizer = task["optimizer"]

        def complete_update(selected_state, selected_inputs, selected_curvature):
            selected_params, optimizer_state = selected_state
            value, gradients = jax.value_and_grad(loss)(
                selected_params, selected_inputs, selected_curvature
            )
            updated_params, updated_optimizer, diagnostics = adam_update(
                selected_params,
                gradients,
                optimizer_state,
                learning_rate=float(optimizer["learning_rate"]),
                weight_decay=float(optimizer["weight_decay"]),
                gradient_clip_global_norm=optimizer["gradient_clip_global_norm"],
            )
            return value, updated_params, updated_optimizer, diagnostics

        function = complete_update
        state = (params, init_adam(params))
    else:
        raise ValueError("model benchmark mode must be forward, backward, or update")
    return function, state, inputs, curvature


def _operator_state(
    workload: dict[str, Any],
    variant: str,
    seed: int,
    protocol: dict[str, Any],
):
    dtype = _dtype(workload["dtype"])
    key = jax.random.PRNGKey(seed)
    shape = workload["shape"]
    depth = int(workload.get("depth", 1))
    curvature = jnp.asarray(workload["curvature"], dtype=dtype)
    keys = jax.random.split(key, depth + 1)
    if workload["kind"] == "conv2d":
        input_shape = (
            shape["batch"],
            shape["height"],
            shape["width"],
            shape["input_channels"],
        )
        inputs = 0.05 * jax.random.normal(keys[0], input_shape, dtype=dtype)
        params = []
        for index in range(depth):
            input_channels = (
                shape["input_channels"] if index == 0 else shape["output_channels"]
            )
            params.append(
                init_conv2d(
                    keys[index + 1],
                    workload["kernel_size"],
                    input_channels,
                    shape["output_channels"],
                    dtype=dtype,
                    initialization="normal",
                )
            )

        def forward(selected_params, selected_inputs, selected_curvature):
            value = selected_inputs
            for layer in selected_params:
                value = conv2d(
                    layer,
                    value,
                    selected_curvature,
                    variant=variant,
                    stride=workload["stride"],
                    padding=workload["padding"],
                )
                if workload.get("activation", "none") == "relu":
                    value = jax.nn.relu(value)
            return value

    elif workload["kind"] == "conv_bn_relu":
        input_shape = (
            shape["batch"],
            shape["height"],
            shape["width"],
            shape["input_channels"],
        )
        inputs = 0.05 * jax.random.normal(keys[0], input_shape, dtype=dtype)
        params = []
        for index in range(depth):
            input_channels = (
                shape["input_channels"] if index == 0 else shape["output_channels"]
            )
            params.append(
                {
                    "conv": init_conv2d(
                        keys[index + 1],
                        workload["kernel_size"],
                        input_channels,
                        shape["output_channels"],
                        dtype=dtype,
                        initialization="normal",
                    ),
                    "bn": init_midpoint_batch_norm(shape["output_channels"], dtype=dtype),
                }
            )

        def forward(selected_params, selected_inputs, selected_curvature):
            value = selected_inputs
            for layer in selected_params:
                value = conv2d(
                    layer["conv"],
                    value,
                    selected_curvature,
                    variant=variant,
                    stride=workload["stride"],
                    padding=workload["padding"],
                )
                if variant == "E":
                    value = euclidean_batch_norm(
                        layer["bn"],
                        value,
                        epsilon=float(workload["normalization_epsilon"]),
                    )
                else:
                    value = midpoint_batch_norm(
                        layer["bn"],
                        value,
                        selected_curvature,
                        epsilon=float(workload["normalization_epsilon"]),
                    )
                value = jax.nn.relu(value)
            return value

    elif workload["kind"] == "dense":
        input_shape = (shape["batch"], shape["input_channels"])
        inputs = 0.05 * jax.random.normal(keys[0], input_shape, dtype=dtype)
        params = []
        for index in range(depth):
            input_channels = (
                shape["input_channels"] if index == 0 else shape["output_channels"]
            )
            params.append(
                init_dense(
                    keys[index + 1],
                    input_channels,
                    shape["output_channels"],
                    dtype=dtype,
                    initialization="normal",
                )
            )

        def forward(selected_params, selected_inputs, selected_curvature):
            value = selected_inputs
            for layer in selected_params:
                value = jax.nn.relu(
                    dense(layer, value, selected_curvature, variant=variant)
                )
            return value

    elif workload["kind"] in {"unet", "resnet20"}:
        return _model_state(workload, variant, seed, protocol)
    else:
        raise ValueError(f"unsupported workload kind: {workload['kind']}")
    coordinate_boundary = workload.get("coordinate_boundary", "tangent_to_tangent")
    if coordinate_boundary == "ball_to_ball":
        tangent_forward = forward
        inputs = expmap0(inputs, curvature)

        def forward(selected_params, selected_inputs, selected_curvature):
            tangent_inputs = logmap0(selected_inputs, selected_curvature)
            tangent_outputs = tangent_forward(
                selected_params, tangent_inputs, selected_curvature
            )
            return expmap0(tangent_outputs, selected_curvature)

    elif coordinate_boundary != "tangent_to_tangent":
        raise ValueError(f"unsupported coordinate boundary: {coordinate_boundary}")
    if workload["mode"] == "forward":
        function = forward
    elif workload["mode"] == "backward":

        def loss(selected_params, selected_inputs, selected_curvature):
            output = forward(selected_params, selected_inputs, selected_curvature)
            return jnp.mean(jnp.square(output))

        function = jax.value_and_grad(loss, argnums=(0, 1, 2))
    else:
        raise ValueError("benchmark mode must be forward or backward")
    return function, params, inputs, curvature


def _sustained_sequence(
    function: Any,
    state: Any,
    inputs: Array,
    curvature: Array,
    calls: int,
) -> dict[str, float | int]:
    """Time one compiled, dependent, shape-preserving operator sequence."""

    def sequence(selected_state, selected_inputs, selected_curvature):
        def body(value, _):
            return function(selected_state, value, selected_curvature), None

        output, _ = jax.lax.scan(body, selected_inputs, xs=None, length=calls)
        return output

    compilation_start = time.perf_counter_ns()
    compiled = jax.jit(sequence).lower(state, inputs, curvature).compile()
    compilation_ms = (time.perf_counter_ns() - compilation_start) / 1.0e6
    first_start = time.perf_counter_ns()
    first_output = compiled(state, inputs, curvature)
    _synchronize(first_output)
    first_call_ms = (time.perf_counter_ns() - first_start) / 1.0e6
    del first_output
    start = time.perf_counter_ns()
    output = compiled(state, inputs, curvature)
    _synchronize(output)
    latency_ms = (time.perf_counter_ns() - start) / 1.0e6
    return {
        "compilation_ms": compilation_ms,
        "first_call_ms": first_call_ms,
        "latency_ms": latency_ms,
        "output_nonfinite_count": _tree_nonfinite_count(output),
        "calls_per_second": 1000.0 * calls / latency_ms,
    }


def benchmark_worker(
    protocol_path: str | Path,
    config_path: str | Path,
    *,
    workload_id: str,
    variant: str,
    session: int,
    output_path: str | Path,
    phase_status_path: str | Path | None = None,
) -> None:
    """Execute one workload/variant/session in the current fresh process."""

    _write_worker_phase(phase_status_path, "initialization")
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    protocol = load_yaml(protocol_path)
    config = load_yaml(config_path)
    workload = next(item for item in config["workloads"] if item["id"] == workload_id)
    experiment_id = _workload_experiment_id(config, workload)
    suite_hash = config_hash(protocol, config, workload)
    seed = int(protocol["seeds"]["measurement_order"]) + session
    function, initial_state, inputs, curvature = _operator_state(
        workload, variant, seed, protocol
    )
    state_before = _state_summary(initial_state)
    _write_worker_phase(phase_status_path, "compilation")
    compilation_start = time.perf_counter_ns()
    compiled = jax.jit(function).lower(initial_state, inputs, curvature).compile()
    compilation_ms = (time.perf_counter_ns() - compilation_start) / 1.0e6
    _write_worker_phase(
        phase_status_path,
        "first_update" if workload["mode"] == "update" else "first_call",
    )
    first_call_start = time.perf_counter_ns()
    first_output = compiled(initial_state, inputs, curvature)
    _synchronize(first_output)
    first_call_ms = (time.perf_counter_ns() - first_call_start) / 1.0e6
    del first_output
    state = initial_state
    _write_worker_phase(
        phase_status_path,
        "warmup_updates" if workload["mode"] == "update" else "warmup_calls",
    )
    for _ in range(int(config["warmup_calls"])):
        output = compiled(state, inputs, curvature)
        _synchronize(output)
        state = _state_after_call(workload["mode"], state, output)
        del output
    latencies = []
    timed_calls = int(config["timed_calls_per_session"])
    if timed_calls < 1:
        raise ValueError("timed_calls_per_session must be positive")
    _write_worker_phase(
        phase_status_path,
        "measured_updates" if workload["mode"] == "update" else "measured_calls",
    )
    for call_index in range(timed_calls):
        start = time.perf_counter_ns()
        output = compiled(state, inputs, curvature)
        _synchronize(output)
        latencies.append((time.perf_counter_ns() - start) / 1.0e6)
        state = _state_after_call(workload["mode"], state, output)
        if call_index + 1 < timed_calls:
            del output
    _write_worker_phase(phase_status_path, "post_measurement")
    memory_stats = jax.devices()[0].memory_stats() or {}
    process_reservation_mib = _process_gpu_memory_mib()
    output_nonfinite_count = _tree_nonfinite_count(output)
    state_after = _state_summary(state)
    result_status = (
        "ok"
        if output_nonfinite_count == 0 and state_after["nonfinite_count"] == 0
        else "nonfinite"
    )
    sustained_calls = (
        int(config["sustained_sequence_calls"])
        if workload.get("sustained_sequence", False)
        else 0
    )
    sustained = (
        _sustained_sequence(
            function,
            initial_state,
            inputs,
            curvature,
            sustained_calls,
        )
        if sustained_calls > 0
        else None
    )
    shared = {
        **base_result_row(
            protocol,
            experiment_id=experiment_id,
            variant=variant,
            configuration_hash=suite_hash,
            dtype=workload["dtype"],
            seed_or_session=session,
            status=result_status,
        ),
        "workload_id": workload_id,
        "shape": canonical_json(workload["shape"]),
        "kernel": str(workload.get("kernel_size")),
        "layout": (
            "BHWC/model-pytree"
            if workload["kind"] in {"unet", "resnet20"}
            else "BHWC/HWIO"
            if workload["kind"] in {"conv2d", "conv_bn_relu"}
            else "BF/FO"
        ),
        "coordinate_boundary": (
            "euclidean_to_euclidean"
            if variant == "E"
            else workload.get("coordinate_boundary", "tangent_to_tangent")
        ),
        "operation_boundary": (
            f"{workload['kind']}:{workload['mode']}:depth={workload.get('depth', 1)}"
        ),
        "warmup_count": int(config["warmup_calls"]),
        "repeat_count": int(config["timed_calls_per_session"]),
        "compiler_mode": "jax.jit lowered+compiled before steady-state timing",
        "synchronization_method": "block_until_ready on every output/state leaf",
        "output_retention_policy": (
            "dereference each completed output before the next invocation; retain "
            "only the final steady-state output for finite-value validation"
        ),
        "state_evolves_across_calls": workload["mode"] == "update",
        "state_summary_before": canonical_json(state_before),
        "state_summary_after": canonical_json(state_after),
        "output_nonfinite_count": output_nonfinite_count,
        "sustained_sequence_call_count": sustained_calls,
        "sustained_compilation_ms": (
            sustained["compilation_ms"] if sustained is not None else None
        ),
        "sustained_first_call_ms": (
            sustained["first_call_ms"] if sustained is not None else None
        ),
        "sustained_output_nonfinite_count": (
            sustained["output_nonfinite_count"] if sustained is not None else None
        ),
        "throughput_calls_per_second": None,
        "measured_peak_phase": (
            "process lifetime through final synchronized steady-state output; "
            "includes initialization, compilation, and warmup; excludes post-run "
            "state summary"
        ),
        "measured_peak_bytes_in_use": memory_stats.get("peak_bytes_in_use"),
        "measured_peak_pool_bytes": memory_stats.get("peak_pool_bytes"),
        "process_reservation_mib": process_reservation_mib,
        **_memory_analysis(compiled),
    }
    rows = [
        {
            **shared,
            "observation": -2,
            "phase": "compilation",
            "latency_ms": compilation_ms,
        },
        {
            **shared,
            "observation": -1,
            "phase": "first_compiled_call",
            "latency_ms": first_call_ms,
        },
        {
            **shared,
            "status": (
                "not_applicable"
                if sustained is None
                else "nonfinite"
                if sustained["output_nonfinite_count"]
                else "ok"
            ),
            "observation": -3,
            "phase": "sustained_sequence",
            "latency_ms": sustained["latency_ms"] if sustained is not None else None,
            "throughput_calls_per_second": (
                sustained["calls_per_second"] if sustained is not None else None
            ),
        },
    ]
    rows.extend(
        {
            **shared,
            "observation": index,
            "phase": "steady_state",
            "latency_ms": latency,
        }
        for index, latency in enumerate(latencies)
    )
    write_json(output_path, {"rows": rows})
    _write_worker_phase(phase_status_path, "complete")


def _selected_physical_gpu() -> int | None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        return None
    first = visible.split(",", maxsplit=1)[0].strip()
    return int(first) if first.isdigit() else None


def _gpu_snapshot(index: int | None) -> dict[str, Any]:
    """Capture an auditable physical-GPU occupancy snapshot."""

    if index is None:
        return {
            "status": "unavailable",
            "reason": "CUDA_VISIBLE_DEVICES does not start with a physical GPU index",
        }
    fields = (
        "index",
        "uuid",
        "name",
        "memory.used",
        "memory.total",
        "utilization.gpu",
        "temperature.gpu",
        "power.draw",
    )
    gpu_result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={index}",
            f"--query-gpu={','.join(fields)}",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if gpu_result.returncode != 0:
        return {
            "status": "failed",
            "index": index,
            "stderr": gpu_result.stderr.strip(),
        }
    values = [value.strip() for value in gpu_result.stdout.strip().split(",")]
    if len(values) != len(fields):
        return {
            "status": "failed",
            "index": index,
            "raw_gpu_query": gpu_result.stdout.strip(),
        }
    snapshot: dict[str, Any] = dict(zip(fields, values, strict=True))
    snapshot.update(
        {
            "status": "ok",
            "index": int(snapshot["index"]),
            "memory.used": int(snapshot["memory.used"]),
            "memory.total": int(snapshot["memory.total"]),
            "utilization.gpu": int(snapshot["utilization.gpu"]),
        }
    )
    process_result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    processes = []
    if process_result.returncode == 0:
        for line in process_result.stdout.splitlines():
            process = [value.strip() for value in line.split(",", maxsplit=3)]
            if len(process) == 4 and process[0] == snapshot["uuid"]:
                processes.append(
                    {
                        "gpu_uuid": process[0],
                        "pid": int(process[1]),
                        "process_name": process[2],
                        "used_memory_mib": int(process[3]),
                    }
                )
    snapshot["compute_processes"] = processes
    snapshot["process_query_status"] = "ok" if process_result.returncode == 0 else "failed"
    return snapshot


def _require_idle_gpu(config: dict[str, Any]) -> dict[str, Any]:
    """Wait for telemetry to settle or reject a contaminated final run."""

    gpu_index = _selected_physical_gpu()
    deadline = time.monotonic() + GPU_IDLE_SETTLE_TIMEOUT_SECONDS
    require_idle = bool(config.get("require_idle_gpu", False))
    memory_limit = int(config["max_preexisting_memory_mib"]) if require_idle else 0
    utilization_limit = (
        int(config["max_preexisting_utilization_percent"]) if require_idle else 0
    )
    while True:
        snapshot = _gpu_snapshot(gpu_index)
        if not require_idle:
            return snapshot
        if snapshot["status"] != "ok" or snapshot["process_query_status"] != "ok":
            raise RuntimeError(f"could not establish GPU idleness: {snapshot}")
        if snapshot["compute_processes"]:
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "selected GPU compute processes did not clear within "
                    f"{GPU_IDLE_SETTLE_TIMEOUT_SECONDS:g} seconds; "
                    "refusing final measurement"
                )
            time.sleep(GPU_IDLE_POLL_INTERVAL_SECONDS)
            continue
        memory_used = int(snapshot["memory.used"])
        utilization = int(snapshot["utilization.gpu"])
        if memory_used <= memory_limit and utilization <= utilization_limit:
            return snapshot
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "selected GPU telemetry did not settle within "
                f"{GPU_IDLE_SETTLE_TIMEOUT_SECONDS:g} seconds "
                f"(memory={memory_used} MiB, utilization={utilization}%); "
                "refusing final measurement"
            )
        time.sleep(GPU_IDLE_POLL_INTERVAL_SECONDS)


def run_benchmark_suite(
    protocol_path: str | Path,
    config_path: str | Path,
    output_path: str | Path,
    *,
    max_workloads: int | None = None,
) -> int:
    """Coordinate paired fresh-process variant sessions and append raw CSV rows."""

    protocol = load_yaml(protocol_path)
    config = load_yaml(config_path)
    require_keys(
        config,
        (
            "experiment_id",
            "variants",
            "sessions",
            "warmup_calls",
            "timed_calls_per_session",
            "workloads",
        ),
        name="benchmark config",
    )
    workloads = config["workloads"][:max_workloads]
    output = Path(output_path).expanduser().resolve()
    failure_output = output.with_name(f"{output.name}.failures.jsonl")
    expected_rows = int(config["timed_calls_per_session"]) + 3
    existing_counts: dict[tuple[str, int, str], int] = {}
    failed_groups: set[tuple[str, int, str]] = set()
    if output.exists():
        with output.open("r", encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                key = (row["workload_id"], int(row["seed_or_session"]), row["variant"])
                existing_counts[key] = existing_counts.get(key, 0) + 1
    if failure_output.exists():
        with failure_output.open("r", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                failed_groups.add(
                    (row["workload_id"], int(row["seed_or_session"]), row["variant"])
                )
    randomizer = random.Random(int(protocol["seeds"]["measurement_order"]))
    row_count = 0
    for workload in workloads:
        for session in range(int(config["sessions"])):
            order = list(config["variants"])
            randomizer.shuffle(order)
            for variant in order:
                existing_key = (workload["id"], session, variant)
                if existing_counts.get(existing_key, 0) == expected_rows:
                    continue
                if existing_key in failed_groups:
                    continue
                if existing_counts.get(existing_key, 0) != 0:
                    raise RuntimeError(
                        f"partial benchmark group requires audit: {existing_key}"
                    )
                snapshot_before = _require_idle_gpu(config)
                with tempfile.TemporaryDirectory(prefix="poincare-benchmark-") as directory:
                    worker_output = Path(directory) / "worker.json"
                    phase_status = Path(directory) / "worker-phase.json"
                    command = [
                        sys.executable,
                        "-m",
                        "poincare_exec.cli",
                        "benchmark-worker",
                        "--protocol",
                        str(protocol_path),
                        "--config",
                        str(config_path),
                        "--workload-id",
                        workload["id"],
                        "--variant",
                        variant,
                        "--session",
                        str(session),
                        "--output",
                        str(worker_output),
                        "--phase-status",
                        str(phase_status),
                    ]
                    result = subprocess.run(
                        command, check=False, capture_output=True, text=True
                    )
                    snapshot_after = _gpu_snapshot(_selected_physical_gpu())
                    if result.returncode != 0:
                        git = git_provenance()
                        experiment_id = _workload_experiment_id(config, workload)
                        append_jsonl(
                            failure_output,
                            [
                                {
                                    "experiment_id": experiment_id,
                                    "protocol_version": protocol["protocol_version"],
                                    "variant": variant,
                                    "config_hash": config_hash(protocol, config, workload),
                                    "code_commit": git["commit"],
                                    "code_dirty": git["dirty"],
                                    "reference_commit": canonical_json(
                                        protocol["reference_commits"]
                                    ),
                                    "device_id": _logical_gpu_device_identity(
                                        snapshot_before
                                    ),
                                    "precision_policy": precision_policy(
                                        str(workload["dtype"])
                                    ),
                                    "seed_or_session": session,
                                    "status": "failed",
                                    "timestamp_utc": timestamp_utc(),
                                    "workload_id": workload["id"],
                                    "shape": workload["shape"],
                                    "operation_boundary": (
                                        f"{workload['kind']}:{workload['mode']}"
                                    ),
                                    "returncode": result.returncode,
                                    "failure_kind": _classify_worker_failure(
                                        stdout=result.stdout,
                                        stderr=result.stderr,
                                        returncode=result.returncode,
                                    ),
                                    "failure_phase": _read_worker_phase(phase_status),
                                    "gpu_snapshot_before": snapshot_before,
                                    "gpu_snapshot_after": snapshot_after,
                                    "stdout": result.stdout,
                                    "stderr": result.stderr,
                                }
                            ],
                        )
                        row_count += 1
                        continue
                    rows = json.loads(worker_output.read_text(encoding="utf-8"))["rows"]
                    for row in rows:
                        row["gpu_snapshot_before"] = canonical_json(snapshot_before)
                        row["gpu_snapshot_after"] = canonical_json(snapshot_after)
                    append_csv_rows(output, rows)
                    row_count += len(rows)
    return row_count
