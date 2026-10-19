"""Paired latency and memory benchmark: HypLL HConvolution2d versus the factorized module.

Protocol (as in the paper): every workload has 5 sessions; in each session the two
variants run in randomized order, each in a fresh process, with 50 warmup and 200 timed
calls. Each call ends with torch.cuda.synchronize(); a session contributes its median.
FP32, TF32 disabled, cudnn.benchmark left at its default (False).

    python benchmark.py run --gpu 1 --run-dir RUN --unet-src Hyperbolic-U-Net   # driver
    python benchmark.py summarize RUN/sessions.jsonl --prefix OUT/e32_pytorch    # tables
The driver launches `python benchmark.py worker ...` for each session.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

SHAPES = {"S1": (16, 16, 256, 256), "S2": (256, 256, 32, 32), "S3": (256, 256, 96, 32)}
WORKLOADS = [f"{s}_{m}" for s in SHAPES for m in ("forward", "backward")]
WORKLOADS += ["unet_b8_forward", "unet_b8_update"]
VARIANTS = ["hypll", "factorized"]
SESSIONS, WARMUP, TIMED, ORDER_SEED = 5, 50, 200, 314159
CURVATURE, BATCH = 0.1, 8
T_975_DF4 = 2.7764451051977987  # Student t quantile, 0.975, 4 degrees of freedom
PROTOCOL = {
    "sessions_per_workload": SESSIONS,
    "warmup_calls": WARMUP,
    "timed_calls": TIMED,
    "variant_order": "shuffled per workload and session by random.Random(314159); "
    "each variant of a session runs in a fresh process",
    "session_seed": "314159 + session, shared by both variants",
    "latency": "time.perf_counter() around each call, which ends with "
    "torch.cuda.synchronize(); a session contributes its median",
    "speedup": "geometric mean of session-median ratios hypll/factorized; "
    "95% log-t interval with df = 4",
    "steady_state_peak": "torch.cuda.max_memory_allocated() over the timed calls, after "
    "reset_peak_memory_stats() following warmup",
    "process_peak": "largest torch.cuda.max_memory_allocated() over the whole process, "
    "including setup and warmup",
    "precision": "FP32; torch.backends.cuda.matmul.allow_tf32 = False; "
    "torch.backends.cudnn.allow_tf32 = False; cudnn.benchmark left at its default",
    "curvature": CURVATURE,
    "batch": BATCH,
}


def gpu_apps():
    out = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [line.split(", ") for line in out.strip().splitlines() if line.strip()]


def gpu_info(index):
    out = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(index),
            "--query-gpu=uuid,name,driver_version,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    uuid, name, driver, used, util = out.split(", ")
    return {
        "uuid": uuid,
        "name": name,
        "driver": driver,
        "memory_used_mib": int(used),
        "utilization_percent": int(util),
    }


# ---------------------------------------------------------------- worker (one session)


def build(workload, variant, seed, unet_src):
    import torch
    from factorized_hconv import factorize_hconvs
    from hypll.manifolds.poincare_ball import Curvature, PoincareBall
    from hypll.manifolds.poincare_ball.math.diffgeom import expmap0
    from hypll.nn import HConvolution2d
    from hypll.tensors import ManifoldTensor
    from torch import nn

    theta = math.log(math.expm1(CURVATURE))  # HypLL curvature is softplus(theta)
    torch.manual_seed(seed)
    if workload.startswith("unet"):
        sys.path.insert(0, unet_src)
        from hypll.optim import RiemannianAdam
        from losses import MULTICLASS_MODE, DiceLoss, FocalLoss
        from unet.hyperbolic_unet.model import HUNet

        model = HUNet(
            n_channels=3,
            n_classes=2,
            bilinear=False,
            curvature=theta,
            trainable=True,
            init_feats=8,
            depth=4,
        )
        if variant == "factorized":
            factorize_hconvs(model)
        model = model.to(memory_format=torch.channels_last).cuda()
        torch.manual_seed(seed + 1)
        images = torch.rand(BATCH, 3, 256, 256, device="cuda")
        images = images.to(memory_format=torch.channels_last)
        masks = (torch.rand(BATCH, 256, 256, device="cuda") > 0.5).long()
        if workload == "unet_b8_forward":

            def call():
                with torch.no_grad():
                    return model(images)

            return call
        optimizer = RiemannianAdam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        dice = DiceLoss(MULTICLASS_MODE, from_logits=True)
        focal = FocalLoss(MULTICLASS_MODE, alpha=0.25, gamma=2.0)

        def call():  # the update of Hyperbolic-U-Net/train.py without AMP and logging
            logits = model(images).contiguous()
            loss = dice(logits, masks) + focal(logits, masks)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            return loss.detach()

        return call

    shape, mode = workload.split("_")
    height, width, cin, cout = SHAPES[shape]
    manifold = PoincareBall(Curvature(theta, requires_grad=True))
    layer = HConvolution2d(cin, cout, 3, manifold, padding=1)  # HypLL init, bias=True
    if variant == "factorized":
        layer = factorize_hconvs(nn.ModuleList([layer]))[0]
    layer.cuda()
    torch.manual_seed(seed + 1)
    tangent = 0.05 * torch.randn(BATCH, cin, height, width, device="cuda")
    x = expmap0(tangent, manifold.c().detach(), dim=1).requires_grad_(mode == "backward")
    if mode == "forward":

        def call():
            with torch.no_grad():
                return layer(ManifoldTensor(x, manifold=manifold, man_dim=1)).tensor

        return call
    leaves = [x, layer.weights.tensor, layer.bias, manifold.c.value]

    def call():  # gradient of the mean squared output, as in the paper's JAX benchmark
        y = layer(ManifoldTensor(x, manifold=manifold, man_dim=1)).tensor
        return torch.autograd.grad(y.square().mean(), leaves)

    return call


def worker(args):
    import hypll
    import numpy
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    call = build(args.workload, args.variant, ORDER_SEED + args.session, args.unet_src)
    for _ in range(WARMUP):
        out = call()
        del out
    torch.cuda.synchronize()
    warmup_peak = torch.cuda.max_memory_allocated()
    warmup_reserved = torch.cuda.max_memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(TIMED):
        start = time.perf_counter()
        out = call()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
        del out  # dereference before the next call
    steady_peak = torch.cuda.max_memory_allocated()
    steady_reserved = torch.cuda.max_memory_reserved()
    smi = {pid: mib for _, pid, mib in gpu_apps()}
    print(
        json.dumps(
            {
                "workload": args.workload,
                "variant": args.variant,
                "session": args.session,
                "median_ms": 1e3 * statistics.median(times),
                "times_s": times,
                "steady_peak_allocated_bytes": steady_peak,
                "process_peak_allocated_bytes": max(warmup_peak, steady_peak),
                "process_peak_reserved_bytes": max(warmup_reserved, steady_reserved),
                "nvidia_smi_process_mib_at_end": int(smi.get(str(os.getpid()), -1)),
                "env": {
                    "python": platform.python_version(),
                    "torch": torch.__version__,
                    "cuda_runtime": torch.version.cuda,
                    "cudnn": torch.backends.cudnn.version(),
                    "hypll": getattr(hypll, "__version__", None) or _dist_version("hypll"),
                    "numpy": numpy.__version__,
                    "device": torch.cuda.get_device_name(0),
                    "cudnn_benchmark": torch.backends.cudnn.benchmark,
                    "cudnn_deterministic": torch.backends.cudnn.deterministic,
                    "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                    "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                },
            }
        )
    )


def _dist_version(name):
    from importlib.metadata import version

    return version(name)


# ---------------------------------------------------------------- driver


def driver(args):
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log = run_dir / "sessions.jsonl"
    done = set()
    if log.exists():
        for line in log.read_text().splitlines():
            r = json.loads(line)
            if r.get("returncode") == 0:
                done.add((r["workload"], r["session"], r["variant"]))
    target = gpu_info(args.gpu)
    workloads = args.workloads or WORKLOADS
    order = random.Random(ORDER_SEED)
    launch = 0
    for workload in WORKLOADS:  # the shuffle sequence does not depend on --workloads
        for session in range(SESSIONS):
            variants = list(VARIANTS)
            order.shuffle(variants)
            for variant in variants:
                launch += 1
                if workload not in workloads or (workload, session, variant) in done:
                    continue
                apps, info = gpu_apps(), gpu_info(args.gpu)
                if (
                    any(app[0] == target["uuid"] for app in apps)
                    or info["memory_used_mib"] > 512
                ):
                    sys.exit(
                        f"GPU {args.gpu} ({target['uuid']}) is not idle: {apps} {info}"
                    )
                env = dict(
                    os.environ,
                    CUDA_DEVICE_ORDER="PCI_BUS_ID",
                    CUDA_VISIBLE_DEVICES=str(args.gpu),
                    OMP_NUM_THREADS="4",
                )
                cmd = [
                    "nice",
                    "-n",
                    "19",
                    sys.executable,
                    __file__,
                    "worker",
                    "--workload",
                    workload,
                    "--variant",
                    variant,
                    "--session",
                    str(session),
                    "--unet-src",
                    args.unet_src,
                ]
                proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
                record = {
                    "workload": workload,
                    "variant": variant,
                    "session": session,
                    "launch_index": launch,
                    "returncode": proc.returncode,
                    "gpu_index": args.gpu,
                    "gpu_uuid": target["uuid"],
                    "gpu_name": target["name"],
                    "driver": target["driver"],
                    "compute_apps_before_launch": apps,
                    "gpu_before_launch": info,
                }
                if proc.returncode == 0:
                    record.update(json.loads(proc.stdout.strip().splitlines()[-1]))
                else:
                    record["stderr_tail"] = proc.stderr[-4000:]
                with log.open("a") as handle:
                    handle.write(json.dumps(record) + "\n")
                median = record.get("median_ms", float("nan"))
                print(
                    f"[{launch:3d}] {workload:16s} s{session} {variant:10s} "
                    f"rc={proc.returncode} median_ms={median:.3f}",
                    flush=True,
                )


# ---------------------------------------------------------------- summary (pure Python)


def summarize(args):
    launches = [json.loads(line) for line in Path(args.sessions).read_text().splitlines()]
    rows = [r for r in launches if r["returncode"] == 0]
    gib = 2.0**30
    session_rows, summary_rows = [], []
    for workload in WORKLOADS:
        by = {(r["variant"], r["session"]): r for r in rows if r["workload"] == workload}
        failed = sum(r["workload"] == workload and r["returncode"] != 0 for r in launches)
        if not by and not failed:
            continue
        for (variant, session), r in sorted(
            by.items(), key=lambda kv: (kv[0][1], kv[0][0])
        ):
            session_rows.append(
                {
                    "workload": workload,
                    "variant": variant,
                    "session": session,
                    "launch_index": r["launch_index"],
                    "median_ms": r["median_ms"],
                    "min_ms": 1e3 * min(r["times_s"]),
                    "max_ms": 1e3 * max(r["times_s"]),
                    "steady_peak_allocated_gib": r["steady_peak_allocated_bytes"] / gib,
                    "process_peak_allocated_gib": r["process_peak_allocated_bytes"] / gib,
                    "process_peak_reserved_gib": r["process_peak_reserved_bytes"] / gib,
                    "nvidia_smi_process_mib_at_end": r["nvidia_smi_process_mib_at_end"],
                }
            )
        paired = sorted(s for v, s in by if v == "hypll" and ("factorized", s) in by)
        logs = [
            math.log(by["hypll", s]["median_ms"] / by["factorized", s]["median_ms"])
            for s in paired
        ]
        row = {
            "workload": workload,
            "paired_sessions": len(paired),
            "failed_launches": failed,
        }
        for v in VARIANTS:
            vs = [by[v, s] for s in range(SESSIONS) if (v, s) in by]
            if not vs:
                continue
            row[f"{v}_median_ms"] = statistics.median(r["median_ms"] for r in vs)
            for key in ("steady_peak_allocated", "process_peak_allocated"):
                row[f"{v}_{key}_gib"] = (
                    statistics.median(r[f"{key}_bytes"] for r in vs) / gib
                )
        if len(logs) >= 2:
            mean, half = (
                statistics.mean(logs),
                T_975_DF4 * statistics.stdev(logs) / math.sqrt(len(logs)),
            )
            if len(logs) != 5:
                half = math.nan  # the hard-coded t quantile assumes 4 degrees of freedom
            row.update(
                speedup=math.exp(mean),
                speedup_ci_low=math.exp(mean - half),
                speedup_ci_high=math.exp(mean + half),
            )
        for key in ("steady_peak_allocated", "process_peak_allocated"):
            if f"hypll_{key}_gib" in row and f"factorized_{key}_gib" in row:
                ratio = row[f"hypll_{key}_gib"] / row[f"factorized_{key}_gib"]
                row[f"{key}_ratio_hypll_over_factorized"] = ratio
        summary_rows.append(row)
    write_csv(Path(f"{args.prefix}_sessions.csv"), session_rows)
    write_csv(Path(f"{args.prefix}_latency_memory.csv"), summary_rows)
    gpus = {(r["gpu_name"], r["gpu_uuid"], r["driver"], r["gpu_index"]) for r in launches}
    summary = {
        "measurement_commit": args.source_commit,
        "hyperbolic_unet_commit": args.unet_commit,
        "environment": [
            json.loads(e) for e in sorted({json.dumps(r["env"]) for r in rows})
        ],
        "gpu": [
            dict(zip(("name", "uuid", "driver", "index"), g, strict=True)) for g in gpus
        ],
        "protocol": PROTOCOL,
        "workloads": summary_rows,
        "inputs": {"sessions_jsonl_sha256": sha256(args.sessions)},
    }
    if args.equivalence:
        summary["equivalence"] = equivalence_digest(args.equivalence)
        summary["inputs"]["equivalence_csv_sha256"] = sha256(args.equivalence)
    Path(f"{args.prefix}_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary_rows, indent=1))


def equivalence_digest(path):
    """Largest finite errors and criterion passes per case group and dtype."""
    columns = ["rel_l2_error", "max_abs_error", "ref_rel_l2_vs_fp64", "fac_rel_l2_vs_fp64"]
    digest = {}
    with open(path, newline="") as handle:
        for r in csv.DictReader(handle):
            group = "S1-S3 and stride 2" if r["regime"] == "smooth" else r["case"]
            d = digest.setdefault(group, {}).setdefault(r["dtype"], {"comparisons": 0})
            d["comparisons"] += 1
            d["paper_criterion_pass"] = d.get("paper_criterion_pass", 0) + (
                r["paper_criterion_pass"] == "True"
            )
            for key in ("nonfinite_ref", "nonfinite_mismatch"):
                d[key] = d.get(key, 0) + int(r[key])
            for key in columns:
                value = float(r.get(key) or "nan")
                if math.isfinite(value):
                    d[f"largest_{key}"] = max(d.get(f"largest_{key}", 0.0), value)
    return digest


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run")
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--unet-src", required=True)
    p.add_argument("--workloads", nargs="*", choices=WORKLOADS)
    p = sub.add_parser("worker")
    p.add_argument("--workload", choices=WORKLOADS, required=True)
    p.add_argument("--variant", choices=VARIANTS, required=True)
    p.add_argument("--session", type=int, required=True)
    p.add_argument("--unet-src", required=True)
    p = sub.add_parser("summarize")
    p.add_argument("sessions")
    p.add_argument("--prefix", required=True)
    p.add_argument("--equivalence", help="equivalence CSV of check_equivalence.py")
    p.add_argument("--source-commit", help="commit of the measured code")
    p.add_argument("--unet-commit", help="commit of the Hyperbolic U-Net sources")
    args = parser.parse_args()
    {"run": driver, "worker": worker, "summarize": summarize}[args.command](args)


if __name__ == "__main__":
    main()
