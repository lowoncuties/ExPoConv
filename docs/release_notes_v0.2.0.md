# Release notes: v0.2.0 (2026-10-19)

Version 0.2.0 accompanies the revised manuscript *Efficient Poincaré
Convolutions through Patch-Statistics Factorization* (L. Jochymek, 2026). It
adds the revision experiments E22--E35 and three exports from existing runs to
v0.1.0 (commit `33f991ad34a0`). Accepted v0.1.0 results are unchanged.

## Revision experiments

Run commands use the GPU environment and `W=./scripts/with_cuda_11_8.sh` from
`docs/reproducibility.md` (Sections 2 and 4). Export commands validate the
archived raw outputs of the accepted runs on CPU. Their output is in
`results/reviewer_requested/`.

| ID | What it is | Run | Export |
|---|---|---|---|
| E22 | V100 allocator-fraction rerun at fraction 0.95: batch 8 at 256 x 256 and 512 x 512, batch 16 at 512 x 512, R/F, five sessions; all 30 groups retained (15 completed, 15 first-update OOMs) | `$W ./scripts/run_e22.sh` | `python -m scripts.export_e22_allocator_fraction --success RAW.csv --failures RAW.csv.failures.jsonl` |
| E23 | V100 batch-one U-Net forward memory diagnostic with default and disabled (`--xla_gpu_autotune_level=0`) convolution autotuning | `$W ./scripts/run_e23.sh default` and `$W ./scripts/run_e23.sh autotune0` | `python -m scripts.export_e23_autotune_memory --default DEFAULT.csv --autotune0 AUTOTUNE0.csv` |
| E24 | A6000 materialized-patch ablation M: R/M/F at batch-1 and batch-8 U-Net forward and batch-8 update, pinned JAX 0.4.25 | `$W ./scripts/run_e24.sh` | `python scripts/export_e24_e25_e27_tables.py --artifact-dir RAW_DIR --output-dir OUT` |
| E25 | A6000 same-topology Euclidean reference E: E/R/F at three U-Net and three ResNet-20 boundaries, pinned JAX 0.4.25 | `$W ./scripts/run_e25.sh` | same exporter as E24; figure: `python -m scripts.plot_e25_cost_context --validated-dir results/reviewer_requested --output-stem OUT/e25_cost_context` |
| E26 | V100 structured elementwise agreement replay (2,720 tensors) with relative-error quantiles | `$W ./.venv/bin/poincare-exec structured-numerics --protocol protocol.yaml --config configs/numerics/e26_elementwise_agreement.yaml --output artifacts/e26_elementwise_agreement.jsonl` | `python scripts/export_e26_elementwise_tables.py --input RAW.jsonl` |
| E27 | A6000 batch-8 U-Net update R/F check under JAX/JAXLIB 0.11.2 and the CUDA-12 plugin, reported separately | `./scripts/run_e27.sh` (separate environment, see `docs/reproducibility.md` Section 2) | same exporter as E24 |
| E28 | V100 replay stability: the affected 408-tensor FP32 prefix of E26 in three fresh processes | `$W ./scripts/run_e28.sh` | `python -m scripts.export_e28_replay_stability --replay-0 R0.jsonl --replay-1 R1.jsonl --replay-2 R2.jsonl` |
| E29 | V100 (GPU-a37e9fd0) Euclidean reference E alone on the three E08 and seven E13 U-Net workloads under the E08/E13 allocator settings; batch-8 update 23.50 ms over the 4 of 5 sessions without a driver stall (all-session sensitivity 25.25 ms) | `$W ./scripts/run_e29.sh` | `python -m scripts.export_e29_e30_tables --artifact-dir RAW_DIR --output-dir OUT`; figure: `python -m scripts.plot_scaling_with_euclidean --output-stem OUT/e29_unet_update_scaling` |
| E30 | V100 (GPU-4e0547c3) paired R/F batch-8 update at U-Net initial widths 16 and 32; width 16: speedup 2.04 [2.01, 2.06] over 3 stall-free pairs (all-pairs sensitivity 2.00 [1.94, 2.06]) and 56% lower F peak; width 32: both paths out of memory in 5/5 sessions | `$W ./scripts/run_e30.sh` | same exporter as E29 |
| E31 | A6000 gather-based reference controls: R/Rg/Mg/F in paired sessions on three U-Net boundaries and the ResNet-20 update, pinned JAX 0.4.25, plus a GPU equivalence check of R/Rg and M/Mg | `$W ./scripts/run_e31.sh`; `python -m scripts.check_e31_gpu_equivalence --output RAW_DIR/e31_gpu_equivalence.json`; `python -m scripts.capture_e31_compiled_memory --output RAW_DIR/e31_compiled_memory.json` | `python -m scripts.export_e31_gather_reference --artifact-dir RAW_DIR --output-dir OUT` |
| E32 | PyTorch cross-check on the A6000: factorized replacement for HypLL 0.1.1 `HConvolution2d`, output/gradient agreement and paired latency/memory on S1--S3 and the upstream Hyperbolic U-Net | in `pytorch_crosscheck/` with its PyTorch environment: `python check_equivalence.py --output equivalence.csv`; `python benchmark.py run --gpu 1 --run-dir RUN --unet-src Hyperbolic-U-Net` | `python benchmark.py summarize RUN/sessions.jsonl --prefix OUT/e32_pytorch` |
| E33 | CPU (TFRT_CPU_0) diagnostics: FP32 R/F/T accuracy against a fixed-margin FP64 oracle at three curvatures, branch trace of the small-U-Net realization behind the E02/E26 misses, and source-radius cap activity in the final E09/E10 checkpoints | `python scripts/run_oracle_accuracy.py --output-dir OUT`; `python scripts/trace_branch_margins.py --output-dir OUT`; `EXPOCONV_TRAINING_ROOT=RUNS EXPOCONV_DATA_ROOT=data/raw python scripts/measure_cap_activity.py --output-dir OUT` | written directly by the run commands |
| E35 | V100 FP64-oracle accuracy at the three retained E07-J2 ResNet-20 states: FP32 R/F/T gradients against R in FP64 with the FP32 margin | `CUDA_VISIBLE_DEVICES=1 ./scripts/with_cuda_11_8.sh ./scripts/run_e35.sh` | `python scripts/export_e35_retained_oracle.py --artifact-dir artifacts/e35_retained_oracle` |
| Curvature | Learned ISIC16 curvature trajectories and final values for E09 and E20 R/F seeds | none (existing training logs) | `python scripts/export_reviewer_requested_analysis.py --e09-training-root E09 --e20-training-root E20 --accepted-structured-results E02.jsonl` |
| TOST | Post-hoc two one-sided tests of R/F quality (1 Dice point, 0.5 accuracy points) for E09/E10/E20/E21 | none (compact training tables) | same exporter as Curvature |
| Training minutes | Absolute per-seed compute and elapsed minutes for R/F in E09/E10/E20/E21 | none (compact training tables) | same exporter as Curvature |

`python` stands for `./.venv/bin/python`, run from the repository root. E22,
E23, and E26 were run at commit `30142ce2` under protocol 1.15; E24, E25, E27,
and E28 at commit `df3f6f55` under protocol 1.17; E29 and E30 at `20a886e4`
under protocol 1.18; E31 at `486b5bd2` under protocol 1.19; E35 at `e98d313f` under protocol 1.20. E32 (measurement
commit `df03198a`) and E33 (commits `620b0bfb`, `55e9c8a2`, `768301c7`) are
post-hoc diagnostics that were not registered in `protocol.yaml`. The exporters
accept only the recorded identities, so a new run must declare its own.

E29 and E30 ran concurrently on the two V100s of one host (E29 on GPU-a37e9fd0,
E30 on GPU-4e0547c3) while the host recorded driver stalls. A rule fixed
before any affected latency was inspected excludes every group whose execution
window overlapped a stall from the primary medians and contrasts; all-session
values are reported beside them as a sensitivity (`docs/reproducibility.md`,
Section 9). For E31, the R/Mg contrast is reported in addition to the five
registered contrasts, and the compiled buffer-assignment bytes, which are zero
in this run, are not reported.

## Other changes

- Title and package metadata use "patch-statistics factorization".
- `protocol.yaml` records amendments 1.15--1.19, which register E22--E31.
  Benchmark-only paths M, E, Rg, and Mg are added for E24, E25, and E31, and an
  optional U-Net `initial_width` workload key for E30. The R/F/T computations
  are unchanged apart from formatting and one shared helper, and every
  configuration file of v0.1.0 is byte-identical.
- Byte-identical protocol snapshots `configs/protocols/protocol_1_14.yaml`,
  `protocol_1_17.yaml`, `protocol_1_18.yaml`, and `protocol_1_19.yaml` keep the
  E20/E21, E24/E25/E27, E29/E30, and E31 configuration hashes verifiable after
  later amendments.
- `pytorch_crosscheck/` adds the PyTorch implementation used by E32.
- `results/claims.csv` adds claims C16--C28 and maps every claim to the
  sections, tables, and figures of the revised manuscript.
- `provenance/v100_accepted_history_manifest.json` lists the E22--E35
  revisions as release-history commits.
- The README follows the companion-repository layout. Detailed protocol,
  provenance, release-package, and E19 rerun instructions, and the definitions
  of the revision analyses, are merged into `docs/reproducibility.md`.
- Removed files that a reader does not need to reproduce the paper: planning
  and status notes (`docs/revision_r1_response_plan.md`,
  `docs/release_scope_disposition.md`), the optional E19 handoff runbook and
  its blank form (`docs/e19_independent_reproduction.md`,
  `docs/e19_reproduction_record_TEMPLATE.md`), `docs/reviewer_requested_analysis.md`
  and `provenance/README.md` (merged), runners nothing referenced
  (`scripts/run_e06_trained_checkpoint.sh`,
  `scripts/run_journal_retained_diagnostics_v2.sh`), the scripts of the unrun
  E12 study and its CUDA-12/JAX 0.5.3 environment (`scripts/run_e12.sh`,
  `scripts/bootstrap_cuda12_a6000.sh`, `scripts/cuda12_a6000_requirements.txt`),
  and the superseded five-seed training figure with its script
  (`results/training/training_efficiency_quality.{pdf,png,svg}`,
  `scripts/plot_training_efficiency_quality.py`).
- Removed unreferenced helpers (`dimension_beta`, `as_real_array`, `asinhc`,
  `count_parameters`, `normalization_parameters`, the E02 `_dtype` duplicate,
  `BENCHMARK_REQUIRED_FIELDS`) and applied `ruff format` to the Python
  sources (layout only; no executed code changes).
- `scripts/with_cuda_11_8.sh` reads the system CUDA 11.8 library directory
  from `EXPOCONV_CUDA_TOOLKIT_LIB`; the default is unchanged.
  `scripts/check_cuda.sh` finds its default `.venv` from the repository root
  instead of the working directory.

## Fixes

- The frozen E20/E21 analysis manifests pin the protocol-1.14 bytes of
  `protocol.yaml`, which no longer matched after amendments 1.15--1.19. The
  byte-identical snapshot `configs/protocols/protocol_1_14.yaml` is added, and
  the hashed-document validators use `configs/protocols/protocol_<version>.yaml`
  when the top-level file no longer matches a versioned pin and the snapshot's
  SHA-256 equals it. The E20/E21 exporter, the checkpoint-inventory exporter,
  and the frozen-manifest test validate again; no recorded result, hash, or
  hash-bound file changes (`docs/reproducibility.md`, Sections 5 and 8).
- The E24/E25/E27, E29/E30, and E31 exporters read the protocol snapshot that
  their recorded configuration hashes bind, so later amendments do not break
  them.
- `results/SHA256SUMS` lists every file in `results/` again; the `claims.csv`
  entry and the E22--E33 files were missing or stale.
- `scripts/measure_cap_activity.py` reads the archived training runs and the
  datasets from `EXPOCONV_TRAINING_ROOT` and `EXPOCONV_DATA_ROOT` (defaults
  `artifacts/training` and `data/raw`) instead of fixed host paths.
- `.venv-jax0112-cuda12/`, the default E27 environment of `scripts/run_e27.sh`,
  is ignored by Git so that it does not mark a rerun as dirty.
