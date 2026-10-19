# Efficient Poincaré Convolutions through Patch-Statistics Factorization

This repository contains the source code, registered experiment protocol and
configurations, data-split manifests, experiment runners, and compact result
tables associated with the manuscript:

Jochymek L.  
**Efficient Poincaré Convolutions through Patch-Statistics Factorization.**  
(2026, manuscript submitted)

This package is intended to make every reported measurement traceable to its
code, configuration, and result file, and to allow the experiments to be rerun.

---

## 1. Overview

A literal HNN++ Poincaré convolution expands every receptive-field patch, maps
it to the Poincaré ball, and evaluates a hyperplane score. The score depends on
the tangent-space patch only through its squared norm (patch energy) and its
inner products with the filters (directional products). A native convolution
and a scalar spatial reduction therefore compute the same clipped score without
ball-valued patches.

The compared execution paths share parameters, clipping, normalization,
curvature, data order, and optimizer:

| Path | Execution | Role |
|---|---|---|
| **R** | materializes the beta-scaled tangent patch and maps it to the ball | reference |
| **F** | directional products by native convolution plus scalar patch energy | primary method |
| **T** | F with direct evaluation of the projected origin-tangent output | ablation |
| **M** | R's explicit patch, scored from its patch statistics without the pointwise ball map | mechanism control (benchmarks only) |
| **Rg**, **Mg** | R and M with the patch built by gathering shifted slices, as the official HNN++ code does | gather-based reference controls (benchmarks only) |
| **E** | Euclidean network of the same topology | cost reference (benchmarks only) |

The repository provides:

- a standalone JAX implementation of the layers and of the Poincaré U-Net and
  ResNet-20 models, written from the published equations without a
  hyperbolic-learning library
- the registered protocol `protocol.yaml` and frozen experiment configurations
- exact ISIC16 and CIFAR-100 split manifests
- runners for numerical agreement, timing and memory benchmarks, profiling,
  and training
- fail-closed exporters and the compact result tables behind every reported
  number
- CPU tests

Spatial patch factorization is the primary contribution. The F/T model paths
also apply the analogous evaluation to HNN++ dense layers;
`docs/derivation.md` maps both cases to code.

---

## 2. Datasets

- **ISIC 2016 challenge, Part 1** — four archives from the
  [ISIC challenge object store](https://isic-archive.s3.amazonaws.com/challenges/2016/)
  ([Gutman et al., 2016](https://arxiv.org/abs/1605.01397)); used for the
  Poincaré U-Net. [Terms of use](https://challenge.isic-archive.com/terms-of-use/).
- **CIFAR-100** — Python archive from the
  [CIFAR page](https://www.cs.toronto.edu/~kriz/cifar.html) (Krizhevsky, 2009);
  used for the Poincaré ResNet-20.

**IMPORTANT:**  
The datasets are not redistributed. `scripts/fetch_core_datasets.sh` downloads
them into `data/raw/` (ignored by Git) and verifies their SHA-256 hashes. The
tracked manifests in `splits/` define the partitions (ISIC16 810/90/379;
CIFAR-100 45,000/5,000/10,000), including example order, per-example hashes,
and duplicate groups.

---

## 3. System Requirements

- Python 3.11 and [uv](https://docs.astral.sh/uv/). Dependencies are pinned in
  `uv.lock`: JAX 0.4.25, NumPy 1.26.4, SciPy 1.11.4, Pillow 11.3.0,
  matplotlib 3.8.4.
- A CPU is sufficient for the tests, checksum verification, and figure
  regeneration.
- GPU reruns: the primary measurements used one NVIDIA Tesla V100-PCIE-32GB
  with JAX/JAXLIB 0.4.25, CUDA 11.8, and cuDNN 8.6. The A6000 training
  replications and attribution controls used one NVIDIA RTX A6000 with the
  same pinned user-space stack. One separate check used JAX 0.11.2 with the
  CUDA-12 plugin, and the PyTorch cross-check used PyTorch 2.4.0 with HypLL
  0.1.1 (`pytorch_crosscheck/README.md`).

Platform records and installation options are in
[`docs/reproducibility.md`](docs/reproducibility.md).

---

## 4. Repository Structure

- `src/poincare_exec/core/`  
  Ball geometry, stable scalar functions, HNN++ layers (R/F/T/M and the
  Euclidean reference), normalization, pooling, residual and transpose operators.
- `src/poincare_exec/models/`  
  Poincaré U-Net and ResNet-20.
- `src/poincare_exec/training/`  
  Fixed-budget training worker, losses, Adam, checkpoints.
- `src/poincare_exec/experiments/`  
  Numerical, timing, profile, checkpoint-replay, and diagnostic runners.
- `src/poincare_exec/analysis/`  
  Table and figure builders.
- `protocol.yaml`  
  Registered protocol with its dated amendment log.
- `configs/`  
  Frozen experiment, analysis, and numerical-case definitions.
- `splits/`  
  ISIC16 and CIFAR-100 split manifests with checksums.
- `scripts/`  
  Environment setup, dataset download, experiment runners (`run_*.sh`),
  exporters (`export_*.py`), and plotting scripts.
- `results/`  
  Compact result tables and figures, the claim map `claims.csv`, and
  `SHA256SUMS`; see `results/README.md`.
- `tests/`  
  CPU tests.
- `docs/reproducibility.md`  
  Detailed reproduction protocol, measurement definitions, and provenance.
- `docs/release_notes_v0.2.0.md`  
  Revision experiments added in this release.
- `docs/derivation.md`  
  Derivation and equation-to-code map.
- `docs/upstream_execution_audit.md`, `docs/reference_resolution.md`  
  Comparison of the reference path with the pinned upstream implementations.
- `pytorch_crosscheck/`  
  PyTorch implementation of F as a replacement for the HypLL convolution, with
  its agreement check and benchmark.
- `provenance/v100_accepted_history_manifest.json`  
  Manifest of the separately distributed Git history of earlier V100 results.

---

## 5. Reproducing the Experiments

New outputs go to `artifacts/` and `out/` (both ignored) and never overwrite
`results/`. GPU runners require exactly one physical GPU index in
`CUDA_VISIBLE_DEVICES` and run sequentially on that device. Publication-scale
training takes hours; nothing is launched by installation or tests.

### Step 1 – Obtain the datasets

```bash
./scripts/fetch_core_datasets.sh
(cd splits && sha256sum --check SHA256SUMS)
```

### Step 2 – Install dependencies

```bash
./scripts/bootstrap.sh                 # CPU environment in ./.venv from uv.lock
./scripts/bootstrap_cuda11_v100.sh     # GPU: CUDA-11.8 jaxlib and pinned NVIDIA libraries
```

### Step 3 – Verify the compact results and run the tests

```bash
(cd results && sha256sum --check SHA256SUMS)
./.venv/bin/pytest
```

### Step 4 – Run numerical agreement

```bash
export CUDA_VISIBLE_DEVICES=0
W=./scripts/with_cuda_11_8.sh
$W ./scripts/run_e02_repair.sh artifacts/numerical_repair/e02_repair
$W ./.venv/bin/poincare-exec structured-numerics --protocol protocol.yaml \
    --config configs/numerics/e02_repair_structured.yaml \
    --output artifacts/numerical_repair/e02_repair_structured.jsonl
```

### Step 5 – Run the execution benchmarks

```bash
$W ./scripts/run_corrected_operator_anchors.sh   # operators and blocks (E03/E05)
$W ./scripts/run_e08.sh                          # complete U-Net (E08)
$W ./scripts/run_e13.sh                          # U-Net update scaling (E13)
$W ./scripts/run_e11.sh                          # device trace (E11)
```

### Step 6 – Run the training campaigns

```bash
$W ./scripts/run_training_campaign.sh configs/training/e09_isic16_unet.yaml
$W ./scripts/run_training_campaign.sh configs/training/e10_cifar100_resnet20.yaml
$W ./scripts/run_training_campaign.sh configs/training/e20_isic16_unet_rf_replication.yaml
$W ./scripts/run_training_campaign.sh configs/training/e21_cifar100_resnet20_rf_replication.yaml
```

### Step 7 – Export tables and render figures

```bash
./.venv/bin/python -m poincare_exec.analysis.build_all \
    --artifact-dir artifacts --table-dir out/tables --figure-dir out/figures
./.venv/bin/python -m scripts.plot_e25_cost_context \
    --validated-dir results/reviewer_requested --output-stem out/figures/e25_cost_context
```

The exporters for each experiment family, and the rules they check, are listed
in `docs/reproducibility.md` (Sections 5 and 6).

### Revision experiments

Version 0.2.0 adds thirteen experiments made for the manuscript revision: E22
allocator-fraction rerun, E23 autotuning memory diagnostic, E24
materialized-patch ablation M, E25 same-topology Euclidean reference E, E26
elementwise agreement replay, E27 current-JAX 0.11.2 check, E28 replay
stability, E29 V100 Euclidean reference, E30 U-Net width check, E31
gather-based reference controls Rg/Mg, E32 PyTorch/HypLL cross-check, E33
FP64-oracle accuracy, branch trace, and cap activity, and E35 FP64-oracle
accuracy at trained ResNet-20 states. It also adds exports of
learned curvature, post-hoc equivalence tests (TOST), and training minutes
from existing runs. The run and export command for each is in
[`docs/release_notes_v0.2.0.md`](docs/release_notes_v0.2.0.md).

---

## 6. Outputs and Results

- `results/` holds one folder per result family; `results/README.md` lists
  each file with its experiment, protocol version, and code revision.
- `results/claims.csv` maps each manuscript claim to its supporting files.
- `results/checkpoints/checkpoint_inventory.csv` lists the 540 checkpoint
  files of the 90 training runs with their SHA-256 hashes.
- Raw result rows written to `artifacts/` record the protocol version,
  configuration hash, code commit, clean-worktree flag, and device of each run.

Datasets, checkpoints, raw per-call timing rows, and profiler traces are not
included; they are archived outside Git and are not needed to inspect any
reported number.

---

## 7. Reproducibility Features

- Registered protocol with a dated amendment log and per-result configuration
  hashes
- Exact split manifests with per-example hashes and duplicate groups
- Fixed training seeds (11, 29, 47, 71, 101 for the five-seed cohorts and 15
  registered seeds for the replications) and a fixed split seed
- Five fresh-process paired sessions with 50 warmup and 200 synchronized calls
  for every timing result
- Out-of-memory outcomes and failures retained as results
- Fail-closed exporters that check protocol version, configuration hash, and
  clean code commit
- SHA-256 checksums for compact results and split manifests
- Figures regenerated on CPU from tracked tables, byte for byte

### Methodology mapping to the manuscript

| Manuscript element | Experiment | Compact evidence |
|---|---|---|
| Method and Figure 1 | E01 | `docs/derivation.md`, `src/poincare_exec/core/hnnpp.py` |
| Table 1 (training settings) | E09, E10, E20, E21 | `configs/training/`, `protocol.yaml` |
| Table 2 (structured agreement, accepted run and full replay) | E02, E26 | `results/numerics/`, `results/reviewer_requested/e26_*` |
| Section 5.1 and Supplementary Table S11 (FP64 oracle, branch trace, cap activity) | E33 | `results/reviewer_requested/e33_*` |
| Appendix C.3, Table 11 and Discussion (FP64 oracle at trained ResNet-20 states) | E35 | `results/reviewer_requested/e35_*` |
| Table 3 (operator and block speedups) | E03, E05 | `results/operators/` |
| Table 4 (complete U-Net on V100) | E08, E29; E23 in the text | `results/unet_timing/`, `results/reviewer_requested/e29_*`, `results/reviewer_requested/e23_*` |
| Figure 2 (U-Net update scaling) | E13, E29; E22 in the text | `results/e13_scaling/`, `results/reviewer_requested/e29_unet_update_scaling*`, `results/reviewer_requested/e22_*` |
| Section 5.3 (U-Net width) | E30 | `results/reviewer_requested/e30_*` |
| Table 5 (A6000 attribution and Euclidean reference) | E31, E25 | `results/reviewer_requested/e31_*`, `results/reviewer_requested/e24_e25_e27_*` |
| Section 5.5 and Supplementary Table S12 (official HypLL layer) | E32 | `results/reviewer_requested/e32_*`, `pytorch_crosscheck/` |
| Table 6 (training time and speedups) | E09, E10, E20, E21 | `results/training/`, `results/e20_e21_training_replication/`, `results/reviewer_requested/training_runtime_minutes_*` |
| Table 7 and Figure 3 (task quality) | E09, E10, E20, E21 | `results/training/`, `results/e20_e21_training_replication/` |
| Appendix: software environment | E19, E27 | `results/environment.json`, `results/e19_preflight/`, `results/reviewer_requested/e24_e25_e27_audit.json` |
| Discussion (retained-state gradient differences) | E07-J2 | `results/diagnostics/` |
| Appendix: sensitivity analyses and post-hoc TOST | E09, E10, E20, E21 | `results/training/`, `results/e20_e21_training_replication/`, `results/reviewer_requested/quality_tost.csv` |
| Supplementary Tables S1–S7 | E03/E05, E08, E09/E10, E20/E21 | as above |
| Supplementary Table S8 (device trace) | E11 | `results/profile/` |
| Supplementary Table S9 (elementwise errors) | E26 | `results/reviewer_requested/e26_*` |
| Supplementary Table S10 and Figure S1 (learned curvature) | E09, E20 | `results/reviewer_requested/isic16_curvature_*` |

---

## 8. Citation

If you use this code, please cite the manuscript and the software release
described in `CITATION.cff`:

```bibtex
@software{jochymek2026expoconv,
  author  = {Jochymek, Luk{\'a}{\v{s}}},
  title   = {Efficient Poincar{\'e} Convolutions through Patch-Statistics Factorization: Experiments},
  version = {0.2.0},
  year    = {2026},
  url     = {https://github.com/lowoncuties/ExPoConv}
}
```

---

## 9. License

This software is distributed under the MIT License. See the `LICENSE` file for
details. Dataset rights remain with their providers (Section 2).

---

## 10. Disclaimer

This work is intended for research purposes only. Timing and memory results
are specific to the recorded hardware, drivers, and software versions. Models
trained with this code on ISIC16 must not be used for clinical
decision-making.
