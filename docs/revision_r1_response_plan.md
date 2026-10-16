# Revision R1 evidence and implementation plan

Date frozen for planning: 2026-10-16 (Europe/Prague)

This plan separates completed evidence from work that is running or proposed.
No proposed result may enter the manuscript until its raw artifacts and
provenance checks pass. Existing data splits, preprocessing, losses, optimizer
order, RNG behavior, and metric aggregation remain unchanged.

## Inspected starting point

- Code: `revision-r1` at `5bb74e9f6d4991ecdc813d78e0aea3d7b2d3cae7`.
- Frozen public-release ancestor: `v0.1.0` at
  `33f991ad34a02a59d07ba231eed3f4d9d501f57c`.
- Canonical manuscript:
  `/data2/joc0027/optimizatio-paper/poincare_latex_project/NN_paper/factorization_revised.tex`,
  SHA-256 `a06e27db8606f8700c43556915f5bb2a792fb09516428992f64aa2e10fca9d04`.
- Accepted original runtime: JAX/JAXLIB 0.4.25, CUDA 11.8, V100 32 GiB.
- Separate accepted replication runtime: JAX/JAXLIB 0.4.25 with pinned CUDA
  11.8 user-space libraries on one RTX A6000.
- Current stable-JAX sanity target: JAX/JAXLIB 0.11.2 with CUDA 12 on the RTX
  A6000. This is a separate portability check and is not pooled with either
  accepted cohort.

## Prioritized work

| Priority | Reviewer question | Smallest credible action | Acceptance criterion | State |
|---|---|---|---|---|
| P0 | Was the 512-by-512 R OOM only the default 0.75 allocator cap? | E22: repeat R/F at batch 8, 512 squared with allocator fraction 0.95; add batch 16, 512 squared as a physical-memory stress point; retain batch 8, 256 squared as a control; five fresh sessions. | Every group has a retained success or explicit failure record. Reframe the result around the jointly feasible 38--55% memory reduction regardless of the stress-point outcome. | Running on V100 GPU 0; one valid F session was already retained before the resumable campaign. |
| P0 | Does R mirror public upstream execution, or is it a straw man? | Inspect pinned HypLL/HyPLL and Poincare ResNet source and cite exact materialization/mapping calls at immutable commits. Search public implementations for an existing factorized path. | A traceable source-level comparison supports every manuscript sentence; any prior factorization narrows the novelty claim. | In progress; literature/provenance workstream. |
| P0 | Is the speedup geometry removal or merely implicit-GEMM versus explicit im2col? | Add M, which retains R's materialized patch but replaces pointwise ball mapping with exact patch statistics. Compare R--M (geometry) and M--F (materialization/backend) under the same benchmark boundary. | Focused value/gradient tests pass; five fresh GPU sessions per registered workload; attribution is reported as two measured effects, not a single causal number. | Implementation and CPU invariance tests in progress. |
| P0 | Will reviewers receive executable code? | Rebuild the verified reviewer archive after revision. Prepare an anonymized public derivative that removes or explicitly approves host/provenance identifiers. | Clean-clone verification passes and the manuscript contains a real reviewer-access URL. | Local release exists; external publication/URL requires an author-approved host and provenance policy. |
| P1 | What is the cost relative to an ordinary Euclidean topology? | Add E, a same-shape Euclidean U-Net/ResNet reference. Benchmark forward and complete update latency plus process-lifetime peak allocation on the same hardware/runtime. | Table 4 and Figure 2 use the same timing/memory definitions for E/R/F; no hyperbolic quality comparison is implied. | Implementation in progress; GPU run follows tests. |
| P1 | Do learned-curvature paths agree despite the known FP32 curvature-gradient residual? | Export complete and downsampled per-seed curvature trajectories plus final R/F values from retained E09/E20 logs. | All registered seeds are present, paired by seed, with no smoothing that hides divergence; text reports agreement or divergence descriptively. | Analysis in progress; no retraining expected. |
| P1 | Does tensor-level relative error hide large elementwise errors? | Export elementwise absolute and stabilized relative errors for outputs and each gradient leaf, with max and prespecified quantiles. Rerun only cases for which raw arrays were not retained. | Denominator convention and near-zero handling are explicit; output and gradient distributions are reported separately by dtype/case family. | Raw-artifact audit in progress. |
| P1 | Is quality practically equivalent? | Exploratory paired TOST using margins of 0.01 Dice and 0.005 accuracy on fixed-final endpoints, separately for each V100/A6000 cohort. | Both one-sided tests, confidence bounds, units, sample size, and the post-hoc/non-prespecified label are reported; failure is retained. | Analysis in progress. |
| P1 | Does the result survive a current compiler/runtime? | Run the registered batch-8 U-Net update R/F sanity point using JAX/JAXLIB 0.11.2 + CUDA 12 on idle A6000 GPU 1 during the weekend. | Environment and device provenance are captured; each path either completes with finite output or has an explicit retained compiler/runtime failure. Timing is a separate portability result. | Isolated environment setup running; GPU launch is weekend-only. |
| P2 | Why does batch-1 F forward use more peak memory? | E23 repeats the exact case with cuDNN autotuning enabled and disabled. | Attribute the difference only if the controlled workspace result supports it; otherwise report it as unresolved implementation workspace behavior. | Running on V100 GPU 1 in a separate work area. |
| P2 | Table 5 reports only ratios. | Add per-run compute and elapsed minutes for both paths, with cohort/platform labels. | Values trace directly to retained summaries and keep compute time distinct from end-to-end elapsed time. | Analysis in progress. |
| P2 | What now dominates F? | Add one bounded sentence from the accepted device trace: reductions/midpoint normalization account for about 65% of F kernel time, so normalization optimization is complementary future work. | Wording remains descriptive and trace-scoped. | Manuscript edit after evidence integration. |
| P2 | Is “sufficient statistics” misleading? | Replace it in the title and narrative with “patch statistics” or “energy--projection factorization”; retain the mathematical sufficiency statement only where defined. | Title and abstract describe the supported mechanism without a statistical-estimation implication. | Manuscript edit after evidence integration. |
| P2 | Where does the memory benefit diminish at larger channel widths? | After P0/P1 runs, add one wider U-Net scaling point only if it reuses the validated harness and fits the remaining weekend budget. | Favorable and unfavorable outcomes are both reported; this optional point cannot delay the central controls. | Deferred behind P0/P1. |

## Execution order

1. Finish E22 and E23 without GPU contention; retain all failure rows.
2. Complete M/E implementations and CPU value/gradient tests.
3. Freeze new benchmark configurations and provenance before GPU execution.
4. Use idle A6000 GPU 1 only during the weekend for the current-JAX sanity
   check and, if validated in time, the M/E benchmark matrix.
5. Export curvature, TOST, elementwise-error, and absolute-runtime analyses from
   retained artifacts.
6. Integrate only completed evidence into the canonical manuscript, figures,
   tables, claims map, and release package.
7. Run focused tests, the complete relevant test suite, checksum/provenance
   verification, three clean LaTeX passes, and full-PDF visual inspection.

## Claim policy

- The central memory claim is the measured R-to-F reduction across jointly
  feasible workloads and the demonstrated device class each implementation
  fits; an OOM remains a configuration-specific observation.
- Performance attribution will be split into geometry work avoided (R--M) and
  explicit materialization/backend effects (M--F).
- TOST, wider scaling, and current-JAX checks are post-hoc or portability
  additions, not retroactively prespecified confirmatory experiments.
- Neither a checklist nor these additions imply or guarantee acceptance.
