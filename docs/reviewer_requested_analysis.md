# Reviewer-requested statistical and trajectory analyses

This supporting analysis addresses learned-curvature trajectories, explicit
post-hoc equivalence tests, absolute training runtimes, and the feasibility of
elementwise agreement summaries. It does not edit the manuscript and does not
pool the V100 and A6000 cohorts.

## Available from retained accepted artifacts

- `results/reviewer_requested/isic16_curvature_trajectories.csv` contains the
  end-of-epoch development record at all 50 epochs for every R/F seed in E09
  and E20 (2,000 rows). The underlying `training_metrics.jsonl` hash is attached
  to every row. E20 hashes are also checked against the accepted run audit.
- `isic16_curvature_finals_by_seed.csv` and
  `isic16_curvature_final_summary.csv` give final matched-seed values and
  descriptive paired 95% t intervals. The E09 mean F-minus-R difference is
  0.00504 (95% CI -0.01042 to 0.02049; five pairs), and the E20 difference is
  0.00348 (-0.00196 to 0.00892; fifteen pairs). The maximum absolute paired
  differences are 0.02177 and 0.02321, respectively. These results show no
  clear cohort-level directional shift, but they do not establish curvature
  equivalence and the seed-level variation should remain visible.
- `quality_tost.csv` reports paired two-one-sided t tests (TOST) on the native
  zero-to-one metric scale. The bounds are +/-0.01 for Dice (one percentage
  point) and +/-0.005 for accuracy (0.5 percentage points). Every row is marked
  `post_hoc_exploratory_not_prespecified`. Each accelerator cohort, endpoint,
  and test partition is analyzed separately. All four E20 and all four E21
  comparisons meet the stated equivalence criterion at alpha 0.05. One of four
  E09 comparisons and none of four E10 comparisons do. Therefore, the newer
  fifteen-seed cohorts support the stated margins, but “confidence interval
  spans zero” must not be retroactively described as equivalence for the older
  five-seed cohorts.
- `training_runtime_minutes_by_seed.csv` reports absolute compute and complete
  worker elapsed wall-clock minutes for every matched R/F seed.
  `training_runtime_minutes_summary.csv` provides descriptive summaries. On
  A6000, mean elapsed wall-clock time is 12.44 versus 7.58 minutes for E20 R/F
  and 35.37 versus 21.38 minutes for E21 R/F. Compute time remains separate
  because it is the registered fixed-budget timing scope.

## Elementwise error and replay stability

The accepted post-repair E02 structured JSONL has 2,720 agreement rows, but it
retains only tensor-level relative L2 error, reference L2 norm, and maximum
absolute error. Elementwise quantiles cannot be reconstructed from those
aggregates. The available NPZ with per-element tensors belongs to the excluded
pre-repair campaign and must not be substituted.

The completed E26 structured-numerics extension records

`abs(candidate - reference) / max(abs(reference), 1e-12)`

with the fraction of denominators affected by the floor, linear p50, p90, p95,
p99, p99.9, and maximum. The median describes the center, p90/p95 the broad
tail, p99/p99.9 the rare tail, and the maximum preserves the worst element. The
floored-denominator fraction exposes sensitivity to zero and near-zero
reference entries. The complete per-tensor export is
`e26_elementwise_tensor_statistics.csv`; scope- and case-level summaries are
retained alongside it.

E26 passes 2,688 of 2,720 tensors but does not reproduce E02's variant-specific
FP32 pattern: 32 small-U-Net realization-2 gradients miss under F, whereas E02
places the analogous pattern under T and adds one T curvature-gradient miss.
E28 therefore repeats the affected 408-tensor deterministic prefix in three
fresh processes. Every E28 process passes 375/408 tensors: all 204 F tensors
pass and the same 33 T tensors miss. The resulting 375 stable passes, 33 stable
failures, and zero mixed outcomes reproduce E02 at the mixed-criterion level,
but the separate E26 outcome prevents a general claim of bitwise identity or
path-specific finite-precision gradient superiority. Neither post-hoc replay
replaces the accepted E02 gate.

## Reproduction

With the external accepted raw roots mounted, regenerate all currently
available tables with:

```bash
./.venv/bin/python scripts/export_reviewer_requested_analysis.py \
  --e09-training-root /path/to/accepted/training/e09 \
  --e20-training-root /path/to/accepted/training/e20 \
  --accepted-structured-results /path/to/e02_repair_structured_ce009c1.jsonl \
  --output-dir results/reviewer_requested
```

The compact training CSVs and E20/E21 run audit default to their tracked paths.
`analysis_summary.json` records their hashes and the hashes of all 40 raw
curvature sources. The exporter never modifies a raw input.

The completed E23 and E28 diagnostics have separate fail-closed exporters:

```bash
./.venv/bin/python scripts/export_e23_autotune_memory.py --help
./.venv/bin/python scripts/export_e28_replay_stability.py --help
```

Their tracked JSON summaries record raw hashes, identities, row counts, and the
scope of each post-hoc interpretation.

## Revision benchmark throughput derivation

`scripts/export_e24_e25_e27_tables.py` validates the complete registered
E24/E25/E27 outcome matrices before producing its combined session, variant,
contrast, and audit outputs. Export schema 1.1 adds the validated batch size and
derives each accepted session's steady-state throughput as

`1000 * batch_size / session_median_latency_ms`

in samples/s. The batch size comes from the registered workload shape and must
be a positive integer. Failed groups retain that validated batch size but have
no latency or throughput value. Variant summaries take the median, minimum, and
maximum over accepted session-level throughputs; they do not invert a rounded
display value. Paired contrast throughput is candidate/reference, so it is
numerically the same speed direction as the retained reference/candidate
latency ratio. Full-precision latency and throughput fields are both preserved.

`scripts/plot_e25_cost_context.py` independently verifies the audit hashes,
batch sizes, and the throughput identity before rendering the U-Net context
figure. Its left panel shows absolute samples/s and its right panel shows
process-lifetime peak live allocation. Small points are the five session values,
large points are medians, and ranges are session minima and maxima rather than
confidence intervals. The figure-data CSV retains the latency summaries and
adds the throughput values used by the plot. No E24/E25, software-stack, or
hardware strata are pooled.

The validated E25 matrix contains all 90 registered groups. At the complete-
update boundaries, the R and F paths are respectively 6.596 and 3.185 times
slower than E for U-Net, and 5.542 and 3.209 times slower than E for ResNet-20.
The paired R-to-F speedups are 2.071 (95% interval 2.059--2.082) and 1.727
(1.696--1.759). These are computational context comparisons only; E25 contains
no task-quality evaluation.

The validated E27 matrix contains all ten registered batch-eight U-Net update
groups under JAX/JAXLIB 0.11.2 and its CUDA-12 plugin. R/F median latencies are
114.710/68.822 ms and median process-lifetime peaks are 4.687/2.386 GiB. The
paired R-to-F speedup is 1.667 (1.662--1.672). E27 is a single-workload post-hoc
portability check and is not pooled with or used as a causal comparison against
the pinned JAX 0.4.25 stack.
