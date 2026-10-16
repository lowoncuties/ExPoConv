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

## Elementwise error requires a fresh structured-numerics rerun

The accepted post-repair E02 structured JSONL has 2,720 agreement rows, but it
retains only tensor-level relative L2 error, reference L2 norm, and maximum
absolute error. Elementwise quantiles cannot be reconstructed from those
aggregates. The available NPZ with per-element tensors belongs to the excluded
pre-repair campaign and must not be substituted.

The E26 structured-numerics extension now prospectively records

`abs(candidate - reference) / max(abs(reference), 1e-12)`

with the fraction of denominators affected by the floor, linear p50, p90, p95,
p99, p99.9, and maximum. The median describes the center, p90/p95 the broad
tail, p99/p99.9 the rare tail, and the maximum preserves the worst element. The
floored-denominator fraction exposes sensitivity to zero and near-zero
reference entries.

Run E26 into a fresh artifact; do not append to or replace the accepted E02
`ce009c1` source:

```bash
./.venv/bin/poincare-exec structured-numerics \
  --protocol protocol.yaml \
  --config configs/numerics/e26_elementwise_agreement.yaml \
  --output /fresh/path/e26_elementwise_agreement.jsonl
```

This rerun changes only diagnostic fields in the raw artifact schema. It does
not change model operations, inputs, gradients, tolerances, RNG, or the
registered acceptance calculation.

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
