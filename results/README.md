# Compact evidence

Deterministic exports from the completed experiments. No value here was
hand-entered; each file retains the experiment, protocol version,
configuration hash, and code revision of the run it was exported from.
Verify the bundle with `sha256sum --check SHA256SUMS` in this directory.

| Folder | Experiment | Contents |
|---|---|---|
| `numerics/` | E02 | bounded repair suite summary, structured agreement overview and the 33 T failures, finite-difference sweeps, exact rows quoted in the appendix |
| `operators/` | E03, E05 | per-session medians, paired speedups with 95% log-t intervals, latency summary |
| `unet_timing/` | E08 | per-session medians, paired speedups, latency and peak-memory summary, manuscript timing table |
| `e13_scaling/` | E13 | all 70 planned session/path outcomes, workload and paired-ratio summaries, and the complete-update scaling and allocator-bounded feasibility figure |
| `profile/` | E11 | module spans, kernel counts, kernel-category attribution |
| `training/` | E09, E10 | all 30 runs and endpoints, per-path summaries, paired quality/IoU/efficiency intervals, and the all-endpoints figure |
| `e20_e21_training_replication/` | E20, E21 (with non-pooled E09/E10 strata) | 60 accepted A6000 runs, joint-gate audit, run-level and paired quality/efficiency tables, four-stratum figure, and analysis provenance |
| `e19_preflight/` | E19 same-machine technical preflight | author-authorized clean-clone record, bounded-numerics summary, five-session R/F/T medians, paired speedups, and raw-evidence audit summary |
| `checkpoints/` | E06 (supporting, not in the manuscript) | replay audit, task agreement, aggregate, and the inventory of archived checkpoints |
| `diagnostics/` | E07-J2 (one Discussion sentence; full detail supporting) | source audit, key metrics, gradient/update/logit agreement, 100-update trajectory and figure |
| `reviewer_requested/` | E22--E35 and retained-run reviewer analyses | E22 allocator sensitivity; E23 workspace control; E24 mechanism ablation; E25 Euclidean cost context; E26 elementwise errors; E27 current-JAX check; E28 replay stability; E29 V100 Euclidean reference and scaling figure; E30 U-Net width; E31 gather-based reference; E32 PyTorch/HypLL cross-check; E33 oracle accuracy, branch trace, and cap activity; learned-curvature trajectories; exploratory TOST; and absolute runtime minutes; E35 FP64 oracle at retained ResNet-20 states |
| `claims.csv` | all | manuscript claims mapped to supporting files |
| `environment.json` | V100 result families | byte-preserved measured V100 hardware/software record used by the registered E13 hash; its historical description predates E20/E21 and must not be read as covering them. Their A6000 runtime is in `e20_e21_training_replication/e20_e21_gate_audit.json` |

## Compact evidence and manuscript elements

| Evidence or manuscript element | File(s) |
|---|---|
| Structured agreement table and appendix worst rows | `numerics/structured_agreement_overview.csv`, `numerics/structured_agreement_failures.csv`, `numerics/numerical_claim_rows.csv`, `numerics/bounded_check_summary.csv` |
| Operator and block speedups | `operators/paired_speedup_summary.csv`, `operators/workload_latency_summary.csv` |
| Complete U-Net latency and peak memory | `unet_timing/paper_timing_table.csv`, `unet_timing/paired_speedup_summary.csv`, `unet_timing/workload_latency_memory_summary.csv` |
| Complete U-Net update scaling and allocator-bounded feasibility | `e13_scaling/paired_ratio_summary.csv`, `e13_scaling/workload_summary.csv`, `e13_scaling/session_outcomes.csv`, `e13_scaling/unet_scaling.{pdf,png}` |
| U-Net update profile | `profile/profile_summary.csv`, `profile/kernel_category_attribution.csv` |
| Five-seed training outcomes and speedups | `training/training_variant_summary.csv`, `training/paired_quality_summary.csv`, `training/paired_training_efficiency_summary.csv`, `training/paired_iou_summary.csv`, `training/training_runs.csv` |
| Original V100 all-endpoints appendix figure | `training/training_all_endpoints.{pdf,png}`, regenerated and cohort-gated by `scripts/plot_training_all_endpoints.py`; `training/figure_training_paired_endpoints.{pdf,png}` is the raw exporter's rendering of the same compact endpoint values |
| Fifteen-seed A6000 replications and current non-pooled four-cohort training figure | `e20_e21_training_replication/e20_e21_training_variant_summary.csv`, `e20_e21_training_replication/e20_e21_paired_quality_summary.csv`, `e20_e21_training_replication/e20_e21_paired_training_efficiency_summary.csv`, `e20_e21_training_replication/e09_e10_e20_e21_platform_strata.csv`; archival exporter figure `e20_e21_training_replication/e09_e10_e20_e21_quality_strata.{pdf,png}`; presentation-only derivative `e20_e21_training_replication/e09_e10_e20_e21_quality_strata_manuscript.{pdf,png}` |
| Same-machine clean-clone technical preflight | `e19_preflight/technical_preflight_record.md`, `e19_preflight/numerics_summary.json`, `e19_preflight/steady_state_session_medians.csv`, `e19_preflight/paired_speedup_summary.csv`, `e19_preflight/summary.json` |
| Reviewer-requested allocator, mechanism, cost-context, width, cross-implementation, numerical, runtime, curvature, and equivalence evidence | `reviewer_requested/e22_*`, `reviewer_requested/e23_*`, `reviewer_requested/e24_e25_e27_*`, `reviewer_requested/e25_cost_context_data.csv`, `reviewer_requested/e26_*`, `reviewer_requested/e28_*`, `reviewer_requested/e29_*`, `reviewer_requested/e30_*`, `reviewer_requested/e31_*`, `reviewer_requested/e32_*`, `reviewer_requested/e33_*`, `reviewer_requested/isic16_curvature_*`, `reviewer_requested/quality_tost.csv`, `reviewer_requested/training_runtime_minutes_*` |
| Provenance appendix | `*/summary.json`, `training/training_runs.csv`, `e20_e21_training_replication/e20_e21_gate_audit.json`, `e20_e21_training_replication/e20_e21_run_audit.csv` |

## Supporting diagnostics retained in full only in the repository

| Diagnostic | Manuscript status | File(s) |
|---|---|---|
| E06 checkpoint replay | Not reported in the current manuscript | `checkpoints/trained_checkpoint_task_agreement.csv`, `checkpoints/trained_checkpoint_compatibility_aggregate.csv`, `checkpoints/trained_checkpoint_campaign_audit.csv` |
| E07-J2 retained-state diagnostic | One Discussion sentence reports the ResNet-20 and U-Net gradient-difference ranges against the repeated-R and one-ULP controls; full detail remains repository-only | `diagnostics/diagnostic_key_metrics.csv`, `diagnostics/*_agreement_summary.csv`, `diagnostics/trajectory.csv`, `diagnostics/diagnostic_source_audit.csv`, figure `diagnostics/figure_retained_state_trajectory.{pdf,png}` from `scripts/export_retained_diagnostic_tables.py` |

## Provenance of each result family

Commit identifiers refer to the development history the code was extracted
from; bounded repairs and later campaigns were frozen at different times, so
families legitimately differ in revision. Full commit objects for the eleven
legacy-development revisions below are preserved in the separate Git bundle
declared in `../provenance/v100_accepted_history_manifest.json`. E13 revision
`75f10dd9`, the E20/E21 training revision `fef30ec`, and the E22--E35
revisions listed below are retained in ExPoConv's own Git history.

| Family | Protocol | Code revision | Identity record |
|---|---|---|---|
| E02 bounded / structured numerics | 1.5 | `d3f3eb80` / `ce009c16` | `numerics/summary.json` |
| E03/E05 operator and block timing | 1.5 | `758d182d` | `operators/summary.json` |
| E08 U-Net forward / update timing | 1.7 / 1.8 | `c0caba16` / `50c399bc` | `unet_timing/summary.json` |
| E11 profile | 1.8 | `33331485` | `profile/summary.json` |
| E13 U-Net update scaling | 1.11 | `75f10dd9` | `e13_scaling/summary.json` |
| E22 allocator sensitivity | 1.15 | `30142ce2` | `reviewer_requested/e22_audit.json` |
| E23 workspace diagnostic | 1.15 | `30142ce2` | `reviewer_requested/e23_memory_summary.json` |
| E26 elementwise replay | 1.15 | `30142ce2` | `reviewer_requested/e26_elementwise_summary.json` |
| E28 replay stability | 1.17 | `df3f6f55` | `reviewer_requested/e28_replay_summary.json` |
| E24 mechanism ablation / E25 Euclidean context / E27 current-JAX check | 1.17 | `df3f6f55` | `reviewer_requested/e24_e25_e27_audit.json` |
| E29 V100 Euclidean reference / E30 U-Net width | 1.18 | `20a886e4` (run), exported at `63175572` | `reviewer_requested/e29_audit.json`, `reviewer_requested/e30_audit.json` |
| E31 gather-based reference | 1.19 | `486b5bd2` (run), exporter `60d47b6f`, export `f7c87c8c` | `reviewer_requested/e31_audit.json` |
| E32 PyTorch/HypLL cross-check | not registered in `protocol.yaml` | `df03198a` (measurement), results `fac215e9` | `reviewer_requested/e32_pytorch_summary.json` |
| E33 oracle accuracy / branch trace / cap activity | 1.17 recorded; not registered | `620b0bfb` / `55e9c8a2` / `768301c7`, results `3be3ee0b` | `reviewer_requested/e33_oracle_accuracy_summary.json`, `reviewer_requested/e33_branch_trace_small_unet_r2_flip_test.json`, `reviewer_requested/e33_cap_activity_summary.json` |
| E35 retained-state FP64 oracle | 1.20 | `e98d313f` (registration), results `fa74c004` | `reviewer_requested/e35_retained_oracle_summary.json` |
| E09 U-Net training | 1.10 | `75426855` (seed 11 R/T), `2d29b525` (others) | `training/training_runs.csv` |
| E10 ResNet-20 training | 1.10 | `c3a1dd92` | `training/training_runs.csv` |
| E20 U-Net R/F replication | 1.14 | `fef30ec` | `e20_e21_training_replication/e20_e21_run_audit.csv` |
| E21 ResNet-20 R/F replication | 1.14 | `fef30ec` | `e20_e21_training_replication/e20_e21_run_audit.csv` |
| E19 same-machine technical preflight | 1.14 | `93fb31a1` | `e19_preflight/technical_preflight_record.md` |
| E06 checkpoint replay | 1.10 | evaluator `e78d3674`; sources keep their E09 identity | `checkpoints/trained_checkpoint_campaign_audit.csv` |
| E07-J2 retained-state diagnostic | 1.10 | evaluator `04aca39f`; sources keep their E09/E10 identity | `diagnostics/diagnostic_source_audit.csv` |

The E08 forward rows come from the protocol-1.7 run and the update rows from
the protocol-1.8 continuation after a dtype-preservation fix; both are
declared in `configs/analysis/e08_unet_accepted.yaml`. The E11 profile and the
timing sessions were run on the same device (`configs/analysis/e11_unet_profile_accepted.yaml`).

E13 accounts for all 70 planned fresh-process path/session groups. All 35 F
groups and 30 of 35 R groups completed; the five R groups at batch 8 and
512 x 512 ended in recorded first-update OOMs. The corresponding five F
groups completed. `e13_scaling/session_outcomes.csv` retains every outcome,
including the five OOM records, and `e13_scaling/summary.json` records the
hashes of the archived raw sources.

E20/E21 account for all 60 registered R/F training runs and 30 matched pairs.
The joint gate opened the official-test endpoints only after every registered
run passed the frozen health, identity, and artifact checks. The clean
`072c9ddd` exporter produced each table and the archival figure in
`e20_e21_training_replication/`; its immutable source and output hashes remain
in `e20_e21_analysis_summary.json`. The presentation-only
`e09_e10_e20_e21_quality_strata_manuscript.{pdf,png}` derivative shortens
labels and adjusts tick formatting without recomputing estimates. From the
repository root, regenerate it on CPU from the unchanged compact CSV with
`./.venv/bin/python scripts/plot_e20_e21_quality_strata.py`; current derivative
hashes are in `results/SHA256SUMS`. The E09/E10 V100 and E20/E21 A6000 rows are
separate strata. No grand pool or causal comparison of accelerators was
performed. `checkpoints/checkpoint_inventory.csv` was generated after the gate
by the hardened exporter at revision `8a233b9`; its five-role historical
E09/E10 projection remains byte-identical to the earlier 150-row inventory.

The E19 same-machine technical preflight cloned the verified release-candidate
Git bundle at clean commit `93fb31a11a008ba1a378434071cb418b504092d9` and used
only physical V100 GPU 1 under the pinned CUDA-11.8 environment. Its bounded
E02 run produced 424 rows and passed the required contract; the batch-eight
complete-update timing retained all 15 R/F/T groups and gave paired R/F
speedup 1.745 [1.557, 1.956] across five sessions. This author-authorized run
checks release mechanics and same-platform repeatability;
it is not an independent-human reproduction, does not include the E08 memory
measurement, and is not pooled with the accepted E08 estimate. Full raw rows,
logs, snapshots, and the environment dump remain in the checksummed external
working artifact named by `e19_preflight/technical_preflight_record.md`.

E22 accounts for all 30 planned workload/path/session groups under allocator
fraction 0.95: 15 completed and 15 ended in explicit first-update OOMs. The
batch-8, 256-squared control completed for both paths; at batch 8 and
512 squared only F completed; at batch 16 and 512 squared neither path
completed. Requested-allocation and BufferAssignment totals in
`reviewer_requested/e22_failure_diagnostics.csv` are compiler diagnostics, not
measured process peaks or minimum-memory requirements. The full raw timing CSV
and failure JSONL remain outside Git; `reviewer_requested/e22_audit.json`
records their names, sizes, and SHA-256 hashes.

E23 retains all 20 registered compiler-setting/path/session cells and binds the
autotuning setting to the frozen wrapper. E24, E25, and E27 retain respectively
45, 90, and 10 complete fresh-process groups in one joint validated export. E24
separates removal of pointwise patch geometry from the remaining materialization/
contraction/backend bundle; E25 supplies computational, not quality, context;
E27 is one non-pooled current-JAX portability point. E26 retains 2,720 tensor
summaries, and E28 retains three independent 408-tensor replay summaries. Their
summary or audit JSON files record raw-source hashes, output row counts, and the
bounded interpretation; the E22 and E24/E25/E27 audits also hash every output.

E29 and E30 ran concurrently on the V100 during periods of host driver stalls.
A rule fixed before any affected latency was inspected (named and hashed in
`reviewer_requested/e29_audit.json` and `e30_audit.json`) flags every group
whose execution window overlapped a recorded stall. Flagged groups are excluded
from the primary medians and paired contrasts, and all-session values are
reported beside them as a sensitivity. The E29/E30 audits record an uncommitted
exporter (`dirty: true`); rerunning the committed exporter on the same raw
files reproduces every E29/E30 table byte for byte.

E31 accounts for all 80 planned R/Rg/Mg/F groups. Its five registered paired
contrasts and the additional post-hoc R/Mg contrast are in
`reviewer_requested/e31_contrasts.csv`. The compiled buffer-assignment byte
fields of this run are zero, so the compile-only memory capture is validated
and hashed in `e31_audit.json` but not reported.

E32 compares the official HypLL 0.1.1 `HConvolution2d` with the factorized
PyTorch layer in `../pytorch_crosscheck/` (PyTorch 2.4.0, CUDA 12.1, RTX A6000).
E33 runs on CPU: FP32 R/F/T accuracy against a fixed-margin FP64 oracle, a
branch trace of the small-U-Net realization behind the E02/E26 misses, and the
fraction of source-radius cap activations in the final E09/E10 checkpoints.
E32 and E33 are post-hoc diagnostics that were not registered in
`protocol.yaml`. E35 recomputes the E07-J2 ResNet-20 gradients on the E07-J2
V100: R, F, and T in FP32 and R in FP64 with the FP32 margin as the oracle.

## Not included

Datasets, the 540 checkpoint artifacts from 90 accepted training runs (listed
with SHA-256 in `checkpoints/checkpoint_inventory.csv`), raw per-call timing
rows (including the 24-MiB E13 table), the E13 JSONL failure sidecar,
the E19 raw timing CSV and numerical JSONL, the raw E22--E32 timing files,
failure sidecars, and numerical JSONL replays, predictions, profiler traces,
and logs are archived outside Git. They are not
needed to inspect any reported number; checkpoints and datasets are needed to
rerun E06 and E07-J2.

Some `summary.json` and audit files retain the hostname, device index, and
absolute paths of the measuring host, because those bytes are part of the
recorded provenance.
