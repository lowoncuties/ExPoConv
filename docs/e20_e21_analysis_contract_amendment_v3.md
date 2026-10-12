# E20/E21 pre-unblinding analysis contract amendment v3

Status: frozen before inspection of any E20/E21 official-test endpoint value

Frozen: 2026-10-12

Base contract: `docs/e20_e21_analysis_contract.md`

Base manifest: `configs/analysis/e20_e21_training_replication.yaml`

Prior amendment: `docs/e20_e21_analysis_contract_amendment_v2.md`

Amendment manifest: `configs/analysis/e20_e21_training_replication_v3.yaml`

Amendment ID: `e20_e21_training_replication_v3`

## Scope

This amendment preserves the statistical analysis frozen in the base contract
and extends only the pre-unblinding provenance registration introduced by
amendment v2. It adds exact dispositions for two E21 snapshots captured after
completed training workers. It does not change a run, seed, model, data split,
training budget, endpoint, estimand, bootstrap stream, or statistical analysis.
The base documents and amendment v2 remain unchanged and are bound by their
SHA-256 values in the v3 manifest.

The registration was made while the joint endpoint embargo remained closed.
Only allowed health fields, infrastructure logs, campaign-attempt metadata,
and endpoint-file path, size, and modification-time metadata were inspected.
No official-test summary content, prediction, individual endpoint, or
aggregate endpoint was opened.

The frozen statistical `analysis_id` remains
`e20_e21_training_replication_v1`. The v3 amendment ID records provenance only
and is not used to derive deterministic bootstrap streams.

## Snapshot boundary retained from amendment v2

The snapshot immediately before each worker remains the exclusivity boundary.
It must identify the registered physical GPU, report a successful process
query, contain no compute process, use at most 512 MiB, and report at most 5%
GPU utilization. A violation closes the joint gate.

The snapshot after a worker is an observation made after that worker process
has returned. It must identify the same registered GPU and have a successful
process query. A nonempty compute-process list is rejected unless an exact
pre-unblinding disposition validates every registered infrastructure file,
hash, attempt identity, snapshot identity, and timing relation. This exception
cannot authorize a non-idle before-worker snapshot.

## Registered events

Amendment v3 carries forward the E20 R/seed 1338995025 event registered by
amendment v2 and adds the following two E21 events.

### E21 R, seed 1619140427

The terminal `summary.json` was atomically finalized at
2026-10-08T08:03:41.508066694Z. The one-second wrapper telemetry then records
the final worker-only allocation at 08:03:42.012Z and an immediately following
foreign-only allocation at 08:03:43.045Z. The post-worker snapshot at
08:03:43.159Z identifies `/usr/local/lib/ollama/llama-server` PID 4026932.
No registered telemetry sample contains both allocations.

### E21 F, seed 646228129

The terminal `summary.json` was atomically finalized at
2026-10-08T09:22:59.428610798Z. The wrapper telemetry records the final
worker-only allocation at 09:23:00.623Z and the immediately following
foreign-only allocation at 09:23:01.663Z. The post-worker snapshot at
09:23:01.021Z identifies `/usr/local/lib/ollama/llama-server` PID 4054197.
No registered telemetry sample contains both allocations.

The foreign allocation then caused the strict idle check before the next E21
worker to fail closed after 30 seconds. No next worker was launched by that
campaign invocation. Before resumption, the 19-row attempt log and the failed
wrapper's command, status, phase, telemetry, and error log were copied to the
immutable pre-resume evidence directory and registered by SHA-256. The guarded
resume uses the same source commit, protocol, configuration, seed order, data,
and physical GPU, and skips only already completed summaries. This is an
infrastructure continuation, not a replacement or endpoint-dependent retry.

## Fail-closed requirements

- The E20 evidence registered in amendment v2 and both E21 registrations must
  validate from their immutable archives.
- Every supporting file must match its registered SHA-256 value.
- Each archived attempt row and its after-worker snapshot must match their
  canonical registered hashes.
- Each endpoint-bearing summary remains unopened during this check. Only its
  registered relative path, size, and nanosecond modification time are used.
- Each registered summary time must precede the first foreign-only telemetry
  sample, and the worker-only and foreign-only samples must be consecutive.
- Every before-worker snapshot must still be idle.
- An unregistered non-idle after-worker snapshot, an unmatched registration,
  or an unused registration keeps the joint endpoint embargo closed.
- All 60 registered E20/E21 runs must independently pass the original health,
  provenance, checkpoint, and completion gate before any endpoint is read.

The original global embargo and every statistical rule in the base contract
remain in force.
