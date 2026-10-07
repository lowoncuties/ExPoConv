# E20/E21 pre-unblinding analysis contract amendment v2

Status: frozen before inspection of any E20/E21 official-test endpoint value

Frozen: 2026-10-07

Base contract: `docs/e20_e21_analysis_contract.md`

Base manifest: `configs/analysis/e20_e21_training_replication.yaml`

Amendment manifest: `configs/analysis/e20_e21_training_replication_v2.yaml`

Amendment ID: `e20_e21_training_replication_v2`

## Scope

This amendment changes only the interpretation and audit of GPU snapshots
captured after a training worker has terminated. It does not change a run,
seed, model, data split, training budget, endpoint, estimand, or statistical
analysis. The base contract and manifest remain unchanged and are bound by
their SHA-256 values in the amendment manifest.

The correction was required by infrastructure-only evidence observed while
the global endpoint embargo remained closed. No official-test summary,
prediction, individual endpoint, or aggregate endpoint was opened to make this
decision.

The frozen statistical `analysis_id` remains
`e20_e21_training_replication_v1`. The amendment ID records provenance only
and is not used to derive deterministic bootstrap streams.

## Corrected snapshot boundary

The snapshot immediately before each worker is the exclusivity boundary. It
must identify the registered physical GPU, report a successful process query,
contain no compute process, use at most 512 MiB, and report at most 5% GPU
utilization. A violation closes the joint gate.

The snapshot after a worker is an observation made after the worker process
has returned. It must identify the same registered GPU and have a successful
process query. An empty compute-process list is accepted even if the sampled
utilization counter has not yet decayed. A nonempty list is rejected unless an
exact disposition was registered before endpoint unblinding and all registered
infrastructure files, hashes, attempt identity, snapshot identity, and timing
relations validate. This exception cannot authorize a non-idle before-worker
snapshot.

## Registered post-worker event

One E20 attempt, reference path R at seed 1338995025, has a non-idle
after-worker snapshot. The after snapshot names
`/usr/local/lib/ollama/llama-server` and was taken at
2026-10-07T20:13:52.671Z. The worker's terminal `summary.json` had already
been atomically finalized at 2026-10-07T20:13:50.720105567Z. In the raw
one-second wrapper telemetry, the last worker-allocation sample is at
20:13:51.779Z and the immediately following foreign-only allocation sample is
at 20:13:52.817Z. No telemetry sample contains both allocations. The foreign
process then caused the next run's existing pre-worker idle gate to abort the
campaign before that next worker started.

The accepted source writes `summary.json` only after fixed-final and
validation-selected official and duplicate-excluded evaluations and their
prediction files are complete. The registered time ordering therefore places
the foreign-only allocation after the completed scientific worker. The run is
not classified as overlapped or restarted. The interrupted campaign wrapper,
attempt snapshot, telemetry, command, status, phase, and error log remain
retained as infrastructure evidence.

## Fail-closed requirements

- The archived pre-v3 attempt log and every supporting wrapper file must match
  its registered SHA-256 value.
- The E20/R/1338995025 attempt row and after snapshot must match their
  canonical registered hashes.
- The endpoint-bearing summary is not opened during this check. Only its
  registered path, size, and nanosecond modification time are inspected.
- The registered telemetry transition must be present as consecutive lines,
  and the summary time must precede the first foreign-only sample.
- Any unregistered non-idle after-worker snapshot, any unmatched disposition,
  or any non-idle before-worker snapshot keeps the joint endpoint embargo
  closed.

The original global 60-run embargo and every other rule in the base contract
remain in force.
