# E19 independent reproduction runbook

## Status, scope, and closure rule

E19 is **open**. This runbook defines a bounded independent check; it does not
claim that anybody has completed it. E19 covers only:

1. the 424-row bounded E02 repair suite in
   `configs/numerics/e02_repair.yaml`; and
2. paired R-versus-F timing for the batch-eight complete-U-Net Adam update in
   `configs/benchmarks/e08_unet_update_bounded.yaml`.

The frozen E08 configuration also runs T. Retain those rows, but T is an
ablation and is not part of the E19 efficiency gate. E19 does **not** reproduce
the E02 structured follow-up, either training campaign, or the E08
process-lifetime peak-allocation result. The timing raw file contains allocator
fields, but this runbook neither exports nor accepts a memory estimate.

A clean clone run by the author, Codex, or another agent on the original host is
only a technical preflight. E19 closes only when another person who did not
implement the method runs an immutable released commit, signs the completed
[`e19_reproduction_record_TEMPLATE.md`](e19_reproduction_record_TEMPLATE.md),
and the signed record plus its evidence are archived outside the author's
working repository.

## Independence and platform

The independent reproducer must:

- obtain a new clone from the released repository or independently clone a
  released Git bundle; a plain source archive is insufficient because the
  evidence capture binds results to Git identity;
- check out the exact 40-character commit named by the frozen release;
- make no source, protocol, configuration, or dependency changes;
- execute every command below in **one persistent Bash shell**, without
  running the numerical and timing parts concurrently;
- record failed, interrupted, and unfavorable attempts without replacement;
- use exactly one otherwise idle physical GPU; and
- retain raw outputs, console logs, status files, environment capture, and
  hashes.

For direct comparison with the paper, use one 32-GB NVIDIA Tesla V100 with the
pinned CUDA 11.8 environment. Another accelerator provides portability
evidence but does not close the same-platform E19 gate. Declare any unavoidable
deviation before inspecting results and classify the reproduction as partial.

## 1. Start one persistent shell and verify the release

Open one Bash process and keep it open through Section 5. Replace all angle
bracket placeholders before running the block. Do not copy an existing working
tree.

```bash
bash --noprofile --norc
set -euo pipefail
umask 077

for command in git python3.11 uv nvidia-smi sha256sum find sort xargs tee; do
  command -v "$command" >/dev/null
done

git clone <released-repository-url-or-bundle-path> ExPoConv-e19
cd ExPoConv-e19
git checkout --detach <immutable-release-commit>
e19_commit=$(git rev-parse HEAD)
[[ "$e19_commit" =~ ^[0-9a-f]{40}$ ]]
test -z "$(git status --porcelain)"

(cd results && sha256sum --check SHA256SUMS)
(cd splits && sha256sum --check SHA256SUMS)

./scripts/bootstrap_cuda11_v100.sh
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu \
  ./.venv/bin/pytest -m "not gpu and not slow and not reference"
test -z "$(git status --porcelain)"

e19_root="artifacts/e19/<reproducer-id>-<utc-date>"
mkdir -p "$e19_root"
export CUDA_VISIBLE_DEVICES="<physical-gpu-index>"
[[ "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+$ ]]

set +e
./scripts/with_cuda_11_8.sh ./scripts/check_cuda.sh \
  2>&1 | tee "$e19_root/cuda-wrapper-check.log"
cuda_wrapper_status=${PIPESTATUS[0]}
set -e
printf '%s\n' "$cuda_wrapper_status" > "$e19_root/cuda-wrapper-check.status"
test "$cuda_wrapper_status" -eq 0
```

The checksum commands replace the unavailable historical `verify_release.py`
workflow. The CPU tests, CUDA wrapper check, initial status, and exact commit
must be copied into the reproduction record. A successful wrapper check proves
that this shell exposes a JAX GPU through the pinned environment; it does not by
itself establish GPU idleness.

## 2. Reproduce the bounded 424-row numerical subset

Continue in the same shell:

```bash
set +e
./scripts/with_cuda_11_8.sh \
  ./scripts/run_e02_repair.sh "$e19_root/numerics" \
  2>&1 | tee "$e19_root/numerics.log"
numerics_status=${PIPESTATUS[0]}
set -e
printf '%s\n' "$numerics_status" > "$e19_root/numerics.status"

set +e
E19_COMMIT="$e19_commit" E19_ROOT="$e19_root" ./.venv/bin/python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["E19_ROOT"])
expected = os.environ["E19_COMMIT"]
summary = json.loads((root / "numerics/summary.json").read_text())
assert summary["status"] == "ok"
assert summary["result_rows"] == 424
assert summary["code_commit"] == expected
assert summary["code_dirty"] is False
assert summary["required_contract_pass"] is True
assert summary["required_same_precision_passes"] == 40
assert summary["required_same_precision_rows"] == 40
assert summary["n088_corrected_finite"] is True
assert summary["n088_mechanism_confirmed"] is True
assert summary["selected_same_precision_nonfinite"] == 0

raw_path = root / "numerics/e02_repair_results.jsonl"
rows = [json.loads(line) for line in raw_path.read_text().splitlines() if line]
assert len(rows) == 424
assert {row["code_commit"] for row in rows} == {expected}
assert {row["code_dirty"] for row in rows} == {False}
print("bounded E02 identity and acceptance checks passed")
PY
numerics_audit_status=$?
set -e
printf '%s\n' "$numerics_audit_status" > "$e19_root/numerics.audit.status"
```

If the process or audit is nonzero, preserve the directory and classify this
part as not reproduced or partial. Do not delete it and rerun only to replace
the outcome. The exact-cap and analytic-zero cases remain diagnostics outside
the smooth acceptance gate; do not silently remove or relabel them.

## 3. Capture the environment and reproduce paired R/F timing

Still in the same shell, capture the environment immediately before timing:

```bash
./scripts/with_cuda_11_8.sh \
  ./.venv/bin/poincare-exec capture-environment \
  --output "$e19_root/environment.json"

E19_COMMIT="$e19_commit" E19_ENV="$e19_root/environment.json" \
  ./.venv/bin/python - <<'PY'
import json
import os
from pathlib import Path

environment = json.loads(Path(os.environ["E19_ENV"]).read_text())
assert environment["git"]["commit"] == os.environ["E19_COMMIT"]
assert environment["git"]["dirty"] is False
assert environment["git"]["status_porcelain"] == []
visible = environment["environment"]["CUDA_VISIBLE_DEVICES"]
assert isinstance(visible, str) and visible.isdigit()
assert environment["jax"]["backend"] == "gpu"
print("environment is bound to the clean release commit")
PY

set +e
POINCARE_EXEC_E08_CONFIG=configs/benchmarks/e08_unet_update_bounded.yaml \
POINCARE_EXEC_E08_OUTPUT="$e19_root/unet_b8_update.csv" \
./scripts/with_cuda_11_8.sh ./scripts/run_e08.sh \
  2>&1 | tee "$e19_root/unet_b8_update.log"
benchmark_status=${PIPESTATUS[0]}
set -e
printf '%s\n' "$benchmark_status" > "$e19_root/unet_b8_update.status"
```

The runner uses five fresh-process paired sessions, 50 warm-up calls, and 200
synchronized observations per path and session. It refuses a non-idle selected
GPU before each worker. Do not lower the counts or distribute variants across
GPUs.

## 4. Audit raw timing evidence and apply the registered estimator

Run the analysis only when the raw CSV exists. The first exporter binds the
checked-out HEAD, environment capture, and every raw-row `code_commit`. Raw-row
status must be `ok` for `compilation`, `first_compiled_call`, and
`steady_state`; `sustained_sequence` may be `not_applicable` (the current
schema) or `ok`. Every other phase/status combination is rejected. The
exporter additionally requires `status == ok` and
`process_query_status == ok` in every before/after GPU snapshot, rejects
snapshots with competing compute processes, and rejects incomplete,
latency-deficient, or mixed-identity groups.

```bash
analysis_status=0
set +e
./.venv/bin/python scripts/export_existing_timing_sessions.py \
  --input "$e19_root/unet_b8_update.csv" \
  --output "$e19_root/unet_b8_update_session_medians.csv" \
  --environment "$e19_root/environment.json" \
  --expected-code-commit "$e19_commit"
analysis_status=$?

if test "$analysis_status" -eq 0; then
  ./.venv/bin/python scripts/export_corrected_anchor_tables.py \
    --raw "$e19_root/unet_b8_update.csv" \
    --session-medians "$e19_root/unet_b8_update_session_medians.csv" \
    --config configs/benchmarks/e08_unet_update_bounded.yaml \
    --environment "$e19_root/environment.json" \
    --log "$e19_root/unet_b8_update.log" \
    --status-file "$e19_root/unet_b8_update.status" \
    --output-dir "$e19_root/unet_b8_update_analysis"
  analysis_status=$?
fi
set -e
printf '%s\n' "$analysis_status" > "$e19_root/unet_b8_update.analysis.status"

if test "$analysis_status" -eq 0; then
  E19_TABLE="$e19_root/unet_b8_update_analysis/paired_speedup_summary.csv" \
  E19_LATENCY_TABLE="$e19_root/unet_b8_update_analysis/workload_latency_summary.csv" \
    ./.venv/bin/python - <<'PY'
import csv
import os
from pathlib import Path

with Path(os.environ["E19_TABLE"]).open(newline="") as stream:
    rows = list(csv.DictReader(stream))
selected = [
    row for row in rows
    if row["workload_id"] == "unet_b8_update"
    and row["baseline_variant"] == "R"
    and row["candidate_variant"] == "F"
]
assert len(selected) == 1
row = selected[0]
assert int(row["paired_sessions"]) == 5

with Path(os.environ["E19_LATENCY_TABLE"]).open(newline="") as stream:
    latency_rows = list(csv.DictReader(stream))
latencies = {
    item["variant"]: item
    for item in latency_rows
    if item["workload_id"] == "unet_b8_update"
}
assert set(latencies) == {"R", "F", "T"}
assert {int(item["sessions"]) for item in latencies.values()} == {5}
print("R median latency (ms):", latencies["R"]["median_latency_ms"])
print("F median latency (ms):", latencies["F"]["median_latency_ms"])
print(
    "R/F speedup:",
    row["geometric_mean_speedup"],
    "95% interval:",
    row["log_t_95_interval_lower"],
    row["log_t_95_interval_upper"],
)
PY
fi
```

Timing is reproduced when the complete raw-evidence audit passes, five R/F
session pairs are present, and the geometric-mean R/F speedup and its unadjusted
95% log-$t$ interval are entirely above one. The estimate need not equal the
paper's `1.923` exactly. If the interval includes or falls below one, retain the
result and reconsider the systems claim; do not replace the sessions because
the outcome is unfavorable.

T rows are retained because the frozen configuration generates them, but they
do not determine the E19 verdict. Do not report E08 peak allocation as
independently reproduced from this runbook.

## 5. Complete, sign, hash, and archive the record

Continue in the same shell. Hash the evidence first. The record is excluded
from `SHA256SUMS`, so the reproducer can put the checksum-file digest into the
record without creating a circular digest. Fill and sign the record only after
the evidence checksum has been printed.

```bash
cp docs/e19_reproduction_record_TEMPLATE.md \
  "$e19_root/reproduction_record.md"

(
  cd "$e19_root"
  find . -type f \
    ! -name SHA256SUMS \
    ! -name reproduction_record.md \
    -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
  sha256sum --check SHA256SUMS
  sha256sum SHA256SUMS
)

# Put the printed SHA256SUMS digest in reproduction_record.md, complete every
# applicable field, and sign the record. Then freeze its separate digest:
(
  cd "$e19_root"
  sha256sum reproduction_record.md > reproduction_record.md.sha256
  sha256sum --check reproduction_record.md.sha256
)

git_status_final=$(git status --porcelain)
test -z "$git_status_final"
printf '%s\n' "$e19_commit"
```

The reproducer must archive the signed record, `SHA256SUMS`, and retained E19
directory in an external release location and place that immutable identifier
in the record. The author may review the package but must not rewrite the
reproducer's observations. Until an independent human has done this, E19 stays
open even if a same-machine clean-clone preflight passes.
