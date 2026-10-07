# E19 independent reproduction record

This is a blank record. Do not mark E19 complete until an independent human
reproducer has filled every applicable field, signed the record, and archived
the record and named evidence outside the author's working repository.

## Reproducer and immutable source identity

- Reproducer name:
- Affiliation or relationship to the project:
- Contact:
- UTC start and completion times:
- Repository URL or confidential release-transport identifier:
- Release tag or archive identifier:
- Full immutable commit SHA (`git rev-parse HEAD`):
- Initial `git status --porcelain` output (must be empty):
- Final `git status --porcelain` output (must be empty):
- `results/SHA256SUMS` verification result:
- `splits/SHA256SUMS` verification result:
- CPU test command and result:
- Did the reproducer contribute to the implementation? (`yes`/`no`; explain):
- Run classification (`independent E19`/`same-machine technical preflight`):

## Environment and CUDA wrapper

- Host operating system:
- CPU and RAM:
- GPU model and VRAM:
- Physical GPU index selected:
- NVIDIA driver:
- CUDA runtime reported by JAX:
- JAX/JAXLIB versions:
- Python version:
- CUDA wrapper-check command and result:
- Environment-record path and SHA-256:
- Environment-record Git commit:
- Environment-record dirty/status fields:
- Competing-process observations:
- Deviations declared before results were inspected:

## Bounded E02 repair suite

- Command used:
- Process exit status:
- Audit exit status:
- `status`:
- Result rows (must be 424):
- Summary `code_commit` and `code_dirty`:
- Unique raw-row `code_commit` and `code_dirty` values:
- `required_contract_pass`:
- Required same-precision passes/rows:
- `n088_corrected_finite`:
- `n088_mechanism_confirmed`:
- `selected_same_precision_nonfinite`:
- Constructed-diagnostic observations:
- `summary.json` path and SHA-256:
- Raw JSONL path and SHA-256:
- Numerical-check verdict (`reproduced`/`not reproduced`/`partial`):
- Notes, including interruptions or failures:

## Batch-eight complete-U-Net update timing

The frozen configuration also emits T rows. The E19 efficiency verdict uses
only paired R/F timing. Peak-allocation fields are outside this reproduction
gate and must not be described as independently reproduced here.

- Command used:
- Process exit status:
- Analysis exit status:
- Raw-evidence audit passed every check (`yes`/`no`):
- Full raw-row `code_commit` set:
- Environment commit equals checked-out HEAD (`yes`/`no`):
- Raw-row commit equals checked-out HEAD (`yes`/`no`):
- All raw rows record `code_dirty == false` (`yes`/`no`):
- Raw-row phase/status policy passed (`yes`/`no`):
- Before/after snapshot `status == ok` for every row (`yes`/`no`):
- Before/after `process_query_status == ok` for every row (`yes`/`no`):
- Snapshots with competing compute processes (must be zero):
- Complete R/F paired-session count:
- R median latency (ms):
- F median latency (ms):
- R/F geometric-mean paired speedup:
- R/F 95% log-$t$ interval:
- R/F interval entirely above one (`yes`/`no`):
- T rows retained (`yes`/`no`; descriptive result optional):
- E08 peak allocation reproduced by this E19 run (`no` is expected):
- Raw CSV path and SHA-256:
- Session-median table path and SHA-256:
- Session-median metadata path and SHA-256:
- Paired-speedup table path and SHA-256:
- Timing-result verdict (`reproduced`/`not reproduced`/`partial`):
- Notes, including interruptions or failures:

## Overall E19 disposition and external archive

- Overall verdict (`reproduced`/`not reproduced`/`partial`):
- Evidence-directory identifier:
- `SHA256SUMS` path and SHA-256:
- External immutable archive or confidential deposit identifier:
- External archive checksum or fixity record:
- Unresolved deviations or questions:
- Reproducer statement: I ran the commands in one persistent shell against the
  immutable release commit and recorded unfavorable, failed, and interrupted
  outcomes without selective omission.
- Reproducer statement: I did not implement the evaluated method or prepare its
  accepted experimental evidence.
- Reproducer signature or authenticated approval:
- UTC approval date:

## Author review (does not replace independent approval)

- Author reviewed the signed record without rewriting observations (`yes`/`no`):
- Author-review UTC date:
- E19 marked closed only after external archival (`yes`/`no`):
