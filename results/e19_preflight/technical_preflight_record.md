# E19 same-machine technical-preflight record

This completed record documents an author-authorized, Codex-operated clean-clone
technical preflight. It is not an independent-human reproduction and does not
claim external archival or closure of the optional independent E19 check.

## Reproducer and immutable source identity

- Reproducer name: Lukáš Jochymek (author authorization); execution operated by OpenAI Codex
- Affiliation or relationship to the project: author and implementation contributor; Codex supplied implementation and orchestration support
- Contact: project-author contact; not duplicated in this evidence record
- UTC start and completion times: 2026-10-15T13:59:32Z to 2026-10-15T16:22:00Z
- Repository URL or confidential release-transport identifier: `ExPoConv-release-93fb31a11a00/repository/ExPoConv-git-93fb31a11a00.bundle`
- Release tag or archive identifier: package `ExPoConv-release-93fb31a11a00`; no release tag
- Full immutable commit SHA (`git rev-parse HEAD`): `93fb31a11a008ba1a378434071cb418b504092d9`
- Initial `git status --porcelain` output (must be empty): empty
- Final `git status --porcelain` output (must be empty): empty
- `results/SHA256SUMS` verification result: all entries passed
- `splits/SHA256SUMS` verification result: all entries passed
- CPU test command and result: `CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu ./.venv/bin/pytest -m "not gpu and not slow and not reference"`; 156 passed, 273 warnings, status 0, 1873.82 s
- Did the reproducer contribute to the implementation? (`yes`/`no`; explain): yes; this is explicitly an author/agent technical preflight, not an independent reproduction
- Run classification (`independent E19`/`same-machine technical preflight`): `same-machine technical preflight`

## Environment and CUDA wrapper

- Host operating system: Linux 6.0.11-1.el7.elrepo.x86_64, glibc 2.17
- CPU and RAM: Intel Xeon Gold 6154; 20 sockets, 18 cores/socket, 720 logical CPUs; 7.1 TiB host RAM reported during the run
- GPU model and VRAM: NVIDIA Tesla V100-PCIE-32GB, 32,768 MiB
- Physical GPU index selected: 1
- NVIDIA driver: 520.61.05
- CUDA runtime reported by JAX: CUDA 11.8 (`nvidia-cuda-runtime-cu11` 11.8.89; cuDNN 8.6.0.163)
- JAX/JAXLIB versions: JAX 0.4.25; `jaxlib` 0.4.25+cuda11.cudnn86
- Python version: 3.11.16
- CUDA wrapper-check command and result: `./scripts/with_cuda_11_8.sh ./scripts/check_cuda.sh`; status 0, `devices=[cuda(id=0)]` under the one-device mask
- Environment-record path and SHA-256: `environment.json`; `10883b286276efd719971799a9284468d25d41b4334276931c104fed3406e68f`
- Environment-record Git commit: `93fb31a11a008ba1a378434071cb418b504092d9`
- Environment-record dirty/status fields: `dirty=false`; `status_porcelain=[]`
- Competing-process observations: both V100s were idle before launch; all 6,090 timed before/after snapshots reported `status=ok`, `process_query_status=ok`, and zero competing compute processes
- Deviations declared before results were inspected: the first no-profile prerequisite shell exited before cloning because the uv-managed `python3.11` directory was not on `PATH`; no experiment or clone had started. The replacement persistent shell prepended the exact uv CPython-3.11.16 directory and ran the frozen commands once. Operation by Codex under author instruction makes this a technical preflight.

## Bounded E02 repair suite

- Command used: `./scripts/with_cuda_11_8.sh ./scripts/run_e02_repair.sh "$e19_root/numerics"`
- Process exit status: 0
- Audit exit status: 0
- `status`: `ok`
- Result rows (must be 424): 424
- Summary `code_commit` and `code_dirty`: `93fb31a11a008ba1a378434071cb418b504092d9`; `false`
- Unique raw-row `code_commit` and `code_dirty` values: only `93fb31a11a008ba1a378434071cb418b504092d9`; only `false`
- `required_contract_pass`: `true`
- Required same-precision passes/rows: 40/40
- `n088_corrected_finite`: `true`
- `n088_mechanism_confirmed`: `true`
- `selected_same_precision_nonfinite`: 0
- Constructed-diagnostic observations: smooth analytic 102/102; exact-cap threshold diagnostic 11/12; N083/N084 same-precision diagnostic 11/20; selected high-precision diagnostic 56/60; N084 analytic-zero curvature rows 0/6. These diagnostics remain outside the required smooth gate.
- `summary.json` path and SHA-256: `numerics/summary.json`; `bdf0b8eb075f156d13cac8af8bcd03f99c199fe1ad2899a2b6f979aec5bd43d4`
- Raw JSONL path and SHA-256: `numerics/e02_repair_results.jsonl`; `48cfc35783872e9b0e843becec2f3444b555e67446ebf0355c54eba6d290b4aa`
- Numerical-check verdict (`reproduced`/`not reproduced`/`partial`): `reproduced` within the bounded technical-preflight scope
- Notes, including interruptions or failures: one execution; status and audit both zero; no numerical interruption or replacement run

## Batch-eight complete-U-Net update timing

The frozen configuration also emits T rows. The E19 efficiency verdict uses
only paired R/F timing. Peak-allocation fields are outside this reproduction
gate and must not be described as independently reproduced here.

- Command used: `POINCARE_EXEC_E08_CONFIG=configs/benchmarks/e08_unet_update_bounded.yaml POINCARE_EXEC_E08_OUTPUT="$e19_root/unet_b8_update.csv" ./scripts/with_cuda_11_8.sh ./scripts/run_e08.sh`
- Process exit status: 0
- Analysis exit status: 0
- Raw-evidence audit passed every check (`yes`/`no`): yes
- Full raw-row `code_commit` set: only `93fb31a11a008ba1a378434071cb418b504092d9`
- Environment commit equals checked-out HEAD (`yes`/`no`): yes
- Raw-row commit equals checked-out HEAD (`yes`/`no`): yes
- All raw rows record `code_dirty == false` (`yes`/`no`): yes
- Raw-row phase/status policy passed (`yes`/`no`): yes; 15 compilation, 15 first-call, 3,000 steady-state, and 15 not-applicable sustained-sequence rows
- Before/after snapshot `status == ok` for every row (`yes`/`no`): yes
- Before/after `process_query_status == ok` for every row (`yes`/`no`): yes
- Snapshots with competing compute processes (must be zero): 0 of 6,090
- Complete R/F paired-session count: 5
- R median latency (ms): 123.235848
- F median latency (ms): 73.883339
- R/F geometric-mean paired speedup: 1.7453064629173736
- R/F 95% log-$t$ interval: [1.5574376287757288, 1.955837327428405]
- R/F interval entirely above one (`yes`/`no`): yes
- T rows retained (`yes`/`no`; descriptive result optional): yes; T median 71.4815775 ms and R/T 1.7471176117718765 [1.5832393093349026, 1.9279586676291822]; outside the E19 gate
- E08 peak allocation reproduced by this E19 run (`no` is expected): no
- Raw CSV path and SHA-256: `unet_b8_update.csv`; `1f79d220e329d33c3315928ca449e57f2f22d4a75604be6329b364460f356c88`
- Session-median table path and SHA-256: `unet_b8_update_session_medians.csv`; `12abec27bbc60f9ec7c9ee785c07452996f439f14fb276090dca529b5747a626`
- Session-median metadata path and SHA-256: `unet_b8_update_session_medians.csv.metadata.json`; `372bb7150c2b1eb473d1d6737f3f784071263ca57237794fda047e09c97e9a83`
- Paired-speedup table path and SHA-256: `unet_b8_update_analysis/paired_speedup_summary.csv`; `3ea5705b7685fa8800b0931c44251d3ff73cab7f069596b6fc02e5d009afc8fc`
- Timing-result verdict (`reproduced`/`not reproduced`/`partial`): `reproduced` within the same-machine technical-preflight scope
- Notes, including interruptions or failures: all 15 fresh-process groups and 3,045 rows completed once; no failure sidecar, missing group, invalid phase, nonfinite row, or replacement session

## Overall E19 disposition and external archive

- Overall verdict (`reproduced`/`not reproduced`/`partial`): `reproduced` as a same-machine clean-clone technical preflight
- Evidence-directory identifier: `/data2/joc0027/e19-preflight-20261015/ExPoConv-e19/artifacts/e19/lukas-codex-preflight-20261015`
- `SHA256SUMS` path and SHA-256: `SHA256SUMS`; `6c56afc3eea2b44d2782a6172aa5eeab229fe9367a873e43c0304952362e25f2`
- External immutable archive or confidential deposit identifier: none; complete raw evidence retained locally outside the working repository
- External archive checksum or fixity record: not applicable to this technical preflight
- Unresolved deviations or questions: no result-level deviation; this run used the original V100 host and an author/agent operator, so it is not evidence of independent usability or portability. It is bound to the historical release-candidate commit named above rather than a future integrated documentation commit.
- Operator statement: OpenAI Codex ran the experiment commands in one persistent
  shell against the immutable release commit and retained failed, interrupted,
  and unfavorable outcomes without selective omission.
- Independence statement: not attested. Lukáš Jochymek and OpenAI Codex
  contributed to the evaluated project, so this record cannot satisfy the
  independent-reproducer declaration in the blank E19 template.
- Reproducer signature or authenticated approval: Lukáš Jochymek authorized execution and recording under his name in the authenticated project workspace; OpenAI Codex operated and transcribed the run. No independent-reproducer signature is claimed.
- UTC approval date: 2026-10-15

## Author review (does not replace independent approval)

- Author reviewed the signed record without rewriting observations (`yes`/`no`): no; prior author authorization is recorded above, but no post-result independent review is claimed
- Author-review UTC date: not applicable
- E19 marked closed only after external archival (`yes`/`no`): no; the independent-human E19 claim remains unmade and is not a publication prerequisite
