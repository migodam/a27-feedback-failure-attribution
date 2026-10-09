# A27 infrastructure handoff

Status: local implementation and synthetic verification PASS. Remote preflight, deployment and scientific stages NOT_RUN by this worker. No commit was created.

## Owned files

- `src/a27/budget.py`: adapted from frozen A26, independent `results/a27/jobs` receipts and `runs/a27-accounting.lock`; CPU hard cap 3600 seconds, GPU occupied-wall hard cap 3600 seconds, zero new full-forward labels.
- `tools/a27_remote.py`: existing private-helper transport interface retained; remote target `D:/AI/A27_FEEDBACK_FAILURE_ATTRIBUTION`; stages `audit/tests/post/diagnostics/observability/latency/report` and driver `tools/a27_driver.py`.
- `tools/a27_driver.py`: paid foreground stage harness, CPU default, import/configuration startup charged, CUBLAS workspace policy preserved, CUDA blocking host wait configured before resource sampling.
- `tests/test_a27_budget.py`: 16 local synthetic tests covering accounting, transport boundary and stage-driver behavior.

## Accounting and immutability

Only A27 accepted receipts restore totals. A26/A25 receipts and old source/results are excluded from A27 history and deployment accounting. Failed and interrupted receipts retain resource costs; dangling started manifests block new work; terminal receipts cannot be overwritten. Durable reservations restore once, truncated reservation evidence blocks new work, and every new-label alias is denied regardless of the supplied action limit. Foreign or changed locks are retained. The shared GPU lock remains exactly `D:/AI/A20_OPM_IMAGING/runs/gpu.lock`.

The native child, controller/import/stop tail, and local transport costs retain the existing A26 rules. CPU work charges max(wall, process CPU) for CPU jobs; CUDA jobs charge process CPU plus entire GPU occupied wall. A CUDA host transport monitor charges process CPU without charging remote GPU occupation twice. Native/controller/local transport records are all routed to A27 results. Controller stop reserves remain 15 CPU seconds and 30 GPU seconds. Nonfinite, zero, negative or Boolean remote timeouts are rejected before any transport call.

## Transport and preflight interface

Actions remain `preflight`, `push`, `run`, `pull`; existing `--transport-helper` and `--private-config` arguments are required/used as before. Config default is `configs/a27/pilot.json`, timeout default 3600 seconds. Source deployment includes `configs/a27`, existing `configs/a26`, existing `configs/a25`, root configs, and Python source/tools/vendor/tests. Development data and frozen A26 result payloads are not transported. Only completed local A27 job receipt/manifest pairs are copied for accounting; the active local transport manifest is excluded.

Preflight is a short inline read-only query: package/device metadata, GPU queue query, shared-lock availability, A26 checkpoint/cache/prediction path availability, and A25 development path availability. It never opens checkpoint/cache/measurement payloads or locked data, and it creates no remote files, job or lock. Outputs include `queue_clear`, `a26_input_availability`, `a25_input_availability`, and `read_only_inputs_available`. A26 root is `D:/AI/A26_OM_FEEDBACK_PRIOR_COMPLETION`; A25 data root remains `D:/AI/A25_ADAPTIVE_OM_FEEDBACK/data`. No checksummed hash computation was introduced.

## Driver interface

Arguments: `--root`, `--data-root`, `--a26-root`, `--config`, required `--job`, required `--stage`, `--device {cpu,cuda}`. Local defaults point to sibling Gaussian/A25 and Gaussian/A26 projects. For every stage except tests, dispatch is:

```python
pipeline.stage_STAGE(root, data_root, a26_root, config, book, device)
```

The parent-owned pipeline loads the frozen A26 configuration separately. The tests stage discovers all inherited and A27 `test_*.py` unittest files under `active_ledger(book)` and writes `results/a27/jobs/JOB/TESTS.log`, `TESTS_SUMMARY.json`, and final `RESULT.json`. Counts include tests/errors/failures/skipped and explicit skipped-test reasons. Stage exceptions preserve a FAILED native receipt and `ERROR.log`. Successful receipts include startup/import/configuration elapsed time; the corresponding deadline is shifted to the same process start.

## Verification

`VERIFICATION.json` and `LOCAL_TESTS.log`: 16 tests passed, 0 errors, 0 failures, 0 skipped. All four owned source/test files compiled without creating bytecode; the driver help interface also ran locally. Generated remote controller and preflight source compiled inside mocked tests; no helper, SSH, transport or remote child was invoked.

The first expanded driver test run had one fixture expectation failure because macOS resolves `/var` temporary paths to `/private/var`. The expected dispatched root was corrected to its canonical path. `VERIFICATION_ATTEMPT1.json` and `LOCAL_TESTS_ATTEMPT1.log` retain that failed attempt. No implementation behavior was relaxed to make it pass.

These are mechanical infrastructure checks. They do not verify live remote inputs, GPU execution, scientific results or any scientific gate. Parent review is needed for scientific claims and actual bounded remote execution.
