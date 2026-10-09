# A27 actual latency output QA

Scope: saved `results/a27/latency` JSON/CSV only, plus read-only inspection of timing boundaries. No new physics, tests, benchmarks, jobs, labels, locked arrays, or source edits. Parent owns budget and scientific claims. This is a factual numerical/timing receipt review, not research acceptance.

## Completeness and recorded failures

| Output | Expected | Observed | Key/status check |
| --- | ---: | ---: | --- |
| Native encoder | 432 | 432 | 9 arms × 16 cases × 3 repeats; unique grid; all OK |
| Stage profiles | 96 | 96 | 6 arms × 16 cases; all OK |
| Raw-to-corrected image | 192 | 192 | 4 arms × 16 cases × 3 repeats; all OK |
| Numerical equivalence | 96 | 96 | 6 arms × 16 cases; all receipt tests pass |
| Failure records | — | 0 | `FAILURES.json` is empty |

All four outputs cover the same 16 case IDs: four families × two backgrounds × two amplitudes, all 20 dB, four distinct parents. They are repeated conditions from a fixed representative cohort, not 48 independent observations per arm. Native/post CSV row keys and times agree exactly with JSON. All full times are positive/finite; all 2,576 stage span records are OK with finite times. Native/stage outputs each report zero new labels; no campaign-budget conclusion is inferred here.

## Equivalence and action counts

All 96 receipts have exactly zero feature and initializer maximum differences. Baseline views and beta-only omission also have zero base/image differences. LU-only and combined variants have maximum base absolute difference `3.469446951953614e-18` and decoder-mass relative difference `4.101467017750089e-17`, within the saved tolerances. Float64 base/image equality is therefore numerical, rather than universal bitwise equality. All 32 beta-omitting receipts explicitly say `EXPLICITLY_OMITTED_NOT_COMPARED`; the other 64 compare the original beta/NaN groups. Omitted beta is never treated as an equal computed beta.

The 288 variant native rows carry `lossless_equivalence_passed=true`; the underlying comparisons are the 96 case/arm receipts from repeat 0, propagated to the corresponding repeated calls. They are not 288 separate saved output comparisons.

Per-image action counts are constant across the 48 native calls of each arm:

| N2 arm | Relative forwards / source RHS | Forward residual audits | Core LU | Core solve RHS | Extra core backward audits |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original / baseline view | 2 / 16 | 2 | 2 | 16 | 0 |
| No-beta only | 1 / 8 | 1 | 2 | 16 | 0 |
| LU reuse only | 2 / 16 | 2 | 1 | 16 | 2 |
| Combined | 1 / 8 | 1 | 1 | 16 | 2 |

Beta omission removes the unused L diagnostic forward: Db forward actions fall 16→8, while Db adjoint actions remain 16. LU reuse has one explicit reuse call/system and no cache invalidations per image; it reduces factor construction, not solved RHS. R core guards remain one per image. No lazy observation SVD is charged to any timed native row, consistent with paid warm-up. The extra retained solve-residual audits are part of the reuse arm's cost.

## Native timing observations

Pooled medians across 48 calls per arm, in milliseconds:

| Arm | Median | Relative to original N2 median |
| --- | ---: | ---: |
| Original N1 | 45.445 | — |
| Original N2 | 99.836 | 1.000 |
| Original N3 | 40.642 | — |
| N2 no-beta only | 86.491 | 0.866; 13.37% lower |
| N2 LU reuse only | 114.493 | 1.147; 14.68% higher |
| N2 combined | 97.693 | 0.979; 2.15% lower |

This saved run shows that fewer LU factorizations did not imply lower full inference time. Beta-only omission has the lowest observed N2 median; combined remains slower than beta-only. Relative to the faster original control median (N3), beta-only is 2.128×, LU-only 2.817×, combined 2.404×, and original N2 2.456×. These are ratios of pooled timing medians, not uncertainty bounds, physical-quality gains, or extension-gate judgments.

## Stage meaning and synchronization opportunities

Stage medians below are from one separate instrumented call per condition. They contain CUDA-event/Torch-dispatch instrumentation and must not replace native latency.

| Profile view | Instrumented full ms | Guard intervals, host ms | Book meter host ms | CUDA host-result/copy opportunities |
| --- | ---: | ---: | ---: | ---: |
| N1 baseline | 83.995 | 10.800 | 6.762 | 25 = 19 scalar + 2 equality + 4 copies |
| N2 baseline | 150.384 | 51.981 | 8.812 | 41 = 32 scalar + 2 equality + 7 copies |
| N3 baseline | 75.638 | 10.497 | 5.835 | 23 = 17 scalar + 2 equality + 4 copies |
| N2 no-beta only | 130.789 | 44.384 | 8.000 | 32 = 24 scalar + 2 equality + 6 copies |
| N2 LU reuse only | 189.208 | 77.258 | 10.460 | 55 = 41 scalar + 5 equality + 9 copies |
| N2 combined | 167.486 | 69.088 | 9.327 | 46 = 33 scalar + 5 equality + 8 copies |

The saved profile distinguishes core assembly, core solve, feedback adjoint, source aggregation, material ratio/denominator guard, output guards, beta, logging, decoder, and transfer. N2 core assembly has about 2.0 ms median host interval. Baseline core solve combines LU and solve (7.809 ms host / 8.147 ms CUDA event); a separate baseline LU duration is correctly unavailable. LU-only records separate LU (3.111 / 3.313 ms) and core solves (5.944 / 6.355 ms). Its feedback-adjoint enclosure is 21.579 ms host versus baseline 9.448 ms; source aggregation is around 1.6–1.7 ms and material-ratio-plus-guard around 6.0–6.5 ms. Omitted-beta arms retain a small named beta bookkeeping span (about 0.195 ms host), which is not computation of a beta value. These are instrumented intervals; they do not isolate pure math or provide causal native cost attribution.

Guard/adjoint/core intervals overlap hierarchically and must not be summed as full latency. Host intervals are elapsed wall, not CPU process time. CUDA events measure current-stream marker intervals, including possible host-induced idle and marker overhead, not pure GPU kernel compute. Opportunity counts are constant within each arm's 16 profiles, but all actual wait-duration fields are null. The observer counts visible Torch-dispatch operations; it does not measure every implicit library synchronization or prove that every opportunity caused a nonzero wait.

## Overhead/reporting limits and post-correction rows

- Native rows have no stage or dispatch observer and no instrumented-full value. Stage rows have no native-full value. All 528 encoder/profile rows explicitly report no per-span budget/file-lock checks; outer budget-check time is about 2 ms and lies outside the native image window while remaining part of the paid job.
- `stage_meter_host_seconds` is only the book/event-recording code's own measured overhead. It is not total instrumentation inflation. For N2 baseline, the median case-matched stage-minus-native difference is 53.469 ms, whereas the median book-meter value is 8.812 ms. Dispatch observation, marker placement, and run variation are not exhaustively attributed. One N1 case even has an instrumented call 8.126 ms below its three-call native case median, so these differences cannot be treated as a fixed subtractable profiling tax.
- Reuse adds more visible host-result/copy opportunities (41→55; combined still 46). Those counts coincide with slower LU-only native timing, but the saved outputs do not establish how many milliseconds came from waiting, guard kernels, snapshots, library behavior, or other overhead. A pure-LU speedup or kernel-purity claim would contradict these receipts.
- All 192 post rows name `post_source_folder='post_fgp'`. Native post medians are NONE 38.928 ms, L 391.038 ms, F 259.714 ms, TASK_R32 251.029 ms. Every corrected row reports a paid full-F acceptance audit and at least two full LU constructions; NONE reports none. No proximal failure counter is recorded. The latency files do not include individual KKT/gap or acceptance/rejection metadata, so those certificates/outcomes require the parent-owned correction artifacts rather than inference from timing status alone.
- Encoder-native rows use outer-only explicit profiling fences. The matching correction implementation also contains two correction-local CUDA timer synchronizations; corrected post full-wall rows therefore include those fences in addition to the outer image boundary. Describing all post rows as the same outer-only synchronization protocol would be inaccurate.

No missing rows, nonfinite equivalence/timing values, saved safety failures, beta-contract contradictions, or unintended timed SVD/cache carries were found. Remaining limits are timing interpretation, comparison scope, repeated-cohort sampling, and absent post-certificate details in these particular files.
