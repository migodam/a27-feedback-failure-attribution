# A27 frozen physics reuse and latency module

Status: IMPLEMENTED; static syntax/whitespace checks passed; runtime CPU tests NOT_RUN by this worker; CUDA NOT_RUN. Parent reported run002 failed during test import because discovery supplies the tests directory directly; the fixture import was corrected to `from test_a26_encoder_decoder import _tiny_cache`. Parent retains that failure log and owns the rerun, charged test/profile jobs, driver, result paths, resources, interpretation, and scientific claims. No inherited A25/A26 file was edited and no commit was created.

## Frozen source and scope

- Working tree starts at A26 commit `2603f3bd16c48e48587fedc4363975779def17f5` (verified locally).
- Source interfaces inspected: `src/a25/torch_physics.py`, `src/a26/encoder.py`, `src/a26/evaluation.py`, `tools/a26_driver.py`, `configs/a26/pilot.json`, and existing CPU fixture/cost contracts.
- No datasets, locked data, labels, teachers, full-current truths, training, full-wave label generation, rank changes, device jobs, or hashes were opened/run by this worker.
- A26's L forward in N2 supplies diagnostic beta only. R supplies its data gradient, physical base, field/current-power features, and matching observation feature. Omitting the L diagnostic forward therefore changes the diagnostic contract, not the intended feature/base equations.

## Public interfaces

`a27.fast_physics.ReusingTorchRelativeModel(reference)` accepts the already validated nonlocal TASK_R32 TorchReference. It retains `forward`, `jvp`, `adjoint`, `vjp_complex`, and `data_gradient` call signatures. Each material state stores its own LU; identical material values in different states never silently share a factor.

`a27.fast_physics.OptimizedFrozenEncoder(background_path, config, book, device='cpu', *, omit_beta=False, reuse_forward_lu=False)` runs the original FrozenEncoder loading/provenance/known-background validation before configuring a variant. `OptimizedFrozenEncoder.from_encoder(existing_encoder, *, ...)` constructs a resident view of the same already validated reference/operators, with fresh model owners. These views must be used sequentially because the reference's book binding is shared.

`a27.latency.build_profile_arms(existing_encoder)` returns six entries, each `(ProfileArm, encoder_view)`:

| Arm | Method | Diagnostic beta | Forward LU reuse |
| --- | --- | --- | --- |
| N1_BASELINE | N1 | Original no-feedback status | No |
| N2_BASELINE | N2 | Computed from matched L/R | No |
| N3_BASELINE | N3 | Original no-feedback status | No |
| N2_NO_BETA_ONLY | N2 | Explicitly omitted | No |
| N2_LU_REUSE_ONLY | N2 | Computed from matched L/R | Yes |
| N2_COMBINED | N2 | Explicitly omitted | Yes |

The parent should also measure the untouched `FrozenEncoder` for the actual A26 native baseline. The instrumented baseline views preserve its operation order/equations and supply finer named spans; their additional host instrumentation is not a claim about untouched native latency.

`a27.latency.profile_inference(encoder, decode, observation, sigma, method, *, book=None, timing_mode='native', transfer=None, metadata=None)` accepts `decode(encoded) -> material_mass_tensor`, so the parent retains checkpoint normalization, model selection, and weights. Optional `transfer(mass, encoder)` owns image formatting. The default returns a complex NumPy `[batch, cells]` image. The result contains `encoded`, `mass`, `image`, and `timing`. It creates no jobs/result files, warms no models, selects no observations, and performs no offline full audit.

`a27.latency.compare_encoder_outputs(reference_output, candidate_output, ...)` produces a numerical equality receipt for features, base, and initializer. It compares beta only when present, preserving `EXPLICITLY_OMITTED_NOT_COMPARED` otherwise. This is a numerical check, not scientific or deployment acceptance.

## Cache and safety behavior

- The first R core solve factors the guarded K with `torch.linalg.lu_factor`; matching forward/tangent/adjoint core solves use `torch.linalg.lu_solve(..., adjoint=...)` without replacing the physics.
- Every cache hit checks state identity, model owner, material and h snapshots, core tensor/version/value snapshot, reference owner/geometry/buffer versions, and book/owner metadata.
- Stale material, core, reference, or model states are invalidated and rejected. A change of cost-book owner invalidates the old LU, rechecks the core, and charges a new factorization under the new owner.
- Reference tensors constructed inside `torch.inference_mode` lack version counters; reuse rejects such references rather than silently missing reference changes. Inference-mode online states with a normally constructed reference remain supported through value snapshots.
- Existing material-domain, core-SVD/condition, finite-solve, A26 full relative-forward residual, material-denominator, observation/OM rank, and finite-output checks remain active. LU solves additionally verify normalized core backward error against the frozen tolerance; those extra guards/actions are included in timing and counts.
- Failure has no jitter, pseudoinverse, rank adjustment, L fallback, or hidden correction.
- `LU_factorizations` and `feedback_K_LU_factorizations` count actual factor constructions only. `linear_solve_rhs` and reverse hooks retain solve/action accounting. Cache hits have explicit `a27_feedback_K_LU_reuse_calls/systems`; invalidations are counted.
- For a batch of B images, baseline N2 performs 2B core factorizations for its forward and explicit adjoint; reuse performs B. Both still solve 2B*s core RHS for that forward/adjoint pair. These are expected action counts to be verified, not measured runtime results.

Beta omission returns `beta=None`, `beta_status='OMITTED_NO_DIAGNOSTIC_L_FORWARD'`, and an explicit list of omitted beta/L residual diagnostics. It does not fabricate a beta value or claim beta equivalence. The matching R forward residual, R core margins, sigma checks/noise-free group, gradient, boxed base, and all twelve output channels remain.

## Timing interpretation and schema

Native and stage profiles are separate calls. Native has `native_full_inference_seconds` and no stage/event observer; stage runs have `instrumented_full_inference_seconds` and no native latency value. End synchronization is included in the outer image window. No intermediate synchronization is inserted. Required scalar numerical decisions and diagnostic/output CPU transfers remain present.

The in-memory profile book matches A26 CountOnlyOnlineBook's scope: counters per operation; parent budget/owner/file-lock checks before and after each outer image; no per-span disk trace, resource sampling, lock read, or device synchronization. Completed/failed action counts are transferred to the existing book and its binding is restored even when encoding/decoding fails.

Named spans cover input transfer, background-adjoint initializer, core assembly, separate LU where available, core solve, feedback adjoint, source aggregation, observation material ratio plus denominator guard, physical-base update, feature assembly, guards, beta, diagnostic logging, decoder, and image transfer. Untouched FrozenEncoder's missing finer spans are explicitly `NOT_SEPARATELY_INSTRUMENTED`; nonapplicable L/N3 stages are explicit. Baseline `torch.linalg.solve` combines factorization and solve, so its separate LU duration remains unavailable, rather than inferred by subtraction.

Stage records include host inclusive/exclusive seconds and CUDA current-stream event inclusive/exclusive seconds. They overlap hierarchically, so summing named stages is not native latency. CUDA stream intervals may include idle time caused by host guards; they are not pure kernel time. CPU GPU-event fields are null with `NOT_APPLICABLE_CPU` status.

A stage-run-only Torch dispatch observer counts CPU/CUDA scalar host results, tensor-equality host results, and CUDA-to-CPU copies. `host_synchronization_summary` reports these as wait opportunities, not measured wait seconds; native timing has no observer. The observer forwards the exact operators and introduces no synchronize calls. CUDA observer/event behavior remains NOT_RUN locally.

No output path is imposed. Parent-owned rows can flatten `timing` plus metadata such as arm/network/background/case/repeat/correction. Useful fields are `status`, `timing_mode`, `method`, native/instrumented full seconds, `action_counts`, `named_stages`, `stage_rows`, `host_synchronization_summary`, `outer_budget_check_host_seconds`, and `stage_meter_host_seconds`.

## Verification supplied and unresolved work

`tests/test_a27_fast_physics.py` supplies 11 charged-CPU tests reusing the unchanged `_tiny_cache` fixture: forward/adjoint numerical equality and actual counters; JVP/VJP plus higher derivative; single/batch state handling and no cross-state reuse; material/core/reference/model invalidation; book-owner revalidation; domain/singular/nonfinite failures; finite corrupt-adjoint residual rejection; all three engineering-arm feature/base equality with explicit omitted beta; inherited forward/denominator guards; separate native/stage metrics with no per-span synchronization; and failed-decode accounting/restoration.

Static AST parsing and whitespace checks passed. The parent run002 fixture import failure was corrected; rerun status is pending at this artifact revision. Runtime execution remains parent-owned. Actual frozen-cache equivalence receipts, actual CPU/GPU latency, and scientific interpretation remain for the parent. No measured speedup, hardware result, reconstruction quality, beta equality for omitted arms, or publication result is claimed here.

Files owned by this worker:

- `src/a27/fast_physics.py`
- `src/a27/latency.py`
- `tests/test_a27_fast_physics.py`
- `research/delegated/a27_latency/summary.md`
