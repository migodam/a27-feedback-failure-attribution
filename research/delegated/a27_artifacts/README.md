# A27 artifact worker handoff

This worker implements saved-output aggregation and plotting only. No physics, training, label generation, remote work, locked split or scientific case verdict is run here. The parent owns numerical certification, active input selection and scientific interpretation.

## Callable and input selection

`a27.artifacts.build_artifacts(root, data_root, a26_root, config, book, device)` has the stage-callback signature; `stage_report` is an alias. The entire function, including imports, reads, bootstrap and rendering, is inside `book.span('a27_build_saved_evidence_artifacts')` and retains periodic budget checks.

If `results/a27/POST_SOURCE.json` exists, its `folder` must be exactly `post` or `post_fgp`. Otherwise the original `post` is used. This is explicit input routing, never quality-based selection. The manifest is copied into the summary. Original inputs are never edited. Original bundles write to `results/a27/artifacts`; selected FGP bundles write to `results/a27/artifacts_post_fgp`, preserving the original bundle.

Required saved inputs are the selected folder's `PER_CASE.csv` and `CORRECTED_PREDICTIONS.npz`, plus `diagnostics/PER_STATE.csv` and `observability/PER_STATE.csv`. Optional native timing comes from `latency/*.csv` or native timing JSON records, with `FAILURES.json` incorporated where failed native rows were omitted. Instrumented stage intervals are not summed into latency.

## Full-cohort tables and statistics

All 16 frozen validation parents, eight conditions each, both N3 and N1, and all four NONE/L/F/TASK_R32 arms are represented. Missing, duplicated, failed, metadata-mismatched or undefined conditions make the parent incomplete; its metric means remain null. No success-only mean replaces it.

`PER_PARENT.csv` averages eight conditions first. `PER_FAMILY.csv` weights parents equally. `PAIRED_STATISTICS.json` uses 2000 paired parent resamples for NONE-to-L/F/R32, L-to-F and F-to-R32. Primary complex N3 statistics use exactly the 12 shell/asymmetric/mixed parents. Error gain is `(control-candidate)/control`; zero control errors remain undefined. SSIM, IoU, edge and passivity comparisons use absolute candidate-minus-control differences. Incomplete paired statistics carry `HOLD`, null point estimates and null intervals.

Metrics include delta/full/real/imag NRMSE, delta SSIM, edge F1, support IoU, full data residuals, full backward residual, passivity and feasibility. Separate counts retain accepted updates, unknown acceptance status, failed/missing cases, and cases whose data residual decreases while delta-material error worsens versus the same-network NONE.

## Subordinate weak-only description

`WEAK_ONLY_DESCRIPTIVE.json` and the corresponding three tables use all four weak-background conditions per parent, with the same strict completeness rules and parent bootstrap. They are explicitly post-hoc and descriptive, and never replace primary full-cohort tables, statistics or HOLD status. No condition or parent is selected by reconstruction quality.

## Figures and provenance

PNG/SVG and raw plot CSVs are generated for N3/N1 fixed delta-real and delta-imag images, primary paired-family correction gain, feedback/noise ratio across fixed states, and offline observability error fractions. An optional latency/quality figure is created only for complete native samples paired to complete scalar quality evidence.

Images use the first parent in every family, amplitude 1.2, 20 dB, and both weak/ring backgrounds: eight fixed cases. `a26.data.load_split(data_root, 'validation')` is called only for those image panels; truth is indexed only at the fixed eight rows. No metrics are rescored from truth. Failed/undefined post records retain their status even when the saved prediction image is finite. Clean-state feedback/noise ratios remain explicitly undefined. Soft confidence energies retain their cross term and are not labeled posterior uncertainty.

Native timing requires the fixed 16 representative cases times three repeats. Missing, duplicated or failed samples produce HOLD and no median. Optimized N2 quality uses frozen A26 scalar metrics only when lossless equivalence is explicitly recorded. Corrected POST timing must have a `post_source_folder` matching the active post input; unmarked timing defaults to original `post`. Mismatched solver/input provenance suppresses the quality pair. NONE and original encoder quality are unaffected by that correction provenance check.

## Verification status

The final saved-data extension adds `CORRECTION_AUDIT.json`, `ARTIFACT_MANIFEST.json`, and eight tables: `CORRECTION_UPDATE_PER_CASE`, `CORRECTION_COMPARISON_PER_CASE`, `CORRECTION_TRIALS`, `CORRECTION_ACTION_COUNTS`, plus update/comparison `PER_PARENT` and `PER_FAMILY` CSVs. Actual returned L/F and R32/F updates use aligned saved case IDs, same-network NONE starts and real mass volume 1/576. Rejected zero updates remain present; zero cosine/reference denominators are explicit. Every reported family mean averages all eight conditions within each parent before equal parent weighting. Missing or undefined conditions yield HOLD rather than a mean of the available cases.

Trial counts, trust radii, acceptance/rejection reasons, recorded prox KKT/gap/iterations and inclusive action counter deltas are retained. The audit does not re-certify the solver. Pre-prox vectors, trial candidate vectors, background data normalization denominators and per-trace algorithm names were not saved and remain unavailable; scalar `step*gradient_mass_norm` is labeled as a derived pre-prox norm. The selected POST_SOURCE manifest is retained without inferring the FGP algorithm from the original PDHG config.

Diagnostic state rows now preserve all six OFFLINE gradient/projected-step alignments and identify the projected-step/box diagnostics as fixed background-step proxies. Only matching CNN_N1/N3 states in the fixed observability cohort receive saved full-gradient norm, strong-energy fraction and OFFLINE gradient/error alignment; other states are explicitly outside that observation scope. Observability aggregate tables also contain those three gradient metrics.

Current saved inputs have 1024 trace rows, 128 unique saved case IDs with all eight (128,576) complex prediction arrays, 1280 diagnostic rows and 32 observability rows. A read-only schema inspection found 1076 trial records: 761 accepted and 315 full-objective/data rejections, with seven L arms returning zero updates. These are receipt/schema facts, not scientific verdicts. The final source passes static syntax compilation. No production build, charged job, physics evaluation or broad test was launched by this worker while the parent controlled remote execution. The parent will execute the report callback on the real saved outputs.

The initial module passed 13 synthetic assertions and rendered eight figures in about 5.1 seconds wall / 4.8 seconds process CPU. Representative PNGs were inspected: labels, legends and the fixed image grid were readable. All preview figures are watermarked SYNTHETIC QA ONLY, and their fake inputs/results remain under this worker's `synthetic_preview/` folder.

After the parent's runtime freeze for controller job 62677, input-manifest routing, subordinate weak-only tables and strict native-cohort/provenance checks were added as source-only changes. No new runtime verification was launched during that freeze. `verify_artifacts.py` contains those additional assertions; the final correction audit extension has static checks only, pending the parent's report run.

Production artifacts have not been generated by this worker. Synthetic QA does not establish scientific success, numerical rescue certification, live GPU behavior or human acceptance.
