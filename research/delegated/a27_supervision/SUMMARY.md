# A27 supervision audit summary

- SUCCESS: frozen source HEAD is `2603f3bd16c48e48587fedc4363975779def17f5`; own-directory audit complete, no source/runtime edits.
- Existing A25 TRAIN has full complex128 induced-current `[512,576,8]`, 64 parents / 256 unique clean states, 8 source columns. Noise variants share identical teachers.
- Total field and relative contrast source are not stored; both reconstruct from existing arrays without a new solve: `e=incident+G c`, `w=(chi-chi_b)e=Lb(c-cb)`. Maximum checked w identity error `7.132e-14`; data reconstruction error `1.226e-15`.
- TRAIN RMS h/c/e/w is `.1694758747/.002803701251/.003684127913/.0005801294179`. Common-material ratio agrees with teacher chi to `9.372e-16`; these prove existing TRAIN content consistency only.
- A26 caches/repackaged inputs/checkpoints contain no current/field/w teachers; actual A26 objective was material MSE+SSIM with physical_backward_calls=0. A25 intended auxiliary supervision never reached scientific training.
- Loss draft uses removable training-only source-shared current head, fixed-G total-field deduction, common-material ratio/residual and optional relative-w loss; every comparison arm needs identical extra supervision rights and training capacity.
- Original nearby SOM-Net text supports current + receiver scattered-field + SSIM + material losses; its field loss is not internal total-field supervision. Proposed shell/feedback benefits remain NOT_RUN.
- Zero locked/validation array reads, new labels, solves, training updates, GPU/remote jobs or hash checks. Two successful audit passes cost CPU1.873889s / wall2.374273s; no training numerical experiment was executed.

Artifacts: `SUPERVISION_AUDIT.md`, `TRAINING_ONLY_LOSS_SPEC.md`, `EVIDENCE_TABLE.md`, `TRAIN_ARRAY_AUDIT.json`, `TRAIN_STATE_SUMMARY.json`, `audit_train_arrays.py`, `COMMAND_RECEIPT.json` in this directory.

Uncertainties: final loss weights/head wiring/cost admission require parent protocol/math review; no new loss has been trained or tested, predicted-field ratio safety is unknown, and no new public A27 link exists.
