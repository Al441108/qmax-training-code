# StageA-Full final trajectory and missing-tensor audits

This additive package contains two read-only locked-validation tasks.

## LR trajectory

Evaluates exact `model_last.pt` checkpoints at epochs 30, 40, 50, and 60 on
the same locked clean and robustness manifests. It reports patient-level
paired bootstrap differences relative to epoch30, q separation, auxiliary RMS
diagnostics, and clean correction-on/off effects. Epoch60 remains the formal
selection; earlier checkpoints are supportive trajectory points only.

## Missing-tensor safety

Fixes availability to `m=0` and replaces the PD tensor with zero, real PD,
deterministically transformed PD, and a deterministic nonzero pattern. It
checks output invariance, exact-zero direct/correction/final auxiliary RMS,
finite outputs, and that DC evidence cannot bypass availability.

Neither task accesses the held-out test set or modifies a checkpoint.
