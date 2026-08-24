# StageA-Full epoch60 component counterfactuals

This additive package performs a read-only, locked-validation evaluation of
`epoch60/model_last.pt`. It does not change checkpoint-bound scientific code.

All modes retain learned q:

- `full`
- `detail_neutral` (`G=1`)
- `alignment_off` (`Delta=0`)
- `correction_off` (`C=0`)
- `dc_zero` (DC evidence zeroed after RMS normalization)

It evaluates the locked clean and robustness cohorts, builds the original
`shift8 + wrong_slice + wrong_patient` composite, and reports patient-level
paired bootstrap intervals for full versus each counterfactual. It also checks
that all modes have exact-zero direct/correction paths and identical patient
metrics when PD availability is zero.

Install from the project root:

```bash
cp stagea_full_epoch60_components/scripts/evaluate_stagea_full_epoch60_components.py scripts/
cp stagea_full_epoch60_components/slurm/submit_stagea_full_epoch60_components.slurm slurm/
python -m py_compile scripts/evaluate_stagea_full_epoch60_components.py
bash -n slurm/submit_stagea_full_epoch60_components.slurm
```

Submit:

```bash
sbatch slurm/submit_stagea_full_epoch60_components.slurm
```

Formal outputs are written to:

```text
/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline/outputs/qmax_stage_a/component_counterfactuals/qmax_full_seed42_epoch60/
```
