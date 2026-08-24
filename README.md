# QMax Training and Evaluation Code

Research code for QMax, a reliability-aware multi-contrast MRI reconstruction framework that uses an auxiliary proton-density (PD) acquisition to support accelerated fat-suppressed proton-density (PD-FS) reconstruction.

This repository is a code and protocol snapshot for the experiments reported in the associated MSc project. It contains the model implementations, training and evaluation programs, Slurm job specifications, public split metadata, environment records, and cryptographic provenance files. Raw MRI data, patient metadata, trained checkpoints, logs, and result directories are intentionally excluded.

## Experimental scope

The repository contains two related but distinct experimental tracks. Their results must not be pooled.

### 1. Primary multicoil QMax study

- Target: multicoil PD-FS reconstruction at nominal 8x acceleration.
- Auxiliary input: PD reconstructed from a nominal 2x acquisition.
- Final model: QMax-Full, a 12-cascade VarNet-style reconstruction model with reliability-controlled auxiliary fusion.
- Formal model seeds: 42, 123, and 2026.
- Formal checkpoint policy: `epoch60/model_last.pt`.
- PD-free comparator: an independently trained PD-FS-only VarNet.
- Main controls: same-checkpoint auxiliary availability-off and reliability/component counterfactuals.

The frozen architecture and training contract are recorded in `QMAX_STAGE_A_PROTOCOL_R8.json`. The replication and held-out contracts are recorded in `QMAX_FULL_MULTISEED_PROTOCOL_R8.json` and `QMAX_FULL_HELDOUT_TEST_PROTOCOL.json`.

### 2. Single-coil FSMNet-public benchmark

- Dataset: paired single-coil knee PD and PD-FS volumes from fastMRI.
- Public grouping unit: FSMNet public pair ID; patient identifiers are not publicly available.
- Training set: 227 pairs and 8,332 slices.
- Train-monitor subset: 16 pairs and 571 slices.
- Locked held-out set: 45 pairs and 1,665 slices.
- Verified train/held-out volume overlap: zero.
- Nominal acceleration: 8x.
- Models: zero-filled reconstruction, single-coil QMax-Full, and QMax-Frequency.
- Training seed: 1337.
- Formal training length: 100,000 optimizer updates in five 20,000-update segments.

QMax-Frequency extends single-coil QMax-Full with a 64-channel multiscale frequency branch. It processes normalized QMax and PD images using Fourier amplitude-phase residual blocks, applies learned cross-modal fusion at five scales, and predicts a residual refinement. The training objective sums the image, Fourier-amplitude, and Fourier-phase losses for the main and frequency-refined outputs.

The single-coil comparison with published FSMNet-context results is contextual rather than a strict head-to-head benchmark because the external models were not retrained and re-evaluated within this repository.

## Repository layout

```text
.
|-- src/                    Model definitions, physics operators and datasets
|-- scripts/                Multicoil training, evaluation and audit programs
|-- external_benchmarks/    Single-coil datasets, training and evaluation code
|-- slurm/                  Historical Isambard Slurm job specifications
|-- protocol/               Public splits, hashes and frozen protocol records
|-- environment.yml         Minimal Python environment declaration
|-- requirements-lock.txt   Audit snapshot of the original Python environment
`-- .gitignore              Data, checkpoint, output and credential exclusions
```

Important model entry points include:

```text
src/m2_prnf_qmax_varnet.py
src/m2_prnf_qmax_singlecoil.py
src/m2_prnf_qmax_singlecoil_freqaux.py
scripts/train_qmax_stage_a.py
scripts/train_qmax_stage_a_continue_31to60.py
scripts/train_varnet_single.py
external_benchmarks/train_singlecoil_qmax.py
external_benchmarks/train_singlecoil_qmax_freqaux.py
external_benchmarks/evaluate_singlecoil_three_arm_heldout.py
```

Files containing `backup`, `before_`, `archive`, or older model-family names are retained for provenance. They are not authoritative entry points for the final QMax experiments.

## Software environment

The original runs used Linux, Slurm, Python 3.11, CUDA-capable NVIDIA GPUs, PyTorch, and `fastmri` 0.3.0.

Create a clean environment:

```bash
conda create -n qmax-reproduction python=3.11
conda activate qmax-reproduction
```

Install a PyTorch build compatible with the CUDA driver on the target system, then install the principal Python dependencies:

```bash
python -m pip install \
  fastmri==0.3.0 \
  h5py \
  matplotlib \
  numpy \
  pandas \
  pytorch-lightning \
  scipy \
  scikit-image \
  tqdm
```

`requirements-lock.txt` records the original Isambard environment, including a development PyTorch build and platform-specific CUDA packages. It is provided for audit purposes and should not be treated as a universally portable lock file. `environment.yml` records the original Python version but also contains the original environment prefix.

Verify imports and Python syntax after installation:

```bash
python -c "import torch, fastmri, h5py, numpy, pandas, skimage; print(torch.__version__)"
python -m compileall -q src scripts external_benchmarks
```

The original formal jobs used GPUs with sufficient memory for a 96 GB Slurm allocation. Batch size and memory requirements should be re-profiled on other hardware before formal training.

## Data access and licensing

fastMRI data are not redistributed in this repository. Obtain access through the official fastMRI process and comply with its data-use agreement. Do not commit raw `.h5` data or patient metadata.

The single-coil benchmark additionally depends on the public FSMNet repository at the pinned commit recorded in `protocol/fsmnet_commit.txt`:

```bash
git clone https://github.com/qic999/FSMNet.git /path/to/FSMNet
git -C /path/to/FSMNet checkout b36e1609046ac081134c7cfb5833d3c7424d691c
```

The public FSMNet split files are included under:

```text
protocol/fsmnet_data_split/singlecoil_train_split_less.csv
protocol/fsmnet_data_split/singlecoil_val_split_less.csv
```

The paired data and manifests must be prepared outside the repository. The single-coil code expects the following benchmark structure:

```text
BENCH_ROOT/
|-- manifests/
|   |-- train.csv
|   |-- train_monitor.csv
|   `-- test_locked.csv
|-- repo/FSMNet/
|-- data/
|-- outputs/
`-- logs/
```

The public split inventory and pairing audit are available in `protocol/manifest_summary.json`, `protocol/fsmnet_pair_audit.csv`, and `protocol/fsmnet_pair_audit.json`. Archive hashes identify the data snapshot used in the original runs but do not grant redistribution rights.

The multicoil study expects a local metadata CSV and pre-generated clean/robustness/condition manifests. These contain dataset-specific paths and are not distributed. Consult the loaders under `src/dataset_paired_multicoil*.py` and the argument definitions in the corresponding training scripts when constructing an equivalent local manifest.

## Required path configuration

The Slurm files preserve the exact Isambard job specifications used in the study. They contain absolute paths such as:

```text
/projects/u6dm/fastmri_project/fastmri_pipeline
/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project
```

Before running on another system, copy the repository to the desired project directory and update at least:

- `PROJECT_ROOT` or `PIPELINE`;
- `SCRATCH_ROOT` or `BENCH_ROOT`;
- `METADATA` and all manifest paths;
- `#SBATCH --partition`, GPU, memory and wall-time requests;
- `#SBATCH --output` and `#SBATCH --error` paths;
- the local FSMNet checkout path.

Do not submit a historical Slurm file unchanged on another cluster.

## Multicoil QMax workflow

The primary multicoil workflow is protocol-gated. The principal sequence is:

1. Prepare the metadata CSV and the frozen clean, robustness and condition manifests.
2. Update all cluster-specific paths in the relevant Slurm files.
3. Run the QMax preflight and smoke checks.
4. Train the Stage-A candidates through epoch 30.
5. Continue the selected QMax-Full model through epoch 60.
6. Repeat the frozen QMax-Full protocol for seeds 123 and 2026.
7. Freeze model development before accessing the held-out cohort.
8. Evaluate the three prespecified epoch-60 checkpoints on the held-out cohort without test-driven checkpoint or seed selection.

Principal entry points:

```text
slurm/submit_qmax_preflight.slurm
slurm/submit_qmax_smoke_p1.slurm
slurm/submit_qmax_smoke_p2.slurm
slurm/submit_qmax_p1_epoch1to30.slurm
slurm/submit_qmax_p2_epoch1to30.slurm
slurm/submit_qmax_stage_a_winner_epoch31to60.slurm
slurm/submit_qmax_full_seed123_epoch1to30.slurm
slurm/submit_qmax_full_seed123_epoch31to60.slurm
slurm/submit_qmax_full_seed2026_epoch1to30.slurm
slurm/submit_qmax_full_seed2026_epoch31to60.slurm
slurm/submit_qmax_full_heldout_panel_freeze.slurm
```

Example preflight submission after path configuration:

```bash
sbatch slurm/submit_qmax_preflight.slurm
```

Several multicoil jobs intentionally require earlier manifests, historical comparator checkpoints, random-initialization templates, and completion locks. A missing gate should be recreated from a new prespecified protocol rather than bypassed silently.

### PD-free comparator

The independent PD-FS-only VarNet is the PD-free learned comparator for the multicoil study. It receives no auxiliary PD image.

```text
scripts/train_varnet_single.py
scripts/evaluate_varnet_single.py
slurm/train_varnet_single_pdfs_R8_c12_ch18_ep30.slurm
slurm/resume_varnet_single_pdfs_R8_ep30_to50.slurm
slurm/evaluate_single_varnet_R4_R6_R8_val.slurm
```

For a formal comparison, evaluate its frozen checkpoint and QMax on the same held-out patients, masks, preprocessing pipeline, metric implementation, and aggregation unit.

## Single-coil workflow

The single-coil pipeline uses raw-grid k-space physics and shape-aware batching. Run each preflight before its corresponding pilot or formal chain.

### 1. QMax-Full preflight

```bash
PREFLIGHT_BATCH_SIZE=4 \
  sbatch slurm/submit_singlecoil_qmax_preflight.slurm
```

The preflight checks raw-grid shapes, forward and adjoint operations, gradient flow, deterministic mask generation, and GPU memory.

### 2. Pilot and resume verification

The following jobs verify one-epoch training and deterministic continuation before formal training:

```text
slurm/submit_singlecoil_qmax_pilot.slurm
slurm/submit_singlecoil_qmax_resume_preflight.slurm
slurm/submit_singlecoil_qmax_resume_branch_check.slurm
```

Inspect each file for required environment variables and gate artifacts before submission.

### 3. Segmented QMax-Full training

After all gates pass, the formal job accepts only the prespecified target updates and resume modes:

```bash
J20=$(sbatch --parsable \
  --export=ALL,TARGET_UPDATE=20000,RESUME_MODE=fresh \
  slurm/submit_singlecoil_qmax_formal_segment.slurm)

J40=$(sbatch --parsable --dependency=afterok:${J20} \
  --export=ALL,TARGET_UPDATE=40000,RESUME_MODE=resume \
  slurm/submit_singlecoil_qmax_formal_segment.slurm)

J60=$(sbatch --parsable --dependency=afterok:${J40} \
  --export=ALL,TARGET_UPDATE=60000,RESUME_MODE=resume \
  slurm/submit_singlecoil_qmax_formal_segment.slurm)

J80=$(sbatch --parsable --dependency=afterok:${J60} \
  --export=ALL,TARGET_UPDATE=80000,RESUME_MODE=resume \
  slurm/submit_singlecoil_qmax_formal_segment.slurm)

J100=$(sbatch --parsable --dependency=afterok:${J80} \
  --export=ALL,TARGET_UPDATE=100000,RESUME_MODE=resume \
  slurm/submit_singlecoil_qmax_formal_segment.slurm)
```

### 4. QMax-Frequency training

First run the frequency-branch preflight, pilot, and resume verification:

```text
slurm/submit_singlecoil_qmax_freqaux_preflight.slurm
slurm/submit_singlecoil_qmax_freqaux_pilot.slurm
slurm/submit_singlecoil_qmax_freqaux_resume_preflight.slurm
```

Then repeat the five-segment submission pattern using:

```text
slurm/submit_singlecoil_qmax_freqaux_formal_segment.slurm
```

with `TARGET_UPDATE` set to 20,000, 40,000, 60,000, 80,000, and 100,000. The first segment must use `RESUME_MODE=fresh`; all later segments must use `RESUME_MODE=resume`.

### 5. Monitor reproduction and held-out evaluation

The held-out evaluator must not be used for model development. Reproduce the frozen train-monitor results first, verify both final checkpoints, and then run the single formal three-arm evaluation:

```text
slurm/submit_singlecoil_zerofill_monitor_audit.slurm
slurm/submit_singlecoil_qmax_formal_monitor_evaluation.slurm
slurm/submit_singlecoil_qmax_freqaux_final_verify.slurm
slurm/submit_singlecoil_three_arm_heldout.slurm
```

The formal evaluator uses the metric implementation from the pinned FSMNet checkout, mask seed 1337, bootstrap seed 1337, and 10,000 bootstrap resamples. It evaluates all 45 locked held-out volumes exactly once after the prespecified gates pass.

## Provenance and integrity checks

The `protocol/` directory records:

- the pinned FSMNet commit;
- public split inventories and pair audits;
- dataset archive hashes;
- formal job-chain identifiers;
- SHA-256 manifests for training, pilot, frequency-branch and held-out code;
- prespecified protocol amendments and access gates.

For the relative-path QMax-Full single-coil training manifest, run from the repository root:

```bash
sha256sum -c protocol/singlecoil_formal_training_code_sha256.txt
```

Some other historical hash files include absolute paths to the original benchmark directory. Their digest values remain useful for provenance, but the paths must be mapped to the corresponding local artifacts before direct verification.

## Outputs and checkpoint policy

Generated outputs are intentionally excluded by `.gitignore`. Typical runs create:

```text
outputs/
|-- model_last.pt
|-- checkpoint_updateXXXXXX.pt
|-- train_metrics.jsonl
|-- training_summary.json
`-- evaluation summaries and per-slice/per-volume CSV files
```

Do not use `model_best.pt` or a test-best seed unless that policy was prespecified for the relevant track. The primary multicoil QMax study uses the frozen epoch-60 checkpoints. The single-coil study uses `model_last.pt` at update 100,000 for both trained models.

## Reproducibility boundaries

This repository enables code inspection and reconstruction of the experimental workflow, but it is not fully self-contained:

- raw fastMRI data are governed by a separate data-use agreement;
- patient-level multicoil metadata and local manifests are not distributed;
- trained checkpoints and outputs are not stored in Git;
- historical Slurm files contain Isambard-specific paths and resource requests;
- several formal jobs depend on frozen gate artifacts generated during earlier stages;
- the original Python lock includes platform-specific and development packages.

These limitations should be reported when sharing or archiving the repository. Reproduction on another cluster requires local path adaptation, data-manifest reconstruction, dependency validation, and fresh preflight checks.

## Security and data hygiene

Before every push, verify that no data, checkpoints, logs, signed download URLs, access tokens, private keys, or patient metadata have been staged:

```bash
git status --short
git diff --cached --name-only
find . -type f -size +90M -print
```

Never commit `.env`, `.pem`, `.key`, `.pt`, `.pth`, `.ckpt`, `.h5`, archive, output, or log files.

## License and citation

No software license has yet been assigned to this repository. Until a license is added, reuse rights are not granted automatically. Third-party components, fastMRI data, and FSMNet remain subject to their original licenses and terms.

When the associated dissertation or paper receives a stable citation, add it here together with the repository release tag and commit hash used for the reported experiments.
