# YBCO fixed-reference formation-energy trial

This entry point uses the author-provided CP2K training XYZ from
[YBCO_ACE_MACE](https://github.com/nicdieugenio/YBCO_ACE_MACE), pinned to commit
`cfbb8a59f6657bc3f31fbadb9e06b254e6a43fbf`. SHA256 and file size are checked;
Git LFS pointer files and changed upstream releases are rejected.

## What the target means

The public XYZ supplies total DFT energies, not an explicit table of defect
formation energies. This trial derives **unrelaxed configuration formation
energies relative to a fixed reference**, in eV per configuration:

```text
E_target = E_configuration - E_host - sum((N_configuration - N_host) * mu)
```

The fixed host is the source frame labelled
`Ba2YCu3O7_vol0_strain0_jiggle0` (936 atoms, 72 formula units).
The Y, Ba and Cu chemical potentials are the energies per atom of the source
frames `Y/Ba/Cu_vol0_strain0_jiggle0`. Oxygen uses the DFT value **−433.99 eV/O**
from Table 4 of the [source paper](https://arxiv.org/html/2511.22592v1), whose
reference is non-spin-polarized O2. This value is published to 0.01 eV precision;
the training XYZ does not contain its O2 calculation. The MACE training `E0s`
are isolated-atom offsets and are **not** substituted for these chemical potentials.

Limitations are saved in the preparation manifest. The host has a maximum force
of about 0.379 eV/Å and cannot be certified fully relaxed. Defective configurations
have strain/atomic perturbations and generally differ from the reference cell.
Their target therefore includes deformation energy. It is **not** a benchmark
of relaxed thermodynamic defect formation energies, nor defect-pair binding.
Charge corrections and phase-stability-constrained chemical potentials are absent.
In the first audited cohort, targets span roughly −5 to 1031 eV and the median
is about 627 eV: extensive perturbation energy is substantial. Do not compare
its MAE directly with the native semiconductor benchmark or present a low MAE
as evidence that defect interactions have been learned. For a conventional
relaxed-DFE study, additional matched reference/defect DFT relaxations are needed.

## Selection and graphs

The pinned file actually has **2705** configurations (the paper reports 2712).
There are 1310 labelled single vacancies, interstitials and antisites. Bulk
YBCO7, ordered YBCO6, amorphous material and elemental/binary subsystems are not
training samples in this experiment. Reference structures are reserved for
the energy definition and never used as train/validation/test observations.

For each defect family, a minimally perturbed exemplar is matched to the host
by species-wise periodic assignment in the fixed host cell. The matching must
agree with the defect's composition change. The atom order and displacements
of all family members are then checked. Assignment or displacement distances
above 1 Å are excluded and recorded; this is a graph-identity rule, not proof
of invalid DFT energies. The first preparation retains **707 frames from 43
families** and excludes 603 of the 1310 candidates. All compared models use
exactly the same cohort. No target-energy trimming is performed.

Vacancies receive an additional X node at the mapped reference position for
`full_x`, `was_x`, `hetero`, and `hetero_was`. Interstitials have WAS=0 and
substitutions have the original host species. `full` includes only real atoms.
Heterogeneous graphs preserve the actual defect pooling mask independently
of the selected local radius.

All volume/strain/jiggle variants of a defect family stay together in the
60/20/20 **group** split. Seed 123 gives **383 train / 128 validation / 196
test structures**. This evaluates unseen defect families, not a random-row
split or concentration transfer. Since the cohort is single-defect only,
this run cannot test the proposed high-concentration defect interactions.

## Commands

Run from the directory containing the `HERA` package, with the existing HERA
environment active. ASE and SciPy are already listed in the project environment.
The default data directory is `dataset/ybco` beside `HERA`, independently of CWD.

```bash
conda activate hera
python -m HERA.scripts.run_ybco_formation --download --prepare-only

# Verify data -> graphs -> one epoch -> held-out predictions/checkpoints.
python -m HERA.scripts.run_ybco_formation --model alignn --mode hetero_was was_x --smoke --epochs 1 --batch-size 1 --device cuda:0 --seed 123 --run-dir HERA/logs/ybco_formation_smoke_v1

# Exploratory trial on all 707 matched, unrelaxed configurations.
python -m HERA.scripts.run_ybco_formation --model alignn --mode hetero_was was_x --epochs 500 --batch-size 1 --device cuda:0 --seed 123 --r 0 --run-dir HERA/logs/ybco_formation_v1
```

Each supercell has about 936 atoms, so start with batch size 1. `--model cgcnn`
and `--model megnet` are also supported. `--smoke` picks one minimally perturbed
sample per family (43 structures); smoke scores are only a pipeline check.
Add `--resume` to skip compatible completed seeds; interrupted seeds restart.
The best validation checkpoint is selected, and the test set is evaluated
after training. A new experiment configuration requires a new run directory.

To prepare from an already-downloaded pinned file:

```bash
python -m HERA.scripts.run_ybco_formation --source /absolute/path/ybco_mace.xyz --data-root /absolute/path/ybco --prepare-only
```

Pass the same `--data-root` to training. The preparation outputs an ASE database
and a JSON manifest containing reference energies, chemical potentials, the
selection, mapping exclusions, source IDs, and target values. Training outputs
`data_contract.json`, `seed123_splits.json`, per-mode history, predictions and
best checkpoints, plus `summary.json`. All MAEs are in eV/configuration.
