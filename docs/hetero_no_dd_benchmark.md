# AA/DD ablations on native, semi and imp2d

`--alignn-hetero-dd drop` removes the main `('defect', 'dd', 'defect')`
relation from graph construction, message passing, radial encoding, residual
adapters and defect-node fusion. Its bond nodes and incident angles are also
removed from the ALIGNN line graph. All remaining edges and their geometry
retain the original neighbor selection and ordering.

The new option is independent of `--alignn-hetero-aa`. Both default to `keep`,
including when the fields are absent from older checkpoint configs. Original
models and the previous no-aa variant remain available. Dropped relations add
`relations_no_aa` / `relations_no_dd` path components and saved run labels.

| Variant | AA | DD | Parameters |
|---|---|---|---:|
| baseline | keep | keep | 679,193 |
| no_aa | drop | keep | 606,819 |
| no_dd | keep | drop | 606,819 |
| no_aa_dd | drop | drop | 534,445 |

Counts use shared_residual rank 8, hidden size 64, 3 ALIGNN + 3 GCN blocks,
LayerNorm and defect_energy_mean. Each dropped relation removes 7,040 radial
encoder parameters, 16,182 adapter parameters and 49,152 fusion parameters.
Train these architectures separately; restore each with its own saved config.

Removing DD retains A -> D and D -> A, including indirect D -> A -> D paths.
Removing AA and DD together leaves this bipartite atom/defect graph. All nodes,
root updates and actual-defect pooling masks remain. With `--r 0`, D is the
actual-defect set. At larger radii it also includes the surrounding defect
region, which changes the meaning of this ablation.

## Ready-to-run benchmark

Sync the modified HERA code to the machine containing all three datasets.
Activate its HERA Python environment and run from HERA's parent directory:

```bash
python -m HERA.scripts.run_hetero_relation_benchmark --dataset native semi imp2d --variant baseline no_aa no_dd no_aa_dd --seed 123 11 1245 --epochs 500 --device cuda:0
```

This runs 4 variants x 3 datasets x 3 seeds = **36 training/test runs**
sequentially. Each variant has its own root directory under
`HERA/logs/hetero_relation_native_semi_imp2d/` and always uses `--resume`.
Checkpoint selection and early stopping use validation MAE; testing happens
automatically at the end of each run.

For only the original and DD-deleted model:

```bash
python -m HERA.scripts.run_hetero_relation_benchmark --dataset native semi imp2d --variant baseline no_dd --seed 123 11 1245 --epochs 500 --device cuda:0
```

For the AA-deleted model versus deleting BOTH AA and DD:

```bash
python -m HERA.scripts.run_hetero_relation_benchmark --dataset native semi imp2d --variant no_aa no_aa_dd --seed 123 11 1245 --epochs 500 --device cuda:0
```

For this Windows machine, use the same options with:

```powershell
Set-Location C:/Users/User/Desktop
$py = 'C:/Users/User/.conda/envs/hera/python.exe'
& $py -m HERA.scripts.run_hetero_relation_benchmark --dataset native semi imp2d --variant baseline no_aa no_dd no_aa_dd --seed 123 11 1245 --epochs 500 --device cuda:0
```

Add `--dry-run` to print exact underlying HERA.main commands and check data
availability without starting training. Use `--dataset native` to run only
the locally available native dataset. All dataset lists are checked before
any training starts, so a missing second dataset cannot fail after hours of
training on the first one.

## Protocol

- Fixed backbone: shared_residual rank 8; LayerNorm; defect_energy_mean;
  physical graph; r = 0; no auxiliary sparse defect residual; FP32.
- Batch size 8 for every dataset/variant, test batch size 1. Override training
  size with `--batch-size`, keeping it equal for the paired comparisons.
- These three datasets use the existing per-structure random 60/20/20
  train/validation/test split. A given seed selects identical samples for all
  variants. This evaluates random-split performance within each dataset.
- Up to 500 epochs, existing early stopping with patience 50 and 0.5% minimum
  relative validation improvement. The three seeds provide repeated-split
  estimates; they are not three folds of cross validation.
- Use `--seed all` for the repository's ten-seed suite: 120 runs for all four
  variants and three datasets. For five-fold CV, use `--cv5 --seed 123` and a
  NEW root such as `--run-dir HERA/logs/hetero_relation_cv5` (60 runs).
- Resume only when using the same protocol; choose a new root if changing
  batch size, precision, split mode or other experiment settings.

Result summaries are at:

```text
HERA/logs/hetero_relation_native_semi_imp2d/<variant>/alignn/<dataset>/summary.txt
```

Detailed per-seed histories, checkpoints and prediction CSVs are in each
dataset's hetero/r0 subdirectories. The summaries report mean/std MAE over
seeds. Units follow each dataset's existing labels; the runner does not change
targets or target scaling. No separate 2dmd_low evaluator is involved.

For an individual HERA.main run, the core new options are:

```bash
python -m HERA.main --model alignn --dataset native semi imp2d --mode hetero --r 0 --alignn-hetero-relations shared_residual --alignn-hetero-pooling defect_energy_mean --alignn-hetero-aa keep --alignn-hetero-dd drop --alignn-train-batch-size 8 --seed 123 11 1245 --epochs 500 --device cuda:0 --run-dir HERA/logs/hetero_no_dd_direct --resume
```

The DD switch removes the **main dd relation**. The separate optional
`sparse_dd` residual is controlled by `--alignn-hetero-defect-residual`; this
benchmark explicitly sets it to `none`. Complete-connectivity edges that
would belong to the dropped dd store are removed too.

## Interpretation and local validation

Some single-defect graphs already have no DD bonds within the cutoff; periodic
images can create exceptions. Eight evenly sampled real native structures
checked locally each had one defect and zero DD edges at r=0. This is a sample
audit, not a full-dataset statement. On graphs with no DD edges, this ablation
primarily removes unused relation parameters and narrows fusion input layers;
it also changes random initialization. It cannot establish that real
defect-defect messages were unnecessary when none were present.

84 related tests passed across the regression and benchmark-runner checks.
All four full-size models passed native GPU forward/backward optimizer steps,
batch consistency and strict checkpoint restoration. Synthetic tests include
multiple defects, periodic DD edges, complete connectivity, empty-edge graphs,
all sharing modes and both relation aggregation modes. Details of the real
native check are in `results/hetero_no_dd_validation_20260917.json`.

Full benchmark accuracy has not been measured. On this local machine:

- Native's data list and sampled structures are available.
- `dataset/Dataset_1/Dataset_1/Neutral/Neutral/id_prop_A_rich.csv` is empty.
- `dataset/imp2d/imp2d/id_prop.csv` is missing.

The three-dataset command will intentionally stop at preflight until the
semi and imp2d data are present. Run it on the complete-data machine after
syncing the code, or restrict the local command to native.
