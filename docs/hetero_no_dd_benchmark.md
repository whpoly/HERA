# Hetero / hetero_was AA/DD benchmarks on native, semi and imp2d

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

### Add hetero_was to the existing baseline directory

Run this from HERA's parent on the machine holding the existing baseline:

```bash
python -m HERA.scripts.run_hetero_relation_benchmark --dataset native semi imp2d --mode hetero hetero_was --variant baseline --seed 123 11 1245 --epochs 500 --device cuda:0 --run-dir HERA/logs/hetero_relation_native_semi_imp2d
```

Both `hetero` and `hetero_was` are selected (18 dataset/mode/seed tasks).
Completed matching `baseline/alignn/<dataset>/hetero/` jobs are reused by resume.
New WAS results go to `baseline/alignn/<dataset>/hetero_was/`, and both modes
appear in the existing dataset/model summaries. Native/semi/imp2d now add a
`<dataset>_reference_v1` subdirectory for true WAS: **old WAS results are
preserved but are not reused as new WAS results**. Completed jobs of the same
version are reused; existing incomplete jobs stop under `--protect-existing`. This uses the
baseline relation configuration, keeping both AA and DD. Pass the common root
shown above, not a path ending in `/baseline`, because the runner appends it.

For a fair native WAS ablation, add `--native-preprocessing reference_v1`.
This also corrects ordinary hetero's defect indexing and vacancy construction,
so both native modes need new versioned runs initially. Full instructions:
[native_true_was.md](native_true_was.md).
For semi/imp2d use `--semi-preprocessing reference_v1` /
`--imp2d-preprocessing reference_v1`; see the
[impurity data audit](imp2d_semi_was_audit_20260920.md) for validation scope and missing semi data.

### Reuse existing runs and merge results (2026-09-19)

Mode roots retain the original layout; reference_v1 runs add a dataset-specific version
subdirectory below the graph/configuration components:

```text
HERA/logs/hetero_relation_native_semi_imp2d/   <-- --run-dir
  baseline/alignn/<dataset>/hetero/...        AA, AD, DA, DD
  no_dd/alignn/<dataset>/hetero/...           AA, AD, DA
  no_dd/alignn/<dataset>/hetero_was/...       AA, AD, DA with WAS features
```

`baseline` and `no_dd` are siblings. Pass their common parent to the benchmark
runner's `--run-dir`; passing `.../baseline` or `.../no_dd` would add another
variant subdirectory and miss the old histories.

**If the old results are under `baseline/`, selecting `--variant no_dd` does
not search them.** A folder name alone does not establish which relations
were trained. The standalone merge command below prints `saved AA=..., DD=...`
from each checkpoint's configuration. `DD=keep` is a baseline model and cannot
be reused as no_dd; `DD=drop` confirms the DD-deleted model, even if its source
directory is named baseline. The command preserves both source directories.

Recommended for combining old baseline-directory results and newer no_dd runs:

```bash
python HERA/scripts/merge_hetero_relation_results.py --run-dir HERA/logs/hetero_relation_native_semi_imp2d --variant baseline no_dd
```

This separate script has no training entry point. It prints its absolute
source path and `MERGE ONLY`, reads trusted HERA checkpoints on CPU, audits
AA/DD settings and merges saved metrics. It can recover a final `test_mae`
from a checkpoint when no completed history/leaf summary survives. Conflicting
saved metrics are reported as errors. No dataset, prediction, optimizer step,
checkpoint move or checkpoint modification is performed. Run by file path as
shown to select the current checkout rather than another installed HERA package.

To merge previously completed baseline/no_dd results, including all saved
hetero and hetero_was modes and seeds, **without any training or evaluation**:

```bash
python -m HERA.scripts.run_hetero_relation_benchmark --variant baseline no_dd --run-dir HERA/logs/hetero_relation_native_semi_imp2d --summary-only
```

For only no_dd, use `--variant no_dd`. `--summary-only` needs no dataset files
or GPU; mode/dataset/seed training selections do not filter saved results.
It rebuilds tables from completed `TEST` histories and per-mode summaries,
retains other existing aggregate rows, and leaves checkpoints/history untouched.
It does not infer completion from a checkpoint alone. Missing/incomplete
results are not trained or inserted as zeroes. If the variant root is missing,
the command reports that rather than creating a new experiment.

Merged output: `<run-dir>/summary.txt`, plus summaries inside each selected
variant at `summary.txt`, `alignn/summary.txt` and `alignn/<dataset>/summary.txt`.
Changed summaries receive timestamped `.bak` copies before atomic replacement.
Old leaf summaries can restore entries removed from an aggregate by old code.

To **add only missing WAS runs** while retaining completed ordinary no_dd runs:

```bash
python -m HERA.scripts.run_hetero_relation_benchmark --dataset native semi imp2d --mode hetero_was --variant no_dd --seed 123 11 1245 --epochs 500 --device cuda:0 --run-dir HERA/logs/hetero_relation_native_semi_imp2d
```

The runner merges existing results after training as well. Requesting both
`--mode hetero hetero_was` is also valid: a matching history with a finite
`TEST` result skips that split, including dataset loading for completed jobs.
If the history is missing/incomplete, resume can also reuse a checkpoint with
a final finite test MAE and matching model/dataset/mode/seed/configuration.
The benchmark runner now supplies `--protect-existing`: an existing split
without a verified final result stops instead of restarting. Entirely missing
splits can still train in an ordinary benchmark command. Resume does **not**
restore an interrupted epoch/optimizer. Use the standalone merge command if
no training at all is desired. Keep the original experiment protocol/root.

### Launch new experiments

Sync the modified HERA code to the machine containing all three datasets.
Activate its HERA Python environment and run from HERA's parent directory.
Use `--mode hetero hetero_was` to run every relation variant with both inputs:

```bash
python -m HERA.scripts.run_hetero_relation_benchmark --dataset native semi imp2d --mode hetero hetero_was --variant baseline no_aa no_dd no_aa_dd --seed 123 11 1245 --epochs 500 --device cuda:0
```

This runs 2 feature modes x 4 variants x 3 datasets x 3 seeds = **72 training/test runs**
sequentially. Each variant has its own root directory under
`HERA/logs/hetero_relation_native_semi_imp2d/` and always uses `--resume`.
Inside a variant, `hetero/r0/...` and `hetero_was/r0/...` are separate folders.
Previously completed hetero runs are skipped when rerunning this expanded
command with the same seed and protocol; their checkpoints remain available.
Checkpoint selection and early stopping use validation MAE; testing happens
automatically at the end of each run.

`hetero` uses the current element's 92-dimensional features. `hetero_was`
uses `was_species`, concatenating the current and reference element features
into 184 dimensions. Graph connectivity, node ordering, geometry, pool masks,
relation settings and the training protocol are paired between these modes
when the same preprocessing version is selected. Explicitly use each dataset's
`--<dataset>-preprocessing reference_v1` for this paired comparison.
The two node-input projections grow; the rest of the configured architecture
is unchanged. The WAS/reference label availability caveat below applies.

Use `--mode hetero_was` to run only the 36 WAS jobs, or `--mode hetero` for
only the 36 ordinary hetero jobs. Omitting `--mode` retains the old hetero-only
behavior. No `--reference full full_x` flag is needed for hetero_was.

For only the original and DD-deleted model:

```bash
python -m HERA.scripts.run_hetero_relation_benchmark --dataset native semi imp2d --mode hetero hetero_was --variant baseline no_dd --seed 123 11 1245 --epochs 500 --device cuda:0
```

For the AA-deleted model versus deleting BOTH AA and DD:

```bash
python -m HERA.scripts.run_hetero_relation_benchmark --dataset native semi imp2d --mode hetero hetero_was --variant no_aa no_aa_dd --seed 123 11 1245 --epochs 500 --device cuda:0
```

For this Windows machine, use the same options with:

```powershell
Set-Location C:/Users/User/Desktop
$py = 'C:/Users/User/.conda/envs/hera/python.exe'
& $py -m HERA.scripts.run_hetero_relation_benchmark --dataset native semi imp2d --mode hetero hetero_was --variant baseline no_aa no_dd no_aa_dd --seed 123 11 1245 --epochs 500 --device cuda:0
```

Add `--dry-run` to print exact underlying HERA.main commands and check data
availability without starting training. Use `--dataset native` to run only
the locally available native dataset. All dataset lists are checked before
any training starts, so a missing second dataset cannot fail after hours of
training on the first one.

## Protocol

- Hetero backbone: shared_residual rank 8; LayerNorm; defect_energy_mean;
  physical graph; r = 0; no auxiliary sparse defect residual; FP32.
- Batch size 8 for every dataset/variant, test batch size 1. Override training
  size with `--batch-size`, keeping it equal for the paired comparisons.
- These three datasets use the existing per-structure random 60/20/20
  train/validation/test split. A given seed selects identical samples for all
  variants. This evaluates random-split performance within each dataset.
- Up to 500 epochs, existing early stopping with patience 50 and 0.5% minimum
  relative validation improvement. The three seeds provide repeated-split
  estimates; they are not three folds of cross validation.
- Use `--seed all` for the repository's ten-seed suite: 240 runs for both
  feature modes, all four variants and three datasets (120 for one mode).
  For five-fold CV, use `--cv5 --seed 123` and a NEW root such as
  `--run-dir HERA/logs/hetero_relation_cv5` (120 runs for both modes).
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

## WAS validation and reference-label availability

**Native update (2026-09-20):** new WAS runs default to `reference_v1`, which
provides original-site element labels and corrects native defect indexing and
vacancy X construction. Results use a new `native_reference_v1` subdirectory;
historical checkpoints and results are preserved. For a paired comparison use
`--mode hetero hetero_was --native-preprocessing reference_v1`, so both modes
receive the same corrected graphs. Both versioned modes need new training on
their first run. Without the flag ordinary hetero keeps its historical inputs
and can reuse its historical results, while WAS uses the new version.
See [native_true_was.md](native_true_was.md) for semantics, checks and commands.

**Semi/imp2d follow-up:** both now implement true WAS in their own reference_v1
paths. Imp2d validation passed on 10,302 converged source-database structures;
the actual training CSV is unavailable. Semi implementation tests passed,
but all local semi CIFs and its CSV are empty, so real-data validation remains
incomplete. See [the detailed audit](imp2d_semi_was_audit_20260920.md).
Explicit `--<dataset>-preprocessing legacy` retains the historical missing-label
fallback `[E(current), E(current)]` and historical result paths.

Historical routing validation: 22 related tests passed after adding WAS routing. They cover all
eight feature/relation combinations, identical topology and geometry across
the paired modes, 184-dimensional inputs, explicit-reference and missing-label
semantics, finite backward passes, strict checkpoint restoration, matching
CLI seeds/configurations, distinct result directories and 72/36 job counts.
No full training benchmark was launched for this change.

## Add ordinary ALIGNN with vacancy X (2026-09-18)

Hetero vacancy inputs already contain a DummySpecies/X node. Adding a second
hetero label called "with X" would repeat the same representation. To add the
existing ordinary ALIGNN full-graph-with-X reference, use `--reference full_x`.
For a paired test of adding X, include both `full` and `full_x`:

```bash
python -m HERA.scripts.run_hetero_relation_benchmark --dataset native semi imp2d --variant baseline no_aa no_dd no_aa_dd --reference full full_x --seed 123 11 1245 --epochs 500 --device cuda:0
```

Run from HERA's parent in its Python environment. On this Windows machine,
replace `python` with `& 'C:/Users/User/.conda/envs/hera/python.exe'`.

| Dataset | Ordinary references | Hetero variants | Runs for 3 seeds |
|---|---|---|---:|
| native | full, full_x | all four | 18 |
| semi | full | all four | 15 |
| imp2d | full | all four | 15 |

Total: **48 training/test runs**. The current semi and imp2d loaders do not
insert vacancy X nodes, so full_x and full have identical inputs there. When
both modes are requested, the existing HERA.main policy runs full once.
No artificial X site is inserted into a structure with no vacancy. Native
substitution/interstitial samples also retain their ordinary full graph;
native vacancy samples gain one X at the dataset's central vacancy position.

Reference runs use the same seeds, splits, batch sizes, epoch limit and
validation-based selection as the hetero runs. They retain the ordinary ALIGNN
architecture: BatchNorm, whole-graph mean pooling and its existing output
layer. Hetero retains its own architecture described above. Interpret full
versus full_x as the X comparison; comparisons to hetero also change the
backbone/readout design.

The references execute before the hetero variants and are saved separately:

```text
HERA/logs/hetero_relation_native_semi_imp2d/references/alignn/native/full/
HERA/logs/hetero_relation_native_semi_imp2d/references/alignn/native/full_x/
HERA/logs/hetero_relation_native_semi_imp2d/references/alignn/semi/full/
HERA/logs/hetero_relation_native_semi_imp2d/references/alignn/imp2d/full/
```

Reusing the same benchmark root resumes/skips completed matching runs. Omitting
`--reference` retains the previous runner behavior and all previous paths.
`--seed all` gives 160 total runs for this expanded suite; `--cv5 --seed 123`
gives 80. Use a separate root when changing the split protocol.

Validation: 27 relevant tests passed, including CLI dispatch to exactly these
four reference dataset/mode combinations, unchanged no-reference behavior,
run counts, output separation, X insertion and finite forward/backward passes.
On the real native `1-ZnO-V_Zn-POSCAR0-Neutral.cif`, the input has 63 real atoms;
full keeps 63 and full_x keeps all 63 plus one X (64 nodes). Existing hetero
also contains X, but its historical native vacancy conversion replaces the
last input site with X (63 total nodes). That pre-existing construction is
preserved by this runner change; it is another reason not to interpret
hetero-versus-full_x differences as an isolated X effect.

The local semi/imp2d availability issues above still apply. No full benchmark
training or new test MAE has been produced by this update.
