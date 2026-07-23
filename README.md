# HERA

HERA: Heterogeneous Region-Aware Message Passing Neural Network for Property Prediction of Crystalline Defects.

This repository contains research code for defect-property prediction on crystalline materials. The current training pipeline supports multiple graph construction and ablation modes.

| Item | Supported Options |
| --- | --- |
| Models | `megnet`, `cgcnn`, `definet`, `alignn`, `all` |
| Modes | `full`, `full_x`, `hetero`, `hetero_fixed_pool`, `attention`, `was_x`, `hetero_was`, `attention_was`, `definet`, `definet_was`, `all` |
| Datasets | `vacancy`, `2dmd_high`, `native`, `och`, `imp2d`, `semi`, `all` |

## Repository Layout

- `main.py`: training CLI entry point
- `config/`: dataset and mode configuration
- `data/`: dataset loading and graph conversion
- `models/`: MEGNet and CGCNN implementations
- `training/`: trainer, loss, and logging utilities
- `utils/`: helper utilities

## Requirements

Recommended environment:

- Python 3.12
- `torch`
- `torch-geometric`
- `numpy`
- `pandas`
- `scikit-learn`
- `tqdm`
- `pymatgen`

Conda installation:

```bash
conda env create -f environment.yml
conda activate hera
```

The provided `environment.yml` follows the known-working pip environment in
`requirements.txt`: Python 3.12, PyTorch 2.5.1 with CUDA 12.4 wheels, matching
PyTorch Geometric CUDA wheels, and only the dependencies needed for HERA
training, explanation, and visualization.

## Data Preparation

This repository does not include the raw datasets or `atom_init.json`.

Before training, make sure the following resources are available in the paths expected by the code:

- `atom_init.json`
- `2d-materials-point-defects-all/...` for `vacancy` and `2dmd_high`
- `Dataset_1/...` for `native` and `semi`
- `../autodl-tmp/rs2re_h_ads/...` for `och`
- `imp2d/imp2d/...` for `imp2d`

If `atom_init.json` is stored elsewhere, pass it with `--atom-init`.

## Quick Start

Because the CLI uses package-relative imports, run commands from the parent directory of `HERA`:

```bash
cd path/to/parent/of/HERA
python -m HERA.main --help
```

Common arguments:

- `--model`: `alignn`, `megnet`, `cgcnn`, `definet`, or `all`; `all` runs the ALIGNN,
  MEGNet, and CGCNN suites in that order, with DeFiNet-style modes included for ALIGNN and CGCNN
- For the default ALIGNN benchmark (`--mode all` or no explicit `--mode`),
  `hetero` runs first and `definet` runs second, followed by their related
  `hetero_fixed_pool`, `hetero_was`, and `definet_was` modes. Explicitly
  listed modes keep the order supplied on the command line.
- `--dataset`: dataset name, or `all` to run every dataset
- `--mode`: one or more of `full`, `full_x`, `hetero`, `attention`, `was_x`,
  `hetero_was`, `attention_was`, `definet`, `definet_was`, or `all`
- `--r`: radius values for hetero local/host boundary sweeps; valid values are
  `0 3 4 5 6 7` or `all`. The graph edge cutoff remains the config value,
  currently `6`; no reduced/cropped graph modes are exposed.
- `full_x` is the old full-graph-with-X comparison: vacancy-style datasets
  such as `vacancy` and `2dmd_high` add DummySpecies/X vacancy sites to the
  full graph; datasets without an X site use the same graph as `full`. When a
  benchmark requests both modes, `full_x` is therefore skipped for datasets
  that contain no vacancy samples (`och`, `imp2d`, and `semi`).
- `was_x` applies WAS atom features to the `full_x` representation. The old
  full-graph-only `was` mode is no longer exposed.
- `hetero` and `hetero_was` use the `--r` values as the local/host boundary
  cutoff while keeping the full model graph and config graph cutoff.
- CGCNN, MEGNet, and ALIGNN support WAS ablation modes `was_x` and `hetero_was`,
  which concatenate current and previous/reference atom features.
- Attention ablations are `attention` and `attention_was`. DeFiNet-style modes
  are `definet` and `definet_was` for CGCNN and ALIGNN.
- `--device`: for example `cpu`, `cuda:0`
- `--epochs`: number of epochs per seed or CV fold
- `--train-batch-size` / `--test-batch-size`: override batch size for all runs
- `--alignn-train-batch-size` / `--alignn-test-batch-size`: override batch size
  only for ALIGNN runs. ALIGNN defaults to `64` for training and `1` for
  validation/test because its bond-angle line graph is much more memory
  intensive.
- `--alignn-max-neighbors` / `--alignn-cutoff`: reduce ALIGNN graph size when
  memory is still too high. `--alignn-max-neighbors 12` is usually the first
  knob to try because ALIGNN angle edges grow roughly with neighbor count
  squared.
- ALIGNN defaults use the original high-level layout with 3 ALIGNN blocks
  followed by 3 graph-conv blocks, while keeping HERA's current feature sizes
  (`hidden=64`, `edge_embed=40`, `angle_embed=40`), `cutoff=6`, and
  `max_neighbors=12`.
- `--alignn-embedding-size` / `--alignn-nblocks` / `--alignn-gcn-blocks` /
  `--alignn-angle-embed-size`: reduce ALIGNN model capacity for lower memory use.
- `--alignn-grad-accum-steps`: keep an effective large batch while using a
  smaller memory-resident micro-batch, e.g. `--alignn-train-batch-size 4
  --alignn-grad-accum-steps 16` gives an effective ALIGNN training batch of 64.
- `--alignn-amp`: train ALIGNN with CUDA automatic mixed precision to reduce
  activation memory without changing graph topology.
- `--seeds`: one or more random seeds for ordinary train/val/test splits; with
  `--cv5`, pass exactly one random state
- `--cv5` / `--five-fold-cv`: use 5-fold cross validation. Each run uses one
  fold for test, the next fold for validation, and the remaining three folds
  for training, so the train/val/test split is roughly 60/20/20.
- `--atom-init`: path to `atom_init.json`
- `--log-dir`: output directory for logs
- `--run-dir`: exact `logs/run_{timestamp}` directory to use instead of creating a new one
- `--resume`: skip a completed mode/radius immediately when its per-mode
  `summary.txt` exists; otherwise skip completed seed/fold tasks inside that mode

Example training commands:

```bash
python -m HERA.main --model megnet --dataset vacancy
python -m HERA.main --model megnet --dataset semi --mode hetero --r 0
python -m HERA.main --model all --dataset vacancy --mode all --r 0
python -m HERA.main --model all --dataset all --mode all --r all
python -m HERA.main --model cgcnn --dataset native --device cuda:0 --epochs 300 --seeds 42 123
python -m HERA.main --model cgcnn --dataset native --mode hetero --r 0 --cv5 --seeds 42
python -m HERA.main --model cgcnn --dataset native --mode hetero --r 0 --resume --run-dir logs/run_YYYYMMDD_HHMMSS
python -m HERA.main --model alignn --dataset 2dmd_high --mode all --r 0 --alignn-train-batch-size 1 --alignn-test-batch-size 1
python -m HERA.main --model alignn --dataset 2dmd_high --mode all --r 0 --alignn-amp
python -m HERA.main --model alignn --dataset 2dmd_high --mode all --r 0 --alignn-train-batch-size 4 --alignn-grad-accum-steps 16 --alignn-amp
```

### Native OOD case studies

To reproduce the GaAs-style leave-one-material-out case study on selected
native-defect hosts, run one held-out material per command. These commands can
be distributed across different machines because each one writes an independent
run directory:

```bash
python -m HERA.native_ood_case_study \
  --model cgcnn megnet definet \
  --mode full full_x hetero attention \
  --material GaN \
  --epochs 500 \
  --seeds 123 11 1245 34 42 80 13232 8 99 101 \
  --device cuda:0 \
  --log-dir HERA/logs \
  --resume \
  --atom-init HERA/atom_init.json
```

Change `--material GaN` to `--material AlN` or `--material SiC` on the other
machines. With a single held-out material, the default output directories are
`logs/native_ood_GaN/`, `logs/native_ood_AlN/`, and `logs/native_ood_SiC/`
under the selected `--log-dir`.
Each run treats that material as a completely unseen test host and uses the
other native-defect structures for training/validation. The outputs include a
single-seed `summary.md`, machine-readable `summary.csv`, full candidate
`seed_metrics.csv`, `selection_summary.csv`, per-sample predictions, and
figures under `<material>/figures/`. The seed list is used to search for one
case where `hetero_r0` performs best; results are not averaged across seeds.
When multiple models are passed, results are kept separate under each
material's `cgcnn/`, `megnet/`, and `definet/` subdirectories. DefiNet is a
defect-aware attention/gating baseline, so it runs through its attention-style
mode while CGCNN and MEGNet run `full`, `full_x`, `hetero_r0`, and `attention`.

The DeFiNet-style defect-aware attention/gating experiment uses the same scalar-property training pipeline as the rest of this repository, not the full coordinate-relaxation target from the paper.

### Native POSCAR0 fine-tuning

To test POSCAR0-based target adaptation without a validation split, hold out
GaN, AlN, and SiC together, train for a fixed 500 epochs on all other native
materials, then fine-tune separately for each target material using only that
material's POSCAR0 configurations:

```bash
python -m HERA.native_poscar0_finetune \
  --model cgcnn megnet definet \
  --mode full full_x hetero attention \
  --material GaN AlN SiC \
  --epochs 500 \
  --finetune-epochs 100 \
  --finetune-lr 1e-4 \
  --seed 42 \
  --device cuda:0 \
  --log-dir HERA/logs \
  --resume \
  --atom-init HERA/atom_init.json
```

This protocol does not use validation data or best-validation checkpoint
selection. For each model and mode, the base checkpoint is the model after the
last training epoch. The base model is tested zero-shot on each target material,
then copied and fine-tuned per material from the same base checkpoint using
that material's POSCAR0 rows only. The main comparison reports zero-shot versus
POSCAR0 fine-tuned performance on the target material's non-POSCAR0
configurations. Outputs include `summary.csv`, `comparison.csv`, prediction
CSVs, per-material POSCAR0 value tables under
`<model>/<mode>/poscar0_values/`, and figures under `figures/`.

### Native initial/relaxed leave-one-out final DFE

To leave out every eligible native-defect material in turn, train on the other
materials, and predict the final relaxed defect formation energy from initial
and relaxed structures:

```bash
python -m HERA.native_initial_relaxed_leave_one_out --seed 42 --model cgcnn megnet definet --mode full hetero attention --epochs 500 --device cuda:0 --run-dir HERA/logs/native_initial_relaxed_seed42 --atom-init HERA/atom_init.json
```

By default the script discovers all materials that have both POSCAR0 initial
structures and non-POSCAR0 relaxed structures. For each held-out material it
runs two comparisons: train on all usable rows from the other materials and
test the held-out POSCAR0 initial structures, or train on all usable rows from
the other materials plus the held-out POSCAR0 initial structures and test the
held-out lowest-energy relaxed structure for each native defect group. The
target is the lowest non-POSCAR0 DFE for each native defect group. Outputs
include `summary.csv`, `summary.md`, per-sample
prediction CSVs, material eligibility metadata, and figures following the
style of `scripts/plot_native_zero_shot_performance.py`. The outer loop is
material-first: after one held-out material finishes across all selected
models/modes, the script writes that material's model-performance figure and
DFT-ordered energy-comparison figure. Outputs are organized as
`<run-dir>/<material>/<model>/<mode>/` for checkpoints and prediction CSVs,
with plots in `<run-dir>/<material>/figures/`. When a fixed `--run-dir` is
reused, existing checkpoints and prediction CSVs are loaded automatically, so
`--resume` is no longer required.

## ALIGNN / HeteroALIGNN

This repository includes a PyTorch Geometric ALIGNN implementation with a
HERA-compatible heterogeneous variant. Use it with:

```bash
python -m HERA.main --model alignn --dataset native --mode hetero --r 0
```

Supported ALIGNN modes are `full`, `full_x`, `hetero`, `hetero_fixed_pool`,
`attention`, `was_x`, `hetero_was`, `attention_was`,
`definet`, and `definet_was`. HeteroALIGNN uses the same
`atom`/`defect` node split and `aa`/`dd`/`ad`/`da` edge split as the existing
HERA hetero models, while dynamically building the ALIGNN bond-angle line graph
from periodic edge vectors during each forward pass.

HeteroALIGNN also embeds the ordered relation pair of every line-graph angle
edge alongside its geometric angle RBF features. Reverse traversals share one
embedding (for example, `aa->ad` and `da->aa`), giving six learnable angle
relation classes for the standard four edge types without introducing an
artificial direction dependence.

HeteroALIGNN uses only physical periodic-neighbor edges. It does not add
synthetic zero-distance self-loops because its gated convolutions already have
root/residual updates. Consequently, a single-defect graph with no physical
defect-defect bond keeps the `dd` edge store empty and skips that relation branch.
The directed `ad` and `da` edge stores are both retained for message routing,
but they share one reciprocal host-defect edge embedding and one convolutional
network at every HeteroALIGNN/GCN layer.
Relation messages are combined per destination node using their accumulated
ALIGNN gate weights. A relation contributes only to nodes with real incoming
edges of that type, and atom/defect nodes each receive one self/root update
after all of their incoming relation messages have been combined. This avoids
cross-graph leakage from relation types that occur elsewhere in a mixed batch.

For a controlled ALIGNN comparison, neither HeteroALIGNN nor DefiNetALIGNN uses
a learned graph-level virtual node. HeteroALIGNN reads out separate atom, defect,
and edge pools; DefiNetALIGNN reads out its shared node and edge pools.

## Smoke Check

There is currently no built-in automated test suite such as `pytest` or `unittest`.

For a quick manual validation:

1. Confirm the CLI starts normally:

```bash
cd path/to/parent/of/HERA
python -m HERA.main --help
```

2. Run a minimal one-epoch experiment after preparing the required dataset files:

```bash
python -m HERA.main \
  --model cgcnn \
  --dataset native \
  --mode hetero \
  --r 0 \
  --device cpu \
  --epochs 500 \
  --seeds 42 \
  --atom-init HERA/atom_init.json \
  --log-dir HERA/logs_hetero_r0_cgcnn
```

This smoke check helps verify:

- imports are working
- dataset paths are correct
- graph conversion can run
- training and logging can start

## Outputs

Training logs are written under:

```text
logs/run_{timestamp}/{model}/{dataset}/{mode}/
logs/run_{timestamp}/{model}/{dataset}/{mode}/r{radius}/
```

The `r{radius}` layer is used for hetero local/host boundary sweeps. It changes
only the local-host node-type boundary, not the graph edge cutoff or graph size.

Each run may contain:

- per-epoch CSV logs
- per-split best-validation checkpoints named `seed{seed}_best_checkpoint.pth`
  or `seed{random_state}_fold{fold}_best_checkpoint.pth`
- per-mode `summary.txt`
- one run-level `summary.txt`

Each checkpoint stores `model`, `scaler`, `config`, dataset/mode labels, test
MAE, best validation MAE, and train/validation/test source metadata so the
trainer can be reconstructed later for explanations.

When `--resume` is enabled, the CLI first reuses a per-mode `summary.txt` when
present, which avoids loading the dataset for that mode/radius. If no summary
exists, it treats a seed/fold CSV as complete only when it contains a valid
final `TEST` row; completed splits are skipped as-is, while missing or
incomplete seed/fold tasks are retrained and write fresh checkpoints.

## Batch Explanations

The training CLI can run `GNNExplainer` after each seed's test prediction and
save notebook-free visualizations for every selected dataset/mode:

```bash
python -m HERA.main \
  --model cgcnn \
  --dataset all \
  --mode full hetero \
  --device cuda:0 \
  --explain
```

By default, each seed writes explanations under:

```text
logs/run_{timestamp}/{model}/{dataset}/{mode}/explanations/seed_{seed}/
logs/run_{timestamp}/{model}/{dataset}/{mode}/r{radius}/explanations/seed_{seed}/
```

Each explanation folder contains:

- `index.csv`: target, prediction, absolute error, attribution range, and file paths
- `{cif_name}.xyz`: OVITO extended XYZ with per-atom `importance`, `Color`, and `Radius`

If two samples share the same CIF stem, the later files get suffixes such as
`_02` to avoid overwriting.

Useful options:

```bash
--explain-max-samples 20      # explain only the first 20 test samples per seed
--explain-epochs 50           # faster GNNExplainer optimization
--explain-formats ovito       # default: OVITO extended XYZ only
--explain-formats csv html png # optionally request the legacy outputs
--explain-cmap viridis_r      # reversed viridis attribution colors
--explain-dir explain_runs    # write explanations outside the log directory
```

## Notes

- Full training depends on external datasets being placed exactly where `data/datasets.py` expects them.
- Use `python -m HERA.main --help` to inspect all available CLI options.
