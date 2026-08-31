# HERA

HERA: Heterogeneous Region-Aware Message Passing Neural Network for Property Prediction of Crystalline Defects.

This repository contains research code for defect-property prediction on crystalline materials. The current training pipeline supports multiple graph construction and ablation modes.

| Item | Supported Options |
| --- | --- |
| Models | `megnet`, `cgcnn`, `definet`, `alignn`, `hypergraph`, `all` |
| Modes | `sparse`, `full`, `full_x`, `hetero`, `hetero_fixed_pool`, `attention`, `was_x`, `hetero_was`, `attention_was`, `definet`, `definet_was`, `hypergraph`, `hypergraph_was`, `all` |
| Datasets | `vacancy`, `2dmd_low`, `2dmd_high`, `native`, `och`, `imp2d`, `semi`, `all` |

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
- `2d-materials-point-defects-all/...` for `vacancy`, `2dmd_low`, and `2dmd_high`
- `Dataset_1/...` for `native` and `semi`
- `../autodl-tmp/rs2re_h_ads/...` for `och`
- `imp2d/imp2d/...` for `imp2d`

`2dmd_low` is the standalone low-density 2DMD benchmark and loads only the
`low_density_defects/MoS2` and `low_density_defects/WSe2` subsets.

If `atom_init.json` is stored elsewhere, pass it with `--atom-init`.

## Quick Start

Because the CLI uses package-relative imports, run commands from the parent directory of `HERA`:

```bash
cd path/to/parent/of/HERA
python -m HERA.main --help
```

Common arguments:

- `--model`: `alignn`, `megnet`, `cgcnn`, `definet`, `hypergraph`, or `all`; `all` runs the ALIGNN,
  MEGNet, and CGCNN suites in that order, with DeFiNet-style modes included for ALIGNN and CGCNN
- For the default ALIGNN benchmark (`--mode all` or no explicit `--mode`),
  `hetero` runs first and `definet` second, followed by their related
  `hetero_fixed_pool`, `hetero_was`, and `definet_was` modes. Explicitly
  listed modes keep the order supplied on the command line. The hybrid
  `hypergraph` mode is included in the default CGCNN, MEGNet, and ALIGNN
  suites.
- Multi-seed ALIGNN benchmarks run in seed-major order: every requested mode
  is completed for one seed before the next seed starts. Each mode still writes
  to its own `<run-dir>/alignn/<dataset>/<mode>/` directory, so histories,
  checkpoints, summaries, and `--resume` remain mode-specific.
- MEGNet runs `sparse` first for the `vacancy`, `2dmd_low`, and `2dmd_high`
  benchmarks, including when `sparse` appears later in an explicit `--mode`
  list. Other datasets do not run the sparse representation.
- `--dataset`: one or more dataset names, or `all` to run every dataset
- `--mode`: one or more of `sparse`, `full`, `full_x`, `hetero`, `hetero_fixed_pool`, `attention`, `was_x`,
  `hetero_was`, `attention_was`, `definet`, `definet_was`, `hypergraph`, `hypergraph_was`, or `all`
- `--r`: radius values for hetero local/host boundary sweeps; valid values are
  `0 3 4 5 6 7` or `all`. The graph edge cutoff remains the config value,
  currently `6`; no reduced/cropped graph modes are exposed.
- `full_x` is the old full-graph-with-X comparison: vacancy-style datasets
  such as `vacancy`, `2dmd_low`, and `2dmd_high` add DummySpecies/X vacancy sites to the
  full graph; datasets without an X site use the same graph as `full`. When a
  benchmark requests both modes, `full_x` is therefore skipped for datasets
  that contain no vacancy samples (`och`, `imp2d`, and `semi`).
- `sparse` reproduces the legacy `MEGNET_SPARSE` representation/model and is only
  available for `--model megnet` on `vacancy`, `2dmd_low`, and `2dmd_high`. It retains only
  defect sites and restores the original two-value current/reference species
  features, 12 Å cutoff, max/max aggregation, hidden size 64, and 3 MEGNet
  blocks. Its training follows the current common benchmark protocol: AdamW,
  validation-MAE scheduling, standard early stopping, and the current
  dataset-specific batch sizes.
- `was_x` applies WAS atom features to the `full_x` representation. The old
  full-graph-only `was` mode is no longer exposed.
- `hetero`, `hetero_fixed_pool`, and `hetero_was` use the `--r` values as the local/host boundary
  cutoff while keeping the full model graph and config graph cutoff.
- `hypergraph` keeps defect neighborhoods independent. Every defect gets a
  singleton core hyperedge and a local hyperedge containing that center defect
  plus pristine atoms within 3.0 A; other defect atoms are excluded. A
  pristine atom may belong to multiple genuinely overlapping neighborhoods,
  and pristine atoms outside every neighborhood share one optional far-field
  hyperedge. Distances use periodic minimum images. The variable hyperedges are
  pooled individually and combined by semantic type only at graph readout.
  With `--model cgcnn`, `megnet`, or `alignn`, the original physical bond (and
  ALIGNN angle) updates are interleaved with hyperedge updates. `--model
  hypergraph` runs the pure hypergraph baseline without a physical backbone.
  Override the threshold with `--hypergraph-radius`.
- `hypergraph_was` keeps the same physical graph and per-defect hyperedges as
  `hypergraph`, but concatenates current and previous/reference (`was`) atom
  embeddings. It is available for CGCNN, MEGNet, and ALIGNN so the two modes
  form a controlled input-feature comparison.
- CGCNN, MEGNet, and ALIGNN support WAS ablation modes `was_x`, `hetero_was`,
  and `hypergraph_was`,
  which concatenate current and previous/reference atom features.
- Attention ablations are `attention` and `attention_was`. DeFiNet-style modes
  are `definet` and `definet_was` for CGCNN and ALIGNN.
- `--device`: for example `cpu`, `cuda:0`
- `--epochs`: number of epochs per seed or CV fold
- All training protocols use AdamW with a default weight decay of `1e-4`.
- The `vacancy` dataset defaults to a training batch size of `8` for every
  model. All other datasets default to `64` for training, and every model uses
  `1` for validation/test.
- `--batch-size` / `--train-batch-size`: override training batch size for every
  selected run. `--test-batch-size` overrides validation/test batch size.
- `--alignn-train-batch-size` / `--alignn-test-batch-size`: override batch size
  only for ALIGNN runs. ALIGNN otherwise defaults to `64` for training and `1`
  for validation/test.
- `--alignn-max-neighbors` / `--alignn-cutoff`: reduce ALIGNN graph size when
  memory is still too high. The neighbor cap is applied independently to each
  target node type for every center node, so `--alignn-max-neighbors 12` keeps
  up to 12 host and 12 defect neighbors. ALIGNN angle edges grow roughly with
  the resulting total neighbor count squared.
- ALIGNN defaults use the original high-level layout with 3 ALIGNN blocks
  followed by 3 graph-conv blocks, while keeping HERA's current feature sizes
  (`hidden=64`, `edge_embed=40`, `angle_embed=40`), `cutoff=6`, and
  `max_neighbors=12`.
- `--alignn-embedding-size` / `--alignn-nblocks` / `--alignn-gcn-blocks` /
  `--alignn-angle-embed-size`: reduce ALIGNN model capacity for lower memory use.
- `--alignn-grad-accum-steps`: keep an effective large batch while using a
  smaller memory-resident micro-batch, e.g. `--alignn-train-batch-size 4
  --alignn-grad-accum-steps 16` gives an effective ALIGNN training batch of 64.
- HeteroALIGNN applies LayerNorm only to each residual delta by default. Use
  `--alignn-hetero-node-norm layernorm`, `batchnorm`, or `none` to compare
  normalization choices while preserving the residual identity path. Pass
  multiple values in one command, for example `--alignn-hetero-node-norm
  layernorm batchnorm none`, to run all choices in one benchmark with separate
  result directories and combined summaries. Any explicitly selected norm,
  including a single value, uses its own `norm_<choice>` directory so
  `--resume` cannot reuse results produced by a different normalization.
- All Attention and DeFiNet variants use LayerNorm instead of batch-dependent
  normalization. Their message-passing residuals follow `root +
  LayerNorm(delta)`; AttentionMEGNet applies this independently to node, edge,
  and global-state deltas.
- ALIGNN, CGCNN, MEGNet, and hypergraph runs all stop early by default after 50 epochs
  without a relative validation MAE improvement greater than `0.5%`. Use
  `--early-stopping-patience` and
  `--early-stopping-min-delta-percent` to tune this for every selected model,
  or set patience to `0` to disable early stopping. The best-validation
  checkpoint is still used for the final test evaluation.
- `--alignn-amp`: train ALIGNN with CUDA automatic mixed precision to reduce
  activation memory without changing graph topology.
- `--seed`: one or more random seeds for ordinary train/val/test splits
  (default: `123`), for example `--seed 123 42 99`; use `--seed all` for the
  standard 10-seed benchmark. With `--cv5`, pass exactly one seed.
- `--cv5` / `--five-fold-cv`: use 5-fold cross validation. Each run uses one
  fold for test, the next fold for validation, and the remaining three folds
  for training, so the train/val/test split is roughly 60/20/20.
- `--atom-init`: path to `atom_init.json`
- `--log-dir`: output directory for logs
- `--run-dir`: exact `logs/run_{timestamp}` directory to use instead of creating a new one
- `--resume`: skip a completed mode/radius immediately when its per-mode
  `summary.txt` exists; otherwise skip completed seed/fold tasks inside that mode
- The `ReduceLROnPlateau` scheduler monitors validation MAE. Protocols that
  intentionally have no validation split, such as native POSCAR0 fixed-epoch
  training, keep a fixed learning rate.

Example training commands:

```bash
python -m HERA.main --model megnet --dataset vacancy
python -m HERA.main --model all --dataset 2dmd_low --mode all --r 0
python -m HERA.main --model megnet --dataset vacancy 2dmd_high --mode sparse --epochs 500 --seed 123 --run-dir HERA/logs/megnet_sparse_reproduction --resume
python -m HERA.main --model megnet --dataset semi --mode hetero --r 0
python -m HERA.main --model all --dataset vacancy --mode all --r 0
python -m HERA.main --model all --dataset all --mode all --r all
python -m HERA.main --model cgcnn --dataset native --device cuda:0 --epochs 300 --seed 123
python -m HERA.main --model cgcnn --dataset native --device cuda:0 --epochs 300 --seed all
python -m HERA.main --model cgcnn --dataset native --mode hetero --r 0 --cv5 --seed 123
python -m HERA.main --model cgcnn --dataset native --mode hetero --r 0 --resume --run-dir logs/run_YYYYMMDD_HHMMSS
python -m HERA.main --model cgcnn --dataset native --mode hypergraph --hypergraph-radius 3.0 --device cuda:0
python -m HERA.main --model megnet --dataset native --mode hypergraph --hypergraph-radius 3.0 --device cuda:0
python -m HERA.main --model alignn --dataset native --mode hypergraph --hypergraph-radius 3.0 --device cuda:0
python -m HERA.main --model all --dataset native --mode hypergraph hypergraph_was --hypergraph-radius 3.0 --device cuda:0
python -m HERA.main --model hypergraph --dataset native --hypergraph-radius 3.0 --device cuda:0
python -m HERA.main --model alignn --dataset 2dmd_high --mode all --r 0 --alignn-train-batch-size 1 --alignn-test-batch-size 1
python -m HERA.main --model alignn --dataset 2dmd_high --mode all --r 0 --alignn-amp
python -m HERA.main --model alignn --dataset 2dmd_high --mode all --r 0 --alignn-train-batch-size 4 --alignn-grad-accum-steps 16 --alignn-amp
```

On the configured Slurm cluster, `scripts/submit_hypergraph_benchmark.sbatch`
runs the four standard hypergraph datasets in parallel, then reuses one GPU for
the seed-123 native leave-one-material-out comparison between full ALIGNN and
Hypergraph ALIGNN. Its LOO results are written under
`HERA/hypergraph/four_gpu/native_leave_one_out/`.

On Windows, the complete MEGNET_SPARSE benchmark is also packaged as:

```powershell
HERA\scripts\run_megnet_sparse_benchmark.ps1
```

It runs `2dmd_high` and `vacancy` for seed 123, writes resumable
outputs to `HERA/logs/megnet_sparse_reproduction/`, and records launcher status,
combined console output, and the final exit code in that directory.

To insert this benchmark after an already-running 2dmd `full_x` mode and then
resume that run, use `scripts/run_sparse_between_2dmd_modes.ps1`. The watcher
waits for both the `TEST` row and best checkpoint before stopping the original
process.

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
  --seed all \
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
  --seed 123 \
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

### Native final-structure leave-one-out and POSCAR0 transfer

To leave out every eligible native-defect material in turn, directly predict
its final relaxed structures, and compare against POSCAR0 one-shot transfer:

```bash
python -m HERA.native_initial_relaxed_leave_one_out --seed 123 --model cgcnn megnet definet --mode full hetero attention --epochs 500 --device cuda:0 --run-dir HERA/logs/native_initial_relaxed_seed123 --atom-init HERA/atom_init.json
```

For the controlled ordinary full-graph ALIGNN versus Hypergraph ALIGNN
comparison:

```bash
python -m HERA.native_initial_relaxed_leave_one_out --seed all --model alignn --mode full hypergraph --hypergraph-radius 3.0 --epochs 500 --device cuda:0 --run-dir HERA/logs/native_alignn_hypergraph_loo --atom-init HERA/atom_init.json
```

This is also the runner's default model/mode pair, so `--model alignn --mode
full hypergraph` may be omitted. The same ALIGNN physical backbone and native
LOO splits are used in both runs; the hypergraph variant additionally applies
the independent per-defect hyperedge updates.

By default the script discovers all materials that have both POSCAR0 initial
structures and non-POSCAR0 relaxed structures. Every CIF retains its own DFE
from the native CSV; labels are never replaced by a group minimum. For each
held-out material the script trains a source model on all rows from the other
materials and evaluates two protocols on the same final structures (the
lowest-DFE non-POSCAR0 CIF in each defect group): direct prediction, and a copy
of the source model fine-tuned using only the held-out POSCAR0 rows. Use
`--finetune-epochs` and `--finetune-lr` to control adaptation. This LOO runner
updates the full model for 20 epochs by default, using discriminative learning
rates of `1e-5` for the GNN backbone and `1e-4` for the prediction head. This
allows representation adaptation while limiting catastrophic forgetting. Use
`--finetune-backbone-lr` to override the backbone rate. Outputs include
`summary.csv`, `overall_mae.csv` (sample-weighted across held-out materials),
`summary.md`, per-sample
prediction CSVs, material eligibility metadata, and figures following the
style of `scripts/plot_native_zero_shot_performance.py`. The outer loop is
material-first: after one held-out material finishes across all selected
models/modes, the script writes that material's model-performance figure and
DFT-ordered energy-comparison figure. Outputs are organized as
`<run-dir>/<material>/<model>/<mode>/` for checkpoints and prediction CSVs,
with plots in `<run-dir>/<material>/figures/`. When a fixed `--run-dir` is
reused, existing checkpoints and prediction CSVs are loaded automatically, so
`--resume` is no longer required.
When both ALIGNN variants are selected, the runner also writes paired
`alignn_hypergraph_comparison.csv`, cross-material/seed
`alignn_hypergraph_aggregate.csv`, and
`figures/hypergraph_alignn_vs_alignn_loo_mae.png`. Positive relative MAE
improvement means Hypergraph ALIGNN performs better than ordinary ALIGNN.

## ALIGNN / HeteroALIGNN

This repository includes a PyTorch Geometric ALIGNN implementation with a
HERA-compatible heterogeneous variant. Use it with:

```bash
python -m HERA.main --model alignn --dataset native --mode hetero --r 0
```

Supported ALIGNN modes are `full`, `full_x`, `hetero`, `hetero_fixed_pool`,
`attention`, `was_x`, `hetero_was`, `attention_was`,
`definet`, `definet_was`, `hypergraph`, and `hypergraph_was`. HeteroALIGNN uses the same
`atom`/`defect` node split and `aa`/`dd`/`ad`/`da` edge split as the existing
HERA hetero models, while dynamically building the ALIGNN bond-angle line graph
from periodic edge vectors during each forward pass.

HeteroALIGNN uses one shared geometric-angle update for every line-graph edge.
The angle convolution does not distinguish `aa`/`dd`/`ad`/`da` relation
combinations.

All CGCNN, MEGNet, and ALIGNN heterogeneous graph conversions use only physical
periodic-neighbor edges. They do not add synthetic zero-distance self-loops
because the backbones already preserve root/node features internally.
Consequently, a single-defect graph with no physical defect-defect bond keeps
the `dd` edge store empty for every backbone.
The directed `ad` and `da` edge stores are both retained for message routing,
and use separate learnable edge embeddings and convolutional networks at every
HeteroALIGNN graph-convolution layer. HeteroMEGNet likewise uses a separate
MEGNet module for each direction. HeteroCGCNN also uses separate modules for
all four directed relation types.
Each heterogeneous backbone aggregates neighbors within each relation first
and preserves those relation-wise aggregates in separate slots. A
node-type-specific FFN then fuses the root state with its incoming slots; no
cross-relation sum or mean is applied before this update. Missing relation
slots are represented by zeros. The host FFN receives `[root, HH, HD]`
(`aa`, `da` in source-to-destination edge naming), while the defect FFN
receives `[root, DH, DD]` (`ad`, `dd`).
HeteroCGCNN and HeteroMEGNet apply LayerNorm to the FFN residual delta before
adding it to the root state. HeteroMEGNet applies the same update to its global
state, so the normalization never depends on batch statistics.
Within each ALIGNN relation, gated messages are normalized only by that
relation's accumulated gate weights. This preserves relation identity and
avoids cross-graph leakage from relation types that occur elsewhere in a mixed
batch.

For a controlled comparison with DefiNetALIGNN, HeteroALIGNN has no learned
global graph feature and does not pool edge embeddings into its prediction
head. It concatenates only the separate atom and defect node pools, while
DefiNetALIGNN uses its homogeneous node pool.

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
  --seed 123 \
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
