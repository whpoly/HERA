# HERA

HERA: Heterogeneous Region-Aware Message Passing Neural Network for Property Prediction of Crystalline Defects.

This repository contains research code for defect-property prediction on crystalline materials. The current training pipeline supports multiple graph construction and ablation modes.

| Item | Supported Options |
| --- | --- |
| Models | `megnet`, `cgcnn`, `definet`, `all` |
| Modes | `full`, `hetero`, `local`, `attention`, `was`, `hetero_was`, `attention_local`, `attention_was`, `attention_local_was`, `definet`, `definet_local`, `definet_was`, `definet_local_was`, `all` |
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

- `--model`: `megnet`, `cgcnn`, `definet`, or `all`; `all` runs the MEGNet
  and CGCNN suites, with DeFiNet included as CGCNN modes
- `--dataset`: dataset name, or `all` to run every dataset
- `--mode`: one or more of `full`, `hetero`, `local`, `attention`, `was`, `hetero_was`, `attention_local`, `attention_was`, `attention_local_was`, `definet`, `definet_local`, `definet_was`, `definet_local_was`, or `all`
- `--r`: radius values for local graph cropping and hetero local/host cutoff
  sweeps; valid values are `0 3 4 5 6 7` or `all`. The graph edge cutoff
  remains the config value, currently `6`.
- `local` uses the homogeneous model path on the union of all defect-centered
  neighborhoods within radius `r`. `r=0` keeps only the defect atoms.
- `attention_local` and `attention_local_was` use the same direct local graph
  cropping as `local`; graph edges are still built with cutoff `6`.
- `hetero` and `hetero_was` use the same `--r` values as the local/host boundary
  cutoff while keeping the model graph cutoff from the config.
- CGCNN and MEGNet also support ablation modes `was` and `hetero_was`, which
  concatenate current and previous/reference atom features. CGCNN uses
  `atom_init.json` embeddings; MEGNet uses direct float atomic-number pairs.
- Attention ablations for CGCNN and MEGNet are `attention`, `attention_local`,
  `attention_was`, and `attention_local_was`. DeFiNet is run inside the CGCNN
  attention suite as `definet`, `definet_local`, `definet_was`, and
  `definet_local_was`.
- `--device`: for example `cpu`, `cuda:0`
- `--epochs`: number of epochs per seed or CV fold
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
python -m HERA.main --model cgcnn --dataset 2dmd_high --mode local --r 0
python -m HERA.main --model megnet --dataset semi --mode local hetero --r 0
python -m HERA.main --model megnet --dataset vacancy --mode full was local hetero hetero_was attention attention_local attention_was attention_local_was --r 0 3 4 5 6 7
python -m HERA.main --model cgcnn --dataset vacancy --mode full was local hetero hetero_was attention attention_local attention_was attention_local_was definet definet_local definet_was definet_local_was --r 0 3 4 5 6 7
python -m HERA.main --model all --dataset all --mode all --r all
python -m HERA.main --model cgcnn --dataset native --device cuda:0 --epochs 300 --seeds 42 123
python -m HERA.main --model cgcnn --dataset native --mode local --r 0 --cv5 --seeds 42
python -m HERA.main --model cgcnn --dataset native --mode local --r 0 --resume --run-dir logs/run_YYYYMMDD_HHMMSS
```

The DeFiNet-style defect-aware attention/gating experiment uses the same scalar-property training pipeline as the rest of this repository, not the full coordinate-relaxation target from the paper.

## ALIGNN Reference

For ALIGNN-related experiments, we use the official GitHub implementation of ALIGNN rather than maintaining a separate local implementation in this repository. If you need to run or reproduce ALIGNN experiments, please refer directly to the upstream project: https://github.com/usnistgov/alignn

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
  --mode local \
  --r 0 \
  --device cpu \
  --epochs 500 \
  --seeds 42 \
  --atom-init HERA/atom_init.json \
  --log-dir HERA/logs_local_r0_cgcnn
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

The `r{radius}` layer is used for local graph and hetero local/host cutoff
sweeps. It changes only the local radius/local-host boundary, not the graph
edge cutoff.

Each run may contain:

- per-epoch CSV logs
- per-mode `summary.txt`
- one run-level `summary.txt`

When `--resume` is enabled, the CLI does not load or save model checkpoints.
It first reuses a per-mode `summary.txt` when present, which avoids loading the
dataset for that mode/radius. If no summary exists, it treats a seed/fold CSV
as complete only when it contains a valid final `TEST` row; partially completed
runs load data once, reuse completed test MAEs, and retrain missing or
incomplete seed/fold tasks from scratch.

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
