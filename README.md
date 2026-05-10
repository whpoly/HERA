# HERA

HERA: Heterogeneous Region-Aware Message Passing Neural Network for Property Prediction of Crystalline Defects.

This repository contains research code for defect-property prediction on crystalline materials. The current training pipeline supports multiple graph construction and ablation modes.

| Item | Supported Options |
| --- | --- |
| Models | `megnet`, `cgcnn`, `definet` |
| Modes | `sparse`, `full`, `hetero`, `local`, `attention`, `was`, `hetero_was` |
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

- `--model`: `megnet`, `cgcnn`, or `definet`
- `--dataset`: dataset name, or `all` to run every dataset
- `--mode`: one or more of `sparse`, `full`, `hetero`, `local`, `attention`, `was`, `hetero_was`
- `--r`: local radius/cutoff values for `--mode local`; valid values are `0 3 4 5 6 7`
- `local` uses the same heterogeneous architecture as hetero mode, but keeps only
  the defect neighborhood within radius `r`. `r=0` is the sparse-equivalent local
  input expanded into the local/hetero format.
- CGCNN also supports ablation modes `was` (`cgcnn_was`) and `hetero_was`
  (`hetero_cgcnn_was`), which concatenate the current atom embedding and
  previous/reference atom embedding from `atom_init.json`. `cgcnn_was` uses the
  complete CGCNN crystal graph, not the sparse graph.
- `--device`: for example `cpu`, `cuda:0`
- `--epochs`: number of epochs per seed
- `--seeds`: one or more random seeds
- `--atom-init`: path to `atom_init.json`
- `--log-dir`: output directory for logs

Example training commands:

```bash
python -m HERA.main --model megnet --dataset vacancy
python -m HERA.main --model cgcnn --dataset 2dmd_high --mode sparse
python -m HERA.main --model megnet --dataset semi --mode sparse hetero
python -m HERA.main --model cgcnn --dataset vacancy --mode local --r 0 3 4 5 6 7
python -m HERA.main --model cgcnn --dataset native --device cuda:0 --epochs 300 --seeds 42 123
python -m HERA.main --model definet --dataset all
```

The `definet` option runs the DeFiNet-style defect-aware attention/gating experiment in `attention` mode across the selected dataset(s). It uses the same scalar-property training pipeline as the rest of this repository, not the full coordinate-relaxation target from the paper.

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
  --mode sparse \
  --device cpu \
  --epochs 500 \
  --seeds 42 \
  --atom-init HERA/atom_init.json \
  --log-dir HERA/logs_sparse_cgcnn
```

This smoke check helps verify:

- imports are working
- dataset paths are correct
- graph conversion can run
- training and logging can start

## Outputs

Training logs are written under:

```text
logs/{model}_{dataset}_{timestamp}/
```

Each run may contain:

- per-epoch CSV logs
- per-mode `summary.txt`
- one run-level `summary.txt`

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
logs/{model}_{dataset}_{timestamp}/{mode}/explanations/seed_{seed}/
logs/{model}_all_{timestamp}/{dataset}/{mode}/explanations/seed_{seed}/
```

Each explanation folder contains:

- `index.csv`: target, prediction, absolute error, attribution range, and file paths
- `{cif_name}.csv`: per-atom attribution values and colors
- `{cif_name}.html`: standalone browser visualization using the same atom-color idea as the notebook
- `{cif_name}.png`: static batch-friendly preview

If two samples share the same CIF stem, the later files get suffixes such as
`_02` to avoid overwriting.

Useful options:

```bash
--explain-max-samples 20      # explain only the first 20 test samples per seed
--explain-epochs 50           # faster GNNExplainer optimization
--explain-formats csv html    # skip PNG generation
--explain-cmap viridis_r      # reversed viridis attribution colors
--explain-dir explain_runs    # write explanations outside the log directory
```

## Notes

- Full training depends on external datasets being placed exactly where `data/datasets.py` expects them.
- Use `python -m HERA.main --help` to inspect all available CLI options.
