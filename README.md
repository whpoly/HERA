# HERA

HERA: Heterogeneous Region-Aware Message Passing for Property Prediction of Crystalline Defects.

This repository contains research code for training crystal graph neural networks on several defect-property datasets. It currently supports two backbones (`MEGNet` and `CGCNN`) and four graph construction modes:

- `sparse`
- `full`
- `hetero`
- `attention`

## What Is In This Repo

- `main.py`: command-line entry point for training experiments
- `config/`: dataset- and mode-specific training configs
- `data/`: dataset loading and graph conversion utilities
- `models/`: MEGNet / CGCNN model definitions
- `training/`: trainer, losses, and logging helpers
- `utils/`: scaling utilities

## Requirements

The project is written for Python 3.10+ and depends on the following Python packages:

- `torch`
- `torch-geometric`
- `numpy`
- `pandas`
- `scikit-learn`
- `tqdm`
- `pymatgen`

Install `torch` and `torch-geometric` with versions that match your CPU/CUDA environment, then install the remaining packages, for example:

```bash
pip install torch
pip install torch-geometric
pip install numpy pandas scikit-learn tqdm pymatgen
```

## Data And Input Files

This repository does not bundle the raw datasets or `atom_init.json`.

Before launching training, make sure the required files and directories exist in the locations expected by the loaders:

- `atom_init.json`
- `2d-materials-point-defects-all/...` for `vacancy` and `2dmd_high`
- `Dataset_1/...` for `native` and `semi`
- `../autodl-tmp/rs2re_h_ads/...` for `och`
- `imp2d/imp2d/...` for `imp2d`

If `atom_init.json` lives somewhere else, pass it explicitly with `--atom-init`.

## How To Run Training

Because the CLI uses package-relative imports, run commands from the parent directory of this repository.

```bash
cd path/to/parent/of/HERA
python -m HERA.main --help
```

Supported arguments:

- `--model`: `megnet` or `cgcnn`
- `--dataset`: `vacancy`, `2dmd_high`, `native`, `och`, `imp2d`, `semi`
- `--mode`: one or more of `sparse`, `full`, `hetero`, `attention`
- `--device`: for example `cpu`, `cuda:0`
- `--epochs`: number of epochs per random seed
- `--seeds`: one or more integer seeds
- `--atom-init`: path to `atom_init.json`
- `--log-dir`: output directory for logs and summaries

Example commands:

```bash
python -m HERA.main --model megnet --dataset vacancy
python -m HERA.main --model cgcnn --dataset 2dmd_high --mode sparse
python -m HERA.main --model megnet --dataset semi --mode sparse hetero
python -m HERA.main --model cgcnn --dataset native --device cuda:0 --epochs 300 --seeds 42 123
```

## Quick Smoke Run

Once the dataset files are available, the smallest practical end-to-end check is a single mode, a single seed, and a single epoch:

```bash
python -m HERA.main \
  --model cgcnn \
  --dataset native \
  --mode sparse \
  --device cpu \
  --epochs 1 \
  --seeds 42 \
  --atom-init HERA/atom_init.json \
  --log-dir HERA/logs_smoke
```

This is useful to verify that:

- the package imports correctly
- the dataset paths are valid
- graph conversion works
- training and logging start without crashing

## How To Run Tests

There is currently no built-in automated test suite such as `pytest` or `unittest` in this repository.

For now, the recommended way to validate the environment is:

1. Check that the CLI can start correctly:

```bash
cd path/to/parent/of/HERA
python -m HERA.main --help
```

2. Run a minimal smoke experiment with a single mode, a single seed, and one epoch after preparing the required dataset files:

```bash
python -m HERA.main \
  --model cgcnn \
  --dataset native \
  --mode sparse \
  --device cpu \
  --epochs 1 \
  --seeds 42 \
  --atom-init HERA/atom_init.json \
  --log-dir HERA/logs_smoke
```

This manual check verifies that:

- package imports work
- dataset paths are correct
- graph conversion succeeds
- training can start and write logs

## Outputs

Training writes logs to:

```text
logs/{model}_{dataset}_{timestamp}/
```

For each selected mode, the run directory contains:

- per-epoch CSV logs
- per-mode `summary.txt`
- a run-level `summary.txt`

## Notes

- There is currently no automated test suite in the repository.
- Full training requires the external datasets to be prepared exactly as expected by `data/datasets.py`.
- If you want to inspect all CLI options, use `python -m HERA.main --help`.
