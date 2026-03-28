# HERA

HERA: Heterogeneous Region-Aware Message Passing Neural Network for Property Prediction of Crystalline Defects.

This repository contains research code for defect-property prediction on crystalline materials. The current training pipeline supports two backbone models and four graph construction modes.

| Item | Supported Options |
| --- | --- |
| Models | `megnet`, `cgcnn` |
| Modes | `sparse`, `full`, `hetero`, `attention` |
| Datasets | `vacancy`, `2dmd_high`, `native`, `och`, `imp2d`, `semi` |

## Repository Layout

- `main.py`: training CLI entry point
- `config/`: dataset and mode configuration
- `data/`: dataset loading and graph conversion
- `models/`: MEGNet and CGCNN implementations
- `training/`: trainer, loss, and logging utilities
- `utils/`: helper utilities

## Requirements

Recommended environment:

- Python 3.10+
- `torch`
- `torch-geometric`
- `numpy`
- `pandas`
- `scikit-learn`
- `tqdm`
- `pymatgen`

Example installation:

```bash
pip install torch
pip install torch-geometric
pip install numpy pandas scikit-learn tqdm pymatgen
```

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

- `--model`: `megnet` or `cgcnn`
- `--dataset`: dataset name
- `--mode`: one or more of `sparse`, `full`, `hetero`, `attention`
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
python -m HERA.main --model cgcnn --dataset native --device cuda:0 --epochs 300 --seeds 42 123
```

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
  --epochs 1 \
  --seeds 42 \
  --atom-init HERA/atom_init.json \
  --log-dir HERA/logs_smoke
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

## Notes

- Full training depends on external datasets being placed exactly where `data/datasets.py` expects them.
- Use `python -m HERA.main --help` to inspect all available CLI options.
