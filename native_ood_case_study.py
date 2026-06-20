#!/usr/bin/env python
"""Leave-one-material-out native-defect case study runner.

This script reproduces the OOD case-study protocol used for GaAs in the
manuscript, but for arbitrary ZBSPD-native host materials such as GaN, AlN,
and SiC. For each held-out material, all of its structures are used only for
testing; the remaining native-defect structures are split into train/validation
sets for model selection.
"""

import argparse
import copy
import csv
import json
import os
from datetime import datetime
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
from pymatgen.core import Structure
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from .config.defaults import VALID_MODES, get_config
from .data.datasets import init_elem_embedding, tag_structure_source
from .data.structure_utils import convert_to_sparse_native
from .main import (
    DEFAULT_SEEDS,
    LOCAL_CUTOFF_CHOICES,
    LOCAL_CUTOFF_SWEEP_MODES,
    LOCAL_GRAPH_SWEEP_MODES,
    parse_radius_values,
    set_seed,
    with_radius,
)
from .training.trainer import MEGNetTrainer


MODE_TO_DATASET_KEY = {
    "full": 0,
    "hetero": 1,
    "local": 1,
    "attention": 2,
    "was": 0,
    "hetero_was": 1,
    "hetero_local": 1,
    "hetero_local_was": 1,
    "attention_local": 2,
    "attention_was": 2,
    "attention_local_was": 2,
    "definet": 2,
    "definet_local": 2,
    "definet_was": 2,
    "definet_local_was": 2,
}

DEFAULT_NATIVE_CSV = (
    r"D:/defects/dataset/Dataset_1/Dataset_1/A_rich/Neutral/"
    r"id_prop_A_rich.csv"
)


def parse_native_filename(filename):
    parts = Path(filename).name.split("-")
    if len(parts) < 5:
        raise ValueError(f"Unexpected native-defect filename: {filename}")
    return {
        "file": Path(filename).name,
        "material": parts[1],
        "defect_label": parts[2],
        "defect_group": "-".join(parts[:3]),
        "configuration": parts[3],
    }


def native_defect_marker(filename):
    defect_label = parse_native_filename(filename)["defect_label"]
    return "vacancy" if defect_label.split("_")[0] == "V" else "others"


def load_native_with_metadata(model_name, csv_path=DEFAULT_NATIVE_CSV, local_cutoff=None):
    csv_path = Path(csv_path)
    data_dir = csv_path.parent
    df = pd.read_csv(csv_path, header=None, names=["file", "target"])

    prep = []
    targets = []
    metadata = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading native CIFs"):
        source_path = data_dir / row["file"]
        structure = Structure.from_file(source_path)
        tag_structure_source(structure, source_path, row["file"])
        meta = parse_native_filename(row["file"])
        meta["source_path"] = str(source_path)
        prep.append([structure, native_defect_marker(row["file"])])
        targets.append(float(row["target"]))
        metadata.append(meta)

    skip_was = True
    dataset_full = [
        convert_to_sparse_native(
            structure, defect, 1, f"{model_name}_full", None, skip_was, False
        )
        for structure, defect in tqdm(prep, desc="Converting full graphs")
    ]
    dataset_hetero = [
        convert_to_sparse_native(
            structure,
            defect,
            1,
            f"{model_name}_hetero",
            None,
            skip_was,
            False,
            local_cutoff=local_cutoff,
        )
        for structure, defect in tqdm(prep, desc="Converting hetero graphs")
    ]
    dataset_attn = [
        convert_to_sparse_native(
            structure,
            defect,
            1,
            f"{model_name}_attention",
            None,
            skip_was,
            False,
            local_cutoff=local_cutoff,
        )
        for structure, defect in tqdm(prep, desc="Converting attention graphs")
    ]

    rows = [
        (full, hetero, attn, target, meta)
        for full, hetero, attn, target, meta in zip(
            dataset_full, dataset_hetero, dataset_attn, targets, metadata
        )
        if hetero is not None
    ]
    if not rows:
        raise ValueError("No valid native structures were loaded.")

    dataset_full, dataset_hetero, dataset_attn, targets, metadata = zip(*rows)
    return (
        [list(dataset_full), list(dataset_hetero), list(dataset_attn)],
        torch.tensor(targets).float(),
        pd.DataFrame(metadata),
    )


def expand_mode_runs(model_name, modes, radii):
    runs = []
    for mode in modes:
        if mode not in VALID_MODES:
            raise ValueError(f"Unknown mode: {mode}")
        if mode in LOCAL_GRAPH_SWEEP_MODES:
            selected_radii = radii or LOCAL_CUTOFF_CHOICES
            for radius in selected_radii:
                runs.append(
                    {
                        "label": f"{mode}_r{radius}",
                        "mode": mode,
                        "config": with_radius(get_config(model_name, "native", mode), radius),
                        "local_cutoff": None,
                    }
                )
        elif mode in LOCAL_CUTOFF_SWEEP_MODES:
            selected_radii = radii or LOCAL_CUTOFF_CHOICES
            for radius in selected_radii:
                runs.append(
                    {
                        "label": f"{mode}_r{radius}",
                        "mode": mode,
                        "config": with_radius(get_config(model_name, "native", mode), radius),
                        "local_cutoff": radius,
                    }
                )
        else:
            runs.append(
                {
                    "label": mode,
                    "mode": mode,
                    "config": get_config(model_name, "native", mode),
                    "local_cutoff": None,
                }
            )
    return runs


def subset(values, indices):
    return [values[int(idx)] for idx in indices]


def tensor_subset(values, indices):
    return torch.stack([values[int(idx)] for idx in indices])


def evaluate_case_metrics(y_true, y_pred, metadata):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    abs_err = np.abs(y_pred - y_true)

    gs_errors = []
    top1_hits = []
    swaps = []
    for _, group in metadata.groupby("defect_group", sort=False):
        idx = group.index.to_numpy()
        true_group = y_true[idx]
        pred_group = y_pred[idx]
        true_rel = int(np.argmin(true_group))
        pred_rel = int(np.argmin(pred_group))
        gs_errors.append(abs(pred_group[true_rel] - true_group[true_rel]))
        top1_hits.append(float(pred_rel == true_rel))
        ranking = np.argsort(pred_group)
        true_rank = int(np.where(ranking == true_rel)[0][0])
        swaps.append(float(true_rank))

    return {
        "mae": float(np.mean(abs_err)),
        "rmse": float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
        "ground_state_mae": float(np.mean(gs_errors)),
        "top1_accuracy": float(np.mean(top1_hits)),
        "min_swaps_to_correct": float(np.mean(swaps)),
    }


def prediction_path(run_dir, material, model_name, run_label, seed):
    return Path(run_dir) / material / model_name / run_label / f"seed_{seed}_predictions.csv"


def load_prediction(path):
    df = pd.read_csv(path)
    metrics = evaluate_case_metrics(df["target"], df["prediction"], df)
    return df, metrics


def save_history(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_mae", "train_mse", "val_mae", "best_val_mae", "lr"],
        )
        writer.writeheader()
        writer.writerows(rows)


def train_one_seed(
    config,
    data,
    targets,
    metadata,
    heldout_material,
    seed,
    epochs,
    device,
    val_fraction,
    history_path,
):
    material_mask = metadata["material"].to_numpy() == heldout_material
    test_idx = np.where(material_mask)[0]
    train_pool_idx = np.where(~material_mask)[0]
    if len(test_idx) == 0:
        raise ValueError(f"No samples found for held-out material {heldout_material}.")

    train_idx, val_idx = train_test_split(
        train_pool_idx,
        test_size=val_fraction,
        random_state=seed,
        shuffle=True,
    )

    set_seed(seed)
    trainer = MEGNetTrainer(config, device, seed=seed)
    trainer.prepare_data(
        subset(data, train_idx),
        tensor_subset(targets, train_idx),
        subset(data, val_idx),
        tensor_subset(targets, val_idx),
        "formation_energy",
    )

    min_val = float("inf")
    best_state = None
    history_rows = []
    for epoch in range(epochs):
        train_mae, train_mse = trainer.train_one_epoch()
        val_mae = trainer.evaluate_on_test()
        cur_lr = trainer.optimizer.param_groups[0]["lr"]
        if val_mae < min_val:
            min_val = val_mae
            best_state = copy.deepcopy(trainer.model.state_dict())
        history_rows.append(
            {
                "epoch": epoch + 1,
                "train_mae": f"{train_mae:.6f}",
                "train_mse": f"{train_mse:.6f}",
                "val_mae": f"{val_mae:.6f}",
                "best_val_mae": f"{min_val:.6f}",
                "lr": f"{cur_lr:.8g}",
            }
        )
        print(
            f"  seed={seed} epoch {epoch + 1}/{epochs} "
            f"train_mae={train_mae:.4f} val_mae={val_mae:.4f}"
        )

    save_history(history_path, history_rows)
    test_mae, predictions = trainer.predict_structures(
        subset(data, test_idx),
        tensor_subset(targets, test_idx),
        best_state,
        return_predictions=True,
    )
    test_meta = metadata.iloc[test_idx].reset_index(drop=True).copy()
    test_meta["target"] = tensor_subset(targets, test_idx).numpy()
    test_meta["prediction"] = predictions
    test_meta["abs_error"] = np.abs(test_meta["prediction"] - test_meta["target"])
    metrics = evaluate_case_metrics(test_meta["target"], test_meta["prediction"], test_meta)
    metrics["test_mae_from_trainer"] = float(test_mae)
    return test_meta, metrics


def aggregate_seed_metrics(seed_metrics):
    out = {}
    metric_names = [
        "mae",
        "rmse",
        "ground_state_mae",
        "top1_accuracy",
        "min_swaps_to_correct",
    ]
    for name in metric_names:
        values = [m[name] for m in seed_metrics]
        out[f"{name}_mean"] = float(np.mean(values))
        out[f"{name}_std"] = float(np.std(values))
    return out


def ensemble_predictions(prediction_frames):
    drop_cols = [
        col
        for col in ("seed", "prediction", "abs_error")
        if col in prediction_frames[0].columns
    ]
    base = prediction_frames[0].drop(columns=drop_cols).copy()
    pred_cols = []
    for frame in prediction_frames:
        col = f"prediction_seed_{int(frame['seed'].iloc[0])}"
        base[col] = frame["prediction"].to_numpy()
        pred_cols.append(col)
    base["ensemble_prediction"] = base[pred_cols].mean(axis=1)
    base["ensemble_abs_error"] = np.abs(base["ensemble_prediction"] - base["target"])
    return base


def format_pm(mean, std, digits=3):
    return f"{mean:.{digits}f} +/- {std:.{digits}f}"


def write_summary(run_dir, summary_rows, seed_rows):
    run_dir = Path(run_dir)
    summary_df = pd.DataFrame(summary_rows)
    seed_df = pd.DataFrame(seed_rows)
    summary_df.to_csv(run_dir / "summary.csv", index=False)
    seed_df.to_csv(run_dir / "seed_metrics.csv", index=False)

    lines = [
        "# Native OOD Case Study",
        "",
        "| Material | Model | Mode | N | Defects | MAE | Ensemble MAE | Ground-state MAE | Ensemble GS MAE | Top-1 acc. | Ensemble Top-1 | Min swaps | Ensemble swaps |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| {material} | {model} | {mode} | {n_samples} | {n_defect_groups} | "
            "{mae} | {ensemble_mae:.3f} | {gs_mae} | {ensemble_gs:.3f} | "
            "{top1} | {ensemble_top1:.3f} | {swaps} | {ensemble_swaps:.3f} |".format(
                material=row["material"],
                model=row["model"],
                mode=row["mode"],
                n_samples=row["n_samples"],
                n_defect_groups=row["n_defect_groups"],
                mae=format_pm(row["mae_mean"], row["mae_std"]),
                ensemble_mae=row["ensemble_mae"],
                gs_mae=format_pm(
                    row["ground_state_mae_mean"], row["ground_state_mae_std"]
                ),
                ensemble_gs=row["ensemble_ground_state_mae"],
                top1=format_pm(row["top1_accuracy_mean"], row["top1_accuracy_std"]),
                ensemble_top1=row["ensemble_top1_accuracy"],
                swaps=format_pm(
                    row["min_swaps_to_correct_mean"],
                    row["min_swaps_to_correct_std"],
                ),
                ensemble_swaps=row["ensemble_min_swaps_to_correct"],
            )
        )
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_settings(run_dir, args, materials):
    settings = vars(args).copy()
    settings["materials"] = list(materials)
    settings["run_dir"] = str(run_dir)
    (run_dir / "settings.json").write_text(
        json.dumps(settings, indent=2), encoding="utf-8"
    )


def default_run_dir(log_dir, materials):
    if len(materials) == 1:
        return Path(log_dir) / f"native_ood_{materials[0]}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(log_dir) / f"native_ood_{timestamp}"


def run_case_study(args, runs, materials, run_dir, dataset_cache):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_settings(run_dir, args, materials)

    summary_rows = []
    seed_rows = []

    for run in runs:
        cache_key = run["local_cutoff"]
        if cache_key not in dataset_cache:
            dataset_cache[cache_key] = load_native_with_metadata(
                args.model, args.native_csv, local_cutoff=run["local_cutoff"]
            )
        datasets, targets, metadata = dataset_cache[cache_key]
        data = datasets[MODE_TO_DATASET_KEY[run["mode"]]]

        for material in materials:
            prediction_frames = []
            seed_metrics = []
            print(
                f"\n=== {args.model} | {run['label']} | held out {material} ==="
            )
            for seed in args.seeds:
                pred_path = prediction_path(run_dir, material, args.model, run["label"], seed)
                history_path = pred_path.with_name(f"seed_{seed}_history.csv")
                if args.resume and pred_path.exists():
                    pred_df, metrics = load_prediction(pred_path)
                    print(f"  seed={seed} resume from {pred_path}")
                else:
                    pred_df, metrics = train_one_seed(
                        run["config"],
                        data,
                        targets,
                        metadata,
                        material,
                        seed,
                        args.epochs,
                        args.device,
                        args.val_fraction,
                        history_path,
                    )
                    pred_path.parent.mkdir(parents=True, exist_ok=True)
                    pred_df.insert(0, "seed", seed)
                    pred_df.to_csv(pred_path, index=False)
                pred_df["seed"] = seed
                prediction_frames.append(pred_df)
                seed_metrics.append(metrics)
                seed_row = {
                    "material": material,
                    "model": args.model,
                    "mode": run["label"],
                    "seed": seed,
                }
                seed_row.update(metrics)
                seed_rows.append(seed_row)

            ensemble_df = ensemble_predictions(prediction_frames)
            ensemble_metrics = evaluate_case_metrics(
                ensemble_df["target"],
                ensemble_df["ensemble_prediction"],
                ensemble_df,
            )
            ensemble_out = (
                run_dir
                / material
                / args.model
                / run["label"]
                / "ensemble_predictions.csv"
            )
            ensemble_df.to_csv(ensemble_out, index=False)

            aggregate = aggregate_seed_metrics(seed_metrics)
            material_meta = metadata[metadata["material"] == material]
            summary_row = {
                "material": material,
                "model": args.model,
                "mode": run["label"],
                "n_samples": int(len(material_meta)),
                "n_defect_groups": int(material_meta["defect_group"].nunique()),
                "ensemble_mae": ensemble_metrics["mae"],
                "ensemble_rmse": ensemble_metrics["rmse"],
                "ensemble_ground_state_mae": ensemble_metrics["ground_state_mae"],
                "ensemble_top1_accuracy": ensemble_metrics["top1_accuracy"],
                "ensemble_min_swaps_to_correct": ensemble_metrics[
                    "min_swaps_to_correct"
                ],
            }
            summary_row.update(aggregate)
            summary_rows.append(summary_row)
            write_summary(run_dir, summary_rows, seed_rows)

    print(f"\nSummary written to {run_dir / 'summary.md'}")
    print(f"CSV summary written to {run_dir / 'summary.csv'}")
    return pd.DataFrame(summary_rows)


def main():
    parser = argparse.ArgumentParser(
        description="Run leave-one-material-out OOD case studies on native defects."
    )
    parser.add_argument("--model", default="cgcnn", choices=["cgcnn", "megnet", "definet"])
    parser.add_argument("--mode", nargs="+", default=["full", "hetero"], choices=VALID_MODES)
    parser.add_argument(
        "--materials",
        "--material",
        dest="materials",
        nargs="+",
        default=["GaN", "AlN", "SiC"],
        help="Held-out native-defect material(s), e.g. --material GaN.",
    )
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--atom-init", default="./HERA/atom_init.json")
    parser.add_argument("--native-csv", default=DEFAULT_NATIVE_CSV)
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--r",
        nargs="+",
        default=None,
        help="Radius values for local/cutoff sweep modes; use all for 0 3 4 5 6 7.",
    )
    args = parser.parse_args()

    if not 0 < args.val_fraction < 1:
        parser.error("--val-fraction must be between 0 and 1.")

    radii = parse_radius_values(args.r, parser)
    init_elem_embedding(args.atom_init)

    runs = expand_mode_runs(args.model, args.mode, radii)
    dataset_cache = {}

    run_dir = Path(args.run_dir) if args.run_dir else default_run_dir(
        args.log_dir, args.materials
    )
    run_case_study(args, runs, args.materials, run_dir, dataset_cache)


if __name__ == "__main__":
    main()
