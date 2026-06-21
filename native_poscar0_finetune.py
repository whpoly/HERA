#!/usr/bin/env python
"""Native-defect POSCAR0 fine-tuning experiment.

Protocol:
1. Hold out the selected native-defect materials as target domains.
2. Train on all other native-defect materials for a fixed number of epochs.
3. Evaluate zero-shot predictions on the target materials.
4. For each target material, fine-tune a copy of the trained model using only
   that material's POSCAR0 configurations.
5. Re-evaluate on that material's non-POSCAR0 configurations and compare.
"""

import argparse
import copy
import json
import os
from datetime import datetime
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from .config.defaults import VALID_MODES
from .main import parse_radius_values, set_seed
from .native_ood_case_study import (
    DEFAULT_NATIVE_CSV,
    MODE_TO_DATASET_KEY,
    evaluate_case_metrics,
    expand_mode_runs,
    load_native_with_metadata,
    modes_for_model,
)
from .training.trainer import MEGNetTrainer, set_attr


def subset(values, indices):
    return [values[int(idx)] for idx in indices]


def tensor_subset(values, indices):
    return torch.stack([values[int(idx)] for idx in indices])


def default_run_dir(log_dir, materials):
    label = "_".join(materials)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(log_dir) / f"native_poscar0_finetune_{label}_{timestamp}"


def save_checkpoint(path, trainer):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": trainer.model.state_dict(),
            "scaler": trainer.scaler.state_dict(),
        },
        path,
    )


def load_checkpoint(path, config, device, seed):
    trainer = MEGNetTrainer(config, device, seed=seed)
    checkpoint = torch.load(path, map_location=device)
    trainer.model.load_state_dict(checkpoint["model"])
    trainer.scaler.load_state_dict(checkpoint["scaler"])
    return trainer


def reset_optimizer(trainer, lr):
    optim_config = trainer.config["optim"]
    trainer.optimizer = torch.optim.Adam(trainer.model.parameters(), lr=lr)
    trainer.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        trainer.optimizer,
        factor=optim_config["factor"],
        patience=optim_config["patience"],
        threshold=optim_config["threshold"],
        min_lr=optim_config["min_lr"],
    )


def set_train_loader_keep_scaler(trainer, train_data, train_targets):
    train_data = [set_attr(s, y, "y") for s, y in zip(train_data, train_targets)]
    trainer.train_structures = [
        trainer.converter.convert(s) for s in tqdm(train_data, desc="Converting fine-tune data")
    ]
    trainer.sampler = None
    trainer.trainloader = DataLoader(
        trainer.train_structures,
        batch_size=trainer.config["model"]["train_batch_size"],
        shuffle=True,
        num_workers=0,
        generator=trainer._make_generator(7),
    )


def train_fixed_epochs(trainer, epochs, history_path, phase):
    rows = []
    for epoch in range(epochs):
        train_mae, train_mse = trainer.train_one_epoch()
        cur_lr = trainer.optimizer.param_groups[0]["lr"]
        rows.append(
            {
                "phase": phase,
                "epoch": epoch + 1,
                "train_mae": f"{train_mae:.6f}",
                "train_mse": f"{train_mse:.6f}",
                "lr": f"{cur_lr:.8g}",
            }
        )
        print(
            f"  [{phase}] Epoch {epoch + 1}/{epochs} "
            f"train_mae={train_mae:.4f}"
        )

    history_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(history_path, index=False)


def train_base_model(config, data, targets, train_idx, epochs, device, seed, out_dir):
    checkpoint_path = out_dir / "base_checkpoint.pth"
    history_path = out_dir / "base_history.csv"

    set_seed(seed)
    trainer = MEGNetTrainer(config, device, seed=seed)
    train_data = subset(data, train_idx)
    train_targets = tensor_subset(targets, train_idx)
    trainer.prepare_data(
        train_data,
        train_targets,
        train_data[:1],
        train_targets[:1],
        "formation_energy",
    )
    train_fixed_epochs(trainer, epochs, history_path, phase="base")
    save_checkpoint(checkpoint_path, trainer)
    return trainer


def finetune_model(base_trainer, data, targets, finetune_idx, epochs, lr, out_dir):
    checkpoint_path = out_dir / "finetuned_checkpoint.pth"
    history_path = out_dir / "finetune_history.csv"

    trainer = base_trainer
    set_train_loader_keep_scaler(
        trainer,
        subset(data, finetune_idx),
        tensor_subset(targets, finetune_idx),
    )
    reset_optimizer(trainer, lr)
    train_fixed_epochs(trainer, epochs, history_path, phase="finetune")
    save_checkpoint(checkpoint_path, trainer)
    return trainer


def predict_dataframe(trainer, data, targets, metadata, indices, model_state_dict):
    if len(indices) == 0:
        return pd.DataFrame(), None

    mae, predictions = trainer.predict_structures(
        subset(data, indices),
        tensor_subset(targets, indices),
        model_state_dict,
        return_predictions=True,
    )
    out = metadata.iloc[indices].reset_index(drop=True).copy()
    out["target"] = tensor_subset(targets, indices).numpy()
    out["prediction"] = predictions
    out["abs_error"] = np.abs(out["prediction"] - out["target"])
    metrics = evaluate_case_metrics(out["target"], out["prediction"], out)
    metrics["test_mae_from_trainer"] = float(mae)
    return out, metrics


def metric_row(material, model_name, mode_label, phase, seed, metrics, n_train, n_finetune, n_test):
    row = {
        "material": material,
        "model": model_name,
        "mode": mode_label,
        "phase": phase,
        "seed": int(seed),
        "n_train_base": int(n_train),
        "n_finetune_poscar0": int(n_finetune),
        "n_test": int(n_test),
    }
    row.update({key: float(value) for key, value in metrics.items()})
    return row


def save_poscar0_values(path, material, zero_df, finetuned_df):
    if zero_df.empty:
        return

    poscar0_df = zero_df.copy()
    poscar0_df = poscar0_df.rename(
        columns={
            "prediction": "zero_shot_prediction",
            "abs_error": "zero_shot_abs_error",
        }
    )
    if not finetuned_df.empty:
        fit_df = finetuned_df[["file", "prediction", "abs_error"]].rename(
            columns={
                "prediction": "finetuned_prediction",
                "abs_error": "finetuned_abs_error",
            }
        )
        poscar0_df = poscar0_df.merge(fit_df, on="file", how="left")
    poscar0_df.insert(0, "fine_tune_material", material)
    poscar0_df.to_csv(path, index=False)


def run_model_mode(args, model_name, run, datasets, targets, metadata, run_dir):
    mode_label = run["label"]
    data = datasets[MODE_TO_DATASET_KEY[run["mode"]]]
    out_dir = run_dir / model_name / mode_label
    out_dir.mkdir(parents=True, exist_ok=True)

    target_mask = metadata["material"].isin(args.materials).to_numpy()
    poscar0_mask = metadata["configuration"].eq("POSCAR0").to_numpy()
    base_train_idx = np.where(~target_mask)[0]
    target_eval_idx = np.where(target_mask & ~poscar0_mask)[0]

    if len(target_eval_idx) == 0:
        raise ValueError("No non-POSCAR0 target samples found for evaluation.")

    base_checkpoint = out_dir / "base_checkpoint.pth"

    if args.resume and base_checkpoint.exists():
        print(f"  Resume base checkpoint: {base_checkpoint}")
        base_trainer = load_checkpoint(base_checkpoint, run["config"], args.device, args.seed)
    else:
        print(f"  Training base model: {model_name} {mode_label}")
        base_trainer = train_base_model(
            run["config"],
            data,
            targets,
            base_train_idx,
            args.epochs,
            args.device,
            args.seed,
            out_dir,
        )

    base_state = copy.deepcopy(base_trainer.model.state_dict())

    summary_rows = []
    comparison_rows = []
    prediction_dir = out_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    poscar0_dir = out_dir / "poscar0_values"
    poscar0_dir.mkdir(parents=True, exist_ok=True)

    for material in args.materials:
        material_mask = metadata["material"].eq(material).to_numpy()
        material_all_idx = np.where(material_mask)[0]
        material_eval_idx = np.where(material_mask & ~poscar0_mask)[0]
        material_ft_idx = np.where(material_mask & poscar0_mask)[0]
        material_ft_dir = out_dir / f"finetune_{material}"
        material_ft_checkpoint = material_ft_dir / "finetuned_checkpoint.pth"

        if len(material_ft_idx) == 0:
            raise ValueError(f"No POSCAR0 samples found for {material}.")
        if len(material_eval_idx) == 0:
            raise ValueError(f"No non-POSCAR0 evaluation samples found for {material}.")

        zero_all_df, zero_all_metrics = predict_dataframe(
            base_trainer, data, targets, metadata, material_all_idx, base_state
        )
        zero_eval_df, zero_eval_metrics = predict_dataframe(
            base_trainer, data, targets, metadata, material_eval_idx, base_state
        )

        zero_poscar0_df, _ = predict_dataframe(
            base_trainer, data, targets, metadata, material_ft_idx, base_state
        )

        if args.resume and material_ft_checkpoint.exists():
            print(f"  Resume {material} fine-tuned checkpoint: {material_ft_checkpoint}")
            ft_trainer = load_checkpoint(
                material_ft_checkpoint,
                run["config"],
                args.device,
                args.seed,
            )
        else:
            print(f"  Fine-tuning {material} on its POSCAR0 samples")
            base_trainer.model.load_state_dict(base_state)
            ft_trainer = finetune_model(
                base_trainer,
                data,
                targets,
                material_ft_idx,
                args.finetune_epochs,
                args.finetune_lr,
                material_ft_dir,
            )

        ft_state = copy.deepcopy(ft_trainer.model.state_dict())
        ft_eval_df, ft_eval_metrics = predict_dataframe(
            ft_trainer, data, targets, metadata, material_eval_idx, ft_state
        )
        ft_poscar0_df, _ = predict_dataframe(
            ft_trainer, data, targets, metadata, material_ft_idx, ft_state
        )

        zero_all_df.to_csv(prediction_dir / f"{material}_zero_shot_all.csv", index=False)
        zero_eval_df.to_csv(prediction_dir / f"{material}_zero_shot_non_poscar0.csv", index=False)
        ft_eval_df.to_csv(prediction_dir / f"{material}_finetuned_non_poscar0.csv", index=False)
        save_poscar0_values(
            poscar0_dir / f"{material}_poscar0_values.csv",
            material,
            zero_poscar0_df,
            ft_poscar0_df,
        )

        summary_rows.append(
            metric_row(
                material,
                model_name,
                mode_label,
                "zero_shot_all",
                args.seed,
                zero_all_metrics,
                len(base_train_idx),
                len(material_ft_idx),
                len(material_all_idx),
            )
        )
        summary_rows.append(
            metric_row(
                material,
                model_name,
                mode_label,
                "zero_shot_non_poscar0",
                args.seed,
                zero_eval_metrics,
                len(base_train_idx),
                len(material_ft_idx),
                len(material_eval_idx),
            )
        )
        summary_rows.append(
            metric_row(
                material,
                model_name,
                mode_label,
                "finetuned_non_poscar0",
                args.seed,
                ft_eval_metrics,
                len(base_train_idx),
                len(material_ft_idx),
                len(material_eval_idx),
            )
        )

        comparison_rows.append(
            {
                "material": material,
                "model": model_name,
                "mode": mode_label,
                "seed": int(args.seed),
                "n_train_base": int(len(base_train_idx)),
                "n_finetune_poscar0": int(len(material_ft_idx)),
                "n_test_non_poscar0": int(len(material_eval_idx)),
                "zero_shot_mae": zero_eval_metrics["mae"],
                "finetuned_mae": ft_eval_metrics["mae"],
                "delta_mae": zero_eval_metrics["mae"] - ft_eval_metrics["mae"],
                "zero_shot_ground_state_mae": zero_eval_metrics["ground_state_mae"],
                "finetuned_ground_state_mae": ft_eval_metrics["ground_state_mae"],
                "delta_ground_state_mae": (
                    zero_eval_metrics["ground_state_mae"]
                    - ft_eval_metrics["ground_state_mae"]
                ),
                "zero_shot_top1_accuracy": zero_eval_metrics["top1_accuracy"],
                "finetuned_top1_accuracy": ft_eval_metrics["top1_accuracy"],
            }
        )

    return summary_rows, comparison_rows


def plot_comparison(comparison_df, run_dir):
    if comparison_df.empty:
        return []

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outputs = []
    figure_dir = Path(run_dir) / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    plot_specs = [
        (
            "zero_shot_mae",
            "finetuned_mae",
            "delta_mae",
            "MAE on non-POSCAR0 target structures (eV)",
            "dMAE",
            "mae",
        ),
        (
            "zero_shot_ground_state_mae",
            "finetuned_ground_state_mae",
            "delta_ground_state_mae",
            "Ground-state MAE on non-POSCAR0 target groups (eV)",
            "dGS",
            "ground_state_mae",
        ),
    ]

    for material, material_df in comparison_df.groupby("material", sort=False):
        material_df = material_df.sort_values(["model", "mode"])
        labels = [f"{row.model} {row.mode}" for row in material_df.itertuples()]
        x = np.arange(len(labels))
        width = 0.38

        for zero_col, fine_col, delta_col, ylabel, delta_label, suffix in plot_specs:
            fig, ax = plt.subplots(figsize=(max(7.5, len(labels) * 0.9), 4.8))
            ax.bar(
                x - width / 2,
                material_df[zero_col],
                width,
                label="Zero-shot",
                color="#64748b",
            )
            ax.bar(
                x + width / 2,
                material_df[fine_col],
                width,
                label="POSCAR0 fine-tuned",
                color="#0f766e",
            )
            ax.set_title(f"{material}: POSCAR0 fine-tuning effect")
            ax.set_ylabel(ylabel)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=35, ha="right")
            ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.4)
            ax.legend(frameon=False)
            for idx, row in enumerate(material_df.itertuples()):
                zero_value = getattr(row, zero_col)
                fine_value = getattr(row, fine_col)
                delta_value = getattr(row, delta_col)
                ax.text(
                    idx,
                    max(zero_value, fine_value),
                    f"{delta_label}={delta_value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
            fig.tight_layout()
            output = figure_dir / f"{material}_poscar0_finetune_{suffix}.png"
            fig.savefig(output, dpi=220)
            plt.close(fig)
            outputs.append(output)

    return outputs


def main():
    parser = argparse.ArgumentParser(
        description="Train native-defect models, POSCAR0 fine-tune target materials, and compare."
    )
    parser.add_argument(
        "--model",
        dest="models",
        nargs="+",
        default=["cgcnn", "megnet", "definet"],
        choices=["cgcnn", "megnet", "definet"],
        help="Model architecture(s), e.g. --model cgcnn megnet definet.",
    )
    parser.add_argument(
        "--mode",
        nargs="+",
        default=["full", "hetero", "attention"],
        choices=VALID_MODES,
    )
    parser.add_argument(
        "--materials",
        "--material",
        dest="materials",
        nargs="+",
        default=["GaN", "AlN", "SiC"],
        help="Target native-defect material(s).",
    )
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--finetune-epochs", type=int, default=100)
    parser.add_argument("--finetune-lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--atom-init", default="./HERA/atom_init.json")
    parser.add_argument("--native-csv", default=DEFAULT_NATIVE_CSV)
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--r",
        nargs="+",
        default=None,
        help=(
            "Radius values for local graph sweep modes; use all for 0 3 4 5 6 7. "
            "Hetero is fixed to r0 in this experiment."
        ),
    )
    args = parser.parse_args()

    radii = parse_radius_values(args.r, parser)
    set_seed(args.seed)

    from .data.datasets import init_elem_embedding

    init_elem_embedding(args.atom_init)

    run_dir = Path(args.run_dir) if args.run_dir else default_run_dir(args.log_dir, args.materials)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "settings.json").write_text(
        json.dumps(vars(args), indent=2), encoding="utf-8"
    )

    dataset_cache = {}
    all_summary_rows = []
    all_comparison_rows = []

    for model_name in args.models:
        model_modes = modes_for_model(model_name, args.mode)
        runs = expand_mode_runs(model_name, model_modes, radii)
        for run in runs:
            cache_key = (model_name, run["local_cutoff"])
            if cache_key not in dataset_cache:
                dataset_cache[cache_key] = load_native_with_metadata(
                    model_name,
                    args.native_csv,
                    local_cutoff=run["local_cutoff"],
                )
            datasets, targets, metadata = dataset_cache[cache_key]

            print(f"\n=== {model_name} | {run['label']} ===")
            summary_rows, comparison_rows = run_model_mode(
                args,
                model_name,
                run,
                datasets,
                targets,
                metadata,
                run_dir,
            )
            all_summary_rows.extend(summary_rows)
            all_comparison_rows.extend(comparison_rows)
            pd.DataFrame(all_summary_rows).to_csv(run_dir / "summary.csv", index=False)
            pd.DataFrame(all_comparison_rows).to_csv(run_dir / "comparison.csv", index=False)

    summary_df = pd.DataFrame(all_summary_rows)
    comparison_df = pd.DataFrame(all_comparison_rows)
    summary_df.to_csv(run_dir / "summary.csv", index=False)
    comparison_df.to_csv(run_dir / "comparison.csv", index=False)
    plot_paths = plot_comparison(comparison_df, run_dir)

    print(f"\nSummary written to {run_dir / 'summary.csv'}")
    print(f"Comparison written to {run_dir / 'comparison.csv'}")
    if plot_paths:
        print("Figures written:")
        for path in plot_paths:
            print(f"  {path}")


if __name__ == "__main__":
    main()
