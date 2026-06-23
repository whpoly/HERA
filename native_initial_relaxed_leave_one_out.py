#!/usr/bin/env python
"""Leave-one-material-out native-defect initial/relaxed DFE tests.

The target in this experiment is the final relaxed defect formation energy
for each native-defect group. By default that value is the lowest non-POSCAR0
DFE observed for the group. Two cross-domain protocols are run for every
eligible held-out material:

1. Train on all usable rows from the other materials, then test the held-out
   POSCAR0 initial structures against the final relaxed DFE.
2. Train on all usable rows from the other materials plus the held-out POSCAR0
   initial structures, then test held-out relaxed structures against the final
   relaxed DFE.
"""

from __future__ import annotations

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
from sklearn.model_selection import train_test_split

from .config.defaults import VALID_MODES
from .data.datasets import dataset_index_for_mode, init_elem_embedding, representation_for_mode
from .main import parse_radius_values, set_seed
from .native_ood_case_study import (
    DEFAULT_NATIVE_CSV,
    evaluate_case_metrics,
    expand_mode_runs,
    load_native_with_metadata,
    mode_display_name,
    model_mode_display,
    modes_for_model,
)
from .training.trainer import MEGNetTrainer


PROTOCOLS = {
    "other_train__initial_test": {
        "display": "Train: other -> Test: initial",
        "train_pool": "other",
        "test_pool": "initial",
    },
    "other_plus_initial_train__relaxed_test": {
        "display": "Train: other+held-out initial -> Test: relaxed",
        "train_pool": "other_plus_initial",
        "test_pool": "relaxed",
    },
}

PROTOCOL_ORDER = list(PROTOCOLS)
PROTOCOL_COLORS = {
    "other_train__initial_test": "#3b82f6",
    "other_plus_initial_train__relaxed_test": "#0f766e",
}


def subset(values, indices):
    return [values[int(idx)] for idx in indices]


def tensor_subset(values, indices):
    return torch.stack([values[int(idx)] for idx in indices])


def default_run_dir(log_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(log_dir) / f"native_initial_relaxed_loo_{timestamp}"


def add_final_relaxed_targets(metadata, targets):
    """Attach final relaxed DFE metadata to every row.

    The final target is the minimum DFE among non-POSCAR0 configurations in the
    same defect group. Groups without any relaxed structure are marked invalid
    for this experiment and are left out of train/test pools.
    """
    out = metadata.reset_index(drop=True).copy()
    out["raw_target"] = np.asarray(targets, dtype=float)
    out["is_initial"] = out["configuration"].astype(str).eq("POSCAR0")
    out["is_relaxed"] = ~out["is_initial"]
    out["has_relaxed_final"] = False
    out["final_target"] = np.nan
    out["final_file"] = None
    out["final_configuration"] = None

    for defect_group, group in out.groupby("defect_group", sort=False):
        relaxed = group[group["is_relaxed"]].copy()
        if relaxed.empty:
            continue
        relaxed = relaxed.sort_values(["raw_target", "file"], ascending=[True, True])
        final_row = relaxed.iloc[0]
        idx = group.index
        out.loc[idx, "has_relaxed_final"] = True
        out.loc[idx, "final_target"] = float(final_row["raw_target"])
        out.loc[idx, "final_file"] = final_row["file"]
        out.loc[idx, "final_configuration"] = final_row["configuration"]

    return out


def eligible_materials(metadata):
    rows = []
    valid = metadata["has_relaxed_final"].to_numpy()
    initial = metadata["is_initial"].to_numpy()
    relaxed = metadata["is_relaxed"].to_numpy()
    for material, group in metadata.groupby("material", sort=True):
        idx = group.index.to_numpy()
        n_initial = int(np.sum(valid[idx] & initial[idx]))
        n_relaxed = int(np.sum(valid[idx] & relaxed[idx]))
        rows.append(
            {
                "material": material,
                "n_initial_with_final": n_initial,
                "n_relaxed_with_final": n_relaxed,
                "eligible": n_initial > 0 and n_relaxed > 0,
            }
        )
    table = pd.DataFrame(rows)
    return table[table["eligible"]]["material"].astype(str).tolist(), table


def split_train_val(train_idx, val_fraction, seed):
    if len(train_idx) < 2:
        raise ValueError("Need at least two training samples for train/validation split.")
    return train_test_split(
        np.asarray(train_idx, dtype=int),
        test_size=val_fraction,
        random_state=seed,
        shuffle=True,
    )


def save_history(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def save_checkpoint(path, trainer, model_state_dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model_state_dict,
            "scaler": trainer.scaler.state_dict(),
        },
        path,
    )


def load_checkpoint(path, config, device, seed):
    trainer = MEGNetTrainer(config, device, seed=seed)
    checkpoint = torch.load(path, map_location=device)
    trainer.model.load_state_dict(checkpoint["model"])
    trainer.scaler.load_state_dict(checkpoint["scaler"])
    return trainer, copy.deepcopy(checkpoint["model"])


def train_with_validation(
    config,
    data,
    targets,
    train_idx,
    val_idx,
    epochs,
    device,
    seed,
    history_path,
):
    set_seed(seed)
    trainer = MEGNetTrainer(config, device, seed=seed)
    trainer.prepare_data(
        subset(data, train_idx),
        tensor_subset(targets, train_idx),
        subset(data, val_idx),
        tensor_subset(targets, val_idx),
        "formation_energy",
    )

    best_val = float("inf")
    best_state = copy.deepcopy(trainer.model.state_dict())
    rows = []
    for epoch in range(epochs):
        train_mae, train_mse = trainer.train_one_epoch()
        val_mae = trainer.evaluate_on_test()
        cur_lr = trainer.optimizer.param_groups[0]["lr"]
        if val_mae < best_val:
            best_val = float(val_mae)
            best_state = copy.deepcopy(trainer.model.state_dict())
        rows.append(
            {
                "epoch": epoch + 1,
                "train_mae": f"{train_mae:.6f}",
                "train_mse": f"{train_mse:.6f}",
                "val_mae": f"{val_mae:.6f}",
                "best_val_mae": f"{best_val:.6f}",
                "lr": f"{cur_lr:.8g}",
            }
        )
        print(
            f"  epoch {epoch + 1}/{epochs} "
            f"train_mae={train_mae:.4f} val_mae={val_mae:.4f}"
        )

    save_history(history_path, rows)
    return trainer, best_state, best_val


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


def metric_row(
    material,
    model_name,
    mode_label,
    protocol,
    seed,
    metrics,
    n_train,
    n_val,
    n_test,
    best_val,
):
    row = {
        "material": material,
        "model": model_name,
        "mode": mode_label,
        "model_mode": model_mode_display(model_name, mode_label),
        "protocol": protocol,
        "protocol_display": PROTOCOLS[protocol]["display"],
        "seed": int(seed),
        "n_train": int(n_train),
        "n_val": int(n_val),
        "n_test": int(n_test),
        "best_val_mae": float(best_val),
    }
    row.update({key: float(value) for key, value in metrics.items()})
    return row


def masks_for_material(metadata, material):
    valid = metadata["has_relaxed_final"].to_numpy()
    material_mask = metadata["material"].astype(str).eq(str(material)).to_numpy()
    initial = metadata["is_initial"].to_numpy()
    relaxed = metadata["is_relaxed"].to_numpy()
    other = ~material_mask

    return {
        "train_other": np.where(other & valid)[0],
        "train_other_plus_initial": np.where((other & valid) | (material_mask & valid & initial))[0],
        "test_initial": np.where(material_mask & valid & initial)[0],
        "test_relaxed": np.where(material_mask & valid & relaxed)[0],
    }


def prediction_path(out_dir, protocol, material):
    return out_dir / "predictions" / protocol / f"{material}.csv"


def run_training_group(
    args,
    run,
    data,
    targets,
    metadata,
    material,
    train_kind,
    train_idx,
    group_dir,
):
    train_idx, val_idx = split_train_val(train_idx, args.val_fraction, args.seed)
    checkpoint_path = group_dir / f"{train_kind}_checkpoint.pth"
    history_path = group_dir / f"{train_kind}_history.csv"

    if args.resume and checkpoint_path.exists():
        print(f"  Resume {train_kind} checkpoint: {checkpoint_path}")
        trainer, state = load_checkpoint(checkpoint_path, run["config"], args.device, args.seed)
        best_val = float("nan")
    else:
        print(
            f"  Train {train_kind} model for held-out {material} "
            f"(train={len(train_idx)}, val={len(val_idx)})"
        )
        trainer, state, best_val = train_with_validation(
            run["config"],
            data,
            targets,
            train_idx,
            val_idx,
            args.epochs,
            args.device,
            args.seed,
            history_path,
        )
        save_checkpoint(checkpoint_path, trainer, state)

    return trainer, state, best_val, train_idx, val_idx


def run_material(args, model_name, run, data, targets, metadata, material, out_dir):
    idx = masks_for_material(metadata, material)
    if len(idx["test_initial"]) == 0 or len(idx["test_relaxed"]) == 0:
        return [], {
            "material": material,
            "reason": "missing initial or relaxed held-out samples with relaxed final target",
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    other_trainer, other_state, other_best_val, other_train_idx, other_val_idx = (
        run_training_group(
            args,
            run,
            data,
            targets,
            metadata,
            material,
            "other_train",
            idx["train_other"],
            out_dir,
        )
    )

    protocol = "other_train__initial_test"
    pred_path = prediction_path(out_dir, protocol, material)
    if args.resume and pred_path.exists():
        pred_df = pd.read_csv(pred_path)
        metrics = evaluate_case_metrics(pred_df["target"], pred_df["prediction"], pred_df)
    else:
        pred_df, metrics = predict_dataframe(
            other_trainer,
            data,
            targets,
            metadata,
            idx["test_initial"],
            other_state,
        )
        pred_df.insert(0, "seed", args.seed)
        pred_df.insert(1, "protocol", protocol)
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(pred_path, index=False)
    rows.append(
        metric_row(
            material,
            model_name,
            run["label"],
            protocol,
            args.seed,
            metrics,
            len(other_train_idx),
            len(other_val_idx),
            len(idx["test_initial"]),
            other_best_val,
        )
    )

    augmented_trainer, augmented_state, augmented_best_val, augmented_train_idx, augmented_val_idx = (
        run_training_group(
            args,
            run,
            data,
            targets,
            metadata,
            material,
            "other_plus_initial_train",
            idx["train_other_plus_initial"],
            out_dir,
        )
    )

    protocol = "other_plus_initial_train__relaxed_test"
    pred_path = prediction_path(out_dir, protocol, material)
    if args.resume and pred_path.exists():
        pred_df = pd.read_csv(pred_path)
        metrics = evaluate_case_metrics(pred_df["target"], pred_df["prediction"], pred_df)
    else:
        pred_df, metrics = predict_dataframe(
            augmented_trainer,
            data,
            targets,
            metadata,
            idx["test_relaxed"],
            augmented_state,
        )
        pred_df.insert(0, "seed", args.seed)
        pred_df.insert(1, "protocol", protocol)
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(pred_path, index=False)
    rows.append(
        metric_row(
            material,
            model_name,
            run["label"],
            protocol,
            args.seed,
            metrics,
            len(augmented_train_idx),
            len(augmented_val_idx),
            len(idx["test_relaxed"]),
            augmented_best_val,
        )
    )

    return rows, None


def write_settings(run_dir, args, materials):
    settings = vars(args).copy()
    settings["materials"] = list(materials)
    settings["run_dir"] = str(run_dir)
    (run_dir / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")


def plot_protocol_mae(summary_df, run_dir):
    if summary_df.empty:
        return []

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = Path(run_dir) / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    outputs = []

    for (model, mode), group in summary_df.groupby(["model", "mode"], sort=False):
        group = group.copy()
        materials = sorted(group["material"].astype(str).unique().tolist())
        x = np.arange(len(materials), dtype=float)
        width = min(0.24, 0.78 / max(len(PROTOCOL_ORDER), 1))
        fig, ax = plt.subplots(figsize=(max(9.0, 0.55 * len(materials) + 3.0), 5.2))

        for p_idx, protocol in enumerate(PROTOCOL_ORDER):
            protocol_df = group[group["protocol"].eq(protocol)].set_index("material")
            values = [
                float(protocol_df.loc[material, "mae"])
                if material in protocol_df.index
                else np.nan
                for material in materials
            ]
            offset = (p_idx - (len(PROTOCOL_ORDER) - 1) / 2.0) * width
            ax.bar(
                x + offset,
                values,
                width,
                label=PROTOCOLS[protocol]["display"],
                color=PROTOCOL_COLORS[protocol],
                edgecolor="black",
                linewidth=0.35,
            )

        ax.set_title(f"{model_mode_display(model, mode)}: leave-one-material-out final DFE")
        ax.set_ylabel("MAE to final relaxed DFE (eV)")
        ax.set_xlabel("Held-out material")
        ax.set_xticks(x)
        ax.set_xticklabels(materials, rotation=45, ha="right")
        ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        output = figure_dir / f"{model}_{mode}_protocol_mae.png"
        fig.savefig(output, dpi=220)
        plt.close(fig)
        outputs.append(output)

    return outputs


def load_prediction_outputs(run_dir):
    run_dir = Path(run_dir)
    prediction_paths = sorted(run_dir.glob("*/*/*/predictions/*/*.csv"))
    if not prediction_paths:
        return pd.DataFrame()

    rows = []
    for path in prediction_paths:
        df = pd.read_csv(path)
        if df.empty:
            continue
        parts = path.relative_to(run_dir).parts
        if len(parts) < 6:
            continue
        df = df.copy()
        df["model"] = parts[0]
        df["mode"] = parts[1]
        df["material"] = parts[2]
        rows.append(df)
    if not rows:
        return pd.DataFrame()

    pred_df = pd.concat(rows, ignore_index=True)
    pred_df["protocol_display"] = pred_df["protocol"].map(
        {key: value["display"] for key, value in PROTOCOLS.items()}
    )
    pred_df["model_mode"] = pred_df.apply(
        lambda row: model_mode_display(row["model"], row["mode"]),
        axis=1,
    )
    return pred_df


def plot_prediction_scatter(run_dir):
    pred_df = load_prediction_outputs(run_dir)
    if pred_df.empty:
        return []

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = run_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    outputs = []

    for (model, mode), group in pred_df.groupby(["model", "mode"], sort=False):
        fig, ax = plt.subplots(figsize=(6.2, 5.6))
        for protocol in PROTOCOL_ORDER:
            p_df = group[group["protocol"].eq(protocol)]
            if p_df.empty:
                continue
            ax.scatter(
                p_df["target"],
                p_df["prediction"],
                s=18,
                alpha=0.68,
                label=PROTOCOLS[protocol]["display"],
                color=PROTOCOL_COLORS[protocol],
                edgecolors="none",
            )
        finite = group[["target", "prediction"]].replace([np.inf, -np.inf], np.nan).dropna()
        if not finite.empty:
            lo = float(finite.min().min())
            hi = float(finite.max().max())
            pad = max(0.2, 0.04 * (hi - lo))
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#111827", linewidth=1)
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(f"{model_mode_display(model, mode)}: predicted vs final DFE")
        ax.set_xlabel("DFT final relaxed DFE (eV)")
        ax.set_ylabel("Predicted DFE (eV)")
        ax.grid(linestyle="--", linewidth=0.6, alpha=0.35)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        output = figure_dir / f"{model}_{mode}_predicted_vs_final_dfe.png"
        fig.savefig(output, dpi=220)
        plt.close(fig)
        outputs.append(output)

    return outputs


def defect_label(row):
    if "defect_label" in row and pd.notna(row["defect_label"]):
        return str(row["defect_label"])
    return str(row["defect_group"])


def select_final_state_rows(pred_df, material):
    material_df = pred_df[pred_df["material"].astype(str).eq(str(material))].copy()
    if material_df.empty:
        raise ValueError(f"No prediction rows found for {material}.")

    optional_cols = [
        col
        for col in [
            "defect_label",
            "configuration",
            "source_path",
            "final_file",
            "final_configuration",
        ]
        if col in material_df.columns
    ]
    candidates = material_df[
        ["material", "defect_group", "target"] + optional_cols
    ].drop_duplicates()
    selected = (
        candidates.sort_values(["defect_group", "target"], ascending=[True, True])
        .groupby("defect_group", sort=False, as_index=False)
        .head(1)
        .copy()
    )
    selected["plot_label"] = selected.apply(defect_label, axis=1)
    return selected.sort_values(["target", "plot_label"], ascending=[True, True]).reset_index(drop=True)


def protocol_label(protocol):
    return PROTOCOLS.get(str(protocol), {}).get("display", str(protocol))


def build_final_state_table(pred_df, material):
    selected = select_final_state_rows(pred_df, material)
    rows = pred_df[pred_df["defect_group"].isin(selected["defect_group"])].copy()
    if rows.empty:
        raise ValueError(f"No selected prediction rows remain for {material}.")

    stats = (
        rows.groupby(["defect_group", "protocol"], sort=False)["prediction"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    stats["std"] = stats["std"].fillna(0.0)

    keep_cols = [
        col
        for col in [
            "plot_label",
            "defect_group",
            "target",
            "final_file",
            "final_configuration",
        ]
        if col in selected.columns
    ]
    plot_table = selected[keep_cols].copy()
    for protocol in PROTOCOL_ORDER:
        protocol_stats = stats[stats["protocol"].eq(protocol)][
            ["defect_group", "mean", "std", "count"]
        ]
        if protocol_stats.empty:
            continue
        label = protocol_label(protocol)
        plot_table = plot_table.merge(protocol_stats, on=["defect_group"], how="left")
        plot_table = plot_table.rename(
            columns={
                "mean": f"{label} prediction",
                "std": f"{label} std",
                "count": f"{label} n",
            }
        )
    return plot_table


def plot_final_state(plot_table, material, model, mode, output_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = plot_table["plot_label"].astype(str).tolist()
    series_labels = [
        col.removesuffix(" prediction")
        for col in plot_table.columns
        if col.endswith(" prediction")
    ]
    x = np.arange(len(labels), dtype=float)
    series_count = 1 + len(series_labels)
    width = min(0.16, 0.82 / max(series_count, 1))
    offsets = (np.arange(series_count) - (series_count - 1) / 2.0) * width
    fig_width = max(8.8, 0.82 * len(labels) + 0.85 * len(series_labels))
    fig, ax = plt.subplots(figsize=(fig_width, 5.4))
    target = plot_table["target"].to_numpy(dtype=float)
    ax.bar(
        x + offsets[0],
        target,
        width,
        label="DFT final relaxed",
        color="#8a8a8a",
        edgecolor="black",
        linewidth=0.45,
    )

    for idx, label in enumerate(series_labels, start=1):
        values = plot_table[f"{label} prediction"].to_numpy(dtype=float)
        std_col = f"{label} std"
        errors = plot_table[std_col].to_numpy(dtype=float) if std_col in plot_table else None
        mae = float(np.nanmean(np.abs(values - target)))
        protocol = next(
            (key for key in PROTOCOL_ORDER if protocol_label(key) == label),
            None,
        )
        color = PROTOCOL_COLORS.get(protocol, "#5aa0c8")
        ax.bar(
            x + offsets[idx],
            values,
            width,
            yerr=errors if errors is not None and np.nanmax(errors) > 0 else None,
            capsize=3,
            label=f"{label} (MAE: {mae:.3f})",
            color=color,
            edgecolor="black",
            linewidth=0.45,
        )

    ax.set_ylabel("Defect Formation Energy (eV)", fontsize=12)
    ax.set_xlabel("Defect Type", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(frameon=True, fontsize=9, loc="best")
    ax.set_title(f"{material}: final relaxed DFE comparison ({model_mode_display(model, mode)})")
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{material}_{model}_{mode}_final_state.png"
    fig.savefig(output, dpi=220)
    plt.close(fig)
    return output


def build_energy_order_table(plot_table):
    rows = []
    dft_df = plot_table[["plot_label", "defect_group", "target"]].copy()
    if "final_file" in plot_table.columns:
        dft_df["file"] = plot_table["final_file"]
    else:
        dft_df["file"] = plot_table["defect_group"]
    dft_df = dft_df.sort_values(["target", "plot_label"], ascending=[True, True]).reset_index(drop=True)
    for order, row in enumerate(dft_df.itertuples(index=False), start=1):
        rows.append(
            {
                "model": "DFT final relaxed",
                "sort_position": order,
                "plot_label": row.plot_label,
                "file": row.file,
                "defect_group": row.defect_group,
                "energy": float(row.target),
            }
        )

    series_labels = [
        col.removesuffix(" prediction")
        for col in plot_table.columns
        if col.endswith(" prediction")
    ]
    for label in series_labels:
        pred_col = f"{label} prediction"
        series_df = dft_df[["plot_label", "file", "defect_group"]].merge(
            plot_table[["defect_group", pred_col]],
            on=["defect_group"],
            how="left",
        )
        for order, (_, row) in enumerate(series_df.iterrows(), start=1):
            rows.append(
                {
                    "model": label,
                    "sort_position": order,
                    "plot_label": row["plot_label"],
                    "file": row["file"],
                    "defect_group": row["defect_group"],
                    "energy": float(row[pred_col]),
                }
            )
    return pd.DataFrame(rows)


def plot_energy_order_comparison(order_table, material, model, mode, output_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = order_table["model"].drop_duplicates().astype(str).tolist()
    n_items = int(order_table["sort_position"].max())
    group_gap = 1.2
    x_ticks = []
    x_labels = []
    fig_width = max(9.5, 0.20 * n_items * len(models) + 1.25 * len(models))
    fig, ax = plt.subplots(figsize=(fig_width, 5.4))

    for model_idx, series in enumerate(models):
        series_df = order_table[order_table["model"].eq(series)].sort_values("sort_position")
        start = model_idx * (n_items + group_gap)
        x = start + np.arange(len(series_df), dtype=float)
        protocol = next(
            (key for key in PROTOCOL_ORDER if protocol_label(key) == series),
            None,
        )
        color = "#8a8a8a" if series == "DFT final relaxed" else PROTOCOL_COLORS.get(protocol, "#5aa0c8")
        ax.bar(
            x,
            series_df["energy"].to_numpy(dtype=float),
            width=0.82,
            color=color,
            edgecolor="black",
            linewidth=0.35,
        )
        x_ticks.append(start + (len(series_df) - 1) / 2.0)
        x_labels.append(series)

    ax.set_ylabel("Defect Formation Energy (eV)", fontsize=12)
    ax.set_xlabel("Protocol (bars ordered by ascending DFT final relaxed DFE)", fontsize=12)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=35, ha="right")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)
    ax.set_title(f"{material}: DFT-ordered final DFE comparison ({model_mode_display(model, mode)})")
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{material}_{model}_{mode}_energy_order_comparison.png"
    fig.savefig(output, dpi=220)
    plt.close(fig)
    return output


def plot_final_state_comparisons(run_dir):
    pred_df = load_prediction_outputs(run_dir)
    if pred_df.empty:
        return []

    output_dir = Path(run_dir) / "figures"
    outputs = []
    for (model, mode, material), group in pred_df.groupby(["model", "mode", "material"], sort=False):
        plot_table = build_final_state_table(group, material)
        table_path = output_dir / f"{material}_{model}_{mode}_final_state.csv"
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_table.to_csv(table_path, index=False)
        outputs.append(plot_final_state(plot_table, material, model, mode, output_dir))

        order_table = build_energy_order_table(plot_table)
        order_path = output_dir / f"{material}_{model}_{mode}_energy_order_comparison.csv"
        order_table.to_csv(order_path, index=False)
        outputs.append(plot_energy_order_comparison(order_table, material, model, mode, output_dir))

    return outputs


def write_summary_markdown(summary_df, skipped_df, run_dir):
    lines = [
        "# Native Initial/Relaxed Leave-One-Out",
        "",
        "| Material | Model | Mode | Protocol | N test | MAE | RMSE | GS MAE | Top-1 |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_df.sort_values(["material", "model", "mode", "protocol"]).itertuples():
        lines.append(
            "| {material} | {model} | {mode} | {protocol} | {n_test} | "
            "{mae:.3f} | {rmse:.3f} | {gs:.3f} | {top1:.3f} |".format(
                material=row.material,
                model=row.model,
                mode=mode_display_name(row.mode),
                protocol=row.protocol_display,
                n_test=row.n_test,
                mae=row.mae,
                rmse=row.rmse,
                gs=row.ground_state_mae,
                top1=row.top1_accuracy,
            )
        )

    if not skipped_df.empty:
        lines.extend(["", "## Skipped Materials", ""])
        for row in skipped_df.itertuples():
            lines.append(f"- {row.material}: {row.reason}")

    (Path(run_dir) / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run native-defect leave-one-material-out tests using initial and "
            "relaxed structures to predict final relaxed DFE."
        )
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--atom-init", default="./HERA/atom_init.json")
    parser.add_argument("--native-csv", default=DEFAULT_NATIVE_CSV)
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--model",
        dest="models",
        nargs="+",
        default=["cgcnn"],
        choices=["cgcnn", "megnet", "definet"],
    )
    parser.add_argument(
        "--mode",
        nargs="+",
        default=["hetero"],
        choices=VALID_MODES,
    )
    parser.add_argument(
        "--materials",
        "--material",
        dest="materials",
        nargs="+",
        default=None,
        help="Optional held-out materials. Defaults to all eligible materials in the native CSV.",
    )
    parser.add_argument(
        "--r",
        nargs="+",
        default=None,
        help=(
            "Radius values for local graph sweep modes; use all for 0 3 4 5 6 7. "
            "Hetero is fixed to r0."
        ),
    )
    args = parser.parse_args()

    if not 0 < args.val_fraction < 1:
        parser.error("--val-fraction must be between 0 and 1.")

    radii = parse_radius_values(args.r, parser)
    init_elem_embedding(args.atom_init)
    set_seed(args.seed)

    run_dir = Path(args.run_dir) if args.run_dir else default_run_dir(args.log_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset_cache = {}
    summary_rows = []
    skipped_rows = []
    material_table_written = False

    for model_name in args.models:
        model_modes = modes_for_model(model_name, args.mode)
        runs = expand_mode_runs(model_name, model_modes, radii)
        for run in runs:
            representation = representation_for_mode(run["mode"])
            cache_key = (model_name, run["local_cutoff"], representation)
            if cache_key not in dataset_cache:
                datasets, raw_targets, raw_metadata = load_native_with_metadata(
                    model_name,
                    args.native_csv,
                    local_cutoff=run["local_cutoff"],
                    representations=[representation],
                )
                metadata = add_final_relaxed_targets(raw_metadata, raw_targets)
                final_targets = torch.tensor(metadata["final_target"].to_numpy(dtype=float)).float()
                dataset_cache[cache_key] = (datasets, final_targets, metadata)
            datasets, targets, metadata = dataset_cache[cache_key]

            all_materials, material_table = eligible_materials(metadata)
            if not material_table_written:
                material_table.to_csv(run_dir / "material_eligibility.csv", index=False)
                material_table_written = True
            materials = args.materials or all_materials
            if not materials:
                raise ValueError("No eligible materials found.")

            data = datasets[dataset_index_for_mode(run["mode"])]

            for material in materials:
                print(f"\n=== {model_name} | {run['label']} | held out {material} ===")
                out_dir = run_dir / model_name / run["label"] / str(material)
                rows, skipped = run_material(
                    args,
                    model_name,
                    run,
                    data,
                    targets,
                    metadata,
                    material,
                    out_dir,
                )
                if skipped is not None:
                    print(f"  Skip {material}: {skipped['reason']}")
                    skipped_rows.append(
                        {
                            "model": model_name,
                            "mode": run["label"],
                            **skipped,
                        }
                    )
                summary_rows.extend(rows)
                pd.DataFrame(summary_rows).to_csv(run_dir / "summary.csv", index=False)
                if skipped_rows:
                    pd.DataFrame(skipped_rows).to_csv(run_dir / "skipped_materials.csv", index=False)

    summary_df = pd.DataFrame(summary_rows)
    skipped_df = pd.DataFrame(skipped_rows)
    summary_df.to_csv(run_dir / "summary.csv", index=False)
    if not skipped_df.empty:
        skipped_df.to_csv(run_dir / "skipped_materials.csv", index=False)
    write_settings(run_dir, args, sorted(set(summary_df["material"])) if not summary_df.empty else [])
    write_summary_markdown(summary_df, skipped_df, run_dir)
    plot_paths = plot_protocol_mae(summary_df, run_dir)
    plot_paths.extend(plot_prediction_scatter(run_dir))
    plot_paths.extend(plot_final_state_comparisons(run_dir))

    print(f"\nSummary written to {run_dir / 'summary.csv'}")
    print(f"Markdown summary written to {run_dir / 'summary.md'}")
    if plot_paths:
        print("Figures written:")
        for path in plot_paths:
            print(f"  {path}")


if __name__ == "__main__":
    main()
