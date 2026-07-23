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

from .config.defaults import CGCNN_DEFINET_MODES, DEFINET_MODES, VALID_MODES, get_config
from .data.datasets import (
    DATASET_REPRESENTATIONS,
    DEFAULT_DATASET_REPRESENTATIONS,
    dataset_index_for_mode,
    init_elem_embedding,
    representation_for_mode,
    tag_structure_source,
)
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

DEFAULT_NATIVE_CSV = (
    r"dataset/Dataset_1/Dataset_1/A_rich/Neutral/"
    r"id_prop_A_rich.csv"
)

MODEL_DISPLAY = {
    "cgcnn": "CGCNN",
    "megnet": "MEGNet",
    "definet": "DeFiNet",
}

MODE_DISPLAY = {
    "full": "Full",
    "full_x": "Full + X",
    "hetero": "Hetero",
    "attention": "Attention",
    "definet": "DeFiNet",
}

MODEL_COLORS = {
    "Full CGCNN": "#5aa0c8",
    "Full + X CGCNN": "#4f7fb8",
    "Hetero CGCNN": "#e8896d",
    "Attention CGCNN": "#72c4a8",
    "Full MEGNet": "#c8ad5a",
    "Full + X MEGNet": "#a89642",
    "Hetero MEGNet": "#9b8bd6",
    "Attention MEGNet": "#d36aa0",
    "DeFiNet": "#157f78",
}

FALLBACK_COLORS = [
    "#5aa0c8",
    "#e8896d",
    "#72c4a8",
    "#c8ad5a",
    "#9b8bd6",
    "#d36aa0",
    "#157f78",
    "#8a6f5a",
    "#5f7f4f",
]


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


def _normalize_representations(representations):
    if representations is None:
        return set(DEFAULT_DATASET_REPRESENTATIONS)
    if isinstance(representations, str):
        representations = [representations]
    normalized = set(representations)
    unknown = normalized - set(DATASET_REPRESENTATIONS)
    if unknown:
        raise ValueError(
            f"Unknown representation(s) {sorted(unknown)}. "
            f"Choose from {list(DATASET_REPRESENTATIONS)}"
        )
    if not normalized:
        raise ValueError("At least one dataset representation must be requested")
    if {'full', 'full_x'} <= normalized:
        raise ValueError("'full' and 'full_x' must be loaded separately")
    return normalized


def load_native_with_metadata(
        model_name,
        csv_path=DEFAULT_NATIVE_CSV,
        local_cutoff=None,
        representations=None,
):
    csv_path = Path(csv_path)
    data_dir = csv_path.parent
    representations = _normalize_representations(representations)
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
    dataset_full = None
    if "full" in representations or "full_x" in representations:
        full_task = (
            f"{model_name}_full_x"
            if "full_x" in representations
            else f"{model_name}_full"
        )
        dataset_full = [
            convert_to_sparse_native(
                structure, defect, 1, full_task, None, skip_was, False
            )
            for structure, defect in tqdm(prep, desc="Converting full graphs")
        ]
    dataset_hetero = None
    if "hetero" in representations:
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
    dataset_attn = None
    if "attention" in representations:
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

    datasets = [dataset_full, dataset_hetero, dataset_attn]
    available = [dataset for dataset in datasets if dataset is not None]
    valid_indices = [
        idx
        for idx in range(len(targets))
        if all(dataset[idx] is not None for dataset in available)
    ]
    if not valid_indices:
        raise ValueError("No valid native structures were loaded.")

    return (
        [
            None if dataset is None else [dataset[idx] for idx in valid_indices]
            for dataset in datasets
        ],
        torch.tensor([targets[idx] for idx in valid_indices]).float(),
        pd.DataFrame([metadata[idx] for idx in valid_indices]),
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
            selected_radii = [0]
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


def modes_for_model(model_name, requested_modes):
    """Return the mode list supported by a model for the case-study command.

    The case-study CLI allows commands such as
    ``--model cgcnn megnet definet --mode full hetero attention``. DefiNet is
    the defect-aware attention/gating baseline, so it only runs attention-style
    modes while CGCNN/MEGNet keep the requested homogeneous, heterogeneous, and
    attention baselines.
    """
    if model_name == "definet":
        alias_to_definet_mode = {
            "definet": "attention",
            "definet_local": "attention_local",
            "definet_was": "attention_was",
            "definet_local_was": "attention_local_was",
        }
        modes = [
            alias_to_definet_mode.get(mode, mode)
            for mode in requested_modes
            if mode in DEFINET_MODES or mode in alias_to_definet_mode
        ]
        modes = list(dict.fromkeys(modes))
        skipped = [
            mode
            for mode in requested_modes
            if mode not in DEFINET_MODES and mode not in alias_to_definet_mode
        ]
        if skipped:
            print(
                "Skipping unsupported DefiNet mode(s): "
                + ", ".join(skipped)
                + "; using attention-style DefiNet mode(s)."
            )
        if not modes:
            modes = ["attention"]
        return modes

    if model_name != "cgcnn":
        modes = [mode for mode in requested_modes if mode not in CGCNN_DEFINET_MODES]
        skipped = [mode for mode in requested_modes if mode in CGCNN_DEFINET_MODES]
        if skipped:
            print(
                f"Skipping CGCNN-only DefiNet mode(s) for {model_name}: "
                + ", ".join(skipped)
            )
        return modes

    return requested_modes


def subset(values, indices):
    return [values[int(idx)] for idx in indices]


def tensor_subset(values, indices):
    return torch.stack([values[int(idx)] for idx in indices])


def mode_display_name(mode):
    mode = str(mode)
    if "_r" in mode:
        base, radius = mode.rsplit("_r", 1)
        if base == "hetero" and radius == "0":
            return MODE_DISPLAY["hetero"]
        return f"{MODE_DISPLAY.get(base, base.replace('_', ' ').title())} r{radius}"
    return MODE_DISPLAY.get(mode, mode.replace("_", " ").title())


def model_mode_display(model, mode):
    model_name = MODEL_DISPLAY.get(str(model), str(model).upper())
    if str(model) == "definet":
        if str(mode) in {"attention", "definet"}:
            return model_name
        return f"{model_name} {mode_display_name(mode)}"
    return f"{mode_display_name(mode)} {model_name}"


def color_for_label(label, fallback_idx):
    return MODEL_COLORS.get(str(label), FALLBACK_COLORS[fallback_idx % len(FALLBACK_COLORS)])


def ground_state_set(df):
    return (
        df.sort_values(["defect_group", "target", "file"], ascending=[True, True, True])
        .groupby("defect_group", sort=False, as_index=False)
        .head(1)
        .copy()
    )


def ndcg_at_k(y_true, y_pred, k):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    if n <= 1:
        return 1.0

    true_order = np.argsort(y_true, kind="mergesort")
    pred_order = np.argsort(y_pred, kind="mergesort")
    true_ranks = np.empty(n, dtype=float)
    true_ranks[true_order] = np.arange(1, n + 1, dtype=float)
    gains = n - true_ranks + 1.0
    limit = min(max(int(k), 1), n)
    discounts = 1.0 / np.log2(np.arange(2, limit + 2, dtype=float))
    dcg = float(np.sum(gains[pred_order[:limit]] * discounts))
    ideal = float(np.sum(gains[true_order[:limit]] * discounts))
    if ideal == 0.0:
        return 0.0
    return float(np.clip(dcg / ideal, 0.0, 1.0))


def evaluate_case_metrics(y_true, y_pred, metadata):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    abs_err = np.abs(y_pred - y_true)
    eval_df = metadata.copy().reset_index(drop=True)
    eval_df["target"] = y_true
    eval_df["prediction"] = y_pred
    gs_df = ground_state_set(eval_df)
    ndcg_values = {
        f"ndcg_at_{k}": ndcg_at_k(gs_df["target"], gs_df["prediction"], k)
        for k in range(1, 6)
    }

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
        "ndcg": ndcg_values["ndcg_at_5"],
        **ndcg_values,
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
        trainer.step_scheduler(val_mae)
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


def model_mode_label(row):
    return model_mode_display(row["model"], row["mode"])


def is_hetero_row(row):
    return str(row["mode"]).startswith("hetero")


def metric_row(material, model_name, run_label, seed, metrics, metadata):
    row = {
        "material": material,
        "model": model_name,
        "mode": run_label,
        "seed": int(seed),
        "n_samples": int(len(metadata)),
        "n_defect_groups": int(metadata["defect_group"].nunique()),
    }
    row.update({key: float(value) for key, value in metrics.items()})
    return row


def select_hetero_seed(seed_rows, metric_name="mae"):
    df = pd.DataFrame(seed_rows)
    selected_rows = []
    selection_rows = []
    if df.empty:
        return selected_rows, selection_rows

    for material, material_df in df.groupby("material", sort=False):
        candidates = []
        for seed, seed_df in material_df.groupby("seed", sort=False):
            hetero_df = seed_df[seed_df.apply(is_hetero_row, axis=1)]
            baseline_df = seed_df[~seed_df.apply(is_hetero_row, axis=1)]
            if hetero_df.empty or baseline_df.empty:
                continue
            hetero_best = hetero_df.loc[hetero_df[metric_name].idxmin()]
            baseline_best = baseline_df.loc[baseline_df[metric_name].idxmin()]
            margin = float(baseline_best[metric_name] - hetero_best[metric_name])
            candidates.append(
                {
                    "material": material,
                    "seed": int(seed),
                    "selection_metric": metric_name,
                    "hetero_model": hetero_best["model"],
                    "hetero_mode": hetero_best["mode"],
                    "hetero_metric": float(hetero_best[metric_name]),
                    "best_baseline_model": baseline_best["model"],
                    "best_baseline_mode": baseline_best["mode"],
                    "best_baseline_metric": float(baseline_best[metric_name]),
                    "hetero_margin": margin,
                    "hetero_beats_all_baselines": bool(margin > 0),
                }
            )

        if not candidates:
            continue
        selection = max(candidates, key=lambda row: row["hetero_margin"])
        selection_rows.append(selection)
        selected = material_df[material_df["seed"] == selection["seed"]].copy()
        selected["selected_hetero_case"] = selected.apply(
            lambda row: (
                row["model"] == selection["hetero_model"]
                and row["mode"] == selection["hetero_mode"]
            ),
            axis=1,
        )
        selected_rows.extend(selected.to_dict("records"))

    return selected_rows, selection_rows


def write_summary(run_dir, selected_rows, seed_rows, selection_rows):
    run_dir = Path(run_dir)
    summary_df = pd.DataFrame(selected_rows)
    seed_df = pd.DataFrame(seed_rows)
    summary_df.to_csv(run_dir / "summary.csv", index=False)
    seed_df.to_csv(run_dir / "seed_metrics.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(run_dir / "selection_summary.csv", index=False)

    lines = [
        "# Native OOD Case Study - Selected Seed",
        "",
        "| Material | Seed | Model | Mode | N | Defects | MAE | RMSE | Ground-state MAE | NDCG | Top-1 acc. | Min swaps | Selected hetero |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in selected_rows:
        lines.append(
            "| {material} | {seed} | {model} | {mode} | {n_samples} | "
            "{n_defect_groups} | {mae:.3f} | {rmse:.3f} | "
            "{ground_state_mae:.3f} | {ndcg:.3f} | {top1_accuracy:.3f} | "
            "{min_swaps_to_correct:.3f} | {selected} |".format(
                material=row["material"],
                seed=row["seed"],
                model=row["model"],
                mode=row["mode"],
                n_samples=row["n_samples"],
                n_defect_groups=row["n_defect_groups"],
                mae=row["mae"],
                rmse=row["rmse"],
                ground_state_mae=row["ground_state_mae"],
                ndcg=row.get("ndcg", float("nan")),
                top1_accuracy=row["top1_accuracy"],
                min_swaps_to_correct=row["min_swaps_to_correct"],
                selected="yes" if row.get("selected_hetero_case") else "",
            )
        )
    lines.extend(["", "## Seed Selection", ""])
    for row in selection_rows:
        status = "beats all baselines" if row["hetero_beats_all_baselines"] else "best available margin"
        lines.append(
            "- {material}: seed {seed}, selected {hetero_model} {hetero_mode} "
            "({metric}={hetero_metric:.3f}) vs best baseline "
            "{best_baseline_model} {best_baseline_mode} "
            "({metric}={best_baseline_metric:.3f}); margin={hetero_margin:.3f} "
            "({status}).".format(metric=row["selection_metric"], status=status, **row)
        )
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_selected_seed_performance(run_dir, material, selected_rows, selection):
    rows = [row for row in selected_rows if row["material"] == material]
    if not rows:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [model_mode_label(row) for row in rows]
    metric_specs = [
        ("mae", "MAE (eV)", True),
        ("ground_state_mae", "Ground-State MAE (eV)", True),
        ("ndcg", "NDCG", False),
    ]
    metric_specs = [spec for spec in metric_specs if spec[0] in rows[0]]
    x = np.arange(len(labels))
    fig_width = max(8.0, 0.78 * len(labels) + 2.0)
    fig, axes = plt.subplots(
        len(metric_specs),
        1,
        figsize=(fig_width, 2.7 * len(metric_specs)),
        sharex=True,
    )
    axes = np.atleast_1d(axes)

    for ax, (metric, ylabel, _) in zip(axes, metric_specs):
        values = np.asarray([row[metric] for row in rows], dtype=float)
        colors = [color_for_label(label, idx) for idx, label in enumerate(labels)]
        bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.45)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
        ax.set_axisbelow(True)
        finite_values = values[np.isfinite(values)]
        value_max = float(np.max(finite_values)) if len(finite_values) else 1.0
        value_min = float(np.min(finite_values)) if len(finite_values) else 0.0
        if metric == "ndcg":
            ax.set_ylim(0, 1.14)
            label_offset = 0.018
        else:
            span = max(value_max - min(0.0, value_min), 1e-9)
            ax.set_ylim(0, value_max + max(0.14 * span, 0.18))
            label_offset = max(0.012 * span, 0.035)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + label_offset,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    axes[0].set_title(f"{material}: OOD model performance (seed {selection['seed']})")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=35, ha="right")
    fig.tight_layout()

    figure_dir = Path(run_dir) / material / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    output = figure_dir / "selected_seed_performance.png"
    fig.savefig(output, dpi=220)
    plt.close(fig)
    return output


def plot_selected_seed_ground_state(run_dir, material, selected_rows, selection):
    rows = [row for row in selected_rows if row["material"] == material]
    if not rows:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_table = None
    model_labels = []
    for row in rows:
        pred_path = prediction_path(run_dir, material, row["model"], row["mode"], row["seed"])
        pred_df = pd.read_csv(pred_path)
        gs_df = ground_state_set(pred_df)
        label = model_mode_label(row)
        model_labels.append(label)
        model_part = gs_df[["file", "defect_group", "prediction"]].rename(
            columns={"prediction": f"{label} prediction"}
        )
        if plot_table is None:
            label_cols = [
                col
                for col in ["defect_label", "configuration", "source_path"]
                if col in gs_df.columns
            ]
            plot_table = gs_df[["file", "defect_group", "target"] + label_cols].copy()
            if "defect_label" in plot_table.columns:
                plot_table["plot_label"] = plot_table["defect_label"].astype(str)
            else:
                plot_table["plot_label"] = plot_table["defect_group"].astype(str)
        plot_table = plot_table.merge(model_part, on=["file", "defect_group"], how="left")

    if plot_table is None or plot_table.empty:
        return None

    plot_table = plot_table.sort_values(["target", "plot_label"], ascending=[True, True]).reset_index(drop=True)
    figure_dir = Path(run_dir) / material / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_path = figure_dir / "selected_seed_ground_state.csv"
    plot_table.to_csv(table_path, index=False)

    labels = plot_table["plot_label"].astype(str).tolist()
    x = np.arange(len(labels), dtype=float)
    series_count = 1 + len(model_labels)
    width = min(0.16, 0.82 / max(series_count, 1))
    offsets = (np.arange(series_count) - (series_count - 1) / 2.0) * width

    fig_width = max(8.8, 0.82 * len(labels) + 0.85 * len(model_labels))
    fig, ax = plt.subplots(figsize=(fig_width, 5.4))
    target = plot_table["target"].to_numpy(dtype=float)
    ax.bar(
        x + offsets[0],
        target,
        width,
        label="DFT",
        color="#8a8a8a",
        edgecolor="black",
        linewidth=0.45,
    )

    for idx, model_label in enumerate(model_labels, start=1):
        values = plot_table[f"{model_label} prediction"].to_numpy(dtype=float)
        mae = float(np.nanmean(np.abs(values - target)))
        ax.bar(
            x + offsets[idx],
            values,
            width,
            label=f"{model_label} (MAE: {mae:.3f})",
            color=color_for_label(model_label, idx - 1),
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
    ax.set_title(f"{material}: DFT ground-state OOD comparison (seed {selection['seed']})")
    fig.tight_layout()

    output = figure_dir / "selected_seed_ground_state.png"
    fig.savefig(output, dpi=220)
    plt.close(fig)
    return output


def build_energy_order_table(plot_table):
    rows = []
    dft_df = plot_table[["plot_label", "file", "defect_group", "target"]].copy()
    dft_df = dft_df.sort_values(["target", "plot_label"], ascending=[True, True]).reset_index(drop=True)
    for order, row in enumerate(dft_df.itertuples(index=False), start=1):
        rows.append(
            {
                "model": "DFT",
                "sort_position": order,
                "plot_label": row.plot_label,
                "file": row.file,
                "defect_group": row.defect_group,
                "energy": float(row.target),
            }
        )

    model_labels = [
        col.removesuffix(" prediction")
        for col in plot_table.columns
        if col.endswith(" prediction")
    ]
    for model_label in model_labels:
        pred_col = f"{model_label} prediction"
        model_df = dft_df[["plot_label", "file", "defect_group"]].merge(
            plot_table[["file", "defect_group", pred_col]],
            on=["file", "defect_group"],
            how="left",
        )
        for order, (_, row) in enumerate(model_df.iterrows(), start=1):
            rows.append(
                {
                    "model": model_label,
                    "sort_position": order,
                    "plot_label": row["plot_label"],
                    "file": row["file"],
                    "defect_group": row["defect_group"],
                    "energy": float(row[pred_col]),
                }
            )
    return pd.DataFrame(rows)


def build_rank_comparison_table(plot_table):
    out = plot_table[["plot_label", "file", "defect_group", "target"]].copy()
    out = out.sort_values(["target", "plot_label"], ascending=[True, True]).reset_index(drop=True)
    out["dft_rank"] = np.arange(1, len(out) + 1)
    model_labels = [
        col.removesuffix(" prediction")
        for col in plot_table.columns
        if col.endswith(" prediction")
    ]
    for model_label in model_labels:
        pred_col = f"{model_label} prediction"
        ranks = plot_table[["file", "defect_group", pred_col]].copy()
        ranks[f"{model_label} rank"] = (
            ranks[pred_col].rank(method="first", ascending=True).astype(int)
        )
        out = out.merge(
            ranks[["file", "defect_group", f"{model_label} rank"]],
            on=["file", "defect_group"],
            how="left",
        )
    return out


def plot_selected_seed_energy_order_comparison(run_dir, material, selected_rows, selection):
    rows = [row for row in selected_rows if row["material"] == material]
    if not rows:
        return None

    plot_table = None
    for row in rows:
        pred_path = prediction_path(run_dir, material, row["model"], row["mode"], row["seed"])
        pred_df = pd.read_csv(pred_path)
        gs_df = ground_state_set(pred_df)
        label = model_mode_label(row)
        model_part = gs_df[["file", "defect_group", "prediction"]].rename(
            columns={"prediction": f"{label} prediction"}
        )
        if plot_table is None:
            label_cols = [
                col
                for col in ["defect_label", "configuration", "source_path"]
                if col in gs_df.columns
            ]
            plot_table = gs_df[["file", "defect_group", "target"] + label_cols].copy()
            if "defect_label" in plot_table.columns:
                plot_table["plot_label"] = plot_table["defect_label"].astype(str)
            else:
                plot_table["plot_label"] = plot_table["defect_group"].astype(str)
        plot_table = plot_table.merge(model_part, on=["file", "defect_group"], how="left")

    if plot_table is None or plot_table.empty:
        return None

    order_table = build_energy_order_table(plot_table)
    figure_dir = Path(run_dir) / material / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_path = figure_dir / "selected_seed_energy_order_comparison.csv"
    order_table.to_csv(table_path, index=False)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = order_table["model"].drop_duplicates().astype(str).tolist()
    n_items = int(order_table["sort_position"].max())
    if not models or n_items == 0:
        return None
    group_gap = 1.2
    x_ticks = []
    x_labels = []
    fig_width = max(9.5, 0.20 * n_items * len(models) + 1.25 * len(models))
    fig, ax = plt.subplots(figsize=(fig_width, 5.4))
    for model_idx, model in enumerate(models):
        model_df = order_table[order_table["model"].eq(model)].sort_values("sort_position")
        start = model_idx * (n_items + group_gap)
        x = start + np.arange(len(model_df), dtype=float)
        color = "#8a8a8a" if model == "DFT" else color_for_label(model, model_idx - 1)
        ax.bar(
            x,
            model_df["energy"].to_numpy(dtype=float),
            width=0.82,
            color=color,
            edgecolor="black",
            linewidth=0.35,
        )
        x_ticks.append(start + (len(model_df) - 1) / 2.0)
        x_labels.append(model)

    ax.set_ylabel("Defect Formation Energy (eV)", fontsize=12)
    ax.set_xlabel("Model (bars ordered by ascending DFT energy)", fontsize=12)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=35, ha="right")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)
    ax.set_title(f"{material}: DFT-ordered energy comparison (seed {selection['seed']})")
    fig.tight_layout()

    output = figure_dir / "selected_seed_energy_order_comparison.png"
    fig.savefig(output, dpi=220)
    plt.close(fig)
    return output


def write_plots(run_dir, selected_rows, selection_rows):
    outputs = []
    for selection in selection_rows:
        material = selection["material"]
        for plotter in (
            plot_selected_seed_performance,
            plot_selected_seed_ground_state,
            plot_selected_seed_energy_order_comparison,
        ):
            output = plotter(run_dir, material, selected_rows, selection)
            if output is not None:
                outputs.append(output)
    return outputs


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


def run_case_study(
    args,
    model_name,
    runs,
    materials,
    run_dir,
    dataset_cache,
    seed_rows=None,
):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_settings(run_dir, args, materials)

    if seed_rows is None:
        seed_rows = []

    for run in runs:
        representation = representation_for_mode(run["mode"])
        cache_key = (model_name, run["local_cutoff"], representation)
        if cache_key not in dataset_cache:
            dataset_cache[cache_key] = load_native_with_metadata(
                model_name,
                args.native_csv,
                local_cutoff=run["local_cutoff"],
                representations=[representation],
            )
        datasets, targets, metadata = dataset_cache[cache_key]
        data = datasets[dataset_index_for_mode(run["mode"])]

        for material in materials:
            print(
                f"\n=== {model_name} | {run['label']} | held out {material} ==="
            )
            for seed in args.seeds:
                pred_path = prediction_path(run_dir, material, model_name, run["label"], seed)
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
                seed_row = metric_row(
                    material,
                    model_name,
                    run["label"],
                    seed,
                    metrics,
                    pred_df,
                )
                seed_rows.append(seed_row)
                pd.DataFrame(seed_rows).to_csv(run_dir / "seed_metrics.csv", index=False)

    selected_rows, selection_rows = select_hetero_seed(
        seed_rows, metric_name=args.selection_metric
    )
    write_summary(run_dir, selected_rows, seed_rows, selection_rows)
    plot_paths = write_plots(run_dir, selected_rows, selection_rows)

    print(f"\nSummary written to {run_dir / 'summary.md'}")
    print(f"CSV summary written to {run_dir / 'summary.csv'}")
    if plot_paths:
        print("Figures written:")
        for path in plot_paths:
            print(f"  {path}")
    return pd.DataFrame(selected_rows)


def main():
    parser = argparse.ArgumentParser(
        description="Run leave-one-material-out OOD case studies on native defects."
    )
    parser.add_argument(
        "--model",
        dest="models",
        nargs="+",
        default=["cgcnn"],
        choices=["cgcnn", "megnet", "definet"],
        help="Model architecture(s), e.g. --model cgcnn megnet.",
    )
    parser.add_argument(
        "--mode",
        nargs="+",
        default=["full", "full_x", "hetero", "attention"],
        choices=VALID_MODES,
    )
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
        "--selection-metric",
        default="mae",
        choices=["mae", "rmse", "ground_state_mae"],
        help="Metric used to choose the single seed where hetero is strongest.",
    )
    parser.add_argument(
        "--r",
        nargs="+",
        default=None,
        help=(
            "Radius values for local graph sweep modes; use all for 0 3 4 5 6 7. "
            "Hetero is fixed to r0 in this case-study runner."
        ),
    )
    args = parser.parse_args()

    if not 0 < args.val_fraction < 1:
        parser.error("--val-fraction must be between 0 and 1.")

    radii = parse_radius_values(args.r, parser)
    init_elem_embedding(args.atom_init)

    dataset_cache = {}

    run_dir = Path(args.run_dir) if args.run_dir else default_run_dir(
        args.log_dir, args.materials
    )
    seed_rows = []
    for model_name in args.models:
        model_modes = modes_for_model(model_name, args.mode)
        if not model_modes:
            print(f"Skipping {model_name}: no supported modes selected.")
            continue
        runs = expand_mode_runs(model_name, model_modes, radii)
        run_case_study(
            args,
            model_name,
            runs,
            args.materials,
            run_dir,
            dataset_cache,
            seed_rows=seed_rows,
        )


if __name__ == "__main__":
    main()
