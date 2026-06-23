#!/usr/bin/env python
"""Plot zero-shot native-defect model performance from completed runs.

This script reads outputs produced by ``native_poscar0_finetune.py`` without
using any fine-tuned predictions:

    <run-dir>/comparison.csv
    <run-dir>/summary.csv
    <run-dir>/<model>/<mode>/predictions/*_zero_shot_all.csv
    <run-dir>/<model>/<mode>/predictions/*_zero_shot_non_poscar0.csv

It writes two kinds of figures:

1. Zero-shot performance comparison across models/modes (MAE, ground-state MAE,
   and rank NDCG across the per-defect ground-state set).
2. DFT ground-state bar plots: for each defect group, keep the lowest-energy
   DFT configuration, then compare DFT with each model/mode zero-shot
   prediction.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_DISPLAY = {
    "cgcnn": "CGCNN",
    "megnet": "MEGNet",
    "definet": "DeFiNet",
}

MODE_DISPLAY = {
    "full": "Full",
    "hetero": "Hetero",
    "attention": "Attention",
    "definet": "DeFiNet",
}

MODEL_COLORS = {
    "Full CGCNN": "#5aa0c8",
    "Hetero CGCNN": "#e8896d",
    "Attention CGCNN": "#72c4a8",
    "Full MEGNet": "#c8ad5a",
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


def mode_display_name(mode: str) -> str:
    mode = str(mode)
    if "_r" in mode:
        base, radius = mode.rsplit("_r", 1)
        if base == "hetero" and radius == "0":
            return MODE_DISPLAY["hetero"]
        return f"{MODE_DISPLAY.get(base, base.replace('_', ' ').title())} r{radius}"
    return MODE_DISPLAY.get(mode, mode.replace("_", " ").title())


def model_mode_display(model: str, mode: str) -> str:
    model_name = MODEL_DISPLAY.get(str(model), str(model).upper())
    if str(model) == "definet":
        if str(mode) in {"attention", "definet"}:
            return model_name
        return f"{model_name} {mode_display_name(mode)}"
    return f"{mode_display_name(mode)} {model_name}"


def color_for_label(label: str, fallback_idx: int) -> str:
    return MODEL_COLORS.get(str(label), FALLBACK_COLORS[fallback_idx % len(FALLBACK_COLORS)])


def model_mode_from_path(path: Path) -> tuple[str, str]:
    try:
        mode = path.parents[1].name
        model = path.parents[2].name
    except IndexError as exc:
        raise ValueError(f"Unexpected prediction path: {path}") from exc
    return model, mode


def add_model_columns(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    model, mode = model_mode_from_path(path)
    out = df.copy()
    out["model"] = model
    out["mode"] = mode
    out["model_mode"] = out.apply(
        lambda row: model_mode_display(row["model"], row["mode"]),
        axis=1,
    )
    out["source_csv"] = str(path)
    return out


def discover_prediction_tables(run_dir: Path, split: str) -> list[Path]:
    pattern = f"*/*/predictions/*_zero_shot_{split}.csv"
    return sorted(run_dir.glob(pattern))


def load_predictions(run_dir: Path, split: str) -> pd.DataFrame:
    rows = []
    for path in discover_prediction_tables(run_dir, split):
        df = pd.read_csv(path)
        if df.empty:
            continue
        if "prediction" not in df.columns:
            continue
        rows.append(add_model_columns(df, path))
    if not rows:
        raise FileNotFoundError(
            f"No zero-shot {split} prediction tables found under {run_dir}."
        )
    out = pd.concat(rows, ignore_index=True)
    required = {"file", "material", "defect_group", "target", "prediction", "model", "mode"}
    missing = sorted(required - set(out.columns))
    if missing:
        raise ValueError(f"Missing required prediction column(s): {', '.join(missing)}")
    return out


def evaluate_metrics(df: pd.DataFrame) -> dict[str, float]:
    y_true = df["target"].to_numpy(dtype=float)
    y_pred = df["prediction"].to_numpy(dtype=float)
    abs_err = np.abs(y_pred - y_true)

    gs_errors = []
    top1_hits = []
    swaps = []
    for _, group in df.groupby("defect_group", sort=False):
        idx = group.index.to_numpy()
        true_group = df.loc[idx, "target"].to_numpy(dtype=float)
        pred_group = df.loc[idx, "prediction"].to_numpy(dtype=float)
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
        "n_samples": int(len(df)),
        "n_defect_groups": int(df["defect_group"].nunique()),
    }


def ndcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """Rank-only NDCG@k with log discount and descending DFT-rank gains."""
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


def ground_state_set(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the lowest-DFT row from each defect group."""
    return (
        df.sort_values(["defect_group", "target", "file"], ascending=[True, True, True])
        .groupby("defect_group", sort=False, as_index=False)
        .head(1)
        .copy()
    )


def add_rank_ndcg(perf_df: pd.DataFrame, run_dir: Path) -> pd.DataFrame:
    """Merge rank NDCG@1..5 across the per-defect DFT ground-state set."""
    try:
        pred_df = load_predictions(run_dir, "all")
    except FileNotFoundError:
        return perf_df

    rows = []
    for (material, model, mode), group in pred_df.groupby(["material", "model", "mode"], sort=False):
        gs_df = ground_state_set(group)
        y_true = gs_df["target"].to_numpy(dtype=float)
        y_pred = gs_df["prediction"].to_numpy(dtype=float)
        ndcg_values = {
            f"ndcg_at_{k}": ndcg_at_k(y_true, y_pred, k)
            for k in range(1, 6)
        }
        rows.append(
            {
                "material": material,
                "model": model,
                "mode": mode,
                "ndcg": ndcg_values["ndcg_at_5"],
                **ndcg_values,
            }
        )

    ndcg_df = pd.DataFrame(rows)
    if ndcg_df.empty:
        return perf_df
    perf_df = perf_df.drop(
        columns=[
            "ndcg",
            "ndcg_at_1",
            "ndcg_at_2",
            "ndcg_at_3",
            "ndcg_at_4",
            "ndcg_at_5",
            "kl_divergence",
            "mrr",
        ],
        errors="ignore",
    )
    return perf_df.merge(ndcg_df, on=["material", "model", "mode"], how="left")


def load_performance_table(run_dir: Path) -> pd.DataFrame:
    summary_path = run_dir / "summary.csv"
    if summary_path.exists():
        df = pd.read_csv(summary_path)
        if not df.empty and "phase" in df.columns:
            out = df[df["phase"].eq("zero_shot_all")].copy()
            if not out.empty:
                out["model_mode"] = out.apply(
                    lambda row: model_mode_display(row["model"], row["mode"]),
                    axis=1,
                )
                return out

    try:
        pred_df = load_predictions(run_dir, "all")
    except FileNotFoundError:
        pred_df = None
    if pred_df is not None:
        rows = []
        for (material, model, mode), group in pred_df.groupby(["material", "model", "mode"], sort=False):
            metrics = evaluate_metrics(group)
            rows.append(
                {
                    "material": material,
                    "model": model,
                    "mode": mode,
                    "model_mode": model_mode_display(model, mode),
                    **metrics,
                }
            )
        return pd.DataFrame(rows)

    comparison_path = run_dir / "comparison.csv"
    if comparison_path.exists():
        df = pd.read_csv(comparison_path)
        if not df.empty and {"material", "model", "mode", "zero_shot_mae"}.issubset(df.columns):
            rename = {
                "zero_shot_mae": "mae",
                "zero_shot_ground_state_mae": "ground_state_mae",
                "zero_shot_top1_accuracy": "top1_accuracy",
            }
            keep = [
                "material",
                "model",
                "mode",
                "seed",
                "n_train_base",
                "n_finetune_poscar0",
                "n_test_non_poscar0",
                *rename.keys(),
            ]
            out = df[[col for col in keep if col in df.columns]].copy()
            out = out.rename(columns={key: value for key, value in rename.items() if key in out.columns})
            out["model_mode"] = out.apply(
                lambda row: model_mode_display(row["model"], row["mode"]),
                axis=1,
            )
            return out

    return pd.DataFrame()


def filter_materials(df: pd.DataFrame, materials: list[str] | None) -> pd.DataFrame:
    if not materials:
        return df
    out = df[df["material"].astype(str).isin([str(material) for material in materials])].copy()
    if out.empty:
        raise ValueError(f"No rows found for requested material(s): {', '.join(materials)}")
    return out


def filter_labels(df: pd.DataFrame, include_labels: set[str] | None) -> pd.DataFrame:
    if include_labels is None:
        return df
    out = df[df["model_mode"].isin(include_labels)].copy()
    if out.empty:
        raise ValueError("No rows remain after --include filtering.")
    return out


def plot_performance(perf_df: pd.DataFrame, output_dir: Path, dpi: int) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric_specs = [
        ("mae", "MAE (eV)", True),
        ("ground_state_mae", "Ground-State MAE (eV)", True),
        ("ndcg", "NDCG", False),
    ]
    metric_specs = [spec for spec in metric_specs if spec[0] in perf_df.columns]
    outputs = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for material, material_df in perf_df.groupby("material", sort=False):
        labels = material_df["model_mode"].astype(str).tolist()
        x = np.arange(len(labels))
        fig_width = max(8.0, 0.78 * len(labels) + 2.0)
        fig, axes = plt.subplots(
            len(metric_specs),
            1,
            figsize=(fig_width, 2.7 * len(metric_specs)),
            sharex=True,
        )
        axes = np.atleast_1d(axes)

        for ax, (metric, ylabel, lower_is_better) in zip(axes, metric_specs):
            values = material_df[metric].to_numpy(dtype=float)
            bar_colors = [color_for_label(label, idx) for idx, label in enumerate(labels)]
            bars = ax.bar(x, values, color=bar_colors, edgecolor="black", linewidth=0.45)
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
            if metric == "top1_accuracy":
                ax.set_ylim(0, max(1.0, float(np.nanmax(values)) * 1.12))

        axes[0].set_title(f"{material}: zero-shot model performance")
        axes[-1].set_xticks(x)
        axes[-1].set_xticklabels(labels, rotation=35, ha="right")
        fig.tight_layout()
        output = output_dir / f"{material}_zero_shot_performance.png"
        fig.savefig(output, dpi=dpi)
        plt.close(fig)
        outputs.append(output)

    return outputs


def defect_label(row: pd.Series) -> str:
    if "defect_label" in row and pd.notna(row["defect_label"]):
        return str(row["defect_label"])
    return str(row["defect_group"])


def select_ground_state_rows(pred_df: pd.DataFrame, material: str) -> pd.DataFrame:
    material_df = pred_df[pred_df["material"].astype(str).eq(str(material))].copy()
    if material_df.empty:
        raise ValueError(f"No zero-shot prediction rows found for {material}.")

    optional_cols = [
        col
        for col in ["defect_label", "configuration", "source_path"]
        if col in material_df.columns
    ]
    candidates = material_df[["file", "material", "defect_group", "target"] + optional_cols]
    candidates = candidates.drop_duplicates()
    selected = (
        candidates.sort_values(["defect_group", "target", "file"], ascending=[True, True, True])
        .groupby("defect_group", sort=False, as_index=False)
        .head(1)
        .copy()
    )
    selected["plot_label"] = selected.apply(defect_label, axis=1)
    return selected.sort_values(["target", "plot_label"], ascending=[True, True]).reset_index(drop=True)


def build_ground_state_table(
    pred_df: pd.DataFrame,
    material: str,
    include_labels: set[str] | None,
) -> pd.DataFrame:
    selected = select_ground_state_rows(pred_df, material)
    rows = pred_df[pred_df["file"].isin(selected["file"])].copy()
    if include_labels is not None:
        rows = rows[rows["model_mode"].isin(include_labels)]
    if rows.empty:
        raise ValueError(f"No selected zero-shot rows remain for {material}.")

    stats = (
        rows.groupby(["file", "defect_group", "model_mode"], sort=False)["prediction"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    stats["std"] = stats["std"].fillna(0.0)

    plot_table = selected[["plot_label", "file", "defect_group", "target"]].copy()
    for label in stats["model_mode"].drop_duplicates():
        label_stats = stats[stats["model_mode"].eq(label)][
            ["file", "defect_group", "mean", "std", "count"]
        ]
        plot_table = plot_table.merge(label_stats, on=["file", "defect_group"], how="left")
        plot_table = plot_table.rename(
            columns={
                "mean": f"{label} prediction",
                "std": f"{label} std",
                "count": f"{label} n",
            }
        )
    return plot_table


def plot_ground_state(plot_table: pd.DataFrame, material: str, output_dir: Path, dpi: int) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = plot_table["plot_label"].astype(str).tolist()
    model_labels = [
        col.removesuffix(" prediction")
        for col in plot_table.columns
        if col.endswith(" prediction")
    ]
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
        std_col = f"{model_label} std"
        errors = plot_table[std_col].to_numpy(dtype=float) if std_col in plot_table else None
        mae = float(np.nanmean(np.abs(values - target)))
        ax.bar(
            x + offsets[idx],
            values,
            width,
            yerr=errors if errors is not None and np.nanmax(errors) > 0 else None,
            capsize=3,
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
    ax.set_title(f"{material}: DFT ground-state zero-shot comparison")
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{material}_zero_shot_ground_state.png"
    fig.savefig(output, dpi=dpi)
    plt.close(fig)
    return output


def build_energy_order_table(plot_table: pd.DataFrame) -> pd.DataFrame:
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


def plot_energy_order_comparison(order_table: pd.DataFrame, material: str, output_dir: Path, dpi: int) -> Path:
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
    ax.set_title(f"{material}: DFT-ordered energy comparison")
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{material}_zero_shot_energy_order_comparison.png"
    fig.savefig(output, dpi=dpi)
    plt.close(fig)
    return output


def build_rank_comparison_table(plot_table: pd.DataFrame) -> pd.DataFrame:
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


def plot_rank_comparison(rank_table: pd.DataFrame, material: str, output_dir: Path, dpi: int) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = rank_table["plot_label"].astype(str).tolist()
    x = np.arange(len(labels), dtype=float)
    model_labels = [
        col.removesuffix(" rank")
        for col in rank_table.columns
        if col.endswith(" rank") and col != "dft_rank"
    ]
    if not model_labels:
        raise ValueError("No model rank columns found for rank comparison plot.")

    fig_width = max(8.8, 0.72 * len(labels) + 1.2)
    fig, ax = plt.subplots(figsize=(fig_width, 5.2))
    width = min(0.12, 0.82 / max(len(model_labels), 1))
    offsets = (np.arange(len(model_labels)) - (len(model_labels) - 1) / 2.0) * width
    for idx, model_label in enumerate(model_labels):
        ax.bar(
            x + offsets[idx],
            rank_table[f"{model_label} rank"],
            width,
            label=model_label,
            color=color_for_label(model_label, idx),
            edgecolor="black",
            linewidth=0.45,
        )

    ax.set_xlabel("Defect Type (sorted by DFT energy)", fontsize=12)
    ax.set_ylabel("Rank by Energy (1 = lowest)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(1, len(labels) + 1))
    ax.set_ylim(0, len(labels) + 0.8)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(frameon=True, fontsize=9, loc="best", ncol=2)
    ax.set_title(f"{material}: DFT-sorted rank comparison")
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{material}_zero_shot_rank_comparison.png"
    fig.savefig(output, dpi=dpi)
    plt.close(fig)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot zero-shot model performance and DFT ground-state comparisons."
    )
    parser.add_argument("--run-dir", required=True, help="native_poscar0_finetune output directory.")
    parser.add_argument(
        "--material",
        nargs="+",
        default=None,
        help="Material(s) to plot. Defaults to all materials found in zero-shot outputs.",
    )
    parser.add_argument(
        "--include",
        nargs="+",
        default=None,
        help="Optional display labels to include, e.g. 'Full CGCNN' 'Hetero CGCNN'.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to <run-dir>/figures_zero_shot.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "figures_zero_shot"
    include_labels = set(args.include) if args.include else None

    perf_df = load_performance_table(run_dir)
    perf_df = add_rank_ndcg(perf_df, run_dir)
    perf_df = filter_materials(perf_df, args.material)
    perf_df = filter_labels(perf_df, include_labels)
    output_dir.mkdir(parents=True, exist_ok=True)
    perf_csv = output_dir / "zero_shot_performance.csv"
    perf_df.to_csv(perf_csv, index=False)
    performance_figures = plot_performance(perf_df, output_dir, args.dpi)

    all_pred_df = load_predictions(run_dir, "all")
    all_pred_df = filter_materials(all_pred_df, args.material)
    all_pred_df = filter_labels(all_pred_df, include_labels)
    ground_state_outputs = []
    energy_order_outputs = []
    for material in sorted(all_pred_df["material"].astype(str).unique()):
        plot_table = build_ground_state_table(all_pred_df, material, include_labels)
        table_path = output_dir / f"{material}_zero_shot_ground_state.csv"
        plot_table.to_csv(table_path, index=False)
        figure_path = plot_ground_state(plot_table, material, output_dir, args.dpi)
        ground_state_outputs.append((figure_path, table_path))
        energy_order_table = build_energy_order_table(plot_table)
        energy_order_table_path = output_dir / f"{material}_zero_shot_energy_order_comparison.csv"
        energy_order_table.to_csv(energy_order_table_path, index=False)
        energy_order_figure_path = plot_energy_order_comparison(
            energy_order_table, material, output_dir, args.dpi
        )
        energy_order_outputs.append((energy_order_figure_path, energy_order_table_path))

    print("Wrote:")
    print(f"  performance table: {perf_csv}")
    for path in performance_figures:
        print(f"  performance figure: {path}")
    for figure_path, table_path in ground_state_outputs:
        print(f"  ground-state figure: {figure_path}")
        print(f"  ground-state table:  {table_path}")
    for figure_path, table_path in energy_order_outputs:
        print(f"  energy-order figure: {figure_path}")
        print(f"  energy-order table:  {table_path}")


if __name__ == "__main__":
    main()
