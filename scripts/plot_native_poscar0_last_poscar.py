#!/usr/bin/env python
"""Plot DFT-ranked final POSCAR configurations for native fine-tuning runs.

The script reads outputs produced by ``native_poscar0_finetune.py``:

    <run-dir>/<model>/<mode>/predictions/*_finetuned_non_poscar0.csv
    <run-dir>/<model>/<mode>/poscar0_values/*_poscar0_values.csv

For each material and defect group, it sorts available POSCAR configurations by
DFT target energy from low to high and keeps the last row. The selected DFT
values are then used as the x-axis order and compared with each model/mode
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

MATERIAL_KEY = "_plot_material"


def mode_display_name(mode: str) -> str:
    """Return a compact display name while preserving radius suffixes."""
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


def discover_poscar0_tables(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("*/*/poscar0_values/*_poscar0_values.csv"))


def discover_prediction_tables(run_dir: Path, phase: str) -> list[Path]:
    if phase == "zero_shot":
        return sorted(run_dir.glob("*/*/predictions/*_zero_shot_all.csv"))
    return sorted(run_dir.glob("*/*/predictions/*_finetuned_non_poscar0.csv"))


def path_model_mode(path: Path) -> tuple[str, str]:
    try:
        mode = path.parents[1].name
        model = path.parents[2].name
    except IndexError as exc:
        raise ValueError(f"Unexpected native fine-tune output path: {path}") from exc
    return model, mode


def add_model_columns(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    model, mode = path_model_mode(path)
    df = df.copy()
    df["model"] = model
    df["mode"] = mode
    df["model_mode"] = df.apply(
        lambda row: model_mode_display(row["model"], row["mode"]),
        axis=1,
    )
    df["source_csv"] = str(path)
    return df


def add_material_key(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "fine_tune_material" in df.columns and "material" in df.columns:
        df[MATERIAL_KEY] = df["fine_tune_material"].combine_first(df["material"])
    elif "fine_tune_material" in df.columns:
        df[MATERIAL_KEY] = df["fine_tune_material"]
    elif "material" in df.columns:
        df[MATERIAL_KEY] = df["material"]
    else:
        raise ValueError("Result tables must include either fine_tune_material or material.")
    return df


def load_poscar0_tables(run_dir: Path, phase: str) -> pd.DataFrame:
    rows = []
    for path in discover_poscar0_tables(run_dir):
        df = pd.read_csv(path)
        if df.empty:
            continue
        prediction_column = (
            "zero_shot_prediction" if phase == "zero_shot" else "finetuned_prediction"
        )
        if prediction_column not in df.columns:
            continue
        df = df.rename(columns={prediction_column: "prediction"})
        rows.append(add_model_columns(df, path))

    if not rows:
        raise FileNotFoundError(
            f"No POSCAR0 value tables found under {run_dir}. "
            "Expected <run-dir>/<model>/<mode>/poscar0_values/*_poscar0_values.csv."
        )

    out = add_material_key(pd.concat(rows, ignore_index=True))
    required = {"file", "target", "defect_group", "model", "mode"}
    missing = sorted(required - set(out.columns))
    if missing:
        raise ValueError(f"Missing required column(s) in POSCAR0 tables: {', '.join(missing)}")
    return out


def load_prediction_tables(run_dir: Path, phase: str) -> pd.DataFrame:
    rows = []
    for path in discover_prediction_tables(run_dir, phase):
        df = pd.read_csv(path)
        if df.empty:
            continue
        if "prediction" not in df.columns:
            continue
        rows.append(add_model_columns(df, path))
    if not rows:
        return pd.DataFrame()
    out = add_material_key(pd.concat(rows, ignore_index=True))
    required = {"file", "target", "defect_group", "prediction", "model", "mode"}
    missing = sorted(required - set(out.columns))
    if missing:
        raise ValueError(f"Missing required column(s) in prediction tables: {', '.join(missing)}")
    return out


def load_plot_source(run_dir: Path, phase: str, selection_source: str) -> pd.DataFrame:
    """Load rows used for both DFT selection and model comparison."""
    if selection_source == "poscar0":
        return load_poscar0_tables(run_dir, phase)

    prediction_df = load_prediction_tables(run_dir, phase)
    if phase == "zero_shot" and not prediction_df.empty:
        return prediction_df

    rows = []
    if not prediction_df.empty:
        rows.append(prediction_df)
    try:
        rows.append(load_poscar0_tables(run_dir, phase))
    except FileNotFoundError:
        pass
    if not rows:
        raise FileNotFoundError(
            f"No usable {phase} prediction tables found under {run_dir}. "
            "Expected predictions/*.csv and/or poscar0_values/*.csv."
        )
    out = pd.concat(rows, ignore_index=True)
    return out.drop_duplicates(subset=["file", "model", "mode"], keep="last")


def material_column(df: pd.DataFrame) -> str:
    if MATERIAL_KEY in df.columns:
        return MATERIAL_KEY
    if "fine_tune_material" in df.columns:
        return "fine_tune_material"
    if "material" in df.columns:
        return "material"
    raise ValueError("Result tables must include either fine_tune_material or material.")


def defect_label(row: pd.Series) -> str:
    if "defect_label" in row and pd.notna(row["defect_label"]):
        return str(row["defect_label"])
    return str(row["defect_group"])


def selected_last_poscar_rows(df: pd.DataFrame, material: str) -> pd.DataFrame:
    """Select the highest-DFT configuration row per defect group for one material."""
    mat_col = material_column(df)
    material_df = df[df[mat_col].astype(str).eq(str(material))].copy()
    if material_df.empty:
        raise ValueError(f"No result rows found for material {material}.")

    base_cols = ["file", "target", "defect_group"]
    optional_cols = [
        col
        for col in ["defect_label", "configuration", "material", "fine_tune_material", "source_path"]
        if col in material_df.columns
    ]
    candidates = material_df[base_cols + optional_cols].drop_duplicates()

    selected = (
        candidates.sort_values(["defect_group", "target", "file"], ascending=[True, True, True])
        .groupby("defect_group", sort=False, as_index=False)
        .tail(1)
        .copy()
    )
    selected["plot_label"] = selected.apply(defect_label, axis=1)
    selected = selected.sort_values(["target", "plot_label"], ascending=[True, True]).reset_index(drop=True)
    return selected


def build_plot_table(
    df: pd.DataFrame,
    material: str,
    include_labels: set[str] | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = selected_last_poscar_rows(df, material)
    keys = ["file", "defect_group"]

    pred_rows = df[df["file"].isin(selected["file"])].copy()
    if include_labels is not None:
        pred_rows = pred_rows[pred_rows["model_mode"].isin(include_labels)]
    if pred_rows.empty:
        raise ValueError(f"No prediction rows remain for {material} after filtering.")
    if "prediction" not in pred_rows.columns:
        raise ValueError("Prediction column not found after loading result tables.")

    pred_rows = pred_rows.dropna(subset=["prediction"])
    if pred_rows.empty:
        raise ValueError(f"No non-empty predictions remain for {material}.")
    stats = (
        pred_rows.groupby(keys + ["model_mode"], sort=False)["prediction"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    stats["std"] = stats["std"].fillna(0.0)

    plot_table = selected[["plot_label", "file", "defect_group", "target"]].copy()
    for label in stats["model_mode"].drop_duplicates():
        label_stats = stats[stats["model_mode"].eq(label)][keys + ["mean", "std", "count"]]
        plot_table = plot_table.merge(label_stats, on=keys, how="left")
        plot_table = plot_table.rename(
            columns={
                "mean": f"{label} prediction",
                "std": f"{label} std",
                "count": f"{label} n",
            }
        )

    return selected, plot_table


def plot_material(
    plot_table: pd.DataFrame,
    material: str,
    phase: str,
    selection_source: str,
    output_dir: Path,
    dpi: int,
    title: str | None,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = plot_table["plot_label"].astype(str).tolist()
    x = np.arange(len(labels), dtype=float)
    model_labels = [
        col.removesuffix(" prediction")
        for col in plot_table.columns
        if col.endswith(" prediction")
    ]
    series_count = 1 + len(model_labels)
    width = min(0.16, 0.82 / max(series_count, 1))

    fig_width = max(8.8, 0.82 * len(labels) + 0.85 * len(model_labels))
    fig, ax = plt.subplots(figsize=(fig_width, 5.4))

    offsets = (np.arange(series_count) - (series_count - 1) / 2.0) * width
    ax.bar(
        x + offsets[0],
        plot_table["target"].to_numpy(dtype=float),
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
        mae = float(np.nanmean(np.abs(values - plot_table["target"].to_numpy(dtype=float))))
        legend = f"{model_label} (MAE: {mae:.3f})"
        ax.bar(
            x + offsets[idx],
            values,
            width,
            yerr=errors if errors is not None and np.nanmax(errors) > 0 else None,
            capsize=3,
            label=legend,
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
    ax.set_title(title or f"{material}: DFT-ranked final POSCAR comparison ({phase})")
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{material}_native_poscar0_last_poscar_{phase}_{selection_source}.png"
    fig.savefig(output, dpi=dpi)
    plt.close(fig)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot native_poscar0_finetune result rows after sorting DFT values "
            "from low to high and keeping the last POSCAR per defect group."
        )
    )
    parser.add_argument("--run-dir", required=True, help="native_poscar0_finetune output directory.")
    parser.add_argument(
        "--material",
        nargs="+",
        default=None,
        help="Material(s) to plot. Defaults to all materials found in the result tables.",
    )
    parser.add_argument(
        "--phase",
        choices=["finetuned", "zero_shot"],
        default="finetuned",
        help="Prediction column to compare against DFT.",
    )
    parser.add_argument(
        "--selection-source",
        choices=["all", "poscar0"],
        default="all",
        help=(
            "Rows used to select the last DFT-ranked POSCAR. 'all' combines "
            "finetuned non-POSCAR0 predictions with POSCAR0 values; 'poscar0' "
            "uses only poscar0_values tables."
        ),
    )
    parser.add_argument(
        "--include",
        nargs="+",
        default=None,
        help=(
            "Optional display labels to include, e.g. 'Full CGCNN' 'Hetero CGCNN'. "
            "By default all discovered model/mode labels are plotted."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Figure/table output directory. Defaults to <run-dir>/figures_last_poscar.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--title", default=None, help="Optional plot title for a single material.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "figures_last_poscar"
    df = load_plot_source(run_dir, args.phase, args.selection_source)
    mat_col = material_column(df)
    materials = args.material or sorted(df[mat_col].astype(str).dropna().unique())
    include_labels = set(args.include) if args.include else None

    outputs = []
    for material in materials:
        _, plot_table = build_plot_table(df, material, include_labels)
        output_dir.mkdir(parents=True, exist_ok=True)
        table_path = (
            output_dir
            / f"{material}_native_poscar0_last_poscar_{args.phase}_{args.selection_source}.csv"
        )
        plot_table.to_csv(table_path, index=False)
        figure_path = plot_material(
            plot_table,
            material,
            args.phase,
            args.selection_source,
            output_dir,
            args.dpi,
            args.title if len(materials) == 1 else None,
        )
        outputs.append((figure_path, table_path))

    print("Wrote:")
    for figure_path, table_path in outputs:
        print(f"  figure: {figure_path}")
        print(f"  table:  {table_path}")


if __name__ == "__main__":
    main()
