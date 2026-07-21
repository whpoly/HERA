#!/usr/bin/env python
"""Plot WAS and hetero-radius ablations from a HERA run-level summary file.

The training CLI writes aggregate summaries such as:

    logs/run_xxx/summary.txt
    logs/run_xxx/<model>/summary.txt
    logs/run_xxx/<model>/<dataset>/summary.txt

This script reads only aggregate ``summary.txt`` files, normalizes mode names
like ``hetero_r3`` and ``hetero_was_r3``, and writes:

1. ``ablation_summary.csv`` with one row per model/dataset/mode result.
2. WAS ablation bar charts comparing paired non-radius modes with and without WAS.
3. Hetero radius line charts for ``hetero_r*`` plus optional
   ``hetero_global_r*`` and ``hetero_was_r*`` comparison lines.
"""

from __future__ import annotations

import argparse
import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


MODE_DISPLAY = {
    "full": "Full",
    "full_x": "Full + X",
    "was_x": "Full + X + WAS",
    "was": "WAS",
    "hetero": "Hetero",
    "hetero_global": "Hetero + Global",
    "hetero_was": "Hetero + WAS",
    "hetero_local": "Hetero Local",
    "hetero_local_was": "Hetero Local + WAS",
    "local": "Local",
    "attention": "Attention",
    "attention_was": "Attention + WAS",
    "attention_local": "Attention Local",
    "attention_local_was": "Attention Local + WAS",
    "definet": "DeFiNet",
    "definet_was": "DeFiNet + WAS",
    "definet_local": "DeFiNet Local",
    "definet_local_was": "DeFiNet Local + WAS",
}

MODEL_DISPLAY = {
    "cgcnn": "CGCNN",
    "megnet": "MEGNet",
    "definet": "DeFiNet",
}

PAIR_ORDER = {
    "full": 0,
    "full_x": 1,
    "hetero": 2,
    "hetero_global": 3,
    "hetero_local": 4,
    "local": 5,
    "attention": 6,
    "attention_local": 7,
    "definet": 8,
    "definet_local": 9,
}

COLORS = {
    "no_was": "#5aa0c8",
    "was": "#e8896d",
    "hetero": "#5aa0c8",
    "hetero_global": "#d99a3e",
    "hetero_was": "#e8896d",
    "hetero_local": "#72c4a8",
    "hetero_local_was": "#9b8bd6",
}

HEADER_RE = re.compile(r"^\s*([A-Za-z0-9_]+)\s*\|\s*([A-Za-z0-9_]+)\s*\|\s*([A-Za-z0-9_]+)\s*$")
SUMMARY_RE = re.compile(r"^\s*SUMMARY:\s+([A-Za-z0-9_]+)\s+on\s+(.+?)\s*$")
MEAN_STD_RE = re.compile(
    r"Mean\s*=\s*(?P<mean>[-+0-9.eE]+)\s+Std\s*=\s*(?P<std>[-+0-9.eE]+)"
)
PER_SPLIT_LOSS_RE = re.compile(
    r"Per-(?:seed|fold) losses\s*:\s*(?P<losses>\[[^\]]*\])",
    re.IGNORECASE,
)
AGGREGATE_LOSS_RE = re.compile(
    r"\b(?:Seeds|Folds)\s*=\s*(?P<losses>\[[^\]]*\])",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SummaryRow:
    model: str
    dataset: str
    mode_label: str
    mean: float
    std: float
    losses: tuple[float, ...]
    source: str
    source_rank: int


def normalize_mode(mode_label: str) -> tuple[str, str, bool, int | None]:
    """Return normalized mode, paired base mode, uses_was flag, and radius."""
    normalized = str(mode_label).strip().lower().replace("were", "was")
    radius = None
    radius_match = re.search(r"_r(\d+)$", normalized)
    if radius_match:
        radius = int(radius_match.group(1))
        normalized = normalized[: radius_match.start()]

    uses_was = normalized in {"was", "was_x"} or normalized.endswith("_was")
    if normalized == "was_x":
        pair_base = "full_x"
    elif normalized == "was":
        pair_base = "full"
    elif normalized.endswith("_was"):
        pair_base = normalized.removesuffix("_was")
    else:
        pair_base = normalized

    return normalized, pair_base, uses_was, radius


def display_model(model: str) -> str:
    return MODEL_DISPLAY.get(str(model).lower(), str(model).upper())


def display_mode(mode: str) -> str:
    return MODE_DISPLAY.get(str(mode).lower(), str(mode).replace("_", " ").title())


def pair_label(pair_base: str, radius: int | None) -> str:
    label = display_mode(pair_base)
    if radius is not None:
        label = f"{label} r{radius}"
    return label


def parse_losses(lines: Iterable[str]) -> tuple[float, ...]:
    line_list = list(lines)
    for pattern in (PER_SPLIT_LOSS_RE, AGGREGATE_LOSS_RE):
        for line in line_list:
            match = pattern.search(line)
            if not match:
                continue
            try:
                values = ast.literal_eval(match.group("losses"))
            except (SyntaxError, ValueError):
                continue
            if isinstance(values, (list, tuple)):
                return tuple(float(value) for value in values)
    return tuple()


def parse_mean_std(lines: Iterable[str]) -> tuple[float, float] | None:
    for line in lines:
        match = MEAN_STD_RE.search(line)
        if match:
            return float(match.group("mean")), float(match.group("std"))
    return None


def parse_per_mode_summary(path: Path, lines: list[str]) -> SummaryRow | None:
    if not lines:
        return None
    header = HEADER_RE.match(lines[0])
    if not header:
        return None
    mean_std = parse_mean_std(lines)
    if mean_std is None:
        return None
    model, dataset, mode_label = header.groups()
    mean, std = mean_std
    return SummaryRow(
        model=model.lower(),
        dataset=dataset.lower(),
        mode_label=mode_label.lower(),
        mean=mean,
        std=std,
        losses=parse_losses(lines),
        source=str(path),
        source_rank=0,
    )


def parse_aggregate_summary(path: Path, lines: list[str]) -> list[SummaryRow]:
    rows: list[SummaryRow] = []
    context_kind = None
    context_model = None
    context_dataset = None

    for line in lines:
        if line.strip().upper() == "SUMMARY: ALL MODELS":
            context_kind = "all_models"
            context_model = None
            context_dataset = None
            continue

        summary_match = SUMMARY_RE.match(line)
        if summary_match:
            context_model = summary_match.group(1).lower()
            context_dataset = summary_match.group(2).strip().lower()
            if context_dataset == "all datasets":
                context_kind = "all_datasets"
            else:
                context_kind = "dataset"
            continue

        mean_match = MEAN_STD_RE.search(line)
        if not mean_match or context_kind is None:
            continue

        prefix = line.split("Mean", 1)[0].strip()
        tokens = prefix.split()
        model = dataset = mode_label = None

        if context_kind == "all_models" and len(tokens) >= 3:
            model, dataset, mode_label = tokens[0], tokens[1], tokens[2]
        elif context_kind == "all_datasets" and len(tokens) >= 2:
            model, dataset, mode_label = context_model, tokens[0], tokens[1]
        elif context_kind == "dataset" and len(tokens) >= 1:
            model, dataset, mode_label = context_model, context_dataset, tokens[0]

        if model is None or dataset is None or mode_label is None:
            continue

        rows.append(
            SummaryRow(
                model=str(model).lower(),
                dataset=str(dataset).lower(),
                mode_label=str(mode_label).lower(),
                mean=float(mean_match.group("mean")),
                std=float(mean_match.group("std")),
                losses=parse_losses([line]),
                source=str(path),
                source_rank=1,
            )
        )

    return rows


def read_summary_file(path: Path) -> list[SummaryRow]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    per_mode = parse_per_mode_summary(path, lines)
    if per_mode is not None:
        return [per_mode]
    return parse_aggregate_summary(path, lines)


def discover_summary_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    summary_path = input_path / "summary.txt"
    if summary_path.is_file():
        return [summary_path]

    aggregate_paths = []
    for path in sorted(input_path.rglob("summary.txt")):
        try:
            first_line = path.read_text(encoding="utf-8", errors="replace").splitlines()[0]
        except IndexError:
            continue
        if first_line.strip().startswith("SUMMARY:"):
            aggregate_paths.append(path)

    if aggregate_paths:
        min_depth = min(len(path.relative_to(input_path).parts) for path in aggregate_paths)
        return [
            path
            for path in aggregate_paths
            if len(path.relative_to(input_path).parts) == min_depth
        ]

    raise FileNotFoundError(
        f"No aggregate summary.txt found under {input_path}. "
        "Expected a summary whose first line starts with 'SUMMARY:'."
    )


def load_summary_table(input_path: Path) -> pd.DataFrame:
    rows: list[SummaryRow] = []
    for path in discover_summary_files(input_path):
        rows.extend(read_summary_file(path))
    if not rows:
        raise ValueError(f"No parseable summary rows found in {input_path}")

    records = []
    for row_idx, row in enumerate(rows):
        mode, pair_base, uses_was, radius = normalize_mode(row.mode_label)
        records.append(
            {
                "_row_order": row_idx,
                "model": row.model,
                "dataset": row.dataset,
                "mode_label": row.mode_label,
                "mode": mode,
                "pair_base": pair_base,
                "uses_was": uses_was,
                "uses_hetero": pair_base.startswith("hetero"),
                "radius": radius,
                "mean": row.mean,
                "std": row.std,
                "losses": ";".join(f"{value:.10g}" for value in row.losses),
                "n_splits": len(row.losses),
                "source": row.source,
                "source_rank": row.source_rank,
            }
        )

    df = pd.DataFrame.from_records(records)
    df = df.drop_duplicates(subset=["model", "dataset", "mode_label"], keep="last")
    df = df.sort_values(
        ["source_rank", "model", "dataset", "mode_label", "source", "_row_order"],
        kind="mergesort",
    )
    df = df.reset_index(drop=True)
    return df


def filter_table(df: pd.DataFrame, models: list[str] | None, datasets: list[str] | None) -> pd.DataFrame:
    out = df
    if models:
        wanted = {value.lower() for value in models}
        out = out[out["model"].isin(wanted)]
    if datasets:
        wanted = {value.lower() for value in datasets}
        out = out[out["dataset"].isin(wanted)]
    if out.empty:
        raise ValueError("No rows remain after filtering by model/dataset.")
    return out.copy()


def clean_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def apply_ylim(ax, ylim: tuple[float, float] | None) -> None:
    if ylim is not None:
        ax.set_ylim(*ylim)


def apply_axis_padding(ax, ylim: tuple[float, float] | None) -> None:
    ax.margins(x=0.08)
    if ylim is None:
        ax.margins(y=0.12)


def plot_was_ablation(
    group: pd.DataFrame,
    output_dir: Path,
    metric_label: str,
    dpi: int,
    ylim: tuple[float, float] | None,
) -> Path | None:
    was_df = group[group["radius"].isna()].copy()
    if was_df.empty:
        return None

    paired = []
    for pair_base, base_df in was_df.groupby("pair_base", sort=False):
        no_was = base_df[~base_df["uses_was"]]
        with_was = base_df[base_df["uses_was"]]
        if no_was.empty or with_was.empty:
            continue
        no_row = no_was.iloc[-1]
        was_row = with_was.iloc[-1]
        paired.append(
            {
                "pair_base": pair_base,
                "label": pair_label(pair_base, None),
                "no_was_mean": float(no_row["mean"]),
                "no_was_std": float(no_row["std"]),
                "was_mean": float(was_row["mean"]),
                "was_std": float(was_row["std"]),
            }
        )

    if not paired:
        return None

    plot_df = pd.DataFrame(paired)
    plot_df["_order"] = plot_df.apply(
        lambda row: PAIR_ORDER.get(str(row["pair_base"]), 99),
        axis=1,
    )
    plot_df = plot_df.sort_values("_order").reset_index(drop=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(plot_df), dtype=float)
    width = 0.30
    fig_width = max(7.2, 1.02 * len(plot_df) + 2.2)
    fig, ax = plt.subplots(figsize=(fig_width, 4.8))
    ax.bar(
        x - width / 2,
        plot_df["no_was_mean"],
        width,
        yerr=plot_df["no_was_std"],
        capsize=3,
        label="Without WAS",
        color=COLORS["no_was"],
        edgecolor="black",
        linewidth=0.45,
    )
    ax.bar(
        x + width / 2,
        plot_df["was_mean"],
        width,
        yerr=plot_df["was_std"],
        capsize=3,
        label="With WAS",
        color=COLORS["was"],
        edgecolor="black",
        linewidth=0.45,
    )

    ax.set_ylabel(metric_label)
    ax.set_xlabel("Model")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["label"], rotation=35, ha="right")
    apply_ylim(ax, ylim)
    apply_axis_padding(ax, ylim)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(frameon=True, loc="upper right")
    fig.tight_layout(pad=1.2)

    output = output_dir / f"{clean_name(group['model'].iloc[0])}_{clean_name(group['dataset'].iloc[0])}_was_ablation.png"
    fig.savefig(output, dpi=dpi)
    plt.close(fig)
    return output


def plot_hetero_radius(
    group: pd.DataFrame,
    output_dir: Path,
    metric_label: str,
    dpi: int,
    ylim: tuple[float, float] | None,
) -> Path | None:
    radius_df = group[
        group["pair_base"].isin(("hetero", "hetero_global"))
        & group["radius"].notna()
    ].copy()
    if radius_df.empty:
        return None
    radius_df = radius_df.drop_duplicates(
        subset=["pair_base", "uses_was", "radius"],
        keep="last",
    ).copy()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    radii = sorted(radius_df["radius"].dropna().astype(int).unique())
    radius_positions = {radius: idx for idx, radius in enumerate(radii)}
    plotted = 0
    for pair_base, uses_was, label, color, marker in [
        ("hetero", False, "Hetero", COLORS["hetero"], "o"),
        ("hetero_global", False, "Hetero + Global", COLORS["hetero_global"], "^"),
        ("hetero", True, "Hetero + WAS", COLORS["hetero_was"], "s"),
    ]:
        series = radius_df[
            radius_df["pair_base"].eq(pair_base)
            & radius_df["uses_was"].eq(uses_was)
        ].copy()
        if series.empty:
            continue
        series = series.sort_values("radius")
        x_positions = [radius_positions[int(radius)] for radius in series["radius"]]
        ax.errorbar(
            x_positions,
            series["mean"],
            yerr=series["std"],
            marker=marker,
            linewidth=1.8,
            capsize=3,
            label=label,
            color=color,
        )
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return None

    ax.set_xticks(range(len(radii)))
    ax.set_xticklabels([str(radius) for radius in radii])
    ax.set_xlabel("Radius")
    ax.set_ylabel(metric_label)
    apply_ylim(ax, ylim)
    apply_axis_padding(ax, ylim)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(frameon=True, loc="upper left")
    fig.tight_layout(pad=1.2)

    output = output_dir / f"{clean_name(group['model'].iloc[0])}_{clean_name(group['dataset'].iloc[0])}_hetero_radius_ablation.png"
    fig.savefig(output, dpi=dpi)
    plt.close(fig)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot WAS/no-WAS and hetero radius ablations from a HERA run-level summary.txt."
    )
    parser.add_argument(
        "--summary-path",
        "--run-dir",
        "--input",
        dest="summary_path",
        required=True,
        help="An aggregate summary.txt file, or a run directory containing aggregate summary.txt files.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to <summary-path>/figures_ablation for directories.",
    )
    parser.add_argument("--model", nargs="+", default=None, help="Optional model filter, e.g. cgcnn megnet.")
    parser.add_argument("--dataset", nargs="+", default=None, help="Optional dataset filter, e.g. vacancy native.")
    parser.add_argument("--metric-label", default="Test MAE", help="Y-axis label for the metric.")
    parser.add_argument(
        "--radius-ylim",
        nargs=2,
        type=float,
        metavar=("YMIN", "YMAX"),
        default=None,
        help="Optional explicit y-axis range for the hetero radius figure, e.g. --radius-ylim 0 0.8.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.summary_path)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif input_path.is_dir():
        output_dir = input_path / "figures_ablation"
    else:
        output_dir = input_path.parent / "figures_ablation"

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df = load_summary_table(input_path)
    summary_df = filter_table(summary_df, args.model, args.dataset)
    radius_ylim = tuple(args.radius_ylim) if args.radius_ylim is not None else None
    if radius_ylim is not None and radius_ylim[0] >= radius_ylim[1]:
        raise ValueError("--radius-ylim requires YMIN < YMAX")

    summary_csv = output_dir / "ablation_summary.csv"
    summary_df.drop(columns=["source_rank", "_row_order"]).to_csv(summary_csv, index=False)

    figure_paths: list[Path] = []
    for (_, _), group in summary_df.groupby(["model", "dataset"], sort=True):
        was_path = plot_was_ablation(group, output_dir, args.metric_label, args.dpi, None)
        radius_path = plot_hetero_radius(group, output_dir, args.metric_label, args.dpi, radius_ylim)
        figure_paths.extend(path for path in [was_path, radius_path] if path is not None)

    print("Wrote:")
    print(f"  table: {summary_csv}")
    for path in figure_paths:
        print(f"  figure: {path}")
    if not figure_paths:
        print("  no figures: no paired WAS rows or hetero radius rows were found")


if __name__ == "__main__":
    main()
