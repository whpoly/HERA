"""Batch GNNExplainer runs and notebook-free visualizations."""

from __future__ import annotations

import csv
import html
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.loader import DataLoader


HETERO_TASKS = {
    "megnet_hetero",
    "megnet_local",
    "cgcnn_hetero",
    "cgcnn_local",
    "hetero_cgcnn_was",
}


@dataclass
class ExplanationRunSummary:
    output_dir: Path
    index_csv: Path
    total: int
    succeeded: int
    failed: int


class PredictionWrapper(nn.Module):
    """Expose trainer models with unscaled property predictions for explainers."""

    def __init__(self, trainer):
        super().__init__()
        self.model = trainer.model
        self.task = trainer.config["task"]
        self.mean = float(trainer.scaler.mean)
        self.std = float(trainer.scaler.std) if abs(float(trainer.scaler.std)) > 1e-7 else 1.0

    def forward(
            self,
            x,
            edge_index,
            edge_attr=None,
            batch=None,
            state=None,
            bond_batch=None,
            node_type=None,
            defect_marker=None,
    ):
        if self.task in ("megnet_hetero", "megnet_local"):
            pred = self.model(x, edge_index, edge_attr, state, batch, bond_batch)
        elif self.task in ("cgcnn_hetero", "cgcnn_local", "hetero_cgcnn_was"):
            pred = self.model(x, edge_index, edge_attr, batch)
        elif self.task in ("cgcnn_sparse", "cgcnn_full", "cgcnn_was"):
            pred = self.model(x, edge_index, edge_attr, batch)
        elif self.task == "cgcnn_attention":
            pred = self.model(x, edge_index, edge_attr, batch, node_type=node_type)
        elif self.task == "definet_attention":
            marker = defect_marker if defect_marker is not None else node_type
            pred = self.model(x, edge_index, edge_attr, batch, defect_marker=marker)
        elif self.task == "megnet_attention":
            pred = self.model(
                x, edge_index, edge_attr, state, batch, bond_batch, node_type=node_type
            )
        else:
            pred = self.model(x, edge_index, edge_attr, state, batch, bond_batch)
        return pred.view(-1, 1) * self.std + self.mean


def explain_trainer_predictions(
        trainer,
        output_dir,
        device,
        max_samples=None,
        formats=("csv", "html", "png"),
        epochs=100,
        lr=0.01,
        cmap="viridis_r",
        strict=False,
):
    """Explain all current ``trainer.test_structures`` and save batch outputs.

    The trainer is expected to have just loaded the model state that produced
    the test predictions. Outputs are written as per-sample files plus an
    ``index.csv`` with prediction/error metadata.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    formats = tuple(dict.fromkeys(formats or ()))
    task = trainer.config["task"]
    wrapper = PredictionWrapper(trainer).to(device)
    wrapper.eval()

    explainer = Explainer(
        model=wrapper,
        algorithm=GNNExplainer(epochs=epochs, lr=lr),
        explanation_type="phenomenon",
        node_mask_type="object",
        model_config={
            "mode": "regression",
            "task_level": "graph",
            "return_type": "raw",
        },
    )

    loader = DataLoader(trainer.test_structures, batch_size=1, shuffle=False)
    total_available = len(trainer.test_structures)
    total = total_available if max_samples is None else min(int(max_samples), total_available)
    rows = []
    used_names = {}
    was_training = trainer.model.training

    try:
        for sample_idx, batch in enumerate(loader):
            if sample_idx >= total:
                break

            row = {
                "sample_index": sample_idx,
                "source_id": "",
                "source_name": "",
                "source_path": "",
                "output_name": "",
                "status": "ok",
                "target": "",
                "prediction": "",
                "abs_error": "",
                "value_min": "",
                "value_max": "",
                "value_mean": "",
                "top_atom_indices": "",
                "csv": "",
                "html": "",
                "png": "",
                "error": "",
            }

            try:
                batch = batch.to(device)
                target = _batch_target(batch, device)
                with torch.no_grad():
                    pred = _call_model(wrapper, batch, task).detach().cpu().view(-1)[0].item()

                explanation = _call_explainer(explainer, batch, task, target)
                structure, type_index = _extract_structure_and_types(batch, task)
                source_meta = _source_metadata(structure, sample_idx)
                sample_name = _unique_output_name(source_meta["source_name"], used_names)
                row.update(source_meta)
                row["output_name"] = sample_name
                values, node_types = _extract_node_values(explanation, batch, task, type_index)
                atoms = _atoms_from_structure(structure, values, node_types, cmap)

                target_value = target.detach().cpu().view(-1)[0].item()
                abs_error = abs(pred - target_value)
                value_arr = np.asarray(values, dtype=float)
                top_atoms = np.argsort(-value_arr)[:10].tolist()

                row.update({
                    "target": target_value,
                    "prediction": pred,
                    "abs_error": abs_error,
                    "value_min": float(np.min(value_arr)),
                    "value_max": float(np.max(value_arr)),
                    "value_mean": float(np.mean(value_arr)),
                    "top_atom_indices": ";".join(str(i) for i in top_atoms),
                })

                if "csv" in formats:
                    csv_path = output_dir / f"{sample_name}.csv"
                    _write_atom_csv(csv_path, atoms)
                    row["csv"] = csv_path.name
                if "html" in formats:
                    html_path = output_dir / f"{sample_name}.html"
                    _write_html(html_path, structure, atoms, row, cmap)
                    row["html"] = html_path.name
                if "png" in formats:
                    png_path = output_dir / f"{sample_name}.png"
                    _write_png(png_path, structure, atoms, row, cmap)
                    row["png"] = png_path.name
            except Exception as exc:
                row["status"] = "failed"
                row["error"] = str(exc)
                if strict:
                    raise
            rows.append(row)
    finally:
        trainer.model.train(was_training)

    index_csv = output_dir / "index.csv"
    _write_index(index_csv, rows)
    succeeded = sum(1 for row in rows if row["status"] == "ok")
    failed = len(rows) - succeeded
    return ExplanationRunSummary(output_dir, index_csv, len(rows), succeeded, failed)


def _batch_target(batch, device):
    target = batch.y
    if not torch.is_tensor(target):
        target = torch.tensor([target], dtype=torch.float32, device=device)
    return target.to(device).float().view(-1, 1)


def _model_args(batch, task):
    if task in HETERO_TASKS:
        kwargs = {
            "edge_attr": batch.edge_attr_dict,
            "batch": batch.batch_dict,
        }
        if task.startswith("megnet"):
            kwargs["state"] = batch.state
            kwargs["bond_batch"] = batch.bond_batch_dict
        return (batch.x_dict, batch.edge_index_dict), kwargs

    kwargs = {
        "edge_attr": batch.edge_attr,
        "batch": batch.batch,
    }
    if task.startswith("megnet"):
        kwargs["state"] = batch.state
        kwargs["bond_batch"] = batch.bond_batch
    if task.endswith("_attention"):
        kwargs["node_type"] = getattr(batch, "node_type", None)
        kwargs["defect_marker"] = getattr(batch, "defect_marker", None)
    return (batch.x, batch.edge_index), kwargs


def _call_model(wrapper, batch, task):
    args, kwargs = _model_args(batch, task)
    return wrapper(*args, **kwargs)


def _call_explainer(explainer, batch, task, target):
    args, kwargs = _model_args(batch, task)
    return explainer(*args, **kwargs, target=target)


def _extract_structure_and_types(batch, task):
    payload = getattr(batch, "structure")
    if task in HETERO_TASKS:
        if isinstance(payload, (list, tuple)) and len(payload) == 1:
            payload = payload[0]
        if isinstance(payload, (list, tuple)) and len(payload) >= 2:
            return payload[0], payload[1]
        raise ValueError("Could not extract hetero structure payload from batch.structure")

    if isinstance(payload, (list, tuple)) and len(payload) == 1:
        payload = payload[0]
    return payload, None


def _source_metadata(structure, sample_idx):
    source_path = str(getattr(structure, "source_path", "") or "")
    source_id = str(getattr(structure, "source_id", "") or "")
    source_name = str(getattr(structure, "source_name", "") or "")
    if not source_name and source_path:
        source_name = Path(source_path).stem
    if not source_name and source_id:
        source_name = Path(source_id).stem
    if not source_name:
        source_name = f"sample_{sample_idx:05d}"
    return {
        "source_id": source_id,
        "source_name": source_name,
        "source_path": source_path,
    }


def _unique_output_name(source_name, used_names):
    safe_name = _safe_filename_stem(source_name)
    count = used_names.get(safe_name, 0)
    used_names[safe_name] = count + 1
    if count == 0:
        return safe_name
    return f"{safe_name}_{count + 1:02d}"


def _safe_filename_stem(value):
    raw = str(value or "sample").strip()
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in raw)
    safe = safe.strip(" ._")
    if not safe:
        safe = "sample"
    reserved = {
        "CON", "PRN", "AUX", "NUL",
        "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
        "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
    }
    if safe.upper() in reserved:
        safe = f"{safe}_file"
    return safe[:180]


def _extract_node_values(explanation, batch, task, type_index):
    if task in HETERO_TASKS:
        type_labels = _tensor_like_to_list(type_index)
        atom_mask = _mask_to_vector(explanation["atom"].node_mask)
        defect_mask = _mask_to_vector(explanation["defect"].node_mask)
        atom_i = 0
        defect_i = 0
        values = []
        for label in type_labels:
            if int(label) == 0:
                values.append(float(atom_mask[atom_i]) if atom_i < len(atom_mask) else 0.0)
                atom_i += 1
            else:
                values.append(float(defect_mask[defect_i]) if defect_i < len(defect_mask) else 0.0)
                defect_i += 1
        return values, type_labels

    values = _mask_to_vector(explanation.node_mask)
    node_types = getattr(batch, "node_type", None)
    if node_types is not None:
        node_types = _tensor_like_to_list(node_types)
    return [float(v) for v in values], node_types


def _mask_to_vector(mask):
    if mask is None:
        raise ValueError("Explainer did not return a node mask")
    return mask.detach().cpu().view(-1).numpy()


def _tensor_like_to_list(value):
    if torch.is_tensor(value):
        return value.detach().cpu().view(-1).tolist()
    if hasattr(value, "tolist"):
        raw = value.tolist()
        return raw if isinstance(raw, list) else [raw]
    return list(value)


def _atoms_from_structure(structure, values, node_types, cmap):
    if len(values) != len(structure):
        raise ValueError(
            f"Node mask length ({len(values)}) does not match structure length ({len(structure)})"
        )
    elements = [_site_element(site) for site in structure.sites]
    radii = _element_radii(elements)
    colors = _value_colors(values, cmap)
    atoms = []
    for idx, site in enumerate(structure.sites):
        node_type = None
        if node_types is not None and idx < len(node_types):
            node_type = int(node_types[idx])
        elif "type" in site.properties:
            node_type = int(site.properties["type"])
        atoms.append({
            "index": idx,
            "element": elements[idx],
            "x": float(site.coords[0]),
            "y": float(site.coords[1]),
            "z": float(site.coords[2]),
            "value": float(values[idx]),
            "node_type": "" if node_type is None else node_type,
            "color": colors[idx],
            "radius": radii[elements[idx]],
        })
    return atoms


def _site_element(site):
    try:
        return site.specie.symbol
    except Exception:
        return site.species_string


def _element_z(element):
    try:
        from pymatgen.core import Element

        return Element(element).Z
    except Exception:
        return 0


def _element_radii(elements):
    unique = sorted(set(elements), key=lambda item: (_element_z(item), item))
    if len(unique) == 1:
        return {unique[0]: 0.8}
    min_radius = 0.45
    max_radius = 0.95
    return {
        element: min_radius + (max_radius - min_radius) * i / (len(unique) - 1)
        for i, element in enumerate(unique)
    }


def _value_colors(values, cmap):
    values = np.asarray(values, dtype=float)
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if math.isclose(vmin, vmax):
        scaled = np.full_like(values, 0.5, dtype=float)
    else:
        scaled = (values - vmin) / (vmax - vmin)
    try:
        from matplotlib import cm

        cmap_fn = cm.get_cmap(cmap)
        rgba = cmap_fn(scaled)
        return [
            "#{:02x}{:02x}{:02x}".format(
                int(row[0] * 255), int(row[1] * 255), int(row[2] * 255)
            )
            for row in rgba
        ]
    except Exception:
        return [_fallback_color(float(v)) for v in scaled]


def _fallback_color(value):
    r = int(255 * value)
    g = int(128 + 80 * value)
    b = int(255 * (1.0 - value))
    return f"#{r:02x}{g:02x}{b:02x}"


def _write_atom_csv(path, atoms):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["index", "element", "x", "y", "z", "value", "node_type", "color", "radius"],
        )
        writer.writeheader()
        writer.writerows(atoms)


def _write_index(path, rows):
    fieldnames = [
        "sample_index",
        "source_id",
        "source_name",
        "source_path",
        "output_name",
        "status",
        "target",
        "prediction",
        "abs_error",
        "value_min",
        "value_max",
        "value_mean",
        "top_atom_indices",
        "csv",
        "html",
        "png",
        "error",
    ]
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_html(path, structure, atoms, row, cmap):
    cell_edges = _unit_cell_edges(structure)
    title = f"Sample {row['sample_index']} explanation"
    atoms_json = json.dumps(atoms)
    cell_json = json.dumps(cell_edges)
    legend_gradient = _cmap_gradient(cmap)
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <script src="https://unpkg.com/ngl@2.0.0-dev.38/dist/ngl.js"></script>
  <style>
    html, body {{ margin: 0; height: 100%; font-family: Arial, sans-serif; }}
    #viewport {{ width: 100%; height: 86vh; }}
    .meta {{ padding: 10px 14px; font-size: 13px; line-height: 1.45; }}
    .bar {{ height: 10px; width: 240px; background: {legend_gradient}; }}
    .legend {{ display: flex; align-items: center; gap: 10px; margin-top: 4px; }}
  </style>
</head>
<body>
  <div id="viewport"></div>
  <div class="meta">
    <strong>{html.escape(title)}</strong>
    target={row['target']} prediction={row['prediction']} abs_error={row['abs_error']}
    <div class="legend">
      <span>low</span><div class="bar" title="{html.escape(cmap)}"></div><span>high</span>
      <span>value range: {row['value_min']} - {row['value_max']}</span>
    </div>
  </div>
  <script>
    const atoms = {atoms_json};
    const cellEdges = {cell_json};
    function hexToRgb01(hex) {{
      const value = hex.replace("#", "");
      return [
        parseInt(value.slice(0, 2), 16) / 255,
        parseInt(value.slice(2, 4), 16) / 255,
        parseInt(value.slice(4, 6), 16) / 255,
      ];
    }}
    document.addEventListener("DOMContentLoaded", function () {{
      const stage = new NGL.Stage("viewport", {{ backgroundColor: "white" }});
      const shape = new NGL.Shape("node attribution");
      atoms.forEach(function (atom) {{
        shape.addSphere(
          [atom.x, atom.y, atom.z],
          hexToRgb01(atom.color),
          atom.radius,
          atom.index + ":" + atom.element + " value=" + atom.value.toFixed(5)
        );
      }});
      cellEdges.forEach(function (edge) {{
        shape.addCylinder(edge[0], edge[1], [0.25, 0.25, 0.25], 0.035, "cell");
      }});
      stage.addComponentFromObject(shape).then(function (component) {{
        component.addRepresentation("buffer");
        stage.autoView();
      }});
      window.addEventListener("resize", function () {{ stage.handleResize(); }});
    }});
  </script>
</body>
</html>
"""
    Path(path).write_text(html_text, encoding="utf-8")


def _cmap_gradient(cmap):
    try:
        from matplotlib import cm

        cmap_fn = cm.get_cmap(cmap)
        stops = []
        for value in np.linspace(0.0, 1.0, 5):
            rgba = cmap_fn(float(value))
            color = "#{:02x}{:02x}{:02x}".format(
                int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
            )
            stops.append(f"{color} {int(value * 100)}%")
        return f"linear-gradient(90deg, {', '.join(stops)})"
    except Exception:
        return "linear-gradient(90deg, #fde725, #21918c, #440154)"


def _unit_cell_edges(structure):
    lattice = np.asarray(structure.lattice.matrix, dtype=float)
    origin = np.zeros(3)
    a, b, c = lattice
    corners = [origin, a, b, c, a + b, a + c, b + c, a + b + c]
    edge_indices = [
        (0, 1), (0, 2), (0, 3),
        (1, 4), (1, 5),
        (2, 4), (2, 6),
        (3, 5), (3, 6),
        (4, 7), (5, 7), (6, 7),
    ]
    return [[corners[i].tolist(), corners[j].tolist()] for i, j in edge_indices]


def _write_png(path, structure, atoms, row, cmap):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors

    coords = np.asarray([[atom["x"], atom["y"], atom["z"]] for atom in atoms], dtype=float)
    values = np.asarray([atom["value"] for atom in atoms], dtype=float)
    sizes = np.asarray([atom["radius"] for atom in atoms], dtype=float)
    sizes = (sizes * 70.0) ** 2

    fig = plt.figure(figsize=(7, 6), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    norm = colors.Normalize(vmin=float(values.min()), vmax=float(values.max()))
    if math.isclose(float(values.min()), float(values.max())):
        norm = colors.Normalize(vmin=float(values.min()) - 0.5, vmax=float(values.max()) + 0.5)
    ax.scatter(
        coords[:, 0], coords[:, 1], coords[:, 2],
        c=values, cmap=cmap, norm=norm, s=sizes,
        edgecolors="black", linewidths=0.25, alpha=0.95,
    )
    for start, end in _unit_cell_edges(structure):
        xs, ys, zs = zip(start, end)
        ax.plot(xs, ys, zs, color="0.35", linewidth=0.8, alpha=0.7)
    _set_axes_equal(ax, coords)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(
        f"sample {row['sample_index']} | target={float(row['target']):.4f} "
        f"pred={float(row['prediction']):.4f}"
    )
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.72, label="node attribution")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _set_axes_equal(ax, coords):
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    centers = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins) / 2.0)
    if radius <= 0:
        radius = 1.0
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)
