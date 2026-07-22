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
    "megnet_hetero_was",
    "megnet_hetero_local",
    "megnet_hetero_local_was",
    "cgcnn_hetero",
    "hetero_cgcnn_was",
    "cgcnn_hetero_local",
    "cgcnn_hetero_local_was",
    "alignn_hetero",
    "alignn_hetero_was",
    "alignn_hetero_local",
    "alignn_hetero_local_was",
}
MEGNET_HETERO_TASKS = {
    "megnet_hetero",
    "megnet_hetero_was",
    "megnet_hetero_local",
    "megnet_hetero_local_was",
}
CGCNN_HETERO_TASKS = {
    "cgcnn_hetero",
    "hetero_cgcnn_was",
    "cgcnn_hetero_local",
    "cgcnn_hetero_local_was",
}
ALIGNN_HOMOGENEOUS_TASKS = {
    "alignn_full",
    "alignn_full_x",
    "alignn_local",
    "alignn_was_x",
}
ALIGNN_HETERO_TASKS = {
    "alignn_hetero",
    "alignn_hetero_was",
    "alignn_hetero_local",
    "alignn_hetero_local_was",
}
ALIGNN_ATTENTION_TASKS = {
    "alignn_attention",
    "alignn_attention_local",
    "alignn_attention_was",
    "alignn_attention_local_was",
}
ALIGNN_DEFINET_TASKS = {
    "alignn_definet",
    "alignn_definet_local",
    "alignn_definet_was",
    "alignn_definet_local_was",
}

HETERO_NODE_TYPES = ("atom", "defect")
HETERO_EDGE_TYPES = (
    ("atom", "aa", "atom"),
    ("defect", "dd", "defect"),
    ("atom", "ad", "defect"),
    ("defect", "da", "atom"),
)

CGCNN_ATTENTION_TASKS = {
    "cgcnn_attention",
    "cgcnn_attention_local",
    "cgcnn_attention_was",
    "cgcnn_attention_local_was",
}

MEGNET_ATTENTION_TASKS = {
    "megnet_attention",
    "megnet_attention_local",
    "megnet_attention_was",
    "megnet_attention_local_was",
}

DEFINET_ATTENTION_TASKS = {
    "definet_attention",
    "definet_attention_local",
    "definet_attention_was",
    "definet_attention_local_was",
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
            edge_vec=None,
    ):
        if self.task in MEGNET_HETERO_TASKS:
            x, edge_index, edge_attr, batch, bond_batch = _complete_hetero_inputs(
                x, edge_index, edge_attr, batch, bond_batch
            )
            pred = self.model(x, edge_index, edge_attr, state, batch, bond_batch)
        elif self.task in CGCNN_HETERO_TASKS:
            x, edge_index, edge_attr, batch, _ = _complete_hetero_inputs(
                x, edge_index, edge_attr, batch, None
            )
            pred = self.model(x, edge_index, edge_attr, batch)
        elif self.task in ALIGNN_HETERO_TASKS:
            x, edge_index, edge_attr, batch, _ = _complete_hetero_inputs(
                x, edge_index, edge_attr, batch, None
            )
            edge_vec = _complete_hetero_edge_vecs(edge_vec, edge_attr)
            pred = self.model(x, edge_index, edge_attr, batch, edge_vec_dict=edge_vec, state=state)
        elif self.task in (
                "cgcnn_sparse", "cgcnn_full", "cgcnn_full_x",
                "cgcnn_local", "cgcnn_was_x",
        ):
            pred = self.model(x, edge_index, edge_attr, batch)
        elif self.task in ALIGNN_HOMOGENEOUS_TASKS:
            pred = self.model(x, edge_index, edge_attr, batch, edge_vec=edge_vec)
        elif self.task in ALIGNN_ATTENTION_TASKS:
            pred = self.model(
                x, edge_index, edge_attr, batch, edge_vec=edge_vec, node_type=node_type
            )
        elif self.task in ALIGNN_DEFINET_TASKS:
            marker = defect_marker if defect_marker is not None else node_type
            pred = self.model(
                x, edge_index, edge_attr, batch, edge_vec=edge_vec, defect_marker=marker
            )
        elif self.task in CGCNN_ATTENTION_TASKS:
            pred = self.model(x, edge_index, edge_attr, batch, node_type=node_type)
        elif self.task in DEFINET_ATTENTION_TASKS:
            marker = defect_marker if defect_marker is not None else node_type
            pred = self.model(x, edge_index, edge_attr, batch, defect_marker=marker)
        elif self.task in MEGNET_ATTENTION_TASKS:
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
        formats=("ovito",),
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
                "ovito": "",
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
                source_meta = _source_metadata(structure, sample_idx, batch)
                sample_name = _unique_output_name(_explain_output_stem(source_meta), used_names)
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

                if "ovito" in formats:
                    ovito_path = output_dir / f"{sample_name}.xyz"
                    _write_ovito_xyz(ovito_path, structure, atoms)
                    row["ovito"] = ovito_path.name
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
        edge_index_dict = _nonempty_edge_dict(batch.edge_index_dict)
        edge_attr_dict = _select_dict_keys(batch.edge_attr_dict, edge_index_dict.keys())
        kwargs = {
            "edge_attr": edge_attr_dict,
            "batch": batch.batch_dict,
        }
        if task.startswith("megnet"):
            kwargs["state"] = batch.state
            kwargs["bond_batch"] = _select_dict_keys(batch.bond_batch_dict, edge_index_dict.keys())
        if task.startswith("alignn"):
            kwargs["state"] = batch.state
            kwargs["edge_vec"] = _select_dict_keys(_collect_hetero_attr(batch, "edge_vec"), edge_index_dict.keys())
        return (batch.x_dict, edge_index_dict), kwargs

    kwargs = {
        "edge_attr": batch.edge_attr,
        "batch": batch.batch,
    }
    if task.startswith("megnet"):
        kwargs["state"] = batch.state
        kwargs["bond_batch"] = batch.bond_batch
    if task.startswith("alignn"):
        kwargs["edge_vec"] = getattr(batch, "edge_vec", None)
    if (
            task.endswith("_attention")
            or task in CGCNN_ATTENTION_TASKS
            or task in MEGNET_ATTENTION_TASKS
            or task in DEFINET_ATTENTION_TASKS
            or task in ALIGNN_ATTENTION_TASKS
            or task in ALIGNN_DEFINET_TASKS
    ):
        kwargs["node_type"] = getattr(batch, "node_type", None)
        kwargs["defect_marker"] = getattr(batch, "defect_marker", None)
    return (batch.x, batch.edge_index), kwargs


def _call_model(wrapper, batch, task):
    args, kwargs = _model_args(batch, task)
    return wrapper(*args, **kwargs)


def _call_explainer(explainer, batch, task, target):
    args, kwargs = _model_args(batch, task)
    return explainer(*args, **kwargs, target=target)


def _nonempty_edge_dict(edge_index_dict):
    return {
        edge_type: edge_index
        for edge_type, edge_index in edge_index_dict.items()
        if edge_index.size(1) > 0
    }


def _select_dict_keys(value_dict, keys):
    return {key: value_dict[key] for key in keys if key in value_dict}


def _collect_hetero_attr(batch, attr):
    try:
        return batch.collect(attr)
    except (AttributeError, KeyError):
        return getattr(batch, f"{attr}_dict", {}) or {}


def _complete_hetero_edge_vecs(edge_vec_dict, edge_attr_dict):
    edge_vec_dict = {} if edge_vec_dict is None else dict(edge_vec_dict)
    for edge_type in HETERO_EDGE_TYPES:
        if edge_type not in edge_vec_dict:
            edge_vec_dict[edge_type] = edge_attr_dict[edge_type].new_zeros(
                (edge_attr_dict[edge_type].size(0), 3)
            )
    return edge_vec_dict


def _complete_hetero_inputs(x_dict, edge_index_dict, edge_attr_dict, batch_dict, bond_batch_dict=None):
    x_dict = dict(x_dict)
    edge_index_dict = dict(edge_index_dict)
    edge_attr_dict = dict(edge_attr_dict)
    batch_dict = dict(batch_dict)
    bond_batch_dict = None if bond_batch_dict is None else dict(bond_batch_dict)

    ref_x = next(iter(x_dict.values()))
    node_feature_dim = ref_x.shape[1] if ref_x.dim() > 1 else 1
    for node_type in HETERO_NODE_TYPES:
        if node_type not in x_dict:
            x_dict[node_type] = ref_x.new_empty((0, node_feature_dim))
        if node_type not in batch_dict:
            batch_dict[node_type] = torch.empty((0,), dtype=torch.long, device=ref_x.device)

    if edge_attr_dict:
        ref_edge_attr = next(iter(edge_attr_dict.values()))
        edge_feature_dim = ref_edge_attr.shape[1] if ref_edge_attr.dim() > 1 else 1
    else:
        ref_edge_attr = ref_x.new_empty((0, 1))
        edge_feature_dim = 1

    for edge_type in HETERO_EDGE_TYPES:
        if edge_type not in edge_index_dict:
            edge_index_dict[edge_type] = torch.empty((2, 0), dtype=torch.long, device=ref_x.device)
        if edge_type not in edge_attr_dict:
            edge_attr_dict[edge_type] = ref_edge_attr.new_empty((0, edge_feature_dim))
        if bond_batch_dict is not None and edge_type not in bond_batch_dict:
            bond_batch_dict[edge_type] = torch.empty((0,), dtype=torch.long, device=ref_x.device)

    return x_dict, edge_index_dict, edge_attr_dict, batch_dict, bond_batch_dict


def _extract_structure_and_types(batch, task):
    payload = getattr(batch, "structure")
    if task in HETERO_TASKS:
        payload = _unwrap_singletons(payload)
        if isinstance(payload, (list, tuple)) and len(payload) >= 2:
            structure = _structure_from_payload(payload[0])
            type_index = _unwrap_singletons(payload[1])
            return structure, type_index
        raise ValueError("Could not extract hetero structure payload from batch.structure")

    return _structure_from_payload(payload), None


def _unwrap_singletons(value):
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    return value


def _structure_from_payload(payload):
    payload = _unwrap_singletons(payload)
    if hasattr(payload, "sites") and hasattr(payload, "lattice"):
        return payload

    if hasattr(payload, "coords") and hasattr(payload, "lattice"):
        from pymatgen.core import Structure

        return Structure.from_sites([payload])

    # PyG may collate pymatgen Structure as a nested list of PeriodicSite objects.
    if isinstance(payload, (list, tuple)) and payload and all(hasattr(site, "coords") for site in payload):
        from pymatgen.core import Structure

        return Structure.from_sites(list(payload))

    raise ValueError(f"Could not extract pymatgen Structure from batch.structure payload: {type(payload)}")


def _source_metadata(structure, sample_idx, batch=None):
    source_path = _metadata_value(batch, "source_path") or str(getattr(structure, "source_path", "") or "")
    source_id = _metadata_value(batch, "source_id") or str(getattr(structure, "source_id", "") or "")
    source_name = _metadata_value(batch, "source_name") or str(getattr(structure, "source_name", "") or "")
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


def _metadata_value(source, attr):
    if source is None or not hasattr(source, attr):
        return ""
    value = getattr(source, attr)
    value = _unwrap_singletons(value)
    if torch.is_tensor(value):
        raw = value.detach().cpu().view(-1).tolist()
        return str(raw[0]) if raw else ""
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else ""
    return str(value or "")


def _explain_output_stem(source_meta):
    """Prefer the input CIF stem for explanation filenames."""
    for key in ("source_name", "source_path", "source_id"):
        value = str(source_meta.get(key, "") or "").strip()
        if value:
            return Path(value).stem
    return "sample"


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
            if _node_type_to_int(label, default=0) == 0:
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
            node_type = _node_type_to_int(node_types[idx], default=None)
        elif "type" in site.properties:
            node_type = _node_type_to_int(site.properties["type"], default=None)
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


def _node_type_to_int(value, default=None):
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("", "none", "null", "nan"):
            return default
        if normalized in ("false", "f", "no"):
            return 0
        if normalized in ("true", "t", "yes"):
            return 1
    return int(value)


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


def _write_ovito_xyz(path, structure, atoms):
    lattice = np.asarray(structure.lattice.matrix, dtype=float).reshape(-1)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        handle.write(f"{len(atoms)}\n")
        handle.write(
            'Lattice="{}" '.format(" ".join(f"{value:.10f}" for value in lattice))
            + "Properties=species:S:1:pos:R:3:id:I:1:importance:R:1:Color:R:3:Radius:R:1 "
            + 'pbc="T T T"\n'
        )
        for atom in atoms:
            red, green, blue = _hex_to_rgb01(atom["color"])
            handle.write(
                f"{atom['element']} "
                f"{float(atom['x']):.10f} {float(atom['y']):.10f} {float(atom['z']):.10f} "
                f"{int(atom['index'])} {float(atom['value']):.10f} "
                f"{red:.6f} {green:.6f} {blue:.6f} {float(atom['radius']):.6f}\n"
            )


def _hex_to_rgb01(value):
    raw = str(value or "#808080").strip().lstrip("#")
    if len(raw) != 6:
        return 0.5, 0.5, 0.5
    try:
        return tuple(int(raw[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    except ValueError:
        return 0.5, 0.5, 0.5


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
        "ovito",
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
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    html, body {{ margin: 0; height: 100%; font-family: Arial, sans-serif; color: #1f2933; }}
    body {{ display: flex; flex-direction: column; background: #ffffff; }}
    #viewport {{ flex: 1 1 auto; width: 100%; min-height: 360px; display: block; cursor: grab; }}
    #viewport:active {{ cursor: grabbing; }}
    .meta {{ flex: 0 0 auto; padding: 10px 14px; font-size: 13px; line-height: 1.45; border-top: 1px solid #d9dee5; }}
    .bar {{ height: 10px; width: 240px; background: {legend_gradient}; }}
    .legend {{ display: flex; align-items: center; gap: 10px; margin-top: 4px; }}
    .reset {{ float: right; border: 1px solid #c8d0da; background: #fff; border-radius: 4px; padding: 4px 8px; cursor: pointer; }}
  </style>
</head>
<body>
  <canvas id="viewport"></canvas>
  <div class="meta">
    <button class="reset" id="reset" type="button">Reset</button>
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
    const canvas = document.getElementById("viewport");
    const ctx = canvas.getContext("2d");
    let rotX = -0.55;
    let rotY = 0.72;
    let zoom = 1.0;
    let dragging = false;
    let lastX = 0;
    let lastY = 0;

    const points = atoms.map(atom => [atom.x, atom.y, atom.z]);
    cellEdges.forEach(edge => {{
      points.push(edge[0], edge[1]);
    }});
    const bounds = points.reduce((acc, point) => {{
      for (let i = 0; i < 3; i += 1) {{
        acc.min[i] = Math.min(acc.min[i], point[i]);
        acc.max[i] = Math.max(acc.max[i], point[i]);
      }}
      return acc;
    }}, {{ min: [Infinity, Infinity, Infinity], max: [-Infinity, -Infinity, -Infinity] }});
    const center = [0, 1, 2].map(i => (bounds.min[i] + bounds.max[i]) / 2);
    const span = Math.max(...[0, 1, 2].map(i => bounds.max[i] - bounds.min[i]), 1);

    function resize() {{
      const rect = canvas.getBoundingClientRect();
      const ratio = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.floor(rect.width * ratio));
      canvas.height = Math.max(1, Math.floor(rect.height * ratio));
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      draw();
    }}

    function rotate(point) {{
      const x0 = point[0] - center[0];
      const y0 = point[1] - center[1];
      const z0 = point[2] - center[2];
      const cx = Math.cos(rotX);
      const sx = Math.sin(rotX);
      const cy = Math.cos(rotY);
      const sy = Math.sin(rotY);
      const y1 = y0 * cx - z0 * sx;
      const z1 = y0 * sx + z0 * cx;
      const x2 = x0 * cy + z1 * sy;
      const z2 = -x0 * sy + z1 * cy;
      return [x2, y1, z2];
    }}

    function project(point) {{
      const rect = canvas.getBoundingClientRect();
      const scale = Math.min(rect.width, rect.height) * 0.78 * zoom / span;
      const rotated = rotate(point);
      return {{
        x: rect.width / 2 + rotated[0] * scale,
        y: rect.height / 2 - rotated[1] * scale,
        z: rotated[2],
        scale,
      }};
    }}

    function drawCell() {{
      ctx.save();
      ctx.strokeStyle = "#8a94a6";
      ctx.lineWidth = 1;
      ctx.globalAlpha = 0.75;
      cellEdges.forEach(edge => {{
        const a = project(edge[0]);
        const b = project(edge[1]);
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
      }});
      ctx.restore();
    }}

    function drawAtom(item) {{
      const atom = item.atom;
      const point = item.point;
      const radius = Math.max(5, atom.radius * point.scale * 0.18);
      const shade = Math.max(0.72, Math.min(1.15, 0.92 + point.z / span * 0.18));
      ctx.save();
      ctx.beginPath();
      ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
      ctx.fillStyle = atom.color;
      ctx.shadowColor = "rgba(17, 24, 39, 0.22)";
      ctx.shadowBlur = Math.max(2, radius * 0.18);
      ctx.shadowOffsetY = Math.max(1, radius * 0.08);
      ctx.fill();
      ctx.shadowColor = "transparent";
      ctx.globalCompositeOperation = "source-atop";
      ctx.fillStyle = `rgba(255, 255, 255, ${{Math.max(0, shade - 1)}})`;
      ctx.fillRect(point.x - radius, point.y - radius, radius * 2, radius * 2);
      ctx.globalCompositeOperation = "source-over";
      ctx.strokeStyle = "#17202a";
      ctx.lineWidth = 0.6;
      ctx.stroke();
      ctx.restore();
    }}

    function drawLabels(projectedAtoms) {{
      const rect = canvas.getBoundingClientRect();
      if (rect.width < 560 || atoms.length > 80) {{
        return;
      }}
      ctx.save();
      ctx.font = "11px Arial, sans-serif";
      ctx.fillStyle = "#1f2933";
      ctx.strokeStyle = "rgba(255, 255, 255, 0.85)";
      ctx.lineWidth = 3;
      projectedAtoms.slice(-40).forEach(item => {{
        const text = `${{item.atom.index}}:${{item.atom.element}}`;
        const x = item.point.x + 7;
        const y = item.point.y - 7;
        ctx.strokeText(text, x, y);
        ctx.fillText(text, x, y);
      }});
      ctx.restore();
    }}

    function draw() {{
      const rect = canvas.getBoundingClientRect();
      ctx.clearRect(0, 0, rect.width, rect.height);
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, rect.width, rect.height);
      drawCell();
      const projectedAtoms = atoms
        .map(atom => ({{ atom, point: project([atom.x, atom.y, atom.z]) }}))
        .sort((a, b) => a.point.z - b.point.z);
      projectedAtoms.forEach(drawAtom);
      drawLabels(projectedAtoms);
    }}

    canvas.addEventListener("pointerdown", event => {{
      dragging = true;
      lastX = event.clientX;
      lastY = event.clientY;
      canvas.setPointerCapture(event.pointerId);
    }});
    canvas.addEventListener("pointermove", event => {{
      if (!dragging) {{
        return;
      }}
      rotY += (event.clientX - lastX) * 0.01;
      rotX += (event.clientY - lastY) * 0.01;
      lastX = event.clientX;
      lastY = event.clientY;
      draw();
    }});
    canvas.addEventListener("pointerup", event => {{
      dragging = false;
      canvas.releasePointerCapture(event.pointerId);
    }});
    canvas.addEventListener("wheel", event => {{
      event.preventDefault();
      zoom *= event.deltaY < 0 ? 1.08 : 0.92;
      zoom = Math.max(0.25, Math.min(5.0, zoom));
      draw();
    }}, {{ passive: false }});
    document.getElementById("reset").addEventListener("click", () => {{
      rotX = -0.55;
      rotY = 0.72;
      zoom = 1.0;
      draw();
    }});
    window.addEventListener("resize", resize);
    resize();
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
