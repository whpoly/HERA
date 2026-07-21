"""Dataset loading functions.

Each function returns (dataset_full, dataset_hetero, dataset_attn, targets).
Pass ``representations`` or ``modes`` to build only the graph variants needed
for a training run.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from pymatgen.io.cif import CifParser
from pymatgen.core import Composition, Structure

from .structure_utils import (
    convert_to_sparse_vacancy,
    convert_to_sparse_2dmd_high,
    convert_to_sparse_native,
    convert_to_sparse_och,
    convert_to_sparse_imp2d,
    convert_to_sparse_semi,
)

# Global atom embeddings (loaded once from atom_init.json)
elem_embedding = {}

DATASET_REPRESENTATIONS = ('full', 'full_x', 'hetero', 'attention')
DEFAULT_DATASET_REPRESENTATIONS = ('full', 'hetero', 'attention')
REPRESENTATION_INDEX = {
    'full': 0,
    'full_x': 0,
    'hetero': 1,
    'attention': 2,
}
MODE_REPRESENTATION = {
    'full': 'full',
    'full_x': 'full_x',
    'was_x': 'full_x',
    'hetero': 'hetero',
    'hetero_global': 'hetero',
    'hetero_fixed_pool': 'hetero',
    'hetero_was': 'hetero',
    'attention': 'attention',
    'attention_was': 'attention',
    'definet': 'attention',
    'definet_was': 'attention',
}


def representation_for_mode(mode):
    try:
        return MODE_REPRESENTATION[mode]
    except KeyError as exc:
        raise ValueError(f"Unknown mode '{mode}'") from exc


def dataset_index_for_mode(mode):
    return REPRESENTATION_INDEX[representation_for_mode(mode)]


def representations_for_modes(modes):
    if modes is None:
        return set(DEFAULT_DATASET_REPRESENTATIONS)
    if isinstance(modes, str):
        modes = [modes]
    if 'all' in modes:
        return set(DEFAULT_DATASET_REPRESENTATIONS)
    return {representation_for_mode(mode) for mode in modes}


def _normalize_representations(representations=None, modes=None):
    if representations is None:
        normalized = representations_for_modes(modes)
    else:
        if isinstance(representations, str):
            representations = [representations]
        normalized = set(representations)
    unknown = normalized - set(DATASET_REPRESENTATIONS)
    if unknown:
        raise ValueError(
            f"Unknown representation(s) {sorted(unknown)}. "
            f"Choose from {list(DATASET_REPRESENTATIONS)}"
        )
    if {'full', 'full_x'} <= normalized:
        raise ValueError("'full' and 'full_x' must be loaded separately")
    return normalized


def _filter_invalid_datasets(datasets, targets):
    available = [dataset for dataset in datasets if dataset is not None]
    if not available:
        raise ValueError('At least one dataset representation must be requested')

    valid_indices = [
        idx
        for idx in range(len(targets))
        if all(dataset[idx] is not None for dataset in available)
    ]
    filtered = [
        None if dataset is None else [dataset[idx] for idx in valid_indices]
        for dataset in datasets
    ]
    targets = torch.tensor([targets[idx] for idx in valid_indices]).float()
    return (*filtered, targets)


def tag_structure_source(structure, source_path, source_id=None):
    """Attach source-file metadata so downstream explanations can use CIF names."""
    source_path = str(source_path)
    structure.source_path = source_path
    structure.source_name = Path(source_path).stem
    structure.source_id = str(source_id if source_id is not None else structure.source_name)
    return structure


def formula_contains_element(formula, element):
    """Return whether a chemical formula contains an element symbol."""
    try:
        return element in Composition(formula).get_el_amt_dict()
    except Exception:
        return element in formula


def init_elem_embedding(path='atom_init.json'):
    """Load atom embeddings from JSON file."""
    global elem_embedding
    with open(path) as f:
        raw = json.load(f)
        elem_embedding = {int(key): value for key, value in raw.items()}


# ------------------------------------------------------------------ #
#  Helper used by vacancy / 2dmd_high
# ------------------------------------------------------------------ #

def get_prepared(path, prepared, is_high=False):
    df_descriptors = pd.read_csv(f"{path}/descriptors.csv", index_col=0)
    df_targets = pd.read_csv(f"{path}/targets.csv.gz", index_col=0)
    missing_files = []
    for index, row in tqdm(df_targets.iterrows()):
        file = Path(path) / 'initial' / f'{index}.cif'
        if not file.exists():
            missing_files.append(str(index))
            continue
        structure = Structure.from_file(file)
        tag_structure_source(structure, file, index)
        if is_high:
            base = df_descriptors.loc[row['descriptor_id']]['base'] + '_500'
            weight = 0.3132058
        else:
            base = df_descriptors.loc[row['descriptor_id']]['base']
            weight = 3.7165
        cell = df_descriptors.loc[row['descriptor_id']]['cell']
        prepared['target'].append(row['formation_energy_per_site'])
        prepared['weight'].append(weight)
        prepared['structure'].append(structure)
        prepared['id'].append(index)
        prepared['base'].append(base)
        prepared['cell'].append(cell)
    if missing_files:
        preview = ', '.join(missing_files[:5])
        suffix = '' if len(missing_files) <= 5 else ', ...'
        print(f'Skipped {len(missing_files)} missing CIF(s) in {path}/initial: {preview}{suffix}')


# ------------------------------------------------------------------ #
#  Per-dataset loaders
# ------------------------------------------------------------------ #

def load_data_vacancy(task_prefix, local_cutoff=None, representations=None):
    representations = _normalize_representations(representations)
    prepared = {'id': [], 'structure': [], 'base': [], 'cell': [], 'target': [], 'weight': []}
    get_prepared('dataset/2d-materials-point-defects-all/low_density_defects/MoS2', prepared)
    get_prepared('dataset/2d-materials-point-defects-all/low_density_defects/WSe2', prepared)
    get_prepared('dataset/2d-materials-point-defects-all/high_density_defects/MoS2_500', prepared, is_high=True)
    get_prepared('dataset/2d-materials-point-defects-all/high_density_defects/WSe2_500', prepared, is_high=True)
    get_prepared('dataset/2d-materials-point-defects-all/high_density_defects/BP_spin_500', prepared, is_high=True)
    get_prepared('dataset/2d-materials-point-defects-all/high_density_defects/GaSe_spin_500', prepared, is_high=True)
    get_prepared('dataset/2d-materials-point-defects-all/high_density_defects/hBN_spin_500', prepared, is_high=True)
    get_prepared('dataset/2d-materials-point-defects-all/high_density_defects/InSe_spin_500', prepared, is_high=True)
    df = pd.DataFrame(prepared)
    df.set_index(["id"], inplace=True)
    unit_cells = {
        'MoS2': CifParser("dataset/2d-materials-point-defects-all/low_density_defects/MoS2/unit_cells/MoS2.cif").get_structures(primitive=False)[0],
        'WSe2': CifParser("dataset/2d-materials-point-defects-all/low_density_defects/MoS2/unit_cells/WSe2.cif").get_structures(primitive=False)[0],
        'MoS2_500': CifParser("dataset/2d-materials-point-defects-all/MoS2.cif").get_structures(primitive=False)[0],
        'WSe2_500': CifParser("dataset/2d-materials-point-defects-all/WSe2.cif").get_structures(primitive=False)[0],
        'BN_500': CifParser("dataset/2d-materials-point-defects-all/BN.cif").get_structures(primitive=False)[0],
        'GaSe_500': CifParser("dataset/2d-materials-point-defects-all/GaSe.cif").get_structures(primitive=False)[0],
        'P_500': CifParser("dataset/2d-materials-point-defects-all/P.cif").get_structures(primitive=False)[0],
        'InSe_500': CifParser("dataset/2d-materials-point-defects-all/InSe.cif").get_structures(primitive=False)[0],
    }
    prep = df.values.tolist()
    prep = [[p[0], p[1], eval(p[2])] for p in prep]
    skip_full_was = task_prefix not in ('cgcnn', 'megnet', 'definet', 'alignn')

    dataset_full = None
    if 'full' in representations or 'full_x' in representations:
        full_task = f'{task_prefix}_full_x' if 'full_x' in representations else f'{task_prefix}_full'
        dataset_full = [convert_to_sparse_vacancy(p[0], unit_cells[p[1]], p[2], full_task, None, skip_full_was, False) for p in tqdm(prep)]
    dataset_hetero = None
    if 'hetero' in representations:
        dataset_hetero = [convert_to_sparse_vacancy(p[0], unit_cells[p[1]], p[2], f'{task_prefix}_hetero', None, skip_full_was, False, local_cutoff=local_cutoff) for p in tqdm(prep)]
    dataset_attn = None
    if 'attention' in representations:
        dataset_attn = [convert_to_sparse_vacancy(p[0], unit_cells[p[1]], p[2], f'{task_prefix}_attention', None, skip_full_was, False, local_cutoff=local_cutoff) for p in tqdm(prep)]

    return _filter_invalid_datasets(
        (dataset_full, dataset_hetero, dataset_attn),
        df['target'].values,
    )


def load_data_2dmd_high(task_prefix, local_cutoff=None, representations=None):
    representations = _normalize_representations(representations)
    prepared = {'id': [], 'structure': [], 'base': [], 'cell': [], 'target': [], 'weight': []}
    get_prepared('dataset/2d-materials-point-defects-all/high_density_defects/MoS2_500', prepared, is_high=True)
    get_prepared('dataset/2d-materials-point-defects-all/high_density_defects/WSe2_500', prepared, is_high=True)
    get_prepared('dataset/2d-materials-point-defects-all/high_density_defects/BP_spin_500', prepared, is_high=True)
    get_prepared('dataset/2d-materials-point-defects-all/high_density_defects/GaSe_spin_500', prepared, is_high=True)
    get_prepared('dataset/2d-materials-point-defects-all/high_density_defects/hBN_spin_500', prepared, is_high=True)
    get_prepared('dataset/2d-materials-point-defects-all/high_density_defects/InSe_spin_500', prepared, is_high=True)
    df = pd.DataFrame(prepared)
    df.set_index(["id"], inplace=True)
    unit_cells = {
        'MoS2': CifParser("dataset/2d-materials-point-defects-all/low_density_defects/MoS2/unit_cells/MoS2.cif").get_structures(primitive=False)[0],
        'WSe2': CifParser("dataset/2d-materials-point-defects-all/low_density_defects/MoS2/unit_cells/WSe2.cif").get_structures(primitive=False)[0],
        'MoS2_500': CifParser("dataset/2d-materials-point-defects-all/MoS2.cif").get_structures(primitive=False)[0],
        'WSe2_500': CifParser("dataset/2d-materials-point-defects-all/WSe2.cif").get_structures(primitive=False)[0],
        'BN_500': CifParser("dataset/2d-materials-point-defects-all/BN.cif").get_structures(primitive=False)[0],
        'GaSe_500': CifParser("dataset/2d-materials-point-defects-all/GaSe.cif").get_structures(primitive=False)[0],
        'P_500': CifParser("dataset/2d-materials-point-defects-all/P.cif").get_structures(primitive=False)[0],
        'InSe_500': CifParser("dataset/2d-materials-point-defects-all/InSe.cif").get_structures(primitive=False)[0],
    }
    prep = df.values.tolist()
    prep = [[p[0], p[1], eval(p[2])] for p in prep]
    skip_full_was = task_prefix not in ('cgcnn', 'megnet', 'definet', 'alignn')

    dataset_full = None
    if 'full' in representations or 'full_x' in representations:
        full_task = f'{task_prefix}_full_x' if 'full_x' in representations else f'{task_prefix}_full'
        dataset_full = [convert_to_sparse_2dmd_high(p[0], unit_cells[p[1]], p[2], full_task, None, skip_full_was, False) for p in tqdm(prep)]
    dataset_hetero = None
    if 'hetero' in representations:
        dataset_hetero = [convert_to_sparse_2dmd_high(p[0], unit_cells[p[1]], p[2], f'{task_prefix}_hetero', None, skip_full_was, False, local_cutoff=local_cutoff) for p in tqdm(prep)]
    dataset_attn = None
    if 'attention' in representations:
        dataset_attn = [convert_to_sparse_2dmd_high(p[0], unit_cells[p[1]], p[2], f'{task_prefix}_attention', None, skip_full_was, False, local_cutoff=local_cutoff) for p in tqdm(prep)]

    return _filter_invalid_datasets(
        (dataset_full, dataset_hetero, dataset_attn),
        df['target'].values,
    )


def load_data_native(task_prefix, local_cutoff=None, representations=None):
    representations = _normalize_representations(representations)
    df_descriptors = pd.read_csv('dataset/Dataset_1/Dataset_1/A_rich/Neutral/id_prop_A_rich.csv', header=None)
    prep = []
    targets = []
    for i, j in tqdm(enumerate(df_descriptors[0])):
        source_path = 'dataset/Dataset_1/Dataset_1/A_rich/Neutral/' + j
        struct = Structure.from_file(source_path)
        tag_structure_source(struct, source_path, j)
        defect = 'vacancy' if j.split('-')[2].split('_')[0] == 'V' else 'others'
        prep.append([struct, defect])
        targets.append(df_descriptors[1][i])

    dataset_full = None
    if 'full' in representations or 'full_x' in representations:
        full_task = f'{task_prefix}_full_x' if 'full_x' in representations else f'{task_prefix}_full'
        dataset_full = [convert_to_sparse_native(p[0], p[1], 1, full_task, None, True, False) for p in tqdm(prep)]
    dataset_hetero = None
    if 'hetero' in representations:
        dataset_hetero = [convert_to_sparse_native(p[0], p[1], 1, f'{task_prefix}_hetero', None, True, False, local_cutoff=local_cutoff) for p in tqdm(prep)]
    dataset_attn = None
    if 'attention' in representations:
        dataset_attn = [convert_to_sparse_native(p[0], p[1], 1, f'{task_prefix}_attention', None, True, False, local_cutoff=local_cutoff) for p in tqdm(prep)]
    return _filter_invalid_datasets((dataset_full, dataset_hetero, dataset_attn), targets)


def load_data_och(task_prefix, local_cutoff=None, representations=None):
    representations = _normalize_representations(representations)
    df_descriptors = pd.read_csv('dataset/rs2re_h/id_prop.csv', header=None)
    prep = []
    targets = []
    for i, j in tqdm(enumerate(df_descriptors[0])):
        if df_descriptors[1][i] > -1.5 and df_descriptors[1][i] < 0.5:
            source_path = 'dataset/rs2re_h/' + j + '.cif'
            struct = Structure.from_file(source_path)
            tag_structure_source(struct, source_path, j)
            prep.append([struct, 'H'])
            targets.append(df_descriptors[1][i])

    dataset_full = None
    if 'full' in representations or 'full_x' in representations:
        full_task = f'{task_prefix}_full_x' if 'full_x' in representations else f'{task_prefix}_full'
        dataset_full = [convert_to_sparse_och(p[0], p[1], 1, full_task, None, True, False) for p in tqdm(prep)]
    dataset_hetero = None
    if 'hetero' in representations:
        dataset_hetero = [convert_to_sparse_och(p[0], p[1], 1, f'{task_prefix}_hetero', None, True, False, local_cutoff=local_cutoff) for p in tqdm(prep)]
    dataset_attn = None
    if 'attention' in representations:
        dataset_attn = [convert_to_sparse_och(p[0], p[1], 1, f'{task_prefix}_attention', None, True, False, local_cutoff=local_cutoff) for p in tqdm(prep)]
    return _filter_invalid_datasets((dataset_full, dataset_hetero, dataset_attn), targets)


def load_data_imp2d(task_prefix, local_cutoff=None, representations=None):
    representations = _normalize_representations(representations)
    df_descriptors = pd.read_csv('dataset/imp2d/imp2d/id_prop.csv', header=None)
    prep = []
    targets = []
    for i, j in tqdm(enumerate(df_descriptors[0])):
        base, impurity, site = j.split('_')
        if df_descriptors[1][i] <= -10 or df_descriptors[1][i] >= 10:
            continue
        source_path = 'dataset/imp2d/imp2d/' + j + '.cif'
        struct = Structure.from_file(source_path)
        tag_structure_source(struct, source_path, j)
        defect_info = {
            'base': base,
            'impurity': impurity,
            'site': site,
            'is_self': formula_contains_element(base, impurity),
        }
        prep.append([struct, defect_info])
        targets.append(df_descriptors[1][i])

    dataset_full = None
    if 'full' in representations or 'full_x' in representations:
        full_task = f'{task_prefix}_full_x' if 'full_x' in representations else f'{task_prefix}_full'
        dataset_full = [convert_to_sparse_imp2d(p[0], p[1], 1, full_task, None, True, False) for p in tqdm(prep)]
    dataset_hetero = None
    if 'hetero' in representations:
        dataset_hetero = [convert_to_sparse_imp2d(p[0], p[1], 1, f'{task_prefix}_hetero', None, True, False, local_cutoff=local_cutoff) for p in tqdm(prep)]
    dataset_attn = None
    if 'attention' in representations:
        dataset_attn = [convert_to_sparse_imp2d(p[0], p[1], 1, f'{task_prefix}_attention', None, True, False, local_cutoff=local_cutoff) for p in tqdm(prep)]
    return _filter_invalid_datasets((dataset_full, dataset_hetero, dataset_attn), targets)


def load_data_semi(task_prefix, local_cutoff=None, representations=None):
    representations = _normalize_representations(representations)
    df_descriptors = pd.read_csv('dataset/Dataset_1/Dataset_1/Neutral/Neutral/id_prop_A_rich.csv', header=None)
    prep = []
    targets = []
    for i, j in tqdm(enumerate(df_descriptors[0])):
        base = j.split('-')[1]
        try:
            source_path = 'dataset/Dataset_1/Dataset_1/Neutral/Neutral/' + j
            struct = Structure.from_file(source_path)
            tag_structure_source(struct, source_path, j)
            base_structure = Structure.from_file('Dataset_1/host_configurations/' + base + '.vasp')
            base_species = {site.species_string for site in base_structure}
            defect_species = {site.species_string for site in struct}
            if len(defect_species - base_species) > 0:
                prep.append([struct, base_structure])
                targets.append(df_descriptors[1][i])
        except Exception:
            continue

    dataset_full = None
    if 'full' in representations or 'full_x' in representations:
        full_task = f'{task_prefix}_full_x' if 'full_x' in representations else f'{task_prefix}_full'
        dataset_full = [convert_to_sparse_semi(p[0], p[1], 1, full_task, None, True, False) for p in tqdm(prep)]
    dataset_hetero = None
    if 'hetero' in representations:
        dataset_hetero = [convert_to_sparse_semi(p[0], p[1], 1, f'{task_prefix}_hetero', None, True, False, local_cutoff=local_cutoff) for p in tqdm(prep)]
    dataset_attn = None
    if 'attention' in representations:
        dataset_attn = [convert_to_sparse_semi(p[0], p[1], 1, f'{task_prefix}_attention', None, True, False, local_cutoff=local_cutoff) for p in tqdm(prep)]
    return _filter_invalid_datasets((dataset_full, dataset_hetero, dataset_attn), targets)


# ------------------------------------------------------------------ #
#  Unified entry point
# ------------------------------------------------------------------ #

_LOADER_REGISTRY = {
    'vacancy': load_data_vacancy,
    '2dmd_high': load_data_2dmd_high,
    'native': load_data_native,
    'och': load_data_och,
    'imp2d': load_data_imp2d,
    'semi': load_data_semi,
}


def load_dataset(
        dataset_name: str,
        model_name: str,
        local_cutoff=None,
        representations=None,
        modes=None,
):
    """Load and return all four graph representations for a dataset.

    Args:
        dataset_name: one of vacancy, 2dmd_high, native, och, imp2d, semi
        model_name: 'megnet', 'cgcnn', or 'definet' (used as task prefix)
        local_cutoff: optional local/host boundary radius for hetero and attention structures
        representations: optional subset of full, hetero, attention to build
        modes: optional training modes; converted to the needed representation subset

    Returns:
        (dataset_full, dataset_hetero, dataset_attn, targets). Unrequested
        representations are returned as None.
    """
    if dataset_name not in _LOADER_REGISTRY:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Choose from {list(_LOADER_REGISTRY.keys())}")
    representations = _normalize_representations(representations, modes)
    return _LOADER_REGISTRY[dataset_name](
        model_name,
        local_cutoff=local_cutoff,
        representations=representations,
    )
