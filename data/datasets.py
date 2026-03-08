"""Dataset loading functions.

Each function returns (dataset_full, dataset_hetero, dataset_sparse, dataset_attn, targets).
"""

import json
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from pymatgen.io.cif import CifParser
from pymatgen.core import Structure

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
    for index, row in tqdm(df_targets.iterrows()):
        file = f'{path}/initial/{index}.cif'
        structure = Structure.from_file(file)
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


# ------------------------------------------------------------------ #
#  Per-dataset loaders
# ------------------------------------------------------------------ #

def load_data_vacancy(task_prefix):
    prepared = {'id': [], 'structure': [], 'base': [], 'cell': [], 'target': [], 'weight': []}
    get_prepared('2d-materials-point-defects-all/low_density_defects/MoS2', prepared)
    get_prepared('2d-materials-point-defects-all/low_density_defects/WSe2', prepared)
    get_prepared('2d-materials-point-defects-all/high_density_defects/MoS2_500', prepared, is_high=True)
    get_prepared('2d-materials-point-defects-all/high_density_defects/WSe2_500', prepared, is_high=True)
    get_prepared('2d-materials-point-defects-all/high_density_defects/BP_spin_500', prepared, is_high=True)
    get_prepared('2d-materials-point-defects-all/high_density_defects/GaSe_spin_500', prepared, is_high=True)
    get_prepared('2d-materials-point-defects-all/high_density_defects/hBN_spin_500', prepared, is_high=True)
    get_prepared('2d-materials-point-defects-all/high_density_defects/InSe_spin_500', prepared, is_high=True)
    df = pd.DataFrame(prepared)
    df.set_index(["id"], inplace=True)
    unit_cells = {
        'MoS2': CifParser("2d-materials-point-defects-all/low_density_defects/MoS2/unit_cells/MoS2.cif").get_structures(primitive=False)[0],
        'WSe2': CifParser("2d-materials-point-defects-all/low_density_defects/MoS2/unit_cells/WSe2.cif").get_structures(primitive=False)[0],
        'MoS2_500': CifParser("2d-materials-point-defects-all/MoS2.cif").get_structures(primitive=False)[0],
        'WSe2_500': CifParser("2d-materials-point-defects-all/WSe2.cif").get_structures(primitive=False)[0],
        'BN_500': CifParser("2d-materials-point-defects-all/BN.cif").get_structures(primitive=False)[0],
        'GaSe_500': CifParser("2d-materials-point-defects-all/GaSe.cif").get_structures(primitive=False)[0],
        'P_500': CifParser("2d-materials-point-defects-all/P.cif").get_structures(primitive=False)[0],
        'InSe_500': CifParser("2d-materials-point-defects-all/InSe.cif").get_structures(primitive=False)[0],
    }
    prep = df.values.tolist()
    prep = [[p[0], p[1], eval(p[2])] for p in prep]

    dataset_full = [convert_to_sparse_vacancy(p[0], unit_cells[p[1]], p[2], f'{task_prefix}_full', None, True, False) for p in tqdm(prep)]
    dataset_hetero = [convert_to_sparse_vacancy(p[0], unit_cells[p[1]], p[2], f'{task_prefix}_hetero', None, True, False) for p in tqdm(prep)]
    dataset_sparse = [convert_to_sparse_vacancy(p[0], unit_cells[p[1]], p[2], f'{task_prefix}_sparse', [1], False, False) for p in tqdm(prep)]
    dataset_attn = [convert_to_sparse_vacancy(p[0], unit_cells[p[1]], p[2], f'{task_prefix}_attention', None, True, False) for p in tqdm(prep)]

    targets = df['target'].values
    pairs = [(xi, yi, ki, ai, mi) for xi, yi, ki, ai, mi in zip(dataset_hetero, dataset_full, dataset_sparse, dataset_attn, targets) if xi is not None]
    dataset_hetero, dataset_full, dataset_sparse, dataset_attn, targets = (list(t) for t in zip(*pairs)) if pairs else ([], [], [], [], [])
    targets = torch.tensor(targets).float()
    return dataset_full, dataset_hetero, dataset_sparse, dataset_attn, targets


def load_data_2dmd_high(task_prefix):
    prepared = {'id': [], 'structure': [], 'base': [], 'cell': [], 'target': [], 'weight': []}
    get_prepared('2d-materials-point-defects-all/high_density_defects/MoS2_500', prepared, is_high=True)
    get_prepared('2d-materials-point-defects-all/high_density_defects/WSe2_500', prepared, is_high=True)
    get_prepared('2d-materials-point-defects-all/high_density_defects/BP_spin_500', prepared, is_high=True)
    get_prepared('2d-materials-point-defects-all/high_density_defects/GaSe_spin_500', prepared, is_high=True)
    get_prepared('2d-materials-point-defects-all/high_density_defects/hBN_spin_500', prepared, is_high=True)
    get_prepared('2d-materials-point-defects-all/high_density_defects/InSe_spin_500', prepared, is_high=True)
    df = pd.DataFrame(prepared)
    df.set_index(["id"], inplace=True)
    unit_cells = {
        'MoS2': CifParser("2d-materials-point-defects-all/low_density_defects/MoS2/unit_cells/MoS2.cif").get_structures(primitive=False)[0],
        'WSe2': CifParser("2d-materials-point-defects-all/low_density_defects/MoS2/unit_cells/WSe2.cif").get_structures(primitive=False)[0],
        'MoS2_500': CifParser("2d-materials-point-defects-all/MoS2.cif").get_structures(primitive=False)[0],
        'WSe2_500': CifParser("2d-materials-point-defects-all/WSe2.cif").get_structures(primitive=False)[0],
        'BN_500': CifParser("2d-materials-point-defects-all/BN.cif").get_structures(primitive=False)[0],
        'GaSe_500': CifParser("2d-materials-point-defects-all/GaSe.cif").get_structures(primitive=False)[0],
        'P_500': CifParser("2d-materials-point-defects-all/P.cif").get_structures(primitive=False)[0],
        'InSe_500': CifParser("2d-materials-point-defects-all/InSe.cif").get_structures(primitive=False)[0],
    }
    prep = df.values.tolist()
    prep = [[p[0], p[1], eval(p[2])] for p in prep]

    dataset_full = [convert_to_sparse_2dmd_high(p[0], unit_cells[p[1]], p[2], f'{task_prefix}_full', None, True, False) for p in tqdm(prep)]
    dataset_hetero = [convert_to_sparse_2dmd_high(p[0], unit_cells[p[1]], p[2], f'{task_prefix}_hetero', None, True, False) for p in tqdm(prep)]
    dataset_sparse = [convert_to_sparse_2dmd_high(p[0], unit_cells[p[1]], p[2], f'{task_prefix}_sparse', [1], False, False) for p in tqdm(prep)]
    dataset_attn = [convert_to_sparse_2dmd_high(p[0], unit_cells[p[1]], p[2], f'{task_prefix}_attention', None, True, False) for p in tqdm(prep)]

    targets = df['target'].values
    targets = torch.tensor(targets).float()
    return dataset_full, dataset_hetero, dataset_sparse, dataset_attn, targets


def load_data_native(task_prefix):
    df_descriptors = pd.read_csv('Dataset_1/Dataset_1/A_rich/Neutral/id_prop_A_rich.csv', header=None)
    prep = []
    targets = []
    for i, j in tqdm(enumerate(df_descriptors[0])):
        struct = Structure.from_file('Dataset_1/Dataset_1/A_rich/Neutral/' + j)
        defect = 'vacancy' if j.split('-')[2].split('_')[0] == 'V' else 'others'
        prep.append([struct, defect])
        targets.append(df_descriptors[1][i])

    dataset_full = [convert_to_sparse_native(p[0], p[1], 1, f'{task_prefix}_full', None, True, False) for p in tqdm(prep)]
    dataset_hetero = [convert_to_sparse_native(p[0], p[1], 1, f'{task_prefix}_hetero', None, True, False) for p in tqdm(prep)]
    dataset_sparse = [convert_to_sparse_native(p[0], p[1], 1, f'{task_prefix}_sparse', None, True, False) for p in tqdm(prep)]
    dataset_attn = [convert_to_sparse_native(p[0], p[1], 1, f'{task_prefix}_attention', None, True, False) for p in tqdm(prep)]
    pairs = [(xi, yi, ki, ai, zi) for xi, yi, ki, ai, zi in zip(dataset_hetero, dataset_full, dataset_sparse, dataset_attn, targets) if xi is not None]
    dataset_hetero, dataset_full, dataset_sparse, dataset_attn, targets = (list(t) for t in zip(*pairs)) if pairs else ([], [], [], [], [])
    targets = torch.tensor(targets).float()
    return dataset_full, dataset_hetero, dataset_sparse, dataset_attn, targets


def load_data_och(task_prefix):
    df_descriptors = pd.read_csv('../autodl-tmp/rs2re_h_ads/id_prop.csv', header=None)
    prep = []
    targets = []
    for i, j in tqdm(enumerate(df_descriptors[0])):
        if df_descriptors[1][i] > -1.5 and df_descriptors[1][i] < 0.5:
            struct = Structure.from_file('../autodl-tmp/rs2re_h_ads/' + j + '.cif')
            prep.append([struct, 'H'])
            targets.append(df_descriptors[1][i])

    dataset_full = [convert_to_sparse_och(p[0], p[1], 1, f'{task_prefix}_full', None, True, False) for p in tqdm(prep)]
    dataset_hetero = [convert_to_sparse_och(p[0], p[1], 1, f'{task_prefix}_hetero', None, True, False) for p in tqdm(prep)]
    dataset_sparse = [convert_to_sparse_och(p[0], p[1], 1, f'{task_prefix}_sparse', None, True, False) for p in tqdm(prep)]
    dataset_attn = [convert_to_sparse_och(p[0], p[1], 1, f'{task_prefix}_attention', None, True, False) for p in tqdm(prep)]
    targets = torch.tensor(targets).float()
    return dataset_full, dataset_hetero, dataset_sparse, dataset_attn, targets


def load_data_imp2d(task_prefix):
    df_descriptors = pd.read_csv('imp2d/imp2d/id_prop.csv', header=None)
    prep = []
    targets = []
    for i, j in tqdm(enumerate(df_descriptors[0])):
        base, impurity, _ = j.split('_')
        if impurity not in base:
            if df_descriptors[1][i] > -10 and df_descriptors[1][i] < 10:
                struct = Structure.from_file('imp2d/imp2d/' + j + '.cif')
                prep.append([struct, impurity])
                targets.append(df_descriptors[1][i])

    dataset_full = [convert_to_sparse_imp2d(p[0], p[1], 1, f'{task_prefix}_full', None, True, False) for p in tqdm(prep)]
    dataset_hetero = [convert_to_sparse_imp2d(p[0], p[1], 1, f'{task_prefix}_hetero', None, True, False) for p in tqdm(prep)]
    dataset_sparse = [convert_to_sparse_imp2d(p[0], p[1], 1, f'{task_prefix}_sparse', None, True, False) for p in tqdm(prep)]
    dataset_attn = [convert_to_sparse_imp2d(p[0], p[1], 1, f'{task_prefix}_attention', None, True, False) for p in tqdm(prep)]
    targets = torch.tensor(targets).float()
    return dataset_full, dataset_hetero, dataset_sparse, dataset_attn, targets


def load_data_semi(task_prefix):
    df_descriptors = pd.read_csv('Dataset_1/Dataset_1/Neutral/Neutral/id_prop_A_rich.csv', header=None)
    prep = []
    targets = []
    for i, j in tqdm(enumerate(df_descriptors[0])):
        base = j.split('-')[1]
        try:
            struct = Structure.from_file('Dataset_1/Dataset_1/Neutral/Neutral/' + j)
            base_structure = Structure.from_file('Dataset_1/host_configurations/' + base + '.vasp')
            base_species = {site.species_string for site in base_structure}
            defect_species = {site.species_string for site in struct}
            if len(defect_species - base_species) > 0:
                prep.append([struct, base_structure])
                targets.append(df_descriptors[1][i])
        except Exception:
            continue

    dataset_full = [convert_to_sparse_semi(p[0], p[1], 1, f'{task_prefix}_full', None, True, False) for p in tqdm(prep)]
    dataset_hetero = [convert_to_sparse_semi(p[0], p[1], 1, f'{task_prefix}_hetero', None, True, False) for p in tqdm(prep)]
    dataset_sparse = [convert_to_sparse_semi(p[0], p[1], 1, f'{task_prefix}_sparse', None, True, False) for p in tqdm(prep)]
    dataset_attn = [convert_to_sparse_semi(p[0], p[1], 1, f'{task_prefix}_attention', None, True, False) for p in tqdm(prep)]
    pairs = [(xi, yi, ki, ai, zi) for xi, yi, ki, ai, zi in zip(dataset_hetero, dataset_full, dataset_sparse, dataset_attn, targets) if xi is not None]
    dataset_hetero, dataset_full, dataset_sparse, dataset_attn, targets = (list(t) for t in zip(*pairs)) if pairs else ([], [], [], [], [])
    targets = torch.tensor(targets).float()
    return dataset_full, dataset_hetero, dataset_sparse, dataset_attn, targets


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


def load_dataset(dataset_name: str, model_name: str):
    """Load and return all four graph representations for a dataset.

    Args:
        dataset_name: one of vacancy, 2dmd_high, native, och, imp2d, semi
        model_name: 'megnet' or 'cgcnn' (used as task prefix)

    Returns:
        (dataset_full, dataset_hetero, dataset_sparse, dataset_attn, targets)
    """
    if dataset_name not in _LOADER_REGISTRY:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Choose from {list(_LOADER_REGISTRY.keys())}")
    return _LOADER_REGISTRY[dataset_name](model_name)
