"""Structure manipulation utilities for each dataset type.

Each dataset has its own convert_to_sparse_* function that produces the
appropriate graph representation (sparse / full / hetero / local / attention).
"""

import numpy as np
from pymatgen.core import Structure
from pymatgen.core.periodic_table import DummySpecies
from pymatgen.core.sites import PeriodicSite


# ================================================================== #
#  Shared helpers
# ================================================================== #

def strucure_to_dict(structure, precision=3):
    res = {}
    for site in structure:
        res[tuple(np.round(site.frac_coords, precision))] = site
    return res


def get_site_index(structure: Structure, site, tolerance=1e-3):
    if hasattr(site, 'coords'):
        coords = site.coords
    else:
        coords = site
    for i, struct_site in enumerate(structure):
        if np.allclose(struct_site.coords, coords, atol=tolerance):
            return i


def set_attr(structure, attr, name):
    setattr(structure, name, attr)
    return structure


def add_was(structure, unit_cell, supercell_size):
    structure = structure.copy()
    reference_supercell = unit_cell.copy()
    reference_supercell.make_supercell(supercell_size)
    sites = []
    structure_dict = strucure_to_dict(structure)
    reference_structure_dict = strucure_to_dict(reference_supercell)
    for coords, reference_site in reference_structure_dict.items():
        if coords not in structure_dict:
            continue
        else:
            cur_site = structure_dict[coords]
            cur_site.properties['was'] = reference_site.specie.Z
            sites.append(
                PeriodicSite(
                    species=cur_site.species,
                    coords=coords,
                    coords_are_cartesian=False,
                    lattice=structure.lattice,
                    properties=cur_site.properties,
                )
            )
    return Structure.from_sites(sites)


def add_unit_cell_properties(structure, unit_cell, supercell_size):
    structure = structure.copy()
    reference_supercell = unit_cell.copy()
    reference_supercell.make_supercell(supercell_size)
    sites = []
    structure_dict = strucure_to_dict(structure)
    reference_structure_dict = strucure_to_dict(reference_supercell)
    for coords, reference_site in reference_structure_dict.items():
        if coords not in structure_dict:
            continue
        else:
            cur_site = structure_dict[coords]
            cur_site.properties.update(reference_site.properties)
            sites.append(
                PeriodicSite(
                    species=cur_site.species,
                    coords=coords,
                    coords_are_cartesian=False,
                    lattice=structure.lattice,
                    properties=cur_site.properties,
                )
            )
    return Structure.from_sites(sites)


def get_full(structure, unit_cell, supercell_size, state):
    return structure.copy()


# ================================================================== #
#  Vacancy
# ================================================================== #

def get_sparse_vacancy(structure, unit_cell, supercell_size):
    structure = structure.copy()
    reference_supercell = unit_cell.copy()
    reference_supercell.make_supercell(supercell_size)
    defects = []
    structure_dict = strucure_to_dict(structure)
    reference_structure_dict = strucure_to_dict(reference_supercell)
    for coords, reference_site in reference_structure_dict.items():
        if coords not in structure_dict:
            defects.append(PeriodicSite(
                species=DummySpecies(), coords=coords,
                coords_are_cartesian=False, lattice=structure.lattice, properties={},
            ))
        elif structure_dict[coords].specie != reference_site.specie:
            defects.append(structure_dict[coords])
    return Structure.from_sites(defects)


def get_hetero_vacancy(structure, unit_cell, supercell_size, state):
    structure = structure.copy()
    reference_supercell = unit_cell.copy()
    reference_supercell.make_supercell(supercell_size)
    sites_raw = []
    structure_dict = strucure_to_dict(structure)
    reference_structure_dict = strucure_to_dict(reference_supercell)
    for coords, reference_site in reference_structure_dict.items():
        if coords not in structure_dict:
            sites_raw.append(PeriodicSite(
                species=DummySpecies(), coords=coords,
                coords_are_cartesian=False, lattice=structure.lattice,
                properties={'type': True},
            ))
        elif structure_dict[coords].specie != reference_site.specie:
            return None
        else:
            cur_site = structure_dict[coords]
            cur_site.properties['type'] = False
            sites_raw.append(cur_site)
    return Structure.from_sites(sites_raw)


def add_state_vacancy(structure, unit_cell):
    reference_species = set(unit_cell.species)
    structure = structure.copy()
    structure.state = [sorted([element.Z for element in reference_species])]
    return structure


def convert_to_sparse_vacancy(structure, unit_cell, supercell_size, task, state,
                               skip_was=False, copy_unit_cell_properties=False):
    structure = structure.copy()
    unit_cell = unit_cell.copy()
    if task == "cgcnn_hetero" or task == "megnet_hetero":
        structure = get_hetero_vacancy(structure, unit_cell, supercell_size, state)
    elif task == 'cgcnn_attention' or task == 'megnet_attention':
        structure = get_hetero_vacancy(structure, unit_cell, supercell_size, state)
    elif task == 'cgcnn_sparse' or task == 'megnet_sparse':
        structure = get_sparse_vacancy(structure, unit_cell, supercell_size)
    else:
        structure = get_full(structure, unit_cell, supercell_size, state)
    if structure is None:
        return None
    if not skip_was:
        structure = add_was(structure, unit_cell, supercell_size)
    if copy_unit_cell_properties:
        structure = add_unit_cell_properties(structure, unit_cell, supercell_size)
    if state is not None:
        structure = add_state_vacancy(structure, unit_cell)
    return structure


# ================================================================== #
#  2dmd_high
# ================================================================== #

def get_sparse_2dmd_high(structure, unit_cell, supercell_size):
    return get_sparse_vacancy(structure, unit_cell, supercell_size)


def get_hetero_2dmd_high(structure, unit_cell, supercell_size, state):
    structure = structure.copy()
    reference_supercell = unit_cell.copy()
    reference_supercell.make_supercell(supercell_size)
    sites_raw = []
    structure_dict = strucure_to_dict(structure)
    reference_structure_dict = strucure_to_dict(reference_supercell)
    for coords, reference_site in reference_structure_dict.items():
        if coords not in structure_dict:
            sites_raw.append(PeriodicSite(
                species=DummySpecies(), coords=coords,
                coords_are_cartesian=False, lattice=structure.lattice,
                properties={'type': True},
            ))
        elif structure_dict[coords].specie != reference_site.specie:
            cur_site = structure_dict[coords]
            cur_site.properties['type'] = True
            sites_raw.append(cur_site)
        else:
            cur_site = structure_dict[coords]
            cur_site.properties['type'] = False
            sites_raw.append(cur_site)
    return Structure.from_sites(sites_raw)


def add_state_2dmd_high(structure, unit_cell):
    return add_state_vacancy(structure, unit_cell)


def convert_to_sparse_2dmd_high(structure, unit_cell, supercell_size, task, state,
                                 skip_was=False, copy_unit_cell_properties=False):
    structure = structure.copy()
    unit_cell = unit_cell.copy()
    if task == "cgcnn_hetero" or task == "megnet_hetero":
        structure = get_hetero_2dmd_high(structure, unit_cell, supercell_size, state)
    elif task == 'cgcnn_attention' or task == 'megnet_attention':
        structure = get_hetero_2dmd_high(structure, unit_cell, supercell_size, state)
    elif task == 'cgcnn_sparse' or task == 'megnet_sparse':
        structure = get_sparse_2dmd_high(structure, unit_cell, supercell_size)
    else:
        structure = get_full(structure, unit_cell, supercell_size, state)
    if not skip_was:
        structure = add_was(structure, unit_cell, supercell_size)
    if copy_unit_cell_properties:
        structure = add_unit_cell_properties(structure, unit_cell, supercell_size)
    if state is not None:
        structure = add_state_2dmd_high(structure, unit_cell)
    return structure


# ================================================================== #
#  Native
# ================================================================== #

def get_sparse_native(structure, unit_cell, supercell_size):
    structure = structure.copy()
    sites_raw = []
    structure_dict = strucure_to_dict(structure)
    l = len(structure_dict)
    for i, (coords, reference_site) in enumerate(structure_dict.items()):
        if i == l - 1:
            if unit_cell == 'vacancy':
                cur_site = PeriodicSite(
                    species=DummySpecies(), coords=(0.5, 0.5, 0.5),
                    coords_are_cartesian=False, lattice=structure.lattice, properties={},
                )
                sites_raw.append(cur_site)
            else:
                sites_raw.append(structure_dict[coords])
    return Structure.from_sites(sites_raw)


def get_hetero_native(structure, unit_cell, supercell_size, state):
    structure = structure.copy()
    sites_raw = []
    structure_dict = strucure_to_dict(structure)
    l = len(structure_dict)
    for i, (coords, reference_site) in enumerate(structure_dict.items()):
        if i == l - 1:
            if unit_cell == 'vacancy':
                cur_site = PeriodicSite(
                    species=DummySpecies(), coords=(0.5, 0.5, 0.5),
                    coords_are_cartesian=False, lattice=structure.lattice, properties={},
                )
                cur_site.properties['type'] = True
                sites_raw.append(cur_site)
            else:
                cur_site = structure_dict[coords]
                cur_site.properties['type'] = True
                sites_raw.append(cur_site)
        else:
            cur_site = structure_dict[coords]
            cur_site.properties['type'] = False
            sites_raw.append(cur_site)
    return Structure.from_sites(sites_raw)


def get_local_native(structure, unit_cell, supercell_size, state):
    structure = structure.copy()
    reference_supercell = unit_cell.copy()
    base_species = [site.species_string for site in reference_supercell]
    reference_supercell.make_supercell(supercell_size)
    sites_raw = []
    base_species = unit_cell
    defect_idx = None
    for idx in range(len(structure)):
        if structure[idx].species_string not in base_species:
            defect_idx = idx
        else:
            continue

    structure_dict = strucure_to_dict(structure)
    for index, (coords, reference_site) in enumerate(structure_dict.items()):
        distance = structure.get_distance(index, defect_idx)
        if distance <= 5:
            cur_site = structure_dict[coords]
            cur_site.properties['type'] = True
            sites_raw.append(cur_site)
        elif distance > 0 and distance < 12:
            cur_site = structure_dict[coords]
            cur_site.properties['type'] = False
            sites_raw.append(cur_site)
    return Structure.from_sites(sites_raw)


def add_state_native(structure, unit_cell):
    reference_species = set(unit_cell.species)
    structure = structure.copy()
    structure.state = [sorted([element.Z for element in reference_species])]
    return structure


def convert_to_sparse_native(structure, unit_cell, supercell_size, task, state,
                              skip_was=False, copy_unit_cell_properties=False):
    structure = structure.copy()
    if task == "cgcnn_hetero" or task == "megnet_hetero":
        structure = get_hetero_native(structure, unit_cell, supercell_size, state)
    elif task == 'cgcnn_attention' or task == 'megnet_attention':
        structure = get_hetero_native(structure, unit_cell, supercell_size, state)
    elif task == 'cgcnn_sparse' or task == 'megnet_sparse':
        structure = get_sparse_native(structure, unit_cell, supercell_size)
    elif task == 'cgcnn_local' or task == 'megnet_local':
        structure = get_local_native(structure, unit_cell, supercell_size, state)
    else:
        structure = get_full(structure, unit_cell, supercell_size, state)
    if not skip_was:
        structure = add_was(structure, unit_cell, supercell_size)
    if copy_unit_cell_properties:
        structure = add_unit_cell_properties(structure, unit_cell, supercell_size)
    if state is not None:
        structure = add_state_native(structure, unit_cell)
    return structure


# ================================================================== #
#  OCH
# ================================================================== #

def get_sparse_och(structure, unit_cell, supercell_size):
    impurity_sites = [*filter(lambda x: x.species_string == unit_cell, structure)]
    return Structure.from_sites(impurity_sites)


def get_hetero_och(structure, unit_cell, supercell_size, state):
    structure = structure.copy()
    base_species = unit_cell
    sites_raw = []
    structure_dict = strucure_to_dict(structure)
    for coords, reference_site in structure_dict.items():
        cur_site = structure_dict[coords]
        if cur_site.species_string == base_species:
            cur_site.properties['type'] = True
        else:
            cur_site.properties['type'] = False
        sites_raw.append(cur_site)
    return Structure.from_sites(sites_raw)


def get_local_och(structure, unit_cell, supercell_size):
    structure = structure.copy()
    sites_raw = []
    base_species = unit_cell
    defect_idx = None
    for idx in range(len(structure)):
        if structure[idx].species_string == base_species:
            defect_idx = idx

    structure_dict = strucure_to_dict(structure)
    for index, (coords, reference_site) in enumerate(structure_dict.items()):
        distance = structure.get_distance(index, defect_idx)
        if distance == 0:
            cur_site = structure_dict[coords]
            cur_site.properties['type'] = True
            sites_raw.append(cur_site)
        elif distance > 0 and distance < 30:
            cur_site = structure_dict[coords]
            cur_site.properties['type'] = False
            sites_raw.append(cur_site)
    return Structure.from_sites(sites_raw)


def add_state_och(structure, state):
    structure.state = [state]
    return structure


def convert_to_sparse_och(structure, unit_cell, supercell_size, task, state,
                           skip_was=False, copy_unit_cell_properties=False):
    structure = structure.copy()
    if task == "cgcnn_hetero" or task == "megnet_hetero":
        structure = get_hetero_och(structure, unit_cell, supercell_size, state)
    elif task == 'cgcnn_attention' or task == 'megnet_attention':
        structure = get_hetero_och(structure, unit_cell, supercell_size, state)
    elif task == 'cgcnn_sparse' or task == 'megnet_sparse':
        structure = get_sparse_och(structure, unit_cell, supercell_size)
    elif task == 'cgcnn_local' or task == 'megnet_local':
        structure = get_local_och(structure, unit_cell, supercell_size)
    else:
        structure = get_full(structure, unit_cell, supercell_size, state)
    if not skip_was:
        structure = add_was(structure, unit_cell, supercell_size)
    if copy_unit_cell_properties:
        structure = add_unit_cell_properties(structure, unit_cell, supercell_size)
    if state is not None:
        structure = add_state_och(structure, state)
    return structure


# ================================================================== #
#  imp2d
# ================================================================== #

def get_sparse_imp2d(structure, unit_cell, supercell_size):
    return get_sparse_och(structure, unit_cell, supercell_size)


def get_hetero_imp2d(structure, unit_cell, supercell_size, state):
    return get_hetero_och(structure, unit_cell, supercell_size, state)


def get_local_imp2d(structure, unit_cell, supercell_size, state):
    structure = structure.copy()
    sites_raw = []
    base_species = unit_cell
    defect_idx = None
    for idx in range(len(structure)):
        if structure[idx].species_string == base_species:
            defect_idx = idx

    structure_dict = strucure_to_dict(structure)
    for index, (coords, reference_site) in enumerate(structure_dict.items()):
        distance = structure.get_distance(index, defect_idx)
        if distance == 0:
            cur_site = structure_dict[coords]
            cur_site.properties['type'] = True
            sites_raw.append(cur_site)
        elif distance > 0 and distance < 12:
            cur_site = structure_dict[coords]
            cur_site.properties['type'] = False
            sites_raw.append(cur_site)
    return Structure.from_sites(sites_raw)


def add_state_imp2d(structure, unit_cell):
    reference_species = set(unit_cell.species)
    structure = structure.copy()
    structure.state = [sorted([element.Z for element in reference_species])]
    return structure


def convert_to_sparse_imp2d(structure, unit_cell, supercell_size, task, state,
                             skip_was=False, copy_unit_cell_properties=False):
    structure = structure.copy()
    if task == "cgcnn_hetero" or task == "megnet_hetero":
        structure = get_hetero_imp2d(structure, unit_cell, supercell_size, state)
    elif task == 'cgcnn_attention' or task == 'megnet_attention':
        structure = get_hetero_imp2d(structure, unit_cell, supercell_size, state)
    elif task == 'cgcnn_sparse' or task == 'megnet_sparse':
        structure = get_sparse_imp2d(structure, unit_cell, supercell_size)
    elif task == 'cgcnn_local' or task == 'megnet_local':
        structure = get_local_imp2d(structure, unit_cell, supercell_size, state)
    else:
        structure = get_full(structure, unit_cell, supercell_size, state)
    if not skip_was:
        structure = add_was(structure, unit_cell, supercell_size)
    if copy_unit_cell_properties:
        structure = add_unit_cell_properties(structure, unit_cell, supercell_size)
    if state is not None:
        structure = add_state_imp2d(structure, unit_cell)
    return structure


# ================================================================== #
#  Semi
# ================================================================== #

def get_sparse_semi(structure, unit_cell, supercell_size):
    base_species = {site.species_string for site in unit_cell}
    impurity_sites = [*filter(lambda x: x.species_string not in base_species, structure)]
    return Structure.from_sites(impurity_sites)


def get_hetero_semi(structure, unit_cell, supercell_size, state):
    structure = structure.copy()
    reference_supercell = unit_cell.copy()
    base_species = [site.species_string for site in reference_supercell]
    reference_supercell.make_supercell(supercell_size)
    sites_raw = []
    structure_dict = strucure_to_dict(structure)
    for coords, reference_site in structure_dict.items():
        cur_site = structure_dict[coords]
        if cur_site.species_string not in base_species:
            cur_site.properties['type'] = True
        else:
            cur_site.properties['type'] = False
        sites_raw.append(cur_site)
    return Structure.from_sites(sites_raw)


def get_local_semi(structure, unit_cell, supercell_size, state):
    structure = structure.copy()
    reference_supercell = unit_cell.copy()
    base_species = [site.species_string for site in reference_supercell]
    reference_supercell.make_supercell(supercell_size)
    sites_raw = []
    defect_idx = None
    for idx in range(len(structure)):
        if structure[idx].species_string not in base_species:
            defect_idx = idx

    structure_dict = strucure_to_dict(structure)
    for index, (coords, reference_site) in enumerate(structure_dict.items()):
        distance = structure.get_distance(index, defect_idx)
        if distance == 0:
            cur_site = structure_dict[coords]
            cur_site.properties['type'] = True
            sites_raw.append(cur_site)
        elif distance > 0 and distance < 12:
            cur_site = structure_dict[coords]
            cur_site.properties['type'] = False
            sites_raw.append(cur_site)
    return Structure.from_sites(sites_raw)


def add_state_semi(structure, unit_cell):
    reference_species = set(unit_cell.species)
    structure = structure.copy()
    structure.state = [sorted([element.Z for element in reference_species])]
    return structure


def convert_to_sparse_semi(structure, unit_cell, supercell_size, task, state,
                            skip_was=False, copy_unit_cell_properties=False):
    structure = structure.copy()
    unit_cell = unit_cell.copy()
    if task == "cgcnn_hetero" or task == "megnet_hetero":
        structure = get_hetero_semi(structure, unit_cell, supercell_size, state)
    elif task == 'cgcnn_attention' or task == 'megnet_attention':
        structure = get_hetero_semi(structure, unit_cell, supercell_size, state)
    elif task == 'cgcnn_sparse' or task == 'megnet_sparse':
        structure = get_sparse_semi(structure, unit_cell, supercell_size)
    elif task == 'cgcnn_local' or task == 'megnet_local':
        structure = get_local_semi(structure, unit_cell, supercell_size, state)
    else:
        structure = get_full(structure, unit_cell, supercell_size, state)
    if not skip_was:
        structure = add_was(structure, unit_cell, supercell_size)
    if copy_unit_cell_properties:
        structure = add_unit_cell_properties(structure, unit_cell, supercell_size)
    if state is not None:
        structure = add_state_semi(structure, unit_cell)
    return structure
