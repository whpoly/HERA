"""Structure manipulation utilities for each dataset type.

Each dataset has its own legacy convert_to_sparse_* helper that produces the
appropriate graph representation (full / hetero / local / attention).
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


def copy_source_metadata(source, target):
    """Preserve original CIF identity through structure transformations."""
    if target is None:
        return None
    for attr in ("source_id", "source_name", "source_path"):
        if hasattr(source, attr):
            setattr(target, attr, getattr(source, attr))
    return target


def site_type_flag(site, default=False):
    raw = site.properties.get('type', default)
    if raw is None:
        return default
    if isinstance(raw, str):
        value = raw.strip().lower()
        if value in ('true', 't', 'yes'):
            return True
        if value in ('false', 'f', 'no', ''):
            return False
    return bool(int(raw))


def mark_local_region(structure, local_cutoff):
    if local_cutoff is None or structure is None:
        return structure

    structure = structure.copy()
    defect_indices = [
        idx for idx, site in enumerate(structure)
        if site_type_flag(site)
    ]
    if not defect_indices:
        return structure

    for idx, site in enumerate(structure):
        if idx in defect_indices:
            site.properties['type'] = True
            continue
        min_distance = min(structure.get_distance(idx, defect_idx) for defect_idx in defect_indices)
        site.properties['type'] = bool(local_cutoff > 0 and min_distance <= local_cutoff)
    return structure


def is_hetero_task(task):
    return (
        task.endswith('_hetero')
        or task.endswith('_hetero_was')
        or task.endswith('_hetero_fixed_pool')
        or task.endswith('_hetero_local')
        or task.endswith('_hetero_local_was')
        or task == 'hetero_cgcnn_was'
    )


def is_hetero_local_task(task):
    return (
        task.endswith('_hetero_local')
        or task.endswith('_hetero_local_was')
    )


def preserve_pool_type(structure):
    structure = structure.copy()
    for site in structure:
        site.properties['pool_type'] = int(site_type_flag(site))
    return structure


def mark_hetero_region_if_needed(structure, task, local_cutoff):
    structure = preserve_pool_type(structure)
    if is_hetero_local_task(task):
        return structure
    return mark_local_region(structure, local_cutoff)


def is_attention_task(task):
    return (
        task.endswith('_attention')
        or task.endswith('_attention_local')
        or task.endswith('_attention_was')
        or task.endswith('_attention_local_was')
    )


def is_sparse_task(task):
    return task.endswith('_sparse')


def is_local_task(task):
    return task.endswith('_local')


def is_full_x_task(task):
    return task.endswith('_full_x') or task.endswith('_was_x') or task == 'full_x' or task == 'was_x'


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


def add_vacancy_dummy_sites(structure, source_structure, unit_cell, supercell_size, include_was=False):
    structure = structure.copy()
    for site in structure:
        if site.properties.get('type') is None:
            site.properties['type'] = False
    reference_supercell = unit_cell.copy()
    reference_supercell.make_supercell(supercell_size)
    structure_dict = strucure_to_dict(source_structure)
    reference_structure_dict = strucure_to_dict(reference_supercell)
    for coords, reference_site in reference_structure_dict.items():
        if coords not in structure_dict:
            properties = {'type': True}
            if include_was:
                properties['was'] = reference_site.specie.Z
            structure.append(
                DummySpecies(),
                coords,
                coords_are_cartesian=False,
                properties=properties,
            )
    return structure


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
                               skip_was=False, copy_unit_cell_properties=False,
                               local_cutoff=None):
    source_structure = structure
    structure = structure.copy()
    unit_cell = unit_cell.copy()
    add_graph_vacancies = False
    if is_hetero_task(task):
        structure = get_hetero_vacancy(structure, unit_cell, supercell_size, state)
        structure = mark_hetero_region_if_needed(structure, task, local_cutoff)
    elif is_attention_task(task):
        structure = get_hetero_vacancy(structure, unit_cell, supercell_size, state)
        structure = mark_local_region(structure, local_cutoff)
    elif is_sparse_task(task):
        structure = get_sparse_vacancy(structure, unit_cell, supercell_size)
    else:
        structure = get_full(structure, unit_cell, supercell_size, state)
        add_graph_vacancies = is_full_x_task(task)
    if structure is None:
        return None
    if not skip_was:
        structure = add_was(structure, unit_cell, supercell_size)
    if add_graph_vacancies:
        structure = add_vacancy_dummy_sites(
            structure, source_structure, unit_cell, supercell_size, include_was=not skip_was
        )
    if copy_unit_cell_properties:
        structure = add_unit_cell_properties(structure, unit_cell, supercell_size)
    if state is not None:
        structure = add_state_vacancy(structure, unit_cell)
    return copy_source_metadata(source_structure, structure)


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
                                 skip_was=False, copy_unit_cell_properties=False,
                                 local_cutoff=None):
    source_structure = structure
    structure = structure.copy()
    unit_cell = unit_cell.copy()
    add_graph_vacancies = False
    if is_hetero_task(task):
        structure = get_hetero_2dmd_high(structure, unit_cell, supercell_size, state)
        structure = mark_hetero_region_if_needed(structure, task, local_cutoff)
    elif is_attention_task(task):
        structure = get_hetero_2dmd_high(structure, unit_cell, supercell_size, state)
        structure = mark_local_region(structure, local_cutoff)
    elif is_sparse_task(task):
        structure = get_sparse_2dmd_high(structure, unit_cell, supercell_size)
    else:
        structure = get_full(structure, unit_cell, supercell_size, state)
        add_graph_vacancies = is_full_x_task(task)
    if not skip_was:
        structure = add_was(structure, unit_cell, supercell_size)
    if add_graph_vacancies:
        structure = add_vacancy_dummy_sites(
            structure, source_structure, unit_cell, supercell_size, include_was=not skip_was
        )
    if copy_unit_cell_properties:
        structure = add_unit_cell_properties(structure, unit_cell, supercell_size)
    if state is not None:
        structure = add_state_2dmd_high(structure, unit_cell)
    return copy_source_metadata(source_structure, structure)


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


def add_native_vacancy_dummy_site(structure, unit_cell):
    if unit_cell != 'vacancy':
        return structure
    structure = structure.copy()
    structure.append(
        DummySpecies(),
        (0.5, 0.5, 0.5),
        coords_are_cartesian=False,
        properties={'type': True},
    )
    return structure


def get_local_native(structure, unit_cell, supercell_size, state, local_cutoff=5):
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
        if distance <= local_cutoff:
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
                              skip_was=False, copy_unit_cell_properties=False,
                              local_cutoff=None):
    source_structure = structure
    structure = structure.copy()
    if is_hetero_task(task):
        structure = get_hetero_native(structure, unit_cell, supercell_size, state)
        structure = mark_hetero_region_if_needed(structure, task, local_cutoff)
    elif is_attention_task(task):
        structure = get_hetero_native(structure, unit_cell, supercell_size, state)
        structure = mark_local_region(structure, local_cutoff)
    elif is_sparse_task(task):
        structure = get_sparse_native(structure, unit_cell, supercell_size)
    elif is_local_task(task):
        cutoff = 5 if local_cutoff is None else local_cutoff
        structure = get_local_native(structure, unit_cell, supercell_size, state, cutoff)
    else:
        structure = get_full(structure, unit_cell, supercell_size, state)
        if is_full_x_task(task):
            structure = add_native_vacancy_dummy_site(structure, unit_cell)
    if not skip_was:
        structure = add_was(structure, unit_cell, supercell_size)
    if copy_unit_cell_properties:
        structure = add_unit_cell_properties(structure, unit_cell, supercell_size)
    if state is not None:
        structure = add_state_native(structure, unit_cell)
    return copy_source_metadata(source_structure, structure)


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


def get_local_och(structure, unit_cell, supercell_size, local_cutoff=0):
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
        if distance <= local_cutoff:
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
                           skip_was=False, copy_unit_cell_properties=False,
                           local_cutoff=None):
    source_structure = structure
    structure = structure.copy()
    if is_hetero_task(task):
        structure = get_hetero_och(structure, unit_cell, supercell_size, state)
        structure = mark_hetero_region_if_needed(structure, task, local_cutoff)
    elif is_attention_task(task):
        structure = get_hetero_och(structure, unit_cell, supercell_size, state)
        structure = mark_local_region(structure, local_cutoff)
    elif is_sparse_task(task):
        structure = get_sparse_och(structure, unit_cell, supercell_size)
    elif is_local_task(task):
        cutoff = 0 if local_cutoff is None else local_cutoff
        structure = get_local_och(structure, unit_cell, supercell_size, cutoff)
    else:
        structure = get_full(structure, unit_cell, supercell_size, state)
    if not skip_was:
        structure = add_was(structure, unit_cell, supercell_size)
    if copy_unit_cell_properties:
        structure = add_unit_cell_properties(structure, unit_cell, supercell_size)
    if state is not None:
        structure = add_state_och(structure, state)
    return copy_source_metadata(source_structure, structure)


# ================================================================== #
#  imp2d
# ================================================================== #

def get_sparse_imp2d(structure, unit_cell, supercell_size):
    return get_sparse_och(structure, unit_cell, supercell_size)


def get_hetero_imp2d(structure, unit_cell, supercell_size, state):
    return get_hetero_och(structure, unit_cell, supercell_size, state)


def get_local_imp2d(structure, unit_cell, supercell_size, state, local_cutoff=0):
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
        if distance <= local_cutoff:
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
                             skip_was=False, copy_unit_cell_properties=False,
                             local_cutoff=None):
    source_structure = structure
    structure = structure.copy()
    if is_hetero_task(task):
        structure = get_hetero_imp2d(structure, unit_cell, supercell_size, state)
        structure = mark_hetero_region_if_needed(structure, task, local_cutoff)
    elif is_attention_task(task):
        structure = get_hetero_imp2d(structure, unit_cell, supercell_size, state)
        structure = mark_local_region(structure, local_cutoff)
    elif is_sparse_task(task):
        structure = get_sparse_imp2d(structure, unit_cell, supercell_size)
    elif is_local_task(task):
        cutoff = 0 if local_cutoff is None else local_cutoff
        structure = get_local_imp2d(structure, unit_cell, supercell_size, state, cutoff)
    else:
        structure = get_full(structure, unit_cell, supercell_size, state)
    if not skip_was:
        structure = add_was(structure, unit_cell, supercell_size)
    if copy_unit_cell_properties:
        structure = add_unit_cell_properties(structure, unit_cell, supercell_size)
    if state is not None:
        structure = add_state_imp2d(structure, unit_cell)
    return copy_source_metadata(source_structure, structure)


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


def get_local_semi(structure, unit_cell, supercell_size, state, local_cutoff=0):
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
        if distance <= local_cutoff:
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
                            skip_was=False, copy_unit_cell_properties=False,
                            local_cutoff=None):
    source_structure = structure
    structure = structure.copy()
    unit_cell = unit_cell.copy()
    if is_hetero_task(task):
        structure = get_hetero_semi(structure, unit_cell, supercell_size, state)
        structure = mark_hetero_region_if_needed(structure, task, local_cutoff)
    elif is_attention_task(task):
        structure = get_hetero_semi(structure, unit_cell, supercell_size, state)
        structure = mark_local_region(structure, local_cutoff)
    elif is_sparse_task(task):
        structure = get_sparse_semi(structure, unit_cell, supercell_size)
    elif is_local_task(task):
        cutoff = 0 if local_cutoff is None else local_cutoff
        structure = get_local_semi(structure, unit_cell, supercell_size, state, cutoff)
    else:
        structure = get_full(structure, unit_cell, supercell_size, state)
    if not skip_was:
        structure = add_was(structure, unit_cell, supercell_size)
    if copy_unit_cell_properties:
        structure = add_unit_cell_properties(structure, unit_cell, supercell_size)
    if state is not None:
        structure = add_state_semi(structure, unit_cell)
    return copy_source_metadata(source_structure, structure)
