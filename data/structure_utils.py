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
    if structure is None:
        return None
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

def get_och_adsorbate_index(structure, adsorbate_species):
    """Return the adsorbed atom index used by the OCH dataset.

    OCH CIFs preserve the dataset convention that the newly adsorbed atom is
    the last site of its species.  This matters for hydrogen-containing hosts:
    selecting every H atom would incorrectly classify host hydrogen as defect
    atoms.
    """
    adsorbate_indices = [
        idx for idx, site in enumerate(structure)
        if site.species_string == adsorbate_species
    ]
    if not adsorbate_indices:
        source_id = getattr(structure, "source_id", "<unknown>")
        raise ValueError(
            f"OCH structure {source_id} has no {adsorbate_species} adsorbate candidate"
        )
    return adsorbate_indices[-1]


def get_sparse_och(structure, unit_cell, supercell_size):
    adsorbate_idx = get_och_adsorbate_index(structure, unit_cell)
    return Structure.from_sites([structure[adsorbate_idx]])


def get_hetero_och(structure, unit_cell, supercell_size, state):
    structure = structure.copy()
    base_species = unit_cell
    adsorbate_idx = get_och_adsorbate_index(structure, base_species)
    sites_raw = []
    for idx, cur_site in enumerate(structure):
        cur_site.properties['type'] = idx == adsorbate_idx
        sites_raw.append(cur_site)
    return Structure.from_sites(sites_raw)


def get_local_och(structure, unit_cell, supercell_size, local_cutoff=0):
    structure = structure.copy()
    sites_raw = []
    base_species = unit_cell
    defect_idx = get_och_adsorbate_index(structure, base_species)

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

def get_imp2d_defect_info(unit_cell):
    if isinstance(unit_cell, dict):
        return unit_cell
    return {'impurity': unit_cell, 'is_self': False}


def resolve_imp2d_self_defect_index(structure, impurity, reference_frac_coords):
    """Match a self-impurity atom to non-self structures at the same site.

    CIF parsing does not preserve a reliable insertion order, so a self-defect
    cannot be identified by taking the first or last atom.  Non-self imp2d
    structures contain exactly one atom of the impurity species and therefore
    provide unambiguous reference coordinates for the same ``(base, site)``.
    """
    candidate_indices = [
        idx
        for idx, site in enumerate(structure)
        if site.species_string == impurity
    ]
    if not candidate_indices:
        raise ValueError(
            f"Cannot locate self-defect {impurity}: the structure contains no "
            "candidate atom of that species"
        )

    references = np.asarray(reference_frac_coords, dtype=float).reshape(-1, 3)
    if references.size == 0:
        raise ValueError(
            f"Cannot locate self-defect {impurity}: no non-self reference "
            "coordinates were provided"
        )

    candidates = np.asarray(
        [structure[idx].frac_coords for idx in candidate_indices],
        dtype=float,
    )
    fractional_deltas = candidates[:, None, :] - references[None, :, :]
    fractional_deltas -= np.round(fractional_deltas)
    cartesian_deltas = fractional_deltas @ np.asarray(
        structure.lattice.matrix,
        dtype=float,
    )
    distances = np.linalg.norm(cartesian_deltas, axis=-1)
    # The median is insensitive to a minority of reference impurities that
    # relax away from the nominal adsorption/interstitial site.
    scores = np.median(distances, axis=1)
    return candidate_indices[int(np.argmin(scores))]


def get_imp2d_defect_indices(structure, unit_cell):
    defect_info = get_imp2d_defect_info(unit_cell)
    if defect_info.get('is_self'):
        defect_index = defect_info.get('defect_index')
        if defect_index is None:
            reference_frac_coords = defect_info.get('reference_frac_coords')
            if reference_frac_coords is None:
                raise ValueError(
                    "Self-impurity defects require an explicit defect_index or "
                    "same-site non-self reference coordinates"
                )
            defect_index = resolve_imp2d_self_defect_index(
                structure,
                defect_info['impurity'],
                reference_frac_coords,
            )
        defect_index = int(defect_index)
        if defect_index < 0 or defect_index >= len(structure):
            raise ValueError(
                f"Self-defect index {defect_index} is outside a structure with "
                f"{len(structure)} sites"
            )
        selected_species = structure[defect_index].species_string
        if selected_species != defect_info['impurity']:
            raise ValueError(
                f"Self-defect index {defect_index} selects {selected_species}, "
                f"expected {defect_info['impurity']}"
            )
        return {defect_index}

    impurity = defect_info['impurity']
    return {
        idx
        for idx, site in enumerate(structure)
        if site.species_string == impurity
    }


def get_sparse_imp2d(structure, unit_cell, supercell_size):
    defect_indices = get_imp2d_defect_indices(structure, unit_cell)
    return Structure.from_sites([
        site for idx, site in enumerate(structure)
        if idx in defect_indices
    ])


def get_hetero_imp2d(structure, unit_cell, supercell_size, state):
    structure = structure.copy()
    defect_indices = get_imp2d_defect_indices(structure, unit_cell)
    sites_raw = []
    for idx, site in enumerate(structure):
        site.properties['type'] = idx in defect_indices
        sites_raw.append(site)
    return Structure.from_sites(sites_raw)


def get_local_imp2d(structure, unit_cell, supercell_size, state, local_cutoff=0):
    structure = structure.copy()
    sites_raw = []
    defect_indices = get_imp2d_defect_indices(structure, unit_cell)
    if not defect_indices:
        return Structure.from_sites(sites_raw)

    structure_dict = strucure_to_dict(structure)
    for index, (coords, reference_site) in enumerate(structure_dict.items()):
        distance = min(structure.get_distance(index, defect_idx) for defect_idx in defect_indices)
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
