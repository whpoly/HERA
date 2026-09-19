"""Versioned native-defect preprocessing and original-site species labels.

Dataset_1 CIF labels retain the original zero-based site index even when
pymatgen sorts sites by element. The inserted/substituted atom has the highest
index. Vacancy structures are centered at fractional (1/2, 1/2, 1/2).
"""

import re
from pathlib import Path

from pymatgen.core import Element
from pymatgen.core.periodic_table import DummySpecies


NATIVE_REFERENCE_VERSION = 'reference_v1'
NATIVE_PREPROCESSING_CHOICES = ('legacy', NATIVE_REFERENCE_VERSION)


def native_run_components(config):
    version = config.get('native_preprocessing')
    return [f'native_{version}'] if version and version != 'legacy' else []


def parse_native_defect(source_id):
    parts = Path(str(source_id)).name.split('-')
    if len(parts) < 5:
        raise ValueError(f'Cannot parse native defect identity: {source_id!r}')
    material, label = parts[1:3]
    # The supplied GaSb/InSb interstitial families have this filename typo.
    # Require the actual CIF defect to be Sb below; never infer arbitrary typos.
    if material in ('GaSb', 'InSb') and label == 'Ab_i_A':
        label = 'Sb_i_A'
    tokens = label.split('_')
    try:
        if len(tokens) == 2 and tokens[0] == 'V':
            return 'vacancy', 0, Element(tokens[1]).Z
        if len(tokens) in (2, 3) and tokens[1] == 'i':
            return 'interstitial', Element(tokens[0]).Z, 0
        if len(tokens) == 2:
            return 'substitution', Element(tokens[0]).Z, Element(tokens[1]).Z
    except ValueError as exc:
        raise ValueError(f'Invalid native defect species: {source_id!r}') from exc
    raise ValueError(f'Unsupported native defect label: {source_id!r}')


def native_defect_index(structure, current_z, source_id):
    indices = []
    for site in structure:
        match = re.fullmatch(r'([A-Z][a-z]?)(\d+)', site.label)
        if match is None or match[1] != site.specie.symbol:
            raise ValueError(f'{source_id}: native CIF site-index labels are missing or invalid')
        indices.append(int(match[2]))
    if sorted(indices) != list(range(len(structure))):
        raise ValueError(f'{source_id}: native CIF site indices must be unique and contiguous from zero')
    defect_index = indices.index(len(structure) - 1)
    if structure[defect_index].specie.Z != current_z:
        raise ValueError(f'{source_id}: indexed defect {structure[defect_index].label} '
                         f'disagrees with the filename species (Z={current_z})')
    return defect_index


def native_reference_structure(structure, source_id, include_vacancy=True):
    """Preserve every real atom, mark the actual defect, and assign WAS.

    The zero reference for an interstitial denotes an originally empty site.
    A vacancy is an additional X node, not a replacement for a surviving atom.
    """
    kind, current_z, previous_z = parse_native_defect(source_id)
    result = structure.copy()
    for site in result:
        if isinstance(site.specie, DummySpecies):
            raise ValueError(f'{source_id}: expected a raw native CIF without X nodes')
        site.properties.update(was=site.specie.Z, type=False)
    if kind == 'vacancy':
        if include_vacancy:
            result.append(DummySpecies(), (.5, .5, .5),
                          properties={'was': previous_z, 'type': True})
    else:
        index = native_defect_index(result, current_z, source_id)
        result[index].properties.update(was=previous_z, type=True)
    result.add_site_property('native_preprocessing', [NATIVE_REFERENCE_VERSION] * len(result))
    return result
