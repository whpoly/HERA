"""Original-site species for extrinsic semiconductor and imp2d impurities."""

from collections import Counter
from pathlib import Path
import re

import numpy as np
from pymatgen.core import Composition, Element


REFERENCE_VERSION = 'reference_v1'
IMPURITY_DATASETS = ('imp2d', 'semi')


def impurity_run_components(config):
    return [f'{dataset}_{config[f"{dataset}_preprocessing"]}'
            for dataset in IMPURITY_DATASETS
            if config.get(f'{dataset}_preprocessing') not in (None, 'legacy')]


def _matches_host(counts, host):
    """Check stoichiometry without assuming a particular conventional cell size."""
    counts = {element: count for element, count in counts.items() if count}
    reference = host.get_el_amt_dict()
    if set(counts) != set(reference):
        return False
    ratios = [counts[element] / amount for element, amount in reference.items()]
    return bool(ratios) and min(ratios) > 0 and np.allclose(ratios, ratios[0], rtol=0, atol=1e-8)


def _annotate(structure, defect_index, previous_z, dataset):
    result = structure.copy()
    for index, site in enumerate(result):
        site.properties.update(was=previous_z if index == defect_index else site.specie.Z,
                               type=index == defect_index,
                               impurity_preprocessing=f'{dataset}_{REFERENCE_VERSION}')
    return result


def imp2d_reference_structure(structure, defect_info, defect_indices):
    source_id = getattr(structure, 'source_id', '<unknown>')
    if not re.fullmatch(r'(ads|int)\d+', str(defect_info.get('site', ''))):
        raise ValueError(f'{source_id}: true imp2d WAS requires a documented ads/int site')
    if len(defect_indices) != 1:
        raise ValueError(f'{source_id}: expected exactly one imp2d impurity, found {len(defect_indices)}')
    index = next(iter(defect_indices))
    if structure[index].specie.symbol != defect_info['impurity']:
        raise ValueError(f'{source_id}: imp2d impurity species mismatch')
    host_counts = Counter(site.specie.symbol for i, site in enumerate(structure) if i != index)
    if not _matches_host(host_counts, Composition(defect_info['base'])):
        raise ValueError(f'{source_id}: removing the added impurity does not recover host stoichiometry')
    # Both adsorption and interstitial insertion start from an unoccupied site.
    return _annotate(structure, index, 0, 'imp2d')


def semi_reference_structure(structure, host):
    source_id = getattr(structure, 'source_id', getattr(structure, 'source_path', '<unknown>'))
    parts = Path(str(source_id)).name.split('-')
    if len(parts) != 5:
        raise ValueError(f'{source_id}: invalid semi filename; expected ID-host-site-impurity-POSCAR.cif')
    _, base, kind, impurity, _ = parts
    if kind not in ('M_A', 'M_B', 'M_i_A', 'M_i_B', 'M_i_neut'):
        raise ValueError(f'{source_id}: unsupported semi defect type {kind!r}')
    if not _matches_host(host.composition.get_el_amt_dict(), Composition(base)):
        raise ValueError(f'{source_id}: host file disagrees with filename composition')
    host_species = set(host.composition.get_el_amt_dict())
    indices = [i for i, site in enumerate(structure) if site.specie.symbol not in host_species]
    if len(indices) != 1 or structure[indices[0]].specie.symbol != impurity:
        raise ValueError(f'{source_id}: expected exactly one extrinsic {impurity} impurity')
    index = indices[0]
    counts = Counter(site.specie.symbol for i, site in enumerate(structure) if i != index)
    if kind.startswith('M_i_'):
        if not _matches_host(counts, host.composition):
            raise ValueError(f'{source_id}: interstitial host stoichiometry mismatch')
        previous_z = 0
    else:
        # Recover the removed species from atom counts, without guessing which
        # element A/B denotes or matching relaxed coordinates to a primitive cell.
        candidates = [symbol for symbol in host_species
                      if _matches_host(counts + Counter({symbol: 1}), host.composition)]
        if len(candidates) != 1:
            raise ValueError(f'{source_id}: original substitution species is not uniquely determined')
        previous_z = Element(candidates[0]).Z
    return _annotate(structure, index, previous_z, 'semi')
