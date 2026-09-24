"""Conservative physical screening for IMP2D and extrinsic semiconductors.

No prediction residuals or distribution-fitted target fences are used here.
Radii are screening heuristics, not a proof of DFT or thermodynamic validity.
"""
from collections import Counter
import math
import re

import numpy as np
from ase.data import atomic_numbers, covalent_radii
from ase.formula import Formula
from ase.neighborlist import neighbor_list


VERSION = 'impurity_physical_v1'
POLICY = {
    'version': VERSION,
    'severe_pair_covalent_ratio': 0.5,
    'review_pair_covalent_ratio': 0.8,
    'review_isolated_covalent_ratio': 1.8,
    'imp2d_final_energy_change_max_ev': 0.05,
    'imp2d_expansion_factor_max': 2.0,
    'energy_identity_tolerance_ev': 1e-4,
    'database_geometry_tolerance_angstrom': 1e-3,
    'statistical_energy_filter': False,
    'imp2d_legacy_energy_window_ev': [-10, 10],
}


def finite(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def matches_formula(counts, formula):
    counts = +Counter(counts)
    expected = Formula(formula).count()
    return (set(counts) == set(expected) and bool(counts)
            and np.allclose([counts[k]/expected[k] for k in counts],
                            next(iter(counts.values()))/expected[next(iter(counts))],
                            rtol=0, atol=1e-8))


def inspect_geometry(atoms):
    """Include periodic images, even same-atom images in very small cells."""
    result = {'quality_reasons': [], 'review_flags': []}
    if (len(atoms) < 2 or not np.isfinite(atoms.positions).all()
            or not np.isfinite(atoms.cell.array).all()
            or abs(np.linalg.det(atoms.cell.array)) < 1e-8
            or (atoms.numbers <= 0).any()):
        result['quality_reasons'].append('invalid_cell_coordinates_or_species')
        return result
    radii = covalent_radii[atoms.numbers]
    if not np.isfinite(radii).all() or (radii <= 0).any():
        result['quality_reasons'].append('undefined_atomic_radius')
        return result
    # This search covers every pair below the rejection/review thresholds;
    # no hard absolute-distance cutoff that would reject normal H bonds.
    i, j, d, shifts = neighbor_list('ijdS', atoms, radii*POLICY['review_isolated_covalent_ratio'])
    result['atom_count'] = len(atoms)
    result['isolated_atom_indices_review_only'] = sorted(set(range(len(atoms)))-set(i.tolist()))
    if result['isolated_atom_indices_review_only']:
        result['review_flags'].append('weakly_bound_or_isolated_atoms_review_only')
    if len(d):
        q = d/(radii[i]+radii[j])
        k = int(np.argmin(q))
        symbols = atoms.get_chemical_symbols()
        result.update(min_pair_covalent_ratio=float(q[k]),
                      compressed_pair=f'{symbols[i[k]]}{i[k]}-{symbols[j[k]]}{j[k]}',
                      compressed_pair_distance_angstrom=float(d[k]),
                      compressed_pair_image_shift=shifts[k].tolist())
        if q[k] < POLICY['severe_pair_covalent_ratio']:
            result['quality_reasons'].append('severe_atomic_overlap')
        elif q[k] < POLICY['review_pair_covalent_ratio']:
            result['review_flags'].append('short_bond_review_only')
    else:
        result['min_pair_covalent_ratio'] = None  # All pairs >= search threshold.
    return result


def inspect_imp2d_metadata(metadata):
    reasons, flags = [], []
    result = {'quality_reasons': reasons, 'review_flags': flags}
    for key in ('eform', 'conv1', 'conv2', 'extension_factor', 'en2', 'hostenergy',
                'dopant_chemical_potential', 'depth'):
        result[key] = finite(metadata.get(key))
    result['reported_converged'] = bool(metadata.get('converged', False))
    result['host_spacegroup'] = metadata.get('host_spacegroup')
    if result['eform'] is None:
        reasons.append('nonfinite_formation_energy')
    conv2 = result['conv2']
    if conv2 is None:
        reasons.append('missing_final_convergence')
    elif abs(conv2) > POLICY['imp2d_final_energy_change_max_ev']:
        reasons.append('final_ionic_energy_not_converged')
    final_ok = conv2 is not None and abs(conv2) <= POLICY['imp2d_final_energy_change_max_ev']
    if final_ok != result['reported_converged']:
        flags.append('source_converged_flag_differs_from_final_stage')
    xf = result['extension_factor']
    if xf is None or xf <= 0:
        reasons.append('invalid_expansion_factor')
    elif xf > POLICY['imp2d_expansion_factor_max']:
        reasons.append('excessive_reconstruction_outside_point_defect_scope')
    terms = [result[k] for k in ('eform', 'en2', 'hostenergy', 'dopant_chemical_potential')]
    if any(x is None for x in terms):
        reasons.append('missing_energy_reference')
    else:
        residual = terms[0]-(terms[1]-terms[2]-terms[3])
        result['energy_identity_residual_ev'] = residual
        if abs(residual) > POLICY['energy_identity_tolerance_ev']:
            reasons.append('inconsistent_energy_metadata')
    depth = result['depth']
    if depth is not None:
        result['relaxed_site_from_depth'] = 'int' if abs(depth) < 1 else 'ads'
        if not str(metadata.get('site', '')).startswith(result['relaxed_site_from_depth']):
            flags.append('site_migration_allowed')
    return result


def inspect_impurity_composition(atoms, source_id, dataset):
    reasons, flags = [], []
    result = {'quality_reasons': reasons, 'review_flags': flags}
    counts = Counter(atoms.get_chemical_symbols())
    try:
        if dataset == 'imp2d':
            base, impurity, site = source_id.split('_')
            if re.fullmatch(r'(ads|int)\d+', site) is None:
                raise ValueError('Unknown imp2d site')
            host_counts = counts-Counter({impurity: 1})
            if counts[impurity] < 1 or not matches_formula(host_counts, base):
                reasons.append('impurity_host_stoichiometry_mismatch')
            previous = 0
            if impurity in Formula(base).count():
                flags.append('self_impurity_requires_checked_site_label')
        else:
            parts = source_id.split('-')
            if len(parts) != 5:
                raise ValueError('Invalid semi filename')
            _, base, site, impurity, _ = parts
            if site not in ('M_A', 'M_B', 'M_i_A', 'M_i_B', 'M_i_neut'):
                raise ValueError('Unknown semi site')
            host_species = set(Formula(base).count())
            extrinsic = {k: v for k, v in counts.items() if k not in host_species}
            result['baseline_eligible'] = bool(extrinsic)
            previous = None
            if extrinsic != {impurity: 1}:
                reasons.append('expected_exactly_one_extrinsic_impurity')
            else:
                host_counts = counts-Counter({impurity: 1})
                if site.startswith('M_i_'):
                    previous = 0
                    if not matches_formula(host_counts, base):
                        reasons.append('interstitial_host_stoichiometry_mismatch')
                else:
                    candidates = [x for x in host_species
                                  if matches_formula(host_counts+Counter({x: 1}), base)]
                    if len(candidates) != 1:
                        reasons.append('ambiguous_or_invalid_substitution_stoichiometry')
                    else:
                        previous = atomic_numbers[candidates[0]]
        result.update(material=base, impurity=impurity, site=site, previous_z=previous)
    except (ValueError, KeyError) as exc:
        reasons.append('invalid_defect_description')
        result['description_error'] = str(exc)
    return result


def combine_checks(*checks):
    result = {'quality_reasons': [], 'review_flags': []}
    for check in checks:
        for key, value in check.items():
            if key in ('quality_reasons', 'review_flags'):
                result[key].extend(x for x in value if x not in result[key])
            else:
                result[key] = value
    return result
