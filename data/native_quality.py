"""Additional native geometry diagnostics and train-only family energy fences.

These are explicit screening heuristics, not DFT convergence tests.
"""
from collections import defaultdict
import re

from ase.data import covalent_radii
from ase.geometry import find_mic
import numpy as np


DEEP_POLICY = {
    'min_pair_covalent_ratio': 0.80,
    'max_nearest_covalent_ratio': 1.80,
    'family_min_train_samples': 4,
    'family_iqr_multiplier': 3.0,
    'family_iqr_floor_ev': 1.0,
    'joint_max_covalent_ratio': 0.90,
    'joint_max_family_ratio_fraction': 0.90,
    'joint_energy_sigma_multiplier': 6.0,
    'joint_energy_min_excess_ev': 1.0,
    'near_geometry_rms_angstrom': 0.015,
    'near_geometry_max_angstrom': 0.05,
    'near_geometry_energy_gap_ev': 1.0,
}


def configuration_family(source_id):
    return re.sub(r'-POSCAR\d+', '', source_id)


def geometry_features(atoms, distances, labels):
    """Use every real-atom pair; no artificial vacancy nodes participate."""
    radii = covalent_radii[atoms.numbers]
    if not np.isfinite(radii).all() or (radii <= 0).any():
        raise ValueError('Missing positive covalent radius for native geometry checks')
    normalized = distances / (radii[:, None] + radii[None, :])
    i, j = np.unravel_index(np.argmin(normalized), normalized.shape)
    nearest = normalized.min(axis=1)
    isolated = int(nearest.argmax())
    return {
        'min_pair_covalent_ratio': float(normalized[i, j]),
        'compressed_pair': f'{labels[i]}-{labels[j]}',
        'compressed_pair_distance_angstrom': float(distances[i, j]),
        'compressed_pair_radius_sum_angstrom': float(radii[i]+radii[j]),
        'max_nearest_covalent_ratio': float(nearest[isolated]),
        'most_isolated_atom': labels[isolated],
        'covalent_radii_used': {str(int(z)): float(covalent_radii[z]) for z in sorted(set(atoms.numbers))},
    }


def geometry_exclusions(row, policy):
    reasons = []
    if row.get('min_pair_covalent_ratio') is not None:
        if row['min_pair_covalent_ratio'] < policy['min_pair_covalent_ratio']:
            reasons.append('compressed_pair_relative_to_covalent_radii')
        if row['max_nearest_covalent_ratio'] > policy['max_nearest_covalent_ratio']:
            reasons.append('isolated_atom_relative_to_covalent_radii')
    return reasons


def fit_family_fences(records, train_ids, policy):
    """Same defect site family, with all POSCAR indices eligible as training data."""
    by_id = {r['source_id']: r for r in records}
    groups = defaultdict(list)
    for sid in train_ids:
        row = by_id[sid]
        if not row['quality_reasons'] and not geometry_exclusions(row, policy):
            groups[configuration_family(sid)].append(row)
    fences = {}
    for family, rows in sorted(groups.items()):
        if len(rows) < policy['family_min_train_samples']:
            fences[family] = {'n_train': len(rows), 'status': 'insufficient_train_samples'}
            continue
        values = np.asarray([r['target'] for r in rows])
        q1, q3 = np.quantile(values, [.25, .75], method='linear')
        width = policy['family_iqr_multiplier']*max(float(q3-q1), policy['family_iqr_floor_ev'])
        fences[family] = {'n_train': len(values), 'status': 'fitted',
                          'q1': float(q1), 'q3': float(q3),
                          'lower_ev': float(q1-width), 'upper_ev': float(q3+width)}
        ratios = [r['min_pair_covalent_ratio'] for r in rows if r.get('min_pair_covalent_ratio') is not None]
        if len(ratios) == len(rows):
            median = float(np.median(values))
            sigma = float(1.4826*np.median(np.abs(values-median)))
            fences[family].update(energy_median_ev=median, energy_mad_sigma_ev=sigma,
                                  min_pair_ratio_median=float(np.median(ratios)),
                                  joint_energy_upper_ev=median+max(policy['joint_energy_sigma_multiplier']*sigma,
                                                                  policy['joint_energy_min_excess_ev']))
    return fences


def joint_distortion_exclusion(row, fence, policy):
    """Require two geometric criteria and an independent energy departure."""
    ratio = row.get('min_pair_covalent_ratio')
    return bool(ratio is not None and 'joint_energy_upper_ev' in fence
                and ratio < policy['joint_max_covalent_ratio']
                and ratio < policy['joint_max_family_ratio_fraction']*fence['min_pair_ratio_median']
                and row['target'] > fence['joint_energy_upper_ev'])


def audit_family_geometry(records, atoms_by_id, policy):
    """Diagnostic only: comparisons across all frames never determine deletions.

    Compare indexed atoms after removing rigid translation, with periodic MIC.
    This is not a symmetry/permutation/rotation exhaustive duplicate search.
    """
    groups = defaultdict(list)
    for row in records:
        if row['source_id'] in atoms_by_id and row['target'] is not None:
            groups[configuration_family(row['source_id'])].append(row)
    pairs, anomalies = [], []
    for family, rows in sorted(groups.items()):
        rows.sort(key=lambda row: int(re.search(r'-POSCAR(\d+)', row['source_id'])[1]))
        reference = atoms_by_id[rows[0]['source_id']]
        compatible = all(np.array_equal(atoms_by_id[r['source_id']].numbers, reference.numbers)
                         and np.allclose(atoms_by_id[r['source_id']].cell.array,
                                         reference.cell.array, atol=1e-6, rtol=0) for r in rows)
        if not compatible:
            anomalies.append({'family': family, 'reason': 'cell_or_indexed_species_changes', 'n_frames': len(rows)})
            continue
        indices = list(zip(*np.triu_indices(len(rows), k=1)))
        for offset in range(0, len(indices), 128):
            chunk = indices[offset:offset+128]
            delta = np.array([atoms_by_id[rows[j]['source_id']].positions
                              - atoms_by_id[rows[i]['source_id']].positions for i, j in chunk])
            vectors, _ = find_mic(delta.reshape(-1, 3), reference.cell, reference.pbc)
            vectors = vectors.reshape(delta.shape)
            vectors -= vectors.mean(axis=1, keepdims=True)
            _, lengths = find_mic(vectors.reshape(-1, 3), reference.cell, reference.pbc)
            lengths = lengths.reshape(len(chunk), len(reference))
            rms = np.sqrt(np.mean(lengths**2, axis=1))
            maximum = lengths.max(axis=1)
            for k, (i, j) in enumerate(chunk):
                left, right = rows[i], rows[j]
                gap = float(right['target']-left['target'])
                near = (rms[k] <= policy['near_geometry_rms_angstrom']
                        and maximum[k] <= policy['near_geometry_max_angstrom'])
                adjacent = j == i+1  # Adjacent available files, not verified ionic steps.
                if adjacent or near:
                    pairs.append({'family': family, 'left_id': left['source_id'], 'right_id': right['source_id'],
                                  'adjacent_available_frames': adjacent,
                                  'rms_angstrom': float(rms[k]), 'max_angstrom': float(maximum[k]),
                                  'energy_change_ev': gap,
                                  'near_geometry_energy_conflict': bool(near and abs(gap) >= policy['near_geometry_energy_gap_ev'])})
    return pairs, anomalies
