"""Read-only ASE audit of native structures, targets and high-error test cases.

This identifies internal inconsistencies and geometric extremes. It cannot
verify DFT electronic convergence without the original electronic-structure logs.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import time
import warnings

from ase.geometry import find_mic
from ase.io import read as ase_read
from ase.formula import Formula
from ase.data import atomic_numbers
import numpy as np
import pandas as pd

from ..data.native_was import parse_native_defect
from ..training.trainer import load_trusted_checkpoint


ROOT = Path(__file__).resolve().parents[1]


def family(sid):
    return re.sub(r'-POSCAR\d+', '', sid)


def geometry_hash(atoms):
    frac = np.round(atoms.get_scaled_positions(wrap=True), 8) % 1.0
    sites = sorted((int(z), *xyz) for z, xyz in zip(atoms.numbers, frac))
    value = {'cell': np.round(atoms.cell.array, 8).tolist(), 'sites': sites}
    return hashlib.sha256(json.dumps(value).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, default=ROOT.parent/'dataset/Dataset_1')
    parser.add_argument('--predictions', type=Path, default=ROOT/'results/native_val_test_audit_20260922/paired_errors.csv')
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'logs/alignn_ref_v2/alignn/native/hetero_was_no_dd/seed123_best_checkpoint.pth')
    parser.add_argument('--output', type=Path, default=ROOT/'results/native_outlier_audit_20260922')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data_dir = args.dataset_root/'Dataset_1/A_rich/Neutral'
    source = pd.read_csv(data_dir/'id_prop_A_rich.csv', header=None, names=['source_id', 'target'])
    if source.source_id.duplicated().any() or not np.isfinite(source.target).all():
        raise ValueError('Duplicate source IDs or invalid labels')
    c = load_trusted_checkpoint(args.checkpoint, 'cpu')
    splits = {r['source_id']: split for split in ['train', 'val', 'test'] for r in c['split'][split+'_sources']}
    if set(splits) != set(source.source_id):
        raise ValueError('Saved split does not exactly cover local dataset')
    predictions = pd.read_csv(args.predictions)
    test = predictions[predictions.split == 'test'].copy()
    test['shared_error'] = (test.absolute_error_hetero+test.absolute_error_attention)/2
    test = test.sort_values('shared_error', ascending=False)
    top_ids = test.source_id.head(50).tolist()
    focus_families = {family(sid) for sid in top_ids}
    hosts = {}
    host_sources = {}
    structures = {}
    rows, failures = [], []
    started = time.monotonic()
    for index, row in enumerate(source.itertuples(index=False)):
        sid, target = row
        material = sid.split('-')[1]
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                atoms = ase_read(str(data_dir/sid), format='cif', store_tags=True)
            labels = atoms.info['_atom_site_label']
            indices = np.array([int(re.fullmatch(r'[A-Z][a-z]?(\d+)', label)[1]) for label in labels])
            labels_valid = len(labels) == len(atoms) and sorted(indices.tolist()) == list(range(len(atoms)))
            symbols_valid = all(re.fullmatch(r'([A-Z][a-z]?)\d+', label)[1] == sym
                                for label, sym in zip(labels, atoms.get_chemical_symbols()))
            positions_finite = bool(np.isfinite(atoms.positions).all() and np.isfinite(atoms.cell.array).all())
            if not (positions_finite and atoms.get_volume() > 0 and labels_valid and symbols_valid):
                raise ValueError('Invalid cell, coordinates or CIF atom indices')
            kind, current_z, previous_z = parse_native_defect(sid)
            if material not in hosts:
                host_path = args.dataset_root/f'host_configurations/{material}.vasp'
                if host_path.is_file() and host_path.stat().st_size:
                    host = ase_read(str(host_path), format='vasp')
                    hosts[material] = Counter(host.numbers.tolist())
                    host_sources[material] = 'host_vasp'
                else:
                    # A missing pristine reference must not skip valid defect CIFs.
                    # Formula fallback checks only stoichiometry, not host geometry.
                    hosts[material] = Counter({atomic_numbers[k]: v for k, v in Formula(material).count().items()})
                    host_sources[material] = 'filename_formula_missing_host_vasp'
            base_n = len(atoms)+(1 if kind == 'vacancy' else (-1 if kind == 'interstitial' else 0))
            multiplier = base_n/sum(hosts[material].values())
            expected = Counter({z: int(round(n*multiplier)) for z, n in hosts[material].items()})
            if previous_z:
                expected[previous_z] -= 1
            if current_z:
                expected[current_z] += 1
            expected = +expected
            counts_match = Counter(atoms.numbers.tolist()) == expected
            distances = atoms.get_all_distances(mic=True)
            np.fill_diagonal(distances, np.inf)
            i, j = np.unravel_index(np.argmin(distances), distances.shape)
            defect_index = int(np.argmax(indices)) if kind != 'vacancy' else None
            deficit_z_ok = defect_index is None or int(atoms.numbers[defect_index]) == current_z
            occupancy = np.array(atoms.info['_atom_site_occupancy'], dtype=float)
            rec = {
                'source_id': sid, 'split': splits[sid], 'material': material,
                'defect_type': kind, 'family': family(sid),
                'poscar_index': int(re.search(r'-POSCAR(\d+)', sid)[1]),
                'target': target, 'n_atoms': len(atoms), 'formula': atoms.get_chemical_formula(),
                'volume': atoms.get_volume(), 'labels_valid': labels_valid and symbols_valid,
                'composition_matches_host_and_defect': counts_match,
                'host_composition_source': host_sources[material],
                'defect_species_matches': deficit_z_ok,
                'occupancies_one': bool(np.all(occupancy == 1)),
                'min_distance_angstrom': float(distances[i, j]),
                'closest_pair': f'{labels[i]}-{labels[j]}',
                'closest_pair_involves_defect': defect_index in (i, j),
                'defect_nearest_distance': float(distances[defect_index].min()) if defect_index is not None else None,
                'geometry_hash': geometry_hash(atoms),
                'read_warnings': '; '.join(sorted({str(w.message) for w in caught})),
            }
            if defect_index is not None:
                rec['closest_pair_nearest_defect_distance'] = float(min(distances[defect_index, i], distances[defect_index, j])) if defect_index not in (i, j) else 0.0
            rows.append(rec)
            if rec['family'] in focus_families:
                structures[sid] = atoms[np.argsort(indices)]
        except Exception as exc:
            failures.append({'source_id': sid, 'error': repr(exc)})
        if (index+1) % 500 == 0:
            print(f'ASE audit {index+1}/{len(source)}, {time.monotonic()-started:.1f}s', flush=True)
    audit = pd.DataFrame(rows)
    audit.to_csv(args.output/'all_structure_checks.csv', index=False)
    (args.output/'parse_failures.json').write_text(json.dumps(failures, indent=2), encoding='utf-8')
    if failures:
        raise RuntimeError(f'{len(failures)} structures failed; inspect parse_failures.json before comparing sample ranks')

    comparisons = []
    for fam, group in audit[audit.family.isin(focus_families)].groupby('family'):
        group = group.sort_values('poscar_index')
        last = group.iloc[-1]
        for previous, current in zip(group.iloc[:-1].itertuples(), group.iloc[1:].itertuples()):
            a, b = structures[previous.source_id], structures[current.source_id]
            if not np.array_equal(a.numbers, b.numbers):
                raise ValueError(f'Original atom indices change species in {fam}')
            _, displacements = find_mic(b.positions-a.positions, a.cell, pbc=a.pbc)
            comparisons.append({
                'family': fam, 'source_id': current.source_id, 'previous_id': previous.source_id,
                'energy_change_ev': current.target-previous.target,
                'indexed_rms_displacement_angstrom': float(np.sqrt(np.mean(displacements**2))),
                'max_displacement_angstrom': float(displacements.max()),
            })
        for row in group.itertuples():
            if row.source_id not in top_ids:
                continue
            index = audit.index[audit.source_id == row.source_id][0]
            audit.loc[index, 'same_family_last_id'] = last.source_id
            audit.loc[index, 'same_family_last_target'] = last.target
            audit.loc[index, 'same_family_last_min_distance'] = last.min_distance_angstrom
            train = group[group.split == 'train']
            audit.loc[index, 'same_family_train_n'] = len(train)
            if len(train):
                audit.loc[index, 'same_family_train_target_min'] = train.target.min()
                audit.loc[index, 'same_family_train_target_max'] = train.target.max()
                candidates = []
                for r in train.itertuples():
                    a, b = structures[row.source_id], structures[r.source_id]
                    if not np.array_equal(a.numbers, b.numbers):
                        continue
                    _, d = find_mic(b.positions-a.positions, a.cell, pbc=a.pbc)
                    candidates.append((float(np.sqrt(np.mean(d**2))), r.source_id, r.target, float(d.max())))
                if candidates:
                    rms, sid, target, maximum = min(candidates)
                    audit.loc[index, 'nearest_indexed_train_id'] = sid
                    audit.loc[index, 'nearest_indexed_train_target'] = target
                    audit.loc[index, 'nearest_indexed_train_rms'] = rms
                    audit.loc[index, 'nearest_indexed_train_max_displacement'] = maximum
    audit.to_csv(args.output/'all_structure_checks.csv', index=False)
    pd.DataFrame(comparisons).to_csv(args.output/'family_adjacent_changes.csv', index=False)
    enriched = test.merge(audit.drop(columns=['split', 'material', 'defect_type']), on='source_id', validate='one_to_one')
    enriched['error_rank'] = np.arange(1, len(enriched)+1)
    enriched.head(50).to_csv(args.output/'top50_test_audit.csv', index=False)
    audit[audit.family.isin(focus_families)].merge(
        predictions[['source_id', 'prediction_hetero', 'prediction_attention']], on='source_id', how='left',
    ).sort_values(['family', 'poscar_index']).to_csv(args.output/'focus_family_series.csv', index=False)

    duplicates = audit.groupby('geometry_hash').filter(lambda g: len(g) > 1)
    duplicates.to_csv(args.output/'duplicate_geometry_candidates.csv', index=False)
    duplicate_stats = duplicates.groupby('geometry_hash').target.agg(['size', 'min', 'max'])
    if len(duplicate_stats):
        duplicate_stats['target_span'] = duplicate_stats['max']-duplicate_stats['min']
    b_path = args.dataset_root/'Dataset_1/B_rich/Neutral/id_prop_B_rich.csv'
    cross = {}
    if b_path.is_file() and b_path.stat().st_size:
        b = pd.read_csv(b_path, header=None, names=['source_id', 'b_target'])
        overlap = source.merge(b, on='source_id', validate='one_to_one')
        overlap['family'] = overlap.source_id.map(family)
        overlap['a_minus_b'] = overlap.target-overlap.b_target
        grouped = overlap.groupby('family').a_minus_b.agg(['size', 'min', 'max'])
        cross = {'overlap': len(overlap), 'families': len(grouped),
                 'max_within_family_offset_range_ev': float((grouped['max']-grouped['min']).max()),
                 'top10_overlap': sum(sid in set(overlap.source_id) for sid in top_ids[:10]),
                 'scope': 'Internal offset consistency only; A-rich and B-rich are not independent DFT calculations.'}
        overlap.to_csv(args.output/'a_b_rich_label_comparison.csv', index=False)
    groups = {}
    for title, part in [('all', audit), ('train', audit[audit.split == 'train']),
                        ('test', audit[audit.split == 'test']), ('top10_test', enriched.head(10)),
                        ('top20_test', enriched.head(20)), ('top50_test', enriched.head(50))]:
        groups[title] = {
            'n': len(part), 'min_distance': float(part.min_distance_angstrom.min()),
            'distance_below_1_angstrom': int((part.min_distance_angstrom < 1).sum()),
            'overlap_below_0_1_angstrom': int((part.min_distance_angstrom < .1).sum()),
            'poscar_0_or_1': int((part.poscar_index <= 1).sum()),
            'composition_mismatch': int((~part.composition_matches_host_and_defect).sum()),
            'defect_species_mismatch': int((~part.defect_species_matches).sum()),
        }
    summary = {
        'source_rows': len(source), 'failures': failures, 'groups': groups,
        'host_composition_sources': host_sources,
        'duplicate_geometry_groups_at_export_precision': len(duplicate_stats),
        'duplicate_geometry_groups_with_target_span_over_0_01_ev': int((duplicate_stats.target_span > .01).sum()) if len(duplicate_stats) else 0,
        'a_b_rich_consistency': cross,
        'raw_dft_logs_found': [str(p) for pattern in ['*OUTCAR*', '*vasprun*', '*OSZICAR*']
                               for p in args.dataset_root.rglob(pattern) if p.is_file() and p.stat().st_size],
        'limitations': ['Geometry duplicate check does not exhaust translations, rotations or crystal symmetries.',
                        'Adjacent POSCAR indices are file ordering, not verified DFT optimization step metadata.',
                        'No label corrections, sample deletions or training were performed.'],
    }
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({**summary, 'failures': len(failures)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
