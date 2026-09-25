"""Audit frozen splits for repeated structures and conflicting labels; no filtering."""
import argparse
from collections import Counter, defaultdict
import itertools
import json
from pathlib import Path
import warnings

from ase.db import connect
from ase.io import read
import numpy as np
import pandas as pd
from pymatgen.io.ase import AseAtomsAdaptor

from ..data.structure_groups import anchor_distances, equivalent
from ..data.impurity_preprocessing import (DATA_DIRS, default_database, digest, identity,
                                          source_path, validate_manifest)

ROOT = Path(__file__).resolve().parents[1]
DISTANCE_TOLERANCE = 1e-3
ENERGY_TOLERANCE = 0.05


def audit(dataset, run_dir, data_root, output):
    manifest = validate_manifest(json.loads((run_dir/f'{dataset}_filter_manifest.json').read_text()))
    if len(manifest['splits']) != 1:
        raise ValueError('Specify a run with one split for this audit')
    split = manifest['splits'][0]
    membership = {sid: part for part in ('train', 'val', 'test') for sid in split['kept'][part]}
    all_ids = [sid for values in split['kept'].values() for sid in values]
    if len(all_ids) != len(membership):
        raise ValueError('The same source ID appears in multiple splits')
    selected = [row for row in manifest['records'] if row['source_id'] in membership]
    folder = data_root/Path(DATA_DIRS[dataset]).relative_to('dataset')
    db_rows = {}
    if dataset == 'imp2d':
        database = default_database(folder)
        if database is None or digest(database) != manifest['database_sha256']:
            raise ValueError('IMP2D database differs from the frozen physical-filter source')
        with connect(str(database)) as db:
            db_rows = {row.name: row for row in db.select()}
    groups = defaultdict(list)
    for index, row in enumerate(selected, 1):
        path = source_path(folder, dataset, row['source_id'])
        if manifest.get('source_format') != 'ase_database' and digest(path) != row['sha256']:
            raise ValueError(f'CIF differs from frozen source: {path}')
        if dataset == 'imp2d':
            atoms = db_rows[row['source_id']].toatoms()
        else:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                atoms = read(str(path), format='cif')
        composition = tuple(sorted(Counter(atoms.numbers).items()))
        key = (row['material'], row['impurity'], composition)
        groups[key].append({'row': row, 'structure': AseAtomsAdaptor.get_structure(atoms),
                            'fingerprint': anchor_distances(atoms), 'volume': atoms.get_volume()})
        if index % 2000 == 0:
            print(f'{dataset}: loaded {index}/{len(selected)} structures', flush=True)
    pairs = []
    compared = 0
    for group_index, members in enumerate(groups.values(), 1):
        for left, right in itertools.combinations(members, 2):
            if abs(left['volume']-right['volume']) > 5e-6*max(left['volume'], right['volume']):
                continue
            fa, fb = left['fingerprint'], right['fingerprint']
            if fa is not None and fb is not None and np.max(np.abs(fa-fb)) > 2*DISTANCE_TOLERANCE + 1e-4:
                continue
            a, b = left['row'], right['row']
            byte_equal = bool(a.get('sha256') and a['sha256'] == b.get('sha256'))
            compared += 1
            near_equal = byte_equal or equivalent(left['structure'], right['structure'], DISTANCE_TOLERANCE)
            if not near_equal:
                continue
            strict_equal = byte_equal or equivalent(left['structure'], right['structure'], 1e-6)
            sid_a, sid_b = a['source_id'], b['source_id']
            energy_gap = abs(a['target']-b['target'])
            pair = {'source_a': sid_a, 'source_b': sid_b, 'split_a': membership[sid_a],
                    'split_b': membership[sid_b], 'target_a': a['target'], 'target_b': b['target'],
                    'energy_gap_ev': energy_gap, 'label_conflict_above_0_05_ev': energy_gap > ENERGY_TOLERANCE,
                    'identical_file_bytes': byte_equal, 'equivalent_within_1e_6_angstrom': strict_equal,
                    'equivalent_within_1e_3_angstrom': True,
                    'cross_split': membership[sid_a] != membership[sid_b],
                    'train_test_pair': {membership[sid_a], membership[sid_b]} == {'train', 'test'},
                    'material': a['material'], 'impurity': a['impurity']}
            if dataset == 'imp2d':
                pair.update(spin_a=db_rows[sid_a].get('spin'), spin_b=db_rows[sid_b].get('spin'),
                            initial_site_a=db_rows[sid_a].get('site'), initial_site_b=db_rows[sid_b].get('site'))
            pairs.append(pair)
        if group_index % 300 == 0:
            print(f'{dataset}: matched {group_index}/{len(groups)} groups, {len(pairs)} equivalent pairs', flush=True)
    columns = ['source_a', 'source_b', 'split_a', 'split_b', 'target_a', 'target_b', 'energy_gap_ev',
               'label_conflict_above_0_05_ev', 'identical_file_bytes', 'equivalent_within_1e_6_angstrom',
               'equivalent_within_1e_3_angstrom', 'cross_split', 'train_test_pair', 'material', 'impurity']
    frame = pd.DataFrame(pairs, columns=columns + (['spin_a', 'spin_b', 'initial_site_a', 'initial_site_b']
                                                 if dataset == 'imp2d' else []))
    frame.to_csv(output/f'{dataset}_equivalent_pairs.csv', index=False)
    frame.loc[frame['cross_split'].eq(True)].to_csv(output/f'{dataset}_cross_split_pairs.csv', index=False)
    frame.loc[frame['label_conflict_above_0_05_ev'].eq(True)].to_csv(output/f'{dataset}_label_conflicts.csv', index=False)
    summaries = {}
    for name, selector in [('identical_file_bytes', lambda p: p['identical_file_bytes']),
                           ('equivalent_1e_6_angstrom', lambda p: p['equivalent_within_1e_6_angstrom']),
                           ('equivalent_1e_3_angstrom', lambda p: True)]:
        subset = [p for p in pairs if selector(p)]
        cross = [p for p in subset if p['cross_split']]
        conflict = [p for p in subset if p['label_conflict_above_0_05_ev']]
        summaries[name] = {'pairs': len(subset),
                           'unique_samples': len({p[k] for p in subset for k in ('source_a', 'source_b')}),
                           'cross_split_pairs': len(cross),
                           'cross_split_samples': len({p[k] for p in cross for k in ('source_a', 'source_b')}),
                           'train_test_pairs': sum(p['train_test_pair'] for p in subset),
                           'conflicting_label_pairs': len(conflict),
                           'conflicting_label_samples': len({p[k] for p in conflict for k in ('source_a', 'source_b')}),
                           'maximum_energy_gap_ev': max((p['energy_gap_ev'] for p in subset), default=0.)}
    result = {'dataset': dataset, 'manifest_identity': identity(manifest),
              'logger_id': split['logger_id'], 'retained_samples': len(selected), 'duplicate_source_ids': 0,
              'same_host_impurity_composition_groups': len(groups), 'geometry_pairs_tested': compared,
              'results': summaries, 'training_performed': False, 'filter_or_splits_modified': False,
              'scope': 'Same atom counts/composition, host and impurity; periodic symmetry/reordering allowed; '
                       'no cell rescaling or supercell expansion. Volume and unique-atom distance prefilters. '
                       'Near-equivalence is a review criterion, not proof of a wrong DFT label.'}
    (output/f'{dataset}_summary.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', nargs='+', choices=['semi', 'imp2d'], default=['semi', 'imp2d'])
    parser.add_argument('--run-dir', type=Path, default=ROOT/'logs/alignn_physical_clean')
    parser.add_argument('--data-root', type=Path, default=ROOT.parent/'dataset')
    parser.add_argument('--output', type=Path, default=ROOT/'results/impurity_split_audit')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = {name: audit(name, args.run_dir, args.data_root, args.output) for name in args.dataset}
    (args.output/'summary.json').write_text(json.dumps(results, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
