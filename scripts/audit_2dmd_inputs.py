"""Audit local 2DMD inputs and graph preprocessing without training or changing data.

ASE independently reads every CIF and matches atoms to a pristine supercell.
Every hetero structure is checked against that reference. The other graph modes
are checked on the first available sample of every descriptor in each subset.
"""

import argparse
import ast
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import warnings

import numpy as np
import pandas as pd
from ase.data import chemical_symbols
from ase.io import read
from ase.io.cif import parse_cif
from pymatgen.core import Structure
from pymatgen.core.periodic_table import DummySpecies
from scipy.spatial import cKDTree

from ..data.converters import AtomFeaturesExtractor
from ..data.datasets import init_elem_embedding
from ..data.structure_utils import convert_to_sparse_2dmd_high


ROOT = Path(__file__).resolve().parents[1]
REFERENCES = {}


def initialize():
    warnings.filterwarnings('ignore', message='Issues encountered while parsing CIF:.*')
    init_elem_embedding(ROOT / 'atom_init.json')


def coord_key(coords):
    return tuple(np.round(coords, 3))


def reference(path, cell):
    key = (path, tuple(cell))
    if key not in REFERENCES:
        atoms = read(path).repeat(cell)
        frac = atoms.get_scaled_positions(wrap=False)
        keys = [coord_key(x) for x in frac]
        assert len(set(keys)) == len(keys), 'reference rounding collision'
        REFERENCES[key] = (atoms, frac, keys, cKDTree(frac % 1, boxsize=1),
                           Structure.from_file(path))
    return REFERENCES[key]


def descriptor_counter(defects):
    result = Counter()
    for defect in defects:
        if defect['type'] == 'vacancy':
            result[('vacancy', defect['element'])] += 1
        elif defect['type'] == 'substitution':
            result[('substitution', defect['from'], defect['to'])] += 1
        else:
            raise ValueError(f'Unsupported descriptor: {defect}')
    return result


def read_initial(path, crosscheck=False):
    blocks = list(parse_cif(str(path)))
    if len(blocks) == 1 and blocks[0].get_spacegroup(subtrans_included=True).no == 1:
        # P1 contains every atom explicitly. Avoid ASE's quadratic symmetry
        # expansion; still use ASE's own CIF parser and Atoms construction.
        atoms = blocks[0].get_unsymmetrized_structure()
        atoms.pbc = True
        if crosscheck:
            standard = read(path)
            assert np.array_equal(atoms.numbers, standard.numbers)
            assert np.allclose(atoms.positions, standard.positions)
        return atoms
    return read(path)


def inspect_sample(job):
    result = {'subset': job['subset'], 'source': job['source'], 'errors': [],
              'sampled_other_modes': job['sampled_other_modes']}
    try:
        atoms = read_initial(job['path'], crosscheck=job['sampled_other_modes'])
        host, host_frac, keys, tree, unit = reference(job['reference'], job['cell'])
        frac = atoms.get_scaled_positions(wrap=False)
        raw_keys = [coord_key(x) for x in frac]
        assert len(set(raw_keys)) == len(atoms), 'raw coordinates collide at three decimals'
        assert set(raw_keys) <= set(keys), 'real atoms do not match rounded reference coordinates'
        _, indices = tree.query(frac % 1)
        assert len(set(indices.tolist())) == len(atoms), 'reference site assignment is not one-to-one'
        delta = frac - host_frac[indices]
        delta -= np.round(delta)
        max_match = float(np.linalg.norm(delta @ atoms.cell.array, axis=1).max())
        assert max_match < 1e-3, f'initial atom is {max_match} A from its reference site'
        now = np.zeros(len(host), dtype=int)
        now[indices] = atoms.numbers
        expected_defects = Counter()
        for previous_z, current_z in zip(host.numbers, now):
            if current_z == 0:
                expected_defects[('vacancy', chemical_symbols[previous_z])] += 1
            elif previous_z != current_z:
                expected_defects[('substitution', chemical_symbols[previous_z], chemical_symbols[current_z])] += 1
        listed_defects = descriptor_counter(job['defects'])
        result['descriptor_mismatch'] = (None if expected_defects == listed_defects else {
            'actual': {str(k): v for k, v in expected_defects.items()},
            'listed': {str(k): v for k, v in listed_defects.items()}})
        expected = {key: (int(z), int(was)) for key, z, was in zip(keys, now, host.numbers)}
        raw = Structure.from_file(job['path'])
        raw_map = {coord_key(s.frac_coords): s.specie.Z for s in raw}
        ase_map = dict(zip(raw_keys, map(int, atoms.numbers)))
        assert len(raw) == len(atoms) and raw_map == ase_map, 'ASE and training CIF reader differ'
        modes = ['hetero'] + (['full', 'full_x', 'attention', 'sparse'] if job['sampled_other_modes'] else [])
        max_rounding = 0.0
        graph_nodes = 0
        for mode in modes:
            task = f'megnet_{mode}' if mode == 'sparse' else f'alignn_{mode}'
            graph = convert_to_sparse_2dmd_high(
                raw, unit, job['cell'], task, [1] if mode == 'sparse' else None,
                skip_was=False, copy_unit_cell_properties=False, local_cutoff=0)
            want = {k: v for k, v in expected.items()
                    if (mode != 'full' or v[0] != 0) and (mode != 'sparse' or v[0] != v[1])}
            observed = {coord_key(s.frac_coords):
                        (0 if isinstance(s.specie, DummySpecies) else s.specie.Z, s.properties.get('was'))
                        for s in graph}
            assert len(graph) == len(want) and observed == want, f'{mode}: nodes/species/WAS mismatch'
            if mode in ('hetero', 'attention'):
                for site in graph:
                    current_z, was_z = want[coord_key(site.frac_coords)]
                    assert bool(site.properties['type']) == (current_z != was_z), f'{mode}: wrong defect marker'
                    if mode == 'hetero':
                        assert bool(site.properties['pool_type']) == (current_z != was_z), 'wrong pooling marker'
            if mode == 'hetero':
                graph_nodes = len(graph)
                features = AtomFeaturesExtractor('was_species', 'alignn_hetero_was').convert(graph)
                plain = AtomFeaturesExtractor('Z', 'alignn_hetero').convert(graph)
                from ..data.datasets import elem_embedding
                expected_features = np.asarray([
                    ([0] * 92 if current_z == 0 else elem_embedding[current_z]) + elem_embedding[was_z]
                    for current_z, was_z in observed.values()])
                assert features.shape == (len(graph), 184), 'WAS input has wrong dimensions'
                assert np.array_equal(features, expected_features), 'WAS feature vectors mismatch'
                assert np.array_equal(features[:, :92], plain), 'plain model unexpectedly uses WAS'
            for site in graph:
                key = coord_key(site.frac_coords)
                if key in ase_map:
                    original = frac[raw_keys.index(key)]
                    displacement = site.frac_coords - original
                    displacement -= np.round(displacement)
                    max_rounding = max(max_rounding, float(np.linalg.norm(displacement @ atoms.cell.array)))
        # Compare lattice-site occupations across concentrations, ignoring vacuum
        # thickness. This detects identical indexed arrangements, not all symmetry equivalents.
        z_levels = sorted(set(np.round(host_frac[:, 2], 6)))
        occupancy = sorted((round(float(x), 3), round(float(y), 3),
                            z_levels.index(round(float(z), 6)), int(current_z))
                           for (x, y, z), current_z in zip(host_frac, now))
        fingerprint = hashlib.sha256(repr(occupancy).encode()).hexdigest()
        result.update(atoms=len(atoms), graph_nodes=graph_nodes,
                      vacancies=int(np.sum(now == 0)), substitutions=int(np.sum((now != 0) & (now != host.numbers))),
                      max_reference_distance_angstrom=max_match,
                      max_rounding_displacement_angstrom=max_rounding, fingerprint=fingerprint)
    except Exception as exc:
        result['errors'].append(f'{type(exc).__name__}: {exc}')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ROOT.parent / 'dataset/2d-materials-point-defects-all')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit-per-subset', type=int, default=0, help='0 checks all target rows')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/2dmd_input_audit.json')
    args = parser.parse_args()
    jobs, subsets, transfer_samples = [], {}, defaultdict(list)
    for density in ('low_density_defects', 'high_density_defects'):
        concentration = 'low' if density.startswith('low') else 'high'
        for folder in sorted((args.data_root / density).iterdir()):
            if not (folder / 'targets.csv.gz').exists():
                continue
            name = f'{concentration}/{folder.name}'
            targets = pd.read_csv(folder / 'targets.csv.gz', index_col=0)
            descriptors = pd.read_csv(folder / 'descriptors.csv', index_col=0)
            record = subsets[name] = {'target_rows': len(targets), 'checked': 0, 'passed': 0,
                'missing_files': [], 'errors': [], 'descriptor_mismatches': [],
                'other_modes_checked': 0, 'raw_atoms': 0,
                'hetero_nodes': 0, 'vacancies': 0, 'substitutions': 0,
                'duplicate_source_ids': int(targets.index.duplicated().sum()),
                'invalid_targets': int((~np.isfinite(pd.to_numeric(targets.formation_energy_per_site, errors='coerce'))).sum()),
                'max_reference_distance_angstrom': 0.0, 'max_rounding_displacement_angstrom': 0.0}
            seen_descriptors = set()
            for source, row in targets.iloc[:args.limit_per_subset or None].iterrows():
                path = folder / 'initial' / f'{source}.cif'
                if not path.exists():
                    record['missing_files'].append(str(source))
                    continue
                descriptor = descriptors.loc[row.descriptor_id]
                defects = ast.literal_eval(descriptor.defects)
                base = descriptor.base
                unit_path = (args.data_root / f'{base}.cif' if concentration == 'high'
                             else folder / 'unit_cells' / f'{base}.cif')
                sampled = row.descriptor_id not in seen_descriptors
                seen_descriptors.add(row.descriptor_id)
                jobs.append({'subset': name, 'source': str(source), 'path': str(path),
                             'reference': str(unit_path), 'cell': ast.literal_eval(descriptor.cell),
                             'defects': defects, 'sampled_other_modes': sampled})
                if base in ('MoS2', 'WSe2'):
                    transfer_samples[base].append(SimpleNamespace(
                        source_id=str(source), concentration=concentration,
                        pure_vacancy=bool(defects) and all(d['type'] == 'vacancy' for d in defects)))
    fingerprints = defaultdict(lambda: defaultdict(list))
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize) as executor:
        for number, result in enumerate(executor.map(inspect_sample, jobs, chunksize=10), 1):
            record = subsets[result['subset']]
            record['checked'] += 1
            if result.get('descriptor_mismatch'):
                record['descriptor_mismatches'].append({'source': result['source'], **result['descriptor_mismatch']})
            if result['errors']:
                record['errors'].append({'source': result['source'], 'errors': result['errors']})
                if len(record['errors']) <= 3:
                    print(f'{result["subset"]}/{result["source"]}: {result["errors"]}', flush=True)
            else:
                record['passed'] += 1
                record['other_modes_checked'] += int(result['sampled_other_modes'])
                for key, result_key in (('raw_atoms', 'atoms'), ('hetero_nodes', 'graph_nodes'),
                                        ('vacancies', 'vacancies'), ('substitutions', 'substitutions')):
                    record[key] += result[result_key]
                for key in ('max_reference_distance_angstrom', 'max_rounding_displacement_angstrom'):
                    record[key] = max(record[key], result[key])
                fingerprints[result['subset']][result['fingerprint']].append(result['source'])
            if number % 250 == 0 or number == len(jobs):
                print(f'Checked {number}/{len(jobs)}; failures={sum(len(s["errors"]) for s in subsets.values())}', flush=True)
    from ..main import iter_concentration_transfer_splits
    transfers = {}
    for material, samples in transfer_samples.items():
        for pure in (False, True):
            selected = [s for s in samples if not pure or s.pure_vacancy]
            if args.limit_per_subset and (sum(s.concentration == 'low' for s in selected) < 2
                                          or not any(s.concentration == 'high' for s in selected)):
                continue
            test_sets, split_sizes = [], []
            for split in iter_concentration_transfer_splits(selected, np.zeros(len(selected)), [123, 11, 1245]):
                assert all(s.concentration == 'low' for s in split['train_X'] + split['val_X'])
                assert all(s.concentration == 'high' for s in split['test_X'])
                sets = [{s.source_id for s in split[f'{part}_X']} for part in ('train', 'val', 'test')]
                assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
                test_sets.append(sets[2])
                split_sizes.append([len(split[f'{part}_X']) for part in ('train', 'val', 'test')])
            assert all(s == test_sets[0] for s in test_sets)
            transfers[f'{material}_{"vacancy_only" if pure else "all_defects"}'] = {
                'seeds': [123, 11, 1245], 'train_val_test_sizes': split_sizes,
                'high_only_in_test': True, 'source_overlap': 0}
    duplicate_report = {}
    for subset, mapping in fingerprints.items():
        groups = [sources for sources in mapping.values() if len(sources) > 1]
        duplicate_report[subset] = {'groups': len(groups), 'extra_rows': sum(len(g) - 1 for g in groups), 'examples': groups[:3]}
    overlaps = {}
    for material in ('MoS2', 'WSe2'):
        low, high = fingerprints[f'low/{material}'], fingerprints[f'high/{material}_500']
        common = low.keys() & high.keys()
        overlaps[material] = {'identical_site_occupation_groups': len(common),
                             'examples': [{'low': low[k], 'high': high[k]} for k in sorted(common)[:3]]}
    report = {'training_performed': False, 'limit_per_subset': args.limit_per_subset,
              'subsets': subsets, 'transfer_splits': transfers,
              'duplicate_site_occupations': duplicate_report, 'low_high_site_occupation_overlap': overlaps,
              'scope': 'All selected initial CIFs: ASE/reference matching, descriptors, actual hetero preprocessing and atom/WAS features. Other four modes: first sample of each descriptor. No complete edge-tensor audit or retraining. Occupation hashes do not identify every rotation/translation equivalent.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'subsets': {k: {f: v[f] for f in ('checked', 'passed', 'other_modes_checked', 'max_rounding_displacement_angstrom')} for k,v in subsets.items()},
                      'overlaps': overlaps, 'duplicates': duplicate_report}, indent=2), flush=True)
    print(f'Saved: {args.output}', flush=True)


if __name__ == '__main__':
    main()
