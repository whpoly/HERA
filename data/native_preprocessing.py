"""Native source inspection shared by automatic training and standalone preprocessing."""
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import time
import warnings

from ase.data import atomic_numbers
from ase.formula import Formula
from ase.io import read as ase_read
import numpy as np
import pandas as pd

from .native_filter import (DEFAULT_POLICY, SCHEMA, fit_split, validate_manifest, filter_applies_to_source,
                            read_manifest, requested_splits, verify_source_table, verify_source_file)
from .native_was import parse_native_defect
from .native_quality import DEEP_POLICY, configuration_family, geometry_features


def native_filter_policy(profile):
    if profile not in ('standard', 'strict'):
        raise ValueError(f'Unknown native outlier profile: {profile}')
    policy = copy.deepcopy(DEFAULT_POLICY)
    if profile == 'strict':
        policy['deep_checks'] = copy.deepcopy(DEEP_POLICY)
    return policy


def build_native_manifest(data_dir, seeds, cv5, policy, split_iterator, geometry_cache=None):
    """Inspect raw data once; use the training entry point's exact split protocol."""
    data_dir = Path(data_dir)
    source = pd.read_csv(data_dir/'id_prop_A_rich.csv', header=None)
    ids = source[0].astype(str).tolist()
    if len(ids) != len(set(ids)) or any('/' in sid or '\\' in sid for sid in ids):
        raise ValueError('Source table requires unique plain filenames')
    rows = []
    started = time.monotonic()
    for i, row in enumerate(source.itertuples(index=False, name=None)):
        rows.append(inspect_source(data_dir/str(row[0]), row[1], policy, geometry_cache))
        if (i+1) % 500 == 0:
            print(f'Checked {i+1}/{len(ids)} raw structures, {time.monotonic()-started:.1f}s', flush=True)
    splits = []
    for original in split_iterator(ids, [0]*len(ids), seeds, cv5=cv5):
        membership = {part: original[f'{part}_X'] for part in ('train','val','test')}
        splits.append({**{key: original[key] for key in ('display','logger_id','seed','explain_id')},
                       **fit_split(rows, membership, policy)})
    return validate_manifest({'schema':SCHEMA, 'dataset':'native', 'policy':policy,
                              'cv5':cv5, 'records':rows, 'splits':splits})


def prepare_native_filter(run_dir, profile, seeds, cv5, split_iterator,
                          data_dir='dataset/Dataset_1/Dataset_1/A_rich/Neutral'):
    """Generate or reuse one frozen filter before graph loading/config validation."""
    run_dir, data_dir = Path(run_dir), Path(data_dir)
    path = run_dir/'native_filter_manifest.json'
    policy = native_filter_policy(profile)
    if path.exists():
        manifest = read_manifest(path)
        if manifest['policy'] != policy:
            raise ValueError('Run directory uses a different native outlier policy; choose a different --run-dir')
        requested_splits(manifest, seeds, cv5=cv5)
        verify_source_table(manifest, pd.read_csv(data_dir/'id_prop_A_rich.csv', header=None))
        # Check excluded sources too: a changed source might no longer be an outlier.
        for row in manifest['records']:
            source_path = data_dir/row['source_id']
            if row['sha256'] is not None:
                verify_source_file(source_path, row)
            elif source_path.exists():
                raise ValueError(f'Cannot verify cached native source: {source_path}; regenerate in a new --run-dir')
        print(f'Native outlier filter: reusing verified {path}', flush=True)
    else:
        print(f'Native outlier filter: inspecting source data ({profile}) before graph loading...', flush=True)
        manifest = build_native_manifest(data_dir, seeds, cv5, policy, split_iterator)
        run_dir.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8') as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write('\n')
    write_manifest_tables(manifest, run_dir, prefix='native_filter_')
    for split in requested_splits(manifest, seeds, cv5=cv5):
        counts = ', '.join(f'{part} {len(split["original"][part])}->{len(split["kept"][part])}'
                           for part in ('train','val','test'))
        print(f'Native outlier filter [{split["display"]}]: {counts}', flush=True)
    return manifest


def inspect_source(path, target, policy, geometry_cache=None):
    sid = path.name
    record = {'source_id': sid, 'material': sid.split('-')[1] if '-' in sid else '',
              'defect_type': '', 'added_z': None, 'removed_z': None,
              'target': None, 'sha256': None,
              'min_distance_angstrom': None, 'quality_reasons': []}
    try:
        value = float(target)
        if math.isfinite(value):
            record['target'] = value
        else:
            record['quality_reasons'].append('nonfinite_target')
    except (ValueError, TypeError):
        record['quality_reasons'].append('invalid_target')
    try:
        kind, current, previous = parse_native_defect(sid)
        record['defect_type'] = kind
        record['added_z'], record['removed_z'] = int(current), int(previous)
        record['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            atoms = ase_read(str(path), format='cif', store_tags=True)
        if len(atoms) < 2 or not np.isfinite(atoms.positions).all() or not np.isfinite(atoms.cell.array).all() or atoms.get_volume() <= 0:
            raise ValueError('Invalid cell or coordinates')
        labels = atoms.info.get('_atom_site_label', [])
        matches = [re.fullmatch(r'([A-Z][a-z]?)(\d+)', str(label)) for label in labels]
        if len(labels) != len(atoms) or not all(matches):
            raise ValueError('Invalid CIF site labels')
        indices = [int(m[2]) for m in matches]
        if sorted(indices) != list(range(len(atoms))) or [m[1] for m in matches] != atoms.get_chemical_symbols():
            raise ValueError('Inconsistent CIF site indices/species')
        if kind != 'vacancy' and atoms.numbers[indices.index(len(atoms)-1)] != current:
            raise ValueError('Defect species differs from filename')
        base = Counter(atoms.numbers.tolist())
        if current:
            base[current] -= 1
        if previous:
            base[previous] += 1
        base = +base
        formula = {atomic_numbers[k]: v for k, v in Formula(record['material']).count().items()}
        ratios = {base.get(z, 0)/n for z, n in formula.items()}
        if set(base) != set(formula) or len(ratios) != 1 or min(ratios) <= 0:
            raise ValueError('Composition differs from host formula and defect')
        if not np.all(np.asarray(atoms.info['_atom_site_occupancy'], dtype=float) == 1):
            raise ValueError('Partial or invalid occupancy')
        distances = atoms.get_all_distances(mic=True)
        np.fill_diagonal(distances, np.inf)
        minimum = float(distances.min())
        record['min_distance_angstrom'] = minimum
        if minimum < policy['min_distance_angstrom']:
            record['quality_reasons'].append('short_interatomic_distance')
        if policy.get('deep_checks'):
            record.update(geometry_features(atoms, distances, labels))
        if geometry_cache is not None:
            geometry_cache[sid] = atoms[np.argsort(indices)]
    except Exception as exc:
        record['quality_reasons'].append('invalid_or_missing_structure')
        record['structure_error'] = f'{type(exc).__name__}: {exc}'
    return record


def write_manifest_tables(manifest, output, prefix=""):
    """Rebuild audit tables from a frozen manifest without refitting or rescanning."""
    rows, splits, policy = manifest["records"], manifest["splits"], manifest["policy"]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report_rows = []
    for row in rows:
        match = re.search(r'-POSCAR(\d+)-', row['source_id'])
        scope = {'filter_scope_applies': filter_applies_to_source(row['source_id'], policy)} if 'filter_poscar_indices' in policy else {}
        report_rows.append({**row, 'poscar_index': int(match[1]) if match else None, **scope})
    pd.DataFrame(report_rows).to_csv(output/f'{prefix}structure_quality.csv', index=False)
    by_id = {r['source_id']: r for r in report_rows}
    summaries = []
    for split in splits:
        def annotated_row(sid):
            row = by_id[sid]
            if not policy.get('deep_checks'):
                return row
            key = '/'.join(str(row[field]) for field in policy['group_by'])
            coarse = split['train_fences'].get(key, {})
            fine = split['family_train_fences'].get(configuration_family(sid), {})
            return {**row, 'configuration_family': configuration_family(sid),
                    **{'chemical_group_'+field: coarse.get(field) for field in ('n_train','lower_ev','upper_ev')},
                    **{'family_'+field: fine.get(field) for field in ('n_train','lower_ev','upper_ev',
                        'energy_median_ev','energy_mad_sigma_ev','min_pair_ratio_median','joint_energy_upper_ev')}}

        retained, removed = [], []
        for part in ('train', 'val', 'test'):
            for sid in split['kept'][part]:
                retained.append({'split': part, **annotated_row(sid)})
            for item in split['excluded'][part]:
                removed.append({'split': part, **annotated_row(item['source_id']), 'exclusion_reasons': ';'.join(item['reasons'])})
            summaries.append({'logger_id': str(split['logger_id']), 'split': part,
                              'original': len(split['original'][part]), 'kept': len(split['kept'][part]),
                              'removed': len(split['excluded'][part])})
        pd.DataFrame(retained).to_csv(output/f'{prefix}seed{split["logger_id"]}_kept.csv', index=False)
        columns = list(dict.fromkeys(key for row in retained+removed for key in row if key not in ('split','exclusion_reasons')))
        pd.DataFrame(removed, columns=['split', *columns, 'exclusion_reasons']).to_csv(output/f'{prefix}seed{split["logger_id"]}_removed.csv', index=False)
    pd.DataFrame(summaries).to_csv(output/f'{prefix}summary.csv', index=False)
    return summaries


