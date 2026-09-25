"""Frozen, physically screened imp2d/semi selections at training-data loading.

The legacy benchmark cohort is split first, then bad records are omitted in
place. Missing files/hosts and incompatible source versions stop the run;
they are not silently interpreted as unphysical structures.
"""
import copy
import hashlib
import json
from pathlib import Path
import warnings

from ase.db import connect
from ase.geometry import find_mic
from ase.io import read as ase_read
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .impurity_quality import (POLICY, combine_checks, finite, inspect_geometry,
                               inspect_imp2d_metadata, inspect_impurity_composition, matches_formula)
from .imp2d_source import download_imp2d_database

SCHEMA = 'impurity_physical_manifest_v1'
PARTS = ('train', 'val', 'test')
SEMI_IDENTICAL_LABEL_TOLERANCE_EV = 0.05
DATA_DIRS = {'imp2d': 'dataset/imp2d/imp2d', 'semi': 'dataset/Dataset_1/Dataset_1/Neutral/Neutral'}
TABLES = {'imp2d': 'id_prop.csv', 'semi': 'id_prop_A_rich.csv'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(manifest):
    value = json.dumps(manifest, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return {'schema': SCHEMA, 'sha256': hashlib.sha256(value).hexdigest(), 'policy': manifest['policy']}


def read_table(path):
    path = Path(path)
    if not path.is_file() or not path.stat().st_size:
        raise ValueError(f'Missing or empty training CSV: {path}. Restore real data before preprocessing.')
    frame = pd.read_csv(path, header=None)
    ids = frame[0].astype(str).tolist()
    if frame.shape[1] < 2 or not ids or len(set(ids)) != len(ids):
        raise ValueError(f'Expected unique IDs and targets: {path}')
    if any(x in ('', '.', '..') or '/' in x or '\\' in x for x in ids):
        raise ValueError('Source IDs must be plain filenames')
    return frame


def verify_table(manifest, frame):
    actual = [(str(r[0]), finite(r[1])) for r in frame.itertuples(index=False, name=None)]
    expected = [(r['source_id'], r['target']) for r in manifest['records']]
    if len(actual) != len(expected) or any(a != b for a, b in zip(actual, expected)):
        raise ValueError('Impurity source IDs, order or targets changed; use a new --run-dir')


def source_path(data_dir, dataset, sid):
    return Path(data_dir)/(sid+'.cif' if dataset == 'imp2d' else sid)


def verify_file(path, record):
    if not Path(path).is_file() or digest(path) != record['sha256']:
        raise ValueError(f'Impurity CIF changed since preprocessing: {record["source_id"]}')


def requested_splits(manifest, seeds, cv5=False):
    if manifest['cv5'] != bool(cv5):
        raise ValueError('Impurity filter CV protocol changed; use a new --run-dir')
    keys = [f'{s}_fold{i}' for s in seeds for i in range(1, 6)] if cv5 else [str(s) for s in seeds]
    by_id = {str(s['logger_id']): s for s in manifest['splits']}
    if set(keys)-set(by_id):
        raise ValueError('Impurity filter lacks requested seeds; use a new --run-dir')
    return [by_id[key] for key in keys]


def retained_ids(manifest):
    return {sid for split in manifest['splits'] for part in PARTS for sid in split['kept'][part]}


def validate_manifest(manifest):
    if manifest.get('schema') != SCHEMA or manifest.get('dataset') not in DATA_DIRS:
        raise ValueError('Expected an impurity_physical_manifest_v1 manifest')
    rows = manifest['records']
    ids = [r['source_id'] for r in rows]
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError('Invalid impurity manifest source identities')
    baseline = {r['source_id'] for r in rows if r['baseline_eligible']}
    by_id = {r['source_id']: r for r in rows}
    loggers = set()
    if not manifest['splits']:
        raise ValueError('Impurity manifest has no splits')
    for split in manifest['splits']:
        key = str(split['logger_id'])
        if key in loggers:
            raise ValueError('Duplicate impurity split')
        loggers.add(key)
        original = [sid for part in PARTS for sid in split['original'][part]]
        if len(original) != len(baseline) or set(original) != baseline:
            raise ValueError('Impurity original split must partition the legacy cohort')
        for part in PARTS:
            expected = [sid for sid in split['original'][part] if not by_id[sid]['quality_reasons']]
            excluded = [{'source_id': sid, 'reasons': by_id[sid]['quality_reasons']}
                        for sid in split['original'][part] if by_id[sid]['quality_reasons']]
            if not expected or split['kept'][part] != expected or split['excluded'][part] != excluded:
                raise ValueError('Impurity retained IDs must match the frozen quality decision and preserve order')
    identity(manifest)
    return manifest


def same_periodic_geometry(actual, reference, tolerance=None):
    """Species-aware matching allowing reorder, rigid rotation and origin shift.

Cell basis/metric must match. Basis-changing re-exports stop for review instead
of attaching potentially unrelated DFT metadata to a different structure.
"""
    tolerance = tolerance or POLICY['database_geometry_tolerance_angstrom']
    if sorted(actual.numbers) != sorted(reference.numbers):
        return False
    a, b = actual.cell.array, reference.cell.array
    if not np.allclose(a@a.T, b@b.T, rtol=1e-6, atol=1e-3):
        return False
    af, bf = actual.get_scaled_positions(wrap=False), reference.get_scaled_positions(wrap=False)
    species = sorted(set(actual.numbers), key=lambda z: sum(actual.numbers == z))
    anchor = np.flatnonzero(actual.numbers == species[0])[0]
    for ref_anchor in np.flatnonzero(reference.numbers == species[0]):
        offset = af[anchor]-bf[ref_anchor]
        matched = True
        for z in species:
            left, right = af[actual.numbers == z], bf[reference.numbers == z]+offset
            delta = (left[:, None, :]-right[None, :, :])@a
            distances = find_mic(delta.reshape(-1, 3), actual.cell, actual.pbc)[1].reshape(len(left), len(right))
            ix, jx = linear_sum_assignment(distances)
            if np.max(distances[ix, jx]) > tolerance:
                matched = False
                break
        if matched:
            return True
    return False


def _read_cif(path):
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f'Missing/empty structure: {path}. Restore data; this is not a physical outlier decision.')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        return ase_read(str(path), format='cif', store_tags=True)


def default_database(data_dir=None):
    data_dir = Path(data_dir or DATA_DIRS['imp2d'])
    candidates = [data_dir.parent/'imp2d.db', data_dir/'imp2d.db',
                  Path(__file__).resolve().parents[1]/'tmp/imp2d_audit/imp2d.db']
    return next((p for p in candidates if p.is_file()), None)


def mark_identical_label_conflicts(records, tolerance=SEMI_IDENTICAL_LABEL_TOLERANCE_EV, hash_key='sha256'):
    """Quarantine all competing labels for exactly identical model inputs.

Only exact file identity within the same host/defect/impurity is considered.
The fixed tolerance is not fitted to a split or to model prediction errors.
No minimum-energy label is silently chosen as the ground truth.
"""
    from collections import defaultdict
    groups = defaultdict(list)
    for row in records:
        if row.get('baseline_eligible') and row.get(hash_key) and row['target'] is not None:
            key = (row[hash_key], row['material'], row['site'], row['impurity'])
            groups[key].append(row)
    conflicts = 0
    for group in groups.values():
        spread = max(r['target'] for r in group)-min(r['target'] for r in group)
        if len(group) > 1 and spread > tolerance:
            conflicts += 1
            ids = [r['source_id'] for r in group]
            for row in group:
                reason = 'identical_structure_inconsistent_target'
                if reason not in row['quality_reasons']:
                    row['quality_reasons'].append(reason)
                row['identical_structure_conflict_sources'] = ids
                row['identical_structure_target_spread_ev'] = spread
    return conflicts


def prepare_impurity_filter(run_dir, dataset, seeds, cv5, split_iterator, data_dir=None,
                            database_path=None, host_dir='dataset/Dataset_1/host_configurations',
                            semi_source_policy='complete', imp2d_source='cif', imp2d_host_filter=None,
                            imp2d_energy_window=None):
    if dataset not in DATA_DIRS:
        raise ValueError(f'Unsupported physical filter dataset: {dataset}')
    if imp2d_source not in ('cif', 'db'):
        raise ValueError(f'Unknown IMP2D source: {imp2d_source}')
    if imp2d_host_filter is not None and (dataset != 'imp2d' or imp2d_source != 'db'):
        raise ValueError('IMP2D host filtering requires the direct database source')
    if imp2d_energy_window is not None and (dataset != 'imp2d' or imp2d_source != 'db'):
        raise ValueError('IMP2D energy window requires the direct database source')
    if dataset == 'imp2d' and imp2d_source == 'db':
        from .imp2d_database import prepare_database_filter
        return prepare_database_filter(run_dir, seeds, cv5, split_iterator, database_path,
                                       host_filter=imp2d_host_filter, energy_window=imp2d_energy_window)
    data_dir, run_dir, host_dir = Path(data_dir or DATA_DIRS[dataset]), Path(run_dir), Path(host_dir)
    table_path = data_dir/TABLES[dataset]
    frame = read_table(table_path)
    db_path = None
    if dataset == 'imp2d':
        db_path = Path(database_path) if database_path is not None else default_database(data_dir)
        if db_path is None:
            db_path = download_imp2d_database(data_dir.parent/'imp2d.db')
        if not db_path.is_file() or not db_path.stat().st_size:
            raise ValueError(f'Missing/empty original IMP2D database: {db_path}. '
                             'Restore that file, or omit --imp2d-source-db for automatic download.')
    path = run_dir/f'{dataset}_filter_manifest.json'
    policy = copy.deepcopy(POLICY)
    if semi_source_policy not in ('complete', 'legacy_available'):
        raise ValueError(f'Unknown semi source policy: {semi_source_policy}')
    if dataset == 'semi':
        policy['semi_source_policy'] = semi_source_policy
        policy['semi_identical_structure_label_tolerance_ev'] = SEMI_IDENTICAL_LABEL_TOLERANCE_EV
    if path.exists():
        manifest = validate_manifest(json.loads(path.read_text(encoding='utf-8')))
        if manifest['dataset'] != dataset or manifest['policy'] != policy:
            raise ValueError('Different impurity filter policy; use a new --run-dir')
        requested_splits(manifest, seeds, cv5)
        verify_table(manifest, frame)
        if dataset == 'imp2d' and manifest['database_sha256'] != digest(db_path):
            raise ValueError('IMP2D source database changed; use a new --run-dir')
        for host, checksum in manifest.get('host_sha256', {}).items():
            host_path = host_dir/f'{host}.vasp'
            actual = digest(host_path) if host_path.is_file() else None
            if actual != checksum:
                raise ValueError(f'Semi host changed: {host}; use a new --run-dir')
        for record in manifest['records']:
            if record['sha256'] is not None:
                verify_file(source_path(data_dir, dataset, record['source_id']), record)
            elif record.get('baseline_exclusion') == 'legacy_missing_cif':
                if source_path(data_dir, dataset, record['source_id']).exists():
                    raise ValueError(f'Previously missing semi CIF appeared: {record["source_id"]}; use a new --run-dir')
        print(f'{dataset}: reusing verified physical filter {path}', flush=True)
    else:
        db_rows = {}
        if dataset == 'imp2d':
            with connect(str(db_path)) as db:
                for row in db.select():
                    if row.name in db_rows:
                        raise ValueError(f'Ambiguous IMP2D database ID: {row.name}')
                    db_rows[row.name] = row
        rows, hosts, host_hashes = [], {}, {}
        for sid, target, *_ in frame.itertuples(index=False, name=None):
            sid, target = str(sid), finite(target)
            record = {'source_id': sid, 'target': target, 'sha256': None,
                      'quality_reasons': [], 'review_flags': []}
            if dataset == 'imp2d' and (target is None or not -10 < target < 10):
                record.update(baseline_eligible=False, baseline_exclusion='legacy_energy_window_or_nonfinite')
                rows.append(record)
                continue
            cif = source_path(data_dir, dataset, sid)
            if dataset == 'semi' and semi_source_policy == 'legacy_available':
                # Reproduce the documented historical loader cohort explicitly.
                # Availability exclusions are never called physical outliers.
                base = sid.split('-')[1]
                record['material'] = base
                if not cif.is_file() or cif.stat().st_size == 0:
                    record.update(baseline_eligible=False,
                                  baseline_exclusion='legacy_empty_cif' if cif.is_file() else 'legacy_missing_cif',
                                  sha256=digest(cif) if cif.is_file() else None)
                    rows.append(record)
                    continue
                host_path = host_dir/f'{base}.vasp'
                if not host_path.is_file() or not host_path.stat().st_size:
                    host_hashes[base] = digest(host_path) if host_path.is_file() else None
                    record.update(baseline_eligible=False, baseline_exclusion='legacy_missing_or_empty_host',
                                  sha256=digest(cif))
                    rows.append(record)
                    continue
            atoms = _read_cif(cif)  # Parser/I/O failure stops; never shrink the dataset silently.
            record['sha256'] = digest(cif)
            check = combine_checks(inspect_geometry(atoms), inspect_impurity_composition(atoms, sid, dataset))
            record = combine_checks(record, check)
            occupancy = atoms.info.get('_atom_site_occupancy')
            if occupancy is not None and not np.all(np.asarray(occupancy, dtype=float) == 1):
                record['quality_reasons'].append('partial_occupancy_unsupported')
            record.setdefault('baseline_eligible', True)
            if target is None:
                record['quality_reasons'].append('nonfinite_target')
            if dataset == 'imp2d':
                row = db_rows.get(sid)
                if row is None or finite(row.get('eform')) is None or abs(target-row.eform) > POLICY['energy_identity_tolerance_ev']:
                    raise ValueError(f'CSV and IMP2D database label/version mismatch: {sid}')
                if not same_periodic_geometry(atoms, row.toatoms()):
                    raise ValueError(f'CIF and IMP2D database geometry/version mismatch: {sid}')
                record = combine_checks(record, inspect_imp2d_metadata(row.key_value_pairs))
                if not record['quality_reasons']:
                    # Verify the exact CIF labels that graph loading uses, including self impurities.
                    from pymatgen.core import Structure
                    from .datasets import _assign_imp2d_self_defect_indices, _load_imp2d_self_defect_labels
                    from .impurity_was import imp2d_reference_structure
                    from .structure_utils import get_imp2d_defect_indices
                    raw = Structure.from_file(cif)
                    raw.source_id = sid
                    info = {'base': record['material'], 'impurity': record['impurity'], 'site': record['site'],
                            'is_self': 'self_impurity_requires_checked_site_label' in record['review_flags']}
                    _assign_imp2d_self_defect_indices([[raw, info]], _load_imp2d_self_defect_labels())
                    imp2d_reference_structure(raw, info, get_imp2d_defect_indices(raw, info))
            else:
                base = sid.split('-')[1]
                if base not in hosts:
                    host_path = host_dir/f'{base}.vasp'
                    if not host_path.is_file() or not host_path.stat().st_size:
                        raise ValueError(f'Missing semi reference host: {host_path}; restore it before filtering')
                    hosts[base] = ase_read(str(host_path), format='vasp')
                    host_hashes[base] = digest(host_path)
                from collections import Counter
                if not matches_formula(Counter(hosts[base].get_chemical_symbols()), base):
                    raise ValueError(f'Semi reference host composition mismatch: {base}')
                record['review_flags'].append('DFT_convergence_unverified_no_metadata')
                if not record['baseline_eligible']:
                    record['baseline_exclusion'] = 'legacy_non_extrinsic'
            rows.append(record)
            if len(rows) % 500 == 0:
                print(f'{dataset} physical filter: inspected {len(rows)}/{len(frame)}', flush=True)
        if dataset == 'semi':
            conflicts = mark_identical_label_conflicts(rows)
            print(f'Semi label consistency: {conflicts} identical-structure groups with conflicting targets', flush=True)
        by_id = {r['source_id']: r for r in rows}
        baseline_ids = [r['source_id'] for r in rows if r['baseline_eligible']]
        if dataset == 'semi' and semi_source_policy == 'legacy_available':
            from collections import Counter
            outside = Counter(r['baseline_exclusion'] for r in rows if not r['baseline_eligible'])
            print(f'Semi explicit historical source cohort: {len(rows)} CSV rows -> {len(baseline_ids)} samples; '
                  f'outside cohort: {dict(outside)}', flush=True)
        splits = []
        for original in split_iterator(baseline_ids, [0]*len(baseline_ids), seeds, cv5=cv5):
            saved = {key: original[key] for key in ('display', 'logger_id', 'seed', 'explain_id')}
            saved.update(original={}, kept={}, excluded={})
            for part in PARTS:
                ids = original[f'{part}_X']
                saved['original'][part] = ids
                saved['kept'][part] = [sid for sid in ids if not by_id[sid]['quality_reasons']]
                saved['excluded'][part] = [{'source_id': sid, 'reasons': by_id[sid]['quality_reasons']}
                                           for sid in ids if by_id[sid]['quality_reasons']]
            splits.append(saved)
        manifest = validate_manifest({'schema': SCHEMA, 'dataset': dataset, 'policy': policy,
                                       'records': rows, 'splits': splits, 'cv5': bool(cv5),
                                       'host_sha256': host_hashes,
                                       'database_sha256': digest(db_path) if dataset == 'imp2d' else None})
        run_dir.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8') as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write('\n')
    write_tables(manifest, run_dir)
    for split in requested_splits(manifest, seeds, cv5):
        print(f'{dataset} physical filter [{split["display"]}]: '+', '.join(
            f'{p} {len(split["original"][p])}->{len(split["kept"][p])}' for p in PARTS), flush=True)
    return manifest


def write_tables(manifest, run_dir):
    run_dir, dataset = Path(run_dir), manifest['dataset']
    rows = {r['source_id']: {**r, 'quality_reasons': ';'.join(r['quality_reasons']),
                           'review_flags': ';'.join(r['review_flags'])} for r in manifest['records']}
    pd.DataFrame(rows.values()).to_csv(run_dir/f'{dataset}_filter_structure_quality.csv', index=False)
    if dataset == 'semi':
        outside = [r for r in rows.values() if not r['baseline_eligible']]
        pd.DataFrame(outside).to_csv(run_dir/'semi_filter_outside_original_cohort.csv', index=False)
        conflicts = [r for r in rows.values() if 'identical_structure_inconsistent_target' in r['quality_reasons']]
        pd.DataFrame(conflicts).to_csv(run_dir/'semi_filter_label_conflicts.csv', index=False)
        review = [r for r in rows.values() if r['baseline_eligible'] and
                  ('short_bond_review_only' in r['review_flags'] or
                   'weakly_bound_or_isolated_atoms_review_only' in r['review_flags'])]
        pd.DataFrame(review).to_csv(run_dir/'semi_filter_geometry_review.csv', index=False)
    summaries = []
    for split in manifest['splits']:
        removed, kept = [], []
        for part in PARTS:
            kept.extend({'split': part, **rows[sid]} for sid in split['kept'][part])
            removed.extend({'split': part, **rows[r['source_id']]} for r in split['excluded'][part])
            summaries.append({'logger_id': split['logger_id'], 'split': part,
                              'original': len(split['original'][part]), 'kept': len(split['kept'][part]),
                              'removed': len(split['excluded'][part])})
        columns = ['split', *next(iter(rows.values())).keys()]
        pd.DataFrame(removed).reindex(columns=list(dict.fromkeys(columns + [k for r in removed for k in r]))).to_csv(
            run_dir/f'{dataset}_filter_seed{split["logger_id"]}_removed.csv', index=False)
        pd.DataFrame(kept).to_csv(run_dir/f'{dataset}_filter_seed{split["logger_id"]}_kept.csv', index=False)
    pd.DataFrame(summaries).to_csv(run_dir/f'{dataset}_filter_summary.csv', index=False)


def filtered_splits(data, targets, manifest, seeds, cv5=False):
    by_id = {}
    for structure, target in zip(data, targets):
        sid = getattr(structure, 'source_id', None)
        if not sid or sid in by_id:
            raise ValueError('Physical-filter training requires unique source IDs')
        by_id[sid] = (structure, target)
    for saved in requested_splits(manifest, seeds, cv5):
        split = {key: saved[key] for key in ('display', 'logger_id', 'seed', 'explain_id')}
        for part in PARTS:
            ids = saved['kept'][part]
            if set(ids)-set(by_id):
                raise ValueError('Filtered impurity sources missing from loaded graphs')
            split[f'{part}_X'] = [by_id[sid][0] for sid in ids]
            split[f'{part}_y'] = [by_id[sid][1] for sid in ids]
        split['impurity_filter_counts'] = {part: {'original': len(saved['original'][part]),
                                                  'kept': len(saved['kept'][part]),
                                                  'removed': len(saved['excluded'][part])} for part in PARTS}
        yield split
