"""Read, physically screen and split original IMP2D ASE rows without CIF/CSV."""
from collections import Counter, defaultdict
import copy
import json
from pathlib import Path

from ase.db import connect
from pymatgen.io.ase import AseAtomsAdaptor
import pandas as pd
import numpy as np

from .impurity_quality import (POLICY, combine_checks, finite, inspect_geometry,
                               inspect_imp2d_metadata, inspect_impurity_composition)
from .structure_groups import assign_structure_groups

SELF_INDICES = Path(__file__).with_name('imp2d_self_defect_indices.json')
SOURCE_FORMAT = 'ase_database'
HOST_FILTERS = {
    'reviewed_v1': {
        'excluded_materials': ['CO2Ti2'],
        'reason': 'host_energy_reference_under_review',
        'basis': 'Host-wide formation-energy offset in Ti2CO2; reference calculation requires review.',
        'selection': 'Fixed reviewed host list; no individual energy trimming or prediction-error filtering.',
    },
}


def resolve_database(database_path=None):
    from .impurity_preprocessing import default_database
    from .imp2d_source import download_imp2d_database
    path = Path(database_path) if database_path is not None else default_database()
    if path is None:
        path = download_imp2d_database('dataset/imp2d/imp2d.db')
    if not path.is_file() or not path.stat().st_size:
        raise ValueError(f'Missing/empty IMP2D database: {path}')
    return path


def database_policy(host_filter=None, energy_window=None):
    from .impurity_preprocessing import digest
    policy = copy.deepcopy(POLICY)
    policy.update(version='imp2d_database_physical_v1', source_format=SOURCE_FORMAT,
                  imp2d_legacy_energy_window_ev=None, source_converged_flag_filter=False,
                  split_protocol='equivalent_structure_groups_v1', structure_tolerance_angstrom=1e-3,
                  self_defect_indices_sha256=digest(SELF_INDICES),
                  unidentified_self_impurities='exclude_from_model_cohort_and_report_separately')
    if host_filter is not None:
        if host_filter not in HOST_FILTERS:
            raise ValueError(f'Unknown IMP2D host filter: {host_filter}')
        policy.update(version='imp2d_database_host_review_v1',
                      host_filter={'name': host_filter, **copy.deepcopy(HOST_FILTERS[host_filter])})
    if energy_window is not None:
        if len(energy_window) != 2 or not (-float('inf') < energy_window[0] < energy_window[1] < float('inf')):
            raise ValueError('IMP2D energy window requires two finite increasing bounds')
        policy.update(version='imp2d_database_energy_window_v1', energy_window={
            'lower_ev': float(energy_window[0]), 'upper_ev': float(energy_window[1]),
            'bounds': 'open', 'reason': 'outside_benchmark_energy_window',
            'purpose': 'Fixed benchmark scope; not a physical validity criterion.',
        })
    return policy


def energy_window_reasons(record, policy):
    window = policy.get('energy_window')
    target = record['target']
    # Nonfinite targets already fail the physical metadata checks.
    if window and target is not None and not window['lower_ev'] < target < window['upper_ev']:
        return [window['reason']]
    return []


def validate_database_manifest(manifest):
    from .impurity_preprocessing import validate_manifest
    validate_manifest(manifest)
    if manifest.get('source_format') != SOURCE_FORMAT or manifest['dataset'] != 'imp2d':
        raise ValueError('Run directory uses another IMP2D data source; use a new --run-dir')
    rows = {r['source_id']: r for r in manifest['records']}
    host_policy = manifest['policy'].get('host_filter')
    window = manifest['policy'].get('energy_window')
    if window and (window.get('bounds') != 'open' or
                   not -float('inf') < window['lower_ev'] < window['upper_ev'] < float('inf')):
        raise ValueError('Invalid IMP2D energy window in frozen policy')
    if host_policy or window:
        for row in rows.values():
            expected = ([host_policy['reason']]
                        if host_policy and row.get('material') in host_policy['excluded_materials'] else [])
            energy_expected = energy_window_reasons(row, manifest['policy'])
            if (row.get('cohort_exclusion_reasons', []) != expected or
                    row['quality_reasons'] != row['physical_quality_reasons'] +
                    row['input_quality_reasons'] + expected + energy_expected):
                raise ValueError('IMP2D host exclusion or energy window disagrees with frozen cohort policy')
            if window and row.get('energy_window_exclusion_reasons') != energy_expected:
                raise ValueError('IMP2D energy window exclusion disagrees with frozen cohort policy')
    for split in manifest['splits']:
        assigned = {}
        for part in ('train', 'val', 'test'):
            for sid in split['original'][part]:
                group = rows[sid]['structure_group']
                if assigned.setdefault(group, part) != part:
                    raise ValueError('Equivalent IMP2D structures cross split boundaries')
    return manifest


def prepare_database_filter(run_dir, seeds, cv5, split_iterator, database_path=None, host_filter=None,
                            energy_window=None):
    from .impurity_preprocessing import (SCHEMA, digest, requested_splits, write_tables)
    database, run_dir = resolve_database(database_path), Path(run_dir)
    database_sha = digest(database)
    policy = database_policy(host_filter, energy_window)
    manifest_path = run_dir/'imp2d_filter_manifest.json'
    if manifest_path.exists():
        manifest = validate_database_manifest(json.loads(manifest_path.read_text(encoding='utf-8')))
        if manifest['policy'] != policy or manifest['database_sha256'] != database_sha:
            raise ValueError('IMP2D database/filter policy changed; use a new --run-dir')
        requested_splits(manifest, seeds, cv5)
        print(f'IMP2D db: reusing verified physical filter {manifest_path}', flush=True)
    else:
        checked = json.loads(SELF_INDICES.read_text(encoding='utf-8'))
        indices = checked['indices'] if checked['database_sha256'] == database_sha else {}
        records, atoms_by_id, seen = [], {}, set()
        with connect(str(database)) as db:
            for row in db.select(sort='id'):
                sid = str(row.name)
                if sid in seen:
                    raise ValueError(f'Ambiguous IMP2D source ID: {sid}')
                seen.add(sid)
                atoms = row.toatoms()
                record = combine_checks(inspect_imp2d_metadata(row.key_value_pairs),
                                        inspect_impurity_composition(atoms, sid, 'imp2d'))
                # Already-rejected DFT records do not need expensive neighbor scans.
                if not record['quality_reasons']:
                    record = combine_checks(record, inspect_geometry(atoms))
                    record['geometry_checked'] = True
                else:
                    record['geometry_checked'] = False
                record.update(source_id=sid, target=finite(row.get('eform')), sha256=None,
                              database_row_id=int(row.id), baseline_eligible=True,
                              physical_quality_reasons=list(record['quality_reasons']),
                              spin=finite(row.get('spin')), defect_index=None,
                              input_quality_reasons=[])
                if not record['quality_reasons']:
                    is_self = 'self_impurity_requires_checked_site_label' in record['review_flags']
                    if is_self:
                        index = indices.get(sid)
                    else:
                        matches = [i for i, symbol in enumerate(atoms.get_chemical_symbols())
                                   if symbol == record['impurity']]
                        index = matches[0] if len(matches) == 1 else None
                    if index is None:
                        record['input_quality_reasons'].append('unverified_self_impurity_identity')
                        record['quality_reasons'].append('unverified_self_impurity_identity')
                    else:
                        if not 0 <= index < len(atoms) or atoms[index].symbol != record['impurity']:
                            raise ValueError(f'Checked defect index disagrees with database: {sid}')
                        record['defect_index'] = int(index)
                        record['defect_index_source'] = 'checked_database_index' if is_self else 'unique_extrinsic_species'
                        atoms_by_id[sid] = atoms
                records.append(record)
                if len(records) % 1000 == 0:
                    print(f'IMP2D db physical filter: {len(records)} rows inspected', flush=True)
        groups, pairs = assign_structure_groups(records, atoms_by_id,
                                                tolerance=policy['structure_tolerance_angstrom'])
        # Build the same structure groups before applying the host restriction.
        # Remaining samples therefore keep their previous split membership.
        host_policy = policy.get('host_filter')
        if host_policy:
            for record in records:
                reasons = ([host_policy['reason']]
                           if record.get('material') in host_policy['excluded_materials'] else [])
                record['cohort_exclusion_reasons'] = reasons
                record['quality_reasons'].extend(reasons)
        if policy.get('energy_window'):
            for record in records:
                reasons = energy_window_reasons(record, policy)
                record['energy_window_exclusion_reasons'] = reasons
                record['quality_reasons'].extend(reasons)
        grouped = defaultdict(list)
        for record in records:
            record['structure_group'] = groups[record['source_id']]
            grouped[record['structure_group']].append(record)
        for members in grouped.values():
            retained = [r for r in members if not r['quality_reasons']]
            if len(retained) > 1:
                gap = max(r['target'] for r in retained)-min(r['target'] for r in retained)
                for r in retained:
                    r['equivalent_group_energy_spread_ev'] = gap
                    if gap > 0.05:
                        r['review_flags'].append('equivalent_structure_energy_disagreement_review')
        splits = []
        group_ids = list(grouped)
        for original in split_iterator(group_ids, [0]*len(group_ids), seeds, cv5=cv5):
            saved = {key: original[key] for key in ('display', 'logger_id', 'seed', 'explain_id')}
            saved.update(original={}, kept={}, excluded={})
            for part in ('train', 'val', 'test'):
                selected_groups = set(original[f'{part}_X'])
                selected = [r for r in records if r['structure_group'] in selected_groups]
                saved['original'][part] = [r['source_id'] for r in selected]
                saved['kept'][part] = [r['source_id'] for r in selected if not r['quality_reasons']]
                saved['excluded'][part] = [{'source_id': r['source_id'], 'reasons': r['quality_reasons']}
                                          for r in selected if r['quality_reasons']]
            splits.append(saved)
        manifest = validate_database_manifest({'schema': SCHEMA, 'dataset': 'imp2d',
                    'source_format': SOURCE_FORMAT, 'policy': policy, 'records': records,
                    'splits': splits, 'cv5': bool(cv5), 'host_sha256': {},
                    'database_sha256': database_sha, 'equivalent_structure_pairs': pairs})
        run_dir.mkdir(parents=True, exist_ok=True)
        with manifest_path.open('x', encoding='utf-8') as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write('\n')
    write_tables(manifest, run_dir)
    rows = manifest['records']
    pd.DataFrame([r for r in rows if r['input_quality_reasons']]).to_csv(
        run_dir/'imp2d_filter_unverified_defect_identity.csv', index=False)
    retained = [r for r in rows if not r['quality_reasons']]
    print(f'IMP2D db: {len(rows)} source rows, '
          f'{sum(not r["physical_quality_reasons"] for r in rows)} pass current physical checks, '
          f'{sum(bool(r["input_quality_reasons"]) for r in rows)} lack verified defect identity, '
          f'{len(retained)} model-ready rows', flush=True)
    if policy.get('host_filter'):
        write_host_filter_report(manifest, run_dir)
    if policy.get('energy_window'):
        write_energy_window_report(manifest, run_dir)
    for split in requested_splits(manifest, seeds, cv5):
        print(f'IMP2D db grouped split [{split["display"]}]: '+', '.join(
            f'{p} {len(split["kept"][p])}' for p in ('train', 'val', 'test')), flush=True)
    return manifest


def write_host_filter_report(manifest, run_dir):
    """Document a fixed material restriction separately from physical failures."""
    from .impurity_preprocessing import identity
    run_dir = Path(run_dir)
    records = manifest['records']
    baseline = [r for r in records if not r['physical_quality_reasons'] and not r['input_quality_reasons']]
    retained = [r for r in records if not r['quality_reasons']]
    removed = [r for r in baseline if r['cohort_exclusion_reasons']]
    grouped = defaultdict(list)
    for row in baseline:
        grouped[row['material']].append(row)
    host_rows = []
    for host, members in sorted(grouped.items()):
        energies = np.array([r['target'] for r in members])
        quantiles = np.quantile(energies, [0, .05, .25, .5, .75, .95, 1])
        host_rows.append({'material': host, 'before': len(members),
                          'retained': sum(not r['quality_reasons'] for r in members),
                          'excluded_host': bool(members[0]['cohort_exclusion_reasons']),
                          **dict(zip(('min_ev', 'p05_ev', 'p25_ev', 'median_ev', 'p75_ev', 'p95_ev', 'max_ev'),
                                     map(float, quantiles))),
                          'outside_legacy_10ev_window': int((np.abs(energies) >= 10).sum())})
    pd.DataFrame(host_rows).to_csv(run_dir/'imp2d_host_distribution.csv', index=False)
    # Includes only additional removals from the previously model-ready cohort.
    pd.DataFrame(removed).to_csv(run_dir/'imp2d_host_excluded.csv', index=False)
    tails = [r for r in retained if abs(r['target']) >= 10]
    pd.DataFrame(tails).to_csv(run_dir/'imp2d_remaining_energy_review.csv', index=False)
    values = [r['target'] for r in retained]
    summary = {'filter_identity': identity(manifest),
               'source_rows': len(records), 'before_host_filter': len(baseline),
               'excluded_materials': manifest['policy']['host_filter']['excluded_materials'],
               'raw_rows_in_excluded_hosts': sum(bool(r['cohort_exclusion_reasons']) for r in records),
               'additional_excluded_rows': len(removed), 'retained_rows': len(retained),
               'retained_materials': len({r['material'] for r in retained}),
               'retained_energy_quantiles_ev': dict(zip(('min', 'p05', 'median', 'p95', 'max'),
                                                        map(float, np.quantile(values, [0, .05, .5, .95, 1])))),
               'retained_outside_legacy_10ev_window': len(tails),
               'individual_energy_trimming_applied': bool(manifest['policy'].get('energy_window')),
               'splits': [{'logger_id': s['logger_id'], **{p: len(s['kept'][p]) for p in ('train','val','test')}}
                          for s in manifest['splits']]}
    (run_dir/'imp2d_host_filter_summary.json').write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False)+'\n', encoding='utf-8')
    print(f'IMP2D host filter: {len(removed)} model-ready rows excluded from '
          f'{manifest["policy"]["host_filter"]["excluded_materials"]}; '
          f'{len(retained)} rows after all filters', flush=True)


def write_energy_window_report(manifest, run_dir):
    from .impurity_preprocessing import identity
    run_dir = Path(run_dir)
    before = [r for r in manifest['records'] if not r['physical_quality_reasons'] and
              not r['input_quality_reasons'] and not r.get('cohort_exclusion_reasons')]
    removed = [r for r in before if r['energy_window_exclusion_reasons']]
    retained = [r for r in before if not r['energy_window_exclusion_reasons']]
    pd.DataFrame(removed).to_csv(run_dir/'imp2d_energy_window_excluded.csv', index=False)
    values = [r['target'] for r in retained]
    summary = {'filter_identity': identity(manifest), 'before_energy_window': len(before),
               'additional_excluded_rows': len(removed), 'retained_rows': len(retained),
               'retained_materials': len({r['material'] for r in retained}),
               'minimum_ev': min(values), 'maximum_ev': max(values),
               'splits': [{'logger_id': s['logger_id'], **{p: len(s['kept'][p]) for p in ('train','val','test')}}
                          for s in manifest['splits']]}
    (run_dir/'imp2d_energy_window_summary.json').write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False)+'\n', encoding='utf-8')
    window = manifest['policy']['energy_window']
    print(f'IMP2D energy window: {window["lower_ev"]} < DFE < {window["upper_ev"]} eV; '
          f'{len(before)} -> {len(retained)} rows ({len(removed)} excluded)', flush=True)


def load_database_inputs(manifest, database_path=None):
    from .impurity_preprocessing import digest, retained_ids
    from .datasets import tag_structure_source
    validate_database_manifest(manifest)
    database = resolve_database(database_path)
    if digest(database) != manifest['database_sha256']:
        raise ValueError('IMP2D database changed since preprocessing')
    wanted = retained_ids(manifest)
    records = {r['source_id']: r for r in manifest['records']}
    prepared, targets = [], []
    with connect(str(database)) as db:
        for row in db.select(sort='id'):
            sid = str(row.name)
            if sid not in wanted:
                continue
            record = records[sid]
            if finite(row.get('eform')) != record['target'] or row.id != record['database_row_id']:
                raise ValueError(f'IMP2D row identity/target changed: {sid}')
            structure = AseAtomsAdaptor.get_structure(row.toatoms())
            tag_structure_source(structure, f'{database}#row={row.id}', sid)
            index = record['defect_index']
            if index is None or structure[index].specie.symbol != record['impurity']:
                raise ValueError(f'Missing or invalid checked IMP2D defect index: {sid}')
            info = {'base': record['material'], 'impurity': record['impurity'], 'site': record['site'],
                    'is_self': 'self_impurity_requires_checked_site_label' in record['review_flags'],
                    'defect_index': index}
            prepared.append([structure, info])
            targets.append(record['target'])
    if {structure.source_id for structure, _ in prepared} != wanted:
        raise ValueError('IMP2D database did not supply every retained source')
    return prepared, targets
