"""Verify downloaded impurity CIFs; never treat empty downloads as bad physics.

The fallback prepared_data.csv is audited by name, never silently installed as
id_prop.csv: its cohort/order may differ from the historical training list.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import warnings

from ase.db import connect
from ase.io import read as ase_read
import pandas as pd

from ..data.impurity_preprocessing import digest, same_periodic_geometry, mark_identical_label_conflicts
from ..data.impurity_quality import (combine_checks, finite, inspect_geometry,
                                    inspect_imp2d_metadata, inspect_impurity_composition, POLICY)
from .audit_impurity_quality import semi_inventory

ROOT = Path(__file__).resolve().parents[1]


def verify_imp2d(data_root, database, output):
    folder = data_root/'imp2d/imp2d'
    primary = folder/'id_prop.csv'
    fallback = folder.parent/'prepared_data.csv'
    source = next((p for p in (primary, fallback) if p.is_file() and p.stat().st_size), None)
    inventory = {'training_csv': str(primary.resolve()),
                 'training_csv_bytes': primary.stat().st_size if primary.is_file() else None,
                 'cif_files': len(list(folder.glob('*.cif'))),
                 'nonempty_cif_files': sum(p.stat().st_size > 0 for p in folder.glob('*.cif'))}
    if source is None:
        return {'status': 'missing_nonempty_label_table', **inventory}
    frame = pd.read_csv(source, header=None)
    if frame.shape[1] < 2 or frame[0].duplicated().any():
        raise ValueError(f'Expected unique source IDs and targets: {source}')
    with connect(str(database)) as db:
        db_rows = {row.name: row for row in db.select()}
    records = []
    for sid, value, *_ in frame.itertuples(index=False, name=None):
        sid, target = str(sid), finite(value)
        if '/' in sid or '\\' in sid or sid in ('', '.', '..'):
            raise ValueError(f'Invalid source ID: {sid}')
        path = folder/(sid+'.cif')
        row = db_rows.get(sid)
        record = {'source_id': sid, 'target': target, 'download_status': 'missing',
                  'download_bytes': None, 'download_sha256': None, 'geometry_verified': False,
                  'label_verified': False, 'quality_reasons': [], 'review_flags': []}
        if row is not None:
            record = combine_checks(record, inspect_imp2d_metadata(row.key_value_pairs))
            record['label_verified'] = bool(target is not None and finite(row.get('eform')) is not None
                                            and abs(target-row.eform) <= POLICY['energy_identity_tolerance_ev'])
        if path.is_file():
            record['download_bytes'] = path.stat().st_size
            if record['download_bytes'] == 0:
                record['download_status'] = 'empty_download'
            else:
                record['download_sha256'] = digest(path)
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', UserWarning)
                        atoms = ase_read(str(path), format='cif', store_tags=True)
                    record['download_status'] = 'readable'
                    record = combine_checks(record, inspect_geometry(atoms),
                                            inspect_impurity_composition(atoms, sid, 'imp2d'))
                    if row is not None:
                        record['geometry_verified'] = bool(same_periodic_geometry(atoms, row.toatoms()))
                        if not record['geometry_verified']:
                            record['download_status'] = 'database_geometry_mismatch'
                except Exception as exc:
                    record['download_status'] = 'unreadable_download'
                    record['download_error'] = f'{type(exc).__name__}: {exc}'
        records.append(record)
        if len(records) % 500 == 0:
            print(f'Download verification: {len(records)}/{len(frame)} records', flush=True)
    flat = [{**r, 'quality_reasons': ';'.join(r['quality_reasons']),
             'review_flags': ';'.join(r['review_flags'])} for r in records]
    table = pd.DataFrame(flat)
    table.to_csv(output/'imp2d_download_verification.csv', index=False)
    table.loc[table['download_status'] != 'readable'].to_csv(output/'imp2d_incomplete_or_mismatched_files.csv', index=False)
    table.loc[table['quality_reasons'] != ''].to_csv(output/'imp2d_candidate_physical_exclusions.csv', index=False)
    counts = Counter(r['download_status'] for r in records)
    return {'status': 'verified' if counts['readable'] == len(records) and all(r['label_verified'] for r in records)
                     and source == primary else 'incomplete_or_alternative_label_table',
            **inventory, 'audited_label_table': str(source.resolve()), 'label_table_sha256': digest(source),
            'used_alternative_label_table': source != primary, 'label_table_rows': len(records),
            'database_sha256': digest(database), 'policy': POLICY,
            'all_labels_match_database': all(r['label_verified'] for r in records),
            'download_status_counts': dict(counts),
            'physical_candidate_exclusions_from_available_checks': sum(bool(r['quality_reasons']) for r in records),
            'physical_candidate_reason_counts_overlapping': dict(Counter(reason for r in records for reason in r['quality_reasons'])),
            'fully_verified_retained_files': sum(r['download_status'] == 'readable' and r['geometry_verified']
                                                 and r['label_verified'] and not r['quality_reasons'] for r in records),
            'training_manifest_created': False,
            'note': 'Missing/empty files are download failures, not physical exclusions. Alternative CSV order is not a verified historical split.'}


def verify_semi(data_root, output):
    folder = data_root/'Dataset_1/Dataset_1/Neutral/Neutral'
    csv = folder/'id_prop_A_rich.csv'
    targets = {}
    label_table_available = csv.is_file() and csv.stat().st_size > 0
    if label_table_available:
        frame = pd.read_csv(csv, header=None)
        if frame.shape[1] < 2 or frame[0].duplicated().any():
            raise ValueError(f'Expected unique semi IDs and targets: {csv}')
        targets = {str(r[0]): finite(r[1]) for r in frame.itertuples(index=False, name=None)}
        ids = list(targets)
    else:
        ids = sorted(p.name for p in folder.glob('*.cif'))
    records = []
    for sid in ids:
        if '/' in sid or '\\' in sid or sid in ('', '.', '..'):
            raise ValueError(f'Invalid semi ID: {sid}')
        path = folder/sid
        record = {'source_id': sid, 'target': targets.get(sid), 'label_available': sid in targets,
                  'download_status': 'missing', 'download_bytes': None, 'download_sha256': None,
                  'quality_reasons': [], 'review_flags': []}
        if path.is_file():
            record['download_bytes'] = path.stat().st_size
            if record['download_bytes'] == 0:
                record['download_status'] = 'empty_download'
            else:
                record['download_sha256'] = digest(path)
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', UserWarning)
                        atoms = ase_read(str(path), format='cif', store_tags=True)
                    record['download_status'] = 'readable'
                    record = combine_checks(record, inspect_geometry(atoms),
                                            inspect_impurity_composition(atoms, sid, 'semi'))
                    if record.get('material'):
                        host = data_root/'Dataset_1/host_configurations'/(record['material']+'.vasp')
                        record['host_file_available'] = host.is_file() and host.stat().st_size > 0
                    record['review_flags'].append('DFT_convergence_unverified_no_metadata')
                    if label_table_available and record['target'] is None:
                        record['quality_reasons'].append('nonfinite_target')
                except Exception as exc:
                    record['download_status'] = 'unreadable_download'
                    record['download_error'] = f'{type(exc).__name__}: {exc}'
        records.append(record)
        if len(records) % 500 == 0:
            print(f'Semi download verification: {len(records)}/{len(ids)} records', flush=True)
    conflicts = mark_identical_label_conflicts(records, hash_key='download_sha256') if label_table_available else 0
    flat = [{**r, 'quality_reasons': ';'.join(r['quality_reasons']),
             'review_flags': ';'.join(r['review_flags'])} for r in records]
    table = pd.DataFrame(flat)
    if not table.empty:
        table.to_csv(output/'semi_download_verification.csv', index=False)
        table.loc[table['download_status'] != 'readable'].to_csv(output/'semi_incomplete_files.csv', index=False)
        candidates = [r for r in flat if r['download_status'] == 'readable'
                      and r.get('baseline_eligible') and r['quality_reasons']]
        pd.DataFrame(candidates).reindex(columns=table.columns).to_csv(output/'semi_candidate_physical_exclusions.csv', index=False)
        review = table.loc[(table.get('baseline_eligible') == True)
                           & table['review_flags'].str.contains('short_bond_review_only|weakly_bound_or_isolated_atoms_review_only')]
        review.to_csv(output/'semi_geometry_review.csv', index=False)
    else:
        candidates = []
    status = ('structure_audit_complete' if label_table_available and records
              and all(r['download_status'] == 'readable' for r in records)
              else 'partial_structure_audit' if any(r['download_status'] == 'readable' for r in records)
              else 'unavailable_for_structure_audit')
    return {**semi_inventory(data_root, output), 'status': status,
            'scope': 'training_CSV' if label_table_available else 'available_CIFs_without_labels',
            'training_csv_available': bool(label_table_available), 'records_checked': len(records),
            'download_status_counts': dict(Counter(r['download_status'] for r in records)),
            'readable_extrinsic_structures': sum(r['download_status'] == 'readable' and bool(r.get('baseline_eligible')) for r in records),
            'legacy_nonextrinsic_structures': sum(r['download_status'] == 'readable' and r.get('baseline_eligible') is False for r in records),
            'candidate_physical_exclusions': len(candidates),
            'identical_structure_conflict_groups': conflicts,
            'review_flag_counts_overlapping': dict(Counter(flag for r in records
                if r['download_status'] == 'readable' and r.get('baseline_eligible') for flag in r['review_flags'])),
            'candidate_reason_counts_overlapping': dict(Counter(reason for r in records
                if r['download_status'] == 'readable' and r.get('baseline_eligible') for reason in r['quality_reasons'])),
            'training_manifest_created': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ROOT.parent/'dataset')
    parser.add_argument('--imp2d-db', type=Path, default=ROOT/'tmp/imp2d_audit/imp2d.db')
    parser.add_argument('--output', type=Path, default=ROOT/'results/impurity_download_verification')
    parser.add_argument('--dataset', nargs='+', choices=['imp2d', 'semi'], default=['imp2d', 'semi'])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'raw_files_modified': False, 'training_performed': False}
    for dataset in args.dataset:
        report[dataset] = (verify_imp2d(args.data_root, args.imp2d_db, args.output) if dataset == 'imp2d'
                           else verify_semi(args.data_root, args.output))
    path = args.output/'summary.json'
    path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
