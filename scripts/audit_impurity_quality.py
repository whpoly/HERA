"""Audit available impurity source data without training or altering raw files."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from ase.db import connect
from ase.io import read as ase_read
import pandas as pd

from ..data.impurity_quality import (POLICY, combine_checks, inspect_geometry, matches_formula,
                                      inspect_imp2d_metadata, inspect_impurity_composition)

ROOT = Path(__file__).resolve().parents[1]


def audit_database(path, output):
    if not path.is_file():
        return {'status': 'unavailable', 'path': str(path)}
    records = []
    with connect(str(path)) as db:
        for row in db.select():
            record = combine_checks(inspect_imp2d_metadata(row.key_value_pairs),
                                    inspect_geometry(row.toatoms()),
                                    inspect_impurity_composition(row.toatoms(), row.name, 'imp2d'))
            record['source_id'] = row.name
            record['legacy_energy_window'] = record['eform'] is not None and -10 < record['eform'] < 10
            records.append(record)
            if len(records) % 1000 == 0:
                print(f'IMP2D physical audit: {len(records)} structures', flush=True)
    cohorts = {}
    selectors = {
        'all_database_rows': lambda r: True,
        'finite_target': lambda r: r['eform'] is not None,
        'legacy_energy_window': lambda r: r['legacy_energy_window'],
        'legacy_energy_window_and_source_converged': lambda r: r['legacy_energy_window'] and r['reported_converged'],
    }
    for name, select in selectors.items():
        subset = [r for r in records if select(r)]
        reasons = Counter(x for r in subset for x in r['quality_reasons'])
        cohorts[name] = {'total': len(subset), 'retained': sum(not r['quality_reasons'] for r in subset),
                         'excluded': sum(bool(r['quality_reasons']) for r in subset),
                         'reason_counts_overlapping': dict(reasons)}
    flag_matches_first = sum(r['reported_converged'] == (r['conv1'] is not None and abs(r['conv1']) <= .05)
                             for r in records)
    frame = pd.DataFrame([{**r, 'quality_reasons': ';'.join(r['quality_reasons']),
                           'review_flags': ';'.join(r['review_flags'])} for r in records])
    frame.to_csv(output/'imp2d_database_quality.csv', index=False)
    frame.loc[frame['quality_reasons'] != ''].to_csv(output/'imp2d_database_quarantine.csv', index=False)
    frame.groupby(['material', 'host_spacegroup'], dropna=False).agg(
        total=('source_id', 'size'), excluded=('quality_reasons', lambda s: sum(s != ''))
    ).to_csv(output/'imp2d_by_host.csv')
    return {'status': 'audited', 'scope': 'original_ASE_database_NOT_actual_training_CSV',
            'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'converged_flag_matches_abs_conv1_le_0_05': flag_matches_first,
            'cohorts': cohorts, 'policy': POLICY,
            'metadata_inconsistency_examples': [r for r in records if 'inconsistent_energy_metadata' in r['quality_reasons']]}


def semi_inventory(data_root, output=None):
    folder = data_root/'Dataset_1/Dataset_1/Neutral/Neutral'
    csv = folder/'id_prop_A_rich.csv'
    files = list(folder.glob('*.cif'))
    present = [p for p in files if p.stat().st_size]
    hosts = data_root/'Dataset_1/host_configurations'
    missing = sorted({p.name.split('-')[1] for p in files if len(p.name.split('-')) >= 2}
                     - {p.stem for p in hosts.glob('*.vasp')})
    filename_counts = Counter(p.name.split('-')[1] for p in files if len(p.name.split('-')) >= 2)
    references = []
    for path in sorted(hosts.glob('*.vasp')):
        try:
            atoms = ase_read(str(path), format='vasp')
            record = {**inspect_geometry(atoms), 'material': path.stem,
                      'formula': atoms.get_chemical_formula(),
                      'matches_filename': bool(matches_formula(Counter(atoms.get_chemical_symbols()), path.stem))}
        except Exception as exc:
            record = {'material': path.stem, 'error': str(exc)}
        references.append(record)
    if output is not None:
        pd.DataFrame(references).to_csv(output/'semi_host_reference_quality.csv', index=False)
    ready = csv.is_file() and csv.stat().st_size and present
    return {'status': 'source_available_use_training_filter_for_full_audit' if ready else 'unavailable_for_physical_audit',
            'csv_path': str(csv), 'csv_bytes': csv.stat().st_size if csv.is_file() else None,
            'cif_files': len(files), 'nonempty_cifs': len(present), 'missing_hosts': missing,
            'host_references_checked': len(references),
            'host_geometry_rejection_count': sum(bool(r.get('quality_reasons') or r.get('error')) for r in references),
            'host_composition_mismatch_count': sum(not r.get('matches_filename') for r in references),
            'missing_host_filename_counts': {key: filename_counts[key] for key in missing}}


def audit_training_sources(dataset, args):
    from ..data.impurity_preprocessing import prepare_impurity_filter, TABLES, DATA_DIRS
    from ..main import iter_train_val_test_splits
    relative = Path(DATA_DIRS[dataset]).relative_to('dataset')
    folder = args.data_root/relative
    table = folder/TABLES[dataset]
    if not table.is_file() or not table.stat().st_size:
        return {'status': 'unavailable', 'csv_path': str(table)}
    try:
        manifest = prepare_impurity_filter(args.output, dataset, args.seed, args.cv5,
                                           iter_train_val_test_splits, data_dir=folder,
                                           database_path=args.imp2d_db,
                                           host_dir=args.data_root/'Dataset_1/host_configurations',
                                           semi_source_policy=getattr(args,'semi_source_policy','complete'))
    except Exception as exc:
        return {'status': 'blocked', 'error': f'{type(exc).__name__}: {exc}'}
    return {'status': 'audited', 'source_rows': len(manifest['records']),
            'baseline_rows': sum(r['baseline_eligible'] for r in manifest['records']),
            'excluded_baseline_rows': sum(r['baseline_eligible'] and bool(r['quality_reasons']) for r in manifest['records'])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--imp2d-db', type=Path, default=ROOT/'tmp/imp2d_audit/imp2d.db')
    parser.add_argument('--data-root', type=Path, default=ROOT.parent/'dataset')
    parser.add_argument('--output', type=Path, default=ROOT/'results/impurity_quality')
    parser.add_argument('--seed', type=int, nargs='+', default=[123])
    parser.add_argument('--cv5', action='store_true')
    parser.add_argument('--semi-source-policy', choices=['complete','legacy_available'], default='complete')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'training_performed': False, 'raw_files_modified': False,
              'imp2d': audit_database(args.imp2d_db, args.output), 'semi': semi_inventory(args.data_root, args.output)}
    report['actual_training_sources'] = {dataset: audit_training_sources(dataset, args) for dataset in ('imp2d', 'semi')}
    (args.output/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2, allow_nan=False))
    if any(r['status'] == 'blocked' for r in report['actual_training_sources'].values()):
        raise SystemExit('Actual training-source audit blocked; see summary.json before training')


if __name__ == '__main__':
    main()
