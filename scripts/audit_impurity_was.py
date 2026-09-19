"""Audit imp2d/semi original-species inputs without training or rewriting data.

Prefer the actual CSV/CIF training files. If imp2d CSVs are absent, an ASE
database can validate source structures, with that narrower scope recorded.
"""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import warnings

import numpy as np
from pymatgen.core import Structure

from ..data.converters import AtomFeaturesExtractor
from ..data.datasets import (init_elem_embedding, tag_structure_source, formula_contains_element,
                             _load_imp2d_self_defect_labels, _assign_imp2d_self_defect_indices,
                             _read_impurity_manifest)
from ..data.structure_utils import convert_to_sparse_imp2d, convert_to_sparse_semi


ROOT = Path(__file__).resolve().parents[1]


def audit_structure(raw, info, dataset, stats):
    convert = convert_to_sparse_imp2d if dataset == 'imp2d' else convert_to_sparse_semi
    options = {f'{dataset}_preprocessing': 'reference_v1'}
    hetero = convert(raw, info, 1, 'alignn_hetero', None, True, **options)
    attention = convert(raw, info, 1, 'alignn_attention', None, True, **options)
    legacy = convert(raw, info, 1, 'alignn_hetero', None, True)
    extractor = AtomFeaturesExtractor('was_species', 'alignn_hetero_was',
                                      impurity_preprocessing=f'{dataset}_reference_v1')
    features = extractor.convert(hetero)
    np.testing.assert_array_equal(features, extractor.convert(attention))
    assert len(hetero) == len(raw), raw.source_id
    np.testing.assert_array_equal(hetero.frac_coords, raw.frac_coords)
    assert hetero.species == raw.species, raw.source_id
    assert hetero.site_properties['type'] == attention.site_properties['type'], raw.source_id
    defect = [i for i, site in enumerate(hetero) if site.properties['pool_type']]
    assert len(defect) == 1, raw.source_id
    changed = np.flatnonzero(np.any(features[:, :92] != features[:, 92:], axis=1)).tolist()
    assert changed == defect, raw.source_id
    old_features = AtomFeaturesExtractor('was_species', 'alignn_hetero_was').convert(legacy)
    assert np.array_equal(old_features[:, :92], old_features[:, 92:]), raw.source_id
    if dataset == 'imp2d':
        assert hetero[defect[0]].properties['was'] == 0, raw.source_id
        assert len(legacy) == len(hetero), raw.source_id
        assert legacy.site_properties['type'] == hetero.site_properties['type'], raw.source_id
    stats['structures_validated'] += 1
    stats['nodes_validated'] += len(hetero)
    stats['legacy_duplicate_was_structures'] += 1
    stats['legacy_changed_atom_count'] += len(legacy) != len(hetero)


def audit_imp2d(args):
    csv_path = args.data_root / 'imp2d/imp2d/id_prop.csv'
    lookup = _load_imp2d_self_defect_labels()
    stats = Counter()
    counts = Counter()
    source = 'training_csv_and_cifs'

    def csv_entries():
        for name, target, *_ in _read_impurity_manifest(csv_path, 'imp2d').itertuples(index=False, name=None):
            if not -10 < target < 10:
                counts['excluded_energy_range'] += 1
                continue
            path = csv_path.parent / f'{name}.cif'
            raw = Structure.from_file(path)
            tag_structure_source(raw, path, name)
            yield name, raw

    def db_entries():
        from ase.db import connect
        from pymatgen.io.ase import AseAtomsAdaptor
        from pymatgen.io.cif import CifWriter
        roundtrip_groups = set()
        for row in connect(str(args.imp2d_db)).select():
            counts['db_rows'] += 1
            energy = row.get('eform')
            valid_energy = energy is not None and math.isfinite(energy) and -10 < energy < 10
            base, impurity, site = row.name.split('_')
            is_self = formula_contains_element(base, impurity)
            if valid_energy and is_self and row.name not in lookup:
                counts['missing_lookup_converged' if row.get('converged') else 'missing_lookup_unconverged'] += 1
            if not valid_energy or not row.get('converged'):
                continue
            raw = AseAtomsAdaptor.get_structure(row.toatoms())
            # Check CIF reorder/label preservation for every self impurity and
            # one non-self structure in each host/site stratum.
            group = (base, site)
            if is_self or group not in roundtrip_groups:
                raw = Structure.from_str(str(CifWriter(raw)), 'cif')
                roundtrip_groups.add(group)
                counts['cif_roundtrips'] += 1
            tag_structure_source(raw, f'{args.imp2d_db}#row={row.id}', row.name)
            yield row.name, raw

    if csv_path.is_file() and csv_path.stat().st_size:
        entries = csv_entries()
    elif args.imp2d_db.is_file():
        source = 'local_ase_db_converged_finite_minus10_to10_NOT_training_csv'
        entries = db_entries()
    else:
        return {'status': 'unavailable', 'training_csv': str(csv_path), 'database': str(args.imp2d_db)}
    for name, raw in entries:
        base, impurity, site = name.split('_')
        info = dict(base=base, impurity=impurity, site=site, is_self=formula_contains_element(base, impurity))
        _assign_imp2d_self_defect_indices([[raw, info]], lookup)
        counts['self_impurities' if info['is_self'] else 'nonself_impurities'] += 1
        counts['adsorbates' if site.startswith('ads') else 'interstitials'] += 1
        audit_structure(raw, info, 'imp2d', stats)
        if stats['structures_validated'] % 500 == 0:
            print(f'imp2d: {stats["structures_validated"]} validated', flush=True)
    return {'status': 'passed', 'source': source, **stats, **counts,
            'self_lookup_entries': len(lookup), 'actual_training_csv_verified': source == 'training_csv_and_cifs'}


def audit_semi(args):
    folder = args.data_root / 'Dataset_1/Dataset_1/Neutral/Neutral'
    csv_path = folder / 'id_prop_A_rich.csv'
    hosts = args.data_root / 'Dataset_1/host_configurations'
    files = list(folder.glob('*.cif'))
    nonempty = sum(path.stat().st_size > 0 for path in files)
    missing_hosts = sorted({path.name.split('-')[1] for path in files if len(path.name.split('-')) >= 2}
                           - {path.stem for path in hosts.glob('*.vasp')})
    inventory = {'cif_files': len(files), 'nonempty_cifs': nonempty, 'missing_host_files': missing_hosts,
                 'csv_path': str(csv_path), 'csv_bytes': csv_path.stat().st_size if csv_path.is_file() else None}
    if not csv_path.is_file() or not csv_path.stat().st_size:
        return {'status': 'unavailable_for_real_structure_validation', **inventory}
    stats = Counter()
    for name, target, *_ in _read_impurity_manifest(csv_path, 'semi').itertuples(index=False, name=None):
        raw = Structure.from_file(folder / name)
        tag_structure_source(raw, folder / name, name)
        host = Structure.from_file(hosts / f'{name.split("-")[1]}.vasp')
        if not (set(raw.composition.get_el_amt_dict()) - set(host.composition.get_el_amt_dict())):
            stats['excluded_non_extrinsic'] += 1
            continue
        audit_structure(raw, host, 'semi', stats)
        if stats['structures_validated'] % 500 == 0:
            print(f'semi: {stats["structures_validated"]} validated', flush=True)
    return {'status': 'passed', **inventory, **stats}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ROOT.parent / 'dataset')
    parser.add_argument('--imp2d-db', type=Path, default=ROOT / 'tmp/imp2d_audit/imp2d.db')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/impurity_was_audit_20260920.json')
    args = parser.parse_args()
    init_elem_embedding(ROOT / 'atom_init.json')
    report = {'training_performed': False, 'preprocessing': 'reference_v1'}
    failures = []
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Issues encountered while parsing CIF:.*')
        for dataset, audit in [('imp2d', audit_imp2d), ('semi', audit_semi)]:
            try:
                report[dataset] = audit(args)
            except Exception as exc:
                report[dataset] = {'status': 'failed', 'error': str(exc)}
                failures.append(dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    print(f'Saved: {args.output}', flush=True)
    if failures:
        raise SystemExit(f'Audit failed: {failures}; inspect the saved report')


if __name__ == '__main__':
    main()
