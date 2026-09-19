"""Audit LEGACY native WAS duplication, retained to reproduce the pre-fix diagnosis."""
import argparse
from collections import Counter, defaultdict
import copy
import csv
import json
from pathlib import Path
import re

import numpy as np
from pymatgen.core import Structure
from pymatgen.core.periodic_table import DummySpecies
import torch
from torch_geometric.data import Batch

from ..config.defaults import get_config, apply_alignn_hetero_options
from ..data.converters import AtomFeaturesExtractor
from ..data.datasets import init_elem_embedding
from ..data.structure_utils import convert_to_sparse_native
from ..main import iter_train_val_test_splits, set_seed
from ..training.trainer import MEGNetTrainer


ROOT = Path(__file__).resolve().parents[1]


def defect_family(filename):
    tag = filename.split('-')[2]
    return 'vacancy' if tag.startswith('V_') else ('interstitial' if '_i' in tag else 'substitution')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-list', type=Path, default=ROOT.parent / 'dataset/Dataset_1/Dataset_1/A_rich/Neutral/id_prop_A_rich.csv')
    parser.add_argument('--samples-per-stratum', type=int, default=2)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/native_was_audit_20260920.json')
    args = parser.parse_args()
    torch.set_num_threads(1)
    init_elem_embedding(ROOT / 'atom_init.json')
    with args.data_list.open(newline='', encoding='utf-8-sig') as stream:
        rows = [(name, float(target)) for name, target, *extra in csv.reader(stream)]
    strata = defaultdict(list)
    for i, (name, _) in enumerate(rows):
        strata[(name.split('-')[1], defect_family(name))].append(i)
    indices = sorted({values[int(i)] for values in strata.values()
                      for i in np.linspace(0, len(values) - 1, min(args.samples_per_stratum, len(values)), dtype=int)})
    structures, audits = [], []
    for index in indices:
        name, target = rows[index]
        raw = Structure.from_file(args.data_list.parent / name)
        family = defect_family(name)
        tag = 'vacancy' if family == 'vacancy' else 'others'
        hetero = convert_to_sparse_native(raw, tag, 1, 'alignn_hetero', None, True, False, local_cutoff=0)
        attention = convert_to_sparse_native(raw, tag, 1, 'alignn_attention', None, True, False, local_cutoff=None)
        plain = AtomFeaturesExtractor('Z', 'alignn_hetero').convert(hetero)
        was = AtomFeaturesExtractor('was_species', 'alignn_hetero_was').convert(hetero)
        was_attention = AtomFeaturesExtractor('was_species', 'alignn_attention_was').convert(attention)
        types = [int(site.properties['type']) for site in hetero]
        audit = {
            'source': name, 'material': name.split('-')[1], 'defect_family': family,
            'target': target, 'raw_sites': len(raw), 'graph_sites': len(hetero),
            'raw_was_sites': sum('was' in site.properties for site in raw),
            'was_sites': sum('was' in site.properties for site in hetero),
            'defects': sum(types), 'dummy_sites': sum(isinstance(site.specie, DummySpecies) for site in hetero),
            'reference_differs_from_current': int(np.count_nonzero(np.any(was[:, :92] != was[:, 92:], axis=1))),
            'first_half_equals_plain': bool(np.array_equal(was[:, :92], plain)),
            'was_halves_identical': bool(np.array_equal(was[:, :92], was[:, 92:])),
            'attention_hetero_features_identical': bool(np.array_equal(was_attention, was)),
            'attention_hetero_sites_identical': attention.species == hetero.species and bool(np.array_equal(attention.frac_coords, hetero.frac_coords)),
        }
        audits.append(audit)
        structures.append((hetero, attention))
    print(f'Audited {len(audits)} structures across {len(strata)} material/defect strata.', flush=True)

    model_checks = {}
    # Use one real sample from each defect family for the full-model equivalence check.
    probes = [next(i for i, item in enumerate(audits) if item['defect_family'] == family)
              for family in ('vacancy', 'substitution', 'interstitial')]
    for base_mode in ('attention', 'hetero'):
        models = {}
        for mode in (base_mode, base_mode + '_was'):
            config = get_config('alignn', 'native', mode)
            config.pop('native_preprocessing', None)  # Intentionally audit historical inputs.
            if base_mode == 'hetero':
                config['model']['local_radius'] = 0
                config['model']['train_batch_size'] = 8
                apply_alignn_hetero_options(config, pooling='defect_energy_mean', aa_mode='keep', dd_mode='keep')
            set_seed(123)
            models[mode] = MEGNetTrainer(config, 'cpu', seed=123)
            models[mode].model.eval()
        ordinary, duplicated = models[base_mode], models[base_mode + '_was']
        normal_state = ordinary.model.state_dict()
        was_state = duplicated.model.state_dict()
        changed = [key for key in normal_state if normal_state[key].shape == was_state[key].shape
                   and not torch.equal(normal_state[key], was_state[key])]
        folded = copy.deepcopy(was_state)
        input_keys = []
        for key in normal_state:
            if normal_state[key].shape != was_state[key].shape:
                assert normal_state[key].shape == (64, 92) and was_state[key].shape == (64, 184), key
                folded[key] = was_state[key][:, :92] + was_state[key][:, 92:]
                input_keys.append(key)
        errors = []
        geometry_matches = []
        ordinary.model.load_state_dict(folded, strict=True)
        for index in probes:
            structure = structures[index][1 if base_mode == 'attention' else 0]
            first = ordinary.converter.convert(structure)
            second = duplicated.converter.convert(structure)
            if base_mode == 'attention':
                geometry_matches.append(all(torch.equal(getattr(first, field), getattr(second, field))
                                            for field in ('edge_index', 'edge_attr', 'edge_vec', 'node_type')))
            else:
                geometry_matches.append(all(torch.equal(first[t][field], second[t][field])
                                            for t in first.edge_types for field in ('edge_index', 'edge_attr', 'edge_vec')))
            with torch.no_grad():
                a = ordinary._forward(Batch.from_data_list([first]))
                b = duplicated._forward(Batch.from_data_list([second]))
            errors.append(float((a - b).abs().max()))
        model_checks[base_mode] = {
            'ordinary_parameters': sum(p.numel() for p in ordinary.model.parameters()),
            'was_parameters': sum(p.numel() for p in duplicated.model.parameters()),
            'input_projection_keys': input_keys,
            'same_seed_changed_same_shape_state_tensors': len(changed),
            'same_seed_changed_examples': changed[:6],
            'ordinary_config': ordinary.config, 'was_config': duplicated.config,
            'probe_sources': [audits[i]['source'] for i in probes],
            'folded_projection_prediction_max_abs_error': max(errors),
            'folded_projection_prediction_errors': errors,
            'paired_geometry_identical': all(geometry_matches),
            'interpretation': 'Untrained forward equivalence only; not a test MAE or trained-model comparison.',
        }
        print(f'{base_mode}: duplicate-feature folding max error {max(errors):.3g}', flush=True)

    # The CLI uses one seed for the split and model initialization. Quantify
    # trajectory-family overlap using identifiers alone; no model is fit.
    names = [name for name, _ in rows]
    labels = [target for _, target in rows]
    groups = [re.sub(r'-POSCAR\d+', '', name) for name in names]
    split_checks = []
    for split in iter_train_val_test_splits(list(range(len(rows))), labels, [123, 11, 1245]):
        train_ids, test_ids = split['train_X'], split['test_X']
        train_groups = {groups[i] for i in train_ids}
        overlap = sum(groups[i] in train_groups for i in test_ids)
        split_checks.append({'seed': split['seed'], 'train': len(train_ids), 'val': len(split['val_X']),
                             'test': len(test_ids), 'test_rows_with_POSCAR_family_in_training': overlap,
                             'fraction': overlap / len(test_ids)})

    report = {
        'scope': 'Legacy native preprocessing and local inputs; remote performance checkpoints are unavailable.',
        'training_performed': False,
        'data_list': str(args.data_list.resolve()), 'dataset_rows': len(rows),
        'materials': len({name.split('-')[1] for name in names}),
        'defect_family_counts': dict(Counter(defect_family(name) for name in names)),
        'unique_POSCAR_families': len(set(groups)),
        'sample_summary': {
            'structures': len(audits), 'strata': len(strata), 'sites': sum(a['graph_sites'] for a in audits),
            'raw_was_sites': sum(a['raw_was_sites'] for a in audits),
            'was_sites': sum(a['was_sites'] for a in audits),
            'all_duplicate_reference_features': all(a['was_halves_identical'] for a in audits),
            'all_one_actual_defect': all(a['defects'] == 1 for a in audits),
            'all_attention_hetero_inputs_equal': all(a['attention_hetero_features_identical'] and a['attention_hetero_sites_identical'] for a in audits),
        },
        'samples': audits, 'model_checks': model_checks,
        'split_family_overlap': split_checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('dataset_rows', 'materials', 'defect_family_counts',
                                                  'sample_summary', 'split_family_overlap')}, indent=2))
    print(f'Saved: {args.output}')


if __name__ == '__main__':
    main()
