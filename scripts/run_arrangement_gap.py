"""Train all non-hypergraph ALIGNN modes against held-out, same-composition DFT energy differences."""
import argparse
import csv
from importlib import import_module
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import torch
from pymatgen.core import Structure
from pymatgen.core.periodic_table import DummySpecies

from ..config.defaults import get_config
from ..arrangement_sources import SOURCES
from ..data.datasets import dataset_index_for_mode, init_elem_embedding, representation_for_mode
from ..data.structure_utils import preserve_pool_type
from ..main import train_single_mode
from ..training.results import ensure_compact_run_config
from .alignn_benchmark_support import audit_checkpoint, fixed_split, persist_contract
from .alignn_convergence import convergence_audit
from .run_sige_complexes import MODES, save_csv


ORDER_THRESHOLD_EV = 0.010
DISCOVERY_SEED = 123
REPLICATION_SEEDS = [456, 789]


def dataset_module(name):
    return import_module('..data.'+SOURCES[name].module,__package__)


def structures(mode, records, geometries, archive):
    rep = representation_for_mode(mode)
    result = []
    for r in records:
        sid = r['source_id']
        a = geometries[sid]
        st = Structure(a.cell.array, a.numbers, a.get_scaled_positions(),
                       site_properties={'was': a.arrays['was_Z'].tolist(),
                                        'type': a.arrays['defect_site'].astype(bool).tolist()})
        if rep != 'full':
            positions = a.info.get('vacancy_fractionals', [])
            if 'vacancy_fractional' in a.info:
                positions = [a.info['vacancy_fractional']]
            for position in positions:
                st.append(DummySpecies(), position, properties={'type': True, 'was': int(a.info['vacancy_was_Z'])})
        if rep == 'hetero':
            st = preserve_pool_type(st)
        st.source_id = st.source_name = sid
        st.source_path = f'{archive}!{sid}'
        st.material = st.defect_family = r['group']
        result.append(st)
    return result


def pair_indices(ids, lookup):
    pairs = [(i, j) for i, j in combinations(range(len(ids)), 2)
             if lookup[ids[i]]['group'] == lookup[ids[j]]['group']]
    if not pairs:
        raise ValueError('No same-composition pairs in this partition')
    return np.array(pairs, dtype=int).T


def validation_metric(ids, lookup):
    left, right = pair_indices(ids, lookup)
    truth = np.array([lookup[s]['energy_eV_cell'] for s in ids])
    dft = truth[right] - truth[left]

    def evaluate(trainer, validation_structures, validation_targets):
        if [s.source_id for s in validation_structures] != ids:
            raise ValueError('Validation order changed')
        labels = np.asarray(validation_targets, dtype=float).reshape(-1)
        if not np.allclose(labels[right] - labels[left], dft, atol=1e-6, rtol=0):
            raise ValueError('Validation energy differences do not match source labels')
        _, raw = trainer.evaluate_on_test(return_predictions=True)
        pred = np.asarray(raw, dtype=float).reshape(-1)
        if len(pred) != len(ids) or not np.isfinite(pred).all():
            raise ValueError('Invalid validation prediction')
        return float(np.mean(np.abs((pred[right] - pred[left]) - dft)))
    return evaluate


def pair_rows(rows):
    out = []
    for a, b in combinations(rows, 2):
        if a['group'] == b['group'] and a['fold'] == b['fold']:
            truth = b['dft_eV_cell'] - a['dft_eV_cell']
            pred = b['predicted_eV_cell'] - a['predicted_eV_cell']
            out.append(dict(fold=a['fold'], group=a['group'], id_a=a['source_id'], id_b=b['source_id'],
                            dft_gap_eV=truth, predicted_gap_eV=pred, gap_error_eV=abs(pred-truth)))
    return out


def fit(mode, fold, ids, records, lookup, dataset, spec, args):
    groups = sorted({r['group'] for r in records})
    offsets = {g: float(np.mean([lookup[s]['energy_eV_cell'] for s in ids['train'] if lookup[s]['group'] == g]))
               for g in groups}
    if not np.isfinite(list(offsets.values())).all():
        raise ValueError('Every composition needs a training-only energy reference')
    targets = torch.tensor([r['energy_eV_cell'] - offsets[r['group']] for r in records], dtype=torch.float32)
    scorer = validation_metric(ids['val'], lookup)
    base = args.run_dir/f'fold{fold}'/f'seed{args.seed}'/mode
    budget = args.initial_budget
    while True:
        folder = base if budget == args.initial_budget else base/f'extended_{budget}'
        config = get_config('alignn', 'och', mode)
        config['model'].update(train_batch_size=args.batch_size, test_batch_size=args.batch_size, local_radius=0.)
        config['optim'].update(amp=False, early_stopping_patience=args.patience)
        config.update(target_name='DFT_relaxed_energy_minus_train_group_mean', target_units='eV/cell',
                      train_group_means_eV=offsets, validation_metric='same_group_signed_energy_gap_MAE_eV',
                      benchmark_schema=spec.SCHEMA, source_sha256=spec.DIGEST, arrangement_fold=fold, max_epochs=budget)
        ensure_compact_run_config(folder, dict(layout='compact_v1', model_name='alignn', dataset_name=spec.SCHEMA,
                                              mode=mode, run_label=mode, config=config))
        if not args.resume and any(folder.glob('*_history.csv')):
            raise FileExistsError(folder)
        print(f'{spec.SCHEMA} fold={fold} seed={args.seed} mode={mode} patience={args.patience}', flush=True)
        train_single_mode(mode, config, dataset, targets, [args.seed], budget, args.device, 'alignn', spec.SCHEMA,
                          log_dir=str(folder), resume=args.resume, validation_score=scorer,
                          split_iterator=fixed_split(ids, spec.SCHEMA, f'held-out arrangements {fold}'))
        plateau = convergence_audit(folder, args.seed)
        if plateau['stopped_on_validation_plateau']:
            break
        budget *= 2
    audit = audit_checkpoint(folder, config, args.seed, ids, 'minimum_validation_gap_MAE_eV')
    with (folder/f'seed{args.seed}_test_predictions.csv').open(newline='') as handle:
        predictions = list(csv.DictReader(handle))
    if [r['source_id'] for r in predictions] != ids['test']:
        raise ValueError('Wrong held-out configurations')
    rows = []
    for p in predictions:
        r = lookup[p['source_id']]
        offset = offsets[r['group']]
        truth = r['energy_eV_cell']
        pred = float(p['prediction']) + offset
        if abs(float(p['target']) + offset - truth) > 1e-6 or not np.isfinite(pred):
            raise ValueError('Test label or prediction mismatch')
        rows.append(dict(fold=fold, group=r['group'], source_id=r['source_id'], dft_eV_cell=truth,
                         predicted_eV_cell=pred, train_mean_eV_cell=offset))
    pairs = pair_rows(rows)
    save_csv(base/'test_structures.csv', rows)
    save_csv(base/'test_pairs.csv', pairs)
    report = dict(mode=mode, fold=fold, seed=args.seed, device=args.device, n_test=len(rows), n_pairs=len(pairs),
                  test_gap_MAE_eV=float(np.mean([p['gap_error_eV'] for p in pairs])),
                  selected_run=folder.relative_to(base).as_posix(), **(audit | plateau))
    (base/'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf8')
    print(report, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=SOURCES, default='ysz')
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--results', type=Path)
    parser.add_argument('--mode', nargs='+', choices=MODES, default=list(MODES))
    parser.add_argument('--fold', nargs='+', type=int, choices=range(3), default=[0,1,2])
    parser.add_argument('--seed', type=int, default=DISCOVERY_SEED)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--initial-budget', type=int, default=1000)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if min(args.batch_size, args.patience) < 1 or args.initial_budget < args.patience:
        parser.error('Invalid convergence or batch settings')
    spec = dataset_module(args.dataset)
    args.results = args.results or spec.RESULTS
    args.archive = args.archive or (spec.ARCHIVE if spec.ARCHIVE.exists() else args.results/spec.BUNDLE)
    args.run_dir = args.run_dir or spec.ROOT/'logs'/spec.SCHEMA
    torch.set_num_threads(1)
    records, geometries, raw, payload = spec.prepare(args.archive)
    folds = spec.partitions(records)
    lookup = {r['source_id']: r for r in records}
    args.run_dir.mkdir(parents=True, exist_ok=True)
    contract = dict(schema=spec.SCHEMA, publication=spec.PAPER, source=spec.SOURCE, source_sha256=spec.DIGEST,
                    model_names=list(MODES), partitions=folds, split_seed=spec.SPLIT_SEED,
                    primary='MAE of (pred_j-pred_i) minus (DFT_j-DFT_i). Only same group and same held-out fold; all eligible pairs.',
                    primary_subject='ordinary hetero, ranked against every non-hypergraph ALIGNN mode',
                    optimization='Fit centered single-structure DFT labels. LR schedule, checkpoint selection and convergence use validation gap MAE.',
                    energy_reference='Mean of training labels within each composition/parent group, in float64 before casting.',
                    convergence=dict(patience=args.patience, minimum_relative_improvement_percent=.5,
                                     guard=args.initial_budget, guard_reached='repeat from same random initialization with twice the budget'),
                    batch_size=args.batch_size, initial_screen_seed=DISCOVERY_SEED, tie_tolerance_eV=1e-8,
                    confirmation=dict(seeds=REPLICATION_SEEDS, modes=list(MODES),
                                      rule='If ordinary hetero ranks first (ties included) at seed 123, run the full slate at both other seeds and report the mean of seed-specific MAEs.'),
                    source_ordering=getattr(spec, 'INPUT_DESCRIPTION', 'Use atomic species, site properties and ideal positions only. All mode-specific markers use the same defect reference.'),
                    secondary_order_threshold_eV=ORDER_THRESHOLD_EV)
    if hasattr(spec, 'MATCHED_PATHS'):
        contract['matched_path_diagnostic'] = spec.MATCHED_PATHS
    persist_contract(args.run_dir/'data_contract.json', contract)
    if hasattr(spec,'inventory_diagnostic'):
        persist_contract(args.run_dir/'inventory_diagnostic.json',spec.inventory_diagnostic(records,geometries,folds))
    bundle = args.results/spec.BUNDLE
    if args.prepare_only or not bundle.exists():
        spec.write_bundle(bundle, records, geometries, raw, payload, folds)
    if args.prepare_only:
        return
    del raw, payload
    init_elem_embedding(spec.ROOT/'atom_init.json')
    for mode in dict.fromkeys(args.mode):
        dataset = [None]*4
        dataset[dataset_index_for_mode(mode)] = structures(mode, records, geometries, args.archive)
        for fold in dict.fromkeys(args.fold):
            fit(mode, fold, folds[fold], records, lookup, dataset, spec, args)


if __name__ == '__main__':
    main()
