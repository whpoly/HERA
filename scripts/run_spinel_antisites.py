"""From-scratch HERA comparison on symmetry-distinct MgAl2O4 cation arrangements."""
import argparse
import csv
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import torch
from pymatgen.core import Structure

from ..config.defaults import get_config
from ..data.datasets import dataset_index_for_mode, init_elem_embedding, representation_for_mode
from ..data.spinel_antisites import (ARCHIVE, DIGEST, DOI, RESULTS, ROOT, SCHEMA, SOURCE,
                                    SPLIT_SEED, STRATA, collect, partitions, write_bundle)
from ..data.structure_utils import preserve_pool_type
from ..main import train_single_mode
from ..training.results import ensure_compact_run_config
from .alignn_benchmark_support import audit_checkpoint, fixed_split, persist_contract
from .run_sige_complexes import MODES, save_csv


DEFAULT_RUN = ROOT / 'logs' / SCHEMA
ORDER_THRESHOLD_EV = 0.010


def structures(mode, cohort, atoms, archive, member_prefix='dataset_MgAl2O4/'):
    result = []
    for row in cohort:
        sid = row['source_id']
        a = atoms[sid]
        st = Structure(a.cell.array, a.numbers, a.get_scaled_positions(),
                       site_properties={'was': a.arrays['was_Z'].tolist(),
                                        'type': a.arrays['is_antisite'].astype(bool).tolist()})
        if representation_for_mode(mode) == 'hetero':
            st = preserve_pool_type(st)
        st.source_id = st.source_name = sid
        st.source_path = str(archive) + '!/' + member_prefix + sid
        st.material = 'MgAl2O4'
        st.defect_family = f"antisites_k{row['k_each']}"
        result.append(st)
    return result


def predictions_for_fold(path, ids, lookup, offsets, fold):
    with path.open(newline='', encoding='utf8') as handle:
        predictions = list(csv.DictReader(handle))
    if [r['source_id'] for r in predictions] != ids['test']:
        raise ValueError('Prediction IDs differ from the held-out occupations')
    rows = []
    for row in predictions:
        sid = row['source_id']
        sample = lookup[sid]
        k, dft = sample['k_each'], sample['energy_eV_cell']
        if abs(float(row['target']) + offsets[k] - dft) > 1e-6:
            raise ValueError(f'Training target and raw DFT energy differ for {sid}')
        energy = float(row['prediction']) + offsets[k]
        if not np.isfinite(energy):
            raise ValueError(f'Nonfinite prediction for {sid}')
        rows.append(dict(fold=fold, k_each=k, source_id=sid, dft_eV_cell=dft,
                         predicted_eV_cell=energy, train_mean_eV_cell=offsets[k]))
    return rows


def equal_count_pairs(rows):
    pairs = []
    for a, b in combinations(rows, 2):
        if a['fold'] == b['fold'] and a['k_each'] == b['k_each']:
            truth = round(b['dft_eV_cell'] - a['dft_eV_cell'], 8)
            pred = b['predicted_eV_cell'] - a['predicted_eV_cell']
            pairs.append(dict(fold=a['fold'], k_each=a['k_each'], id_a=a['source_id'], id_b=b['source_id'],
                              dft_gap_eV_cell=truth, predicted_gap_eV_cell=pred,
                              abs_gap_error_eV_cell=abs(pred - truth)))
    return pairs


def train_fold(mode, fold, ids, lookup, offsets, targets, dataset, args, *, schema=SCHEMA, digest=DIGEST,
               batch_size=8, accumulation=1):
    folder = args.run_dir / f'fold{fold}' / f'seed{args.seed}' / mode
    # This registry key supplies the common ALIGNN architecture. Only the structures
    # assembled from the pinned spinel archive below are passed to train_single_mode.
    config = get_config('alignn', 'och', mode)
    config['model'].update(train_batch_size=batch_size, test_batch_size=batch_size, local_radius=0.)
    config['optim'].update(amp=False, early_stopping_patience=args.patience)
    if accumulation != 1:
        config['optim']['grad_accum_steps'] = accumulation
    config.update(target_name='energy_relative_to_training_mean_at_fixed_inversion', target_units='eV/cell',
                  training_offsets_eV_cell={str(k): v for k, v in offsets.items()}, benchmark_schema=schema,
                  cohort_sha256=digest, arrangement_fold=fold, max_epochs=args.epochs)
    ensure_compact_run_config(folder, dict(layout='compact_v1', model_name='alignn', dataset_name=schema,
                                           mode=mode, run_label=mode, config=config))
    if not args.resume and any(folder.glob('*_history.csv')):
        raise FileExistsError(folder)
    print(f'MgAl2O4 fold={fold} mode={mode}', flush=True)
    train_single_mode(mode, config, dataset, targets, [args.seed], args.epochs, args.device, 'alignn', schema,
                      log_dir=str(folder), resume=args.resume,
                      split_iterator=fixed_split(ids, schema, f'arrangement holdout {fold}'))
    test = predictions_for_fold(folder / f'seed{args.seed}_test_predictions.csv', ids, lookup, offsets, fold)
    pairs = equal_count_pairs(test)
    audit = audit_checkpoint(folder, config, args.seed, ids, 'best_val_mae_eV_cell')
    save_csv(folder / 'test_structures.csv', test)
    save_csv(folder / 'test_pairs.csv', pairs)
    summary = dict(mode=mode, fold=fold, seed=args.seed, device=args.device, structures=len(test), pairs=len(pairs),
                   pair_mae=float(np.mean([r['abs_gap_error_eV_cell'] for r in pairs])), **audit)
    (folder / 'metrics.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='utf8')
    print(summary, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=ARCHIVE)
    parser.add_argument('--run-dir', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--results', type=Path, default=RESULTS)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--mode', choices=MODES, nargs='+', default=list(MODES))
    parser.add_argument('--fold', choices=range(3), type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    raw, atoms, records = collect(args.archive)
    cohort = [r for r in records if r['is_representative'] and r['k_each'] in STRATA]
    lookup = {r['source_id']: r for r in cohort}
    splits = partitions(cohort)
    contract = dict(schema=SCHEMA, json_sha256=DIGEST, source=SOURCE, publication=DOI, split_seed=SPLIT_SEED,
                    partitions=splits, modes=list(MODES), k_each=list(STRATA), primary='MAE of signed DFT energy gaps in eV per 56-atom cell. Pairs must share test fold and k_each.',
                    target='DFT total energy minus training mean at the same k_each (invertible per fold)',
                    secondary='Absolute energy MAE and fraction of correctly ordered pairs above the threshold',
                    order_threshold_eV_cell=ORDER_THRESHOLD_EV, selection='Minimum validation structure MAE',
                    primary_comparators=['hetero', 'full', 'full_x'], inference='Relaxed crystal structure',
                    uncertainty='Paired bootstrap of held-out structures within each (fold,k_each), fixed trained models')
    args.run_dir.mkdir(parents=True, exist_ok=True)
    persist_contract(args.run_dir / 'data_contract.json', contract)
    if args.prepare_only:
        write_bundle(args.results / 'MgAl2O4_antisite_DFT.zip', raw, atoms, records, splits)
        return
    init_elem_embedding(ROOT / 'atom_init.json')
    for mode in args.mode:
        dataset = [None] * 4
        dataset[dataset_index_for_mode(mode)] = structures(mode, cohort, atoms, args.archive)
        for fold in args.fold:
            ids = splits[fold]
            offsets = {k: float(np.mean([lookup[s]['energy_eV_cell'] for s in ids['train']
                                         if lookup[s]['k_each'] == k])) for k in STRATA}
            targets = torch.tensor([r['energy_eV_cell'] - offsets[r['k_each']] for r in cohort], dtype=torch.float32)
            train_fold(mode, fold, ids, lookup, offsets, targets, dataset, args)


if __name__ == '__main__':
    main()
