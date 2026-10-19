"""Prepare and train HERA on explicitly defined unrelaxed YBCO formation energies."""
import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ..config.defaults import get_config
from ..data.datasets import init_elem_embedding, dataset_index_for_mode
from ..data.ybco import (DEFAULT_ROOT, SCHEMA, download_source, prepare,
                         read_prepared, grouped_indices, load_structures)
from ..main import train_single_mode


def select_records(records, smoke=False):
    if not smoke:
        return records
    # One minimally perturbed member from every family: a small I/O/graph/train
    # check, never a benchmark score. Families still never cross split boundaries.
    from ..data.ybco import LABEL_RE
    selected = {}
    for record in records:
        m = LABEL_RE.fullmatch(record['label'])
        priority = (int(m['jiggle']), int(m['strain']), abs(float(m['volume'])))
        family = record['family']
        if family not in selected or priority < selected[family][0]:
            selected[family] = priority, record
    return sorted([r for _, r in selected.values()], key=lambda r: r['source_frame'])


def make_splitter(records):
    by_id = {r['source_id']: r for r in records}

    def iterator(data, targets, seeds, cv5=False):
        if cv5:
            raise ValueError('YBCO uses grouped 60/20/20 splits, not row-wise CV')
        ordered = [by_id[s.source_id] for s in data]
        for seed in seeds:
            indices = grouped_indices(ordered, seed)
            split = {'display': f'seed={seed} grouped_by_defect_family', 'logger_id': seed,
                     'explain_id': f'seed_{seed}', 'seed': int(seed),
                     'ood_split': {'schema': SCHEMA, 'grouping': 'defect_family',
                                   'group_fractions': [0.6, 0.2, 0.2]}}
            for part, ids in indices.items():
                split[f'{part}_X'] = [data[i] for i in ids]
                split[f'{part}_y'] = [targets[i] for i in ids]
            print('Split structures: ' + ', '.join(f'{p}={len(ii)}' for p, ii in indices.items()), flush=True)
            yield split
    return iterator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--source', type=Path, help='Existing pinned author XYZ for preparation')
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--model', choices=('alignn', 'cgcnn', 'megnet'), default='alignn')
    parser.add_argument('--mode', nargs='+', choices=('full', 'full_x', 'was_x', 'hetero', 'hetero_was'),
                        default=['hetero_was', 'was_x'])
    parser.add_argument('--seed', type=int, nargs='+', default=[123])
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--r', type=float, default=0, help='Local region radius in Angstrom')
    parser.add_argument('--smoke', action='store_true', help='One sample per family; two epochs by default')
    parser.add_argument('--resume', action='store_true', help='Skip compatible completed seeds')
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--atom-init', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'atom_init.json')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or not np.isfinite(args.r) or args.r < 0:
        parser.error('epochs/batch-size must be positive and r must be finite and nonnegative')
    if args.download:
        if args.source is not None:
            parser.error('Use either --download or --source')
        args.source = download_source(args.data_root)
    if args.prepare_only or not (args.data_root / f'{SCHEMA}.json').exists():
        prepare(args.data_root, args.source)
    manifest = read_prepared(args.data_root)
    records = select_records(manifest['records'], args.smoke)
    print(f'Target: {manifest["target"]}; {manifest["units"]}', flush=True)
    print('This task includes strain/displacement energies; it is NOT relaxed DFE or pair binding.', flush=True)
    for seed in args.seed:
        indices = grouped_indices(records, seed)
        print(f'Seed {seed}: ' + ', '.join(f'{part}={len(ids)}' for part, ids in indices.items()), flush=True)
    if args.prepare_only:
        return
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        parser.error('CUDA is unavailable; select --device cpu')
    if args.smoke and args.epochs == 500:
        args.epochs = 2
    default_name = 'ybco_formation_smoke_v1' if args.smoke else 'ybco_formation_v1'
    run_dir = args.run_dir or Path(__file__).resolve().parents[1] / 'logs' / default_name
    run_dir.mkdir(parents=True, exist_ok=True)
    # A reproducible cohort and references, independent of download locations.
    identity = {k: manifest[k] for k in ('schema', 'source_sha256', 'target', 'units',
                                        'formula', 'references', 'chemical_potentials_ev',
                                        'mapping_policy', 'graph_mapping_sha256')}
    identity['records'] = records
    identity['smoke'] = args.smoke
    contract = json.dumps(identity, sort_keys=True)
    contract_path = run_dir / 'data_contract.json'
    if contract_path.exists() and json.loads(contract_path.read_text()) != identity:
        raise ValueError('Run directory has a different cohort/reference definition; choose a new --run-dir')
    contract_path.write_text(json.dumps(identity, indent=2) + '\n', encoding='utf-8')
    for seed in args.seed:
        splits = grouped_indices(records, seed)
        membership = {part: [records[i]['source_id'] for i in ids] for part, ids in splits.items()}
        (run_dir / f'seed{seed}_splits.json').write_text(json.dumps(membership, indent=2) + '\n', encoding='utf-8')
    init_elem_embedding(args.atom_init)
    results = []
    for mode in dict.fromkeys(args.mode):
        config = get_config(args.model, 'native', mode)
        # Reuse the existing model architecture, not native CIF preprocessing.
        config.pop('native_preprocessing', None)
        config['target_name'] = manifest['target']
        config['target_units'] = manifest['units']
        config['ybco_data_sha256'] = hashlib.sha256(contract.encode()).hexdigest()
        config['ybco_reference_schema'] = SCHEMA
        config['model']['train_batch_size'] = args.batch_size
        config['model']['test_batch_size'] = 1
        config['model']['local_radius'] = args.r
        path = run_dir / args.model / mode / f'r{args.r:g}'
        path.mkdir(parents=True, exist_ok=True)
        config_path = path / 'config.json'
        if config_path.exists() and json.loads(config_path.read_text()) != config:
            raise ValueError(f'{path}: model configuration changed; choose a new --run-dir')
        if not args.resume and list(path.glob('seed*_history.csv')):
            raise ValueError(f'{path}: results already exist; use --resume or a new --run-dir')
        config_path.write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
        structures = load_structures(args.data_root, records, mode, args.r)
        dataset = [None] * 4
        dataset[dataset_index_for_mode(mode)] = structures
        targets = torch.tensor([r['target_ev'] for r in records], dtype=torch.float32)
        losses = train_single_mode(mode, config, dataset, targets, args.seed, args.epochs,
                                   args.device, args.model, 'ybco_formation', log_dir=str(path),
                                   resume=args.resume, protect_existing=True,
                                   split_iterator=make_splitter(records))
        for seed, loss in zip(args.seed, losses):
            results.append({'model': args.model, 'mode': mode, 'seed': seed,
                            'test_mae_ev': float(loss), 'smoke': args.smoke})
        del dataset, structures
        gc.collect()
        if args.device.startswith('cuda'):
            torch.cuda.empty_cache()
    summary_path = run_dir / 'summary.json'
    if summary_path.exists():
        previous = json.loads(summary_path.read_text())
        current_keys = {(r['model'], r['mode'], r['seed']) for r in results}
        results = [r for r in previous if (r['model'], r['mode'], r['seed']) not in current_keys] + results
    summary_path.write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
    print(f'Results: {summary_path}', flush=True)


if __name__ == '__main__':
    main()
