#!/usr/bin/env python
"""Evaluate trained 2dmd_low checkpoints on material-specific high tests.

This command performs inference only. It restores the model configuration,
weights, and target scaler stored in each trusted HERA checkpoint, loads only
the 500 high-concentration structures for the requested material, and writes
MoS2 and WSe2 results to separate log trees.
"""

import argparse
import copy
import csv
import re
from pathlib import Path

import torch

from .config.defaults import VALID_MODES
from .data.datasets import (
    _load_data_2dmd_material_transfer,
    dataset_index_for_mode,
    init_elem_embedding,
    representation_for_mode,
)
from .main import clear_cuda_cache, write_test_predictions
from .training.trainer import MEGNetTrainer, load_trusted_checkpoint


MATERIAL_DATASETS = {
    'mos2': ('MoS2', '2dmd_mos2'),
    'wse2': ('WSe2', '2dmd_wse2'),
}
NON_WAS_ALIGNN_MODES = tuple(
    mode for mode in (
        'full',
        'full_x',
        'hetero',
        'hetero_fixed_pool',
        'definet',
        'attention',
        'hypergraph',
    )
    if mode in VALID_MODES
)
CHECKPOINT_PATTERN = 'seed*_best_checkpoint.pth'


def checkpoint_seed(checkpoint, path):
    split = checkpoint.get('split') or {}
    if split.get('seed') is not None:
        return int(split['seed'])
    match = re.search(r'seed(\d+)', Path(path).name)
    if match:
        return int(match.group(1))
    raise ValueError(f'Cannot determine checkpoint seed: {path}')


def checkpoint_identity(checkpoint, path):
    required = ('model', 'scaler', 'config', 'model_name', 'dataset_name', 'mode')
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise ValueError(
            f'Checkpoint lacks required HERA fields {missing}: {path}'
        )
    mode = str(checkpoint['mode'])
    return {
        'path': Path(path),
        'checkpoint': checkpoint,
        'model_name': str(checkpoint['model_name']),
        'dataset_name': str(checkpoint['dataset_name']),
        'mode': mode,
        'run_label': str(checkpoint.get('run_label') or mode),
        'seed': checkpoint_seed(checkpoint, path),
    }


def discover_low_checkpoints(
        checkpoint_root,
        alignn_modes=None,
        seeds=None,
):
    """Discover MEGNet Sparse and selected non-WAS ALIGNN checkpoints."""
    checkpoint_root = Path(checkpoint_root)
    if not checkpoint_root.is_dir():
        raise FileNotFoundError(
            f'Checkpoint root does not exist: {checkpoint_root}'
        )
    alignn_modes = (
        set(NON_WAS_ALIGNN_MODES)
        if alignn_modes is None
        else set(alignn_modes)
    )
    seeds = None if seeds is None else {int(seed) for seed in seeds}
    selected = []

    for path in sorted(checkpoint_root.rglob(CHECKPOINT_PATTERN)):
        checkpoint = load_trusted_checkpoint(path, map_location='cpu')
        try:
            record = checkpoint_identity(checkpoint, path)
        except ValueError:
            continue
        if record['dataset_name'] != '2dmd_low':
            continue
        if seeds is not None and record['seed'] not in seeds:
            continue
        is_sparse = (
            record['model_name'] == 'megnet'
            and record['mode'] == 'sparse'
        )
        is_alignn = (
            record['model_name'] == 'alignn'
            and record['mode'] in alignn_modes
            and 'was' not in record['mode']
            and 'was' not in record['run_label']
        )
        if is_sparse or is_alignn:
            selected.append(record)

    if not selected:
        raise FileNotFoundError(
            'No matching 2dmd_low MEGNet Sparse or non-WAS ALIGNN best '
            f'checkpoints found under {checkpoint_root}'
        )

    identities = {}
    for record in selected:
        identity = (
            record['model_name'],
            record['mode'],
            record['run_label'],
            record['seed'],
        )
        if identity in identities:
            raise ValueError(
                f'Duplicate checkpoint identity {identity}; use a narrower '
                f'--checkpoint-root. Found {identities[identity]} and '
                f'{record["path"]}'
            )
        identities[identity] = record['path']

    selected.sort(key=lambda record: (
        0 if record['model_name'] == 'megnet' else 1,
        record['mode'],
        record['run_label'],
        record['seed'],
    ))
    return selected


def load_material_high_test(material_key, model_name, mode, config):
    """Load only one material's fixed high-concentration test structures."""
    material, _ = MATERIAL_DATASETS[material_key]
    representation = representation_for_mode(mode)
    local_cutoff = None
    if mode in ('hetero', 'hetero_fixed_pool'):
        local_cutoff = config['model'].get('local_radius')
    dataset = _load_data_2dmd_material_transfer(
        material,
        model_name,
        local_cutoff=local_cutoff,
        representations=[representation],
        pure_vacancy=False,
        include_low=False,
    )
    structures = dataset[dataset_index_for_mode(mode)]
    targets = dataset[-1]
    if structures is None or not structures:
        raise ValueError(
            f'No {material} high structures available for {model_name}/{mode}'
        )
    invalid = [
        getattr(structure, 'source_id', index)
        for index, structure in enumerate(structures)
        if getattr(structure, 'concentration', None) != 'high'
        or getattr(structure, 'material', None) != material
    ]
    if invalid:
        raise ValueError(
            f'{material} inference data contain non-matching samples: '
            f'{invalid[:5]}'
        )
    return structures, targets


def safe_variant_name(record):
    raw = record['run_label'] or record['mode']
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(raw)).strip('._')


def apply_checkpoint_model_compatibility(config, record):
    """Recreate legacy normalization when its saved buffers identify it."""
    state_dict = record['checkpoint']['model']
    legacy_modes = {'attention', 'definet'}
    has_batchnorm_buffers = any(
        key.endswith('.running_mean') for key in state_dict
    )
    if (
            record['model_name'] == 'alignn'
            and record['mode'] in legacy_modes
            and has_batchnorm_buffers
    ):
        model_config = config.setdefault('model', {})
        model_config['alignn_feature_normalization'] = 'batchnorm'
        model_config['alignn_legacy_residual_norm'] = True
        return 'legacy_batchnorm'
    return 'checkpoint_native'


def write_material_summary(path, rows):
    fields = (
        'training_dataset',
        'test_dataset',
        'material',
        'model',
        'mode',
        'run_label',
        'seed',
        'n_test',
        'test_mae',
        'model_compatibility',
        'checkpoint',
        'prediction_csv',
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def predict_checkpoint(record, material_key, output_root, device,
                       test_batch_size=None, amp=False, dataset_cache=None):
    """Run one checkpoint on one material and write its prediction CSV."""
    checkpoint = record['checkpoint']
    config = copy.deepcopy(checkpoint['config'])
    if test_batch_size is not None:
        config['model']['test_batch_size'] = int(test_batch_size)
    if amp:
        config.setdefault('optim', {})['amp'] = True
    compatibility = apply_checkpoint_model_compatibility(config, record)

    trainer = MEGNetTrainer(config, device, seed=record['seed'])
    trainer.scaler.load_state_dict(checkpoint['scaler'])
    # Validate architecture compatibility before converting hundreds of CIFs.
    trainer.model.load_state_dict(checkpoint['model'])

    cache_key = (
        material_key,
        record['model_name'],
        representation_for_mode(record['mode']),
        config['model'].get('local_radius'),
    )
    if dataset_cache is not None and cache_key in dataset_cache:
        structures, targets = dataset_cache[cache_key]
    else:
        structures, targets = load_material_high_test(
            material_key,
            record['model_name'],
            record['mode'],
            config,
        )
        if dataset_cache is not None:
            dataset_cache[cache_key] = (structures, targets)

    mae, predictions = trainer.predict_structures(
        structures,
        targets,
        checkpoint['model'],
        return_predictions=True,
    )

    material, test_dataset = MATERIAL_DATASETS[material_key]
    result_dir = (
        Path(output_root)
        / test_dataset
        / record['model_name']
        / safe_variant_name(record)
    )
    prediction_path = write_test_predictions(
        result_dir,
        record['seed'],
        structures,
        targets,
        predictions,
    )
    row = {
        'training_dataset': record['dataset_name'],
        'test_dataset': test_dataset,
        'material': material,
        'model': record['model_name'],
        'mode': record['mode'],
        'run_label': record['run_label'],
        'seed': record['seed'],
        'n_test': len(structures),
        'test_mae': f'{float(mae):.10g}',
        'model_compatibility': compatibility,
        'checkpoint': str(record['path'].resolve()),
        'prediction_csv': str(Path(prediction_path).resolve()),
    }
    del trainer
    clear_cuda_cache(device)
    return row


def parse_seed_values(values, parser):
    if values == ['all']:
        return None
    if 'all' in values:
        parser.error('--seed all cannot be combined with explicit seeds')
    try:
        seeds = list(dict.fromkeys(int(value) for value in values))
    except ValueError:
        parser.error('--seed must contain integers or the single value all')
    if any(seed < 0 for seed in seeds):
        parser.error('--seed values must be non-negative')
    return seeds


def parse_args(argv=None):
    package_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            'Use trained 2dmd_low checkpoints to predict the separate MoS2 '
            'and WSe2 high-concentration test sets without retraining.'
        )
    )
    parser.add_argument(
        '--checkpoint-root',
        type=Path,
        required=True,
        help='Root of the completed 2dmd_low training logs/checkpoints.',
    )
    parser.add_argument(
        '--output-root',
        type=Path,
        default=None,
        help='Default: <checkpoint-root>/high_test_predictions.',
    )
    parser.add_argument(
        '--material',
        nargs='+',
        choices=tuple(MATERIAL_DATASETS),
        default=list(MATERIAL_DATASETS),
    )
    parser.add_argument(
        '--alignn-mode',
        nargs='+',
        choices=(*NON_WAS_ALIGNN_MODES, 'available'),
        default=['available'],
        help=(
            'Non-WAS ALIGNN modes to evaluate. "available" selects every '
            'non-WAS ALIGNN checkpoint found (default).'
        ),
    )
    parser.add_argument('--seed', nargs='+', default=['all'])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument(
        '--atom-init',
        type=Path,
        default=package_root / 'atom_init.json',
    )
    parser.add_argument('--test-batch-size', type=int, default=None)
    parser.add_argument('--amp', action='store_true')
    args = parser.parse_args(argv)

    if 'available' in args.alignn_mode:
        if len(args.alignn_mode) != 1:
            parser.error(
                '--alignn-mode available cannot be combined with named modes'
            )
        args.alignn_modes = None
    else:
        args.alignn_modes = list(dict.fromkeys(args.alignn_mode))
    args.seeds = parse_seed_values(args.seed, parser)
    args.material = list(dict.fromkeys(args.material))
    if args.test_batch_size is not None and args.test_batch_size < 1:
        parser.error('--test-batch-size must be >= 1')
    args.checkpoint_root = args.checkpoint_root.resolve()
    args.output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else args.checkpoint_root / 'high_test_predictions'
    )
    args.atom_init = args.atom_init.resolve()
    return args


def main():
    args = parse_args()
    init_elem_embedding(args.atom_init)
    records = discover_low_checkpoints(
        args.checkpoint_root,
        alignn_modes=args.alignn_modes,
        seeds=args.seeds,
    )
    sparse_count = sum(
        record['model_name'] == 'megnet' and record['mode'] == 'sparse'
        for record in records
    )
    alignn_count = sum(record['model_name'] == 'alignn' for record in records)
    if sparse_count == 0:
        raise FileNotFoundError(
            'No 2dmd_low MEGNet Sparse checkpoint matched this selection.'
        )
    if alignn_count == 0:
        raise FileNotFoundError(
            'No selected non-WAS 2dmd_low ALIGNN checkpoint was found.'
        )

    print(
        f'Found {len(records)} checkpoints: MEGNet Sparse={sparse_count}, '
        f'ALIGNN={alignn_count}'
    )
    print(f'Inference output root: {args.output_root}')
    args.output_root.mkdir(parents=True, exist_ok=True)

    dataset_cache = {}
    for material_key in args.material:
        material, test_dataset = MATERIAL_DATASETS[material_key]
        print(f'\n=== {material}: {test_dataset} high test only ===')
        summary_rows = []
        for index, record in enumerate(records, start=1):
            print(
                f'[{index}/{len(records)}] {record["model_name"]}/'
                f'{record["run_label"]}/seed{record["seed"]}'
            )
            row = predict_checkpoint(
                record,
                material_key,
                args.output_root,
                args.device,
                test_batch_size=args.test_batch_size,
                amp=args.amp,
                dataset_cache=dataset_cache,
            )
            summary_rows.append(row)
            print(
                f'  high-test MAE={row["test_mae"]} eV, '
                f'n={row["n_test"]}, '
                f'compatibility={row["model_compatibility"]}'
            )
        summary_path = write_material_summary(
            args.output_root / test_dataset / 'test_summary.csv',
            summary_rows,
        )
        print(f'{material} summary: {summary_path}')


if __name__ == '__main__':
    main()
