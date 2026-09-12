#!/usr/bin/env python
"""Run Sparse MEGNet and ALIGNN, then merge fixed-test predictions.

This suite exists because ``sparse`` is a MEGNet-only representation.  It runs
the two model families through their valid modes in separate HERA invocations
and writes one wide CSV containing their predictions on the identical fixed
high-density test structures.
"""

import argparse
import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from .config.defaults import get_config, hypergraph_run_components, alignn_hetero_run_components
from .main import (
    ALIGNN_DEFAULT_MODES,
    ALL_BENCHMARK_SEEDS,
    LOCAL_CUTOFF_SWEEP_MODES,
)


VALID_TRANSFER_DATASETS = (
    '2dmd_mos2',
    '2dmd_wse2',
    'vacancy_mos2',
    'vacancy_wse2',
)
VALID_ALIGNN_MODES = tuple(ALIGNN_DEFAULT_MODES)
VALID_ALIGNN_RADII = ('0', '3', '4', '5', '6', '7', 'all')


def parse_seed_values(values, parser):
    values = [str(value).strip() for value in values]
    if any(value.lower() == 'all' for value in values):
        if len(values) != 1:
            parser.error('--seed all cannot be combined with explicit seeds')
        return list(ALL_BENCHMARK_SEEDS), ['all']
    try:
        seeds = [int(value) for value in values]
    except ValueError:
        parser.error('--seed must contain integers or the single value all')
    if any(seed < 0 for seed in seeds):
        parser.error('--seed values must be non-negative')
    return seeds, [str(seed) for seed in seeds]


def prediction_file_path(run_dir, model, dataset, mode, seed):
    mode_parts = [Path(run_dir), model, dataset, mode]
    if mode == 'hetero':
        mode_parts.append(Path('r0'))
        if model == 'alignn':
            mode_parts.extend(alignn_hetero_run_components(get_config(model, dataset, mode)['model']))
    elif mode in ('hypergraph', 'hypergraph_was'):
        mode_parts.extend(hypergraph_run_components(get_config(model, dataset, mode)['model']))
    return Path(*mode_parts) / f'seed{seed}_test_predictions.csv'


def result_prefix(model, mode):
    if model == 'alignn' and mode == 'hetero':
        return 'alignn_hetero_r0'
    return f'{model}_{mode}'


def alignn_result_prefix(dataset_root, prediction_path):
    """Return a stable column prefix from an ALIGNN result directory."""
    relative_parent = Path(prediction_path).parent.relative_to(dataset_root)
    parts = relative_parent.parts
    if not parts:
        raise ValueError(
            f'ALIGNN prediction file is not inside a mode directory: '
            f'{prediction_path}'
        )
    mode = parts[0]
    suffixes = [
        part for part in parts[1:]
        if part.startswith(('r', 'norm_', 'pool_', 'features_', 'updates_'))
    ]
    return '_'.join(('alignn', mode, *suffixes))


def build_training_command(args, model, modes):
    command = [
        sys.executable,
        '-m',
        'HERA.main',
        '--model',
        model,
        '--dataset',
        *args.dataset,
        '--mode',
        *modes,
        '--epochs',
        str(args.epochs),
        '--seed',
        *args.seed_cli,
        '--device',
        args.device,
        '--atom-init',
        str(args.atom_init),
        '--run-dir',
        str(args.run_dir),
    ]
    if model == 'alignn' and any(
            mode in LOCAL_CUTOFF_SWEEP_MODES for mode in modes
    ):
        command.extend(('--r', *args.alignn_r))
    if args.resume:
        command.append('--resume')
    if args.early_stopping_patience is not None:
        command.extend((
            '--early-stopping-patience',
            str(args.early_stopping_patience),
        ))
    if model == 'alignn':
        if args.alignn_train_batch_size is not None:
            command.extend((
                '--alignn-train-batch-size',
                str(args.alignn_train_batch_size),
            ))
        if args.alignn_amp:
            command.append('--alignn-amp')
    return command


def _read_prediction_rows(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def discover_prediction_files(
        run_dir, dataset, seed, alignn_modes, alignn_r=None,
):
    """Find Sparse MEGNet plus every requested ALIGNN prediction file."""
    run_dir = Path(run_dir)
    sources = [(
        'megnet_sparse',
        prediction_file_path(run_dir, 'megnet', dataset, 'sparse', seed),
    )]
    alignn_root = run_dir / 'alignn' / dataset
    found_modes = set()
    seen_prefixes = {'megnet_sparse'}
    allowed_radius_labels = None
    if alignn_r is not None:
        radius_values = (
            VALID_ALIGNN_RADII[:-1] if 'all' in alignn_r else alignn_r
        )
        allowed_radius_labels = {f'r{radius}' for radius in radius_values}

    if alignn_root.is_dir():
        pattern = f'seed{seed}_test_predictions.csv'
        paths = sorted(alignn_root.rglob(pattern))
    else:
        paths = []
    mode_order = {mode: index for index, mode in enumerate(alignn_modes)}
    paths.sort(key=lambda path: (
        mode_order.get(path.relative_to(alignn_root).parts[0], len(mode_order)),
        path.as_posix(),
    ))
    for path in paths:
        relative_parts = path.relative_to(alignn_root).parts
        mode = relative_parts[0]
        if mode not in alignn_modes:
            continue
        if mode in LOCAL_CUTOFF_SWEEP_MODES and allowed_radius_labels is not None:
            radius_labels = {
                part for part in relative_parts[1:] if part.startswith('r')
            }
            if not radius_labels.intersection(allowed_radius_labels):
                continue
        prefix = alignn_result_prefix(alignn_root, path)
        if prefix in seen_prefixes:
            raise ValueError(
                f'Multiple prediction files map to {prefix} for '
                f'{dataset}, seed={seed}: {path}'
            )
        seen_prefixes.add(prefix)
        found_modes.add(mode)
        sources.append((prefix, path))

    # At r=0, main.py intentionally skips hetero_fixed_pool because its output
    # is identical to hetero_r0. Every other requested mode must be present.
    fixed_pool_is_redundant = allowed_radius_labels == {'r0'}
    missing_modes = [
        mode for mode in alignn_modes
        if mode not in found_modes
        and not (mode == 'hetero_fixed_pool' and fixed_pool_is_redundant)
    ]
    if missing_modes:
        raise FileNotFoundError(
            f'Missing ALIGNN predictions for {dataset}, seed={seed}: '
            f'{", ".join(missing_modes)} (searched under {alignn_root})'
        )
    return sources


def merge_prediction_files(
        run_dir, datasets, seeds, alignn_modes, output_path=None,
        alignn_r=None,
):
    """Merge all model predictions by dataset, seed, and source structure."""
    run_dir = Path(run_dir)
    output_path = Path(
        output_path or run_dir / 'model_comparison_predictions.csv'
    )
    metadata_fields = (
        'source_id',
        'source_name',
        'source_path',
        'material',
        'concentration',
        'defect_family',
    )
    records = {}
    key_order = []
    prefixes = []
    group_prefixes = {}

    for dataset in datasets:
        for seed in seeds:
            group = (dataset, int(seed))
            sources = discover_prediction_files(
                run_dir, dataset, seed, alignn_modes, alignn_r=alignn_r,
            )
            group_prefixes[group] = []
            for prefix, prediction_path in sources:
                if not prediction_path.is_file():
                    raise FileNotFoundError(
                        f'Missing prediction file for {prefix}/{dataset}: '
                        f'{prediction_path}'
                    )
                if prefix not in prefixes:
                    prefixes.append(prefix)
                group_prefixes[group].append(prefix)
                source_rows = _read_prediction_rows(prediction_path)
                if not source_rows:
                    raise ValueError(f'Prediction file is empty: {prediction_path}')
                for source_row in source_rows:
                    key = (dataset, int(seed), source_row['source_id'])
                    if key not in records:
                        records[key] = {
                            'dataset': dataset,
                            'seed': int(seed),
                            **{
                                field: source_row.get(field, '')
                                for field in metadata_fields
                            },
                            'target': source_row['target'],
                        }
                        key_order.append(key)
                    else:
                        existing_target = float(records[key]['target'])
                        incoming_target = float(source_row['target'])
                        if abs(existing_target - incoming_target) > 1e-6:
                            raise ValueError(
                                f'Inconsistent targets for {key}: '
                                f'{existing_target} versus {incoming_target}'
                            )
                    records[key][f'{prefix}_prediction'] = source_row['prediction']
                    records[key][f'{prefix}_absolute_error'] = source_row['absolute_error']

    for group, present_prefixes in group_prefixes.items():
        missing_prefixes = [
            prefix for prefix in prefixes if prefix not in present_prefixes
        ]
        if missing_prefixes:
            raise FileNotFoundError(
                f'Prediction variants differ across datasets/seeds; {group} is '
                f'missing: {", ".join(missing_prefixes)}'
            )

    result_fields = ['dataset', 'seed', *metadata_fields, 'target']
    for prefix in prefixes:
        result_fields.extend((
            f'{prefix}_prediction',
            f'{prefix}_absolute_error',
            f'{prefix}_test_mae',
        ))
    for dataset in datasets:
        for seed in seeds:
            group_keys = [
                key for key in key_order
                if key[0] == dataset and key[1] == int(seed)
            ]
            if not group_keys:
                raise ValueError(
                    f'No merged prediction rows for {dataset}, seed={seed}'
                )
            for prefix in prefixes:
                missing_keys = [
                    key for key in group_keys
                    if f'{prefix}_absolute_error' not in records[key]
                ]
                if missing_keys:
                    raise ValueError(
                        f'{prefix} has {len(missing_keys)} missing test samples '
                        f'for {dataset}, seed={seed}'
                    )
                mae = sum(
                    float(records[key][f'{prefix}_absolute_error'])
                    for key in group_keys
                ) / len(group_keys)
                for key in group_keys:
                    records[key][f'{prefix}_test_mae'] = f'{mae:.10g}'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=result_fields)
        writer.writeheader()
        writer.writerows(records[key] for key in key_order)
    return output_path, [records[key] for key in key_order], prefixes


def print_mae_summary(rows, prefixes):
    print('\nCombined fixed high-density test MAE:')
    datasets = list(dict.fromkeys(row['dataset'] for row in rows))
    seeds = list(dict.fromkeys(int(row['seed']) for row in rows))
    for dataset in datasets:
        for seed in seeds:
            subset = [
                row for row in rows
                if row['dataset'] == dataset and int(row['seed']) == seed
            ]
            if not subset:
                continue
            for prefix in prefixes:
                errors = [
                    float(row[f'{prefix}_absolute_error'])
                    for row in subset
                ]
                print(
                    f'  {dataset} | seed={seed} | {prefix}: '
                    f'MAE={sum(errors) / len(errors):.6f} eV '
                    f'(n={len(errors)})'
                )


def parse_args(argv=None):
    package_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            'Run Sparse MEGNet plus ALIGNN on the same low-to-high transfer '
            'dataset and merge their high-density predictions into one CSV.'
        )
    )
    parser.add_argument(
        '--dataset',
        choices=VALID_TRANSFER_DATASETS,
        required=True,
        help='Run exactly one material per command/output directory.',
    )
    parser.add_argument(
        '--alignn-mode',
        nargs='+',
        choices=(*VALID_ALIGNN_MODES, 'all'),
        required=True,
        help='Your selected ALIGNN modes, or all for the configured suite.',
    )
    parser.add_argument(
        '--alignn-r',
        nargs='+',
        choices=VALID_ALIGNN_RADII,
        default=['0'],
        help=(
            'Radius values for ALIGNN hetero modes (default: 0). Use all '
            'for the complete 0/3/4/5/6/7 sweep.'
        ),
    )
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--seed', nargs='+', default=['123'])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--atom-init', type=Path, default=package_root / 'atom_init.json')
    parser.add_argument('--run-dir', type=Path, default=None)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--early-stopping-patience', type=int, default=None)
    parser.add_argument('--alignn-train-batch-size', type=int, default=None)
    parser.add_argument('--alignn-amp', action='store_true')
    args = parser.parse_args(argv)
    if args.epochs < 1:
        parser.error('--epochs must be >= 1')
    if args.alignn_train_batch_size is not None and args.alignn_train_batch_size < 1:
        parser.error('--alignn-train-batch-size must be >= 1')
    dataset_name = args.dataset
    args.dataset = [dataset_name]
    if 'all' in args.alignn_mode:
        if len(args.alignn_mode) != 1:
            parser.error('--alignn-mode all cannot be combined with specific modes')
        args.alignn_mode = list(VALID_ALIGNN_MODES)
    else:
        args.alignn_mode = list(dict.fromkeys(args.alignn_mode))
    if 'all' in args.alignn_r:
        if len(args.alignn_r) != 1:
            parser.error('--alignn-r all cannot be combined with specific radii')
    else:
        args.alignn_r = list(dict.fromkeys(args.alignn_r))
    args.seeds, args.seed_cli = parse_seed_values(args.seed, parser)
    if args.run_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.run_dir = (
            package_root / 'logs' /
            f'{dataset_name}_sparse_megnet_alignn_{timestamp}'
        )
    else:
        args.run_dir = args.run_dir.resolve()
    args.atom_init = args.atom_init.resolve()
    return args


def main():
    args = parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    phases = (
        ('megnet', ['sparse']),
        ('alignn', args.alignn_mode),
    )
    for model, modes in phases:
        command = build_training_command(args, model, modes)
        print(f'\nRunning {model.upper()} modes: {modes}')
        subprocess.run(command, check=True)

    output_path, rows, prefixes = merge_prediction_files(
        args.run_dir,
        args.dataset,
        args.seeds,
        args.alignn_mode,
        alignn_r=args.alignn_r,
    )
    print_mae_summary(rows, prefixes)
    print(f'\nCombined prediction file: {output_path}')


if __name__ == '__main__':
    main()
