"""Compare native HeteroALIGNN readouts and optional hidden-width ablations.

Run on the training server from HERA's parent directory. --dry-run only
prints commands; --summary-only only reads completed checkpoints.
"""
import argparse
import copy
import csv
import json
import math
from pathlib import Path
import shlex
import statistics
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
POOLINGS = ('defect_energy_mean', 'global_mean', 'defect_global_mean')


def build_variants(poolings=POOLINGS, embedding_sizes=(64,)):
    """Keep historical paths for the default 64-wide readout experiment."""
    return {
        pooling if list(embedding_sizes) == [64] else f'{pooling}_h{width}':
        dict(pooling=pooling, embedding_size=width)
        for width in embedding_sizes for pooling in poolings
    }


def build_command(args, pooling, embedding_size=64):
    variants = build_variants(args.pooling, args.embedding_sizes)
    name = next(name for name, variant in variants.items()
                if variant == dict(pooling=pooling, embedding_size=embedding_size))
    return [
        sys.executable, '-u', '-m', 'HERA.main', '--model', 'alignn',
        '--dataset', 'native', '--mode', *args.mode,
        '--native-preprocessing', 'reference_v1',
        '--native-filter-manifest', str(args.run_dir / 'native_filter_manifest.json'),
        '--alignn-hetero-dd', 'drop', '--r', '0',
        '--alignn-hetero-feature-norm', 'layernorm',
        '--alignn-hetero-node-norm', 'layernorm',
        '--alignn-hetero-relations', 'shared_residual',
        '--alignn-hetero-adapter-rank', '8',
        '--alignn-hetero-pooling', pooling,
        '--alignn-embedding-size', str(embedding_size),
        '--seed', *[str(seed) for seed in args.seed],
        '--epochs', str(args.epochs),
        '--early-stopping-patience', str(args.early_stopping_patience),
        '--alignn-train-batch-size', str(args.batch_size),
        '--alignn-test-batch-size', '1', '--device', args.device,
        '--atom-init', str(ROOT / 'atom_init.json'),
        '--run-dir', str(args.run_dir / name), '--compact-logs', '--resume',
    ]


def compare_checkpoints(checkpoints, modes, seeds, variants=None):
    """Require paired experiments; select width/readout using validation only."""
    variants = build_variants() if variants is None else variants
    rows, recommendations = [], []
    for mode in modes:
        for seed in seeds:
            reference = None
            for name, variant in variants.items():
                key = (mode, seed, name)
                if key not in checkpoints:
                    raise ValueError(f'Missing completed checkpoint: {key}')
                checkpoint = checkpoints[key]
                config = copy.deepcopy(checkpoint['config'])
                actual_pooling = config['model'].pop('hetero_pooling', 'type_mean')
                actual_width = config['model'].pop('embedding_size', 64)
                if (actual_pooling != variant['pooling'] or actual_width != variant['embedding_size']
                        or checkpoint['mode'] != mode
                        or checkpoint['dataset_name'] != 'native'
                        or checkpoint['split']['seed'] != seed):
                    raise ValueError(f'Checkpoint identity mismatch: {key}')
                sources = {
                    part: [row['source_id'] for row in checkpoint['split'][f'{part}_sources']]
                    for part in ('train', 'val', 'test')
                }
                if not all(sources.values()):
                    raise ValueError(f'Empty split in {key}')
                if any(len(ids) != len(set(ids)) for ids in sources.values()):
                    raise ValueError(f'Duplicate sources in {key}')
                if any(set(sources[a]) & set(sources[b])
                       for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test'))):
                    raise ValueError(f'Overlapping splits in {key}')
                identity = (config, sources, checkpoint['epochs'], checkpoint.get('scaler'))
                if reference is None:
                    reference = identity
                elif identity != reference:
                    raise ValueError(f'Non-ablation config, scaler or ordered split mismatch: {key}')
                val_mae, test_mae = float(checkpoint['best_val_mae']), float(checkpoint['test_mae'])
                if not all(math.isfinite(value) and value >= 0 for value in (val_mae, test_mae)):
                    raise ValueError(f'Invalid metrics in {key}')
                rows.append(dict(mode=mode, seed=seed, variant=name, **variant,
                                 best_val_mae_ev=val_mae, test_mae_ev=test_mae,
                                 best_epoch=checkpoint['best_epoch'],
                                 epochs_completed=checkpoint['epochs_completed'],
                                 train_count=len(sources['train']), val_count=len(sources['val']),
                                 test_count=len(sources['test'])))
        grouped = []
        for name, variant in variants.items():
            group = [row for row in rows if row['mode'] == mode and row['variant'] == name]
            vals = [row['best_val_mae_ev'] for row in group]
            tests = [row['test_mae_ev'] for row in group]
            grouped.append(dict(variant=name, **variant, seeds=len(group),
                                mean_val_mae_ev=statistics.mean(vals),
                                mean_test_mae_ev=statistics.mean(tests),
                                std_test_mae_ev=statistics.stdev(tests) if len(tests) > 1 else None))
        winner = min(grouped, key=lambda row: row['mean_val_mae_ev'])
        recommendations.append(dict(mode=mode, selected_by='mean validation MAE',
                                    recommended_variant=winner['variant'],
                                    recommended_pooling=winner['pooling'],
                                    recommended_embedding_size=winner['embedding_size'], results=grouped))
    return rows, recommendations


def summarize(args):
    from ..training.trainer import load_trusted_checkpoint

    checkpoints = {}
    variants = build_variants(args.pooling, args.embedding_sizes)
    for mode in args.mode:
        for seed in args.seed:
            for name in variants:
                path = (args.run_dir / name / 'alignn' / 'native' / f'{mode}_no_dd'
                        / f'seed{seed}_best_checkpoint.pth')
                if not path.is_file():
                    raise FileNotFoundError(f'Complete all declared variants before ranking: {path}')
                checkpoint = load_trusted_checkpoint(path, 'cpu')
                # Metrics/config/split are sufficient; do not retain model tensors.
                checkpoints[mode, seed, name] = {
                    field: value for field, value in checkpoint.items() if field != 'model'
                }
    rows, recommendations = compare_checkpoints(checkpoints, args.mode, args.seed, variants)
    with (args.run_dir / 'readout_comparison.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = dict(same_ordered_splits=True, recommendations=recommendations,
                  caveats=[
                      'Validation MAE selects epochs, readout and width. Test MAE is reporting only.',
                      'Native random structure splits measure interpolation, not unseen-host generalization.',
                      'Changing seed changes both the data split and model initialization.',
                      'Concatenation adds H*H head parameters (4096 at H=64).',
                      'Changing hidden width changes the whole backbone and its initialization.',
                      'One seed is a pilot comparison, not a stable ranking.',
                  ])
    (args.run_dir / 'readout_selection.json').write_text(
        json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    lines = ['# Native readout comparison', '',
             'Matched ordered train/validation/test sources, scalers and non-ablation configurations.', '',
             '| Input mode | Readout | Hidden width | Seeds | Validation MAE (eV) | Test MAE (eV) | Test SD |',
             '|---|---|---:|---:|---:|---:|---:|']
    for recommendation in recommendations:
        for row in recommendation['results']:
            sd = 'n/a' if row['std_test_mae_ev'] is None else f"{row['std_test_mae_ev']:.6f}"
            lines.append(f"| {recommendation['mode']} | {row['pooling']} | {row['embedding_size']} | {row['seeds']} | "
                         f"{row['mean_val_mae_ev']:.6f} | {row['mean_test_mae_ev']:.6f} | {sd} |")
    for recommendation in recommendations:
        lines.extend(['', f"Validation-selected variant for {recommendation['mode']}: "
                      f"**{recommendation['recommended_variant']}**."])
    lines.extend(['', *[f'- {note}' for note in result['caveats']], ''])
    report = '\n'.join(lines)
    (args.run_dir / 'readout_comparison.md').write_text(report, encoding='utf-8')
    print(report, flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', nargs='+', choices=('hetero', 'hetero_was'), default=['hetero'])
    parser.add_argument('--pooling', nargs='+', choices=POOLINGS, default=list(POOLINGS))
    parser.add_argument('--embedding-sizes', nargs='+', type=int, default=[64],
                        help='Cross each readout with these hidden widths (default: 64)')
    parser.add_argument('--seed', nargs='+', type=int, default=[123])
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--early-stopping-patience', type=int, default=0)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--run-dir', type=Path, default=ROOT / 'logs/native_readout_benchmark')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--dry-run', action='store_true')
    group.add_argument('--summary-only', action='store_true')
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.early_stopping_patience < 0 or min(args.seed) < 0:
        parser.error('epochs/batch-size must be positive; patience/seeds must be non-negative')
    if min(args.embedding_sizes) < 2:
        parser.error('embedding-sizes must be at least 2')
    args.mode, args.seed = list(dict.fromkeys(args.mode)), list(dict.fromkeys(args.seed))
    args.pooling = list(dict.fromkeys(args.pooling))
    args.embedding_sizes = list(dict.fromkeys(args.embedding_sizes))
    args.run_dir = args.run_dir.resolve()
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.summary_only:
        summarize(args)
        return
    variants = build_variants(args.pooling, args.embedding_sizes)
    commands = [build_command(args, **variant) for variant in variants.values()]
    if args.dry_run:
        print('No training or preprocessing. On execution: freeze one strict native filter, then run:')
        for command in commands:
            print(shlex.join(command))
        return

    # The user's explicit training command is the only path that reaches here.
    import os
    from ..data.native_preprocessing import prepare_native_filter
    from ..main import iter_train_val_test_splits

    os.chdir(ROOT.parent)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    protocol = dict(poolings=args.pooling, modes=args.mode, seeds=args.seed, epochs=args.epochs,
                    batch_size=args.batch_size, early_stopping_patience=args.early_stopping_patience,
                    preprocessing='reference_v1', filter='strict', dd='drop', local_radius=0,
                    selection='mean validation MAE', schema='native_readout_v1')
    if args.embedding_sizes != [64]:
        protocol.update(embedding_sizes=args.embedding_sizes, schema='native_readout_capacity_v1')
    protocol_path = args.run_dir / 'readout_protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding='utf-8')) != protocol:
            raise ValueError('Different benchmark protocol; use a new --run-dir')
    else:
        protocol_path.write_text(json.dumps(protocol, indent=2) + '\n', encoding='utf-8')
    prepare_native_filter(args.run_dir, 'strict', args.seed, False, iter_train_val_test_splits)
    for command in commands:
        print(shlex.join(command), flush=True)
        subprocess.run(command, cwd=ROOT.parent, check=True)
    summarize(args)


if __name__ == '__main__':
    main()
