"""Compare actual-defect, all-node and concatenated readouts on native.

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


def build_command(args, pooling):
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
        '--seed', *[str(seed) for seed in args.seed],
        '--epochs', str(args.epochs),
        '--early-stopping-patience', str(args.early_stopping_patience),
        '--alignn-train-batch-size', str(args.batch_size),
        '--alignn-test-batch-size', '1', '--device', args.device,
        '--atom-init', str(ROOT / 'atom_init.json'),
        '--run-dir', str(args.run_dir / pooling), '--compact-logs', '--resume',
    ]


def compare_checkpoints(checkpoints, modes, seeds):
    """Require paired completed experiments; select readouts using validation only."""
    rows, recommendations = [], []
    for mode in modes:
        for seed in seeds:
            reference = None
            for pooling in POOLINGS:
                key = (mode, seed, pooling)
                if key not in checkpoints:
                    raise ValueError(f'Missing completed checkpoint: {key}')
                checkpoint = checkpoints[key]
                config = copy.deepcopy(checkpoint['config'])
                actual_pooling = config['model'].pop('hetero_pooling', 'type_mean')
                if (actual_pooling != pooling or checkpoint['mode'] != mode
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
                identity = (config, sources, checkpoint['epochs'])
                if reference is None:
                    reference = identity
                elif identity != reference:
                    raise ValueError(f'Non-readout config or ordered split mismatch: {key}')
                val_mae, test_mae = float(checkpoint['best_val_mae']), float(checkpoint['test_mae'])
                if not all(math.isfinite(value) and value >= 0 for value in (val_mae, test_mae)):
                    raise ValueError(f'Invalid metrics in {key}')
                rows.append(dict(mode=mode, seed=seed, pooling=pooling,
                                 best_val_mae_ev=val_mae, test_mae_ev=test_mae,
                                 best_epoch=checkpoint['best_epoch'],
                                 epochs_completed=checkpoint['epochs_completed'],
                                 train_count=len(sources['train']), val_count=len(sources['val']),
                                 test_count=len(sources['test'])))
        grouped = []
        for pooling in POOLINGS:
            group = [row for row in rows if row['mode'] == mode and row['pooling'] == pooling]
            vals = [row['best_val_mae_ev'] for row in group]
            tests = [row['test_mae_ev'] for row in group]
            grouped.append(dict(pooling=pooling, seeds=len(group),
                                mean_val_mae_ev=statistics.mean(vals),
                                mean_test_mae_ev=statistics.mean(tests),
                                std_test_mae_ev=statistics.stdev(tests) if len(tests) > 1 else None))
        winner = min(grouped, key=lambda row: row['mean_val_mae_ev'])
        recommendations.append(dict(mode=mode, selected_by='mean validation MAE',
                                    recommended_pooling=winner['pooling'], results=grouped))
    return rows, recommendations


def summarize(args):
    from ..training.trainer import load_trusted_checkpoint

    checkpoints = {}
    for mode in args.mode:
        for seed in args.seed:
            for pooling in POOLINGS:
                path = (args.run_dir / pooling / 'alignn' / 'native' / f'{mode}_no_dd'
                        / f'seed{seed}_best_checkpoint.pth')
                if not path.is_file():
                    raise FileNotFoundError(f'Complete all three readouts before ranking: {path}')
                checkpoint = load_trusted_checkpoint(path, 'cpu')
                # Metrics/config/split are sufficient; do not retain model tensors.
                checkpoints[mode, seed, pooling] = {
                    name: value for name, value in checkpoint.items() if name not in ('model', 'scaler')
                }
    rows, recommendations = compare_checkpoints(checkpoints, args.mode, args.seed)
    with (args.run_dir / 'readout_comparison.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = dict(same_ordered_splits=True, recommendations=recommendations,
                  caveats=[
                      'Validation MAE selects both epochs and readout. Test MAE is reporting only.',
                      'Native random structure splits measure interpolation, not unseen-host generalization.',
                      'Changing seed changes both the data split and model initialization.',
                      'Concatenation adds H*H head parameters (4096 at H=64).',
                      'One seed is a pilot comparison, not a stable ranking.',
                  ])
    (args.run_dir / 'readout_selection.json').write_text(
        json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    lines = ['# Native readout comparison', '',
             'Matched ordered train/validation/test sources and non-readout configurations.', '',
             '| Input mode | Readout | Seeds | Validation MAE (eV) | Test MAE (eV) | Test SD |',
             '|---|---|---:|---:|---:|---:|']
    for recommendation in recommendations:
        for row in recommendation['results']:
            sd = 'n/a' if row['std_test_mae_ev'] is None else f"{row['std_test_mae_ev']:.6f}"
            lines.append(f"| {recommendation['mode']} | {row['pooling']} | {row['seeds']} | "
                         f"{row['mean_val_mae_ev']:.6f} | {row['mean_test_mae_ev']:.6f} | {sd} |")
    for recommendation in recommendations:
        lines.extend(['', f"Validation-selected readout for {recommendation['mode']}: "
                      f"**{recommendation['recommended_pooling']}**."])
    lines.extend(['', *[f'- {note}' for note in result['caveats']], ''])
    report = '\n'.join(lines)
    (args.run_dir / 'readout_comparison.md').write_text(report, encoding='utf-8')
    print(report, flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', nargs='+', choices=('hetero', 'hetero_was'), default=['hetero'])
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
    args.mode, args.seed = list(dict.fromkeys(args.mode)), list(dict.fromkeys(args.seed))
    args.run_dir = args.run_dir.resolve()
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.summary_only:
        summarize(args)
        return
    commands = [build_command(args, pooling) for pooling in POOLINGS]
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
    protocol = dict(poolings=list(POOLINGS), modes=args.mode, seeds=args.seed, epochs=args.epochs,
                    batch_size=args.batch_size, early_stopping_patience=args.early_stopping_patience,
                    preprocessing='reference_v1', filter='strict', dd='drop', local_radius=0,
                    selection='mean validation MAE', schema='native_readout_v1')
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
