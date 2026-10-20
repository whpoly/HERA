"""Run the five selected ALIGNN datasets and compare monitored early stopping.

This runner never enables Hypergraph. Dataset-specific graph settings and
curation policies match the previously requested benchmark commands.
"""
import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ('native', 'semi', 'imp2d', '2dmd_low', '2dmd_high')
MODES = ('full', 'full_x', 'attention', 'attention_was', 'hetero',
         'hetero_fixed_pool', 'hetero_was', 'was_x', 'definet', 'definet_was')
HETERO_MODES = ('hetero', 'hetero_fixed_pool', 'hetero_was')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', nargs='+', choices=DATASETS, default=list(DATASETS))
    parser.add_argument('--seed', nargs='+', type=int, default=[11, 42, 1245])
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--early-stopping-patience', type=int, default=50)
    parser.add_argument('--early-stopping-min-delta-percent', type=float, default=0.5)
    parser.add_argument('--point-loss', choices=('mse', 'huber_ev'), default='mse')
    parser.add_argument('--huber-delta-ev', type=float, default=1.0,
                        help='Global Huber transition in eV; fallback for datasets without an override')
    parser.add_argument('--huber-delta-ev-by-dataset', nargs='+', action='extend', default=[],
                        metavar='DATASET=EV',
                        help='Per-dataset transitions, e.g. native=1 2dmd_low=0.01; requires huber_ev')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--run-dir', type=Path,
                        default=ROOT / 'logs/all5_3seeds_early_stop_monitor')
    parser.add_argument('--native-preparation-dir', type=Path,
                        help='Reuse a native preparation directory with the same seeds/protocol')
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--dry-run', action='store_true')
    actions.add_argument('--summary-only', action='store_true')
    args = parser.parse_args(argv)
    if min(args.seed) < 0 or args.epochs < 1 or args.early_stopping_patience < 1:
        parser.error('Seeds must be nonnegative; epochs and monitoring patience must be positive')
    if not 0 <= args.early_stopping_min_delta_percent < 100:
        parser.error('--early-stopping-min-delta-percent must be in [0, 100)')
    if not math.isfinite(args.huber_delta_ev) or args.huber_delta_ev <= 0:
        parser.error('--huber-delta-ev must be finite and positive')
    args.seed = list(dict.fromkeys(args.seed))
    args.dataset = list(dict.fromkeys(args.dataset))
    if args.huber_delta_ev_by_dataset and args.point_loss != 'huber_ev':
        parser.error('--huber-delta-ev-by-dataset requires --point-loss huber_ev')
    overrides = {}
    for item in args.huber_delta_ev_by_dataset:
        dataset, separator, value = item.partition('=')
        if not separator or dataset not in args.dataset:
            parser.error(f'Expected selected DATASET=EV for --huber-delta-ev-by-dataset: {item!r}')
        if dataset in overrides:
            parser.error(f'Duplicate Huber transition for dataset {dataset!r}')
        try:
            delta = float(value)
        except ValueError:
            parser.error(f'Invalid Huber transition: {item!r}')
        if not math.isfinite(delta) or delta <= 0:
            parser.error(f'Huber transition must be finite and positive: {item!r}')
        overrides[dataset] = delta
    args.huber_delta_ev_by_dataset = overrides
    args.run_dir = args.run_dir.resolve()
    args.native_preparation_dir = (
        args.native_preparation_dir.resolve() if args.native_preparation_dir
        else args.run_dir / 'native_preparation'
    )
    return args


def huber_deltas(args):
    """Resolve physical thresholds once per dataset, shared by all modes/seeds."""
    if args.point_loss != 'huber_ev':
        return {}
    return {dataset: args.huber_delta_ev_by_dataset.get(dataset, args.huber_delta_ev)
            for dataset in args.dataset}


def benchmark_protocol(args):
    protocol = dict(
        schema='all5_no_hypergraph_monitor_v1', datasets=args.dataset, modes=list(MODES),
        seeds=args.seed, epochs=args.epochs, patience=args.early_stopping_patience,
        min_delta_percent=args.early_stopping_min_delta_percent, point_loss=args.point_loss,
        huber_delta_ev=None,
    )
    deltas = huber_deltas(args)
    if deltas:
        if len(set(deltas.values())) == 1:
            # Keep existing uniform-Huber protocol files compatible.
            protocol['huber_delta_ev'] = next(iter(deltas.values()))
        else:
            protocol['huber_delta_ev_by_dataset'] = deltas
    return protocol


def validate_protocol(args):
    protocol = benchmark_protocol(args)
    path = args.run_dir / 'benchmark_protocol.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != protocol:
        raise ValueError(f'Existing benchmark protocol differs; choose another --run-dir: {path}')
    return protocol


def modes_for_dataset(dataset):
    # Full-X is identical to Full for these two vacancy-free datasets.
    return [mode for mode in MODES if mode != 'full_x' or dataset not in ('semi', 'imp2d')]


def mode_directory(args, dataset, mode):
    label = f'{mode}_no_dd' if dataset in ('native', 'semi', 'imp2d') and mode in HETERO_MODES else mode
    return args.run_dir / 'alignn' / dataset / label


def build_commands(args):
    commands = []
    deltas = huber_deltas(args)
    if 'native' in args.dataset:
        commands.append([
            sys.executable, '-u', '-m', 'HERA.scripts.run_native_filter_benchmark',
            '--suite', 'filtered_dataset', '--max-energy-ev', '15', '--exclude-poscar0',
            '--seed', *map(str, args.seed), '--epochs', str(args.epochs),
            '--run-dir', str(args.native_preparation_dir), '--prepare-only',
        ])
    for dataset in args.dataset:
        command = [
            sys.executable, '-u', '-m', 'HERA.main', '--model', 'alignn',
            '--dataset', dataset, '--mode', *modes_for_dataset(dataset), '--r', '0',
            '--alignn-embedding-size', '64', '--alignn-cutoff', '6',
            '--alignn-max-neighbors', '12', '--alignn-nblocks', '3',
            '--alignn-gcn-blocks', '3', '--alignn-test-batch-size', '1',
            '--alignn-train-batch-size',
            str({'native': 16, 'semi': 64, 'imp2d': 64, '2dmd_low': 8, '2dmd_high': 16}[dataset]),
            '--alignn-hetero-node-norm', 'layernorm',
            '--point-loss', args.point_loss, '--seed', *map(str, args.seed),
            '--epochs', str(args.epochs),
            '--early-stopping-patience', str(args.early_stopping_patience),
            '--early-stopping-min-delta-percent', str(args.early_stopping_min_delta_percent),
            '--early-stopping-monitor-only',
            '--device', args.device, '--atom-init', str(ROOT / 'atom_init.json'),
            '--run-dir', str(args.run_dir), '--compact-logs', '--resume',
        ]
        if args.point_loss == 'huber_ev':
            command += ['--huber-delta-ev', str(deltas[dataset])]
        if dataset in ('native', 'semi', 'imp2d'):
            command += [
                f'--{dataset}-preprocessing', 'reference_v1',
                '--alignn-hetero-feature-norm', 'layernorm',
                '--alignn-hetero-relations', 'shared_residual',
                '--alignn-hetero-adapter-rank', '8',
                '--alignn-hetero-pooling', 'defect_energy_mean',
                '--alignn-hetero-aa', 'keep', '--alignn-hetero-dd', 'drop',
            ]
        else:
            command += [
                '--alignn-hetero-feature-norm', 'batchnorm',
                '--alignn-hetero-relations', 'independent',
                '--alignn-hetero-pooling', 'type_mean',
                '--alignn-hetero-aa', 'keep', '--alignn-hetero-dd', 'keep',
            ]
        if dataset == 'native':
            command += ['--native-filter-manifest',
                        str(args.native_preparation_dir / 'population_mse/native_filter_manifest.json')]
        elif dataset == 'semi':
            command += ['--semi-quality-filter', 'physical', '--semi-source-policy', 'legacy_available']
        elif dataset == 'imp2d':
            command += [
                '--imp2d-source', 'db', '--imp2d-quality-filter', 'physical',
                '--imp2d-host-filter', 'reviewed_v1', '--imp2d-energy-window', '-10', '10',
                '--imp2d-defect-type', 'interstitial',
            ]
        commands.append(command)
    return commands


def _checkpoint_record(path):
    """Recover a completed comparison if interruption preceded its JSON write."""
    import torch

    saved = torch.load(path, map_location='cpu', weights_only=False)
    optim = saved['config']['optim']
    if saved.get('evaluation_stage') != 'final' or not optim.get('early_stopping_monitor_only'):
        raise ValueError(f'Checkpoint is not a completed monitored experiment: {path}')
    return dict(
        schema='early_stopping_comparison_v1', model=saved['model_name'],
        dataset=saved['dataset_name'], mode=saved['mode'], run_label=saved['run_label'],
        seed=saved['split']['seed'], logger_id=saved['split']['logger_id'],
        epochs_requested=saved['epochs'], patience=optim['early_stopping_patience'],
        min_delta_percent=optim['early_stopping_min_delta_percent'],
        early_stopping=saved.get('early_stopping_result'),
        final={key: saved[key] for key in ('epochs_completed', 'best_epoch', 'best_val_mae', 'test_mae')},
    )


def _mean_std(values):
    if not values:
        return None
    return dict(mean=statistics.mean(values),
                std=statistics.stdev(values) if len(values) > 1 else None, n=len(values))


def _display(stats):
    if stats is None:
        return '—'
    return (f'{stats["mean"]:.6f} ± {stats["std"]:.6f}'
            if stats['std'] is not None else f'{stats["mean"]:.6f} (n=1)')


def summarize(args):
    from ..training.results import atomic_write_text

    protocol = validate_protocol(args)
    deltas = huber_deltas(args)
    records, missing = [], []
    groups = defaultdict(list)
    for dataset in args.dataset:
        for mode in modes_for_dataset(dataset):
            for seed in args.seed:
                directory = mode_directory(args, dataset, mode)
                path = directory / f'seed{seed}_evaluation_comparison.json'
                record = json.loads(path.read_text(encoding='utf-8')) if path.exists() else None
                checkpoint = directory / f'seed{seed}_best_checkpoint.pth'
                if (record is None or record.get('final') is None) and checkpoint.exists():
                    record = _checkpoint_record(checkpoint)
                if record is None or record.get('final') is None:
                    missing.append(dict(dataset=dataset, mode=mode, seed=seed))
                    continue
                if (record.get('schema') != 'early_stopping_comparison_v1'
                        or record.get('model') != 'alignn'
                        or (record['dataset'], record['mode'], record['seed']) != (dataset, mode, seed)
                        or record['epochs_requested'] != args.epochs
                        or record['final']['epochs_completed'] != args.epochs
                        or record['patience'] != args.early_stopping_patience
                        or record['min_delta_percent'] != args.early_stopping_min_delta_percent):
                    raise ValueError(f'Comparison identity/protocol mismatch: {path}')
                records.append(record)
                groups[dataset, mode].append(record)
    rows = []
    for (dataset, mode), values in groups.items():
        paired = [r for r in values if r['early_stopping'] is not None]
        rows.append(dict(
            dataset=dataset, mode=mode, completed=len(values), triggered=len(paired),
            early_test_mae=_mean_std([r['early_stopping']['test_mae'] for r in paired]),
            final_test_mae=_mean_std([r['final']['test_mae'] for r in values]),
            paired_delta=_mean_std([
                r['final']['test_mae'] - r['early_stopping']['test_mae'] for r in paired
            ]),
        ))
    output = dict(schema='all_dataset_early_stopping_summary_v1', protocol=protocol,
                  records=records, summary=rows, missing=missing)
    atomic_write_text(args.run_dir / 'early_stopping_comparison.json',
                      json.dumps(output, indent=2, allow_nan=False) + '\n')
    lines = [
        '# Early-stopping monitor versus full training', '',
        f'Point loss: `{args.point_loss}`.',
        'Huber transitions (eV): ' + (
            ', '.join(f'{dataset}={delta:g}' for dataset, delta in deltas.items())
            if deltas else 'not used'
        ) + '.', '',
        'Both stages select weights using validation MAE only. Test scores do not affect training.',
        'Final means the best validation checkpoint over the entire training budget, not the last-epoch weights.',
        'Early-stage means include only triggered seeds; paired delta uses those same seeds (final minus early).',
        'A missing trigger is not an early test result. Standard deviations use ddof=1.', '',
        '| Dataset | Mode | Completed | Triggered | Early test MAE | Final test MAE (all completed) | Paired delta |',
        '|---|---|---:|---:|---:|---:|---:|',
    ]
    for row in rows:
        lines.append(f'| {row["dataset"]} | {row["mode"]} | {row["completed"]} | '
                     f'{row["triggered"]} | {_display(row["early_test_mae"])} | '
                     f'{_display(row["final_test_mae"])} | {_display(row["paired_delta"])} |')
    lines += ['', '| Dataset | Mode | Seed | Trigger epoch | Early best epoch | Early test MAE | Final best epoch | Final test MAE |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for record in records:
        early, final = record['early_stopping'], record['final']
        early_cells = (f'{early["trigger_epoch"]} | {early["best_epoch"]} | {early["test_mae"]:.6f}'
                       if early is not None else '— | — | —')
        lines.append(f'| {record["dataset"]} | {record["mode"]} | {record["seed"]} | '
                     f'{early_cells} | {final["best_epoch"]} | {final["test_mae"]:.6f} |')
    lines += ['', f'Incomplete/missing runs: {len(missing)}.', '']
    path = args.run_dir / 'early_stopping_comparison.md'
    atomic_write_text(path, '\n'.join(lines))
    print(f'Completed {len(records)} runs; incomplete/missing {len(missing)}. Comparison: {path}', flush=True)
    return output


def main(argv=None):
    args = parse_args(argv)
    if args.summary_only:
        if not args.run_dir.exists():
            raise FileNotFoundError(args.run_dir)
        summarize(args)
        return
    commands = build_commands(args)
    print(f'Datasets: {args.dataset}; seeds: {args.seed}; epochs: {args.epochs}; '
          f'loss: {args.point_loss}; Hypergraph disabled; early stopping only records/tests.', flush=True)
    if args.point_loss == 'huber_ev':
        print(f'Huber transitions (eV): {huber_deltas(args)}', flush=True)
    if args.dry_run:
        for command in commands:
            print(subprocess.list2cmdline(command) if os.name == 'nt' else shlex.join(command))
        return
    from ..training.results import atomic_write_text

    protocol = validate_protocol(args)
    path = args.run_dir / 'benchmark_protocol.json'
    atomic_write_text(path, json.dumps(protocol, indent=2) + '\n')
    for index, command in enumerate(commands, 1):
        print(f'[{index}/{len(commands)}] '
              + (subprocess.list2cmdline(command) if os.name == 'nt' else shlex.join(command)), flush=True)
        subprocess.run(command, cwd=ROOT.parent, check=True)
    summarize(args)


if __name__ == '__main__':
    main()
