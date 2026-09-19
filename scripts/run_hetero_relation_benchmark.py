"""Paired hetero/hetero_was AA/DD ablations and optional ALIGNN references.

Run from HERA's parent; each HERA.main invocation trains and evaluates one
variant on every requested dataset. All outputs are isolated by variant.
"""
import argparse
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {
    'baseline': ('keep', 'keep'),
    'no_aa': ('drop', 'keep'),
    'no_dd': ('keep', 'drop'),
    'no_aa_dd': ('drop', 'drop'),
}
REFERENCE_MODES = ('full', 'full_x')
HETERO_MODES = ('hetero', 'hetero_was')
DATA_LISTS = {
    'native': Path('dataset/Dataset_1/Dataset_1/A_rich/Neutral/id_prop_A_rich.csv'),
    'semi': Path('dataset/Dataset_1/Dataset_1/Neutral/Neutral/id_prop_A_rich.csv'),
    'imp2d': Path('dataset/imp2d/imp2d/id_prop.csv'),
}


def data_errors(datasets, workspace=ROOT.parent):
    errors = []
    for dataset in datasets:
        path = Path(workspace) / DATA_LISTS[dataset]
        if not path.is_file():
            errors.append(f'{dataset}: missing data list: {path}')
        elif not path.read_text(encoding='utf-8-sig').strip():
            errors.append(f'{dataset}: empty data list: {path}')
    if 'semi' in datasets:
        hosts = Path(workspace) / 'dataset/Dataset_1/host_configurations'
        if not hosts.is_dir() or not next(hosts.glob('*.vasp'), None):
            errors.append(f'semi: missing host structures (*.vasp): {hosts}')
    return errors


def training_command(args, variant):
    aa, dd = VARIANTS[variant]
    command = [
        sys.executable, '-m', 'HERA.main', '--model', 'alignn',
        '--dataset', *args.dataset, '--mode', *getattr(args, 'mode', ['hetero']), '--r', '0',
        '--alignn-hetero-aa', aa, '--alignn-hetero-dd', dd,
        '--alignn-hetero-feature-norm', 'layernorm',
        '--alignn-hetero-node-norm', 'layernorm',
        '--alignn-hetero-relations', 'shared_residual', '--alignn-hetero-adapter-rank', '8',
        '--alignn-hetero-pooling', 'defect_energy_mean',
        '--alignn-hetero-defect-connectivity', 'physical', '--alignn-hetero-defect-residual', 'none',
        '--seed', *args.seed, '--epochs', str(args.epochs), '--device', args.device,
        '--alignn-train-batch-size', str(args.batch_size), '--alignn-test-batch-size', '1',
        '--atom-init', str(ROOT / 'atom_init.json'),
        '--run-dir', str(args.run_dir / variant), '--resume', '--protect-existing',
    ]
    if args.cv5:
        command.append('--cv5')
    for dataset in ('native', 'semi', 'imp2d'):
        preprocessing = getattr(args, f'{dataset}_preprocessing', None)
        if preprocessing:
            command.extend([f'--{dataset}-preprocessing', preprocessing])
    return command


def reference_command(args):
    """Ordinary ALIGNN baselines use their own graph modes and output roots."""
    command = [
        sys.executable, '-m', 'HERA.main', '--model', 'alignn',
        '--dataset', *args.dataset, '--mode', *args.reference,
        '--seed', *args.seed, '--epochs', str(args.epochs), '--device', args.device,
        '--alignn-train-batch-size', str(args.batch_size), '--alignn-test-batch-size', '1',
        '--atom-init', str(ROOT / 'atom_init.json'),
        '--run-dir', str(args.run_dir / 'references'), '--resume', '--protect-existing',
    ]
    if args.cv5:
        command.append('--cv5')
    for dataset in ('native', 'semi', 'imp2d'):
        preprocessing = getattr(args, f'{dataset}_preprocessing', None)
        if preprocessing:
            command.extend([f'--{dataset}-preprocessing', preprocessing])
    return command


def merge_saved_results(root, variants):
    from ..training.results import merge_benchmark_results
    merge_benchmark_results(root, variants)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', nargs='+', choices=tuple(DATA_LISTS), default=list(DATA_LISTS))
    parser.add_argument('--variant', nargs='+', choices=tuple(VARIANTS), default=['baseline', 'no_dd'])
    parser.add_argument('--mode', nargs='+', choices=HETERO_MODES, default=['hetero'],
                        help='Input feature modes for every AA/DD variant: hetero and/or hetero_was')
    parser.add_argument('--native-preprocessing', choices=('legacy', 'reference_v1'),
                        help='Use reference_v1 on both native modes for a fair WAS ablation; '
                        'versioned directories keep historical checkpoints intact')
    for dataset in ('semi', 'imp2d'):
        parser.add_argument(f'--{dataset}-preprocessing', choices=('legacy', 'reference_v1'),
                            help=f'Use reference_v1 on both {dataset} modes for a paired WAS comparison; '
                            'new outputs are versioned and historical checkpoints are retained')
    parser.add_argument('--reference', nargs='+', choices=REFERENCE_MODES, default=[],
                        help='Also run ordinary ALIGNN full/full_x references; duplicate no-vacancy graphs are skipped when both are requested')
    parser.add_argument('--seed', nargs='+', default=['123', '11', '1245'])
    parser.add_argument('--cv5', action='store_true')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--run-dir', type=Path, default=ROOT / 'logs/hetero_relation_native_semi_imp2d')
    parser.add_argument('--dry-run', action='store_true', help='Print commands and data availability without starting training')
    parser.add_argument('--summary-only', action='store_true',
                        help='Merge saved results for selected variants; never load datasets or train (includes all saved modes/seeds)')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error('--epochs and --batch-size must be positive')
    from ..main import parse_seed_values, modes_for_dataset
    seeds = parse_seed_values(args.seed, parser)
    if args.cv5 and len(seeds) != 1:
        parser.error('--cv5 requires exactly one --seed value')
    args.dataset = list(dict.fromkeys(args.dataset))
    args.variant = list(dict.fromkeys(args.variant))
    args.mode = list(dict.fromkeys(args.mode))
    args.reference = list(dict.fromkeys(args.reference))
    args.run_dir = args.run_dir.resolve()
    if args.summary_only:
        roots = [args.run_dir / variant for variant in args.variant]
        if args.reference:
            roots.append(args.run_dir / 'references')
        existing = [root for root in roots if root.is_dir()]
        if not existing:
            parser.error(f'No saved variant directories found under {args.run_dir}. '
                         '--run-dir must be the parent of baseline/no_dd, not the variant directory itself.')
        if args.dry_run:
            for root in existing:
                print(f'Would merge saved results in {root} (no training).', flush=True)
        else:
            merge_saved_results(args.run_dir, [root.name for root in roots])
        return
    errors = data_errors(args.dataset)
    if errors and not args.dry_run:
        parser.error('Data preflight failed before training:\n' + '\n'.join(errors))
    if errors:
        print('Data unavailable (dry run only):\n' + '\n'.join(errors), flush=True)
    ref_count = sum(len(modes_for_dataset(args.reference, d)) for d in args.dataset)
    count = (len(args.dataset) * len(args.variant) * len(args.mode) + ref_count) * (5 if args.cv5 else len(seeds))
    print(f'{count} training/test runs; variants={args.variant}; modes={args.mode}; references={args.reference}; datasets={args.dataset}', flush=True)
    if 'hetero_was' in args.mode:
        print('hetero_was uses 184-D current/reference features. Native/semi/imp2d default to true '
              'original-site labels (reference_v1), with separate versioned outputs. Use each dataset\'s '
              '--<dataset>-preprocessing reference_v1 on both modes for a paired comparison.', flush=True)
        for dataset in args.dataset:
            if getattr(args, f'{dataset}_preprocessing') is None and 'hetero' in args.mode:
                print(f'For a paired {dataset} comparison use --{dataset}-preprocessing reference_v1. '
                      'Without it, ordinary hetero keeps its historical inputs/results.', flush=True)
    for dataset in args.dataset:
        if getattr(args, f'{dataset}_preprocessing') == 'legacy':
            print(f'{dataset} legacy requested: historical graphs and WAS fallback.', flush=True)
    if args.reference:
        print('Hetero vacancy inputs already contain X. full_x is an ordinary ALIGNN reference.', flush=True)
        for dataset in args.dataset:
            selected = modes_for_dataset(args.reference, dataset)
            if selected != args.reference:
                print(f'{dataset}: full_x equals full (no vacancy X); running {selected} once.', flush=True)
        command = reference_command(args)
        print(f'\n[references] {shlex.join(command)}', flush=True)
        if not args.dry_run:
            try:
                subprocess.run(command, cwd=ROOT.parent, check=True)
            finally:
                merge_saved_results(args.run_dir, ['references'])
    for variant in args.variant:
        command = training_command(args, variant)
        print(f'\n[{variant}] {shlex.join(command)}', flush=True)
        if not args.dry_run:
            try:
                subprocess.run(command, cwd=ROOT.parent, check=True)
            finally:
                # Include older variants even if this call adds only WAS/no_dd.
                saved_variants = [name for name in (*VARIANTS, 'references')
                                  if (args.run_dir / name).is_dir()]
                merge_saved_results(args.run_dir, saved_variants)


if __name__ == '__main__':
    main()
