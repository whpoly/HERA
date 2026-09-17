"""Paired AA/DD graph ablations on native, semi and imp2d.

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
        '--dataset', *args.dataset, '--mode', 'hetero', '--r', '0',
        '--alignn-hetero-aa', aa, '--alignn-hetero-dd', dd,
        '--alignn-hetero-feature-norm', 'layernorm',
        '--alignn-hetero-node-norm', 'layernorm',
        '--alignn-hetero-relations', 'shared_residual', '--alignn-hetero-adapter-rank', '8',
        '--alignn-hetero-pooling', 'defect_energy_mean',
        '--alignn-hetero-defect-connectivity', 'physical', '--alignn-hetero-defect-residual', 'none',
        '--seed', *args.seed, '--epochs', str(args.epochs), '--device', args.device,
        '--alignn-train-batch-size', str(args.batch_size), '--alignn-test-batch-size', '1',
        '--atom-init', str(ROOT / 'atom_init.json'),
        '--run-dir', str(args.run_dir / variant), '--resume',
    ]
    if args.cv5:
        command.append('--cv5')
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', nargs='+', choices=tuple(DATA_LISTS), default=list(DATA_LISTS))
    parser.add_argument('--variant', nargs='+', choices=tuple(VARIANTS), default=['baseline', 'no_dd'])
    parser.add_argument('--seed', nargs='+', default=['123', '11', '1245'])
    parser.add_argument('--cv5', action='store_true')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--run-dir', type=Path, default=ROOT / 'logs/hetero_relation_native_semi_imp2d')
    parser.add_argument('--dry-run', action='store_true', help='Print commands and data availability without starting training')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error('--epochs and --batch-size must be positive')
    from ..main import parse_seed_values
    seeds = parse_seed_values(args.seed, parser)
    if args.cv5 and len(seeds) != 1:
        parser.error('--cv5 requires exactly one --seed value')
    args.dataset = list(dict.fromkeys(args.dataset))
    args.variant = list(dict.fromkeys(args.variant))
    args.run_dir = args.run_dir.resolve()
    errors = data_errors(args.dataset)
    if errors and not args.dry_run:
        parser.error('Data preflight failed before training:\n' + '\n'.join(errors))
    if errors:
        print('Data unavailable (dry run only):\n' + '\n'.join(errors), flush=True)
    count = len(args.dataset) * len(args.variant) * (5 if args.cv5 else len(seeds))
    print(f'{count} training/test runs; variants={args.variant}; datasets={args.dataset}', flush=True)
    for variant in args.variant:
        command = training_command(args, variant)
        print(f'\n[{variant}] {shlex.join(command)}', flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT.parent, check=True)


if __name__ == '__main__':
    main()
