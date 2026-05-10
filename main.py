#!/usr/bin/env python
"""
Crystal Graph Neural Network Training CLI
==========================================

Usage examples:

  # Train MEGNet on vacancy dataset with the default modes
  python -m HERA.main --model megnet --dataset vacancy

  # Train CGCNN on 2dmd_high with only the local r=0 defect-only input
  python -m HERA.main --model cgcnn --dataset 2dmd_high --mode local --r 0

  # Train MEGNet on semi dataset with local r=0 + hetero modes
  python -m HERA.main --model megnet --dataset semi --mode local hetero --r 0

  # Run local radius/cutoff ablation; r=0 is sparse-equivalent local input
  python -m HERA.main --model cgcnn --dataset vacancy --mode local --r 0 3 4 5 6 7

  # Run every configured dataset/mode/radius for MEGNet and CGCNN
  python -m HERA.main --model all --dataset all --mode all --r all

  # Custom device, epochs, and random seeds
  python -m HERA.main --model cgcnn --dataset native --device cuda:1 --epochs 300 --seeds 42 123

  # Five-fold cross validation; --seeds must contain exactly one random state
  python -m HERA.main --model cgcnn --dataset native --mode local --r 0 --cv5 --seeds 42

Supported combinations:
  Models  : megnet, cgcnn, definet, all
  Modes   : full, hetero, local, attention, was, hetero_was,
            attention_local, attention_was, attention_local_was,
            definet, definet_local, definet_was, definet_local_was, all
  Datasets: vacancy, 2dmd_high, native, och, imp2d, semi, all
"""

import argparse
import copy
import os
import random
import warnings
from datetime import datetime

import numpy as np
import torch
from sklearn.model_selection import KFold, train_test_split

from .config.defaults import get_config, VALID_DATASETS, VALID_MODELS, VALID_MODES
from .data.datasets import load_dataset, init_elem_embedding
from .training.trainer import MEGNetTrainer
from .training.history import TrainingLogger


LOCAL_CUTOFF_CHOICES = [0, 3, 4, 5, 6, 7]
DEFAULT_SEEDS = [123, 11, 1245, 34, 42, 80, 13232, 8, 99, 101]
ALL_MODEL_SUITES = ('megnet', 'cgcnn')
CGCNN_DEFINET_MODES = (
    'definet',
    'definet_local',
    'definet_was',
    'definet_local_was',
)
LOCAL_GRAPH_SWEEP_MODES = (
    'local',
    'attention_local',
    'attention_local_was',
    'definet_local',
    'definet_local_was',
)
LOCAL_CUTOFF_SWEEP_MODES = ('hetero', 'hetero_was')
DEFINET_MODES = ('attention', 'attention_local', 'attention_was', 'attention_local_was')
WAS_ABLATION_MODELS = ('cgcnn', 'megnet')
WAS_ABLATION_MODES = (
    'was',
    'hetero_was',
)
ATTENTION_ABLATION_MODELS = ('cgcnn', 'megnet', 'definet')
ATTENTION_ABLATION_MODES = (
    'attention_local',
    'attention_was',
    'attention_local_was',
)
CGCNN_DEFAULT_MODES = [
    'full',
    'hetero',
    'local',
    'attention',
    'was',
    'hetero_was',
    'attention_local',
    'attention_was',
    'attention_local_was',
    'definet',
    'definet_local',
    'definet_was',
    'definet_local_was',
]
MEGNET_DEFAULT_MODES = [
    'full',
    'hetero',
    'local',
    'attention',
    'was',
    'hetero_was',
    'attention_local',
    'attention_was',
    'attention_local_was',
]


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def with_radius(config, radius, update_graph_cutoff=False):
    config = copy.deepcopy(config)
    if update_graph_cutoff:
        config['model']['cutoff'] = radius
    config['model']['local_radius'] = radius
    return config


def radius_summary(mode, config):
    radius = config['model']['local_radius']
    if mode in LOCAL_GRAPH_SWEEP_MODES:
        return f'local_radius = cutoff = {radius}'
    return f'local_cutoff = {radius} (local/host boundary), graph cutoff = {config["model"]["cutoff"]}'


def subset_by_indices(values, indices):
    return [values[int(idx)] for idx in indices]


def iter_train_val_test_splits(data, targets, random_seeds, cv5=False):
    if cv5:
        if len(random_seeds) != 1:
            raise ValueError('5-fold cross validation requires exactly one random state in --seeds')
        if len(data) < 5:
            raise ValueError('5-fold cross validation requires at least 5 valid structures')

        random_state = random_seeds[0]
        splitter = KFold(n_splits=5, shuffle=True, random_state=random_state)
        folds = [test_idx for _, test_idx in splitter.split(np.arange(len(data)))]
        for fold_idx, test_idx in enumerate(folds):
            val_fold_idx = (fold_idx + 1) % len(folds)
            val_idx = folds[val_fold_idx]
            train_idx = np.concatenate([
                fold
                for idx, fold in enumerate(folds)
                if idx not in (fold_idx, val_fold_idx)
            ])
            yield {
                'display': f'seed={random_state} fold={fold_idx + 1}/5',
                'logger_id': f'{random_state}_fold{fold_idx + 1}',
                'explain_id': f'seed_{random_state}_fold_{fold_idx + 1}',
                'train_X': subset_by_indices(data, train_idx),
                'train_y': subset_by_indices(targets, train_idx),
                'val_X': subset_by_indices(data, val_idx),
                'val_y': subset_by_indices(targets, val_idx),
                'test_X': subset_by_indices(data, test_idx),
                'test_y': subset_by_indices(targets, test_idx),
            }
        return

    for rs in random_seeds:
        train_X, test_X, train_y, test_y = train_test_split(data, targets, test_size=0.4, random_state=rs)
        val_X, test_X, val_y, test_y = train_test_split(test_X, test_y, test_size=0.5, random_state=rs)
        yield {
            'display': f'seed={rs}',
            'logger_id': rs,
            'explain_id': f'seed_{rs}',
            'train_X': train_X,
            'train_y': train_y,
            'val_X': val_X,
            'val_y': val_y,
            'test_X': test_X,
            'test_y': test_y,
        }


def train_single_mode(mode, config, dataset, targets, random_seeds, epochs, device,
                      model_name, dataset_name, log_dir='logs', explain_options=None,
                      run_label=None, cv5=False):
    """Train a single mode and return per-split test losses."""
    mode_to_dataset_key = {
        'full': 0,      # dataset_full
        'hetero': 1,    # dataset_hetero
        'local': 1,     # dataset_hetero, cropped by SimpleCrystalConverter at conversion time
        'attention': 2, # dataset_attn
        'was': 0,       # dataset_full with current+reference atom embeddings
        'hetero_was': 1, # dataset_hetero with current+reference atom embeddings
        'attention_local': 2, # dataset_attn, cropped by SimpleCrystalConverter
        'attention_was': 2, # dataset_attn with current+reference atom embeddings
        'attention_local_was': 2, # dataset_attn with local crop and WAS features
        'definet': 2,
        'definet_local': 2,
        'definet_was': 2,
        'definet_local_was': 2,
    }
    log_mode = run_label or mode
    data = dataset[mode_to_dataset_key[mode]]
    data_targets = [(s, y) for s, y in zip(data, targets) if s is not None]
    if not data_targets:
        raise ValueError(f'No valid structures found for mode {mode}')
    data, targets = zip(*data_targets)
    targets = torch.stack(list(targets)) if isinstance(data_targets[0][1], torch.Tensor) else list(targets)
    losses = []

    for split in iter_train_val_test_splits(data, targets, random_seeds, cv5=cv5):
        trainer = MEGNetTrainer(config, device)
        trainer.prepare_data(
            split['train_X'],
            split['train_y'],
            split['val_X'],
            split['val_y'],
            'formation_energy',
        )

        logger = TrainingLogger(log_dir, model_name, dataset_name, log_mode, split['logger_id'])

        min_loss = 1e8
        model_best = None
        for epoch in range(epochs):
            mae, mse = trainer.train_one_epoch()
            loss = trainer.evaluate_on_test()
            cur_lr = trainer.optimizer.param_groups[0]['lr']
            print(f'  [{split["display"]}] Epoch {epoch + 1}/{epochs}  train_mae={mae:.4f}  val_mae={loss:.4f}')
            if loss < min_loss:
                min_loss = loss
                model_best = copy.deepcopy(trainer.model.state_dict())
            logger.log(epoch + 1, mae, mse, loss, min_loss, cur_lr)

        loss_test = trainer.predict_structures(split['test_X'], split['test_y'], model_best)
        logger.log_test_result(loss_test)
        print(f'  [{split["display"]}] Test MAE: {loss_test:.4f}')

        if explain_options is not None:
            from .explain.batch import explain_trainer_predictions

            explain_dir = os.path.join(explain_options['root_dir'], split['explain_id'])
            print(f'  [{split["display"]}] Explaining {dataset_name}/{log_mode} test predictions -> {explain_dir}')
            summary = explain_trainer_predictions(
                trainer,
                output_dir=explain_dir,
                device=device,
                max_samples=explain_options.get('max_samples'),
                formats=explain_options.get('formats'),
                epochs=explain_options.get('epochs', 100),
                lr=explain_options.get('lr', 0.01),
                cmap=explain_options.get('cmap', 'viridis_r'),
                strict=explain_options.get('strict', False),
            )
            print(
                f'  [{split["display"]}] Explanations saved: '
                f'{summary.succeeded}/{summary.total} ok, {summary.failed} failed '
                f'({summary.index_csv})'
            )
        losses.append(loss_test)

    return losses


def split_run_summary(seeds, cv5):
    if cv5:
        return f'5-fold CV random_state: {seeds[0]}'
    return f'Seeds: {seeds}'


def default_modes_for_model(model_name):
    if model_name == 'definet':
        return list(DEFINET_MODES)
    if model_name == 'cgcnn':
        return list(CGCNN_DEFAULT_MODES)
    if model_name == 'megnet':
        return list(MEGNET_DEFAULT_MODES)
    return ['full', 'hetero', 'local', 'attention']


def validate_modes_for_model(model_name, modes, parser):
    if model_name != 'cgcnn' and any(mode in CGCNN_DEFINET_MODES for mode in modes):
        parser.error('The definet modes are run under --model cgcnn')
    if model_name == 'definet' and any(mode not in DEFINET_MODES for mode in modes):
        parser.error('The definet model only supports --mode attention attention_local attention_was attention_local_was')
    if model_name not in WAS_ABLATION_MODELS and any(mode in WAS_ABLATION_MODES for mode in modes):
        parser.error('The was and hetero_was modes are only supported with --model cgcnn or --model megnet')
    if model_name not in ATTENTION_ABLATION_MODELS and any(mode in ATTENTION_ABLATION_MODES for mode in modes):
        parser.error('The attention ablation modes are only supported with --model cgcnn, --model megnet, or --model definet')


def resolve_modes(requested_modes, model_name, parser):
    if requested_modes is None or requested_modes == ['all']:
        return default_modes_for_model(model_name)
    if 'all' in requested_modes:
        parser.error('--mode all cannot be combined with specific modes')
    validate_modes_for_model(model_name, requested_modes, parser)
    return requested_modes


def parse_radius_values(raw_values, parser):
    if raw_values is None:
        return None
    values = [str(value).lower() for value in raw_values]
    if 'all' in values:
        if len(values) != 1:
            parser.error('--r all cannot be combined with specific radius values')
        return list(LOCAL_CUTOFF_CHOICES)

    radii = []
    for raw_value in values:
        try:
            radius = int(raw_value)
        except ValueError:
            parser.error(f"--r values must be one of {LOCAL_CUTOFF_CHOICES} or 'all'")
        if radius not in LOCAL_CUTOFF_CHOICES:
            parser.error(f"--r values must be one of {LOCAL_CUTOFF_CHOICES} or 'all'")
        radii.append(radius)
    return radii


def write_dataset_summary(model_name, dataset_name, modes, results, epochs, seeds, device, out_dir, cv5=False):
    split_label = 'Folds' if cv5 else 'Seeds'
    summary_lines = [
        f'SUMMARY: {model_name.upper()} on {dataset_name}',
        f'Device: {device} | Epochs: {epochs} | {split_run_summary(seeds, cv5)}',
        '-' * 50,
    ]
    for mode in modes:
        losses = results[mode]
        line = f'  {mode.upper():12s}  Mean={np.mean(losses):.4f}  Std={np.std(losses):.4f}  {split_label}={losses}'
        summary_lines.append(line)

    print(f'\n{"=" * 60}')
    for line in summary_lines:
        print(line)
    print(f'{"=" * 60}')
    print()

    summary_path = os.path.join(out_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines) + '\n')
    print(f'Summary saved to {summary_path}')


def main():
    parser = argparse.ArgumentParser(
        description='Train crystal GNN models (CGCNN / MEGNet) with different graph modes.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--model', required=True, choices=VALID_MODELS + ['all'],
                        help='Model architecture: megnet, cgcnn, definet, or all')
    parser.add_argument('--dataset', required=True, choices=VALID_DATASETS + ['all'],
                        help='Dataset to use, or all for every dataset')
    parser.add_argument('--mode', nargs='+', default=None, choices=VALID_MODES + ['all'],
                        help='Graph mode(s) to train. Use all for all modes supported by each model')
    parser.add_argument('--device', default='cuda:0',
                        help='Torch device (default: cuda:0)')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of training epochs per seed or CV fold (default: 500)')
    parser.add_argument('--seeds', nargs='+', type=int,
                        default=None,
                        help='Random seeds for train/test splits, or one random state for --cv5')
    parser.add_argument('--cv5', '--five-fold-cv', action='store_true',
                        help='Use 5-fold cross validation. Requires exactly one --seeds value.')
    parser.add_argument('--atom-init', default='./HERA/atom_init.json',
                        help='Path to atom_init.json (default: atom_init.json)')
    parser.add_argument('--log-dir', default='logs',
                        help='Directory to save training history CSVs (default: logs)')
    parser.add_argument('--r', nargs='+', default=None,
                        help=('Shared radius values for local graph and local/host cutoff sweeps. '
                              "Use all for 0 3 4 5 6 7"))
    parser.add_argument('--explain', action='store_true',
                        help='Run GNNExplainer after each seed prediction and save batch visualizations')
    parser.add_argument('--explain-dir', default=None,
                        help='Optional root directory for explanations (default: under each mode log directory)')
    parser.add_argument('--explain-max-samples', type=int, default=None,
                        help='Maximum test samples to explain per seed (default: all)')
    parser.add_argument('--explain-epochs', type=int, default=100,
                        help='GNNExplainer optimization epochs per sample (default: 100)')
    parser.add_argument('--explain-lr', type=float, default=0.01,
                        help='GNNExplainer learning rate (default: 0.01)')
    parser.add_argument('--explain-formats', nargs='+', choices=['ovito', 'csv', 'html', 'png'],
                        default=['ovito'],
                        help='Explanation outputs to save (default: ovito extended XYZ)')
    parser.add_argument('--explain-cmap', default='viridis_r',
                        help='Matplotlib colormap for attribution colors (default: viridis_r)')
    parser.add_argument('--explain-strict', action='store_true',
                        help='Stop immediately if any sample explanation fails')

    args = parser.parse_args()
    warnings.filterwarnings('ignore')
    if args.seeds is None:
        args.seeds = [42] if args.cv5 else DEFAULT_SEEDS
    if args.cv5 and len(args.seeds) != 1:
        parser.error('--cv5 requires exactly one --seeds value, e.g. --cv5 --seeds 42')
    args.r = parse_radius_values(args.r, parser)
    set_seed(42)

    model_names = list(ALL_MODEL_SUITES) if args.model == 'all' else [args.model]
    dataset_names = VALID_DATASETS if args.dataset == 'all' else [args.dataset]
    requested_modes = args.mode

    init_elem_embedding(args.atom_init)

    mode_label = 'all' if requested_modes is None or requested_modes == ['all'] else requested_modes
    print(f'=== Model: {args.model} | Dataset: {args.dataset} | Modes: {mode_label} ===')
    print(f'    Device: {args.device} | Epochs: {args.epochs} | {split_run_summary(args.seeds, args.cv5)}')
    print()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(args.log_dir, f'run_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    print(f'Run directory: {run_dir}\n')

    all_results = {}
    for model_name in model_names:
        modes = resolve_modes(requested_modes, model_name, parser)
        model_dir = os.path.join(run_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)

        print(f'\n{"#" * 60}')
        print(f'  Model: {model_name.upper()} | Modes: {modes}')
        print(f'{"#" * 60}')

        model_results = {}
        for dataset_name in dataset_names:
            dataset_dir = os.path.join(model_dir, dataset_name)
            os.makedirs(dataset_dir, exist_ok=True)

            print(f'\n{"#" * 60}')
            print(f'  Dataset: {dataset_name}')
            print(f'{"#" * 60}')

            dataset_cache = {}

            def dataset_for_cutoff(local_cutoff=None):
                if local_cutoff not in dataset_cache:
                    dataset_cache[local_cutoff] = load_dataset(
                        dataset_name, model_name, local_cutoff=local_cutoff
                    )
                return dataset_cache[local_cutoff]

            base_dataset = dataset_for_cutoff(None)
            targets = base_dataset[3]

            results = {}
            result_labels = []
            for mode in modes:
                mode_runs = [{
                    'label': mode,
                    'mode': mode,
                    'config': get_config(model_name, dataset_name, mode),
                    'local_cutoff': None,
                    'radius_label': None,
                }]
                if mode in LOCAL_GRAPH_SWEEP_MODES:
                    radii = args.r if args.r is not None else LOCAL_CUTOFF_CHOICES
                    mode_runs = [
                        {
                            'label': f'{mode}_r{radius}',
                            'mode': mode,
                            'config': with_radius(
                                get_config(model_name, dataset_name, mode),
                                radius,
                                update_graph_cutoff=True,
                            ),
                            'local_cutoff': None,
                            'radius_label': f'r{radius}',
                        }
                        for radius in radii
                    ]
                elif mode in LOCAL_CUTOFF_SWEEP_MODES:
                    radii = args.r if args.r is not None else LOCAL_CUTOFF_CHOICES
                    mode_runs = [
                        {
                            'label': f'{mode}_r{radius}',
                            'mode': mode,
                            'config': with_radius(
                                get_config(model_name, dataset_name, mode),
                                radius,
                                update_graph_cutoff=False,
                            ),
                            'local_cutoff': radius,
                            'radius_label': f'r{radius}',
                        }
                        for radius in radii
                    ]

                for run in mode_runs:
                    run_label = run['label']
                    train_mode = run['mode']
                    config = run['config']
                    run_dataset = (
                        dataset_for_cutoff(run['local_cutoff'])
                        if run['local_cutoff'] is not None
                        else base_dataset
                    )
                    result_labels.append(run_label)

                    print(f'\n{"=" * 60}')
                    print(f'  Training {model_name.upper()} - {run_label.upper()} mode')
                    if mode in LOCAL_GRAPH_SWEEP_MODES + LOCAL_CUTOFF_SWEEP_MODES:
                        print(f'  {radius_summary(mode, config)}')
                    print(f'{"=" * 60}')
                    mode_parts = [dataset_dir, train_mode]
                    if run['radius_label'] is not None:
                        mode_parts.append(run['radius_label'])
                    mode_dir = os.path.join(*mode_parts)
                    os.makedirs(mode_dir, exist_ok=True)
                    explain_options = None
                    if args.explain:
                        if args.explain_dir is not None:
                            explain_parts = [args.explain_dir, model_name, dataset_name, train_mode]
                            if run['radius_label'] is not None:
                                explain_parts.append(run['radius_label'])
                            explain_root = os.path.join(*explain_parts)
                        else:
                            explain_root = os.path.join(mode_dir, 'explanations')
                        explain_options = {
                            'root_dir': explain_root,
                            'max_samples': args.explain_max_samples,
                            'epochs': args.explain_epochs,
                            'lr': args.explain_lr,
                            'formats': args.explain_formats,
                            'cmap': args.explain_cmap,
                            'strict': args.explain_strict,
                        }
                    losses = train_single_mode(
                        train_mode, config, run_dataset[:3], targets, args.seeds, args.epochs, args.device,
                        model_name=model_name, dataset_name=dataset_name,
                        log_dir=mode_dir, explain_options=explain_options,
                        run_label=run_label, cv5=args.cv5,
                    )
                    results[run_label] = losses

                    split_label = 'Per-fold losses' if args.cv5 else 'Per-seed losses'
                    mode_summary = [
                        f'{model_name.upper()} | {dataset_name} | {run_label.upper()}',
                        f'Epochs: {args.epochs} | {split_run_summary(args.seeds, args.cv5)}',
                        f'Mean={np.mean(losses):.4f}  Std={np.std(losses):.4f}',
                        f'{split_label}: {losses}',
                    ]
                    if mode in LOCAL_GRAPH_SWEEP_MODES + LOCAL_CUTOFF_SWEEP_MODES:
                        mode_summary.insert(1, radius_summary(mode, config))
                    mode_summary_path = os.path.join(mode_dir, 'summary.txt')
                    with open(mode_summary_path, 'w') as f:
                        f.write('\n'.join(mode_summary) + '\n')

            model_results[dataset_name] = results
            all_results[(model_name, dataset_name)] = results
            write_dataset_summary(
                model_name, dataset_name, result_labels, results,
                args.epochs, args.seeds, args.device, dataset_dir, cv5=args.cv5,
            )

        if len(dataset_names) > 1:
            summary_lines = [
                f'SUMMARY: {model_name.upper()} on ALL DATASETS',
                f'Device: {args.device} | Epochs: {args.epochs} | {split_run_summary(args.seeds, args.cv5)}',
                '-' * 50,
            ]
            split_label = 'Folds' if args.cv5 else 'Seeds'
            for dataset_name, results in model_results.items():
                for mode, losses in results.items():
                    line = (
                        f'  {dataset_name:12s} {mode.upper():12s} '
                        f'Mean={np.mean(losses):.4f}  Std={np.std(losses):.4f}  {split_label}={losses}'
                    )
                    summary_lines.append(line)

            summary_path = os.path.join(model_dir, 'summary.txt')
            with open(summary_path, 'w') as f:
                f.write('\n'.join(summary_lines) + '\n')
            print(f'All-dataset summary saved to {summary_path}')

    if len(model_names) > 1:
        summary_lines = [
            'SUMMARY: ALL MODELS',
            f'Device: {args.device} | Epochs: {args.epochs} | {split_run_summary(args.seeds, args.cv5)}',
            '-' * 50,
        ]
        split_label = 'Folds' if args.cv5 else 'Seeds'
        for (model_name, dataset_name), results in all_results.items():
            for mode, losses in results.items():
                line = (
                    f'  {model_name:6s} {dataset_name:12s} {mode.upper():12s} '
                    f'Mean={np.mean(losses):.4f}  Std={np.std(losses):.4f}  {split_label}={losses}'
                )
                summary_lines.append(line)

        summary_path = os.path.join(run_dir, 'summary.txt')
        with open(summary_path, 'w') as f:
            f.write('\n'.join(summary_lines) + '\n')
        print(f'All-model summary saved to {summary_path}')


if __name__ == '__main__':
    main()
