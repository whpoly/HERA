#!/usr/bin/env python
"""
Crystal Graph Neural Network Training CLI
==========================================

Usage examples:

  # Train MEGNet on vacancy dataset with ALL 4 modes (sparse, full, hetero, attention)
  python -m HERA.main --model megnet --dataset vacancy

  # Train CGCNN on 2dmd_high with only the sparse mode
  python -m HERA.main --model cgcnn --dataset 2dmd_high --mode sparse

  # Train MEGNet on semi dataset with sparse + hetero modes
  python -m HERA.main --model megnet --dataset semi --mode sparse hetero

  # Run local radius/cutoff ablation; r=0 is sparse-equivalent local input
  python -m HERA.main --model cgcnn --dataset vacancy --mode local --r 0 3 4 5 6 7

  # Run the DeFiNet-style attention experiment on every dataset
  python -m HERA.main --model definet --dataset all

  # Custom device, epochs, and random seeds
  python -m HERA.main --model cgcnn --dataset native --device cuda:1 --epochs 300 --seeds 42 123

Supported combinations:
  Models  : megnet, cgcnn, definet
  Modes   : sparse, full, hetero, local, attention, was, hetero_was
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
from sklearn.model_selection import train_test_split

from .config.defaults import get_config, VALID_DATASETS, VALID_MODELS, VALID_MODES
from .data.datasets import load_dataset, init_elem_embedding
from .training.trainer import MEGNetTrainer
from .training.history import TrainingLogger


LOCAL_CUTOFF_CHOICES = [0, 3, 4, 5, 6, 7]


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def with_local_cutoff(config, cutoff):
    config = copy.deepcopy(config)
    config['model']['cutoff'] = cutoff
    config['model']['local_radius'] = cutoff
    return config


def train_single_mode(mode, config, dataset, targets, random_seeds, epochs, device,
                      model_name, dataset_name, log_dir='logs', explain_options=None,
                      run_label=None):
    """Train a single mode across multiple random seeds and return per-seed test losses."""
    mode_to_dataset_key = {
        'sparse': 2,    # dataset_sparse
        'full': 0,      # dataset_full
        'hetero': 1,    # dataset_hetero
        'local': 1,     # dataset_hetero, cropped by SimpleCrystalConverter at conversion time
        'attention': 3, # dataset_attn
        'was': 0,       # dataset_full with current+reference atom embeddings
        'hetero_was': 1, # dataset_hetero with current+reference atom embeddings
    }
    log_mode = run_label or mode
    data = dataset[mode_to_dataset_key[mode]]
    data_targets = [(s, y) for s, y in zip(data, targets) if s is not None]
    if not data_targets:
        raise ValueError(f'No valid structures found for mode {mode}')
    data, targets = zip(*data_targets)
    targets = torch.stack(list(targets)) if isinstance(data_targets[0][1], torch.Tensor) else list(targets)
    losses = []

    for rs in random_seeds:
        train_X, test_X, train_y, test_y = train_test_split(data, targets, test_size=0.4, random_state=rs)
        val_X, test_X, val_y, test_y = train_test_split(test_X, test_y, test_size=0.5, random_state=rs)

        trainer = MEGNetTrainer(config, device)
        trainer.prepare_data(train_X, train_y, val_X, val_y, 'formation_energy')

        logger = TrainingLogger(log_dir, model_name, dataset_name, log_mode, rs)

        min_loss = 1e8
        model_best = None
        for epoch in range(epochs):
            mae, mse = trainer.train_one_epoch()
            loss = trainer.evaluate_on_test()
            cur_lr = trainer.optimizer.param_groups[0]['lr']
            print(f'  [seed={rs}] Epoch {epoch + 1}/{epochs}  train_mae={mae:.4f}  val_mae={loss:.4f}')
            if loss < min_loss:
                min_loss = loss
                model_best = copy.deepcopy(trainer.model.state_dict())
            logger.log(epoch + 1, mae, mse, loss, min_loss, cur_lr)

        loss_test = trainer.predict_structures(test_X, test_y, model_best)
        logger.log_test_result(loss_test)
        print(f'  [seed={rs}] Test MAE: {loss_test:.4f}')

        if explain_options is not None:
            from .explain.batch import explain_trainer_predictions

            explain_dir = os.path.join(explain_options['root_dir'], f'seed_{rs}')
            print(f'  [seed={rs}] Explaining {dataset_name}/{log_mode} test predictions -> {explain_dir}')
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
                f'  [seed={rs}] Explanations saved: '
                f'{summary.succeeded}/{summary.total} ok, {summary.failed} failed '
                f'({summary.index_csv})'
            )
        losses.append(loss_test)

    return losses


def write_dataset_summary(model_name, dataset_name, modes, results, epochs, seeds, device, out_dir):
    summary_lines = [
        f'SUMMARY: {model_name.upper()} on {dataset_name}',
        f'Device: {device} | Epochs: {epochs} | Seeds: {seeds}',
        '-' * 50,
    ]
    for mode in modes:
        losses = results[mode]
        line = f'  {mode.upper():12s}  Mean={np.mean(losses):.4f}  Std={np.std(losses):.4f}  Seeds={losses}'
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
    parser.add_argument('--model', required=True, choices=VALID_MODELS,
                        help='Model architecture: megnet, cgcnn, or definet')
    parser.add_argument('--dataset', required=True, choices=VALID_DATASETS + ['all'],
                        help='Dataset to use, or all for every dataset')
    parser.add_argument('--mode', nargs='+', default=None, choices=VALID_MODES,
                        help='Graph mode(s) to train. Default: all modes, or attention for definet')
    parser.add_argument('--device', default='cuda:0',
                        help='Torch device (default: cuda:0)')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of training epochs per seed (default: 500)')
    parser.add_argument('--seeds', nargs='+', type=int,
                        default=[123, 11, 1245, 34, 42, 80, 13232, 8, 99, 101],
                        help='Random seeds for train/test splits')
    parser.add_argument('--atom-init', default='./HERA/atom_init.json',
                        help='Path to atom_init.json (default: atom_init.json)')
    parser.add_argument('--log-dir', default='logs',
                        help='Directory to save training history CSVs (default: logs)')
    parser.add_argument('--r', nargs='+', type=int, choices=LOCAL_CUTOFF_CHOICES, default=None,
                        help='Local radius/cutoff values for --mode local. Default: 0 3 4 5 6 7')
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
    parser.add_argument('--explain-formats', nargs='+', choices=['csv', 'html', 'png'],
                        default=['csv', 'html', 'png'],
                        help='Explanation outputs to save (default: csv html png)')
    parser.add_argument('--explain-cmap', default='viridis_r',
                        help='Matplotlib colormap for attribution colors (default: viridis_r)')
    parser.add_argument('--explain-strict', action='store_true',
                        help='Stop immediately if any sample explanation fails')

    args = parser.parse_args()
    warnings.filterwarnings('ignore')
    set_seed(42)

    dataset_names = VALID_DATASETS if args.dataset == 'all' else [args.dataset]
    modes = args.mode
    if modes is None:
        if args.model == 'definet':
            modes = ['attention']
        elif args.model == 'cgcnn':
            modes = VALID_MODES
        else:
            modes = ['sparse', 'full', 'hetero', 'attention']
    if args.model == 'definet' and any(mode != 'attention' for mode in modes):
        parser.error('The definet model is the paper-style attention experiment and only supports --mode attention')
    if args.model != 'cgcnn' and any(mode in ('was', 'hetero_was') for mode in modes):
        parser.error('The was and hetero_was modes are only supported with --model cgcnn')

    init_elem_embedding(args.atom_init)

    print(f'=== Model: {args.model} | Dataset: {args.dataset} | Modes: {modes} ===')
    print(f'    Device: {args.device} | Epochs: {args.epochs} | Seeds: {args.seeds}')
    print()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    dataset_label = 'all' if len(dataset_names) > 1 else dataset_names[0]
    run_dir = os.path.join(args.log_dir, f'{args.model}_{dataset_label}_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    print(f'Run directory: {run_dir}\n')

    all_results = {}
    for dataset_name in dataset_names:
        dataset_dir = os.path.join(run_dir, dataset_name) if len(dataset_names) > 1 else run_dir
        os.makedirs(dataset_dir, exist_ok=True)

        print(f'\n{"#" * 60}')
        print(f'  Dataset: {dataset_name}')
        print(f'{"#" * 60}')

        dataset_full, dataset_hetero, dataset_sparse, dataset_attn, targets = load_dataset(
            dataset_name, args.model
        )
        dataset = (dataset_full, dataset_hetero, dataset_sparse, dataset_attn)

        results = {}
        result_labels = []
        for mode in modes:
            mode_runs = [(mode, get_config(args.model, dataset_name, mode))]
            if mode == 'local':
                radii = args.r if args.r is not None else LOCAL_CUTOFF_CHOICES
                mode_runs = [
                    (f'local_r{radius}', with_local_cutoff(get_config(args.model, dataset_name, 'local'), radius))
                    for radius in radii
                ]

            for run_label, config in mode_runs:
                train_mode = 'local' if mode == 'local' else mode
                result_labels.append(run_label)

                print(f'\n{"=" * 60}')
                print(f'  Training {args.model.upper()} - {run_label.upper()} mode')
                if mode == 'local':
                    print(f'  local_radius = cutoff = {config["model"]["cutoff"]}')
                print(f'{"=" * 60}')
                mode_dir = os.path.join(dataset_dir, run_label)
                os.makedirs(mode_dir, exist_ok=True)
                explain_options = None
                if args.explain:
                    explain_root = (
                        os.path.join(args.explain_dir, dataset_name, run_label)
                        if args.explain_dir is not None
                        else os.path.join(mode_dir, 'explanations')
                    )
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
                    train_mode, config, dataset, targets, args.seeds, args.epochs, args.device,
                    model_name=args.model, dataset_name=dataset_name,
                    log_dir=mode_dir, explain_options=explain_options,
                    run_label=run_label,
                )
                results[run_label] = losses

                mode_summary = [
                    f'{args.model.upper()} | {dataset_name} | {run_label.upper()}',
                    f'Epochs: {args.epochs} | Seeds: {args.seeds}',
                    f'Mean={np.mean(losses):.4f}  Std={np.std(losses):.4f}',
                    f'Per-seed losses: {losses}',
                ]
                if mode == 'local':
                    mode_summary.insert(1, f'local_radius = cutoff = {config["model"]["cutoff"]}')
                mode_summary_path = os.path.join(mode_dir, 'summary.txt')
                with open(mode_summary_path, 'w') as f:
                    f.write('\n'.join(mode_summary) + '\n')

        all_results[dataset_name] = results
        write_dataset_summary(
            args.model, dataset_name, result_labels, results,
            args.epochs, args.seeds, args.device, dataset_dir,
        )

    if len(dataset_names) > 1:
        summary_lines = [
            f'SUMMARY: {args.model.upper()} on ALL DATASETS',
            f'Device: {args.device} | Epochs: {args.epochs} | Seeds: {args.seeds}',
            '-' * 50,
        ]
        for dataset_name, results in all_results.items():
            for mode, losses in results.items():
                line = (
                    f'  {dataset_name:12s} {mode.upper():12s} '
                    f'Mean={np.mean(losses):.4f}  Std={np.std(losses):.4f}  Seeds={losses}'
                )
                summary_lines.append(line)

        summary_path = os.path.join(run_dir, 'summary.txt')
        with open(summary_path, 'w') as f:
            f.write('\n'.join(summary_lines) + '\n')
        print(f'All-dataset summary saved to {summary_path}')


if __name__ == '__main__':
    main()
