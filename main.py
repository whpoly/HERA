#!/usr/bin/env python
"""
Crystal Graph Neural Network Training CLI
==========================================

Usage examples:

  # Train MEGNet on vacancy dataset with ALL 4 modes (sparse, full, hetero, attention)
  python -m test.main --model megnet --dataset vacancy

  # Train CGCNN on 2dmd_high with only the sparse mode
  python -m test.main --model cgcnn --dataset 2dmd_high --mode sparse

  # Train MEGNet on semi dataset with sparse + hetero modes
  python -m test.main --model megnet --dataset semi --mode sparse hetero

  # Custom device, epochs, and random seeds
  python -m test.main --model cgcnn --dataset native --device cuda:1 --epochs 300 --seeds 42 123

Supported combinations:
  Models  : megnet, cgcnn
  Modes   : sparse, full, hetero, attention
  Datasets: vacancy, 2dmd_high, native, och, imp2d, semi
"""

import argparse
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


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_single_mode(mode, config, dataset, targets, random_seeds, epochs, device,
                      model_name, dataset_name, log_dir='logs'):
    """Train a single mode across multiple random seeds and return per-seed test losses."""
    mode_to_dataset_key = {
        'sparse': 2,    # dataset_sparse
        'full': 0,      # dataset_full
        'hetero': 1,    # dataset_hetero
        'attention': 3, # dataset_attn
    }
    data = dataset[mode_to_dataset_key[mode]]
    losses = []

    for rs in random_seeds:
        train_X, test_X, train_y, test_y = train_test_split(data, targets, test_size=0.4, random_state=rs)
        val_X, test_X, val_y, test_y = train_test_split(test_X, test_y, test_size=0.5, random_state=rs)

        trainer = MEGNetTrainer(config, device)
        trainer.prepare_data(train_X, train_y, val_X, val_y, 'formation_energy')

        # Training history logger
        logger = TrainingLogger(log_dir, model_name, dataset_name, mode, rs)

        min_loss = 1e8
        model_best = None
        for epoch in range(epochs):
            mae, mse = trainer.train_one_epoch()
            loss = trainer.evaluate_on_test()
            cur_lr = trainer.optimizer.param_groups[0]['lr']
            print(f'  [seed={rs}] Epoch {epoch+1}/{epochs}  train_mae={mae:.4f}  val_mae={loss:.4f}')
            if loss < min_loss:
                min_loss = loss
                model_best = trainer.model.state_dict()
            logger.log(epoch + 1, mae, mse, loss, min_loss, cur_lr)

        loss_test = trainer.predict_structures(test_X, test_y, model_best)
        logger.log_test_result(loss_test)
        print(f'  [seed={rs}] Test MAE: {loss_test:.4f}')
        losses.append(loss_test)

    return losses


def main():
    parser = argparse.ArgumentParser(
        description='Train crystal GNN models (CGCNN / MEGNet) with different graph modes.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--model', required=True, choices=VALID_MODELS,
                        help='Model architecture: megnet or cgcnn')
    parser.add_argument('--dataset', required=True, choices=VALID_DATASETS,
                        help='Dataset to use')
    parser.add_argument('--mode', nargs='+', default=VALID_MODES, choices=VALID_MODES,
                        help='Graph mode(s) to train. Default: all four modes')
    parser.add_argument('--device', default='cuda:0',
                        help='Torch device (default: cuda:0)')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of training epochs per seed (default: 500)')
    parser.add_argument('--seeds', nargs='+', type=int,
                        default=[123, 11, 1245, 34, 42, 80, 13232, 8, 99, 101],
                        help='Random seeds for train/test splits')
    parser.add_argument('--atom-init', default='atom_init.json',
                        help='Path to atom_init.json (default: atom_init.json)')
    parser.add_argument('--log-dir', default='logs',
                        help='Directory to save training history CSVs (default: logs)')

    args = parser.parse_args()
    warnings.filterwarnings('ignore')
    set_seed(42)

    # Load atom embeddings
    init_elem_embedding(args.atom_init)

    print(f'=== Model: {args.model} | Dataset: {args.dataset} | Modes: {args.mode} ===')
    print(f'    Device: {args.device} | Epochs: {args.epochs} | Seeds: {args.seeds}')
    print()

    # Create a run-specific subfolder: logs/{model}_{dataset}_{timestamp}/
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(args.log_dir, f'{args.model}_{args.dataset}_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    print(f'Run directory: {run_dir}\n')

    # Load all four dataset variants at once
    dataset_full, dataset_hetero, dataset_sparse, dataset_attn, targets = load_dataset(
        args.dataset, args.model
    )
    dataset = (dataset_full, dataset_hetero, dataset_sparse, dataset_attn)

    # Train each requested mode
    results = {}
    for mode in args.mode:
        print(f'\n{"="*60}')
        print(f'  Training {args.model.upper()} — {mode.upper()} mode')
        print(f'{"="*60}')
        config = get_config(args.model, args.dataset, mode)
        mode_dir = os.path.join(run_dir, mode)
        os.makedirs(mode_dir, exist_ok=True)
        losses = train_single_mode(
            mode, config, dataset, targets, args.seeds, args.epochs, args.device,
            model_name=args.model, dataset_name=args.dataset,
            log_dir=mode_dir,
        )
        results[mode] = losses

        # Per-mode summary
        mode_summary = [
            f'{args.model.upper()} | {args.dataset} | {mode.upper()}',
            f'Epochs: {args.epochs} | Seeds: {args.seeds}',
            f'Mean={np.mean(losses):.4f}  Std={np.std(losses):.4f}',
            f'Per-seed losses: {losses}',
        ]
        mode_summary_path = os.path.join(mode_dir, 'summary.txt')
        with open(mode_summary_path, 'w') as f:
            f.write('\n'.join(mode_summary) + '\n')

    # Summary
    summary_lines = []
    summary_lines.append(f'SUMMARY: {args.model.upper()} on {args.dataset}')
    summary_lines.append(f'Device: {args.device} | Epochs: {args.epochs} | Seeds: {args.seeds}')
    summary_lines.append('-' * 50)
    for mode, losses in results.items():
        line = f'  {mode.upper():12s}  Mean={np.mean(losses):.4f}  Std={np.std(losses):.4f}  Seeds={losses}'
        summary_lines.append(line)

    print(f'\n{"="*60}')
    for line in summary_lines:
        print(line)
    print(f'{"="*60}')
    print()

    # Write summary to txt
    summary_path = os.path.join(run_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines) + '\n')
    print(f'Summary saved to {summary_path}')


if __name__ == '__main__':
    main()
