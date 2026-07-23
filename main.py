#!/usr/bin/env python
"""
Crystal Graph Neural Network Training CLI
==========================================

Usage examples:

  # Train MEGNet on vacancy dataset with the default modes
  python -m HERA.main --model megnet --dataset vacancy

  # Run every configured dataset/mode/radius for MEGNet, CGCNN, and ALIGNN
  python -m HERA.main --model all --dataset all --mode all --r all

  # Custom device, epochs, and random seeds
  python -m HERA.main --model cgcnn --dataset native --device cuda:1 --epochs 300 --seeds 42 123

  # Resume an existing run; completed mode summaries are skipped
  python -m HERA.main --model cgcnn --dataset native --mode hetero --r 0 --resume --run-dir logs/run_YYYYMMDD_HHMMSS

Supported combinations:
  Models  : megnet, cgcnn, definet, alignn, all
  Modes   : full, full_x, hetero, hetero_fixed_pool, attention,
            was_x, hetero_was, attention_was, definet, definet_was, all
  Datasets: vacancy, 2dmd_high, native, och, imp2d, semi, all
"""

import argparse
import ast
import copy
import os
import random
import warnings
from datetime import datetime

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from sklearn.model_selection import KFold, train_test_split

from .config.defaults import get_config, VALID_DATASETS, VALID_MODELS, VALID_MODES
from .data.datasets import (
    dataset_index_for_mode,
    load_dataset,
    init_elem_embedding,
    representation_for_mode,
)
from .training.trainer import MEGNetTrainer
from .training.history import TrainingLogger


LOCAL_CUTOFF_CHOICES = [0, 3, 4, 5, 6, 7]
DEFAULT_SEEDS = [123, 11, 1245, 34, 42, 80, 13232, 8, 99, 101]
ALL_MODEL_SUITES = ('alignn', 'megnet', 'cgcnn')
DEFINET_HOST_MODELS = ('cgcnn', 'alignn')
CGCNN_DEFINET_MODES = (
    'definet',
    'definet_was',
)
LOCAL_GRAPH_SWEEP_MODES = ()
LOCAL_CUTOFF_SWEEP_MODES = (
    'hetero', 'hetero_fixed_pool', 'hetero_was'
)
FIXED_POOL_MODES = ('hetero_fixed_pool',)
DEFINET_MODES = ('attention', 'attention_was')
ALIGNN_MODES = (
    'full',
    'full_x',
    'hetero',
    'hetero_fixed_pool',
    'attention',
    'was_x',
    'hetero_was',
    'attention_was',
    'definet',
    'definet_was',
)
WAS_ABLATION_MODELS = ('cgcnn', 'megnet', 'alignn')
WAS_ABLATION_MODES = (
    'was_x',
    'hetero_was',
)
ATTENTION_ABLATION_MODELS = ('cgcnn', 'megnet', 'definet', 'alignn')
ATTENTION_ABLATION_MODES = (
    'attention_was',
)
FULL_X_DISTINCT_DATASETS = frozenset(('vacancy', '2dmd_high', 'native'))
CGCNN_DEFAULT_MODES = [
    'full',
    'full_x',
    'hetero',
    'hetero_fixed_pool',
    'attention',
    'was_x',
    'hetero_was',
    'attention_was',
    'definet',
    'definet_was',
]
MEGNET_DEFAULT_MODES = [
    'full',
    'full_x',
    'hetero',
    'hetero_fixed_pool',
    'attention',
    'was_x',
    'hetero_was',
    'attention_was',
]
ALIGNN_DEFAULT_MODES = [
    'hetero',
    'definet',
    'hetero_fixed_pool',
    'hetero_was',
    'definet_was',
    'full',
    'full_x',
    'attention',
    'was_x',
    'attention_was',
]


def set_seed(seed):
    seed = int(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def with_radius(config, radius):
    config = copy.deepcopy(config)
    config['model']['local_radius'] = radius
    return config


def apply_batch_size_overrides(config, args, model_name):
    config = copy.deepcopy(config)
    if args.train_batch_size is not None:
        config['model']['train_batch_size'] = args.train_batch_size
    if args.test_batch_size is not None:
        config['model']['test_batch_size'] = args.test_batch_size
    if model_name == 'alignn':
        if args.alignn_train_batch_size is not None:
            config['model']['train_batch_size'] = args.alignn_train_batch_size
        if args.alignn_test_batch_size is not None:
            config['model']['test_batch_size'] = args.alignn_test_batch_size
        if args.alignn_cutoff is not None:
            config['model']['cutoff'] = args.alignn_cutoff
        if args.alignn_max_neighbors is not None:
            config['model']['max_neighbors'] = args.alignn_max_neighbors
        if args.alignn_embedding_size is not None:
            config['model']['embedding_size'] = args.alignn_embedding_size
        if args.alignn_nblocks is not None:
            config['model']['nblocks'] = args.alignn_nblocks
        if args.alignn_gcn_blocks is not None:
            config['model']['gcn_blocks'] = args.alignn_gcn_blocks
        if args.alignn_angle_embed_size is not None:
            config['model']['angle_embed_size'] = args.alignn_angle_embed_size
        if args.alignn_grad_accum_steps is not None:
            config['optim']['grad_accum_steps'] = args.alignn_grad_accum_steps
        if args.alignn_amp:
            config['optim']['amp'] = True
    return config


def radius_summary(mode, config):
    radius = config['model']['local_radius']
    if mode in LOCAL_GRAPH_SWEEP_MODES:
        return f'local_radius = {radius}, graph cutoff = {config["model"]["cutoff"]}'
    return f'local_cutoff = {radius} (local/host boundary), graph cutoff = {config["model"]["cutoff"]}'


def subset_by_indices(values, indices):
    return [values[int(idx)] for idx in indices]


def cpu_state_dict(model):
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def clear_cuda_cache(device):
    if str(device).startswith('cuda') and torch.cuda.is_available():
        torch.cuda.empty_cache()


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
                'seed': int(random_state) + fold_idx,
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
            'seed': int(rs),
            'train_X': train_X,
            'train_y': train_y,
            'val_X': val_X,
            'val_y': val_y,
            'test_X': test_X,
            'test_y': test_y,
        }


def train_single_mode(mode, config, dataset, targets, random_seeds, epochs, device,
                      model_name, dataset_name, log_dir='logs', explain_options=None,
                      run_label=None, cv5=False, resume=False):
    """Train a single mode and return per-split test losses."""
    log_mode = run_label or mode
    data = dataset[dataset_index_for_mode(mode)]
    if data is None:
        representation = representation_for_mode(mode)
        raise ValueError(f'Dataset representation {representation!r} was not loaded for mode {mode}')
    data_targets = [(s, y) for s, y in zip(data, targets) if s is not None]
    if not data_targets:
        raise ValueError(f'No valid structures found for mode {mode}')
    data, targets = zip(*data_targets)
    targets = torch.stack(list(targets)) if isinstance(data_targets[0][1], torch.Tensor) else list(targets)
    losses = []
    skipped = 0
    trained = 0

    for split in iter_train_val_test_splits(data, targets, random_seeds, cv5=cv5):
        completed_loss = TrainingLogger.completed_test_mae(log_dir, split['logger_id']) if resume else None
        if completed_loss is not None:
            print(
                f'  [{split["display"]}] Resume: completed history found, '
                f'skipping train/test (test_mae={completed_loss:.4f})'
            )
            losses.append(completed_loss)
            skipped += 1
            continue

        set_seed(split['seed'])
        trainer = MEGNetTrainer(config, device, seed=split['seed'])
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
            trainer.step_scheduler(loss)
            cur_lr = trainer.optimizer.param_groups[0]['lr']
            print(f'  [{split["display"]}] Epoch {epoch + 1}/{epochs}  train_mae={mae:.4f}  val_mae={loss:.4f}')
            if loss < min_loss:
                min_loss = loss
                model_best = cpu_state_dict(trainer.model)
            logger.log(epoch + 1, mae, mse, loss, min_loss, cur_lr)

        loss_test = trainer.predict_structures(split['test_X'], split['test_y'], model_best)
        logger.log_test_result(loss_test)
        checkpoint_path = split_checkpoint_path(log_dir, split['logger_id'])
        save_best_checkpoint(
            checkpoint_path,
            trainer,
            model_best,
            config,
            split,
            model_name,
            dataset_name,
            mode,
            log_mode,
            min_loss,
            loss_test,
            epochs,
        )
        print(f'  [{split["display"]}] Best checkpoint saved: {checkpoint_path}')
        print(f'  [{split["display"]}] Test MAE: {loss_test:.4f}')
        trained += 1

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
        del trainer
        del model_best
        clear_cuda_cache(device)

    if resume and skipped:
        total = skipped + trained
        print(f'  Resume: skipped {skipped}/{total} completed split(s) for {log_mode}')

    return losses


def _checkpoint_safe_id(value):
    return str(value).replace(os.sep, '_').replace('/', '_').replace('\\', '_')


def split_checkpoint_path(log_dir, logger_id):
    return os.path.join(log_dir, f'seed{_checkpoint_safe_id(logger_id)}_best_checkpoint.pth')


def _structure_source_metadata(structures):
    return [
        {
            'source_id': getattr(structure, 'source_id', ''),
            'source_name': getattr(structure, 'source_name', ''),
            'source_path': getattr(structure, 'source_path', ''),
        }
        for structure in structures
    ]


def save_best_checkpoint(path, trainer, model_state_dict, config, split,
                         model_name, dataset_name, mode, run_label,
                         best_val_mae, test_mae, epochs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            'model': model_state_dict,
            'scaler': trainer.scaler.state_dict(),
            'config': copy.deepcopy(config),
            'model_name': model_name,
            'dataset_name': dataset_name,
            'mode': mode,
            'run_label': run_label,
            'task': config.get('task'),
            'best_val_mae': float(best_val_mae),
            'test_mae': float(test_mae),
            'epochs': int(epochs),
            'split': {
                'display': split['display'],
                'logger_id': split['logger_id'],
                'seed': int(split['seed']),
                'train_sources': _structure_source_metadata(split['train_X']),
                'val_sources': _structure_source_metadata(split['val_X']),
                'test_sources': _structure_source_metadata(split['test_X']),
            },
        },
        path,
    )


def split_run_summary(seeds, cv5):
    if cv5:
        return f'5-fold CV random_state: {seeds[0]}'
    return f'Seeds: {seeds}'


def expected_split_logger_ids(seeds, cv5):
    if cv5:
        return [f'{seeds[0]}_fold{fold_idx + 1}' for fold_idx in range(5)]
    return list(seeds)


def completed_resume_losses(log_dir, seeds, cv5):
    losses = []
    for logger_id in expected_split_logger_ids(seeds, cv5):
        loss = TrainingLogger.completed_test_mae(log_dir, logger_id)
        if loss is None:
            return None
        losses.append(loss)
    return losses


def read_mode_summary_losses(path):
    if not os.path.isfile(path):
        return None

    try:
        with open(path, 'r') as f:
            for line in f:
                if not (line.startswith('Per-seed losses:') or line.startswith('Per-fold losses:')):
                    continue
                raw_losses = line.split(':', 1)[1].strip()
                losses = ast.literal_eval(raw_losses)
                if not isinstance(losses, (list, tuple)):
                    return None
                losses = [float(loss) for loss in losses]
                if all(np.isfinite(losses)):
                    return losses
                return None
    except (OSError, SyntaxError, ValueError, TypeError):
        return None

    return None


def write_mode_summary(path, model_name, dataset_name, run_label, losses,
                       epochs, seeds, config, radius_label=None, cv5=False):
    split_label = 'Per-fold losses' if cv5 else 'Per-seed losses'
    mode_summary = [
        f'{model_name.upper()} | {dataset_name} | {run_label.upper()}',
        f'Epochs: {epochs} | {split_run_summary(seeds, cv5)}',
        f'Mean={np.mean(losses):.4f}  Std={np.std(losses):.4f}',
        f'{split_label}: {losses}',
    ]
    if radius_label is not None:
        mode_summary.insert(1, radius_summary(run_label.rsplit('_r', 1)[0], config))
    with open(path, 'w') as f:
        f.write('\n'.join(mode_summary) + '\n')


def latest_run_dir(log_dir):
    if not os.path.isdir(log_dir):
        return None

    candidates = [
        os.path.join(log_dir, name)
        for name in os.listdir(log_dir)
        if name.startswith('run_') and os.path.isdir(os.path.join(log_dir, name))
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: (os.path.getmtime(path), path))


def default_modes_for_model(model_name):
    if model_name == 'definet':
        return list(DEFINET_MODES)
    if model_name == 'alignn':
        return list(ALIGNN_DEFAULT_MODES)
    if model_name == 'cgcnn':
        return list(CGCNN_DEFAULT_MODES)
    if model_name == 'megnet':
        return list(MEGNET_DEFAULT_MODES)
    return ['full', 'full_x', 'hetero', 'attention']


def validate_modes_for_model(model_name, modes, parser):
    if model_name not in DEFINET_HOST_MODELS and any(mode in CGCNN_DEFINET_MODES for mode in modes):
        parser.error('The definet modes are run under --model cgcnn or --model alignn')
    if model_name == 'definet' and any(mode not in DEFINET_MODES for mode in modes):
        parser.error('The definet model only supports --mode attention attention_was')
    if model_name == 'alignn' and any(mode not in ALIGNN_MODES for mode in modes):
        parser.error(
            'The alignn model supports --mode full full_x hetero '
            'hetero_fixed_pool attention was_x hetero_was attention_was '
            'definet definet_was'
        )
    if model_name not in WAS_ABLATION_MODELS and any(mode in WAS_ABLATION_MODES for mode in modes):
        parser.error('The was_x and hetero_was modes are only supported with --model cgcnn, --model megnet, or --model alignn')
    if model_name not in ATTENTION_ABLATION_MODELS and any(mode in ATTENTION_ABLATION_MODES for mode in modes):
        parser.error('The attention ablation modes are only supported with --model cgcnn, --model megnet, --model definet, or --model alignn')


def resolve_modes(requested_modes, model_name, parser):
    if requested_modes is None or requested_modes == ['all']:
        return default_modes_for_model(model_name)
    if 'all' in requested_modes:
        parser.error('--mode all cannot be combined with specific modes')
    validate_modes_for_model(model_name, requested_modes, parser)
    return requested_modes


def modes_for_dataset(modes, dataset_name):
    """Avoid duplicate full/full_x benchmarks when a dataset has no vacancies."""
    if dataset_name in FULL_X_DISTINCT_DATASETS or not {'full', 'full_x'} <= set(modes):
        return list(modes)
    return [mode for mode in modes if mode != 'full_x']


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
        description='Train crystal GNN models (CGCNN / MEGNet / ALIGNN) with different graph modes.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--model', required=True, choices=VALID_MODELS + ['all'],
                        help='Model architecture: megnet, cgcnn, definet, alignn, or all')
    parser.add_argument('--dataset', required=True, choices=VALID_DATASETS + ['all'],
                        help='Dataset to use, or all for every dataset')
    parser.add_argument('--mode', nargs='+', default=None, choices=VALID_MODES + ['all'],
                        help='Graph mode(s) to train. Use all for all modes supported by each model')
    parser.add_argument('--device', default='cuda:0',
                        help='Torch device (default: cuda:0)')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of training epochs per seed or CV fold (default: 500)')
    parser.add_argument('--train-batch-size', type=int, default=None,
                        help='Override training batch size for every selected run')
    parser.add_argument('--test-batch-size', type=int, default=None,
                        help='Override validation/test batch size for every selected run')
    parser.add_argument('--alignn-train-batch-size', type=int, default=None,
                        help='Override training batch size only for ALIGNN runs')
    parser.add_argument('--alignn-test-batch-size', type=int, default=None,
                        help='Override validation/test batch size only for ALIGNN runs')
    parser.add_argument('--alignn-cutoff', type=float, default=None,
                        help='Override graph edge cutoff only for ALIGNN runs')
    parser.add_argument('--alignn-max-neighbors', type=int, default=None,
                        help='Cap neighbors per atom only for ALIGNN graph construction')
    parser.add_argument('--alignn-embedding-size', type=int, default=None,
                        help='Override hidden dimension only for ALIGNN runs')
    parser.add_argument('--alignn-nblocks', type=int, default=None,
                        help='Override number of ALIGNN blocks only for ALIGNN runs')
    parser.add_argument('--alignn-gcn-blocks', type=int, default=None,
                        help='Override number of post-ALIGNN graph-conv blocks only for ALIGNN runs')
    parser.add_argument('--alignn-angle-embed-size', type=int, default=None,
                        help='Override angle basis size only for ALIGNN runs')
    parser.add_argument('--alignn-grad-accum-steps', type=int, default=None,
                        help='Accumulate ALIGNN gradients over this many micro-batches')
    parser.add_argument('--alignn-amp', action='store_true',
                        help='Use CUDA automatic mixed precision only for ALIGNN runs')
    parser.add_argument('--seeds', nargs='+', type=int,
                        default=None,
                        help='Random seeds for train/test splits, or one random state for --cv5')
    parser.add_argument('--cv5', '--five-fold-cv', action='store_true',
                        help='Use 5-fold cross validation. Requires exactly one --seeds value.')
    parser.add_argument('--atom-init', default='./HERA/atom_init.json',
                        help='Path to atom_init.json (default: atom_init.json)')
    parser.add_argument('--log-dir', default='logs',
                        help='Directory to save training history CSVs (default: logs)')
    parser.add_argument('--run-dir', default=None,
                        help='Specific run directory to write/read instead of creating logs/run_{timestamp}')
    parser.add_argument('--resume', action='store_true',
                        help='Skip completed seed/fold tasks whose history CSV already has a TEST result')
    parser.add_argument('--r', nargs='+', default=None,
                        help=('Radius values for hetero local/host cutoff sweeps; '
                              'graph edge cutoff stays at the config value. Use all for 0 3 4 5 6 7'))
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
    for arg_name in (
            'train_batch_size',
            'test_batch_size',
            'alignn_train_batch_size',
            'alignn_test_batch_size',
            'alignn_max_neighbors',
            'alignn_embedding_size',
            'alignn_nblocks',
            'alignn_gcn_blocks',
            'alignn_angle_embed_size',
            'alignn_grad_accum_steps',
    ):
        arg_value = getattr(args, arg_name)
        if arg_value is not None and arg_value < 1:
            parser.error(f'--{arg_name.replace("_", "-")} must be >= 1')
    if args.alignn_cutoff is not None and args.alignn_cutoff <= 0:
        parser.error('--alignn-cutoff must be > 0')
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

    if args.run_dir is not None:
        run_dir = args.run_dir
    elif args.resume:
        run_dir = latest_run_dir(args.log_dir)
        if run_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            run_dir = os.path.join(args.log_dir, f'run_{timestamp}')
            print(f'No existing run_* directory found under {args.log_dir}; starting a new run.')
        else:
            print(f'Resuming latest run directory under {args.log_dir}: {run_dir}')
    else:
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
            dataset_modes = modes_for_dataset(modes, dataset_name)
            dataset_dir = os.path.join(model_dir, dataset_name)
            os.makedirs(dataset_dir, exist_ok=True)

            print(f'\n{"#" * 60}')
            print(f'  Dataset: {dataset_name}')
            print(f'{"#" * 60}')
            if dataset_modes != modes:
                print('  Skipping FULL_X: this dataset has no vacancies, so it is identical to FULL.')

            dataset_cache = {}

            def dataset_for_run(local_cutoff, mode):
                representation = representation_for_mode(mode)
                cache_key = (local_cutoff, representation)
                if cache_key not in dataset_cache:
                    dataset_cache[cache_key] = load_dataset(
                        dataset_name,
                        model_name,
                        local_cutoff=local_cutoff,
                        representations=[representation],
                    )
                return dataset_cache[cache_key]

            results = {}
            result_labels = []
            for mode in dataset_modes:
                def config_for_mode(mode_name):
                    return apply_batch_size_overrides(
                        get_config(model_name, dataset_name, mode_name),
                        args,
                        model_name,
                    )

                mode_runs = [{
                    'label': mode,
                    'mode': mode,
                    'config': config_for_mode(mode),
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
                                config_for_mode(mode),
                                radius,
                            ),
                            'local_cutoff': None,
                            'radius_label': f'r{radius}',
                        }
                        for radius in radii
                    ]
                elif mode in LOCAL_CUTOFF_SWEEP_MODES:
                    radii = args.r if args.r is not None else LOCAL_CUTOFF_CHOICES
                    if mode in FIXED_POOL_MODES:
                        radii = [radius for radius in radii if radius != 0]
                        if not radii:
                            print(
                                f'\nSkipping {model_name.upper()} - {mode.upper()}: '
                                'r=0 is identical to hetero_r0 for fixed pooling.'
                            )
                    mode_runs = [
                        {
                            'label': f'{mode}_r{radius}',
                            'mode': mode,
                            'config': with_radius(
                                config_for_mode(mode),
                                radius,
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
                    result_labels.append(run_label)
                    mode_parts = [dataset_dir, train_mode]
                    if run['radius_label'] is not None:
                        mode_parts.append(run['radius_label'])
                    mode_dir = os.path.join(*mode_parts)
                    os.makedirs(mode_dir, exist_ok=True)
                    mode_summary_path = os.path.join(mode_dir, 'summary.txt')

                    if args.resume:
                        summary_losses = read_mode_summary_losses(mode_summary_path)
                        if summary_losses is not None:
                            print(
                                f'\nResume: summary found for '
                                f'{model_name.upper()} - {run_label.upper()}; '
                                'skipping dataset load/train.'
                            )
                            results[run_label] = summary_losses
                            continue

                        completed_losses = completed_resume_losses(mode_dir, args.seeds, args.cv5)
                        if completed_losses is not None:
                            print(
                                f'\nResume: all completed histories found for '
                                f'{model_name.upper()} - {run_label.upper()}; '
                                'skipping dataset load/train.'
                            )
                            results[run_label] = completed_losses
                            write_mode_summary(
                                mode_summary_path,
                                model_name, dataset_name, run_label, completed_losses,
                                args.epochs, args.seeds, config,
                                radius_label=run['radius_label'], cv5=args.cv5,
                            )
                            continue

                    run_dataset = dataset_for_run(run['local_cutoff'], train_mode)

                    print(f'\n{"=" * 60}')
                    print(f'  Training {model_name.upper()} - {run_label.upper()} mode')
                    print(
                        '  Batch size: '
                        f'train={config["model"]["train_batch_size"]}, '
                        f'val/test={config["model"]["test_batch_size"]}'
                    )
                    print(
                        '  Graph/model: '
                        f'cutoff={config["model"]["cutoff"]}, '
                        f'max_neighbors={config["model"].get("max_neighbors", "all")}, '
                        f'hidden={config["model"]["embedding_size"]}, '
                        f'nblocks={config["model"]["nblocks"]}, '
                        f'gcn_blocks={config["model"].get("gcn_blocks", 0)}, '
                        f'angle_embed={config["model"].get("angle_embed_size", config["model"]["edge_embed_size"])}, '
                        f'grad_accum={config["optim"].get("grad_accum_steps", 1)}, '
                        f'amp={config["optim"].get("amp", False)}'
                    )
                    if mode in LOCAL_GRAPH_SWEEP_MODES + LOCAL_CUTOFF_SWEEP_MODES:
                        print(f'  {radius_summary(mode, config)}')
                    print(f'{"=" * 60}')
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
                        train_mode, config, run_dataset[:3], run_dataset[3], args.seeds, args.epochs, args.device,
                        model_name=model_name, dataset_name=dataset_name,
                        log_dir=mode_dir, explain_options=explain_options,
                        run_label=run_label, cv5=args.cv5, resume=args.resume,
                    )
                    results[run_label] = losses
                    write_mode_summary(
                        mode_summary_path,
                        model_name, dataset_name, run_label, losses,
                        args.epochs, args.seeds, config,
                        radius_label=run['radius_label'], cv5=args.cv5,
                    )

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
