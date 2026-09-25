#!/usr/bin/env python
"""
Crystal Graph Neural Network Training CLI
==========================================

Usage examples:

  # Train MEGNet on vacancy dataset with the default modes
  python -m HERA.main --model megnet --dataset vacancy

  # Reproduce MEGNET_SPARSE with the current training protocol on both datasets
  python -m HERA.main --model megnet --dataset vacancy 2dmd_high --mode sparse --seed 123

  # Run every configured dataset/mode/radius for MEGNet, CGCNN, and ALIGNN
  python -m HERA.main --model all --dataset all --mode all --r all

  # Custom device, epochs, and random seed
  python -m HERA.main --model cgcnn --dataset native --device cuda:1 --epochs 300 --seed 42

  # Run the standard 10-seed benchmark
  python -m HERA.main --model cgcnn --dataset native --seed all

  # Reuse completed seed/fold results in an existing run
  python -m HERA.main --model cgcnn --dataset native --mode hetero --r 0 --resume --run-dir logs/run_YYYYMMDD_HHMMSS

Supported combinations:
  Models  : megnet, cgcnn, definet, alignn, hypergraph, all
  Modes   : sparse, full, full_x, hetero, hetero_fixed_pool, attention,
            was_x, hetero_was, attention_was, definet, definet_was,
            hypergraph, hypergraph_was, all
  Datasets: vacancy, vacancy_mos2, vacancy_wse2, 2dmd_low, 2dmd_high,
            2dmd_mos2, 2dmd_wse2, native, och, imp2d, semi, all
"""

import argparse
import ast
import copy
import csv
import os
import json
import random
import warnings
from datetime import datetime

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from sklearn.model_selection import KFold, train_test_split

from .config.defaults import (
    get_config, VALID_DATASETS, VALID_MODELS, VALID_MODES, HYPERGRAPH_POOLING_MODES,
    apply_hypergraph_options, hypergraph_run_components, resolve_hypergraph_pooling,
    ALIGNN_HETERO_FEATURE_NORMS, ALIGNN_HETERO_POOLING_MODES,
    ALIGNN_HETERO_RELATION_MODES,
    ALIGNN_HETERO_MESSAGE_MODES, ALIGNN_HETERO_DISTANCE_MODES, ALIGNN_HETERO_ABLATIONS,
    ALIGNN_HETERO_AGGREGATION_MODES, ALIGNN_HETERO_DEFECT_RESIDUAL_MODES,
    ALIGNN_HETERO_DEFECT_CONNECTIVITY_MODES,
    ALIGNN_HETERO_AA_MODES,
    ALIGNN_HETERO_DD_MODES,
    apply_alignn_hetero_options, alignn_hetero_run_components,
    HYPERGRAPH_UPDATE_MODES, resolve_hypergraph_updates,
)
from .data.datasets import (
    dataset_index_for_mode,
    load_dataset,
    init_elem_embedding,
    representation_for_mode,
)
from .training.trainer import MEGNetTrainer
from .data.native_was import NATIVE_PREPROCESSING_CHOICES, native_run_components
from .data.native_filter import read_manifest, manifest_identity, requested_splits, filtered_splits
from .data.native_preprocessing import prepare_native_filter
from .data.impurity_preprocessing import (prepare_impurity_filter, identity as impurity_filter_identity,
                                          filtered_splits as impurity_filtered_splits)
from .data.impurity_was import impurity_run_components
from .training.history import TrainingLogger
from .training.results import (
    atomic_write_text, merge_aggregate_summary, merge_mode_losses,
    prefixed_summary_rows, saved_dataset_rows, completed_split_result,
    validate_compact_run_config, ensure_compact_run_config,
    compact_mode_directory,
)


LOCAL_CUTOFF_CHOICES = [0, 3, 4, 5, 6, 7]
DEFAULT_SEED = 123
DEFAULT_SEEDS = [DEFAULT_SEED]
ALL_BENCHMARK_SEEDS = [123, 11, 1245, 34, 42, 80, 13232, 8, 99, 101]
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
HYPERGRAPH_MODES = ('hypergraph', 'hypergraph_was')
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
    'hypergraph',
    'hypergraph_was',
)
WAS_ABLATION_MODELS = ('cgcnn', 'megnet', 'alignn')
WAS_ABLATION_MODES = (
    'was_x',
    'hetero_was',
    'hypergraph_was',
)
ATTENTION_ABLATION_MODELS = ('cgcnn', 'megnet', 'definet', 'alignn')
ATTENTION_ABLATION_MODES = (
    'attention_was',
)
FULL_X_DISTINCT_DATASETS = frozenset(
    (
        'vacancy', 'vacancy_mos2', 'vacancy_wse2',
        '2dmd_low', '2dmd_high', '2dmd_mos2', '2dmd_wse2', 'native',
    )
)
MEGNET_SPARSE_DATASETS = frozenset(
    (
        'vacancy', 'vacancy_mos2', 'vacancy_wse2',
        '2dmd_low', '2dmd_high', '2dmd_mos2', '2dmd_wse2',
    )
)
CONCENTRATION_TRANSFER_DATASETS = frozenset(
    ('vacancy_mos2', 'vacancy_wse2', '2dmd_mos2', '2dmd_wse2')
)
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
    'hypergraph',
    'hypergraph_was',
]
MEGNET_DEFAULT_MODES = [
    'sparse',
    'full',
    'full_x',
    'hetero',
    'hetero_fixed_pool',
    'attention',
    'was_x',
    'hetero_was',
    'attention_was',
    'hypergraph',
    'hypergraph_was',
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
    'hypergraph',
    'hypergraph_was',
]
ALIGNN_NODE_NORM_MODES = frozenset((
    'hetero',
    'hetero_fixed_pool',
    'hetero_was',
))


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


def parse_seed_values(values, parser):
    """Resolve CLI seed tokens to one seed or the standard 10-seed suite."""
    if values is None:
        return [DEFAULT_SEED]
    if not isinstance(values, (list, tuple)):
        values = [values]
    tokens = [str(value).strip() for value in values]
    if any(token.lower() == 'all' for token in tokens):
        if len(tokens) != 1:
            parser.error('--seed all cannot be combined with explicit seed values')
        return list(ALL_BENCHMARK_SEEDS)
    try:
        seeds = [int(token) for token in tokens]
    except ValueError:
        parser.error('--seed must be an integer or all')
    if any(seed < 0 for seed in seeds):
        parser.error('--seed must be non-negative')
    return seeds


def with_radius(config, radius):
    config = copy.deepcopy(config)
    config['model']['local_radius'] = radius
    return config


def expand_alignn_node_norm_runs(mode_runs, norm_values, enabled):
    """Expand HeteroALIGNN run specs into isolated normalization ablations."""
    if not enabled or norm_values is None:
        return mode_runs
    if isinstance(norm_values, str):
        norm_values = [norm_values]
    norm_values = list(dict.fromkeys(norm_values))

    expanded = []
    for run in mode_runs:
        for normalization in norm_values:
            norm_run = copy.deepcopy(run)
            norm_run['label'] = f'{run["label"]}_norm_{normalization}'
            norm_run['norm_label'] = f'norm_{normalization}'
            norm_run['config']['model']['hetero_node_norm'] = normalization
            expanded.append(norm_run)
    return expanded


def expand_alignn_hetero_encoder_runs(mode_runs, ablations, enabled):
    """Run explicitly requested, independent encoder/interaction ablations."""
    if not enabled or ablations is None:
        return mode_runs
    if isinstance(ablations, str):
        ablations = [ablations]
    expanded = []
    for run in mode_runs:
        for ablation in dict.fromkeys(ablations):
            if ablation not in ALIGNN_HETERO_ABLATIONS:
                raise ValueError(f'Unknown hetero encoder ablation: {ablation}')
            variant = copy.deepcopy(run)
            message_mode, distance_mode, aggregation, residual = ALIGNN_HETERO_ABLATIONS[ablation]
            apply_alignn_hetero_options(variant['config'], message_mode=message_mode,
                                       distance_mode=distance_mode, aggregation_mode=aggregation,
                                       defect_residual=residual)
            expanded.append(variant)
    return expanded


def expand_hypergraph_update_runs(mode_runs, update_values, enabled):
    """Change only active hypergraph messages; keep model and graph inputs fixed."""
    if not enabled or update_values is None:
        return mode_runs
    if isinstance(update_values, str):
        update_values = [update_values]
    expanded = []
    for run in mode_runs:
        for updates in dict.fromkeys(update_values):
            update_run = copy.deepcopy(run)
            model = update_run['config']['model']
            pooling = resolve_hypergraph_pooling(model['hypergraph_schema'], model.get('hypergraph_pooling'))
            model['hypergraph_updates'] = resolve_hypergraph_updates(
                model['hypergraph_schema'], pooling, updates,
            )
            update_run['label'] += f'_updates_{updates}'
            expanded.append(update_run)
    return expanded


def apply_training_overrides(config, args, model_name):
    config = copy.deepcopy(config)
    if args.train_batch_size is not None:
        config['model']['train_batch_size'] = args.train_batch_size
    if args.test_batch_size is not None:
        config['model']['test_batch_size'] = args.test_batch_size
    if args.early_stopping_patience is not None:
        config['optim']['early_stopping_patience'] = args.early_stopping_patience
    if args.early_stopping_min_delta_percent is not None:
        config['optim']['early_stopping_min_delta_percent'] = (
            args.early_stopping_min_delta_percent
        )
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
        if args.alignn_hetero_node_norm is not None:
            norm_values = args.alignn_hetero_node_norm
            if isinstance(norm_values, str):
                norm_values = [norm_values]
            if len(norm_values) == 1:
                config['model']['hetero_node_norm'] = norm_values[0]
        if args.alignn_amp:
            config['optim']['amp'] = True
        apply_alignn_hetero_options(
            config, getattr(args, 'alignn_hetero_feature_norm', None),
            getattr(args, 'alignn_hetero_pooling', None),
            getattr(args, 'alignn_hetero_relations', None),
            getattr(args, 'alignn_hetero_adapter_rank', None),
            message_mode=getattr(args, 'alignn_hetero_message', None),
            distance_mode=getattr(args, 'alignn_hetero_distance', None),
            aggregation_mode=getattr(args, 'alignn_hetero_aggregation', None),
            defect_residual=getattr(args, 'alignn_hetero_defect_residual', None),
            defect_cutoff=getattr(args, 'alignn_hetero_defect_cutoff', None),
            defect_connectivity=getattr(args, 'alignn_hetero_defect_connectivity', None),
            aa_mode=getattr(args, 'alignn_hetero_aa', None),
            dd_mode=getattr(args, 'alignn_hetero_dd', None),
        )
    if config['task'].endswith(('_hypergraph', '_hypergraph_was')) and args.hypergraph_radius is not None:
        config['model']['hypergraph_radius'] = args.hypergraph_radius
    if config['task'].endswith(('_hypergraph', '_hypergraph_was')):
        apply_hypergraph_options(config, getattr(args, 'hypergraph_schema', None),
                                 getattr(args, 'hypergraph_pooling', None))
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


def is_meaningful_relative_improvement(value, reference, min_delta_percent):
    """Return whether a non-negative metric improved by the requested percent."""
    if not np.isfinite(reference):
        return True
    return value < reference * (1.0 - min_delta_percent / 100.0)


def clear_cuda_cache(device):
    if str(device).startswith('cuda') and torch.cuda.is_available():
        torch.cuda.empty_cache()


def iter_train_val_test_splits(data, targets, random_seeds, cv5=False):
    if cv5:
        if len(random_seeds) != 1:
            raise ValueError('5-fold cross validation requires exactly one --seed value')
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


def iter_concentration_transfer_splits(
        data,
        targets,
        random_seeds,
        cv5=False,
):
    """Split low-concentration samples into train/val and keep high fixed as test."""
    concentrations = [getattr(structure, 'concentration', None) for structure in data]
    unknown = [idx for idx, value in enumerate(concentrations) if value not in ('low', 'high')]
    if unknown:
        raise ValueError(
            'Concentration-transfer datasets require every structure to be tagged '
            f"as 'low' or 'high'; missing/invalid tags at indices {unknown[:5]}"
        )

    low_indices = np.asarray([
        idx for idx, value in enumerate(concentrations) if value == 'low'
    ], dtype=int)
    high_indices = np.asarray([
        idx for idx, value in enumerate(concentrations) if value == 'high'
    ], dtype=int)
    if len(low_indices) < 2:
        raise ValueError('Concentration transfer requires at least 2 low-concentration samples')
    if len(high_indices) == 0:
        raise ValueError('Concentration transfer requires at least 1 high-concentration test sample')

    fixed_test_X = subset_by_indices(data, high_indices)
    fixed_test_y = subset_by_indices(targets, high_indices)
    if cv5:
        if len(random_seeds) != 1:
            raise ValueError('5-fold cross validation requires exactly one --seed value')
        if len(low_indices) < 5:
            raise ValueError(
                '5-fold concentration transfer requires at least 5 low-concentration samples'
            )
        random_state = random_seeds[0]
        splitter = KFold(n_splits=5, shuffle=True, random_state=random_state)
        for fold_idx, (train_positions, val_positions) in enumerate(
                splitter.split(low_indices)
        ):
            train_indices = low_indices[train_positions]
            val_indices = low_indices[val_positions]
            yield {
                'display': f'seed={random_state} fold={fold_idx + 1}/5',
                'logger_id': f'{random_state}_fold{fold_idx + 1}',
                'explain_id': f'seed_{random_state}_fold_{fold_idx + 1}',
                'seed': int(random_state) + fold_idx,
                'train_X': subset_by_indices(data, train_indices),
                'train_y': subset_by_indices(targets, train_indices),
                'val_X': subset_by_indices(data, val_indices),
                'val_y': subset_by_indices(targets, val_indices),
                'test_X': fixed_test_X,
                'test_y': fixed_test_y,
            }
        return

    for rs in random_seeds:
        train_indices, val_indices = train_test_split(
            low_indices,
            test_size=0.2,
            random_state=rs,
        )
        yield {
            'display': f'seed={rs}',
            'logger_id': rs,
            'explain_id': f'seed_{rs}',
            'seed': int(rs),
            'train_X': subset_by_indices(data, train_indices),
            'train_y': subset_by_indices(targets, train_indices),
            'val_X': subset_by_indices(data, val_indices),
            'val_y': subset_by_indices(targets, val_indices),
            'test_X': fixed_test_X,
            'test_y': fixed_test_y,
        }


def train_single_mode(mode, config, dataset, targets, random_seeds, epochs, device,
                      model_name, dataset_name, log_dir='logs', explain_options=None,
                      run_label=None, cv5=False, resume=False, protect_existing=False,
                      native_filter_manifest=None, impurity_filter_manifest=None):
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

    if config.get('native_data_filter') is not None:
        if dataset_name != 'native' or native_filter_manifest is None:
            raise ValueError('Filtered native training requires its frozen preprocessing manifest')
        if config['native_data_filter'] != manifest_identity(native_filter_manifest):
            raise ValueError('Native filter identity differs from the experiment configuration')
        splits = filtered_splits(data, targets, native_filter_manifest, random_seeds, cv5=cv5)
    elif native_filter_manifest is not None:
        raise ValueError('Native filter manifest must be recorded in the experiment configuration')
    elif config.get('impurity_data_filter') is not None:
        if (impurity_filter_manifest is None or dataset_name not in ('imp2d', 'semi')
                or impurity_filter_manifest['dataset'] != dataset_name):
            raise ValueError('Filtered impurity training requires its frozen preprocessing manifest')
        if config['impurity_data_filter'] != impurity_filter_identity(impurity_filter_manifest):
            raise ValueError('Impurity filter identity differs from the experiment configuration')
        splits = impurity_filtered_splits(data, targets, impurity_filter_manifest, random_seeds, cv5=cv5)
    elif impurity_filter_manifest is not None:
        raise ValueError('Impurity filter must be recorded in the experiment configuration')
    elif dataset_name in CONCENTRATION_TRANSFER_DATASETS:
        low_count = sum(
            getattr(structure, 'concentration', None) == 'low'
            for structure in data
        )
        high_count = sum(
            getattr(structure, 'concentration', None) == 'high'
            for structure in data
        )
        print(
            '  Concentration transfer: '
            f'low={low_count} (train/validation), '
            f'high={high_count} (fixed test)'
        )
        splits = iter_concentration_transfer_splits(
            data,
            targets,
            random_seeds,
            cv5=cv5,
        )
    else:
        splits = iter_train_val_test_splits(
            data,
            targets,
            random_seeds,
            cv5=cv5,
        )

    for split in splits:
        if 'impurity_filter_counts' in split:
            counts = split['impurity_filter_counts']
            print(f'  [{split["display"]}] {dataset_name} physical filter (split membership preserved): '
                  + ', '.join(f'{part} {counts[part]["original"]}->{counts[part]["kept"]}'
                              for part in ('train', 'val', 'test')))
        if 'native_filter_counts' in split:
            counts = split['native_filter_counts']
            print(f'  [{split["display"]}] Native filter (original split membership preserved): '
                  + ', '.join(f'{part} {counts[part]["original"]}->{counts[part]["kept"]}'
                              for part in ('train', 'val', 'test')))
        completed_loss = completed_split_result(
            log_dir, split['logger_id'],
            expected={'config': config, 'model_name': model_name, 'dataset_name': dataset_name, 'mode': mode},
            protect_existing=protect_existing,
        ) if resume else None
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
        best_epoch = 0
        early_stopping_patience = int(config['optim'].get('early_stopping_patience', 0))
        early_stopping_min_delta_percent = float(
            config['optim'].get('early_stopping_min_delta_percent', 0.0)
        )
        early_stopping_best = float('inf')
        epochs_without_improvement = 0
        epochs_completed = 0
        stopped_early = False
        for epoch in range(epochs):
            mae, mse = trainer.train_one_epoch()
            loss = trainer.evaluate_on_test()
            trainer.step_scheduler(loss)
            cur_lr = trainer.optimizer.param_groups[0]['lr']
            print(f'  [{split["display"]}] Epoch {epoch + 1}/{epochs}  train_mae={mae:.4f}  val_mae={loss:.4f}')
            if loss < min_loss:
                min_loss = loss
                model_best = cpu_state_dict(trainer.model)
                best_epoch = epoch + 1
            if is_meaningful_relative_improvement(
                    loss,
                    early_stopping_best,
                    early_stopping_min_delta_percent,
            ):
                early_stopping_best = loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            logger.log(epoch + 1, mae, mse, loss, min_loss, cur_lr)
            epochs_completed = epoch + 1
            if (
                    early_stopping_patience > 0
                    and epochs_without_improvement >= early_stopping_patience
            ):
                stopped_early = True
                print(
                    f'  [{split["display"]}] Early stopping at epoch '
                    f'{epochs_completed}/{epochs}: no validation MAE improvement '
                    f'> {early_stopping_min_delta_percent:g}% for '
                    f'{early_stopping_patience} epochs '
                    f'(best={min_loss:.4f} at epoch {best_epoch}).'
                )
                break

        if dataset_name in CONCENTRATION_TRANSFER_DATASETS:
            loss_test, predictions = trainer.predict_structures(
                split['test_X'],
                split['test_y'],
                model_best,
                return_predictions=True,
            )
            prediction_path = write_test_predictions(
                log_dir,
                split['logger_id'],
                split['test_X'],
                split['test_y'],
                predictions,
            )
            print(
                f'  [{split["display"]}] High-concentration predictions saved: '
                f'{prediction_path}'
            )
        else:
            loss_test = trainer.predict_structures(
                split['test_X'], split['test_y'], model_best
            )
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
            epochs_completed,
            best_epoch,
            stopped_early,
            early_stopping_patience,
            early_stopping_min_delta_percent,
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


def _scalar_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().reshape(-1)[0].item())
    return float(value)


def write_test_predictions(log_dir, logger_id, structures, targets, predictions):
    """Write per-structure fixed-test predictions for concentration transfer."""
    path = os.path.join(
        log_dir,
        f'seed{_checkpoint_safe_id(logger_id)}_test_predictions.csv',
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = (
        'source_id',
        'source_name',
        'source_path',
        'material',
        'concentration',
        'defect_family',
        'target',
        'prediction',
        'absolute_error',
    )
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for structure, target, prediction in zip(structures, targets, predictions):
            target = _scalar_float(target)
            prediction = _scalar_float(prediction)
            writer.writerow({
                'source_id': getattr(structure, 'source_id', ''),
                'source_name': getattr(structure, 'source_name', ''),
                'source_path': getattr(structure, 'source_path', ''),
                'material': getattr(structure, 'material', ''),
                'concentration': getattr(structure, 'concentration', ''),
                'defect_family': getattr(structure, 'defect_family', ''),
                'target': f'{target:.10g}',
                'prediction': f'{prediction:.10g}',
                'absolute_error': f'{abs(prediction - target):.10g}',
            })
    return path


def _structure_source_metadata(structures):
    return [
        {
            'source_id': getattr(structure, 'source_id', ''),
            'source_name': getattr(structure, 'source_name', ''),
            'source_path': getattr(structure, 'source_path', ''),
            'material': getattr(structure, 'material', ''),
            'concentration': getattr(structure, 'concentration', ''),
            'defect_family': getattr(structure, 'defect_family', ''),
        }
        for structure in structures
    ]


def save_best_checkpoint(path, trainer, model_state_dict, config, split,
                         model_name, dataset_name, mode, run_label,
                         best_val_mae, test_mae, epochs, epochs_completed,
                         best_epoch, stopped_early, early_stopping_patience,
                         early_stopping_min_delta_percent):
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
            'epochs_completed': int(epochs_completed),
            'best_epoch': int(best_epoch),
            'stopped_early': bool(stopped_early),
            'early_stopping_patience': int(early_stopping_patience),
            'early_stopping_min_delta_percent': float(early_stopping_min_delta_percent),
            'split': {
                'display': split['display'],
                'logger_id': split['logger_id'],
                'seed': int(split['seed']),
                'train_sources': _structure_source_metadata(split['train_X']),
                'val_sources': _structure_source_metadata(split['val_X']),
                'test_sources': _structure_source_metadata(split['test_X']),
                **({'native_filter_counts': split['native_filter_counts']}
                   if 'native_filter_counts' in split else {}),
                **({'impurity_filter_counts': split['impurity_filter_counts']}
                   if 'impurity_filter_counts' in split else {}),
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


def completed_resume_losses(log_dir, seeds, cv5, expected=None, protect_existing=False):
    losses = []
    for logger_id in expected_split_logger_ids(seeds, cv5):
        loss = completed_split_result(log_dir, logger_id, expected=expected, protect_existing=protect_existing)
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
    seeds, losses = merge_mode_losses(path, seeds, losses, cv5=cv5)
    split_label = 'Per-fold losses' if cv5 else 'Per-seed losses'
    mode_summary = [
        f'{model_name.upper()} | {dataset_name} | {run_label.upper()}',
        f'Epochs: {epochs} | {split_run_summary(seeds, cv5)}',
        f'Mean={np.mean(losses):.4f}  Std={np.std(losses):.4f}',
        f'{split_label}: {losses}',
    ]
    preprocessing = config.get(f'{dataset_name}_preprocessing')
    if preprocessing:
        mode_summary.insert(1, f'{dataset_name} preprocessing: {preprocessing}; '
                            f'atom features: {config["model"]["atom_features"]}')
    if config['task'].endswith(('_hypergraph', '_hypergraph_was')):
        mode_summary.insert(
            1,
            f'Hypergraph defect-neighbor radius: '
            f'{config["model"]["hypergraph_radius"]} A; pooling: '
            f'{resolve_hypergraph_pooling(config["model"]["hypergraph_schema"], config["model"].get("hypergraph_pooling"))}',
        )
        if 'hypergraph_updates' in config['model']:
            mode_summary.insert(2, f'Hypergraph message updates: {config["model"]["hypergraph_updates"]}')
    elif radius_label is not None:
        mode_summary.insert(1, radius_summary(run_label.rsplit('_r', 1)[0], config))
    if config['task'] in ('alignn_hetero', 'alignn_hetero_was', 'alignn_hetero_fixed_pool'):
        mode_summary.insert(
            1, f'Hetero feature norm: {config["model"].get("hetero_feature_norm", "batchnorm")}; '
            f'node delta norm: {config["model"].get("hetero_node_norm", "layernorm")}; '
            f'pooling: {config["model"].get("hetero_pooling", "type_mean")}; '
            f'relations: {config["model"].get("hetero_relation_mode", "independent")}; '
            f'adapter rank: {config["model"].get("hetero_relation_rank", 8)}; '
            f'message: {config["model"].get("hetero_message_mode", "linear")}; '
            f'distance: {config["model"].get("hetero_distance_mode", "independent")}; '
            f'aggregation: {config["model"].get("hetero_aggregation_mode", "relation_mean")}; '
            f'defect residual: {config["model"].get("hetero_defect_residual", "none")}; '
            f'defect cutoff: {config["model"].get("hetero_defect_cutoff", 12.0)} A; '
            f'defect connectivity: {config["model"].get("hetero_defect_connectivity", "physical")}; '
            f'atom-atom relation: {config["model"].get("hetero_aa_mode", "keep")}; '
            f'defect-defect relation: {config["model"].get("hetero_dd_mode", "keep")}',
        )
    atomic_write_text(path, '\n'.join(mode_summary) + '\n', backup=True)


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
    if model_name == 'hypergraph':
        return ['hypergraph']
    if model_name == 'alignn':
        return list(ALIGNN_DEFAULT_MODES)
    if model_name == 'cgcnn':
        return list(CGCNN_DEFAULT_MODES)
    if model_name == 'megnet':
        return list(MEGNET_DEFAULT_MODES)
    return ['full', 'full_x', 'hetero', 'attention']


def validate_modes_for_model(model_name, modes, parser):
    if model_name == 'hypergraph' and any(mode != 'hypergraph' for mode in modes):
        parser.error('The hypergraph model only supports --mode hypergraph')
    if model_name not in ('cgcnn', 'megnet', 'alignn', 'hypergraph') and any(
            mode in HYPERGRAPH_MODES for mode in modes
    ):
        parser.error(
            'The hypergraph mode is only supported with --model cgcnn, '
            '--model megnet, --model alignn, or --model hypergraph'
        )
    if model_name != 'megnet' and 'sparse' in modes:
        parser.error(
            'The sparse mode is the MEGNET_SPARSE reproduction and only '
            'supports --model megnet'
        )
    if model_name not in DEFINET_HOST_MODELS and any(mode in CGCNN_DEFINET_MODES for mode in modes):
        parser.error('The definet modes are run under --model cgcnn or --model alignn')
    if model_name == 'definet' and any(mode not in DEFINET_MODES for mode in modes):
        parser.error('The definet model only supports --mode attention attention_was')
    if model_name == 'alignn' and any(mode not in ALIGNN_MODES for mode in modes):
        parser.error(
            'The alignn model supports --mode full full_x hetero '
            'hetero_fixed_pool attention was_x hetero_was attention_was '
            'definet definet_was hypergraph hypergraph_was'
        )
    if model_name not in WAS_ABLATION_MODELS and any(mode in WAS_ABLATION_MODES for mode in modes):
        parser.error('The was_x, hetero_was, and hypergraph_was modes are only supported with --model cgcnn, --model megnet, or --model alignn')
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
    """Drop model modes that are not meaningful for a selected dataset."""
    dataset_modes = list(modes)
    if dataset_name not in MEGNET_SPARSE_DATASETS:
        dataset_modes = [mode for mode in dataset_modes if mode != 'sparse']
    elif 'sparse' in dataset_modes:
        dataset_modes = [
            'sparse',
            *(mode for mode in dataset_modes if mode != 'sparse'),
        ]
    if (
            dataset_name not in FULL_X_DISTINCT_DATASETS
            and {'full', 'full_x'} <= set(dataset_modes)
    ):
        dataset_modes = [mode for mode in dataset_modes if mode != 'full_x']
    return dataset_modes


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
        f'Latest invocation: Device: {device} | Epochs: {epochs} | {split_run_summary(seeds, cv5)}',
        'Accumulated completed results; consult per-mode summaries for training settings.',
        '-' * 50,
    ]
    for mode in modes:
        losses = results[mode]
        line = f'  {mode.upper():12s}  Mean={np.mean(losses):.4f}  Std={np.std(losses):.4f}  {split_label}={losses}'
        summary_lines.append(line)

    summary_path = os.path.join(out_dir, 'summary.txt')
    # Leaf summaries/history also restore rows lost by older CLI versions.
    merge_aggregate_summary(summary_path, summary_lines + saved_dataset_rows(out_dir))
    print(f'\n{"=" * 60}')
    with open(summary_path, encoding='utf-8') as stream:
        print(stream.read(), end='')
    print(f'{"=" * 60}\n')
    print(f'Summary saved to {summary_path}')


def main():
    parser = argparse.ArgumentParser(
        description='Train crystal GNN and region-hypergraph models for defect properties.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--model', required=True, choices=VALID_MODELS + ['all'],
                        help='Model architecture: megnet, cgcnn, definet, alignn, hypergraph, or all')
    parser.add_argument(
        '--dataset',
        required=True,
        nargs='+',
        choices=VALID_DATASETS + ['all'],
        help='One or more datasets to use, or all for every dataset',
    )
    parser.add_argument('--mode', nargs='+', default=None, choices=VALID_MODES + ['all'],
                        help='Graph mode(s) to train. Use all for all modes supported by each model')
    parser.add_argument('--native-preprocessing', choices=NATIVE_PREPROCESSING_CHOICES,
                        default=None, help='Native only: reference_v1 gives true WAS and correct '
                        'defect indices/X nodes to every selected mode for paired comparisons. '
                        'Default: reference_v1 for WAS modes, legacy for ordinary modes. '
                        'legacy reproduces historical inputs/checkpoints.')
    for impurity_dataset in ('semi', 'imp2d'):
        parser.add_argument(f'--{impurity_dataset}-preprocessing', choices=NATIVE_PREPROCESSING_CHOICES,
                            default=None, help=f'{impurity_dataset}: reference_v1 assigns true WAS '
                            'and validates defect identity; legacy reproduces historical inputs. '
                            'Default: reference_v1 for WAS modes, legacy for ordinary modes.')
    parser.add_argument('--device', default='cuda:0',
                        help='Torch device (default: cuda:0)')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of training epochs per seed or CV fold (default: 500)')
    parser.add_argument('--batch-size', '--train-batch-size',
                        dest='train_batch_size', type=int, default=None,
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
                        help='Cap neighbors per atom and target node type only for ALIGNN graph construction')
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
    parser.add_argument(
        '--alignn-hetero-feature-norm', choices=ALIGNN_HETERO_FEATURE_NORMS,
        default=None, help='HeteroALIGNN embedding, bond and angle normalization (default: layernorm)',
    )
    parser.add_argument(
        '--alignn-hetero-pooling', choices=ALIGNN_HETERO_POOLING_MODES,
        default=None, help='HeteroALIGNN readout: defect_mean = MLP(mean(h)), defect_energy_mean = mean(MLP(h)), type_mean = concatenated type means (default: defect_mean)',
    )
    parser.add_argument(
        '--alignn-hetero-relations', choices=ALIGNN_HETERO_RELATION_MODES,
        default=None, help='HeteroALIGNN message parameters (default: shared_residual)',
    )
    parser.add_argument(
        '--alignn-hetero-adapter-rank', type=int, default=None,
        help='Bottleneck width for shared_residual relations (default: 8)',
    )
    parser.add_argument(
        '--alignn-hetero-node-norm',
        nargs='+',
        choices=('layernorm', 'batchnorm', 'none'),
        default=None,
        help=(
            'One or more normalizations applied to HeteroALIGNN residual '
            'deltas: layernorm, batchnorm, none. Multiple values run an '
            'isolated benchmark for each choice (default: layernorm)'
        ),
    )
    parser.add_argument('--alignn-hetero-message', choices=ALIGNN_HETERO_MESSAGE_MODES,
                        help='Hetero message content: linear source or pair-conditioned MLP')
    parser.add_argument('--alignn-hetero-distance', choices=ALIGNN_HETERO_DISTANCE_MODES,
                        help='Hetero radial encoder: independent or shared with relation vectors')
    parser.add_argument('--alignn-hetero-aggregation', choices=ALIGNN_HETERO_AGGREGATION_MODES,
                        help='Separate relation means or joint softmax over all incoming relations')
    parser.add_argument('--alignn-hetero-defect-residual', choices=ALIGNN_HETERO_DEFECT_RESIDUAL_MODES,
                        help='Optional direct sparse defect residual after the local backbone')
    parser.add_argument('--alignn-hetero-defect-cutoff', type=float,
                        help='Auxiliary sparse defect edge cutoff in angstrom (default: 12)')
    parser.add_argument('--alignn-hetero-defect-connectivity',
                        choices=ALIGNN_HETERO_DEFECT_CONNECTIVITY_MODES,
                        help='Complete adds missing directed actual-defect pairs to the main graph using periodic minimum images')
    parser.add_argument('--alignn-hetero-aa', choices=ALIGNN_HETERO_AA_MODES,
                        help='Keep (default) or drop the atom-to-atom relation, its parameters and line-graph bonds')
    parser.add_argument('--alignn-hetero-dd', choices=ALIGNN_HETERO_DD_MODES,
                        help='Keep (default) or drop the defect-to-defect main relation, its parameters and line-graph bonds')
    parser.add_argument('--alignn-hetero-ablation', nargs='+', choices=tuple(ALIGNN_HETERO_ABLATIONS),
                        help='Sequential independent hetero ablations; cannot combine with explicit '
                             'message/distance/aggregation/defect-residual options')
    parser.add_argument('--early-stopping-patience', type=int, default=None,
                        help=('Stop any model after this many epochs without meaningful '
                              'validation improvement (default: 50; 0 disables)'))
    parser.add_argument('--early-stopping-min-delta-percent', type=float, default=None,
                        help=('Minimum relative validation MAE decrease that resets '
                              'early stopping patience, in percent (default: 0.5)'))
    parser.add_argument('--alignn-amp', action='store_true',
                        help='Use CUDA automatic mixed precision only for ALIGNN runs')
    parser.add_argument(
        '--hypergraph-radius',
        type=float,
        default=None,
        help='Defect-neighbor radius for hypergraph regions in angstrom (default: 3.0)',
    )
    from .config.defaults import HYPERGRAPH_SCHEMAS
    parser.add_argument(
        '--hypergraph-schema', choices=HYPERGRAPH_SCHEMAS, default=None,
        help='Hypergraph version (default: defect_global_attention_v3; v2 remains reproducible)',
    )
    parser.add_argument(
        '--hypergraph-pooling', choices=HYPERGRAPH_POOLING_MODES, default=None,
        help='V3: defect_mean = MLP(mean(h)) (default), defect_energy_mean = mean(MLP(h)), or hierarchical_attention; v2: region_mean',
    )
    parser.add_argument(
        '--hypergraph-updates', nargs='+', choices=HYPERGRAPH_UPDATE_MODES, default=None,
        help='HyperALIGNN v3 mean-readout ablations: none, local, global_only, local_global; multiple choices run sequentially',
    )
    parser.add_argument(
        '--seed',
        dest='seeds',
        nargs='+',
        default=None,
        metavar='SEED|all',
        help=(
            'One or more random seeds for train/val/test splits (default: 123). '
            'Use all for the standard 10-seed benchmark.'
        ),
    )
    parser.add_argument('--cv5', '--five-fold-cv', action='store_true',
                        help='Use 5-fold cross validation. Requires exactly one --seed value.')
    native_filter_options = parser.add_mutually_exclusive_group()
    native_filter_options.add_argument('--native-filter-manifest', default=None,
                                       help='Frozen native outlier manifest; preserves original split assignments')
    native_filter_options.add_argument('--native-outlier-filter', choices=['standard','strict'], default=None,
                                       help='Automatically inspect/filter native data at load time and save the frozen manifest in --run-dir')
    for impurity_dataset in ('imp2d', 'semi'):
        parser.add_argument(f'--{impurity_dataset}-quality-filter', choices=['physical'], default=None,
                            help='Screen raw structures at loading; preserve original splits and save reasons')
    parser.add_argument('--imp2d-source-db', default=None,
                        help='Optional override for the original IMP2D ASE database; auto-detects '
                             'dataset/imp2d/imp2d.db or dataset/imp2d/imp2d/imp2d.db, '
                             'then the local audit cache; downloads the verified official release if absent')
    parser.add_argument('--imp2d-source', choices=['cif', 'db'], default='cif',
                        help='db reads all original ASE rows without CSV/CIF, physically screens them and '
                             'splits equivalent structures together; requires --imp2d-quality-filter physical')
    parser.add_argument('--imp2d-host-filter', choices=['reviewed_v1'], default=None,
                        help='With --imp2d-source db, exclude the reviewed Ti2CO2 host (CO2Ti2) '
                             'with a systematic energy offset; record exclusions without trimming individual energies')
    parser.add_argument('--imp2d-energy-window', nargs=2, type=float, default=None, metavar=('LOW', 'HIGH'),
                        help='With --imp2d-source db, retain LOW < DFE < HIGH in eV at loading; '
                             'fixed benchmark scope, not a physical validity test (example: -10 10)')
    parser.add_argument('--semi-source-policy', choices=['complete', 'legacy_available'], default='complete',
                        help='With --semi-quality-filter, explicitly preserve the historical readable-source cohort; '
                             'missing files/hosts are logged separately from physical exclusions')
    parser.add_argument('--atom-init', default='./HERA/atom_init.json',
                        help='Path to atom_init.json (default: atom_init.json)')
    parser.add_argument('--log-dir', default='logs',
                        help='Directory to save training history CSVs (default: logs)')
    parser.add_argument('--run-dir', default=None,
                        help='Specific run directory to write/read instead of creating logs/run_{timestamp}')
    parser.add_argument('--compact-logs', action='store_true',
                        help='Store results under model/dataset/mode with config.json instead of '
                        'nested configuration directories; AA/DD ablations use sibling names '
                        'such as hetero_no_dd. Requires one configuration per mode '
                        '(e.g. --r 0); conflicting configurations stop before training.')
    parser.add_argument('--resume', action='store_true',
                        help='Reuse final TEST history or matching completed checkpoint results')
    parser.add_argument('--protect-existing', action='store_true',
                        help='With --resume, stop instead of restarting existing incomplete or unverified splits')
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
    if args.protect_existing and not args.resume:
        parser.error('--protect-existing requires --resume')
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
    if (
            args.early_stopping_patience is not None
            and args.early_stopping_patience < 0
    ):
        parser.error('--early-stopping-patience must be >= 0')
    if (
            args.early_stopping_min_delta_percent is not None
            and not 0 <= args.early_stopping_min_delta_percent < 100
    ):
        parser.error('--early-stopping-min-delta-percent must be in [0, 100)')
    if args.alignn_cutoff is not None and args.alignn_cutoff <= 0:
        parser.error('--alignn-cutoff must be > 0')
    if args.hypergraph_radius is not None and args.hypergraph_radius < 0:
        parser.error('--hypergraph-radius must be >= 0')
    if any(value is not None for value in (
            args.alignn_hetero_message, args.alignn_hetero_distance, args.alignn_hetero_ablation,
            args.alignn_hetero_aggregation, args.alignn_hetero_defect_residual, args.alignn_hetero_defect_cutoff,
            args.alignn_hetero_defect_connectivity, args.alignn_hetero_aa, args.alignn_hetero_dd)):
        if args.model != 'alignn':
            parser.error('Hetero encoder options require --model alignn')
        if args.mode is not None and not any(
                mode in (*ALIGNN_NODE_NORM_MODES, 'all') for mode in args.mode):
            parser.error('Hetero encoder options require a hetero mode')
        if args.alignn_hetero_ablation is not None and any(value is not None for value in (
                args.alignn_hetero_message, args.alignn_hetero_distance,
                args.alignn_hetero_aggregation, args.alignn_hetero_defect_residual)):
            parser.error('Use either --alignn-hetero-ablation or explicit message/distance/aggregation/defect-residual options')
        if args.alignn_hetero_defect_cutoff is not None and (
                not np.isfinite(args.alignn_hetero_defect_cutoff) or args.alignn_hetero_defect_cutoff <= 0):
            parser.error('--alignn-hetero-defect-cutoff must be finite and > 0')
    if args.alignn_hetero_relations is not None or args.alignn_hetero_adapter_rank is not None:
        if args.model not in ('alignn', 'all'):
            parser.error('Hetero relation options require --model alignn or all')
        if args.mode is not None and not any(
                mode in ('hetero', 'hetero_was', 'hetero_fixed_pool', 'all') for mode in args.mode):
            parser.error('Hetero relation options require a hetero mode')
        if args.alignn_hetero_adapter_rank is not None:
            if args.alignn_hetero_adapter_rank < 1:
                parser.error('--alignn-hetero-adapter-rank must be >= 1')
            if args.alignn_hetero_relations not in (None, 'shared_residual'):
                parser.error('Relation adapter rank requires shared_residual relations')
    try:
        resolve_hypergraph_pooling(args.hypergraph_schema or HYPERGRAPH_SCHEMAS[-1],
                                  args.hypergraph_pooling)
    except ValueError as exc:
        parser.error(str(exc))
    if args.hypergraph_updates is not None:
        if args.model != 'alignn':
            parser.error('--hypergraph-updates requires --model alignn')
        if args.mode is not None and not any(mode in (*HYPERGRAPH_MODES, 'all') for mode in args.mode):
            parser.error('--hypergraph-updates requires a hypergraph mode')
        try:
            for updates in args.hypergraph_updates:
                resolve_hypergraph_updates(
                    args.hypergraph_schema or HYPERGRAPH_SCHEMAS[-1],
                    args.hypergraph_pooling or 'defect_mean', updates,
                )
        except ValueError as exc:
            parser.error(str(exc))
    args.seeds = parse_seed_values(args.seeds, parser)
    if args.alignn_hetero_node_norm is not None:
        args.alignn_hetero_node_norm = list(dict.fromkeys(
            args.alignn_hetero_node_norm
        ))
    if args.cv5 and len(args.seeds) != 1:
        parser.error('--cv5 requires exactly one --seed value, e.g. --cv5 --seed 123')
    args.r = parse_radius_values(args.r, parser)
    set_seed(args.seeds[0])

    model_names = list(ALL_MODEL_SUITES) if args.model == 'all' else [args.model]
    if 'all' in args.dataset:
        if len(args.dataset) != 1:
            parser.error('--dataset all cannot be combined with specific datasets')
        dataset_names = list(VALID_DATASETS)
    else:
        dataset_names = list(dict.fromkeys(args.dataset))
    native_filter_manifest = None
    impurity_manifests = {}
    if args.imp2d_source == 'db' and ('imp2d' not in dataset_names or args.imp2d_quality_filter != 'physical'):
        parser.error('--imp2d-source db requires --dataset imp2d and --imp2d-quality-filter physical')
    if args.imp2d_host_filter is not None and args.imp2d_source != 'db':
        parser.error('--imp2d-host-filter requires --imp2d-source db')
    if args.imp2d_energy_window is not None:
        if args.imp2d_source != 'db':
            parser.error('--imp2d-energy-window requires --imp2d-source db')
        if not -float('inf') < args.imp2d_energy_window[0] < args.imp2d_energy_window[1] < float('inf'):
            parser.error('--imp2d-energy-window requires two finite increasing bounds')
    if args.semi_source_policy != 'complete' and ('semi' not in dataset_names or args.semi_quality_filter is None):
        parser.error('--semi-source-policy legacy_available requires --dataset semi and --semi-quality-filter physical')
    for impurity_dataset in ('imp2d', 'semi'):
        if getattr(args, f'{impurity_dataset}_quality_filter') and impurity_dataset not in dataset_names:
            parser.error(f'--{impurity_dataset}-quality-filter requires that dataset')
    if args.native_outlier_filter is not None and 'native' not in dataset_names:
        parser.error('--native-outlier-filter requires native among the requested datasets')
    if args.native_filter_manifest is not None:
        if 'native' not in dataset_names:
            parser.error('--native-filter-manifest requires native among the requested datasets')
        try:
            native_filter_manifest = read_manifest(args.native_filter_manifest)
            requested_splits(native_filter_manifest, args.seeds, cv5=args.cv5)
        except (OSError, ValueError, KeyError) as exc:
            parser.error(str(exc))
    requested_modes = args.mode
    if (
            requested_modes is not None
            and 'sparse' in requested_modes
            and any(
                dataset_name not in MEGNET_SPARSE_DATASETS
                for dataset_name in dataset_names
            )
    ):
        parser.error(
            'The sparse mode is only defined for vacancy-family 2DMD datasets'
        )

    init_elem_embedding(args.atom_init)

    mode_label = 'all' if requested_modes is None or requested_modes == ['all'] else requested_modes
    print(
        f'=== Model: {args.model} | Datasets: {dataset_names} | '
        f'Modes: {mode_label} ==='
    )
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
    if native_filter_manifest is not None:
        manifest_copy = os.path.join(run_dir, 'native_filter_manifest.json')
        if os.path.exists(manifest_copy):
            if manifest_identity(read_manifest(manifest_copy)) != manifest_identity(native_filter_manifest):
                raise RuntimeError('Run directory already uses another native filter; choose a different --run-dir')
        else:
            with open(manifest_copy, 'x', encoding='utf-8') as handle:
                json.dump(native_filter_manifest, handle, indent=2, sort_keys=True)
                handle.write('\n')
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
                skipped_modes = set(modes) - set(dataset_modes)
                if 'sparse' in skipped_modes:
                    print(
                        '  Skipping SPARSE: the MEGNET_SPARSE '
                        'representation is only defined for vacancy-family '
                        '2DMD datasets.'
                    )
                if 'full_x' in skipped_modes:
                    print(
                        '  Skipping FULL_X: this dataset has no vacancies, '
                        'so it is identical to FULL.'
                    )

            if dataset_name == 'native' and args.native_outlier_filter is not None and native_filter_manifest is None:
                native_filter_manifest = prepare_native_filter(
                    run_dir, args.native_outlier_filter, args.seeds, args.cv5, iter_train_val_test_splits,
                )

            dataset_cache = {}
            if dataset_name in ('imp2d', 'semi') and getattr(args, f'{dataset_name}_quality_filter'):
                if dataset_name not in impurity_manifests:
                    impurity_manifests[dataset_name] = prepare_impurity_filter(
                        run_dir, dataset_name, args.seeds, args.cv5, iter_train_val_test_splits,
                        database_path=args.imp2d_source_db,
                        **({'semi_source_policy': args.semi_source_policy} if dataset_name == 'semi' else {}),
                        **({'imp2d_source': args.imp2d_source, 'imp2d_host_filter': args.imp2d_host_filter,
                            'imp2d_energy_window': args.imp2d_energy_window}
                           if dataset_name == 'imp2d' else {}),
                    )
            impurity_manifest = impurity_manifests.get(dataset_name)

            def dataset_for_run(local_cutoff, mode, config):
                representation = representation_for_mode(mode)
                preprocessing_key = f'{dataset_name}_preprocessing'
                preprocessing = config.get(preprocessing_key)
                cache_key = (local_cutoff, representation, preprocessing)
                if cache_key not in dataset_cache:
                    dataset_cache[cache_key] = load_dataset(
                        dataset_name,
                        model_name,
                        local_cutoff=local_cutoff,
                        representations=[representation],
                        **({preprocessing_key: preprocessing} if dataset_name in ('native', 'semi', 'imp2d') else {}),
                        **({'native_filter_manifest': native_filter_manifest}
                           if dataset_name == 'native' and native_filter_manifest is not None else {}),
                        **({'impurity_filter_manifest': impurity_manifest} if impurity_manifest is not None else {}),
                        **({'imp2d_source_db': args.imp2d_source_db}
                           if dataset_name == 'imp2d' and args.imp2d_source == 'db' else {}),
                    )
                return dataset_cache[cache_key]

            def config_for_mode(mode_name):
                config = apply_training_overrides(
                    get_config(model_name, dataset_name, mode_name),
                    args,
                    model_name,
                )
                preprocessing_key = f'{dataset_name}_preprocessing'
                preprocessing = getattr(args, preprocessing_key, None)
                if preprocessing is not None:
                    if preprocessing == 'legacy':
                        config.pop(preprocessing_key, None)
                    else:
                        config[preprocessing_key] = preprocessing
                if dataset_name == 'native' and native_filter_manifest is not None:
                    config['native_data_filter'] = manifest_identity(native_filter_manifest)
                if impurity_manifest is not None:
                    config['impurity_data_filter'] = impurity_filter_identity(impurity_manifest)
                return config

            run_specs = []
            for mode in dataset_modes:
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

                mode_runs = expand_alignn_node_norm_runs(
                    mode_runs,
                    args.alignn_hetero_node_norm,
                    model_name == 'alignn' and mode in ALIGNN_NODE_NORM_MODES,
                )
                mode_runs = expand_hypergraph_update_runs(
                    mode_runs, args.hypergraph_updates,
                    model_name == 'alignn' and mode in HYPERGRAPH_MODES,
                )
                mode_runs = expand_alignn_hetero_encoder_runs(
                    mode_runs, args.alignn_hetero_ablation,
                    model_name == 'alignn' and mode in ALIGNN_NODE_NORM_MODES,
                )

                for run in mode_runs:
                    if model_name == 'alignn' and mode in ALIGNN_NODE_NORM_MODES:
                        parts = alignn_hetero_run_components(run['config']['model'])
                        if parts:
                            run['label'] += '_' + '_'.join(parts)
                    preprocessing_parts = native_run_components(run['config']) + impurity_run_components(run['config'])
                    if run['config'].get('native_data_filter'):
                        preprocessing_parts.append('clean_' + run['config']['native_data_filter']['sha256'][:8])
                    if run['config'].get('impurity_data_filter'):
                        preprocessing_parts.append('physical_' + run['config']['impurity_data_filter']['sha256'][:8])
                    if preprocessing_parts:
                        run['label'] += '_' + '_'.join(preprocessing_parts)
                    run_label = run['label']
                    train_mode = run['mode']
                    mode_parts = [dataset_dir, train_mode]
                    if not args.compact_logs:
                        if train_mode in HYPERGRAPH_MODES:
                            mode_parts.extend(hypergraph_run_components(run['config']['model']))
                        if run['radius_label'] is not None:
                            mode_parts.append(run['radius_label'])
                        if run.get('norm_label') is not None:
                            mode_parts.append(run['norm_label'])
                        if model_name == 'alignn' and mode in ALIGNN_NODE_NORM_MODES:
                            mode_parts.extend(alignn_hetero_run_components(run['config']['model']))
                        mode_parts.extend(preprocessing_parts)
                    mode_dir = (compact_mode_directory(dataset_dir, train_mode, run['config'])
                                if args.compact_logs else os.path.join(*mode_parts))
                    if not args.compact_logs:
                        os.makedirs(mode_dir, exist_ok=True)
                    run['mode_dir'] = mode_dir
                    run['summary_path'] = os.path.join(mode_dir, 'summary.txt')
                    run['resume_expected'] = {
                        'config': run['config'], 'model_name': model_name,
                        'dataset_name': dataset_name, 'mode': train_mode,
                    }
                    run_specs.append(run)

            if args.compact_logs:
                seen_directories = set()
                for run in run_specs:
                    if run['mode_dir'] in seen_directories:
                        parser.error('--compact-logs requires one configuration per mode. '
                                     'Select one radius/norm/ablation (e.g. --r 0), use separate '
                                     '--run-dir values, or omit --compact-logs for sweeps.')
                    seen_directories.add(run['mode_dir'])
                    run['compact_config'] = {
                        'layout': 'compact_v1', **run['resume_expected'],
                        'run_label': run['label'], 'epochs': args.epochs, 'cv5': args.cv5,
                    }
                    validate_compact_run_config(run['mode_dir'], run['compact_config'])
                for run in run_specs:
                    ensure_compact_run_config(run['mode_dir'], run['compact_config'])

            if args.resume and args.protect_existing:
                # Inspect every requested split before training anything for this dataset.
                for run in run_specs:
                    for logger_id in expected_split_logger_ids(args.seeds, args.cv5):
                        completed_split_result(run['mode_dir'], logger_id,
                                               expected=run['resume_expected'], protect_existing=True)

            def train_run_for_seeds(run, selected_seeds):
                if args.resume:
                    completed = completed_resume_losses(run['mode_dir'], selected_seeds, args.cv5,
                                                        expected=run['resume_expected'],
                                                        protect_existing=args.protect_existing)
                    if completed is not None:
                        print(f'  Resume: {run["label"]}, seeds={selected_seeds} already completed; skipping.')
                        return completed
                run_label = run['label']
                train_mode = run['mode']
                config = run['config']
                mode_dir = run['mode_dir']
                run_dataset = dataset_for_run(run['local_cutoff'], train_mode, config)

                print(f'\n{"=" * 60}')
                print(f'  Training {model_name.upper()} - {run_label.upper()} mode')
                if dataset_name in ('native', 'semi', 'imp2d'):
                    print(f'  {dataset_name} preprocessing: {config.get(f"{dataset_name}_preprocessing", "legacy")}; '
                          f'atom features: {config["model"]["atom_features"]}')
                print(
                    '  Batch size: '
                    f'train={config["model"]["train_batch_size"]}, '
                    f'val/test={config["model"]["test_batch_size"]}'
                )
                if train_mode in HYPERGRAPH_MODES:
                    physical_summary = (
                        'physical_backbone=none, '
                        if model_name == 'hypergraph'
                        else (
                            f'physical_backbone={model_name}, '
                            f'physical_cutoff={config["model"]["cutoff"]}, '
                            f'max_neighbors={config["model"].get("max_neighbors", "all")}, '
                        )
                    )
                    print(
                        '  Hypergraph/model: '
                        f'pooling={config["model"]["hypergraph_pooling"]}, '
                        f'{physical_summary}'
                        f'updates={config["model"].get("hypergraph_updates", "schema default")}, '
                        f'schema={config["model"]["hypergraph_schema"]}, '
                        f'radius={config["model"]["hypergraph_radius"]} A, '
                        f'hidden={config["model"]["embedding_size"]}, '
                        f'nblocks={config["model"]["nblocks"]}, '
                        f'heads={config["model"].get("n_heads", 4)}'
                    )
                else:
                    print(
                        '  Graph/model: '
                        f'cutoff={config["model"]["cutoff"]}, '
                        f'max_neighbors={config["model"].get("max_neighbors", "all")}, '
                        f'hidden={config["model"]["embedding_size"]}, '
                        f'nblocks={config["model"]["nblocks"]}, '
                        f'gcn_blocks={config["model"].get("gcn_blocks", 0)}, '
                        f'angle_embed={config["model"].get("angle_embed_size", config["model"]["edge_embed_size"])}, '
                        f'hetero_node_norm={config["model"].get("hetero_node_norm", "layernorm")}, '
                        f'hetero_feature_norm={config["model"].get("hetero_feature_norm", "n/a")}, '
                        f'hetero_pooling={config["model"].get("hetero_pooling", "n/a")}, '
                        f'hetero_relations={config["model"].get("hetero_relation_mode", "independent")}, '
                        f'hetero_adapter_rank={config["model"].get("hetero_relation_rank", 8)}, '
                        f'hetero_message={config["model"].get("hetero_message_mode", "linear")}, '
                        f'hetero_distance={config["model"].get("hetero_distance_mode", "independent")}, '
                        f'hetero_aggregation={config["model"].get("hetero_aggregation_mode", "relation_mean")}, '
                        f'hetero_defect_residual={config["model"].get("hetero_defect_residual", "none")}, '
                        f'hetero_defect_cutoff={config["model"].get("hetero_defect_cutoff", 12.0)}, '
                        f'hetero_defect_connectivity={config["model"].get("hetero_defect_connectivity", "physical")}, '
                        f'hetero_aa={config["model"].get("hetero_aa_mode", "keep")}, '
                        f'hetero_dd={config["model"].get("hetero_dd_mode", "keep")}, '
                        f'grad_accum={config["optim"].get("grad_accum_steps", 1)}, '
                        f'amp={config["optim"].get("amp", False)}'
                    )
                early_stopping_patience = config['optim'].get('early_stopping_patience', 0)
                if early_stopping_patience:
                    print(
                        '  Early stopping: '
                        f'patience={early_stopping_patience}, '
                        'minimum relative improvement='
                        f'{config["optim"].get("early_stopping_min_delta_percent", 0):g}%'
                    )
                if train_mode in LOCAL_GRAPH_SWEEP_MODES + LOCAL_CUTOFF_SWEEP_MODES:
                    print(f'  {radius_summary(train_mode, config)}')
                print(f'{"=" * 60}')

                explain_options = None
                if args.explain:
                    if args.explain_dir is not None:
                        explain_mode = os.path.basename(mode_dir) if args.compact_logs else train_mode
                        explain_parts = [args.explain_dir, model_name, dataset_name, explain_mode]
                        if not args.compact_logs:
                            if run['radius_label'] is not None:
                                explain_parts.append(run['radius_label'])
                            if run.get('norm_label') is not None:
                                explain_parts.append(run['norm_label'])
                            if model_name == 'alignn' and train_mode in ALIGNN_NODE_NORM_MODES:
                                explain_parts.extend(alignn_hetero_run_components(config['model']))
                            explain_parts.extend(native_run_components(config))
                            explain_parts.extend(impurity_run_components(config))
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

                return train_single_mode(
                    train_mode,
                    config,
                    run_dataset[:-1],
                    run_dataset[-1],
                    selected_seeds,
                    args.epochs,
                    args.device,
                    model_name=model_name,
                    dataset_name=dataset_name,
                    log_dir=mode_dir,
                    explain_options=explain_options,
                    run_label=run_label,
                    cv5=args.cv5,
                    resume=args.resume,
                    protect_existing=args.protect_existing,
                    **({'native_filter_manifest': native_filter_manifest}
                       if dataset_name == 'native' and native_filter_manifest is not None else {}),
                    **({'impurity_filter_manifest': impurity_manifest} if impurity_manifest is not None else {}),
                )

            results = {}
            result_labels = [run['label'] for run in run_specs]
            if model_name == 'alignn' and not args.cv5:
                completed_labels = set()
                losses_by_seed = {run['label']: {} for run in run_specs}

                if args.resume:
                    for run in run_specs:
                        completed_losses = completed_resume_losses(
                            run['mode_dir'],
                            args.seeds,
                            cv5=False,
                            expected=run['resume_expected'], protect_existing=args.protect_existing,
                        )
                        if completed_losses is not None:
                            print(
                                f'\nResume: all completed histories found for '
                                f'{model_name.upper()} - {run["label"].upper()}; '
                                'skipping all seeds.'
                            )
                            results[run['label']] = completed_losses
                            completed_labels.add(run['label'])

                for seed in args.seeds:
                    print(f'\n{"#" * 60}')
                    print(f'  ALIGNN benchmark seed: {seed}')
                    print(f'{"#" * 60}')
                    for run in run_specs:
                        if run['label'] in completed_labels:
                            continue
                        seed_losses = train_run_for_seeds(run, [seed])
                        if len(seed_losses) != 1:
                            raise RuntimeError(
                                f'Expected one loss for seed {seed}, got {seed_losses}'
                            )
                        losses_by_seed[run['label']][seed] = seed_losses[0]

                for run in run_specs:
                    run_label = run['label']
                    if run_label not in results:
                        results[run_label] = [
                            losses_by_seed[run_label][seed]
                            for seed in args.seeds
                        ]
                    write_mode_summary(
                        run['summary_path'],
                        model_name,
                        dataset_name,
                        run_label,
                        results[run_label],
                        args.epochs,
                        args.seeds,
                        run['config'],
                        radius_label=run['radius_label'],
                        cv5=False,
                    )
            else:
                for run in run_specs:
                    run_label = run['label']
                    if args.resume:
                        completed_losses = completed_resume_losses(
                            run['mode_dir'],
                            args.seeds,
                            args.cv5,
                            expected=run['resume_expected'], protect_existing=args.protect_existing,
                        )
                        if completed_losses is not None:
                            print(
                                f'\nResume: all completed histories found for '
                                f'{model_name.upper()} - {run_label.upper()}; '
                                'skipping dataset load/train.'
                            )
                            results[run_label] = completed_losses
                            write_mode_summary(
                                run['summary_path'],
                                model_name,
                                dataset_name,
                                run_label,
                                completed_losses,
                                args.epochs,
                                args.seeds,
                                run['config'],
                                radius_label=run['radius_label'],
                                cv5=args.cv5,
                            )
                            continue

                    losses = train_run_for_seeds(run, args.seeds)
                    results[run_label] = losses
                    write_mode_summary(
                        run['summary_path'],
                        model_name,
                        dataset_name,
                        run_label,
                        losses,
                        args.epochs,
                        args.seeds,
                        run['config'],
                        radius_label=run['radius_label'],
                        cv5=args.cv5,
                    )

            model_results[dataset_name] = results
            all_results[(model_name, dataset_name)] = results
            write_dataset_summary(
                model_name, dataset_name, result_labels, results,
                args.epochs, args.seeds, args.device, dataset_dir, cv5=args.cv5,
            )

        if len(dataset_names) > 1 or os.path.isfile(os.path.join(model_dir, 'summary.txt')):
            summary_lines = [
                f'SUMMARY: {model_name.upper()} on ALL DATASETS',
                'Accumulated completed results; consult per-mode summaries for training settings.',
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
            merge_aggregate_summary(summary_path, summary_lines + prefixed_summary_rows(model_dir))
            print(f'All-dataset summary saved to {summary_path}')

    if len(model_names) > 1 or os.path.isfile(os.path.join(run_dir, 'summary.txt')):
        summary_lines = [
            'SUMMARY: ALL MODELS',
            'Accumulated completed results; consult per-mode summaries for training settings.',
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
        merge_aggregate_summary(summary_path, summary_lines + prefixed_summary_rows(run_dir))
        print(f'All-model summary saved to {summary_path}')


if __name__ == '__main__':
    main()
