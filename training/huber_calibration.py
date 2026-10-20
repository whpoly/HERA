"""Shared Huber thresholds from training-only, out-of-fold MSE residuals."""

import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .results import atomic_write_text


SCHEMA = 'huber_training_oof_v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode('utf-8')).hexdigest()


def source_key(structure):
    sid = str(getattr(structure, 'source_id', ''))
    if not sid:
        raise ValueError('Huber calibration requires stable source IDs for every structure')
    return json.dumps([sid, str(getattr(structure, 'material', '')),
                       str(getattr(structure, 'concentration', ''))], separators=(',', ':'))


def split_identity(split):
    """Hash memberships and training labels; never inspect outer val/test labels."""
    parts = {part: [source_key(s) for s in split[f'{part}_X']]
             for part in ('train', 'val', 'test')}
    keys = [key for values in parts.values() for key in values]
    if len(keys) != len(set(keys)):
        raise ValueError('Huber calibration requires distinct source IDs across the split')
    targets = np.asarray([float(y) for y in split['train_y']], dtype=float)
    if len(targets) != len(parts['train']) or not np.isfinite(targets).all():
        raise ValueError('Invalid training targets for Huber calibration')
    return dict(seed=int(split['seed']), membership_sha256=digest(parts),
                train_targets_sha256=digest(targets.tolist()),
                counts={part: len(values) for part, values in parts.items()})


def residual_threshold(residuals, multiplier=1.345, floor_ev=1e-6):
    residuals = np.asarray(residuals, dtype=float).reshape(-1)
    if not len(residuals) or not np.isfinite(residuals).all():
        raise ValueError('Calibration needs nonempty, finite residuals')
    if not all(math.isfinite(v) and v > 0 for v in (multiplier, floor_ev)):
        raise ValueError('Calibration multiplier and floor must be finite and positive')
    center = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - center)))
    scale, method = mad / 0.6744897501960817, 'normal_consistent_mad'
    if scale == 0:
        scale, method = float(np.median(np.abs(residuals))), 'median_absolute_residual'
    if scale == 0:
        scale, method = float(np.sqrt(np.mean(residuals ** 2))), 'rms_residual'
    delta = max(multiplier * scale, floor_ev)
    if not math.isfinite(delta):
        raise ValueError('Non-finite calibrated Huber threshold')
    return dict(delta_ev=delta, scale_ev=scale, scale_method=method,
                median_residual_ev=center, mad_ev=mad, n=len(residuals),
                mae=float(np.mean(np.abs(residuals))),
                rmse=float(np.sqrt(np.mean(residuals ** 2))),
                floor_applied=multiplier * scale < floor_ev)


def grouped_folds(groups, n_splits, seed):
    """Seeded, size-balanced group assignment, preserving known equivalences."""
    if n_splits < 2:
        raise ValueError('Calibration requires at least two folds')
    groups = np.asarray(groups, dtype=str)
    unique, counts = np.unique(groups, return_counts=True)
    if len(unique) < n_splits + 1:
        raise ValueError('Too few training groups for OOF folds and an inner validation set')
    order = np.random.default_rng(seed).permutation(len(unique))
    order = sorted(order, key=lambda i: -counts[i])
    sizes, assignment = [0] * n_splits, {}
    for i in order:
        fold = min(range(n_splits), key=lambda j: sizes[j])
        assignment[unique[i]] = fold
        sizes[fold] += int(counts[i])
    result = []
    for fold in range(n_splits):
        holdout = np.flatnonzero([assignment[g] == fold for g in groups])
        fit_pool = np.flatnonzero([assignment[g] != fold for g in groups])
        if not len(holdout) or len(set(groups[fit_pool])) < 2:
            raise ValueError('Each calibration fold needs held-out and inner fit/validation groups')
        result.append((fit_pool, holdout))
    return result


def inner_partition(pool, groups, seed):
    """Reserve approximately 10% of the fold's fit pool for model selection."""
    unique = list(dict.fromkeys(groups[i] for i in pool))
    order = np.random.default_rng(seed).permutation(len(unique))
    selected, count = set(), 0
    for i in order[:-1]:  # Always retain at least one fitting group.
        selected.add(unique[i])
        count += sum(groups[j] == unique[i] for j in pool)
        if count >= max(1, math.ceil(.1 * len(pool))):
            break
    fit = [int(i) for i in pool if groups[i] not in selected]
    val = [int(i) for i in pool if groups[i] in selected]
    return fit, val


def calibration_identity(manifest):
    if manifest.get('schema') != SCHEMA or not manifest.get('splits'):
        raise ValueError('Invalid or incomplete Huber calibration manifest')
    if manifest.get('sha256') != digest({k: v for k, v in manifest.items() if k != 'sha256'}):
        raise ValueError('Huber calibration manifest checksum mismatch')
    for key, record in manifest['splits'].items():
        delta = record['statistics']['delta_ev']
        if not math.isfinite(delta) or delta <= 0 or key != str(record['identity']['seed']):
            raise ValueError('Invalid calibrated threshold or seed')
    return dict(schema=SCHEMA, dataset=manifest['dataset'], sha256=manifest['sha256'],
                splits={key: dict(identity=record['identity'],
                                  delta_ev=record['statistics']['delta_ev'])
                        for key, record in manifest['splits'].items()})


def read_calibration(path):
    manifest = json.loads(Path(path).read_text(encoding='utf-8'))
    calibration_identity(manifest)
    return manifest


def resolved_delta(config, split, dataset_name):
    calibration = config['optim'].get('huber_calibration')
    if calibration is None:
        return config['optim'].get('huber_delta_ev', 1.0)
    record = calibration['splits'].get(str(split['seed']))
    if (calibration['dataset'] != dataset_name or record is None
            or record['identity'] != split_identity(split)):
        raise ValueError('Training split differs from the shared Huber calibration')
    return record['delta_ev']


def calibrate(splits, config, dataset_name, output, epochs, folds, multiplier, device,
              *, trainer_factory, seed_fn, weights_fn, clear_fn, group_by_source=None):
    """Fit one fixed reference per inner fold; outer val/test never reach a trainer."""
    if config.get('task') != 'alignn_was_x' or config['optim'].get('point_loss', 'mse') != 'mse':
        raise ValueError('Calibration reference must be ALIGNN WAS-X with MSE')
    if epochs < 1 or folds < 2:
        raise ValueError('Calibration epochs must be positive and folds >= 2')
    residual_threshold([0.], multiplier)  # Validate the coefficient before fitting.
    output = Path(output)
    protocol = dict(schema=SCHEMA, dataset=dataset_name, reference='alignn/was_x',
                    config=copy.deepcopy(config), epochs=epochs, folds=folds,
                    multiplier=multiplier, floor_ev=1e-6,
                    inner_validation_fraction=.1, selection='inner validation MAE only',
                    groups='known structure_group; otherwise source identity')
    path = output / 'protocol.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != protocol:
        raise ValueError('Existing Huber calibration protocol differs; use a new run directory')
    atomic_write_text(path, json.dumps(protocol, indent=2, allow_nan=False) + '\n')
    records = {}
    group_by_source = group_by_source or {}
    for split in splits:
        identity = split_identity(split)
        seed = identity['seed']
        structures = list(split['train_X'])
        targets = np.asarray([float(y) for y in split['train_y']], dtype=float)
        keys = [source_key(s) for s in structures]
        groups = [str(group_by_source.get(str(getattr(s, 'source_id')), key))
                  for s, key in zip(structures, keys)]
        predictions = np.full(len(targets), np.nan)
        fold_records = []
        for fold, (pool, holdout) in enumerate(grouped_folds(groups, folds, seed)):
            fold_seed = (seed + 104729 * (fold + 1)) % (2 ** 32)
            fit, val = inner_partition(pool, groups, fold_seed)
            holdout = holdout.tolist()
            expected = dict(protocol_sha256=digest(protocol), split=identity, fold=fold,
                            seed=fold_seed, groups_sha256=digest(groups),
                            fit=fit, val=val, holdout=holdout)
            fold_path = output / f'seed{seed}' / f'fold{fold + 1}.json'
            if fold_path.exists():
                saved = json.loads(fold_path.read_text(encoding='utf-8'))
                if (saved.get('identity') != expected
                        or saved.get('sha256') != digest({k: v for k, v in saved.items() if k != 'sha256'})):
                    raise ValueError(f'Cached calibration fold differs: {fold_path}')
                predicted = np.asarray(saved['predictions'], dtype=float)
                print(f'  Calibration seed={seed} fold={fold + 1}: reusing completed fold', flush=True)
            else:
                seed_fn(fold_seed)
                trainer = trainer_factory(copy.deepcopy(config), device, seed=fold_seed)
                try:
                    trainer.prepare_data(
                        [structures[i] for i in fit], targets[fit],
                        [structures[i] for i in val], targets[val],
                        config.get('target_name', 'formation_energy'),
                    )
                    best, weights, best_epoch = float('inf'), None, 0
                    history = []
                    for epoch in range(epochs):
                        train_mae, _ = trainer.train_one_epoch()
                        val_mae = float(trainer.evaluate_on_test())
                        if not math.isfinite(val_mae):
                            raise ValueError('Non-finite inner validation MAE during calibration')
                        trainer.step_scheduler(val_mae)
                        history.append(dict(epoch=epoch + 1, train_mae=float(train_mae), val_mae=val_mae))
                        if val_mae < best:
                            best, weights, best_epoch = val_mae, weights_fn(trainer.model), epoch + 1
                        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
                            print(f'  Calibration seed={seed} fold={fold + 1}/{folds} '
                                  f'epoch={epoch + 1}/{epochs} inner_val_mae={val_mae:.6f}', flush=True)
                    _, predicted = trainer.predict_structures(
                        [structures[i] for i in holdout], targets[holdout], weights,
                        return_predictions=True,
                    )
                    predicted = np.asarray(predicted, dtype=float).reshape(-1)
                    if len(predicted) != len(holdout) or not np.isfinite(predicted).all():
                        raise ValueError('Invalid out-of-fold predictions')
                    saved = dict(identity=expected, predictions=predicted.tolist(),
                                 best_epoch=best_epoch, best_inner_val_mae=best, history=history)
                    saved['sha256'] = digest(saved)
                    atomic_write_text(fold_path, json.dumps(saved, indent=2, allow_nan=False) + '\n')
                finally:
                    del trainer
                    clear_fn(device)
            if len(predicted) != len(holdout) or not np.isfinite(predicted).all():
                raise ValueError('Invalid cached out-of-fold predictions')
            predictions[holdout] = predicted
            fold_records.append(dict(fold=fold + 1, sha256=saved['sha256'],
                                     best_epoch=saved['best_epoch']))
        statistics = residual_threshold(predictions - targets, multiplier)
        records[str(seed)] = dict(identity=identity, statistics=statistics, folds=fold_records)
        rows = [dict(source_key=key, target=float(y), prediction=float(pred), residual=float(pred - y))
                for key, y, pred in zip(keys, targets, predictions)]
        atomic_write_text(output / f'seed{seed}' / 'oof_predictions.json',
                          json.dumps(rows, indent=2, allow_nan=False) + '\n')
        print(f'  Shared Huber delta: {dataset_name} seed={seed}: {statistics["delta_ev"]:.8g}', flush=True)
    manifest = dict(schema=SCHEMA, dataset=dataset_name, protocol=protocol, splits=records,
                    target_unit='eV/site' if dataset_name.startswith('2dmd_') else 'eV')
    manifest['sha256'] = digest(manifest)
    calibration_identity(manifest)
    path = output / 'manifest.json'
    if path.exists() and read_calibration(path) != manifest:
        raise ValueError('Completed Huber calibration changed; use a new run directory')
    atomic_write_text(path, json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    return manifest
