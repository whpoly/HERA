"""Merge saved experiment summaries without loading data or training models."""

import ast
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import tempfile
from datetime import datetime

from .history import TrainingLogger


COMPACT_CONFIG_FILENAME = 'config.json'


def read_compact_run_config(log_dir):
    path = Path(log_dir) / COMPACT_CONFIG_FILENAME
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(record, dict) or record.get('layout') != 'compact_v1':
            raise ValueError('missing compact_v1 layout marker')
        if not isinstance(record.get('config'), dict) or not isinstance(record.get('run_label'), str):
            raise ValueError('missing experiment configuration or run label')
    except (OSError, ValueError) as exc:
        raise RuntimeError(f'Cannot verify existing experiment configuration: {path}: {exc}') from exc
    return record


def validate_compact_run_config(log_dir, expected):
    """Require a single experiment per mode directory, including CSV-only resume."""
    directory = Path(log_dir)
    saved = read_compact_run_config(directory)
    if saved is not None:
        changed = [key for key, value in expected.items() if saved.get(key) != value]
        if changed:
            raise RuntimeError(f'Experiment configuration differs in {directory}: {", ".join(changed)}. '
                               'Existing results were not changed. Use a different --run-dir '
                               'for a different experiment.')
    elif directory.exists() and any(directory.iterdir()):
        raise RuntimeError(f'Existing outputs without {COMPACT_CONFIG_FILENAME} in {directory}. '
                           'Compact layout cannot adopt or overwrite them. Keep the original '
                           'layout to resume them, or select a new --run-dir.')


def ensure_compact_run_config(log_dir, record):
    validate_compact_run_config(log_dir, record)
    directory = Path(log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / COMPACT_CONFIG_FILENAME
    # Exclusive creation prevents a second invocation from replacing the
    # experiment identity between validation and starting training.
    try:
        with path.open('x', encoding='utf-8', newline='') as stream:
            json.dump(record, stream, indent=2, ensure_ascii=False, sort_keys=True)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        validate_compact_run_config(log_dir, record)


def read_checkpoint_result(path):
    """Read metadata from a user's locally generated HERA checkpoint on CPU."""
    from .trainer import load_trusted_checkpoint
    payload = load_trusted_checkpoint(path, map_location='cpu')
    if not isinstance(payload, dict) or not isinstance(payload.get('config'), dict):
        raise ValueError(f'No HERA configuration in checkpoint: {path}')
    config = payload['config']
    aa = config.get('model', {}).get('hetero_aa_mode', 'keep')
    dd = config.get('model', {}).get('hetero_dd_mode', 'keep')
    variants = {('keep', 'keep'): 'baseline', ('drop', 'keep'): 'no_aa',
                ('keep', 'drop'): 'no_dd', ('drop', 'drop'): 'no_aa_dd'}
    loss = payload.get('test_mae')
    try:
        loss = float(loss)
    except (TypeError, ValueError):
        loss = None
    if loss is not None and not math.isfinite(loss):
        loss = None
    return {
        'path': str(Path(path).resolve()), 'config': config,
        'model_name': payload.get('model_name'), 'dataset_name': payload.get('dataset_name'),
        'mode': payload.get('mode'), 'run_label': payload.get('run_label'),
        'logger_id': str(payload.get('split', {}).get('logger_id', '')),
        'test_mae': loss, 'aa': aa, 'dd': dd,
        'relation_variant': variants.get((aa, dd), 'unknown'),
    }


def completed_split_result(log_dir, logger_id, expected=None, protect_existing=False):
    """Reuse a final test result; never restart over an unreadable checkpoint."""
    if expected is not None and (Path(log_dir) / COMPACT_CONFIG_FILENAME).exists():
        validate_compact_run_config(log_dir, expected)
    loss = TrainingLogger.completed_test_mae(log_dir, logger_id)
    if loss is not None:
        return loss
    directory = Path(log_dir)
    checkpoint = directory / f'seed{logger_id}_best_checkpoint.pth'
    if checkpoint.is_file() and expected is not None:
        try:
            result = read_checkpoint_result(checkpoint)
            for key, value in {**expected, 'logger_id': str(logger_id)}.items():
                if result.get(key) != value:
                    raise ValueError(f'saved {key} does not match the requested experiment')
            if result['test_mae'] is None:
                raise ValueError('checkpoint has no finite final test_mae')
        except Exception as exc:
            raise RuntimeError(f'Resume stopped; existing checkpoint will not be retrained: '
                               f'{checkpoint}\nReason: {exc}') from exc
        print(f'  Resume: completed checkpoint found: {checkpoint} '
              f'(test_mae={result["test_mae"]:.6f}); skipping training.', flush=True)
        return result['test_mae']
    summary = read_saved_mode_summary(directory / 'summary.txt') if protect_existing else None
    if protect_existing and (
            checkpoint.exists() or Path(TrainingLogger.filepath_for(log_dir, logger_id)).exists()
            or (summary and str(logger_id) in summary['splits'])):
        raise RuntimeError(f'Resume stopped: saved files exist for seed{logger_id} in {directory}, '
                           'but no verified final test result was found. No restart was launched. '
                           'Use merge_hetero_relation_results.py to merge existing results only.')
    return None


def atomic_write_text(path, text, backup=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        if path.read_text(encoding='utf-8') == text:
            return
        if backup:
            stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            shutil.copy2(path, path.with_name(f'{path.name}.{stamp}.bak'))
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='',
                                         dir=path.parent, delete=False) as stream:
            temporary = stream.name
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def summary_rows(path):
    """Read table rows, keeping their experiment identity before Mean=."""
    path = Path(path)
    if not path.is_file():
        return {}
    rows = {}
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        if line.startswith('  ') and 'Mean=' in line and 'Std=' in line:
            key = ' '.join(line.split('Mean=', 1)[0].split())
            rows[key] = line
    return rows


def merge_aggregate_summary(path, lines):
    """Update selected rows while retaining other saved modes/datasets."""
    rows = summary_rows(path)
    header = []
    for line in lines:
        if line.startswith('  ') and 'Mean=' in line and 'Std=' in line:
            key = ' '.join(line.split('Mean=', 1)[0].split())
            rows[key] = line
        else:
            header.append(line)
    atomic_write_text(path, '\n'.join(header + list(rows.values())) + '\n', backup=True)


def read_saved_mode_summary(path):
    """Read split identities as well as losses; never guess seed ordering."""
    path = Path(path)
    if not path.is_file():
        return None
    lines = path.read_text(encoding='utf-8-sig').splitlines()
    if not lines or lines[0].startswith('SUMMARY:'):
        return None
    identity = lines[0].split(' | ')
    if len(identity) != 3:
        return None
    try:
        loss_line = next(line for line in lines if line.startswith(('Per-seed losses:', 'Per-fold losses:')))
        losses = [float(value) for value in ast.literal_eval(loss_line.split(':', 1)[1].strip())]
        if not losses or not all(math.isfinite(value) for value in losses):
            return None
        cv5 = loss_line.startswith('Per-fold')
        if cv5:
            seed = next(re.search(r'5-fold CV random_state: (\d+)', line).group(1)
                        for line in lines if re.search(r'5-fold CV random_state: (\d+)', line))
            if len(losses) != 5:
                return None
            ids = [f'{seed}_fold{i + 1}' for i in range(5)]
        else:
            seed_text = next(line.split('Seeds:', 1)[1] for line in lines if 'Seeds:' in line)
            ids = [str(int(value)) for value in ast.literal_eval(seed_text.strip())]
        if len(ids) != len(losses) or len(set(ids)) != len(ids):
            return None
    except (StopIteration, SyntaxError, ValueError, TypeError):
        return None
    return {'label': identity[2].lower(), 'splits': dict(zip(ids, losses)), 'cv5': cv5}


def merge_mode_losses(path, seeds, losses, cv5=False):
    """Adding seeds must not remove previously reported seeds in this mode."""
    old = read_saved_mode_summary(path)
    if cv5 or old is None or old['cv5']:
        return seeds, losses
    merged = dict(old['splits'])
    merged.update((str(seed), float(loss)) for seed, loss in zip(seeds, losses))
    return [int(seed) for seed in merged], list(merged.values())


def saved_dataset_rows(dataset_dir, checkpoint_results=None):
    """Recover rows from leaf summaries and completed TEST histories.

    Missing/incomplete histories never become zero-valued results. Different
    feature/relation paths and CV random states remain distinct experiments.
    """
    dataset_dir = Path(dataset_dir)
    leaves = {path.parent for path in dataset_dir.rglob('seed*_history.csv')}
    checkpoint_splits = defaultdict(dict)
    checkpoint_relations = defaultdict(set)
    for record in checkpoint_results or []:
        path = Path(record['path'])
        if not path.is_relative_to(dataset_dir.resolve()) or record['test_mae'] is None:
            continue
        split_id = record['logger_id']
        if not re.fullmatch(r'\d+(?:_fold[1-5])?', split_id):
            continue
        if path.name != f'seed{split_id}_best_checkpoint.pth':
            raise ValueError(f'Checkpoint split identity disagrees with filename: {path}')
        checkpoint_splits[path.parent][split_id] = record['test_mae']
        checkpoint_relations[path.parent].add((record['aa'], record['dd']))
        leaves.add(path.parent)
    saved = {}
    for path in dataset_dir.rglob('summary.txt'):
        record = read_saved_mode_summary(path)
        if record is not None:
            leaves.add(path.parent)
            saved[path.parent] = record
    rows = []
    for leaf in sorted(leaves):
        relations = checkpoint_relations[leaf]
        if len(relations) > 1:
            raise ValueError(f'Checkpoints in {leaf} have different AA/DD configurations; '
                             'they cannot be averaged as one experiment.')
        relation_text = ''
        if relations:
            aa, dd = next(iter(relations))
            relation_text = f'  Saved AA={aa}, DD={dd}'
        record = saved.get(leaf)
        compact_config = read_compact_run_config(leaf) if record is None else None
        label = (record['label'] if record else compact_config['run_label'] if compact_config
                 else '_'.join(leaf.relative_to(dataset_dir).parts))
        splits = dict(record['splits']) if record else {}
        for path in sorted(leaf.glob('seed*_history.csv')):
            split_id = path.name[len('seed'):-len('_history.csv')]
            if not re.fullmatch(r'\d+(?:_fold[1-5])?', split_id):
                continue
            value = TrainingLogger.completed_test_mae(leaf, split_id)
            if value is not None:
                # A mode summary retains more digits than the history CSV.
                if split_id not in splits or not math.isclose(splits[split_id], value, abs_tol=1e-6):
                    splits[split_id] = value
        for split_id, value in checkpoint_splits[leaf].items():
            if split_id in splits and not math.isclose(splits[split_id], value, abs_tol=1e-6):
                raise ValueError(f'Conflicting saved test MAEs for seed{split_id} in {leaf}: '
                                 f'history/summary={splits[split_id]}, checkpoint={value}. '
                                 'Inspect the saved artifacts before merging.')
            splits.setdefault(split_id, value)
        groups = defaultdict(dict)
        for split_id, value in splits.items():
            protocol = f'cv5_seed{split_id.split("_fold")[0]}' if '_fold' in split_id else 'seeds'
            groups[protocol][split_id] = value
        for protocol, values in sorted(groups.items()):
            # Keep the historical label for the usual single-protocol case.
            name = label if len(groups) == 1 else f'{label}_{protocol}'
            ids = sorted(values, key=lambda value: tuple(int(part) for part in value.split('_fold')))
            losses = [values[split_id] for split_id in ids]
            split_label = 'Folds' if protocol.startswith('cv5_') else 'Seeds'
            rows.append(f'  {name.upper():12s}  Mean={statistics.mean(losses):.4f}  '
                        f'Std={statistics.pstdev(losses):.4f}  {split_label}={losses}  Split IDs={ids}'
                        f'{relation_text}')
    return rows


def prefixed_summary_rows(parent):
    rows = []
    for path in sorted(Path(parent).glob('*/summary.txt')):
        for line in summary_rows(path).values():
            rows.append(f'  {path.parent.name:12s} {line.strip()}')
    return rows


def rebuild_run_summaries(run_dir, checkpoint_results=None):
    """Rebuild model/dataset tables using saved metrics, with zero training.

    run_dir is a HERA.main root, e.g. <benchmark>/no_dd. Existing table-only
    rows are retained too. Source histories/checkpoints are never modified.
    """
    root = Path(run_dir)
    if not root.is_dir():
        raise FileNotFoundError(f'Existing run directory not found: {root}')
    count = 0
    for model_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for dataset_dir in sorted(path for path in model_dir.iterdir() if path.is_dir()):
            rows = saved_dataset_rows(dataset_dir, checkpoint_results=checkpoint_results)
            if rows or summary_rows(dataset_dir / 'summary.txt'):
                merge_aggregate_summary(dataset_dir / 'summary.txt', [
                    f'SUMMARY: {model_dir.name.upper()} on {dataset_dir.name}',
                    'Accumulated saved results; no training or test evaluation performed.',
                    '-' * 50, *rows,
                ])
                count += len(summary_rows(dataset_dir / 'summary.txt'))
        rows = prefixed_summary_rows(model_dir)
        if rows:
            merge_aggregate_summary(model_dir / 'summary.txt', [
                f'SUMMARY: {model_dir.name.upper()} on ALL DATASETS',
                'Accumulated saved results; consult per-mode summaries for training settings.',
                '-' * 50, *rows,
            ])
    rows = prefixed_summary_rows(root)
    if rows:
        merge_aggregate_summary(root / 'summary.txt', [
            'SUMMARY: ALL MODELS',
            'Accumulated saved results; consult per-mode summaries for training settings.',
            '-' * 50, *rows,
        ])
    return count


def merge_benchmark_results(root, variants, checkpoint_results=None):
    root = Path(root)
    for variant in variants:
        variant_dir = root / variant
        if not variant_dir.is_dir():
            print(f'No saved directory: {variant_dir}; skipped (no training).', flush=True)
            continue
        count = rebuild_run_summaries(variant_dir, checkpoint_results=checkpoint_results)
        print(f'[{variant}] {count} saved result rows merged.', flush=True)
    rows = prefixed_summary_rows(root)
    if rows:
        merge_aggregate_summary(root / 'summary.txt', [
            'HETERO RELATION BENCHMARK: accumulated saved results',
            'Rows: source directory, model, dataset, mode, MAE statistics and split results.',
            '-' * 50, *rows,
        ])
        print(f'Combined summary: {root / "summary.txt"}', flush=True)
