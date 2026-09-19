"""Merge saved experiment summaries without loading data or training models."""

import ast
from collections import defaultdict
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import tempfile
from datetime import datetime

from .history import TrainingLogger


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


def saved_dataset_rows(dataset_dir):
    """Recover rows from leaf summaries and completed TEST histories.

    Missing/incomplete histories never become zero-valued results. Different
    feature/relation paths and CV random states remain distinct experiments.
    """
    dataset_dir = Path(dataset_dir)
    leaves = {path.parent for path in dataset_dir.rglob('seed*_history.csv')}
    saved = {}
    for path in dataset_dir.rglob('summary.txt'):
        record = read_saved_mode_summary(path)
        if record is not None:
            leaves.add(path.parent)
            saved[path.parent] = record
    rows = []
    for leaf in sorted(leaves):
        record = saved.get(leaf)
        label = record['label'] if record else '_'.join(leaf.relative_to(dataset_dir).parts)
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
                        f'Std={statistics.pstdev(losses):.4f}  {split_label}={losses}  Split IDs={ids}')
    return rows


def prefixed_summary_rows(parent):
    rows = []
    for path in sorted(Path(parent).glob('*/summary.txt')):
        for line in summary_rows(path).values():
            rows.append(f'  {path.parent.name:12s} {line.strip()}')
    return rows


def rebuild_run_summaries(run_dir):
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
            rows = saved_dataset_rows(dataset_dir)
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
