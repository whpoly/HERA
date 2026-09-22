"""Frozen, train-fitted native outlier manifests with preserved split membership."""
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

from .native_quality import configuration_family, fit_family_fences, geometry_exclusions, joint_distortion_exclusion


SCHEMA = 'native_outliers_v1'
DEFAULT_POLICY = {
    'min_distance_angstrom': 1.0,
    'iqr_multiplier': 3.0,
    'iqr_floor_ev': 1.0,
    'min_group_train_samples': 8,
    'group_by': ['material', 'defect_type', 'added_z', 'removed_z'],
}


def manifest_identity(manifest):
    encoded = json.dumps(manifest, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return {'schema': SCHEMA, 'sha256': hashlib.sha256(encoded).hexdigest(), 'policy': manifest['policy']}


def filter_applies_to_source(source_id, policy):
    indices = policy.get('filter_poscar_indices')
    if indices is None:
        return True
    match = re.search(r'-POSCAR(\d+)-', source_id)
    if match is None:
        raise ValueError(f'Cannot determine POSCAR filter scope: {source_id}')
    return int(match[1]) in indices


def validate_manifest(manifest):
    if manifest.get('schema') != SCHEMA or manifest.get('dataset') != 'native':
        raise ValueError('Expected a native_outliers_v1 manifest')
    scope = manifest['policy'].get('filter_poscar_indices')
    if scope is not None and (not isinstance(scope, list) or not scope
                              or any(type(i) is not int or i < 0 for i in scope)
                              or scope != sorted(set(scope))):
        raise ValueError('filter_poscar_indices must be a sorted, unique list of nonnegative integers')
    records = manifest['records']
    ids = [r['source_id'] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate IDs in native filter manifest')
    for sid in ids:
        if '/' in sid or '\\' in sid or sid in ('', '.', '..'):
            raise ValueError('Native filter source IDs must be filenames')
    if not manifest['splits']:
        raise ValueError('Native filter manifest has no splits')
    loggers = []
    for split in manifest['splits']:
        loggers.append(str(split['logger_id']))
        original = [sid for part in ('train', 'val', 'test') for sid in split['original'][part]]
        if len(original) != len(ids) or set(original) != set(ids):
            raise ValueError('Native filter split must partition every original source exactly once')
        for part in ('train', 'val', 'test'):
            kept = split['kept'][part]
            keep_set = set(kept)
            if not kept or [sid for sid in split['original'][part] if sid in keep_set] != kept:
                raise ValueError('Native filter retained IDs must be a nonempty ordered subset')
            excluded = {r['source_id'] for r in split['excluded'][part]}
            if any(not filter_applies_to_source(sid, manifest['policy']) for sid in excluded):
                raise ValueError('Native manifest excludes a source outside its POSCAR filter scope')
            if keep_set & excluded or keep_set | excluded != set(split['original'][part]):
                raise ValueError('Native filter kept/excluded IDs do not partition the split')
    if len(loggers) != len(set(loggers)):
        raise ValueError('Duplicate split identities in native filter manifest')
    manifest_identity(manifest)  # Reject non-finite JSON numbers.
    return manifest


def read_manifest(path):
    return validate_manifest(json.loads(Path(path).read_text(encoding='utf-8')))


def requested_splits(manifest, seeds, cv5=False):
    if bool(manifest['cv5']) != bool(cv5):
        raise ValueError('Native filter CV protocol differs; regenerate the manifest for this protocol')
    required = ([f'{seed}_fold{i}' for seed in seeds for i in range(1, 6)]
                if cv5 else [str(seed) for seed in seeds])
    by_id = {str(s['logger_id']): s for s in manifest['splits']}
    missing = set(required)-set(by_id)
    if missing:
        raise ValueError(f'Native filter manifest lacks requested splits: {sorted(missing)}')
    return [by_id[key] for key in required]


def retained_source_ids(manifest):
    return {sid for split in manifest['splits'] for part in ('train', 'val', 'test')
            for sid in split['kept'][part]}


def verify_source_table(manifest, dataframe):
    rows = list(dataframe.itertuples(index=False, name=None))
    expected = manifest['records']
    if [str(r[0]) for r in rows] != [r['source_id'] for r in expected]:
        raise ValueError('Native source IDs/order changed since preprocessing; regenerate the manifest')
    for row, record in zip(rows, expected):
        try:
            actual = float(row[1])
            actual = actual if math.isfinite(actual) else None
        except (ValueError, TypeError):
            actual = None
        target = record['target']
        if actual != target and (actual is None or target is None
                                 or not math.isclose(actual, target, rel_tol=0, abs_tol=1e-10)):
            raise ValueError(f'Native target changed since preprocessing: {row[0]}')


def verify_source_file(path, record):
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != record['sha256']:
        raise ValueError(f'Native CIF changed since preprocessing: {record["source_id"]}')


def fit_split(records, original_split, policy):
    """Fit once on geometry-valid training labels; never fit on val/test labels."""
    def group_key(row):
        return '/'.join(str(row[field]) for field in policy['group_by'])

    by_id = {r['source_id']: r for r in records}
    groups = {}
    for sid in original_split['train']:
        row = by_id[sid]
        if not row['quality_reasons']:
            key = group_key(row)
            groups.setdefault(key, []).append(row['target'])
    bounds = {}
    for key, values in sorted(groups.items()):
        if len(values) < policy['min_group_train_samples']:
            bounds[key] = {'n_train': len(values), 'status': 'insufficient_train_samples'}
            continue
        q1, q3 = np.quantile(values, [.25, .75], method='linear')
        width = policy['iqr_multiplier']*max(float(q3-q1), policy['iqr_floor_ev'])
        bounds[key] = {'n_train': len(values), 'status': 'fitted', 'q1': float(q1), 'q3': float(q3),
                       'lower_ev': float(q1-width), 'upper_ev': float(q3+width)}
    deep_policy = policy.get('deep_checks')
    family_fences = fit_family_fences(records, original_split['train'], deep_policy) if deep_policy else {}
    kept, excluded = {}, {}
    for part in ('train', 'val', 'test'):
        kept[part], excluded[part] = [], []
        for sid in original_split[part]:
            row = by_id[sid]
            if not filter_applies_to_source(sid, policy):
                kept[part].append(sid)
                continue
            reasons = list(row['quality_reasons'])
            group = bounds.get(group_key(row), {}) if not reasons else {}
            if not reasons and group.get('status') == 'fitted':
                if row['target'] < group['lower_ev']:
                    reasons.append('energy_below_train_fence')
                elif row['target'] > group['upper_ev']:
                    reasons.append('energy_above_train_fence')
            if deep_policy and not row['quality_reasons']:
                reasons.extend(geometry_exclusions(row, deep_policy))
                fence = family_fences.get(configuration_family(sid), {})
                if fence.get('status') == 'fitted':
                    if row['target'] < fence['lower_ev']:
                        reasons.append('energy_below_same_family_train_fence')
                    elif row['target'] > fence['upper_ev']:
                        reasons.append('energy_above_same_family_train_fence')
                    if joint_distortion_exclusion(row, fence, deep_policy):
                        reasons.append('compressed_and_high_energy_vs_same_family_train')
            if reasons:
                excluded[part].append({'source_id': sid, 'reasons': reasons})
            else:
                kept[part].append(sid)
        if not kept[part]:
            raise ValueError(f'Native filtering emptied {part}; review the policy before training')
    result = {'original': original_split, 'kept': kept, 'excluded': excluded, 'train_fences': bounds}
    if deep_policy:
        result['family_train_fences'] = family_fences
    return result


def filtered_splits(data, targets, manifest, seeds, cv5=False):
    by_id = {}
    for structure, target in zip(data, targets):
        sid = getattr(structure, 'source_id', None)
        if not sid or sid in by_id:
            raise ValueError('Native filtered training requires unique source IDs')
        by_id[sid] = (structure, target)
    for saved in requested_splits(manifest, seeds, cv5):
        split = {key: saved[key] for key in ('display', 'logger_id', 'seed', 'explain_id')}
        for part in ('train', 'val', 'test'):
            ids = saved['kept'][part]
            missing = set(ids)-set(by_id)
            if missing:
                raise ValueError(f'Filtered native sources missing from loaded graphs: {sorted(missing)[:5]}')
            split[f'{part}_X'] = [by_id[sid][0] for sid in ids]
            split[f'{part}_y'] = [by_id[sid][1] for sid in ids]
        split['native_filter_counts'] = {
            part: {'original': len(saved['original'][part]), 'kept': len(saved['kept'][part]),
                   'removed': len(saved['excluded'][part])} for part in ('train', 'val', 'test')
        }
        yield split
