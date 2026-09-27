"""Label-independent grouped train/validation/test splits for native defects."""
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATASETS = ('native',)
GROUPINGS = ('family', 'material')
PARTS = ('train', 'val', 'test')


def source_group(source_id, dataset, grouping):
    if dataset not in DATASETS or grouping not in GROUPINGS:
        raise ValueError(f'Unsupported OOD grouping: {dataset}/{grouping}')
    if not isinstance(source_id, str) or '/' in source_id or '\\' in source_id:
        raise ValueError(f'Invalid OOD source ID: {source_id!r}')
    parts = source_id.split('-')
    if (len(parts) < 4 or not parts[0].isdigit() or not parts[1]
            or re.search(r'-POSCAR\d+(?=-|\.cif$)', source_id) is None):
        raise ValueError(f'Invalid {dataset} source ID: {source_id!r}')
    return parts[1] if grouping == 'material' else re.sub(r'-POSCAR\d+(?=-|\.cif$)', '', source_id)


def protocol_for(dataset, grouping):
    if dataset not in DATASETS or grouping not in GROUPINGS:
        raise ValueError(f'Unsupported OOD grouping: {dataset}/{grouping}')
    return {
        'schema': 'defect_group_ood_v1', 'dataset': dataset, 'grouping': grouping,
        'group_definition': ('host_material' if grouping == 'material' else
                             'source_id_without_POSCAR_index'),
        'group_fractions': {'train': 0.6, 'val': 0.2, 'test': 0.2},
        'algorithm': 'sorted_groups_two_stage_train_test_split',
        'row_order': 'source_order', 'label_dependent_assignment': False,
    }


def iter_ood_splits(data, targets, random_seeds, cv5=False, *, dataset, grouping):
    if cv5:
        raise ValueError('OOD uses grouped 60/20/20 train/val/test; do not combine with --cv5')
    if len(data) != len(targets):
        raise ValueError('OOD data and target lengths differ')
    ids = [item if isinstance(item, str) else getattr(item, 'source_id', None) for item in data]
    groups = [source_group(sid, dataset, grouping) for sid in ids]
    if len(ids) != len(set(ids)):
        raise ValueError('OOD requires unique source IDs')
    unique = sorted(set(groups))
    if len(unique) < 5:
        raise ValueError('OOD requires at least five distinct groups')
    for seed in random_seeds:
        train, held = train_test_split(unique, test_size=0.4, random_state=int(seed))
        val, test = train_test_split(held, test_size=0.5, random_state=int(seed))
        split = {'display': f'seed={seed} OOD={grouping}', 'logger_id': int(seed),
                 'explain_id': f'seed_{seed}', 'seed': int(seed)}
        for part, selected in zip(PARTS, (train, val, test)):
            selected = set(selected)
            indices = [i for i, group in enumerate(groups) if group in selected]
            split[f'{part}_X'] = [data[i] for i in indices]
            split[f'{part}_y'] = [targets[i] for i in indices]
        yield split


def ood_splitter(dataset, grouping):
    protocol = protocol_for(dataset, grouping)
    def iterator(data, targets, random_seeds, cv5=False):
        return iter_ood_splits(data, targets, random_seeds, cv5=cv5, dataset=dataset, grouping=grouping)
    iterator.ood_protocol = protocol
    return iterator


def with_ood_policy(policy, split_iterator):
    protocol = getattr(split_iterator, 'ood_protocol', None)
    return {**policy, 'ood_split': protocol} if protocol else policy


def validate_ood_manifest(manifest):
    """Reject any crossing in original membership, even if filtering hides it."""
    protocol = manifest['policy'].get('ood_split')
    if protocol is None:
        return manifest
    dataset, grouping = manifest['dataset'], protocol.get('grouping')
    if protocol != protocol_for(dataset, grouping) or manifest['cv5']:
        raise ValueError('Unsupported frozen OOD split protocol')
    for saved in manifest['splits']:
        for scope in ('original', 'kept'):
            groups = {part: {source_group(sid, dataset, grouping) for sid in saved[scope][part]} for part in PARTS}
            for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')):
                if groups[a] & groups[b]:
                    raise ValueError(f'OOD group leakage: seed={saved["seed"]}/{scope}, {a}/{b}')
    return manifest


def write_ood_tables(manifest, run_dir):
    protocol = manifest['policy'].get('ood_split')
    if protocol is None:
        return
    validate_ood_manifest(manifest)
    dataset, grouping = manifest['dataset'], protocol['grouping']
    memberships, counts = [], []
    for saved in manifest['splits']:
        for part in PARTS:
            original, kept = saved['original'][part], set(saved['kept'][part])
            counts.append({'seed': saved['seed'], 'split': part,
                           'original_groups': len({source_group(sid, dataset, grouping) for sid in original}),
                           'kept_groups': len({source_group(sid, dataset, grouping) for sid in kept}),
                           'original_structures': len(original), 'kept_structures': len(kept),
                           'kept_materials': len({source_group(sid, dataset, 'material') for sid in kept})})
            memberships.extend({'seed': saved['seed'], 'split': part, 'source_id': sid,
                                'group': source_group(sid, dataset, grouping), 'kept': sid in kept}
                               for sid in original)
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(memberships).to_csv(root/f'{dataset}_ood_split_membership.csv', index=False)
    pd.DataFrame(counts).to_csv(root/f'{dataset}_ood_split_counts.csv', index=False)
    print(f'{dataset} OOD grouping={grouping}; no groups cross train/val/test. '
          '60/20/20 fractions apply to groups, not structure counts.', flush=True)
