"""Frozen-checkpoint native validation/test audit; never trains a model.

Run from HERA's parent with ``python -m HERA.scripts.compare_native_checkpoint_errors``.
Uses the ordered source IDs saved in checkpoints, not a newly generated split.
"""
import argparse
import copy
import csv
import hashlib
import json
import re
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pymatgen.core import Structure
from torch_geometric.data import Batch

from ..data.datasets import init_elem_embedding, tag_structure_source, representation_for_mode
from ..data.native_was import parse_native_defect
from ..data.structure_utils import convert_to_sparse_native
from ..main import set_seed
from ..training.trainer import MEGNetTrainer, load_trusted_checkpoint


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def identity(source_id):
    parts = source_id.split('-')
    return {
        'material': parts[1],
        'defect_type': parse_native_defect(source_id)[0],
        'defect_label': parts[2],
        'configuration_family': re.sub(r'-POSCAR\d+', '', source_id),
    }


def paired_bootstrap(frame, repetitions=20000):
    """Conditional on these trained models; does not estimate training-seed variance."""
    rng = np.random.default_rng(20260922)
    values = frame['delta'].to_numpy()
    means = np.concatenate([
        values[rng.integers(0, len(values), (min(500, repetitions-i), len(values)))].mean(axis=1)
        for i in range(0, repetitions, 500)
    ])
    grouped = frame.groupby('configuration_family')['delta'].agg(['sum', 'count'])
    cluster_means = []
    for i in range(0, repetitions, 500):
        indices = rng.integers(0, len(grouped), (min(500, repetitions-i), len(grouped)))
        cluster_means.extend((grouped['sum'].to_numpy()[indices].sum(axis=1)
                              / grouped['count'].to_numpy()[indices].sum(axis=1)).tolist())
    return {
        'delta_definition': 'hetero_was_no_dd absolute error minus attention_was absolute error, eV',
        'mean': float(values.mean()),
        'paired_sample_ci95': np.quantile(means, [.025, .975]).tolist(),
        'configuration_family_cluster_ci95': np.quantile(cluster_means, [.025, .975]).tolist(),
        'configuration_families': len(grouped),
        'replicates': repetitions,
        'limitation': 'Conditional resampling of this split and these fixed checkpoints; not across training seeds. Validation estimates also have checkpoint-selection bias.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-root', type=Path, default=ROOT/'logs/alignn_ref_v2/alignn/native')
    parser.add_argument('--data-dir', type=Path, default=ROOT.parent/'dataset/Dataset_1/Dataset_1/A_rich/Neutral')
    parser.add_argument('--output', type=Path, default=ROOT/'results/native_val_test_audit_20260922')
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--allow-reproduction-drift', action='store_true',
                        help='Continue a diagnostic run after recording saved/local MAE discrepancies')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    init_elem_embedding(ROOT/'atom_init.json')
    labels = {}
    with (args.data_dir/'id_prop_A_rich.csv').open(encoding='utf-8-sig', newline='') as handle:
        for name, target, *_ in csv.reader(handle):
            if name in labels:
                raise ValueError(f'Duplicate target ID: {name}')
            labels[name] = float(np.float32(target))
    names = ['hetero_was_no_dd', 'attention_was']
    paths = {name: args.checkpoint_root/name/'seed123_best_checkpoint.pth' for name in names}
    checkpoints = {name: load_trusted_checkpoint(path, 'cpu') for name, path in paths.items()}
    reference = checkpoints[names[0]]
    source_ids = {s: [r['source_id'] for r in reference['split'][s+'_sources']]
                  for s in ['train', 'val', 'test']}
    for split, ids in source_ids.items():
        if len(ids) != len(set(ids)) or any(sid not in labels for sid in ids):
            raise ValueError(f'Invalid {split} IDs or missing local labels')
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        if set(source_ids[a]) & set(source_ids[b]):
            raise ValueError(f'Overlapping splits: {a}/{b}')
    for name, checkpoint in checkpoints.items():
        if checkpoint['dataset_name'] != 'native' or checkpoint['config'].get('native_preprocessing') != 'reference_v1':
            raise ValueError('This audit expects native/reference_v1 checkpoints')
        for split in source_ids:
            if [r['source_id'] for r in checkpoint['split'][split+'_sources']] != source_ids[split]:
                raise ValueError(f'Ordered split differs: {name}/{split}')
        if checkpoint['split']['seed'] != reference['split']['seed']:
            raise ValueError('Seeds differ')
        for key in ['mean', 'std']:
            if float(checkpoint['scaler'][key]) != float(reference['scaler'][key]):
                raise ValueError('Target scalers differ')
    provenance = {
        'training_performed': False, 'torch': torch.__version__, 'device': args.device,
        'seed': reference['split']['seed'], 'identical_ordered_split_ids': True,
        'split_counts': {s: len(ids) for s, ids in source_ids.items()},
        'target_csv_sha256': sha256(args.data_dir/'id_prop_A_rich.csv'),
        'atom_init_sha256': sha256(ROOT/'atom_init.json'),
        'checkpoints': {},
    }
    all_predictions = []
    for name in names:
        checkpoint = checkpoints[name]
        config = copy.deepcopy(checkpoint['config'])
        if config['model']['test_batch_size'] != 1:
            raise ValueError('This audit reproduces checkpoints evaluated with batch size 1')
        set_seed(checkpoint['split']['seed'])
        trainer = MEGNetTrainer(config, args.device, seed=checkpoint['split']['seed'])
        trainer.scaler.load_state_dict(checkpoint['scaler'])
        trainer.model.load_state_dict(checkpoint['model'], strict=True)
        trainer.model.eval().requires_grad_(False)
        provenance['checkpoints'][name] = {
            'path': str(paths[name].resolve()), 'sha256': sha256(paths[name]),
            'best_epoch': checkpoint['best_epoch'], 'saved_val_mae': checkpoint['best_val_mae'],
            'saved_test_mae': checkpoint['test_mae'], 'config': config,
            'parameter_count': sum(p.numel() for p in trainer.model.parameters()),
        }
        representation = representation_for_mode(checkpoint['mode'])
        for split in ['val', 'test']:
            rows = []
            start = time.monotonic()
            for i, sid in enumerate(source_ids[split]):
                path = args.data_dir/sid
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', UserWarning)
                    raw = Structure.from_file(path)
                tag_structure_source(raw, str(path), sid)
                meta = identity(sid)
                structure = convert_to_sparse_native(
                    raw, 'vacancy' if meta['defect_type'] == 'vacancy' else 'others',
                    1, f'alignn_{representation}', None, True, False,
                    local_cutoff=config['model'].get('local_radius'), native_preprocessing='reference_v1',
                )
                structure.y = torch.tensor(labels[sid], dtype=torch.float32)
                graph = trainer.converter.convert(structure)
                batch = Batch.from_data_list([graph]).to(args.device)
                with torch.no_grad(), trainer._autocast():
                    prediction = float(trainer.scaler.inverse_transform(trainer._forward(batch)).item())
                if not np.isfinite(prediction):
                    raise ValueError(f'Non-finite prediction: {name}/{sid}')
                rows.append({
                    'model': name, 'split': split, 'source_id': sid, **meta,
                    'target': labels[sid], 'prediction': prediction,
                    'absolute_error': abs(prediction-labels[sid]),
                    'signed_error': prediction-labels[sid], 'raw_atoms': len(raw),
                    'structure_sha256': sha256(path),
                })
                if (i+1) % 100 == 0:
                    print(f'{name}/{split}: {i+1}/{len(source_ids[split])}, {time.monotonic()-start:.1f}s', flush=True)
            frame = pd.DataFrame(rows)
            frame.to_csv(args.output/f'{name}_{split}_predictions.csv', index=False)
            all_predictions.extend(rows)
            actual = float(frame.absolute_error.mean())
            saved = checkpoint['best_val_mae' if split == 'val' else 'test_mae']
            provenance['checkpoints'][name][f'recomputed_{split}_mae'] = actual
            provenance['checkpoints'][name][f'{split}_reproduction_difference'] = actual-saved
            provenance['checkpoints'][name][f'{split}_reproduced_within_0_05_mev'] = abs(actual-saved) <= 5e-5
            print(f'{name}/{split}: MAE={actual:.10f}, saved={saved:.10f}, difference={actual-saved:+.3g}', flush=True)
            write_json(args.output/'provenance.json', provenance)
            if abs(actual-saved) > 5e-5 and not args.allow_reproduction_drift:
                raise RuntimeError('Saved MAE not reproduced within 0.05 meV; inspect data/runtime before comparison')
        del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    predictions = pd.DataFrame(all_predictions)
    a = predictions[predictions.model == names[0]].drop(columns='model')
    b = predictions[predictions.model == names[1]][['split', 'source_id', 'prediction', 'absolute_error', 'structure_sha256', 'target']]
    paired = a.merge(b, on=['split', 'source_id'], suffixes=('_hetero', '_attention'), validate='one_to_one')
    assert (paired.structure_sha256_hetero == paired.structure_sha256_attention).all()
    assert (paired.target_hetero == paired.target_attention).all()
    paired['delta'] = paired.absolute_error_hetero-paired.absolute_error_attention
    paired.to_csv(args.output/'paired_errors.csv', index=False)
    summary = {'provenance': provenance, 'splits': {}, 'composition_decomposition': {}}
    for split in ['val', 'test']:
        f = paired[paired.split == split]
        shared_difficulty = (f.absolute_error_hetero + f.absolute_error_attention)/2
        worst = f.loc[shared_difficulty.sort_values(ascending=False).index]
        record = {
            'n': len(f), 'hetero_mae': float(f.absolute_error_hetero.mean()),
            'attention_mae': float(f.absolute_error_attention.mean()),
            'hetero_wins': int((f.delta < 0).sum()), 'attention_wins': int((f.delta > 0).sum()),
            'hetero_median_error': float(f.absolute_error_hetero.median()),
            'attention_median_error': float(f.absolute_error_attention.median()),
            'bootstrap': paired_bootstrap(f),
            'top_shared_difficulty_sensitivity': {
                str(k): {'remaining_n': len(f)-k, 'remaining_delta': float(worst.iloc[k:].delta.mean()),
                         'removed_delta_sum': float(worst.iloc[:k].delta.sum()),
                         'hetero_total_error_share': float(worst.iloc[:k].absolute_error_hetero.sum()/f.absolute_error_hetero.sum()),
                         'attention_total_error_share': float(worst.iloc[:k].absolute_error_attention.sum()/f.absolute_error_attention.sum())}
                for k in [1, 5, 10, 20]
            },
        }
        summary['splits'][split] = record
        worst.to_csv(args.output/f'{split}_shared_difficulty.csv', index=False)
        f.reindex(f.delta.abs().sort_values(ascending=False).index).to_csv(args.output/f'{split}_largest_model_differences.csv', index=False)
    for group in ['defect_type', 'material', 'configuration_family']:
        g = paired.groupby(['split', group]).agg(
            n=('delta', 'size'), hetero_mae=('absolute_error_hetero', 'mean'),
            attention_mae=('absolute_error_attention', 'mean'), delta=('delta', 'mean'),
        ).reset_index()
        g.to_csv(args.output/f'by_{group}.csv', index=False)
        val = g[g.split == 'val'].set_index(group)
        test = g[g.split == 'test'].set_index(group)
        if set(val.index) == set(test.index):
            pval, ptest = val.n/val.n.sum(), test.n/test.n.sum()
            summary['composition_decomposition'][group] = {
                'reference': 'validation within-group differences',
                'composition_change': float(((ptest-pval)*val.delta).sum()),
                'within_group_change': float((ptest*(test.delta-val.delta)).sum()),
            }
    train_families = {identity(sid)['configuration_family'] for sid in source_ids['train']}
    summary['split_characteristics'] = {
        s: {'defect_type_counts': dict(Counter(identity(sid)['defect_type'] for sid in ids)),
            'material_counts': dict(Counter(identity(sid)['material'] for sid in ids)),
            'target_mean': float(np.mean([labels[sid] for sid in ids])),
            'target_std': float(np.std([labels[sid] for sid in ids])),
            'configuration_family_seen_in_train': sum(identity(sid)['configuration_family'] in train_families for sid in ids)}
        for s, ids in source_ids.items()
    }
    write_json(args.output/'summary.json', summary)
    print(json.dumps({s: summary['splits'][s] for s in ['val', 'test']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
