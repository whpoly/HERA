"""Run HyperALIGNN-WAS on exactly the saved native HeteroALIGNN-WAS split."""
import argparse
import copy
import csv
import gc
import json
import math
import os
from pathlib import Path

import torch

from ..config.defaults import get_config, HYPERGRAPH_UPDATE_MODES, resolve_hypergraph_updates
from ..data.datasets import init_elem_embedding, load_data_native
from ..data.native_filter import read_manifest, manifest_identity, filtered_splits
from ..main import train_single_mode
from ..training.trainer import load_trusted_checkpoint, MEGNetTrainer
from .run_official_alignn_native import digest, metrics, write_json

ROOT = Path(__file__).resolve().parents[1]
CORE_KEYS = ('train_batch_size', 'test_batch_size', 'atom_features', 'cutoff',
             'edge_embed_size', 'embedding_size', 'nblocks', 'gcn_blocks',
             'max_neighbors', 'hetero_node_norm')


def paired_config(reference, *, hypergraph_radius=3., hypergraph_updates='local_global'):
    if reference['dataset_name'] != 'native' or reference['config']['task'] != 'alignn_hetero_was':
        raise ValueError('Expected the native HeteroALIGNN-WAS reference checkpoint.')
    if not math.isfinite(hypergraph_radius) or hypergraph_radius <= 0:
        raise ValueError('Hypergraph radius must be finite and positive.')
    resolve_hypergraph_updates('defect_global_attention_v3', 'defect_energy_mean', hypergraph_updates)
    config = get_config('alignn', 'native', 'hypergraph_was')
    for key in CORE_KEYS:
        config['model'][key] = reference['config']['model'][key]
    config['model'].update(local_radius=0, hypergraph_radius=float(hypergraph_radius),
                           hypergraph_schema='defect_global_attention_v3',
                           hypergraph_pooling='defect_energy_mean',
                           hypergraph_updates=hypergraph_updates)
    config['optim'] = copy.deepcopy(reference['config']['optim'])
    config['native_preprocessing'] = reference['config']['native_preprocessing']
    config['native_data_filter'] = copy.deepcopy(reference['config']['native_data_filter'])
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, default=ROOT / 'logs/alignn_v4/alignn/native/hetero_was_no_dd/seed123_best_checkpoint.pth')
    parser.add_argument('--filter-manifest', type=Path, default=ROOT / 'logs/alignn_v4/native_filter_manifest.json')
    parser.add_argument('--output', type=Path,
                        help='Separate result directory; required for non-default radius/update variants.')
    parser.add_argument('--hypergraph-radius', type=float, default=3.,
                        help='Local hyperedge radius in angstroms; physical graph cutoff stays fixed.')
    parser.add_argument('--hypergraph-updates', choices=HYPERGRAPH_UPDATE_MODES, default='local_global',
                        help='local excludes global incidences from both attention directions; defect identity metadata is retained.')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if not math.isfinite(args.hypergraph_radius) or args.hypergraph_radius <= 0:
        parser.error('--hypergraph-radius must be finite and positive')
    if args.output is None:
        if args.hypergraph_radius != 3. or args.hypergraph_updates != 'local_global':
            parser.error('Use --output for a new variant to preserve the existing local_global radius3 experiment')
        args.output = ROOT / 'results/native_hypergraph_paired_20260929'
    args.output = args.output.resolve()
    reference = load_trusted_checkpoint(args.reference, 'cpu')
    config = paired_config(reference, hypergraph_radius=args.hypergraph_radius,
                           hypergraph_updates=args.hypergraph_updates)
    manifest = read_manifest(args.filter_manifest)
    if manifest_identity(manifest) != config['native_data_filter']:
        raise ValueError('Filter does not match the trained hetero baseline.')
    seed = reference['split']['seed']
    kept = next(s['kept'] for s in manifest['splits'] if s['seed'] == seed)
    for part in ('train', 'val', 'test'):
        if kept[part] != [r['source_id'] for r in reference['split'][part + '_sources']]:
            raise ValueError(f'{part} split/order differs from the baseline.')
    protocol = dict(reference_sha256=digest(args.reference), config=config,
                    epochs=reference['epochs'], seed=seed, split_counts={k:len(v) for k,v in kept.items()},
                    checkpoint_selection='validation MAE', model='HyperALIGNN-WAS v3',
                    hypergraph_radius_angstrom=args.hypergraph_radius, input_features='same WAS features as hetero',
                    note='Backbone sizes, optimizer, budget and split match; hypergraph and heterogeneous message rules differ.')
    args.output.mkdir(parents=True, exist_ok=True)
    protocol_path = args.output / 'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError('Output directory has a different experiment protocol.')
    write_json(protocol_path, protocol)
    if args.prepare_only:
        print(json.dumps(protocol, indent=2))
        return
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; CPU fallback disabled.')
    torch.set_num_threads(1)
    os.chdir(ROOT.parent)
    init_elem_embedding(ROOT / 'atom_init.json')
    loaded = load_data_native('alignn', local_cutoff=0, representations={'hetero'},
                              native_preprocessing=config['native_preprocessing'], native_filter_manifest=manifest)
    dataset, targets = loaded[:-1], loaded[-1]
    split = next(filtered_splits(dataset[1], targets, manifest, [seed]))
    train_single_mode('hypergraph_was', config, dataset, targets, [seed], reference['epochs'],
                      'cuda:0', 'alignn', 'native', log_dir=str(args.output),
                      resume=True, protect_existing=True, native_filter_manifest=manifest)
    gc.collect()
    torch.cuda.empty_cache()
    path = args.output / f'seed{seed}_best_checkpoint.pth'
    trained = load_trusted_checkpoint(path, 'cpu')
    for part in ('train', 'val', 'test'):
        if trained['split'][part + '_sources'] != reference['split'][part + '_sources']:
            # Paths can differ across hosts; source ID order must still be identical.
            if [r['source_id'] for r in trained['split'][part + '_sources']] != kept[part]:
                raise ValueError('Saved split identity changed.')
    if trained['scaler'] != reference['scaler']:
        raise ValueError('Saved target scaler differs from the hetero baseline.')
    trainer = MEGNetTrainer(config, 'cuda:0', seed=seed)
    trainer.scaler.load_state_dict(trained['scaler'])
    rows, scores = [], {}
    for part in ('val', 'test'):
        mae, predictions = trainer.predict_structures(split[part + '_X'], split[part + '_y'],
                                                       trained['model'], return_predictions=True)
        part_rows = [dict(split=part, source_id=sid, target=float(target), prediction=float(pred))
                     for sid, target, pred in zip(kept[part], split[part + '_y'], predictions)]
        if len(part_rows) != len(kept[part]):
            raise ValueError('Missing predictions.')
        scores[part] = metrics(part_rows)
        expected = trained['best_val_mae'] if part == 'val' else trained['test_mae']
        if abs(scores[part]['mae_ev'] - expected) > 5e-5:
            raise ValueError('Recomputed MAE does not match the saved checkpoint.')
        rows.extend(part_rows)
    with (args.output / 'predictions.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['split', 'source_id', 'target', 'prediction'])
        writer.writeheader()
        writer.writerows(rows)
    result = dict(metrics=scores, checkpoint_sha256=digest(path), best_epoch=trained['best_epoch'],
                  epochs_completed=trained['epochs_completed'], identical_ordered_splits=True,
                  identical_scaler=True, protocol=protocol)
    write_json(args.output / 'metrics.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
