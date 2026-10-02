"""Prepare and train a native benchmark using unmodified upstream ALIGNN on CUDA.

No HERA network or graph implementation is used. The exact historical native
checkpoint/split was not published, so this is an independent reproduction.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_COMMIT = 'd1415cf824a7d6edd7a044446ae7ddbfdf55316a'  # v2023.08.01
RAW_LABELS_SHA256 = '9e994cf38c31a1694dadaecaf910f9f0ddac6826cefec92532a0d1e30f14466b'
PARTS = ('train', 'val', 'test')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def labels_at(data_dir):
    path = data_dir / 'id_prop_A_rich.csv'
    if digest(path) != RAW_LABELS_SHA256:
        raise ValueError('Raw label table differs from the verified author release.')
    rows = list(csv.reader(path.open(encoding='utf-8-sig', newline='')))
    labels = {sid: float(value) for sid, value in rows}
    if len(labels) != len(rows) or len(rows) != 3070:
        raise ValueError('Expected 3070 unique native source IDs.')
    return labels


def raw_split(ids, seed):
    """Same Python shuffle and indexing as upstream get_id_train_val_test."""
    ids = list(ids)
    random.Random(seed).shuffle(ids)
    nval = ntest = int(len(ids) * .2)
    ntrain = len(ids) - nval - ntest
    return dict(train=ids[:ntrain], val=ids[ntrain:ntrain + nval], test=ids[-ntest:])


def validate_split(splits, labels):
    seen = set()
    for part in PARTS:
        ids = splits[part]
        if not ids or len(set(ids)) != len(ids) or seen.intersection(ids):
            raise ValueError(f'Empty, duplicate or overlapping {part} IDs.')
        if set(ids) - labels.keys():
            raise ValueError(f'Unknown {part} source IDs.')
        seen.update(ids)


def prepare(args):
    labels = labels_at(args.data_dir)
    manifests = []
    for protocol in args.protocol:
        provenance = {}
        if protocol == 'raw':
            splits = raw_split(labels, args.seed)
            provenance['split_algorithm'] = 'upstream Python random.shuffle; 60/20/20'
        else:
            # Repository-owned checkpoint: only metadata is loaded, no model executes.
            import torch
            checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
            if checkpoint['dataset_name'] != 'native' or checkpoint['split']['seed'] != args.seed:
                raise ValueError('Paired checkpoint must be native and match the requested seed.')
            splits = {part: [r['source_id'] for r in checkpoint['split'][part + '_sources']]
                      for part in PARTS}
            provenance = dict(checkpoint_sha256=digest(args.checkpoint),
                              filter=checkpoint['config'].get('native_data_filter'))
        validate_split(splits, labels)
        manifest = dict(protocol=protocol, seed=args.seed, upstream_commit=UPSTREAM_COMMIT,
                        labels_sha256=RAW_LABELS_SHA256, splits=splits,
                        structure_sha256={sid: digest(args.data_dir / sid)
                                          for ids in splits.values() for sid in ids},
                        provenance=provenance)
        out = args.run_dir / f'{protocol}_seed{args.seed}'
        out.mkdir(parents=True, exist_ok=True)
        path = out / 'manifest.json'
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise ValueError(f'Refusing to replace a different manifest: {path}')
        write_json(path, manifest)
        manifests.append(path)
        print(f'{path}: ' + ', '.join(f'{k}={len(v)}' for k, v in splits.items()))
    return manifests


def config_for(manifest, out, epochs, workers):
    return dict(dataset='user_data', target='target', atom_features='cgcnn', id_tag='jid',
                neighbor_strategy='k-nearest', random_seed=manifest['seed'],
                n_train=len(manifest['splits']['train']), n_val=len(manifest['splits']['val']),
                n_test=len(manifest['splits']['test']), train_ratio=.6, val_ratio=.2, test_ratio=.2,
                keep_data_order=True, epochs=epochs, batch_size=16, weight_decay=1e-5,
                learning_rate=.001, criterion='mse', optimizer='adamw', scheduler='onecycle',
                cutoff=8., max_neighbors=12, use_canonize=True, num_workers=workers,
                pin_memory=True, save_dataloader=False, write_checkpoint=True,
                write_predictions=False, store_outputs=True, progress=True,
                standard_scalar_and_pca=False, n_early_stopping=None, output_dir=str(out),
                model=dict(name='alignn', alignn_layers=4, gcn_layers=4,
                           atom_input_features=92, edge_input_features=80,
                           triplet_input_features=40, embedding_features=64,
                           hidden_features=256, output_features=1, link='identity',
                           zero_inflated=False, classification=False))


def metrics(rows):
    if not rows:
        raise ValueError('Cannot score an empty set.')
    errors = [r['prediction'] - r['target'] for r in rows]
    if not all(math.isfinite(e) for e in errors):
        raise ValueError('Nonfinite predictions; no silent filtering is allowed.')
    return dict(n=len(rows), mae_ev=sum(abs(e) for e in errors) / len(rows),
                rmse_ev=math.sqrt(sum(e * e for e in errors) / len(rows)))


def check_upstream(source):
    actual = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != UPSTREAM_COMMIT:
        raise ValueError(f'Expected ALIGNN {UPSTREAM_COMMIT}, got {actual}')
    dirty = subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain',
                                     '--untracked-files=no', '--', 'alignn'], text=True).strip()
    if dirty:
        raise ValueError('Upstream ALIGNN source has tracked modifications.')
    sys.path.insert(0, str(source.resolve()))


def train(args):
    check_upstream(args.alignn_source)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is required. CPU fallback is disabled.')
    if args.disable_graphbolt:
        # DGL 2.2 wheels eagerly import an optional extension built against old
        # PyTorch. Original ALIGNN uses DGLGraph, not GraphBolt or distributed
        # sampling. Keep the native DGL graph kernels and upstream model intact.
        sys.modules['dgl.graphbolt'] = types.ModuleType('dgl.graphbolt')
    from alignn.config import TrainingConfig
    from alignn.data import get_train_val_loaders
    from alignn.models.alignn import ALIGNN
    from alignn.train import train_dgl
    from jarvis.core.atoms import Atoms
    from torch.utils.data import DataLoader
    labels = labels_at(args.data_dir)
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    validate_split(manifest['splits'], labels)
    if manifest['upstream_commit'] != UPSTREAM_COMMIT or manifest['labels_sha256'] != RAW_LABELS_SHA256:
        raise ValueError('Manifest source/version mismatch.')
    out = args.manifest.parent.resolve() / 'training'
    if out.exists() and any(out.iterdir()):
        raise ValueError(f'Training output is nonempty; use a fresh run directory: {out}')
    out.mkdir(parents=True, exist_ok=True)
    config = TrainingConfig(**config_for(manifest, out, args.epochs, args.workers))
    provenance = dict(upstream_commit=UPSTREAM_COMMIT, manifest_sha256=digest(args.manifest),
                      gpu=torch.cuda.get_device_name(0), packages={
                          name: importlib.metadata.version(name) for name in
                          ('torch', 'dgl', 'jarvis-tools', 'pytorch-ignite', 'pydantic', 'numpy')},
                      independent_reproduction=True, original_paper_split_available=False,
                      unused_graphbolt_disabled=args.disable_graphbolt,
                      cif_parser='JARVIS native parser; use_cif2cell=False',
                      validation_selection='upstream validation MAE, with original drop_last=True',
                      reporting='complete validation/test sets for both best-val-MAE and last epoch')
    write_json(out / 'provenance.json', provenance)
    dataset = []
    for part in PARTS:
        for sid in manifest['splits'][part]:
            path = args.data_dir / sid
            if digest(path) != manifest['structure_sha256'][sid]:
                raise ValueError(f'Structure changed: {sid}')
            dataset.append(dict(jid=sid, target=labels[sid],
                                atoms=Atoms.from_cif(str(path), use_cif2cell=False).to_dict()))
    loaders = get_train_val_loaders(
        dataset_array=dataset, target='target', atom_features='cgcnn', id_tag='jid',
        n_train=config.n_train, n_val=config.n_val, n_test=config.n_test,
        batch_size=16, standardize=False, line_graph=True, split_seed=manifest['seed'],
        workers=args.workers, pin_memory=True, save_dataloader=False, use_canonize=True,
        cutoff=8., max_neighbors=12, standard_scalar_and_pca=False, keep_data_order=True,
        output_dir=str(out))
    actual = json.loads((out / 'ids_train_val_test.json').read_text())
    for part in PARTS:
        if actual['id_' + part] != manifest['splits'][part]:
            raise ValueError(f'Upstream changed {part} split/order.')
    train_dgl(config, train_val_test_loaders=list(loaders))
    # Official training exports last-epoch predictions by default. Explicitly score
    # both last and validation-selected weights to avoid mislabelling either one.
    history = json.loads((out / 'history_val.json').read_text())
    if len(history['mae']) != args.epochs:
        raise ValueError('Training did not complete the specified epoch budget.')
    last = out / f'checkpoint_{args.epochs}.pt'
    if not last.exists():
        raise FileNotFoundError(f'Expected upstream final checkpoint: {last}')
    score = dict(protocol=manifest['protocol'], seed=manifest['seed'], epochs=args.epochs,
                 independent_reproduction=True, results={})
    for tag, checkpoint_path in [('best_val_mae', out / 'best_model.pt'), ('last_epoch', last)]:
        net = ALIGNN(config.model).cuda()
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        net.load_state_dict(checkpoint['model'], strict=True)
        net.eval()
        predicted = []
        for part, loader in [('val', loaders[1]), ('test', loaders[2])]:
            complete = DataLoader(loader.dataset, batch_size=1, shuffle=False,
                                  collate_fn=loader.collate_fn, drop_last=False)
            expected = manifest['splits'][part]
            part_rows = []
            with torch.no_grad():
                for sid, batch in zip(expected, complete):
                    inputs, target = loaders[3](batch, device='cuda')
                    value = float(net(inputs).reshape(-1).item())
                    if abs(float(target.item()) - labels[sid]) > 1e-4:
                        raise ValueError(f'Target alignment changed: {sid}')
                    part_rows.append(dict(split=part, source_id=sid, target=labels[sid], prediction=value))
            if len(part_rows) != len(expected):
                raise ValueError('Evaluation dropped samples.')
            score['results'][tag + '_' + part] = metrics(part_rows)
            predicted.extend(part_rows)
        with (out / f'{tag}_predictions.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['split', 'source_id', 'target', 'prediction'])
            writer.writeheader()
            writer.writerows(predicted)
        del net
    score['checkpoint_sha256'] = {p.name: digest(p) for p in (out / 'best_model.pt', last)}
    write_json(out / 'metrics.json', score)
    print(json.dumps(score, indent=2))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    default_data = ROOT.parent / 'dataset/Dataset_1/Dataset_1/A_rich/Neutral'
    prep = sub.add_parser('prepare', help='Metadata/file checks only; no training or inference')
    prep.add_argument('--data-dir', type=Path, default=default_data)
    prep.add_argument('--run-dir', type=Path, default=ROOT / 'results/official_alignn_native')
    prep.add_argument('--protocol', nargs='+', choices=['raw', 'paired'], default=['raw', 'paired'])
    prep.add_argument('--seed', type=int, default=123)
    prep.add_argument('--checkpoint', type=Path, default=ROOT / 'logs/alignn_v4/alignn/native/hetero_was_no_dd/seed123_best_checkpoint.pth')
    run = sub.add_parser('train', help='CUDA training; unmodified upstream model and trainer')
    run.add_argument('--data-dir', type=Path, default=default_data)
    run.add_argument('--manifest', type=Path, required=True)
    run.add_argument('--alignn-source', type=Path, required=True)
    run.add_argument('--epochs', type=int, default=150)
    run.add_argument('--workers', type=int, default=0)
    run.add_argument('--disable-graphbolt', action='store_true',
                     help='Disable unused GraphBolt import for newer PyTorch; native DGL ops remain unchanged')
    return parser.parse_args(argv)


if __name__ == '__main__':
    options = parse_args()
    prepare(options) if options.action == 'prepare' else train(options)
