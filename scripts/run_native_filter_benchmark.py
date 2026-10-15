"""Compare native filtering and robust loss using frozen, auditable sample lists.

The filtered_dataset suite freezes one screened population before creating
fresh splits. Earlier suites fit thresholds on training data only and retain
their original held-out membership. Filters do not assert DFT labels are wrong.
"""
import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / 'dataset/Dataset_1/Dataset_1/A_rich/Neutral'
UPPER_EV = 15.0
VARIANTS = ('strict', 'train_lt15')
MODES = ('hetero_was', 'was_x')
FAMILY_POLICY = dict(min_train_samples=4, mad_sigma_multiplier=6.0, min_excess_ev=1.0,
                     mad_normalization=1.4826, direction='upper only', refit_iterations=1)


def variant_specs(args):
    if args.suite == 'filtered_dataset':
        return {
            'population_mse': dict(training_filter='prefiltered', point_loss='mse'),
            'population_huber': dict(training_filter='prefiltered', point_loss='huber_ev'),
        }
    if args.suite == 'outlier_control':
        return {
            'strict': dict(training_filter='strict', point_loss='mse'),
            'family_mad': dict(training_filter='family_mad', point_loss='mse'),
            'strict_huber': dict(training_filter='strict', point_loss='huber_ev'),
            'family_mad_huber': dict(training_filter='family_mad', point_loss='huber_ev'),
        }
    return {name: dict(training_filter=name, point_loss='mse') for name in VARIANTS}


def save_json(path, value, frozen=False):
    path = Path(path)
    if frozen and path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != value:
            raise ValueError(f'Different experiment identity; choose a new --run-dir: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    def scalar(item):
        if hasattr(item, 'item'):
            return item.item()
        raise TypeError(f'Cannot serialize {type(item).__name__}')
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=scalar) + '\n', encoding='utf-8')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def derive_training_scope(base):
    """Restrict only the frozen training IDs; never refit fences or inspect errors."""
    from ..data.native_filter import manifest_identity, validate_manifest
    from ..data.native_preprocessing import native_filter_policy

    validate_manifest(base)
    if base['policy'] != native_filter_policy('strict') or base['cv5']:
        raise ValueError('Expected the unmodified strict random-split native manifest')
    result = copy.deepcopy(base)
    result['policy']['training_energy_scope'] = dict(
        schema='native_training_energy_scope_v1', upper_exclusive_ev=UPPER_EV,
        applies_to=['train'], parent_manifest_sha256=manifest_identity(base)['sha256'],
        interpretation='Exploratory energy scope; not confirmed erroneous DFT labels',
    )
    by_id = {row['source_id']: row for row in base['records']}
    for split in result['splits']:
        keep = []
        removed = {row['source_id']: row for row in split['excluded']['train']}
        for sid in split['kept']['train']:
            value = by_id[sid]['target']
            if value is None or not math.isfinite(value):
                raise ValueError(f'Invalid retained training label: {sid}')
            if value < UPPER_EV:
                keep.append(sid)
            else:
                removed[sid] = dict(source_id=sid, reasons=['outside_training_energy_scope_ge_15_ev'])
        split['kept']['train'] = keep
        split['excluded']['train'] = [removed[sid] for sid in split['original']['train'] if sid in removed]
    return validate_manifest(result)


def derive_family_mad(base, filter_splits=('train',)):
    """Fit once on strict train; apply frozen fences to the declared splits."""
    from ..data.native_filter import manifest_identity, validate_manifest
    from ..data.native_preprocessing import native_filter_policy
    from ..data.native_quality import configuration_family

    validate_manifest(base)
    filter_splits = tuple(filter_splits)
    if filter_splits not in (('train',), ('train', 'val', 'test')):
        raise ValueError('Family MAD scope must be train only or all three splits')
    if base['policy'] != native_filter_policy('strict') or base['cv5']:
        raise ValueError('Expected the unmodified strict random-split native manifest')
    result = copy.deepcopy(base)
    all_splits = len(filter_splits) == 3
    result['policy']['family_mad_filter' if all_splits else 'training_family_mad'] = dict(
        schema='native_family_mad_all_splits_v1' if all_splits else 'native_training_family_mad_v1',
        **FAMILY_POLICY, applies_to=list(filter_splits),
        parent_manifest_sha256=manifest_identity(base)['sha256'],
        interpretation='Statistical upper-tail sensitivity; not confirmed bad DFT data')
    if all_splits:
        result['policy']['family_mad_filter']['fit_on'] = 'strict retained train only'
    records = {row['source_id']: row for row in base['records']}
    for split in result['splits']:
        groups = {}
        for sid in split['kept']['train']:
            value = records[sid]['target']
            if value is None or not math.isfinite(value):
                raise ValueError(f'Invalid retained training label: {sid}')
            groups.setdefault(configuration_family(sid), []).append(value)
        fences = {}
        for family, values in sorted(groups.items()):
            fence = dict(n_train=len(values), status='insufficient_train_samples')
            if len(values) >= FAMILY_POLICY['min_train_samples']:
                median = statistics.median(values)
                sigma = FAMILY_POLICY['mad_normalization'] * statistics.median(abs(v-median) for v in values)
                fence.update(status='fitted', median_ev=median, mad_sigma_ev=sigma,
                             upper_ev=median+max(FAMILY_POLICY['mad_sigma_multiplier']*sigma,
                                                 FAMILY_POLICY['min_excess_ev']))
            fences[family] = fence
        split['training_family_mad_fences'] = fences
        for part in filter_splits:
            removed = {row['source_id']: row for row in split['excluded'][part]}
            keep = []
            for sid in split['kept'][part]:
                if family_flagged(sid, records[sid]['target'], fences):
                    removed[sid] = dict(source_id=sid, reasons=['above_strict_train_family_mad_upper'])
                else:
                    keep.append(sid)
            split['kept'][part] = keep
            split['excluded'][part] = [removed[sid] for sid in split['original'][part] if sid in removed]
    return validate_manifest(result)


def family_flagged(sid, target, fences):
    from ..data.native_quality import configuration_family
    fence = fences.get(configuration_family(sid), {})
    return fence.get('status') == 'fitted' and target > fence['upper_ev']


def write_family_review(base, scoped, output):
    from ..data.native_quality import configuration_family
    records = {r['source_id']: r for r in base['records']}
    splits = {str(s['logger_id']): s for s in base['splits']}
    for split in scoped['splits']:
        rows = []
        for part, ids in splits[str(split['logger_id'])]['kept'].items():
            for sid in ids:
                record = records[sid]
                fence = split['training_family_mad_fences'].get(configuration_family(sid), {})
                flagged = family_flagged(sid, record['target'], split['training_family_mad_fences'])
                rows.append(dict(split=part, source_id=sid, family=configuration_family(sid),
                                 target_ev=record['target'], n_train=fence.get('n_train', 0),
                                 status=fence.get('status', 'unseen_family'),
                                 median_ev=fence.get('median_ev'), mad_sigma_ev=fence.get('mad_sigma_ev'),
                                 upper_ev=fence.get('upper_ev'), flagged=flagged,
                                 removed_from_training=part == 'train' and flagged,
                                 removed_from_split=sid not in split['kept'][part],
                                 min_pair_covalent_ratio=record.get('min_pair_covalent_ratio')))
        write_csv(output / f'seed{split["seed"]}_family_mad_review.csv', rows)


def build_command(args, variant):
    directory = args.run_dir / variant
    command = [sys.executable, '-u', '-m', 'HERA.main', '--model', 'alignn',
            '--dataset', 'native', '--mode', *MODES,
            '--native-preprocessing', 'reference_v1',
            '--native-filter-manifest', str(directory / 'native_filter_manifest.json'),
            '--alignn-hetero-dd', 'drop', '--r', '0',
            '--alignn-hetero-feature-norm', 'layernorm', '--alignn-hetero-node-norm', 'layernorm',
            '--alignn-hetero-relations', 'shared_residual', '--alignn-hetero-adapter-rank', '8',
            '--alignn-hetero-pooling', 'defect_energy_mean', '--alignn-embedding-size', '64',
            '--seed', *map(str, args.seed), '--epochs', str(args.epochs),
            '--early-stopping-patience', '0', '--alignn-train-batch-size', str(args.batch_size),
            '--alignn-test-batch-size', '1', '--device', args.device,
            '--atom-init', str(ROOT / 'atom_init.json'), '--run-dir', str(directory),
            '--compact-logs', '--resume']
    if variant_specs(args)[variant]['point_loss'] == 'huber_ev':
        command += ['--point-loss', 'huber_ev', '--huber-delta-ev', '1.0']
    return command


def prepare(args):
    from ..data.native_filter import read_manifest
    from ..data.native_preprocessing import prepare_native_filter, write_manifest_tables
    from ..main import iter_train_val_test_splits

    variants = variant_specs(args)
    protocol = dict(schema='native_training_filter_benchmark_v1', modes=list(MODES),
                    variants=list(variants), seeds=args.seed, epochs=args.epochs,
                    batch_size=args.batch_size, upper_exclusive_ev=UPPER_EV,
                    additional_filter_applies_to='train only',
                    epoch_selection='full strict validation MAE',
                    report_scopes=['strict_all', 'target_lt15', 'target_ge15'],
                    architecture='unchanged hidden64 defect_energy_mean no_dd')
    if args.suite == 'outlier_control':
        protocol.pop('upper_exclusive_ev')
        protocol.update(schema='native_outlier_control_benchmark_v1', variants=variants,
                        family_policy=FAMILY_POLICY, huber_delta_ev=1.0, huber_multiplier=2.0,
                        report_scopes=['strict_all', 'family_high', 'family_not_flagged'])
    elif args.suite == 'filtered_dataset':
        protocol.pop('upper_exclusive_ev')
        from ..data.native_population import population_policy
        protocol.update(schema=('native_filter_then_split_benchmark_v4' if args.exclude_poscar0 else
                                'native_filter_then_split_benchmark_v3' if args.max_energy_ev is not None
                                else 'native_filter_then_split_benchmark_v2'), variants=variants,
                        population_policy=population_policy(args.max_energy_ev, args.exclude_poscar0), huber_delta_ev=1.0, huber_multiplier=2.0,
                        additional_filter_applies_to='entire population before splitting',
                        threshold_fit='full eligible population; independent of seed',
                        epoch_selection='filtered validation MAE',
                        report_scopes=['filtered_all'], split_membership='fresh 60/20/20 after population freeze')
    protocol_path = args.run_dir / 'filter_protocol.json'
    if args.evaluate_only and (not protocol_path.exists() or any(
            not (args.run_dir / v / 'native_filter_manifest.json').exists() for v in variants)):
        raise FileNotFoundError('Evaluation requires an already prepared experiment')
    save_json(protocol_path, protocol, frozen=True)
    if args.suite == 'filtered_dataset':
        from ..data.native_population import prepare_population, split_population
        if args.evaluate_only and not (args.run_dir / 'native_population_filter.json').exists():
            raise FileNotFoundError('Evaluation requires a frozen pre-split population')
        population = prepare_population(args.run_dir, DATA, max_energy_ev=args.max_energy_ev,
                                        exclude_poscar0=args.exclude_poscar0)
        cohorts = dict(prefiltered=split_population(population, args.seed, iter_train_val_test_splits))
    else:
        base = prepare_native_filter(args.run_dir / 'strict', 'strict', args.seed, False,
                                     iter_train_val_test_splits, data_dir=DATA)
        scoped = (dict(family_mad=derive_family_mad(base)) if args.suite == 'outlier_control'
                  else dict(train_lt15=derive_training_scope(base)))
        cohorts = dict(strict=base, **scoped)
    manifests = {}
    for name, spec in variants.items():
        path = args.run_dir / name / 'native_filter_manifest.json'
        save_json(path, cohorts[spec['training_filter']], frozen=True)
        manifests[name] = read_manifest(path)
        write_manifest_tables(manifests[name], path.parent, prefix='native_filter_')
    if args.suite == 'outlier_control':
        write_family_review(base, scoped['family_mad'], args.run_dir)
    for name, manifest in manifests.items():
        for split in manifest['splits']:
            print(f'{name}/seed{split["seed"]}: ' + ', '.join(
                f'{part}={len(split["kept"][part])}' for part in ('train', 'val', 'test')), flush=True)
    return manifests


def validate_checkpoint(checkpoint, manifest, mode, seed, epochs, batch_size, point_loss='mse'):
    from ..data.native_filter import manifest_identity, requested_splits

    if (checkpoint['dataset_name'] != 'native' or checkpoint['mode'] != mode
            or checkpoint['split']['seed'] != seed
            or checkpoint['epochs'] != epochs or checkpoint['epochs_completed'] != epochs
            or checkpoint.get('early_stopping_patience', 0) != 0):
        raise ValueError('Checkpoint identity or completed training budget differs')
    split = requested_splits(manifest, [seed])[0]
    for part in ('train', 'val', 'test'):
        actual = [row['source_id'] for row in checkpoint['split'][part + '_sources']]
        if actual != split['kept'][part]:
            raise ValueError(f'Checkpoint ordered {part} sources differ from frozen manifest')
    config = copy.deepcopy(checkpoint['config'])
    if config.pop('native_data_filter', None) != manifest_identity(manifest):
        raise ValueError('Checkpoint native filter identity differs')
    model = config['model']
    if (config.get('native_preprocessing') != 'reference_v1'
            or model['embedding_size'] != 64 or model['train_batch_size'] != batch_size
            or model['test_batch_size'] != 1):
        raise ValueError('Checkpoint architecture, preprocessing or batch size differs')
    if (mode == 'hetero_was' and any(model.get(k) != v for k, v in {
            'hetero_pooling': 'defect_energy_mean', 'hetero_dd_mode': 'drop',
            'hetero_relation_mode': 'shared_residual', 'hetero_relation_rank': 8,
            'hetero_node_norm': 'layernorm', 'hetero_feature_norm': 'layernorm',
            'local_radius': 0}.items())):
        raise ValueError('Unexpected hetero architecture')
    scaler = checkpoint['scaler']
    if not all(math.isfinite(float(scaler[k])) for k in ('mean', 'std')) or float(scaler['std']) <= 0:
        raise ValueError('Invalid checkpoint target scaler')
    optim = config.setdefault('optim', {})
    if optim.get('point_loss', 'mse') != point_loss:
        raise ValueError('Checkpoint point loss differs from declared experiment')
    if point_loss == 'huber_ev' and optim.get('huber_delta_ev') != 1.0:
        raise ValueError('Checkpoint Huber transition differs from declared experiment')
    if point_loss == 'mse' and 'huber_delta_ev' in optim:
        raise ValueError('Unexpected Huber transition in MSE checkpoint')
    optim.pop('point_loss', None)
    optim.pop('huber_delta_ev', None)
    return config


def predict(checkpoint, source_ids, records, device):
    """Reproduce the repository's native checkpoint evaluation, batch size one."""
    import warnings
    import numpy as np
    import torch
    from pymatgen.core import Structure
    from torch_geometric.data import Batch
    from ..data.datasets import init_elem_embedding, tag_structure_source, representation_for_mode
    from ..data.native_filter import verify_source_file
    from ..data.native_was import parse_native_defect
    from ..data.structure_utils import convert_to_sparse_native
    from ..main import set_seed
    from ..training.trainer import MEGNetTrainer

    init_elem_embedding(ROOT / 'atom_init.json')
    seed = checkpoint['split']['seed']
    set_seed(seed)
    config = copy.deepcopy(checkpoint['config'])
    trainer = MEGNetTrainer(config, device, seed=seed)
    trainer.scaler.load_state_dict(checkpoint['scaler'])
    trainer.model.load_state_dict(checkpoint['model'], strict=True)
    trainer.model.eval().requires_grad_(False)
    representation = representation_for_mode(checkpoint['mode'])
    rows = []
    for part, ids in source_ids.items():
        for index, sid in enumerate(ids, 1):
            path, record = DATA / sid, records[sid]
            verify_source_file(path, record)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                raw = Structure.from_file(path)
            tag_structure_source(raw, str(path), sid)
            kind = parse_native_defect(sid)[0]
            structure = convert_to_sparse_native(
                raw, 'vacancy' if kind == 'vacancy' else 'others',
                1, f'alignn_{representation}', None, True, False,
                local_cutoff=config['model'].get('local_radius'), native_preprocessing='reference_v1')
            target = float(np.float32(record['target']))
            structure.y = torch.tensor(target, dtype=torch.float32)
            batch = Batch.from_data_list([trainer.converter.convert(structure)]).to(device)
            with torch.no_grad(), trainer._autocast():
                value = float(trainer.scaler.inverse_transform(trainer._forward(batch)).item())
            if not math.isfinite(value):
                raise ValueError(f'Nonfinite prediction: {sid}')
            rows.append(dict(split=part, source_id=sid, target=target, source_target=record['target'],
                             prediction=value, structure_sha256=record['sha256']))
            if index % 100 == 0:
                print(f'{checkpoint["mode"]}/seed{seed}/{part}: {index}/{len(ids)}', flush=True)
    return rows


def score(rows):
    if not rows:
        return dict(n=0, mae_ev=None, rmse_ev=None)
    errors = [float(row['prediction']) - float(row['target']) for row in rows]
    if not all(math.isfinite(e) for e in errors):
        raise ValueError('Nonfinite prediction/target in metrics')
    return dict(n=len(errors), mae_ev=sum(abs(e) for e in errors) / len(errors),
                rmse_ev=math.sqrt(sum(e * e for e in errors) / len(errors)))


def evaluate(args, manifests):
    import torch
    from ..data.native_filter import requested_splits
    from ..training.trainer import load_trusted_checkpoint

    metrics, provenance, reference_configs, scalers = [], {}, {}, {}
    variants = variant_specs(args)
    reference_manifest = manifests[next(iter(variants))]
    if args.suite == 'filtered_dataset':
        from ..data.native_filter import PREFILTER_SCHEMA, manifest_identity, validate_manifest
        from ..data.native_population import population_policy
        validate_manifest(reference_manifest)
        expected_identity = manifest_identity(reference_manifest)
        if reference_manifest['schema'] != PREFILTER_SCHEMA:
            raise ValueError('Filtered-dataset evaluation requires filtering before splitting')
        if reference_manifest['policy'] != population_policy(args.max_energy_ev, args.exclude_poscar0):
            raise ValueError('Evaluation population policy differs from the requested population')
        if any(manifest_identity(manifests[name]) != expected_identity for name in variants):
            raise ValueError('Filtered-dataset variants must share the identical frozen manifest')
    records = {r['source_id']: r for r in reference_manifest['records']}
    for variant, spec in variants.items():
        for seed in args.seed:
            split = requested_splits(manifests[variant], [seed])[0]
            sources = {p: split['kept'][p] for p in ('val', 'test')}
            for mode in MODES:
                directory = args.run_dir / variant / 'alignn/native' / (mode + '_no_dd' if mode == 'hetero_was' else mode)
                path = directory / f'seed{seed}_best_checkpoint.pth'
                checkpoint = load_trusted_checkpoint(path, 'cpu')
                config = validate_checkpoint(checkpoint, manifests[variant], mode, seed, args.epochs,
                                             args.batch_size, point_loss=spec['point_loss'])
                key = (mode, seed)
                if key in reference_configs and config != reference_configs[key]:
                    raise ValueError(f'Non-ablation configuration changed: {mode}/seed{seed}')
                reference_configs[key] = config
                # Filtering changes the training scaler legitimately. Modes within a cohort must agree.
                scaler_key = (spec['training_filter'], seed)
                if scaler_key in scalers and checkpoint['scaler'] != scalers[scaler_key]:
                    raise ValueError(f'Models use different target scalers within {variant}/seed{seed}')
                scalers[scaler_key] = checkpoint['scaler']
                rows = predict(checkpoint, sources, records, args.device)
                for part, ids in sources.items():
                    subset = [r for r in rows if r['split'] == part]
                    if [r['source_id'] for r in subset] != ids:
                        raise ValueError(f'Incomplete or reordered {part} predictions')
                    actual = score(subset)['mae_ev']
                    saved = float(checkpoint['best_val_mae' if part == 'val' else 'test_mae'])
                    if not math.isfinite(saved) or abs(actual - saved) > 5e-5:
                        raise ValueError(f'{variant}/{mode}/{part}: saved MAE not reproduced within 0.05 meV')
                    scopes = (
                            ('strict_all', subset),
                            ('target_lt15', [r for r in subset if r['source_target'] < UPPER_EV]),
                            ('target_ge15', [r for r in subset if r['source_target'] >= UPPER_EV]))
                    if args.suite == 'outlier_control':
                        fences = requested_splits(manifests['family_mad'], [seed])[0]['training_family_mad_fences']
                        flags = [family_flagged(r['source_id'], r['source_target'], fences) for r in subset]
                        scopes = (('strict_all', subset),
                                  ('family_high', [r for r, flag in zip(subset, flags) if flag]),
                                  ('family_not_flagged', [r for r, flag in zip(subset, flags) if not flag]))
                    elif args.suite == 'filtered_dataset':
                        scopes = (('filtered_all', subset),)
                    for scope, selected in scopes:
                        metrics.append(dict(variant=variant, mode=mode, seed=seed, split=part,
                                            scope=scope, train_count=len(split['kept']['train']), **score(selected)))
                output = directory / f'seed{seed}_filter_predictions.csv'
                write_csv(output, rows)
                provenance[f'{variant}/{mode}/seed{seed}'] = dict(
                    checkpoint_sha256=sha256(path), predictions_sha256=sha256(output),
                    best_epoch=checkpoint['best_epoch'], scaler=checkpoint['scaler'], config=checkpoint['config'])
                del checkpoint
    write_csv(args.run_dir / 'filter_comparison.csv', metrics)
    validation_scope = 'filtered' if args.suite == 'filtered_dataset' else 'full strict'
    save_json(args.run_dir / 'filter_evaluation.json', dict(
        status='complete', source_csv_sha256=sha256(DATA / 'id_prop_A_rich.csv'),
        atom_init_sha256=sha256(ROOT / 'atom_init.json'), device=args.device, torch_version=torch.__version__,
        selection=f'All runs reported; epochs selected by {validation_scope} validation MAE',
        checkpoints=provenance, metrics=metrics))
    description = ('All runs share the same newly filtered train/validation/test population.'
                   if args.suite == 'filtered_dataset' else
                   'Declared training-filter/loss changes only. Validation/test are fixed strict splits.')
    lines = ['# Native filter comparison', '', description,
             'Filters are exploratory statistical scopes, not confirmed erroneous DFT labels.',
             f'All runs are reported; checkpoints use {validation_scope} validation MAE. No test-based selection.', '',
             '| Training | Model | Seed | Evaluation | Scope | N | MAE (eV) | RMSE (eV) |',
             '|---|---|---:|---|---|---:|---:|---:|']
    for row in metrics:
        values = ['n/a' if row[k] is None else f'{row[k]:.6f}' for k in ('mae_ev', 'rmse_ev')]
        lines.append(f'| {row["variant"]} | {row["mode"]} | {row["seed"]} | {row["split"]} | '
                     f'{row["scope"]} | {row["n"]} | {values[0]} | {values[1]} |')
    if args.suite == 'filtered_dataset':
        lines += ['', 'The entire population was screened and frozen BEFORE creating fresh 60/20/20 splits.',
                  'Curation statistics use the full eligible population and do not depend on the split seed.',
                  'The target scaler and model are fitted on the new training split only.',
                  'These scores describe the new filtered dataset, not the previous 604-sample test population.',
                  'Compare models and losses within this new shared population; do not attribute a change',
                  'from historical unfiltered scores solely to better learning.']
    else:
        lines += ['', 'Compare training variants within the SAME evaluation scope to measure a retraining effect.',
                  'Comparing full and subgroup scores changes the evaluation population.',
                  'The same epoch budget has fewer optimizer updates after training filtering.',
                  'Excluded training regimes remain present in full validation/test evaluation.']
    lines += [
              'Huber may improve typical errors while worsening tail RMSE; inspect both metrics.',
              'One seed is exploratory. No additional filter threshold was searched.', '']
    if args.suite == 'filtered_dataset' and args.max_energy_ev is not None:
        lines += [f'The frozen dataset retains E <= {args.max_energy_ev:g} eV; no lower energy bound is applied.',
                  'This bound was applied after strict/MAD screening and before generating fresh splits.', '']
    if args.suite == 'filtered_dataset' and args.exclude_poscar0:
        lines += ['All filenames containing -POSCAR0- were excluded from the population before splitting.',
                  'Remaining frames are not necessarily fully relaxed structures.', '']
    (args.run_dir / 'filter_comparison.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines), flush=True)


def write_csv(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=['energy_scope', 'outlier_control', 'filtered_dataset'], default='energy_scope',
                        help='energy_scope: <15 train cap; outlier_control: train MAD x loss; '
                             'filtered_dataset: screen full population, then split 60/20/20 and compare losses')
    parser.add_argument('--seed', nargs='+', type=int, default=[123])
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-energy-ev', type=float, default=None,
                        help='filtered_dataset only: retain E <= this bound after strict/MAD and before splitting')
    parser.add_argument('--exclude-poscar0', action='store_true',
                        help='filtered_dataset only: exclude the exact -POSCAR0- token after other filters, before splitting')
    parser.add_argument('--run-dir', type=Path, default=None)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--dry-run', action='store_true')
    group.add_argument('--prepare-only', action='store_true')
    group.add_argument('--evaluate-only', action='store_true')
    args = parser.parse_args(argv)
    if min(args.seed) < 0 or args.epochs < 1 or args.batch_size < 1:
        parser.error('Nonnegative seeds and positive epochs/batch size required')
    if args.max_energy_ev is not None and (args.suite != 'filtered_dataset' or not math.isfinite(args.max_energy_ev)):
        parser.error('--max-energy-ev requires filtered_dataset and a finite value')
    if args.exclude_poscar0 and args.suite != 'filtered_dataset':
        parser.error('--exclude-poscar0 requires filtered_dataset')
    args.seed = list(dict.fromkeys(args.seed))
    if args.run_dir is None:
        directories = dict(energy_scope='native_training_filter_v1',
                           outlier_control='native_outlier_control_v1',
                           filtered_dataset='native_filter_then_split_v2')
        args.run_dir = ROOT / 'logs' / directories[args.suite]
        if args.max_energy_ev is not None:
            bound = f'{args.max_energy_ev:g}'.replace('-', 'minus').replace('.', 'p')
            args.run_dir = ROOT / 'logs' / f'native_filter_then_split_le{bound}_v3'
        if args.exclude_poscar0:
            energy = f'_le{bound}' if args.max_energy_ev is not None else ''
            args.run_dir = ROOT / 'logs' / f'native_filter_then_split{energy}_no_poscar0_v4'
    args.run_dir = args.run_dir.resolve()
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.dry_run:
        print(f'No preprocessing/training/inference. Freeze manifests for {args.suite}; run:')
        if args.suite == 'filtered_dataset':
            print('Screen the full population first, freeze retained IDs, THEN create fresh seeded 60/20/20 splits.')
            if args.max_energy_ev is not None:
                print(f'After strict/MAD, retain E <= {args.max_energy_ev:g} eV before splitting; no lower bound.')
            if args.exclude_poscar0:
                print('Then exclude all -POSCAR0- filenames from the population BEFORE splitting.')
        for variant in variant_specs(args):
            print(shlex.join(build_command(args, variant)))
        print('Then evaluate both models on the same newly filtered validation/test sets.' if args.suite == 'filtered_dataset'
              else 'Then evaluate both models on the same full validation/test sets and fixed diagnostic subgroups.')
        return
    previous = Path.cwd()
    try:
        os.chdir(ROOT.parent)
        manifests = prepare(args)
        if args.prepare_only:
            return
        if not args.evaluate_only:
            for variant in variant_specs(args):
                command = build_command(args, variant)
                print(shlex.join(command), flush=True)
                subprocess.run(command, cwd=ROOT.parent, check=True)
        evaluate(args, manifests)
    finally:
        os.chdir(previous)


if __name__ == '__main__':
    main()
