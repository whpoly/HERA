"""Check MEGNet sparse repeatability in independent Python processes.

Uses a trusted HERA checkpoint for configuration, split and scaler. Tests
saved-weight inference, then fresh initialization with fixed-LR FP32 updates.
This diagnostic does not alter the model or resume a benchmark training run.
"""
import argparse
from datetime import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import uuid

ROOT = Path(__file__).resolve().parent


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def tensor_hash(tensors):
    """Hash values, names, shapes and dtypes; device is intentionally excluded."""
    digest = hashlib.sha256()
    for key, value in sorted(tensors.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(json.dumps([key, str(tensor.dtype), list(tensor.shape)]).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def plain_cpu(value):
    import torch
    # MyTensor can propagate from graph bond_batch into predictions/gradients.
    # Keep audit snapshots as ordinary tensors for weights_only deserialization.
    return value.detach().as_subclass(torch.Tensor).cpu().clone()


def model_state(model):
    return {key: plain_cpu(value) for key, value in model.state_dict().items()}


def gradients(model):
    return {key: plain_cpu(p.grad) for key, p in model.named_parameters() if p.grad is not None}


def compare_tensors(left, right):
    import torch
    if set(left) != set(right):
        return {'exact': False, 'max_abs': None, 'reason': 'different tensor names'}
    exact, maximum = True, 0.0
    for key in left:
        a, b = left[key], right[key]
        if a.shape != b.shape or a.dtype != b.dtype:
            return {'exact': False, 'max_abs': None, 'reason': f'shape/dtype mismatch: {key}'}
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            return {'exact': False, 'max_abs': None, 'reason': f'nonfinite tensor: {key}'}
        exact = exact and torch.equal(a, b)
        if a.numel():
            maximum = max(maximum, float((a.double()-b.double()).abs().max()))
    return {'exact': bool(exact), 'max_abs': maximum}


def source_location(source):
    path = str(source.get('source_path', '')).replace('\\', '/')
    material = source.get('material')
    if material not in ('MoS2', 'WSe2'):
        matches = [m for m in ('MoS2', 'WSe2') if m in path or str(source['source_id']).startswith(m+'_')]
        if len(matches) != 1:
            raise ValueError(f'Cannot identify material for {source["source_id"]}')
        material = matches[0]
    concentration = source.get('concentration')
    if concentration not in ('low', 'high'):
        concentration = ('low' if 'low_density_defects/' in path else
                         'high' if 'high_density_defects/' in path else None)
    if concentration is None:
        raise ValueError(f'Cannot identify concentration for {source["source_id"]}')
    return material, concentration


def build_graphs(trainer, sources, data_root):
    import ast
    import pandas as pd
    import torch
    from pymatgen.core import Structure
    from .data.structure_utils import convert_to_sparse_2dmd_high
    tables, graphs = {}, []
    for index, source in enumerate(sources):
        material, density = source_location(source)
        key = material, density
        directory = data_root / ('low_density_defects/'+material if density == 'low'
                                 else 'high_density_defects/'+material+'_500')
        if key not in tables:
            unit_path = directory / 'unit_cells' / f'{material}.cif' if density == 'low' else data_root / f'{material}.cif'
            tables[key] = (pd.read_csv(directory / 'targets.csv.gz', index_col=0),
                           pd.read_csv(directory / 'descriptors.csv', index_col=0),
                           Structure.from_file(unit_path))
        targets, descriptors, unit = tables[key]
        sid = source['source_id']
        cell = ast.literal_eval(descriptors.loc[targets.loc[sid, 'descriptor_id'], 'cell'])
        structure = convert_to_sparse_2dmd_high(
            Structure.from_file(directory / 'initial' / f'{sid}.cif'), unit, cell,
            'megnet_sparse', [1], False, False)
        structure.y = float(targets.loc[sid, 'formation_energy_per_site'])
        graph = trainer.converter.convert(structure)
        graph.y = torch.tensor([structure.y], dtype=torch.float32)
        graph.audit_id = torch.tensor([index])
        # Store tensors only: the model does not consume the pymatgen object.
        del graph.structure
        graphs.append(graph)
    return graphs


def graph_hash(graph):
    import torch
    return tensor_hash({k: v for k, v in graph if isinstance(v, torch.Tensor)})


def git_output(*args):
    try:
        return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True,
                                       stderr=subprocess.DEVNULL, timeout=10).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def provenance(args):
    import torch
    code = [ROOT/'main.py', Path(__file__), ROOT/'__init__.py']
    for directory in ('models', 'data', 'training', 'utils', 'config'):
        code.extend((ROOT/directory).rglob('*.py'))
    content = {str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(code)}
    versions = {}
    for name in ('torch', 'torch-geometric', 'torch-scatter', 'numpy', 'pymatgen', 'scikit-learn'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {'git_commit': git_output('rev-parse', 'HEAD'), 'git_status': git_output('status', '--short'),
            'code_sha256': hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest(),
            'atom_init_sha256': hashlib.sha256(args.atom_init.read_bytes()).hexdigest(),
            'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            'python': sys.version, 'versions': versions, 'cuda': torch.version.cuda,
            'cudnn': torch.backends.cudnn.version(), 'device': args.device[0],
            'gpu': torch.cuda.get_device_name(args.device[0]) if args.device[0].startswith('cuda') else None,
            'pythonhashseed_at_startup': args.startup_hash_seed,
            'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            'cpu_threads': torch.get_num_threads(), 'determinism': args.determinism}


def worker(args):
    args.startup_hash_seed = os.environ.get('PYTHONHASHSEED')
    import copy
    import warnings
    import torch
    from torch_geometric.loader import DataLoader
    from .main import set_seed
    from .data.datasets import init_elem_embedding
    from .training.trainer import MEGNetTrainer, load_trusted_checkpoint

    report = {'status': 'ERROR', 'stage': 'setup', 'trace': []}
    output = args.worker
    try:
        torch.set_num_threads(1)
        init_elem_embedding(args.atom_init)
        checkpoint = load_trusted_checkpoint(args.checkpoint, map_location='cpu')
        if checkpoint['config']['task'] != 'megnet_sparse':
            raise ValueError('This diagnostic currently supports MEGNet sparse checkpoints only.')
        seed = int(checkpoint['split']['seed'])
        set_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=args.determinism == 'warn')
        config = copy.deepcopy(checkpoint['config'])
        if config['optim'].get('grad_accum_steps', 1) != 1 or config['optim'].get('point_loss', 'mse') != 'mse':
            raise ValueError('Diagnostic requires the standard sparse MSE protocol without gradient accumulation.')
        config['optim']['amp'] = False
        trainer = MEGNetTrainer(config, args.device[0], seed=seed)
        trainer.scaler.load_state_dict(checkpoint['scaler'])
        initial = model_state(trainer.model)
        report.update(provenance=provenance(args), seed=seed, config=config,
                      scaler={k: float(v) for k,v in checkpoint['scaler'].items()},
                      protocol='FP32, fresh initialization, fixed LR, no scheduler or early stopping',
                      initial_hash=tensor_hash(initial), stage='graphs')
        selected = {group: checkpoint['split'][group+'_sources'][:args.samples if group == 'train' else args.eval_samples]
                    for group in ('train', 'val')}
        if any(not sources for sources in selected.values()):
            raise ValueError('Checkpoint must contain nonempty train_sources and val_sources.')
        graphs = {group: build_graphs(trainer, sources, args.data_root) for group, sources in selected.items()}
        report['sources'] = {g: [s['source_id'] for s in sources] for g, sources in selected.items()}
        report['graph_hashes'] = {g: [graph_hash(graph) for graph in data] for g,data in graphs.items()}
        evaluation = list(DataLoader(graphs['val'], batch_size=config['model']['test_batch_size']))

        def predict():
            values = []
            trainer.model.eval()
            with torch.no_grad():
                for batch in evaluation:
                    pred = trainer.scaler.inverse_transform(trainer._forward(batch.clone().to(args.device[0])))
                    values.append(plain_cpu(pred).reshape(-1))
            return torch.cat(values)

        report['stage'] = 'checkpoint_inference'
        trainer.model.load_state_dict(checkpoint['model'], strict=True)
        frozen = [predict() for _ in range(3)]
        report['inference_within_process'] = [compare_tensors({'pred': frozen[0]}, {'pred': pred}) for pred in frozen[1:]]
        report['checkpoint_inference_hash'] = tensor_hash({'pred': frozen[0]})
        trainer.model.load_state_dict(initial, strict=True)
        set_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=args.determinism == 'warn')
        loader = DataLoader(graphs['train'], batch_size=config['model']['train_batch_size'], shuffle=True,
                            num_workers=0, generator=trainer._make_generator(2))
        iterator = iter(loader)
        first_gradient = None
        report['stage'] = 'training'
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            for step in range(1, args.steps+1):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    batch = next(iterator)
                row = {'step': step, 'batch_ids': batch.audit_id.tolist(), 'batch_hash': graph_hash(batch)}
                batch = batch.to(args.device[0])
                trainer.model.train()
                trainer.optimizer.zero_grad(set_to_none=True)
                pred = trainer._forward(batch).reshape(-1)
                target = trainer.scaler.transform(batch.y).reshape(-1)
                loss = torch.nn.functional.mse_loss(pred, target)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite training loss at step {step}')
                row.update(prediction_hash=tensor_hash({'pred': pred}), loss=float(loss.detach()))
                loss.backward()
                grad = gradients(trainer.model)
                if any(not torch.isfinite(v).all() for v in grad.values()):
                    raise FloatingPointError(f'Nonfinite gradient at step {step}')
                if first_gradient is None:
                    first_gradient = grad
                row['gradient_hash'] = tensor_hash(grad)
                trainer.optimizer.step()
                row['updated_weight_hash'] = tensor_hash(trainer.model.state_dict())
                report['trace'].append(row)
            report['training_warnings'] = sorted(set(str(w.message) for w in caught))
        final = model_state(trainer.model)
        final_predictions = predict()
        if any(not torch.isfinite(v).all() for v in [*final.values(), final_predictions, *frozen]):
            raise FloatingPointError('Nonfinite final weights or evaluation predictions')
        target = torch.cat([graph.y for graph in graphs['val']])
        report['small_val_mae'] = float((final_predictions-target).abs().mean())
        report['final_hash'] = tensor_hash(final)
        torch.save({'initial': initial, 'first_gradient': first_gradient, 'final': final,
                    'checkpoint_predictions': {'pred': frozen[0]}, 'final_predictions': {'pred': final_predictions}},
                   output.with_suffix('.pth'))
        report.update(status='COMPLETE', stage='complete')
    except Exception as exc:
        report.update(error=str(exc), traceback=traceback.format_exc())
    write_json(output, report)
    return 0 if report['status'] == 'COMPLETE' else 1


def compare_reports(left, right, left_state, right_state):
    checks = {
        'code': left['provenance']['code_sha256'] == right['provenance']['code_sha256'],
        'environment': left['provenance'] == right['provenance'],
        'config': left['config'] == right['config'], 'scaler': left['scaler'] == right['scaler'],
        'sources': left['sources'] == right['sources'],
        'graphs': left['graph_hashes'] == right['graph_hashes'],
        'initial_weights': left['initial_hash'] == right['initial_hash'],
        'batch_order_and_inputs': [(r['batch_ids'], r['batch_hash']) for r in left['trace']] ==
                                  [(r['batch_ids'], r['batch_hash']) for r in right['trace']],
    }
    first = None
    for a, b in zip(left['trace'], right['trace']):
        for field in ('batch_hash', 'prediction_hash', 'loss', 'gradient_hash', 'updated_weight_hash'):
            if a[field] != b[field]:
                first = {'step': a['step'], 'stage': field}
                break
        if first is not None:
            break
    differences = {key: compare_tensors(left_state[key], right_state[key]) for key in left_state}
    inference_exact = all(d['exact'] for report in (left, right) for d in report['inference_within_process'])
    status = ('INPUT_OR_ENVIRONMENT_MISMATCH' if not all(checks.values()) else
              'EXACT_SHORT_RUN' if first is None and inference_exact and all(d['exact'] for d in differences.values())
              else 'NUMERICAL_DIFFERENCES')
    return {'status': status, 'controls': checks, 'first_training_difference': first,
            'tensor_differences': differences, 'inference_exact_within_process': inference_exact,
            'max_loss_difference': max(abs(a['loss']-b['loss']) for a,b in zip(left['trace'], right['trace'])),
            'small_val_mae': [left['small_val_mae'], right['small_val_mae']]}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True, help='Trusted HERA MEGNet sparse checkpoint')
    parser.add_argument('--device', nargs='+', default=['cuda:0', 'cpu'])
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--samples', type=int, default=32, help='First N saved training IDs, for a bounded diagnostic')
    parser.add_argument('--eval-samples', type=int, default=16)
    parser.add_argument('--determinism', choices=('strict', 'warn'), default='strict')
    parser.add_argument('--data-root', type=Path, default=ROOT.parent/'dataset/2d-materials-point-defects-all')
    parser.add_argument('--atom-init', type=Path, default=ROOT/'atom_init.json')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'logs/sparse_reproducibility')
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if min(args.steps, args.samples, args.eval_samples) < 1:
        parser.error('steps, samples and eval-samples must be positive')
    for name in ('checkpoint', 'data_root', 'atom_init', 'output_dir'):
        setattr(args, name, getattr(args, name).resolve())
    if args.worker:
        args.worker = args.worker.resolve()
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.worker:
        return worker(args)
    # Workers start with hash/CUBLAS settings already in their environment.
    env = os.environ.copy()
    env.update(PYTHONHASHSEED='0', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    folder = args.output_dir/(datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
    folder.mkdir(parents=True)
    result = {'scope': 'Independent-process short-run repeatability; not full-training or cross-seed stability.',
              'steps': args.steps, 'devices': {}}
    failures = False
    for device_index, device in enumerate(args.device):
        paths, reports = [], []
        for repeat in range(2):
            path = folder/f'device{device_index}_repeat{repeat}.json'
            cmd = [sys.executable, '-m', 'HERA.check_sparse_reproducibility', '--worker', str(path),
                   '--checkpoint', str(args.checkpoint), '--device', device, '--steps', str(args.steps),
                   '--samples', str(args.samples), '--eval-samples', str(args.eval_samples),
                   '--determinism', args.determinism, '--data-root', str(args.data_root),
                   '--atom-init', str(args.atom_init)]
            print(f'{device}: independent process {repeat+1}/2, {args.steps} updates', flush=True)
            with path.with_suffix('.log').open('w', encoding='utf-8') as log:
                process = subprocess.run(cmd, cwd=ROOT.parent, env=env, stdout=log, stderr=subprocess.STDOUT)
            report = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'status': 'ERROR', 'error': 'Worker did not write a report; inspect its log.'}
            if process.returncode or report['status'] != 'COMPLETE':
                failures = True
                result['devices'][device] = {'status': 'ERROR', 'stage': report.get('stage'),
                                              'error': report.get('error'), 'worker_report': str(path)}
                break
            paths.append(path)
            reports.append(report)
        if len(reports) == 2:
            import torch
            try:
                states = [torch.load(p.with_suffix('.pth'), map_location='cpu', weights_only=True) for p in paths]
                result['devices'][device] = compare_reports(*reports, *states)
            except Exception as exc:
                failures = True
                result['devices'][device] = {'status': 'ERROR', 'stage': 'comparison', 'error': str(exc)}
        print(device, json.dumps(result['devices'][device]), flush=True)
        write_json(folder/'report.json', result)
    print(f'Report: {folder / "report.json"}', flush=True)
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
