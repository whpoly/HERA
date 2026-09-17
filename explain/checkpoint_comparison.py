"""Compare frozen ALIGNN checkpoint explanations on deterministic high-density cases.

Run from HERA's parent: python -m HERA.explain.checkpoint_comparison --help
Node masks act on the initial learned node embeddings by default. This keeps
vacancy placeholders explainable even though their raw 92-D features are zero.
Geometry, types and graph connectivity stay fixed. Model predictions, rather
than experimental targets, are explained.
"""
from __future__ import annotations

import argparse
import ast
import copy
import csv
import hashlib
import itertools
import json
import time
import warnings
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ase import Atoms
from ase.io import read as ase_read, write as ase_write
from pymatgen.core import Structure
from scipy.stats import spearmanr
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.explain import Explainer, GNNExplainer

from ..data.datasets import init_elem_embedding, tag_structure_source
from ..data.structure_utils import convert_to_sparse_2dmd_high
from ..models.alignn import SharedHeteroRelations
from ..predict_2dmd_low_checkpoints import apply_checkpoint_model_compatibility, checkpoint_identity
from ..training.trainer import MEGNetTrainer, load_trusted_checkpoint

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / 'dataset/2d-materials-point-defects-all'
CHECKPOINTS = {
    'attention': 'logs/bench_2dmd_low_all_models/alignn/2dmd_low/attention/seed123_best_checkpoint.pth',
    'hetero_old': 'logs/bench_2dmd_low_all_models/alignn/2dmd_low/hetero/r0/norm_layernorm/seed123_best_checkpoint.pth',
    'hetero_shared': 'logs/hetero_energy_mean_benchmark/alignn/2dmd_low/hetero/r0/features_layernorm/pool_defect_energy_mean/relations_shared_residual_rank8/seed123_best_checkpoint.pth',
}
RELATIONS = ('aa', 'ad', 'da', 'dd')


def json_write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)), encoding='utf-8')


def selected_cases(material, counts, per_count=1, selection_seed=20260917):
    directory = DATA / 'high_density_defects' / f'{material}_500'
    desc = pd.read_csv(directory / 'descriptors.csv', index_col=0)
    targets = pd.read_csv(directory / 'targets.csv.gz', index_col=0)
    groups = {count: [] for count in counts}
    for sid, row in targets.iterrows():
        descriptor = desc.loc[row.descriptor_id]
        defects = ast.literal_eval(descriptor.defects)
        count = len(defects)
        if count in groups:
            groups[count].append({'id': str(sid), 'material': material, 'defect_count': count,
                                 'target': float(row.formation_energy_per_site),
                                 'cell': ast.literal_eval(descriptor.cell), 'defects': defects,
                                 'path': str(directory / 'initial' / f'{sid}.cif')})
    cases = []
    for count, items in groups.items():
        items.sort(key=lambda x: hashlib.sha256(f'{selection_seed}:{x["id"]}'.encode()).hexdigest())
        if len(items) < per_count:
            raise ValueError(f'Insufficient {material} cases at {count} defects')
        for index, item in enumerate(items[:per_count]):
            item['key'] = f'{material.lower()}_d{count:02d}_{index+1}'
            cases.append(item)
    return cases


def load_models(names, device, cohort='mixed_low', material=None):
    trainers, checkpoints, provenance = {}, {}, {}
    for name in names:
        if cohort == 'mixed_low':
            path = ROOT / CHECKPOINTS[name]
        elif name == 'hetero_shared':
            if material != 'WSe2':
                raise ValueError('No verified local MoS2-only shared checkpoint is available')
            path = ROOT / 'logs/hetero_energy_mean_benchmark/alignn/2dmd_wse2/hetero/r0/features_layernorm/pool_defect_energy_mean/relations_shared_residual_rank8/seed123_best_checkpoint.pth'
        else:
            dataset = f'2dmd_{material.lower()}'
            suffix = 'attention' if name == 'attention' else 'hetero/r0'
            path = ROOT / f'logs/{dataset}/alignn/{dataset}/{suffix}/seed123_best_checkpoint.pth'
        ckpt = load_trusted_checkpoint(path, map_location='cpu')
        record = checkpoint_identity(ckpt, path)
        config = copy.deepcopy(ckpt['config'])
        compat = apply_checkpoint_model_compatibility(config, record)
        trainer = MEGNetTrainer(config, device, seed=record['seed'])
        trainer.scaler.load_state_dict(ckpt['scaler'])
        trainer.model.load_state_dict(ckpt['model'], strict=True)
        trainer.model.eval().requires_grad_(False)
        trainers[name], checkpoints[name] = trainer, ckpt
        provenance[name] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                            'dataset': ckpt['dataset_name'], 'seed': record['seed'],
                            'best_epoch': ckpt.get('best_epoch'), 'saved_test_mae': ckpt['test_mae'],
                            'effective_config': config, 'compatibility': compat,
                            'scaler_mean': float(trainer.scaler.mean), 'scaler_std': float(trainer.scaler.std),
                            'parameter_count': sum(p.numel() for p in trainer.model.parameters())}
    reference = next(iter(checkpoints.values()))
    for name, ckpt in checkpoints.items():
        checks = {}
        for split in ('train_sources', 'val_sources', 'test_sources'):
            a = [x['source_id'] for x in reference['split'][split]]
            b = [x['source_id'] for x in ckpt['split'][split]]
            checks[split] = a == b
        if not all(checks.values()):
            raise ValueError(f'{name}: training-domain split differs')
        if ckpt['dataset_name'] != reference['dataset_name']:
            raise ValueError('Training datasets differ')
        provenance[name]['identical_ordered_split_ids'] = checks
        if any(float(ckpt['scaler'][k]) != float(reference['scaler'][k]) for k in ('mean', 'std')):
            raise ValueError('Target scalers differ')
    return trainers, provenance


def prepare_case(case, trainers, device, mask_space='embedding'):
    raw = Structure.from_file(case['path'])
    real_atoms = ase_read(case['path'])
    if len(real_atoms) != len(raw):
        raise ValueError('ASE and model CIF readers disagree on atom count')
    tag_structure_source(raw, case['path'], case['id'], concentration='high', material=case['material'])
    unit = Structure.from_file(DATA / f'{case["material"]}.cif')
    wrappers, reference = {}, None
    for name, trainer in trainers.items():
        mode = 'attention' if trainer.config['task'] == 'alignn_attention' else 'hetero'
        structure = convert_to_sparse_2dmd_high(raw, unit, case['cell'], f'alignn_{mode}', None,
                                               False, False, local_cutoff=0)
        structure.y = case['target']
        types = np.array([int(s.properties['type']) for s in structure], dtype=int)
        pool_types = np.array([int(s.properties.get('pool_type', s.properties['type'])) for s in structure], dtype=int)
        if int(pool_types.sum()) != case['defect_count']:
            raise ValueError('Descriptor and actual-defect masks disagree')
        graph = trainer.converter.convert(structure)
        batch = Batch.from_data_list([graph]).to(device)
        wrapper = FrozenGraphWrapper(trainer, batch, types, mask_space).to(device).eval()
        with torch.no_grad():
            y_ref = trainer._forward(batch).reshape(-1)
            y_flat = wrapper(wrapper.x).reshape(-1)
        torch.testing.assert_close(y_flat, y_ref, atol=1e-6, rtol=1e-6)
        coordinates = np.asarray(structure.cart_coords)
        species = [s.specie.symbol for s in structure]
        current = {'positions': coordinates.tolist(), 'cell': np.asarray(structure.lattice.matrix).tolist(),
                   'species': species, 'node_type': types.tolist(), 'pool_type': pool_types.tolist()}
        if reference is not None:
            if current != reference:
                raise ValueError('Models use different site order, species or type labels')
            torch.testing.assert_close(wrapper.raw_x.cpu(), next(iter(wrappers.values())).raw_x.cpu(), atol=0, rtol=0)
        reference = current
        wrappers[name] = wrapper
    atoms = Atoms(symbols=['X' if str(s).startswith('X') else s for s in reference['species']],
                  positions=reference['positions'], cell=reference['cell'], pbc=True)
    atoms.new_array('node_type', np.asarray(reference['node_type']))
    atoms.new_array('site_index', np.arange(len(atoms)))
    return wrappers, reference, atoms


class FrozenGraphWrapper(nn.Module):
    """One mask per physical lattice site, with the same regularizer for every model.

    Flattening the heterogeneous node stores avoids summing one mean penalty
    per node type, which would overweight the much smaller defect store.
    """
    def __init__(self, trainer, batch, types, mask_space='embedding'):
        super().__init__()
        self.model = trainer.model
        self.graph = batch
        self.hetero = trainer.config['task'] == 'alignn_hetero'
        self.mean, self.std = float(trainer.scaler.mean), float(trainer.scaler.std)
        self.mask_space = mask_space
        self.indices = {}
        if self.hetero:
            reference = next(iter(batch.x_dict.values()))
            self.x = reference.new_zeros((len(types), reference.size(1)))
            for label, code in [('atom', 0), ('defect', 1)]:
                ids = torch.as_tensor(np.flatnonzero(np.asarray(types) == code), device=reference.device)
                self.indices[label] = ids
                torch.testing.assert_close(torch.tensor(len(ids)), torch.tensor(batch[label].num_nodes))
                self.x[ids] = batch[label].x
            self.edge_index = torch.cat([torch.stack([self.indices[k[0]][v[0]], self.indices[k[2]][v[1]]])
                                         for k, v in batch.edge_index_dict.items()], dim=1)
        else:
            self.x = batch.x
            self.edge_index = batch.edge_index
        self.raw_x = self.x
        if mask_space == 'embedding':
            with torch.no_grad():
                if self.hetero:
                    self.x = self.raw_x.new_zeros((len(types), self.model.hidden_dim))
                    for key, ids in self.indices.items():
                        self.x[ids] = self.model.node_embedding[key](self.raw_x[ids])
                else:
                    self.x = self.model.node_embedding(self.raw_x).detach()

    def forward(self, x, edge_index=None):
        g = self.graph
        handles = []
        model_x = x
        if self.mask_space == 'embedding':
            model_x = self.raw_x
            if self.hetero:
                for key, ids in self.indices.items():
                    handles.append(self.model.node_embedding[key].register_forward_hook(
                        lambda module, args, output, ids=ids: x[ids]))
            else:
                handles.append(self.model.node_embedding.register_forward_hook(lambda module, args, output: x))
        try:
            if self.hetero:
                result = self.model({k: model_x[v] for k, v in self.indices.items()}, g.edge_index_dict,
                                    g.edge_attr_dict, g.batch_dict, edge_vec_dict=g.collect('edge_vec'),
                                    state=g.state, pool_type=g.collect('pool_type'))
            else:
                result = self.model(model_x, g.edge_index, g.edge_attr, g.batch, edge_vec=g.edge_vec, node_type=g.node_type)
        finally:
            for handle in handles:
                handle.remove()
        return result.reshape(-1, 1)

    def physical(self, x):
        return self(x) * self.std + self.mean


@contextmanager
def ablate_relation(model, relation):
    """Zero one relation's node-message numerator at every layer.

    Gates/denominators and edge updates are left intact. This is an internal
    computation intervention, not deletion of physical atoms or bonds.
    """
    handles = []
    def zero_result(module, args, output):
        return (torch.zeros_like(output[0]), output[1], output[2])
    def shared_hook(module, args, output):
        return zero_result(module, args, output) if args[0].split('__')[1] == relation else output
    try:
        for layer in [*model.layers, *model.gcn_layers]:
            bank = layer.atom_convs
            if isinstance(bank, SharedHeteroRelations):
                handles.append(bank.register_forward_hook(shared_hook))
            else:
                for key, conv in bank.items():
                    if key.split('__')[1] == relation:
                        handles.append(conv.register_forward_hook(zero_result))
        yield
    finally:
        for handle in handles:
            handle.remove()


def relation_effects(wrapper):
    if not wrapper.hetero:
        return None
    with torch.no_grad():
        base = float(wrapper.physical(wrapper.x).item())
        effects = {}
        for relation in RELATIONS:
            with ablate_relation(wrapper.model, relation):
                ablated = float(wrapper.physical(wrapper.x).item())
            effects[relation] = {'ablated_prediction': ablated, 'delta_mev': 1000 * (ablated-base)}
        torch.testing.assert_close(wrapper.physical(wrapper.x), torch.tensor([[base]], device=wrapper.x.device), atol=1e-6, rtol=1e-6)
    return effects


def native_readout(wrapper):
    if wrapper.hetero and wrapper.model.pooling == 'defect_energy_mean':
        captured = []
        handle = wrapper.model.readout.register_forward_hook(lambda module, args, out: captured.append(out.detach()))
        try:
            with torch.no_grad():
                pred = float(wrapper.physical(wrapper.x).item())
        finally:
            handle.remove()
        values = captured[0].reshape(-1).cpu().numpy() * wrapper.std + wrapper.mean
        if abs(float(values.mean()) - pred) > 1e-5:
            raise ValueError('Per-defect readout does not reconstruct graph prediction')
        ids = []
        for label in wrapper.model.node_types:
            mask = wrapper.graph[label].pool_type.bool()
            ids.extend(wrapper.indices[label][mask].cpu().tolist())
        return {'kind': 'latent_defect_readout', 'site_indices': ids, 'values_ev': values.tolist(),
                'mean_reconstruction_error_ev': abs(float(values.mean())-pred)}
    if not wrapper.hetero:
        with torch.no_grad():
            wrapper(wrapper.x)
        weights = wrapper.model.node_readout.get_attention_weights().detach().cpu().reshape(-1).numpy()
        return {'kind': 'global_attention_weights', 'site_indices': list(range(len(weights))), 'values': weights.tolist()}
    return None


def explain_one(wrapper, epochs, seeds, lr, size_coeff, ent_coeff):
    masks, runs = [], []
    with torch.no_grad():
        full = float(wrapper.physical(wrapper.x).item())
    for seed in seeds:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        start = time.perf_counter()
        explainer = Explainer(model=wrapper,
                              algorithm=GNNExplainer(epochs=epochs, lr=lr, node_feat_size=size_coeff,
                                                     node_feat_ent=ent_coeff, node_feat_reduction='mean'),
                              explanation_type='model', node_mask_type='object', edge_mask_type=None,
                              model_config={'mode': 'regression', 'task_level': 'graph', 'return_type': 'raw'})
        explanation = explainer(wrapper.x, wrapper.edge_index)
        mask = explanation.node_mask.detach().reshape(-1)
        with torch.no_grad():
            masked = float(wrapper.physical(wrapper.x * mask[:, None]).item())
        masks.append(mask.cpu().numpy())
        runs.append({'seed': seed, 'seconds': time.perf_counter()-start, 'masked_prediction': masked,
                     'prediction_preservation_error_mev': 1000*abs(masked-full), 'mask_mean': float(mask.mean()),
                     'mask_min': float(mask.min()), 'mask_max': float(mask.max())})
    matrix = np.stack(masks)
    stability = []
    k = max(1, int(np.ceil(0.1*matrix.shape[1])))
    for i, j in itertools.combinations(range(len(seeds)), 2):
        ai, aj = set(np.argsort(-matrix[i])[:k]), set(np.argsort(-matrix[j])[:k])
        corr = spearmanr(matrix[i], matrix[j]).statistic
        stability.append({'seed_a': seeds[i], 'seed_b': seeds[j], 'spearman': float(corr) if np.isfinite(corr) else None,
                          'top10pct_jaccard': len(ai & aj)/len(ai | aj)})
    return matrix, runs, stability


def deletion_checks(wrapper, scores, random_trials=12, seed=813):
    rng = np.random.default_rng(seed)
    n = len(scores)
    with torch.no_grad():
        base = float(wrapper.physical(wrapper.x).item())
        rows = []
        for fraction in (0.05, 0.1, 0.2):
            k = max(1, int(np.ceil(n*fraction)))
            top = np.argsort(-scores)[:k]
            low = np.argsort(scores)[:k]
            changes = {}
            for label, ids in [('top', top), ('bottom', low)]:
                masked = wrapper.x.clone()
                masked[torch.as_tensor(ids, device=masked.device)] = 0
                changes[f'{label}_change_mev'] = 1000*abs(float(wrapper.physical(masked).item())-base)
            random = []
            for _ in range(random_trials):
                masked = wrapper.x.clone()
                masked[torch.as_tensor(rng.choice(n, k, replace=False), device=masked.device)] = 0
                random.append(1000*abs(float(wrapper.physical(masked).item())-base))
            rows.append({'fraction': fraction, 'k': k, **changes, 'random_mean_mev': float(np.mean(random)),
                         'random_std_mev': float(np.std(random)), 'random_changes_mev': random})
    return rows


def run(args):
    torch.set_num_threads(2)
    warnings.filterwarnings('ignore', message='.*Issues encountered while parsing CIF.*')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    init_elem_embedding(ROOT / 'atom_init.json')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.cohort == 'single_material' and len(args.materials) != 1:
        raise ValueError('Single-material cohort requires exactly one material per run')
    trainers, provenance = load_models(args.models, args.device, args.cohort, args.materials[0])
    cases = [c for m in args.materials for c in selected_cases(m, args.counts, args.per_count)]
    protocol = {'explanation': 'PyG GNNExplainer, model prediction, one node-state mask per lattice site',
                'mask_space': args.mask_space, 'cohort': args.cohort,
                'geometry_types_and_edges': 'fixed', 'mask_regularizer': 'one mean over all physical lattice sites',
                'optimization_output': 'standardized prediction, identical scaler across models',
                'epochs': args.epochs, 'seeds': args.seeds, 'lr': args.lr,
                'random_trials': args.random_trials,
                'node_feat_size': args.size, 'node_feat_ent': args.entropy,
                'selection': 'first SHA256(20260917:source_id) per defect-count stratum, independent of errors',
                'torch': torch.__version__, 'device': args.device,
                'caveat': 'Comparison of existing checkpoints, not an isolated architectural ablation'}
    json_write(output / 'manifest.json', {'protocol': protocol, 'checkpoints': provenance, 'cases': cases})
    all_results = []
    for case in cases:
        case_dir = output / 'cases' / case['key']
        case_dir.mkdir(parents=True, exist_ok=True)
        wrappers, geometry, atoms = prepare_case(case, trainers, args.device, args.mask_space)
        json_write(case_dir / 'structure.json', {**case, **geometry})
        ase_write(case_dir / 'structure.extxyz', atoms)
        for name, wrapper in wrappers.items():
            record_path = case_dir / f'{name}.json'
            if args.resume and record_path.exists():
                record = json.loads(record_path.read_text(encoding='utf-8'))
                if record['protocol'] != protocol or record['checkpoint_sha256'] != provenance[name]['sha256']:
                    raise ValueError('Resume protocol or checkpoint differs')
                all_results.append(record)
                print(f'REUSED {case["key"]} {name}', flush=True)
                continue
            start = time.perf_counter()
            with torch.no_grad():
                pred = float(wrapper.physical(wrapper.x).item())
            print(f'START {case["key"]} {name} prediction={pred:.6f} target={case["target"]:.6f}', flush=True)
            matrix, runs, stability = explain_one(wrapper, args.epochs, args.seeds, args.lr, args.size, args.entropy)
            consensus = matrix.mean(axis=0)
            with torch.no_grad():
                consensus_prediction = float(wrapper.physical(wrapper.x * torch.as_tensor(consensus, device=wrapper.x.device)[:, None]).item())
                zero_prediction = float(wrapper.physical(torch.zeros_like(wrapper.x)).item())
            fidelity = deletion_checks(wrapper, consensus, args.random_trials)
            relation = relation_effects(wrapper)
            native = native_readout(wrapper)
            record = {'case': case, 'model': name, 'checkpoint_sha256': provenance[name]['sha256'],
                      'protocol': protocol, 'prediction': pred, 'target': case['target'],
                      'consensus_preservation_error_mev': 1000*abs(consensus_prediction-pred),
                      'zero_embedding_change_mev': 1000*abs(zero_prediction-pred),
                      'abs_error_mev': 1000*abs(pred-case['target']), 'runs': runs,
                      'stability': stability, 'deletion': fidelity, 'relation_ablation': relation,
                      'native_readout': native, 'seconds': time.perf_counter()-start,
                      'wrapper_parity': 'passed', 'n_nodes': len(consensus),
                      'raw_zero_feature_sites': torch.where(wrapper.raw_x.abs().sum(1).eq(0))[0].cpu().tolist(),
                      'n_edges': wrapper.edge_index.size(1)}
            np.savez_compressed(case_dir / f'{name}_masks.npz', masks=matrix, mean=consensus, std=matrix.std(axis=0))
            rows = []
            for i in range(len(consensus)):
                rows.append({'site_index': i, 'species': geometry['species'][i], 'node_type': geometry['node_type'][i],
                             'x': geometry['positions'][i][0], 'y': geometry['positions'][i][1], 'z': geometry['positions'][i][2],
                             'mask_mean': float(consensus[i]), 'mask_std': float(matrix[:, i].std()),
                             **{f'mask_seed_{s}': float(matrix[j, i]) for j, s in enumerate(args.seeds)}})
            with (case_dir / f'{name}_nodes.csv').open('w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
            at = atoms.copy()
            at.new_array('importance', consensus)
            at.new_array('importance_std', matrix.std(axis=0))
            ase_write(case_dir / f'{name}.extxyz', at)
            json_write(record_path, record)
            all_results.append(record)
            print(f'DONE {case["key"]} {name} {record["seconds"]:.1f}s preservation_meV={np.mean([r["prediction_preservation_error_mev"] for r in runs]):.3f}', flush=True)
            json_write(output / 'results.json', all_results)
        del wrappers
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    json_write(output / 'results.json', all_results)
    print(f'COMPLETE: {len(all_results)} case/model combinations at {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=str(ROOT / 'results/hetero_explainer_20260917'))
    parser.add_argument('--materials', nargs='+', choices=['MoS2','WSe2'], default=['MoS2','WSe2'])
    parser.add_argument('--cohort', choices=['mixed_low','single_material'], default='mixed_low')
    parser.add_argument('--mask-space', choices=['embedding','input'], default='embedding')
    parser.add_argument('--models', nargs='+', choices=list(CHECKPOINTS), default=list(CHECKPOINTS))
    parser.add_argument('--counts', nargs='+', type=int, default=[4,9,14,19,24])
    parser.add_argument('--per-count', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--seeds', nargs='+', type=int, default=[123,456,789])
    parser.add_argument('--lr', type=float, default=0.03)
    parser.add_argument('--size', type=float, default=0.01)
    parser.add_argument('--entropy', type=float, default=0.001)
    parser.add_argument('--random-trials', type=int, default=12)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--resume', action='store_true')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
