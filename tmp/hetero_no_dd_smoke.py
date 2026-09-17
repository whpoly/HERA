"""Real-native GPU integration check; does not run or estimate a benchmark."""
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

import numpy as np
import pandas as pd
import torch
from pymatgen.core import Structure
from torch_geometric.data import Batch

from HERA.config.defaults import get_config, apply_alignn_hetero_options
from HERA.data.datasets import init_elem_embedding, tag_structure_source
from HERA.data.structure_utils import convert_to_sparse_native
from HERA.training.trainer import MEGNetTrainer
from HERA.scripts.run_hetero_relation_benchmark import VARIANTS, data_errors

torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
init_elem_embedding(ROOT / 'atom_init.json')
directory = ROOT.parent / 'dataset/Dataset_1/Dataset_1/A_rich/Neutral'
table = pd.read_csv(directory / 'id_prop_A_rich.csv', header=None)
structures, selected = [], []
for index in np.linspace(0, len(table) - 1, 8, dtype=int):
    name, target = str(table.iloc[index, 0]), float(table.iloc[index, 1])
    raw = Structure.from_file(directory / name)
    tag_structure_source(raw, directory / name, name)
    defect = 'vacancy' if name.split('-')[2].split('_')[0] == 'V' else 'others'
    structure = convert_to_sparse_native(raw, defect, 1, 'alignn_hetero', None, True, False, local_cutoff=0)
    if structure is None:
        continue
    structures.append(structure)
    selected.append({'source_id': name, 'target': target,
                     'defects': sum(int(s.properties['pool_type']) for s in structure)})
assert len(structures) >= 2
report = {'purpose': 'GPU integration only; not a test MAE benchmark',
          'device': torch.cuda.get_device_name(), 'sampled_native': selected,
          'data_preflight': data_errors(['native', 'semi', 'imp2d']), 'variants': {}}
baselines = None
for name, (aa, dd) in VARIANTS.items():
    config = get_config('alignn', 'native', 'hetero')
    config['model']['local_radius'] = 0
    apply_alignn_hetero_options(config, aa_mode=aa, dd_mode=dd, pooling='defect_energy_mean')
    torch.manual_seed(123)
    trainer = MEGNetTrainer(config, 'cuda:0', seed=123)
    graphs = [trainer.converter.convert(s) for s in structures]
    if baselines is None:
        baselines = graphs
    else:
        for graph, before in zip(graphs, baselines):
            for t in graph.edge_types:
                for field in ('edge_index', 'edge_attr', 'edge_vec'):
                    torch.testing.assert_close(graph[t][field], before[t][field], rtol=0, atol=0)
    batch = Batch.from_data_list(graphs[:2]).to('cuda:0')
    trainer.model.eval()
    with torch.no_grad():
        together = trainer._forward(batch)
        single = torch.cat([trainer._forward(Batch.from_data_list([g]).to('cuda:0')) for g in graphs[:2]])
    torch.testing.assert_close(together, single, atol=1e-5, rtol=2e-5)
    optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=1e-3)
    trainer.model.train()
    optimizer.zero_grad(set_to_none=True)
    target = torch.tensor([s['target'] for s in selected[:2]], device='cuda:0')
    loss = (trainer._forward(batch) - target).square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in trainer.model.parameters() if p.grad is not None)
    optimizer.step()
    restored = MEGNetTrainer(copy.deepcopy(config), 'cuda:0', seed=123)
    restored.model.load_state_dict(trainer.model.state_dict(), strict=True)
    restored.model.eval()
    trainer.model.eval()
    with torch.no_grad():
        torch.testing.assert_close(trainer._forward(batch), restored._forward(batch), atol=2e-6, rtol=2e-5)
    report['variants'][name] = {
        'parameters': sum(p.numel() for p in trainer.model.parameters()),
        'edges': [{t[1]: g[t].num_edges for t in g.edge_types} for g in graphs],
        'batch_single_max_difference': float((together - single).abs().max()),
        'finite_forward_backward_optimizer_step': 'passed', 'strict_restore': 'passed',
    }
    del restored, trainer, optimizer, batch
    torch.cuda.empty_cache()
output = ROOT / 'results/hetero_no_dd_validation_20260917.json'
output.write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
