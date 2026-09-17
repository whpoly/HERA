"""Short GPU integration check on real high-density inputs; not a benchmark."""
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

import torch
from pymatgen.core import Structure
from torch_geometric.data import Batch

from HERA.config.defaults import apply_alignn_hetero_options
from HERA.data.datasets import init_elem_embedding
from HERA.data.structure_utils import convert_to_sparse_2dmd_high
from HERA.explain.checkpoint_comparison import selected_cases, CHECKPOINTS, DATA
from HERA.training.trainer import MEGNetTrainer, load_trusted_checkpoint

torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
device = 'cuda:0'
init_elem_embedding(ROOT / 'atom_init.json')
checkpoint = load_trusted_checkpoint(ROOT / CHECKPOINTS['hetero_shared'], 'cpu')
structures, cases = [], []
for material in ('MoS2', 'WSe2'):
    for case in selected_cases(material, (4, 24)):
        raw = Structure.from_file(case['path'])
        unit = Structure.from_file(DATA / f'{material}.cif')
        structure = convert_to_sparse_2dmd_high(raw, unit, case['cell'], 'alignn_hetero',
                                               None, False, False, local_cutoff=0)
        structure.y = case['target']
        structures.append(structure)
        cases.append(case)

report = {'device': torch.cuda.get_device_name(), 'purpose': 'integration only, not accuracy evaluation',
          'cases': [c['key'] for c in cases], 'variants': {}}
baseline_graphs = None
for aa in ('keep', 'drop'):
    config = copy.deepcopy(checkpoint['config'])
    apply_alignn_hetero_options(config, aa_mode=aa)
    torch.manual_seed(123)
    trainer = MEGNetTrainer(config, device, seed=123)
    graphs = [trainer.converter.convert(s) for s in structures]
    if aa == 'keep':
        baseline_graphs = graphs
        trainer.model.load_state_dict(checkpoint['model'], strict=True)
        trainer.model.eval()
        with torch.no_grad():
            scaled = trainer._forward(Batch.from_data_list(graphs).to(device))
        predictions = scaled.cpu() * float(checkpoint['scaler']['std']) + float(checkpoint['scaler']['mean'])
        errors = []
        for case, prediction in zip(cases, predictions):
            prior = json.loads((ROOT / 'results/relation_message_concentration_20260917/cases' / f"{case['key']}.json").read_text())
            # The stored concentration experiment uses this same checkpoint.
            reference = prior['prediction']
            errors.append(abs(float(prediction) - reference))
        assert max(errors) < 2e-6, errors
        report['legacy_prediction_max_difference_ev'] = max(errors)
        torch.manual_seed(123)
        trainer = MEGNetTrainer(config, device, seed=123)
    else:
        for graph, baseline in zip(graphs, baseline_graphs):
            assert ('atom', 'aa', 'atom') not in graph.edge_types
            for t in graph.edge_types:
                for field in ('edge_index', 'edge_attr', 'edge_vec'):
                    torch.testing.assert_close(graph[t][field], baseline[t][field], rtol=0, atol=0)
    batch = Batch.from_data_list(graphs).to(device)
    trainer.model.eval()
    with torch.no_grad():
        together = trainer._forward(batch)
        alone = torch.cat([trainer._forward(Batch.from_data_list([g]).to(device)) for g in graphs])
    torch.testing.assert_close(together, alone, atol=1e-5, rtol=2e-5)
    target = torch.tensor([c['target'] for c in cases], device=device)
    target = (target - float(checkpoint['scaler']['mean'])) / float(checkpoint['scaler']['std'])
    optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=1e-3)
    losses = []
    trainer.model.train()
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = (trainer._forward(batch) - target).square().mean()
        loss.backward()
        assert torch.isfinite(loss)
        assert all(torch.isfinite(p.grad).all() for p in trainer.model.parameters() if p.grad is not None)
        optimizer.step()
        losses.append(float(loss.detach()))
    trainer.model.eval()
    restored = MEGNetTrainer(copy.deepcopy(config), device, seed=123)
    restored.model.load_state_dict(trainer.model.state_dict(), strict=True)
    restored.model.eval()
    with torch.no_grad():
        torch.testing.assert_close(trainer._forward(batch), restored._forward(batch), atol=2e-6, rtol=2e-5)
    report['variants'][aa] = {
        'parameters': sum(p.numel() for p in trainer.model.parameters()),
        'batch_single_max_difference': float((together - alone).abs().max()),
        'two_training_steps_losses': losses,
        'edges': [{t[1]: g[t].num_edges for t in g.edge_types} for g in graphs],
        'strict_restore': 'passed', 'finite_gradients': 'passed',
    }
    del restored, trainer, optimizer, batch
    torch.cuda.empty_cache()

out = ROOT / 'results/hetero_no_aa_validation_20260917.json'
out.write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
