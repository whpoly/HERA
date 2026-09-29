"""Readout membership, all-node weighting, batch isolation and checkpoint safety."""
import copy
from pathlib import Path
import unittest

import torch
from torch_geometric.loader import DataLoader

from HERA.config.defaults import get_config, alignn_hetero_run_components
from HERA.data.datasets import init_elem_embedding
from HERA.training.trainer import MEGNetTrainer
from HERA.tests.test_hypergraph import (
    make_two_independent_defect_structure, make_periodic_three_region_structure,
)


class HeteroGlobalReadoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        init_elem_embedding(Path(__file__).resolve().parents[1] / 'atom_init.json')

    def trainer(self, pooling):
        config = get_config('alignn', 'native', 'hetero')
        config['model'].update(embedding_size=8, nblocks=1, gcn_blocks=1,
                               edge_embed_size=4, angle_embed_size=4, local_radius=0,
                               hetero_pooling=pooling, hetero_dd_mode='drop')
        torch.manual_seed(123)
        return MEGNetTrainer(config, 'cpu', seed=123)

    def graphs(self, trainer):
        structure = make_two_independent_defect_structure()
        structure.add_site_property('pool_type', [1, 0, 1, 0, 0])
        structure.add_site_property('type', [1, 1, 1, 0, 0])
        return [trainer.converter.convert(structure),
                trainer.converter.convert(make_periodic_three_region_structure())]

    def test_union_mean_is_node_weighted_with_empty_stores_and_batches(self):
        model = self.trainer('global_mean').model
        nodes = {'atom': torch.tensor([[0.], [2.], [4.], [10.]]),
                 'defect': torch.tensor([[10.], [20.], [30.]])}
        batches = {'atom': torch.tensor([0, 0, 0, 1]), 'defect': torch.tensor([0, 1, 1])}
        result = model._pool_all_nodes(nodes, batches, 3, nodes['atom'])
        torch.testing.assert_close(result.flatten(), torch.tensor([4., 20., 0.]))
        nodes['atom'] = nodes['atom'][:0]
        batches['atom'] = batches['atom'][:0]
        result = model._pool_all_nodes(nodes, batches, 3, nodes['defect'])
        torch.testing.assert_close(result.flatten(), torch.tensor([10., 25., 0.]))

    def test_backbone_initialization_identical_and_defaults_unchanged(self):
        baseline = self.trainer('defect_energy_mean').model
        for pooling in ('global_mean', 'defect_global_mean'):
            candidate = self.trainer(pooling).model
            for key, value in baseline.state_dict().items():
                if not key.startswith('readout.'):
                    torch.testing.assert_close(value, candidate.state_dict()[key], atol=0, rtol=0)
            delta = sum(p.numel() for p in candidate.parameters()) - sum(p.numel() for p in baseline.parameters())
            self.assertEqual(delta, 64 if pooling == 'defect_global_mean' else 0)
        global_model = self.trainer('global_mean').model
        for key, value in baseline.state_dict().items():
            torch.testing.assert_close(value, global_model.state_dict()[key], atol=0, rtol=0)
        self.assertEqual(get_config('alignn', 'native', 'hetero')['model']['hetero_pooling'], 'defect_mean')

    def test_forward_membership_batch_isolation_and_host_gradients(self):
        for pooling in ('global_mean', 'defect_global_mean'):
            trainer = self.trainer(pooling)
            graphs = self.graphs(trainer)
            batch = next(iter(DataLoader(graphs, batch_size=2)))
            captured = {}
            handles = [trainer.model.gcn_layers[-1].register_forward_hook(
                lambda module, args, result: captured.update(nodes=result[0])),
                trainer.model.readout.register_forward_pre_hook(
                lambda module, args: captured.update(pooled=args[0]))]
            try:
                together = trainer._forward(batch)
            finally:
                for handle in handles:
                    handle.remove()
            expected = []
            for graph_id in range(2):
                all_nodes, defects = [], []
                for node_type, nodes in captured['nodes'].items():
                    mask = batch[node_type].batch.eq(graph_id)
                    all_nodes.append(nodes[mask])
                    defects.append(nodes[mask & batch[node_type].pool_type.eq(1)])
                global_mean = torch.cat(all_nodes).mean(0)
                defect_mean = torch.cat(defects).mean(0)
                expected.append(torch.cat([defect_mean, global_mean])
                                if pooling == 'defect_global_mean' else global_mean)
            torch.testing.assert_close(captured['pooled'], torch.stack(expected))
            alone = torch.cat([trainer._forward(b) for b in DataLoader(graphs, batch_size=1)])
            torch.testing.assert_close(together, alone, atol=3e-6, rtol=2e-5)
            together.square().sum().backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in trainer.model.parameters() if p.grad is not None))
            self.assertGreater(trainer.model.node_embedding['atom'].layer[0].weight.grad.abs().sum(), 0)

    def test_restore_and_output_paths_keep_readouts_separate(self):
        paths = []
        for pooling in ('defect_energy_mean', 'global_mean', 'defect_global_mean'):
            trainer = self.trainer(pooling)
            restored = MEGNetTrainer(copy.deepcopy(trainer.config), 'cpu', seed=123)
            restored.model.load_state_dict(trainer.model.state_dict(), strict=True)
            self.assertEqual(restored.model.pooling, pooling)
            batch = next(iter(DataLoader(self.graphs(trainer), batch_size=2)))
            torch.testing.assert_close(trainer._forward(batch), restored._forward(batch))
            parts = alignn_hetero_run_components(trainer.config['model'])
            self.assertIn(f'pool_{pooling}', parts)
            paths.append(tuple(parts))
        self.assertEqual(len(set(paths)), 3)

    def test_concat_requires_actual_defect_even_with_promoted_region_nodes(self):
        trainer = self.trainer('defect_global_mean')
        structure = make_periodic_three_region_structure()
        structure.add_site_property('pool_type', [0, 0, 0])
        graph = trainer.converter.convert(structure)
        with self.assertRaisesRegex(ValueError, 'at least one defect per graph'):
            trainer._forward(next(iter(DataLoader([graph], batch_size=1))))


if __name__ == '__main__':
    unittest.main()
