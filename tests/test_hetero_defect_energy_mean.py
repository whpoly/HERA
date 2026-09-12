"""Per-defect readout: nonlinear composition, actual-defect masks and gradients."""
import unittest
from pathlib import Path

import torch
from torch import nn
from torch_geometric.loader import DataLoader

from HERA.config.defaults import get_config
from HERA.data.datasets import init_elem_embedding
from HERA.native_initial_relaxed_leave_one_out import expand_leave_one_out_runs
from HERA.native_ood_case_study import model_mode_display
from HERA.training.trainer import MEGNetTrainer
from HERA.tests.test_hypergraph import (
    make_two_independent_defect_structure, make_periodic_three_region_structure,
)


class SquareFirstFeature(nn.Module):
    def forward(self, x):
        return x[:, :1].square()


class HeteroDefectEnergyMeanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        init_elem_embedding(Path(__file__).resolve().parents[1] / 'atom_init.json')

    def trainer(self, pooling='defect_energy_mean', relation='shared_residual'):
        config = get_config('alignn', '2dmd_mos2', 'hetero')
        config['model'].update(embedding_size=8, nblocks=1, gcn_blocks=1,
                               edge_embed_size=4, angle_embed_size=4, local_radius=0,
                               hetero_pooling=pooling, hetero_relation_mode=relation)
        torch.manual_seed(123)
        return MEGNetTrainer(config, 'cpu', seed=123)

    def graphs(self, trainer):
        structure = make_two_independent_defect_structure()
        # A pristine member of the enlarged defect region must not enter readout.
        structure.add_site_property('pool_type', [1, 0, 1, 0, 0])
        structure.add_site_property('type', [1, 1, 1, 0, 0])
        return [trainer.converter.convert(structure),
                trainer.converter.convert(make_periodic_three_region_structure())]

    def test_same_initial_parameters_and_backbone_for_both_readouts(self):
        for relation in ('independent', 'shared', 'shared_residual'):
            old = self.trainer('defect_mean', relation)
            new = self.trainer('defect_energy_mean', relation)
            old_state, new_state = old.model.state_dict(), new.model.state_dict()
            self.assertEqual(set(old_state), set(new_state))
            for key in old_state:
                torch.testing.assert_close(old_state[key], new_state[key], atol=0, rtol=0)
            batch = next(iter(DataLoader(self.graphs(new), batch_size=2)))
            captured = []
            for trainer in (old, new):
                handle = trainer.model.gcn_layers[-1].register_forward_hook(
                    lambda module, args, result: captured.append(result[0]))
                try:
                    trainer._forward(batch)
                finally:
                    handle.remove()
            for node_type in captured[0]:
                torch.testing.assert_close(captured[0][node_type], captured[1][node_type], atol=0, rtol=0)

    def test_forward_reads_actual_defects_before_mean_and_preserves_gradients(self):
        for relation in ('independent', 'shared', 'shared_residual'):
            trainer = self.trainer(relation=relation)
            graphs = self.graphs(trainer)
            batch = next(iter(DataLoader(graphs, batch_size=2)))
            captured = {}
            handles = [
                trainer.model.gcn_layers[-1].register_forward_hook(
                    lambda module, args, result: captured.update(nodes=result[0])),
                trainer.model.readout.register_forward_hook(
                    lambda module, args, result: captured.update(inputs=args[0], energies=result)),
            ]
            try:
                together = trainer._forward(batch)
            finally:
                for handle in handles:
                    handle.remove()
            features, graph_ids = [], []
            for node_type, nodes in captured['nodes'].items():
                mask = batch[node_type].pool_type.eq(1)
                features.append(nodes[mask])
                graph_ids.append(batch[node_type].batch[mask])
            graph_ids = torch.cat(graph_ids)
            torch.testing.assert_close(captured['inputs'], torch.cat(features), atol=0, rtol=0)
            self.assertEqual(captured['energies'].shape, (3, 1))
            expected = torch.stack([captured['energies'][graph_ids.eq(i)].mean() for i in range(2)])
            torch.testing.assert_close(together, expected)
            alone = torch.cat([trainer._forward(b) for b in DataLoader(graphs, batch_size=1)])
            torch.testing.assert_close(together, alone, atol=3e-6, rtol=2e-5)
            trainer.model.eval()
            torch.testing.assert_close(together, trainer._forward(batch), atol=3e-6, rtol=2e-5)
            together.square().sum().backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in trainer.model.parameters() if p.grad is not None))
            self.assertGreater(trainer.model.readout[0].weight.grad.abs().sum(), 0)
            self.assertGreater(trainer.model.node_embedding['atom'].layer[0].weight.grad.abs().sum(), 0)

    def test_nonlinear_readout_obeys_defect_weighted_composition(self):
        model = self.trainer().model
        # Graphs [0, 2], [1, 1], [2, 2, 2] have scalar contributions [2, 1, 4].
        # Include one ignored host in each node store and an actual defect in atom.
        atom = torch.tensor([[0.], [99.]], requires_grad=True)
        defect = torch.tensor([[2.], [1.], [1.], [2.], [2.], [2.], [99.]], requires_grad=True)
        nodes = {'atom': atom, 'defect': defect}
        batches = {'atom': torch.tensor([0, 0]), 'defect': torch.tensor([0, 1, 1, 2, 2, 2, 2])}
        masks = {'atom': torch.tensor([1, 0]), 'defect': torch.tensor([1, 1, 1, 1, 1, 1, 0])}
        transform = SquareFirstFeature()
        separate = model._pool_fixed_type(nodes, batches, masks, 1, 3, atom,
                                         require_nonempty=True, node_transform=transform)
        torch.testing.assert_close(separate.flatten(), torch.tensor([2., 1., 4.]))
        feature_mean = model._pool_fixed_type(nodes, batches, masks, 1, 3, atom, require_nonempty=True)
        torch.testing.assert_close(transform(feature_mean).flatten(), torch.tensor([1., 1., 4.]))
        merged = model._pool_fixed_type(nodes, {k: torch.zeros_like(v) for k, v in batches.items()},
                                       masks, 1, 1, atom, require_nonempty=True, node_transform=transform)
        torch.testing.assert_close(merged.flatten(), (separate.flatten() * torch.tensor([2., 2., 3.])).sum().view(1) / 7)
        separate.sum().backward()
        torch.testing.assert_close(atom.grad.flatten(), torch.tensor([0., 0.]))
        torch.testing.assert_close(defect.grad.flatten(), torch.tensor([2., 1., 1., 4/3, 4/3, 4/3, 0.]))

    def test_empty_defects_rejected_and_no_edges_or_host_supported(self):
        trainer = self.trainer()
        for kind in ('no_edges', 'no_host', 'no_defects'):
            structure = make_periodic_three_region_structure()
            if kind == 'no_edges':
                structure.remove_sites([1, 2])
            else:
                structure.add_site_property('type', [int(kind == 'no_host')] * 3)
            graph = trainer.converter.convert(structure)
            graphs = [self.graphs(trainer)[0], graph]
            batch = next(iter(DataLoader(graphs, batch_size=2)))
            if kind == 'no_defects':
                with self.assertRaisesRegex(ValueError, 'at least one defect per graph'):
                    trainer._forward(batch)
            else:
                self.assertTrue(torch.isfinite(trainer._forward(batch)).all())

    def test_config_restoration_and_native_display(self):
        for relation in ('independent', 'shared', 'shared_residual'):
            trainer = self.trainer(relation=relation)
            restored = MEGNetTrainer(trainer.config, 'cpu', seed=123)
            restored.model.load_state_dict(trainer.model.state_dict(), strict=True)
            self.assertEqual(restored.model.pooling, 'defect_energy_mean')
            batch = next(iter(DataLoader(self.graphs(trainer), batch_size=2)))
            torch.testing.assert_close(trainer._forward(batch), restored._forward(batch))
        run = expand_leave_one_out_runs('alignn', ['hetero'], None,
                                       hetero_pooling='defect_energy_mean')[0]
        self.assertIn('_pool_defect_energy_mean_', run['label'])
        self.assertIn('mean defect energy', model_mode_display('alignn', run['label']))
        self.assertEqual(get_config('alignn', '2dmd_mos2', 'hetero')['model']['hetero_pooling'], 'defect_mean')


if __name__ == '__main__':
    unittest.main()
