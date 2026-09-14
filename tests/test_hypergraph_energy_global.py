"""Hypergraph energy readout and isolation of the global-only message branch."""
import unittest
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter

from HERA.config.defaults import (
    HYPERGRAPH_SCHEMA, HYPERGRAPH_UPDATE_MODES, get_config, hypergraph_run_components,
    resolve_hypergraph_pooling,
)
from HERA.data.datasets import init_elem_embedding
from HERA.models.hypergraph import RegionHypergraphInteraction
from HERA.native_initial_relaxed_leave_one_out import (
    expand_leave_one_out_runs, build_alignn_hypergraph_comparison,
)
from HERA.native_ood_case_study import model_mode_display
from HERA.sparse_megnet_alignn import alignn_result_prefix
from HERA.training.trainer import MEGNetTrainer
from HERA.tests.test_hypergraph import (
    make_two_independent_defect_structure, make_periodic_three_region_structure,
    make_overlapping_defect_neighborhood_structure,
)
from HERA.tests.test_hypergraph_v3 import converter


POOLS = ('defect_mean', 'defect_energy_mean')


class HypergraphEnergyGlobalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        init_elem_embedding(Path(__file__).resolve().parents[1] / 'atom_init.json')

    def trainer(self, pool, updates='local_global', name='alignn'):
        config = get_config(name, '2dmd_mos2', 'hypergraph')
        config['model'].update(embedding_size=8, nblocks=1, gcn_blocks=1,
                               n_heads=2, edge_embed_size=4, angle_embed_size=4,
                               hypergraph_pooling=pool)
        if name == 'alignn':
            config['model']['hypergraph_updates'] = updates
        torch.manual_seed(123)
        return MEGNetTrainer(config, 'cpu', seed=123)

    def test_global_only_softmaxes_exclude_local_and_host_nodes(self):
        torch.manual_seed(123)
        interaction = RegionHypergraphInteraction(8, 1, heads=2, schema=HYPERGRAPH_SCHEMA,
                                                  pooling='defect_mean', updates='global_only')
        graph = converter().convert(make_two_independent_defect_structure())
        x = torch.randn(5, 8, requires_grad=True)
        output = interaction.update(0, x, graph.hyperedge_index, graph.hyperedge_type, 3)
        active = interaction.blocks[0]._active_hyperedge_index
        alpha, beta = interaction.blocks[0]._attention_weights
        torch.testing.assert_close(active[0], torch.tensor([0, 2]))
        self.assertTrue(torch.all(graph.hyperedge_type[active[1]].eq(0)))
        torch.testing.assert_close(scatter(alpha, active[1], dim=0, dim_size=3, reduce='sum'),
                                   torch.tensor([[1., 1.], [0., 0.], [0., 0.]]))
        torch.testing.assert_close(beta, torch.ones(2, 2))
        torch.testing.assert_close(output[[1, 3, 4]], x[[1, 3, 4]], atol=0, rtol=0)
        output[0, 0].backward()
        self.assertGreater(x.grad[2].abs().sum(), 0)
        torch.testing.assert_close(x.grad[[1, 3, 4]], torch.zeros(3, 8), atol=0, rtol=0)
        # Local edge contexts are computed as metadata but cannot affect updates.
        changed = x.detach().clone()
        changed[[1, 3, 4]] = torch.randn(3, 8) * 100
        after = interaction.update(0, changed, graph.hyperedge_index, graph.hyperedge_type, 3)
        torch.testing.assert_close(after[[0, 2]], output.detach()[[0, 2]], atol=0, rtol=0)

    def test_energy_readout_nonlinearity_counts_and_gradients(self):
        interaction = RegionHypergraphInteraction(1, 1, schema=HYPERGRAPH_SCHEMA,
                                                  pooling='defect_energy_mean')
        batch = next(iter(DataLoader([converter().convert(s) for s in (
            make_overlapping_defect_neighborhood_structure(), make_periodic_three_region_structure(),
        )], batch_size=2)))
        x = torch.tensor([[1.], [30.], [5.], [90.], [7.], [300.], [900.]], requires_grad=True)
        result = interaction.pool(x, batch.hyperedge_index, batch.hyperedge_type, batch.batch,
                                  2, 5, node_transform=torch.square)
        torch.testing.assert_close(result, torch.tensor([[13.], [49.]]))
        # Merging unchanged defect environments must weight by actual counts.
        merged = interaction.pool(x, torch.tensor([[0, 2, 4], [0, 0, 0]]), torch.tensor([0]),
                                  torch.zeros(7, dtype=torch.long), 1, 1, node_transform=torch.square)
        torch.testing.assert_close(merged, (2 * result[:1] + result[1:]) / 3)
        result.sum().backward()
        torch.testing.assert_close(x.grad, torch.tensor([[1.], [0.], [5.], [0.], [14.], [0.], [0.]]))
        with self.assertRaisesRegex(ValueError, 'prediction head'):
            interaction.pool(x, batch.hyperedge_index, batch.hyperedge_type, batch.batch, 2, 5)
        with self.assertRaisesRegex(ValueError, 'at least one defect'):
            interaction.pool(torch.ones(2, 1), torch.tensor([[0, 1], [0, 0]]), torch.tensor([1]),
                             torch.zeros(2, dtype=torch.long), 1, 1, node_transform=torch.square)

    def test_all_readout_update_combinations_keep_initial_weights_and_inputs(self):
        reference_state, reference_graph = None, None
        for pool in POOLS:
            for updates in HYPERGRAPH_UPDATE_MODES:
                trainer = self.trainer(pool, updates)
                state = trainer.model.state_dict()
                graph = trainer.converter.convert(make_overlapping_defect_neighborhood_structure())
                if reference_state is not None:
                    self.assertEqual(set(state), set(reference_state))
                    for key in state:
                        torch.testing.assert_close(state[key], reference_state[key], atol=0, rtol=0)
                    for key in ('x', 'edge_index', 'edge_attr', 'edge_vec', 'hyperedge_index',
                                'hyperedge_type', 'region_type'):
                        torch.testing.assert_close(graph[key], reference_graph[key], atol=0, rtol=0)
                reference_state, reference_graph = state, graph
        for name in ('hypergraph', 'cgcnn', 'megnet'):
            old, new = self.trainer(POOLS[0], name=name), self.trainer(POOLS[1], name=name)
            self.assertEqual(set(old.model.state_dict()), set(new.model.state_dict()))
            for key, value in old.model.state_dict().items():
                torch.testing.assert_close(value, new.model.state_dict()[key], atol=0, rtol=0)

    def test_energy_forward_uses_each_defect_once_for_all_update_modes(self):
        for updates in HYPERGRAPH_UPDATE_MODES:
            trainer = self.trainer('defect_energy_mean', updates)
            graphs = [trainer.converter.convert(s) for s in (
                make_overlapping_defect_neighborhood_structure(), make_periodic_three_region_structure(),
            )]
            batch = next(iter(DataLoader(graphs, batch_size=2)))
            batch.x.requires_grad_(True)
            captured = {}
            handles = [
                trainer.model.hypergraph.readout.register_forward_pre_hook(
                    lambda module, args: captured.update(x=args[0])),
                trainer.model.readout.register_forward_hook(
                    lambda module, args, result: captured.update(head_input=args[0], energies=result)),
            ]
            try:
                together = trainer._forward(batch)
            finally:
                for handle in handles:
                    handle.remove()
            mask = batch.region_type.eq(0)
            torch.testing.assert_close(captured['head_input'], captured['x'][mask], atol=0, rtol=0)
            self.assertEqual(captured['energies'].shape, (3, 1))
            expected = scatter(captured['energies'], batch.batch[mask], dim=0, dim_size=2, reduce='mean')
            torch.testing.assert_close(together, expected.flatten())
            alone = torch.cat([trainer._forward(b) for b in DataLoader(graphs, batch_size=1)])
            torch.testing.assert_close(together, alone, atol=3e-6, rtol=2e-5)
            trainer.model.eval()
            torch.testing.assert_close(together, trainer._forward(batch), atol=3e-6, rtol=2e-5)
            together.square().sum().backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in trainer.model.parameters() if p.grad is not None))
            # Even global_only keeps the physical host-to-defect message path.
            self.assertGreater(batch.x.grad[~mask].abs().sum(), 0)
            restored = MEGNetTrainer(trainer.config, 'cpu', seed=123)
            restored.model.load_state_dict(trainer.model.state_dict(), strict=True)
            self.assertTrue(restored.model.hypergraph.defect_energy_mean)
            self.assertEqual(restored.model.hypergraph.updates_mode, updates)
            torch.testing.assert_close(together, restored._forward(batch))

    def test_global_only_single_defect_without_bonds_or_hosts(self):
        for pool in POOLS:
            trainer = self.trainer(pool, 'global_only')
            structure = make_periodic_three_region_structure()
            structure.remove_sites([1, 2])
            trainer.converter.cutoff = .01
            trainer.converter.hypergraph_radius = 0
            batch = next(iter(DataLoader([trainer.converter.convert(structure)], batch_size=1)))
            value = trainer._forward(batch)
            self.assertTrue(torch.isfinite(value).all())
            value.sum().backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in trainer.model.parameters() if p.grad is not None))

    def test_eight_experiments_have_distinct_paths_and_comparison_rows(self):
        paths, labels, prefixes = set(), set(), set()
        rows = [dict(model='alignn', mode='full', material='MoS2', protocol='p', seed=123, mae=.2)]
        for pool in POOLS:
            for run in expand_leave_one_out_runs('alignn', ['hypergraph'], None,
                                                hypergraph_pooling=pool, hypergraph_updates=HYPERGRAPH_UPDATE_MODES):
                path = Path('root/hypergraph').joinpath(*hypergraph_run_components(run['config']['model']))
                paths.add(path)
                labels.add(run['label'])
                prefixes.add(alignn_result_prefix(Path('root'), path / 'seed123_test_predictions.csv'))
                display = model_mode_display('alignn', run['label'])
                if pool == 'defect_energy_mean':
                    self.assertIn('mean defect energy', display)
                if run['config']['model']['hypergraph_updates'] == 'global_only':
                    self.assertIn('global-only updates', display)
                rows.append(dict(model='alignn', mode=run['label'], material='MoS2', protocol='p', seed=123, mae=.1))
        self.assertEqual((len(paths), len(labels), len(prefixes)), (8, 8, 8))
        self.assertEqual(len(build_alignn_hypergraph_comparison(pd.DataFrame(rows))), 8)
        with self.assertRaisesRegex(ValueError, 'supports pooling'):
            resolve_hypergraph_pooling('per_defect_neighborhood_v2', 'defect_energy_mean')
        config = get_config('alignn', '2dmd_mos2', 'hypergraph')['model']
        self.assertEqual(config['hypergraph_pooling'], 'defect_mean')
        self.assertNotIn('hypergraph_updates', config)


if __name__ == '__main__':
    unittest.main()
