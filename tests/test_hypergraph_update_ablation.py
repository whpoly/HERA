"""Controlled HyperALIGNN ablations: identity, local, and local+global updates."""
import copy
import unittest
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter

from HERA.config.defaults import (
    HYPERGRAPH_SCHEMA, get_config, hypergraph_run_components, resolve_hypergraph_updates,
)
from HERA.data.datasets import init_elem_embedding
from HERA.main import expand_hypergraph_update_runs
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


MODES = ('none', 'local', 'local_global')


class HypergraphUpdateAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        init_elem_embedding(Path(__file__).resolve().parents[1] / 'atom_init.json')

    def interaction(self, updates):
        torch.manual_seed(123)
        return RegionHypergraphInteraction(8, 1, heads=2, schema=HYPERGRAPH_SCHEMA,
                                          pooling='defect_mean', updates=updates)

    def trainer(self, updates):
        config = get_config('alignn', '2dmd_mos2', 'hypergraph')
        config['model'].update(embedding_size=8, nblocks=1, gcn_blocks=1,
                               n_heads=2, edge_embed_size=4, angle_embed_size=4)
        if updates is not None:
            config['model']['hypergraph_updates'] = updates
        torch.manual_seed(123)
        return MEGNetTrainer(config, 'cpu', seed=123)

    def test_none_is_exact_identity_and_skips_the_ffn(self):
        interaction = self.interaction('none')
        graph = converter().convert(make_two_independent_defect_structure())
        x = torch.randn(5, 8, requires_grad=True)
        output = interaction.update(0, x, graph.hyperedge_index, graph.hyperedge_type, 3)
        self.assertIs(output, x)
        output.sum().backward()
        torch.testing.assert_close(x.grad, torch.ones_like(x))
        self.assertTrue(all(p.grad is None for p in interaction.blocks.parameters()))

    def test_local_has_no_message_path_between_disjoint_defect_environments(self):
        graph = converter().convert(make_two_independent_defect_structure())
        for updates in ('local', 'local_global'):
            interaction = self.interaction(updates)
            x = torch.randn(5, 8, requires_grad=True)
            output = interaction.update(0, x, graph.hyperedge_index, graph.hyperedge_type, 3)
            torch.testing.assert_close(output[4], x[4], rtol=0, atol=0)
            output[0, 0].backward()
            if updates == 'local':
                torch.testing.assert_close(x.grad[2], torch.zeros(8), rtol=0, atol=0)
                self.assertGreater(x.grad[1].abs().sum(), 0)
            else:
                self.assertGreater(x.grad[2].abs().sum(), 0)

    def test_local_softmax_excludes_global_in_both_directions(self):
        graph = converter().convert(make_overlapping_defect_neighborhood_structure())
        interaction = self.interaction('local')
        interaction.update(0, torch.randn(4, 8), graph.hyperedge_index, graph.hyperedge_type, 3)
        block = interaction.blocks[0]
        active = block._active_hyperedge_index
        self.assertTrue(torch.all(graph.hyperedge_type[active[1]] == 1))
        alpha, beta = block._attention_weights
        sums = scatter(alpha, active[1], dim=0, dim_size=3, reduce='sum')
        torch.testing.assert_close(sums, torch.tensor([[0., 0.], [1., 1.], [1., 1.]]))
        sums = scatter(beta, active[0], dim=0, dim_size=4, reduce='sum')
        torch.testing.assert_close(sums, torch.tensor([[1., 1.], [1., 1.], [1., 1.], [0., 0.]]))

    def test_local_with_no_local_incidences_is_identity(self):
        interaction = self.interaction('local')
        x = torch.randn(2, 8)
        output = interaction.update(0, x, torch.tensor([[0, 1], [0, 0]]), torch.tensor([0]), 1)
        self.assertIs(output, x)

    def test_variants_have_identical_initial_weights_and_graph_inputs(self):
        reference_state = None
        reference_graph = None
        for updates in MODES:
            trainer = self.trainer(updates)
            model = trainer.model
            state = model.state_dict()
            graph = trainer.converter.convert(make_two_independent_defect_structure())
            self.assertFalse(any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in model.modules()))
            self.assertEqual(model.readout[0].in_features, 8)
            self.assertFalse(hasattr(model, 'node_readout'))
            if reference_state is not None:
                self.assertEqual(list(state), list(reference_state))
                for key in state:
                    torch.testing.assert_close(state[key], reference_state[key], rtol=0, atol=0)
                for key in ('x', 'edge_index', 'edge_attr', 'edge_vec', 'hyperedge_index',
                            'hyperedge_type', 'region_type'):
                    torch.testing.assert_close(graph[key], reference_graph[key], rtol=0, atol=0)
            reference_state, reference_graph = state, graph

    def test_all_variants_pool_actual_defects_and_train_without_batch_leakage(self):
        for updates in MODES:
            with self.subTest(updates=updates):
                trainer = self.trainer(updates)
                graphs = [trainer.converter.convert(s) for s in (
                    make_two_independent_defect_structure(), make_periodic_three_region_structure(),
                )]
                batch = next(iter(DataLoader(graphs, batch_size=2)))
                captured = {}
                handles = [
                    trainer.model.hypergraph.readout.register_forward_pre_hook(
                        lambda module, args: captured.update(x=args[0])),
                    trainer.model.readout.register_forward_pre_hook(
                        lambda module, args: captured.update(pooled=args[0])),
                ]
                together = trainer._forward(batch)
                for handle in handles:
                    handle.remove()
                mask = batch.region_type.eq(0)
                expected = scatter(captured['x'][mask], batch.batch[mask], dim=0, reduce='mean')
                torch.testing.assert_close(captured['pooled'], expected)
                alone = torch.cat([trainer._forward(b) for b in DataLoader(graphs, batch_size=1)])
                torch.testing.assert_close(together, alone, atol=2e-6, rtol=2e-5)
                together.square().sum().backward()
                gradients = [p.grad for p in trainer.model.parameters() if p.grad is not None]
                self.assertTrue(gradients)
                self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
                used = any(p.grad is not None for p in trainer.model.hypergraph.blocks.parameters())
                self.assertEqual(used, updates != 'none')
                restored = MEGNetTrainer(copy.deepcopy(trainer.config), 'cpu', seed=123)
                restored.model.load_state_dict(trainer.model.state_dict(), strict=True)
                self.assertEqual(restored.model.hypergraph.updates_mode, updates)
                torch.testing.assert_close(together, restored._forward(batch))

    def test_missing_saved_option_preserves_the_old_local_global_model(self):
        old = self.trainer(None)
        explicit = self.trainer('local_global')
        explicit.model.load_state_dict(old.model.state_dict(), strict=True)
        graph = old.converter.convert(make_two_independent_defect_structure())
        batch = next(iter(DataLoader([graph], batch_size=1)))
        torch.testing.assert_close(old._forward(batch), explicit._forward(batch), rtol=0, atol=0)

    def test_sweep_paths_labels_and_prediction_columns_are_isolated(self):
        run = {'label': 'hypergraph', 'mode': 'hypergraph',
               'config': get_config('alignn', '2dmd_mos2', 'hypergraph')}
        runs = expand_hypergraph_update_runs([run], [*MODES, 'local'], True)
        self.assertEqual(len(runs), 3)
        self.assertNotIn('hypergraph_updates', run['config']['model'])
        prefixes = []
        for updates, run in zip(MODES, runs):
            self.assertEqual(run['label'], f'hypergraph_updates_{updates}')
            parts = hypergraph_run_components(run['config']['model'])
            self.assertEqual(parts, [HYPERGRAPH_SCHEMA, 'pool_defect_mean', f'updates_{updates}'])
            path = Path('root/hypergraph').joinpath(*parts, 'seed123_test_predictions.csv')
            prefixes.append(alignn_result_prefix(Path('root'), path))
        self.assertEqual(len(set(prefixes)), 3)
        self.assertEqual(expand_hypergraph_update_runs([run], MODES, False), [run])

    def test_incompatible_readouts_schemas_and_backbones_are_rejected(self):
        for schema, pooling in (('per_defect_neighborhood_v2', 'region_mean'),
                                (HYPERGRAPH_SCHEMA, 'hierarchical_attention')):
            with self.assertRaisesRegex(ValueError, 'v3 and defect_mean'):
                resolve_hypergraph_updates(schema, pooling, 'local')
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            resolve_hypergraph_updates(HYPERGRAPH_SCHEMA, 'defect_mean', 'typo')
        config = get_config('megnet', '2dmd_mos2', 'hypergraph')
        config['model']['hypergraph_updates'] = 'local'
        with self.assertRaisesRegex(ValueError, 'only for HyperALIGNN'):
            MEGNetTrainer(config, 'cpu', seed=123)

    def test_native_runner_and_comparison_keep_all_update_variants(self):
        runs = expand_leave_one_out_runs('alignn', ['full', 'hypergraph'], None,
                                        hypergraph_updates=MODES)
        self.assertEqual(len(runs), 4)
        rows = []
        for run in runs:
            rows.append({'model': 'alignn', 'mode': run['label'], 'material': 'MoS2',
                         'protocol': 'direct__final_test', 'seed': 123, 'mae': .1})
        comparison = build_alignn_hypergraph_comparison(pd.DataFrame(rows))
        self.assertEqual(len(comparison), 3)
        self.assertIn('local + global updates', model_mode_display('alignn', runs[-1]['label']))


if __name__ == '__main__':
    unittest.main()
