"""Transfer-sensitive invariants for the defect-global hypergraph version."""
import unittest
from pathlib import Path

import torch
from torch import nn
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter

from HERA.config.defaults import (
    HYPERGRAPH_SCHEMA, LEGACY_HYPERGRAPH_SCHEMA, get_config,
    apply_hypergraph_options, hypergraph_run_components,
)
from HERA.data.converters import SimpleCrystalConverter
from HERA.data.datasets import init_elem_embedding
from HERA.models.hypergraph import RegionHypergraphInteraction, DefectHypergraphBlock
from HERA.training.trainer import MEGNetTrainer
from HERA.tests.test_hypergraph import (
    AtomicNumberConverter, make_two_independent_defect_structure,
    make_periodic_three_region_structure, make_overlapping_defect_neighborhood_structure,
)


def converter(schema=HYPERGRAPH_SCHEMA):
    return SimpleCrystalConverter('alignn_hypergraph', atom_converter=AtomicNumberConverter(),
                                  hypergraph_schema=schema)


class DefectGlobalConversionTests(unittest.TestCase):
    def test_global_defects_and_local_edges_without_far_edge(self):
        structure = make_two_independent_defect_structure()
        graph = converter().convert(structure)
        self.assertEqual(graph.hyperedge_type.tolist(), [0, 1, 1])
        self.assertEqual(graph.hyperedge_index.tolist(), [[0, 2, 0, 1, 2, 3], [0, 0, 1, 1, 2, 2]])
        self.assertEqual(graph.region_type.tolist(), [0, 1, 0, 1, 2])
        self.assertNotIn(4, graph.hyperedge_index[0].tolist())
        legacy = converter(LEGACY_HYPERGRAPH_SCHEMA).convert(structure)
        for attr in ('x', 'edge_index', 'edge_attr', 'edge_vec', 'region_type'):
            torch.testing.assert_close(getattr(graph, attr), getattr(legacy, attr))

    def test_variable_batch_offsets_and_uncovered_node_inference(self):
        graphs = [converter().convert(s) for s in (
            make_two_independent_defect_structure(), make_periodic_three_region_structure(),
        )]
        batch = next(iter(DataLoader(graphs, batch_size=2)))
        self.assertEqual(batch.hyperedge_type.tolist(), [0, 1, 1, 0, 1])
        self.assertEqual(batch.hyperedge_index[1].tolist(), [0, 0, 1, 1, 2, 2, 3, 4, 4])
        interaction = RegionHypergraphInteraction(8, 1, schema=HYPERGRAPH_SCHEMA)
        normalized = interaction.normalize_inputs(
            batch.x, batch.hyperedge_index, batch=batch.batch,
            hyperedge_type=batch.hyperedge_type,
        )
        torch.testing.assert_close(normalized[-1], batch.region_type)

    def test_cross_graph_edge_is_rejected_before_message_passing(self):
        interaction = RegionHypergraphInteraction(8, 1, schema=HYPERGRAPH_SCHEMA)
        with self.assertRaisesRegex(ValueError, 'multiple graphs'):
            interaction.normalize_inputs(torch.randn(2, 8), torch.tensor([[0, 1], [0, 0]]),
                                         batch=torch.tensor([0, 1]), hyperedge_type=torch.tensor([0]))

    def test_legacy_graph_cannot_silently_be_used_as_v3(self):
        graph = converter(LEGACY_HYPERGRAPH_SCHEMA).convert(make_two_independent_defect_structure())
        interaction = RegionHypergraphInteraction(8, 1, schema=HYPERGRAPH_SCHEMA)
        with self.assertRaisesRegex(ValueError, 'reconvert'):
            interaction.normalize_inputs(graph.x, graph.hyperedge_index,
                                         hyperedge_type=graph.hyperedge_type)


class DefectHypergraphAttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.graph = converter().convert(make_overlapping_defect_neighborhood_structure())
        self.interaction = RegionHypergraphInteraction(8, 1, heads=2, schema=HYPERGRAPH_SCHEMA)

    def test_two_softmaxes_normalize_in_their_own_direction(self):
        graph = self.graph
        self.interaction.update(0, torch.randn(4, 8), graph.hyperedge_index,
                                graph.hyperedge_type, 3)
        alpha, beta = self.interaction.blocks[0]._attention_weights
        torch.testing.assert_close(scatter(alpha, graph.hyperedge_index[1], dim=0, reduce='sum'),
                                   torch.ones(3, 2))
        sums = scatter(beta, graph.hyperedge_index[0], dim=0, dim_size=4, reduce='sum')
        torch.testing.assert_close(sums[:3], torch.ones(3, 2))
        torch.testing.assert_close(sums[3], torch.zeros(2))

    def test_global_edge_connects_defects_and_far_node_stays_unchanged(self):
        graph = converter().convert(make_two_independent_defect_structure())
        x = torch.randn(5, 8, requires_grad=True)
        y = self.interaction.update(0, x, graph.hyperedge_index, graph.hyperedge_type, 3)
        torch.testing.assert_close(y[4], x[4], rtol=0, atol=0)
        y[0, 0].backward()
        self.assertGreater(x.grad[2].abs().sum().item(), 0)
        torch.testing.assert_close(x.grad[4], torch.zeros(8))

    def test_identical_defects_do_not_shrink_messages_with_cardinality(self):
        block = DefectHypergraphBlock(8, heads=2).eval()
        x = torch.randn(1, 8)
        attr = torch.randn(1, 8)
        outputs = []
        for count in (1, 2, 16):
            incidence = torch.stack([torch.arange(count), torch.zeros(count, dtype=torch.long)])
            outputs.append(block(x.repeat(count, 1), incidence, attr, 1, torch.tensor([0]))[0])
        for output in outputs[1:]:
            torch.testing.assert_close(output, outputs[0])

    def test_pool_is_permutation_invariant_and_counts_overlap(self):
        graph = self.graph
        x = torch.randn(4, 8, requires_grad=True)
        batch = torch.zeros(4, dtype=torch.long)
        args = (graph.hyperedge_index, graph.hyperedge_type, batch, 1, 3)
        expected = self.interaction.pool(x, *args)
        fractions = self.interaction.readout._attention_weights['fractions']
        torch.testing.assert_close(fractions, torch.tensor([[.5, .25, .25]]))
        permutation = torch.tensor([3, 1, 0, 2])
        inverse = torch.argsort(permutation)
        edge_permutation = torch.tensor([2, 0, 1])
        edge_inverse = torch.argsort(edge_permutation)
        index = torch.stack([inverse[graph.hyperedge_index[0]],
                             edge_inverse[graph.hyperedge_index[1]]]).flip(1)
        actual = self.interaction.pool(x[permutation], index,
                                       graph.hyperedge_type[edge_permutation], batch, 1, 3)
        torch.testing.assert_close(actual, expected)
        expected.square().sum().backward()
        for name, parameter in self.interaction.readout.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_pool_does_not_count_a_shared_host_twice_in_background(self):
        graph = self.graph
        # Uniform scoring makes the exact hierarchical averages inspectable.
        for scorer in (self.interaction.readout.node_score, self.interaction.readout.local_score,
                       self.interaction.readout.host_score):
            for parameter in scorer.parameters():
                nn.init.zeros_(parameter)
        x = torch.tensor([1., 3., 5., 9.]).unsqueeze(1).repeat(1, 8)
        pooled = self.interaction.pool(x, graph.hyperedge_index, graph.hyperedge_type,
                                       torch.zeros(4, dtype=torch.long), 1, 3)
        # Defects=3; local mean(mean(1,3), mean(5,3))=3; unique hosts=mean(3,9)=6.
        torch.testing.assert_close(pooled, torch.tensor([[3., 3., 6.]]).repeat_interleave(8, dim=1))


class DefectHypergraphIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_elem_embedding(Path(__file__).resolve().parents[1] / 'atom_init.json')

    def make_trainer(self, name, schema=HYPERGRAPH_SCHEMA, pooling='defect_mean'):
        config = get_config(name, '2dmd_mos2', 'hypergraph')
        config['model'].update(embedding_size=8, nblocks=1, gcn_blocks=1,
                               edge_embed_size=4, angle_embed_size=4, n_heads=2,
                               hypergraph_schema=schema)
        if pooling is None:
            config['model'].pop('hypergraph_pooling', None)
        else:
            config['model']['hypergraph_pooling'] = pooling
        return MEGNetTrainer(config, 'cpu', seed=123)

    def test_all_backbones_backward_and_batch_independence(self):
        self.check_backbones('defect_mean')

    def test_hierarchical_ablation_backward_and_batch_independence(self):
        self.check_backbones('hierarchical_attention')

    def check_backbones(self, pooling):
        for name in ('hypergraph', 'cgcnn', 'megnet', 'alignn'):
            with self.subTest(model=name):
                torch.manual_seed(123)
                trainer = self.make_trainer(name, pooling=pooling)
                graphs = [trainer.converter.convert(s) for s in (
                    make_two_independent_defect_structure(), make_periodic_three_region_structure(),
                )]
                batched = next(iter(DataLoader(graphs, batch_size=2)))
                trainer.model.train()
                together = trainer._forward(batched)
                self.assertEqual(tuple(together.shape), (2,))
                alone = torch.cat([trainer._forward(b) for b in DataLoader(graphs, batch_size=1)])
                torch.testing.assert_close(together, alone, atol=2e-6, rtol=2e-5)
                together.square().sum().backward()
                for param_name, parameter in trainer.model.named_parameters():
                    if parameter.requires_grad and param_name.startswith('hypergraph.'):
                        self.assertIsNotNone(parameter.grad, param_name)
                    if parameter.grad is not None:
                        self.assertTrue(torch.isfinite(parameter.grad).all(), param_name)
                self.assertTrue(any(p.grad is not None for key, p in trainer.model.named_parameters()
                                    if not key.startswith('hypergraph.')))
                self.assertFalse(any(isinstance(m, nn.BatchNorm1d) for m in trainer.model.modules()))
                # A saved v3 model must recreate strictly through its config.
                restored = self.make_trainer(name, pooling=pooling)
                restored.model.load_state_dict(trainer.model.state_dict(), strict=True)

    def test_all_defects_and_single_defect_without_bonds_are_finite(self):
        for name in ('hypergraph', 'cgcnn', 'megnet', 'alignn'):
            for count in (1, 3):
                with self.subTest(model=name, count=count):
                    trainer = self.make_trainer(name)
                    structure = make_periodic_three_region_structure()
                    structure.remove_sites(list(range(count, 3)))
                    structure.add_site_property('type', [1] * count)
                    trainer.converter.cutoff = .01
                    trainer.converter.hypergraph_radius = 0
                    graph = trainer.converter.convert(structure)
                    batch = next(iter(DataLoader([graph], batch_size=1)))
                    output = trainer._forward(batch)
                    self.assertEqual(tuple(output.shape), (1,))
                    self.assertTrue(torch.isfinite(output).all())
                    output.sum().backward()

    def test_checkpoint_without_pooling_preserves_both_old_architectures(self):
        for schema, pooling in ((LEGACY_HYPERGRAPH_SCHEMA, 'region_mean'),
                                (HYPERGRAPH_SCHEMA, 'hierarchical_attention')):
            for name in ('hypergraph', 'cgcnn', 'megnet', 'alignn'):
                with self.subTest(schema=schema, model=name):
                    explicit = self.make_trainer(name, schema=schema, pooling=pooling)
                    missing = self.make_trainer(name, schema=schema, pooling=None)
                    missing.model.load_state_dict(explicit.model.state_dict(), strict=True)
                    self.assertEqual(missing.model.hypergraph.pooling_mode, pooling)
                    graph = explicit.converter.convert(make_two_independent_defect_structure())
                    batch = next(iter(DataLoader([graph], batch_size=1)))
                    explicit.model.eval()
                    missing.model.eval()
                    torch.testing.assert_close(explicit._forward(batch), missing._forward(batch))

    def test_mean_readout_uses_only_mean_final_defect_features(self):
        for name in ('hypergraph', 'cgcnn', 'megnet', 'alignn'):
            with self.subTest(model=name):
                trainer = self.make_trainer(name)
                graphs = [trainer.converter.convert(s) for s in (
                    make_overlapping_defect_neighborhood_structure(),
                    make_periodic_three_region_structure(),
                )]
                batch = next(iter(DataLoader(graphs, batch_size=2)))
                captured = {}

                def capture(module, args):
                    captured['x'] = args[0]

                hook = trainer.model.hypergraph.readout.register_forward_pre_hook(capture)
                actual = trainer._forward(batch)
                hook.remove()
                defect = batch.region_type.eq(0)
                pooled = scatter(captured['x'][defect], batch.batch[defect], dim=0,
                                 dim_size=2, reduce='mean')
                if name == 'cgcnn':
                    from torch.nn import functional as F
                    h = F.softplus(trainer.model.conv_to_fc(F.softplus(pooled)))
                    for fc, activation in zip(trainer.model.fcs, trainer.model.softpluses):
                        h = activation(fc(h))
                    expected = trainer.model.fc_out(h)
                elif name == 'megnet':
                    expected = trainer.model.hiddens(pooled)
                else:
                    expected = trainer.model.readout(pooled)
                torch.testing.assert_close(actual, expected.flatten())
                self.assertFalse(any(key.startswith(('node_readout.', 'sv.', 'se.', 'state_embedding.'))
                                     for key in trainer.model.state_dict()))


class DefectMeanPoolingTests(unittest.TestCase):
    def setUp(self):
        self.interaction = RegionHypergraphInteraction(1, 1, schema=HYPERGRAPH_SCHEMA,
                                                       pooling='defect_mean')

    def test_each_graph_averages_only_its_defects_with_correct_gradients(self):
        graphs = [converter().convert(s) for s in (
            make_overlapping_defect_neighborhood_structure(),
            make_periodic_three_region_structure(),
        )]
        batch = next(iter(DataLoader(graphs, batch_size=2)))
        x = torch.tensor([[1.], [30.], [5.], [90.], [7.], [300.], [900.]], requires_grad=True)
        pool = self.interaction.pool(x, batch.hyperedge_index, batch.hyperedge_type,
                                    batch.batch, 2, 5)
        torch.testing.assert_close(pool, torch.tensor([[3.], [7.]]))
        pool.sum().backward()
        torch.testing.assert_close(x.grad, torch.tensor([[.5], [0.], [.5], [0.], [1.], [0.], [0.]]))

    def test_defect_count_host_count_and_incidence_overlap_do_not_rescale_pool(self):
        for copies in (1, 2, 12):
            for hosts in (0, 1, 100):
                with self.subTest(defects=2*copies, hosts=hosts):
                    defect_count = 2 * copies
                    x = torch.cat([torch.tensor([[1.], [5.]]).repeat(copies, 1),
                                   torch.full((hosts, 1), 1000.)])
                    index = torch.stack([torch.arange(defect_count), torch.zeros(defect_count, dtype=torch.long)])
                    # Every defect is also incident to its own singleton local edge.
                    local = torch.stack([torch.arange(defect_count), torch.arange(1, defect_count + 1)])
                    index = torch.cat([index, local], dim=1)
                    types = torch.tensor([0] + [1] * defect_count)
                    result = self.interaction.pool(x, index, types,
                                                   torch.zeros(x.size(0), dtype=torch.long),
                                                   1, defect_count + 1)
                    torch.testing.assert_close(result, torch.tensor([[3.]]))
                    perm = torch.randperm(x.size(0))
                    inverse = torch.argsort(perm)
                    permuted_index = torch.stack([inverse[index[0]], index[1]]).flip(1)
                    actual = self.interaction.pool(x[perm], permuted_index, types,
                                                   torch.zeros(x.size(0), dtype=torch.long),
                                                   1, defect_count + 1)
                    torch.testing.assert_close(actual, result)

    def test_pooling_configuration_and_paths_are_unambiguous(self):
        for name in ('hypergraph', 'cgcnn', 'megnet', 'alignn'):
            config = get_config(name, '2dmd_mos2', 'hypergraph')
            self.assertEqual(config['model']['hypergraph_pooling'], 'defect_mean')
            self.assertEqual(hypergraph_run_components(config['model']),
                             [HYPERGRAPH_SCHEMA, 'pool_defect_mean'])
            apply_hypergraph_options(config, pooling='hierarchical_attention')
            self.assertEqual(hypergraph_run_components(config['model']), [HYPERGRAPH_SCHEMA])
            apply_hypergraph_options(config, schema=LEGACY_HYPERGRAPH_SCHEMA)
            self.assertEqual(config['model']['hypergraph_pooling'], 'region_mean')
            self.assertEqual(hypergraph_run_components(config['model']), [LEGACY_HYPERGRAPH_SCHEMA])
            with self.assertRaisesRegex(ValueError, 'supports pooling'):
                apply_hypergraph_options(config, pooling='defect_mean')

    def test_pristine_only_graph_is_not_silently_predicted_as_zero(self):
        with self.assertRaisesRegex(ValueError, 'at least one defect'):
            self.interaction.pool(torch.ones(2, 1), torch.tensor([[0, 1], [0, 0]]),
                                  torch.tensor([1]), torch.zeros(2, dtype=torch.long), 1, 1)


if __name__ == '__main__':
    unittest.main()
