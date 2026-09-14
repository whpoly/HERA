"""Joint relation attention and independent sparse defect residual controls."""
import copy
import io
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch_geometric.loader import DataLoader
from pymatgen.core import Lattice, Structure

from HERA import main as training_cli
from HERA.config.defaults import get_config, apply_alignn_hetero_options, alignn_hetero_run_components
from HERA.data.datasets import init_elem_embedding
from HERA.models.alignn import HeteroRelationConv, _hetero_relation_update, _edge_parameter_key
from HERA.native_ood_case_study import model_mode_display
from HERA.sparse_megnet_alignn import alignn_result_prefix
from HERA.training.trainer import MEGNetTrainer, HETERO_EDGE_TYPES
from HERA.tests.test_hypergraph import make_periodic_three_region_structure


ROOT = Path(__file__).resolve().parents[1]


def distant_defects():
    structure = Structure(Lattice.cubic(30), ['Mo', 'S', 'W', 'S'],
                          [[1, 1, 1], [2, 1, 1], [9, 1, 1], [10, 1, 1]], coords_are_cartesian=True)
    # True defects in BOTH node stores, and a pristine atom in the defect store.
    structure.add_site_property('type', [1, 1, 0, 0])
    structure.add_site_property('pool_type', [1, 0, 1, 0])
    return structure


class CaptureSlots(nn.Module):
    def forward(self, x, slots):
        self.slots = slots
        return x


class HeteroInteractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        init_elem_embedding(ROOT / 'atom_init.json')

    def trainer(self, aggregation='relation_mean', residual='none', relation='shared_residual', cutoff=12.):
        config = get_config('alignn', '2dmd_wse2', 'hetero')
        config['model'].update(embedding_size=8, nblocks=1, gcn_blocks=1,
                               edge_embed_size=4, angle_embed_size=4, local_radius=0)
        apply_alignn_hetero_options(config, pooling='defect_energy_mean', relation_mode=relation,
                                   aggregation_mode=aggregation, defect_residual=residual, defect_cutoff=cutoff)
        torch.manual_seed(123)
        return MEGNetTrainer(config, 'cpu', seed=123)

    def test_joint_attention_preserves_relative_relation_mass(self):
        nodes = {'atom': torch.tensor([[1., 0.]]), 'defect': torch.tensor([[0., 1.]])}
        convs = nn.ModuleDict({_edge_parameter_key(t): HeteroRelationConv(2, 2, normalization='layernorm')
                               for t in HETERO_EDGE_TYPES})
        with torch.no_grad():
            for conv in convs.values():
                conv.message_update.weight.copy_(torch.eye(2))
                conv.message_update.bias.zero_()
                for gate in (conv.src_gate, conv.dst_gate, conv.edge_gate):
                    gate.weight.zero_()
                    gate.bias.zero_()
        updates = nn.ModuleDict({t: CaptureSlots() for t in nodes})
        for count in (1, 4):
            edges = {t: torch.empty(2, 0, dtype=torch.long) for t in HETERO_EDGE_TYPES}
            edges[('atom', 'ad', 'defect')] = torch.zeros(2, 1, dtype=torch.long)
            edges[('defect', 'dd', 'defect')] = torch.zeros(2, count, dtype=torch.long)
            features = {t: torch.zeros(e.size(1), 2) for t, e in edges.items()}
            _hetero_relation_update(nodes, edges, features, tuple(nodes), HETERO_EDGE_TYPES,
                                    convs, updates, 'relation_mean')
            torch.testing.assert_close(updates['defect'].slots[0], torch.tensor([[1., 0.]]), atol=3e-6, rtol=0)
            torch.testing.assert_close(updates['defect'].slots[1], torch.tensor([[0., 1.]]), atol=3e-6, rtol=0)
            _hetero_relation_update(nodes, edges, features, tuple(nodes), HETERO_EDGE_TYPES,
                                    convs, updates, 'cross_relation_attention')
            torch.testing.assert_close(updates['defect'].slots[0], torch.tensor([[1 / (1 + count), 0.]]))
            torch.testing.assert_close(updates['defect'].slots[1], torch.tensor([[0., count / (1 + count)]]))
            # Softmax must remain stable at extreme, shared gate-logit offsets.
            with torch.no_grad():
                for conv in convs.values():
                    conv.edge_gate.bias.fill_(1000.)
            _hetero_relation_update(nodes, edges, features, tuple(nodes), HETERO_EDGE_TYPES,
                                    convs, updates, 'cross_relation_attention')
            torch.testing.assert_close(sum(updates['defect'].slots).sum(), torch.tensor(1.))
            with torch.no_grad():
                for conv in convs.values():
                    conv.edge_gate.bias.zero_()

    def test_sparse_edges_have_correct_cutoff_masks_and_preserve_physical_graph(self):
        original = self.trainer()
        candidate = self.trainer(residual='sparse')
        graph = candidate.converter.convert(distant_defects())
        baseline = original.converter.convert(distant_defects())
        for key in HETERO_EDGE_TYPES:
            for field in ('edge_index', 'edge_attr', 'edge_vec'):
                torch.testing.assert_close(graph[key][field], baseline[key][field], atol=0, rtol=0)
        self.assertEqual(graph.node_types, baseline.node_types)
        count = 0
        for src, relation, dst in graph.edge_types:
            if relation != 'sparse_dd':
                continue
            store = graph[src, relation, dst]
            count += store.edge_index.size(1)
            self.assertNotIn('edge_attr', store)
            self.assertTrue(graph[src].pool_type[store.edge_index[0]].eq(1).all())
            self.assertTrue(graph[dst].pool_type[store.edge_index[1]].eq(1).all())
            torch.testing.assert_close(store.edge_vec.norm(dim=1), torch.full((store.edge_index.size(1),), 8.))
        self.assertEqual(count, 2)
        smaller = self.trainer(residual='sparse', cutoff=7).converter.convert(distant_defects())
        self.assertEqual(sum(smaller[t].edge_index.size(1) for t in smaller.edge_types if t[1] == 'sparse_dd'), 0)

    def test_sparse_edges_respect_periodic_images_and_batch_offsets(self):
        trainer = self.trainer(residual='sparse')
        structure = Structure(Lattice.orthorhombic(20, 40, 40), ['Mo', 'W'],
                              [[1, 1, 1], [19, 1, 1]], coords_are_cartesian=True)
        structure.add_site_property('type', [1, 1])
        graph = trainer.converter.convert(structure)
        key = ('defect', 'sparse_dd', 'defect')
        torch.testing.assert_close(graph[key].edge_vec.norm(dim=1), torch.tensor([2., 2.]))
        single = Structure(Lattice.orthorhombic(5, 40, 40), ['Mo'], [[1, 1, 1]], coords_are_cartesian=True)
        single.add_site_property('type', [1])
        images = trainer.converter.convert(single)[key]
        self.assertEqual(images.edge_index.size(1), 4)
        self.assertTrue(images.edge_vec.norm(dim=1).gt(0).all())
        batch = next(iter(DataLoader([graph, trainer.converter.convert(distant_defects())], batch_size=2)))
        for src, relation, dst in batch.edge_types:
            if relation == 'sparse_dd':
                edges = batch[src, relation, dst].edge_index
                torch.testing.assert_close(batch[src].batch[edges[0]], batch[dst].batch[edges[1]])

    def test_unchanged_initialization_and_zero_residual_baseline(self):
        baseline = self.trainer()
        graph = baseline.converter.convert(distant_defects())
        output = baseline._forward(next(iter(DataLoader([graph], batch_size=1))))
        for aggregation, residual in [('cross_relation_attention', 'none'), ('relation_mean', 'sparse')]:
            trainer = self.trainer(aggregation, residual)
            state = trainer.model.state_dict()
            for key, value in baseline.model.state_dict().items():
                torch.testing.assert_close(value, state[key], rtol=0, atol=0, msg=key)
            if residual == 'sparse':
                batch = next(iter(DataLoader([trainer.converter.convert(distant_defects())], batch_size=1)))
                torch.testing.assert_close(output, trainer._forward(batch), rtol=0, atol=0)

    def test_residual_starts_learning_and_uses_geometry(self):
        trainer = self.trainer(residual='sparse')
        batch = next(iter(DataLoader([trainer.converter.convert(distant_defects())], batch_size=1)))
        branch = trainer.model.sparse_defect_residual
        loss = trainer._forward(batch).sum()
        loss.backward()
        self.assertGreater(branch.output.weight.grad.abs().sum(), 0)
        with torch.no_grad():
            branch.output.weight.add_(-.01 * branch.output.weight.grad)
        trainer.model.zero_grad(set_to_none=True)
        output = trainer._forward(batch)
        output.square().sum().backward()
        self.assertGreater(branch.message[0].weight.grad.abs().sum(), 0)
        self.assertGreater(branch.scale_logit.grad.abs(), 0)
        # Test geometry with a visibly active branch, independent of the tiny
        # projection update caused by one synthetic graph's readout gradient.
        with torch.no_grad():
            branch.output.weight.copy_(torch.eye(8))
            branch.scale_logit.zero_()
        output = trainer._forward(batch)
        changed = batch.clone()
        for key in changed.edge_types:
            if key[1] == 'sparse_dd':
                changed[key].edge_vec *= .7
        self.assertGreater((trainer._forward(changed) - output).abs().max(), 1e-8)

    def test_batching_readout_gradients_and_strict_restore(self):
        for relation in ('independent', 'shared', 'shared_residual'):
            for aggregation, residual in [('cross_relation_attention', 'none'), ('relation_mean', 'sparse'),
                                          ('cross_relation_attention', 'sparse')]:
                with self.subTest(relation=relation, aggregation=aggregation, residual=residual):
                    trainer = self.trainer(aggregation, residual, relation)
                    # Exercise nonzero residuals, not just an initially disabled branch.
                    if residual == 'sparse':
                        nn.init.normal_(trainer.model.sparse_defect_residual.output.weight, std=.03)
                    graphs = [trainer.converter.convert(distant_defects()),
                              trainer.converter.convert(make_periodic_three_region_structure())]
                    batch = next(iter(DataLoader(graphs, batch_size=2)))
                    captured = {}
                    hook = trainer.model.readout.register_forward_hook(lambda m, a, out: captured.update(energy=out))
                    together = trainer._forward(batch)
                    hook.remove()
                    graph_ids = torch.cat([batch[t].batch[batch[t].pool_type.eq(1)] for t in batch.node_types])
                    expected = torch.stack([captured['energy'][graph_ids.eq(i)].mean() for i in range(2)])
                    torch.testing.assert_close(together, expected)
                    alone = torch.cat([trainer._forward(b) for b in DataLoader(graphs, batch_size=1)])
                    torch.testing.assert_close(together, alone, atol=3e-6, rtol=2e-5)
                    together.square().sum().backward()
                    self.assertTrue(all(torch.isfinite(p.grad).all() for p in trainer.model.parameters() if p.grad is not None))
                    restored = MEGNetTrainer(copy.deepcopy(trainer.config), 'cpu', seed=123)
                    restored.model.load_state_dict(trainer.model.state_dict(), strict=True)
                    torch.testing.assert_close(together, restored._forward(batch))

    def test_empty_relations_and_missing_auxiliary_graph(self):
        for aggregation, residual in [('cross_relation_attention', 'none'), ('relation_mean', 'sparse')]:
            trainer = self.trainer(aggregation, residual)
            for no_host in (False, True):
                structure = make_periodic_three_region_structure()
                if no_host:
                    structure.add_site_property('type', [1, 1, 1])
                else:
                    structure.remove_sites([1, 2])
                graph = trainer.converter.convert(structure)
                batch = next(iter(DataLoader([graph], batch_size=1)))
                output = trainer._forward(batch)
                self.assertTrue(torch.isfinite(output).all())
                output.sum().backward()
                if residual == 'sparse':
                    del batch['defect', 'sparse_dd', 'defect']
                    with self.assertRaisesRegex(ValueError, 'rebuild graphs'):
                        trainer._forward(batch)

    def test_one_command_dispatches_two_independent_variants_and_distinct_paths(self):
        with tempfile.TemporaryDirectory() as output:
            argv = ['HERA.main', '--model', 'alignn', '--dataset', '2dmd_wse2', '--mode', 'hetero', '--r', '0',
                    '--alignn-hetero-pooling', 'defect_energy_mean', '--alignn-hetero-ablation',
                    'cross_relation_attention', 'sparse_residual', '--seed', '123', '--epochs', '1',
                    '--device', 'cpu', '--atom-init', str(ROOT / 'atom_init.json'), '--run-dir', output]
            with patch('sys.argv', argv), patch.object(training_cli, 'load_dataset', return_value=(None, [], None, None, [])), \
                    patch.object(training_cli, 'train_single_mode', return_value=[.123]) as train, redirect_stdout(io.StringIO()):
                training_cli.main()
            self.assertEqual(train.call_count, 2)
            configs = [call.args[1]['model'] for call in train.call_args_list]
            self.assertEqual([(m['hetero_aggregation_mode'], m['hetero_defect_residual']) for m in configs],
                             [('cross_relation_attention', 'none'), ('relation_mean', 'sparse')])
            self.assertTrue(all(m['hetero_message_mode'] == 'linear' and m['hetero_distance_mode'] == 'independent' for m in configs))
            names = []
            for model in [self.trainer().config['model'], *configs, self.trainer(residual='sparse', cutoff=10).config['model']]:
                parts = alignn_hetero_run_components(model)
                names.append((str(Path('dataset/hetero/r0').joinpath(*parts)),
                              model_mode_display('alignn', 'hetero_r0_' + '_'.join(parts)),
                              alignn_result_prefix(Path('dataset'), Path('dataset/hetero/r0').joinpath(*parts, 'seed123_test_predictions.csv'))))
            for column in zip(*names):
                self.assertEqual(len(set(column)), 4)
            for conflicting in (['--alignn-hetero-aggregation', 'relation_mean'], ['--alignn-hetero-defect-residual', 'sparse'],
                                ['--alignn-hetero-defect-cutoff', 'nan']):
                with patch('sys.argv', argv + conflicting), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    training_cli.main()


if __name__ == '__main__':
    unittest.main()
