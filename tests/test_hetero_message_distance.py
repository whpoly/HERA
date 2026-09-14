"""Independent hetero message/radial-encoder ablations and saved-run compatibility."""
import copy
import io
from contextlib import redirect_stdout, redirect_stderr
from itertools import product
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch_geometric.loader import DataLoader

from HERA import main as training_cli
from HERA.config.defaults import get_config, apply_alignn_hetero_options, alignn_hetero_run_components
from HERA.data.datasets import init_elem_embedding
from HERA.models.alignn import HeteroRelationConv, SharedHeteroDistanceEmbedding
from HERA.native_ood_case_study import model_mode_display
from HERA.sparse_megnet_alignn import alignn_result_prefix
from HERA.training.trainer import MEGNetTrainer
from HERA.tests.test_hypergraph import make_two_independent_defect_structure, make_periodic_three_region_structure


ROOT = Path(__file__).resolve().parents[1]


class HeteroMessageDistanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        init_elem_embedding(ROOT / 'atom_init.json')

    def trainer(self, message='linear', distance='independent', relation='shared_residual'):
        config = get_config('alignn', '2dmd_wse2', 'hetero')
        config['model'].update(embedding_size=8, nblocks=1, gcn_blocks=1,
                               edge_embed_size=4, angle_embed_size=4, local_radius=0)
        apply_alignn_hetero_options(config, pooling='defect_energy_mean',
                                   relation_mode=relation, message_mode=message, distance_mode=distance)
        torch.manual_seed(123)
        return MEGNetTrainer(config, 'cpu', seed=123)

    def graphs(self, trainer):
        structure = make_two_independent_defect_structure()
        structure.add_site_property('pool_type', [1, 0, 1, 0, 0])
        structure.add_site_property('type', [1, 1, 1, 0, 0])
        return [trainer.converter.convert(structure),
                trainer.converter.convert(make_periodic_three_region_structure())]

    def test_pair_content_uses_destination_and_geometry_even_with_constant_gates(self):
        for mode in ('linear', 'pair_mlp'):
            torch.manual_seed(123)
            layer = HeteroRelationConv(8, 8, normalization='layernorm', message_mode=mode)
            with torch.no_grad():
                for gate in (layer.src_gate, layer.dst_gate, layer.edge_gate):
                    gate.weight.zero_()
                    gate.bias.zero_()
            src = torch.randn(3, 8, requires_grad=True)
            dst = torch.randn(2, 8, requires_grad=True)
            features = torch.randn(3, 8, requires_grad=True)
            edges = torch.tensor([[0, 1, 2], [0, 0, 1]])
            messages, gates, _ = layer((src, dst), edges, features)
            torch.testing.assert_close(gates[:, 0], torch.tensor([1., .5]))
            messages.square().sum().backward()
            self.assertGreater(src.grad.abs().sum(), 0)
            for value in (dst, features):
                if mode == 'pair_mlp':
                    self.assertGreater(value.grad.abs().sum(), 0)
                else:
                    torch.testing.assert_close(value.grad, torch.zeros_like(value))

    def test_shared_radial_encoder_gets_gradients_from_every_relation(self):
        model = self.trainer(distance='shared').model
        bank = model.edge_embedding
        self.assertIsInstance(bank, SharedHeteroDistanceEmbedding)
        self.assertEqual(len(bank.relation_embedding), 4)
        features = torch.randn(4, 4)
        outputs = [bank(key, features) for key in bank.relation_embedding]
        for output in outputs[1:]:
            torch.testing.assert_close(output, outputs[0], rtol=0, atol=0)
        for key in bank.relation_embedding:
            bank.zero_grad(set_to_none=True)
            bank(key, features).square().sum().backward()
            self.assertGreater(bank.shared[0].layer[0].weight.grad.abs().sum(), 0)
            self.assertGreater(bank.relation_embedding[key].grad.abs().sum(), 0)
            for other in set(bank.relation_embedding) - {key}:
                self.assertIsNone(bank.relation_embedding[other].grad)

    def test_unchanged_parameters_and_physical_inputs_have_identical_initialization(self):
        baseline = self.trainer()
        original = baseline.model.state_dict()
        reference_graphs = self.graphs(baseline)
        for message, distance in (('pair_mlp', 'independent'), ('linear', 'shared')):
            candidate = self.trainer(message, distance)
            state = candidate.model.state_dict()
            for key in set(original) & set(state):
                torch.testing.assert_close(original[key], state[key], rtol=0, atol=0, msg=key)
            if distance == 'shared':
                for key, value in candidate.model.edge_embedding.shared.state_dict().items():
                    torch.testing.assert_close(value, original[f'edge_embedding.atom__aa__atom.{key}'], rtol=0, atol=0)
            for graph, reference in zip(self.graphs(candidate), reference_graphs):
                self.assertEqual(graph.node_types, reference.node_types)
                self.assertEqual(graph.edge_types, reference.edge_types)
                for node_type in graph.node_types:
                    for field in ('x', 'pool_type'):
                        torch.testing.assert_close(graph[node_type][field], reference[node_type][field], rtol=0, atol=0)
                for edge_type in graph.edge_types:
                    for field in ('edge_index', 'edge_attr', 'edge_vec'):
                        torch.testing.assert_close(graph[edge_type][field], reference[edge_type][field], rtol=0, atol=0)

    def test_combinations_preserve_energy_mean_batching_and_strict_restore(self):
        for message, distance, relation in product(('linear', 'pair_mlp'), ('independent', 'shared'),
                                                   ('independent', 'shared', 'shared_residual')):
            with self.subTest(message=message, distance=distance, relation=relation):
                trainer = self.trainer(message, distance, relation)
                graphs = self.graphs(trainer)
                batch = next(iter(DataLoader(graphs, batch_size=2)))
                captured = {}
                hook = trainer.model.readout.register_forward_hook(
                    lambda module, args, result: captured.update(energies=result))
                try:
                    together = trainer._forward(batch)
                finally:
                    hook.remove()
                ids = torch.cat([batch[t].batch[batch[t].pool_type.eq(1)] for t in batch.node_types])
                expected = torch.stack([captured['energies'][ids.eq(i)].mean() for i in range(2)])
                torch.testing.assert_close(together, expected)
                alone = torch.cat([trainer._forward(b) for b in DataLoader(graphs, batch_size=1)])
                torch.testing.assert_close(together, alone, atol=3e-6, rtol=2e-5)
                together.square().sum().backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in trainer.model.parameters() if p.grad is not None))
                self.assertGreater(trainer.model.node_embedding['atom'].layer[0].weight.grad.abs().sum(), 0)
                config = copy.deepcopy(trainer.config)
                if message == 'linear' and distance == 'independent':
                    config['model'].pop('hetero_message_mode')
                    config['model'].pop('hetero_distance_mode')
                restored = MEGNetTrainer(config, 'cpu', seed=123)
                restored.model.load_state_dict(trainer.model.state_dict(), strict=True)
                torch.testing.assert_close(together, restored._forward(batch))

    def test_empty_relations_and_single_defect_remain_supported(self):
        for message, distance in (('pair_mlp', 'independent'), ('linear', 'shared'), ('pair_mlp', 'shared')):
            trainer = self.trainer(message, distance)
            for no_host in (False, True):
                structure = make_periodic_three_region_structure()
                if no_host:
                    structure.add_site_property('type', [1, 1, 1])
                else:
                    structure.remove_sites([1, 2])
                batch = next(iter(DataLoader([trainer.converter.convert(structure)], batch_size=1)))
                output = trainer._forward(batch)
                self.assertTrue(torch.isfinite(output).all())
                output.sum().backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in trainer.model.parameters() if p.grad is not None))

    def test_sweep_has_only_requested_independent_changes_and_unique_result_names(self):
        run = {'label': 'hetero_r0', 'mode': 'hetero', 'config': self.trainer().config,
               'local_cutoff': 0, 'radius_label': 'r0'}
        before = copy.deepcopy(run)
        variants = training_cli.expand_alignn_hetero_encoder_runs(
            [run], ['baseline', 'pair_message', 'shared_distance', 'pair_message'], True)
        self.assertEqual(run, before)
        self.assertEqual(len(variants), 3)
        paths, labels, prefixes = set(), set(), set()
        for variant, expected in zip(variants, [('linear', 'independent'), ('pair_mlp', 'independent'), ('linear', 'shared')]):
            model = variant['config']['model']
            self.assertEqual((model['hetero_message_mode'], model['hetero_distance_mode']), expected)
            parts = alignn_hetero_run_components(model)
            path = Path('dataset/hetero/r0').joinpath(*parts)
            paths.add(path)
            labels.add(model_mode_display('alignn', variant['label'] + '_' + '_'.join(parts)))
            prefixes.add(alignn_result_prefix(Path('dataset'), path / 'seed123_test_predictions.csv'))
        self.assertEqual((len(paths), len(labels), len(prefixes)), (3, 3, 3))
        for config in (get_config('alignn', '2dmd_wse2', 'attention'), get_config('megnet', '2dmd_wse2', 'hetero')):
            original = copy.deepcopy(config)
            apply_alignn_hetero_options(config, message_mode='pair_mlp', distance_mode='shared')
            self.assertEqual(config, original)
        with self.assertRaises(ValueError):
            self.trainer(message='invalid')
        with self.assertRaises(ValueError):
            self.trainer(distance='invalid')

    def test_one_cli_command_dispatches_exactly_two_runs_and_separates_outputs(self):
        with tempfile.TemporaryDirectory() as output:
            argv = ['HERA.main', '--model', 'alignn', '--dataset', '2dmd_wse2', '--mode', 'hetero',
                    '--r', '0', '--alignn-hetero-pooling', 'defect_energy_mean',
                    '--alignn-hetero-ablation', 'pair_message', 'shared_distance',
                    '--seed', '123', '--epochs', '1', '--device', 'cpu',
                    '--atom-init', str(ROOT / 'atom_init.json'), '--run-dir', output]
            fake_data = (None, [], None, None, [])
            with patch('sys.argv', argv), patch.object(training_cli, 'load_dataset', return_value=fake_data), \
                    patch.object(training_cli, 'train_single_mode', return_value=[.123]) as train, redirect_stdout(io.StringIO()):
                training_cli.main()
            self.assertEqual(train.call_count, 2)
            modes = [(call.args[1]['model']['hetero_message_mode'], call.args[1]['model']['hetero_distance_mode'])
                     for call in train.call_args_list]
            self.assertEqual(modes, [('pair_mlp', 'independent'), ('linear', 'shared')])
            summaries = list(Path(output).rglob('summary.txt'))
            self.assertEqual(len(summaries), 3)  # two variants and one dataset summary
            for conflicting in (['--alignn-hetero-message', 'pair_mlp'], ['--alignn-hetero-distance', 'shared']):
                with patch('sys.argv', argv + conflicting), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    training_cli.main()
                self.assertEqual(error.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
