"""Remove aa topology and parameters while preserving legacy heterogeneous runs."""
import copy
import io
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from pymatgen.core import Lattice, Structure
from torch_geometric.loader import DataLoader

from HERA import main as cli
from HERA.config.defaults import get_config, apply_alignn_hetero_options, alignn_hetero_run_components
from HERA.data.converters import SimpleCrystalConverter
from HERA.data.datasets import init_elem_embedding
from HERA.models.alignn import build_hetero_line_graph
from HERA.native_ood_case_study import model_mode_display
from HERA.sparse_megnet_alignn import alignn_result_prefix
from HERA.training.trainer import MEGNetTrainer, HETERO_EDGE_TYPES


ROOT = Path(__file__).resolve().parents[1]
AA = ('atom', 'aa', 'atom')
RETAINED = tuple(t for t in HETERO_EDGE_TYPES if t != AA)


def sample(all_defects=False, isolated=False):
    lattice = Lattice.cubic(40) if isolated else Lattice.orthorhombic(10, 12, 15)
    coords = [[1, 1, 1], [3, 1, 1], [5, 1, 1], [7, 1, 1]]
    if isolated:
        coords = [[1, 1, 1], [3, 1, 1], [20, 20, 20]]
    types = [1] * len(coords) if all_defects else ([0, 0, 1] if isolated else [0, 0, 1, 1])
    return Structure(lattice, ['Mo', 'S', 'W', 'Se'][:len(coords)], coords,
                     coords_are_cartesian=True, site_properties={'type': types, 'pool_type': types})


class NoAtomAtomTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        init_elem_embedding(ROOT / 'atom_init.json')

    def trainer(self, aa_mode=None, relation='shared_residual', aggregation='relation_mean',
                connectivity='physical', distance='independent', full=False):
        config = get_config('alignn', '2dmd_wse2', 'hetero')
        config['model']['local_radius'] = 0
        if not full:
            config['model'].update(embedding_size=8, nblocks=1, gcn_blocks=1,
                                   edge_embed_size=4, angle_embed_size=4)
        apply_alignn_hetero_options(config, pooling='defect_energy_mean', aa_mode=aa_mode,
                                   relation_mode=relation, aggregation_mode=aggregation,
                                   defect_connectivity=connectivity, distance_mode=distance)
        torch.manual_seed(123)
        return MEGNetTrainer(config, 'cpu', seed=123)

    def test_graph_removes_only_aa_and_keeps_periodic_geometry_and_node_order(self):
        for connectivity in ('physical', 'complete'):
            old = self.trainer(connectivity=connectivity)
            new = self.trainer('drop', connectivity=connectivity)
            for cap in (1, 12):
                old.converter.max_neighbors = new.converter.max_neighbors = cap
                baseline = old.converter.convert(sample())
                graph = new.converter.convert(sample())
                self.assertGreater(baseline[AA].num_edges, 0)
                self.assertEqual(tuple(graph.edge_types), RETAINED)
                self.assertEqual(graph.num_nodes, baseline.num_nodes)
                for node_type in baseline.node_types:
                    for field in ('x', 'pool_type'):
                        torch.testing.assert_close(graph[node_type][field], baseline[node_type][field], rtol=0, atol=0)
                for relation in RETAINED:
                    for field in ('edge_index', 'edge_attr', 'edge_vec', 'bond_batch'):
                        torch.testing.assert_close(graph[relation][field], baseline[relation][field], rtol=0, atol=0)

    def test_line_graph_is_exactly_baseline_without_aa_bonds_and_incident_angles(self):
        old, new = self.trainer(), self.trainer('drop')
        baseline, graph = old.converter.convert(sample()), new.converter.convert(sample())

        def line(g, model):
            offset, offsets = 0, {}
            for relation in model.edge_types:
                offsets[relation] = offset
                offset += g[relation].num_edges
            vectors = torch.cat([g[t].edge_vec for t in model.edge_types])
            return build_hetero_line_graph(g.edge_index_dict, vectors, offsets,
                                          model.edge_types, model.angle_expansion)

        before, before_angles = line(baseline, old.model)
        after, after_angles = line(graph, new.model)
        aa_count = baseline[AA].num_edges
        retained = (before >= aa_count).all(dim=0)
        self.assertGreater(int((~retained).sum()), 0)
        torch.testing.assert_close(after, before[:, retained] - aa_count, rtol=0, atol=0)
        torch.testing.assert_close(after_angles, before_angles[retained], rtol=0, atol=0)

    def test_trainable_batched_models_and_strict_restore_for_relation_modes(self):
        for relation in ('independent', 'shared', 'shared_residual'):
            for aggregation in ('relation_mean', 'cross_relation_attention'):
                for distance in ('independent', 'shared'):
                    with self.subTest(relation=relation, aggregation=aggregation, distance=distance):
                        trainer = self.trainer('drop', relation, aggregation, distance=distance)
                        self.assertEqual(trainer.model.edge_types, RETAINED)
                        self.assertFalse(any('atom__aa__atom' in key for key in trainer.model.state_dict()))
                        for layer in [*trainer.model.layers, *trainer.model.gcn_layers]:
                            self.assertEqual(layer.node_updates['atom'].fusion.num_relations, 1)
                            self.assertEqual(layer.node_updates['defect'].fusion.num_relations, 2)
                        graphs = [trainer.converter.convert(sample()), trainer.converter.convert(sample(all_defects=True))]
                        batch = next(iter(DataLoader(graphs, batch_size=2)))
                        trainer.model.eval()
                        output = trainer._forward(batch)
                        single = torch.cat([trainer._forward(b) for b in DataLoader(graphs, batch_size=1)])
                        torch.testing.assert_close(output, single, atol=3e-6, rtol=2e-5)
                        output.square().sum().backward()
                        self.assertTrue(all(torch.isfinite(p.grad).all() for p in trainer.model.parameters() if p.grad is not None))
                        self.assertGreater(trainer.model.node_embedding['atom'].layer[0].weight.grad.abs().sum(), 0)
                        restored = MEGNetTrainer(copy.deepcopy(trainer.config), 'cpu', seed=123)
                        restored.model.load_state_dict(trainer.model.state_dict(), strict=True)
                        restored.model.eval()
                        torch.testing.assert_close(output, restored._forward(batch), rtol=0, atol=0)
        # Removing aa can leave the ENTIRE graph edgeless, with hosts present.
        trainer = self.trainer('drop')
        graph = trainer.converter.convert(sample(isolated=True))
        self.assertEqual(sum(graph[t].num_edges for t in graph.edge_types), 0)
        output = trainer._forward(next(iter(DataLoader([graph], batch_size=1))))
        self.assertTrue(torch.isfinite(output).all())
        output.sum().backward()

    def test_keep_is_identical_to_legacy_configs_and_no_aa_has_fewer_parameters(self):
        old, explicit, new = self.trainer(full=True), self.trainer('keep', full=True), self.trainer('drop', full=True)
        self.assertNotIn('hetero_aa_mode', old.config['model'])
        self.assertEqual(old.model.edge_types, HETERO_EDGE_TYPES)
        explicit.model.load_state_dict(old.model.state_dict(), strict=True)
        for key, value in old.model.state_dict().items():
            torch.testing.assert_close(value, explicit.model.state_dict()[key], rtol=0, atol=0)
        batch = next(iter(DataLoader([old.converter.convert(sample())], batch_size=1)))
        torch.testing.assert_close(old._forward(batch), explicit._forward(batch), rtol=0, atol=0)
        self.assertEqual(sum(p.numel() for p in old.model.parameters()), 679193)
        self.assertLess(sum(p.numel() for p in new.model.parameters()), 679193)
        self.assertEqual(alignn_hetero_run_components(old.config['model']),
                         alignn_hetero_run_components(explicit.config['model']))

    def test_cli_runs_both_materials_and_keeps_variant_outputs_separate(self):
        with tempfile.TemporaryDirectory() as output:
            paths, labels = [], []
            argv = ['HERA.main', '--model', 'alignn', '--dataset', '2dmd_mos2', '2dmd_wse2',
                    '--mode', 'hetero', '--r', '0', '--alignn-hetero-pooling', 'defect_energy_mean',
                    '--seed', '123', '--epochs', '1', '--device', 'cpu',
                    '--atom-init', str(ROOT / 'atom_init.json'), '--run-dir', output]
            for aa in ('keep', 'drop'):
                with patch('sys.argv', argv + ['--alignn-hetero-aa', aa]), \
                        patch.object(cli, 'load_dataset', return_value=(None, [], None, None, [])), \
                        patch.object(cli, 'train_single_mode', return_value=[.123]) as train, redirect_stdout(io.StringIO()):
                    cli.main()
                self.assertEqual(train.call_count, 2)
                for call in train.call_args_list:
                    self.assertEqual(call.args[1]['model']['hetero_aa_mode'], aa)
                    paths.append(call.kwargs['log_dir'])
                    labels.append((call.kwargs['dataset_name'], call.kwargs['run_label']))
            self.assertEqual(len(set(paths)), 4)
            self.assertEqual(len(set(labels)), 4)
            self.assertTrue(all((Path(path) / 'summary.txt').exists() for path in paths))
            root = Path(output) / 'alignn/2dmd_wse2'
            old_parts = alignn_hetero_run_components(self.trainer('keep').config['model'])
            new_parts = alignn_hetero_run_components(self.trainer('drop').config['model'])
            self.assertIn('no A-A relation', model_mode_display('alignn', 'hetero_r0_' + '_'.join(new_parts)))
            self.assertNotEqual(alignn_result_prefix(root, root.joinpath('hetero', *old_parts, 'seed123_test_predictions.csv')),
                                alignn_result_prefix(root, root.joinpath('hetero', *new_parts, 'seed123_test_predictions.csv')))
            for model, mode in (('megnet', 'hetero'), ('alignn', 'full')):
                invalid = list(argv)
                invalid[invalid.index('--model') + 1] = model
                invalid[invalid.index('--mode') + 1] = mode
                with patch('sys.argv', invalid + ['--alignn-hetero-aa', 'drop']), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    cli.main()

    def test_invalid_options_are_rejected(self):
        with self.assertRaises(ValueError):
            self.trainer('invalid')
        with self.assertRaises(ValueError):
            SimpleCrystalConverter('alignn_full', hetero_aa_mode='drop')
        with self.assertRaises(ValueError):
            SimpleCrystalConverter('alignn_hetero', hetero_aa_mode='invalid')


if __name__ == '__main__':
    unittest.main()
