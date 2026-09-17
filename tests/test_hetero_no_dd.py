"""Independent DD removal, AA/DD combinations and three-dataset CLI routing."""
import copy
import io
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

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
from HERA.tests.test_hetero_no_aa import sample
from HERA.training.trainer import MEGNetTrainer, HETERO_EDGE_TYPES


ROOT = Path(__file__).resolve().parents[1]
DD = ('defect', 'dd', 'defect')
VARIANTS = [('keep', 'keep'), ('drop', 'keep'), ('keep', 'drop'), ('drop', 'drop')]


class NoDefectDefectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        init_elem_embedding(ROOT / 'atom_init.json')

    def trainer(self, aa='keep', dd=None, relation='shared_residual', aggregation='relation_mean',
                connectivity='physical', distance='independent', full=False):
        config = get_config('alignn', 'native', 'hetero')
        config['model']['local_radius'] = 0
        if not full:
            config['model'].update(embedding_size=8, nblocks=1, gcn_blocks=1,
                                   edge_embed_size=4, angle_embed_size=4)
        apply_alignn_hetero_options(config, pooling='defect_energy_mean', aa_mode=aa, dd_mode=dd,
                                   relation_mode=relation, aggregation_mode=aggregation,
                                   defect_connectivity=connectivity, distance_mode=distance)
        torch.manual_seed(123)
        return MEGNetTrainer(config, 'cpu', seed=123)

    def test_delete_only_dd_including_complete_and_periodic_edges(self):
        periodic = Structure(Lattice.orthorhombic(5, 20, 20), ['Mo'], [[0, 0, 0]],
                             site_properties={'type': [1], 'pool_type': [1]})
        for aa in ('keep', 'drop'):
            for connectivity in ('physical', 'complete'):
                old, new = self.trainer(aa, connectivity=connectivity), self.trainer(aa, 'drop', connectivity=connectivity)
                for structure in (sample(), periodic):
                    before, after = old.converter.convert(structure), new.converter.convert(structure)
                    self.assertGreater(before[DD].num_edges, 0)
                    self.assertNotIn(DD, after.edge_types)
                    self.assertEqual(after.num_nodes, before.num_nodes)
                    for t in before.node_types:
                        for field in ('x', 'pool_type'):
                            torch.testing.assert_close(before[t][field], after[t][field], rtol=0, atol=0)
                    for t in after.edge_types:
                        for field in ('edge_index', 'edge_attr', 'edge_vec', 'bond_batch'):
                            torch.testing.assert_close(before[t][field], after[t][field], rtol=0, atol=0)

    def test_line_graph_has_exactly_the_retained_angles(self):
        def line(graph, model):
            offsets, size = {}, 0
            for relation in model.edge_types:
                offsets[relation] = size
                size += graph[relation].num_edges
            result = build_hetero_line_graph(graph.edge_index_dict,
                                            torch.cat([graph[t].edge_vec for t in model.edge_types]),
                                            offsets, model.edge_types, model.angle_expansion)
            return *result, offsets, size

        for aa in ('keep', 'drop'):
            old, new = self.trainer(aa), self.trainer(aa, 'drop')
            before, after = old.converter.convert(sample()), new.converter.convert(sample())
            old_edges, old_angles, offsets, size = line(before, old.model)
            edges, angles, _, _ = line(after, new.model)
            keep = torch.ones(size, dtype=torch.bool)
            keep[offsets[DD]: offsets[DD] + before[DD].num_edges] = False
            retained = keep[old_edges].all(dim=0)
            remap = torch.full((size,), -1, dtype=torch.long)
            remap[keep] = torch.arange(int(keep.sum()))
            self.assertGreater(int((~retained).sum()), 0)
            torch.testing.assert_close(edges, remap[old_edges[:, retained]], atol=0, rtol=0)
            torch.testing.assert_close(angles, old_angles[retained], atol=0, rtol=0)

    def test_forward_backward_empty_relations_and_strict_restore(self):
        for aa in ('keep', 'drop'):
            for relation in ('independent', 'shared', 'shared_residual'):
                for aggregation in ('relation_mean', 'cross_relation_attention'):
                    with self.subTest(aa=aa, relation=relation, aggregation=aggregation):
                        trainer = self.trainer(aa, 'drop', relation, aggregation)
                        self.assertFalse(any('defect__dd__defect' in k for k in trainer.model.state_dict()))
                        for layer in [*trainer.model.layers, *trainer.model.gcn_layers]:
                            self.assertEqual(layer.node_updates['defect'].fusion.num_relations, 1)
                            self.assertEqual(layer.node_updates['atom'].fusion.num_relations, 2 if aa == 'keep' else 1)
                        graphs = [trainer.converter.convert(s) for s in
                                  (sample(), sample(all_defects=True), sample(isolated=True))]
                        # An all-defect structure becomes an edgeless graph.
                        self.assertEqual(sum(graphs[1][t].num_edges for t in graphs[1].edge_types), 0)
                        batch = next(iter(DataLoader(graphs, batch_size=3)))
                        trainer.model.eval()
                        output = trainer._forward(batch)
                        separate = torch.cat([trainer._forward(b) for b in DataLoader(graphs, batch_size=1)])
                        torch.testing.assert_close(output, separate, atol=3e-6, rtol=2e-5)
                        output.square().sum().backward()
                        self.assertTrue(all(torch.isfinite(p.grad).all() for p in trainer.model.parameters() if p.grad is not None))
                        self.assertGreater(trainer.model.node_embedding['atom'].layer[0].weight.grad.abs().sum(), 0)
                        restored = MEGNetTrainer(copy.deepcopy(trainer.config), 'cpu', seed=123)
                        restored.model.load_state_dict(trainer.model.state_dict(), strict=True)
                        restored.model.eval()
                        torch.testing.assert_close(output, restored._forward(batch), atol=0, rtol=0)
        trainer = self.trainer('drop', 'drop', distance='shared')
        output = trainer._forward(next(iter(DataLoader([trainer.converter.convert(sample())], batch_size=1))))
        self.assertTrue(torch.isfinite(output).all())

    def test_keep_matches_legacy_and_four_variants_have_distinct_identities(self):
        legacy, explicit = self.trainer(), self.trainer(dd='keep')
        self.assertNotIn('hetero_dd_mode', legacy.config['model'])
        for key, value in legacy.model.state_dict().items():
            torch.testing.assert_close(value, explicit.model.state_dict()[key], atol=0, rtol=0)
        batch = next(iter(DataLoader([legacy.converter.convert(sample())], batch_size=1)))
        torch.testing.assert_close(legacy._forward(batch), explicit._forward(batch), atol=0, rtol=0)
        identities, names, counts = [], [], []
        for aa, dd in VARIANTS:
            trainer = self.trainer(aa, dd, full=True)
            parts = alignn_hetero_run_components(trainer.config['model'])
            path = Path('native/hetero/r0').joinpath(*parts, 'seed123_test_predictions.csv')
            identities.append(alignn_result_prefix(Path('native'), path))
            names.append(model_mode_display('alignn', 'hetero_r0_' + '_'.join(parts)))
            counts.append(sum(p.numel() for p in trainer.model.parameters()))
        self.assertEqual(len(set(identities)), 4)
        self.assertEqual(len(set(names)), 4)
        self.assertEqual(counts, [679193, 606819, 606819, 534445])

    def test_cli_dispatches_three_datasets_and_preserves_paired_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for aa, dd in VARIANTS:
                argv = ['HERA.main', '--model', 'alignn', '--dataset', 'native', 'semi', 'imp2d',
                        '--mode', 'hetero', '--r', '0', '--alignn-hetero-aa', aa, '--alignn-hetero-dd', dd,
                        '--alignn-hetero-pooling', 'defect_energy_mean', '--seed', '123', '11', '1245',
                        '--epochs', '1', '--device', 'cpu', '--atom-init', str(ROOT / 'atom_init.json'),
                        '--run-dir', str(Path(directory) / f'{aa}_{dd}')]
                with patch('sys.argv', argv), patch.object(cli, 'load_dataset', return_value=(None, [], None, None, [])), \
                        patch.object(cli, 'train_single_mode', return_value=[.123]) as train, redirect_stdout(io.StringIO()):
                    cli.main()
                self.assertEqual(train.call_count, 9)
                self.assertEqual({call.kwargs['dataset_name'] for call in train.call_args_list}, {'native', 'semi', 'imp2d'})
                for dataset in ('native', 'semi', 'imp2d'):
                    calls = [c for c in train.call_args_list if c.kwargs['dataset_name'] == dataset]
                    self.assertEqual([c.args[4] for c in calls], [[123], [11], [1245]])
                for call in train.call_args_list:
                    self.assertEqual(call.args[1]['model']['hetero_aa_mode'], aa)
                    self.assertEqual(call.args[1]['model']['hetero_dd_mode'], dd)
                    paths.append(call.kwargs['log_dir'])
            self.assertEqual(len(set(paths)), 12)

    def test_invalid_modes_and_nonhetero_cli_are_rejected(self):
        with self.assertRaises(ValueError):
            self.trainer(dd='invalid')
        with self.assertRaises(ValueError):
            SimpleCrystalConverter('alignn_hetero', hetero_dd_mode='invalid')
        with self.assertRaises(ValueError):
            SimpleCrystalConverter('alignn_full', hetero_dd_mode='drop')
        for model, mode in (('megnet', 'hetero'), ('alignn', 'full')):
            argv = ['HERA.main', '--model', model, '--dataset', 'native', '--mode', mode, '--alignn-hetero-dd', 'drop']
            with patch('sys.argv', argv), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                cli.main()

    def test_benchmark_runner_preflight_and_paired_commands(self):
        from HERA.scripts import run_hetero_relation_benchmark as runner
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.assertEqual(len(runner.data_errors(['native', 'imp2d'], root)), 2)
            native = root / runner.DATA_LISTS['native']
            native.parent.mkdir(parents=True)
            native.write_text('', encoding='utf-8')
            self.assertIn('empty data list', runner.data_errors(['native'], root)[0])
            native.write_text('example.cif,1.0\n', encoding='utf-8')
            self.assertEqual(runner.data_errors(['native'], root), [])
            args = SimpleNamespace(dataset=['native', 'semi', 'imp2d'], seed=['123', '11', '1245'],
                                   epochs=500, device='cuda:0', batch_size=8, run_dir=root, cv5=False)
            commands = [runner.training_command(args, name) for name in runner.VARIANTS]
            for name, command in zip(runner.VARIANTS, commands):
                aa, dd = runner.VARIANTS[name]
                self.assertEqual(command[command.index('--alignn-hetero-aa') + 1], aa)
                self.assertEqual(command[command.index('--alignn-hetero-dd') + 1], dd)
                self.assertEqual(command[command.index('--run-dir') + 1], str(root / name))
                self.assertIn('--resume', command)
            with patch('sys.argv', ['benchmark', '--dry-run']), \
                    patch.object(runner, 'data_errors', return_value=['missing data']), \
                    patch.object(runner.subprocess, 'run') as launch, redirect_stdout(io.StringIO()):
                runner.main()
                launch.assert_not_called()
            with patch('sys.argv', ['benchmark']), \
                    patch.object(runner, 'data_errors', return_value=['missing data']), \
                    patch.object(runner.subprocess, 'run') as launch, redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                runner.main()
            launch.assert_not_called()


if __name__ == '__main__':
    unittest.main()
