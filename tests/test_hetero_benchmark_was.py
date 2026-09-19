"""WAS variants retain topology, paired protocols and separate checkpoint paths."""
import copy
import io
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from pymatgen.core import Lattice, Structure
from pymatgen.core.periodic_table import DummySpecies
from torch_geometric.data import Batch

from HERA import main as cli
from HERA.config.defaults import get_config, apply_alignn_hetero_options
from HERA.data.converters import AtomFeaturesExtractor
from HERA.data.datasets import init_elem_embedding
from HERA.scripts import run_hetero_relation_benchmark as runner
from HERA.training.trainer import MEGNetTrainer


ROOT = Path(__file__).resolve().parents[1]


class HeteroBenchmarkWasTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        init_elem_embedding(ROOT / 'atom_init.json')

    def test_cli_dispatches_both_modes_for_all_four_variants_and_three_datasets(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = SimpleNamespace(dataset=['native', 'semi', 'imp2d'], mode=['hetero', 'hetero_was'],
                                   seed=['123'], epochs=1, device='cpu', batch_size=8,
                                   run_dir=Path(temporary), cv5=False)
            paths = []
            for variant in runner.VARIANTS:
                command = runner.training_command(args, variant)
                with patch('sys.argv', ['HERA.main', *command[3:]]), \
                        patch.object(cli, 'load_dataset', return_value=(None, [], None, None, [])), \
                        patch.object(cli, 'train_single_mode', return_value=[.123]) as train, redirect_stdout(io.StringIO()):
                    cli.main()
                self.assertEqual(train.call_count, 6)
                for dataset in args.dataset:
                    calls = [c for c in train.call_args_list if c.kwargs['dataset_name'] == dataset]
                    self.assertEqual({c.args[0] for c in calls}, {'hetero', 'hetero_was'})
                    configurations = {c.args[0]: c.args[1] for c in calls}
                    self.assertEqual(configurations['hetero']['model']['atom_features'], 'Z')
                    self.assertEqual(configurations['hetero_was']['model']['atom_features'], 'was_species')
                    first, second = copy.deepcopy(configurations['hetero']), copy.deepcopy(configurations['hetero_was'])
                    first.pop('task'); second.pop('task')
                    first['model'].pop('atom_features'); second['model'].pop('atom_features')
                    self.assertEqual(first, second)
                    self.assertTrue(all(c.args[4] == [123] for c in calls))
                paths.extend(c.kwargs['log_dir'] for c in train.call_args_list)
            self.assertEqual(len(set(paths)), 24)
            self.assertTrue(all((Path(p) / 'summary.txt').is_file() for p in paths))

    def test_was_features_use_reference_labels_and_keep_legacy_missing_label_fallback(self):
        from HERA.data.datasets import elem_embedding
        structure = Structure(Lattice.cubic(12), [DummySpecies(), 'W', 'S'],
                              [[.5, .5, .5], [.4, .5, .5], [.5, .4, .5]],
                              site_properties={'type': [1, 1, 0], 'pool_type': [1, 1, 0], 'was': [42, 42, 16]})
        normal = AtomFeaturesExtractor('Z', 'alignn_hetero').convert(structure)
        was = AtomFeaturesExtractor('was_species', 'alignn_hetero_was').convert(structure)
        self.assertEqual(was.shape, (3, 184))
        np.testing.assert_array_equal(was[:, :92], normal)
        np.testing.assert_array_equal(was[0, 92:], elem_embedding[42])
        np.testing.assert_array_equal(was[1, 92:], elem_embedding[42])
        untagged = structure.copy()
        untagged.remove_site_property('was')
        fallback = AtomFeaturesExtractor('was_species', 'alignn_hetero_was').convert(untagged)
        np.testing.assert_array_equal(fallback[:, :92], fallback[:, 92:])
        self.assertTrue(np.all(fallback[0] == 0))

    def test_was_all_relation_ablations_have_identical_graphs_and_trainable_models(self):
        structure = Structure(Lattice.cubic(12), [DummySpecies(), 'W', 'S', 'S'],
                              [[.5, .5, .5], [.4, .5, .5], [.5, .4, .5], [.5, .5, .6]],
                              site_properties={'type': [1, 1, 0, 0], 'pool_type': [1, 1, 0, 0], 'was': [42, 42, 16, 16]})
        for variant, (aa, dd) in runner.VARIANTS.items():
            original_graph = None
            for mode in runner.HETERO_MODES:
                config = get_config('alignn', 'native', mode)
                config['model'].update(embedding_size=8, nblocks=1, gcn_blocks=1,
                                       edge_embed_size=4, angle_embed_size=4, local_radius=0)
                apply_alignn_hetero_options(config, aa_mode=aa, dd_mode=dd, pooling='defect_energy_mean')
                trainer = MEGNetTrainer(config, 'cpu', seed=123)
                graph = trainer.converter.convert(structure)
                if original_graph is None:
                    original_graph = graph
                else:
                    self.assertEqual(graph.edge_types, original_graph.edge_types)
                    for t in graph.edge_types:
                        for field in ('edge_index', 'edge_attr', 'edge_vec'):
                            torch.testing.assert_close(graph[t][field], original_graph[t][field], rtol=0, atol=0)
                    for t in graph.node_types:
                        self.assertEqual(graph[t].x.size(1), 184)
                        torch.testing.assert_close(graph[t].x[:, :92], original_graph[t].x, rtol=0, atol=0)
                        torch.testing.assert_close(graph[t].pool_type, original_graph[t].pool_type, rtol=0, atol=0)
                trainer.model.eval()
                batch = Batch.from_data_list([graph, graph])
                prediction = trainer._forward(batch)
                prediction.square().sum().backward()
                self.assertTrue(torch.isfinite(prediction).all())
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in trainer.model.parameters() if p.grad is not None))
                restored = MEGNetTrainer(copy.deepcopy(config), 'cpu', seed=123)
                restored.model.load_state_dict(trainer.model.state_dict(), strict=True)
                restored.model.eval()
                torch.testing.assert_close(prediction, restored._forward(batch))

    def test_benchmark_counts_and_was_only_command(self):
        base = ['benchmark', '--variant', *runner.VARIANTS, '--dry-run']
        for modes, expected in ((['hetero', 'hetero_was'], 72), (['hetero_was'], 36), (['hetero'], 36)):
            output = io.StringIO()
            with patch('sys.argv', base + ['--mode', *modes]), patch.object(runner, 'data_errors', return_value=[]), \
                    patch.object(runner.subprocess, 'run') as launch, redirect_stdout(output):
                runner.main()
            self.assertIn(f'{expected} training/test runs;', output.getvalue())
            self.assertIn('--mode ' + ' '.join(modes) + ' --r 0', output.getvalue())
            self.assertNotIn('[references]', output.getvalue())
            launch.assert_not_called()


if __name__ == '__main__':
    unittest.main()
