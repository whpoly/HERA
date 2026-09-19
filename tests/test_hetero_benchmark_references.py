"""Full/full_x reference routing and real vacancy-placeholder semantics."""
import io
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from pymatgen.core import Lattice, Structure
from pymatgen.core.periodic_table import DummySpecies
import torch
from torch_geometric.data import Batch

from HERA import main as cli
from HERA.config.defaults import get_config
from HERA.data.datasets import init_elem_embedding
from HERA.data.structure_utils import convert_to_sparse_native
from HERA.scripts import run_hetero_relation_benchmark as runner
from HERA.training.trainer import MEGNetTrainer


ROOT = Path(__file__).resolve().parents[1]


class HeteroBenchmarkReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        init_elem_embedding(ROOT / 'atom_init.json')

    def args(self, directory):
        return SimpleNamespace(dataset=['native', 'semi', 'imp2d'], reference=['full', 'full_x'],
                               seed=['123'], epochs=1, device='cpu', batch_size=8,
                               run_dir=Path(directory), cv5=False)

    def test_reference_command_uses_homogeneous_modes_and_separate_output(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(directory)
            command = runner.reference_command(args)
            self.assertEqual(command[command.index('--mode') + 1: command.index('--seed')], ['full', 'full_x'])
            self.assertFalse(any(flag.startswith('--alignn-hetero-') for flag in command))
            self.assertEqual(command[command.index('--run-dir') + 1], str(Path(directory) / 'references'))
            hetero = runner.training_command(args, 'no_dd')
            for flag in ('--seed', '--epochs', '--device', '--alignn-train-batch-size', '--alignn-test-batch-size'):
                self.assertEqual(command[command.index(flag) + 1], hetero[hetero.index(flag) + 1])
            args.cv5 = True
            self.assertIn('--cv5', runner.reference_command(args))

    def test_cli_skips_redundant_full_x_only_on_nonvacancy_datasets(self):
        with tempfile.TemporaryDirectory() as directory:
            command = runner.reference_command(self.args(directory))
            with patch('sys.argv', ['HERA.main', *command[3:]]), \
                    patch.object(cli, 'load_dataset', return_value=(None, [], None, None, [])), \
                    patch.object(cli, 'train_single_mode', return_value=[.123]) as train, redirect_stdout(io.StringIO()):
                cli.main()
            self.assertEqual([(c.kwargs['dataset_name'], c.args[0]) for c in train.call_args_list],
                             [('native', 'full'), ('native', 'full_x'), ('semi', 'full'), ('imp2d', 'full')])
            paths = [c.kwargs['log_dir'] for c in train.call_args_list]
            self.assertEqual(len(set(paths)), 4)
            self.assertTrue(all((Path(p) / 'summary.txt').is_file() for p in paths))

    def test_runner_run_counts_and_optional_references(self):
        variants = list(runner.VARIANTS)
        base = ['benchmark', '--variant', *variants, '--dry-run']
        for extra, expected in (([], 36), (['--reference', 'full', 'full_x'], 48)):
            output = io.StringIO()
            with patch('sys.argv', base + extra), patch.object(runner, 'data_errors', return_value=[]), \
                    patch.object(runner.subprocess, 'run') as launch, redirect_stdout(output):
                runner.main()
            self.assertIn(f'{expected} training/test runs;', output.getvalue())
            self.assertEqual('[references]' in output.getvalue(), bool(extra))
            launch.assert_not_called()

    def test_full_x_adds_a_vacancy_site_and_preserves_every_original_atom(self):
        raw = Structure(Lattice.cubic(12), ['Zn', 'O', 'Zn'],
                        [[.4, .5, .5], [.5, .4, .5], [.5, .5, .6]])
        raw.source_id = 'native_vacancy_fixture'
        full = convert_to_sparse_native(raw, 'vacancy', 1, 'alignn_full', None, True, False)
        full_x = convert_to_sparse_native(raw, 'vacancy', 1, 'alignn_full_x', None, True, False)
        hetero = convert_to_sparse_native(raw, 'vacancy', 1, 'alignn_hetero', None, True, False, local_cutoff=0)
        self.assertEqual(len(full), len(raw))
        self.assertEqual(len(full_x), len(raw) + 1)
        self.assertEqual(sum(isinstance(s.specie, DummySpecies) for s in full), 0)
        self.assertEqual(sum(isinstance(s.specie, DummySpecies) for s in full_x), 1)
        self.assertEqual(sum(isinstance(s.specie, DummySpecies) for s in hetero), 1)
        np.testing.assert_array_equal(full_x.frac_coords[:-1], raw.frac_coords)
        self.assertEqual(full_x.species[:-1], raw.species)
        np.testing.assert_array_equal(full_x[-1].frac_coords, [.5, .5, .5])
        self.assertEqual(full_x.source_id, raw.source_id)
        for mode, structure in (('full', full), ('full_x', full_x)):
            config = get_config('alignn', 'native', mode)
            config['model'].update(embedding_size=8, nblocks=1, gcn_blocks=1,
                                   edge_embed_size=4, angle_embed_size=4)
            trainer = MEGNetTrainer(config, 'cpu', seed=123)
            graph = trainer.converter.convert(structure)
            if mode == 'full_x':
                self.assertTrue(graph.x[-1].eq(0).all())
                self.assertTrue(graph.edge_index.eq(len(raw)).any())
            output = trainer._forward(Batch.from_data_list([graph]))
            self.assertTrue(torch.isfinite(output).all())
            output.sum().backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in trainer.model.parameters() if p.grad is not None))

    def test_full_x_does_not_add_x_to_native_nonvacancies(self):
        raw = Structure(Lattice.cubic(12), ['Zn', 'O', 'S'],
                        [[.4, .5, .5], [.5, .4, .5], [.5, .5, .5]])
        full = convert_to_sparse_native(raw, 'others', 1, 'alignn_full', None, True, False)
        full_x = convert_to_sparse_native(raw, 'others', 1, 'alignn_full_x', None, True, False)
        self.assertEqual(full, full_x)
        self.assertFalse(any(isinstance(s.specie, DummySpecies) for s in full_x))


if __name__ == '__main__':
    unittest.main()
