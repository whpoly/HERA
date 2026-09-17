"""Complete actual-defect connectivity: geometry, caps, batching and replay."""
import copy
import io
import json
import os
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from pymatgen.core import Lattice, Structure
from torch_geometric.loader import DataLoader

from HERA import main as cli
from HERA.config.defaults import get_config, apply_alignn_hetero_options, alignn_hetero_run_components
from HERA.data.converters import SimpleCrystalConverter, DummyConverter
from HERA.data.datasets import init_elem_embedding
from HERA.training.trainer import MEGNetTrainer

ROOT = Path(__file__).resolve().parents[1]
DD = ('defect', 'dd', 'defect')


class Ones:
    def convert(self, structure):
        return np.ones((len(structure), 1))


def structure(coords, types=None, pools=None, lattice=None):
    types = [1] * len(coords) if types is None else types
    return Structure(lattice or Lattice.cubic(80), ['W'] * len(coords), coords,
                     coords_are_cartesian=True,
                     site_properties={'type': types, 'pool_type': types if pools is None else pools})


class CompleteDefectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        init_elem_embedding(ROOT / 'atom_init.json')

    def converter(self, connectivity='complete', cutoff=6, cap=12):
        return SimpleCrystalConverter('alignn_hetero', Ones(), DummyConverter(),
                                      cutoff=cutoff, max_neighbors=cap,
                                      hetero_defect_connectivity=connectivity)

    def test_all_pairs_bypass_cutoff_and_neighbor_cap_without_duplicate_edges(self):
        s = structure([[2 + i, 2, 2] for i in range(16)])
        base = self.converter('physical', cutoff=2, cap=1).convert(s)
        candidate = self.converter(cutoff=2, cap=1).convert(s)
        edges = candidate[DD].edge_index
        self.assertEqual(edges.size(1), 16 * 15)
        self.assertEqual(set(map(tuple, edges.T.tolist())),
                         {(i, j) for i in range(16) for j in range(16) if i != j})
        count = base[DD].edge_index.size(1)
        for field in ('edge_attr', 'edge_vec'):
            torch.testing.assert_close(candidate[DD][field][:count], base[DD][field], rtol=0, atol=0)
        torch.testing.assert_close(edges[:, :count], base[DD].edge_index, rtol=0, atol=0)
        torch.testing.assert_close(candidate[DD].edge_attr[:, 0], candidate[DD].edge_vec.norm(dim=1))
        self.assertGreater(candidate[DD].edge_attr.max(), 12)

    def test_periodic_minimum_images_and_opposite_vectors_in_skew_cell(self):
        lattice = Lattice.from_parameters(20, 22, 40, 90, 90, 65)
        s = structure(lattice.get_cartesian_coords([[.04, .05, .5], [.93, .89, .5]]), lattice=lattice)
        edges = self.converter(cutoff=.1).convert(s)[DD]
        self.assertEqual(edges.edge_index.size(1), 2)
        torch.testing.assert_close(edges.edge_vec[0], -edges.edge_vec[1])
        torch.testing.assert_close(edges.edge_attr[:, 0], torch.full((2,), s.get_distance(0, 1)))
        images = structure([[1, 1, 1]], lattice=Lattice.orthorhombic(5, 30, 30))
        base, complete = self.converter('physical').convert(images), self.converter().convert(images)
        torch.testing.assert_close(base[DD].edge_vec, complete[DD].edge_vec)
        self.assertEqual(complete[DD].edge_index.size(1), 2)  # existing nonzero self-images retained

    def test_only_actual_defects_are_completed_and_other_physical_edges_are_unchanged(self):
        s = structure([[1, 1, 1], [9, 1, 1], [1, 2, 1], [30, 1, 1], [31, 1, 1]],
                      [1, 1, 1, 0, 0], [1, 1, 0, 0, 0])
        base, complete = self.converter('physical').convert(s), self.converter().convert(s)
        for key in base.edge_types:
            if key == DD:
                self.assertEqual(set(map(tuple, complete[key].edge_index.T.tolist())) -
                                 set(map(tuple, base[key].edge_index.T.tolist())), {(0, 1), (1, 0)})
            else:
                for field in ('edge_index', 'edge_attr', 'edge_vec'):
                    torch.testing.assert_close(base[key][field], complete[key][field], rtol=0, atol=0)
        for key in base.node_types:
            torch.testing.assert_close(base[key].pool_type, complete[key].pool_type)
        for types in ([1], [0]):
            graph = self.converter().convert(structure([[1, 1, 1]], types))
            self.assertEqual(graph[DD].edge_index.size(1), 0)

    def test_model_initialization_batching_gradients_and_saved_config(self):
        config = get_config('alignn', '2dmd_wse2', 'hetero')
        config['model'].update(embedding_size=8, nblocks=1, gcn_blocks=1,
                               edge_embed_size=8, angle_embed_size=8, local_radius=0)
        apply_alignn_hetero_options(config, pooling='defect_energy_mean', relation_mode='shared_residual')
        torch.manual_seed(123)
        base = MEGNetTrainer(copy.deepcopy(config), 'cpu', seed=123)
        apply_alignn_hetero_options(config, defect_connectivity='complete')
        torch.manual_seed(123)
        trainer = MEGNetTrainer(config, 'cpu', seed=123)
        for key, value in base.model.state_dict().items():
            torch.testing.assert_close(value, trainer.model.state_dict()[key], atol=0, rtol=0)
        samples = [structure([[1, 1, 1], [9, 1, 1], [1, 2, 1]], [1, 1, 0]),
                   structure([[1, 1, 1]])]
        graphs = [trainer.converter.convert(s) for s in samples]
        batch = next(iter(DataLoader(graphs, batch_size=2)))
        for src, rel, dst in batch.edge_types:
            edges = batch[src, rel, dst].edge_index
            torch.testing.assert_close(batch[src].batch[edges[0]], batch[dst].batch[edges[1]])
        trainer.model.eval()
        output = trainer._forward(batch)
        individual = torch.cat([trainer._forward(b) for b in DataLoader(graphs, batch_size=1)])
        torch.testing.assert_close(output, individual, atol=2e-6, rtol=2e-5)
        output.square().sum().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in trainer.model.parameters() if p.grad is not None))
        restored = MEGNetTrainer(copy.deepcopy(config), 'cpu', seed=123)
        restored.model.load_state_dict(trainer.model.state_dict(), strict=True)
        restored.model.eval()
        replay = next(iter(DataLoader([restored.converter.convert(s) for s in samples], batch_size=2)))
        torch.testing.assert_close(output, restored._forward(replay))
        self.assertEqual(base.converter.hetero_defect_connectivity, 'physical')
        self.assertEqual(alignn_hetero_run_components(config['model'])[-1], 'defect_edges_complete')

    def test_cli_dispatch_and_mode_validation(self):
        with tempfile.TemporaryDirectory() as output:
            argv = ['HERA.main', '--model', 'alignn', '--dataset', '2dmd_wse2', '--mode', 'hetero',
                    '--r', '0', '--alignn-hetero-defect-connectivity', 'complete', '--seed', '123',
                    '--epochs', '1', '--device', 'cpu', '--atom-init', str(ROOT / 'atom_init.json'),
                    '--run-dir', output]
            with patch('sys.argv', argv), patch.object(cli, 'load_dataset', return_value=(None, [], None, None, [])), \
                    patch.object(cli, 'train_single_mode', return_value=[.123]) as train, redirect_stdout(io.StringIO()):
                cli.main()
            self.assertEqual(train.call_count, 1)
            self.assertEqual(train.call_args.args[1]['model']['hetero_defect_connectivity'], 'complete')
            self.assertTrue(any(p.parent.name == 'defect_edges_complete' for p in Path(output).rglob('summary.txt')))
            argv[argv.index('hetero')] = 'full'
            with patch('sys.argv', argv), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                cli.main()
        with self.assertRaises(ValueError):
            self.converter('invalid')
        with self.assertRaises(ValueError):
            SimpleCrystalConverter('alignn_full', hetero_defect_connectivity='complete')

    def test_benchmark_runner_writes_comparison_and_rejects_mismatched_splits(self):
        from HERA.scripts import run_wse2_complete_defects as runner
        original_cwd = Path.cwd()
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                config = get_config('alignn', '2dmd_wse2', 'hetero')
                config['model']['local_radius'] = 0
                sources = {'train': ['train'], 'val': ['val'], 'test': [str(i) for i in range(500)]}
                checkpoint = dict(dataset_name='2dmd_wse2', task='alignn_hetero', config=config,
                                  epochs=500, test_mae=.05, best_val_mae=.004, best_epoch=20, epochs_completed=70,
                                  split={'seed': 123, **{p+'_sources': [{'source_id': s} for s in ids]
                                                        for p, ids in sources.items()}})
                baseline_path = root / 'baseline.pth'
                torch.save(checkpoint, baseline_path)
                predictions = pd.DataFrame({'source_id': sources['test'], 'target': np.ones(500),
                                            'prediction': np.full(500, 1.05), 'absolute_error': np.full(500, .05)})
                predictions.to_csv(root / 'seed123_test_predictions.csv', index=False)
                split = {p+'_X': [SimpleNamespace(source_id=s) for s in ids] for p, ids in sources.items()}

                def completed_training(*args, **kwargs):
                    output = Path(kwargs['log_dir'])
                    saved = copy.deepcopy(checkpoint)
                    saved['config'] = args[1]
                    saved['test_mae'] = .04
                    torch.save(saved, output / 'seed123_best_checkpoint.pth')
                    candidate = predictions.copy()
                    candidate['prediction'], candidate['absolute_error'] = 1.04, .04
                    candidate.to_csv(output / 'seed123_test_predictions.csv', index=False)
                    return [.04]

                argv = ['runner', '--baseline-checkpoint', str(baseline_path), '--run-dir', str(root / 'run')]
                with patch('sys.argv', argv), patch.object(runner, 'load_dataset', return_value=(None, [], None, None, [])), \
                        patch.object(runner, 'iter_concentration_transfer_splits', side_effect=lambda *a: iter([split])), \
                        patch.object(runner, 'train_single_mode', side_effect=completed_training) as train, redirect_stdout(io.StringIO()):
                    runner.main()
                    result = json.loads((root / 'run/comparison.json').read_text())
                    self.assertAlmostEqual(result['relative_mae_improvement_percent'], 20)
                    self.assertEqual(json.loads((root / 'run/status.json').read_text())['phase'], 'complete')
                    self.assertTrue((root / 'run/comparison.md').exists())
                    self.assertEqual(len(pd.read_csv(root / 'run/paired_predictions.csv')), 500)
                    split['val_X'][0].source_id = 'wrong'
                    with self.assertRaisesRegex(ValueError, 'val source order differs'):
                        runner.main()
                    self.assertEqual(train.call_count, 1)
        finally:
            os.chdir(original_cwd)


if __name__ == '__main__':
    unittest.main()
