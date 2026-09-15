import copy
import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from HERA.config.defaults import get_config
from HERA import main as training_cli
from HERA.predict_2dmd_low_checkpoints import (
    apply_checkpoint_model_compatibility,
    discover_low_checkpoints,
    load_material_high_test,
    parse_args,
    predict_checkpoint,
)
from HERA.training.trainer import MEGNetTrainer


def fake_checkpoint(model, mode, dataset='2dmd_low', seed=123):
    return {
        'model': {},
        'scaler': {'mean': 0.0, 'std': 1.0},
        'config': {'model': {}},
        'model_name': model,
        'dataset_name': dataset,
        'mode': mode,
        'run_label': mode,
        'split': {'seed': seed},
    }


class PredictLowCheckpointsTests(unittest.TestCase):
    def test_explicit_all_includes_every_backbone_and_was_but_excludes_high_training(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            expected, payloads = set(), {}
            for model in training_cli.ALL_MODEL_SUITES:
                for mode in training_cli.default_modes_for_model(model):
                    path = root / model / mode / 'seed123_best_checkpoint.pth'
                    payloads[path] = fake_checkpoint(model, mode)
                    expected.add((model, mode))
            payloads[root/'high/seed123_best_checkpoint.pth'] = fake_checkpoint('cgcnn', 'full', '2dmd_high')
            payloads[root/'other_seed/seed42_best_checkpoint.pth'] = fake_checkpoint('cgcnn', 'full', seed=42)
            for path in payloads:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            with patch('HERA.predict_2dmd_low_checkpoints.load_trusted_checkpoint',
                       side_effect=lambda path, map_location: payloads[Path(path)]):
                records = discover_low_checkpoints(root, models=training_cli.ALL_MODEL_SUITES, seeds=[123])
                filtered = discover_low_checkpoints(root, models=['cgcnn'], modes=['definet'], seeds=[123])
            self.assertEqual({(r['model_name'], r['mode']) for r in records}, expected)
            self.assertEqual([(r['model_name'], r['mode']) for r in filtered], [('cgcnn', 'definet')])

    def test_all_cli_selection_and_conflicts(self):
        args = parse_args(['--checkpoint-root', '.', '--model', 'all', '--mode', 'all'])
        self.assertEqual(args.models, list(training_cli.ALL_MODEL_SUITES))
        self.assertIsNone(args.modes)
        for flags in (['--mode', 'full'], ['--model', 'all', 'cgcnn'],
                      ['--model', 'all', '--mode', 'all', 'attention'],
                      ['--model', 'all', '--alignn-mode', 'hetero']):
            with self.subTest(flags=flags), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(['--checkpoint-root', '.', *flags])

    def test_full_training_command_dispatches_both_datasets_and_preserves_hetero_config(self):
        with tempfile.TemporaryDirectory() as output:
            argv = ['HERA.main', '--model', 'all', '--dataset', '2dmd_low', '2dmd_high', '--mode', 'all',
                    '--r', '0', '--alignn-hetero-feature-norm', 'layernorm',
                    '--alignn-hetero-pooling', 'defect_energy_mean', '--alignn-hetero-relations', 'shared_residual',
                    '--alignn-hetero-adapter-rank', '8', '--seed', '123', '--epochs', '500',
                    '--device', 'cpu', '--atom-init', str(Path(__file__).resolve().parents[1]/'atom_init.json'),
                    '--run-dir', output, '--resume']
            with patch('sys.argv', argv), patch.object(training_cli, 'load_dataset', return_value=([], [], [], [], [])), \
                    patch.object(training_cli, 'train_single_mode', return_value=[.123]) as train, redirect_stdout(io.StringIO()):
                training_cli.main()
            actual = {(c.kwargs['model_name'], c.kwargs['dataset_name'], c.args[0]) for c in train.call_args_list}
            expected = {(model, dataset, mode) for model in training_cli.ALL_MODEL_SUITES
                        for dataset in ('2dmd_low', '2dmd_high')
                        for mode in training_cli.default_modes_for_model(model) if mode != 'hetero_fixed_pool'}
            self.assertEqual(actual, expected)
            self.assertEqual(train.call_count, 64)
            for call in train.call_args_list:
                model = call.args[1]['model']
                if call.kwargs['model_name'] == 'alignn' and call.args[0] in ('hetero', 'hetero_was'):
                    self.assertEqual(model['hetero_pooling'], 'defect_energy_mean')
                    self.assertEqual(model['hetero_relation_mode'], 'shared_residual')
                    self.assertEqual(model['hetero_relation_rank'], 8)
                elif call.kwargs['model_name'] != 'alignn':
                    self.assertNotIn('hetero_pooling', model)

    def test_nonfinite_predictions_do_not_write_a_success_result(self):
        checkpoint = fake_checkpoint('cgcnn', 'attention')
        record = {'path': Path('fake.pth'), 'checkpoint': checkpoint, 'model_name': 'cgcnn',
                  'mode': 'attention', 'run_label': 'attention', 'seed': 123}
        trainer = MagicMock()
        trainer.predict_structures.return_value = (float('nan'), [float('nan')])
        with patch('HERA.predict_2dmd_low_checkpoints.MEGNetTrainer', return_value=trainer), \
                patch('HERA.predict_2dmd_low_checkpoints.load_material_high_test', return_value=([object()], [1.])), \
                patch('HERA.predict_2dmd_low_checkpoints.write_test_predictions') as writer:
            with self.assertRaisesRegex(ValueError, 'Nonfinite'):
                predict_checkpoint(record, 'mos2', Path('.'), 'cpu')
        writer.assert_not_called()

    def test_discovery_selects_sparse_and_non_was_alignn_only(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            payloads = {
                root / 'megnet/sparse/seed123_best_checkpoint.pth':
                    fake_checkpoint('megnet', 'sparse'),
                root / 'alignn/full/seed123_best_checkpoint.pth':
                    fake_checkpoint('alignn', 'full'),
                root / 'alignn/hetero_was/seed123_best_checkpoint.pth':
                    fake_checkpoint('alignn', 'hetero_was'),
                root / 'alignn/high/seed123_best_checkpoint.pth':
                    fake_checkpoint('alignn', 'hetero', dataset='2dmd_high'),
            }
            for path in payloads:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()

            with patch(
                'HERA.predict_2dmd_low_checkpoints.load_trusted_checkpoint',
                side_effect=lambda path, map_location: payloads[Path(path)],
            ):
                records = discover_low_checkpoints(root)

        self.assertEqual(
            [(record['model_name'], record['mode']) for record in records],
            [('megnet', 'sparse'), ('alignn', 'full')],
        )

    def test_material_loader_requests_high_only(self):
        structure = SimpleNamespace(
            source_id='high-1',
            concentration='high',
            material='MoS2',
        )
        dataset = ([structure], None, None, None, torch.tensor([1.25]))
        config = {'model': {'local_radius': 0}}
        with patch(
            'HERA.predict_2dmd_low_checkpoints.'
            '_load_data_2dmd_material_transfer',
            return_value=dataset,
        ) as loader:
            structures, targets = load_material_high_test(
                'mos2', 'alignn', 'full', config,
            )

        self.assertEqual(structures, [structure])
        self.assertEqual(float(targets[0]), 1.25)
        loader.assert_called_once_with(
            'MoS2',
            'alignn',
            local_cutoff=None,
            representations=['full'],
            pure_vacancy=False,
            include_low=False,
        )

    def test_hetero_was_preserves_local_radius_during_loading(self):
        structure = SimpleNamespace(source_id='high-1', concentration='high', material='WSe2')
        with patch('HERA.predict_2dmd_low_checkpoints._load_data_2dmd_material_transfer',
                   return_value=(None, [structure], None, None, [1.])) as loader:
            load_material_high_test('wse2', 'cgcnn', 'hetero_was', {'model': {'local_radius': 5}})
        self.assertEqual(loader.call_args.kwargs['local_cutoff'], 5)
        self.assertEqual(loader.call_args.kwargs['representations'], ['hetero'])

    def test_cli_defaults_to_both_materials_and_available_non_was_modes(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            args = parse_args(['--checkpoint-root', str(root)])

        self.assertEqual(args.material, ['mos2', 'wse2'])
        self.assertIsNone(args.alignn_modes)
        self.assertIsNone(args.seeds)
        self.assertEqual(
            args.output_root,
            root.resolve() / 'high_test_predictions',
        )

    def test_legacy_alignn_batchnorm_checkpoint_is_reconstructed_strictly(self):
        for mode in ('attention', 'definet'):
            with self.subTest(mode=mode):
                legacy_config = get_config('alignn', '2dmd_low', mode)
                legacy_config['model']['alignn_feature_normalization'] = (
                    'batchnorm'
                )
                legacy_config['model']['alignn_legacy_residual_norm'] = True
                legacy_trainer = MEGNetTrainer(legacy_config, 'cpu', seed=123)
                state_dict = legacy_trainer.model.state_dict()
                checkpoint = fake_checkpoint('alignn', mode)
                checkpoint['model'] = state_dict
                checkpoint['config'] = get_config('alignn', '2dmd_low', mode)
                record = {
                    'model_name': 'alignn',
                    'mode': mode,
                    'checkpoint': checkpoint,
                }
                restored_config = copy.deepcopy(checkpoint['config'])

                compatibility = apply_checkpoint_model_compatibility(
                    restored_config,
                    record,
                )
                restored_trainer = MEGNetTrainer(
                    restored_config,
                    'cpu',
                    seed=123,
                )
                result = restored_trainer.model.load_state_dict(state_dict)

                self.assertEqual(compatibility, 'legacy_batchnorm')
                self.assertFalse(result.missing_keys)
                self.assertFalse(result.unexpected_keys)


if __name__ == '__main__':
    unittest.main()
