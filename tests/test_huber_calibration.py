"""Calibration must be training-only, reusable and shared by every model."""

import copy
from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock

import numpy as np
import torch

from HERA import main as cli
from HERA.scripts import run_all_dataset_benchmark as runner
from HERA.training import huber_calibration as calibration
from HERA.tests.test_early_stopping_monitor import TinyTrainer


class CalibrationTrainer:
    """Small CPU regression with the production calibration orchestration."""
    instances = []

    def __init__(self, config, device, seed):
        self.config = config
        self.model = torch.nn.Linear(1, 1)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=.01)
        self.instances.append(self)
        self.scheduled = []

    def prepare_data(self, train_X, train_y, val_X, val_y, target_name):
        self.fit_ids = [s.source_id for s in train_X]
        self.val_ids = [s.source_id for s in val_X]
        self.x = torch.tensor([[s.x] for s in train_X], dtype=torch.float32)
        self.y = torch.tensor(train_y, dtype=torch.float32).reshape(-1, 1)
        self.vx = torch.tensor([[s.x] for s in val_X], dtype=torch.float32)
        self.vy = torch.tensor(val_y, dtype=torch.float32).reshape(-1, 1)
        self.fit_mean = float(self.y.mean())

    def train_one_epoch(self):
        self.optimizer.zero_grad()
        prediction = self.model(self.x)
        loss = (prediction - self.y).square().mean()
        loss.backward()
        self.optimizer.step()
        return float((prediction.detach() - self.y).abs().mean()), float(loss.detach())

    def evaluate_on_test(self):
        with torch.no_grad():
            return float((self.model(self.vx) - self.vy).abs().mean())

    def step_scheduler(self, value):
        self.scheduled.append(value)

    def predict_structures(self, X, y, weights, return_predictions=False):
        self.heldout_ids = [s.source_id for s in X]
        self.model.load_state_dict(weights)
        with torch.no_grad():
            predictions = self.model(torch.tensor([[s.x] for s in X], dtype=torch.float32)).numpy().ravel()
        return float(np.abs(predictions - y).mean()), predictions


def example_data(n=40):
    structures = [SimpleNamespace(source_id=f's{i}', x=i / n) for i in range(n)]
    targets = [2 * s.x + (i % 3 - 1) * .15 for i, s in enumerate(structures)]
    return structures, targets


def example_split(seed=11):
    structures, targets = example_data()
    return next(cli.iter_train_val_test_splits(structures, targets, [seed]))


class HuberCalibrationTests(unittest.TestCase):
    def setUp(self):
        CalibrationTrainer.instances = []

    def calibrate(self, output, split=None, **overrides):
        options = dict(epochs=3, folds=3, multiplier=1.345, device='cpu',
                       trainer_factory=CalibrationTrainer, seed_fn=cli.set_seed,
                       weights_fn=cli.cpu_state_dict, clear_fn=lambda device: None)
        options.update(overrides)
        with redirect_stdout(io.StringIO()):
            return calibration.calibrate(
                [split or example_split()], dict(task='alignn_was_x', optim={}, model={}),
                'native', output, **options,
            )

    def test_mad_threshold_units_and_degenerate_fallbacks(self):
        residuals = np.array([-2., -1., 0., 1., 2.])
        result = calibration.residual_threshold(residuals)
        self.assertAlmostEqual(result['delta_ev'], 1.345 / .6744897501960817)
        self.assertAlmostEqual(calibration.residual_threshold(residuals * .01)['delta_ev'],
                               result['delta_ev'] * .01)
        self.assertEqual(calibration.residual_threshold([2., 2., 2.])['scale_method'],
                         'median_absolute_residual')
        self.assertEqual(calibration.residual_threshold([0., 0., 2.])['scale_method'], 'rms_residual')
        self.assertEqual(calibration.residual_threshold([0., 0.])['delta_ev'], 1e-6)
        for residuals in ([], [float('nan')], [float('inf')]):
            with self.assertRaises(ValueError):
                calibration.residual_threshold(residuals)

    def test_all_training_samples_get_out_of_fold_predictions_without_outer_holdouts(self):
        split = example_split()
        split['val_y'], split['test_y'] = object(), object()  # Reading these must fail.
        with tempfile.TemporaryDirectory() as directory:
            manifest = self.calibrate(directory, split)
            expected = {s.source_id for s in split['train_X']}
            predicted = []
            for trainer in CalibrationTrainer.instances:
                fit, val, heldout = map(set, (trainer.fit_ids, trainer.val_ids, trainer.heldout_ids))
                self.assertFalse(fit & val or fit & heldout or val & heldout)
                self.assertEqual(fit | val | heldout, expected)
                self.assertEqual(len(trainer.scheduled), 3)
                predicted += trainer.heldout_ids
            self.assertEqual(len(predicted), len(expected))
            self.assertEqual(set(predicted), expected)
            self.assertEqual(manifest['splits']['11']['statistics']['n'], len(expected))
            rows = json.loads((Path(directory) / 'seed11/oof_predictions.json').read_text())
            expected_delta = calibration.residual_threshold([r['residual'] for r in rows])['delta_ev']
            self.assertEqual(manifest['splits']['11']['statistics']['delta_ev'], expected_delta)
            self.assertEqual(calibration.read_calibration(Path(directory) / 'manifest.json'), manifest)

    def test_equivalent_groups_stay_together_in_outer_and_inner_folds(self):
        groups = [f'g{i // 3}' for i in range(36)]
        folds = calibration.grouped_folds(groups, 3, 11)
        heldout_indices = []
        for pool, holdout in folds:
            fit, val = calibration.inner_partition(pool, groups, 42)
            group_sets = [set(groups[i] for i in indices) for indices in (fit, val, holdout)]
            self.assertFalse(group_sets[0] & group_sets[1])
            self.assertFalse(group_sets[0] & group_sets[2])
            self.assertFalse(group_sets[1] & group_sets[2])
            heldout_indices.extend(holdout)
        self.assertEqual(sorted(heldout_indices), list(range(36)))

    def test_resume_reuses_folds_and_rejects_changed_labels_or_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            first = self.calibrate(directory)
            no_training = Mock(side_effect=AssertionError('Cached folds must not retrain'))
            changed = example_split()
            changed['val_y'], changed['test_y'] = object(), object()
            self.assertEqual(first, self.calibrate(directory, changed, trainer_factory=no_training))
            changed['train_y'][0] += 1
            with self.assertRaisesRegex(ValueError, 'Cached calibration fold differs'):
                self.calibrate(directory, changed, trainer_factory=no_training)
            with self.assertRaisesRegex(ValueError, 'protocol differs'):
                self.calibrate(directory, multiplier=2., trainer_factory=no_training)
            no_training.assert_not_called()

    def test_interrupted_fold_restarts_only_unfinished_fold(self):
        with tempfile.TemporaryDirectory() as directory:
            factory = Mock(side_effect=[CalibrationTrainer({}, 'cpu', 1), RuntimeError('interrupted')])
            with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                self.calibrate(directory, trainer_factory=factory)
            self.assertTrue((Path(directory) / 'seed11/fold1.json').exists())
            self.assertFalse((Path(directory) / 'manifest.json').exists())
            CalibrationTrainer.instances = []
            self.calibrate(directory)
            self.assertEqual(len(CalibrationTrainer.instances), 2)

    def test_tampered_cache_and_manifest_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            self.calibrate(directory)
            path = Path(directory) / 'seed11/fold1.json'
            record = json.loads(path.read_text())
            record['predictions'][0] += 1
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, 'Cached calibration fold differs'):
                self.calibrate(directory)
            path = Path(directory) / 'manifest.json'
            record = json.loads(path.read_text())
            record['splits']['11']['statistics']['delta_ev'] *= 2
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, 'checksum'):
                calibration.read_calibration(path)

    def test_modes_use_same_delta_and_changed_split_is_rejected_before_training(self):
        split = example_split()
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            manifest = self.calibrate(Path(directory) / 'calibration', split)
            identity = calibration.calibration_identity(manifest)
            for mode in ('hetero', 'was_x'):
                config = dict(task=f'alignn_{mode}', model={}, optim=dict(
                    point_loss='huber_ev', huber_calibration=identity,
                    early_stopping_patience=1, early_stopping_monitor_only=True))
                target = Path(directory) / mode
                instances = []

                def factory(*args, **kwargs):
                    trainer = TinyTrainer(*args, **kwargs, values=[.2, .3])
                    instances.append(trainer)
                    return trainer

                data, y = example_data()
                with patch.object(cli, 'MEGNetTrainer', side_effect=factory):
                    cli.train_single_mode(
                        mode, config, [data] * 4, y, [11], 2, 'cpu', 'alignn', 'native',
                        log_dir=str(target), split_iterator=lambda *a, **k: iter([split]),
                    )
                delta = identity['splits']['11']['delta_ev']
                self.assertEqual(instances[0].config['optim']['huber_delta_ev'], delta)
                self.assertNotIn('huber_delta_ev', config['optim'])
                saved = torch.load(target / 'seed11_best_checkpoint.pth', weights_only=False)
                self.assertEqual(saved['resolved_huber_delta_ev'], delta)
                self.assertEqual(saved['config'], config)
                report = json.loads((target / 'seed11_evaluation_comparison.json').read_text())
                self.assertEqual(report['huber_delta_ev'], delta)
            changed = copy.deepcopy(split)
            changed['train_X'] = list(reversed(changed['train_X']))
            with patch.object(cli, 'MEGNetTrainer') as trainer:
                with self.assertRaisesRegex(ValueError, 'split differs'):
                    cli.train_single_mode(
                        'was_x', config, [data] * 4, y, [11], 2, 'cpu', 'alignn', 'native',
                        split_iterator=lambda *a, **k: iter([changed]),
                    )
                trainer.assert_not_called()

    def test_real_cli_calibrates_then_passes_shared_manifest_to_all_modes(self):
        data, y = example_data()
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            args = runner.parse_args([
                '--dataset', '2dmd_low', '--seed', '11', '--point-loss', 'huber_ev',
                '--huber-auto-calibrate', '--huber-calibration-epochs', '2',
                '--run-dir', directory, '--device', 'cpu',
            ])
            reference, training = runner.build_commands(args)
            with patch('sys.argv', reference[3:]), \
                    patch.object(cli, 'load_dataset', return_value=(*([data] * 4), y)), \
                    patch.object(cli, 'MEGNetTrainer', CalibrationTrainer), \
                    patch.object(cli, 'train_single_mode') as train:
                cli.main()
                train.assert_not_called()
            self.assertFalse(list(Path(directory).rglob('*best_checkpoint.pth')))
            manifest = calibration.read_calibration(
                runner.calibration_directory(args, '2dmd_low') / 'manifest.json')
            with patch('sys.argv', training[3:]), \
                    patch.object(cli, 'load_dataset', return_value=(*([data] * 4), y)), \
                    patch.object(cli, 'train_single_mode', return_value=[.1]) as train:
                cli.main()
            self.assertEqual(train.call_count, 9)
            for call in train.call_args_list:
                config = call.args[1]
                self.assertEqual(config['optim']['huber_calibration']['sha256'], manifest['sha256'])
                self.assertNotIn('huber_delta_ev', config['optim'])
                if call.args[0] in runner.HETERO_MODES:
                    self.assertEqual(config['model']['hetero_dd_mode'], 'keep')

    def test_auto_runner_preserves_dd_policy_and_stages_calibration_before_training(self):
        args = runner.parse_args(['--point-loss', 'huber_ev', '--huber-auto-calibrate'])
        commands = runner.build_commands(args)
        self.assertEqual(len(commands), 11)
        for dataset, reference, training in zip(runner.DATASETS, commands[1::2], commands[2::2]):
            self.assertIn('--calibrate-huber-only', reference)
            self.assertEqual(reference[reference.index('--point-loss') + 1], 'mse')
            self.assertNotIn('--early-stopping-monitor-only', reference)
            self.assertNotIn('--huber-delta-ev', training)
            self.assertIn('--huber-calibration-manifest', training)
            dd = training[training.index('--alignn-hetero-dd') + 1]
            self.assertEqual(dd, 'keep' if dataset.startswith('2dmd_') else 'drop')
            self.assertNotIn('hypergraph', training)
            self.assertNotIn('hypergraph_was', training)
        with redirect_stderr(io.StringIO()):
            for flags in (['--huber-auto-calibrate'],
                          ['--point-loss', 'huber_ev', '--huber-auto-calibrate', '--huber-delta-ev', '1'],
                          ['--point-loss', 'huber_ev', '--huber-auto-calibrate',
                           '--huber-delta-ev-by-dataset', 'native=1']):
                with self.assertRaises(SystemExit):
                    runner.parse_args(flags)

    def test_summary_reports_threshold_and_rejects_another_models_delta(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            args = runner.parse_args([
                '--dataset', 'native', '--seed', '11', '--point-loss', 'huber_ev',
                '--huber-auto-calibrate', '--huber-calibration-epochs', '3', '--run-dir', directory,
            ])
            manifest = self.calibrate(runner.calibration_directory(args, 'native'))
            record = dict(
                schema='early_stopping_comparison_v1', model='alignn', dataset='native',
                mode='was_x', seed=11, epochs_requested=500, patience=50, min_delta_percent=.5,
                huber_delta_ev=manifest['splits']['11']['statistics']['delta_ev'],
                huber_calibration_sha256=manifest['sha256'], early_stopping=None,
                final=dict(epochs_completed=500, best_epoch=400, best_val_mae=.1, test_mae=.2),
            )
            target = runner.mode_directory(args, 'native', 'was_x')
            target.mkdir(parents=True)
            path = target / 'seed11_evaluation_comparison.json'
            path.write_text(json.dumps(record))
            summary = runner.summarize(args)
            self.assertEqual(summary['huber_calibrations']['native']['sha256'], manifest['sha256'])
            record['huber_delta_ev'] *= 2
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, 'shared Huber threshold'):
                runner.summarize(args)


if __name__ == '__main__':
    unittest.main()
