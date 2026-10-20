"""Intermediate held-out evaluation must not change the training trajectory."""
from contextlib import nullcontext, redirect_stdout, redirect_stderr
import csv
import io
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from HERA import main as cli
from HERA.utils.scaler import Scaler
from HERA.training.history import TrainingLogger
from HERA.training.results import completed_split_result
from HERA.training.trainer import MEGNetTrainer
from HERA.tests import test_compact_logs as compact_tests


class TinyTrainer:
    """Exercise the real CLI loop with tiny SGD updates and a controlled val curve."""
    def __init__(self, config, device, seed, values, fail_epoch=None, test_bias=0.0):
        self.config, self.device, self.values = config, device, values
        self.fail_epoch, self.test_bias = fail_epoch, test_bias
        self.model = torch.nn.Linear(1, 1)
        self.model.register_buffer('epoch_counter', torch.tensor(0))
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=.01, momentum=.9)
        self.scaler = Scaler()
        self.scaler.mean, self.scaler.std = 0., 1.
        self.epoch, self.evaluations = 0, []
        self.trajectory, self.draws, self.scheduled = [], [], []

    def prepare_data(self, train_X, train_y, val_X, val_y, target_name):
        self.test_structures = val_X
        self.validation_data = val_X

    def train_one_epoch(self):
        self.epoch += 1
        if self.epoch == self.fail_epoch:
            raise RuntimeError('simulated interruption')
        self.model.train()
        self.optimizer.zero_grad()
        draw = random.random() + np.random.random() + torch.rand(()).item()
        self.draws.append(draw)
        loss = self.model(torch.tensor([[draw]])).square().mean()
        loss.backward()
        self.optimizer.step()
        self.model.epoch_counter.fill_(self.epoch)
        self.trajectory.append(cli.cpu_state_dict(self.model))
        return float(loss.detach()), float(loss.detach())

    def evaluate_on_test(self):
        # predict_structures replaces this attribute, so restoring only weights is insufficient.
        assert self.test_structures is self.validation_data
        self.model.eval()
        return self.values[self.epoch - 1]

    def step_scheduler(self, value):
        self.scheduled.append(value)

    def predict_structures(self, structures, targets, weights, return_predictions=False):
        self.model.load_state_dict(weights)
        self.model.eval()
        self.test_structures = structures
        # Like a newly constructed DataLoader, evaluation consumes random state.
        random.random()
        np.random.random()
        torch.rand(5)
        best_epoch = self.model.epoch_counter.item()
        mae = .01 * best_epoch + self.test_bias
        self.evaluations.append((self.epoch, best_epoch, mae))
        return (mae, np.full(len(targets), mae)) if return_predictions else mae


class EarlyStoppingMonitorTests(unittest.TestCase):
    def run_training(self, directory, monitor=True, patience=2, values=None,
                     min_delta=0., fail_epoch=None, test_bias=0.):
        values = values or [1., .8, .81, .82, .7, .72, .73]
        config = dict(task='alignn_hetero', model={}, optim=dict(
            early_stopping_patience=patience,
            early_stopping_min_delta_percent=min_delta,
        ))
        if monitor:
            config['optim']['early_stopping_monitor_only'] = True
        structures = [SimpleNamespace(source_id=f's{i}') for i in range(8)]
        split = dict(display='seed=11', logger_id=11, seed=11,
                     train_X=structures[:3], train_y=[0.] * 3,
                     val_X=structures[3:5], val_y=[0.] * 2,
                     test_X=structures[5:], test_y=[0.] * 3)
        instances = []

        def factory(*args, **kwargs):
            trainer = TinyTrainer(*args, **kwargs, values=values,
                                  fail_epoch=fail_epoch, test_bias=test_bias)
            instances.append(trainer)
            return trainer

        with patch.object(cli, 'MEGNetTrainer', side_effect=factory), redirect_stdout(io.StringIO()):
            cli.train_single_mode(
                'hetero', config, [structures] * 4, [0.] * len(structures),
                [11], len(values), 'cpu', 'alignn', 'native', log_dir=directory,
                split_iterator=lambda *args, **kwargs: iter([split]),
            )
        checkpoint = torch.load(Path(directory) / 'seed11_best_checkpoint.pth',
                                map_location='cpu', weights_only=False)
        return instances[0], checkpoint

    def test_first_trigger_and_final_use_validation_best_weights_and_distinct_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer, final = self.run_training(directory)
            self.assertEqual(trainer.epoch, 7)
            self.assertEqual([(e, best) for e, best, _ in trainer.evaluations], [(4, 2), (7, 5)])
            self.assertEqual(final['epochs_completed'], 7)
            self.assertEqual(final['best_epoch'], 5)
            self.assertFalse(final['stopped_early'])
            self.assertEqual(final['evaluation_stage'], 'final')
            early = torch.load(Path(directory) / 'seed11_early_stopping_checkpoint.pth',
                               map_location='cpu', weights_only=False)
            self.assertEqual(early['evaluation_stage'], 'early_stopping')
            self.assertEqual(early['model']['epoch_counter'].item(), 2)
            self.assertEqual(final['model']['epoch_counter'].item(), 5)
            record = json.loads((Path(directory) / 'seed11_evaluation_comparison.json').read_text())
            self.assertEqual(record['early_stopping']['trigger_epoch'], 4)
            self.assertAlmostEqual(record['early_stopping']['test_mae'], .02)
            self.assertAlmostEqual(record['final']['test_mae'], .05)
            for filename, prediction in [('seed11_early_stopping_test_predictions.csv', .02),
                                         ('seed11_test_predictions.csv', .05)]:
                with (Path(directory) / filename).open(newline='') as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(len(rows), 3)
                self.assertTrue(all(float(row['prediction']) == prediction for row in rows))
            with (Path(directory) / 'seed11_history.csv').open(newline='') as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(sum(r['epoch'] == 'EARLY_STOP_TEST' for r in rows), 1)
            self.assertEqual(rows[-1]['epoch'], 'TEST')
            self.assertAlmostEqual(TrainingLogger.completed_test_mae(directory, 11), .05)

    def test_monitoring_preserves_exact_training_trajectory_and_does_not_select_on_test(self):
        with tempfile.TemporaryDirectory() as baseline_dir, tempfile.TemporaryDirectory() as monitored_dir:
            baseline, _ = self.run_training(baseline_dir, monitor=False, patience=0)
            monitored, _ = self.run_training(monitored_dir, test_bias=100.)
            self.assertEqual(monitored.draws, baseline.draws)
            self.assertEqual(monitored.scheduled, baseline.scheduled)
            for actual, expected in zip(monitored.trajectory, baseline.trajectory):
                for key in expected:
                    torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
            for param, state in baseline.optimizer.state_dict()['state'].items():
                torch.testing.assert_close(
                    monitored.optimizer.state_dict()['state'][param]['momentum_buffer'],
                    state['momentum_buffer'], rtol=0, atol=0,
                )

    def test_untriggered_monitor_records_no_early_result(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer, saved = self.run_training(directory, patience=20)
            self.assertEqual(len(trainer.evaluations), 1)
            self.assertIsNone(saved['early_stopping_result'])
            record = json.loads((Path(directory) / 'seed11_evaluation_comparison.json').read_text())
            self.assertIsNone(record['early_stopping'])
            self.assertFalse((Path(directory) / 'seed11_early_stopping_checkpoint.pth').exists())

    def test_original_stopping_behavior_remains_available(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer, saved = self.run_training(directory, monitor=False)
            self.assertEqual(trainer.epoch, 4)
            self.assertTrue(saved['stopped_early'])
            self.assertEqual(trainer.evaluations, [(4, 2, .02)])
            self.assertNotIn('evaluation_stage', saved)

    def test_relative_threshold_and_trigger_on_last_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer, _ = self.run_training(
                directory, values=[1., .995, .991], patience=2, min_delta=1.,
            )
            self.assertEqual([(e, best) for e, best, _ in trainer.evaluations], [(3, 3), (3, 3)])

    def test_interruption_after_early_evaluation_is_not_a_completed_run(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                self.run_training(directory, fail_epoch=5)
            self.assertTrue((Path(directory) / 'seed11_early_stopping_checkpoint.pth').exists())
            self.assertIsNone(TrainingLogger.completed_test_mae(directory, 11))
            self.assertIsNone(completed_split_result(directory, 11))
            # Restarting archives the previous attempt's intermediate artifacts.
            self.run_training(directory, patience=20)
            self.assertFalse((Path(directory) / 'seed11_early_stopping_checkpoint.pth').exists())
            self.assertEqual(len(list(Path(directory).glob('*early_stopping_checkpoint.pth.*.bak'))), 1)

    def test_exception_during_prediction_restores_weights_modes_data_and_random_state(self):
        cli.set_seed(11)
        trainer = TinyTrainer(dict(model={}), 'cpu', 11, [1.])
        trainer.test_structures = ['validation']
        trainer.model.train()
        before = cli.cpu_state_dict(trainer.model)
        python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
        snapshot = {key: torch.zeros_like(value) for key, value in before.items()}

        def fail(*args, **kwargs):
            trainer.model.load_state_dict(snapshot)
            trainer.model.eval()
            trainer.test_structures = ['heldout']
            random.random()
            np.random.random()
            torch.rand(2)
            raise RuntimeError('prediction failed')

        with patch.object(trainer, 'predict_structures', side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, 'prediction failed'):
                cli.predict_preserving_training_state(trainer, [], [], snapshot)
        for key, value in trainer.model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        self.assertTrue(trainer.model.training)
        self.assertEqual(trainer.test_structures, ['validation'])
        self.assertEqual(random.getstate(), python_rng)
        np.testing.assert_array_equal(np.random.get_state()[1], numpy_rng[1])
        torch.testing.assert_close(torch.get_rng_state(), torch_rng, rtol=0, atol=0)

    def test_real_predict_structures_preserves_validation_data_mixed_modes_and_loader_rng(self):
        from torch_geometric.data import Data

        trainer = MEGNetTrainer.__new__(MEGNetTrainer)
        trainer.config = dict(model=dict(test_batch_size=1))
        trainer.device = 'cpu'
        trainer.model = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.Dropout(.2))
        trainer.model.train()
        trainer.model[1].eval()
        trainer.converter = SimpleNamespace(convert=lambda s: Data(
            x=torch.tensor([[float(s.x)]]), y=torch.tensor([float(s.y)]),
            weight=torch.ones(1),
        ))
        trainer.scaler = Scaler()
        trainer._autocast = nullcontext
        trainer._forward = lambda batch: trainer.model(batch.x).reshape(-1)
        validation_data = [object(), object()]
        trainer.test_structures = validation_data
        before = cli.cpu_state_dict(trainer.model)
        rng_before = torch.get_rng_state()
        snapshot = {key: torch.zeros_like(value) for key, value in before.items()}
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            mae, predictions = cli.predict_preserving_training_state(
                trainer, [SimpleNamespace(x=i) for i in range(3)], [1.] * 3, snapshot,
            )
        self.assertAlmostEqual(float(mae), 1.)
        np.testing.assert_array_equal(predictions, np.zeros(3))
        self.assertIs(trainer.test_structures, validation_data)
        self.assertTrue(trainer.model.training)
        self.assertTrue(trainer.model[0].training)
        self.assertFalse(trainer.model[1].training)
        for key, value in trainer.model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        torch.testing.assert_close(torch.get_rng_state(), rng_before, rtol=0, atol=0)

    def test_cli_records_monitoring_and_rejects_disabled_patience_or_reusing_other_protocol(self):
        harness = compact_tests.CompactLogsTests()
        with tempfile.TemporaryDirectory() as directory:
            _, train = harness.run_cli(directory, ['--early-stopping-monitor-only',
                                                  '--early-stopping-patience', '2'])
            self.assertTrue(train.call_args.args[1]['optim']['early_stopping_monitor_only'])
            harness.completed_histories(train)
            with self.assertRaisesRegex(RuntimeError, 'configuration differs'):
                harness.run_cli(directory, ['--early-stopping-patience', '2'])
        with tempfile.TemporaryDirectory() as directory, redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                harness.run_cli(directory, ['--early-stopping-monitor-only', '--early-stopping-patience', '0'])


if __name__ == '__main__':
    unittest.main()
