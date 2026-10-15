"""Physical-unit Huber transitions, trainer wiring and experiment identity."""
from contextlib import nullcontext, redirect_stderr
import io
import tempfile
from types import SimpleNamespace
import unittest

import torch
import torch.nn.functional as F

from HERA.training.losses import huber_ev_loss
from HERA.training.trainer import MEGNetTrainer
from HERA.utils.scaler import Scaler
from HERA.tests import test_compact_logs as compact_tests


class RobustPointLossTests(unittest.TestCase):
    def test_matches_mse_value_and_gradient_below_physical_transition(self):
        predictions = torch.tensor([-.2, .05, .2], dtype=torch.float64, requires_grad=True)
        targets = torch.zeros_like(predictions)
        loss = huber_ev_loss(predictions, targets, target_std=4., delta_ev=1.)
        mse = F.mse_loss(predictions, targets)
        self.assertEqual(loss.item(), mse.item())
        actual = torch.autograd.grad(loss, predictions)[0]
        expected = torch.autograd.grad(mse, predictions)[0]
        torch.testing.assert_close(actual, expected)

    def test_caps_large_error_gradients_and_scales_transition_in_ev(self):
        for scale in (1., 4., 10.):
            physical = torch.tensor([-.5, 1., 2., 100.], dtype=torch.float64)
            predictions = (physical/scale).requires_grad_()
            loss = huber_ev_loss(predictions, torch.zeros_like(predictions), scale, 1.)
            loss.backward()
            # Gradient is 2*clamp(residual_eV, -delta, delta)/scale / N.
            expected = 2*physical.clamp(-1., 1.)/scale/len(physical)
            torch.testing.assert_close(predictions.grad, expected)

    def test_constant_target_fallback_and_invalid_values(self):
        predictions, target = torch.tensor([2.]), torch.tensor([0.])
        self.assertEqual(huber_ev_loss(predictions, target, 0.).item(), 3.)
        self.assertEqual(huber_ev_loss(predictions, target, 1e-8).item(), 3.)
        for std, delta in ((float('nan'),1.),(-1.,1.),(1.,0.),(1.,float('inf'))):
            with self.assertRaises(ValueError):
                huber_ev_loss(predictions,target,std,delta)

    def train_step(self, point_loss):
        trainer = MEGNetTrainer.__new__(MEGNetTrainer)
        trainer.device, trainer.use_amp, trainer.grad_accum_steps = 'cpu', False, 1
        trainer.config = dict(optim=dict(point_loss=point_loss, huber_delta_ev=1.))
        trainer.model = torch.nn.Linear(1,1,bias=False)
        with torch.no_grad():
            trainer.model.weight.fill_(7.)
        trainer.optimizer = torch.optim.SGD(trainer.model.parameters(),lr=.1)
        trainer.scaler = Scaler()
        trainer.scaler.std = 2.
        batch = SimpleNamespace(x=torch.ones(1,1), y=torch.zeros(1), weight=torch.ones(1))
        batch.to = lambda device:batch
        trainer.trainloader, trainer.train_structures = [batch], [None]
        trainer._forward = lambda b:trainer.model(b.x).reshape(-1)
        trainer._autocast = nullcontext
        mae, mse = trainer.train_one_epoch()
        self.assertAlmostEqual(float(mae),14.)
        self.assertAlmostEqual(float(mse),49.)
        return trainer.model.weight.item()

    def test_real_trainer_uses_robust_loss_but_keeps_reported_mae_and_mse(self):
        self.assertAlmostEqual(self.train_step('mse'),5.6,places=5)
        self.assertAlmostEqual(self.train_step('huber_ev'),6.9,places=5)
        # Historical smooth_l1 and its legacy huber alias retain beta=1 in scaled units.
        self.assertAlmostEqual(self.train_step('smooth_l1'),6.9,places=5)
        self.assertAlmostEqual(self.train_step('huber'),6.9,places=5)

    def test_cli_records_loss_and_rejects_resume_from_other_loss(self):
        cli = compact_tests.CompactLogsTests()
        with tempfile.TemporaryDirectory() as directory:
            _, train = cli.run_cli(directory, ['--point-loss','huber_ev','--huber-delta-ev','1.'])
            config = train.call_args.args[1]
            self.assertEqual(config['optim']['point_loss'],'huber_ev')
            self.assertEqual(config['optim']['huber_delta_ev'],1.)
            cli.completed_histories(train)
            for changed in (['--point-loss','mse'],['--point-loss','huber_ev','--huber-delta-ev','2.']):
                with self.assertRaisesRegex(RuntimeError,'configuration differs'):
                    cli.run_cli(directory,changed)
        with tempfile.TemporaryDirectory() as directory:
            _, train = cli.run_cli(directory)
            cli.completed_histories(train)
            loader, repeated = cli.run_cli(directory,['--point-loss','mse'])
            loader.assert_not_called()
            repeated.assert_not_called()

    def test_invalid_cli_delta_rejected(self):
        cli = compact_tests.CompactLogsTests()
        with tempfile.TemporaryDirectory() as directory, redirect_stderr(io.StringIO()):
            for flags in (['--huber-delta-ev','1.'],
                          ['--point-loss','huber_ev','--huber-delta-ev','nan'],
                          ['--point-loss','huber_ev','--huber-delta-ev','0']):
                with self.assertRaises(SystemExit):
                    cli.run_cli(directory,flags)


if __name__ == '__main__':
    unittest.main()
