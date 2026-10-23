"""Deferring scalar transfers must preserve weighting, update order and predictions."""
from contextlib import nullcontext, redirect_stdout
from copy import deepcopy
import io
from types import SimpleNamespace
import unittest

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from HERA.training.losses import MAELoss
from HERA.training.trainer import MEGNetTrainer
from HERA.utils.scaler import Scaler


class DeferredMetricTests(unittest.TestCase):
    def trainer(self):
        torch.manual_seed(5)
        trainer = MEGNetTrainer.__new__(MEGNetTrainer)
        trainer.device, trainer.use_amp, trainer.grad_accum_steps = 'cpu', False, 2
        trainer.config = dict(optim=dict(point_loss='mse'), model=dict(test_batch_size=2))
        trainer.model = torch.nn.Linear(1, 1)
        trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=.02)
        trainer.scaler = Scaler()
        trainer.scaler.mean, trainer.scaler.std = .7, 2.
        trainer.trainloader = [
            Data(x=torch.tensor([[float(i)], [float(i + 1)]]), y=torch.tensor([i / 2., -float(i)]),
                 weight=torch.tensor([.5, 2.], dtype=torch.float64)) for i in range(3)
        ]
        trainer.train_structures = [None] * 6
        trainer.testloader = trainer.trainloader
        trainer.test_structures = trainer.train_structures
        trainer._forward = lambda batch: trainer.model(batch.x).reshape(-1)
        trainer._autocast = nullcontext
        return trainer

    def test_weighted_metrics_and_incomplete_accumulation_match_legacy_loop(self):
        trainer = self.trainer()
        model = deepcopy(trainer.model)
        optimizer = torch.optim.SGD(model.parameters(), lr=.02)
        optimizer.zero_grad(set_to_none=True)
        errors, mses = [], []
        for index, batch in enumerate(trainer.trainloader):
            prediction = model(batch.x).reshape(-1)
            loss = F.mse_loss(trainer.scaler.transform(batch.y), prediction)
            (loss / 2).backward()
            if (index + 1) % 2 == 0 or index + 1 == len(trainer.trainloader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            mses.append(loss.detach().cpu().numpy())
            errors.append(MAELoss(trainer.scaler.inverse_transform(prediction), batch.y,
                                  weights=batch.weight, reduction='sum').cpu().numpy())
        mae, mse = trainer.train_one_epoch()
        self.assertEqual(mae, sum(errors) / 6)
        self.assertEqual(mse, np.mean(mses))
        self.assertEqual(trainer.last_train_rank_loss, 0.)
        for actual, expected in zip(trainer.model.parameters(), model.parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_validation_metrics_and_prediction_order(self):
        trainer = self.trainer()
        expected, errors = [], []
        for batch in trainer.testloader:
            values = trainer.scaler.inverse_transform(trainer._forward(batch))
            expected.append(values)
            errors.append(MAELoss(values, batch.y, weights=batch.weight, reduction='sum').numpy())
        expected_mae = sum(errors) / 6
        self.assertEqual(trainer.evaluate_on_test(), expected_mae)
        mae, predictions = trainer.evaluate_on_test(return_predictions=True)
        self.assertEqual(mae, expected_mae)
        torch.testing.assert_close(predictions, torch.cat(expected).reshape(-1, 1), rtol=0, atol=0)

    def test_test_predictions_keep_source_order_and_weights(self):
        trainer = self.trainer()
        structures = [Data(x=torch.tensor([[float(i)]])) for i in range(5)]
        targets, weights = [0., .1, .2, .3, .4], [.5, 1., 2., 1., .5]
        trainer.converter = SimpleNamespace(convert=lambda structure: structure)
        with redirect_stdout(io.StringIO()):
            mae, values = trainer.predict_structures(
                structures, targets, trainer.model.state_dict(), test_weights=weights,
                return_predictions=True,
            )
        expected = trainer.scaler.inverse_transform(trainer.model(torch.arange(5.).reshape(-1, 1))).numpy().reshape(-1)
        np.testing.assert_array_equal(values, expected)
        self.assertAlmostEqual(float(mae), float(np.sum(np.abs(expected - targets) * weights) / 5), places=6)


if __name__ == '__main__':
    unittest.main()
