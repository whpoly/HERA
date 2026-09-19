"""Missing TEST CSVs and mislabeled folders must not silently retrain models."""
from contextlib import redirect_stdout, redirect_stderr
import copy
import io
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from HERA import main as cli
from HERA.config.defaults import get_config, apply_alignn_hetero_options
from HERA.scripts import merge_hetero_relation_results as merger
from HERA.scripts import run_hetero_relation_benchmark as runner
from HERA.training.history import TrainingLogger
from HERA.training.results import completed_split_result


ROOT = Path(__file__).resolve().parents[1]


class ResumeSavedCheckpointTests(unittest.TestCase):
    @staticmethod
    def config(dd='drop'):
        config = get_config('alignn', 'native', 'hetero')
        apply_alignn_hetero_options(config, dd_mode=dd)
        return config

    @classmethod
    def checkpoint(cls, directory, config=None, seed=123, loss=.123):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f'seed{seed}_best_checkpoint.pth'
        payload = dict(config=config or cls.config(), model={'weight': torch.zeros(1)},
                       scaler={'mean': np.float64(1.), 'std': np.float64(2.)},
                       model_name='alignn', dataset_name='native', mode='hetero',
                       split={'logger_id': seed, 'seed': seed}, test_mae=loss)
        torch.save(payload, path)
        return path

    def expected(self, config=None):
        return dict(config=config or self.config(), model_name='alignn', dataset_name='native', mode='hetero')

    def test_checkpoint_test_result_is_reused_without_csv(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            path = self.checkpoint(directory)
            before = path.read_bytes()
            self.assertEqual(completed_split_result(directory, 123, self.expected(), True), .123)
            self.assertEqual(path.read_bytes(), before)
            self.assertFalse((Path(directory) / 'seed123_history.csv').exists())

    def test_completed_checkpoint_survives_an_overwritten_incomplete_history(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            self.checkpoint(directory)
            logger = TrainingLogger(directory, 'alignn', 'native', 'hetero', 123)
            logger.log(1, 1., 1., 1., 1., .001)
            before = Path(logger.filepath).read_bytes()
            self.assertEqual(completed_split_result(directory, 123, self.expected(), True), .123)
            self.assertEqual(Path(logger.filepath).read_bytes(), before)

    def test_baseline_checkpoint_cannot_be_reused_as_no_dd(self):
        with tempfile.TemporaryDirectory() as directory:
            self.checkpoint(directory, config=self.config('keep'))
            with self.assertRaisesRegex(RuntimeError, 'saved config does not match'):
                completed_split_result(directory, 123, self.expected(), True)

    def test_invalid_checkpoint_stops_instead_of_retraining(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'seed123_best_checkpoint.pth'
            path.write_bytes(b'invalid checkpoint')
            with self.assertRaisesRegex(RuntimeError, 'will not be retrained'):
                completed_split_result(directory, 123, self.expected(), True)
            self.assertEqual(path.read_bytes(), b'invalid checkpoint')

    def test_protect_existing_stops_partial_history_but_allows_new_split(self):
        with tempfile.TemporaryDirectory() as directory:
            logger = TrainingLogger(directory, 'alignn', 'native', 'hetero', 123)
            logger.log(1, 1., 1., 1., 1., .001)
            with self.assertRaisesRegex(RuntimeError, 'No restart was launched'):
                completed_split_result(directory, 123, self.expected(), True)
            self.assertIsNone(completed_split_result(directory, 11, self.expected(), True))

    def test_training_cli_skips_checkpoint_only_results_before_dataset_load(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(dataset=['native'], mode=['hetero'], seed=['123'], epochs=500,
                                   device='cpu', batch_size=8, run_dir=Path(directory), cv5=False)
            command = runner.training_command(args, 'no_dd')
            with patch('sys.argv', ['HERA.main', *command[3:]]), \
                    patch.object(cli, 'load_dataset', return_value=(None, [], None, None, [])), \
                    patch.object(cli, 'train_single_mode', return_value=[.123]) as train, \
                    redirect_stdout(io.StringIO()):
                cli.main()
            call = train.call_args
            self.checkpoint(call.kwargs['log_dir'], config=copy.deepcopy(call.args[1]))
            with patch('sys.argv', ['HERA.main', *command[3:]]), \
                    patch.object(cli, 'load_dataset', side_effect=AssertionError('must not load data')), \
                    patch.object(cli, 'train_single_mode', side_effect=AssertionError('must not train')), \
                    redirect_stdout(io.StringIO()):
                cli.main()

    def test_standalone_merger_reads_actual_dd_and_does_not_rename_or_train(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.checkpoint(root / 'baseline/alignn/native/hetero/r0', config=self.config('drop'))
            before = path.read_bytes()
            output = io.StringIO()
            with patch.object(cli, 'train_single_mode', side_effect=AssertionError('no training')), \
                    patch.object(runner.subprocess, 'run', side_effect=AssertionError('no subprocess')), \
                    redirect_stdout(output):
                merger.main(['--run-dir', str(root)])
            self.assertIn('saved AA=keep, DD=drop => no_dd', output.getvalue())
            self.assertIn('MERGE ONLY:', output.getvalue())
            self.assertIn('Mean=0.1230', (root / 'summary.txt').read_text())
            self.assertIn('Saved AA=keep, DD=drop', (root / 'summary.txt').read_text())
            self.assertFalse((root / 'no_dd').exists())
            self.assertEqual(path.read_bytes(), before)

    def test_direct_script_command_merges_both_directories_in_fresh_process(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for variant, dd in (('baseline', 'keep'), ('no_dd', 'drop')):
                self.checkpoint(root / variant / 'alignn/native/hetero/r0', config=self.config(dd))
            result = subprocess.run([sys.executable, str(ROOT / 'scripts/merge_hetero_relation_results.py'),
                                     '--run-dir', str(root)], capture_output=True, text=True, check=True)
            self.assertIn('DD=keep => baseline', result.stdout)
            self.assertIn('DD=drop => no_dd', result.stdout)
            self.assertNotIn('Epoch 1/', result.stdout)
            text = (root / 'summary.txt').read_text()
            self.assertIn('baseline', text)
            self.assertIn('no_dd', text)

    def test_conflicting_history_and_checkpoint_results_are_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            leaf = Path(directory) / 'baseline/alignn/native/hetero/r0'
            self.checkpoint(leaf)
            TrainingLogger(leaf, 'alignn', 'native', 'hetero', 123).log_test_result(.9)
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                merger.main(['--run-dir', directory])

    def test_different_dd_configs_in_one_leaf_are_not_averaged(self):
        with tempfile.TemporaryDirectory() as directory:
            leaf = Path(directory) / 'baseline/alignn/native/hetero/r0'
            self.checkpoint(leaf, config=self.config('keep'), seed=123)
            self.checkpoint(leaf, config=self.config('drop'), seed=11)
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                merger.main(['--run-dir', directory])


if __name__ == '__main__':
    unittest.main()
