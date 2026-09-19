"""Completed experiments survive adding modes/seeds and summary-only repair."""
from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from HERA import main as cli
from HERA.config.defaults import get_config
from HERA.scripts import run_hetero_relation_benchmark as runner
from HERA.training.history import TrainingLogger
from HERA.training.results import (
    merge_aggregate_summary, read_saved_mode_summary, rebuild_run_summaries,
    saved_dataset_rows,
)


class BenchmarkResultMergeTests(unittest.TestCase):
    @staticmethod
    def complete(directory, seed, loss):
        logger = TrainingLogger(directory, 'alignn', 'native', 'hetero', seed)
        logger.log(1, .2, .04, .1, .1, .001)
        logger.log_test_result(loss)
        checkpoint = Path(directory) / f'seed{seed}_best_checkpoint.pth'
        checkpoint.write_bytes(b'checkpoint preserved verbatim')

    @classmethod
    def fake_train(cls, *args, **kwargs):
        for seed in args[4]:
            cls.complete(kwargs['log_dir'], seed, seed / 1000)
        return [seed / 1000 for seed in args[4]]

    def invoke_main(self, root, modes, seeds, forbid_data=False):
        args = SimpleNamespace(dataset=['native', 'semi', 'imp2d'], mode=modes,
                               seed=[str(seed) for seed in seeds], epochs=500,
                               device='cpu', batch_size=8, run_dir=root, cv5=False)
        command = runner.training_command(args, 'no_dd')
        with patch('sys.argv', ['HERA.main', *command[3:]]), \
                patch.object(cli, 'load_dataset', return_value=(None, [], None, None, []),
                             side_effect=AssertionError('must not load data') if forbid_data else None), \
                patch.object(cli, 'train_single_mode', side_effect=self.fake_train) as train, \
                redirect_stdout(io.StringIO()):
            cli.main()
        return train.call_args_list

    def test_adding_was_and_missing_seeds_never_retrains_completed_hetero(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calls = self.invoke_main(root, ['hetero'], [123, 11])
            self.assertEqual(len(calls), 6)
            old_files = {path: path.read_bytes() for path in root.rglob('seed*') if path.is_file()}
            calls = self.invoke_main(root, ['hetero_was'], [123])
            self.assertEqual(len(calls), 3)
            self.assertTrue(all(call.args[0] == 'hetero_was' for call in calls))
            # Original files are untouched even when the second command selects only WAS.
            self.assertEqual(old_files, {path: path.read_bytes() for path in old_files})
            for dataset in ('native', 'semi', 'imp2d'):
                text = (root / 'no_dd/alignn' / dataset / 'summary.txt').read_text()
                self.assertIn('HETERO_R0_', text)
                self.assertIn('HETERO_WAS_R0_', text)
            text = (root / 'no_dd/alignn/summary.txt').read_text()
            self.assertIn('HETERO_R0_', text)
            self.assertIn('HETERO_WAS_R0_', text)
            calls = self.invoke_main(root, ['hetero', 'hetero_was'], [123, 11])
            self.assertEqual(len(calls), 3)
            self.assertTrue(all(call.args[0] == 'hetero_was' and call.args[4] == [11] for call in calls))
            self.assertEqual(self.invoke_main(root, ['hetero', 'hetero_was'], [123, 11], forbid_data=True), [])

    def test_summary_only_rebuilds_old_and_new_without_training_or_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for variant in ('baseline', 'no_dd'):
                for mode in ('hetero', 'hetero_was'):
                    leaf = root / variant / 'alignn/native' / mode / 'r0'
                    self.complete(leaf, 123, .123)
                    self.complete(leaf, 11, .011)
            # A half-finished task is preserved, but contributes no test MAE.
            leaf = root / 'no_dd/alignn/semi/hetero/r0'
            TrainingLogger(leaf, 'alignn', 'semi', 'hetero', 123).log(1, 1., 1., 1., 1., .001)
            old_files = {p: p.read_bytes() for p in root.rglob('seed*') if p.is_file()}
            args = ['benchmark', '--summary-only', '--run-dir', str(root)]
            with patch('sys.argv', args), patch.object(runner, 'data_errors', side_effect=AssertionError('no data needed')), \
                    patch.object(runner.subprocess, 'run') as launch, \
                    patch.object(cli, 'train_single_mode') as train, redirect_stdout(io.StringIO()):
                runner.main()
            launch.assert_not_called()
            train.assert_not_called()
            text = (root / 'summary.txt').read_text()
            self.assertIn('baseline', text)
            self.assertIn('no_dd', text)
            self.assertIn('HETERO_R0', text)
            self.assertIn('HETERO_WAS_R0', text)
            self.assertNotIn('semi', text)
            self.assertEqual(old_files, {path: path.read_bytes() for path in old_files})

    def test_old_leaf_summaries_restore_rows_even_without_histories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            leaf = root / 'alignn/native/hetero/r0'
            leaf.mkdir(parents=True)
            config = get_config('alignn', 'native', 'hetero')
            cli.write_mode_summary(leaf / 'summary.txt', 'alignn', 'native', 'hetero_r0',
                                   [.123, .011], 500, [123, 11], config)
            self.assertEqual(rebuild_run_summaries(root), 1)
            self.assertIn('Mean=0.0670', (root / 'alignn/native/summary.txt').read_text())
            cli.write_mode_summary(leaf / 'summary.txt', 'alignn', 'native', 'hetero_r0',
                                   [.42], 500, [42], config)
            saved = read_saved_mode_summary(leaf / 'summary.txt')
            self.assertEqual(saved['splits'], {'123': .123, '11': .011, '42': .42})

    def test_merge_keeps_other_rows_and_backs_up_previous_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'summary.txt'
            original = 'SUMMARY: ALIGNN on native\n  HETERO Mean=0.1 Std=0.0 Seeds=[0.1]\n'
            path.write_text(original)
            lines = ['SUMMARY: ALIGNN on native', '  HETERO_WAS Mean=0.2 Std=0.0 Seeds=[0.2]']
            merge_aggregate_summary(path, lines)
            self.assertIn('HETERO Mean=', path.read_text())
            self.assertIn('HETERO_WAS Mean=', path.read_text())
            backups = list(path.parent.glob('summary.txt.*.bak'))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_text(), original)
            merge_aggregate_summary(path, lines)
            self.assertEqual(len(list(path.parent.glob('summary.txt.*.bak'))), 1)

    def test_cv_and_random_splits_are_not_averaged_together(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            leaf = root / 'hetero/r0'
            self.complete(leaf, 123, .1)
            self.complete(leaf, '123_fold1', .3)
            self.complete(leaf, '42_fold1', .5)
            self.complete(leaf, 11, float('nan'))
            rows = saved_dataset_rows(root)
            self.assertEqual(len(rows), 3)
            self.assertTrue(any('CV5_SEED123' in row and 'Mean=0.3000' in row for row in rows))
            self.assertTrue(any('CV5_SEED42' in row and 'Mean=0.5000' in row for row in rows))
            self.assertTrue(any('R0_SEEDS' in row and 'Mean=0.1000' in row for row in rows))

    def test_restart_preserves_history_and_failed_atomic_write_keeps_original(self):
        with tempfile.TemporaryDirectory() as directory:
            self.complete(directory, 123, .1)
            path = Path(directory) / 'seed123_history.csv'
            original = path.read_bytes()
            logger = TrainingLogger(directory, 'alignn', 'native', 'hetero', 123)
            self.assertEqual(next(path.parent.glob('seed123_history.csv.*.bak')).read_bytes(), original)
            with patch('HERA.training.history.os.replace', side_effect=OSError('simulated write failure')), \
                    self.assertRaises(OSError):
                logger.log(1, 1., 1., 1., 1., .001)
            self.assertEqual(path.read_bytes(), original)

    def test_summary_only_with_wrong_parent_fails_without_training(self):
        with tempfile.TemporaryDirectory() as directory:
            args = ['benchmark', '--summary-only', '--variant', 'no_dd', '--run-dir', directory]
            with patch('sys.argv', args), patch.object(runner.subprocess, 'run') as launch, \
                    redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                runner.main()
            launch.assert_not_called()


if __name__ == '__main__':
    unittest.main()
