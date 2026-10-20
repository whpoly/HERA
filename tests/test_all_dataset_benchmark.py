"""The single-command suite must retain dataset policies and report paired stages."""
from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from HERA.scripts import run_all_dataset_benchmark as runner


class AllDatasetBenchmarkTests(unittest.TestCase):
    def test_commands_preserve_data_policies_and_architectures_without_hypergraph(self):
        args = runner.parse_args([])
        commands = runner.build_commands(args)
        self.assertEqual(len(commands), 6)
        self.assertIn('--prepare-only', commands[0])
        self.assertIn('--exclude-poscar0', commands[0])
        for command, dataset in zip(commands[1:], runner.DATASETS):
            self.assertEqual(command[command.index('--dataset') + 1], dataset)
            self.assertNotIn('hypergraph', command)
            self.assertNotIn('hypergraph_was', command)
            self.assertIn('--early-stopping-monitor-only', command)
            self.assertIn('--resume', command)
            self.assertEqual(command[command.index('--early-stopping-patience') + 1], '50')
            self.assertEqual(command[command.index('--epochs') + 1], '500')
            batch = dict(native=16, semi=64, imp2d=64, **{'2dmd_low': 8, '2dmd_high': 16})[dataset]
            self.assertEqual(command[command.index('--alignn-train-batch-size') + 1], str(batch))
            if dataset == 'native':
                self.assertEqual(command[command.index('--native-filter-manifest') + 1],
                                 str(args.native_preparation_dir / 'population_mse/native_filter_manifest.json'))
            elif dataset == 'semi':
                self.assertIn('legacy_available', command)
                self.assertIn('--semi-quality-filter', command)
            elif dataset == 'imp2d':
                self.assertIn('interstitial', command)
                self.assertIn('reviewed_v1', command)
                self.assertIn('--imp2d-source', command)
            else:
                self.assertEqual(command[command.index('--alignn-hetero-pooling') + 1], 'type_mean')
                self.assertEqual(command[command.index('--alignn-hetero-dd') + 1], 'keep')

    def test_three_seed_huber_override_reaches_every_training_job(self):
        args = runner.parse_args(['--seed', '7', '8', '9', '--point-loss', 'huber_ev',
                                  '--huber-delta-ev', '.2', '--early-stopping-patience', '25',
                                  '--early-stopping-min-delta-percent', '.1'])
        for command in runner.build_commands(args)[1:]:
            pos = command.index('--seed')
            self.assertEqual(command[pos + 1:pos + 4], ['7', '8', '9'])
            self.assertEqual(command[command.index('--huber-delta-ev') + 1], '0.2')
            self.assertEqual(command[command.index('--early-stopping-patience') + 1], '25')

    def test_dry_run_never_prepares_data_trains_or_writes_files(self):
        with patch.object(runner.subprocess, 'run') as run, patch.object(Path, 'mkdir') as mkdir, \
                patch.object(Path, 'write_text') as write, redirect_stdout(io.StringIO()):
            runner.main(['--dry-run'])
        run.assert_not_called()
        mkdir.assert_not_called()
        write.assert_not_called()

    def test_preparation_failure_stops_before_any_training(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), \
                patch.object(runner.subprocess, 'run',
                             side_effect=subprocess.CalledProcessError(1, 'prepare')) as run:
            with self.assertRaises(subprocess.CalledProcessError):
                runner.main(['--run-dir', directory])
            self.assertEqual(run.call_count, 1)
            self.assertIn('--prepare-only', run.call_args.args[0])

    def test_summary_excludes_missing_and_untriggered_seeds_from_paired_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            args = runner.parse_args(['--run-dir', directory, '--dataset', 'native'])
            path = runner.mode_directory(args, 'native', 'hetero')
            path.mkdir(parents=True)
            for seed, early, final in [(11, .2, .1), (42, None, .3)]:
                record = dict(schema='early_stopping_comparison_v1', model='alignn',
                              dataset='native', mode='hetero', seed=seed,
                              epochs_requested=500, patience=50, min_delta_percent=.5,
                              early_stopping=(dict(trigger_epoch=80, best_epoch=30, test_mae=early,
                                                   best_val_mae=.1) if early is not None else None),
                              final=dict(epochs_completed=500, best_epoch=400,
                                         best_val_mae=.08, test_mae=final))
                (path / f'seed{seed}_evaluation_comparison.json').write_text(json.dumps(record))
            with redirect_stdout(io.StringIO()):
                result = runner.summarize(args)
            row = result['summary'][0]
            self.assertEqual(row['completed'], 2)
            self.assertEqual(row['triggered'], 1)
            self.assertEqual(row['early_test_mae']['mean'], .2)
            self.assertEqual(row['final_test_mae']['mean'], .2)
            self.assertAlmostEqual(row['paired_delta']['mean'], -.1)
            self.assertEqual(len(result['missing']), 28)
            self.assertTrue((args.run_dir / 'early_stopping_comparison.md').exists())

    def test_invalid_patience_rejected(self):
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                runner.parse_args(['--early-stopping-patience', '0'])

    def test_missing_comparison_json_is_recovered_from_completed_checkpoint(self):
        import torch

        with tempfile.TemporaryDirectory() as directory:
            args = runner.parse_args(['--run-dir', directory, '--dataset', 'native'])
            path = runner.mode_directory(args, 'native', 'hetero')
            path.mkdir(parents=True)
            torch.save(dict(
                config=dict(optim=dict(early_stopping_monitor_only=True,
                                       early_stopping_patience=50,
                                       early_stopping_min_delta_percent=.5)),
                evaluation_stage='final', model_name='alignn', dataset_name='native',
                mode='hetero', run_label='hetero_no_dd', epochs=500, epochs_completed=500,
                best_epoch=400, best_val_mae=.1, test_mae=.2,
                split=dict(seed=11, logger_id=11),
                early_stopping_result=dict(trigger_epoch=100, best_epoch=50, test_mae=.3,
                                           best_val_mae=.2),
            ), path / 'seed11_best_checkpoint.pth')
            with redirect_stdout(io.StringIO()):
                result = runner.summarize(args)
            self.assertEqual(len(result['records']), 1)
            self.assertEqual(result['records'][0]['final']['test_mae'], .2)
            self.assertAlmostEqual(result['summary'][0]['paired_delta']['mean'], -.1)


if __name__ == '__main__':
    unittest.main()
