"""Experiment pairing and validation-only selection without training or data."""
import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from HERA.scripts.run_native_readout_benchmark import (
    POOLINGS, build_command, build_variants, compare_checkpoints, main, parse_args, summarize,
)


class NativeReadoutBenchmarkTests(unittest.TestCase):
    def checkpoints(self):
        checkpoints = {}
        for pooling, val_mae, test_mae in zip(POOLINGS, (.3, .1, .2), (.01, .5, .2)):
            checkpoints['hetero', 123, pooling] = dict(
                config=dict(task='alignn_hetero', model=dict(hetero_pooling=pooling, local_radius=0)),
                mode='hetero', dataset_name='native', epochs=500, best_epoch=10, epochs_completed=500,
                best_val_mae=val_mae, test_mae=test_mae,
                split=dict(seed=123, **{f'{part}_sources': [dict(source_id=part)]
                                       for part in ('train', 'val', 'test')}))
        return checkpoints

    def test_selection_uses_validation_not_test(self):
        rows, selections = compare_checkpoints(self.checkpoints(), ['hetero'], [123])
        self.assertEqual(len(rows), 3)
        self.assertEqual(selections[0]['recommended_pooling'], 'global_mean')
        self.assertIsNone(selections[0]['results'][0]['std_test_mae_ev'])

    def test_mismatched_splits_backbone_or_budget_rejected(self):
        for kind in ('order', 'backbone', 'budget', 'overlap', 'missing'):
            checkpoints = self.checkpoints()
            candidate = checkpoints['hetero', 123, 'global_mean']
            if kind == 'order':
                candidate['split']['train_sources'] = [dict(source_id='different_train')]
            elif kind == 'backbone':
                candidate['config']['model']['local_radius'] = 3
            elif kind == 'budget':
                candidate['epochs'] = 100
            elif kind == 'overlap':
                candidate['split']['test_sources'] = copy.deepcopy(candidate['split']['train_sources'])
            else:
                del checkpoints['hetero', 123, 'defect_global_mean']
            with self.assertRaises(ValueError, msg=kind):
                compare_checkpoints(checkpoints, ['hetero'], [123])

    def test_commands_share_manifest_and_differ_only_in_readout_and_output(self):
        args = parse_args(['--seed', '123', '11', '--mode', 'hetero', 'hetero_was'])
        normalized = []
        for pooling in POOLINGS:
            command = build_command(args, pooling)
            self.assertEqual(command[command.index('--alignn-hetero-pooling') + 1], pooling)
            self.assertEqual(command[command.index('--native-filter-manifest') + 1],
                             str(args.run_dir / 'native_filter_manifest.json'))
            self.assertIn('--resume', command)
            command[command.index('--alignn-hetero-pooling') + 1] = '<pool>'
            command[command.index('--run-dir') + 1] = '<output>'
            normalized.append(command)
        self.assertTrue(all(command == normalized[0] for command in normalized))

    def test_dry_run_has_no_execution_or_output_side_effects(self):
        with patch('HERA.scripts.run_native_readout_benchmark.subprocess.run') as run, \
                patch.object(Path, 'mkdir') as mkdir, redirect_stdout(io.StringIO()) as output:
            main(['--dry-run'])
        run.assert_not_called()
        mkdir.assert_not_called()
        self.assertIn('--alignn-hetero-pooling defect_global_mean', output.getvalue())

    def test_report_reads_completed_checkpoints_and_records_validation_winner(self):
        import torch

        with tempfile.TemporaryDirectory() as temporary:
            args = parse_args(['--run-dir', temporary, '--summary-only'])
            for (_, seed, pooling), checkpoint in self.checkpoints().items():
                directory = args.run_dir / pooling / 'alignn' / 'native' / 'hetero_no_dd'
                directory.mkdir(parents=True)
                torch.save(checkpoint, directory / f'seed{seed}_best_checkpoint.pth')
            with redirect_stdout(io.StringIO()):
                summarize(args)
            result = json.loads((args.run_dir / 'readout_selection.json').read_text(encoding='utf-8'))
            self.assertTrue(result['same_ordered_splits'])
            self.assertEqual(result['recommendations'][0]['recommended_pooling'], 'global_mean')
            self.assertIn('0.500000', (args.run_dir / 'readout_comparison.md').read_text(encoding='utf-8'))

    def test_new_poolings_reach_training_cli_without_loading_real_data(self):
        from HERA.tests.test_compact_logs import CompactLogsTests

        cli_test = CompactLogsTests()
        for pooling in ('global_mean', 'defect_global_mean'):
            with tempfile.TemporaryDirectory() as directory:
                _, train = cli_test.run_cli(directory, ['--alignn-hetero-pooling', pooling])
                self.assertTrue(train.called)
                for call in train.call_args_list:
                    self.assertEqual(call.args[1]['model']['hetero_pooling'], pooling)
                    self.assertIn(f'pool_{pooling}', call.kwargs['run_label'])

    def capacity_checkpoints(self):
        variants = build_variants(('defect_energy_mean', 'defect_global_mean'), (64, 32))
        template = self.checkpoints()['hetero', 123, 'defect_energy_mean']
        checkpoints = {}
        for (name, variant), val, test in zip(variants.items(), (.3,.2,.1,.15), (.01,.3,.5,.2)):
            checkpoint = copy.deepcopy(template)
            checkpoint['config']['model'].update(hetero_pooling=variant['pooling'],
                                                  embedding_size=variant['embedding_size'])
            checkpoint.update(best_val_mae=val, test_mae=test, scaler={'mean':1., 'std':2.})
            checkpoints['hetero', 123, name] = checkpoint
        return variants, checkpoints

    def test_capacity_matrix_selects_validation_winner_and_reports_all_candidates(self):
        variants, checkpoints = self.capacity_checkpoints()
        rows, selections = compare_checkpoints(checkpoints, ['hetero'], [123], variants)
        self.assertEqual(len(rows), 4)
        self.assertEqual(selections[0]['recommended_variant'], 'defect_energy_mean_h32')
        self.assertEqual(selections[0]['recommended_embedding_size'], 32)
        # The best-test candidate is deliberately different from the validation winner.
        self.assertEqual(min(rows, key=lambda r:r['test_mae_ev'])['variant'], 'defect_energy_mean_h64')
        with tempfile.TemporaryDirectory() as temporary:
            args = parse_args(['--run-dir', temporary, '--summary-only', '--embedding-sizes', '64', '32',
                               '--pooling', 'defect_energy_mean', 'defect_global_mean'])
            for (_, seed, name), checkpoint in checkpoints.items():
                import torch
                directory = args.run_dir/name/'alignn/native/hetero_no_dd'
                directory.mkdir(parents=True)
                torch.save(checkpoint, directory/f'seed{seed}_best_checkpoint.pth')
            with redirect_stdout(io.StringIO()):
                summarize(args)
            selection = json.loads((args.run_dir/'readout_selection.json').read_text())
            self.assertEqual(selection['recommendations'][0]['recommended_embedding_size'], 32)
            report = (args.run_dir/'readout_comparison.md').read_text()
            self.assertIn('| defect_global_mean | 32 |', report)
            self.assertIn('0.500000', report)

    def test_capacity_matrix_rejects_unplanned_width_scaler_and_config_changes(self):
        for kind in ('width', 'scaler', 'norm', 'missing'):
            variants, checkpoints = self.capacity_checkpoints()
            key = ('hetero', 123, 'defect_global_mean_h32')
            if kind == 'width':
                checkpoints[key]['config']['model']['embedding_size'] = 48
            elif kind == 'scaler':
                checkpoints[key]['scaler']['mean'] = 3.
            elif kind == 'norm':
                checkpoints[key]['config']['model']['hetero_feature_norm'] = 'batchnorm'
            else:
                del checkpoints[key]
            with self.assertRaises(ValueError, msg=kind):
                compare_checkpoints(checkpoints, ['hetero'], [123], variants)

    def test_capacity_commands_share_data_and_isolate_each_architecture(self):
        args = parse_args(['--pooling', 'defect_energy_mean', 'defect_global_mean',
                           '--embedding-sizes', '64', '32'])
        outputs, normalized = set(), []
        for name, variant in build_variants(args.pooling, args.embedding_sizes).items():
            command = build_command(args, **variant)
            self.assertEqual(command[command.index('--alignn-embedding-size')+1], str(variant['embedding_size']))
            outputs.add(command[command.index('--run-dir')+1])
            self.assertTrue(command[command.index('--run-dir')+1].endswith(name))
            for option in ('--alignn-hetero-pooling', '--alignn-embedding-size', '--run-dir'):
                command[command.index(option)+1] = '<variant>'
            normalized.append(command)
        self.assertEqual(len(outputs), 4)
        self.assertTrue(all(command == normalized[0] for command in normalized))


if __name__ == '__main__':
    unittest.main()
