"""Compact result paths keep experiment identity, resume and saved metrics safe."""

from contextlib import redirect_stdout, redirect_stderr
import copy
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from HERA import main as cli
from HERA.scripts import run_hetero_relation_benchmark as runner
from HERA.training.history import TrainingLogger
from HERA.training.results import (completed_split_result, read_compact_run_config,
                                   saved_dataset_rows)


ROOT = Path(__file__).resolve().parents[1]
NON_HYPERGRAPH_MODES = ['full', 'full_x', 'attention', 'attention_was', 'hetero',
                       'hetero_was', 'was_x', 'definet', 'definet_was']


class CompactLogsTests(unittest.TestCase):
    def command(self, directory, extra=()):
        return ['HERA.main', '--model', 'alignn', '--dataset', 'native', '--mode', 'hetero',
                '--r', '0', '--native-preprocessing', 'reference_v1',
                '--device', 'cpu', '--epochs', '1', '--seed', '123',
                '--atom-init', str(ROOT / 'atom_init.json'), '--run-dir', str(directory),
                '--compact-logs', '--resume', '--protect-existing', *extra]

    def run_cli(self, directory, extra=()):
        with patch('sys.argv', self.command(directory, extra)), redirect_stdout(io.StringIO()), \
                patch.object(cli, 'load_dataset', return_value=([], [], [], None, [])) as loader, \
                patch.object(cli, 'train_single_mode', return_value=[.123]) as train:
            cli.main()
        return loader, train

    @staticmethod
    def completed_histories(train):
        for call in train.call_args_list:
            for seed in call.args[4]:
                TrainingLogger(call.kwargs['log_dir'], 'alignn', call.kwargs['dataset_name'],
                               call.kwargs['run_label'], seed).log_test_result(.123)

    def test_all_requested_modes_have_only_model_dataset_mode_directories(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, train = self.run_cli(temporary, [
                '--dataset', 'native', 'semi', 'imp2d', '--mode', *NON_HYPERGRAPH_MODES,
                '--semi-preprocessing', 'reference_v1', '--imp2d-preprocessing', 'reference_v1',
                '--alignn-hetero-node-norm', 'layernorm', '--alignn-hetero-relations', 'shared_residual',
                '--alignn-hetero-pooling', 'defect_energy_mean', '--seed', '123', '11', '1245',
            ])
            self.assertEqual(train.call_count, 75)
            directories = set()
            for call in train.call_args_list:
                directory = Path(call.kwargs['log_dir'])
                dataset, mode = call.kwargs['dataset_name'], call.args[0]
                self.assertEqual(directory.relative_to(temporary).parts, ('alignn', dataset, mode))
                record = read_compact_run_config(directory)
                self.assertEqual(record['config'], call.args[1])
                self.assertEqual(record['config'][f'{dataset}_preprocessing'], 'reference_v1')
                self.assertEqual(record['run_label'], call.kwargs['run_label'])
                self.assertFalse(any(path.is_dir() for path in directory.iterdir()))
                directories.add(directory)
            self.assertEqual(len(directories), 25)

    def test_csv_resume_skips_completed_jobs_and_allows_adding_seeds(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, train = self.run_cli(temporary)
            self.completed_histories(train)
            directory = Path(train.call_args.kwargs['log_dir'])
            config_bytes = (directory / 'config.json').read_bytes()
            history_bytes = (directory / 'seed123_history.csv').read_bytes()
            loader, repeated = self.run_cli(temporary)
            loader.assert_not_called()
            repeated.assert_not_called()
            _, additional = self.run_cli(temporary, ['--seed', '123', '11', '1245'])
            self.assertEqual([call.args[4] for call in additional.call_args_list], [[11], [1245]])
            self.assertEqual((directory / 'config.json').read_bytes(), config_bytes)
            self.assertEqual((directory / 'seed123_history.csv').read_bytes(), history_bytes)

    def test_changed_relation_or_protocol_is_rejected_before_csv_resume_or_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, train = self.run_cli(temporary)
            self.completed_histories(train)
            directory = Path(train.call_args.kwargs['log_dir'])
            before = {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}
            for changed in (['--alignn-hetero-dd', 'drop'], ['--epochs', '500']):
                with self.subTest(changed=changed), patch('sys.argv', self.command(temporary, changed)), \
                        redirect_stdout(io.StringIO()), patch.object(cli, 'load_dataset') as loader, \
                        patch.object(cli, 'train_single_mode') as repeat:
                    with self.assertRaisesRegex(RuntimeError, 'configuration differs'):
                        cli.main()
                    loader.assert_not_called()
                    repeat.assert_not_called()
            self.assertEqual({p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}, before)

    def test_expected_config_is_checked_even_outside_compact_cli_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, train = self.run_cli(temporary)
            self.completed_histories(train)
            directory = Path(train.call_args.kwargs['log_dir'])
            record = read_compact_run_config(directory)
            expected = {k: copy.deepcopy(record[k]) for k in ('config', 'model_name', 'dataset_name', 'mode')}
            expected['config']['native_preprocessing'] = 'legacy'
            with self.assertRaisesRegex(RuntimeError, 'configuration differs'):
                completed_split_result(directory, 123, expected)

    def test_radius_sweeps_stop_before_creating_configs_or_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch('sys.argv', self.command(temporary, ['--r', '0', '3'])), \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), \
                    patch.object(cli, 'load_dataset') as loader, patch.object(cli, 'train_single_mode') as train:
                with self.assertRaises(SystemExit):
                    cli.main()
                loader.assert_not_called()
                train.assert_not_called()
            self.assertFalse(list(Path(temporary).rglob('config.json')))

    def test_existing_nested_results_are_preserved_and_not_adopted(self):
        with tempfile.TemporaryDirectory() as temporary:
            old = Path(temporary) / 'alignn/native/hetero/r0'
            old.mkdir(parents=True)
            history = old / 'seed123_history.csv'
            history.write_text('historical data', encoding='utf-8')
            with self.assertRaisesRegex(RuntimeError, 'Existing outputs without config.json'):
                self.run_cli(temporary)
            self.assertEqual(history.read_text(encoding='utf-8'), 'historical data')
            self.assertFalse((old.parent / 'config.json').exists())

    def test_history_only_aggregate_uses_saved_label_not_an_ambiguous_mode_name(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, train = self.run_cli(temporary)
            self.completed_histories(train)
            directory = Path(train.call_args.kwargs['log_dir'])
            (directory / 'summary.txt').unlink()
            rows = saved_dataset_rows(directory.parent)
            self.assertEqual(len(rows), 1)
            self.assertIn(train.call_args.kwargs['run_label'].upper(), rows[0])
            self.assertIn('Mean=0.1230', rows[0])

    def test_corrupt_identity_is_not_replaced(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, train = self.run_cli(temporary)
            config_file = Path(train.call_args.kwargs['log_dir']) / 'config.json'
            config_file.write_text('{broken', encoding='utf-8')
            with self.assertRaisesRegex(RuntimeError, 'Cannot verify'):
                self.run_cli(temporary)
            self.assertEqual(config_file.read_text(encoding='utf-8'), '{broken')

    def test_explanation_paths_and_relation_runner_support_compact_layout(self):
        with tempfile.TemporaryDirectory() as temporary:
            explanations = str(Path(temporary) / 'plots')
            _, train = self.run_cli(temporary, ['--explain', '--explain-dir', explanations])
            self.assertEqual(Path(train.call_args.kwargs['explain_options']['root_dir']),
                             Path(explanations) / 'alignn/native/hetero')
            args = SimpleNamespace(dataset=['native'], mode=['hetero', 'hetero_was'], reference=['full'],
                                   seed=['123'], epochs=1, device='cpu', batch_size=8,
                                   run_dir=Path(temporary), cv5=False, compact_logs=True)
            self.assertIn('--compact-logs', runner.training_command(args, 'baseline'))
            self.assertIn('--compact-logs', runner.reference_command(args))


if __name__ == '__main__':
    unittest.main()
