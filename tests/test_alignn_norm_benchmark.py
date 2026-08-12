import csv
import os
import tempfile
import unittest

from HERA.main import completed_resume_losses, expand_alignn_node_norm_runs
from HERA.training.history import TrainingLogger


class AlignnNormBenchmarkTests(unittest.TestCase):
    @staticmethod
    def _runs():
        return [{
            'label': 'hetero_r0',
            'mode': 'hetero',
            'config': {'model': {'hetero_node_norm': 'layernorm'}},
            'local_cutoff': 0,
            'radius_label': 'r0',
        }]

    def test_multiple_norms_create_isolated_run_specs(self):
        runs = expand_alignn_node_norm_runs(
            self._runs(),
            ['layernorm', 'batchnorm', 'none'],
            enabled=True,
        )

        self.assertEqual(
            [run['label'] for run in runs],
            [
                'hetero_r0_norm_layernorm',
                'hetero_r0_norm_batchnorm',
                'hetero_r0_norm_none',
            ],
        )
        self.assertEqual(
            [run['norm_label'] for run in runs],
            ['norm_layernorm', 'norm_batchnorm', 'norm_none'],
        )
        self.assertEqual(
            [run['config']['model']['hetero_node_norm'] for run in runs],
            ['layernorm', 'batchnorm', 'none'],
        )

    def test_single_explicit_norm_uses_an_isolated_resume_path(self):
        runs = expand_alignn_node_norm_runs(
            self._runs(),
            ['batchnorm'],
            enabled=True,
        )

        self.assertEqual(runs[0]['label'], 'hetero_r0_norm_batchnorm')
        self.assertEqual(runs[0]['norm_label'], 'norm_batchnorm')
        self.assertEqual(
            runs[0]['config']['model']['hetero_node_norm'],
            'batchnorm',
        )

    def test_non_hetero_run_is_not_expanded(self):
        runs = expand_alignn_node_norm_runs(
            self._runs(),
            ['layernorm', 'none'],
            enabled=False,
        )

        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]['label'], 'hetero_r0')

    @staticmethod
    def _write_completed_history(directory, logger_id, test_mae):
        os.makedirs(directory, exist_ok=True)
        with open(
                TrainingLogger.filepath_for(directory, logger_id),
                'w',
                newline='',
        ) as history_file:
            writer = csv.DictWriter(
                history_file,
                fieldnames=TrainingLogger.HEADER,
            )
            writer.writeheader()
            writer.writerow({'epoch': 'TEST', 'test_mae': test_mae})

    def test_resume_results_are_isolated_by_norm_directory(self):
        with tempfile.TemporaryDirectory() as root:
            layernorm_dir = os.path.join(root, 'norm_layernorm')
            batchnorm_dir = os.path.join(root, 'norm_batchnorm')
            self._write_completed_history(layernorm_dir, 123, 0.1)
            self._write_completed_history(batchnorm_dir, 123, 0.2)

            self.assertEqual(
                completed_resume_losses(layernorm_dir, [123], cv5=False),
                [0.1],
            )
            self.assertEqual(
                completed_resume_losses(batchnorm_dir, [123], cv5=False),
                [0.2],
            )

    def test_incomplete_cv_resume_requires_all_five_folds(self):
        with tempfile.TemporaryDirectory() as norm_dir:
            for fold in range(1, 5):
                self._write_completed_history(
                    norm_dir,
                    f'123_fold{fold}',
                    fold / 100,
                )
            self.assertIsNone(
                completed_resume_losses(norm_dir, [123], cv5=True)
            )

            self._write_completed_history(norm_dir, '123_fold5', 0.05)
            self.assertEqual(
                completed_resume_losses(norm_dir, [123], cv5=True),
                [0.01, 0.02, 0.03, 0.04, 0.05],
            )


if __name__ == '__main__':
    unittest.main()
