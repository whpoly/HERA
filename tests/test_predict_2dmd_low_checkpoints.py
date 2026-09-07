import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from HERA.predict_2dmd_low_checkpoints import (
    discover_low_checkpoints,
    load_material_high_test,
    parse_args,
)


def fake_checkpoint(model, mode, dataset='2dmd_low', seed=123):
    return {
        'model': {},
        'scaler': {'mean': 0.0, 'std': 1.0},
        'config': {'model': {}},
        'model_name': model,
        'dataset_name': dataset,
        'mode': mode,
        'run_label': mode,
        'split': {'seed': seed},
    }


class PredictLowCheckpointsTests(unittest.TestCase):
    def test_discovery_selects_sparse_and_non_was_alignn_only(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            payloads = {
                root / 'megnet/sparse/seed123_best_checkpoint.pth':
                    fake_checkpoint('megnet', 'sparse'),
                root / 'alignn/full/seed123_best_checkpoint.pth':
                    fake_checkpoint('alignn', 'full'),
                root / 'alignn/hetero_was/seed123_best_checkpoint.pth':
                    fake_checkpoint('alignn', 'hetero_was'),
                root / 'alignn/high/seed123_best_checkpoint.pth':
                    fake_checkpoint('alignn', 'hetero', dataset='2dmd_high'),
            }
            for path in payloads:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()

            with patch(
                'HERA.predict_2dmd_low_checkpoints.load_trusted_checkpoint',
                side_effect=lambda path, map_location: payloads[Path(path)],
            ):
                records = discover_low_checkpoints(root)

        self.assertEqual(
            [(record['model_name'], record['mode']) for record in records],
            [('megnet', 'sparse'), ('alignn', 'full')],
        )

    def test_material_loader_requests_high_only(self):
        structure = SimpleNamespace(
            source_id='high-1',
            concentration='high',
            material='MoS2',
        )
        dataset = ([structure], None, None, None, torch.tensor([1.25]))
        config = {'model': {'local_radius': 0}}
        with patch(
            'HERA.predict_2dmd_low_checkpoints.'
            '_load_data_2dmd_material_transfer',
            return_value=dataset,
        ) as loader:
            structures, targets = load_material_high_test(
                'mos2', 'alignn', 'full', config,
            )

        self.assertEqual(structures, [structure])
        self.assertEqual(float(targets[0]), 1.25)
        loader.assert_called_once_with(
            'MoS2',
            'alignn',
            local_cutoff=None,
            representations=['full'],
            pure_vacancy=False,
            include_low=False,
        )

    def test_cli_defaults_to_both_materials_and_available_non_was_modes(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            args = parse_args(['--checkpoint-root', str(root)])

        self.assertEqual(args.material, ['mos2', 'wse2'])
        self.assertIsNone(args.alignn_modes)
        self.assertIsNone(args.seeds)
        self.assertEqual(
            args.output_root,
            root.resolve() / 'high_test_predictions',
        )


if __name__ == '__main__':
    unittest.main()
