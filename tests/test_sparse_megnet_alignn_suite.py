import csv
import tempfile
import unittest
from pathlib import Path

from HERA.config.defaults import HYPERGRAPH_SCHEMA, get_config, hypergraph_run_components
from HERA.sparse_megnet_alignn import (
    VALID_ALIGNN_MODES,
    alignn_result_prefix,
    discover_prediction_files,
    merge_prediction_files,
    parse_args,
    prediction_file_path,
    result_prefix,
)


class SparseMegnetAlignnSuiteTests(unittest.TestCase):
    @staticmethod
    def write_predictions(path, predictions):
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = (
            'source_id', 'source_name', 'source_path', 'material',
            'concentration', 'defect_family', 'target', 'prediction',
            'absolute_error',
        )
        with path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for source_id, target, prediction in predictions:
                writer.writerow({
                    'source_id': source_id,
                    'source_name': source_id,
                    'source_path': f'high/{source_id}.cif',
                    'material': 'MoS2',
                    'concentration': 'high',
                    'defect_family': 'point_defect',
                    'target': target,
                    'prediction': prediction,
                    'absolute_error': abs(prediction - target),
                })

    @staticmethod
    def alignn_prediction_path(root, dataset, mode, seed, radius='0'):
        path = Path(root) / 'alignn' / dataset / mode
        if mode in ('hetero', 'hetero_fixed_pool', 'hetero_was'):
            path /= f'r{radius}'
        elif mode in ('hypergraph', 'hypergraph_was'):
            path = path.joinpath(*hypergraph_run_components(get_config('alignn', dataset, mode)['model']))
        return path / f'seed{seed}_test_predictions.csv'

    def test_mode_paths_match_main_output_layout(self):
        root = Path('run')
        self.assertEqual(
            prediction_file_path(root, 'megnet', '2dmd_mos2', 'sparse', 123),
            root / 'megnet/2dmd_mos2/sparse/seed123_test_predictions.csv',
        )
        self.assertEqual(
            prediction_file_path(root, 'alignn', '2dmd_mos2', 'hetero', 123),
            root / 'alignn/2dmd_mos2/hetero/r0/features_layernorm/pool_defect_mean/relations_shared_residual_rank8/seed123_test_predictions.csv',
        )
        self.assertEqual(
            prediction_file_path(root, 'alignn', '2dmd_mos2', 'hypergraph', 123),
            root / (
                'alignn/2dmd_mos2/hypergraph/'
                'defect_global_attention_v3/pool_defect_mean/seed123_test_predictions.csv'
            ),
        )

    def test_predictions_from_all_models_are_merged_into_one_wide_csv(self):
        dataset = '2dmd_mos2'
        seed = 123
        model_modes = (
            ('megnet', 'sparse', [("a", 1.0, 1.1), ("b", 2.0, 2.2)]),
            ('alignn', 'full', [("a", 1.0, 0.9), ("b", 2.0, 2.1)]),
            ('alignn', 'hetero', [("a", 1.0, 1.0), ("b", 2.0, 2.05)]),
            ('alignn', 'hypergraph', [("a", 1.0, 1.2), ("b", 2.0, 1.8)]),
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            for model, mode, predictions in model_modes:
                self.write_predictions(
                    prediction_file_path(root, model, dataset, mode, seed),
                    predictions,
                )

            output_path, rows, merged_prefixes = merge_prediction_files(
                root,
                [dataset],
                [seed],
                ['full', 'hetero', 'hypergraph'],
            )
            with output_path.open(newline='') as handle:
                written_rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 2)
        self.assertEqual(len(written_rows), 2)
        self.assertEqual(
            merged_prefixes,
            [
                'megnet_sparse',
                'alignn_full',
                'alignn_hetero_r0_features_layernorm_pool_defect_mean_relations_shared_residual_rank8',
                'alignn_hypergraph_pool_defect_mean',
            ],
        )
        self.assertEqual(written_rows[0]['source_id'], 'a')
        self.assertEqual(float(written_rows[0]['megnet_sparse_prediction']), 1.1)
        self.assertEqual(float(written_rows[0]['alignn_full_prediction']), 0.9)
        self.assertEqual(float(written_rows[0]['alignn_hetero_r0_features_layernorm_pool_defect_mean_relations_shared_residual_rank8_prediction']), 1.0)
        self.assertEqual(float(written_rows[0]['alignn_hypergraph_pool_defect_mean_prediction']), 1.2)
        self.assertAlmostEqual(float(written_rows[0]['megnet_sparse_test_mae']), 0.15)
        self.assertAlmostEqual(float(written_rows[0]['alignn_full_test_mae']), 0.1)

    def test_result_prefix_marks_the_fixed_hetero_radius(self):
        self.assertEqual(result_prefix('alignn', 'hetero'), 'alignn_hetero_r0')

    def test_alignn_discovery_keeps_each_radius_and_ignores_schema_folder(self):
        dataset = '2dmd_wse2'
        seed = 11
        predictions = [('a', 1.0, 1.1)]
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            self.write_predictions(
                prediction_file_path(root, 'megnet', dataset, 'sparse', seed),
                predictions,
            )
            alignn_root = root / 'alignn' / dataset
            radius_path = (
                alignn_root / 'hetero/r3' /
                f'seed{seed}_test_predictions.csv'
            )
            hypergraph_path = (
                alignn_root / 'hypergraph' / HYPERGRAPH_SCHEMA /
                f'seed{seed}_test_predictions.csv'
            )
            self.write_predictions(radius_path, predictions)
            self.write_predictions(hypergraph_path, predictions)

            sources = discover_prediction_files(
                root, dataset, seed, ['hetero', 'hypergraph'],
            )

        self.assertEqual(
            [prefix for prefix, _ in sources],
            ['megnet_sparse', 'alignn_hetero_r3', 'alignn_hypergraph'],
        )
        self.assertEqual(
            alignn_result_prefix(alignn_root, hypergraph_path),
            'alignn_hypergraph',
        )

    def test_cli_defaults_to_all_alignn_modes_at_radius_zero(self):
        args = parse_args([
            '--dataset', '2dmd_mos2',
            '--alignn-mode', 'all',
        ])
        self.assertEqual(args.dataset, ['2dmd_mos2'])
        self.assertEqual(args.alignn_mode, list(VALID_ALIGNN_MODES))
        self.assertEqual(args.alignn_r, ['0'])

    def test_pooling_variants_keep_distinct_prediction_columns(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            schema_dir = root / 'alignn/2dmd_mos2/hypergraph' / HYPERGRAPH_SCHEMA
            for folder, prediction in [(schema_dir, 1.2), (schema_dir / 'pool_defect_mean', 1.1)]:
                self.write_predictions(folder / 'seed123_test_predictions.csv', [('a', 1., prediction)])
            self.write_predictions(prediction_file_path(root, 'megnet', '2dmd_mos2', 'sparse', 123),
                                   [('a', 1., 1.3)])
            _, rows, prefixes = merge_prediction_files(root, ['2dmd_mos2'], [123], ['hypergraph'])
        self.assertIn('alignn_hypergraph', prefixes)
        self.assertIn('alignn_hypergraph_pool_defect_mean', prefixes)
        self.assertEqual(float(rows[0]['alignn_hypergraph_prediction']), 1.2)
        self.assertEqual(float(rows[0]['alignn_hypergraph_pool_defect_mean_prediction']), 1.1)

    def test_all_alignn_modes_and_only_selected_radius_are_merged(self):
        dataset = '2dmd_mos2'
        seed = 123
        predictions = [('a', 1.0, 1.1)]
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            self.write_predictions(
                prediction_file_path(root, 'megnet', dataset, 'sparse', seed),
                predictions,
            )
            for mode in VALID_ALIGNN_MODES:
                if mode == 'hetero_fixed_pool':
                    continue
                self.write_predictions(
                    self.alignn_prediction_path(
                        root, dataset, mode, seed, radius='0',
                    ),
                    predictions,
                )
            # A stale result from another radius must not enter an r=0 merge.
            self.write_predictions(
                self.alignn_prediction_path(
                    root, dataset, 'hetero', seed, radius='3',
                ),
                predictions,
            )

            _, _, prefixes = merge_prediction_files(
                root,
                [dataset],
                [seed],
                list(VALID_ALIGNN_MODES),
                alignn_r=['0'],
            )

        self.assertEqual(prefixes[0], 'megnet_sparse')
        self.assertIn('alignn_hetero_r0', prefixes)
        self.assertNotIn('alignn_hetero_r3', prefixes)
        self.assertNotIn('alignn_hetero_fixed_pool_r0', prefixes)
        self.assertEqual(len(prefixes), len(VALID_ALIGNN_MODES))


if __name__ == '__main__':
    unittest.main()
