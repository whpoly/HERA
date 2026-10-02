"""Check independent split identity and scoring, without model training."""
import copy
import unittest

from HERA.scripts.run_official_alignn_native import raw_split, validate_split, metrics, config_for


class OfficialNativeProtocolTests(unittest.TestCase):
    def test_raw_split_is_complete_disjoint_and_reproducible(self):
        labels = {str(i): float(i) for i in range(3070)}
        splits = raw_split(labels, 123)
        validate_split(splits, labels)
        self.assertEqual([len(splits[k]) for k in ('train', 'val', 'test')], [1842, 614, 614])
        self.assertEqual(splits, raw_split(labels, 123))
        self.assertEqual(set().union(*map(set, splits.values())), set(labels))
        broken = copy.deepcopy(splits)
        broken['test'][0] = broken['train'][0]
        with self.assertRaises(ValueError):
            validate_split(broken, labels)

    def test_rmse_keeps_large_errors_and_rejects_nonfinite(self):
        rows = [dict(target=0., prediction=0.), dict(target=0., prediction=10.)]
        result = metrics(rows)
        self.assertEqual(result['n'], 2)
        self.assertEqual(result['mae_ev'], 5.)
        self.assertAlmostEqual(result['rmse_ev'], 50 ** .5)
        rows[1]['prediction'] = float('nan')
        with self.assertRaises(ValueError):
            metrics(rows)

    def test_config_distinguishes_embedding_and_hidden_sizes(self):
        manifest = dict(seed=123, splits=dict(train=['a']*1842, val=['b']*614, test=['c']*614))
        config = config_for(manifest, 'out', 150, 0)
        self.assertEqual(config['model']['embedding_features'], 64)
        self.assertEqual(config['model']['hidden_features'], 256)
        self.assertEqual(config['model']['alignn_layers'], 4)
        self.assertEqual(config['model']['gcn_layers'], 4)
        self.assertEqual(config['cutoff'], 8.)
        self.assertTrue(config['keep_data_order'])
        self.assertFalse(config['standard_scalar_and_pca'])
        self.assertIsNone(config['n_early_stopping'])


if __name__ == '__main__':
    unittest.main()
