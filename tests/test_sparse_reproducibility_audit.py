import copy
import io
import unittest

import torch

from HERA.check_sparse_reproducibility import compare_reports, compare_tensors, plain_cpu, source_location, tensor_hash
from HERA.utils.scaler import MyTensor


class SparseReproducibilityAuditTests(unittest.TestCase):
    def test_snapshots_strip_graph_tensor_subclasses(self):
        value = MyTensor([1.0, 2.0])
        output = io.BytesIO()
        torch.save({'prediction': plain_cpu(value)}, output)
        output.seek(0)
        restored = torch.load(output, weights_only=True)['prediction']
        self.assertIs(type(restored), torch.Tensor)
        self.assertTrue(torch.equal(restored, value))

    def reports(self):
        state = {k: {'weight': torch.tensor([1.0])} for k in
                 ('initial', 'first_gradient', 'final', 'checkpoint_predictions', 'final_predictions')}
        report = {
            'provenance': {'code_sha256': 'code'}, 'config': {}, 'scaler': {'mean': 0},
            'sources': {'train': ['sample']}, 'graph_hashes': {'train': ['graph']},
            'initial_hash': 'initial', 'small_val_mae': 0.1,
            'inference_within_process': [{'exact': True, 'max_abs': 0}],
            'trace': [{'step': 1, 'batch_ids': [0], 'batch_hash': 'batch', 'prediction_hash': 'prediction',
                       'loss': 0.1, 'gradient_hash': 'gradient', 'updated_weight_hash': 'updated'}],
        }
        return report, state

    def test_exact_forward_does_not_hide_backward_divergence(self):
        left, state = self.reports()
        right = copy.deepcopy(left)
        right['trace'][0]['gradient_hash'] = 'different'
        result = compare_reports(left, right, state, state)
        self.assertEqual(result['status'], 'NUMERICAL_DIFFERENCES')
        self.assertEqual(result['first_training_difference'], {'step': 1, 'stage': 'gradient_hash'})

    def test_different_inputs_are_not_called_numerical_noise(self):
        left, state = self.reports()
        right = copy.deepcopy(left)
        right['graph_hashes']['train'][0] = 'different'
        self.assertEqual(compare_reports(left, right, state, state)['status'], 'INPUT_OR_ENVIRONMENT_MISMATCH')

    def test_inference_variation_is_detected_even_if_training_matches(self):
        left, state = self.reports()
        right = copy.deepcopy(left)
        right['inference_within_process'][0]['exact'] = False
        self.assertEqual(compare_reports(left, right, state, state)['status'], 'NUMERICAL_DIFFERENCES')

    def test_nonfinite_values_never_pass(self):
        value = {'weight': torch.tensor([float('nan')])}
        self.assertFalse(compare_tensors(value, value)['exact'])

    def test_hash_preserves_dtype_shape_and_tiny_differences(self):
        value = torch.tensor([1.0, 2.0])
        base = tensor_hash({'x': value})
        self.assertNotEqual(base, tensor_hash({'x': value.reshape(1, 2)}))
        self.assertNotEqual(base, tensor_hash({'x': value.double()}))
        self.assertNotEqual(base, tensor_hash({'x': torch.nextafter(value, value+1)}))

    def test_material_and_density_require_evidence(self):
        self.assertEqual(source_location({'source_id': 'WSe2_a', 'source_path': '/low_density_defects/WSe2/a.cif'}), ('WSe2', 'low'))
        self.assertEqual(source_location({'source_id': 'b', 'material': 'MoS2', 'concentration': 'high'}), ('MoS2', 'high'))
        with self.assertRaises(ValueError):
            source_location({'source_id': 'WSe2_a'})


if __name__ == '__main__':
    unittest.main()
