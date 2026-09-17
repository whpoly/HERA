"""Integration checks for frozen checkpoint explanations (local artifacts required)."""
import unittest
import warnings
import torch
from HERA.explain.checkpoint_comparison import (
    ROOT, CHECKPOINTS, init_elem_embedding, load_models, selected_cases,
    prepare_case, explain_one, ablate_relation, native_readout,
)


@unittest.skipUnless(all((ROOT / p).exists() for p in CHECKPOINTS.values()), 'Local checkpoints required')
class CheckpointExplainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        warnings.filterwarnings('ignore', message='.*Issues encountered while parsing CIF.*')
        init_elem_embedding(ROOT / 'atom_init.json')
        trainers, _ = load_models(list(CHECKPOINTS), 'cpu')
        cls.wrappers, cls.geometry, _ = prepare_case(selected_cases('MoS2', [4])[0], trainers, 'cpu')

    def test_vacancy_masks_have_gradients_and_model_stays_frozen(self):
        for name, wrapper in self.wrappers.items():
            with self.subTest(model=name):
                before = {k: v.clone() for k, v in wrapper.model.state_dict().items()}
                mask = torch.ones((len(wrapper.x), 1), requires_grad=True)
                wrapper(wrapper.x * mask).sum().backward()
                vacant = wrapper.raw_x.abs().sum(1).eq(0)
                self.assertTrue(vacant.any())
                self.assertGreater(float(mask.grad[vacant].abs().sum()), 0)
                masks, _, _ = explain_one(wrapper, 2, [123], .03, .01, .001)
                self.assertEqual(masks.shape, (1, len(wrapper.x)))
                for key, value in wrapper.model.state_dict().items():
                    torch.testing.assert_close(value, before[key], atol=0, rtol=0)
                self.assertTrue(all(p.grad is None for p in wrapper.model.parameters()))

    def test_relation_hooks_restore_after_exception_and_readout_reconstructs(self):
        for wrapper in self.wrappers.values():
            if not wrapper.hetero:
                continue
            before = wrapper(wrapper.x).detach()
            with self.assertRaisesRegex(RuntimeError, 'deliberate'):
                with ablate_relation(wrapper.model, 'ad'):
                    self.assertFalse(torch.allclose(before, wrapper(wrapper.x)))
                    raise RuntimeError('deliberate')
            torch.testing.assert_close(wrapper(wrapper.x), before, rtol=0, atol=0)
        readout = native_readout(self.wrappers['hetero_shared'])
        self.assertEqual(len(readout['values_ev']), 4)
        self.assertLess(readout['mean_reconstruction_error_ev'], 1e-5)


if __name__ == '__main__':
    unittest.main()
