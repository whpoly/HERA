"""Regression checks for LayerNorm + actual-defect mean HeteroALIGNN."""
import copy
import unittest
from pathlib import Path

import torch
from torch import nn
from torch_geometric.loader import DataLoader

from HERA.config.defaults import (
    get_config, apply_alignn_hetero_options, alignn_hetero_run_components,
)
from HERA.data.datasets import init_elem_embedding
from HERA.native_initial_relaxed_leave_one_out import expand_leave_one_out_runs
from HERA.native_ood_case_study import model_mode_display
from HERA.sparse_megnet_alignn import alignn_result_prefix
from HERA.training.trainer import MEGNetTrainer
from HERA.tests.test_hypergraph import (
    make_two_independent_defect_structure, make_periodic_three_region_structure,
)


class HeteroAlignnTransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        init_elem_embedding(Path(__file__).resolve().parents[1] / 'atom_init.json')

    def trainer(self, feature_norm='layernorm', pooling='defect_mean', mode='hetero'):
        config = get_config('alignn', '2dmd_mos2', mode)
        config['model'].update(embedding_size=8, nblocks=1, gcn_blocks=1,
                               edge_embed_size=4, angle_embed_size=4, local_radius=0)
        apply_alignn_hetero_options(config, feature_norm, pooling)
        return MEGNetTrainer(config, 'cpu', seed=123)

    def graphs(self, trainer):
        return [trainer.converter.convert(s) for s in (
            make_two_independent_defect_structure(), make_periodic_three_region_structure(),
        )]

    def test_new_defaults_are_scoped_to_alignn_hetero(self):
        for mode in ('hetero', 'hetero_was', 'hetero_fixed_pool'):
            config = get_config('alignn', '2dmd_mos2', mode)
            self.assertEqual(config['model']['hetero_feature_norm'], 'layernorm')
            self.assertEqual(config['model']['hetero_pooling'], 'defect_mean')
        for model, mode in (('cgcnn', 'hetero'), ('megnet', 'hetero'),
                            ('alignn', 'attention'), ('alignn', 'hypergraph'), ('alignn', 'full')):
            config = get_config(model, '2dmd_mos2', mode)
            original = copy.deepcopy(config)
            apply_alignn_hetero_options(config, 'batchnorm', 'type_mean')
            self.assertEqual(config, original)
            self.assertNotIn('hetero_pooling', config['model'])

    def test_all_norms_are_layernorm_and_batching_does_not_change_output(self):
        trainer = self.trainer()
        self.assertFalse(any(isinstance(m, nn.modules.batchnorm._BatchNorm)
                             for m in trainer.model.modules()))
        self.assertEqual(trainer.model.readout[0].in_features, 8)
        graphs = self.graphs(trainer)
        batch = next(iter(DataLoader(graphs, batch_size=2)))
        trainer.model.train()
        together = trainer._forward(batch)
        alone = torch.cat([trainer._forward(b) for b in DataLoader(graphs, batch_size=1)])
        torch.testing.assert_close(together, alone, atol=2e-6, rtol=2e-5)
        trainer.model.eval()
        torch.testing.assert_close(together, trainer._forward(batch), atol=2e-6, rtol=2e-5)
        together.square().sum().backward()
        gradients = [p.grad for p in trainer.model.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
        # Pristine atoms still affect the prediction through physical messages.
        self.assertGreater(trainer.model.node_embedding['atom'].layer[0].weight.grad.abs().sum(), 0)

    def test_readout_uses_only_actual_defects_in_an_enlarged_region(self):
        for mode in ('hetero', 'hetero_fixed_pool'):
            trainer = self.trainer(mode=mode)
            structure = make_two_independent_defect_structure()
            structure.add_site_property('pool_type', [1, 0, 1, 0, 0])
            structure.add_site_property('type', [1, 1, 1, 0, 0])
            graph = trainer.converter.convert(structure)
            batch = next(iter(DataLoader([graph], batch_size=1)))
            captured = {}
            handles = [
                trainer.model.gcn_layers[-1].register_forward_hook(
                    lambda module, args, result: captured.update(nodes=result[0])),
                trainer.model.readout.register_forward_pre_hook(
                    lambda module, args: captured.update(pooled=args[0])),
            ]
            try:
                trainer._forward(batch)
            finally:
                for handle in handles:
                    handle.remove()
            nodes = captured['nodes']['defect']
            mask = batch['defect'].pool_type.bool()
            self.assertEqual(int(mask.sum()), 2)
            self.assertEqual(nodes.size(0), 3)
            torch.testing.assert_close(captured['pooled'], nodes[mask].mean(0, keepdim=True))

    def test_empty_relations_no_bonds_and_no_host_are_finite(self):
        trainer = self.trainer()
        for kind in ('no_bonds', 'no_host'):
            structure = make_periodic_three_region_structure()
            if kind == 'no_bonds':
                structure.remove_sites([1, 2])
            else:
                structure.add_site_property('type', [1, 1, 1])
            graph = trainer.converter.convert(structure)
            batch = next(iter(DataLoader([graph], batch_size=1)))
            trainer.model.zero_grad(set_to_none=True)
            output = trainer._forward(batch)
            self.assertTrue(torch.isfinite(output).all())
            output.square().sum().backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in trainer.model.parameters()
                                if p.grad is not None))

    def test_a_zero_defect_graph_in_a_mixed_batch_is_rejected(self):
        trainer = self.trainer()
        pristine = make_periodic_three_region_structure()
        pristine.add_site_property('type', [0, 0, 0])
        graphs = [*self.graphs(trainer), trainer.converter.convert(pristine)]
        with self.assertRaisesRegex(ValueError, 'at least one defect per graph'):
            trainer._forward(next(iter(DataLoader(graphs, batch_size=3))))

    def test_saved_configs_restore_both_poolings_and_missing_legacy_fields(self):
        for feature_norm in ('batchnorm', 'layernorm'):
            for pooling in ('type_mean', 'defect_mean'):
                trainer = self.trainer(feature_norm, pooling)
                config = copy.deepcopy(trainer.config)
                if feature_norm == 'batchnorm' and pooling == 'type_mean':
                    config['model'].pop('hetero_feature_norm')
                    config['model'].pop('hetero_pooling')
                restored = MEGNetTrainer(config, 'cpu', seed=123)
                restored.model.load_state_dict(trainer.model.state_dict(), strict=True)
                trainer.model.eval()
                restored.model.eval()
                batch = next(iter(DataLoader(self.graphs(trainer), batch_size=2)))
                torch.testing.assert_close(trainer._forward(batch), restored._forward(batch))

    def test_ablations_have_distinct_paths_labels_and_prediction_columns(self):
        labels, prefixes, paths = set(), set(), set()
        for feature_norm in ('batchnorm', 'layernorm'):
            for pooling in ('type_mean', 'defect_mean'):
                run = expand_leave_one_out_runs(
                    'alignn', ['hetero'], None, hetero_feature_norm=feature_norm,
                    hetero_pooling=pooling,
                )[0]
                parts = alignn_hetero_run_components(run['config']['model'])
                relative = Path('hetero/r0').joinpath(*parts)
                paths.add(relative)
                labels.add(run['label'])
                prefixes.add(alignn_result_prefix(Path('dataset'), Path('dataset') / relative / 'seed123_test_predictions.csv'))
        self.assertEqual(len(paths), 4)
        self.assertEqual(len(labels), 4)
        self.assertEqual(len(prefixes), 4)
        self.assertIn('hetero_r0', labels)
        display = model_mode_display('alignn', 'hetero_r0_features_layernorm_pool_defect_mean_norm_layernorm')
        self.assertIn('defect mean', display)
        self.assertIn('feature LayerNorm', display)


if __name__ == '__main__':
    unittest.main()
