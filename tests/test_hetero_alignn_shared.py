"""Shared hetero messages: trainability, isolation, pooling and legacy restore."""
import copy
import unittest
from pathlib import Path

import torch
from torch import nn
from torch_geometric.loader import DataLoader

from HERA.config.defaults import get_config, apply_alignn_hetero_options, alignn_hetero_run_components
from HERA.data.datasets import init_elem_embedding
from HERA.models.alignn import SharedHeteroRelations, HeteroRelationConv
from HERA.native_initial_relaxed_leave_one_out import expand_leave_one_out_runs
from HERA.native_ood_case_study import model_mode_display
from HERA.sparse_megnet_alignn import alignn_result_prefix
from HERA.training.trainer import MEGNetTrainer
from HERA.tests.test_hypergraph import (
    make_two_independent_defect_structure, make_periodic_three_region_structure,
    make_overlapping_defect_neighborhood_structure,
)


MODES = ('independent', 'shared', 'shared_residual')
KEYS = ('atom__aa__atom', 'defect__dd__defect', 'atom__ad__defect', 'defect__da__atom')


class HeteroSharedMessageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        init_elem_embedding(Path(__file__).resolve().parents[1] / 'atom_init.json')

    def trainer(self, mode, rank=2, width=8):
        config = get_config('alignn', '2dmd_mos2', 'hetero')
        config['model'].update(embedding_size=width, nblocks=1, gcn_blocks=1,
                               edge_embed_size=4, angle_embed_size=4, local_radius=0,
                               hetero_relation_mode=mode, hetero_relation_rank=rank)
        torch.manual_seed(123)
        return MEGNetTrainer(config, 'cpu', seed=123)

    def batch(self, trainer):
        graphs = [trainer.converter.convert(s) for s in (
            make_two_independent_defect_structure(),
            make_overlapping_defect_neighborhood_structure(),
        )]
        return next(iter(DataLoader(graphs, batch_size=2)))

    def test_unaffected_initial_weights_and_graph_inputs_are_identical(self):
        reference = None
        graph_reference = None
        for mode in MODES:
            trainer = self.trainer(mode)
            state = {key: value for key, value in trainer.model.state_dict().items()
                     if '.atom_convs.' not in key}
            graph = trainer.converter.convert(make_overlapping_defect_neighborhood_structure())
            self.assertEqual(trainer.model.readout[0].in_features, 8)
            self.assertFalse(any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in trainer.model.modules()))
            if reference is not None:
                self.assertEqual(set(state), set(reference))
                for key in state:
                    torch.testing.assert_close(state[key], reference[key], rtol=0, atol=0)
                for key in graph.node_types:
                    torch.testing.assert_close(graph[key].x, graph_reference[key].x, rtol=0, atol=0)
                for key in graph.edge_types:
                    for field in ('edge_index', 'edge_attr', 'edge_vec'):
                        torch.testing.assert_close(graph[key][field], graph_reference[key][field], rtol=0, atol=0)
            reference, graph_reference = state, graph

    def test_zero_adapters_make_the_initial_model_exactly_shared(self):
        shared = self.trainer('shared')
        residual = self.trainer('shared_residual')
        residual_state = residual.model.state_dict()
        for key, value in shared.model.state_dict().items():
            torch.testing.assert_close(value, residual_state[key], rtol=0, atol=0)
        torch.testing.assert_close(shared._forward(self.batch(shared)),
                                   residual._forward(self.batch(residual)), rtol=0, atol=0)

    def test_shared_core_is_registered_once_and_parameters_are_reduced(self):
        counts = {}
        for mode in MODES:
            model = self.trainer(mode, rank=8, width=64).model
            counts[mode] = sum(p.numel() for p in model.parameters())
            for block in [*model.layers, *model.gcn_layers]:
                convs = [m for m in block.atom_convs.modules() if isinstance(m, HeteroRelationConv)]
                self.assertEqual(len(convs), 4 if mode == 'independent' else 1)
                if mode == 'shared_residual':
                    self.assertEqual(set(block.atom_convs.adapters), set(KEYS))
                    self.assertEqual(len({id(m.up.weight) for m in block.atom_convs.adapters.values()}), 4)
        self.assertLess(counts['shared'], counts['shared_residual'])
        self.assertLess(counts['shared_residual'], counts['independent'])

    def test_every_relation_trains_the_common_core_and_only_its_own_adapter(self):
        bank = self.trainer('shared_residual').model.layers[0].atom_convs
        x_src, x_dst = torch.randn(3, 8), torch.randn(2, 8)
        edges = torch.tensor([[0, 1, 2], [0, 0, 1]])
        attrs = torch.randn(3, 8)
        for key in KEYS:
            bank.zero_grad(set_to_none=True)
            messages, gates, bonds = bank(key, (x_src, x_dst), edges, attrs)
            (messages.square().sum() + gates.square().sum() + bonds.square().sum()).backward()
            self.assertGreater(bank.shared.message_update.weight.grad.abs().sum(), 0)
            self.assertGreater(bank.shared.src_gate.weight.grad.abs().sum(), 0)
            self.assertGreater(bank.adapters[key].up.weight.grad.abs().sum(), 0)
            for other in set(KEYS) - {key}:
                self.assertTrue(all(p.grad is None for p in bank.adapters[other].parameters()))
        # After the zero output projection starts learning, the bottleneck
        # also receives gradients; the residual branch is not permanently off.
        with torch.no_grad():
            bank.adapters[KEYS[-1]].up.weight.add_(-.01 * bank.adapters[KEYS[-1]].up.weight.grad)
        bank.zero_grad(set_to_none=True)
        messages, _, _ = bank(KEYS[-1], (x_src, x_dst), edges, attrs)
        messages.square().sum().backward()
        self.assertGreater(bank.adapters[KEYS[-1]].down.weight.grad.abs().sum(), 0)

    def test_relation_residual_changes_only_that_relation(self):
        bank = self.trainer('shared_residual').model.layers[0].atom_convs
        x = torch.randn(2, 8)
        edges, attrs = torch.tensor([[0, 1], [1, 0]]), torch.randn(2, 8)
        before = {key: bank(key, x, edges, attrs)[0].detach().clone() for key in KEYS}
        with torch.no_grad():
            bank.adapters[KEYS[2]].up.bias[8:].fill_(1.0)
        for key in KEYS:
            after = bank(key, x, edges, attrs)[0]
            if key == KEYS[2]:
                self.assertGreater((after - before[key]).abs().max(), 0)
            else:
                torch.testing.assert_close(after, before[key], rtol=0, atol=0)

    def test_models_preserve_actual_defect_mean_and_batch_consistency(self):
        for mode in MODES:
            trainer = self.trainer(mode)
            structure = make_two_independent_defect_structure()
            structure.add_site_property('pool_type', [1, 0, 1, 0, 0])
            structure.add_site_property('type', [1, 1, 1, 0, 0])
            graphs = [trainer.converter.convert(structure),
                      trainer.converter.convert(make_periodic_three_region_structure())]
            batch = next(iter(DataLoader(graphs, batch_size=2)))
            captured = {}
            handles = [
                trainer.model.gcn_layers[-1].register_forward_hook(
                    lambda module, args, result: captured.update(nodes=result[0])),
                trainer.model.readout.register_forward_pre_hook(
                    lambda module, args: captured.update(pool=args[0])),
            ]
            together = trainer._forward(batch)
            for handle in handles:
                handle.remove()
            for index in range(2):
                mask = batch['defect'].pool_type.eq(1) & batch['defect'].batch.eq(index)
                torch.testing.assert_close(captured['pool'][index], captured['nodes']['defect'][mask].mean(0))
            alone = torch.cat([trainer._forward(b) for b in DataLoader(graphs, batch_size=1)])
            torch.testing.assert_close(together, alone, atol=3e-6, rtol=2e-5)
            trainer.model.eval()
            torch.testing.assert_close(together, trainer._forward(batch), atol=3e-6, rtol=2e-5)
            together.square().sum().backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in trainer.model.parameters() if p.grad is not None))
            self.assertGreater(trainer.model.node_embedding['atom'].layer[0].weight.grad.abs().sum(), 0)

    def test_empty_relations_and_no_host_graphs_remain_finite(self):
        for mode in ('shared', 'shared_residual'):
            trainer = self.trainer(mode)
            for no_host in (False, True):
                structure = make_periodic_three_region_structure()
                if no_host:
                    structure.add_site_property('type', [1, 1, 1])
                else:
                    structure.remove_sites([1, 2])
                batch = next(iter(DataLoader([trainer.converter.convert(structure)], batch_size=1)))
                trainer.model.zero_grad(set_to_none=True)
                result = trainer._forward(batch)
                self.assertTrue(torch.isfinite(result).all())
                result.square().sum().backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in trainer.model.parameters() if p.grad is not None))

    def test_all_modes_and_omitted_legacy_config_strictly_restore(self):
        for mode in MODES:
            trainer = self.trainer(mode)
            config = copy.deepcopy(trainer.config)
            if mode == 'independent':
                config['model'].pop('hetero_relation_mode')
                config['model'].pop('hetero_relation_rank')
            restored = MEGNetTrainer(config, 'cpu', seed=123)
            restored.model.load_state_dict(trainer.model.state_dict(), strict=True)
            self.assertEqual(restored.model.relation_mode, mode)
            torch.testing.assert_close(trainer._forward(self.batch(trainer)), restored._forward(self.batch(restored)))

    def test_default_scope_rank_validation_and_isolated_result_paths(self):
        for model, mode in (('alignn', 'attention'), ('alignn', 'hypergraph'), ('megnet', 'hetero')):
            config = get_config(model, '2dmd_mos2', mode)
            self.assertNotIn('hetero_relation_mode', config['model'])
        paths, labels, prefixes = set(), set(), set()
        for mode, rank in (('independent', None), ('shared', None), ('shared_residual', 4), ('shared_residual', 8)):
            run = expand_leave_one_out_runs('alignn', ['hetero'], None,
                hetero_relation_mode=mode, hetero_relation_rank=rank)[0]
            parts = alignn_hetero_run_components(run['config']['model'])
            path = Path('dataset/hetero/r0').joinpath(*parts)
            paths.add(path)
            labels.add(run['label'])
            prefixes.add(alignn_result_prefix(Path('dataset'), path / 'seed123_test_predictions.csv'))
            display = model_mode_display('alignn', run['label'] + '_norm_layernorm')
            if mode == 'shared_residual':
                self.assertIn(f'rank {rank}', display)
            self.assertIn('defect mean', display)
        self.assertEqual((len(paths), len(labels), len(prefixes)), (4, 4, 4))
        for invalid in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                self.trainer('shared_residual', rank=invalid)
        with self.assertRaises(ValueError):
            self.trainer('unsupported')
        with self.assertRaises(ValueError):
            apply_alignn_hetero_options(get_config('alignn', '2dmd_mos2', 'hetero'),
                                       relation_mode='shared', relation_rank=4)


if __name__ == '__main__':
    unittest.main()
