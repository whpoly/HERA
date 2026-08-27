import unittest

import torch

from HERA.models.alignn import AttentionALIGNN, DefiNetALIGNN
from HERA.models.cgcnn import AttentionCGCNN, DefiNet
from HERA.models.megnet import AttentionMEGNet
from HERA.models.modules import (
    AtomTypeAttentionMegnetModule,
    AttentionCGConv,
    DefectAwareGateConv,
)


class AttentionDefiNetLayerNormTests(unittest.TestCase):
    def assert_uses_only_layernorm(self, model):
        batch_norms = [
            module
            for module in model.modules()
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
        ]
        layer_norms = [
            module
            for module in model.modules()
            if isinstance(module, torch.nn.LayerNorm)
        ]
        self.assertFalse(batch_norms)
        self.assertTrue(layer_norms)

    def test_all_attention_backbones_use_layernorm_without_batchnorm(self):
        models = [
            AttentionCGCNN(
                orig_atom_fea_len=6,
                nbr_fea_len=4,
                atom_fea_len=8,
                n_conv=2,
            ),
            AttentionMEGNet(
                edge_input_shape=4,
                node_input_shape=6,
                state_input_shape=2,
                embedding_size=8,
                n_blocks=2,
            ),
            AttentionALIGNN(
                node_input_shape=6,
                edge_input_shape=4,
                hidden_dim=8,
                n_blocks=1,
                gcn_blocks=1,
                angle_embed_size=4,
            ),
        ]

        for model in models:
            with self.subTest(model=type(model).__name__):
                self.assert_uses_only_layernorm(model)

    def test_all_definet_backbones_use_layernorm_without_batchnorm(self):
        models = [
            DefiNet(
                orig_atom_fea_len=6,
                nbr_fea_len=4,
                atom_fea_len=8,
                n_conv=2,
            ),
            DefiNetALIGNN(
                node_input_shape=6,
                edge_input_shape=4,
                hidden_dim=8,
                n_blocks=1,
                gcn_blocks=1,
                angle_embed_size=4,
            ),
        ]

        for model in models:
            with self.subTest(model=type(model).__name__):
                self.assert_uses_only_layernorm(model)

    def test_attention_cgcnn_normalizes_only_the_residual_delta(self):
        conv = AttentionCGConv(channels=8, dim=4, n_heads=2)
        root = torch.randn(1, 8) + 5.0
        edge_index = torch.tensor([[0], [0]])
        edge_attr = torch.randn(1, 4)

        output = conv(root, edge_index, edge_attr)
        delta = output - root

        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.allclose(delta.mean(dim=-1), torch.zeros(1), atol=1e-5))
        self.assertTrue(torch.allclose(output.mean(dim=-1), root.mean(dim=-1), atol=1e-5))

    def test_definet_normalizes_only_the_residual_delta(self):
        conv = DefectAwareGateConv(channels=8, dim=4, n_marker_types=2)
        root = torch.randn(1, 8) + 5.0
        edge_index = torch.tensor([[0], [0]])
        edge_attr = torch.randn(1, 4)

        output = conv(root, edge_index, edge_attr, defect_marker=torch.ones(1, dtype=torch.long))
        delta = output - root

        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.allclose(delta.mean(dim=-1), torch.zeros(1), atol=1e-5))
        self.assertTrue(torch.allclose(output.mean(dim=-1), root.mean(dim=-1), atol=1e-5))

    def test_attention_megnet_normalizes_node_edge_and_state_deltas(self):
        block = AtomTypeAttentionMegnetModule(
            edge_input_shape=8,
            node_input_shape=8,
            state_input_shape=8,
            embed_size=8,
            n_heads=2,
        )
        root_node = torch.randn(1, 8) + 5.0
        root_edge = torch.randn(1, 8) + 5.0
        root_state = torch.randn(1, 8) + 5.0
        edge_index = torch.tensor([[0], [0]])
        batch = torch.zeros(1, dtype=torch.long)
        bond_batch = torch.zeros(1, dtype=torch.long)

        node, edge, state = block(
            root_node,
            edge_index,
            root_edge,
            root_state,
            batch,
            bond_batch,
        )

        for output, root in (
                (node, root_node),
                (edge, root_edge),
                (state, root_state)):
            delta = output - root
            self.assertTrue(torch.isfinite(output).all())
            self.assertTrue(torch.allclose(
                delta.mean(dim=-1),
                torch.zeros(1),
                atol=1e-5,
            ))

    def test_all_attention_and_definet_models_forward_with_layernorm(self):
        x = torch.randn(3, 6)
        edge_index = torch.tensor([
            [0, 1, 1, 2, 2, 0],
            [1, 0, 2, 1, 0, 2],
        ])
        edge_attr = torch.randn(edge_index.size(1), 4)
        edge_vec = torch.randn(edge_index.size(1), 3)
        batch = torch.zeros(3, dtype=torch.long)
        bond_batch = torch.zeros(edge_index.size(1), dtype=torch.long)
        node_type = torch.tensor([0, 0, 1])

        models_and_calls = [
            (
                AttentionCGCNN(6, 4, atom_fea_len=8, n_conv=1),
                lambda model: model(x, edge_index, edge_attr, batch, node_type),
            ),
            (
                AttentionMEGNet(4, 6, 2, embedding_size=8, n_blocks=1),
                lambda model: model(
                    x,
                    edge_index,
                    edge_attr,
                    torch.zeros(1, 2),
                    batch,
                    bond_batch,
                    node_type,
                ),
            ),
            (
                AttentionALIGNN(
                    6,
                    4,
                    hidden_dim=8,
                    n_blocks=1,
                    gcn_blocks=1,
                    angle_embed_size=4,
                ),
                lambda model: model(
                    x,
                    edge_index,
                    edge_attr,
                    batch,
                    edge_vec,
                    node_type,
                ),
            ),
            (
                DefiNet(6, 4, atom_fea_len=8, n_conv=1),
                lambda model: model(
                    x,
                    edge_index,
                    edge_attr,
                    batch,
                    node_type,
                ),
            ),
            (
                DefiNetALIGNN(
                    6,
                    4,
                    hidden_dim=8,
                    n_blocks=1,
                    gcn_blocks=1,
                    angle_embed_size=4,
                ),
                lambda model: model(
                    x,
                    edge_index,
                    edge_attr,
                    batch,
                    edge_vec,
                    node_type,
                ),
            ),
        ]

        for model, call in models_and_calls:
            with self.subTest(model=type(model).__name__):
                model.train()
                output = call(model)
                self.assertEqual(tuple(output.shape), (1, 1))
                self.assertTrue(torch.isfinite(output).all())
                output.sum().backward()
                gradients = [
                    parameter.grad
                    for parameter in model.parameters()
                    if parameter.grad is not None
                ]
                self.assertTrue(gradients)
                self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))


if __name__ == "__main__":
    unittest.main()
