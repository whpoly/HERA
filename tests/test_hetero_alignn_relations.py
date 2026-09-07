import unittest

import torch

from HERA.models.alignn import (
    GatedGraphConv,
    HeteroALIGNN,
    HeteroNodeUpdate,
    HeteroRelationConv,
    _pool_mean_or_zeros,
)


METADATA = (
    ["atom", "defect"],
    [
        ("atom", "aa", "atom"),
        ("defect", "dd", "defect"),
        ("atom", "ad", "defect"),
        ("defect", "da", "atom"),
    ],
)

AD_KEY = "atom__ad__defect"
DA_KEY = "defect__da__atom"


class HeteroAlignnRelationTests(unittest.TestCase):
    def test_ad_and_da_use_distinct_parameters(self):
        model = HeteroALIGNN(
            node_input_shape=92,
            edge_input_shape=40,
            metadata=METADATA,
            hidden_dim=32,
            n_blocks=1,
            gcn_blocks=1,
        )

        self.assertIn(AD_KEY, model.edge_embedding)
        self.assertIn(DA_KEY, model.edge_embedding)
        self.assertIsNot(model.edge_embedding[AD_KEY], model.edge_embedding[DA_KEY])
        self.assertIsNot(
            model.layers[0].atom_convs[AD_KEY],
            model.layers[0].atom_convs[DA_KEY],
        )
        self.assertIsNot(
            model.gcn_layers[0].atom_convs[AD_KEY],
            model.gcn_layers[0].atom_convs[DA_KEY],
        )

    def test_node_update_layer_normalizes_only_residual_delta(self):
        update = HeteroNodeUpdate(channels=8, num_relations=2)
        update.train()
        x = (torch.randn(4, 8) + 5.0).requires_grad_()
        relation_inputs = [torch.zeros_like(x), torch.randn_like(x)]

        output = update(x, relation_inputs)

        self.assertIsInstance(update.layer_norm, torch.nn.LayerNorm)
        self.assertIsInstance(update.batch_norm, torch.nn.Identity)
        self.assertEqual(tuple(output.shape), (4, 8))
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.allclose(
            (output - x).detach().mean(dim=-1),
            torch.zeros(4),
            atol=1e-5,
        ))
        self.assertTrue(torch.allclose(
            output.detach().mean(dim=-1),
            x.detach().mean(dim=-1),
            atol=1e-5,
        ))

        output.square().mean().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_node_update_supports_one_defect_node(self):
        update = HeteroNodeUpdate(channels=8, num_relations=2)
        update.train()
        x = torch.randn(1, 8, requires_grad=True)

        output = update(x, [torch.zeros_like(x), torch.randn_like(x)])

        self.assertEqual(tuple(output.shape), (1, 8))
        self.assertTrue(torch.isfinite(output).all())
        output.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_node_update_can_disable_layer_norm(self):
        update = HeteroNodeUpdate(
            channels=8,
            num_relations=2,
            normalization="none",
        )
        x = torch.randn(4, 8)
        relation_inputs = [torch.zeros_like(x), torch.randn_like(x)]

        expected = update.fusion(x, relation_inputs)
        output = update(x, relation_inputs)

        self.assertIsInstance(update.layer_norm, torch.nn.Identity)
        self.assertIsInstance(update.batch_norm, torch.nn.Identity)
        self.assertTrue(torch.allclose(output, expected))

    def test_node_update_can_use_safe_batch_norm(self):
        update = HeteroNodeUpdate(
            channels=8,
            num_relations=2,
            normalization="batchnorm",
        )
        update.train()
        x = torch.randn(1, 8)
        relation_inputs = [torch.zeros_like(x), torch.randn_like(x)]

        output = update(x, relation_inputs)

        self.assertIsInstance(update.layer_norm, torch.nn.Identity)
        self.assertIsInstance(update.batch_norm, torch.nn.BatchNorm1d)
        self.assertTrue(torch.isfinite(output).all())

    def test_model_propagates_disabled_node_norm_to_all_hetero_blocks(self):
        model = HeteroALIGNN(
            node_input_shape=92,
            edge_input_shape=40,
            metadata=METADATA,
            hidden_dim=32,
            n_blocks=1,
            gcn_blocks=1,
            node_delta_norm="none",
        )

        updates = [
            *model.layers[0].node_updates.values(),
            *model.gcn_layers[0].node_updates.values(),
        ]
        self.assertTrue(updates)
        self.assertTrue(all(
            isinstance(update.layer_norm, torch.nn.Identity)
            for update in updates
        ))

    def test_hetero_relation_amp_accumulates_in_root_dtype(self):
        conv = HeteroRelationConv(channels=8, edge_dim=8).eval()
        x = torch.randn(3, 8)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
        edge_attr = torch.randn(3, 8)

        with torch.autocast('cpu', dtype=torch.bfloat16):
            message_sum, gate_sum, edge_update = conv(
                (x, x), edge_index, edge_attr,
            )

        self.assertEqual(message_sum.dtype, x.dtype)
        self.assertEqual(gate_sum.dtype, x.dtype)
        self.assertTrue(torch.isfinite(message_sum).all())
        self.assertTrue(torch.isfinite(gate_sum).all())
        self.assertTrue(torch.isfinite(edge_update).all())

    def test_homogeneous_gate_amp_accumulates_in_root_dtype(self):
        conv = GatedGraphConv(channels=8, edge_dim=8).eval()
        x = torch.randn(3, 8)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
        edge_attr = torch.randn(3, 8)

        with torch.autocast('cpu', dtype=torch.bfloat16):
            node_update, edge_update = conv(
                x,
                edge_index,
                edge_attr,
                return_edge_attr=True,
            )

        self.assertEqual(node_update.dtype, x.dtype)
        self.assertTrue(torch.isfinite(node_update).all())
        self.assertTrue(torch.isfinite(edge_update).all())

    def test_pooling_accepts_mixed_feature_and_reference_dtypes(self):
        features = torch.randn(4, 8).to(torch.bfloat16)
        batch = torch.tensor([0, 0, 1, 1])
        reference = torch.randn(1, 8)

        pooled = _pool_mean_or_zeros(
            features,
            batch,
            dim_size=2,
            width=8,
            reference=reference,
        )

        self.assertEqual(pooled.dtype, reference.dtype)
        self.assertTrue(torch.isfinite(pooled).all())


if __name__ == "__main__":
    unittest.main()
