import unittest

import torch

from HERA.models.alignn import HeteroALIGNN, HeteroNodeUpdate


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
        self.assertFalse(hasattr(update, "batch_norm"))
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


if __name__ == "__main__":
    unittest.main()
