import unittest

import torch

from HERA.models.alignn import HeteroALIGNN, HeteroNodeUpdate, SafeBatchNorm1d


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

    def test_node_update_batch_normalizes_complete_residual(self):
        update = HeteroNodeUpdate(channels=8, num_relations=2)
        update.train()
        x = torch.randn(4, 8, requires_grad=True)
        relation_inputs = [torch.zeros_like(x), torch.randn_like(x)]

        output = update(x, relation_inputs)

        self.assertIsInstance(update.batch_norm, SafeBatchNorm1d)
        self.assertFalse(hasattr(update, "layer_norm"))
        self.assertEqual(tuple(output.shape), (4, 8))
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.allclose(
            output.detach().mean(dim=0),
            torch.zeros(8),
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
