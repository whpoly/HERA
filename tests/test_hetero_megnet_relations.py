import unittest

import torch

from HERA.models.modules import HeteroMegnetLayer, RelationFusionUpdate


METADATA = (
    ["atom", "defect"],
    [
        ("atom", "aa", "atom"),
        ("defect", "dd", "defect"),
        ("atom", "ad", "defect"),
        ("defect", "da", "atom"),
    ],
)


class HeteroMegnetRelationTests(unittest.TestCase):
    def make_layer(self):
        return HeteroMegnetLayer(
            edge_input_shape=40,
            node_input_shape=92,
            state_input_shape=2,
            metadata=METADATA,
            embedding_size=32,
            vertex_aggregation="sum",
            global_aggregation="mean",
            inner_skip=True,
        )

    def test_ad_and_da_use_distinct_megnet_modules(self):
        layer = self.make_layer()
        ad_key = layer.relation_module_keys[("atom", "ad", "defect")]
        da_key = layer.relation_module_keys[("defect", "da", "atom")]

        self.assertNotEqual(ad_key, da_key)
        self.assertIsNot(layer.megnets[ad_key], layer.megnets[da_key])
        self.assertEqual(len(layer.megnets), 4)
        self.assertEqual(
            layer.incoming_edge_types["atom"],
            [
                ("atom", "aa", "atom"),
                ("defect", "da", "atom"),
            ],
        )
        self.assertEqual(
            layer.incoming_edge_types["defect"],
            [
                ("atom", "ad", "defect"),
                ("defect", "dd", "defect"),
            ],
        )

    def test_relation_fusion_preserves_ordered_slots(self):
        fusion = RelationFusionUpdate(channels=2, num_relations=2)
        captured = []
        handle = fusion.ffn[0].register_forward_pre_hook(
            lambda _module, inputs: captured.append(inputs[0].detach().clone())
        )
        root = torch.tensor([[1.0, 2.0]])
        first_relation = torch.tensor([[3.0, 4.0]])
        second_relation = torch.tensor([[5.0, 6.0]])

        fusion(root, [first_relation, second_relation])
        handle.remove()

        self.assertEqual(len(captured), 1)
        self.assertTrue(torch.equal(
            captured[0],
            torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]),
        ))

    def test_empty_dd_relation_does_not_train_its_module(self):
        layer = self.make_layer()
        x_dict = {
            "atom": torch.randn(2, 92),
            "defect": torch.randn(1, 92),
        }
        edge_index_dict = {
            ("atom", "aa", "atom"): torch.tensor([[0, 1], [1, 0]]),
            ("defect", "dd", "defect"): torch.empty((2, 0), dtype=torch.long),
            ("atom", "ad", "defect"): torch.tensor([[0], [0]]),
            ("defect", "da", "atom"): torch.tensor([[0], [0]]),
        }
        edge_attr_dict = {
            edge_type: torch.randn(edge_index.size(1), 40)
            for edge_type, edge_index in edge_index_dict.items()
        }
        batch_dict = {
            "atom": torch.zeros(2, dtype=torch.long),
            "defect": torch.zeros(1, dtype=torch.long),
        }
        bond_batch_dict = {
            edge_type: torch.zeros(edge_index.size(1), dtype=torch.long)
            for edge_type, edge_index in edge_index_dict.items()
        }

        x_out, edge_out, state_out = layer(
            x_dict,
            edge_index_dict,
            edge_attr_dict,
            torch.zeros(1, 2),
            batch_dict,
            bond_batch_dict,
        )
        loss = (
            sum(value.sum() for value in x_out.values())
            + sum(value.sum() for value in edge_out.values())
            + state_out.sum()
        )
        loss.backward()

        dd_key = layer.relation_module_keys[("defect", "dd", "defect")]
        self.assertTrue(
            all(parameter.grad is None for parameter in layer.megnets[dd_key].parameters())
        )


if __name__ == "__main__":
    unittest.main()
