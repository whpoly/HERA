import unittest

from HERA.models.alignn import HeteroALIGNN


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


if __name__ == "__main__":
    unittest.main()
