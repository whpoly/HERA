import unittest

import numpy as np
import torch
from pymatgen.core import Lattice, Structure
from torch_geometric.loader import DataLoader

from HERA.config.defaults import get_config
from HERA.data import datasets
from HERA.data.converters import DummyConverter, SimpleCrystalConverter
from HERA.training.trainer import MEGNetTrainer


class _OneFeatureAtomConverter:
    def convert(self, structure):
        return np.ones((len(structure), 1), dtype=float)


class HeteroPhysicalEdgeTests(unittest.TestCase):
    @staticmethod
    def make_single_defect_structure():
        structure = Structure(
            Lattice.cubic(20),
            ["H", "Si", "Si"],
            [
                [10.0, 10.0, 10.0],
                [11.0, 10.0, 10.0],
                [10.0, 11.0, 10.0],
            ],
            coords_are_cartesian=True,
            site_properties={
                "type": [1, 0, 0],
                "pool_type": [1, 0, 0],
            },
        )
        structure.y = torch.tensor(0.0)
        return structure

    def test_all_backbones_receive_the_same_physical_edges(self):
        edge_counts = {}
        for model_name in ("cgcnn", "megnet", "alignn"):
            with self.subTest(model=model_name):
                graph = SimpleCrystalConverter(
                    task=f"{model_name}_hetero",
                    atom_converter=_OneFeatureAtomConverter(),
                    bond_converter=DummyConverter(),
                    cutoff=2.0,
                ).convert(self.make_single_defect_structure())

                counts = {
                    edge_type: int(edge_index.size(1))
                    for edge_type, edge_index in graph.edge_index_dict.items()
                }
                edge_counts[model_name] = counts
                self.assertEqual(
                    counts[("defect", "dd", "defect")],
                    0,
                )
                for edge_vec in graph.edge_vec_dict.values():
                    if edge_vec.numel() > 0:
                        self.assertTrue(torch.all(torch.linalg.vector_norm(edge_vec, dim=1) > 0))

        self.assertEqual(edge_counts["cgcnn"], edge_counts["megnet"])
        self.assertEqual(edge_counts["cgcnn"], edge_counts["alignn"])

    def test_cgcnn_ad_and_da_use_distinct_convolutions(self):
        config = get_config("cgcnn", "imp2d", "hetero")
        trainer = MEGNetTrainer(
            config,
            "cpu",
            seed=123,
        )
        conv = trainer.model.base_model.convs[0]
        ad_conv = conv["atom__ad__defect"]
        da_conv = conv["defect__da__atom"]

        self.assertEqual(config["model"]["embedding_size"], 32)
        self.assertEqual(trainer.model.atom_fea_len, 32)
        self.assertEqual(ad_conv.channels, 32)
        self.assertTrue(all(
            embedding.out_features == 32
            for embedding in trainer.model.base_model.embedding.values()
        ))
        self.assertEqual(
            trainer.model.base_model.incoming_edge_types["atom"],
            [
                ("atom", "aa", "atom"),
                ("defect", "da", "atom"),
            ],
        )
        self.assertEqual(
            trainer.model.base_model.incoming_edge_types["defect"],
            [
                ("atom", "ad", "defect"),
                ("defect", "dd", "defect"),
            ],
        )
        self.assertIsNot(ad_conv, da_conv)
        self.assertIsNot(
            next(ad_conv.parameters()),
            next(da_conv.parameters()),
        )

    def test_all_backbones_forward_with_an_empty_dd_relation(self):
        original_embedding = datasets.elem_embedding
        datasets.elem_embedding = {
            1: [0.0] * 92,
            14: [0.0] * 92,
        }
        try:
            for model_name in ("cgcnn", "megnet", "alignn"):
                with self.subTest(model=model_name):
                    trainer = MEGNetTrainer(
                        get_config(model_name, "imp2d", "hetero"),
                        "cpu",
                        seed=123,
                    )
                    graph = trainer.converter.convert(
                        self.make_single_defect_structure()
                    )
                    self.assertEqual(
                        graph[("defect", "dd", "defect")].edge_index.size(1),
                        0,
                    )

                    batch = next(iter(DataLoader([graph], batch_size=1)))
                    trainer.model.train()
                    prediction = trainer._forward(batch)
                    self.assertEqual(tuple(prediction.shape), (1,))
                    self.assertTrue(torch.isfinite(prediction).all())
                    prediction.sum().backward()
                    gradients = [
                        parameter.grad
                        for parameter in trainer.model.parameters()
                        if parameter.grad is not None
                    ]
                    self.assertTrue(gradients)
                    self.assertTrue(all(
                        torch.isfinite(gradient).all()
                        for gradient in gradients
                    ))
        finally:
            datasets.elem_embedding = original_embedding


if __name__ == "__main__":
    unittest.main()
