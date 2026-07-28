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

    def test_cgcnn_forward_handles_an_empty_dd_relation(self):
        original_embedding = datasets.elem_embedding
        datasets.elem_embedding = {
            1: [0.0] * 92,
            14: [0.0] * 92,
        }
        try:
            trainer = MEGNetTrainer(
                get_config("cgcnn", "imp2d", "hetero"),
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
            trainer.model.eval()
            with torch.no_grad():
                prediction = trainer._forward(batch)
            self.assertEqual(tuple(prediction.shape), (1,))
            self.assertTrue(torch.isfinite(prediction).all())
        finally:
            datasets.elem_embedding = original_embedding


if __name__ == "__main__":
    unittest.main()
