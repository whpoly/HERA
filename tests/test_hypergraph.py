import unittest
from types import SimpleNamespace

import numpy as np
import torch
from pymatgen.core import Lattice, Structure
from torch_geometric.loader import DataLoader

from HERA.config.defaults import get_config
from HERA.data.converters import SimpleCrystalConverter
from HERA.data.datasets import dataset_index_for_mode, representation_for_mode
from HERA.models.cgcnn import HyperCGCNN
from HERA.models.megnet import HyperMEGNet
from HERA.models.alignn import HyperALIGNN
from HERA.models.hypergraph import RegionHypergraphInteraction, RegionHypergraphNet
from HERA.main import apply_training_overrides, default_modes_for_model


class AtomicNumberConverter:
    def convert(self, structure):
        return np.asarray([[site.specie.Z] for site in structure], dtype=float)

    @staticmethod
    def get_shape():
        return 1


def make_periodic_three_region_structure():
    structure = Structure(
        Lattice.cubic(10.0),
        ["Si", "Si", "Si"],
        [
            [0.95, 0.0, 0.0],  # defect
            [0.05, 0.0, 0.0],  # 1 A away through the periodic boundary
            [0.50, 0.50, 0.50],  # far pristine
        ],
        site_properties={"type": [1, 0, 0]},
    )
    structure.state = [[0.0, 0.0]]
    structure.y = 0.0
    return structure


def make_two_independent_defect_structure():
    structure = Structure(
        Lattice.cubic(20.0),
        ["Si", "Si", "Si", "Si", "Si"],
        [
            [0.10, 0.50, 0.50],  # defect 0
            [0.20, 0.50, 0.50],  # pristine near defect 0 only
            [0.60, 0.50, 0.50],  # defect 1
            [0.70, 0.50, 0.50],  # pristine near defect 1 only
            [0.35, 0.50, 0.50],  # far from both defects
        ],
        site_properties={"type": [1, 0, 1, 0, 0]},
    )
    structure.state = [[0.0, 0.0]]
    structure.y = 0.0
    return structure


def make_overlapping_defect_neighborhood_structure():
    structure = Structure(
        Lattice.cubic(20.0),
        ["Si", "Si", "Si", "Si"],
        [
            [0.20, 0.50, 0.50],  # defect 0
            [0.30, 0.50, 0.50],  # pristine within 2 A of both defects
            [0.40, 0.50, 0.50],  # defect 1
            [0.70, 0.50, 0.50],  # far pristine
        ],
        site_properties={"type": [1, 0, 1, 0]},
    )
    structure.state = [[0.0, 0.0]]
    structure.y = 0.0
    return structure


class HypergraphConversionTests(unittest.TestCase):
    def setUp(self):
        self.converter = SimpleCrystalConverter(
            "hypergraph_hypergraph",
            atom_converter=AtomicNumberConverter(),
            hypergraph_radius=3.0,
        )

    def test_local_hyperedge_contains_its_center_defect(self):
        graph = self.converter.convert(make_periodic_three_region_structure())

        self.assertEqual(graph.region_type.tolist(), [0, 1, 2])
        self.assertEqual(
            graph.hyperedge_index.tolist(),
            [[0, 0, 1, 2], [0, 1, 1, 2]],
        )
        self.assertEqual(graph.hyperedge_type.tolist(), [0, 1, 2])
        self.assertGreater(graph.edge_index.size(1), 0)
        self.assertEqual(graph.edge_attr.size(0), graph.edge_index.size(1))
        self.assertEqual(graph.edge_vec.size(0), graph.edge_index.size(1))

    def test_multiple_defect_neighborhoods_remain_separate(self):
        graph = self.converter.convert(make_two_independent_defect_structure())

        self.assertEqual(graph.region_type.tolist(), [0, 1, 0, 1, 2])
        self.assertEqual(graph.hyperedge_type.tolist(), [0, 1, 0, 1, 2])
        self.assertEqual(
            graph.hyperedge_index.tolist(),
            [[0, 0, 1, 2, 2, 3, 4], [0, 1, 1, 2, 3, 3, 4]],
        )

    def test_overlap_is_allowed_only_through_shared_pristine_atoms(self):
        graph = self.converter.convert(
            make_overlapping_defect_neighborhood_structure()
        )
        memberships = {
            edge_id: set(graph.hyperedge_index[0, graph.hyperedge_index[1].eq(edge_id)].tolist())
            for edge_id in range(graph.hyperedge_type.numel())
        }

        self.assertEqual(graph.hyperedge_type.tolist(), [0, 1, 0, 1, 2])
        self.assertEqual(memberships[0], {0})
        self.assertEqual(memberships[1], {0, 1})
        self.assertEqual(memberships[2], {2})
        self.assertEqual(memberships[3], {1, 2})
        self.assertEqual(memberships[4], {3})

    def test_batching_offsets_by_each_graphs_variable_hyperedge_count(self):
        graphs = [
            self.converter.convert(make_two_independent_defect_structure()),
            self.converter.convert(make_periodic_three_region_structure()),
        ]
        batch = next(iter(DataLoader(graphs, batch_size=2, shuffle=False)))

        self.assertEqual(
            batch.hyperedge_index[1].tolist(),
            [0, 1, 1, 2, 3, 3, 4, 5, 6, 6, 7],
        )
        self.assertEqual(
            batch.hyperedge_type.tolist(),
            [0, 1, 0, 1, 2, 0, 1, 2],
        )

    def test_missing_defect_marker_is_rejected(self):
        structure = Structure(
            Lattice.cubic(10.0),
            ["Si"],
            [[0.0, 0.0, 0.0]],
            site_properties={"type": [0]},
        )
        with self.assertRaisesRegex(ValueError, "at least one defect atom"):
            self.converter.convert(structure)


class HypergraphModelTests(unittest.TestCase):
    def test_readout_pools_each_defect_neighborhood_before_type_reduction(self):
        converter = SimpleCrystalConverter(
            "hypergraph_hypergraph",
            atom_converter=AtomicNumberConverter(),
            hypergraph_radius=3.0,
        )
        graph = converter.convert(make_two_independent_defect_structure())
        interaction = RegionHypergraphInteraction(hidden_dim=1, n_steps=1)
        node_features = torch.tensor([[1.0], [3.0], [5.0], [7.0], [9.0]])
        batch = torch.zeros(5, dtype=torch.long)

        pooled = interaction.pool(
            node_features,
            graph.hyperedge_index,
            graph.hyperedge_type,
            batch,
            num_graphs=1,
            num_hyperedges=5,
        )

        # Core: mean(1, 5)=3; local: mean(mean(1,3), mean(5,7))=4;
        # far: 9. Local environments are reduced separately before combining.
        torch.testing.assert_close(pooled, torch.tensor([[3.0, 4.0, 9.0]]))

    def test_model_returns_one_prediction_per_batched_crystal(self):
        converter = SimpleCrystalConverter(
            "hypergraph_hypergraph",
            atom_converter=AtomicNumberConverter(),
            hypergraph_radius=3.0,
        )
        graphs = [
            converter.convert(make_periodic_three_region_structure()),
            converter.convert(make_periodic_three_region_structure()),
        ]
        batch = next(iter(DataLoader(graphs, batch_size=2, shuffle=False)))
        model = RegionHypergraphNet(
            node_input_shape=1,
            hidden_dim=8,
            n_blocks=2,
            heads=2,
            state_input_shape=2,
        )

        prediction = model(
            batch.x,
            batch.hyperedge_index,
            batch.batch,
            state=batch.state,
            hyperedge_type=batch.hyperedge_type,
            region_type=batch.region_type,
        )

        self.assertEqual(tuple(prediction.shape), (2, 1))
        self.assertTrue(torch.isfinite(prediction).all())

    def test_model_handles_an_omitted_empty_far_field(self):
        structure = Structure(
            Lattice.cubic(10.0),
            ["Si", "Si"],
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]],
            site_properties={"type": [1, 0]},
        )
        structure.y = 0.0
        converter = SimpleCrystalConverter(
            "hypergraph_hypergraph",
            atom_converter=AtomicNumberConverter(),
            hypergraph_radius=3.0,
        )
        batch = next(iter(DataLoader([converter.convert(structure)], batch_size=1)))
        self.assertNotIn(2, batch.region_type.tolist())
        self.assertNotIn(2, batch.hyperedge_type.tolist())

        model = RegionHypergraphNet(
            node_input_shape=1,
            hidden_dim=8,
            n_blocks=1,
            heads=2,
            state_input_shape=2,
        )
        prediction = model(
            batch.x,
            batch.hyperedge_index,
            batch.batch,
            state=batch.state,
            hyperedge_type=batch.hyperedge_type,
            region_type=batch.region_type,
        )
        self.assertEqual(tuple(prediction.shape), (1, 1))
        self.assertTrue(torch.isfinite(prediction).all())

    def test_hypergraph_config_and_dataset_mapping(self):
        for model_name in ("cgcnn", "megnet", "alignn", "hypergraph"):
            config = get_config(model_name, "native", "hypergraph")
            self.assertEqual(config["task"], f"{model_name}_hypergraph")
            self.assertEqual(config["model"]["hypergraph_radius"], 3.0)
            self.assertEqual(
                config["model"]["hypergraph_schema"],
                "per_defect_neighborhood_v2",
            )
        self.assertEqual(representation_for_mode("hypergraph"), "hetero")
        self.assertEqual(dataset_index_for_mode("hypergraph"), 1)
        self.assertEqual(representation_for_mode("hypergraph_was"), "hetero")
        self.assertEqual(dataset_index_for_mode("hypergraph_was"), 1)
        with self.assertRaisesRegex(ValueError, "only supports mode 'hypergraph'"):
            get_config("hypergraph", "native", "full")
        for model_name in ("cgcnn", "megnet", "alignn"):
            self.assertIn("hypergraph", default_modes_for_model(model_name))
            self.assertIn("hypergraph_was", default_modes_for_model(model_name))
            was_config = get_config(model_name, "native", "hypergraph_was")
            self.assertEqual(was_config["task"], f"{model_name}_hypergraph_was")
            self.assertEqual(was_config["model"]["atom_features"], "was_species")
            self.assertEqual(was_config["model"]["hypergraph_radius"], 3.0)
        with self.assertRaisesRegex(ValueError, "only supports mode 'hypergraph'"):
            get_config("hypergraph", "native", "hypergraph_was")
        overridden = apply_training_overrides(
            get_config("cgcnn", "native", "hypergraph_was"),
            SimpleNamespace(
                train_batch_size=None,
                test_batch_size=None,
                early_stopping_patience=None,
                early_stopping_min_delta_percent=None,
                hypergraph_radius=4.5,
            ),
            "cgcnn",
        )
        self.assertEqual(overridden["model"]["hypergraph_radius"], 4.5)

    def test_all_physical_backbones_forward_as_hybrid_hypergraphs(self):
        converter = SimpleCrystalConverter(
            "cgcnn_hypergraph",
            atom_converter=AtomicNumberConverter(),
            hypergraph_radius=3.0,
        )
        graphs = [
            converter.convert(make_two_independent_defect_structure()),
            converter.convert(make_periodic_three_region_structure()),
        ]
        batch = next(iter(DataLoader(graphs, batch_size=2, shuffle=False)))
        shared_hypergraph_kwargs = {
            "hyperedge_type": batch.hyperedge_type,
            "region_type": batch.region_type,
        }

        cgcnn = HyperCGCNN(
            orig_atom_fea_len=1,
            nbr_fea_len=1,
            atom_fea_len=8,
            n_conv=2,
            h_fea_len=16,
            n_heads=2,
        )
        cgcnn_prediction = cgcnn(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch,
            batch.hyperedge_index,
            **shared_hypergraph_kwargs,
        )

        megnet = HyperMEGNet(
            edge_input_shape=1,
            node_input_shape=1,
            state_input_shape=2,
            embedding_size=8,
            n_blocks=2,
            n_heads=2,
        )
        megnet_prediction = megnet(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.state,
            batch.batch,
            batch.bond_batch,
            batch.hyperedge_index,
            **shared_hypergraph_kwargs,
        )

        alignn = HyperALIGNN(
            node_input_shape=1,
            edge_input_shape=1,
            hidden_dim=8,
            n_blocks=1,
            gcn_blocks=1,
            angle_embed_size=4,
            n_heads=2,
            cutoff=6.0,
        )
        alignn_prediction = alignn(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch,
            batch.hyperedge_index,
            edge_vec=batch.edge_vec,
            **shared_hypergraph_kwargs,
        )

        for model, prediction, physical_prefix in (
                (cgcnn, cgcnn_prediction, "convs"),
                (megnet, megnet_prediction, "m1"),
                (alignn, alignn_prediction, "layers"),
        ):
            self.assertEqual(tuple(prediction.shape), (2, 1))
            self.assertTrue(torch.isfinite(prediction).all())
            prediction.sum().backward()
            self.assertTrue(any(
                name.startswith(physical_prefix) and parameter.grad is not None
                for name, parameter in model.named_parameters()
            ))
            self.assertTrue(any(
                name.startswith("hypergraph.blocks") and parameter.grad is not None
                for name, parameter in model.named_parameters()
            ))


if __name__ == "__main__":
    unittest.main()
