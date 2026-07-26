import unittest

from pymatgen.core import Lattice, Structure

from HERA.data.converters import SimpleCrystalConverter


class NeighborTypeCapTests(unittest.TestCase):
    @staticmethod
    def make_structure():
        return Structure(
            Lattice.cubic(20),
            ["Si"] * 7,
            [
                [10.0, 10.0, 10.0],
                [11.0, 10.0, 10.0],
                [12.0, 10.0, 10.0],
                [13.0, 10.0, 10.0],
                [10.0, 11.5, 10.0],
                [10.0, 12.5, 10.0],
                [10.0, 13.5, 10.0],
            ],
            coords_are_cartesian=True,
            site_properties={"type": [0, 0, 0, 0, 1, 1, 1]},
        )

    def test_neighbor_cap_is_applied_independently_per_target_type(self):
        converter = SimpleCrystalConverter(
            task="full",
            cutoff=4.0,
            max_neighbors=2,
        )

        center_neighbors = converter._neighbor_lists(self.make_structure())[0]
        neighbor_indices = [int(neighbor[2]) for neighbor in center_neighbors]
        neighbor_types = [
            int(neighbor.properties["type"])
            for neighbor in center_neighbors
        ]

        self.assertEqual(neighbor_indices, [1, 4, 2, 5])
        self.assertEqual(neighbor_types.count(0), 2)
        self.assertEqual(neighbor_types.count(1), 2)

    def test_single_type_structures_keep_the_original_total_cap(self):
        structure = self.make_structure()
        for site in structure:
            site.properties["type"] = 0
        converter = SimpleCrystalConverter(
            task="full",
            cutoff=4.0,
            max_neighbors=2,
        )

        center_neighbors = converter._neighbor_lists(structure)[0]

        self.assertEqual([int(neighbor[2]) for neighbor in center_neighbors], [1, 4])

    def test_no_neighbor_cap_still_keeps_every_neighbor_within_cutoff(self):
        converter = SimpleCrystalConverter(
            task="full",
            cutoff=4.0,
            max_neighbors=None,
        )

        center_neighbors = converter._neighbor_lists(self.make_structure())[0]

        self.assertEqual(len(center_neighbors), 6)


if __name__ == "__main__":
    unittest.main()
