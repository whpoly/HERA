import unittest

from pymatgen.core import Lattice, Structure

from HERA.data.structure_utils import (
    get_hetero_och,
    get_och_adsorbate_index,
    get_sparse_och,
    site_type_flag,
)


def make_hydrogen_host():
    return Structure(
        Lattice.cubic(10),
        ["H", "Pd", "H", "Pd", "H"],
        [
            [0.10, 0.10, 0.10],
            [0.25, 0.25, 0.25],
            [0.40, 0.40, 0.40],
            [0.60, 0.60, 0.60],
            [0.85, 0.85, 0.85],
        ],
        labels=["H1", "Pd1", "H2", "Pd2", "H3"],
    )


class OchAdsorbateTests(unittest.TestCase):
    def test_last_hydrogen_is_the_adsorbate(self):
        self.assertEqual(get_och_adsorbate_index(make_hydrogen_host(), "H"), 4)

    def test_hetero_marks_only_the_adsorbed_hydrogen(self):
        hetero = get_hetero_och(make_hydrogen_host(), "H", 1, None)

        defect_indices = [
            idx for idx, site in enumerate(hetero)
            if site_type_flag(site)
        ]
        self.assertEqual(defect_indices, [4])

    def test_sparse_keeps_only_the_adsorbed_hydrogen(self):
        sparse = get_sparse_och(make_hydrogen_host(), "H", 1)

        self.assertEqual(len(sparse), 1)
        self.assertEqual(sparse[0].label, "H3")

    def test_missing_adsorbate_fails_clearly(self):
        structure = Structure(
            Lattice.cubic(10),
            ["Pd"],
            [[0.5, 0.5, 0.5]],
        )

        with self.assertRaisesRegex(ValueError, "has no H adsorbate candidate"):
            get_och_adsorbate_index(structure, "H")


if __name__ == "__main__":
    unittest.main()
