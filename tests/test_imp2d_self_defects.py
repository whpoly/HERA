import unittest

import numpy as np
from pymatgen.core import Lattice, Structure

from HERA.data.datasets import _assign_imp2d_self_defect_indices
from HERA.data.structure_utils import (
    get_imp2d_defect_indices,
    resolve_imp2d_self_defect_index,
)


def make_structure(species, frac_coords):
    return Structure(
        Lattice.cubic(10),
        species,
        frac_coords,
        coords_are_cartesian=False,
    )


class Imp2dSelfDefectTests(unittest.TestCase):
    def test_self_defect_is_matched_periodically_to_same_site_references(self):
        structure = make_structure(
            ["S", "S", "S", "W"],
            [
                [0.95, 0.20, 0.70],
                [0.20, 0.20, 0.40],
                [0.70, 0.70, 0.60],
                [0.50, 0.50, 0.50],
            ],
        )
        references = np.array([
            [0.02, 0.21, 0.69],
            [0.98, 0.19, 0.71],
            [0.01, 0.20, 0.70],
        ])

        self.assertEqual(
            resolve_imp2d_self_defect_index(structure, "S", references),
            0,
        )

    def test_explicit_self_defect_index_must_select_the_impurity_species(self):
        structure = make_structure(
            ["Nb", "S", "S"],
            [[0.1, 0.1, 0.5], [0.3, 0.3, 0.4], [0.6, 0.6, 0.6]],
        )
        defect_info = {
            "impurity": "Nb",
            "is_self": True,
            "defect_index": 2,
        }

        with self.assertRaisesRegex(ValueError, "selects S, expected Nb"):
            get_imp2d_defect_indices(structure, defect_info)

    def test_nonself_defect_selection_is_unchanged(self):
        structure = make_structure(
            ["S", "Cl", "Sn"],
            [[0.1, 0.1, 0.4], [0.2, 0.2, 0.5], [0.3, 0.3, 0.6]],
        )

        self.assertEqual(
            get_imp2d_defect_indices(
                structure,
                {"impurity": "Cl", "is_self": False},
            ),
            {1},
        )

    def test_dataset_preparation_assigns_self_index_from_lookup_label(self):
        self_defect = Structure(
            Lattice.cubic(10),
            ["S", "S", "W"],
            [[0.96, 0.2, 0.7], [0.2, 0.2, 0.4], [0.5, 0.5, 0.5]],
            labels=["S_defect", "S_host", "W_host"],
        )
        self_defect.source_id = "S2W_S_ads2"
        prep = [
            [
                self_defect,
                {
                    "base": "S2W",
                    "impurity": "S",
                    "site": "ads2",
                    "is_self": True,
                },
            ],
        ]

        _assign_imp2d_self_defect_indices(
            prep,
            {"S2W_S_ads2": "S_defect"},
        )

        self.assertEqual(prep[0][1]["defect_index"], 0)
        self.assertEqual(
            get_imp2d_defect_indices(self_defect, prep[0][1]),
            {0},
        )


if __name__ == "__main__":
    unittest.main()
