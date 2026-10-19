import unittest
from types import SimpleNamespace

import numpy as np
from ase import Atoms

from HERA.data.ybco import (ELEMENTS, expected_delta, formation_energy,
                             grouped_indices, parse_family, reference_mapping)
from HERA.scripts.run_ybco_formation import make_splitter, select_records


class YBCOTests(unittest.TestCase):
    def host(self):
        return Atoms('YBaCuO', positions=[[1, 1, 1], [4, 1, 1], [1, 4, 1], [4, 4, 1]],
                     cell=[10, 10, 10], pbc=True)

    def test_vacancy_and_interstitial_energy_signs(self):
        mu = dict(zip(ELEMENTS, (-10., -20., -30., -4.)))
        self.assertAlmostEqual(formation_energy(-94, -100, expected_delta('O1_vac'), mu), 2)
        self.assertAlmostEqual(formation_energy(-103, -100, expected_delta('O_int_2'), mu), 1)
        self.assertAlmostEqual(formation_energy(-89, -100, expected_delta('Y_Ba_antisite'), mu), 1)

    def test_reference_mapping_handles_pbc_and_atom_permutations(self):
        host = self.host()
        defect = host[[2, 0, 1]]
        defect.positions[0] += [10, 0, 0]
        result = reference_mapping(defect, host, 'O1_vac')
        self.assertEqual(result['vacancy_z'], [8])
        np.testing.assert_allclose(result['vacancy_frac'], [[.4, .4, .1]])
        self.assertEqual(result['was'], [29, 39, 56])
        self.assertFalse(any(result['defect']))

    def test_substitution_and_interstitial_original_species(self):
        host = self.host()
        substitution = host.copy()
        substitution[1].symbol = 'Y'
        result = reference_mapping(substitution, host, 'Y_Ba_antisite')
        self.assertEqual(result['was'][1], 56)
        self.assertEqual(result['defect'], [False, True, False, False])
        self.assertEqual(result['vacancy_z'], [])
        interstitial = host.copy()
        interstitial += Atoms('O', positions=[[7, 7, 7]])
        result = reference_mapping(interstitial, host, 'O_int_1')
        self.assertEqual(result['was'][-1], 0)
        self.assertEqual(result['defect'], [False] * 4 + [True])

    def test_wrong_composition_and_ambiguous_geometry_fail(self):
        host = self.host()
        with self.assertRaisesRegex(ValueError, 'composition'):
            reference_mapping(host.copy(), host, 'O1_vac')
        moved = host[:3]
        moved.positions[0] += [2, 0, 0]
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            reference_mapping(moved, host, 'O1_vac')

    def test_no_defect_family_leaks_across_splits(self):
        records = [{'source_id': f'{f}_{j}', 'family': f'f{f}'} for f in range(10) for j in range(3)]
        split = grouped_indices(records, 123)
        groups = {p: {records[i]['family'] for i in ids} for p, ids in split.items()}
        self.assertEqual([len(groups[p]) for p in ('train', 'val', 'test')], [6, 2, 2])
        self.assertFalse(groups['train'] & groups['val'] or groups['train'] & groups['test'] or groups['val'] & groups['test'])
        reordered = records[::-1]
        data = [SimpleNamespace(source_id=r['source_id']) for r in reordered]
        result = next(make_splitter(records)(data, list(range(30)), [123]))
        self.assertEqual({s.source_id for s in result['test_X']}, {records[i]['source_id'] for i in split['test']})

    def test_cohort_does_not_treat_oxygen6_or_bulk_as_single_defect(self):
        self.assertIsNone(parse_family('Ba2YCu3O6_vol0_strain0_jiggle0'))
        self.assertIsNone(parse_family('Ba2YCu3O7_vol0_strain0_jiggle0'))
        self.assertEqual(parse_family('O3_vac_vol-5_strain4_jiggle1'), 'O3_vac')

    def test_smoke_selects_geometry_without_looking_at_energy(self):
        records = [{'family': 'O1_vac', 'label': f'O1_vac_vol0_strain0_jiggle{j}',
                    'source_frame': 5-j, 'target_ev': 100-j} for j in (2, 1, 0)]
        self.assertEqual(select_records(records, smoke=True)[0]['label'], 'O1_vac_vol0_strain0_jiggle0')


if __name__ == '__main__':
    unittest.main()
