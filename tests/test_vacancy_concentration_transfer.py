import csv
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import ANY, call, patch

from HERA.config.defaults import VALID_DATASETS, get_config
from HERA.data.datasets import (
    is_pure_vacancy_descriptor,
    load_data_2dmd_mos2,
    load_data_2dmd_wse2,
    load_data_vacancy_mos2,
    load_data_vacancy_wse2,
)
from HERA.main import (
    CONCENTRATION_TRANSFER_DATASETS,
    MEGNET_SPARSE_DATASETS,
    iter_concentration_transfer_splits,
    write_test_predictions,
)


class VacancyConcentrationTransferTests(unittest.TestCase):
    def test_only_descriptors_made_entirely_of_vacancies_are_selected(self):
        self.assertTrue(is_pure_vacancy_descriptor(
            "[{'type': 'vacancy', 'element': 'S'}]"
        ))
        self.assertTrue(is_pure_vacancy_descriptor([
            {'type': 'vacancy', 'element': 'Mo'},
            {'type': 'vacancy', 'element': 'S'},
        ]))
        self.assertFalse(is_pure_vacancy_descriptor([]))
        self.assertFalse(is_pure_vacancy_descriptor([
            {'type': 'vacancy', 'element': 'S'},
            {'type': 'substitution', 'from': 'S', 'to': 'Se'},
        ]))

    def test_new_datasets_are_registered_without_replacing_existing_ones(self):
        for dataset in ('vacancy', '2dmd_low', '2dmd_high'):
            self.assertIn(dataset, VALID_DATASETS)
        for dataset in (
            'vacancy_mos2',
            'vacancy_wse2',
            '2dmd_mos2',
            '2dmd_wse2',
        ):
            self.assertIn(dataset, VALID_DATASETS)
            self.assertIn(dataset, CONCENTRATION_TRANSFER_DATASETS)
            self.assertIn(dataset, MEGNET_SPARSE_DATASETS)
            self.assertEqual(
                get_config('cgcnn', dataset, 'full')['model']['train_batch_size'],
                8,
            )

    @patch('HERA.data.datasets.convert_to_sparse_2dmd_high')
    @patch('HERA.data.datasets.CifParser')
    @patch('HERA.data.datasets.get_prepared')
    def test_material_loaders_pair_low_and_matching_high_only(
        self,
        get_prepared,
        cif_parser,
        convert,
    ):
        def add_sample(path, prepared, **kwargs):
            material = kwargs['material']
            concentration = kwargs['concentration']
            structure = SimpleNamespace(
                concentration=concentration,
                material=material,
                defect_family='vacancy',
            )
            prepared['id'].append(f'{material}-{concentration}')
            prepared['structure'].append(structure)
            prepared['base'].append(
                f'{material}_500' if kwargs.get('is_high') else material
            )
            prepared['cell'].append('[8, 8, 1]')
            prepared['target'].append(2.0 if concentration == 'high' else 1.0)
            prepared['weight'].append(1.0)

        get_prepared.side_effect = add_sample
        cif_parser.return_value.get_structures.return_value = ['unit-cell']
        convert.side_effect = lambda structure, *_args, **_kwargs: structure
        root = 'dataset/2d-materials-point-defects-all'

        for material, loader, pure_vacancy in (
            ('MoS2', load_data_vacancy_mos2, True),
            ('WSe2', load_data_vacancy_wse2, True),
            ('MoS2', load_data_2dmd_mos2, False),
            ('WSe2', load_data_2dmd_wse2, False),
        ):
            with self.subTest(material=material, pure_vacancy=pure_vacancy):
                get_prepared.reset_mock()
                full, hetero, attention, sparse, targets = loader(
                    'cgcnn',
                    representations=['full'],
                )
                self.assertEqual(
                    get_prepared.call_args_list,
                    [
                        call(
                            f'{root}/low_density_defects/{material}',
                            ANY,
                            pure_vacancy=pure_vacancy,
                            concentration='low',
                            material=material,
                        ),
                        call(
                            f'{root}/high_density_defects/{material}_500',
                            ANY,
                            is_high=True,
                            pure_vacancy=pure_vacancy,
                            concentration='high',
                            material=material,
                        ),
                    ],
                )
                self.assertEqual(
                    [structure.concentration for structure in full],
                    ['low', 'high'],
                )
                self.assertEqual(targets.tolist(), [1.0, 2.0])
                self.assertIsNone(hetero)
                self.assertIsNone(attention)
                self.assertIsNone(sparse)

    def test_high_concentration_samples_are_fixed_test_only(self):
        low = [SimpleNamespace(concentration='low', sample_id=i) for i in range(10)]
        high = [SimpleNamespace(concentration='high', sample_id=10 + i) for i in range(3)]
        data = low + high
        targets = list(range(len(data)))

        splits = list(iter_concentration_transfer_splits(
            data,
            targets,
            random_seeds=[123, 42],
        ))

        self.assertEqual(len(splits), 2)
        for split in splits:
            self.assertEqual(
                [sample.sample_id for sample in split['test_X']],
                [10, 11, 12],
            )
            self.assertEqual(split['test_y'], [10, 11, 12])
            self.assertTrue(all(
                sample.concentration == 'low'
                for sample in split['train_X'] + split['val_X']
            ))
            self.assertEqual(len(split['train_X']), 8)
            self.assertEqual(len(split['val_X']), 2)
            self.assertEqual(
                sorted(sample.sample_id for sample in split['train_X'] + split['val_X']),
                list(range(10)),
            )

    def test_missing_concentration_metadata_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'tagged'):
            list(iter_concentration_transfer_splits(
                [SimpleNamespace(), SimpleNamespace(concentration='high')],
                [1.0, 2.0],
                random_seeds=[123],
            ))

    def test_fixed_test_predictions_are_written_per_structure(self):
        structures = [SimpleNamespace(
            source_id='high-1',
            source_name='high-1',
            source_path='high/high-1.cif',
            material='MoS2',
            concentration='high',
            defect_family='vacancy',
        )]
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = write_test_predictions(
                tmp_dir,
                123,
                structures,
                [2.5],
                [2.75],
            )
            with open(path, newline='') as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['source_id'], 'high-1')
        self.assertEqual(rows[0]['material'], 'MoS2')
        self.assertEqual(rows[0]['concentration'], 'high')
        self.assertEqual(float(rows[0]['target']), 2.5)
        self.assertEqual(float(rows[0]['prediction']), 2.75)
        self.assertEqual(float(rows[0]['absolute_error']), 0.25)


if __name__ == '__main__':
    unittest.main()
