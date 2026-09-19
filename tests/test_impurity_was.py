"""True WAS for imp2d/semi, dataset integrity, and versioned output routing."""

from contextlib import redirect_stdout
import copy
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from pymatgen.core import Lattice, Structure
import torch
from torch_geometric.data import Batch

from HERA import main as cli
from HERA.config.defaults import get_config
from HERA.data.converters import AtomFeaturesExtractor
from HERA.data.datasets import init_elem_embedding, load_dataset, _read_impurity_manifest
from HERA.data.impurity_was import semi_reference_structure, imp2d_reference_structure
from HERA.data.structure_utils import convert_to_sparse_semi, convert_to_sparse_imp2d
from HERA.scripts import run_hetero_relation_benchmark as runner
from HERA.training.trainer import MEGNetTrainer


ROOT = Path(__file__).resolve().parents[1]


def make_structure(species, source_id):
    structure = Structure(Lattice.cubic(8), species,
                          [[.3, .5, .5], [.5, .3, .5], [.5, .5, .7], [.5, .5, .5], [.7, .5, .5]][:len(species)])
    structure.source_id = source_id
    return structure


def host_structure():
    return Structure(Lattice.cubic(4), ['Zn', 'O'], [[0, 0, 0], [.25, .25, .25]])


class ImpurityWasTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        init_elem_embedding(ROOT / 'atom_init.json')

    def test_semi_recovers_substitution_species_and_empty_interstitial_site(self):
        for tag, species, was in [('M_A', ['Zn', 'O', 'O', 'Al'], 30),
                                  ('M_B', ['Zn', 'Zn', 'O', 'Al'], 8),
                                  ('M_i_A', ['Zn', 'Zn', 'O', 'O', 'Al'], 0)]:
            with self.subTest(tag=tag):
                raw = make_structure(species, f'1-ZnO-{tag}-Al-POSCAR1.cif')
                structure = semi_reference_structure(raw, host_structure())
                self.assertEqual(len(structure), len(raw))
                self.assertEqual(structure[-1].properties['was'], was)
                np.testing.assert_array_equal(structure.frac_coords, raw.frac_coords)
                permuted = Structure.from_sites(list(reversed(raw.sites)))
                permuted.source_id = raw.source_id
                second = semi_reference_structure(permuted, host_structure())
                self.assertEqual(second[0].properties['was'], was)
                self.assertFalse(any('was' in s.properties for s in raw))

    def test_imp2d_added_atoms_have_zero_was_including_self_impurities(self):
        for impurity in ('Al', 'W'):
            for site in ('ads0', 'int2'):
                raw = make_structure(['W', 'S', 'S', impurity], f'S2W_{impurity}_{site}')
                info = dict(base='S2W', impurity=impurity, site=site,
                            is_self=impurity == 'W', defect_index=3)
                structure = convert_to_sparse_imp2d(raw, info, 1, 'alignn_hetero', None,
                    skip_was=True, local_cutoff=3, imp2d_preprocessing='reference_v1')
                self.assertEqual(structure[-1].properties['was'], 0)
                self.assertEqual(sum(s.properties['pool_type'] for s in structure), 1)
                self.assertGreater(sum(s.properties['type'] for s in structure), 1)
                self.assertEqual(structure[0].properties['was'], 74)

    def test_wrong_species_stoichiometry_and_unknown_site_fail_closed(self):
        raw = make_structure(['Zn', 'Zn', 'O', 'Al'], '1-ZnO-M_i_A-Al-POSCAR1.cif')
        with self.assertRaisesRegex(ValueError, 'stoichiometry'):
            semi_reference_structure(raw, host_structure())
        raw.source_id = '1-ZnO-M_B-Ga-POSCAR1.cif'
        with self.assertRaisesRegex(ValueError, 'extrinsic Ga'):
            semi_reference_structure(raw, host_structure())
        imp = make_structure(['W', 'S', 'S', 'Al'], 'S2W_Al_sub0')
        info = dict(base='S2W', impurity='Al', site='sub0')
        with self.assertRaisesRegex(ValueError, 'ads/int'):
            imp2d_reference_structure(imp, info, {3})
        info['site'] = 'ads0'
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            imp2d_reference_structure(imp, info, {2, 3})

    def test_legacy_checkpoint_cannot_silently_consume_new_was(self):
        raw = make_structure(['Zn', 'O', 'O', 'Al'], '1-ZnO-M_A-Al-POSCAR1.cif')
        old = convert_to_sparse_semi(raw, host_structure(), 1, 'alignn_hetero', None, True)
        new = convert_to_sparse_semi(raw, host_structure(), 1, 'alignn_hetero', None, True,
                                     semi_preprocessing='reference_v1')
        old_extractor = AtomFeaturesExtractor('was_species', 'alignn_hetero_was')
        old_features = old_extractor.convert(old)
        np.testing.assert_array_equal(old_features[:, :92], old_features[:, 92:])
        with self.assertRaisesRegex(ValueError, 'preprocessing differs'):
            old_extractor.convert(new)
        extractor = AtomFeaturesExtractor('was_species', 'alignn_hetero_was',
                                          impurity_preprocessing='semi_reference_v1')
        with self.assertRaisesRegex(ValueError, 'requires a WAS label'):
            extractor.convert(old)
        from HERA.data.datasets import elem_embedding
        np.testing.assert_array_equal(extractor.convert(new)[-1, 92:], elem_embedding[30])

    def test_bad_targets_and_missing_semi_structures_do_not_silently_shrink_data(self):
        for target in (float('nan'), float('inf'), 'bad'):
            with patch('HERA.data.datasets.pd.read_csv', return_value=pd.DataFrame([['sample', target]])):
                with self.assertRaisesRegex(ValueError, 'non-finite'):
                    _read_impurity_manifest('unused.csv', 'semi')
        with patch('HERA.data.datasets.pd.read_csv', side_effect=pd.errors.EmptyDataError):
            with self.assertRaisesRegex(ValueError, 'missing or empty'):
                load_dataset('semi', 'alignn', representations=['hetero'])
        with patch('HERA.data.datasets.pd.read_csv', return_value=pd.DataFrame([['1-GeSn-M_A-Al-POSCAR1.cif', 1.]])), \
                patch('HERA.data.datasets.Structure.from_file', side_effect=FileNotFoundError('host missing')):
            with self.assertRaisesRegex(ValueError, 'no sample was silently skipped'):
                load_dataset('semi', 'alignn', representations=['hetero'])

    def test_full_was_imp2d_loader_also_resolves_self_defect_labels(self):
        raw = make_structure(['W', 'S', 'S', 'W'], 'S2W_W_ads0')
        for idx, site in enumerate(raw):
            site.label = f'{site.specie.symbol}{idx}'
        with patch('HERA.data.datasets.pd.read_csv', return_value=pd.DataFrame([[raw.source_id, 1.]])), \
                patch('HERA.data.datasets.Structure.from_file', return_value=raw), \
                patch('HERA.data.datasets._load_imp2d_self_defect_labels', return_value={raw.source_id: 'W3'}):
            full, _, _, _, _ = load_dataset('imp2d', 'alignn', representations=['full_x'],
                                            imp2d_preprocessing='reference_v1')
        self.assertEqual(full[0][3].properties['was'], 0)
        self.assertEqual(full[0][0].properties['was'], 74)

    def test_paired_graphs_train_and_restore_for_both_datasets(self):
        for dataset in ('semi', 'imp2d'):
            for base_mode in ('hetero', 'attention'):
                if dataset == 'semi':
                    raw = make_structure(['Zn', 'O', 'O', 'Al'], '1-ZnO-M_A-Al-POSCAR1.cif')
                    structure = convert_to_sparse_semi(raw, host_structure(), 1, f'alignn_{base_mode}',
                        None, True, local_cutoff=0, semi_preprocessing='reference_v1')
                else:
                    raw = make_structure(['W', 'S', 'S', 'Al'], 'S2W_Al_ads0')
                    info = dict(base='S2W', impurity='Al', site='ads0', is_self=False)
                    structure = convert_to_sparse_imp2d(raw, info, 1, f'alignn_{base_mode}',
                        None, True, local_cutoff=0, imp2d_preprocessing='reference_v1')
                reference = None
                for mode in (base_mode, base_mode + '_was'):
                    config = get_config('alignn', dataset, mode)
                    config[f'{dataset}_preprocessing'] = 'reference_v1'
                    config['model'].update(embedding_size=8, nblocks=1, gcn_blocks=1,
                                           edge_embed_size=4, angle_embed_size=4, local_radius=0)
                    trainer = MEGNetTrainer(config, 'cpu', seed=123)
                    graph = trainer.converter.convert(structure)
                    if reference is not None:
                        pairs = [(graph[t], reference[t]) for t in graph.edge_types] if base_mode == 'hetero' else [(graph, reference)]
                        for current, old in pairs:
                            for field in ('edge_index', 'edge_attr', 'edge_vec'):
                                torch.testing.assert_close(current[field], old[field], rtol=0, atol=0)
                    reference = graph
                    batch = Batch.from_data_list([graph, graph])
                    trainer.model.eval()
                    prediction = trainer._forward(batch)
                    prediction.square().sum().backward()
                    self.assertTrue(torch.isfinite(prediction).all())
                    self.assertTrue(all(torch.isfinite(p.grad).all() for p in trainer.model.parameters() if p.grad is not None))
                    restored = MEGNetTrainer(copy.deepcopy(config), 'cpu', seed=123)
                    restored.model.load_state_dict(trainer.model.state_dict(), strict=True)
                    restored.model.eval()
                    torch.testing.assert_close(prediction, restored._forward(batch))

    def test_runner_routes_versioned_pairs_and_legacy_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(dataset=['semi', 'imp2d'], mode=['hetero', 'hetero_was'],
                seed=['123'], epochs=1, device='cpu', batch_size=8, cv5=False,
                run_dir=Path(directory), semi_preprocessing='reference_v1', imp2d_preprocessing='reference_v1')
            for version in ('reference_v1', 'legacy'):
                args.semi_preprocessing = args.imp2d_preprocessing = version
                command = runner.training_command(args, 'baseline')
                with patch('sys.argv', ['HERA.main', *command[3:]]), redirect_stdout(io.StringIO()), \
                        patch.object(cli, 'load_dataset', return_value=(None, [], [], None, [])) as loader, \
                        patch.object(cli, 'train_single_mode', return_value=[.12]) as train:
                    cli.main()
                self.assertEqual(train.call_count, 4)
                self.assertEqual(loader.call_count, 2)
                for call in train.call_args_list:
                    dataset = call.kwargs['dataset_name']
                    if version == 'reference_v1':
                        self.assertEqual(call.args[1][f'{dataset}_preprocessing'], version)
                        self.assertIn(f'{dataset}_{version}', call.kwargs['log_dir'])
                    else:
                        self.assertNotIn(f'{dataset}_preprocessing', call.args[1])
                        self.assertNotIn(f'{dataset}_reference_v1', call.kwargs['log_dir'])


if __name__ == '__main__':
    unittest.main()
