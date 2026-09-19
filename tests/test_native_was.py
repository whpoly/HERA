"""Native WAS labels, corrected defect identities and legacy result isolation."""

import copy
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from pymatgen.core import Lattice, Structure
from pymatgen.core.periodic_table import DummySpecies
import torch
from torch_geometric.data import Batch

from HERA import main as cli
from HERA.config.defaults import get_config
from HERA.data.converters import AtomFeaturesExtractor
from HERA.data.datasets import init_elem_embedding, load_dataset
from HERA.data.native_was import parse_native_defect
from HERA.data.structure_utils import convert_to_sparse_native
from HERA.training.history import TrainingLogger
from HERA.training.trainer import MEGNetTrainer


ROOT = Path(__file__).resolve().parents[1]


def raw_structure(tag='Zn_O'):
    # The original final Zn3 has been sorted ahead of the two O atoms by pymatgen.
    structure = Structure(Lattice.cubic(8), ['Zn', 'Zn', 'O', 'O'],
                          [[.2, .5, .5], [.5, .5, .5], [.5, .3, .5], [.5, .5, .8]],
                          labels=['Zn0', 'Zn3', 'O1', 'O2'])
    structure.source_id = f'4-ZnO-{tag}-POSCAR0-Neutral.cif'
    return structure


def converted(raw, mode='hetero', radius=0, version='reference_v1'):
    kind = 'vacancy' if '-V_' in raw.source_id else 'others'
    return convert_to_sparse_native(raw, kind, 1, f'alignn_{mode}', None,
                                    skip_was=True, local_cutoff=radius,
                                    native_preprocessing=version)


class NativeWasTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        init_elem_embedding(ROOT / 'atom_init.json')

    def test_true_labels_and_features_for_each_defect_kind(self):
        from HERA.data.datasets import elem_embedding
        for tag, previous_z, defect_index in [('V_Zn', 30, 4), ('Zn_O', 8, 1), ('Zn_i_A', 0, 1)]:
            with self.subTest(tag=tag):
                raw = raw_structure(tag)
                structure = converted(raw)
                self.assertEqual(structure.source_id, raw.source_id)
                self.assertEqual(len(structure), len(raw) + int(tag.startswith('V_')))
                self.assertEqual([i for i, s in enumerate(structure) if s.properties['pool_type']], [defect_index])
                self.assertEqual(structure[defect_index].properties['was'], previous_z)
                np.testing.assert_array_equal(structure.frac_coords[:len(raw)], raw.frac_coords)
                self.assertEqual(structure.species[:len(raw)], raw.species)
                extractor = AtomFeaturesExtractor('was_species', 'alignn_hetero_was', 'reference_v1')
                features = extractor.convert(structure)
                self.assertEqual(features.shape, (len(structure), 184))
                expected = elem_embedding[previous_z] if previous_z else [0] * 92
                np.testing.assert_array_equal(features[defect_index, 92:], expected)
                if tag.startswith('V_'):
                    self.assertIsInstance(structure[-1].specie, DummySpecies)
                    self.assertFalse(features[-1, :92].any())
                for idx in range(len(structure)):
                    if idx != defect_index:
                        np.testing.assert_array_equal(features[idx, :92], features[idx, 92:])
                self.assertTrue(all('was' not in s.properties for s in raw))

    def test_site_order_is_irrelevant_and_region_expansion_keeps_reference_identity(self):
        raw = raw_structure('Zn_i_B')
        first = converted(raw, radius=3)
        permuted = Structure.from_sites([raw[i] for i in [3, 2, 0, 1]])
        permuted.source_id = raw.source_id
        second = converted(permuted, radius=3)
        labels = lambda s: {site.label: (site.properties['was'], site.properties['pool_type']) for site in s}
        self.assertEqual(labels(first), labels(second))
        self.assertGreater(sum(s.properties['type'] for s in first), 1)
        self.assertEqual(sum(s.properties['pool_type'] for s in first), 1)
        self.assertEqual(sum(s.properties['was'] == 0 for s in first), 1)

    def test_missing_or_inconsistent_identity_fails_without_guessing(self):
        for source in ('fixture', '1-ZnO-Zz_O-POSCAR0-Neutral.cif'):
            with self.assertRaises(ValueError):
                parse_native_defect(source)
        for kind in ('missing_label', 'duplicate_index', 'wrong_species'):
            raw = raw_structure()
            if kind == 'missing_label':
                raw[0].label = 'Zn'
            elif kind == 'duplicate_index':
                raw[0].label = 'Zn3'
            else:
                raw.source_id = '4-ZnO-O_Zn-POSCAR0-Neutral.cif'
            with self.assertRaises(ValueError):
                converted(raw)
        self.assertEqual(parse_native_defect('198-GaSb-Ab_i_A-POSCAR0-Neutral.cif'), ('interstitial', 51, 0))
        with self.assertRaises(ValueError):
            parse_native_defect('1-ZnO-Ab_i_A-POSCAR0-Neutral.cif')

    def test_legacy_features_remain_duplicated_and_versions_cannot_be_mixed(self):
        raw = raw_structure()
        legacy = converted(raw, version=None)
        old_extractor = AtomFeaturesExtractor('was_species', 'alignn_hetero_was')
        features = old_extractor.convert(legacy)
        np.testing.assert_array_equal(features[:, :92], features[:, 92:])
        self.assertTrue(legacy[-1].properties['type'])
        with self.assertRaisesRegex(ValueError, 'preprocessing differs'):
            old_extractor.convert(converted(raw))
        new_extractor = AtomFeaturesExtractor('was_species', 'alignn_hetero_was', 'reference_v1')
        with self.assertRaisesRegex(ValueError, 'requires a WAS label'):
            new_extractor.convert(legacy)
        self.assertEqual(len(converted(raw_structure('V_Zn'), version=None)), 4)

    def test_full_and_full_x_preserve_vacancy_convention(self):
        raw = raw_structure('V_Zn')
        self.assertEqual(len(converted(raw, 'full')), len(raw))
        self.assertEqual(len(converted(raw, 'full_x')), len(raw) + 1)

    def test_actual_cif_roundtrip_and_unified_loader(self):
        raw = raw_structure('Zn_i_A')
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / raw.source_id
            raw.to(filename=path, fmt='cif')
            parsed = Structure.from_file(path)
            parsed.source_id = raw.source_id
            corrected = converted(parsed)
            defect = [s for s in corrected if s.properties['pool_type']][0]
            self.assertEqual(defect.label, 'Zn3')
            self.assertEqual(defect.properties['was'], 0)
        import pandas as pd
        with patch('HERA.data.datasets.pd.read_csv', return_value=pd.DataFrame([[raw.source_id, 1.2]])), \
                patch('HERA.data.datasets.Structure.from_file', return_value=raw):
            _, hetero, attention, _, target = load_dataset('native', 'alignn',
                representations=['hetero', 'attention'], native_preprocessing='reference_v1')
        self.assertAlmostEqual(target.item(), 1.2, places=6)
        self.assertEqual(hetero[0].site_properties['was'], attention[0].site_properties['was'])
        self.assertEqual(hetero[0].site_properties['type'], attention[0].site_properties['type'])

    def test_paired_graphs_and_forward_backward_checkpoint_restore(self):
        for base_mode in ('hetero', 'attention'):
            reference_graph = None
            for mode in (base_mode, base_mode + '_was'):
                config = get_config('alignn', 'native', mode)
                config['native_preprocessing'] = 'reference_v1'
                config['model'].update(embedding_size=8, nblocks=1, gcn_blocks=1,
                                       edge_embed_size=4, angle_embed_size=4, local_radius=0)
                trainer = MEGNetTrainer(config, 'cpu', seed=123)
                graph = trainer.converter.convert(converted(raw_structure(), base_mode))
                if reference_graph is not None:
                    if base_mode == 'hetero':
                        for t in graph.edge_types:
                            for field in ('edge_index', 'edge_attr', 'edge_vec'):
                                torch.testing.assert_close(graph[t][field], reference_graph[t][field], rtol=0, atol=0)
                    else:
                        for field in ('edge_index', 'edge_attr', 'edge_vec', 'node_type'):
                            torch.testing.assert_close(graph[field], reference_graph[field], rtol=0, atol=0)
                reference_graph = graph
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

    def test_cli_preserves_legacy_results_and_versions_new_was_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            command = ['HERA.main', '--model', 'alignn', '--dataset', 'native', '--mode',
                       'attention', 'attention_was', '--r', '0', '--device', 'cpu',
                       '--atom-init', str(ROOT / 'atom_init.json'), '--run-dir', temporary,
                       '--seed', '123', '--resume', '--protect-existing']
            def run(extra):
                with patch('sys.argv', command + extra), redirect_stdout(io.StringIO()), \
                        patch.object(cli, 'load_dataset', return_value=([], [], [], None, [])) as loader, \
                        patch.object(cli, 'train_single_mode', return_value=[.123]) as train:
                    cli.main()
                return loader, train
            _, old = run(['--native-preprocessing', 'legacy'])
            files = {}
            for call in old.call_args_list:
                folder = Path(call.kwargs['log_dir'])
                logger = TrainingLogger(folder, 'alignn', 'native', call.args[0], 123)
                logger.log_test_result(.123)
                files[Path(logger.filepath)] = Path(logger.filepath).read_bytes()
            loader, new = run([])
            self.assertEqual(new.call_count, 1)
            self.assertEqual(new.call_args.args[0], 'attention_was')
            self.assertIn('native_reference_v1', new.call_args.kwargs['log_dir'])
            self.assertEqual(loader.call_args.kwargs['native_preprocessing'], 'reference_v1')
            self.assertTrue(all(p.read_bytes() == content for p, content in files.items()))
            loader, paired = run(['--native-preprocessing', 'reference_v1', '--seed', '11'])
            self.assertEqual(paired.call_count, 2)
            self.assertEqual(loader.call_count, 1)
            for call in paired.call_args_list:
                self.assertEqual(call.args[1]['native_preprocessing'], 'reference_v1')
                self.assertIn('native_reference_v1', call.kwargs['log_dir'])


if __name__ == '__main__':
    unittest.main()
