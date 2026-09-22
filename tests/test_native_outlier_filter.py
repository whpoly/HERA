"""Outlier preprocessing must not fit on held-out labels or reshuffle retained IDs."""
import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd
import numpy as np
from ase import Atoms
from pymatgen.core import Lattice, Structure

from HERA import main as cli
from HERA.data.datasets import load_dataset
from HERA.data.native_filter import (
    DEFAULT_POLICY, SCHEMA, fit_split, filtered_splits, manifest_identity,
    requested_splits, validate_manifest, verify_source_table,
)
from HERA.scripts.preprocess_native_outliers import inspect_source, write_manifest_tables
from HERA.data.native_quality import DEEP_POLICY, audit_family_geometry, geometry_features, geometry_exclusions, joint_distortion_exclusion
from HERA.training.results import validate_compact_run_config, ensure_compact_run_config


def fixture():
    values = [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 100, 1.2, 50, 1.1, 60]
    rows = [{'source_id': f'1-ZnO-Zn_i_A-POSCAR{i}-Neutral.cif', 'material': 'ZnO',
             'defect_type': 'interstitial', 'added_z': 30, 'removed_z': 0,
             'target': y, 'sha256': 'unused',
             'quality_reasons': ['short_interatomic_distance'] if i == 8 else []}
            for i, y in enumerate(values)]
    ids = [r['source_id'] for r in rows]
    original = {'train': ids[:9], 'val': ids[9:11], 'test': ids[11:]}
    return rows, original


def manifest_for(rows, original):
    split = {'display': 'seed=123', 'logger_id': 123, 'seed': 123, 'explain_id': 'seed_123',
             **fit_split(rows, original, DEFAULT_POLICY)}
    return validate_manifest({'schema': SCHEMA, 'dataset': 'native', 'cv5': False,
                              'policy': DEFAULT_POLICY, 'records': rows, 'splits': [split]})


class NativeOutlierTests(unittest.TestCase):
    def test_poscar_scope_only_restricts_exclusions_and_restores_later_outliers(self):
        rows, original = fixture()
        # Make POSCAR1 abnormal, while retaining the existing later outliers.
        rows[1]['quality_reasons'] = ['short_interatomic_distance']
        policy = {**DEFAULT_POLICY, 'deep_checks':DEEP_POLICY}
        unrestricted = fit_split(rows, original, policy)
        scoped = fit_split(rows, original, {**policy, 'filter_poscar_indices':[0,1]})
        self.assertEqual(scoped['train_fences'], unrestricted['train_fences'])
        self.assertEqual(scoped['family_train_fences'], unrestricted['family_train_fences'])
        self.assertEqual([r['source_id'] for r in scoped['excluded']['train']], [rows[1]['source_id']])
        self.assertIn(rows[0]['source_id'], scoped['kept']['train'])
        self.assertIn(rows[8]['source_id'], scoped['kept']['train'])
        self.assertEqual(scoped['kept']['val'], original['val'])
        self.assertEqual(scoped['kept']['test'], original['test'])

    def test_manifest_rejects_exclusion_outside_declared_poscar_scope(self):
        rows, original = fixture()
        manifest = manifest_for(rows, original)
        manifest['policy'] = {**DEFAULT_POLICY, 'filter_poscar_indices':[0,1]}
        with self.assertRaisesRegex(ValueError, 'outside its POSCAR filter scope'):
            validate_manifest(manifest)

    def test_holdout_labels_never_change_fitted_fences_or_training_selection(self):
        rows, original = fixture()
        first = fit_split(rows, original, DEFAULT_POLICY)
        changed = copy.deepcopy(rows)
        for index in (10, 12):
            changed[index]['target'] *= 10000
        changed[9]['target'] = 1.8
        changed[11]['target'] = 1.9
        second = fit_split(changed, original, DEFAULT_POLICY)
        self.assertEqual(first['train_fences'], second['train_fences'])
        self.assertEqual(first['kept']['train'], second['kept']['train'])
        self.assertNotIn(rows[8]['source_id'], first['kept']['train'])
        self.assertEqual(len(first['kept']['val']), 1)
        self.assertEqual(len(first['kept']['test']), 1)

    def test_no_filter_by_filename_and_nonzero_floor_for_near_constant_labels(self):
        rows, original = fixture()
        for row in rows[:8]:
            row['target'] = 1.0
        rows[9]['target'] = 1.5
        split = fit_split(rows, original, DEFAULT_POLICY)
        self.assertIn(rows[1]['source_id'], split['kept']['train'])  # POSCAR1 is retained.
        self.assertIn(rows[9]['source_id'], split['kept']['val'])
        self.assertEqual(split['train_fences']['ZnO/interstitial/30/0']['upper_ev'], 4.0)

    def test_insufficient_training_group_is_not_fitted_to_holdout(self):
        rows, original = fixture()
        rows[12]['material'] = 'GaN'
        split = fit_split(rows, original, DEFAULT_POLICY)
        self.assertIn(rows[12]['source_id'], split['kept']['test'])
        self.assertNotIn('GaN/interstitial/30/0', split['train_fences'])

    def test_different_defect_species_have_independent_energy_baselines(self):
        rows, original = fixture()
        # Oxygen interstitials can have a different baseline from Zn interstitials.
        oxygen = [dict(rows[i], source_id=f'2-ZnO-O_i_B-POSCAR{i}-Neutral.cif',
                       added_z=8, target=20.0+i/10) for i in range(8)]
        rows.extend(oxygen)
        original['train'].extend(r['source_id'] for r in oxygen)
        rows[10]['added_z'], rows[10]['target'] = 8, 20.5
        split = fit_split(rows, original, DEFAULT_POLICY)
        self.assertIn(rows[10]['source_id'], split['kept']['val'])
        self.assertLess(split['train_fences']['ZnO/interstitial/30/0']['upper_ev'], 10)
        self.assertGreater(split['train_fences']['ZnO/interstitial/8/0']['lower_ev'], 10)

    def test_original_split_membership_and_order_survive_loader_order_changes(self):
        rows, original = fixture()
        manifest = manifest_for(rows, original)
        data = [SimpleNamespace(source_id=r['source_id']) for r in reversed(rows)]
        targets = [r['target'] for r in reversed(rows)]
        split = next(filtered_splits(data, targets, manifest, [123]))
        for part in ('train', 'val', 'test'):
            self.assertEqual([s.source_id for s in split[part+'_X']], manifest['splits'][0]['kept'][part])
            self.assertTrue(set(manifest['splits'][0]['kept'][part]) <= set(original[part]))
        with self.assertRaisesRegex(ValueError, 'lacks requested'):
            requested_splits(manifest, [11])
        with self.assertRaisesRegex(ValueError, 'CV protocol'):
            requested_splits(manifest, [123], True)
        with self.assertRaisesRegex(ValueError, 'missing from loaded'):
            next(filtered_splits(data[:1], targets[:1], manifest, [123]))

    def test_manifest_identity_detects_changed_inputs_policy_and_old_results(self):
        rows, original = fixture()
        manifest = manifest_for(rows, original)
        altered = copy.deepcopy(manifest)
        altered['policy'] = {**DEFAULT_POLICY, 'iqr_multiplier': 4}
        self.assertNotEqual(manifest_identity(manifest), manifest_identity(altered))
        frame = pd.DataFrame([[r['source_id'], r['target']] for r in rows])
        verify_source_table(manifest, frame)
        frame.iloc[0, 1] += .1
        with self.assertRaisesRegex(ValueError, 'target changed'):
            verify_source_table(manifest, frame)
        with tempfile.TemporaryDirectory() as tmp:
            old = {'layout': 'compact_v1', 'config': {'task': 'alignn_full'}, 'run_label': 'full'}
            ensure_compact_run_config(tmp, old)
            new = copy.deepcopy(old)
            new['config']['native_data_filter'] = manifest_identity(manifest)
            with self.assertRaisesRegex(RuntimeError, 'configuration differs'):
                validate_compact_run_config(tmp, new)

    def test_structure_scan_detects_short_distance_and_preserves_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'1-ZnO-Zn_i_A-POSCAR1-Neutral.cif'
            raw = Structure(Lattice.cubic(10), ['Zn', 'O', 'Zn'],
                            [[.1,.1,.1], [.5,.5,.5], [.11,.1,.1]], labels=['Zn0', 'O1', 'Zn2'])
            raw.to(filename=path, fmt='cif')
            before = path.read_bytes()
            row = inspect_source(path, 2.0, DEFAULT_POLICY)
            self.assertEqual(row['quality_reasons'], ['short_interatomic_distance'])
            self.assertAlmostEqual(row['min_distance_angstrom'], .1)
            self.assertEqual(path.read_bytes(), before)
            invalid = inspect_source(path, float('nan'), DEFAULT_POLICY)
            self.assertIn('nonfinite_target', invalid['quality_reasons'])

    def test_strict_scan_catches_compression_above_one_angstrom(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'1-ZnO-Zn_i_A-POSCAR1-Neutral.cif'
            raw = Structure(Lattice.cubic(10), ['Zn', 'O', 'Zn'],
                            [[.1,.1,.1], [.24,.1,.1], [.5,.5,.5]], labels=['Zn0','O1','Zn2'])
            raw.to(filename=path, fmt='cif')
            normal = inspect_source(path, 2., DEFAULT_POLICY)
            strict = inspect_source(path, 2., {**DEFAULT_POLICY, 'deep_checks':DEEP_POLICY})
            self.assertEqual(normal['quality_reasons'], [])
            self.assertEqual(strict['quality_reasons'], [])
            self.assertGreater(strict['min_distance_angstrom'], 1.)
            self.assertIn('compressed_pair_relative_to_covalent_radii', geometry_exclusions(strict, DEEP_POLICY))

    def test_normalized_geometry_checks_every_pair_not_only_absolute_shortest(self):
        atoms = Atoms('NNSnSn', positions=[[0,0,0],[1.4,0,0],[5,5,0],[7,5,0]], cell=[20,20,20], pbc=True)
        distances = atoms.get_all_distances(mic=True)
        np.fill_diagonal(distances, np.inf)
        features = geometry_features(atoms, distances, ['N0','N1','Sn2','Sn3'])
        self.assertEqual(features['compressed_pair'], 'Sn2-Sn3')
        self.assertAlmostEqual(features['compressed_pair_distance_angstrom'], 2.)

    def test_strict_families_preserve_distinct_site_baselines_and_train_only_fitting(self):
        rows, original = fixture()
        second_site = [dict(rows[i], source_id=f'2-ZnO-Zn_i_B-POSCAR{i}-Neutral.cif',
                            target=20+i/10) for i in range(4)]
        rows.extend(second_site)
        original['train'].extend(r['source_id'] for r in second_site)
        high_site = dict(second_site[0], source_id='2-ZnO-Zn_i_B-POSCAR4-Neutral.cif',target=20.5)
        rows.append(high_site)
        original['val'].append(high_site['source_id'])
        policy = {**DEFAULT_POLICY, 'deep_checks':DEEP_POLICY}
        first = fit_split(rows, original, policy)
        self.assertIn(high_site['source_id'], first['kept']['val'])
        self.assertIn('energy_above_same_family_train_fence', first['excluded']['val'][0]['reasons'])
        changed = copy.deepcopy(rows)
        changed[10]['target'] *= 10
        changed[12]['target'] *= 10
        changed[9]['target'] += .1
        second = fit_split(changed, original, policy)
        self.assertEqual(first['family_train_fences'], second['family_train_fences'])
        self.assertEqual(first['kept']['train'], second['kept']['train'])
        baseline = fit_split(rows, original, DEFAULT_POLICY)
        self.assertEqual(first['train_fences'], baseline['train_fences'])
        for part in ('train','val','test'):
            self.assertTrue(set(first['kept'][part]) <= set(baseline['kept'][part]))

    def test_near_geometry_conflicts_are_diagnostic_and_remove_translation(self):
        a = Atoms('ZnO',positions=[[1,1,1],[3,1,1]],cell=[10,10,10],pbc=True)
        b = a.copy()
        b.positions += [3,0,0]
        rows = [{'source_id':f'1-ZnO-V_Zn-POSCAR{i}-Neutral.cif','target':float(i)*2} for i in range(2)]
        pairs, anomalies = audit_family_geometry(rows, {rows[0]['source_id']:a,rows[1]['source_id']:b}, DEEP_POLICY)
        self.assertEqual(anomalies, [])
        self.assertTrue(pairs[0]['near_geometry_energy_conflict'])
        self.assertAlmostEqual(pairs[0]['rms_angstrom'],0.)
        self.assertNotIn('quality_reasons', rows[0])

    def test_joint_rule_preserves_stable_short_bonds_and_requires_energy_evidence(self):
        fence = {'min_pair_ratio_median':1., 'joint_energy_upper_ev':5.}
        self.assertTrue(joint_distortion_exclusion({'min_pair_covalent_ratio':.85,'target':6},fence,DEEP_POLICY))
        self.assertFalse(joint_distortion_exclusion({'min_pair_covalent_ratio':.85,'target':4},fence,DEEP_POLICY))
        self.assertFalse(joint_distortion_exclusion({'min_pair_covalent_ratio':.98,'target':6},fence,DEEP_POLICY))
        stable_short = {**fence,'min_pair_ratio_median':.88}
        self.assertFalse(joint_distortion_exclusion({'min_pair_covalent_ratio':.87,'target':6},stable_short,DEEP_POLICY))

    def test_strict_removal_table_includes_frozen_thresholds(self):
        rows, original = fixture()
        for row in rows:
            row.update(min_pair_covalent_ratio=1., max_nearest_covalent_ratio=1.)
        policy = {**DEFAULT_POLICY, 'deep_checks':DEEP_POLICY}
        split = {'logger_id':123, **fit_split(rows, original, policy)}
        manifest = {'records':rows,'splits':[split],'policy':policy}
        with tempfile.TemporaryDirectory() as tmp:
            write_manifest_tables(manifest, tmp)
            excluded = pd.read_csv(Path(tmp)/'seed123_removed.csv')
            val = excluded[excluded.split == 'val'].iloc[0]
            self.assertIn('family_joint_energy_upper_ev', excluded.columns)
            expected = split['family_train_fences']['1-ZnO-Zn_i_A-Neutral.cif']['upper_ev']
            self.assertAlmostEqual(val.family_upper_ev,expected)
            self.assertEqual(len(excluded),3)

    def test_loader_masks_before_parsing_and_all_representations_share_ids(self):
        rows, original = fixture()
        manifest = manifest_for(rows, original)
        frame = pd.DataFrame([[r['source_id'], r['target']] for r in rows])
        def structure(path):
            s = Structure(Lattice.cubic(8), ['Zn','O','Zn'],
                          [[.1,.1,.1],[.5,.5,.5],[.8,.8,.8]], labels=['Zn0','O1','Zn2'])
            return s
        with patch('HERA.data.datasets.pd.read_csv', return_value=frame), \
                patch('HERA.data.native_filter.verify_source_file'), \
                patch('HERA.data.datasets.Structure.from_file', side_effect=structure) as parse, redirect_stdout(io.StringIO()):
            result = load_dataset('native', 'alignn', representations=['hetero','attention'],
                                  native_preprocessing='reference_v1', native_filter_manifest=manifest)
        expected = set().union(*map(set, manifest['splits'][0]['kept'].values()))
        self.assertEqual(parse.call_count, len(expected))
        self.assertEqual([s.source_id for s in result[1]], [s.source_id for s in result[2]])
        self.assertEqual(set(s.source_id for s in result[1]), expected)
        self.assertEqual(len(result[-1]), len(expected))

    def test_cli_records_filter_identity_and_passes_manifest_to_training(self):
        rows, original = fixture()
        manifest = manifest_for(rows, original)
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'manifest.json'
            path.write_text(json.dumps(manifest), encoding='utf-8')
            argv = ['hera', '--model', 'alignn', '--dataset', 'native', '--mode', 'attention_was',
                    '--epochs', '1', '--seed', '123', '--run-dir', str(Path(tmp)/'run'),
                    '--compact-logs', '--atom-init', str(root/'atom_init.json'),
                    '--native-filter-manifest', str(path)]
            with patch('sys.argv', argv), patch.object(cli, 'load_dataset', return_value=(None,None,[],None,[])) as loader, \
                    patch.object(cli, 'train_single_mode', return_value=[.2]) as train, redirect_stdout(io.StringIO()):
                cli.main()
            self.assertEqual(train.call_args.args[1]['native_data_filter'], manifest_identity(manifest))
            self.assertEqual(train.call_args.kwargs['native_filter_manifest'], manifest)
            self.assertEqual(loader.call_args.kwargs['native_filter_manifest'], manifest)
            self.assertTrue((Path(tmp)/'run/native_filter_manifest.json').is_file())


if __name__ == '__main__':
    unittest.main()
