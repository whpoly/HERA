"""Physical screening, source provenance and split preservation regressions."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ase import Atoms
from ase.db import connect
from ase.io import write as ase_write
import numpy as np
import pandas as pd
from pymatgen.io.ase import AseAtomsAdaptor

from HERA import main as cli
from HERA.data import impurity_preprocessing as prep
from HERA.data.datasets import _impurity_filtered_table
from HERA.data.impurity_quality import inspect_geometry, inspect_imp2d_metadata, inspect_impurity_composition


def metadata(eform=1., **updates):
    result = dict(eform=eform, en2=-11.+eform, hostenergy=-10., dopant_chemical_potential=-1.,
                  conv1=.001, conv2=.001, extension_factor=1., converged=True,
                  host_spacegroup=1, site='ads0', depth=2.)
    result.update(updates)
    return result


class PhysicalRulesTests(unittest.TestCase):
    def test_final_convergence_overrides_first_stage_flag(self):
        bad = inspect_imp2d_metadata(metadata(conv1=.001, conv2=-.1, converged=True))
        self.assertIn('final_ionic_energy_not_converged', bad['quality_reasons'])
        good = inspect_imp2d_metadata(metadata(conv1=.1, conv2=.001, converged=False))
        self.assertEqual(good['quality_reasons'], [])
        self.assertIn('source_converged_flag_differs_from_final_stage', good['review_flags'])

    def test_energy_identity_and_expansion_without_target_trimming(self):
        self.assertEqual(inspect_imp2d_metadata(metadata(eform=-20.))['quality_reasons'], [])
        self.assertIn('inconsistent_energy_metadata', inspect_imp2d_metadata(metadata(en2=0.))['quality_reasons'])
        self.assertIn('excessive_reconstruction_outside_point_defect_scope',
                      inspect_imp2d_metadata(metadata(extension_factor=2.1))['quality_reasons'])
        self.assertIn('missing_final_convergence', inspect_imp2d_metadata(metadata(conv2=float('nan')))['quality_reasons'])
        self.assertEqual(inspect_imp2d_metadata(metadata(site='int1', depth=3.))['quality_reasons'], [])

    def test_geometry_preserves_hydrogen_bond_and_weak_adsorption(self):
        h2 = Atoms('H2', positions=[[0,0,0],[.74,0,0]], cell=[10]*3, pbc=True)
        self.assertEqual(inspect_geometry(h2)['quality_reasons'], [])
        h2.positions[1,0] = .1
        self.assertIn('severe_atomic_overlap', inspect_geometry(h2)['quality_reasons'])
        weak = Atoms('ZnOH', positions=[[0,0,0],[2,0,0],[1,1,8]], cell=[20]*3, pbc=True)
        self.assertEqual(inspect_geometry(weak)['quality_reasons'], [])
        self.assertIn('weakly_bound_or_isolated_atoms_review_only', inspect_geometry(weak)['review_flags'])

    def test_periodic_self_overlap_is_detected(self):
        atoms = Atoms('ZnO', positions=[[0,0,0],[0,3,0]], cell=[.2,10,10], pbc=True)
        self.assertIn('severe_atomic_overlap', inspect_geometry(atoms)['quality_reasons'])

    def test_substitution_species_inferred_from_counts(self):
        atoms = Atoms('FeO', positions=[[0,0,0],[2,0,0]], cell=[10]*3, pbc=True)
        record = inspect_impurity_composition(atoms, '1-ZnO-M_A-Fe-POSCAR0.cif', 'semi')
        self.assertEqual(record['previous_z'], 30)
        self.assertEqual(record['quality_reasons'], [])
        wrong = inspect_impurity_composition(atoms, '1-ZnO-M_i_A-Fe-POSCAR0.cif', 'semi')
        self.assertIn('interstitial_host_stoichiometry_mismatch', wrong['quality_reasons'])

    def test_database_geometry_match_allows_reorder_rotation_translation(self):
        atoms = Atoms('ZnOH', positions=[[1,1,1],[3,1,1],[2,2,2]], cell=[10]*3, pbc=True)
        other = atoms[[2,0,1]]
        other.positions += [7,3,2]
        other.rotate(37, 'z', rotate_cell=True)
        self.assertTrue(prep.same_periodic_geometry(other, atoms))
        other.positions[0,0] += .1
        self.assertFalse(prep.same_periodic_geometry(other, atoms))


class FrozenFilterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.hosts = self.root/'hosts'
        self.hosts.mkdir()
        ase_write(str(self.hosts/'ZnO.vasp'), Atoms('ZnO', positions=[[1,1,1],[3,1,1]], cell=[10]*3, pbc=True), format='vasp')
        self.db = self.root/'imp2d.db'
        self.dirs = {}
        for dataset in ('semi', 'imp2d'):
            folder = self.root/dataset
            folder.mkdir()
            self.dirs[dataset] = folder
            entries = []
            for i in range(20):
                atoms = Atoms('FeO' if dataset == 'semi' else 'ZnOH',
                              positions=[[1,1,1],[3,1,1]] if dataset == 'semi' else [[1,1,1],[3,1,1],[2,2,2]],
                              cell=[10]*3, pbc=True)
                target = 100. if dataset == 'semi' and i == 18 else .1*i
                if dataset == 'semi':
                    atoms.positions[1,0] += .005*i  # Distinct physical inputs for the energy-fence test.
                sid = f'1-ZnO-M_A-Fe-POSCAR{i}.cif' if dataset == 'semi' else f'ZnO_H_ads{i}'
                if i == 19:
                    atoms.positions[1] = atoms.positions[0]+.05
                if dataset == 'imp2d':
                    if i == 18:
                        target = 20.  # Outside the pre-existing benchmark cohort.
                    kv = metadata(eform=target, site=f'ads{i}')
                    if i == 17:
                        kv['conv2'] = .2
                    if i == 16:
                        kv.update(converged=False, conv1=.2)
                    with connect(str(self.db)) as db:
                        db.write(atoms, name=sid, **kv)
                AseAtomsAdaptor.get_structure(atoms).to(filename=prep.source_path(folder,dataset,sid), fmt='cif')
                entries.append([sid,target])
            pd.DataFrame(entries).to_csv(folder/prep.TABLES[dataset],header=False,index=False)

    def prepare(self, dataset='semi', run=None, seeds=None, cv5=False):
        with redirect_stdout(io.StringIO()):
            return prep.prepare_impurity_filter(run or self.root/f'run_{dataset}', dataset, seeds or [123], cv5,
                                                cli.iter_train_val_test_splits, data_dir=self.dirs[dataset],
                                                database_path=self.db, host_dir=self.hosts)

    def test_semi_keeps_extreme_energy_and_preserves_original_splits(self):
        manifest = self.prepare()
        original = list(cli.iter_train_val_test_splits(
            [r['source_id'] for r in manifest['records']], [0]*20, [123]))[0]
        saved = manifest['splits'][0]
        for part in prep.PARTS:
            self.assertEqual(saved['original'][part], original[f'{part}_X'])
            self.assertEqual(saved['kept'][part], [s for s in original[f'{part}_X'] if 'POSCAR19.' not in s])
        self.assertIn('1-ZnO-M_A-Fe-POSCAR18.cif', prep.retained_ids(manifest))
        second = self.prepare()
        self.assertEqual(prep.identity(manifest), prep.identity(second))

    def test_imp2d_final_stage_and_legacy_cohort(self):
        manifest = self.prepare('imp2d')
        kept = prep.retained_ids(manifest)
        self.assertIn('ZnO_H_ads16', kept)  # Initial convergence false but final valid.
        self.assertNotIn('ZnO_H_ads17', kept)
        self.assertNotIn('ZnO_H_ads18', kept)
        self.assertNotIn('ZnO_H_ads19', kept)
        self.assertEqual(sum(len(x) for x in manifest['splits'][0]['original'].values()), 19)
        frame, _ = _impurity_filtered_table(self.dirs['imp2d']/'id_prop.csv','imp2d',manifest)
        self.assertEqual(set(frame[0]),kept)
        self.assertEqual(prep.identity(manifest), prep.identity(self.prepare('imp2d')))

    def test_missing_default_database_downloads_before_screening(self):
        with patch.object(prep, 'default_database', return_value=None), \
                patch.object(prep, 'download_imp2d_database', return_value=self.db) as download, \
                redirect_stdout(io.StringIO()):
            manifest = prep.prepare_impurity_filter(self.root/'automatic', 'imp2d', [123], False,
                cli.iter_train_val_test_splits, data_dir=self.dirs['imp2d'])
        download.assert_called_once_with(self.dirs['imp2d'].parent/'imp2d.db')
        self.assertNotIn('ZnO_H_ads17', prep.retained_ids(manifest))

    def test_semi_never_downloads_the_imp2d_database(self):
        with patch.object(prep, 'default_database') as locate, \
                patch.object(prep, 'download_imp2d_database') as download:
            self.prepare('semi')
        locate.assert_not_called()
        download.assert_not_called()

    def test_identical_semi_structures_with_conflicting_labels_are_all_quarantined(self):
        folder = self.dirs['semi']
        original = folder/'1-ZnO-M_A-Fe-POSCAR0.cif'
        (folder/'1-ZnO-M_A-Fe-POSCAR1.cif').write_bytes(original.read_bytes())
        (folder/'1-ZnO-M_A-Fe-POSCAR3.cif').write_bytes((folder/'1-ZnO-M_A-Fe-POSCAR2.cif').read_bytes())
        table = folder/prep.TABLES['semi']
        frame = pd.read_csv(table,header=None)
        frame.iloc[3,1] = frame.iloc[2,1]+1e-5  # Small numerical variation is not a conflict.
        frame.to_csv(table,index=False,header=False)
        manifest = self.prepare()
        rows = {r['source_id']:r for r in manifest['records']}
        for i in (0,1):
            self.assertIn('identical_structure_inconsistent_target',rows[f'1-ZnO-M_A-Fe-POSCAR{i}.cif']['quality_reasons'])
            self.assertNotIn(f'1-ZnO-M_A-Fe-POSCAR{i}.cif',prep.retained_ids(manifest))
        for i in (2,3):
            self.assertIn(f'1-ZnO-M_A-Fe-POSCAR{i}.cif',prep.retained_ids(manifest))

    def test_filtered_graph_splits_and_cv_keep_frozen_membership(self):
        manifest = self.prepare(cv5=True)
        records = [r for r in manifest['records'] if r['source_id'] in prep.retained_ids(manifest)]
        graphs = [SimpleNamespace(source_id=r['source_id']) for r in reversed(records)]
        targets = [r['target'] for r in reversed(records)]
        loaded = list(prep.filtered_splits(graphs,targets,manifest,[123],cv5=True))
        self.assertEqual(len(loaded),5)
        by_id = {r['source_id']:r['target'] for r in records}
        for actual,saved in zip(loaded,manifest['splits']):
            for part in prep.PARTS:
                self.assertEqual([s.source_id for s in actual[f'{part}_X']],saved['kept'][part])
                self.assertEqual(actual[f'{part}_y'],[by_id[s] for s in saved['kept'][part]])
        with self.assertRaisesRegex(ValueError,'missing from loaded graphs'):
            list(prep.filtered_splits(graphs[:-1],targets[:-1],manifest,[123],cv5=True))

    def test_changed_excluded_cif_and_host_are_rejected(self):
        manifest = self.prepare()
        excluded = next(r for r in manifest['records'] if r['quality_reasons'])
        path = prep.source_path(self.dirs['semi'],'semi',excluded['source_id'])
        original = path.read_bytes()
        path.write_bytes(original+b'\n')
        with self.assertRaisesRegex(ValueError,'CIF changed'):
            self.prepare()
        path.write_bytes(original)
        host = self.hosts/'ZnO.vasp'
        host.write_bytes(host.read_bytes()+b'\n')
        with self.assertRaisesRegex(ValueError,'host changed'):
            self.prepare()

    def test_missing_files_are_not_removed_as_physical_outliers(self):
        (self.hosts/'ZnO.vasp').unlink()
        with self.assertRaisesRegex(ValueError,'Missing semi reference host'):
            self.prepare()
        self.assertFalse((self.root/'run_semi/semi_filter_manifest.json').exists())

    def test_explicit_legacy_cohort_records_missing_sources_separately_and_freezes_them(self):
        table = self.dirs['semi']/prep.TABLES['semi']
        frame = pd.read_csv(table,header=None)
        missing = '1-ZnO-M_A-Fe-POSCAR100.cif'
        no_host = '2-GeSn-M_A-Fe-POSCAR1.cif'
        atoms = Atoms('FeSn',positions=[[1,1,1],[3,1,1]],cell=[10]*3,pbc=True)
        AseAtomsAdaptor.get_structure(atoms).to(filename=self.dirs['semi']/no_host,fmt='cif')
        frame.loc[len(frame)] = [missing,1.]
        frame.loc[len(frame)] = [no_host,2.]
        frame.to_csv(table,header=False,index=False)
        def legacy():
            with redirect_stdout(io.StringIO()):
                return prep.prepare_impurity_filter(self.root/'legacy','semi',[123],False,
                    cli.iter_train_val_test_splits,data_dir=self.dirs['semi'],host_dir=self.hosts,
                    semi_source_policy='legacy_available')
        manifest = legacy()
        rows = {r['source_id']:r for r in manifest['records']}
        self.assertEqual(rows[missing]['baseline_exclusion'],'legacy_missing_cif')
        self.assertEqual(rows[no_host]['baseline_exclusion'],'legacy_missing_or_empty_host')
        self.assertFalse(rows[missing]['quality_reasons'])
        self.assertFalse(rows[no_host]['quality_reasons'])
        self.assertEqual(sum(len(x) for x in manifest['splits'][0]['original'].values()),20)
        self.assertEqual(prep.identity(manifest),prep.identity(legacy()))
        (self.dirs['semi']/missing).write_bytes(b'newly appeared')
        with self.assertRaisesRegex(ValueError,'Previously missing semi CIF appeared'):
            legacy()
        (self.dirs['semi']/missing).unlink()
        ase_write(str(self.hosts/'GeSn.vasp'),Atoms('GeSn',positions=[[1,1,1],[3,1,1]],cell=[10]*3,pbc=True),format='vasp')
        with self.assertRaisesRegex(ValueError,'Semi host changed'):
            legacy()

    def test_wrong_csv_database_version_stops(self):
        path = self.dirs['imp2d']/'id_prop.csv'
        frame = pd.read_csv(path,header=None)
        frame.iloc[0,1] += 1
        frame.to_csv(path,header=False,index=False)
        with self.assertRaisesRegex(ValueError,'label/version mismatch'):
            self.prepare('imp2d')

    def test_seed_cv_and_source_changes_cannot_reuse(self):
        self.prepare()
        with self.assertRaisesRegex(ValueError,'lacks requested seeds'):
            self.prepare(seeds=[42])
        with self.assertRaisesRegex(ValueError,'CV protocol'):
            self.prepare(cv5=True)
        path = self.dirs['semi']/prep.TABLES['semi']
        frame = pd.read_csv(path,header=None)
        frame.iloc[0,1] += 1
        frame.to_csv(path,header=False,index=False)
        with self.assertRaisesRegex(ValueError,'targets changed'):
            self.prepare()

    def test_cli_applies_same_manifest_to_all_modes_and_preserves_config_identity(self):
        run = self.root/'cli_run'
        argv = ['hera','--model','alignn','--dataset','semi','imp2d',
                '--mode','attention_was','hetero_was','--r','0','--epochs','1','--seed','123',
                '--compact-logs','--run-dir',str(run), '--atom-init',str(Path(cli.__file__).parent/'atom_init.json'),
                '--semi-quality-filter','physical','--imp2d-quality-filter','physical',
                '--semi-source-policy','legacy_available',
                '--semi-preprocessing','reference_v1','--imp2d-preprocessing','reference_v1']
        def prepare(run_dir,dataset,*args,**kwargs):
            return prep.prepare_impurity_filter(run_dir,dataset,*args,data_dir=self.dirs[dataset],
                                                 host_dir=self.hosts,database_path=self.db,
                                                 semi_source_policy=kwargs.get('semi_source_policy','complete'))
        def loader(dataset,*args,**kwargs):
            self.assertEqual(kwargs['impurity_filter_manifest']['dataset'],dataset)
            return None,[],[],None,[]
        with patch('sys.argv',argv), patch.object(cli,'prepare_impurity_filter',side_effect=prepare) as build, \
                patch.object(cli,'load_dataset',side_effect=loader) as load, \
                patch.object(cli,'train_single_mode',return_value=[.2]) as train, redirect_stdout(io.StringIO()):
            cli.main()
        self.assertEqual(build.call_count,2)
        self.assertEqual(build.call_args_list[0].kwargs['semi_source_policy'],'legacy_available')
        self.assertEqual(load.call_count,4)
        self.assertEqual(train.call_count,4)
        for call in train.call_args_list:
            manifest = call.kwargs['impurity_filter_manifest']
            self.assertEqual(call.args[1]['impurity_data_filter'],prep.identity(manifest))
        self.assertTrue((run/'semi_filter_seed123_removed.csv').is_file())


if __name__ == '__main__':
    unittest.main()
