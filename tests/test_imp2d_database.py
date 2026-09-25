"""DB-only inputs, physical decisions and group-disjoint splits."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ase import Atoms
from ase.db import connect

from HERA import main as cli
from HERA.data import datasets
from HERA.data.impurity_preprocessing import prepare_impurity_filter, identity, filtered_splits
from HERA.data.imp2d_database import load_database_inputs, validate_database_manifest, database_policy


class DatabaseInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.db = self.root/'imp2d.db'
        with connect(str(self.db)) as db:
            for i in range(24):
                displacement = 0 if i in (0, 1) else .08*i
                atoms = Atoms('ZnOH', positions=[[1,1,1],[3,1,1],[2,2+displacement,2]],
                              cell=[10,11,12], pbc=True)
                target = 20. if i == 2 else .01*i
                db.write(atoms, name=f'ZnO_H_ads{i}', eform=target, en2=-11.+target if i != 4 else 0.,
                         hostenergy=-10., dopant_chemical_potential=-1., conv1=.1 if i == 5 else .001,
                         conv2=.2 if i == 3 else .001, converged=i != 5,
                         extension_factor=1., site=f'ads{i}', depth=2., spin=0)
            db.write(Atoms('ZnOZn', positions=[[1,1,1],[3,1,1],[2,3,3]], cell=[10,11,12], pbc=True),
                     name='ZnO_Zn_int0', eform=1., en2=-10., hostenergy=-10.,
                     dopant_chemical_potential=-1., conv1=.001, conv2=.001, converged=True,
                     extension_factor=1., site='int0', depth=0., spin=0)

    def prepare(self, cv5=False, name='run', host_filter=None, energy_window=None):
        with redirect_stdout(io.StringIO()):
            return prepare_impurity_filter(self.root/name, 'imp2d', [123], cv5,
                cli.iter_train_val_test_splits, database_path=self.db, imp2d_source='db',
                imp2d_host_filter=host_filter, imp2d_energy_window=energy_window)

    def test_energy_window_open_bounds_preserve_labels_and_split_membership(self):
        self.add_reviewed_host()
        targets = [-10., 10., -9.999, 9.999, -11., 11.]
        with connect(str(self.db)) as db:
            for i, target in enumerate(targets, 6):
                db.update(db.get(name=f'ZnO_H_ads{i}').id, eform=target, en2=-11.+target)
        for cv5 in (False, True):
            original = self.prepare(cv5=cv5, name=f'base_{cv5}', host_filter='reviewed_v1')
            clean = self.prepare(cv5=cv5, name=f'window_{cv5}', host_filter='reviewed_v1',
                                 energy_window=(-10,10))
            by_id = {r['source_id']:r for r in clean['records']}
            for before, after in zip(original['splits'], clean['splits']):
                self.assertEqual(before['original'], after['original'])
                for part in ('train','val','test'):
                    self.assertEqual(after['kept'][part], [sid for sid in before['kept'][part]
                                                         if -10 < by_id[sid]['target'] < 10])
            for i, target in enumerate(targets, 6):
                row = by_id[f'ZnO_H_ads{i}']
                self.assertEqual(row['target'], target)
                self.assertFalse(row['physical_quality_reasons'])
                self.assertEqual(bool(row['energy_window_exclusion_reasons']), not -10 < target < 10)
            self.assertEqual(identity(clean), identity(self.prepare(cv5=cv5, name=f'window_{cv5}',
                host_filter='reviewed_v1', energy_window=(-10,10))))
            summary = json.loads((self.root/f'window_{cv5}'/'imp2d_energy_window_summary.json').read_text())
            self.assertEqual((summary['before_energy_window'], summary['additional_excluded_rows'],
                              summary['retained_rows']), (22,5,17))
        # Window-only selection also works without adding a host policy.
        window_only = self.prepare(name='window_only', energy_window=(-10,10))
        self.assertEqual(sum(not r['quality_reasons'] for r in window_only['records']),17)

    def test_energy_window_rejects_invalid_bounds_source_and_cached_policy_changes(self):
        for bounds in ((10,-10),(10,10),(float('nan'),10),(-10,float('inf'))):
            with self.assertRaisesRegex(ValueError, 'finite increasing'):
                database_policy(energy_window=bounds)
        self.prepare()
        with self.assertRaisesRegex(ValueError, 'policy changed'):
            self.prepare(energy_window=(-10,10))
        with self.assertRaisesRegex(ValueError, 'direct database'):
            prepare_impurity_filter(self.root/'bad', 'semi', [123], False,
                cli.iter_train_val_test_splits, imp2d_energy_window=(-10,10))
        manifest = self.prepare(name='window', energy_window=(-10,10))
        next(r for r in manifest['records'] if r['source_id']=='ZnO_H_ads2')['energy_window_exclusion_reasons']=[]
        with self.assertRaisesRegex(ValueError, 'energy window'):
            validate_database_manifest(manifest)

    def add_reviewed_host(self):
        with connect(str(self.db)) as db:
            for i in range(3):
                atoms = Atoms('Ti2CO2H', positions=[[1,1,1],[4,1,1],[1,4,1],
                              [4,4,1],[1,1,4],[4,4,4]], cell=[14,15,16], pbc=True)
                db.write(atoms, name=f'CO2Ti2_H_ads{i}', eform=-170.+i, en2=-181.+i,
                         hostenergy=-10., dopant_chemical_potential=-1., conv1=.001,
                         conv2=.001, converged=True, extension_factor=1., site=f'ads{i}', depth=2., spin=0)

    def test_host_filter_preserves_labels_groups_and_existing_split_membership(self):
        self.add_reviewed_host()
        for cv5 in (False, True):
            original = self.prepare(cv5=cv5, name=f'all_{cv5}')
            clean = self.prepare(cv5=cv5, name=f'clean_{cv5}', host_filter='reviewed_v1')
            records = {r['source_id']: r for r in clean['records']}
            for row in original['records']:
                updated = records[row['source_id']]
                self.assertEqual(row['target'], updated['target'])
                self.assertEqual(row['structure_group'], updated['structure_group'])
                if row['material'] == 'CO2Ti2':
                    self.assertFalse(updated['physical_quality_reasons'])
                    self.assertEqual(updated['cohort_exclusion_reasons'], ['host_energy_reference_under_review'])
                else:
                    self.assertEqual(row['quality_reasons'], updated['quality_reasons'])
            self.assertFalse(records['ZnO_H_ads2']['quality_reasons'])  # Other host's Eform=20 stays.
            for before, after in zip(original['splits'], clean['splits']):
                self.assertEqual(before['original'], after['original'])
                for part in ('train','val','test'):
                    self.assertEqual(after['kept'][part], [sid for sid in before['kept'][part]
                                                         if not sid.startswith('CO2Ti2_')])
            self.assertEqual(identity(clean), identity(self.prepare(cv5=cv5, name=f'clean_{cv5}',
                                                                   host_filter='reviewed_v1')))
            summary = json.loads((self.root/f'clean_{cv5}'/'imp2d_host_filter_summary.json').read_text())
            self.assertEqual((summary['additional_excluded_rows'], summary['retained_rows']), (3,22))

    def test_host_filter_rejects_other_sources_and_policy_changes(self):
        self.add_reviewed_host()
        self.prepare()
        with self.assertRaisesRegex(ValueError, 'policy changed'):
            self.prepare(host_filter='reviewed_v1')
        with self.assertRaisesRegex(ValueError, 'direct database'):
            prepare_impurity_filter(self.root/'bad', 'imp2d', [123], False,
                cli.iter_train_val_test_splits, imp2d_host_filter='reviewed_v1')
        clean = self.prepare(name='clean', host_filter='reviewed_v1')
        next(r for r in clean['records'] if r['material']=='CO2Ti2')['cohort_exclusion_reasons'] = []
        with self.assertRaisesRegex(ValueError, 'host exclusion'):
            validate_database_manifest(clean)

    def test_full_database_filter_needs_no_csv_and_keeps_finite_extreme_energy(self):
        with patch.object(datasets.pd, 'read_csv', side_effect=AssertionError('CSV forbidden')):
            manifest = self.prepare()
        records = {r['source_id']:r for r in manifest['records']}
        self.assertEqual(len(records), 25)
        self.assertEqual(sum(not r['quality_reasons'] for r in records.values()), 22)
        self.assertFalse(records['ZnO_H_ads2']['quality_reasons'])  # Eform=20 is not trimmed.
        self.assertFalse(records['ZnO_H_ads5']['quality_reasons'])  # Final stage converged.
        self.assertIn('final_ionic_energy_not_converged', records['ZnO_H_ads3']['physical_quality_reasons'])
        self.assertFalse(records['ZnO_Zn_int0']['physical_quality_reasons'])
        self.assertEqual(records['ZnO_Zn_int0']['input_quality_reasons'], ['unverified_self_impurity_identity'])
        self.assertEqual(identity(manifest), identity(self.prepare()))

    def test_grouped_random_and_cv_splits_never_separate_duplicate_structures(self):
        for cv5, name in ((False,'random'),(True,'cv')):
            manifest = self.prepare(cv5=cv5, name=name)
            for split in manifest['splits']:
                membership = {sid:part for part in ('train','val','test') for sid in split['kept'][part]}
                self.assertEqual(membership['ZnO_H_ads0'], membership['ZnO_H_ads1'])
            self.assertEqual(len(manifest['splits']), 5 if cv5 else 1)

    def test_all_representations_load_direct_db_with_correct_labels_and_splits(self):
        manifest = self.prepare()
        with patch.object(datasets.Structure, 'from_file', side_effect=AssertionError('CIF forbidden')), \
                patch.object(datasets.pd, 'read_csv', side_effect=AssertionError('CSV forbidden')):
            full, hetero, attention, _, targets = datasets.load_data_imp2d('alignn', local_cutoff=0,
                representations=['full','hetero','attention'], imp2d_preprocessing='reference_v1',
                impurity_filter_manifest=manifest, imp2d_source_db=self.db)
        for representation in (full, hetero, attention):
            self.assertEqual(len(representation), 22)
            for structure in representation:
                marked = [s for s in structure if s.properties['type']]
                self.assertEqual(len(marked), 1)
                self.assertEqual(marked[0].specie.symbol, 'H')
                self.assertEqual(marked[0].properties['was'], 0)
            splits = list(filtered_splits(representation, targets, manifest, [123]))
            self.assertEqual(sum(len(splits[0][f'{p}_X']) for p in ('train','val','test')),22)

    def test_changed_database_and_tampered_group_membership_are_rejected(self):
        manifest = self.prepare()
        split = manifest['splits'][0]
        part = next(p for p in ('train','val','test') if 'ZnO_H_ads1' in split['kept'][p])
        destination = next(p for p in ('train','val','test') if p != part)
        for field in ('original','kept'):
            split[field][part].remove('ZnO_H_ads1')
            split[field][destination].append('ZnO_H_ads1')
        with self.assertRaisesRegex(ValueError, 'cross split'):
            validate_database_manifest(manifest)
        fresh = self.prepare()
        with connect(str(self.db)) as db:
            db.update(1, eform=2.)
        with self.assertRaisesRegex(ValueError, 'changed'):
            load_database_inputs(fresh, self.db)

    def test_cli_db_source_passes_database_path_and_one_shared_manifest(self):
        self.add_reviewed_host()
        argv = ['hera','--model','alignn','--dataset','imp2d','--mode','attention_was','hetero_was',
                '--imp2d-source','db','--imp2d-quality-filter','physical','--imp2d-source-db',str(self.db),
                '--imp2d-host-filter','reviewed_v1',
                '--imp2d-energy-window','-10','10',
                '--imp2d-preprocessing','reference_v1','--r','0','--epochs','1','--compact-logs',
                '--run-dir',str(self.root/'cli'),'--atom-init',str(Path(cli.__file__).parent/'atom_init.json')]
        with patch('sys.argv', argv), patch.object(cli,'train_single_mode',return_value=[.2]) as train, \
                patch.object(datasets.Structure,'from_file',side_effect=AssertionError('CIF forbidden')), \
                redirect_stdout(io.StringIO()):
            cli.main()
        self.assertEqual(train.call_count,2)
        manifests = [call.kwargs['impurity_filter_manifest'] for call in train.call_args_list]
        self.assertEqual(identity(manifests[0]), identity(manifests[1]))
        self.assertEqual(manifests[0]['policy']['host_filter']['name'], 'reviewed_v1')
        kept = [sid for s in manifests[0]['splits'] for part in ('train','val','test') for sid in s['kept'][part]]
        self.assertFalse(any(sid.startswith('CO2Ti2_') for sid in kept))
        self.assertNotIn('ZnO_H_ads2',kept)
        self.assertEqual(len(kept),21)
        self.assertEqual(manifests[0]['policy']['energy_window']['lower_ev'],-10.)


if __name__ == '__main__':
    unittest.main()
