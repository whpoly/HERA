"""Native OOD integrity, train-only filtering, and mixed-dataset CLI routing."""
import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from HERA import main as cli
from HERA.data.native_filter import fit_split, validate_manifest, filtered_splits, manifest_identity
from HERA.data.native_preprocessing import native_filter_policy, prepare_native_filter
from HERA.data.native_splits import (
    source_group, iter_ood_splits, ood_splitter, protocol_for, with_ood_policy,
)


def source_ids():
    hosts = ['ZnO', 'ZnS', 'ZnSe', 'ZnTe', 'CdO', 'CdS', 'CdSe', 'CdTe', 'GaN', 'AlN']
    return [f'{h*2+k+1}-{host}-V_A-POSCAR{j}-Neutral.cif'
            for h, host in enumerate(hosts) for k in range(2) for j in range(3)]


def manifest_fixture(grouping='family'):
    ids = source_ids()
    rows = [{'source_id': sid, 'target': 1. + i/1000, 'material': sid.split('-')[1],
             'defect_type': 'vacancy', 'added_z': 0, 'removed_z': 30,
             'sha256': 'fixture', 'quality_reasons': []} for i, sid in enumerate(ids)]
    iterator = ood_splitter('native', grouping)
    policy = with_ood_policy(native_filter_policy('strict'), iterator)
    splits = []
    for original in iterator(ids, [r['target'] for r in rows], [123]):
        membership = {p: original[p+'_X'] for p in ('train', 'val', 'test')}
        splits.append({**{k: original[k] for k in ('display','logger_id','seed','explain_id')},
                       **fit_split(rows, membership, policy)})
    return validate_manifest({'schema': 'native_outliers_v1', 'dataset': 'native', 'cv5': False,
                              'policy': policy, 'records': rows, 'splits': splits})


class NativeOODTests(unittest.TestCase):
    def test_whole_series_and_whole_materials_do_not_cross_splits(self):
        ids = source_ids()
        for grouping in ('family', 'material'):
            for split in iter_ood_splits(ids, range(len(ids)), [123,11,1245],
                                         dataset='native', grouping=grouping):
                groups = [{source_group(sid, 'native', grouping) for sid in split[p+'_X']}
                          for p in ('train','val','test')]
                self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
                self.assertEqual([len(g) for g in groups], [12,4,4] if grouping == 'family' else [6,2,2])
                self.assertCountEqual([sid for p in ('train','val','test') for sid in split[p+'_X']], ids)

    def test_assignments_ignore_targets_representation_order_and_frame_count(self):
        ids = source_ids()
        def membership(data, labels):
            split = next(iter_ood_splits(data, labels, [123], dataset='native', grouping='family'))
            return {getattr(x,'source_id',x): p for p in ('train','val','test') for x in split[p+'_X']}
        expected = membership(ids, [0]*len(ids))
        self.assertEqual(expected, membership(list(reversed(ids)), [1e9]*len(ids)))
        self.assertEqual(expected, membership([SimpleNamespace(source_id=s) for s in ids], range(len(ids))))
        extra = ids[0].replace('POSCAR0','POSCAR999')
        extended = membership(ids+[extra], [0]*(len(ids)+1))
        self.assertEqual(expected, {k:v for k,v in extended.items() if k!=extra})
        self.assertEqual(extended[extra], expected[ids[0]])

    def test_filter_fences_use_only_new_ood_training_ids(self):
        manifest = manifest_fixture()
        saved = manifest['splits'][0]
        changed = copy.deepcopy(manifest['records'])
        held = set(saved['original']['val']+saved['original']['test'])
        for row in changed:
            if row['source_id'] in held:
                row['target'] += 10
        altered = fit_split(changed, saved['original'], manifest['policy'])
        self.assertEqual(saved['train_fences'], altered['train_fences'])
        self.assertEqual(saved['family_train_fences'], altered['family_train_fences'])
        self.assertEqual(saved['kept']['train'], altered['kept']['train'])
        held_families = {source_group(s,'native','family') for s in held}
        self.assertFalse(held_families & set(saved['family_train_fences']))

    def test_manifest_rejects_group_leakage_even_when_source_ids_are_distinct(self):
        manifest = manifest_fixture()
        saved = manifest['splits'][0]
        a, b = saved['original']['train'][0], saved['original']['test'][0]
        for scope in ('original','kept'):
            saved[scope]['train'][0], saved[scope]['test'][0] = b, a
        with self.assertRaisesRegex(ValueError, 'group leakage'):
            validate_manifest(manifest)

    def test_loader_uses_saved_membership_and_records_protocol(self):
        manifest = manifest_fixture()
        structs = [SimpleNamespace(source_id=r['source_id']) for r in reversed(manifest['records'])]
        splits = list(filtered_splits(structs, [0]*len(structs), manifest, [123]))
        self.assertEqual(splits[0]['ood_split'], protocol_for('native','family'))
        for part in ('train','val','test'):
            self.assertEqual([s.source_id for s in splits[0][part+'_X']], manifest['splits'][0]['kept'][part])

    def test_random_and_ood_manifests_cannot_reuse_each_other(self):
        ood = manifest_fixture()
        plain = copy.deepcopy(ood)
        plain['policy'].pop('ood_split')
        self.assertNotEqual(manifest_identity(plain), manifest_identity(ood))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'native_filter_manifest.json'
            path.write_text(json.dumps(plain), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'different native outlier policy'):
                prepare_native_filter(tmp,'strict',[123],False,ood_splitter('native','family'))
            path.write_text(json.dumps(ood), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'different native outlier policy'):
                prepare_native_filter(tmp,'strict',[123],False,cli.iter_train_val_test_splits)

    def test_mixed_cli_changes_native_only_and_preserves_other_splitters(self):
        native = manifest_fixture()
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            argv = ['hera','--model','alignn','--dataset','native','semi','imp2d',
                    '--mode','was_x','hetero','hetero_was','--r','0','--seed','123','--epochs','1',
                    '--native-split','family','--native-outlier-filter','strict',
                    '--semi-quality-filter','physical','--imp2d-quality-filter','physical',
                    '--imp2d-source','db','--run-dir',tmp,'--compact-logs','--atom-init',str(root/'atom_init.json')]
            def impurity(run, dataset, seeds, cv5, splitter, **kwargs):
                self.assertIs(splitter, cli.iter_train_val_test_splits)
                return {'schema':'impurity_physical_manifest_v1','dataset':dataset,'policy':{},'records':[],'splits':[]}
            def prepare(run, profile, seeds, cv5, splitter):
                self.assertEqual(splitter.ood_protocol, protocol_for('native','family'))
                return native
            with patch('sys.argv',argv), patch.object(cli,'prepare_native_filter',side_effect=prepare), \
                    patch.object(cli,'prepare_impurity_filter',side_effect=impurity) as imp, \
                    patch.object(cli,'load_dataset',return_value=([1],[1],None,None,[0])), \
                    patch.object(cli,'train_single_mode',return_value=[.2]) as train, redirect_stdout(io.StringIO()):
                cli.main()
            self.assertEqual(imp.call_count,2)
            self.assertEqual(train.call_count,9)
            for call in train.call_args_list:
                config, dataset_name = call.args[1], call.kwargs['dataset_name']
                self.assertEqual('ood_split' in config, dataset_name=='native')

    def test_invalid_ids_too_few_groups_and_cv_are_rejected(self):
        for ids, cv in [(['not-a-native-id'],False), (source_ids()[:3],False), (source_ids(),True)]:
            with self.assertRaises(ValueError):
                list(iter_ood_splits(ids,[0]*len(ids),[123],cv5=cv,dataset='native',grouping='family'))


if __name__ == '__main__':
    unittest.main()
