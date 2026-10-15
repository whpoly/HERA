"""Regression checks for globally frozen curation BEFORE random splitting."""
import copy
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

from HERA.data.native_filter import filtered_splits, validate_manifest, verify_source_table
from HERA.data.native_population import (prepare_population, screen_population, split_population,
                                         validate_population)
from HERA.main import iter_train_val_test_splits
from HERA.scripts import run_native_filter_benchmark as benchmark


def records_fixture():
    values = [1.]*10 + [2., 2.5, 100., 1.]
    return [dict(source_id=f'1-BN-B_i_A-POSCAR{i}-Neutral.cif', target=value,
                 material='BN', defect_type='interstitial', added_z=5, removed_z=0,
                 sha256=hashlib.sha256(b'unchanged').hexdigest(),
                 quality_reasons=['invalid_structure'] if i==13 else []) for i,value in enumerate(values)]


def energy_cap_fixture():
    rows = records_fixture()[:9]
    for index, (row, value) in enumerate(zip(rows, [-.5, 1., 5., 10., 14.999, 15., 15.001, 18., 20.])):
        row.update(source_id=f'{index+1}-BN-B_i_A-POSCAR0-Neutral.cif', target=value)
    return rows


def poscar0_fixture():
    rows = energy_cap_fixture()
    for row, frame in zip(rows, ('0', '1', '10', '100', '01', '5', '0', '7', '8')):
        row['source_id'] = row['source_id'].replace('-POSCAR0-', f'-POSCAR{frame}-')
    return rows


class NativePopulationTests(unittest.TestCase):
    def test_poscar0_exclusion_follows_existing_filters_and_matches_exact_token(self):
        rows = poscar0_fixture()
        original = copy.deepcopy(rows)
        for cap in (None, 15.):
            with self.subTest(cap=cap):
                before = screen_population(rows, cap)
                after = screen_population(rows, cap, exclude_poscar0=True)
                self.assertEqual(rows, original)
                self.assertEqual(after['pre_poscar0_kept'], before['kept'])
                self.assertEqual(after['kept'], [sid for sid in before['kept'] if '-POSCAR0-' not in sid])
                for key in ('strict_kept', 'group_fences', 'family_fences', 'family_mad_fences'):
                    self.assertEqual(after[key], before[key])
                # POSCAR1, POSCAR10, POSCAR100 and POSCAR01 must remain untouched.
                self.assertTrue(all(r['source_id'] in after['kept'] for r in rows[1:6]))
                reasons = {r['source_id']:r['reasons'] for r in after['excluded']}
                self.assertEqual(reasons[rows[0]['source_id']], ['excluded_population_poscar0'])
                if cap is not None:
                    self.assertEqual(reasons[rows[6]['source_id']], ['above_population_energy_upper_bound'])
                manifests = [split_population(after, [seed], iter_train_val_test_splits) for seed in (123, 456)]
                self.assertEqual(manifests[0]['population_filter'], manifests[1]['population_filter'])
                for manifest in manifests:
                    ids = sum(manifest['splits'][0]['kept'].values(), [])
                    self.assertEqual(set(ids), set(after['kept']))
                    self.assertFalse(any('-POSCAR0-' in sid for sid in ids))
                bad = copy.deepcopy(after)
                bad['kept'].insert(0, rows[0]['source_id'])
                with self.assertRaisesRegex(ValueError, 'POSCAR0 exclusion'):
                    validate_population(bad)
                if cap is not None:
                    bad = copy.deepcopy(after)
                    bad['pre_poscar0_kept'].append(rows[6]['source_id'])
                    with self.assertRaisesRegex(ValueError, 'inclusive energy upper bound'):
                        validate_population(bad)

    def test_poscar0_cache_rejects_switching_population_and_detects_removed_file_changes(self):
        rows = poscar0_fixture()
        with tempfile.TemporaryDirectory() as directory:
            data, output = Path(directory)/'data', Path(directory)/'output'
            data.mkdir()
            pd.DataFrame([(r['source_id'], r['target']) for r in rows]).to_csv(
                data/'id_prop_A_rich.csv', header=False, index=False)
            for row in rows:
                (data/row['source_id']).write_bytes(b'unchanged')
            with patch('HERA.data.native_population.inspect_source', side_effect=rows), redirect_stdout(io.StringIO()):
                before = prepare_population(output, data, 15., exclude_poscar0=True)
                self.assertEqual(before, prepare_population(output, data, 15., exclude_poscar0=True))
                with self.assertRaisesRegex(ValueError, 'policy differs'):
                    prepare_population(output, data, 15.)
                (data/rows[0]['source_id']).write_bytes(b'changed excluded POSCAR0')
                with self.assertRaisesRegex(ValueError, 'CIF changed'):
                    prepare_population(output, data, 15., exclude_poscar0=True)

    def test_poscar0_benchmark_freezes_same_new_split_for_all_four_runs(self):
        population = screen_population(poscar0_fixture(), 15., exclude_poscar0=True)
        with tempfile.TemporaryDirectory() as directory:
            args = benchmark.parse_args(['--suite','filtered_dataset','--max-energy-ev','15',
                                         '--exclude-poscar0','--run-dir',directory,'--prepare-only'])
            with patch('HERA.data.native_population.prepare_population', return_value=population) as prepare, \
                    redirect_stdout(io.StringIO()):
                manifests = benchmark.prepare(args)
            self.assertEqual(prepare.call_args.kwargs, dict(max_energy_ev=15., exclude_poscar0=True))
            self.assertEqual(manifests['population_mse'], manifests['population_huber'])
            protocol = json.loads((args.run_dir/'filter_protocol.json').read_text())
            self.assertEqual(protocol['schema'], 'native_filter_then_split_benchmark_v4')
            self.assertTrue(protocol['population_policy']['exclude_poscar0'])
            args.exclude_poscar0 = False
            with self.assertRaisesRegex(ValueError, 'Different experiment identity'):
                benchmark.prepare(args)

    def test_energy_cap_preserves_boundary_and_negative_before_new_splits(self):
        rows = energy_cap_fixture()
        uncapped = screen_population(rows)
        capped = screen_population(rows, max_energy_ev=15.)
        self.assertEqual(capped['uncapped_kept'], uncapped['kept'])
        for key in ('strict_kept', 'group_fences', 'family_fences', 'family_mad_fences'):
            self.assertEqual(capped[key], uncapped[key])
        self.assertEqual(capped['kept'], [r['source_id'] for r in rows[:6]])
        self.assertEqual([r['reasons'] for r in capped['excluded']],
                         [['above_population_energy_upper_bound']]*3)
        for seed in (123, 456):
            manifest = split_population(capped, [seed], iter_train_val_test_splits)
            membership = manifest['splits'][0]['kept']
            self.assertEqual(set().union(*map(set, membership.values())), set(capped['kept']))
            self.assertEqual({p:len(ids) for p,ids in membership.items()}, dict(train=3, val=1, test=2))
        bad = copy.deepcopy(capped)
        bad['kept'].append(rows[6]['source_id'])
        with self.assertRaisesRegex(ValueError, 'inclusive energy upper bound'):
            validate_population(bad)

    def test_capped_population_cache_rejects_a_different_or_missing_cap(self):
        rows = energy_cap_fixture()
        with tempfile.TemporaryDirectory() as directory:
            data, output = Path(directory)/'data', Path(directory)/'output'
            data.mkdir()
            pd.DataFrame([(r['source_id'], r['target']) for r in rows]).to_csv(
                data/'id_prop_A_rich.csv', header=False, index=False)
            for row in rows:
                (data/row['source_id']).write_bytes(b'unchanged')
            with patch('HERA.data.native_population.inspect_source', side_effect=rows), redirect_stdout(io.StringIO()):
                first = prepare_population(output, data, max_energy_ev=15.)
                self.assertEqual(first, prepare_population(output, data, max_energy_ev=15.))
                for cap in (None, 14.):
                    with self.assertRaisesRegex(ValueError, 'policy differs'):
                        prepare_population(output, data, max_energy_ev=cap)

    def test_global_stages_and_boundary_before_any_split(self):
        rows=records_fixture()
        before=copy.deepcopy(rows)
        population=screen_population(rows)
        self.assertEqual(rows,before)
        self.assertEqual(population['strict_kept'],[r['source_id'] for r in rows[:12]])
        self.assertEqual(population['kept'],[r['source_id'] for r in rows[:11]])
        reasons={r['source_id']:r['reasons'] for r in population['excluded']}
        self.assertEqual(reasons[rows[11]['source_id']],['above_strict_population_family_mad_upper'])
        self.assertIn('energy_above_population_fence',reasons[rows[12]['source_id']])
        self.assertEqual(reasons[rows[13]['source_id']],['invalid_structure'])
        fence=next(iter(population['family_mad_fences'].values()))
        self.assertEqual(fence['n_population'],12)
        self.assertEqual(fence['upper_ev'],2.)

    def test_splitter_receives_only_frozen_retained_ids_and_seed_changes_only_assignment(self):
        population=screen_population(records_fixture())
        frozen=copy.deepcopy(population)
        with patch('HERA.data.native_population.screen_population',side_effect=AssertionError('must not refit')):
            one=split_population(population,[123],iter_train_val_test_splits)
            two=split_population(population,[456],iter_train_val_test_splits)
        self.assertEqual(population,frozen)
        self.assertEqual(one['population_filter'],two['population_filter'])
        self.assertNotEqual(one['splits'][0]['kept'],two['splits'][0]['kept'])
        seen=[]
        def splitter(ids,targets,seeds,cv5):
            seen.extend(ids)
            yield from iter_train_val_test_splits(ids,targets,seeds,cv5=cv5)
        split_population(population,[123],splitter)
        self.assertEqual(seen,population['kept'])
        split=one['splits'][0]
        self.assertEqual({p:len(ids) for p,ids in split['kept'].items()},dict(train=6,val=2,test=3))
        self.assertEqual(set().union(*map(set,split['kept'].values())),set(population['kept']))
        self.assertEqual(split['original'],split['kept'])
        raw=next(iter_train_val_test_splits([r['source_id'] for r in population['records']], [0]*14,[123]))
        old_train=[sid for sid in raw['train_X'] if sid in set(population['kept'])]
        self.assertNotEqual(split['kept']['train'],old_train)

    def test_manifest_loader_uses_new_order_but_verifies_entire_raw_source_table(self):
        population=screen_population(records_fixture())
        manifest=split_population(population,[123],iter_train_val_test_splits)
        rows=population['records']
        verify_source_table(manifest,pd.DataFrame([(r['source_id'],r['target']) for r in rows]))
        retained=set(population['kept'])
        data=[SimpleNamespace(source_id=r['source_id']) for r in rows if r['source_id'] in retained]
        targets=[r['target'] for r in rows if r['source_id'] in retained]
        loaded=list(filtered_splits(data,targets,manifest,[123]))[0]
        by_id={r['source_id']:r['target'] for r in rows}
        for part,ids in manifest['splits'][0]['kept'].items():
            self.assertEqual([r.source_id for r in loaded[part+'_X']],ids)
            self.assertEqual(loaded[part+'_y'],[by_id[sid] for sid in ids])

    def test_curation_partition_and_post_split_removal_are_validated(self):
        population=screen_population(records_fixture())
        bad=copy.deepcopy(population)
        bad['excluded'].pop()
        with self.assertRaisesRegex(ValueError,'exactly account'):
            validate_population(bad)
        manifest=split_population(population,[123],iter_train_val_test_splits)
        split=manifest['splits'][0]
        sid=split['kept']['train'].pop()
        split['excluded']['train'].append(dict(source_id=sid,reasons=['late_deletion']))
        with self.assertRaisesRegex(ValueError,'further exclusions'):
            validate_manifest(manifest)
        tiny=records_fixture()[:2]
        with self.assertRaisesRegex(ValueError,'at least three'):
            screen_population(tiny)

    def test_prepare_freezes_population_and_rejects_modified_excluded_source(self):
        rows=records_fixture()
        with tempfile.TemporaryDirectory() as directory:
            data=Path(directory)/'data'
            output=Path(directory)/'output'
            data.mkdir()
            pd.DataFrame([(r['source_id'],r['target']) for r in rows]).to_csv(data/'id_prop_A_rich.csv',header=False,index=False)
            for row in rows:
                (data/row['source_id']).write_bytes(b'unchanged')
            with patch('HERA.data.native_population.inspect_source',side_effect=rows) as inspect,redirect_stdout(io.StringIO()):
                first=prepare_population(output,data)
                self.assertEqual(inspect.call_count,len(rows))
            frozen=(output/'native_population_filter.json').read_bytes()
            with patch('HERA.data.native_population.inspect_source',side_effect=AssertionError('must reuse')),redirect_stdout(io.StringIO()):
                self.assertEqual(first,prepare_population(output,data))
                self.assertEqual(frozen,(output/'native_population_filter.json').read_bytes())
                (data/rows[-1]['source_id']).write_bytes(b'changed excluded source')
                with self.assertRaisesRegex(ValueError,'CIF changed'):
                    prepare_population(output,data)
            self.assertTrue((output/'filtered_id_prop_A_rich.csv').is_file())

    def test_benchmark_prepares_identical_new_splits_without_old_split_filtering(self):
        population=screen_population(records_fixture())
        with tempfile.TemporaryDirectory() as directory:
            args=benchmark.parse_args(['--suite','filtered_dataset','--run-dir',directory,'--prepare-only'])
            def frozen_population(output,data,max_energy_ev=None,exclude_poscar0=False):
                self.assertIsNone(max_energy_ev)
                self.assertFalse(exclude_poscar0)
                benchmark.save_json(output/'native_population_filter.json',population,frozen=True)
                return population
            with patch('HERA.data.native_population.prepare_population',side_effect=frozen_population), \
                    patch('HERA.data.native_preprocessing.prepare_native_filter',side_effect=AssertionError('old split path')), \
                    redirect_stdout(io.StringIO()):
                manifests=benchmark.prepare(args)
            self.assertEqual(manifests['population_mse'],manifests['population_huber'])
            self.assertFalse((args.run_dir/'strict_reference').exists())
            protocol=json.loads((args.run_dir/'filter_protocol.json').read_text())
            self.assertEqual(protocol['schema'],'native_filter_then_split_benchmark_v2')
            self.assertEqual(protocol['population_policy']['order'],'filter_then_split')
            with patch('HERA.data.native_population.prepare_population') as prepare:
                protocol['schema']='native_filtered_dataset_benchmark_v1'
                benchmark.save_json(args.run_dir/'filter_protocol.json',protocol)
                with self.assertRaisesRegex(ValueError,'Different experiment identity'):
                    benchmark.prepare(args)
                prepare.assert_not_called()


if __name__ == '__main__':
    unittest.main()
