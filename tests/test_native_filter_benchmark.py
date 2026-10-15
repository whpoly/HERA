"""Verify training-only filtering and fair fixed-population evaluation."""
import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from HERA.data.native_filter import manifest_identity, validate_manifest, filtered_splits
from HERA.data.native_preprocessing import native_filter_policy
from HERA.scripts import run_native_filter_benchmark as benchmark


class NativeFilterBenchmarkTests(unittest.TestCase):
    def fixture(self):
        ids = [f'1-BN-B_i_A-POSCAR{i}-Neutral.cif' for i in range(9)]
        rows = [dict(source_id=sid, target=target, sha256='fake', quality_reasons=[])
                for sid, target in zip(ids, (-.5, 14.999, 15., 18., 100., 10., 25., 12., 20.))]
        original = dict(train=ids[:5], val=ids[5:7], test=ids[7:])
        kept = dict(train=ids[:4], val=ids[5:7], test=ids[7:])
        excluded = dict(train=[dict(source_id=ids[4], reasons=['existing_strict_reason'])], val=[], test=[])
        manifest = dict(schema='native_outliers_v1', dataset='native', cv5=False,
                        policy=native_filter_policy('strict'), records=rows,
                        splits=[dict(seed=123, logger_id=123, display='123', explain_id='seed_123',
                                     original=original, kept=kept, excluded=excluded,
                                     train_fences={'fixed':1}, family_train_fences={'fixed':2})])
        return validate_manifest(manifest)

    def checkpoint(self, manifest, mode='hetero_was'):
        model = dict(embedding_size=64, train_batch_size=16, test_batch_size=1,
                     hetero_pooling='defect_energy_mean', hetero_dd_mode='drop',
                     hetero_relation_mode='shared_residual', hetero_relation_rank=8,
                     hetero_node_norm='layernorm', hetero_feature_norm='layernorm', local_radius=0)
        split = manifest['splits'][0]
        return dict(dataset_name='native', mode=mode, epochs=500, epochs_completed=500,
                    early_stopping_patience=0, best_epoch=10, best_val_mae=1., test_mae=1.,
                    config=dict(task='alignn_' + mode, native_preprocessing='reference_v1',
                                native_data_filter=manifest_identity(manifest), model=model),
                    scaler=dict(mean=2., std=3.),
                    split=dict(seed=123, **{p+'_sources': [dict(source_id=s) for s in split['kept'][p]]
                                           for p in ('train','val','test')}))

    def test_boundary_negative_labels_and_frozen_heldout_membership(self):
        base = self.fixture()
        before = copy.deepcopy(base)
        scoped = benchmark.derive_training_scope(base)
        self.assertEqual(base, before)
        a, b = base['splits'][0], scoped['splits'][0]
        self.assertEqual(b['kept']['train'], a['kept']['train'][:2])
        for p in ('val', 'test'):
            self.assertEqual(b['kept'][p], a['kept'][p])
            self.assertEqual(b['excluded'][p], a['excluded'][p])
        for k in ('original', 'train_fences', 'family_train_fences'):
            self.assertEqual(a[k], b[k])
        self.assertEqual(b['excluded']['train'][-1]['reasons'], ['existing_strict_reason'])
        self.assertNotEqual(manifest_identity(base), manifest_identity(scoped))

    def test_heldout_labels_cannot_change_training_selection(self):
        base = self.fixture()
        first = benchmark.derive_training_scope(base)
        for row in base['records'][5:]:
            row['target'] += 1000
        second = benchmark.derive_training_scope(base)
        self.assertEqual(first['splits'], second['splits'])

    def test_manifest_loader_preserves_train_and_heldout_order(self):
        from types import SimpleNamespace
        manifest = benchmark.derive_training_scope(self.fixture())
        data = [SimpleNamespace(source_id=r['source_id']) for r in manifest['records']]
        values = [r['target'] for r in manifest['records']]
        result = list(filtered_splits(data, values, manifest, [123]))[0]
        for part in ('train','val','test'):
            self.assertEqual([r.source_id for r in result[part+'_X']], manifest['splits'][0]['kept'][part])
        self.assertEqual(result['train_y'], [-.5, 14.999])

    def test_rejects_empty_training_and_non_strict_parent(self):
        base = self.fixture()
        base['policy']['unplanned'] = True
        with self.assertRaises(ValueError):
            benchmark.derive_training_scope(base)
        base = self.fixture()
        for row in base['records'][:4]:
            row['target'] = 15.
        with self.assertRaisesRegex(ValueError, 'nonempty'):
            benchmark.derive_training_scope(base)

    def test_checkpoint_identity_budget_order_and_filter(self):
        manifest = self.fixture()
        original = self.checkpoint(manifest)
        for kind in ('budget', 'order', 'filter', 'width', 'readout', 'scaler'):
            checkpoint = copy.deepcopy(original)
            if kind == 'budget':
                checkpoint['epochs_completed'] = 499
            elif kind == 'order':
                checkpoint['split']['test_sources'].reverse()
            elif kind == 'filter':
                checkpoint['config']['native_data_filter']['sha256'] = 'wrong'
            elif kind == 'width':
                checkpoint['config']['model']['embedding_size'] = 32
            elif kind == 'readout':
                checkpoint['config']['model']['hetero_pooling'] = 'global_mean'
            else:
                checkpoint['scaler']['std'] = 0.
            with self.assertRaises(ValueError, msg=kind):
                benchmark.validate_checkpoint(checkpoint, manifest, 'hetero_was', 123, 500, 16)

    def test_dry_run_has_no_side_effects_and_commands_only_change_data_and_output(self):
        with patch.object(benchmark, 'prepare') as prepare, patch.object(benchmark.subprocess, 'run') as run, \
                patch.object(benchmark, 'evaluate') as evaluate, redirect_stdout(io.StringIO()) as output:
            benchmark.main(['--dry-run'])
        prepare.assert_not_called()
        run.assert_not_called()
        evaluate.assert_not_called()
        self.assertIn('--mode hetero_was was_x', output.getvalue())
        args = benchmark.parse_args([])
        commands = [benchmark.build_command(args, variant) for variant in benchmark.VARIANTS]
        for cmd in commands:
            for option in ('--run-dir', '--native-filter-manifest'):
                cmd[cmd.index(option)+1] = '<varies>'
        self.assertEqual(*commands)

    def test_fixed_scopes_report_mae_rmse_and_changed_training_scaler(self):
        base = self.fixture()
        manifests = dict(strict=base, train_lt15=benchmark.derive_training_scope(base))
        with tempfile.TemporaryDirectory() as directory:
            args = benchmark.parse_args(['--run-dir', directory])
            def load(path, device):
                variant = path.relative_to(args.run_dir).parts[0]
                mode = 'hetero_was' if path.parent.name == 'hetero_was_no_dd' else 'was_x'
                checkpoint = self.checkpoint(manifests[variant], mode)
                if variant == 'train_lt15':
                    checkpoint['scaler'] = dict(mean=1., std=2.)
                return checkpoint
            def predict(checkpoint, sources, records, device):
                return [dict(split=part, source_id=sid, target=records[sid]['target'],
                             source_target=records[sid]['target'], prediction=records[sid]['target']+1)
                        for part, ids in sources.items() for sid in ids]
            for variant in benchmark.VARIANTS:
                for mode in ('hetero_was_no_dd','was_x'):
                    (args.run_dir/variant/'alignn/native'/mode).mkdir(parents=True)
            with patch('HERA.training.trainer.load_trusted_checkpoint', side_effect=load), \
                    patch.object(benchmark, 'predict', side_effect=predict), \
                    patch.object(benchmark, 'sha256', return_value='hash'), redirect_stdout(io.StringIO()):
                benchmark.evaluate(args, manifests)
            result = json.loads((args.run_dir/'filter_evaluation.json').read_text())
            self.assertEqual(len(result['metrics']), 24)
            for row in result['metrics']:
                self.assertEqual(row['n'], 2 if row['scope'] == 'strict_all' else 1)
                self.assertEqual(row['mae_ev'], 1.)
                self.assertEqual(row['rmse_ev'], 1.)
                self.assertEqual(row['train_count'], 4 if row['variant'] == 'strict' else 2)

    def test_scores_empty_scope_and_squared_tail(self):
        self.assertEqual(benchmark.score([]), dict(n=0, mae_ev=None, rmse_ev=None))
        metrics = benchmark.score([dict(target=0, prediction=0), dict(target=0, prediction=4)])
        self.assertEqual(metrics['mae_ev'], 2.)
        self.assertAlmostEqual(metrics['rmse_ev'], 8**.5)

    def test_frozen_protocol_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'protocol.json'
            benchmark.save_json(path, dict(epochs=500), frozen=True)
            with self.assertRaisesRegex(ValueError, 'Different experiment identity'):
                benchmark.save_json(path, dict(epochs=1), frozen=True)
            self.assertEqual(json.loads(path.read_text()), dict(epochs=500))

    def test_family_rule_uses_train_only_and_keeps_high_energy_stable_family(self):
        base = self.fixture()
        for row, value in zip(base['records'], [1.,1.,1.,9.]):
            row['target'] = value
        result = benchmark.derive_family_mad(base)
        self.assertEqual(result['splits'][0]['kept']['train'], base['splits'][0]['kept']['train'][:3])
        for part in ('val','test'):
            self.assertEqual(result['splits'][0]['kept'][part], base['splits'][0]['kept'][part])
        for row in base['records'][5:]:
            row['target'] += 10000.
        self.assertEqual(result['splits'], benchmark.derive_family_mad(base)['splits'])
        # Stable high energy is not an outlier. No global 15 eV cap is used.
        for row in base['records'][:4]:
            row['target'] = 20.
        self.assertEqual(benchmark.derive_family_mad(base)['splits'][0]['kept'], base['splits'][0]['kept'])

    def test_family_rule_retains_boundary_and_small_or_unseen_families(self):
        base = self.fixture()
        for row, value in zip(base['records'], [1.,1.,1.,2.]):
            row['target'] = value
        self.assertEqual(benchmark.derive_family_mad(base)['splits'][0]['kept'], base['splits'][0]['kept'])
        self.assertFalse(benchmark.family_flagged('unseen',1e9,{}))
        from HERA.data.native_quality import configuration_family
        family = configuration_family(base['records'][0]['source_id'])
        self.assertFalse(benchmark.family_flagged(base['records'][0]['source_id'],1e9,
                                                 {family:dict(status='insufficient_train_samples',n_train=3)}))

    def test_outlier_suite_is_full_factorial_with_separate_default_directory(self):
        args = benchmark.parse_args(['--suite','outlier_control'])
        self.assertEqual(args.run_dir.name,'native_outlier_control_v1')
        specs = benchmark.variant_specs(args)
        self.assertEqual({(s['training_filter'],s['point_loss']) for s in specs.values()},
                         {('strict','mse'),('strict','huber_ev'),('family_mad','mse'),('family_mad','huber_ev')})
        for name,spec in specs.items():
            command = benchmark.build_command(args,name)
            self.assertEqual('--point-loss' in command, spec['point_loss']=='huber_ev')
            if spec['point_loss']=='huber_ev':
                self.assertEqual(command[command.index('--huber-delta-ev')+1],'1.0')

    def test_checkpoint_requires_declared_loss_and_physical_delta(self):
        base = self.fixture()
        mse = self.checkpoint(base)
        huber = copy.deepcopy(mse)
        huber['config']['optim'] = dict(point_loss='huber_ev',huber_delta_ev=1.)
        common_mse = benchmark.validate_checkpoint(mse,base,'hetero_was',123,500,16)
        common_huber = benchmark.validate_checkpoint(huber,base,'hetero_was',123,500,16,'huber_ev')
        self.assertEqual(common_mse,common_huber)
        with self.assertRaisesRegex(ValueError,'point loss'):
            benchmark.validate_checkpoint(huber,base,'hetero_was',123,500,16)
        huber['config']['optim']['huber_delta_ev'] = 2.
        with self.assertRaisesRegex(ValueError,'transition'):
            benchmark.validate_checkpoint(huber,base,'hetero_was',123,500,16,'huber_ev')

    def test_outlier_evaluation_reports_all_eight_runs_and_shared_subgroups(self):
        base = self.fixture()
        for row, value in zip(base['records'], [1.,1.,1.,9.]):
            row['target'] = value
        # Keep one flagged and one unflagged sample in each held-out split.
        base['records'][5]['target'] = base['records'][7]['target'] = 1.
        scoped = benchmark.derive_family_mad(base)
        with tempfile.TemporaryDirectory() as directory:
            args = benchmark.parse_args(['--suite','outlier_control','--run-dir',directory])
            specs = benchmark.variant_specs(args)
            manifests = {name:base if spec['training_filter']=='strict' else scoped for name,spec in specs.items()}
            def load(path, device):
                name = path.relative_to(args.run_dir).parts[0]
                mode = 'hetero_was' if path.parent.name=='hetero_was_no_dd' else 'was_x'
                checkpoint = self.checkpoint(manifests[name], mode)
                if specs[name]['point_loss']=='huber_ev':
                    checkpoint['config']['optim'] = dict(point_loss='huber_ev',huber_delta_ev=1.)
                if specs[name]['training_filter']=='family_mad':
                    checkpoint['scaler'] = dict(mean=1.,std=.1)
                return checkpoint
            def predict(checkpoint, sources, records, device):
                return [dict(split=part, source_id=sid, target=records[sid]['target'],
                             source_target=records[sid]['target'],prediction=records[sid]['target']+1)
                        for part,ids in sources.items() for sid in ids]
            for name in specs:
                for mode in ('hetero_was_no_dd','was_x'):
                    (args.run_dir/name/'alignn/native'/mode).mkdir(parents=True)
            with patch('HERA.training.trainer.load_trusted_checkpoint',side_effect=load), \
                    patch.object(benchmark,'predict',side_effect=predict), \
                    patch.object(benchmark,'sha256',return_value='hash'), redirect_stdout(io.StringIO()):
                benchmark.evaluate(args,manifests)
            result = json.loads((args.run_dir/'filter_evaluation.json').read_text())
            self.assertEqual(len(result['checkpoints']),8)
            self.assertEqual(len(result['metrics']),48)
            self.assertEqual({r['scope'] for r in result['metrics']}, {'strict_all','family_high','family_not_flagged'})
            for row in result['metrics']:
                self.assertEqual(row['n'],2 if row['scope']=='strict_all' else 1)
                self.assertEqual(row['mae_ev'],1.)
                self.assertEqual(row['rmse_ev'],1.)
                self.assertEqual(row['train_count'],4 if specs[row['variant']]['training_filter']=='strict' else 3)

    def all_split_fixture(self):
        base = self.fixture()
        for row, value in zip(base['records'], [1.,1.,1.,9.]):
            row['target'] = value
        base['records'][5]['target'] = base['records'][7]['target'] = 1.
        return base

    def test_all_split_filter_removes_train_val_test_with_one_frozen_fence(self):
        from types import SimpleNamespace
        base = self.all_split_fixture()
        before = copy.deepcopy(base)
        scoped = benchmark.derive_family_mad(base,('train','val','test'))
        train_only = benchmark.derive_family_mad(base)
        split = scoped['splits'][0]
        self.assertEqual(base,before)
        self.assertEqual(split['training_family_mad_fences'],train_only['splits'][0]['training_family_mad_fences'])
        self.assertEqual(split['kept'],dict(train=base['splits'][0]['kept']['train'][:3],
                                          val=base['splits'][0]['kept']['val'][:1],
                                          test=base['splits'][0]['kept']['test'][:1]))
        self.assertEqual(scoped['policy']['family_mad_filter']['applies_to'],['train','val','test'])
        self.assertEqual(split['original'],base['splits'][0]['original'])
        structures=[SimpleNamespace(source_id=r['source_id']) for r in base['records']]
        targets=[r['target'] for r in base['records']]
        loaded=list(filtered_splits(structures,targets,scoped,[123]))[0]
        for part in ('train','val','test'):
            self.assertEqual([s.source_id for s in loaded[part+'_X']],split['kept'][part])
            self.assertEqual(split['excluded'][part][-1]['reasons'],
                             ['existing_strict_reason'] if part=='train' else ['above_strict_train_family_mad_upper'])

    def test_all_split_heldout_labels_change_only_their_own_selection(self):
        base=self.all_split_fixture()
        before=benchmark.derive_family_mad(base,('train','val','test'))['splits'][0]
        base['records'][6]['target']=1.
        after=benchmark.derive_family_mad(base,('train','val','test'))['splits'][0]
        self.assertEqual(before['training_family_mad_fences'],after['training_family_mad_fences'])
        for part in ('train','test'):
            self.assertEqual(before['kept'][part],after['kept'][part])
            self.assertEqual(before['excluded'][part],after['excluded'][part])
        self.assertEqual(len(before['kept']['val']),1)
        self.assertEqual(len(after['kept']['val']),2)

    def test_all_split_filter_rejects_empty_heldout_and_old_checkpoint(self):
        base=self.all_split_fixture()
        train_only=benchmark.derive_family_mad(base)
        scoped=benchmark.derive_family_mad(base,('train','val','test'))
        checkpoint=self.checkpoint(train_only)
        with self.assertRaisesRegex(ValueError,'val sources'):
            benchmark.validate_checkpoint(checkpoint,scoped,'hetero_was',123,500,16)
        base['records'][5]['target']=1000.
        with self.assertRaisesRegex(ValueError,'nonempty'):
            benchmark.derive_family_mad(base,('train','val','test'))

    def test_filtered_dataset_suite_has_only_new_cohort_and_separate_directory(self):
        args=benchmark.parse_args(['--suite','filtered_dataset'])
        self.assertEqual(args.run_dir.name,'native_filter_then_split_v2')
        specs=benchmark.variant_specs(args)
        self.assertEqual(len(specs),2)
        self.assertEqual({s['training_filter'] for s in specs.values()},{'prefiltered'})
        self.assertEqual({s['point_loss'] for s in specs.values()},{'mse','huber_ev'})
        with patch.object(benchmark,'prepare') as prepare,patch.object(benchmark.subprocess,'run') as run, \
                redirect_stdout(io.StringIO()) as output:
            benchmark.main(['--suite','filtered_dataset','--dry-run'])
        prepare.assert_not_called()
        run.assert_not_called()
        self.assertIn('newly filtered validation/test',output.getvalue())
        self.assertIn('THEN create fresh seeded 60/20/20 splits',output.getvalue())

    def test_population_cap_cli_is_explicit_finite_and_uses_new_directory(self):
        args = benchmark.parse_args(['--suite', 'filtered_dataset', '--max-energy-ev', '15'])
        self.assertEqual(args.run_dir.name, 'native_filter_then_split_le15_v3')
        for argv in (['--max-energy-ev', '15'],
                     ['--suite', 'filtered_dataset', '--max-energy-ev', 'nan'],
                     ['--suite', 'filtered_dataset', '--max-energy-ev', 'inf']):
            from contextlib import redirect_stderr
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                benchmark.parse_args(argv)
        with patch.object(benchmark, 'prepare') as prepare, redirect_stdout(io.StringIO()) as output:
            benchmark.main(['--suite', 'filtered_dataset', '--max-energy-ev', '15', '--dry-run'])
        prepare.assert_not_called()
        self.assertIn('E <= 15 eV before splitting; no lower bound', output.getvalue())

    def test_evaluation_and_checkpoint_cannot_mix_capped_and_uncapped_populations(self):
        from HERA.data.native_population import screen_population, split_population
        from HERA.tests.test_native_population import energy_cap_fixture
        from HERA.main import iter_train_val_test_splits
        rows = energy_cap_fixture()
        uncapped = split_population(screen_population(rows), [123], iter_train_val_test_splits)
        capped = split_population(screen_population(rows, 15.), [123], iter_train_val_test_splits)
        for cap_args, wrong in ((['--max-energy-ev', '15'], uncapped), ([], capped)):
            args = benchmark.parse_args(['--suite', 'filtered_dataset'] + cap_args)
            with self.assertRaisesRegex(ValueError, 'Evaluation population policy differs'):
                benchmark.evaluate(args, {name:wrong for name in benchmark.variant_specs(args)})
        with self.assertRaisesRegex(ValueError, 'sources differ'):
            benchmark.validate_checkpoint(self.checkpoint(uncapped), capped, 'hetero_was', 123, 500, 16)

    def test_poscar0_cli_uses_separate_directory_and_requires_global_suite(self):
        args = benchmark.parse_args(['--suite','filtered_dataset','--max-energy-ev','15','--exclude-poscar0'])
        self.assertEqual(args.run_dir.name, 'native_filter_then_split_le15_no_poscar0_v4')
        args = benchmark.parse_args(['--suite','filtered_dataset','--exclude-poscar0'])
        self.assertEqual(args.run_dir.name, 'native_filter_then_split_no_poscar0_v4')
        from contextlib import redirect_stderr
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            benchmark.parse_args(['--exclude-poscar0'])
        with patch.object(benchmark,'prepare') as prepare, patch.object(benchmark.subprocess,'run') as run, \
                redirect_stdout(io.StringIO()) as output:
            benchmark.main(['--suite','filtered_dataset','--max-energy-ev','15','--exclude-poscar0','--dry-run'])
        prepare.assert_not_called()
        run.assert_not_called()
        self.assertIn('exclude all -POSCAR0-', output.getvalue())

    def test_poscar0_evaluation_and_checkpoint_reject_previous_cohort(self):
        from HERA.data.native_population import screen_population, split_population
        from HERA.tests.test_native_population import poscar0_fixture
        from HERA.main import iter_train_val_test_splits
        rows = poscar0_fixture()
        old = split_population(screen_population(rows, 15.), [123], iter_train_val_test_splits)
        new = split_population(screen_population(rows, 15., exclude_poscar0=True), [123], iter_train_val_test_splits)
        for flags, wrong in ((['--exclude-poscar0'], old), ([], new)):
            args = benchmark.parse_args(['--suite','filtered_dataset','--max-energy-ev','15']+flags)
            with self.assertRaisesRegex(ValueError, 'Evaluation population policy differs'):
                benchmark.evaluate(args, {name:wrong for name in benchmark.variant_specs(args)})
        with self.assertRaisesRegex(ValueError, 'sources differ'):
            benchmark.validate_checkpoint(self.checkpoint(old), new, 'hetero_was', 123, 500, 16)

    def test_filtered_dataset_evaluation_uses_filtered_checkpoint_metrics(self):
        from HERA.data.native_population import screen_population, split_population
        from HERA.main import iter_train_val_test_splits
        rows=self.fixture()['records']
        for index,row in enumerate(rows):
            row.update(target=100. if index==4 else 1., material='BN', defect_type='interstitial',
                       added_z=5,removed_z=0)
        scoped=split_population(screen_population(rows),[123],iter_train_val_test_splits)
        counts={p:len(ids) for p,ids in scoped['splits'][0]['kept'].items()}
        with tempfile.TemporaryDirectory() as directory:
            args=benchmark.parse_args(['--suite','filtered_dataset','--run-dir',directory])
            specs=benchmark.variant_specs(args)
            manifests={name:scoped for name in specs}
            def load(path,device):
                name=path.relative_to(args.run_dir).parts[0]
                mode='hetero_was' if path.parent.name=='hetero_was_no_dd' else 'was_x'
                checkpoint=self.checkpoint(scoped,mode)
                if specs[name]['point_loss']=='huber_ev':
                    checkpoint['config']['optim']=dict(point_loss='huber_ev',huber_delta_ev=1.)
                return checkpoint
            def predict(checkpoint,sources,records,device):
                self.assertEqual({p:len(ids) for p,ids in sources.items()},{p:counts[p] for p in ('val','test')})
                return [dict(split=part,source_id=sid,target=records[sid]['target'],
                             source_target=records[sid]['target'],prediction=records[sid]['target']+1)
                        for part,ids in sources.items() for sid in ids]
            for name in specs:
                for mode in ('hetero_was_no_dd','was_x'):
                    (args.run_dir/name/'alignn/native'/mode).mkdir(parents=True)
            with patch('HERA.training.trainer.load_trusted_checkpoint',side_effect=load), \
                    patch.object(benchmark,'predict',side_effect=predict), \
                    patch.object(benchmark,'sha256',return_value='hash'),redirect_stdout(io.StringIO()):
                benchmark.evaluate(args,manifests)
            result=json.loads((args.run_dir/'filter_evaluation.json').read_text())
            self.assertEqual(len(result['checkpoints']),4)
            self.assertEqual(len(result['metrics']),8)
            self.assertIn('filtered validation MAE',result['selection'])
            for row in result['metrics']:
                self.assertEqual(row['scope'],'filtered_all')
                self.assertEqual(row['train_count'],counts['train'])
                self.assertEqual(row['n'],counts[row['split']])
                self.assertEqual(row['mae_ev'],1.)
                self.assertEqual(row['rmse_ev'],1.)
            report=(args.run_dir/'filter_comparison.md').read_text()
            self.assertIn('new filtered dataset',report)
            self.assertIn('BEFORE creating fresh 60/20/20 splits',report)
            self.assertNotIn('full strict validation MAE',report)
            broken={name:copy.deepcopy(manifest) for name,manifest in manifests.items()}
            broken['population_huber']['records'][0]['target']+=.01
            with self.assertRaisesRegex(ValueError,'identical frozen manifest'):
                benchmark.evaluate(args,broken)

    def test_filtered_dataset_rejects_previous_split_then_filter_protocol(self):
        scoped=benchmark.derive_family_mad(self.all_split_fixture(),('train','val','test'))
        args=benchmark.parse_args(['--suite','filtered_dataset'])
        with self.assertRaisesRegex(ValueError,'before splitting'):
            benchmark.evaluate(args,{name:scoped for name in benchmark.variant_specs(args)})


if __name__ == '__main__':
    unittest.main()
