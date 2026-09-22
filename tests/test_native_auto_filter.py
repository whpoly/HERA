"""One-command native preprocessing preserves frozen experiment identity."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from pymatgen.core import Lattice, Structure

from HERA import main as cli
from HERA.data import native_preprocessing as preprocessing
from HERA.data.native_filter import manifest_identity


class NativeAutoFilterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.data = Path(self.temp.name)/'data'
        self.data.mkdir()
        self.run = Path(self.temp.name)/'run'
        structure = Structure(Lattice.cubic(10), ['Zn','O','Zn'],
                              [[.1,.1,.1],[.3,.1,.1],[.3,.3,.1]], labels=['Zn0','O1','Zn2'])
        rows = []
        for i in range(20):
            sid = f'1-ZnO-Zn_i_A-POSCAR{i}-Neutral.cif'
            structure.to(filename=self.data/sid, fmt='cif')
            rows.append([sid, 100. if i == 19 else 1.+.01*i])
        pd.DataFrame(rows).to_csv(self.data/'id_prop_A_rich.csv',index=False,header=False)

    def prepare(self, profile='strict', seeds=None, cv5=False):
        with redirect_stdout(io.StringIO()):
            return preprocessing.prepare_native_filter(self.run,profile,seeds or [123],cv5,
                                                       cli.iter_train_val_test_splits,data_dir=self.data)

    def test_first_load_builds_once_and_second_reuses_exact_manifest(self):
        with patch.object(preprocessing,'build_native_manifest',wraps=preprocessing.build_native_manifest) as build:
            first = self.prepare()
            second = self.prepare()
        self.assertEqual(build.call_count,1)
        self.assertEqual(manifest_identity(first),manifest_identity(second))
        self.assertTrue((self.run/'native_filter_manifest.json').is_file())
        self.assertTrue((self.run/'native_filter_seed123_removed.csv').is_file())
        self.assertTrue((self.run/'native_filter_summary.csv').is_file())
        self.assertEqual(first['policy'],preprocessing.native_filter_policy('strict'))

    def test_changed_excluded_raw_file_cannot_silently_reuse_cache(self):
        manifest = self.prepare()
        removed = [r['source_id'] for items in manifest['splits'][0]['excluded'].values() for r in items]
        self.assertTrue(removed)
        path = self.data/removed[0]
        path.write_bytes(path.read_bytes()+b'\n')
        before = (self.run/'native_filter_manifest.json').read_bytes()
        with self.assertRaisesRegex(ValueError,'CIF changed'):
            self.prepare()
        self.assertEqual((self.run/'native_filter_manifest.json').read_bytes(),before)

    def test_changed_labels_and_filter_protocol_are_rejected(self):
        self.prepare()
        with self.assertRaisesRegex(ValueError,'different native outlier policy'):
            self.prepare(profile='standard')
        with self.assertRaisesRegex(ValueError,'lacks requested splits'):
            self.prepare(seeds=[42])
        with self.assertRaisesRegex(ValueError,'CV protocol'):
            self.prepare(cv5=True)
        table = self.data/'id_prop_A_rich.csv'
        frame = pd.read_csv(table,header=None)
        frame.iloc[0,1] += 1
        frame.to_csv(table,index=False,header=False)
        with self.assertRaisesRegex(ValueError,'target changed'):
            self.prepare()

    def test_main_automatically_prepares_once_before_loading_multiple_modes(self):
        root = Path(__file__).resolve().parents[1]
        argv = ['hera','--model','alignn','--dataset','native','--mode','attention_was','hetero_was',
                '--r','0','--epochs','1','--seed','123','--compact-logs',
                '--run-dir',str(self.run),'--atom-init',str(root/'atom_init.json'),
                '--native-outlier-filter','strict']
        real_prepare = preprocessing.prepare_native_filter
        def prepare(*args):
            return real_prepare(*args,data_dir=self.data)
        def load(*args,**kwargs):
            saved = json.loads((self.run/'native_filter_manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(kwargs['native_filter_manifest'],saved)
            return None,[],[],None,[]
        with patch('sys.argv',argv), patch.object(cli,'prepare_native_filter',side_effect=prepare) as prep, \
                patch.object(cli,'load_dataset',side_effect=load) as loader, \
                patch.object(cli,'train_single_mode',return_value=[.2]) as train, redirect_stdout(io.StringIO()):
            cli.main()
        self.assertEqual(prep.call_count,1)
        self.assertEqual(loader.call_count,2)
        self.assertEqual(train.call_count,2)
        for call in train.call_args_list:
            self.assertEqual(call.args[1]['native_data_filter'],manifest_identity(call.kwargs['native_filter_manifest']))


if __name__ == '__main__':
    unittest.main()
