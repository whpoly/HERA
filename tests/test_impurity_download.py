"""Download failures must remain distinct from physical exclusion decisions."""
from pathlib import Path
import tempfile
import unittest

from ase import Atoms
from ase.db import connect
from ase.io import write
import pandas as pd

from HERA.scripts.verify_impurity_download import verify_imp2d, verify_semi


class DownloadVerificationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.folder = self.root/'imp2d/imp2d'
        self.folder.mkdir(parents=True)
        self.output = self.root/'audit'
        self.output.mkdir()
        self.db = self.root/'source.db'
        atoms = Atoms('ZnOH', positions=[[1,1,1],[3,1,1],[2,2,2]],cell=[10]*3,pbc=True)
        for index in range(3):
            sid = f'ZnO_H_ads{index}'
            with connect(str(self.db)) as db:
                db.write(atoms,name=sid,eform=1.,en2=-10.,hostenergy=-10.,dopant_chemical_potential=-1.,
                         conv1=.001,conv2=.2 if index == 2 else .001,converged=True,
                         extension_factor=1.,site=f'ads{index}',depth=2.)
            path = self.folder/(sid+'.cif')
            if index == 1:
                path.write_bytes(b'')
            else:
                write(str(path),atoms,format='cif')
        (self.folder/'id_prop.csv').write_bytes(b'')
        pd.DataFrame([[f'ZnO_H_ads{i}',1.] for i in range(3)]).to_csv(
            self.folder.parent/'prepared_data.csv',index=False,header=False)

    def test_empty_download_not_a_physical_exclusion_and_no_raw_overwrite(self):
        result = verify_imp2d(self.root,self.db,self.output)
        self.assertEqual(result['download_status_counts'],{'readable':2,'empty_download':1})
        self.assertEqual(result['physical_candidate_exclusions_from_available_checks'],1)
        self.assertEqual(result['fully_verified_retained_files'],1)
        self.assertTrue(result['used_alternative_label_table'])
        self.assertFalse(result['training_manifest_created'])
        self.assertEqual((self.folder/'id_prop.csv').read_bytes(),b'')
        self.assertEqual((self.folder/'ZnO_H_ads1.cif').read_bytes(),b'')
        report = pd.read_csv(self.output/'imp2d_download_verification.csv').fillna('')
        self.assertEqual(report.loc[report.source_id=='ZnO_H_ads1','quality_reasons'].item(),'')

    def test_nonempty_primary_has_priority_and_label_mismatch_is_reported(self):
        (self.folder/'id_prop.csv').write_text('ZnO_H_ads0,2.0\n',encoding='utf-8')
        result = verify_imp2d(self.root,self.db,self.output)
        self.assertEqual(result['label_table_rows'],1)
        self.assertFalse(result['used_alternative_label_table'])
        self.assertFalse(result['all_labels_match_database'])
        self.assertNotEqual(result['status'],'verified')

    def test_semi_without_labels_can_check_geometry_but_cannot_make_training_manifest(self):
        folder = self.root/'Dataset_1/Dataset_1/Neutral/Neutral'
        folder.mkdir(parents=True)
        (folder/'id_prop_A_rich.csv').write_bytes(b'')
        atoms = Atoms('HgO',positions=[[1,1,1],[3,1,1]],cell=[10]*3,pbc=True)
        write(str(folder/'1-ZnO-M_A-Hg-POSCAR1.cif'),atoms,format='cif')
        (folder/'1-ZnO-M_A-Hg-POSCAR2.cif').write_bytes(b'')
        result = verify_semi(self.root,self.output)
        self.assertEqual(result['download_status_counts'],{'readable':1,'empty_download':1})
        self.assertEqual(result['candidate_physical_exclusions'],0)
        self.assertEqual(result['readable_extrinsic_structures'],1)
        self.assertEqual(result['scope'],'available_CIFs_without_labels')
        self.assertEqual(result['status'],'partial_structure_audit')
        self.assertFalse(result['training_manifest_created'])
        self.assertEqual(result['missing_hosts'],['ZnO'])


if __name__ == '__main__':
    unittest.main()
