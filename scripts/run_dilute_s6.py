"""From-scratch fixed-composition validation on the largest dilute MoS2 inventory."""
import argparse
import ast
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import torch
from pymatgen.core import Structure

from ..config.defaults import get_config
from ..data.datasets import dataset_index_for_mode, init_elem_embedding, representation_for_mode
from ..data.structure_utils import convert_to_sparse_2dmd_high
from ..main import train_single_mode
from ..training.results import ensure_compact_run_config
from .alignn_benchmark_support import audit_checkpoint, fixed_split, persist_contract
from .run_sige_complexes import MODES, save_csv

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT.parent/'dataset'/'2d-materials-point-defects-all'/'low_density_defects'/'MoS2'
EVIDENCE=ROOT/'results'/'hetero_multidefect_validation_20261109'/'2dmd_low'
SCHEMA='dilute_mos2_s6_arrangement_v1'
DEFAULT_RUN=ROOT/'logs'/SCHEMA
COHORT_DIGEST='f587a08e9f4dde47058dfef034e15884178124be89e57eebcf6ddcd419a4f32b'
PARENT_DIGEST='a462bb4b8014b9829bed4b51b017c9f1b8b04cc84d9392aa3a425ec78952b27f'
DFT_REFERENCE_EV_CELL=-1395.4255


def cohort(root,archive):
    fingerprints=pd.read_csv(archive/'occupation_fingerprints.csv')
    old_split=json.loads((archive/'membership.json').read_text())
    info=pd.read_csv(root/'descriptors.csv',index_col=0)
    desc=info[(info.base=='MoS2')&(info.description=='S6')]
    if len(desc)!=1:raise ValueError('Expected exactly one MoS2 S6 descriptor')
    descriptor_id=desc.index[0]
    inventory=ast.literal_eval(desc.iloc[0]['defects'])
    expected=[{'type':'substitution','from':'Mo','to':'W'},{'type':'vacancy','element':'S'},
              {'type':'substitution','from':'S','to':'Se'}]
    if inventory!=expected or ast.literal_eval(desc.iloc[0]['cell'])!=(8,8,1):
        raise ValueError('Unexpected MoS2 cell or defect inventory')
    table=pd.read_csv(root/'targets.csv.gz',index_col=0)
    table=table[table.descriptor_id==descriptor_id].copy()
    source=fingerprints[(fingerprints.host=='MoS2') & (fingerprints.part=='train')]
    source=source[source.source_id.isin(table.index)].sort_values('source_id')
    source=source.drop_duplicates('symmetry_fingerprint')
    eligible=source.source_id.tolist()
    shuffled=np.random.default_rng(20261109).permutation(eligible).tolist()
    ids=dict(train=shuffled[:96],val=shuffled[96:120],test=shuffled[120:168])
    selected=sum(ids.values(),[])
    if len(set(selected))!=168 or not set(selected)<=set(old_split['train']):
        raise ValueError('The new cohort must be entirely drawn from unique old training occupations')
    if set(selected)&set(old_split['val']+old_split['test']):
        raise ValueError('Overlap with the historical validation or test set')
    subset=table.loc[selected]
    offsets=subset.energy-subset.formation_energy_per_site*3
    if not np.allclose(offsets.to_numpy(),DFT_REFERENCE_EV_CELL,rtol=0,atol=1e-6):
        raise ValueError('Raw DFT energy differences and per-defect formation energies disagree')
    geometries=[root/'initial'/f'{sid}.cif' for sid in selected]
    digest=hashlib.sha256()
    for idx,(sid,path) in enumerate(zip(selected,geometries,strict=True)):
        digest.update(sid.encode()+path.read_bytes()+np.float64(subset.energy.iloc[idx]).tobytes())
    if digest.hexdigest()!=COHORT_DIGEST:raise ValueError('Selected raw cases differ from the pinned cohort')
    contract=dict(schema=SCHEMA,host='MoS2',formula=dict(Mo=63,W=1,S=126,Se=1),
                  descriptor=descriptor_id,defects=inventory,cell=[8,8,1],n_eligible=len(eligible),
                  selection='Largest held-out inventory in the archived benchmark by size; current experiment uses only its old training pool',
                  split_seed=20261109,membership=ids,cohort_sha256=digest.hexdigest(),
                  primary='MAE of per-defect energy differences among all 48 held-out occupations',
                  target='Published DFT formation energy per defect minus the training-only mean',
                  selection_metric='Minimum validation structure-energy MAE',ordering_margin_ev=.001)
    return contract,subset,geometries


def paired_scores(y,p):
    i,j=np.triu_indices(len(y),1)
    reference=y[i]-y[j];estimate=p[i]-p[j]
    resolved=np.abs(reference)>.001
    return dict(test_cases=len(y),point_mae_ev_defect=float(np.mean(abs(y-p))),test_pairs=len(i),
                gap_mae_ev_defect=float(np.mean(abs(reference-estimate))),
                zero_gap_mae_ev_defect=float(np.mean(abs(reference))),
                ranked_correct=int(((reference*estimate>0)&resolved).sum()),ranked_total=int(resolved.sum()))


def from_bundle(path):
    with zipfile.ZipFile(path) as archive:
        contract=json.loads(archive.read('data_contract.json'))
        table=pd.read_csv(io.BytesIO(archive.read('records.csv')),float_precision='round_trip').set_index('source_id')
        ids=contract['membership']
        ordered=[sid for part in ('train','val','test') for sid in ids[part]]
        if (contract['schema']!=SCHEMA or [len(ids[p]) for p in ('train','val','test')]!=[96,24,48]
                or not table.index.is_unique or set(ordered)!=set(table.index) or len(table)!=168):
            raise ValueError('Invalid bundled cohort or split')
        table=table.loc[ordered]
        if (table.symmetry_fingerprint.nunique()!=168 or not (table.old_split=='train').all()
                or table.split.tolist()!=['train']*96+['val']*24+['test']*48):
            raise ValueError('Bundled occupations and partition metadata differ')
        table=table.rename(columns={'formation_ev_defect':'formation_energy_per_site','dft_total_ev_cell':'energy'})
        digest=hashlib.sha256()
        structures=[]
        for sid,row in table.iterrows():
            geometry=archive.read(row.cif)
            digest.update(sid.encode()+geometry+np.float64(row.energy).tobytes())
            structures.append((str(path)+'!/'+row.cif,geometry))
        if digest.hexdigest()!=COHORT_DIGEST or contract['cohort_sha256']!=COHORT_DIGEST:
            raise ValueError('The bundled geometries or reference energies differ from the experiment')
        offsets=table.energy-3*table.formation_energy_per_site
        if not np.allclose(offsets.to_numpy(),DFT_REFERENCE_EV_CELL,rtol=0,atol=1e-6):
            raise ValueError('Inconsistent total and per-defect DFT energies')
        return contract,table,archive.read('parents/MoS2.cif'),structures


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,default=SOURCE)
    parser.add_argument('--archive',type=Path,default=EVIDENCE)
    parser.add_argument('--bundle',type=Path,help='Train directly from MoS2_S6_168_structures_DFT.zip')
    parser.add_argument('--run-dir',type=Path,default=DEFAULT_RUN)
    parser.add_argument('--mode',nargs='+',choices=MODES,default=list(MODES))
    parser.add_argument('--epochs',type=int,default=500)
    parser.add_argument('--patience',type=int,default=50)
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--seed',type=int,default=123)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.bundle:
        contract,table,parent_cif,entries=from_bundle(args.bundle)
    else:
        contract,table,paths=cohort(args.data_root,args.archive)
        parent_cif=(args.data_root/'unit_cells'/'MoS2.cif').read_bytes()
        entries=[(str(path),path.read_bytes()) for path in paths]
    if hashlib.sha256(parent_cif).hexdigest()!=PARENT_DIGEST:raise ValueError('Unexpected MoS2 parent CIF')
    args.run_dir.mkdir(parents=True,exist_ok=True)
    persist_contract(args.run_dir/'data_contract.json',contract)
    ids=contract['membership']
    print('MoS2 S6:',{p:len(v) for p,v in ids.items()},'old training pool',contract['n_eligible'],flush=True)
    if args.prepare_only:return
    init_elem_embedding(ROOT/'atom_init.json')
    host=Structure.from_str(parent_cif.decode(),'cif',primitive=False)
    source=[]
    for sid,(path,geometry) in zip(table.index,entries,strict=True):
        st=Structure.from_str(geometry.decode(),'cif',primitive=False)
        if st.composition.get_el_amt_dict()!={'Mo':63,'W':1,'S':126,'Se':1}:
            raise ValueError(f'Chemical counts differ in {sid}')
        st.source_id=st.source_name=sid;st.source_path=str(path);source.append(st)
    offset=float(table.loc[ids['train'],'formation_energy_per_site'].mean())
    targets=torch.tensor(table.formation_energy_per_site.to_numpy()-offset,dtype=torch.float32)
    for mode in dict.fromkeys(args.mode):
        folder=args.run_dir/f'seed{args.seed}'/mode
        config=get_config('alignn','2dmd_low',mode)
        config['model'].update(train_batch_size=args.batch_size,test_batch_size=1,local_radius=0.)
        config['optim'].update(amp=False,early_stopping_patience=args.patience)
        config.update(target_name='formation_energy_per_defect_minus_training_mean',target_units='eV/defect',
                      energy_offset_ev=offset,benchmark_schema=SCHEMA,cohort_sha256=contract['cohort_sha256'],
                      max_epochs=args.epochs)
        ensure_compact_run_config(folder,dict(layout='compact_v1',model_name='alignn',dataset_name=SCHEMA,
                                               mode=mode,run_label=mode,config=config))
        if not args.resume and any(folder.glob('*_history.csv')):raise ValueError(f'{folder} already has a fit')
        rep=representation_for_mode(mode)
        graphs=[convert_to_sparse_2dmd_high(s,host,(8,8,1),'alignn_'+rep,None,False,False,local_cutoff=0.)
                for s in source]
        dataset=[None]*4;dataset[dataset_index_for_mode(mode)]=graphs
        print(f'MoS2 fixed formula seed={args.seed} mode={mode}',flush=True)
        train_single_mode(mode,config,dataset,targets,[args.seed],args.epochs,args.device,'alignn',SCHEMA,
                          log_dir=str(folder),resume=args.resume,split_iterator=fixed_split(ids,SCHEMA,'new occupations'))
        with (folder/f'seed{args.seed}_test_predictions.csv').open(newline='') as f:rows=list(csv.DictReader(f))
        if [r['source_id'] for r in rows]!=ids['test']:raise ValueError('Incorrect test membership')
        y=table.loc[ids['test'],'formation_energy_per_site'].to_numpy()
        p=offset+np.array([float(r['prediction']) for r in rows])
        if not np.all(np.isfinite(p)) or not np.allclose(y,[float(r['target'])+offset for r in rows],rtol=0,atol=1e-7):
            raise ValueError('Non-finite prediction or mismatched DFT targets')
        result=dict(mode=mode,seed=args.seed,**paired_scores(y,p),**audit_checkpoint(folder,config,args.seed,ids))
        (folder/'metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        save_csv(folder/'heldout.csv',[dict(source_id=s,formation_ev_defect=float(v),prediction_ev_defect=float(w),
                     dft_total_ev_cell=float(table.at[s,'energy'])) for s,v,w in zip(ids['test'],y,p,strict=True)])
        print(result,flush=True)


if __name__=='__main__':main()
