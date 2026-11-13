"""From-scratch ALIGNN comparison on published, composition-matched Si/Ge complexes."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import torch
from ase.data import atomic_numbers
from ase.io import read
from pymatgen.core import Structure
from pymatgen.core.periodic_table import DummySpecies

from ..config.defaults import ALIGNN_MODES, get_config
from ..data.datasets import dataset_index_for_mode, init_elem_embedding, representation_for_mode
from ..data.sige_complexes import DATA_ROOT, ROOT, SCHEMA, prepare
from ..data.structure_utils import preserve_pool_type
from ..main import train_single_mode
from ..training.results import ensure_compact_run_config
from .alignn_benchmark_support import audit_checkpoint, fixed_split, persist_contract

MODES=tuple(m for m in ALIGNN_MODES if m not in ('hypergraph','hypergraph_was'))
DEFAULT_RUN=ROOT/'logs'/'sige_bc_sv_pbe_element_holdout_v1'


def membership(records, split_seed):
    dopants=sorted({r['dopant'] for r in records},key=atomic_numbers.get)
    shuffled=np.random.default_rng(split_seed).permutation(dopants).tolist()
    ntest=round(len(shuffled)*0.20)
    nval=round(len(shuffled)*0.20)
    elements=dict(test=shuffled[:ntest],val=shuffled[ntest:ntest+nval],train=shuffled[ntest+nval:])
    ids={p:[r['source'] for r in records if r['dopant'] in keys] for p,keys in elements.items()}
    for p in ids:
        if len(ids[p])!=len(set(ids[p])) or len(ids[p])%2:
            raise ValueError(f'{p}: incomplete or duplicate complex pairs')
    if len(set().union(*ids.values()))!=len(records):
        raise ValueError('Held-out elements do not cover the original cohort')
    return elements,ids


def structures_for_mode(mode, records, geometries, source):
    result=[]
    rep=representation_for_mode(mode)
    for r in records:
        a=geometries[r['source']]
        imp=r['dopant_index']
        flags=np.zeros(len(a),dtype=bool)
        flags[imp]=True
        was=a.numbers.copy()
        host_z=atomic_numbers[r['host']]
        was[imp]=host_z if r['dopant_occupies_host_site'] else 0
        st=Structure(a.cell.array,a.numbers,a.positions,coords_are_cartesian=True,
                     site_properties={'was':was.tolist(),'type':flags.tolist()})
        if rep!='full':
            for site in r['vacancy_frac']:
                st.append(DummySpecies(),site,properties={'was':host_z,'type':True})
        if rep=='hetero':
            st=preserve_pool_type(st)
        st.source_id=r['source']
        st.source_name=f"{r['host']}-{r['kind']}-{r['dopant']}"
        st.source_path=str(source)
        st.material=r['host']
        st.defect_family=r['kind']
        result.append(st)
    return result


def paired_metrics(rows, records):
    by_id={r['source']:r for r in records}
    pairs={}
    errors=[]
    for row in rows:
        r=by_id[row['source_id']]
        target=r['formation_enthalpy_ev']
        pred=float(row['prediction'])
        if not np.isfinite(pred) or abs(float(row['target'])-target)>2e-6:
            raise ValueError('Nonfinite prediction or label mismatch')
        errors.append(abs(pred-target))
        pairs.setdefault((r['host'],r['dopant']),{})[r['kind']]=(target,pred)
    comparison=[]
    for (host,dopant),p in sorted(pairs.items()):
        truth=round(p['SV'][0]-p['BC'][0],2)
        prediction=p['SV'][1]-p['BC'][1]
        comparison.append(dict(host=host,dopant=dopant,dft_gap_ev=truth,predicted_gap_ev=prediction,
                               absolute_gap_error_ev=abs(truth-prediction)))
    informative=[r for r in comparison if abs(r['dft_gap_ev'])>0.02]
    correct=sum(r['dft_gap_ev']*r['predicted_gap_ev']>0 for r in informative)
    return (dict(test_mae_ev_complex=float(np.mean(errors)),test_cases=len(errors),test_pairs=len(pairs),
                 gap_mae_ev=float(np.mean([r['absolute_gap_error_ev'] for r in comparison])),
                 zero_gap_mae_ev=float(np.mean([abs(r['dft_gap_ev']) for r in comparison])),
                 resolved_pair_correct=correct,resolved_pairs=len(informative)),comparison)


def save_csv(path,rows):
    with path.open('w',newline='',encoding='utf8') as handle:
        writer=csv.DictWriter(handle,fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,default=DATA_ROOT)
    parser.add_argument('--run-dir',type=Path,default=DEFAULT_RUN)
    parser.add_argument('--mode',nargs='+',choices=MODES,default=list(MODES))
    parser.add_argument('--seed',type=int,nargs='+',default=[123],help='Weight initialization seeds')
    parser.add_argument('--split-seed',type=int,default=123)
    parser.add_argument('--epochs',type=int,default=500)
    parser.add_argument('--patience',type=int,default=50)
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--resume',action='store_true',help='Skip complete matching fits; restart incomplete fits from scratch')
    args=parser.parse_args()
    if min(args.epochs,args.patience,args.batch_size)<1:
        parser.error('epochs, patience, and batch size must be positive')
    descriptor=prepare(args.data_root)
    records=descriptor['records']
    elements,ids=membership(records,args.split_seed)
    args.run_dir.mkdir(parents=True,exist_ok=True)
    contract=dict(schema=SCHEMA, split_seed=args.split_seed, dopant_partition=elements,membership=ids,
                  source_dois=[descriptor['source_doi_structure'],descriptor['source_doi_labels']],
                  labels_sha256=descriptor['labels_sha256'], cohort_sha256=hashlib.sha256(
                      json.dumps(records,sort_keys=True).encode()).hexdigest(),
                  target=descriptor['target'], primary_metric='MAE of SV minus BC enthalpy differences across matched pairs',
                  secondary_metrics=['formation enthalpy MAE','ordering accuracy for |DFT gap| > 0.02 eV'],
                  selection='minimum validation formation enthalpy MAE', site_rule=descriptor['site_rule'],
                  excluded=descriptor['excluded'])
    persist_contract(args.run_dir/'data_contract.json',contract)
    print('Sources:',{p:len(v) for p,v in ids.items()},'dopants:',{p:len(v) for p,v in elements.items()},flush=True)
    if args.prepare_only:return
    init_elem_embedding(ROOT/'atom_init.json')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        parser.error('CUDA unavailable; select --device cpu')
    source=args.data_root/'cm6b03368_si_001.zip'
    with zipfile.ZipFile(source) as archive:
        geometries={r['source']:read(io.BytesIO(archive.read(r['source'])),format='cif') for r in records}
    targets=torch.tensor([r['formation_enthalpy_ev'] for r in records],dtype=torch.float32)
    for seed in dict.fromkeys(args.seed):
        for mode in dict.fromkeys(args.mode):
            directory=args.run_dir/f'seed{seed}'/mode
            config=get_config('alignn','2dmd_low',mode)
            config['model'].update(train_batch_size=args.batch_size,test_batch_size=1,local_radius=0.0)
            config['optim'].update(amp=False,early_stopping_patience=args.patience)
            config.update(target_name='PBE_defect_formation_enthalpy',target_units='eV/complex',
                          benchmark_schema=SCHEMA,cohort_sha256=contract['cohort_sha256'],
                          element_split_seed=args.split_seed,max_epochs=args.epochs)
            ensure_compact_run_config(directory,dict(layout='compact_v1',model_name='alignn',dataset_name=SCHEMA,
                                                     mode=mode,run_label=mode,config=config))
            if not args.resume and any(directory.glob('*_history.csv')):
                raise ValueError(f'{directory}: history exists; specify --resume or use a new run directory')
            dataset=[None]*4
            dataset[dataset_index_for_mode(mode)]=structures_for_mode(mode,records,geometries,source)
            print(f'Si/Ge seed={seed} mode={mode}',flush=True)
            train_single_mode(mode,config,dataset,targets,[seed],args.epochs,args.device,'alignn',SCHEMA,
                              log_dir=str(directory),resume=args.resume,
                              split_iterator=fixed_split(ids,SCHEMA,'held-out dopant elements'))
            with (directory/f'seed{seed}_test_predictions.csv').open(newline='') as handle:
                rows=list(csv.DictReader(handle))
            if [r['source_id'] for r in rows]!=ids['test']:
                raise ValueError(f'{mode}: test membership changed')
            metrics,pairs=paired_metrics(rows,records)
            audit=audit_checkpoint(directory,config,seed,ids,'best_val_mae_ev_complex')
            report=dict(mode=mode,seed=seed,**metrics,**audit)
            (directory/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf8')
            save_csv(directory/'bc_sv_test_pairs.csv',pairs)
            result=[json.loads(p.read_text(encoding='utf8')) for p in sorted(args.run_dir.glob('seed*/*/metrics.json'))]
            save_csv(args.run_dir/'summary.csv',result)
            print(report,flush=True)


if __name__=='__main__':
    main()

