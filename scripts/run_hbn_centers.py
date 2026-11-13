"""From-scratch three-fold comparison on distinct hBN color-center arrangements."""
import argparse
from collections import defaultdict
import csv
import hashlib
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import torch
from pymatgen.core import Structure
from pymatgen.core.periodic_table import DummySpecies

from ..config.defaults import get_config
from ..data.datasets import dataset_index_for_mode, init_elem_embedding, representation_for_mode
from ..data.hbn_centers import DATA_ROOT, DIGEST, REVISION, ROOT, SCHEMA, load_records
from ..data.structure_utils import preserve_pool_type
from ..main import train_single_mode
from ..training.results import ensure_compact_run_config
from .alignn_benchmark_support import audit_checkpoint, fixed_split, persist_contract
from .run_sige_complexes import MODES, save_csv

DEFAULT_RUN = ROOT/'logs'/'hbn_neutral_triplet_composition_cv_v1'


def representatives(records):
    # A given defect pattern can come from multiple starting structures. Use one
    # release ID per occupation irrespective of the electronic energies.
    unique={}
    for row in sorted(records,key=lambda r:r['source']):
        key=row['formula'],row['signature']
        unique.setdefault(key,row)
    return list(unique.values())


def partitions(records):
    groups=defaultdict(list)
    for r in records:groups[r['formula']].append(r)
    counts={f:sum(r['defect_markers']>=2 for r in group) for f,group in groups.items()}
    informative=[f for f in sorted(groups) if counts[f]>=2]
    other=[f for f in sorted(groups) if counts[f]<2]
    rng=np.random.default_rng(123)
    rng.shuffle(informative);rng.shuffle(other)
    # Balance composition counts first and pair counts second, using only site identities.
    folds=[[] for _ in range(3)]
    loads=[0,0,0]
    for f in sorted(informative,key=lambda k:-counts[k]):
        idx=min(range(3),key=lambda j:(len(folds[j]),loads[j],j))
        folds[idx].append(f);loads[idx]+=counts[f]*(counts[f]-1)//2
    partitions=[]
    for j in range(3):
        eligible=[f for i,fold in enumerate(folds) for f in fold if i!=j]
        rand=np.random.default_rng(123+j)
        nval=max(2,round(len(eligible)*.24))
        valid=set(rand.permutation(eligible)[:nval])
        valid.update(rand.permutation(other)[:max(1,round(len(other)*.15))])
        test=set(folds[j])
        part={s:[] for s in ['train','val','test']}
        for row in records:
            f=row['formula']; section='test' if f in test else ('val' if f in valid else 'train')
            part[section].append(row['source'])
        partitions.append(part)
    return partitions


def structures(mode,records,geometries,path):
    rep=representation_for_mode(mode)
    result=[]
    for row in records:
        a=geometries[row['source']]
        st=Structure(a.cell.array,a.numbers,a.positions,coords_are_cartesian=True,
                     site_properties={'type':row['type'],'was':row['was']})
        if rep!='full':
            for pos,z in zip(row['vacancy_frac'],row['vacancy_species'],strict=True):
                st.append(DummySpecies(),pos,properties={'was':z,'type':True})
        if rep=='hetero':st=preserve_pool_type(st)
        st.source_id=st.source_name=row['source'];st.source_path=str(path)
        st.material='hBN';st.defect_family=row['formula'];result.append(st)
    return result


def pair_rows(test,lookup,offset):
    groups=defaultdict(list)
    for row in test:
        sample=lookup[row['source_id']]
        if abs(float(row['target'])+offset-sample['energy'])>5e-5:
            raise ValueError(f'Label mismatch for {row["source_id"]}')
        if sample['defect_markers']>=2:
            groups[sample['formula']].append((sample,float(row['prediction'])))
    result=[]
    for formula,items in sorted(groups.items()):
        for (a,pa),(b,pb) in combinations(sorted(items,key=lambda item:item[0]['source']),2):
            if a['signature']==b['signature']:raise ValueError('Symmetry-equivalent test occupation')
            observed=b['energy']-a['energy']; predicted=pb-pa
            result.append(dict(formula=formula,id_a=a['source'],id_b=b['source'],dft_gap_ev=observed,
                               predicted_gap_ev=predicted,absolute_gap_error_ev=abs(predicted-observed)))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,default=DATA_ROOT)
    parser.add_argument('--run-dir',type=Path,default=DEFAULT_RUN)
    parser.add_argument('--fold',type=int,choices=(0,1,2),nargs='+',default=[0,1,2])
    parser.add_argument('--mode',choices=MODES,nargs='+',default=list(MODES))
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--seed',type=int,default=123)
    parser.add_argument('--epochs',type=int,default=500)
    parser.add_argument('--patience',type=int,default=50)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    raw,geometries=load_records(args.data_root)
    records=representatives(raw)
    splits=partitions(records);lookup={r['source']:r for r in records}
    args.run_dir.mkdir(parents=True,exist_ok=True)
    cohort_sha=hashlib.sha256(json.dumps(raw,sort_keys=True).encode()).hexdigest()
    contract=dict(schema=SCHEMA,revision=REVISION,database_sha256=DIGEST,cohort_sha256=cohort_sha,
                  source_rows=sum(r['source_rows'] for r in raw),distinct_names=len(raw),
                  retained=[r['source'] for r in records],partitions=splits,
                  geometry='Neutral monolayer triplet ground-state CIFs and total energies in eV',
                  site_rule='7 x 7 B/N parent with fitted in-plane origin, 0.56 first-neighbor lateral assignment cutoff; one label-independent representative per parent-symmetry occupation and off-site element inventory',
                  validation='Lowest validation structure-energy MAE',
                  primary='Out-of-fold MAE of within-composition energy gaps between distinct multi-defect occupations',
                  sign_threshold_ev=0.02)
    persist_contract(args.run_dir/'data_contract.json',contract)
    for j,ids in enumerate(splits):
        fake=[dict(source_id=s,target=lookup[s]['energy'],prediction=0.) for s in ids['test']]
        pair_count=len(pair_rows(fake,lookup,0.))
        print('fold',j,'structure counts',{p:len(v) for p,v in ids.items()},'distinct pairs',pair_count,flush=True)
    if args.prepare_only:return
    init_elem_embedding(ROOT/'atom_init.json')
    for j in dict.fromkeys(args.fold):
        ids=splits[j]
        offset=float(np.mean([lookup[s]['energy'] for s in ids['train']]))
        targets=torch.tensor([r['energy']-offset for r in records],dtype=torch.float32)
        for mode in dict.fromkeys(args.mode):
            folder=args.run_dir/f'fold{j}'/f'seed{args.seed}'/mode
            config=get_config('alignn','2dmd_low',mode)
            config['model'].update(train_batch_size=8,test_batch_size=1,local_radius=0.)
            config['optim'].update(amp=False,early_stopping_patience=args.patience)
            config.update(target_name='triplet_ground_energy_minus_training_mean',target_units='eV/cell',
                          energy_offset_ev=offset,benchmark_schema=SCHEMA,cohort_sha256=cohort_sha,
                          composition_fold=j,max_epochs=args.epochs)
            ensure_compact_run_config(folder,dict(layout='compact_v1',model_name='alignn',dataset_name=SCHEMA,
                                                   mode=mode,run_label=mode,config=config))
            if not args.resume and any(folder.glob('*_history.csv')):raise ValueError(f'{folder} already exists')
            dataset=[None]*4
            dataset[dataset_index_for_mode(mode)]=structures(mode,records,geometries,args.data_root/'hbn_defects_database.db')
            print(f'hBN fold={j} mode={mode}',flush=True)
            train_single_mode(mode,config,dataset,targets,[args.seed],args.epochs,args.device,'alignn',SCHEMA,
                              log_dir=str(folder),resume=args.resume,
                              split_iterator=fixed_split(ids,SCHEMA,f'composition holdout {j}'))
            with (folder/f'seed{args.seed}_test_predictions.csv').open(newline='') as f:rows=list(csv.DictReader(f))
            if [r['source_id'] for r in rows]!=ids['test']:raise ValueError('Wrong held-out structures')
            pairs=pair_rows(rows,lookup,offset)
            audit=audit_checkpoint(folder,config,args.seed,ids,'best_val_mae_ev_cell')
            report=dict(mode=mode,fold=j,seed=args.seed,test_cases=len(rows),test_pairs=len(pairs),
                        gap_mae_ev=float(np.mean([r['absolute_gap_error_ev'] for r in pairs])),
                        zero_gap_mae_ev=float(np.mean([abs(r['dft_gap_ev']) for r in pairs])),
                        test_mae_ev_cell=float(np.mean([abs(float(r['prediction'])-float(r['target'])) for r in rows])),
                        **audit)
            save_csv(folder/'test_pairs.csv',pairs)
            (folder/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf8')
            print(report,flush=True)


if __name__=='__main__':main()
