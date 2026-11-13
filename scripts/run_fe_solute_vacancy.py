"""From-scratch ALIGNN comparison for solute-vacancy shell ordering in bcc Fe."""
import argparse
from collections import defaultdict
import csv
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import torch
from pymatgen.core import Structure
from pymatgen.core.periodic_table import DummySpecies

from ..config.defaults import get_config
from ..data.datasets import dataset_index_for_mode, init_elem_embedding, representation_for_mode
from ..data.fe_solute_vacancy import (ARCHIVE, LATTICE_A, MASTER, RESULTS, ROOT, SCHEMA, SOURCE,
                                     SPLIT_SEED, partitions, prepare, write_bundle)
from ..data.structure_utils import preserve_pool_type
from ..main import train_single_mode
from ..training.results import ensure_compact_run_config
from .alignn_benchmark_support import audit_checkpoint, fixed_split, persist_contract
from .alignn_convergence import convergence_audit
from .run_sige_complexes import MODES, save_csv


DEFAULT_RUN = ROOT/'logs'/SCHEMA
THRESHOLD = 0.010


def structures(mode, records, atoms, archive):
    rep = representation_for_mode(mode)
    result = []
    for row in records:
        sid = row['source_id']
        a = atoms[sid]
        st = Structure(a.cell.array,a.numbers,a.get_scaled_positions(),site_properties={
            'type':a.arrays['is_solute'].astype(bool).tolist(), 'was':a.arrays['was_Z'].tolist()})
        if rep != 'full':
            st.append(DummySpecies(),a.info['vacancy_fractional'],properties={'was':26,'type':True})
        if rep == 'hetero':
            st = preserve_pool_type(st)
        st.source_id=st.source_name=sid
        st.source_path=str(archive) + '!/basic/energies.xlsx#Substitution-vacancy!'
        st.source_path+=f"{row['workbook_col']}{row['workbook_row']}"
        st.material='bcc Fe'
        st.defect_family=f"substitutional {row['solute']} + Fe vacancy"
        result.append(st)
    return result


def equal_solute_pairs(rows):
    groups=defaultdict(list)
    for r in rows:
        groups[r['fold'],r['solute']].append(r)
    pairs=[]
    for (fold,solute),g in sorted(groups.items()):
        if sorted(r['shell'] for r in g) != [1,2,3]:
            raise ValueError('Incomplete triple of held-out configurations')
        for a,b in combinations(sorted(g,key=lambda r:r['shell']),2):
            # Stable configuration has larger binding. Sources A and B have identical references.
            dft=round(a['dft_binding_eV']-b['dft_binding_eV'],6)
            pred=a['predicted_binding_eV']-b['predicted_binding_eV']
            pairs.append(dict(fold=fold,solute=solute,id_a=a['source_id'],id_b=b['source_id'],
                              dft_gap_eV=dft,predicted_gap_eV=pred,abs_gap_error_eV=abs(pred-dft)))
    return pairs


def run_fit(mode,fold,ids,lookup,targets,dataset,args):
    base=args.run_dir/f'fold{fold}'/f'seed{args.seed}'/mode
    budget=args.initial_budget
    while True:
        folder=base if budget==args.initial_budget else base/f'extended_{budget}'
        # This key supplies HERA's common ALIGNN defaults. The only training
        # examples passed to the trainer are the Fe occupations in this fold.
        config=get_config('alignn','och',mode)
        config['model'].update(train_batch_size=8,test_batch_size=8,local_radius=0.)
        config['optim'].update(amp=False,early_stopping_patience=args.patience)
        config.update(target_name='DFT_solute_vacancy_binding',target_units='eV/complex',benchmark_schema=SCHEMA,
                      workbook_sha256=MASTER,element_fold=fold,max_epochs=budget)
        ensure_compact_run_config(folder,dict(layout='compact_v1',model_name='alignn',dataset_name=SCHEMA,
                                              mode=mode,run_label=mode,config=config))
        if not args.resume and any(folder.glob('*_history.csv')):
            raise FileExistsError(folder)
        print(f'Fe-X-vacancy mode={mode} fold={fold} budget={budget} patience={args.patience}',flush=True)
        train_single_mode(mode,config,dataset,targets,[args.seed],budget,args.device,'alignn',SCHEMA,
                          log_dir=str(folder),resume=args.resume,split_iterator=fixed_split(ids,SCHEMA,f'held-out solutes {fold}'))
        plateau=convergence_audit(folder,args.seed)
        if plateau['stopped_on_validation_plateau']:
            break
        # No optimizer state is supplied by HERA's best-validation checkpoint. Keep the old
        # trajectory and rerun from its original seed with twice the budget if the guard is reached.
        budget*=2
    audit=audit_checkpoint(folder,config,args.seed,ids,'best_val_binding_MAE_eV')
    with (folder/f'seed{args.seed}_test_predictions.csv').open(newline='') as f:
        predictions=list(csv.DictReader(f))
    if [r['source_id'] for r in predictions]!=ids['test']:
        raise ValueError('Wrong held-out identities')
    structs=[]
    for p in predictions:
        r=lookup[p['source_id']]
        value=float(p['prediction'])
        if abs(float(p['target'])-r['binding_eV'])>1e-6 or not np.isfinite(value):
            raise ValueError('Test prediction or label mismatch')
        structs.append(dict(fold=fold,source_id=r['source_id'],solute=r['solute'],shell=r['shell'],
                            dft_binding_eV=r['binding_eV'],predicted_binding_eV=value))
    pairs=equal_solute_pairs(structs)
    save_csv(base/'test_structures.csv',structs)
    save_csv(base/'test_pairs.csv',pairs)
    report=dict(mode=mode,fold=fold,seed=args.seed,device=args.device,test_cases=len(structs),pairs=len(pairs),
                gap_MAE_eV=float(np.mean([r['abs_gap_error_eV'] for r in pairs])),
                selected_run=folder.relative_to(base).as_posix(),**(audit | plateau))
    (base/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(report,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,default=ARCHIVE)
    parser.add_argument('--run-dir',type=Path,default=DEFAULT_RUN)
    parser.add_argument('--results',type=Path,default=RESULTS)
    parser.add_argument('--mode',nargs='+',choices=MODES,default=list(MODES))
    parser.add_argument('--fold',nargs='+',type=int,choices=range(3),default=[0,1,2])
    parser.add_argument('--seed',type=int,default=123)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--patience',type=int,default=100)
    parser.add_argument('--initial-budget',type=int,default=1000)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.patience<1 or args.initial_budget<args.patience:
        parser.error('Initial budget must be at least the positive patience')
    torch.set_num_threads(1)
    records,atoms,raw=prepare(args.archive)
    splits=partitions(records)
    args.run_dir.mkdir(parents=True,exist_ok=True)
    contract=dict(schema=SCHEMA,source=SOURCE,master_sha256=MASTER,split_seed=SPLIT_SEED,
                  modes=list(MODES),partitions=splits,cohort='all 45 non-Fe solutes, three shells each',
                  input=f'Ideal 54-site bcc parent, ASE a={LATTICE_A} Å. Nuclear species and nominal defect sites only.',
                  target='Published DFT solute-vacancy binding energy in eV. Higher binding means lower total energy.',
                  primary='MAE of signed energy differences within held-out solutes over all three shell pairs',
                  ordering_threshold_eV=THRESHOLD,init_seed=args.seed,validation_patience=args.patience,
                  min_relative_improvement_pct=.5,initial_budget=args.initial_budget,
                  stopping='Actual validation plateau required; double the budget and repeat same seed if needed.',
                  primary_comparators=['hetero','full','full_x'],baselines=['composition-only','training mean for each shell'])
    persist_contract(args.run_dir/'data_contract.json',contract)
    bundle=args.results/'Fe_solute_vacancy_DFT.zip'
    if args.prepare_only or not bundle.exists():
        write_bundle(bundle,records,atoms,raw,splits)
    if args.prepare_only:
        return
    init_elem_embedding(ROOT/'atom_init.json')
    lookup={r['source_id']:r for r in records}
    targets=torch.tensor([r['binding_eV'] for r in records],dtype=torch.float32)
    for mode in dict.fromkeys(args.mode):
        dataset=[None]*4
        dataset[dataset_index_for_mode(mode)]=structures(mode,records,atoms,args.archive)
        for fold in dict.fromkeys(args.fold):
            run_fit(mode,fold,splits[fold],lookup,targets,dataset,args)


if __name__=='__main__':main()
