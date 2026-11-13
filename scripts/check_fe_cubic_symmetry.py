"""Check held-out predictions for proper cubic rotations of ideal Fe vacancy occupations."""
import argparse
from itertools import permutations, product
import json
from pathlib import Path
from types import MethodType

import numpy as np
import pandas as pd
import torch

from ..data.datasets import init_elem_embedding
from ..data.fe_solute_vacancy import RESULTS, ROOT, prepare
from ..training.trainer import MEGNetTrainer, load_trusted_checkpoint
from .run_fe_solute_vacancy import DEFAULT_RUN, MODES, THRESHOLD, equal_solute_pairs, structures


ROTATIONS={'published_input_reconstruction':np.eye(3),
           'quarter_turn_about_z':np.array([[0,-1,0],[1,0,0],[0,0,1]]),
           'cyclic_permutation_of_axes':np.array([[0,1,0],[0,0,1],[1,0,0]])}
DIRS={1:np.array([1,1,1]),2:np.array([2,0,0]),3:np.array([2,2,0])}  # units of a/2


def distinct_placements():
    identity=np.eye(3,dtype=int)
    operations=[identity]
    for perm in permutations(range(3)):
        for signs in product((-1,1),repeat=3):
            op=identity[list(perm)]*np.array(signs)[:,None]
            if round(np.linalg.det(op))==1 and not np.array_equal(op,identity):
                operations.append(op)
    if len(operations)!=24:
        raise ValueError('Incomplete proper cubic group')
    seen={k:set() for k in DIRS}
    placements=[]
    for op in operations:
        fresh={}
        for k,v in DIRS.items():
            direction=tuple((op@v).tolist())
            if direction not in seen[k]:
                seen[k].add(direction)
                fresh[k]=direction
        if fresh:
            placements.append((op,fresh))
    if [len(seen[k]) for k in DIRS]!=[8,6,12]:
        raise ValueError('Incorrect BCC cubic shell multiplicities')
    return placements


def plot_shell_spread(path,data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(6.6,4.2))
    rng=np.random.default_rng(123)
    for mode,offset,color,marker in [('hetero',-.15,'#19746b','o'),('full_x',.15,'#768ca4','s')]:
        for k in (1,2,3):
            values=data.loc[(data['mode']==mode)&(data.shell==k),'symmetry_range_eV'].to_numpy()*1000
            x=k+offset+rng.uniform(-.07,.07,size=len(values))
            ax.scatter(x,values,alpha=.5,s=16,color=color,marker=marker,edgecolors='none',
                       label=mode if k==1 else None)
            ax.plot([k+offset-.1,k+offset+.1],[values.mean()]*2,color=color,lw=2.3)
    ax.set(yscale='symlog',xticks=[1,2,3],xticklabels=['1NN · 8 sites','2NN · 6 sites','3NN · 12 sites'],
           ylabel='Within-shell prediction range (meV)',xlabel='Symmetry-equivalent vacancy positions')
    ax.set_yscale('symlog',linthresh=.015,linscale=.7)
    ax.set_yticks([0,.01,.1,1,10,100],['0','0.01','0.1','1','10','100'])
    ax.set_ylim(bottom=-.002)
    ax.plot([],[],color='#48535d',lw=2.3,label='Shell mean')
    ax.legend(frameon=False,ncol=3,loc='upper center',bbox_to_anchor=(.5,1.04))
    ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(path,dpi=180,bbox_inches='tight')
    plt.close(fig)


def complete_boundary_shell(converter, structure):
    """Inference-only control. Expand the last retained distance shell per neighbor type."""
    all_neighbors=structure.get_all_neighbors(converter.cutoff,include_index=True)
    result=[]
    for neighbors in all_neighbors:
        ordered=sorted(neighbors,key=lambda n:n[1])
        by_type={}
        for n in ordered:
            kind=int(structure[n[2]].properties.get('type',0))
            by_type.setdefault(kind,[]).append(n[1])
        thresholds={kind:(values[converter.max_neighbors-1] if len(values)>converter.max_neighbors else np.inf)
                    for kind,values in by_type.items()}
        result.append([n for n in ordered
                       if n[1]<=thresholds[int(structure[n[2]].properties.get('type',0))]+1e-6])
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=DEFAULT_RUN)
    parser.add_argument('--results',type=Path,default=RESULTS)
    parser.add_argument('--seed',type=int,default=123)
    parser.add_argument('--mode',nargs='+',choices=MODES,help='Defaults to all trained modes, or the matched pair for the cutoff control')
    parser.add_argument('--complete-boundary-shell',action='store_true',
                        help='Diagnostic only: add all distance ties at the 12th neighbor, keeping original trained weights')
    args=parser.parse_args()
    torch.set_num_threads(2)
    init_elem_embedding(ROOT/'atom_init.json')
    archive=args.results/'Fe_solute_vacancy_DFT.zip'
    contract=json.loads((args.run_dir/'data_contract.json').read_text(encoding='utf8'))
    poses=[(fresh,*prepare(archive,op)[:2]) for op,fresh in distinct_placements()]
    records=poses[0][1]
    lookup={r['source_id']:r for r in records}
    scores=[]
    outputs=[]
    modes=args.mode or (('hetero','full_x') if args.complete_boundary_shell else MODES)
    for mode in modes:
        graphs={sid:[] for sid in lookup}
        for fresh,rs,atoms in poses:
            rs=[r for r in rs if r['shell'] in fresh]
            gs=structures(mode,rs,atoms,archive)
            for g in gs:
                graphs[g.source_id].append((fresh[lookup[g.source_id]['shell']],g))
        for fold,ids in enumerate(contract['partitions']):
            base=args.run_dir/f'fold{fold}'/f'seed{args.seed}'/mode
            record=json.loads((base/'metrics.json').read_text(encoding='utf8'))
            checkpoint=load_trusted_checkpoint(base/record['selected_run']/f'seed{args.seed}_best_checkpoint.pth',map_location='cpu')
            if not checkpoint['stopped_early'] or [s['source_id'] for s in checkpoint['split']['test_sources']]!=ids['test']:
                raise ValueError('Symmetry check must use the converged model on its own held-out solutes')
            trainer=MEGNetTrainer(checkpoint['config'],'cpu',args.seed)
            trainer.scaler.load_state_dict(checkpoint['scaler'])
            if args.complete_boundary_shell:
                if trainer.converter.max_neighbors!=12:
                    raise ValueError('Expected the original 12-neighbor cap')
                trainer.converter._neighbor_lists=MethodType(complete_boundary_shell,trainer.converter)
            reference=pd.read_csv(base/'test_structures.csv').set_index('source_id')
            cases=[(sid,d,g) for sid in ids['test'] for d,g in graphs[sid]]
            if len(cases)!=390:
                raise ValueError('Every held-out element must have 26 distinct equivalent placements')
            true=torch.tensor([lookup[s]['binding_eV'] for s,_,_ in cases],dtype=torch.float32)
            _,pred=trainer.predict_structures([g for _,_,g in cases],true,checkpoint['model'],return_predictions=True)
            canonical={}
            for (sid,d,_),v in zip(cases,np.asarray(pred).reshape(-1),strict=True):
                r=lookup[sid]
                is_canonical=d==tuple(DIRS[r['shell']])
                if is_canonical:
                    canonical[sid]=float(v)
                    if not args.complete_boundary_shell and abs(v-reference.loc[sid,'predicted_binding_eV'])>1e-3:
                        raise ValueError(f'Original CPU and GPU predictions differ on {sid}')
                outputs.append(dict(mode=mode,fold=fold,source_id=sid,solute=r['solute'],shell=r['shell'],
                                    dx=d[0],dy=d[1],dz=d[2],canonical=is_canonical,
                                    dft_binding_eV=r['binding_eV'],predicted_binding_eV=float(v),
                                    change_from_canonical_eV=float(v)-canonical[sid]))
    all_rows=pd.DataFrame(outputs)
    shells=[]; orbit_scores=[]
    for mode,mode_rows in all_rows.groupby('mode'):
        if len(mode_rows)!=1170:
            raise ValueError('Incomplete BCC orientation coverage')
        for pose,op in ROTATIONS.items():
            wanted=np.array([op@DIRS[k] for k in mode_rows.shell])
            g=mode_rows[(mode_rows[['dx','dy','dz']].to_numpy()==wanted).all(axis=1)]
            p=pd.DataFrame(equal_solute_pairs(g.to_dict('records')))
            informative=p.dft_gap_eV.abs()>THRESHOLD
            scores.append(dict(mode=mode,pose=pose,structures=len(g),pairs=len(p),
                               binding_MAE_eV=(g.predicted_binding_eV-g.dft_binding_eV).abs().mean(),
                               gap_MAE_eV=p.abs_gap_error_eV.mean(),
                               correctly_ordered=int(((p.dft_gap_eV*p.predicted_gap_eV>0)&informative).sum()),
                               informative_pairs=int(informative.sum()),mean_abs_prediction_change_eV=g.change_from_canonical_eV.abs().mean(),
                               max_abs_prediction_change_eV=g.change_from_canonical_eV.abs().max()))
        averaged=(mode_rows.groupby(['mode','fold','solute','shell','source_id'],as_index=False)
                  .agg(dft_binding_eV=('dft_binding_eV','first'),predicted_binding_eV=('predicted_binding_eV','mean'),
                       minimum_eV=('predicted_binding_eV','min'),maximum_eV=('predicted_binding_eV','max'),
                       placements=('dx','size')))
        averaged['symmetry_range_eV']=averaged.maximum_eV-averaged.minimum_eV
        pair=pd.DataFrame(equal_solute_pairs(averaged.to_dict('records')))
        shells.append(averaged)
        orbit_scores.append(dict(mode=mode,independent_structures=len(averaged),equivalent_placements=len(mode_rows),
                                 mean_symmetry_range_eV=averaged.symmetry_range_eV.mean(),
                                 median_symmetry_range_eV=averaged.symmetry_range_eV.median(),
                                 max_symmetry_range_eV=averaged.symmetry_range_eV.max(),
                                 symmetry_averaged_binding_MAE_eV=(averaged.predicted_binding_eV-averaged.dft_binding_eV).abs().mean(),
                                 symmetry_averaged_gap_MAE_eV=pair.abs_gap_error_eV.mean()))
    shell_rows=pd.concat(shells,ignore_index=True)
    suffix='_complete_boundary_shell' if args.complete_boundary_shell else ''
    shell_rows.to_csv(args.results/f'all_shell_orientations_by_structure{suffix}.csv',index=False)
    if not args.complete_boundary_shell:
        plot_shell_spread(args.results/'cubic_symmetry_spread.png',shell_rows)
    orbit_table=pd.DataFrame(orbit_scores)
    orbit_table.to_csv(args.results/f'all_shell_orientations{suffix}.csv',index=False)
    all_rows.to_csv(args.results/f'cubic_symmetry_predictions{suffix}.csv',index=False)
    score_table=pd.DataFrame(scores)
    score_table.to_csv(args.results/f'cubic_symmetry_check{suffix}.csv',index=False)
    print(score_table.to_string(index=False),flush=True)
    print(orbit_table.to_string(index=False),flush=True)


if __name__=='__main__':
    main()
