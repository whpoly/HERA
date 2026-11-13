"""Paired errors and cluster bootstrap intervals for the Si/Ge and hBN comparisons."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .run_sige_complexes import DEFAULT_RUN as SIGE_RUN, MODES
from .run_hbn_centers import DEFAULT_RUN as HBN_RUN

OUTPUT = Path(__file__).resolve().parents[1]/'results'/'hetero_multidefect_validation_20261109'


def inputs(kind,run,mode):
    points=[]; pairs=[]; fits=[]
    if kind=='sige':
        dirs=[run/'seed123'/mode]
    else:
        dirs=[run/f'fold{fold}'/'seed123'/mode for fold in range(3)]
    for folder in dirs:
        meta=json.loads((folder/'metrics.json').read_text())
        fits.append(meta)
        t=pd.read_csv(folder/'seed123_test_predictions.csv')
        if kind=='sige':
            g=pd.read_csv(folder/'bc_sv_test_pairs.csv')
            g['comparison'] = g.host+'-'+g.dopant
            g['cluster'] = g.dopant
            t['cluster'] = t.source_id.map(lambda sid:Path(sid).stem.split('-')[-1])
        else:
            g=pd.read_csv(folder/'test_pairs.csv')
            g['comparison'] = g.id_a+' / '+g.id_b
            g['cluster'] = g.formula
            contract=json.loads((run/'data_contract.json').read_text())
            if t.source_id.tolist()!=contract['partitions'][meta['fold']]['test']:
                raise ValueError('Test predictions differ from the pre-specified hBN fold')
            names_to_formula={row[k]:row.formula for _,row in g.iterrows() for k in ('id_a','id_b')}
            t['cluster']=t.source_id.map(names_to_formula)
            if t.cluster.isna().any():raise ValueError('Missing composition in the hBN test fold')
        points.append(t[['source_id','cluster','target','prediction']])
        pairs.append(g[['comparison','cluster','dft_gap_ev','predicted_gap_ev']])
    p=pd.concat(points).set_index('source_id').sort_index()
    g=pd.concat(pairs).set_index('comparison').sort_index()
    if not p.index.is_unique or not g.index.is_unique:raise ValueError('Repeated out-of-fold records')
    return p,g,fits


def paired_interval(excess,groups):
    by_group=defaultdict(list)
    for value,group in zip(excess,groups,strict=True):by_group[group].append(value)
    names=sorted(by_group)
    counts=np.array([len(by_group[n]) for n in names])
    sums=np.array([sum(by_group[n]) for n in names])
    weights=np.random.default_rng(123).multinomial(len(names),np.full(len(names),1/len(names)),size=10000)
    return np.quantile((weights@sums)/(weights@counts),[.025,.975]).tolist()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset',choices=['sige','hbn'])
    parser.add_argument('--run-dir',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--modes',nargs='+',choices=MODES,default=list(MODES))
    args=parser.parse_args()
    which=args.dataset
    run=args.run_dir or (SIGE_RUN if which=='sige' else HBN_RUN)
    out=args.output or OUTPUT/which;out.mkdir(parents=True,exist_ok=True)
    pref,gref,_=inputs(which,run,'hetero')
    summary=[];groups=[];cohort=[]
    for mode in args.modes:
        pt,g,fits=inputs(which,run,mode)
        if not pt[['cluster','target']].equals(pref[['cluster','target']]) or not g[['cluster','dft_gap_ev']].equals(gref[['cluster','dft_gap_ev']]):
            raise ValueError(f'{mode}: predictions on different test cases')
        pe=np.abs(pt.prediction-pt.target)
        ge=np.abs(g.predicted_gap_ev-g.dft_gap_ev)
        pex=pe-np.abs(pref.prediction-pref.target)
        gex=ge-np.abs(gref.predicted_gap_ev-gref.dft_gap_ev)
        pl,ph=paired_interval(pex.to_numpy(),pt.cluster)
        gl,gh=paired_interval(gex.to_numpy(),g.cluster)
        informative=np.abs(g.dft_gap_ev)>0.02
        correct=g.loc[informative,'predicted_gap_ev']*g.loc[informative,'dft_gap_ev']>0
        ref_correct=gref.loc[informative,'predicted_gap_ev']*gref.loc[informative,'dft_gap_ev']>0
        rank_difference=correct.astype(float)-ref_correct.astype(float)
        rl,rh=paired_interval(rank_difference.to_numpy(),g.loc[informative,'cluster'])
        for name,index in g.groupby('cluster').groups.items():
            groups.append(dict(mode=mode,group=name,pairs=len(index),gap_mae_ev=ge.loc[index].mean(),
                               zero_gap_mae_ev=np.abs(g.dft_gap_ev.loc[index]).mean()))
        frame=g.copy();frame['mode']=mode;cohort.append(frame)
        summary.append(dict(mode=mode,structures=len(pt),point_mae_ev=pe.mean(),
                            excess_point_mae_vs_hetero_ev=pex.mean(),point_excess_ci_low=pl,point_excess_ci_high=ph,
                            pairs=len(g),clusters=g.cluster.nunique(),gap_mae_ev=ge.mean(),
                            macro_cluster_gap_mae_ev=ge.groupby(g.cluster).mean().mean(),
                            zero_gap_mae_ev=np.abs(g.dft_gap_ev).mean(),
                            excess_gap_mae_vs_hetero_ev=gex.mean(),gap_excess_ci_low=gl,gap_excess_ci_high=gh,
                            ranked_correct=int(correct.sum()),ranked_total=int(informative.sum()),
                            rank_accuracy_excess_vs_hetero=rank_difference.mean(),
                            rank_accuracy_excess_ci_low=rl,rank_accuracy_excess_ci_high=rh,
                            trained_fits=len(fits),epochs_completed=sum(f['epochs_completed'] for f in fits),
                            min_updated_tensors=min(f['updated_trainable_tensors'] for f in fits)))
    pd.DataFrame(summary).sort_values('gap_mae_ev').to_csv(out/'comparison.csv',index=False)
    pd.DataFrame(groups).to_csv(out/'by_cluster.csv',index=False)
    pd.concat(cohort).to_csv(out/'all_test_pairs.csv')
    print(pd.DataFrame(summary)[['mode','point_mae_ev','gap_mae_ev','ranked_correct','ranked_total']].to_string(index=False))


if __name__=='__main__':main()
