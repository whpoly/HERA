"""Score the fixed-composition refits and resample held-out occupations."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .run_dilute_s6 import DEFAULT_RUN, EVIDENCE, MODES


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=DEFAULT_RUN)
    parser.add_argument('--output',type=Path,default=EVIDENCE.parent/'dilute_s6')
    parser.add_argument('--seed',type=int,default=123)
    args=parser.parse_args()
    contract=json.loads((args.run_dir/'data_contract.json').read_text())
    ids=contract['membership']['test']
    args.output.mkdir(parents=True,exist_ok=True)
    data=None
    fits={}
    for mode in MODES:
        folder=args.run_dir/f'seed{args.seed}'/mode
        t=pd.read_csv(folder/'heldout.csv').set_index('source_id')
        if t.index.tolist()!=ids:raise ValueError(f'{mode}: unexpected test IDs')
        if data is None:data=t[['formation_ev_defect','dft_total_ev_cell']].copy()
        if not data[['formation_ev_defect','dft_total_ev_cell']].equals(t[['formation_ev_defect','dft_total_ev_cell']]):
            raise ValueError('Different target for the same source ID')
        data[mode]=t.prediction_ev_defect
        fits[mode]=json.loads((folder/'metrics.json').read_text())
    y=data.formation_ev_defect.to_numpy()
    i,j=np.triu_indices(len(y),1)
    gap=y[i]-y[j];total=data.dft_total_ev_cell.to_numpy()
    if not np.allclose(total[i]-total[j],3*gap,rtol=0,atol=1e-6):
        raise ValueError('Raw DFT cell energies do not give the same pair differences')
    weights=np.random.default_rng(123).multinomial(len(y),np.full(len(y),1/len(y)),size=10000)
    pair_weights=weights[:,i]*weights[:,j]
    # Avoid self-comparisons when an occupation is drawn several times.
    denominator=pair_weights.sum(axis=1)
    assert denominator.min()>0
    point0=np.abs(data.hetero-y).to_numpy()
    p0=data.hetero.to_numpy();eg0=np.abs((p0[i]-p0[j])-gap)
    mask=np.abs(gap)>contract['ordering_margin_ev']
    ranks0=(p0[i]-p0[j])[mask]*gap[mask]>0
    rank_weights=pair_weights[:,mask]
    rank_denominator=rank_weights.sum(axis=1)
    assert rank_denominator.min()>0
    rows=[]
    for mode in MODES:
        p=data[mode].to_numpy();pg=p[i]-p[j]
        pe=np.abs(p-y);ge=np.abs(pg-gap)
        if not np.isclose(ge.mean(),fits[mode]['gap_mae_ev_defect'],atol=1e-10):
            raise ValueError(f'{mode}: mismatch in pairwise error')
        point_ci=np.quantile((weights@(pe-point0))/len(y),[.025,.975])
        gap_ci=np.quantile((pair_weights@(ge-eg0))/denominator,[.025,.975])
        correct=pg[mask]*gap[mask]>0
        rank_difference=correct.astype(float)-ranks0.astype(float)
        rank_ci=np.quantile((rank_weights@rank_difference)/rank_denominator,[.025,.975])
        rows.append(dict(mode=mode,seed=args.seed,test_cases=len(y),test_pairs=len(i),point_mae_ev_defect=pe.mean(),
                         gap_mae_ev_defect=ge.mean(),zero_gap_mae_ev_defect=np.abs(gap).mean(),
                         point_excess_vs_hetero_ev=(pe-point0).mean(),point_excess_ci_low=point_ci[0],point_excess_ci_high=point_ci[1],
                         gap_excess_vs_hetero_ev=(ge-eg0).mean(),gap_excess_ci_low=gap_ci[0],gap_excess_ci_high=gap_ci[1],
                         ranked_correct=int(correct.sum()),ranked_total=int(mask.sum()),
                         rank_accuracy_excess_vs_hetero=rank_difference.mean(),
                         rank_accuracy_excess_ci_low=rank_ci[0],rank_accuracy_excess_ci_high=rank_ci[1],
                         epochs_completed=fits[mode]['epochs_completed'],updated_tensors=fits[mode]['updated_trainable_tensors']))
    data.to_csv(args.output/'all_test_structures.csv')
    table=pd.DataFrame(rows).sort_values('gap_mae_ev_defect')
    table.to_csv(args.output/'comparison.csv',index=False)
    print(table[['mode','point_mae_ev_defect','gap_mae_ev_defect','ranked_correct','ranked_total']].to_string(index=False))


if __name__=='__main__':main()
