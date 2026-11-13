"""Check structural overlap and compare 2DMD-high predictions on identical held-out cases."""
import argparse
import ast
from collections import Counter
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import spglib

from .audit_2dmd_inputs import reference, read_initial
from .replay_2dmd_high_benchmark import DEFAULT_OUTPUT, REPO


def parent_permutations(unit, cell):
    host, frac, _, tree, _=reference(str(unit), cell)
    sym=spglib.get_symmetry((host.cell.array,frac,host.numbers),symprec=1e-3)
    permutations=[]
    for rot, trans in zip(sym['rotations'],sym['translations']):
        dist, idx=tree.query((frac@rot.T+trans)%1.)
        if dist.max()>2e-4 or len(set(idx))!=len(frac) or not np.array_equal(host.numbers[idx],host.numbers):
            raise ValueError(f'Invalid symmetry operation for {unit} supercell {cell}')
        permutations.append(idx)
    return host, tree, np.asarray(permutations)


def fingerprints(data_root, membership, path, density):
    frames=[]
    for folder in sorted((data_root/(density+'_density_defects')).iterdir()):
        if not (folder/'targets.csv.gz').exists():continue
        targets=pd.read_csv(folder/'targets.csv.gz',index_col=0)
        descriptors=pd.read_csv(folder/'descriptors.csv',index_col=0)
        parents={}
        for idx,target in targets.iterrows():
            if idx not in membership:
                continue
            desc=descriptors.loc[target.descriptor_id]; host=desc['base']; shape=tuple(ast.literal_eval(desc['cell']))
            key=host,shape
            if key not in parents:
                unit=(data_root/f'{host}.cif' if density=='high' else
                      data_root/'low_density_defects'/'MoS2'/'unit_cells'/f'{host}.cif')
                parents[key]=parent_permutations(unit,shape)
            parent,tree,perms=parents[key]
            atoms=read_initial(folder/'initial'/f'{idx}.cif')
            dist, positions=tree.query(atoms.get_scaled_positions()%1.)
            if np.max(dist)>2e-3 or len(set(positions))!=len(positions):
                raise ValueError(f'{idx}: structure does not match its parent')
            decoration=np.zeros(len(parent),dtype=np.uint8)
            decoration[positions]=atoms.numbers
            canonical=min(v.tobytes() for v in decoration[perms])
            sig=hashlib.sha256(str(key).encode()+canonical).hexdigest()
            frames.append(dict(source_id=idx,host=host,part=membership[idx],symmetry_fingerprint=sig))
        print('Symmetry audit:',folder.name,len(targets),flush=True)
    table=pd.DataFrame(frames)
    if set(table['source_id'])!=set(membership) or table['source_id'].duplicated().any():
        raise ValueError('Symmetry audit and benchmark membership differ')
    table.to_csv(path,index=False)
    return table


def clustered_intervals(pairs_by_cluster,host_by_cluster,seed=123,replicates=5000):
    """Resample descriptor groups within each host; differences are paired by structure."""
    draws=np.random.default_rng(seed)
    weights=np.zeros((replicates,len(host_by_cluster)),dtype=np.int16)
    for host in sorted(set(host_by_cluster)):
        ids=np.flatnonzero(np.array(host_by_cluster)==host)
        weights[:,ids]=draws.multinomial(len(ids),np.full(len(ids),1/len(ids)),size=replicates)
    count=pairs_by_cluster[:,0]
    denom=weights@count
    scores=(weights@pairs_by_cluster[:,1]) / np.maximum(denom,1)
    return np.quantile(scores[denom>0],[.025,.975]).tolist()


def analyze_subset(t,available,scope):
    out=[]; hosts=[]
    groups=t.groupby(['host','descriptor'],sort=True)
    for mode in available:
        group_rows=[]
        err=np.abs(t[mode]-t['target_ev_per_defect'])
        point0=np.abs(t['hetero']-t['target_ev_per_defect'])
        all_sign=[];all_gap=[];ref_gap=[];zero_error=0.;group_gap_errors=[]
        rank_counts={.001:[0,0],.010:[0,0]}
        for (host,_),g in groups:
            truth=g['target_ev_per_defect'].to_numpy()
            estimate=g[mode].to_numpy();reference_est=g['hetero'].to_numpy()
            i,j=np.triu_indices(len(g),1)
            gap=truth[i]-truth[j]
            zero_error+=np.abs(gap).sum()
            ep=np.abs((estimate[i]-estimate[j])-gap)
            eh=np.abs((reference_est[i]-reference_est[j])-gap)
            resolved=np.abs(gap)>1.e-6
            correct=gap*(estimate[i]-estimate[j])>0
            all_sign.extend(correct[resolved].tolist())
            for threshold,counts in rank_counts.items():
                usable=np.abs(gap)>threshold
                counts[0]+=int(correct[usable].sum());counts[1]+=int(usable.sum())
            all_gap.extend(ep.tolist()); ref_gap.extend(eh.tolist())
            if len(ep):group_gap_errors.append(float(ep.mean()))
            group_rows.append([host,len(g),(err.loc[g.index]-point0.loc[g.index]).sum(),len(i),sum(ep-eh)])
        host_names=[r[0] for r in group_rows]
        vals=np.array([r[1:] for r in group_rows])
        low,high=clustered_intervals(vals[:,:2],host_names)
        gl,gh=clustered_intervals(vals[:,2:],host_names)
        out.append(dict(scope=scope,mode=mode,structures=len(t),mae_ev_defect=float(err.mean()),
                        mean_error_excess_over_hetero=float((err-point0).mean()),
                        excess_95_low=low,excess_95_high=high,
                        matched_arrangement_pairs=len(all_gap),pair_gap_mae_ev_defect=np.mean(all_gap),
                        zero_gap_mae_ev_defect=zero_error/len(all_gap),
                        macro_descriptor_gap_mae_ev_defect=float(np.mean(group_gap_errors)),
                        pair_gap_excess_over_hetero=np.mean(all_gap)-np.mean(ref_gap),
                        pair_excess_95_low=gl,pair_excess_95_high=gh,
                        resolvable_pairs=len(all_sign),resolved_pair_order_accuracy=np.mean(all_sign),
                        pairs_gt_1mev=rank_counts[.001][1],correct_gt_1mev=rank_counts[.001][0],
                        pairs_gt_10mev=rank_counts[.010][1],correct_gt_10mev=rank_counts[.010][0]))
        for host,sub in t.groupby('host'):
            hosts.append(dict(scope=scope,host=host,mode=mode,n=len(sub),
                              mae_ev_defect=float(err.loc[sub.index].mean())))
    return out,hosts


def largest_group(t,available,root,output):
    """Describe the most populated fixed composition, chosen by case count alone."""
    group=t.groupby(['host','descriptor']).size().sort_values(ascending=False)
    host,desc=group.index[0]
    cases=t[(t.host==host)&(t.descriptor==desc)].copy()
    source=root/'low_density_defects'/host
    descriptors=pd.read_csv(source/'descriptors.csv',index_col=0)
    targets=pd.read_csv(source/'targets.csv.gz',index_col=0)
    cases['dft_total_ev_cell']=targets.loc[cases.index,'energy']
    cases.to_csv(output/'largest_fixed_composition_cases.csv')
    pairs,_=analyze_subset(cases,available,'largest_fixed_composition')
    descriptive=['mode','structures','mae_ev_defect','matched_arrangement_pairs','pair_gap_mae_ev_defect',
                 'pairs_gt_1mev','correct_gt_1mev','pairs_gt_10mev','correct_gt_10mev']
    metrics=pd.DataFrame(pairs)[descriptive].set_index('mode')
    n=len(cases)
    weights=np.random.default_rng(123).multinomial(n,np.full(n,1/n),size=5000).astype(float)
    denominator=n*n-np.sum(weights*weights,axis=1)
    y=cases.target_ev_per_defect.to_numpy()
    truth=y[:,None]-y[None,:]
    h=cases.hetero.to_numpy()
    ref_error=np.abs((h[:,None]-h[None,:])-truth)
    for mode in available:
        p=cases[mode].to_numpy()
        excess=np.abs((p[:,None]-p[None,:])-truth)-ref_error
        samples=np.sum((weights@excess)*weights,axis=1)/denominator
        lo,hi=np.quantile(samples,[.025,.975])
        metrics.loc[mode,['pair_gap_excess_vs_hetero','occupation_resample_95_low','occupation_resample_95_high']]=[
            excess.sum()/(n*(n-1)),lo,hi]
    metrics.to_csv(output/'largest_fixed_composition_metrics.csv')
    original=descriptors.loc[desc]
    detail=dict(selection='Largest number of eligible symmetry-distinct held-out arrangements',
                host=host,descriptor=desc,cell=ast.literal_eval(original['cell']),
                defect_inventory=ast.literal_eval(original['defects']),test_cases=len(cases),
                unordered_pairs=len(cases)*(len(cases)-1)//2,
                uncertainty='Resample whole held-out occupations with replacement, exclude self comparisons',
                dft_cell_energy_range_ev=float(cases.dft_total_ev_cell.max()-cases.dft_total_ev_cell.min()))
    (output/'largest_fixed_composition.json').write_text(json.dumps(detail,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--density',choices=('high','low'),default='high')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--data-root',type=Path,default=REPO.parent/'dataset'/'2d-materials-point-defects-all')
    parser.add_argument('--symmetry-only',action='store_true')
    args=parser.parse_args()
    args.output=args.output or DEFAULT_OUTPUT.parent/('2dmd_'+args.density)
    membership=json.loads((args.output/'membership.json').read_text())
    flat={k:p for p,ids in membership.items() for k in ids}
    sp=args.output/'occupation_fingerprints.csv'
    t=pd.read_csv(sp) if sp.exists() else fingerprints(args.data_root,flat,sp,args.density)
    if dict(zip(t.source_id,t.part))!=flat:raise ValueError('Saved fingerprints belong to another split')
    if args.symmetry_only:return
    replay=json.loads((args.output/'replay.json').read_text())
    available=[r['mode'] for r in replay if r['status']=='replayed']
    all_data=None
    for m in available:
        tab=pd.read_csv(args.output/(m+'_predictions.csv')).set_index('source_id')
        assert list(tab.index)==membership['test']
        if all_data is None:
            all_data=tab[['host','descriptor','composition','defect_count','target_ev_per_defect']].copy()
        else:
            assert tab[['host','descriptor','composition','defect_count','target_ev_per_defect']].equals(
                all_data[['host','descriptor','composition','defect_count','target_ev_per_defect']])
        all_data[m]=tab.prediction_ev_per_defect
    all_data['symmetry_fingerprint']=t.set_index('source_id').loc[all_data.index,'symmetry_fingerprint']
    seen=set(t.loc[t.part!='test','symmetry_fingerprint'])
    novel=all_data.loc[~all_data.symmetry_fingerprint.isin(seen)].drop_duplicates('symmetry_fingerprint')
    summary=[];byhost=[]
    scopes=[('all_saved_test',all_data),('novel_occupations_only',novel)]
    if args.density=='low':
        scopes.append(('novel_multi_defect',novel[novel.defect_count>=2]))
    for label,tab in scopes:
        m,h=analyze_subset(tab,available,label)
        summary.extend(m);byhost.extend(h)
    pd.DataFrame(summary).to_csv(args.output/'paired_summary.csv',index=False)
    pd.DataFrame(byhost).to_csv(args.output/'all_hosts.csv',index=False)
    if args.density=='low':
        largest_group(novel[novel.defect_count>=2],available,args.data_root,args.output)
    review=dict(n=len(t),split_counts=Counter(t.part),test=len(all_data),novel_test=len(novel),
                test_seen_in_train_or_validation=int(all_data.symmetry_fingerprint.isin(seen).sum()),
                novel_duplicate_test_rows=int((~all_data.symmetry_fingerprint.isin(seen)).sum()-len(novel)),
                interpretation='New occupations are inequivalent under operations preserving the typed pristine parent and supercell. The bootstrap resamples defect descriptors within host and is conditional on these trained weights.')
    (args.output/'structural_overlap.json').write_text(json.dumps(review,indent=2)+'\n')
    print(review,flush=True)


if __name__=='__main__':
    main()

