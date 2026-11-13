"""Repeat the Fe binding-energy comparison at fixed new seeds and audit paired results."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from ..data.fe_solute_vacancy import RESULTS, ROOT, SCHEMA
from .alignn_convergence import convergence_audit
from .run_fe_solute_vacancy import DEFAULT_RUN, THRESHOLD


def load_results(run, seed, modes, partitions):
    by_mode={}
    rows=[]
    fits=[]
    for mode in modes:
        all_s, all_p=[], []
        for fold,ids in enumerate(partitions):
            base=run/f'fold{fold}'/f'seed{seed}'/mode
            report=json.loads((base/'metrics.json').read_text(encoding='utf8'))
            audit=convergence_audit(base/report['selected_run'],seed)
            if (not audit['stopped_on_validation_plateau'] or report['updated_trainable_tensors']<1
                    or report['device']!='cuda:0' or report['seed']!=seed):
                raise ValueError(f'Unverified replication training: {base}')
            s=pd.read_csv(base/'test_structures.csv')
            p=pd.read_csv(base/'test_pairs.csv')
            if s.source_id.tolist()!=ids['test'] or len(p)!=45 or set(p.fold)!={fold}:
                raise ValueError(f'Replication holdout changed: {base}')
            all_s.append(s)
            all_p.append(p)
            fits.append(dict(mode=mode,seed=seed,fold=fold,**audit))
        s=pd.concat(all_s,ignore_index=True)
        p=pd.concat(all_p,ignore_index=True)
        point=(s.predicted_binding_eV-s.dft_binding_eV).abs()
        good=p.dft_gap_eV.abs()>THRESHOLD
        correct=good & (p.predicted_gap_eV*p.dft_gap_eV>0)
        ranks=p.assign(order=correct,informative=good).groupby(['fold','solute'])[['order','informative']].sum()
        groups=pd.concat([s.assign(binding=point).groupby(['fold','solute']).binding.mean(),
                          p.groupby(['fold','solute']).abs_gap_error_eV.mean().rename('gap'),ranks],axis=1).sort_index()
        if len(groups)!=45 or len(s)!=135 or len(p)!=135 or s.source_id.nunique()!=135:
            raise ValueError('Incomplete replication')
        by_mode[mode]=groups
        rows.append(dict(seed=seed,mode=mode,binding_MAE_eV=point.mean(),gap_MAE_eV=p.abs_gap_error_eV.mean(),
                         correct_pairs=int(correct.sum()),informative_pairs=int(good.sum())))
    if not by_mode['hetero'].index.equals(by_mode['full_x'].index):
        raise ValueError('Cannot pair replication results')
    return by_mode, rows, fits


def plot_binding_comparison(out,metrics,result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    table=pd.DataFrame(metrics).pivot(index='seed',columns='mode',values='binding_MAE_eV')
    fig,(ax,bx)=plt.subplots(1,2,figsize=(9.2,3.8),gridspec_kw={'width_ratios':[1,1.25]})
    pos=np.arange(len(table))
    for shift,mode,color in [(-.18,'hetero','#19746b'),(.18,'full_x','#8093aa')]:
        ax.bar(pos+shift,table[mode],width=.34,color=color,label=mode)
    ax.set(xticks=pos,xticklabels=[str(i) for i in table.index],xlabel='Initialization seed',
           ylabel='Binding-energy MAE (eV)',ylim=(0,table.to_numpy().max()*1.28))
    ax.legend(frameon=False,loc='upper center',ncol=2)
    ax.set_title('Held-out Fe solute–vacancy binding',loc='left',fontsize=11)
    bx.axvline(0,lw=.9,ls='--',color='#777')
    for i,row in enumerate(result):
        avg=row['hetero_minus_full_x_binding_MAE_eV']
        color='#12544d' if row['seed']=='mean_over_seeds' else '#608c86'
        bx.plot([row['binding_ci_low_eV'],row['binding_ci_high_eV']],[i,i],color=color,lw=2)
        bx.plot(avg,i,'o',color=color)
    bx.set_yticks(range(len(result)),[f"Seed {r['seed']}" if r['seed']!='mean_over_seeds' else 'Three-seed mean' for r in result])
    bx.invert_yaxis()
    bx.set_xlabel('MAE difference, hetero minus full_x (eV)')
    bx.set_title('Paired 95% intervals across 45 solutes',loc='left',fontsize=11)
    for axis in (ax,bx):
        axis.spines[['top','right']].set_visible(False)
        axis.tick_params(labelsize=9)
    fig.tight_layout()
    fig.savefig(out/'binding_energy_replication.png',dpi=185,bbox_inches='tight')
    plt.close(fig)


def summarize(out, primary, log_parent, protocol):
    spec=protocol['focused_replication']
    primary_contract=json.loads((primary/'data_contract.json').read_text(encoding='utf8'))
    models, metrics, fits = {}, [], []
    for seed in [protocol['primary_suite']['seed'],*spec['seeds']]:
        run=primary if seed==protocol['primary_suite']['seed'] else log_parent/f'{SCHEMA}_seed{seed}'
        contract=json.loads((run/'data_contract.json').read_text(encoding='utf8'))
        if (contract|{'init_seed':primary_contract['init_seed']}) != primary_contract:
            raise ValueError('Frozen features, optimization or split protocol changed between seeds')
        models[seed], m, f=load_results(run,seed,spec['modes'],contract['partitions'])
        metrics.extend(m)
        fits.extend(f)
    seeds=list(models)
    weights=np.zeros((10000,45))
    rng=np.random.default_rng(20261110)
    index=models[seeds[0]]['hetero'].index
    for fold in range(3):
        ii=np.flatnonzero(index.get_level_values('fold')==fold)
        weights[:,ii]=rng.multinomial(len(ii),np.full(len(ii),1/len(ii)),10000)/45
    delta={seed:models[seed]['hetero'][['binding','gap','order']].to_numpy()-
                models[seed]['full_x'][['binding','gap','order']].to_numpy() for seed in seeds}
    delta['mean_over_seeds']=np.mean(list(delta.values()),axis=0)
    denominators=models[seeds[0]]['hetero'].informative.to_numpy()
    result=[]
    for seed,v in delta.items():
        b=weights@v
        b[:,2]/=weights@denominators
        lo,hi=np.quantile(b,[.025,.975],axis=0)
        result.append(dict(seed=seed,hetero_minus_full_x_binding_MAE_eV=v[:,0].mean(),
                           binding_ci_low_eV=lo[0],binding_ci_high_eV=hi[0],
                           solutes_with_lower_hetero_binding_MAE=int((v[:,0]<0).sum()),
                           hetero_minus_full_x_gap_MAE_eV=v[:,1].mean(),gap_ci_low_eV=lo[1],gap_ci_high_eV=hi[1],
                           hetero_minus_full_x_order_fraction=v[:,2].sum()/denominators.sum(),
                           order_ci_low=lo[2],order_ci_high=hi[2]))
    pd.DataFrame(metrics).to_csv(out/'replication_by_seed.csv',index=False)
    pd.DataFrame(fits).to_csv(out/'replication_convergence.csv',index=False)
    pd.DataFrame(result).to_csv(out/'replication_differences.csv',index=False)
    plot_binding_comparison(out,metrics,result)
    mean_result=result[-1]
    lines=['# Fe 溶质–空位结合能的初始化复核','',
           '主试验以种子 123 比较全部 10 种 ALIGNN。随后固定种子 456 和 789，单独复核普通 hetero 与 full_x，'
           '使用相同 GPU、相同输入和相同三折，每次从随机权重开始并达到同一验证集早停条件。','',
           '| 种子 | 模型 | 单构型结合能 MAE（eV） | 同溶质能量差 MAE（eV） | 两两排序 |',
           '|---:|---|---:|---:|---:|']
    for m in metrics:
        lines.append(f'| {m["seed"]} | {m["mode"]} | {m["binding_MAE_eV"]:.5f} | {m["gap_MAE_eV"]:.5f} | {m["correct_pairs"]}/{m["informative_pairs"]} |')
    lines.extend(['', '先在每种溶质内对三个种子的绝对误差取平均，再按留出折分层重采样 45 种溶质。'
                  '这样会保留相同构型在不同训练中的配对关系。以下量为 hetero 减 full_x。','',
                  f'- 结合能 MAE 差为 {mean_result["hetero_minus_full_x_binding_MAE_eV"]:+.5f} eV，'
                  f'配对 95% 区间为 [{mean_result["binding_ci_low_eV"]:+.5f}, {mean_result["binding_ci_high_eV"]:+.5f}] eV。'
                  f'在 {mean_result["solutes_with_lower_hetero_binding_MAE"]}/45 种溶质上，hetero 的种子平均结合能 MAE 更低。',
                  f'- 同成分能量差 MAE 差为 {mean_result["hetero_minus_full_x_gap_MAE_eV"]:+.5f} eV，'
                  f'配对 95% 区间为 [{mean_result["gap_ci_low_eV"]:+.5f}, {mean_result["gap_ci_high_eV"]:+.5f}] eV。','',
                  '[逐种子结果](replication_by_seed.csv) · [结合能配对误差图](binding_energy_replication.png) · [收敛记录](replication_convergence.csv)',''])
    (out/'replication.md').write_text('\n'.join(lines),encoding='utf8')
    print(pd.DataFrame(metrics).to_string(index=False),flush=True)
    print(pd.DataFrame(result).to_string(index=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=RESULTS)
    parser.add_argument('--primary-run',type=Path,default=DEFAULT_RUN)
    parser.add_argument('--log-parent',type=Path,default=ROOT/'logs')
    parser.add_argument('--protocol',type=Path,default=RESULTS/'replication_protocol.json')
    parser.add_argument('--analyze-only',action='store_true')
    args=parser.parse_args()
    args.results=args.results.resolve()
    args.primary_run=args.primary_run.resolve()
    args.log_parent=args.log_parent.resolve()
    protocol=json.loads(args.protocol.read_text(encoding='utf8'))
    spec=protocol['focused_replication']
    if not args.analyze_only:
        for seed in spec['seeds']:
            run=args.log_parent/f'{SCHEMA}_seed{seed}'
            run.mkdir(parents=True,exist_ok=True)
            cmd=[sys.executable,'-u','-m','HERA.scripts.run_fe_solute_vacancy',
                 '--archive',str(args.results/'Fe_solute_vacancy_DFT.zip'),'--run-dir',str(run),
                 '--results',str(args.results),'--device',spec['device'],'--patience',str(spec['validation_patience']),
                 '--seed',str(seed),'--mode',*spec['modes']]
            if any(run.glob('fold*/seed*/*/*_history.csv')):
                cmd.append('--resume')
            print('Starting paired seed',seed,flush=True)
            with (run/'training.out').open('a',encoding='utf8') as stdout, (run/'training.err').open('a',encoding='utf8') as stderr:
                subprocess.run(cmd,cwd=ROOT.parent,stdout=stdout,stderr=stderr,check=True,
                               creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    summarize(args.results,args.primary_run,args.log_parent,protocol)


if __name__=='__main__':main()
