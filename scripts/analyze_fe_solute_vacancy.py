"""Verify converged out-of-fold binding predictions and compare shell preferences in Fe."""
import argparse
import io
from itertools import combinations
import json
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..data.fe_solute_vacancy import RESULTS, SOURCE
from .alignn_convergence import convergence_audit, plot_learning_curves
from .run_fe_solute_vacancy import DEFAULT_RUN, MODES, THRESHOLD, equal_solute_pairs


def gather(run, seed, labels, folds):
    structs, pairs, reports, panels = [], [], [], []
    for mode in MODES:
        for fold, ids in enumerate(folds):
            base = run/f'fold{fold}'/f'seed{seed}'/mode
            report = json.loads((base/'metrics.json').read_text(encoding='utf8'))
            folder = base/report['selected_run']
            audit = convergence_audit(folder, seed)
            s = pd.read_csv(base/'test_structures.csv',keep_default_na=False)
            p = pd.read_csv(base/'test_pairs.csv',keep_default_na=False)
            if (s.source_id.tolist() != ids['test'] or s.source_id.nunique()!=45 or len(p)!=45
                    or report['updated_trainable_tensors']<=0 or report['mode']!=mode or report['fold']!=fold
                    or not audit['stopped_on_validation_plateau'] or audit['epochs_completed']!=report['epochs_completed']
                    or not np.allclose(s.dft_binding_eV,labels.loc[s.source_id,'binding_eV'],atol=1e-8,rtol=0)):
                raise ValueError(f'Incorrect test cohort, DFT label or training termination in {mode} fold {fold}')
            reference = pd.DataFrame(equal_solute_pairs(s.to_dict('records')))
            keys=['fold','solute','id_a','id_b']
            errors=['dft_gap_eV','predicted_gap_eV','abs_gap_error_eV']
            if not reference[keys].equals(p[keys]) or not np.allclose(reference[errors],p[errors],atol=1e-7,rtol=0):
                raise ValueError(f'Pair table disagrees with the held-out structures: {mode} fold {fold}')
            structs.append(s.assign(mode=mode))
            pairs.append(p.assign(mode=mode))
            reports.append(report)
            if mode in ('hetero','full'):
                panels.append((f'{mode}, fold {fold+1}',folder,audit))
    return pd.concat(structs,ignore_index=True),pd.concat(pairs,ignore_index=True),pd.DataFrame(reports),panels


def baselines(labels, folds):
    structs, pairs=[],[]
    for fold,ids in enumerate(folds):
        train=labels.loc[ids['train']]
        shell=train.groupby('shell').binding_eV.mean()
        for mode in ('shell_mean','composition_only'):
            rows=[]
            for sid in ids['test']:
                row=labels.loc[sid]
                pred=shell[row.shell] if mode=='shell_mean' else train.binding_eV.mean()
                rows.append(dict(fold=fold,source_id=sid,solute=row.solute,shell=int(row.shell),
                                 dft_binding_eV=row.binding_eV,predicted_binding_eV=float(pred)))
            structs.append(pd.DataFrame(rows).assign(mode=mode))
            pairs.append(pd.DataFrame(equal_solute_pairs(rows)).assign(mode=mode))
    return pd.concat(structs,ignore_index=True),pd.concat(pairs,ignore_index=True)


def results_table(structs,pairs):
    by_solute=(pairs.groupby(['fold','solute','mode']).abs_gap_error_eV.mean().unstack('mode').sort_index())
    point=(structs.assign(error=(structs.predicted_binding_eV-structs.dft_binding_eV).abs())
           .groupby(['fold','solute','mode']).error.mean().unstack('mode').reindex(by_solute.index))
    informative=pairs.dft_gap_eV.abs()>THRESHOLD
    signs=(pairs.assign(correct=informative & (pairs.predicted_gap_eV*pairs.dft_gap_eV>0))
           .groupby(['fold','solute','mode']).correct.sum().unstack('mode').reindex(by_solute.index))
    counts=(pairs[informative & (pairs['mode']=='hetero')].groupby(['fold','solute']).size()
            .reindex(by_solute.index,fill_value=0).to_numpy())
    if by_solute.shape != (45,len(MODES)+2) or by_solute.isna().any().any():
        raise ValueError('Missing solute-level paired comparisons')
    # Bootstrap entire solutes stratified by held-out fold. The three shell gaps
    # belonging to one solute always move together.
    rng=np.random.default_rng(20261110)
    weights=np.zeros((10000,45))
    for fold in range(3):
        idx=np.flatnonzero(by_solute.index.get_level_values('fold')==fold)
        weights[:,idx]=rng.multinomial(len(idx),np.full(len(idx),1/len(idx)),size=10000)
    weights/=45
    rows=[]
    preferences=[]
    for mode in (*MODES,'shell_mean','composition_only'):
        p=pairs[pairs['mode']==mode]
        s=structs[structs['mode']==mode]
        if len(p)!=135 or len(s)!=135 or s.source_id.nunique()!=135:
            raise ValueError(f'Incomplete three-shell coverage: {mode}')
        good=p[p.dft_gap_eV.abs()>THRESHOLD]
        products=good.dft_gap_eV*good.predicted_gap_eV
        true=s.pivot(index='solute',columns='shell',values='dft_binding_eV')
        pred=s.pivot(index='solute',columns='shell',values='predicted_binding_eV')
        best=np.sort(true.to_numpy(),axis=1)
        resolved=np.round(best[:,-1]-best[:,-2],6)>THRESHOLD
        predicted_unique=(pred.eq(pred.max(axis=1),axis='index').sum(axis=1)==1)
        true_shell=true.idxmax(axis=1)
        predicted_shell=pred.idxmax(axis=1).where(predicted_unique)
        choose=(true_shell==predicted_shell) & predicted_unique
        second=resolved & (true_shell==2)
        for i,solute in enumerate(true.index):
            preferences.append(dict(mode=mode,solute=solute,dft_preferred_shell=true_shell.loc[solute],
                                    predicted_preferred_shell=predicted_shell.loc[solute],
                                    dft_best_vs_next_eV=round(best[i,-1]-best[i,-2],6),
                                    resolved=bool(resolved[i]),correct=bool(choose.loc[solute])))
        delta=by_solute.hetero.to_numpy()-by_solute[mode].to_numpy()
        lo,hi=np.quantile(weights@delta,[.025,.975])
        point_delta=(point.hetero-point[mode]).to_numpy()
        plo,phi=np.quantile(weights@point_delta,[.025,.975])
        score_delta=(signs.hetero-signs[mode]).to_numpy()
        slo,shi=np.quantile((weights@score_delta)/(weights@counts),[.025,.975])
        rows.append(dict(mode=mode,gap_MAE_eV=p.abs_gap_error_eV.mean(),
                         binding_MAE_eV=(s.predicted_binding_eV-s.dft_binding_eV).abs().mean(),
                         correct_pairs=int((products>0).sum()),informative_pairs=len(good),
                         correct_preferred_shell=int(choose[resolved].sum()),resolved_solutes=int(resolved.sum()),
                         correct_2NN_preference=int(choose[second].sum()),resolved_2NN_solutes=int(second.sum()),
                         tied_pair_predictions=int((products==0).sum()),total_pairs=len(p),
                         hetero_minus_mode_eV=delta.mean(),paired_ci_low_eV=lo,paired_ci_high_eV=hi,
                         hetero_minus_mode_binding_eV=point_delta.mean(),binding_ci_low_eV=plo,binding_ci_high_eV=phi,
                         solutes_with_lower_hetero_binding_MAE=int((point_delta<0).sum()),
                         solutes_with_lower_hetero_gap_MAE=int((delta<0).sum()),
                         hetero_minus_mode_order_fraction=score_delta.sum()/counts.sum(),
                         order_ci_low_fraction=slo,order_ci_high_fraction=shi))
    pairwise=[]
    for left,right in combinations(MODES,2):
        errors=np.column_stack((point[left]-point[right],by_solute[left]-by_solute[right],signs[left]-signs[right]))
        boot=weights@errors
        boot[:,2]/=weights@counts
        lo,hi=np.quantile(boot,[.025,.975],axis=0)
        pairwise.append(dict(model_a=left,model_b=right,binding_MAE_a_minus_b_eV=errors[:,0].mean(),
                             binding_ci_low_eV=lo[0],binding_ci_high_eV=hi[0],gap_MAE_a_minus_b_eV=errors[:,1].mean(),
                             gap_ci_low_eV=lo[1],gap_ci_high_eV=hi[1],order_a_minus_b_fraction=errors[:,2].sum()/counts.sum(),
                             order_ci_low_fraction=lo[2],order_ci_high_fraction=hi[2]))
    return pd.DataFrame(rows).sort_values('gap_MAE_eV'),by_solute,point,pd.DataFrame(preferences),pd.DataFrame(pairwise)


def plot_results(path,table,pairs):
    fig,(ax,bx)=plt.subplots(1,2,figsize=(10.6,5.0),gridspec_kw={'width_ratios':[1,1.35]})
    modes=['hetero','full_x']
    selected=pairs[pairs['mode'].isin(modes)]
    limit=1.05*selected[['dft_gap_eV','predicted_gap_eV']].abs().max().max()
    ax.plot([-limit,limit],[-limit,limit],ls='--',color='#6b6f73',lw=1)
    for mode, marker, color in [('full_x','o','#788aa2'),('hetero','^','#166a60')]:
        p=selected[selected['mode']==mode]
        ax.scatter(p.dft_gap_eV,p.predicted_gap_eV,s=19,alpha=.58,marker=marker,color=color,label=mode)
    ax.set(xlabel='DFT shell energy gap (eV)',ylabel='Predicted shell energy gap (eV)',
           xlim=(-limit,limit),ylim=(-limit,limit),aspect='equal')
    ax.legend(frameon=False)
    bars=table.iloc[::-1]
    positions=np.arange(len(bars))
    bx.barh(positions+.17,bars.binding_MAE_eV,height=.33,color='#8196ac',label='Binding energy')
    bx.barh(positions-.17,bars.gap_MAE_eV,height=.33,color='#26796f',label='Shell energy gap')
    bx.set_yticks(positions,bars['mode'])
    bx.set_xlabel('Held-out MAE (eV/complex)')
    bx.legend(frameon=False,loc='upper right',fontsize=8)
    ax.set_title('Fe solute–vacancy pairs',loc='left',fontsize=11)
    bx.set_title('Unseen solutes, three shell arrangements',loc='left',fontsize=11)
    for axis in (ax,bx):
        axis.spines[['top','right']].set_visible(False)
        axis.tick_params(labelsize=9)
    fig.tight_layout()
    fig.savefig(path,dpi=180,bbox_inches='tight')
    plt.close(fig)


def check_literature_labels(structs, reference):
    labels=reference.set_index('source_id').reference_binding_eV
    if len(labels)!=66 or not labels.index.is_unique:
        raise ValueError('Expected the complete 22-solute repeat-label cohort')
    summaries=[]
    for mode,s in structs[structs.source_id.isin(labels.index)].groupby('mode'):
        if len(s)!=66 or s.source_id.nunique()!=66:
            raise ValueError(f'Missing independent reference for {mode}')
        repeated=s.copy()
        repeated['dft_binding_eV']=labels.loc[s.source_id].to_numpy()
        pairs=pd.DataFrame(equal_solute_pairs(repeated.to_dict('records')))
        informative=pairs.dft_gap_eV.abs()>THRESHOLD
        scores=pairs.loc[informative,'dft_gap_eV']*pairs.loc[informative,'predicted_gap_eV']
        summaries.append(dict(mode=mode,solutes=22,structures=66,pairs=66,
                              gap_MAE_eV=pairs.abs_gap_error_eV.mean(),
                              binding_MAE_eV=(repeated.predicted_binding_eV-repeated.dft_binding_eV).abs().mean(),
                              pairs_correct=int((scores>0).sum()),informative_pairs=int(informative.sum()),
                              between_DFT_binding_MAE_eV=float(np.abs(repeated.dft_binding_eV.to_numpy()-s.dft_binding_eV.to_numpy()).mean())))
    return pd.DataFrame(summaries).sort_values('gap_MAE_eV')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=DEFAULT_RUN)
    parser.add_argument('--results',type=Path,default=RESULTS)
    parser.add_argument('--seed',type=int,default=123)
    args=parser.parse_args()
    out=args.results
    contract=json.loads((args.run_dir/'data_contract.json').read_text(encoding='utf8'))
    folds=contract['partitions']
    with zipfile.ZipFile(out/'Fe_solute_vacancy_DFT.zip') as z:
        labels=pd.read_csv(io.BytesIO(z.read('records.csv')),keep_default_na=False).set_index('source_id')
        if folds!=json.loads(z.read('folds.json')):
            raise ValueError('Model and released data use different folds')
    s,p,audit,panels=gather(args.run_dir,args.seed,labels,folds)
    b_s,b_p=baselines(labels,folds)
    s=pd.concat([s,b_s],ignore_index=True)
    p=pd.concat([p,b_p],ignore_index=True)
    table,per_solute,per_binding,preferences,pairwise=results_table(s,p)
    literature=None
    reference_path=out/'olsson_2010_reference.csv'
    if reference_path.exists():
        literature=check_literature_labels(s,pd.read_csv(reference_path))
        literature.to_csv(out/'independent_DFT_check.csv',index=False)
    for df, name in ((s,'oof_structures'),(p,'matched_energy_gaps'),(audit,'convergence'),
                     (table,'model_comparison'),(per_solute.reset_index(),'by_solute'),
                     (per_binding.reset_index(),'binding_by_solute'),(preferences,'preferred_shells'),
                     (pairwise,'paired_model_comparisons')):
        df.to_csv(out/(name+'.csv'),index=False)
    plot_results(out/'heldout_energy_gaps.png',table,p)
    plot_learning_curves(panels,args.seed,out/'learning_curves.png')
    lines=['# BCC Fe 中溶质–空位的近邻排布', '',
           'Kartamyshev 等人公布了铁空位与置换溶质处在第一、第二和第三近邻时的 DFT 结合能，单位为 eV。'
           f'工作簿包括 45 种异种溶质及一行全零的 Fe 自身对照。另一个 CSV 的 120 个重叠数值与母表全部一致。[作者数据]({SOURCE})', '',
           '作者补充方法使用自旋极化 VASP、PAW-PBE、400 eV 平面波截断及 3×3×3 k 点网格。'
           '每种溶质 X 的三个构型均为 Fe₅₂X 加一个空位。输入坐标按作者指定的 3×3×3 常规 bcc 超胞和三个壳层重建，'
           '理想晶格常数统一采用 ASE 的 Fe 参考值 2.87 Å。模型只接收结构以及相应模式可见的缺陷标记。'
           '源数据给出结合能，结合越强，构型总能越低。固定 X 后，E(B)−E(A) 等于结合能(A)−结合能(B)。', '',
           '例如 Ni 的三个结合能为 0.105、0.220、0.046 eV，Cu 为 0.254、0.219、0.054 eV。'
           '45 种溶质中有 39 种偏好第一近邻，H、Be、B、F、Co 和 Ni 偏好第二近邻。表中单列这 6 种元素的识别结果。', '',
           '## 留出比较', '',
           '三折按溶质元素划分。同种元素的三个壳层始终在同一分区，每折有 24 种训练、6 种验证、15 种测试元素。'
           '测试中的 45 种元素各留出一次，共 135 个同成分排布对。按训练集中各壳层平均结合能预测是一个可复算的参照。'
           '排序统计使用 DFT 能隙大于 0.010 eV 的构型对，最优壳层正确率也只计算第一与第二名相差超过该门槛的元素。', '',
           '| 模型 | 能量差 MAE（eV） | 结合能 MAE（eV） | 两两排序 | 首选壳层 | 次近邻偏好识别 |',
           '|---|---:|---:|---:|---:|---:|']
    for r in table.itertuples(index=False):
        pair=f'{r.correct_pairs}/{r.informative_pairs}' if r.mode!='composition_only' else '全平局'
        top=f'{r.correct_preferred_shell}/{r.resolved_solutes}' if r.mode!='composition_only' else '全平局'
        second=f'{r.correct_2NN_preference}/{r.resolved_2NN_solutes}' if r.mode!='composition_only' else '全平局'
        lines.append(f'| {r.mode} | {r.gap_MAE_eV:.5f} | {r.binding_MAE_eV:.5f} | {pair} | {top} | {second} |')
    if (out/'replication_differences.csv').exists():
        repeated=pd.read_csv(out/'replication_by_seed.csv').groupby('mode')[['binding_MAE_eV','gap_MAE_eV']].mean()
        d=pd.read_csv(out/'replication_differences.csv',dtype={'seed':str}).set_index('seed').loc['mean_over_seeds']
        best_binding=table.loc[table.binding_MAE_eV.idxmin(),'mode']
        lines[2:2]=[f'三种随机初始化下，普通 hetero 的留出结合能 MAE 平均为 {repeated.loc["hetero","binding_MAE_eV"]:.5f} eV，'
                    f'显式空位对照 full_x 为 {repeated.loc["full_x","binding_MAE_eV"]:.5f} eV。'
                    f'平均降幅为 {-d.hetero_minus_full_x_binding_MAE_eV/repeated.loc["full_x","binding_MAE_eV"]:.1%}，'
                    f'按溶质配对的误差降低量 95% 区间为 [{-d.binding_ci_high_eV:.5f}, {-d.binding_ci_low_eV:.5f}] eV。'
                    f'能量差 MAE 的三种子差值为 {d.hetero_minus_full_x_gap_MAE_eV:+.5f} eV，'
                    f'区间 [{d.gap_ci_low_eV:+.5f}, {d.gap_ci_high_eV:+.5f}] eV。十种模型的主比较中，'
                    f'{best_binding} 的结合能 MAE 最低。','']
    indexed=table.set_index('mode')
    lines.extend(['','[能量预测图](heldout_energy_gaps.png) · [逐构型预测](oof_structures.csv) · [同成分能量差](matched_energy_gaps.csv)',
                  '', '普通 hetero 相对两个基础模型的误差差（hetero 减对照）按留出元素整组、在各折内配对重采样 10,000 次。', ''])
    for m in ('full','full_x'):
        row=indexed.loc[m]
        lines.append(f'- 相对 {m}，能量差 MAE 的差值为 {row.hetero_minus_mode_eV:+.5f} eV，'
                     f'95% 区间为 [{row.paired_ci_low_eV:+.5f}, {row.paired_ci_high_eV:+.5f}] eV。'
                     f'单构型结合能 MAE 的差值为 {row.hetero_minus_mode_binding_eV:+.5f} eV，'
                     f'区间为 [{row.binding_ci_low_eV:+.5f}, {row.binding_ci_high_eV:+.5f}] eV。'
                     f'在 {int(row.solutes_with_lower_hetero_binding_MAE)}/45 种溶质上，hetero 的结合能 MAE 较低。')
    symmetry_path=out/'all_shell_orientations.csv'
    if symmetry_path.exists():
        symmetry=pd.read_csv(symmetry_path).set_index('mode')
        if not {'hetero','full_x'}<=set(symmetry.index) or (symmetry.equivalent_placements!=1170).any():
            raise ValueError('Incomplete symmetry comparison')
        lines.extend(['','## 立方对称等价位置','',
                      '固定溶质和壳层，第一、第二、第三近邻分别有 8、6、12 个由立方对称操作联系的空位位置。'
                      f'用种子 123 的同一组留出权重预测每种溶质的全部位置，已复核的 {len(symmetry)} 个模型各得到 1,170 个方向预测。'
                      '这些位置共对应 135 个原始 DFT 标签。每个构型先计算所有方向中的最大预测值减最小预测值，再对 135 个构型等权取平均。'
                      '这项检查衡量当前 HERA 模型及近邻截断的联合表现。', '',
                      '| 模型 | 平均等价方向跨度（meV） | 最大跨度（meV） | 方向平均后结合能 MAE（eV） | 方向平均后能量差 MAE（eV） |',
                      '|---|---:|---:|---:|---:|'])
        for m in symmetry.mean_symmetry_range_eV.sort_values().index:
            rr=symmetry.loc[m]
            lines.append(f'| {m} | {rr.mean_symmetry_range_eV*1000:.3f} | {rr.max_symmetry_range_eV*1000:.3f} | '
                         f'{rr.symmetry_averaged_binding_MAE_eV:.5f} | {rr.symmetry_averaged_gap_MAE_eV:.5f} |')
        lines.extend(['','[等价方向散点图](cubic_symmetry_spread.png) · [逐溶质、逐壳层的方向数、均值和范围](all_shell_orientations_by_structure.csv)'])
        boundary_path=out/'all_shell_orientations_complete_boundary_shell.csv'
        if boundary_path.exists():
            control=pd.read_csv(boundary_path).set_index('mode')
            if set(control.index)!={'hetero','full_x'} or (control.equivalent_placements!=1170).any():
                raise ValueError('Incomplete equal-distance cutoff control')
            lines.extend(['','保持原训练权重，只在推理时将每类邻居在第 12 个位置处并列的整个距离壳层纳入。'
                          f'平均等价方向跨度降为 hetero {control.loc["hetero","mean_symmetry_range_eV"]:.2e}、'
                          f'full_x {control.loc["full_x","mean_symmetry_range_eV"]:.2e} eV。'
                          '补齐截断边界后两者都保持立方等价性。这定位了默认结果对截断并列邻居的敏感性。'
                          '该控制使用原权重，新的邻居表未经重新训练。'])
    if (out/'replication.md').exists():
        lines.extend(['','两个额外初始化种子的配对复验见[三种子结果](replication.md)。'])
    if literature is not None:
        lt=literature.set_index('mode')
        lines.extend(['', '## 独立 DFT 标注复核', '',
                      '作者补充表 S3 括号内列出 Olsson 等人对相同近邻占位的独立 DFT 计算。'
                      '两边都有完整三个构型且有留出预测的 22 种过渡金属全部纳入，共 66 个重标注构型。'
                      '各预测仍出自将该溶质整组留出的模型。复核复用原预测，覆盖与主试验重合的构型。'
                      '两组计算的单构型结合能平均相差 '
                      f'{lt.iloc[0].between_DFT_binding_MAE_eV:.5f} eV。[原研究](https://doi.org/10.1103/PhysRevB.81.054102)', '',
                      f'独立标注下，能量差 MAE 分别为 hetero {lt.loc["hetero","gap_MAE_eV"]:.5f}、'
                      f'full {lt.loc["full","gap_MAE_eV"]:.5f}、full_x {lt.loc["full_x","gap_MAE_eV"]:.5f} eV。'
                      '全部模式及排序见 independent_DFT_check.csv。'])
    lines.extend(['', '## 训练停止条件', '',
                  f'{len(audit)} 次独立训练共完成 {int(audit.epochs_completed.sum())} 个 epoch，每次 {int(audit.epochs_completed.min())}–{int(audit.epochs_completed.max())} 轮。'
                  f'各模型从随机种子 {args.seed} 初始化。当验证 MAE 连续 {contract["validation_patience"]} 轮没有达到 0.5% 的相对改善时早停。'
                  '测试使用各自验证 MAE 最低轮次的权重。每一条已报告结果均已从 checkpoint 检查早停标记、历史轮数、最佳验证权重及准确的留出身份。', '',
                  '| 模型 | 第 1 折最佳/停止轮 | 第 2 折最佳/停止轮 | 第 3 折最佳/停止轮 |',
                  '|---|---:|---:|---:|'])
    for m in MODES:
        runs=audit[audit['mode']==m].sort_values('fold')
        lines.append('| '+m+' | '+' | '.join(f'{v.best_epoch}/{v.epochs_completed}' for v in runs.itertuples(index=False))+' |')
    lines.extend(['', '单个初始化种子和固定三折只能刻画本次测量。区间条件为这些模型和保留的 45 种元素。'
                  '该任务检验对未知溶质的近邻排序，标签为 DFT 结合能，几何为理想初始占位。', '',
                  '数据包内含原始能量工作簿、源 CSV、补充方法、初始结构及三折身份。[收敛记录](convergence.csv) · [学习曲线](learning_curves.png)', '',
                  '在 HERA 的父目录可用以下命令从公开数据包复现。', '',
                  '```text','python -m HERA.scripts.run_fe_solute_vacancy --archive HERA/results/fe_solute_vacancy_20261110/Fe_solute_vacancy_DFT.zip --run-dir HERA/logs/fe_shell_reproduction --results HERA/results/fe_shell_reproduction',
                  'python -m HERA.scripts.analyze_fe_solute_vacancy --run-dir HERA/logs/fe_shell_reproduction --results HERA/results/fe_shell_reproduction',
                  '```','',
                  '追加独立文献标注时，可在带 pypdf 的 Python 环境运行 HERA/scripts/extract_fe_literature_reference.py，'
                  '以数据包作为 --bundle、结果目录中的 olsson_2010_reference.csv 作为 --output，然后重新生成分析。'
                  'HERA.scripts.check_fe_cubic_symmetry 使用主试验的已训练权重检查完整等价壳层。',''])
    (out/'results.md').write_text('\n'.join(lines),encoding='utf8')
    print(table[['mode','gap_MAE_eV','correct_pairs','correct_preferred_shell']].to_string(index=False))


if __name__=='__main__':main()
