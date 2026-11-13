"""Collect audited same-composition energy-gap experiments and their public data sources."""
import argparse
import csv
import json
import os
from pathlib import Path

from ..arrangement_sources import ROOT, SOURCES


def read_table(path):
    with path.open(encoding='utf8',newline='') as handle:
        return list(csv.DictReader(handle))


def collect(name):
    spec = SOURCES[name]
    report = spec.results/'results.md'
    table = read_table(spec.results/'model_comparison.csv')
    fits = read_table(spec.results/'convergence.csv')
    per_seed = read_table(spec.results/'per_seed_comparison.csv')
    control = read_table(spec.results/'physical_control_pairs.csv')
    baseline_results = {r['baseline']:r for r in read_table(spec.results/'baseline_comparison.csv')}
    contract = json.loads((spec.logs/'data_contract.json').read_text(encoding='utf8'))
    scores = {r['mode']:r for r in table}
    if len(table)!=10 or set(scores)!=set(contract['model_names']):
        raise ValueError(f'{name}: incomplete model comparison')
    h = scores['hetero']
    initial = contract['initial_screen_seed']
    screening = next(r for r in per_seed if r['mode']=='hetero' and int(r['seed'])==initial)
    seeds = [initial]
    if int(screening['rank'])==1:
        seeds.extend(contract['confirmation']['seeds'])
    positions = {int(r['seed']):r for r in per_seed if r['mode']=='hetero'}
    if set(positions)!=set(seeds) or len(per_seed)!=len(scores)*len(seeds):
        raise ValueError(f'{name}: incomplete individual-seed rankings')
    fit_ids = {(r['mode'],int(r['fold']),int(r['seed'])) for r in fits}
    expected = {(m,f,s) for m in scores for f in range(3) for s in seeds}
    if (fit_ids!=expected or len(fit_ids)!=len(fits) or int(h['n_seeds'])!=len(seeds)
            or any(r['stopped_on_validation_plateau']!='True' or int(r['updated_trainable_tensors'])<1
                   or int(r['validation_patience'])!=contract['convergence']['patience']
                   or float(r['min_relative_improvement_pct'])!=contract['convergence']['minimum_relative_improvement_percent']
                   for r in fits)):
        raise ValueError(f'{name}: fits or required seed confirmations are incomplete')
    if len(control)!=int(h['n_unique_pairs']):
        raise ValueError(f'{name}: missing common held-out pairs')
    twins = {'hetero':{},'hetero_fixed_pool':{}}
    with (spec.results/'heldout_energy_gaps.csv').open(encoding='utf8',newline='') as handle:
        for item in csv.DictReader(handle):
            mode = item['mode']
            if mode in twins:
                key = (item['seed'],item['fold'],item['group'],item['id_a'],item['id_b'])
                if key in twins[mode]:
                    raise ValueError(f'{name}: duplicated energy difference for {mode}')
                twins[mode][key] = float(item['predicted_gap_eV'])
    if twins['hetero'].keys()!=twins['hetero_fixed_pool'].keys() or len(twins['hetero'])!=len(seeds)*len(control):
        raise ValueError(f'{name}: incomplete predictions for the two default defect-mean modes')
    twin_delta = max(abs(value-twins['hetero_fixed_pool'][key]) for key,value in twins['hetero'].items())
    zero = sum(abs(float(r['dft_gap_eV'])) for r in control)/len(control)
    physical = sum(float(r['gap_error_eV']) for r in control)/len(control)
    other = min((r for m,r in scores.items() if m not in ('hetero','hetero_fixed_pool')),
                key=lambda r:float(r['gap_MAE_eV_cell']))
    ordinary = float(h['gap_MAE_eV_cell'])
    competing = float(other['gap_MAE_eV_cell'])
    zero_check = baseline_results['zero_gap']
    if (int(zero_check['n_seeds'])!=len(seeds) or abs(float(zero_check['hetero_MAE_eV_cell'])-ordinary)>1e-8
            or abs(float(zero_check['baseline_MAE_eV_cell'])-zero)>1e-8):
        raise ValueError(f'{name}: zero-gap check uses a different cohort or seed set')
    cofirst = (int(h['rank'])==1 and len(seeds)>1 and
               ordinary+contract['tie_tolerance_eV']<min(zero,competing))
    epochs = [int(r['epochs_completed']) for r in fits]
    return dict(dataset=name,title=report.read_text(encoding='utf8').splitlines()[0].removeprefix('# '),
                compositions=' / '.join(sorted({r['group'] for r in control})),arrangements=int(h['n_structures']),
                pairs=len(control),seeds=','.join(map(str,seeds)),converged_fits=len(fits),min_epochs=min(epochs),
                max_epochs=max(epochs),hetero_rank=int(h['rank']),hetero_meV_cell=1000*ordinary,
                hetero_ranks_by_seed=' / '.join(f"{s}={positions[s]['rank']}" for s in seeds),
                hetero_MAE_meV_by_seed=' / '.join(f"{s}={1000*float(positions[s]['gap_MAE_eV_cell']):.6f}" for s in seeds),
                first_modes=' / '.join(r['mode'] for r in table if int(r['rank'])==1),
                first_meV_cell=1000*min(float(r['gap_MAE_eV_cell']) for r in table),
                best_distinct_competitor=other['mode'],best_distinct_meV_cell=1000*competing,
                hetero_minus_best_distinct_meV_cell=1000*(ordinary-competing),
                paired_ci_low_meV=1000*float(other['paired_ci_low_eV']),
                paired_ci_high_meV=1000*float(other['paired_ci_high_eV']),
                zero_gap_meV_cell=1000*zero,physical_control_meV_cell=1000*physical,
                hetero_minus_zero_meV_cell=1000*(ordinary-zero),paired_ci_vs_zero_low_meV=1000*float(zero_check['paired_ci_low_eV']),
                paired_ci_vs_zero_high_meV=1000*float(zero_check['paired_ci_high_eV']),
                twin_gap_predictions=len(twins['hetero']),twin_max_predicted_gap_delta_eV=twin_delta,
                qualifies=cofirst,paper=contract['publication'],data=contract['source'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=ROOT/'results'/'arrangement_gap_screen_20261111')
    args = parser.parse_args()
    rows = [collect(name) for name in SOURCES]
    args.results.mkdir(parents=True,exist_ok=True)
    with (args.results/'benchmark_summary.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    lines=['# 同成分缺陷排布的 DFT 能量差', '',
           f"本轮完成 {len(rows)} 组公开来源、{sum(r['converged_fits'] for r in rows)} 次随机初始化训练，"
           f"涉及 {sum(r['arrangements'] for r in rows)} 个独立构型和 {sum(r['pairs'] for r in rows)} 对留出构型。"
           "表中误差为 meV/超胞，每对比较同一测试折、同一实际成分的两个结构，由同一份权重预测。"
           " [逐数据集汇总表](benchmark_summary.csv)", '',
           '| 成分 | 构型 | 留出对 | 普通 hetero | 名次 | 第一名 | 第一名 MAE | 零能差 MAE | 物理对照 MAE |',
           '|---|---:|---:|---:|---:|---|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['compositions']} | {r['arrangements']} | {r['pairs']} | {r['hetero_meV_cell']:.2f} | "
                     f"{r['hetero_rank']} | {r['first_modes']} | {r['first_meV_cell']:.2f} | "
                     f"{r['zero_gap_meV_cell']:.2f} | {r['physical_control_meV_cell']:.2f} |")
    lines+=['','每组包含 HERA 登记的全部十种非超图 ALIGNN。普通 hetero 与 hetero_fixed_pool 在默认 '
            'defect_mean 读出下具有相同的前向计算，两者各自独立训练。每折只用训练结构确定能量参考。'
            '验证集构型对能差 MAE 控制学习率和权重选择，连续 100 轮没有取得 0.5% 的相对改善时停止。'
            '筛选种子为 123。若普通 hetero 在该种子并列或单列第一，十种模型均以 456、789 再训练，'
            '最终比较各随机种子 MAE 的算术平均。', '',
            '寻找正例的判据还要求普通 hetero 的误差低于另外八个登记模式和零能差基线。'
            '本轮按实验结果寻找外部实例。各次独立初始化共用固定的结构划分。配对区间衡量留出构型的抽样波动。'
            '各来源的输入几何、缺陷定义和 DFT 目标不同，详见各自的对照报告。', '']
    qualified = [r for r in rows if r['qualifies']]
    if qualified:
        lines.extend(['达到上述数值判据的组为 '+'、'.join(r['compositions'] for r in qualified)+'。',''])
        for r in qualified:
            lines.extend([f"{r['compositions']} 中，普通 hetero 减去 {r['best_distinct_competitor']} 的 MAE 差为 "
                          f"{r['hetero_minus_best_distinct_meV_cell']:.2f} meV/超胞，配对 95% 区间为 "
                          f"[{r['paired_ci_low_meV']:.2f}, {r['paired_ci_high_meV']:.2f}]。"
                          f"相对零能差基线的差为 {r['hetero_minus_zero_meV_cell']:.2f}，区间为 "
                          f"[{r['paired_ci_vs_zero_low_meV']:.2f}, {r['paired_ci_vs_zero_high_meV']:.2f}]。"
                          f"逐种子名次为 {r['hetero_ranks_by_seed']}。",''])
    else:
        lines.extend(['本轮各组均未达到上述正例判据。',''])
    for r in rows:
        spec = SOURCES[r['dataset']]
        relative = Path(os.path.relpath(spec.results/'results.md',args.results)).as_posix()
        lines.append(f"- {r['title']}。{r['converged_fits']} 次训练，实际 {r['min_epochs']}–{r['max_epochs']} 轮。"
                     f" [十种模型及逐对预测]({relative}) · [论文]({r['paper']}) · [数据]({r['data']})")
    (args.results/'results.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(f'{len(rows)} completed sources, {sum(r["converged_fits"] for r in rows)} converged fits')


if __name__=='__main__':
    main()
