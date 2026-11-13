"""Verify held-out memberships and compare all ten models for MgAl2O4 arrangements."""
import argparse
from collections import Counter
import json
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..data.spinel_antisites import DOI, RESULTS, SOURCE, STRATA
from .alignn_convergence import convergence_audit, plot_learning_curves
from .run_spinel_antisites import DEFAULT_RUN, MODES, ORDER_THRESHOLD_EV, equal_count_pairs


def bootstrap_weights(pairs, replicates=10000):
    # A configuration contributes to several gaps. Draw whole structures inside
    # their test fold and inversion stratum, with identical draws for every model.
    rng = np.random.default_rng(20261110)
    weights = np.zeros((replicates, len(pairs)), dtype=np.int16)
    for _, part in pairs.groupby(['fold', 'k_each'], sort=True):
        keys = sorted(set(part.id_a) | set(part.id_b))
        counts = rng.multinomial(len(keys), np.full(len(keys), 1 / len(keys)), size=replicates)
        index = {key: i for i, key in enumerate(keys)}
        a, b = part.id_a.map(index), part.id_b.map(index)
        weights[:, part.index] = counts[:, a] * counts[:, b]
    weights = weights.astype(float)
    return weights / weights.sum(axis=1, keepdims=True)


def shell_baseline(records, folds):
    """Supplementary physical control using three first-shell counts on the ideal parent."""
    names = ['antisite_contacts_' + key for key in ('tet_tet', 'tet_oct', 'oct_oct')]
    data = records.set_index('source_id')
    preds, coefficients = [], []
    for fold, ids in enumerate(folds):
        train = data.loc[ids['train']]
        means = train.groupby('k_each')[names + ['energy_eV_cell']].mean()
        centered = train[names].to_numpy() - means.loc[train.k_each, names].to_numpy()
        y = train.energy_eV_cell.to_numpy() - means.loc[train.k_each, 'energy_eV_cell'].to_numpy()
        beta = np.linalg.lstsq(centered, y, rcond=None)[0]
        coefficients.append(dict(fold=fold, **dict(zip(names, beta, strict=True))))
        for sid in ids['test']:
            row = data.loc[sid]
            k = int(row.k_each)
            x = row[names].to_numpy(dtype=float) - means.loc[k, names].to_numpy()
            preds.append(dict(fold=fold, k_each=k, source_id=sid, dft_eV_cell=row.energy_eV_cell,
                              predicted_eV_cell=float(means.loc[k, 'energy_eV_cell'] + x @ beta)))
    return pd.DataFrame(equal_count_pairs(preds)), pd.DataFrame(coefficients)


def verified_fits(run, seed, order_threshold=ORDER_THRESHOLD_EV):
    contract = json.loads((run / 'data_contract.json').read_text(encoding='utf8'))
    splits = contract['partitions']
    if set(contract['modes']) != set(MODES) or contract['order_threshold_eV_cell'] != order_threshold:
        raise ValueError('Protocol does not match this analysis')
    structs, pairs, fits = [], [], []
    for mode in MODES:
        for fold, ids in enumerate(splits):
            folder = run / f'fold{fold}' / f'seed{seed}' / mode
            report = json.loads((folder / 'metrics.json').read_text(encoding='utf8'))
            stop_record = convergence_audit(folder, seed)
            if not stop_record['stopped_on_validation_plateau']:
                raise ValueError(f'Training has not reached the validation stopping criterion: {folder}')
            report.update(stop_record)
            max_epochs = json.loads((folder / 'config.json').read_text(encoding='utf8'))['config']['max_epochs']
            s = pd.read_csv(folder / 'test_structures.csv')
            p = pd.read_csv(folder / 'test_pairs.csv')
            if (report['updated_trainable_tensors'] < 1 or report['best_epoch'] < 1
                    or not 0 < report['epochs_completed'] <= max_epochs or report['mode'] != mode
                    or report['fold'] != fold or report['seed'] != seed
                    or s.source_id.tolist() != ids['test'] or not (s.fold == fold).all()):
                raise ValueError(f'Incomplete audited fit {mode}, fold {fold}')
            s = s.set_index('source_id', drop=False)
            derived_truth = s.loc[p.id_b, 'dft_eV_cell'].to_numpy() - s.loc[p.id_a, 'dft_eV_cell'].to_numpy()
            derived_pred = s.loc[p.id_b, 'predicted_eV_cell'].to_numpy() - s.loc[p.id_a, 'predicted_eV_cell'].to_numpy()
            if (not np.allclose(derived_truth, p.dft_gap_eV_cell, atol=1e-7, rtol=0)
                    or not np.allclose(derived_pred, p.predicted_gap_eV_cell, atol=1e-7, rtol=0)
                    or not np.allclose(np.abs(derived_pred-derived_truth), p.abs_gap_error_eV_cell, atol=1e-7, rtol=0)
                    or not (p.fold == fold).all() or not (p.id_a < p.id_b).all()
                    or not np.array_equal(s.loc[p.id_a, 'k_each'].to_numpy(), p.k_each)
                    or not np.array_equal(s.loc[p.id_b, 'k_each'].to_numpy(), p.k_each)):
                raise ValueError(f'Pair identity or energy mismatch in {mode}, fold {fold}')
            structs.append(s.assign(mode=mode).reset_index(drop=True))
            pairs.append(p.assign(mode=mode))
            fits.append(report)
    return pd.concat(structs, ignore_index=True), pd.concat(pairs, ignore_index=True), pd.DataFrame(fits)


def compare(structures, pairs, structures_expected=51, pairs_by_k=None, order_threshold=ORDER_THRESHOLD_EV):
    if pairs_by_k is None:
        pairs_by_k = {2: 26, 4: 45, 6: 51}
    keys = ['fold', 'k_each', 'id_a', 'id_b']
    ref = pairs[pairs['mode'] == 'hetero'].sort_values(keys).reset_index(drop=True)
    if ref.duplicated(keys).any() or Counter(ref.k_each) != pairs_by_k:
        raise ValueError('Incorrect or missing within-fold, equal-inversion comparisons')
    good = ref.dft_gap_eV_cell.abs() > order_threshold
    weights = bootstrap_weights(ref)
    het = ref.abs_gap_error_eV_cell.to_numpy()
    rows = []
    for mode in (*MODES, 'inversion_only'):
        if mode == 'inversion_only':
            p = ref.assign(predicted_gap_eV_cell=0., abs_gap_error_eV_cell=ref.dft_gap_eV_cell.abs())
            s = structures[structures['mode'] == 'hetero'].copy()
            s['predicted_eV_cell'] = s.train_mean_eV_cell
        else:
            p = pairs[pairs['mode'] == mode].sort_values(keys).reset_index(drop=True)
            s = structures[structures['mode'] == mode]
        if (not p[keys].equals(ref[keys]) or len(s) != structures_expected or s.source_id.nunique() != structures_expected
                or not np.allclose(p.dft_gap_eV_cell, ref.dft_gap_eV_cell, atol=1e-10)):
            raise ValueError(f'{mode} did not predict the same held-out cohort')
        err = p.abs_gap_error_eV_cell.to_numpy()
        lo, hi = np.quantile(weights @ (het - err), [0.025, 0.975])
        sign = p.loc[good, 'predicted_gap_eV_cell'] * ref.loc[good, 'dft_gap_eV_cell']
        rows.append(dict(mode=mode, gap_MAE_eV_cell=err.mean(),
                         abs_energy_MAE_eV_cell=(s.predicted_eV_cell-s.dft_eV_cell).abs().mean(),
                         informative_pairs=int(good.sum()), correctly_ordered=int((sign > 0).sum()),
                         abstained_ties=int((sign == 0).sum()), all_pairs=len(ref), structures=len(s),
                         hetero_minus_mode_eV_cell=(het-err).mean(),
                         paired_ci_low_eV_cell=lo, paired_ci_high_eV_cell=hi))
    return pd.DataFrame(rows).sort_values('gap_MAE_eV_cell').reset_index(drop=True)


def plot(path, records, comparison, shell_mae):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.3, 4.25), gridspec_kw={'width_ratios': [0.9, 1.5]})
    for k, group in records.groupby('k_each'):
        y = np.sort(group.delta_to_normal_eV_cell)
        jitter = np.random.default_rng(123+k).uniform(-0.021, 0.021, size=len(y))
        ax.scatter(k/8 + jitter, y, s=23, color='#176c6a' if k in STRATA else '#73767c',
                   alpha=0.8, edgecolors='white', linewidths=0.35)
    ax.set(xticks=[0, .25, .5, .75, 1], xlabel='Cation inversion fraction', ylabel='DFT energy above normal (eV/cell)')
    ax.set_title('MgAl₂O₄ arrangements', loc='left', fontsize=12)
    bars = comparison[['mode','gap_MAE_eV_cell']].copy()
    bars.loc[len(bars)] = ['first-shell linear', shell_mae]
    bars.sort_values('gap_MAE_eV_cell', ascending=False, inplace=True)
    colors = ['#176c6a' if m == 'hetero' else '#c1c5cb' if m in ('inversion_only','first-shell linear')
              else '#6681a1' for m in bars['mode']]
    bx.barh(bars['mode'], bars.gap_MAE_eV_cell, color=colors, height=0.69)
    bx.set(xlabel='Held-out energy-gap MAE (eV/cell)', xlim=(0, bars.gap_MAE_eV_cell.max()*1.16))
    bx.set_title('Equal-count antisite pairs', loc='left', fontsize=12)
    for i, v in enumerate(bars.gap_MAE_eV_cell):
        bx.text(v + bars.gap_MAE_eV_cell.max()*0.015, i, f'{v:.3f}', va='center', fontsize=8)
    for a in (ax, bx):
        a.spines[['top', 'right']].set_visible(False)
        a.tick_params(labelsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--results', type=Path, default=RESULTS)
    parser.add_argument('--archive', type=Path, help='Prepared MgAl2O4_antisite_DFT.zip with records.csv')
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()
    out = args.results
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.archive or out / 'MgAl2O4_antisite_DFT.zip') as z, z.open('records.csv') as f:
        raw = pd.read_csv(f)
    unique = raw[raw.is_representative]
    structures, pairs, fits = verified_fits(args.run_dir, args.seed)
    dft = unique.set_index('source_id').energy_eV_cell.loc[structures.source_id].to_numpy()
    if not np.allclose(dft, structures.dft_eV_cell, atol=1e-7, rtol=0):
        raise ValueError('Reported DFT energies differ from the source archive')
    comparison = compare(structures, pairs)
    informative = pairs.dft_gap_eV_cell.abs() > ORDER_THRESHOLD_EV
    by_inversion = (pairs.assign(informative=informative,
                                correct=informative & (pairs.dft_gap_eV_cell * pairs.predicted_gap_eV_cell > 0))
                    .groupby(['k_each', 'mode'], as_index=False)
                    .agg(gap_MAE_eV_cell=('abs_gap_error_eV_cell', 'mean'), all_pairs=('id_a', 'size'),
                         informative_pairs=('informative', 'sum'), correctly_ordered=('correct', 'sum'))
                    .sort_values(['k_each', 'gap_MAE_eV_cell']))
    folds = json.loads((args.run_dir / 'data_contract.json').read_text(encoding='utf8'))['partitions']
    shell_pairs, shell_coefficients = shell_baseline(unique, folds)
    if len(shell_pairs) != 122:
        raise ValueError('Physical control must cover all matched comparisons')
    for frame, name in ((comparison, 'model_comparison'), (structures, 'oof_structures'),
                        (pairs, 'oof_energy_gaps'), (fits, 'training_audits'), (by_inversion, 'by_inversion'),
                        (shell_pairs, 'first_shell_gaps'), (shell_coefficients, 'first_shell_coefficients')):
        frame.to_csv(out / (name + '.csv'), index=False)
    fits[['mode','fold','best_epoch','epochs_completed','epoch_budget','validation_patience',
          'min_relative_improvement_pct','stopped_on_validation_plateau','trailing_epochs_from_rounded_log',
          'train_MAE_at_best','val_MAE_at_best','final_train_MAE','final_val_MAE']].to_csv(out/'convergence.csv',index=False)
    panels=[(f'hetero, fold {r.fold+1}', args.run_dir/f'fold{r.fold}'/f'seed{args.seed}'/'hetero',
             {'best_epoch':r.best_epoch}) for r in fits[fits['mode']=='hetero'].itertuples()]
    plot_learning_curves(panels,args.seed,out/'hetero_learning_curves.png')
    plot(out / 'spinel_DFT_and_model_errors.png', unique, comparison, shell_pairs.abs_gap_error_eV_cell.mean())
    lines = ['# MgAl₂O₄ 阳离子反位排布与 DFT 能量', '',
             f'Fang、Ohodnicki 与 Wang 公开了 Mg₈Al₁₆O₃₂ 超胞中 Mg、Al 交换晶格位置后的弛豫结构和 DFT 总能。[论文]({DOI}) · [作者数据]({SOURCE})', '',
             '原始 JSON 共 62 条。按正常尖晶石母结构的 192 个对称操作归并后，有 53 个不等价排布。两个端点各只有一个排布，另外 51 个用于训练和留出评估。', '',
             '| 每类反位原子数 k | 反位度 | 独立排布数 | DFT 总能范围（eV/56 原子超胞） | 能量跨度（eV/超胞） |',
             '|---:|---:|---:|---:|---:|']
    for k, g in unique.groupby('k_each'):
        e = g.energy_eV_cell
        lines.append(f'| {k} | {k/8:g} | {len(g)} | {e.min():.5f} 至 {e.max():.5f} | {e.max()-e.min():.5f} |')
    lines.extend(['', 'k 是占据 Mg 四面体位的 Al 数，也等于占据 Al 八面体位的 Mg 数，反位原子共 2k 个。'
                  '例如 k=4 时，源数据中的 CONTCAR-8 和 CONTCAR-13 分别为 −396.30981 和 −395.60620 eV/超胞。'
                  '前者只有 1 对最近邻八面体 Mg 反位，后者有 4 对（按正常母晶格约 2.889 Å 的第一壳层计数）。'
                  '在化学式和反位浓度相同的条件下，DFT 仍分出 0.70361 eV 的能量差。', '',
                  '## 留出比较', '',
                  f'三个固定折的训练/验证/测试数分别为 {"、".join("/".join(str(len(f[s])) for s in ("train","val","test")) for f in folds)}。'
                  f'各模型从随机初始化训练，权重种子 {args.seed}，上限 500 轮，早停耐心 50 轮，按验证集最低 MAE 选权重。'
                  '每折仅用训练集估计各 k 的平均能量，所有模型拟合该均值之上的能量偏差，再加回均值得到总能预测。'
                  '主指标采用同折、同 k 的 122 对构型，统计预测能量差与 DFT 能量差的平均绝对误差。'
                  '每个对称不等价排布只出现在一个测试折。排序仅统计 DFT 能隙大于 0.010 eV 的对。', '',
                  '| 模型 | 能量差 MAE（eV/超胞） | 总能 MAE（eV/超胞） | 排序正确/可分对 |',
                  '|---|---:|---:|---:|'])
    for r in comparison.itertuples(index=False):
        order = f'{r.correctly_ordered}/{r.informative_pairs}' if r.mode != 'inversion_only' else '全部预测为平局'
        lines.append(f'| {r.mode} | {r.gap_MAE_eV_cell:.5f} | {r.abs_energy_MAE_eV_cell:.5f} | {order} |')
    lines.extend(['', '不同反位度的能量差 MAE（eV/超胞）如下。by_inversion.csv 包含全部 10 种网络。', '',
                  '| 反位度 | 同折构型对 | hetero | full | full_x |', '|---:|---:|---:|---:|---:|'])
    for k, group in by_inversion.groupby('k_each'):
        scores = group.set_index('mode').gap_MAE_eV_cell
        lines.append(f'| {k/8:g} | {int(group.all_pairs.iloc[0])} | {scores["hetero"]:.5f} '
                     f'| {scores["full"]:.5f} | {scores["full_x"]:.5f} |')
    shell_good = shell_pairs[shell_pairs.dft_gap_eV_cell.abs() > ORDER_THRESHOLD_EV]
    shell_correct = int((shell_good.dft_gap_eV_cell * shell_good.predicted_gap_eV_cell > 0).sum())
    lines.extend(['', '作为物理对照，另用正常母晶格第一壳层的三种反位接触数（四面体–四面体、四面体–八面体、八面体–八面体）'
                  '进行线性拟合。各折只用训练结构求反位度均值和三个回归系数。'
                  f'同一批留出对的能量差 MAE 为 {shell_pairs.abs_gap_error_eV_cell.mean():.5f} eV/超胞，'
                  f'排序正确 {shell_correct}/{len(shell_good)} 对。其逐对结果见 first_shell_gaps.csv。', '',
                  'model_comparison.csv 同时给出 hetero 与每个模型的配对误差差值和 95% 区间。'
                  '这些区间用同折、同 k 的留出结构整组重采样 10,000 次计算，条件为这次划分和已训练的模型。'
                  '不同构型对可共享结构，122 对的独立结构数为 51。单个初始化种子无法衡量跨种子方差。'
                  '所有模型输入都包含作者给出的弛豫坐标，本实验衡量该几何条件下的能量预测。', '',
                  f'10 种模型各做 3 折独立训练，共 {int(fits.epochs_completed.sum())} 个 epoch。'
                  f'单次实际训练 {int(fits.epochs_completed.min())}–{int(fits.epochs_completed.max())} 个 epoch。'
                  '各次 checkpoint 的早停标记和验证日志已核对，逐次结果见 [convergence.csv](convergence.csv)，'
                  '普通 hetero 的三折过程见[学习曲线](hetero_learning_curves.png)。', '',
                  '数据包中的 records.csv 保留每条源文件，structures.extxyz 含 53 个独立排布，folds.json 记录三折 ID。'
                  'oof_structures.csv 和 oof_energy_gaps.csv 可分别复算结构总能和配对指标。', '',
                  '## 复现', '', '在包含 HERA 目录的父目录运行以下命令。两条命令使用相同的新日志目录。', '',
                  '```text', 'python -m HERA.scripts.run_spinel_antisites --archive HERA/results/spinel_antisites_20261110/MgAl2O4_antisite_DFT.zip --run-dir HERA/logs/spinel_reproduction',
                  'python -m HERA.scripts.analyze_spinel_antisites --run-dir HERA/logs/spinel_reproduction --archive HERA/results/spinel_antisites_20261110/MgAl2O4_antisite_DFT.zip --results HERA/results/spinel_reproduction',
                  '```', ''])
    (out / 'results.md').write_text('\n'.join(lines), encoding='utf8')
    print(comparison[['mode', 'gap_MAE_eV_cell', 'correctly_ordered']].to_string(index=False))


if __name__ == '__main__':
    main()
