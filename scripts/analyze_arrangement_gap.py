"""Audit converged ALIGNN fits and rank predictions of held-out DFT energy gaps."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .alignn_convergence import convergence_audit
from ..arrangement_sources import SOURCES
from .run_arrangement_gap import DISCOVERY_SEED, MODES, ORDER_THRESHOLD_EV, REPLICATION_SEEDS, dataset_module, pair_rows


KEYS = ['fold', 'group', 'id_a', 'id_b']


def verify(run, spec, archive, seeds):
    contract = json.loads((run / 'data_contract.json').read_text(encoding='utf8'))
    records, _, _, _ = spec.prepare(archive)
    lookup = {r['source_id']: r for r in records}
    folds = spec.partitions(records)
    if (contract['partitions'] != folds or contract['model_names'] != list(MODES)
            or contract['source_sha256'] != spec.DIGEST or contract['schema'] != spec.SCHEMA):
        raise ValueError('Run contract differs from the source data or model list')
    if Counter(s for f in folds for s in f['test']) != Counter(lookup.keys()):
        raise ValueError('Test folds must partition the independent source configurations')
    structures, comparisons, fits = [], [], []
    for seed in seeds:
        for mode in MODES:
            for fold, ids in enumerate(folds):
                base = run / f'fold{fold}' / f'seed{seed}' / mode
                report = json.loads((base / 'metrics.json').read_text(encoding='utf8'))
                selected = base / report['selected_run']
                ck = convergence_audit(selected, seed)
                audit = json.loads((selected / 'training_audit.json').read_text(encoding='utf8'))
                if (any(report[k] != v for k, v in (ck | audit).items()) or not ck['stopped_on_validation_plateau']
                        or report['updated_trainable_tensors'] < 1 or (report['mode'], report['fold'], report['seed'])
                        != (mode, fold, seed) or ck['validation_patience'] != contract['convergence']['patience']
                        or ck['min_relative_improvement_pct'] != contract['convergence']['minimum_relative_improvement_percent']):
                    raise ValueError(f'Invalid convergence or weight audit in {base}')
                s = pd.read_csv(base / 'test_structures.csv')
                if s.source_id.tolist() != ids['test'] or not (s.fold == fold).all() or len(s) != report['n_test']:
                    raise ValueError(f'Unexpected held-out sample in {base}')
                source = [lookup[key] for key in s.source_id]
                means = {g: np.mean([lookup[i]['energy_eV_cell'] for i in ids['train'] if lookup[i]['group'] == g])
                         for g in s.group.unique()}
                if (s.group.tolist() != [r['group'] for r in source] or not np.isfinite(s.predicted_eV_cell).all()
                        or not np.allclose(s.dft_eV_cell, [r['energy_eV_cell'] for r in source], atol=1e-8, rtol=0)
                        or not np.allclose(s.train_mean_eV_cell, s.group.map(means), atol=1e-8, rtol=0)):
                    raise ValueError(f'Invalid source labels, energy references or predictions in {base}')
                pairs = pd.DataFrame(pair_rows(s.to_dict('records')))
                saved = pd.read_csv(base / 'test_pairs.csv')
                physical = ['dft_gap_eV', 'predicted_gap_eV', 'gap_error_eV']
                if (not pairs[KEYS].equals(saved[KEYS]) or pairs.duplicated(KEYS).any()
                        or len(pairs) != report['n_pairs'] or not np.allclose(pairs[physical], saved[physical],
                        atol=1e-8, rtol=0) or abs(pairs.gap_error_eV.mean() - report['test_gap_MAE_eV']) > 1e-8):
                    raise ValueError(f'Energy differences are inconsistent in {base}')
                structures.append(s.assign(seed=seed, mode=mode))
                comparisons.append(pairs.assign(seed=seed, mode=mode))
                fits.append(report)
    return pd.concat(structures, ignore_index=True), pd.concat(comparisons, ignore_index=True), pd.DataFrame(fits), contract


def bootstrap_contrasts(pairs, differences, replicas=10000, families=None):
    """Resample independent structures or entire parent families jointly for all modes."""
    rng = np.random.default_rng(20261110)
    groups = []
    strata = ['fold','group'] if families is None else ['fold']
    for _, group in pairs.groupby(strata, sort=True):
        left = group.id_a if families is None else group.id_a.map(families)
        right = group.id_b if families is None else group.id_b.map(families)
        if left.isna().any() or right.isna().any():
            raise ValueError('Unknown family in pair bootstrap')
        ids = sorted(set(left) | set(right))
        index = {sid: i for i, sid in enumerate(ids)}
        if families is None and len(group) != len(ids) * (len(ids) - 1) // 2:
            raise ValueError('Incomplete set of distinct within-fold arrangement pairs')
        groups.append((group.index.to_numpy(),left.map(index).to_numpy(),right.map(index).to_numpy(),len(ids)))
    draws = []
    for start in range(0, replicas, 200):
        size = min(200, replicas - start)
        weights = np.zeros((size, len(pairs)))
        for cols, left, right, n in groups:
            counts = rng.multinomial(n, np.full(n, 1/n), size=size)
            # A pair within a parent travels once with each copy of that parent.
            weights[:, cols] = np.where(left==right,counts[:,left],counts[:,left]*counts[:,right])
        totals = weights.sum(axis=1)
        draws.append((weights[totals > 0] @ differences) / totals[totals > 0, None])
    return np.quantile(np.concatenate(draws), [.025, .975], axis=0)


def rank(frame, tolerance):
    scores = frame.gap_MAE_eV_cell.to_numpy()
    return frame.assign(rank=[1 + int((scores < s-tolerance).sum()) for s in scores]).sort_values(
        ['rank', 'gap_MAE_eV_cell'], kind='stable').reset_index(drop=True)


def baseline_comparison(pairs,control,seeds,families=None):
    """Compare ordinary hetero with fixed baselines using the same resampled test structures."""
    reference = control.sort_values(KEYS).reset_index(drop=True)
    errors = []
    for seed in seeds:
        pred = pairs[(pairs['mode']=='hetero')&(pairs.seed==seed)].sort_values(KEYS).reset_index(drop=True)
        if (not pred[KEYS].equals(reference[KEYS]) or
                not np.allclose(pred.dft_gap_eV,reference.dft_gap_eV,atol=1e-8,rtol=0)):
            raise ValueError('Hetero and baseline energy differences use different test pairs')
        errors.append(pred.gap_error_eV.to_numpy())
    ordinary = np.mean(errors,axis=0)
    baselines = np.column_stack([reference.dft_gap_eV.abs(),reference.gap_error_eV])
    differences = ordinary[:,None]-baselines
    low, high = bootstrap_contrasts(reference,differences,families=families)
    return pd.DataFrame(dict(baseline=['zero_gap','physical_linear'],baseline_MAE_eV_cell=baselines.mean(axis=0),
                             hetero_MAE_eV_cell=ordinary.mean(),hetero_minus_baseline_eV_cell=differences.mean(axis=0),
                             paired_ci_low_eV=low,paired_ci_high_eV=high,n_unique_pairs=len(reference),n_seeds=len(seeds)))


def physical_control(dataset, spec, archive):
    """Fit a training-only physical baseline on the same inputs and held-out pairs."""
    records, geometries, _, _ = spec.prepare(archive)
    if dataset == 'ysz':
        from pymatgen.analysis.ewald import EwaldSummation
        from pymatgen.core import Structure
        features = []
        for row in records:
            a = geometries[row['source_id']]
            st = Structure(a.cell.array, a.numbers, a.get_scaled_positions())
            st.add_oxidation_state_by_element({'Zr':4, 'Y':3, 'O':-2})
            if abs(st.charge) > 1e-8:
                raise ValueError('The YSZ formal-charge supercell must be neutral')
            features.append([EwaldSummation(st).total_energy])
        label = '理想结构的形式电荷 Ewald 能量，训练折拟合斜率和截距'
    elif dataset == 'hzo':
        features = [[r['first_shell_cations'], r['first_shell_Zr']] for r in records]
        label = '空位近邻阳离子数及 Zr 数，训练折拟合线性回归'
    elif dataset == 'pyrochlore':
        features = [[r['first_shell_Sn_pairs'], r['second_shell_Sn_pairs']] for r in records]
        label = '第一和第二近邻 Sn–Sn 对数，训练折拟合线性回归'
    elif dataset == 'nasicon':
        features = [[r['occupied_Na1'],r['first_shell_vac_pairs']] for r in records]
        label = 'Na1 占位数及最近邻空位对数，训练折拟合线性回归'
    elif dataset == 'nb_split':
        features = [[r['vacancy_distance_A'],r['mean_interstitial_vacancy_distance_A']] for r in records]
        label = '空位间距及间隙 Nb 到两个空位的平均距离，训练折拟合线性回归'
    elif dataset == 'nb_h8':
        features = [[r['closest_HH_A'],r['relaxed_volume_A3']] for r in records]
        label = '最近 H–H 间距及弛豫超胞体积，训练折拟合线性回归'
    elif dataset == 'feni':
        features = [[r['vacancy_first_shell_Fe'],r['Fe_Ni_nearest_pairs']] for r in records]
        label = '空位最近邻 Fe 数及全胞 Fe–Ni 最近邻对数，在训练折内按实际成分分别拟合线性回归'
    elif dataset == 'ceria4':
        features = [[r['top_layer_vacancies'],r['second_layer_vacancies'],r['closest_O_vacancies_A']] for r in records]
        label = '最上两层各自的氧空位数及最近空位间距，训练折拟合线性回归'
    else:
        raise ValueError(dataset)
    features = np.asarray(features, dtype=float)
    targets = np.asarray([r['energy_eV_cell'] for r in records])
    index = {r['source_id']:i for i,r in enumerate(records)}
    pairs = []
    for fold, ids in enumerate(spec.partitions(records)):
        test = [index[s] for s in ids['test']]
        predicted = np.full(len(test),np.nan)
        for group in sorted({records[i]['group'] for i in test}):
            train = [index[s] for s in ids['train'] if records[index[s]]['group']==group]
            keep = np.array([records[i]['group']==group for i in test])
            mean_x, mean_y = features[train].mean(axis=0), targets[train].mean()
            slopes = np.linalg.lstsq(features[train] - mean_x, targets[train] - mean_y, rcond=None)[0]
            predicted[keep] = mean_y + (features[np.asarray(test)[keep]] - mean_x) @ slopes
        if not np.isfinite(predicted).all():
            raise ValueError('Missing composition-specific physical baseline')
        items = [dict(fold=fold, group=records[i]['group'], source_id=records[i]['source_id'],
                      dft_eV_cell=targets[i], predicted_eV_cell=e) for i, e in zip(test,predicted,strict=True)]
        pairs.extend(pair_rows(items))
    control = pd.DataFrame(pairs)
    if dataset == 'ceria4':
        layers = {}
        for record in records:
            top = record['top_layer_vacancies']
            second = record['second_layer_vacancies']
            layers[record['source_id']] = f'{top},{second},{4-top-second}'
        control['layer_counts_a'] = control.id_a.map(layers)
        control['layer_counts_b'] = control.id_b.map(layers)
    if dataset == 'nb_h8':
        energies_e0 = {r['source_id']: r['electronic_E0_eV'] for r in records}
        control['dft_gap_E0_eV'] = control.id_b.map(energies_e0) - control.id_a.map(energies_e0)
    return label, control


def summarize(structures, pairs, seeds, tolerance, families=None):
    ref = pairs[(pairs['mode'] == 'hetero') & (pairs.seed == seeds[0])].sort_values(KEYS).reset_index(drop=True)
    errors, order, by_seed = [], [], []
    good = ref.dft_gap_eV.abs() > ORDER_THRESHOLD_EV
    for seed in seeds:
        rows = []
        for mode in MODES:
            sample = pairs[(pairs['mode'] == mode) & (pairs.seed == seed)].sort_values(KEYS).reset_index(drop=True)
            if (not sample[KEYS].equals(ref[KEYS]) or not np.allclose(sample.dft_gap_eV, ref.dft_gap_eV, atol=1e-8, rtol=0)):
                raise ValueError(f'{mode} has a different evaluation cohort')
            errors.append(sample.gap_error_eV.to_numpy())
            correct = (sample.loc[good, 'predicted_gap_eV'] * ref.loc[good, 'dft_gap_eV'] > 0).mean()
            order.append(correct)
            rows.append(dict(seed=seed, mode=mode, gap_MAE_eV_cell=sample.gap_error_eV.mean(),
                             ordering_accuracy_over_10meV=correct))
        by_seed.append(rank(pd.DataFrame(rows), tolerance))
    matrix = np.asarray(errors).reshape(len(seeds), len(MODES), len(ref)).mean(axis=0).T
    diff = matrix[:, [MODES.index('hetero')]] - matrix
    lower, upper = bootstrap_contrasts(ref, diff, families=families)
    results = rank(pd.DataFrame(dict(mode=MODES, gap_MAE_eV_cell=matrix.mean(axis=0),
        hetero_minus_mode_eV_cell=diff.mean(axis=0), paired_ci_low_eV=lower, paired_ci_high_eV=upper,
        ordering_accuracy_over_10meV=np.asarray(order).reshape(len(seeds),len(MODES)).mean(axis=0),
        n_structures=structures.source_id.nunique(), n_unique_pairs=len(ref),
        order_eligible_unique_pairs=int(good.sum()), n_seeds=len(seeds))), tolerance)
    return results, pd.concat(by_seed, ignore_index=True), float(ref.dft_gap_eV.abs().mean())


def plot_results(path, dataset, results, pairs, zero):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.2, 4.2), layout='constrained')
    names = results['mode'].tolist()
    ax.barh(range(len(names)), results.gap_MAE_eV_cell * 1000,
            color=['#176c60' if x == 'hetero' else '#9ca4ac' if x == 'hetero_fixed_pool' else '#6683a0'
                   for x in names])
    ax.set_yticks(range(len(names)), names)
    ax.invert_yaxis()
    ax.axvline(1000*zero, lw=1, color='#954d22', ls='--', label='Zero-gap baseline')
    ax.set_xlabel('Held-out energy-gap MAE (meV/cell)')
    ax.legend(loc='lower left', bbox_to_anchor=(0,1.005), frameon=False, fontsize=8)
    example = pairs[(pairs['mode'] == 'hetero') & (pairs.seed == DISCOVERY_SEED)]
    for fold, part in example.groupby('fold', sort=True):
        bx.scatter(1000*part.dft_gap_eV, 1000*part.predicted_gap_eV, s=10, alpha=0.55,
                   color=['#225c70','#a45527','#698235'][fold], label=f'Fold {fold+1}', linewidth=0)
    lo = min(example.dft_gap_eV.min(), example.predicted_gap_eV.min()) * 1000
    hi = max(example.dft_gap_eV.max(), example.predicted_gap_eV.max()) * 1000
    bx.plot([lo,hi], [lo,hi], ls='--', lw=0.9, color='#383c40')
    bx.set(xlabel='DFT energy gap (meV/cell)', ylabel='Predicted energy gap (meV/cell)',
           aspect='equal', title=f'hetero, seed {DISCOVERY_SEED}')
    bx.legend(frameon=False, fontsize=8)
    for axis in (ax, bx):
        axis.spines[['top','right']].set_visible(False)
    formulas = {'ysz':'Zr$_{30}$Y$_2$O$_{63}$','hzo':'Hf$_{16}$Zr$_{16}$O$_{63}$',
                'pyrochlore':'La$_{16}$Zr$_8$Sn$_8$O$_{56}$','nasicon':'Na$_8$V$_8$P$_{12}$O$_{48}$',
                'nb_split':'Ti$_{24}$Nb$_{71}$O$_{144}$, charge −3','nb_h8':'Nb$_{16}$H$_8$',
                'feni':'Fe$_{15}$Ni$_{16}$ and Fe$_{16}$Ni$_{15}$','ceria4':'Ce$_{64}$O$_{124}$, (111)'}
    fig.suptitle(formulas[dataset], fontsize=13)
    fig.savefig(path, dpi=190)
    plt.close(fig)


def jump_comparison(out, records, structures, pairs, seeds, tolerance, families, inventory=None):
    source = pd.DataFrame(records).set_index('source_id')
    paths = pairs.id_a.map(source.path_id)
    matched = pairs[paths==pairs.id_b.map(source.path_id)].copy()
    matched['path_id'] = paths[matched.index]
    if (not (matched.groupby(['mode','seed']).size()==32).all()
            or matched.duplicated(['mode','seed','path_id']).any()):
        raise ValueError('Expected all 32 published vacancy hops once per model and seed')
    # Store the physical orientation PRO minus REA, even when hashed IDs order the pair differently.
    backwards = matched.id_a.map(source.endpoint)=='PRO'
    if not (matched.id_a.map(source.endpoint)!=matched.id_b.map(source.endpoint)).all():
        raise ValueError('Both endpoints have the same phase')
    for column in ['dft_gap_eV','predicted_gap_eV']:
        matched.loc[backwards,column] *= -1
    matched.loc[backwards,['id_a','id_b']] = matched.loc[backwards,['id_b','id_a']].to_numpy()
    summary, _, zero = summarize(structures,matched,seeds,tolerance,families)
    matched.rename(columns={'id_a':'reactant_id','id_b':'product_id'}).to_csv(out/'jump_predictions.csv',index=False)
    summary.to_csv(out/'jump_comparison.csv',index=False)
    by_group = (pairs.groupby(['group','mode','seed']).gap_error_eV.mean().groupby(['group','mode']).mean()
                .rename('gap_MAE_eV_cell').reset_index())
    by_group.to_csv(out/'by_composition.csv',index=False)
    h = summary[summary['mode']=='hetero'].iloc[0]
    text = (f"32 条作者指定跃迁的端点能量差（PRO − REA）另行统计，使用主指标选择的同一组权重。"
            f"普通 hetero 为 {h.gap_MAE_eV_cell:.6f} eV/超胞，第 {int(h['rank'])}，零能差为 {zero:.6f}。"
            " [十种模型](jump_comparison.csv) · [逐跃迁预测](jump_predictions.csv) · [按实际成分](by_composition.csv)")
    if inventory is not None:
        keep = pd.DataFrame(inventory['pairs'])
        subset = pairs.merge(keep,on=KEYS,validate='many_to_one')
        members = set(keep.id_a)|set(keep.id_b)
        scores, _, base = summarize(structures[structures.source_id.isin(members)],subset,seeds,tolerance,families)
        scores.to_csv(out/'same_inventory_comparison.csv',index=False)
        subset.to_csv(out/'same_inventory_predictions.csv',index=False)
        h = scores[scores['mode']=='hetero'].iloc[0]
        text += (f"\n\n进一步固定两种反位缺陷数目和空位所在亚晶格，得到 {len(keep)} 对、{len(members)} 个排布。"
                 f"普通 hetero 为 {h.gap_MAE_eV_cell:.6f} eV/超胞，第 {int(h['rank'])}，零能差为 {base:.6f}。"
                 " [同缺陷数排名](same_inventory_comparison.csv) · [逐对预测](same_inventory_predictions.csv)")
    return text


def compare_within_oxygen_layers(out, pairs, control, tolerance):
    """Describe the held-out pairs that also match the number of vacancies in each layer."""
    matched = control.loc[control.layer_counts_a == control.layer_counts_b].copy()
    members = set(matched.id_a) | set(matched.id_b)
    if matched.groupby('fold').size().to_dict() != {0: 2, 1: 2, 2: 2} or len(members) != 12:
        raise ValueError('Unexpected fixed-layer comparison in the published split')
    labels = matched[KEYS + ['layer_counts_a', 'predicted_gap_eV']].rename(
        columns={'layer_counts_a': 'vacancies_top_to_third', 'predicted_gap_eV': 'physical_control_gap_eV'})
    subset = pairs.merge(labels, on=KEYS, validate='many_to_one')
    if not (subset.groupby(['mode', 'seed']).size() == len(matched)).all():
        raise ValueError('Different fixed-layer cohorts across models or seeds')
    per_seed = subset.groupby(['mode', 'seed']).gap_error_eV.mean()
    scores = rank(per_seed.groupby('mode').mean().rename('gap_MAE_eV_cell').reset_index(), tolerance)
    scores.to_csv(out/'same_layer_counts_comparison.csv', index=False)
    subset.to_csv(out/'same_layer_counts_predictions.csv', index=False)
    ordinary = scores.loc[scores['mode'] == 'hetero'].iloc[0]
    full_mae = scores.loc[scores['mode'] == 'full', 'gap_MAE_eV_cell'].iloc[0]
    return (f"事后检查四个空位在三层中的个数也完全相同的排布。原划分中共有 {len(matched)} 对，"
            f"涉及 {len(members)} 个构型，每折两对。普通 hetero 的 MAE 为 {ordinary.gap_MAE_eV_cell:.6f} eV/超胞，"
            f"列第 {int(ordinary['rank'])}，full 为 {full_mae:.6f}，零能差为 {matched.dft_gap_eV.abs().mean():.6f}，"
            f"物理对照为 {matched.gap_error_eV.mean():.6f}。"
            "这一小组仅作描述，所有预测沿用主比较的权重。"
            " [十种模型](same_layer_counts_comparison.csv) · [六对构型及预测](same_layer_counts_predictions.csv)")


def hydride_energy_sensitivity(out, pairs, control, tolerance):
    """Score the same F-trained predictions against the archived extrapolated VASP energies."""
    sample = pairs.merge(control[KEYS + ['dft_gap_E0_eV']], on=KEYS, validate='many_to_one')
    if len(sample) != len(pairs) or not np.isfinite(sample.dft_gap_E0_eV).all():
        raise ValueError('Incomplete alternative VASP energy comparison')
    sample['error_e0'] = (sample.predicted_gap_eV - sample.dft_gap_E0_eV).abs()
    scores = sample.groupby(['seed', 'mode'], sort=False).agg(
        MAE_published_F_eV_cell=('gap_error_eV','mean'),
        MAE_against_E0_same_predictions_eV_cell=('error_e0','mean')).reset_index()
    means = scores.groupby('mode', sort=False)[scores.columns[2:]].mean().reset_index().assign(seed='mean')
    scores = pd.concat([scores,means],ignore_index=True)
    scores['zero_gap_F_eV_cell'] = control.dft_gap_eV.abs().mean()
    scores['zero_gap_E0_eV_cell'] = control.dft_gap_E0_eV.abs().mean()
    for field, ranking in [('MAE_published_F_eV_cell','rank_F'),
                           ('MAE_against_E0_same_predictions_eV_cell','rank_E0_same_predictions')]:
        scores[ranking] = scores.groupby('seed',sort=False)[field].transform(
            lambda x: [1 + int((x < value-tolerance).sum()) for value in x])
    scores.to_csv(out/'energy_definition_sensitivity.csv',index=False)
    h = scores[(scores.seed == 'mean') & (scores['mode'] == 'hetero')].iloc[0]
    return (f"采用作者 OSZICAR 的外推能量 E₀ 重新计算留出构型的参考能量差。沿用以 F 训练的全部预测，"
            f"普通 hetero 的 MAE 为 {1000*h.MAE_against_E0_same_predictions_eV_cell:.3f} meV/超胞，"
            f"第 {int(h.rank_E0_same_predictions)}，零能差为 {1000*h.zero_gap_E0_eV_cell:.3f} meV/超胞。"
            " [两种能量定义下的全部排名](energy_definition_sensitivity.csv)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=SOURCES, required=True)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--results', type=Path)
    parser.add_argument('--seeds', nargs='+', type=int, default=[DISCOVERY_SEED])
    args = parser.parse_args()
    if args.seeds not in ([DISCOVERY_SEED], [DISCOVERY_SEED, *REPLICATION_SEEDS]):
        parser.error('Use the prespecified screening seed or the complete three-seed confirmation set')
    spec = dataset_module(args.dataset)
    run = args.run_dir or spec.ROOT / 'logs' / spec.SCHEMA
    out = args.results or spec.RESULTS
    archive = args.archive or (spec.ARCHIVE if spec.ARCHIVE.exists() else out/spec.BUNDLE)
    s, p, fits, contract = verify(run, spec, archive, args.seeds)
    if fits.device.nunique() != 1:
        raise ValueError('Compute backends differ within the all-mode comparison')
    families, path_records, inventory = None, None, None
    if 'matched_path_diagnostic' in contract:
        path_records, atoms, _, _ = spec.prepare(archive)
        families = {r['source_id']:r['parent_id'] for r in path_records}
        inventory = spec.inventory_diagnostic(path_records,atoms,contract['partitions'])
        if json.loads((run/'inventory_diagnostic.json').read_text(encoding='utf8'))!=inventory:
            raise ValueError('Fixed-inventory pairs differ from the frozen source geometry selection')
    results, seed_table, zero = summarize(s, p, args.seeds, contract['tie_tolerance_eV'], families)
    control_label, control = physical_control(args.dataset, spec, archive)
    reference = p[(p['mode'] == 'hetero') & (p.seed == DISCOVERY_SEED)].reset_index(drop=True)
    if not control[KEYS].equals(reference[KEYS]) or not np.allclose(control.dft_gap_eV, reference.dft_gap_eV, atol=1e-8, rtol=0):
        raise ValueError('Physical baseline is evaluated on different pairs')
    baselines = baseline_comparison(p,control,args.seeds,families)
    out.mkdir(parents=True, exist_ok=True)
    for name, table in [('model_comparison',results), ('per_seed_comparison',seed_table),
                        ('heldout_structures',s), ('heldout_energy_gaps',p), ('convergence',fits),
                        ('physical_control_pairs',control),('baseline_comparison',baselines)]:
        table.to_csv(out / f'{name}.csv', index=False)
    h = results[results['mode'] == 'hetero'].iloc[0]
    a = results[results['mode'] == 'attention_was'].iloc[0]
    if args.seeds==[DISCOVERY_SEED] and int(h['rank'])==1:
        results.to_csv(out/f'discovery_seed{DISCOVERY_SEED}_comparison.csv',index=False)
    winners = results.loc[results['rank'] == 1, 'mode'].tolist()
    plot_results(out/'energy_gap_comparison.png', args.dataset, results, p, zero)
    log_link = Path(os.path.relpath(run, out)).as_posix()
    bundle = out/spec.BUNDLE if (out/spec.BUNDLE).exists() else spec.RESULTS/spec.BUNDLE
    bundle_link = Path(os.path.relpath(bundle, out)).as_posix()
    membership = '、'.join('/'.join(str(len(f[k])) for k in ['train','val','test']) for f in contract['partitions'])
    resampling = ('每次在原测试折内对完整母排布重采样，一个母体的所有跃迁和两个端点一起进入样本。'
                  if families is not None else '每次重采样在原测试折和成分组内抽取完整排布，相同排布涉及的能量差一起重采样。')
    lines = [f'# {spec.TITLE}', '', spec.INTERPRETATION, '',
             f"主指标为同成分、同一测试折内全部独立排布对的有符号能量差 MAE，单位为 eV/超胞。"
             f"共 {int(h.n_structures)} 个独立排布、{int(h.n_unique_pairs)} 个不同测试对。"
             f"种子为 {', '.join(map(str,args.seeds))}，多种子时先分别计算再取均值。", '',
             f"第一名为 {', '.join(winners)}。普通 hetero 第 {int(h['rank'])}，MAE 为 {h.gap_MAE_eV_cell:.6f}。"
             f"attention_was 为 {a.gap_MAE_eV_cell:.6f}。令所有能量差为零得到的 MAE 为 {zero:.6f}。"
             f"物理对照（{control_label}）为 {control.gap_error_eV.mean():.6f}。", '',
             '| 名次 | ALIGNN 模式 | 能量差 MAE (eV/超胞) |',
             '|---:|---|---:|']
    lines.extend(f"| {r.rank} | {r.mode} | {r.gap_MAE_eV_cell:.6f} |" for r in results.itertuples(index=False))
    if path_records is not None:
        lines += ['',jump_comparison(out,path_records,s,p,args.seeds,contract['tie_tolerance_eV'],families,inventory)]
    if args.dataset == 'ceria4':
        lines += ['', compare_within_oxygen_layers(out, p, control, contract['tie_tolerance_eV'])]
    if args.dataset == 'nb_h8':
        lines += ['', hydride_energy_sensitivity(out, p, control, contract['tie_tolerance_eV'])]
    marker_audit = run/'homogeneous_vacancy_marker_check.json'
    if args.dataset=='ceria4' and marker_audit.is_file():
        checks = json.loads(marker_audit.read_text(encoding='utf8'))
        if (set(checks)!={'attention','attention_was','definet','definet_was'} or
                any(q['node_and_vacancy_counts']!=[[192,4]] or q['distinct_input_tensor_hashes']!=34
                    or q['min_edges_per_vacancy']<1 for q in checks.values())):
            raise ValueError('Unexpected vacancy markers in the homogeneous input audit')
        lines += ['', '四种 attention 和 definet 模式的 34 张输入图都标出了四个氧空位，每个空位都有邻接边。'
                  '各模式均有 34 组不同的输入张量。'
                  f' [图输入核查]({log_link}/{marker_audit.name})']
    discovery_table = out/f'discovery_seed{DISCOVERY_SEED}_comparison.csv'
    if discovery_table.is_file():
        lines += ['',f'[触发重复训练的种子 {DISCOVERY_SEED} 排名与配对区间]({discovery_table.name})']
    lines += ['', f"{len(fits)} 次训练均从随机权重开始，通过验证集能量差早停审核。"
              f"三折的训练/验证/测试构型数依次为 {membership}。"
              f"训练轮数为 {fits.epochs_completed.min()}–{fits.epochs_completed.max()}，验证最优轮数为 "
              f"{fits.best_epoch.min()}–{fits.best_epoch.max()}。每折按训练集均值平移能量。"
              f"验证指标连续 {contract['convergence']['patience']} 轮未出现 0.5% 相对改善就停止，测试使用验证误差最低的权重。", '',
              'model_comparison.csv 列出配对 bootstrap 的 95% 区间，差值定义为 hetero MAE 减去相应模型 MAE。'
              + resampling + '区间描述这批构型的波动，跨材料优势需要另行验证。'
              ' [与零能差及物理对照的配对区间](baseline_comparison.csv)', '',
              '默认 defect_mean 读出下，hetero 和 hetero_fixed_pool 使用相同计算路径。名次保留全部十种模式，'
              '数值并列容差为 0.00000001 eV/超胞。', '',
              '复现时在 HERA 的父目录运行以下命令，将运行目录替换为新的目录。'
              '如有多个报告种子，分别替换训练命令中的种子。', '',
              '```sh',
              f"python -m HERA.scripts.run_arrangement_gap --dataset {args.dataset} --seed 123 "
              f"--batch-size {contract['batch_size']} --archive HERA/results/{spec.RESULTS.name}/{spec.BUNDLE} --run-dir REPRO_RUN",
              f"python -m HERA.scripts.analyze_arrangement_gap --dataset {args.dataset} --seeds "
              f"{' '.join(map(str,args.seeds))} --archive HERA/results/{spec.RESULTS.name}/{spec.BUNDLE} "
              '--run-dir REPRO_RUN --results REPRO_RESULTS',
              '```', '',
              f'[原论文]({spec.PAPER}) · [作者数据]({spec.SOURCE}) · [数据包]({bundle_link})',
              '', '[完整排名](model_comparison.csv) · [逐种子排名](per_seed_comparison.csv) · '
              '[逐排布对预测](heldout_energy_gaps.csv) · [训练审核](convergence.csv) · [比较图](energy_gap_comparison.png) · '
              f'[逐轮日志与权重]({log_link}/)', '']
    (out/'results.md').write_text('\n'.join(lines), encoding='utf8')
    print(results[['rank','mode','gap_MAE_eV_cell']].to_string(index=False))


if __name__ == '__main__':
    main()
