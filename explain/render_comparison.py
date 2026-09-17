"""Render checkpoint explanation results as scientific figures and a standalone viewer."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, FuncNorm
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from ase.io import read
from ase.utils import rotate

NAMES = {'attention': 'Attention', 'hetero_old': 'Independent hetero', 'hetero_shared': 'Shared hetero'}
COLORS = {'attention': '#315a96', 'hetero_old': '#cf8a35', 'hetero_shared': '#238773'}
COHORTS = {'mixed_low': 'Mixed-low training', 'mos2_single': 'MoS2-only training', 'wse2_single': 'WSe2-only training'}


def mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def load_bundle(root):
    bundle, rows = [], []
    for directory in (root / key for key in COHORTS):
        if not (directory / 'results.json').exists():
            continue
        manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
        results = json.loads((directory / 'results.json').read_text(encoding='utf-8'))
        for case in manifest['cases']:
            folder = directory / 'cases' / case['key']
            geometry = json.loads((folder / 'structure.json').read_text(encoding='utf-8'))
            atoms = read(folder / 'structure.extxyz')
            transform = rotate('22x,0y,8z')
            coordinates = atoms.positions.copy()
            center = coordinates.mean(0)
            projected = (coordinates-center) @ transform
            corners = np.array([np.zeros(3), atoms.cell[0], atoms.cell[0]+atoms.cell[1], atoms.cell[1], np.zeros(3)])
            corners[:, 2] = center[2]
            outline = (corners-center) @ transform
            entry = {'cohort': directory.name, 'case': case, 'geometry': geometry,
                     'projected': projected.tolist(), 'outline': outline.tolist(), 'models': {}}
            for record in results:
                if record['case']['key'] != case['key']:
                    continue
                name = record['model']
                masks = np.load(folder / f'{name}_masks.npz')
                metrics = {'cohort': directory.name, 'material': case['material'], 'case': case['key'],
                           'defect_count': case['defect_count'], 'model': name,
                           'prediction_error_mev': record['abs_error_mev'],
                           'preservation_mev': mean([v['prediction_preservation_error_mev'] for v in record['runs']]),
                           'consensus_preservation_mev': record['consensus_preservation_error_mev'],
                           'rank_spearman': mean([v['spearman'] for v in record['stability']]),
                           'top10_jaccard': mean([v['top10pct_jaccard'] for v in record['stability']]),
                           'mask_mean': float(masks['mean'].mean()),
                           'zero_embedding_change_mev': record['zero_embedding_change_mev']}
                deletion = next(v for v in record['deletion'] if v['fraction'] == .1)
                metrics.update({k: v for k, v in deletion.items() if k.endswith('_mev') and not isinstance(v, list)})
                metrics['top_beats_random'] = deletion['top_change_mev'] > deletion['random_mean_mev']
                metrics['top_beats_bottom'] = deletion['top_change_mev'] > deletion['bottom_change_mev']
                rows.append(metrics)
                entry['models'][name] = {**record, 'mask_mean': masks['mean'].tolist(), 'mask_std': masks['std'].tolist(),
                                         'metrics': metrics, 'checkpoint': manifest['checkpoints'][name]}
            bundle.append(entry)
    return bundle, pd.DataFrame(rows)


def draw_sites(ax, entry, values, norm, cmap, label_defects=False):
    coordinates = np.array(entry['projected'])
    geometry = entry['geometry']
    types = np.array(geometry['node_type'])
    vacancy = np.array([str(s).startswith('X') for s in geometry['species']])
    outline = np.array(entry['outline'])
    ax.plot(outline[:, 0], outline[:, 1], '--', color='#9ba6b3', lw=.8, zorder=0)
    for ids, marker, size in [(np.where(types == 0)[0], 'o', 21),
                              (np.where((types == 1) & ~vacancy)[0], 's', 48),
                              (np.where(vacancy)[0], 'X', 70)]:
        ax.scatter(coordinates[ids, 0], coordinates[ids, 1], c=np.asarray(values)[ids],
                   cmap=cmap, norm=norm, marker=marker, s=size, linewidths=.55, edgecolors='#697180', zorder=3)
    if label_defects:
        for i in np.flatnonzero(types):
            ax.annotate(str(i), coordinates[i, :2], xytext=(3, 3), textcoords='offset points',
                        fontsize=6, color='#333b49', zorder=4)
    ax.set_aspect('equal'); ax.set_axis_off()


def case_figure(entry, output):
    models = entry['models']
    if not models:
        return
    n = len(models)
    fig, axes = plt.subplots(2, n, figsize=(5*n, 9), squeeze=False, layout='constrained')
    max_effect = max(max(np.abs(m['single_site_ablation']['signed_delta_mev'])) for m in models.values())
    sensitivity_norm = FuncNorm((np.log1p, np.expm1), vmin=0, vmax=max(1, max_effect))
    for j, (name, record) in enumerate(models.items()):
        metrics = record['metrics']
        draw_sites(axes[0, j], entry, record['mask_mean'], Normalize(0, 1), 'viridis')
        axes[0, j].set_title(f'{NAMES[name]}\nPrediction {record["prediction"]:.4f} eV/defect | error {record["abs_error_mev"]:.1f} meV', fontsize=11)
        axes[0, j].text(.5, -.03, f'Soft-mask error {metrics["preservation_mev"]:.2f} meV\nSeed rank correlation {metrics["rank_spearman"]:.2f}; top-10% Jaccard {metrics["top10_jaccard"]:.2f}',
                       transform=axes[0, j].transAxes, ha='center', va='top', fontsize=9)
        draw_sites(axes[1, j], entry, np.abs(record['single_site_ablation']['signed_delta_mev']), sensitivity_norm, 'magma', True)
        axes[1, j].set_title('Single-site embedding ablation', fontsize=11)
    case = entry['case']
    fig.suptitle(f'{case["material"]} | {case["defect_count"]} defects | {COHORTS[entry["cohort"]]}\nDFT target {case["target"]:.4f} eV/defect; site IDs are zero-based', fontsize=15)
    fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap='viridis'), ax=list(axes[0]), shrink=.72, label='GNNExplainer mask (0–1; 3-seed mean)')
    cbar = fig.colorbar(ScalarMappable(norm=sensitivity_norm, cmap='magma'), ax=list(axes[1]), shrink=.72, label='Absolute prediction change (meV; log1p scale)')
    ticks = [v for v in [0, 1, 10, 100, 1000, 10000] if v <= max_effect]
    cbar.set_ticks(ticks); cbar.set_ticklabels([str(t) for t in ticks])
    handles = [Line2D([], [], marker=m, ls='', color='#738392', label=l, markersize=7) for m, l in [('o', 'Pristine'), ('s', 'Substitution'), ('X', 'Vacancy placeholder')]]
    fig.legend(handles=handles, loc='outside lower center', ncol=3, frameon=False)
    filename = f'{entry["cohort"]}_{case["key"]}'
    fig.savefig(output / f'{filename}.png', dpi=180)
    fig.savefig(output / f'{filename}.pdf')
    plt.close(fig)
    # A separate semantic view keeps native quantities distinct from explainer masks.
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4.6), layout='constrained', squeeze=False)
    for ax, (name, record) in zip(axes[0], models.items()):
        effects = record['relation_ablation']
        if effects:
            labels = ['atom→atom', 'atom→defect', 'defect→atom', 'defect→defect']
            values = [effects[k]['delta_mev'] for k in ('aa', 'ad', 'da', 'dd')]
            ax.barh(labels, values, color=[COLORS[name] if v >= 0 else '#bd6358' for v in values])
            ax.axvline(0, color='#444', lw=.7)
            ax.set_xlabel('Prediction after ablation − full prediction (meV)')
            ax.set_title(NAMES[name] + '\nRelation-message ablation')
            ax.invert_yaxis()
        else:
            weights = np.array(record['native_readout']['values'])
            ids = np.argsort(-weights)[:8]
            labels = [f'#{i} {entry["geometry"]["species"][i]}' for i in ids]
            ax.barh(labels, weights[ids], color=COLORS[name]); ax.invert_yaxis()
            ax.set_xlabel('Global pooling weight (sums to 1 over all sites)')
            ax.set_title('Attention\nNative readout weights (top 8)')
    fig.suptitle(f'{case["material"]}, {case["defect_count"]} defects — native model diagnostics\nAttention weights and relation ablations have different meanings', fontsize=13)
    fig.savefig(output / f'{filename}_relations.png', dpi=180)
    fig.savefig(output / f'{filename}_relations.pdf')
    plt.close(fig)


def overview(df, output):
    keys = [(c, m) for c in COHORTS for m in ['MoS2', 'WSe2'] if len(df[(df.cohort == c) & (df.material == m)])]
    fig, axes = plt.subplots(2, len(keys), figsize=(4.7*len(keys), 7.7), layout='constrained', squeeze=False)
    for j, (cohort, material) in enumerate(keys):
        subset = df[(df.cohort == cohort) & (df.material == material)]
        for name in NAMES:
            group = subset[subset.model == name]
            if group.empty: continue
            axes[0,j].plot(group.defect_count, group.preservation_mev, 'o-', c=COLORS[name], label=NAMES[name])
            axes[1,j].plot(group.defect_count, group.top10_jaccard, 'o-', c=COLORS[name])
        axes[0,j].set_title(f'{material}\n{COHORTS[cohort]}', fontsize=11)
        axes[0,j].set_yscale('symlog', linthresh=1)
        axes[0,j].set_ylabel('Soft-mask prediction error (meV)')
        axes[1,j].set_ylabel('Top-10% Jaccard across seeds')
        axes[1,j].set_ylim(0, 1); axes[1,j].set_xlabel('Defect count')
        for ax in axes[:,j]:
            ax.set_xticks([4,9,14,19,24]); ax.grid(alpha=.18)
            ax.spines[['top','right']].set_visible(False)
    axes[0,0].legend(fontsize=8, frameon=False)
    fig.suptitle('Explanation quality: preservation and repeatability\nOne predetermined structure per defect count; each mask repeated with 3 seeds', fontsize=14)
    fig.savefig(output / 'quality_overview.png', dpi=180)
    fig.savefig(output / 'quality_overview.pdf')
    plt.close(fig)


def write_report(root, bundle, df):
    aggregates = df.groupby(['cohort','material','model'], sort=False).agg(
        n=('case','size'), selected_mae=('prediction_error_mev','mean'), preservation=('preservation_mev','mean'),
        rank=('rank_spearman','mean'), jaccard=('top10_jaccard','mean'),
        top_random=('top_beats_random','sum'), top_bottom=('top_beats_bottom','sum')).reset_index()
    df.to_csv(root / 'quality_metrics.csv', index=False)
    aggregates.to_csv(root / 'quality_summary.csv', index=False)
    table = ['|训练组|材料|模型|样本数|样本 MAE / meV|掩码保真误差 / meV|排名相关|Top10% Jaccard|Top>随机|Top>Bottom|',
             '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in aggregates.itertuples():
        table.append(f'|{row.cohort}|{row.material}|{NAMES[row.model]}|{row.n}|{row.selected_mae:.2f}|{row.preservation:.2f}|{row.rank:.3f}|{row.jaccard:.3f}|{row.top_random}/{row.n}|{row.top_bottom}/{row.n}|')
    comparisons = []
    for (cohort, material), group in aggregates.groupby(['cohort', 'material'], sort=False):
        by_model = group.set_index('model')
        if 'hetero_shared' in by_model.index and 'attention' in by_model.index:
            shared, attention = by_model.loc['hetero_shared'], by_model.loc['attention']
            comparisons.append(f'- **{cohort} / {material}**：shared vs attention 的 Top10% Jaccard 为 **{shared.jaccard:.3f} vs {attention.jaccard:.3f}**；软掩码保真误差为 **{shared.preservation:.2f} vs {attention.preservation:.2f} meV**。')
    provenance = []
    seen = set()
    for entry in bundle:
        for name, record in entry['models'].items():
            key = (entry['cohort'], name)
            if key in seen: continue
            seen.add(key); ckpt = record['checkpoint']; cfg = ckpt['effective_config']['model']
            provenance.append(f'- **{entry["cohort"]} / {NAMES[name]}**：`{ckpt["path"]}`；epoch {ckpt["best_epoch"]}；SHA256 `{ckpt["sha256"]}`；归一化配置 `{json.dumps({k:v for k,v in cfg.items() if "norm" in k}, ensure_ascii=False)}`。')
    text = f'''# Hetero 与 attention 的 checkpoint 解释对照

## 结论

**Hetero 提供了更清晰的关系类别；这不等于 GNNExplainer 的节点解释更稳定或更可信。** 每张图同时给出预测、解释保真度和跨随机种子的稳定性。Shared hetero 还可以输出逐缺陷的潜在 readout 值，其均值严格还原模型预测，但这些值不是可唯一确定的物理缺陷能量。

本次完成 MoS₂、WSe₂ 各 5 个预先固定的 high-density 结构（4、9、14、19、24 缺陷），合计 {len(df)} 个结构/模型组合。所有预测模型使用现有 checkpoint，未重新训练；优化的仅是解释掩码。

- 打开 `viewer.html`：切换训练组、材料、缺陷数和解释方式；悬停节点查看位点 ID、元素、类型、掩码不确定性及单点干预结果。
- `figures/`：每个对照的 PNG 和 PDF，可用于 PPT；`quality_overview.png` 汇总解释质量。
- 各训练组 `cases/`：逐节点 CSV、3 次掩码 NPZ、含重要性的 ASE/OVITO extxyz、逐样本 JSON。`quality_metrics.csv` 为完整数值表。

网页预览说明：当前内置浏览器拒绝加载 `file://` 本地路径，尚未完成交互验证。HTML 脚本已通过语法检查，PNG/PDF 已生成并检查；请优先使用这些静态图查看结果。

## 对照口径

1. **mixed_low**：采用本次新下载的 mixed-low attention、independent hetero，与 mixed-low shared hetero 对照；训练/验证/测试 ID 及顺序、标签 scaler 均已核对相同。新下载的 attention 是 BatchNorm 兼容版本，下载时间不代表网络架构更新。它保存的 4.052 meV 属于 low 测试集，不能与 high 测试集的 51.59 meV 混用。
2. **mos2_single**：较新的全程 LayerNorm attention（checkpoint 保存的 high MAE 56.992 meV）与同一 MoS₂ 划分的 independent hetero。未找到可核验的 MoS₂-only shared 权重，不用 mixed 权重代替该列。
3. **wse2_single**：全程 LayerNorm attention（46.233 meV）、independent hetero、shared hetero；训练划分相同。
4. 各组使用完全相同的对应 high 结构、位点顺序、缺陷标签和坐标。不同模型的构图/归一化/readout 可能不同，因此这是**现有 checkpoint 的比较**，无法单独归因于参数共享。全程保留各模型自己的边，未改成全缺陷连接图。
5. 新 attention 51.59 meV 那次仍需对应 checkpoint 才能建立准确关联；不把其他权重改名成该次运行。

## 解释质量实测

下面的 MAE 仅来自每种材料选定的 5 个结构，不是完整 500 个结构的测试 MAE。掩码保真误差越小代表更接近原模型输出；排名相关、Jaccard 越高代表重复性越好。Top>随机/Bottom 表示清零前 10% 高分节点造成的输出变化超过随机/低分节点的样本数。

{chr(10).join(table)}

{chr(10).join(comparisons)}

软掩码能重现预测，并不能单独证明节点排名正确：在 LayerNorm 和冗余信息存在时，很多不同掩码可能保留同一预测。若 Top10% 稳定性低，或高分节点的清零效果不优于低分/随机，就不宜把颜色最亮的位点当成已验证的物理机制。

## 图的读法

- **GNNExplainer**：解释冻结模型的预测，而非 DFT 标签；每个原始晶格位点一个标量掩码。所有模型共用 0–1 色标，无逐图 min-max 放大。展示 3 个 seed 的均值，CSV 中保留各次值及标准差。
- **单节点干预**：逐个将初始节点 embedding 清零，记录输出变化。图用绝对变化，悬停保留正负号；每个结构的多个模型共用色标，使用 log(1+x) 以显示数量级差异。它提供无需随机优化的模型敏感度核对，但不等于删除真实原子的能量。
- **关系类别**：aa=普通→普通，ad=普通→缺陷，da=缺陷→普通，dd=缺陷→缺陷。分别在全部消息层清零该关系的节点消息分子，保留门控分母、边更新和几何，比较预测变化。它不等于删键，四个变化也不能相加当作总能量。
- **逐缺陷 readout**：shared energy-mean 模型的每个实际缺陷都有一个潜在标量，满足预测 = 这些值的平均数。可组织“哪些缺陷的内部输出偏高”的叙述；不可视为真实单缺陷形成能，也不可与 GNNExplainer 分数混用。
- **Attention native 权重**：另外展示全图 pooling 权重。Attention 本身也使用节点类型信息，故所有模型的图都标注普通节点、替位和空位；不会只给 hetero 加标签而制造可读性优势。权重本身不是已验证的因果归因。

## 方法与验证

采用 [GNNExplainer](https://arxiv.org/abs/1903.03894) 的 [PyG 实现](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/explain/algorithm/gnn_explainer.html)，`explanation_type=model`，node mask=object，不优化 edge mask。150 epochs、lr=0.03、size=0.01、entropy=0.001，seed=123/456/789；每个位点同一 mean 正则，不按节点类型重复平均。超参数对所有模型相同，本次未进行大规模超参数搜索。

掩码施加在初始的 64 维 learned embedding 上。原始 92 维空位特征全为零，直接乘原始特征会让空位无论如何都不受掩码影响；改在 embedding 层干预后，测试确认空位掩码存在梯度。模型参数全部冻结且 eval 模式，结构、节点类型、边和距离固定。原始模型与 wrapper 未加掩码的输出逐例一致；集成测试覆盖三种模型，确认解释前后权重逐张量不变、关系 hook 遇到异常也会清理；全部逐缺陷 readout 的均值已校验还原预测。

样本选择为每个缺陷数中 SHA256(`20260917:source_id`) 最小的一个，与模型误差和热图外观无关。随机干预重复 12 次（5%、10%、20% 节点比例）。这是小样本可视化与机制诊断，不是用户可读性实验，也不是完整的因果验证。

3 个随机种子是 **explainer 的初始化种子**；现有预测 checkpoint 的训练 seed 均为 123，因此这里没有验证不同训练种子之间解释是否一致。100-epoch 流程预检与最终 150-epoch 结果有差异，表明掩码仍会受优化时长影响；本报告统一采用后者，不把预检数值混入正式对照。

## 复现

在 HERA 的父目录运行（已有 ML 环境）：

```powershell
python -m HERA.explain.checkpoint_comparison --output HERA/results/hetero_explainer_20260917/mixed_low
python -m HERA.explain.checkpoint_comparison --cohort single_material --materials MoS2 --models attention hetero_old --output HERA/results/hetero_explainer_20260917/mos2_single
python -m HERA.explain.checkpoint_comparison --cohort single_material --materials WSe2 --output HERA/results/hetero_explainer_20260917/wse2_single
python -m HERA.explain.enrich_comparison HERA/results/hetero_explainer_20260917/mixed_low HERA/results/hetero_explainer_20260917/mos2_single HERA/results/hetero_explainer_20260917/wse2_single
python -m HERA.explain.render_comparison HERA/results/hetero_explainer_20260917
python -m unittest HERA.tests.test_checkpoint_explainer -v
```

## 权重来源

{chr(10).join(provenance)}
'''
    (root / 'README.md').write_text(text, encoding='utf-8')
    return aggregates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    root = args.root.resolve(); output = root / 'figures'; output.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.titleweight': 'medium', 'pdf.fonttype': 42})
    bundle, df = load_bundle(root)
    for entry in bundle:
        case_figure(entry, output)
        print(f'RENDERED {entry["cohort"]} {entry["case"]["key"]}', flush=True)
    overview(df, output)
    summary = write_report(root, bundle, df)
    template = Path(__file__).with_name('comparison_viewer.html').read_text(encoding='utf-8')
    payload = json.dumps({'entries': bundle, 'summary': json.loads(summary.to_json(orient='records'))}, ensure_ascii=False).replace('</', '<\\/')
    (root / 'viewer.html').write_text(template.replace('__PAYLOAD__', payload), encoding='utf-8')
    print(summary.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
