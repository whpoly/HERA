"""Average relation-message interventions over complete concentration strata.

Run from HERA's parent: python -m HERA.explain.relation_concentration
Uses frozen mixed-low shared hetero, 500 high-density structures per material.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .checkpoint_comparison import (
    ROOT, RELATIONS, init_elem_embedding, load_models, selected_cases,
    prepare_case, relation_effects, json_write,
)

COUNTS = [4, 9, 14, 19, 24]
LABELS = {'aa': 'A → A', 'ad': 'A → D', 'da': 'D → A', 'dd': 'D → D'}


def reference_predictions(material):
    directory = ROOT / 'logs/hetero_energy_mean_benchmark/high_test_predictions_fp32'
    analysis = json.loads((directory / 'low_to_high_analysis.json').read_text(encoding='utf-8'))
    frame = pd.read_csv(analysis['materials'][material]['prediction_csv']).set_index('source_id')
    return analysis, frame


def summarize(records):
    rows = []
    for r in records:
        # relation_effects evaluates its own unperturbed baseline. GPU scatter
        # reductions may differ by float32 rounding from the earlier replay.
        references = [e['ablated_prediction'] - e['delta_mev']/1000 for e in r['relations'].values()]
        if np.ptp(references) > 1e-10:
            raise ValueError('Relations do not share one intervention baseline')
        for rel, effect in r['relations'].items():
            rows.append({
                'material': r['material'], 'source_id': r['source_id'],
                'n_defects': r['n_defects'], 'n_lattice_sites': r['n_lattice_sites'],
                'concentration_pct': r['concentration_pct'], 'relation': rel,
                'relation_edges': r['relation_edges'][rel], 'prediction_ev_per_defect': r['prediction'],
                'ablation_reference_prediction_ev_per_defect': references[0],
                'target_ev_per_defect': r['target'],
                'ablated_prediction_ev_per_defect': effect['ablated_prediction'],
                'delta_mev_per_defect': effect['delta_mev'],
                'abs_delta_mev_per_defect': abs(effect['delta_mev']),
            })
    raw = pd.DataFrame(rows)
    summary = raw.groupby(['material', 'n_defects', 'concentration_pct', 'relation'], sort=True).agg(
        n=('source_id', 'size'), mean_delta_mev=('delta_mev_per_defect', 'mean'),
        mean_abs_delta_mev=('abs_delta_mev_per_defect', 'mean'),
        std_delta_mev=('delta_mev_per_defect', 'std'),
        median_delta_mev=('delta_mev_per_defect', 'median'),
        min_delta_mev=('delta_mev_per_defect', 'min'), max_delta_mev=('delta_mev_per_defect', 'max'),
        mean_relation_edges=('relation_edges', 'mean'),
    ).reset_index()
    return raw, summary


def render(output, summary, absolute=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm, Normalize
    from matplotlib.cm import ScalarMappable

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42})
    materials = list(summary.material.unique())
    field = 'mean_abs_delta_mev' if absolute else 'mean_delta_mev'
    limit = max(1, float(summary[field].abs().max()))
    norm = Normalize(0, limit) if absolute else TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    cmap = plt.get_cmap('YlOrRd' if absolute else 'RdBu_r')
    def panel(ax, material):
        frame = summary[summary.material == material]
        matrix = frame.pivot(index='n_defects', columns='relation', values=field).reindex(index=COUNTS, columns=RELATIONS)
        ax.imshow(matrix.values, cmap=cmap, norm=norm, aspect='auto')
        ax.set_xticks(range(4), [LABELS[k] for k in RELATIONS], fontsize=12)
        ax.xaxis.tick_top(); ax.tick_params(axis='both', which='both', length=0, pad=10)
        ax.set_yticks(range(5), [f'{100*c/192:.2f}%  ({c}/192)' for c in COUNTS], fontsize=11)
        for i in range(5):
            for j in range(4):
                value = matrix.iloc[i, j]
                ax.text(j, i, f'{value:.2f}' if absolute else f'{value:+.2f}', ha='center', va='center', fontsize=13,
                        fontweight='medium', color='white' if abs(value) > .58*limit else '#1d3045')
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.set_title(material, fontsize=19, fontweight='bold', pad=42, color='#203a50')
        ax.set_ylabel('Defect concentration', fontsize=11)

    def save(materials_to_draw, name):
        fig, axes = plt.subplots(1, len(materials_to_draw), figsize=(9 if len(materials_to_draw)==1 else 16, 5.9), squeeze=False)
        fig.subplots_adjust(left=.24 if len(materials_to_draw)==1 else .14, right=.82 if len(materials_to_draw)==1 else .90,
                            top=.71, bottom=.18, wspace=.63)
        for ax, material in zip(axes[0], materials_to_draw): panel(ax, material)
        fig.suptitle('Mean absolute relation-message effect by concentration' if absolute else 'Mean relation-message intervention by concentration', fontsize=17, y=.98, color='#203a50')
        fig.text(.5, .905, 'Shared hetero · mixed-low checkpoint · 100 structures per concentration / material', ha='center', fontsize=11, color='#506981')
        cb_ax = fig.add_axes([.865 if len(materials_to_draw)==1 else .925, .18, .025 if len(materials_to_draw)==1 else .015, .53])
        cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cb_ax)
        cb.set_label('Mean |Δ prediction| (meV / defect)' if absolute else 'Mean Δ prediction (meV / defect)', fontsize=10)
        fig.text(.5, .095, 'A = ordinary atom; D = actual defect.  Δ = prediction after intervention − original prediction.', ha='center', fontsize=10)
        fig.text(.5, .043, 'Zero one relation’s node-message numerator at every layer; gates, edge updates and geometry remain.', ha='center', fontsize=9, color='#61758a')
        if absolute: name += '_absolute'
        for suffix in ('png', 'pdf'): fig.savefig(output / f'{name}.{suffix}', dpi=180)
        plt.close(fig)

    save(materials, 'relation_mean_by_concentration')
    for material in materials: save([material], f'{material.lower()}_relation_mean')


def run(args):
    start = time.perf_counter()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    warnings.filterwarnings('ignore', message='.*Issues encountered while parsing CIF.*')
    init_elem_embedding(ROOT / 'atom_init.json')
    trainers, provenance = load_models(['hetero_shared'], args.device, 'mixed_low')
    checkpoint = provenance['hetero_shared']
    cases = [c for material in args.materials for c in selected_cases(material, COUNTS, per_count=100)]
    protocol = {
        'checkpoint_sha256': checkpoint['sha256'], 'cohort': 'mixed_low',
        'relation_intervention': 'Zero message numerator for one relation in all 3 ALIGNN + 3 GCN layers; retain gate denominator and edge updates',
        'quantity': '1000 * (ablated_prediction - original_prediction), meV per defect',
        'aggregation': 'Arithmetic mean over all 100 structures per material and defect-count stratum',
        'concentration': '100 * actual_defect_count / pristine_lattice_sites; includes vacancies and substitutions',
        'counts': COUNTS, 'materials': args.materials, 'expected_n': len(cases),
        'torch': torch.__version__, 'device': args.device, 'precision': 'FP32, TF32 disabled',
        'no_model_training': True,
    }
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'manifest.json'
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding='utf-8'))
        if old['protocol'] != protocol: raise ValueError('Existing output protocol differs')
    json_write(manifest_path, {'protocol': protocol, 'checkpoint': checkpoint, 'cases': cases})
    references = {}
    for material in args.materials:
        info, reference = reference_predictions(material)
        if info['checkpoint_sha256'] != checkpoint['sha256']: raise ValueError('Reference prediction checkpoint differs')
        references[material] = reference
    records = []
    for index, case in enumerate(cases):
        record_path = output / 'cases' / f'{case["key"]}.json'
        if record_path.exists():
            record = json.loads(record_path.read_text(encoding='utf-8'))
            if record['source_id'] != case['id'] or record['checkpoint_sha256'] != checkpoint['sha256']:
                raise ValueError('Existing record differs')
        else:
            with torch.inference_mode():
                wrappers, geometry, _ = prepare_case(case, trainers, args.device)
                wrapper = wrappers['hetero_shared']
                prediction = float(wrapper.physical(wrapper.x).item())
                effects = relation_effects(wrapper)
            baseline = references[case['material']].loc[case['id']]
            difference = abs(prediction-float(baseline.prediction))
            if difference > 5e-6: raise ValueError(f'Baseline prediction replay differs: {case["id"]}: {difference}')
            n_sites = len(geometry['positions'])
            if n_sites != 192: raise ValueError('Unexpected pristine site count')
            record = {'source_id': case['id'], 'material': case['material'], 'case_key': case['key'],
                      'n_defects': case['defect_count'], 'n_lattice_sites': n_sites,
                      'concentration_pct': 100*case['defect_count']/n_sites,
                      'checkpoint_sha256': checkpoint['sha256'], 'prediction': prediction,
                      'target': case['target'], 'baseline_replay_abs_difference_ev': difference,
                      'wrapper_parity': 'passed', 'restored_prediction_after_hooks': 'passed',
                      'relation_edges': {edge_type[1]: int(edge_index.shape[1]) for edge_type, edge_index in wrapper.graph.edge_index_dict.items()},
                      'relations': effects}
            json_write(record_path, record)
            del wrappers, wrapper
        records.append(record)
        if (index+1) % 25 == 0:
            elapsed = time.perf_counter()-start
            print(f'{index+1}/{len(cases)} complete | {case["material"]} {case["defect_count"]} defects | {elapsed:.1f}s elapsed', flush=True)
            json_write(output / 'progress.json', {'completed': index+1, 'total': len(cases), 'seconds': elapsed})
    raw, summary = summarize(records)
    if len(records) != len(cases) or not (summary.n == 100).all(): raise ValueError('Incomplete strata')
    raw.to_csv(output / 'per_structure_relations.csv', index=False)
    summary.to_csv(output / 'relation_summary.csv', index=False)
    for material in args.materials:
        matrix = summary[summary.material == material].pivot(index=['n_defects','concentration_pct'], columns='relation', values='mean_delta_mev').reindex(columns=RELATIONS)
        matrix.to_csv(output / f'{material.lower()}_mean_signed_mev.csv')
    validation = {'n_structures': len(records), 'n_interventions': len(raw), 'n_per_stratum': 100,
                  'max_baseline_replay_difference_ev': max(r['baseline_replay_abs_difference_ev'] for r in records),
                  'checkpoint_sha256_final': hashlib.sha256(Path(checkpoint['path']).read_bytes()).hexdigest(),
                  'prediction_mae_mev': {material: float(np.mean([1000*abs(r['prediction']-r['target']) for r in records if r['material']==material])) for material in args.materials}}
    if validation['checkpoint_sha256_final'] != checkpoint['sha256']: raise ValueError('Checkpoint file changed')
    json_write(output / 'validation.json', validation)
    render(output, summary)
    render(output, summary, absolute=True)
    lines = ['# 按缺陷浓度汇总的 relation-message 平均干预结果', '',
             '使用同一个 mixed-low shared hetero checkpoint，分别评估 MoS₂、WSe₂ high-density 测试集，每种材料 500 个结构，每档浓度 100 个结构。', '',
             '**主结果为带符号的算术平均**：Δ = 清零某类关系的节点消息后预测 − 原预测，单位 meV/defect。每个结构同等权重，不按边数加权。', '',
             '浓度 = 缺陷数 / 192 个完整晶格位点；缺陷包含空位和替位。A 表示普通原子，D 表示实际缺陷。', '',
             '逐类清零全部 3 个 ALIGNN 层与 3 个 GCN 层中的节点消息分子；保留门控分母、边更新、坐标和图拓扑。四种干预独立进行。该值是模型输出变化，不能相加解释成物理能量贡献。', '',
             '正值：清零后预测升高；负值：清零后预测降低。带符号平均可能发生正负抵消，完整 CSV 同时保存 mean_abs_delta_mev、标准差、中位数及逐样本数据。', '',
             f'权重：`{checkpoint["path"]}`', f'SHA256：`{checkpoint["sha256"]}`', '',
             '这不是之前仅有训练记录的 MoS₂-only 36.838 meV checkpoint。两种材料统一使用可核验的 mixed-low 权重。', '']
    for material in args.materials:
        lines += [f'## {material}', '', '|缺陷数|浓度|普通→普通|普通→缺陷|缺陷→普通|缺陷→缺陷|', '|---:|---:|---:|---:|---:|---:|']
        for count in COUNTS:
            row = summary[(summary.material==material)&(summary.n_defects==count)].set_index('relation')
            lines.append(f'|{count}|{100*count/192:.2f}%|'+'|'.join(f'{row.loc[k,"mean_delta_mev"]:+.2f}' for k in RELATIONS)+'|')
        lines += ['']
    lines += ['## 验证', '', f'未干预预测与已有完整测试结果逐结构核对，最大差异 {validation["max_baseline_replay_difference_ev"]:.3g} eV。',
              f'完整测试 MAE：{validation["prediction_mae_mev"]} meV/defect。', '',
              '复现：在 HERA 父目录执行 `python -m HERA.explain.relation_concentration`。已有逐样本结果会自动复用。']
    (output / 'README.md').write_text('\n'.join(lines), encoding='utf-8')
    print(summary[['material','n_defects','relation','n','mean_delta_mev','mean_abs_delta_mev']].to_string(index=False), flush=True)
    print(f'COMPLETE {len(records)} structures at {output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=str(ROOT / 'results/relation_message_concentration_20260917'))
    parser.add_argument('--materials', nargs='+', choices=['MoS2','WSe2'], default=['MoS2','WSe2'])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--render-only', action='store_true')
    args = parser.parse_args()
    if args.render_only:
        output = Path(args.output).resolve()
        summary = pd.read_csv(output / 'relation_summary.csv')
        render(output, summary)
        render(output, summary, absolute=True)
    else:
        run(args)
