"""Recompute and explain the frozen native local-r6 experiment; no training."""
import json
from datetime import datetime
import re

import numpy as np

from . import finalize_native_hypergraph_paired as audit_tool


ROOT = audit_tool.ROOT
OUT = ROOT / 'results/native_hypergraph_local_r6_seed123_prepared_20260930'
OLD = audit_tool.DEFAULT_OUT
BASE = audit_tool.BASELINE_PREDICTIONS
KEYS = ('local_r6', 'local_global_r3', 'hetero', 'was_x')
NAMES = {'local_r6': 'Hypergraph local、6 Å',
         'local_global_r3': 'Hypergraph local_global、3 Å',
         'hetero': 'hetero_was_no_dd', 'was_x': 'was_x'}


def stats(errors):
    e = np.asarray(errors, dtype=np.float64)
    assert len(e) and np.isfinite(e).all()
    squared = e ** 2
    return dict(n=len(e), mae_ev=float(np.mean(abs(e))),
                rmse_ev=float(np.sqrt(np.mean(squared))),
                median_absolute_error_ev=float(np.median(abs(e))),
                p90_absolute_error_ev=float(np.quantile(abs(e), .9)),
                p95_absolute_error_ev=float(np.quantile(abs(e), .95)),
                max_absolute_error_ev=float(max(abs(e))),
                n_absolute_error_over_1ev=int(np.sum(abs(e) > 1)),
                sum_squared_error_ev2=float(squared.sum()),
                top5_squared_error_percent=float(np.sort(squared)[-5:].sum() / squared.sum() * 100),
                top20_squared_error_percent=float(np.sort(squared)[-20:].sum() / squared.sum() * 100))


def main():
    audits = {}
    for key, directory in (('local_r6', OUT), ('local_global_r3', OLD)):
        audit_tool.OUT = directory
        audits[key] = audit_tool.audit()
    inputs = [OUT / 'predictions.csv', OLD / 'predictions.csv', BASE,
              OUT / 'protocol.json', OLD / 'protocol.json']
    new = {r['source_id']: r for r in audit_tool.read_csv(inputs[0])}
    old = {r['source_id']: r for r in audit_tool.read_csv(inputs[1])}
    baseline = audit_tool.read_csv(BASE)
    reference = audit_tool.load_trusted_checkpoint(audit_tool.REFERENCE, 'cpu')
    family = lambda sid: re.sub(r'-POSCAR\d+', '', sid)
    train_families = {family(r['source_id']) for r in reference['split']['train_sources']}
    result = {'generated_at': datetime.now().astimezone().isoformat(),
              'source_sha256': {str(path): audit_tool.sha256(path) for path in inputs},
              'audits': audits, 'splits': {}}
    for part in ('val', 'test'):
        rows = [r for r in baseline if r['split'] == part]
        assert len(rows) == 604 and len({r['source_id'] for r in rows}) == 604
        errors = {key: [] for key in KEYS}
        details = []
        for row in rows:
            sid, target = row['source_id'], float(row['target'])
            assert row['configuration_family'] == family(sid)
            assert (row['family_seen_in_train'] == 'True') == (family(sid) in train_families)
            for other in (new[sid], old[sid]):
                assert other['split'] == part and abs(float(other['target']) - target) < 1e-5
            predictions = dict(local_r6=float(new[sid]['prediction']),
                               local_global_r3=float(old[sid]['prediction']),
                               hetero=float(row['hetero_was_no_dd_prediction']),
                               was_x=float(row['was_x_prediction']))
            detail = {key: row[key] for key in ('source_id', 'material', 'defect_type',
                                               'configuration_family', 'family_seen_in_train')}
            detail.update(target_ev=target, predictions_ev=predictions,
                          errors_ev={key: prediction - target for key, prediction in predictions.items()})
            for key in KEYS:
                errors[key].append(detail['errors_ev'][key])
            detail['sse_improvement_vs_old_ev2'] = detail['errors_ev']['local_global_r3'] ** 2 - detail['errors_ev']['local_r6'] ** 2
            details.append(detail)
        errors = {key: np.asarray(values) for key, values in errors.items()}
        metrics = {key: stats(values) for key, values in errors.items()}
        pairs = {}
        for key in ('local_global_r3', 'hetero'):
            pairs[key] = dict(
                new_better_count=int(np.sum(abs(errors['local_r6']) < abs(errors[key]))),
                new_worse_count=int(np.sum(abs(errors['local_r6']) > abs(errors[key]))),
                rmse_change_ev=metrics['local_r6']['rmse_ev'] - metrics[key]['rmse_ev'],
                rmse_change_percent=(metrics['local_r6']['rmse_ev'] / metrics[key]['rmse_ev'] - 1) * 100,
                mae_change_percent=(metrics['local_r6']['mae_ev'] / metrics[key]['mae_ev'] - 1) * 100)
        grouped = {}
        for field in ('defect_type', 'material', 'family_seen_in_train'):
            grouped[field] = {}
            for name in sorted({row[field] for row in rows}):
                mask = np.array([row[field] == name for row in rows])
                grouped[field][name] = {key: stats(e[mask]) for key, e in errors.items()}
            assert sum(item['local_r6']['n'] for item in grouped[field].values()) == 604
        result['splits'][part] = dict(metrics=metrics, comparisons=pairs, groups=grouped,
                                     per_sample=details)

    audit_tool.write_text(OUT / 'analysis.json', json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    test = result['splits']['test']
    val = result['splits']['val']
    metrics = test['metrics']
    lines = ['## 完整结果分析', '', f'分析时间：{result["generated_at"]}。以下结果来自全部固定测试样本的逐条对齐比较，没有删除难例或重新训练。', '',
             '**local-only、6 Å 的组合修改改善了原 Hypergraph 的总体测试误差，但仍未超过 hetero。**', '',
             '| 模型 | Validation MAE | Validation RMSE | Test MAE | Test RMSE |',
             '|---|---:|---:|---:|---:|']
    for key in KEYS:
        v, t = val['metrics'][key], metrics[key]
        lines.append(f'| {NAMES[key]} | {v["mae_ev"]:.6f} | {v["rmse_ev"]:.6f} | {t["mae_ev"]:.6f} | {t["rmse_ev"]:.6f} |')
    lines += ['', '单位均为 eV。四者使用相同的 604 条验证及 604 条测试样本。新模型完成 500 轮，使用验证 MAE 最优的第 492 轮权重；测试集未参与权重选择。', '',
              f'四个模型中，was_x 的 Test MAE 最低（{metrics["was_x"]["mae_ev"]:.6f} eV），hetero 的 Test RMSE 最低（{metrics["hetero"]["rmse_ev"]:.6f} eV）。'
              f'hetero 的 RMSE 比 was_x 的 {metrics["was_x"]["rmse_ev"]:.6f} eV 低 '
              f'{(1-metrics["hetero"]["rmse_ev"]/metrics["was_x"]["rmse_ev"])*100:.2f}%。'
              'MAE 直接平均绝对误差，而 RMSE 先平方再平均，对大误差更敏感，因此两种指标的模型排名可以不同。', '']
    for key in ('local_global_r3', 'hetero'):
        p = test['comparisons'][key]
        lines.append(f'相对 {NAMES[key]}：Test MAE 变化 {p["mae_change_percent"]:+.2f}%，RMSE 变化 {p["rmse_change_percent"]:+.2f}%（{p["rmse_change_ev"]:+.6f} eV）。')
    lines += ['', '### 改善集中在部分大误差样本', '',
              f'相对旧 Hypergraph，新模型在 {test["comparisons"]["local_global_r3"]["new_better_count"]}/604 条样本上绝对误差更小，其余更大。'
              '因此总体分数提升不等于多数结构都预测得更好。', '',
              '| 模型 | 绝对误差中位数 | 绝对误差 P90 | 误差 >1 eV 数量 | 最差 20 条的平方误差占比 |',
              '|---|---:|---:|---:|---:|']
    for key in KEYS:
        t = metrics[key]
        lines.append(f'| {NAMES[key]} | {t["median_absolute_error_ev"]:.6f} | {t["p90_absolute_error_ev"]:.6f} | {t["n_absolute_error_over_1ev"]} | {t["top20_squared_error_percent"]:.2f}% |')
    lines += ['', '旧版到新版的中位误差和 P90 均升高，而 >1 eV 的样本从 19 条降到 16 条，说明主要收益来自部分大误差结构改善。'
              '新版最差 20/604 条（3.31%）仍贡献约 93.16% 的平方误差，长尾误差问题依然突出。', '',
              '下表为新版绝对误差最大的五条，数值为预测减标签（eV）；负数表示低估。', '',
              '| source ID | local 6 Å 误差 | local_global 3 Å 误差 | hetero 误差 |',
              '|---|---:|---:|---:|']
    for row in sorted(test['per_sample'], key=lambda r: abs(r['errors_ev']['local_r6']), reverse=True)[:5]:
        e = row['errors_ev']
        lines.append(f'| {row["source_id"]} | {e["local_r6"]:.6f} | {e["local_global_r3"]:.6f} | {e["hetero"]:.6f} |')
    lines += ['', 'BN 间隙和 SiC 替位的两个最大误差样本都有改善，但与 hetero 的差距仍很大。'
              '最差样本多为低估，不足以据此判断标签有误、构图有误或信息传播不足；需要单独检查结构与标签来源。', '',
              '### 不同缺陷类型的变化并不一致', '',
              '| 缺陷类型 | Test n | local 6 Å RMSE | local_global 3 Å RMSE | hetero RMSE |',
              '|---|---:|---:|---:|---:|']
    for name, label in (('interstitial', '间隙'), ('substitution', '替位'), ('vacancy', '空位')):
        g = test['groups']['defect_type'][name]
        lines.append(f'| {label} | {g["local_r6"]["n"]} | {g["local_r6"]["rmse_ev"]:.6f} | {g["local_global_r3"]["rmse_ev"]:.6f} | {g["hetero"]["rmse_ev"]:.6f} |')
    lines += ['', '相对旧版，间隙和替位 RMSE 降低，空位 RMSE 升高；相对 hetero，新版空位略好，间隙和替位仍较差。'
              '替位仅 46 条且含极端误差，组间差异不能直接等同于架构对该缺陷类型的普遍优劣。', '',
              '### 验证、测试与构型家族', '',
              '新旧 Hypergraph 验证 MAE 几乎持平（0.074379 对 0.074526），验证 RMSE 还略升（0.214903 对 0.213568）。'
              '新版验证集优于 hetero，测试集却相反，说明当前单划分上的模型排序不稳定，不能只看验证分数推断测试优势。', '',
              '| 构型家族是否在训练中出现 | Test n | local 6 Å RMSE | local_global 3 Å RMSE | hetero RMSE |',
              '|---|---:|---:|---:|---:|']
    for seen, label in (('False', '未出现'), ('True', '已出现')):
        g = test['groups']['family_seen_in_train'][seen]
        lines.append(f'| {label} | {g["local_r6"]["n"]} | {g["local_r6"]["rmse_ev"]:.6f} | {g["local_global_r3"]["rmse_ev"]:.6f} | {g["hetero"]["rmse_ev"]:.6f} |')
    lines += ['', '构型家族按 source ID 去掉 `-POSCAR数字` 定义，本次已用参考 checkpoint 的全部训练 source ID 重新核验。'
              '训练未见家族的 16 条上新版更好，是一个值得后续验证的局部信号；样本太少，且这不是独立设计的家族留出实验，不能宣称已证明更强的外推能力。', '',
              '### 对架构的判断与后续对照', '',
              '当前证据支持这次组合改动相对旧版有小幅收益，尚不支持 hypergraph 在 native 上优于 hetero。'
              '两版均使用 defect_energy_mean，本实验没有对比全图 readout，不能回答整图 embedding 是否更合适。', '',
              '本次同时关闭 global 消息传递并扩大 local 半径；扩大半径也改变 near/far 区域标签。'
              '不能把收益单独归因于删除 global，也不能认定半径越大越好。要区分作用，可补齐 `local、3 Å` 和 `local_global、6 Å` 两个对照，形成 2×2 实验；'
              '它们目前尚未运行。跨 seed 的稳定性仍需独立检验，后续架构选择应使用验证集，避免反复按这批测试结果调参。', '',
              f'[机器可读分析与逐样本误差]({(OUT / "analysis.json").as_posix()}) · '
              f'[可重算脚本]({(ROOT / "scripts/analyze_native_hypergraph_local_r6.py").as_posix()})', '']
    report_path = OUT / 'report.md'
    content = report_path.read_text(encoding='utf-8').split('\n## 完整结果分析\n')[0].rstrip()
    audit_tool.write_text(report_path, content + '\n\n' + '\n'.join(lines))
    print(json.dumps({'test': metrics, 'comparisons': test['comparisons'],
                      'report': str(report_path)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
