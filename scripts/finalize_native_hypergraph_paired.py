"""Audit a finished native HyperALIGNN run and publish its local report."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re

from ..training.trainer import load_trusted_checkpoint


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / 'results/native_hypergraph_paired_20260929'
OUT = DEFAULT_OUT
MAIN_REPORT = ROOT / 'results/official_alignn_native_20260929/report.md'
REFERENCE = ROOT / 'logs/alignn_v4/alignn/native/hetero_was_no_dd/seed123_best_checkpoint.pth'
BASELINE_PREDICTIONS = ROOT / 'results/native_hetero_vs_was_x_20260928/paired_predictions.csv'


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_text(path, content):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(content, encoding='utf-8')
    temporary.replace(path)


def read_csv(path):
    with path.open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def score(rows):
    errors = [float(row['prediction']) - float(row['target']) for row in rows]
    if not errors or not all(math.isfinite(value) for value in errors):
        raise ValueError('Missing or nonfinite per-sample predictions.')
    return {'n': len(errors), 'mae_ev': sum(map(abs, errors)) / len(errors),
            'rmse_ev': math.sqrt(sum(value * value for value in errors) / len(errors))}


def audit():
    result = json.loads((OUT / 'metrics.json').read_text(encoding='utf-8'))
    protocol = json.loads((OUT / 'protocol.json').read_text(encoding='utf-8'))
    if result['protocol'] != protocol or protocol['reference_sha256'] != sha256(REFERENCE):
        raise ValueError('Protocol or reference checkpoint hash changed.')
    if protocol['split_counts'] != {'train': 1818, 'val': 604, 'test': 604}:
        raise ValueError('Unexpected split counts.')
    if protocol['epochs'] != 500 or protocol['seed'] != 123:
        raise ValueError('Unexpected training budget or seed.')
    if protocol['checkpoint_selection'] != 'validation MAE':
        raise ValueError('Unexpected checkpoint selection.')
    if result['epochs_completed'] != 500 or not 1 <= result['best_epoch'] <= 500:
        raise ValueError('Training did not finish all 500 epochs.')
    if not result['identical_ordered_splits'] or not result['identical_scaler']:
        raise ValueError('Runner rejected the paired split or scaler.')
    checkpoint = OUT / 'seed123_best_checkpoint.pth'
    if result['checkpoint_sha256'] != sha256(checkpoint):
        raise ValueError('Best checkpoint hash mismatch.')

    trained = load_trusted_checkpoint(checkpoint, 'cpu')
    reference = load_trusted_checkpoint(REFERENCE, 'cpu')
    if trained['scaler'] != reference['scaler'] or trained['config'] != protocol['config']:
        raise ValueError('Checkpoint scaler or configuration changed.')
    if trained['best_epoch'] != result['best_epoch'] or trained['epochs_completed'] != 500:
        raise ValueError('Checkpoint epoch differs from metrics.')
    for part, count in protocol['split_counts'].items():
        ids = [row['source_id'] for row in trained['split'][part + '_sources']]
        if len(ids) != count or ids != [row['source_id'] for row in reference['split'][part + '_sources']]:
            raise ValueError(f'{part} checkpoint split differs from the reference.')

    rows = read_csv(OUT / 'predictions.csv')
    baseline = read_csv(BASELINE_PREDICTIONS)
    if len(rows) != 1208 or len({row['source_id'] for row in rows}) != 1208:
        raise ValueError('Missing or duplicate validation/test predictions.')
    checked = {}
    for part in ('val', 'test'):
        observed = [row for row in rows if row['split'] == part]
        expected = [row for row in baseline if row['split'] == part]
        checkpoint_ids = [row['source_id'] for row in trained['split'][part + '_sources']]
        reference_ids = [row['source_id'] for row in reference['split'][part + '_sources']]
        ids = [row['source_id'] for row in observed]
        if len(observed) != 604 or ids != [row['source_id'] for row in expected]:
            raise ValueError(f'{part} source ID count/order differs from HERA baseline.')
        if ids != checkpoint_ids or ids != reference_ids:
            raise ValueError(f'{part} checkpoint source IDs differ from predictions.')
        for current, prior in zip(observed, expected):
            if not math.isfinite(float(current['target'])) or abs(float(current['target']) - float(prior['target'])) > 1e-5:
                raise ValueError(f'{part} target differs from HERA baseline.')
        checked[part] = score(observed)
        saved = result['metrics'][part]
        if saved['n'] != checked[part]['n'] or any(
            abs(saved[key] - checked[part][key]) > 1e-8
            for key in ('mae_ev', 'rmse_ev')
        ):
            raise ValueError(f'{part} saved scores differ from per-sample recomputation.')
    if abs(checked['val']['mae_ev'] - trained['best_val_mae']) > 5e-5:
        raise ValueError('Validation MAE differs from checkpoint selection record.')
    if abs(checked['test']['mae_ev'] - trained['test_mae']) > 5e-5:
        raise ValueError('Test MAE differs from checkpoint selection record.')
    history = read_csv(OUT / 'seed123_history.csv')
    if len([row for row in history if row['epoch'].isdigit()]) != 500:
        raise ValueError('Training history does not cover 500 epochs.')
    return {'metrics': checked, 'best_epoch': result['best_epoch'],
            'epochs_completed': 500, 'checkpoint_sha256': result['checkpoint_sha256'],
            'reference_sha256': protocol['reference_sha256'],
            'split_counts': protocol['split_counts'], 'identical_scaler': True,
            'identical_source_ids_and_targets': True}


def update_main_report(audit_result, error=None):
    if not MAIN_REPORT.exists():
        return
    content = MAIN_REPORT.read_text(encoding='utf-8')
    now = datetime.now().astimezone().strftime('%Y-%m-%d %H:%M')
    if audit_result:
        test = audit_result['metrics']['test']
        status = f'已完成 500 轮；验证 MAE 最优权重在第 {audit_result["best_epoch"]} 轮；604 条测试预测已核验'
        row = f'| HyperALIGNN-WAS：paired | 沿用 hetero 的 1818/604/604 固定划分 | {test["mae_ev"]:.6f} | **{test["rmse_ev"]:.6f}** | {status} |'
        headline = f'**Hypergraph paired 已完成，Test RMSE {test["rmse_ev"]:.6f} eV；原版 ALIGNN raw 仍未完成。**'
    else:
        row = '| HyperALIGNN-WAS：paired | 沿用 hetero 的 1818/604/604 固定划分 | — | — | 运行或核验失败；详情见 Hypergraph 专项报告 |'
        headline = '**Hypergraph paired 运行或核验失败；原版 ALIGNN raw 仍未完成。**'
    content, matches = re.subn(r'^\| HyperALIGNN-WAS：paired \|.*$', row, content, count=1, flags=re.MULTILINE)
    if matches != 1:
        raise ValueError('Cannot locate Hypergraph result row in main report.')
    content = re.sub(r'^更新：.*$', f'更新：{now}（香港时间）。{headline}', content, count=1, flags=re.MULTILINE)
    content = content.replace('**目前不能判断原版模型能否达到 0.15 eV，也不能判断 Hypergraph 是否优于 hetero。**',
                              '**原版 ALIGNN 能否达到论文的 0.15 eV 仍待完整测试；Hypergraph 的同划分比较见上表与专项报告。**')
    content = content.replace('Hypergraph 正在训练，尚无完整训练产生的 RMSE。',
                              'Hypergraph 的完整结果见上表及专项报告。')
    content = content.replace('raw、paired 和 Hypergraph 的正式 `metrics.json` 尚不存在；Hypergraph 已开始完整训练。',
                              '原版 ALIGNN raw 与 paired 的正式 `metrics.json` 尚不存在；Hypergraph 已另行完成或核验失败，详见上表。')
    write_text(MAIN_REPORT, content)


def variant_report(audit_result, error=None):
    """Publish a distinct variant without replacing the completed radius3 row."""
    protocol = json.loads((OUT / 'protocol.json').read_text(encoding='utf-8'))
    model = protocol['config']['model']
    variant = f'{model["hypergraph_updates"]} r{model["hypergraph_radius"]:g}'
    label = f'HyperALIGNN-WAS：{variant}'
    now = datetime.now().astimezone().strftime('%Y-%m-%d %H:%M')
    prefix = OUT.as_posix()
    content = f'# Native Hypergraph {variant} 测试结果\n\n更新：{now}（香港时间）。\n\n'
    if audit_result:
        val, test = (audit_result['metrics'][part] for part in ('val', 'test'))
        original = json.loads((DEFAULT_OUT / 'metrics.json').read_text(encoding='utf-8'))['metrics']
        hetero_rows = [dict(prediction=float(row['hetero_was_no_dd_prediction']), target=float(row['target']))
                      for row in read_csv(BASELINE_PREDICTIONS) if row['split'] == 'test']
        hetero = score(hetero_rows)
        content += (
            f'**500 轮训练完成；按验证 MAE 选择第 {audit_result["best_epoch"]} 轮权重。**\n\n'
            '| 模型 | Test n | Test MAE (eV) | Test RMSE (eV) |\n'
            '|---|---:|---:|---:|\n'
            f'| {variant} | {test["n"]} | {test["mae_ev"]:.6f} | **{test["rmse_ev"]:.6f}** |\n'
            f'| local_global r3 | {original["test"]["n"]} | {original["test"]["mae_ev"]:.6f} | {original["test"]["rmse_ev"]:.6f} |\n'
            f'| hetero_was_no_dd | {hetero["n"]} | {hetero["mae_ev"]:.6f} | {hetero["rmse_ev"]:.6f} |\n\n'
            f'本次完整验证集 n={val["n"]}，MAE={val["mae_ev"]:.6f} eV，RMSE={val["rmse_ev"]:.6f} eV。\n\n'
        )
        for name, baseline in [('原 local_global r3', original['test']), ('hetero', hetero)]:
            delta = test['rmse_ev'] - baseline['rmse_ev']
            direction = '降低' if delta < 0 else '升高' if delta > 0 else '持平'
            content += f'相对{name}，测试 RMSE {direction} {abs(delta):.6f} eV（{abs(delta)/baseline["rmse_ev"]*100:.2f}%）。\n\n'
        content += (
            '核验通过：训练/验证/测试划分与 hetero 相同；604 条验证及 604 条测试预测完整、无重复，'
            '标签与 source ID 顺序一致；逐样本重算 MAE/RMSE、权重 SHA256、目标 scaler 与配置均一致。\n\n'
            '这次同时调整 global 消息传递和局部半径，结果仅评价两项合并后的效果；'
            '一个 seed 不能证明跨随机种子的稳定优势，不能据此分别归因于 global 或 radius。\n\n'
            f'权重 SHA256：`{audit_result["checkpoint_sha256"]}`。\n\n'
        )
        row = f'| {label} | strict seed123；1818/604/604；与基线同划分 | {test["mae_ev"]:.6f} | **{test["rmse_ev"]:.6f}** | 500 轮完成；最优验证 MAE 第 {audit_result["best_epoch"]} 轮；完整测试已核验 |'
        headline = f'**{label} 已完成，Test RMSE {test["rmse_ev"]:.6f} eV。其他实验状态见下表。**'
    else:
        content += f'**运行或核验失败，尚无可报告的完整测试结果。**\n\n原因：{error}\n\n'
        row = f'| {label} | strict seed123；1818/604/604；与基线同划分 | — | — | 运行或核验失败；见专项报告 |'
        headline = f'**{label} 运行或核验失败。其他实验结果保留。**'
    content += ' · '.join(f'[{name}]({prefix}/{filename})' for name, filename in (
        ('实验配置', 'protocol.json'), ('运行状态', 'run_state.json'), ('逐样本预测', 'predictions.csv'),
        ('指标', 'metrics.json'), ('核验记录', 'audit.json'), ('训练日志', 'train.log'),
    )) + '\n'
    write_text(OUT / 'report.md', content)
    if MAIN_REPORT.exists():
        main = MAIN_REPORT.read_text(encoding='utf-8')
        pattern = '^' + re.escape(f'| {label} |') + r'.*$'
        main, matches = re.subn(pattern, lambda match: row, main, count=1, flags=re.MULTILINE)
        if matches == 0:
            main, matches = re.subn(r'^\| HyperALIGNN-WAS：paired \|.*$',
                                   lambda match: match[0] + '\n' + row, main, count=1, flags=re.MULTILINE)
        if matches != 1:
            raise ValueError('Cannot place variant row in the main report.')
        main = re.sub(r'^更新：.*$', f'更新：{now}（香港时间）。{headline}', main, count=1, flags=re.MULTILINE)
        if model['hypergraph_updates'] == 'local' and model['hypergraph_radius'] == 6:
            pending = '新 local-only、6 Å 的效果尚待完整测试；'
            conclusion = (f'新 local-only、6 Å 的完整测试 MAE {test["mae_ev"]:.6f}、'
                          f'RMSE {test["rmse_ev"]:.6f} eV；' if audit_result
                          else '新 local-only、6 Å 运行或核验失败，尚无可信的完整测试指标；')
            main = main.replace(pending, conclusion)
            main = re.sub(
                r'^- local-only、6 Å 新实验使用独立结果目录；.*$',
                '- local-only、6 Å 使用独立结果目录，已完成训练和完整测试核验，详见专项报告。' if audit_result
                else '- local-only、6 Å 使用独立结果目录，运行或核验失败，详见专项报告。',
                main, count=1, flags=re.MULTILINE,
            )
        if prefix + '/report.md' not in main:
            main += f'\n[{label} 专项结果]({prefix}/report.md)\n'
        write_text(MAIN_REPORT, main)


def report(audit_result, error=None):
    if OUT != DEFAULT_OUT:
        return variant_report(audit_result, error)
    now = datetime.now().astimezone().strftime('%Y-%m-%d %H:%M')
    header = f'# Native HyperALIGNN-WAS 测试结果\n\n更新：{now}（香港时间）。\n\n'
    if audit_result:
        val, test = (audit_result['metrics'][part] for part in ('val', 'test'))
        baseline_mae, baseline_rmse = 0.10703324353852808, 0.37392596579410314
        delta = test['rmse_ev'] - baseline_rmse
        comparison = '低于' if delta < 0 else '高于' if delta > 0 else '等于'
        body = (
            f'**已完成全部 500 轮，按验证 MAE 选择第 {audit_result["best_epoch"]} 轮权重。**\n\n'
            '| 模型 | 验证样本数 | Validation MAE (eV) | Validation RMSE (eV) | 测试样本数 | Test MAE (eV) | Test RMSE (eV) |\n'
            '|---|---:|---:|---:|---:|---:|---:|\n'
            f'| HyperALIGNN-WAS paired | 604 | {val["mae_ev"]:.6f} | {val["rmse_ev"]:.6f} | 604 | {test["mae_ev"]:.6f} | **{test["rmse_ev"]:.6f}** |\n'
            f'| HERA hetero_was_no_dd | 604 | 0.077493 | 0.230631 | 604 | {baseline_mae:.6f} | **{baseline_rmse:.6f}** |\n\n'
            f'Hypergraph 的测试 RMSE 比 hetero {comparison} **{abs(delta):.6f} eV**（{abs(delta)/baseline_rmse*100:.2f}%）。两者使用相同的 604 条测试结构。'
            '训练骨干尺寸、原子特征、优化器、训练预算、划分和目标标准化对齐，消息传递架构不同；这一次比较不能单独归因于 readout。\n\n'
            '核验：604 条验证和 604 条测试逐样本预测完整、无重复，source ID 顺序及标签与 hetero 一致；'
            '重新计算的 MAE/RMSE 与保存结果一致，checkpoint SHA256 和目标 scaler 与参考协议一致。'
            '这是一组 seed123 实验，不能据此断言跨种子稳定排名。\n\n'
            f'权重 SHA256：`{audit_result["checkpoint_sha256"]}`。\n\n'
            '[逐样本预测](C:/Users/User/Desktop/HERA/results/native_hypergraph_paired_20260929/predictions.csv) · '
            '[原始指标](C:/Users/User/Desktop/HERA/results/native_hypergraph_paired_20260929/metrics.json) · '
            '[核验记录](C:/Users/User/Desktop/HERA/results/native_hypergraph_paired_20260929/audit.json) · '
            '[训练日志](C:/Users/User/Desktop/HERA/results/native_hypergraph_paired_20260929/train.log)\n'
        )
    else:
        body = ('**训练或结果核验失败，尚无可信的测试 RMSE。**\n\n'
                f'原因：{error}\n\n'
                '[训练日志](C:/Users/User/Desktop/HERA/results/native_hypergraph_paired_20260929/train.log) · '
                '[运行状态](C:/Users/User/Desktop/HERA/results/native_hypergraph_paired_20260929/run_state.json)\n')
    write_text(OUT / 'report.md', header + body)
    update_main_report(audit_result, error)


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    OUT = args.output.resolve()
    try:
        checked = audit()
        write_text(OUT / 'audit.json', json.dumps(checked, indent=2, ensure_ascii=False) + '\n')
        report(checked)
        print(json.dumps(checked, ensure_ascii=False))
    except Exception as exc:
        report(None, f'{type(exc).__name__}: {exc}')
        raise


if __name__ == '__main__':
    main()
