"""Test-only evaluation of frozen v4 native models, plus audited Hypergraph runs."""
import copy
import csv
from datetime import datetime
import gc
import json
import time
import warnings

import numpy as np
import torch
from pymatgen.core import Structure
from torch_geometric.data import Batch

from . import finalize_native_hypergraph_paired as audit
from .compare_native_checkpoint_errors import identity
from ..data.datasets import init_elem_embedding, tag_structure_source, representation_for_mode
from ..data.structure_utils import convert_to_sparse_native
from ..main import set_seed
from ..training.trainer import MEGNetTrainer, load_trusted_checkpoint


ROOT = audit.ROOT
OUT = ROOT / 'results/native_all_models_rmse_20260930'
CHECKPOINTS = ROOT / 'logs/alignn_v4/alignn/native'
DATA = ROOT.parent / 'dataset/Dataset_1/Dataset_1/A_rich/Neutral'
NEW_MODELS = ('full', 'attention', 'full_x', 'attention_was', 'definet', 'definet_was', 'hetero_no_dd')


def save_json(path, value):
    def scalar(item):
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(f'Cannot serialize {type(item).__name__}')
    audit.write_text(path, json.dumps(value, indent=2, ensure_ascii=False,
                                     allow_nan=False, default=scalar) + '\n')


def publish(summary):
    save_json(OUT / 'summary.json', summary)
    ranked = sorted(summary['models'].items(), key=lambda item: item[1]['rmse_ev'])
    lines = ['# Native 全部同协议模型的测试 RMSE 排名', '',
             f'更新：{datetime.now().astimezone().isoformat()}；状态：{summary["status"]}。', '',
             '同一 strict seed123 划分：1818 train / 604 validation / 604 test。'
             '所有权重均已训练 500 轮、按验证 MAE 选择；本次只补做固定权重测试推理，使用完整测试集，无删样或重新训练。', '',
             '| RMSE 排名 | 模型 | Test n | Test RMSE (eV) | Test MAE (eV) | 最优轮次 |',
             '|---:|---|---:|---:|---:|---:|']
    for rank, (name, values) in enumerate(ranked, 1):
        lines.append(f'| {rank} | {name} | {values["n"]} | {values["rmse_ev"]:.6f} | {values["mae_ev"]:.6f} | {values["best_epoch"]} |')
    if summary['status'] != 'complete':
        lines += ['', '**正在补齐模型；上表仅为已核验部分，尚不是完整排名。**']
    else:
        lines += ['', f'完整排名覆盖 9 个 alignn_v4 基线和 2 个 Hypergraph 变体。Test RMSE 最低为 `{ranked[0][0]}`。',
                  '', '核验：每个模型的 train/val/test source ID 顺序、scaler、strict filter 与参考一致；'
                  '本地标签和 CIF 哈希与已核验基线逐样本匹配；权重严格加载；重算测试 MAE 与各自 checkpoint 保存值相差不超过 0.00005 eV。'
                  'MAE 与 RMSE 都从每个模型全部 604 条测试预测重新计算。', '',
                  '`full` 与 `attention` 是仓库中的两个不同模式，表中沿用原模式名。'
                  '特征和架构在不同模式间有差异，不能把排名单独解释为 readout 的作用。'
                  '这是固定 seed123 的结果，不能视作跨种子稳定排名；不混入旧版未筛选的 614 条测试集结果。']
    lines += ['', f'[指标及权重/数据哈希]({(OUT / "summary.json").as_posix()}) · '
              f'[重算脚本]({(ROOT / "scripts/audit_native_all_models_rmse.py").as_posix()})', '']
    audit.write_text(OUT / 'report.md', '\n'.join(lines))


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    if (OUT / 'summary.json').exists():
        raise FileExistsError('Existing audit output; refusing to overwrite it.')
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable.')
    init_elem_embedding(ROOT / 'atom_init.json')
    reference = load_trusted_checkpoint(audit.REFERENCE, 'cpu')
    ids = {p: [r['source_id'] for r in reference['split'][p + '_sources']] for p in ('train', 'val', 'test')}
    for part in ids:
        assert len(ids[part]) == len(set(ids[part]))
    assert not (set(ids['train']) & set(ids['val']) or set(ids['train']) & set(ids['test']) or set(ids['val']) & set(ids['test']))
    labels = {}
    with (DATA / 'id_prop_A_rich.csv').open(newline='', encoding='utf-8-sig') as f:
        for sid, target, *_ in csv.reader(f):
            assert sid not in labels
            labels[sid] = float(np.float32(target))
    cached_dir = ROOT / 'results/native_hetero_vs_was_x_20260928'
    cached = [r for r in audit.read_csv(cached_dir / 'paired_predictions.csv') if r['split'] == 'test']
    assert [r['source_id'] for r in cached] == ids['test']
    source_provenance = json.loads((cached_dir / 'summary.json').read_text(encoding='utf-8'))['provenance']['models']
    structures, structure_hashes = {}, {}
    for row in cached:
        sid = row['source_id']
        assert abs(labels[sid] - float(row['target'])) < 1e-5
        structure_hashes[sid] = audit.sha256(DATA / sid)
        assert structure_hashes[sid] == row['structure_sha256']
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            structures[sid] = Structure.from_file(DATA / sid)
    summary = dict(status='running', training_performed=False, inference_models=list(NEW_MODELS),
                   seed=123, split_counts={p: len(v) for p, v in ids.items()},
                   checkpoint_selection='validation MAE', target_scaler=reference['scaler'],
                   target_csv_sha256=audit.sha256(DATA / 'id_prop_A_rich.csv'),
                   atom_init_sha256=audit.sha256(ROOT / 'atom_init.json'),
                   structure_sha256=structure_hashes, runtime=dict(torch=torch.__version__, device='cuda:0'), models={})

    def validate_checkpoint(checkpoint):
        assert checkpoint['dataset_name'] == 'native' and checkpoint['epochs_completed'] == 500
        assert checkpoint['split']['seed'] == 123
        assert checkpoint['config']['native_preprocessing'] == 'reference_v1'
        assert checkpoint['config']['native_data_filter'] == reference['config']['native_data_filter']
        assert checkpoint['scaler'] == reference['scaler']
        assert all([r['source_id'] for r in checkpoint['split'][p + '_sources']] == ids[p] for p in ids)

    def register(name, path, checkpoint, rows, prediction_path, origin):
        validate_checkpoint(checkpoint)
        assert [r['source_id'] for r in rows] == ids['test']
        assert len(rows) == 604
        assert all(abs(float(r['target']) - labels[r['source_id']]) < 1e-5 for r in rows)
        metrics = audit.score(rows)
        discrepancy = metrics['mae_ev'] - checkpoint['test_mae']
        if abs(discrepancy) > 5e-5:
            raise ValueError(f'{name}: test MAE mismatch {discrepancy:+.9f} eV')
        summary['models'][name] = dict(**metrics, best_epoch=checkpoint['best_epoch'],
                                      checkpoint_path=str(path), checkpoint_sha256=audit.sha256(path),
                                      saved_test_mae_ev=checkpoint['test_mae'], reproduction_difference_ev=discrepancy,
                                      predictions_path=str(prediction_path), predictions_sha256=audit.sha256(prediction_path),
                                      prediction_origin=origin, config=checkpoint['config'])
        publish(summary)
        print(f'{name}: n=604 MAE={metrics["mae_ev"]:.9f}, RMSE={metrics["rmse_ev"]:.9f}, saved-MAE delta={discrepancy:+.3g}', flush=True)

    try:
        for name in ('was_x', 'hetero_was_no_dd'):
            path = CHECKPOINTS / name / 'seed123_best_checkpoint.pth'
            checkpoint = load_trusted_checkpoint(path, 'cpu')
            assert audit.sha256(path) == source_provenance[name]['checkpoint_sha256']
            rows = [dict(source_id=r['source_id'], target=r['target'], prediction=r[name + '_prediction']) for r in cached]
            register(name, path, checkpoint, rows, cached_dir / 'paired_predictions.csv', 'previous audited inference')
        for name, directory in (
            ('hypergraph_local_global_r3', ROOT / 'results/native_hypergraph_paired_20260929'),
            ('hypergraph_local_r6', ROOT / 'results/native_hypergraph_local_r6_seed123_prepared_20260930'),
        ):
            audit.OUT = directory
            audit.audit()
            path = directory / 'seed123_best_checkpoint.pth'
            rows = [r for r in audit.read_csv(directory / 'predictions.csv') if r['split'] == 'test']
            register(name, path, load_trusted_checkpoint(path, 'cpu'), rows, directory / 'predictions.csv', 'previous audited inference')
        for name in NEW_MODELS:
            path = CHECKPOINTS / name / 'seed123_best_checkpoint.pth'
            checkpoint = load_trusted_checkpoint(path, 'cpu')
            validate_checkpoint(checkpoint)
            config = copy.deepcopy(checkpoint['config'])
            assert config['model']['test_batch_size'] == 1
            set_seed(123)
            trainer = MEGNetTrainer(config, 'cuda:0', seed=123)
            trainer.scaler.load_state_dict(checkpoint['scaler'])
            trainer.model.load_state_dict(checkpoint['model'], strict=True)
            trainer.model.eval().requires_grad_(False)
            representation = representation_for_mode(checkpoint['mode'])
            rows = []
            start = time.monotonic()
            for index, sid in enumerate(ids['test'], 1):
                raw = copy.deepcopy(structures[sid])
                tag_structure_source(raw, str(DATA / sid), sid)
                meta = identity(sid)
                structure = convert_to_sparse_native(
                    raw, 'vacancy' if meta['defect_type'] == 'vacancy' else 'others',
                    1, f'alignn_{representation}', None, True, False,
                    local_cutoff=config['model'].get('local_radius'), native_preprocessing='reference_v1')
                structure.y = torch.tensor(labels[sid], dtype=torch.float32)
                graph = trainer.converter.convert(structure)
                batch = Batch.from_data_list([graph]).to('cuda:0')
                with torch.no_grad(), trainer._autocast():
                    prediction = float(trainer.scaler.inverse_transform(trainer._forward(batch)).item())
                assert np.isfinite(prediction)
                rows.append(dict(model=name, source_id=sid, target=labels[sid], prediction=prediction,
                                 structure_sha256=structure_hashes[sid]))
                if index % 200 == 0:
                    print(f'{name}: {index}/604 test samples, {time.monotonic()-start:.1f}s', flush=True)
            destination = OUT / f'{name}_test_predictions.csv'
            with destination.open('x', encoding='utf-8', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            register(name, path, checkpoint, rows, destination, 'test-only inference this audit')
            del trainer, checkpoint, batch, graph
            gc.collect()
            torch.cuda.empty_cache()
        assert len(summary['models']) == 11
        summary['status'] = 'complete'
        publish(summary)
    except BaseException as exc:
        summary.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        publish(summary)
        raise


if __name__ == '__main__':
    main()
