# IMP2D 固定 −10～10 eV 的 benchmark

2026-09-25。用户选择固定开区间 `−10 < DFE < 10 eV`。
通过 `--imp2d-energy-window -10 10` 在读取原始 DB 时自动筛选，无需预先生成 CSV/CIF。
这是研究范围限制，不把区间外记录一概判为物理错误。

## 实测结果

- 保留原有物理检查和 Ti2CO2 母体排除（`--imp2d-host-filter reviewed_v1`）。
- 从 10,715 条再排除区间外的 240 条，保留 **10,475 条、43 种母体**。
- 实际 DFE 范围：**−9.86772122～9.98396280 eV**。
- seed 123：**train 6,328 / val 2,099 / test 2,048**。
- 两个端点恰好为 −10 或 10 eV 时也排除；不对标签做截断、平均或改写。
- 原始 DB、旧清单和旧结果保留，剩余样本的结构组及集合归属保持一致。

## 一条命令：IMP2D

同步更新后的 HERA 代码到服务器后，在 `/home/wuhao`（HERA 的父目录）执行：

```bash
python -m HERA.main --model alignn --dataset imp2d --mode full attention attention_was hetero hetero_was was_x definet definet_was --imp2d-preprocessing reference_v1 --imp2d-source db --imp2d-quality-filter physical --imp2d-host-filter reviewed_v1 --imp2d-energy-window -10 10 --alignn-hetero-dd drop --r 0 --alignn-hetero-feature-norm layernorm --alignn-hetero-node-norm layernorm --alignn-hetero-relations shared_residual --alignn-hetero-adapter-rank 8 --alignn-hetero-pooling defect_energy_mean --seed 123 --epochs 500 --device cuda:0 --atom-init HERA/atom_init.json --run-dir HERA/logs/imp2d_10ev_v1 --compact-logs --resume
```

默认 batch size；hetero/hetero_was 为 no_dd；不包含 hypergraph。
直接读取 `dataset/imp2d/imp2d.db`，缺失时自动下载并验证已有固定官方版本。
`--resume` 跳过已验证完成的结果，未完成的 split 从头训练。
使用新根目录，避免与不同样本范围的旧检查点混用，子目录仍为 `alignn/dataset/mode`。

## 同时跑三个数据集

以下命令只给 IMP2D 增加上述范围，native 和 semi 沿用原来的预处理：

```bash
python -m HERA.main --model alignn --dataset native semi imp2d --mode full full_x attention attention_was hetero hetero_was was_x definet definet_was --native-preprocessing reference_v1 --semi-preprocessing reference_v1 --imp2d-preprocessing reference_v1 --native-outlier-filter strict --semi-quality-filter physical --semi-source-policy legacy_available --imp2d-source db --imp2d-quality-filter physical --imp2d-host-filter reviewed_v1 --imp2d-energy-window -10 10 --alignn-hetero-dd drop --r 0 --alignn-hetero-feature-norm layernorm --alignn-hetero-node-norm layernorm --alignn-hetero-relations shared_residual --alignn-hetero-adapter-rank 8 --alignn-hetero-pooling defect_energy_mean --seed 123 --epochs 500 --device cuda:0 --atom-init HERA/atom_init.json --run-dir HERA/logs/imp2d_10ev_v1 --compact-logs --resume
```

no_aa_dd 消融可把 mode 改为 `hetero hetero_was` 并增加 `--alignn-hetero-aa drop`，
沿用相同 run-dir，保存到同级的关系消融目录。

## 验证和输出

36 项相关测试通过，包括区间边界、范围内样本保留、随机/五折划分归属保持、
缓存和旧配置冲突保护，以及 CLI 将同一筛选数据传给 attention_was 与 hetero_was。
完整 17,364 条原始 DB 已实测筛查，缓存复用通过，没有启动训练。

- `logs/imp2d_10ev_v1/imp2d_filter_manifest.json`：固定筛选规则与划分。
- `logs/imp2d_10ev_v1/imp2d_filter_seed123_kept.csv`：最终样本。
- `logs/imp2d_10ev_v1/imp2d_energy_window_excluded.csv`：本次额外排除的 240 条。
- `logs/imp2d_10ev_v1/imp2d_energy_window_summary.json`：范围与数量。
- `results/imp2d_10ev_v1_verification.json`：标签、集合归属、旧清单不变及缓存验证。
