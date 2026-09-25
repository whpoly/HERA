# IMP2D 固定 −10～10 eV 的 benchmark

2026-09-25。用户选择固定开区间 `−10 < DFE < 10 eV`。
通过 `--imp2d-energy-window -10 10` 在读取原始 DB 时自动筛选，无需预先生成 CSV/CIF。
这是研究范围限制，不把区间外记录一概判为物理错误。

## DB 与原 CIF 路径使用相同物理标准

两条路径共用 `data/impurity_quality.py` 的检查函数和阈值。
`--imp2d-source db` 必须同时启用 `--imp2d-quality-filter physical`；
能量窗口和母体筛选在这些物理规则之上进一步限制样本范围。

| 检查 | 保留条件 |
|---|---|
| 最终弛豫能量变化 | `abs(conv2) <= 0.05 eV`，包括正负两个方向 |
| 膨胀因子 | `0 < extension_factor <= 2` |
| 能量分项一致性 | `abs(eform - (en2 - hostenergy - dopant_chemical_potential)) <= 1e-4 eV` |
| 严重原子重叠 | 周期镜像也参与检查，所有原子对 `d / (r_cov_i + r_cov_j) >= 0.5` |
| 输入完整性 | 形成能与必要元数据有限、晶胞与坐标有效、单杂质组成正确 |

缺少必要元数据的记录排除。短键、弱结合和位点迁移继续仅作复查提示。
已经因 DFT 元数据失败的记录直接排除，无需继续几何扫描。
原始 `converged` 布尔值不能替代最终 `conv2` 检查。

2026-09-25 对当前冻结清单重新核对：

- 旧 CIF 起始范围内 **10,302 条的物理通过/排除结论完全一致**。
- 旧版保留的 **9,683 条全部仍在当前 DB 队列中**。
- 当前多出的 **792 条**均为旧 CSV 未包含、原始 `converged=False` 的记录，
  最终 `conv2` 与其余物理条件均通过，且缺陷原子身份可用。
- 当前全部 **10,475 条**重新检查能量元数据通过，冻结的几何检查结果也均符合上述阈值。

因此数量增加来自原始 DB 的来源范围，不是修改物理阈值。
DB 版本另使用等价结构分组划分，不能视为与旧 CIF 版本完全相同的 train/val/test benchmark。
核对结果：`results/imp2d_shared_physical_standard_verification.json`。
回归测试包含相同记录分别从 DB 与 CIF 进入筛选、阈值边界、缺失元数据、
严重重叠排除，以及短键/弱结合提示保留。

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
