# IMP2D interstitial-only benchmark

2026-10-08。通过 `--imp2d-defect-type interstitial` 在加载原始 ASE 数据库时，
仅保留弛豫后位于单层内部的杂质。此选项要求 `--imp2d-source db` 和
`--imp2d-quality-filter physical`，省略选项仍为原来的混合队列。

## 固定研究范围

- 沿用当前物理质量检查、可靠缺陷原子标记、Ti2CO2 母体排除，以及严格能量区间 `-10 < DFE < 10 eV`。
- 额外要求数据库的弛豫后 `depth` 有限且 `abs(depth) < 1`。边界 `depth = ±1` 属于吸附结构；缺失或非有限 depth 单列排除原因。
- 不根据初始文件名的 `ads/int` 或模型预测误差筛选。初始 ads 弛豫到内部的样本保留；初始 int 弛豫到表面的样本排除。
- 定义来自 [原论文 Section 2.3](https://arxiv.org/html/2207.05353v2#S2.SS3)。这是 benchmark 范围限制，不是将吸附结构认定为错误数据。
- 先按原流程构建等价结构组，再应用范围限制，保留原 seed 对应的 train/val/test 归属。同组不跨集合，各模型使用相同清单。
- 使用独立目录 `logs/imp2d_interstitial_v1`。筛选政策与完整清单进入模型配置身份；混用旧清单或 checkpoint 会报错。

## 实测验证

从官方 17,364 条原始记录重新完成物理检查和结构分组，在原 10,475 条混合队列中
保留 **3,515 条 interstitial、43 种母体**，排除 6,960 条 adsorbate。
其中保留 412 条初始 ads 弛豫为 interstitial 的记录，排除 375 条初始 int 弛豫为 adsorbate 的记录。
seed 123 的最终划分为 **train 2,132 / val 690 / test 693**。

45 项相关回归测试通过，覆盖两侧深度边界、缺失深度、位点迁移、随机和五折分组划分、
所有结构表示共用队列、CLI 传参、旧政策缓存冲突和冻结清单校验。
真实数据库的 3,515 个结构全部加载验证通过；标签、原结构组及保留样本的集合归属不变，
缓存复用身份一致，原混合清单的文件哈希不变。没有启动模型训练。
完整核对记录保存在 `results/imp2d_interstitial_v1_verification.json`。

## 运行

同步代码后，在 HERA 的父目录（服务器为 `/home/wuhao`）运行：

```bash
python -m HERA.main --model alignn --dataset imp2d --mode full attention attention_was hetero hetero_was was_x definet definet_was --imp2d-preprocessing reference_v1 --imp2d-source db --imp2d-quality-filter physical --imp2d-host-filter reviewed_v1 --imp2d-energy-window -10 10 --imp2d-defect-type interstitial --alignn-hetero-dd drop --r 0 --alignn-hetero-feature-norm layernorm --alignn-hetero-node-norm layernorm --alignn-hetero-relations shared_residual --alignn-hetero-adapter-rank 8 --alignn-hetero-pooling defect_energy_mean --seed 123 --epochs 500 --device cuda:0 --atom-init HERA/atom_init.json --run-dir HERA/logs/imp2d_interstitial_v1 --compact-logs --resume
```

保持原 ALIGNN benchmark 的八种模式、默认 batch size、hetero no_dd 和 seed 123。
WAS 仍将增加原子的原占位设为 0。形成能标签不改写。
`--resume` 复用同配置的已完成结果，未完成 split 从头训练。

## 审计输出

- `imp2d_filter_manifest.json`：来源、物理检查、范围排除原因、结构分组、冻结划分。
- `imp2d_filter_seed123_kept.csv` / `imp2d_filter_seed123_removed.csv`：保留和排除样本。
- `imp2d_defect_type_summary.json`：类型筛选前后数量、初始位置迁移统计、最终划分数量。
- `imp2d_defect_type_excluded.csv`：通过前置筛选、仅因 interstitial 范围限制而排除的样本。
- `imp2d_energy_window_summary.json` 的 `retained_rows` 为类型筛选前的能量窗口队列数量，
  `benchmark_retained_rows` 和 `splits` 为最终 benchmark 数量。

模型结果应标注 `IMP2D-interstitial`，与原混合队列的总体 MAE 属于不同评估范围。
