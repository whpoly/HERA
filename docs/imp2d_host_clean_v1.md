# IMP2D：按母体排除整组能量偏移的版本

2026-09-25。版本名 `imp2d_host_clean_v1`，加载参数 `--imp2d-host-filter reviewed_v1`。
使用原始 ASE db 的结构、DFE 和计算元数据；加载时自动完成筛选，无需提前处理 CIF/CSV。

后续已按用户选择加入固定 −10～10 eV 范围，保留 10,475 条。
当前运行命令见 [imp2d_10ev_v1](imp2d_10ev_v1.md)；本页保留仅排除母体的旧版本说明。

## 排除哪些材料

排除 **Ti2CO2，DB 名称 `CO2Ti2`** 的全部记录。
此前输入完整、通过现有筛查的该母体 231 条记录，DFE 全部位于
−176.47024～−156.16324 eV，中位数为 −170.85014 eV。
其他 43 种母体的中位数为 0.195～3.978 eV，未发现另一种类似的整组偏移。

这是基于本次来源审查冻结的母体排除名单。母体参考计算尚待核实，
**不把偏移原因当作已证实，也不把所有高能样本认定为错误数据。**
固定名单在划分前定义，不根据模型测试误差或每次训练结果改变。
原始 db、旧清单、已有模型结果均保留。native、semi 的预处理不受此选项影响。

## 新数据集

| 项目 | 数量 |
|---|---:|
| 上一版通过现有筛查且输入完整 | 10,946 |
| 本次额外排除 Ti2CO2 | 231 |
| **新版本保留** | **10,715** |
| 保留母体种类 | 43 |
| seed 123：train / val / test | 6,470 / 2,150 / 2,095 |

原始库中 Ti2CO2 共 335 条；其中部分原先已因其他原因排除，
所以相对于原 10,946 条队列，本次减少的是 231 条，不能重复计数。
物理检查、缺陷原子标记不足、母体范围限制分别记录原因。

仍保留的 DFE 范围为 **−10.20108～153.22290 eV**，5%～95% 分位为 **−1.42805～7.55654 eV**。
另有 **240 条**在旧 `−10 < E_form < 10 eV` 区间之外，仅列入复查表，不在本版本单独删除。
例如 Te2W 和 S2Ti 仍有少数极大正值；它们的母体整体分布没有 Ti2CO2 的偏移特征。
本版本没有按原子数平均 DFE、改写标签或应用额外能量截断。

### 高能尾部补充复查

本版本只完成母体范围限制，**没有完成高能样本的可靠性审查**。
仍有 47 条不小于 20 eV、9 条不小于 50 eV、3 条不小于 100 eV。

| 样本 | DFE / eV | 原始 converged | conv2 / eV |
|---|---:|---|---:|
| Te2W_Sc_ads2 | 153.22290 | false | −0.00206761 |
| S2Ti_Mn_int0 | 119.77252 | true | −0.00465398 |
| MoTe2_As_int2 | 119.70526 | true | −0.0423592 |

这些记录通过当前最后一步能量变化判据，不能据此证明参考能和电子态正确。
仅要求原始 converged 为 true 也无法排除后两条。
Te2W 的 277 条保留记录中位数为 3.185 eV，95% 分位为 7.777 eV，
因此正向尾部需要样本级复查；不能只凭最高值认定整个母体都无效。
另有 Te2W_Hg_ads2 的两阶段总能量相差 +60.170 eV；
这是复查线索，但两阶段计算设置不同，不能仅凭差值判定错误原因。

原始字段和统计保存在 `results/imp2d_positive_tail_review.json`。
目前未应用额外排除。若明确采用旧 benchmark 的固定开区间
`−10 < E_form < 10 eV`，会再排除 240 条并保留 10,475 条；
应另设版本并注明能量范围，不能把这个边界解释为物理真假判据。

结构分组沿用原始 DB 路径的 0.001 Å 等价判据，同组不会跨 train/val/test。
先构建与旧版本相同的组，再应用母体限制；剩余样本保持原来的集合归属。
新旧 MAE 对应不同样本范围，应标注版本，不能把差异全部归因于模型改进。

## 直接运行：只跑 IMP2D

将更新后的代码同步到服务器后，在 `/home/wuhao`（HERA 的父目录）运行：

```bash
python -m HERA.main --model alignn --dataset imp2d --mode full attention attention_was hetero hetero_was was_x definet definet_was --imp2d-preprocessing reference_v1 --imp2d-source db --imp2d-quality-filter physical --imp2d-host-filter reviewed_v1 --alignn-hetero-dd drop --r 0 --alignn-hetero-feature-norm layernorm --alignn-hetero-node-norm layernorm --alignn-hetero-relations shared_residual --alignn-hetero-adapter-rank 8 --alignn-hetero-pooling defect_energy_mean --seed 123 --epochs 500 --device cuda:0 --atom-init HERA/atom_init.json --run-dir HERA/logs/imp2d_host_clean_v1 --compact-logs --resume
```

自动读取 `dataset/imp2d/imp2d.db`；缺失时按已有规则下载并验证固定的官方版本。
不需要 CIF 或 CSV。使用默认 batch size，hetero 为 no_dd，不含 hypergraph。
`--resume` 跳过已验证完成的结果；未完成的 split 从头训练，不恢复到中断 epoch。
新版本使用独立根目录，避免混入旧队列检查点；目录结构仍为 `alignn/dataset/mode`。

如需同时跑 native、semi，保留它们原来的筛选，使用：

```bash
python -m HERA.main --model alignn --dataset native semi imp2d --mode full full_x attention attention_was hetero hetero_was was_x definet definet_was --native-preprocessing reference_v1 --semi-preprocessing reference_v1 --imp2d-preprocessing reference_v1 --native-outlier-filter strict --semi-quality-filter physical --semi-source-policy legacy_available --imp2d-source db --imp2d-quality-filter physical --imp2d-host-filter reviewed_v1 --alignn-hetero-dd drop --r 0 --alignn-hetero-feature-norm layernorm --alignn-hetero-node-norm layernorm --alignn-hetero-relations shared_residual --alignn-hetero-adapter-rank 8 --alignn-hetero-pooling defect_energy_mean --seed 123 --epochs 500 --device cuda:0 --atom-init HERA/atom_init.json --run-dir HERA/logs/imp2d_host_clean_v1 --compact-logs --resume
```

no_aa_dd 消融可在同一根目录将 mode 改为 `hetero hetero_was`，再加 `--alignn-hetero-aa drop`。

## 输出

均位于 `logs/imp2d_host_clean_v1/`：

- `imp2d_filter_manifest.json`：原始 DB 身份、筛选版本、标签、结构组、固定划分。
- `imp2d_filter_seed123_kept.csv`：新数据集的完整样本清单及集合归属。
- `imp2d_host_distribution.csv`：原 44 种母体的分布统计与排除状态。
- `imp2d_host_excluded.csv`：本次额外排除的 231 条记录。
- `imp2d_remaining_energy_review.csv`：仍保留的 240 条超出旧能量区间的复查记录。
- `imp2d_host_filter_summary.json`：数量、分布、划分与筛选身份。

相关 34 项测试通过，覆盖按母体完整排除、其他母体高能样本保留、
随机和五折划分的归属保持、缓存身份、配置冲突保护及 CLI 两种模型模式共享数据。
本地实际处理验证记录在 `results/imp2d_host_clean_v1_verification.json`；没有启动训练。
