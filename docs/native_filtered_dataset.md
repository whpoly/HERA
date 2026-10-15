# Native 先筛选、后划分与重训

2026-10-13，v4。当前请求对应 `--suite filtered_dataset --max-energy-ev 15 --exclude-poscar0`：
在 strict + MAD、形成能 ≤15 eV 的数据中，排除所有文件名包含精确标记 `-POSCAR0-`
的样本，**先完成总体筛选，再按 seed 重新划分 train、valid、test**。
只准备服务器重训；本地不启动 benchmark。上一轮含 POSCAR0 的 v3 结果已经完成，
保存在 `logs/native_filter_then_split_le15_v3/`；本轮使用新的目录和清单。

## 冻结筛选规则

先检查完整原始数据集，使用全体合格样本拟合 strict 中的统计阈值并执行筛选。
然后在全体 strict 保留样本上，按同一缺陷构型系列拟合一次 MAD 上尾阈值。
从文件名去掉 POSCAR 序号定义系列；每个系列至少需要 4 条样本：

```text
m = median(E_strict_population)
sigma_MAD = 1.4826 * median(abs(E_strict_population - m))
upper = m + max(6 * sigma_MAD, 1 eV)
```

从总体中排除 `E > upper`，保留边界；不足 4 条的系列不执行新增 MAD 排除。
接着应用 `E <= 15 eV`，再排除 `-POSCAR0-`。15 eV 边界保留，不添加下限。
POSCAR1、POSCAR10、POSCAR100 等不受文件名规则影响；剩余帧不一定是完全弛豫结构。
新增排除步骤不改变 strict 和 MAD 的统计参照，不重拟合或迭代收紧阈值。
此时没有 train、valid、test，也没有使用任何 split seed。将保留清单按源表顺序
冻结后，使用现有随机 splitter 重新按 60%/20%/20% 划分；seed 只影响划分归属。
规则本身不使用模型预测或残差逐条选点；此次排除 POSCAR0 的实验动机来自之前的
误差分析，因此仍属于探索性实验。
被排除的样本可能包含有效的高能结构，这项统计筛选不能证明其 DFT 标签错误。

真实数据的总体筛选结果如下：

| 阶段 | 样本数 |
|---|---:|
| 原始数据 | 3070 |
| 全体样本拟合 strict 后 | 3021 |
| 再应用全体 strict 样本拟合的同系列 MAD | 2940 |
| 再排除形成能 >15 eV | 2811 |
| 再排除 221 条 POSCAR0 | 2590 |
| 从原始数据总计排除 | 480 |

随后 seed123 重新划分为 **train 1554 / valid 518 / test 518**。
上一轮含 POSCAR0 的划分为 1686 / 562 / 563；本轮不会保留旧划分后直接删点。
strict 和 MAD 仍使用全体样本的筛选流程，不复用历史 train-only 拟合阈值。

阈值及逐条保留/排除原因写入 manifest 和审计表；两个损失目录保存完全相同的
manifest。独立的 `native_population_filter.json` 保存不含 split 的总体筛选结果，
两个模型和两种损失共用该结果。`filtered_id_prop_A_rich.csv` 保存筛选后的源 ID 和标签。
本地清单准备目录为 `results/native_filter_then_split_le15_no_poscar0_20261013_prepared/`。
保留原始 CSV 和 CIF，通过清单排除训练和评估样本，并保存逐条排除原因。

## 四次服务器训练

Hetero-WAS no_dd 和 WAS_X 各训练 MSE、Huber 两组，共四次训练。Huber 转折点为
实际 1 eV，采用标准化空间的 `2 * Huber`；具体定义见
[先前的损失说明](native_outlier_control.md)。四组共用同一数据清单和训练目标 scaler。

固定 hidden64、原有 readout、batch16、500 epochs；用**筛选后的 valid MAE**
选择 checkpoint，在**筛选后的 test** 上报告 MAE 和 RMSE。汇总必须核验 checkpoint
记录的三个划分的 ID、顺序和清单身份，并验证保存的 MAE 可由重新推理重现。
旧的 train-only 筛选或含 POSCAR0 的 checkpoint 不能用于该协议。

同步 `main.py`、`training/losses.py`、`training/trainer.py` 和
`scripts/run_native_filter_benchmark.py`、`data/native_filter.py` 和新增的
`data/native_population.py`，在 **HERA 父目录**运行：

```bash
python -m HERA.scripts.run_native_filter_benchmark --suite filtered_dataset --max-energy-ev 15 --exclude-poscar0 --seed 123 --epochs 500 --device cuda:0 --run-dir HERA/logs/native_filter_then_split_le15_no_poscar0_v4
```

命令自动先筛选、再划分、顺序训练四组模型并汇总。输出包括 `filter_protocol.json`、
`native_population_filter.json`、`population_filter_review.csv`、各组 manifest、逐结构预测及
`filter_comparison.csv` / `filter_comparison.md` / `filter_evaluation.json`。

追加 `--dry-run` 只显示计划，`--prepare-only` 只准备清单，`--evaluate-only` 在
训练完成后重算评估。重复执行可跳过兼容的已完成训练；未完成训练依现有 resume
机制从头重启。不同协议必须使用不同运行目录。

现在 test 为重新随机划分的 518 条，上一轮 ≤15 eV test 为 563 条。分数变化同时包含评价人群变化，
不能全部归因于模型学习改善。模型与损失应在这份相同的新清单上比较；尚不能承诺
Hetero-WAS 的 MAE 低于 WAS_X。总体筛选使用标签分布来定义新数据集，评价描述的是
这个经筛选的数据范围；模型和目标 scaler 仍只拟合新训练集。此前已查看过旧测试
误差，这仍是探索性对照。

## 与论文筛选的关系

[论文第 5.2 节](https://arxiv.org/html/2309.06423v2)描述对合并数据集删除少量明显
离群点后重训；其 Table 3 的 neutral A-rich test RMSE 为 0.32 eV。
Native-only 的 0.15 eV 来自 Table 2，是另一项实验。在已核对的公开材料中，未找到
可精确重建相应训练/验证/测试数据的完整删除 ID、数值阈值及最终划分清单。
论文有意包含未优化和中间弛豫结构，因此不能仅按早期 POSCAR 序号认定数据无效。

[作者 notebook](https://github.com/msehabibur/defect_GNN_gen_1/blob/main/Codes/Defect_GNN_Notebook.ipynb)
有合并数据预测结果的 `target < 15`，以及同时限制 target 和 prediction 的后处理。
这些单元不能证实 native Table 2 的训练前筛选方法；按各模型预测删除测试点还会
导致两模型评价样本不一致。2026-10-12 重新下载的公开 notebook 与前次审计哈希
一致，核对记录保存在旧准备目录
`results/native_filtered_dataset_20261012_prepared/paper_notebook_recheck.json`。

因此，本实现是明确、可审计的全数据集统计筛选对照，不能标为论文原筛选的严格
复现。严格复现还需要作者的最终样本及划分清单，或等价的完整筛选代码和参数。
MAD 和 Huber 都是本项目的实验设置，不是从论文中确认的原方法。

## 验证范围

相关测试涵盖 native manifest、统计筛选、稳健损失及日志协议。新增检查覆盖先冻结
总体再传给 splitter、seed 只改变划分、严格保留阈值边界、完整排除审计、划分后
禁止再删样本、旧协议拒绝复用、变更被排除源文件时拒绝缓存及四组模型相同清单。
真实数据准备核验了源表与全部结构文件哈希，并确认共用筛选代码的重构未改变旧版
train-fitted strict 的真实结果。没有在本地启动 benchmark 训练。
