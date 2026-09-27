**Native OOD；semi / imp2d 保留原划分**

同步当前 HERA 代码后，在 HERA 的父目录执行。新增参数只影响 native：

- `--native-split family`：同一缺陷配置系列的全部 POSCAR 一起分配，约 60%/20%/20% 的系列用于 train/val/test。
- `--native-split material`：按完整宿主材料分配到 train/val/test，材料互不重叠。
- 默认 `--native-split random`：保留原随机构型划分。

比例按组计算，结构数量不保证恰好 60/20/20。family 允许不同缺陷系列共享宿主，不能称为新材料外推。两个 OOD 选项均不能与 `--cv5` 混用。

本次推荐命令（3 个模式 × 3 个数据集 × 3 个 seed，共 27 次训练）：

```bash
python -m HERA.main --model alignn --dataset native semi imp2d --mode was_x hetero hetero_was --native-split family --native-preprocessing reference_v1 --semi-preprocessing reference_v1 --imp2d-preprocessing reference_v1 --native-outlier-filter strict --semi-quality-filter physical --semi-source-policy legacy_available --imp2d-source db --imp2d-quality-filter physical --imp2d-host-filter reviewed_v1 --imp2d-energy-window -10 10 --alignn-hetero-dd drop --r 0 --alignn-hetero-feature-norm layernorm --alignn-hetero-node-norm layernorm --alignn-hetero-relations shared_residual --alignn-hetero-adapter-rank 8 --alignn-hetero-pooling defect_energy_mean --seed 123 11 1245 --epochs 500 --early-stopping-patience 0 --device cuda:0 --atom-init HERA/atom_init.json --run-dir HERA/logs/alignn_native_family_ood_v1 --compact-logs --resume
```

只想先跑一个 seed，把 `--seed 123 11 1245` 换成 `--seed 123`。若之后增加 seed，应在首次预处理时就提供全部计划 seed，或者为新的 seed 集合指定新目录；冻结清单不会被静默扩写。

如果希望按材料 OOD，只将 `--native-split family` 改成 `--native-split material`，并将输出目录改为 `HERA/logs/alignn_native_material_ood_v1`。不要复用原随机划分目录。

Native 使用新分组训练集重新拟合 strict 清洗阈值，验证/测试不参与阈值拟合。划分协议进入过滤清单、配置和 checkpoint，随机/family/material 不能互相复用旧结果。原始数据与已有实验保留。

Semi 仍按现有 60/20/20 随机构型划分，并保留 `legacy_available` 的来源处理；IMP2D 的 db 路径仍沿用已有的等价结构组划分、物理过滤、宿主筛选和 -10 < E < 10 eV 范围。这两者都不会因 `--native-split` 改成 OOD。

输出包含：

- `native_ood_split_counts.csv`、`native_ood_split_membership.csv`：native 分组数量和逐源文件归属。
- `native_filter_manifest.json`：冻结划分、清洗规则及训练集拟合阈值。
- `alignn/<dataset>/summary.txt`：各模式测试 MAE 和跨 seed 汇总。
- `alignn/native/<mode>/seed<seed>_test_predictions.csv`：native OOD 逐样本测试预测。
- 各模式的 history、config 和最佳 checkpoint。

只有验证集用于学习率调度和最佳 checkpoint 选择，测试集在训练完成后评估。更换 split 会改变任务，OOD MAE 不应与旧 IID MAE 直接解释为架构优劣。
