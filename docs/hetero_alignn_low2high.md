# HeteroALIGNN：全程 LayerNorm + defect mean

2026-09-12 更新：新训练默认已增加共享消息主体与小型关系修正，详见
[共享关系版本与命令](hetero_alignn_shared_relations.md)。下文记录前一版
独立关系参数的 LN + defect mean 架构。

前一版实现：保留四种有向关系、每种关系的独立参数、
关系内 gate 归一化、节点融合 FFN、物理邻居和角度图；修改归一化与最终 pooling。
当时尚未加入共享关系参数或跨关系 attention。后续用户提供的这一版完整
训练日志显示 high MAE 为 0.129310，attention 为 0.051590，详见
[实际结果分析](hetero_relation_shift_review.md)。

## 前一版架构

`get_config('alignn', dataset, 'hetero')` 默认设置：

```python
hetero_node_norm = 'layernorm'
hetero_feature_norm = 'layernorm'
hetero_pooling = 'defect_mean'
```

同样适用于 ALIGNN 的 `hetero_was` 和 `hetero_fixed_pool`；其他模型和模式不受影响。
LayerNorm 覆盖节点 embedding、各关系的两层距离 embedding、两层角度 embedding、
line-graph 的节点/边更新、所有 ALIGNN/GCN 关系边更新和节点残差更新。
默认模型中不含 BatchNorm，也不累计 running statistics。

最终读出为：

\[
\bar h_D=\frac{1}{N_D}\sum_{i\in D}h_i^{(L)},\qquad
\hat y=\mathrm{MLP}(\bar h_D).
\]

默认隐藏维度 64 时，预测头是 `64 → 64 → 32 → 1`，隐藏层用 SiLU。
先平均特征，再经过非线性 MLP；不是先逐缺陷预测能量再平均。
在 `2dmd_mos2` 中，标签仍然是平均缺陷形成能（eV/defect），没有修改标签或 scaler。

`D` 根据 `pool_type` 取真实缺陷；缺少该标记的直接输入退回节点存储类型。
当 `r > 0` 把局部 pristine 节点也标为 defect 区域时，它们不参与最终缺陷均值。
每个图必须至少有一个真实缺陷，否则报错。pristine 节点仍参与物理消息传递。
`defect_mean` 下，`hetero_fixed_pool` 与 `hetero` 的读出相同。

## 运行与消融

在 HERA 的上一级目录、激活 hera Python 环境后运行：

```bash
python -m HERA.main --model alignn --dataset 2dmd_mos2 --mode hetero --r 0 --alignn-hetero-feature-norm layernorm --alignn-hetero-pooling defect_mean --alignn-hetero-relations independent --seed 123 --epochs 500 --device cuda:0 --run-dir HERA/logs/2dmd_mos2_hetero_ln_mean
```

如需同一入口同时训练 attention 对照，使用 `--mode attention hetero`。
上述两个新参数只应用于 HeteroALIGNN，attention 不受影响。
`--epochs 500` 是最大轮数，现有 early stopping 仍然生效。

| 实验 | `--alignn-hetero-feature-norm` | `--alignn-hetero-pooling` | r0 下的子目录 |
|---|---|---|---|
| 旧架构 | `batchnorm` | `type_mean` | 无新增子目录 |
| 只改归一化 | `layernorm` | `type_mean` | `features_layernorm` |
| 只改 pooling | `batchnorm` | `defect_mean` | `pool_defect_mean` |
| 新版默认 | `layernorm` | `defect_mean` | `features_layernorm/pool_defect_mean` |

节点残差默认保持 `layernorm`；已有 `--alignn-hetero-node-norm` 仍可单独消融，
显式指定后增加 `norm_<choice>` 目录。若改为 `batchnorm`，模型就不再是全程 LN。
Native LOO 入口也支持这两个新参数；结果标签和合并预测列标识不同架构。

新版上述命令的 checkpoint 位于：

```text
HERA/logs/2dmd_mos2_hetero_ln_mean/alignn/2dmd_mos2/hetero/r0/features_layernorm/pool_defect_mean/seed123_best_checkpoint.pth
```

保持相同 split、seed 和训练协议，用 low 验证集选 checkpoint，high 留作最终比较。
更改归一化和预测头后需要重新训练，不能直接套用旧权重获得新版模型。

## 兼容性与检查

底层 `HeteroALIGNN` 直接构造的默认参数、以及缺少新字段的旧 checkpoint config，
仍使用 `batchnorm` feature norm 和 `type_mean` pooling；原参数名称保持不变。
新配置会把两个字段保存在主训练入口生成的 checkpoint 中。

`tests/test_hetero_alignn_transfer.py` 检查：无 BN、训练/推理及单图/批量一致性、
真实缺陷 pooling、pristine 消息梯度、空关系/无键/无 host 图、零缺陷拒绝、
四种组合的严格恢复，以及独立输出路径和预测列。

本次验证结果：完整 unittest 套件 129 项通过。真实旧 hetero checkpoint 严格恢复
成功，抽查一条原测试预测的差异约为 `7.3e-11` eV/defect。
RTX 5060 Ti 上用 8 条真实 low 数据做 10 步 AdamW 检查，loss/梯度及 FP16
autocast 前向/反向均为有限值；单图与批量预测最大差异 `3.58e-7`，峰值 allocated
显存约 2.16 GiB。新版参数量为 916,289。记录见
`logs/hetero_validation/ln_defect_mean_cuda_smoke.json`。
短训练只验证模型能正常训练，不代表 low→high 泛化成绩。

原模型的诊断结果保存在 `logs/hetero_validation/low2high_diagnosis.json`：
同一 seed/split 下 hetero high MAE 为 0.116433，attention 为 0.056992。
这些数字不属于本次新版模型。
