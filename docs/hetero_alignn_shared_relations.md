# HeteroALIGNN：共享消息主体与关系修正

2026-09-12。针对前一版 hetero 在 low 验证集优于 attention、high 测试集却
更差的结果，新增关系参数共享对照。用户提供的前一版成绩是 low 最佳验证
MAE 0.001807、high MAE 0.129310；这些数字不属于本次新模型。

## 修改范围

每个 ALIGNN/GCN 层内，aa、dd、ad、da 四种有向关系共享一个
`HeteroRelationConv` 主体。不同深度的层仍有各自的参数。共享主体包含源/目标
节点 gate、边 gate、消息值投影和边更新的 LayerNorm。

每层每种关系另有一个瓶颈宽度为 8 的小型修正网络，以源节点、目标节点和
当前边特征为输入，分别修正 gate logits 和消息值：

\[
[\delta z_r,\delta v_r]
=\sigma(a_r)B_r\operatorname{SiLU}(A_r[h_j,h_i,e_{ij}]),
\]
\[
z_{ij}=W_sh_j+W_dh_i+W_ee_{ij},\qquad
m_{ij,r}=\sigma(z_{ij}+\delta z_r)\odot(W_mh_j+\delta v_r).
\]

输出投影 `B_r` 的权重和 bias 初始化为零，`a_r=-3`，对应初始缩放约
0.0474。初始修正严格为零；输出投影能立即获得梯度，随后瓶颈投影也能学习。
`shared` 与 `shared_residual` 在同一 seed 下初始预测一致。

本次只改变上述消息模块的参数组织及其关系修正：

- 节点 embedding、四类关系各自的距离 embedding、角度/线图、物理邻居表均沿用原实现。
- 关系仍各自以 gate 总量归一化，固定关系槽仍由原来的节点类型 FFN 融合。
- 保持全程 LayerNorm、实际 defect 表示均值和 `64 → 64 → 32 → 1` 预测头。
- 同一 seed 下，三个版本未改动部分的初始权重一致。

因此，这次不是跨关系 attention 或配位数特征实验；关系间数量比例的处理
仍需后续单独验证。最终仍是 `MLP(mean(h_defect))`。

## 选项与参数量

`get_config` 对 ALIGNN 的 `hetero / hetero_was / hetero_fixed_pool` 默认选择
`shared_residual`。其他模型、attention 和 hypergraph 的配置不受影响。

| `--alignn-hetero-relations` | 消息模块 | MoS₂ 默认配置参数量 |
|---|---|---:|
| `independent` | 原来的四套独立参数 | 916,289 |
| `shared` | 一套共享主体 | 614,465 |
| `shared_residual` | 共享主体 + 四个关系修正分支，默认 | 679,193 |

新默认模型参数量减少约 25.9%。以上计数使用 hidden=64、3 个 ALIGNN block、
3 个 GCN block、LayerNorm 和 defect mean。可用
`--alignn-hetero-adapter-rank 4` 调整瓶颈宽度；该选项仅用于 `shared_residual`。

## 训练命令

先把修改后的项目代码同步到实际训练节点。在 `HERA` 的父目录、已安装依赖的
Python 环境中运行：

```bash
python -m HERA.main --model alignn --dataset 2dmd_mos2 --mode hetero --r 0 --alignn-hetero-feature-norm layernorm --alignn-hetero-pooling defect_mean --alignn-hetero-relations shared_residual --seed 123 --epochs 500 --device cuda:0 --run-dir HERA/logs/2dmd_mos2_hetero_shared
```

在同一命令中将 `shared_residual` 换成 `shared` 或 `independent` 即可做对照。
如需重新训练 attention 参考，把 `--mode hetero` 改为 `--mode attention hetero`。
模型仍由 low 验证集选择 checkpoint，high 用于最终评估。

新默认结果目录：

```text
HERA/logs/2dmd_mos2_hetero_shared/alignn/2dmd_mos2/hetero/r0/
  features_layernorm/pool_defect_mean/relations_shared_residual_rank8/
```

`shared` 使用 `relations_shared`；`independent` 不追加关系目录，保留前一版
结果路径。不同 adapter rank 分开保存。Native LOO 入口也支持这两个新选项。

## 兼容与验证

已保存配置缺少 `hetero_relation_mode` 时按 `independent` 恢复。
底层构造函数也保留旧默认；只有新训练配置选择新版。不能把旧独立权重
不经转换直接载入共享版本，也不能把旧的 MAE 当作新版成绩。

- 完整 unittest：148 项通过。
- 检查共享主体只注册一份、四关系均能训练它、每种修正只影响对应关系、
  零输出初始化及后续瓶颈梯度、真实缺陷 pooling、空关系、批处理一致性、
  checkpoint 恢复和输出路径隔离。
- 三种配置各用 8 个真实 low 结构完成 5 步 CUDA 训练，loss/梯度有限，
  FP16 autocast 前向/反向均正常；单图与批量预测最大差异不超过 `2.39e-7`。
- 真实旧 hetero checkpoint 严格恢复，抽查 high 预测与保存值相差 `7.3e-11`。

记录在 `logs/hetero_validation/shared_relations_cuda_smoke.json`。
短训练只验证工程正确性，尚未完成新版 low→high 实验，不保证性能改善。
