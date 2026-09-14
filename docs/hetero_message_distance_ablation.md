# HeteroALIGNN：消息内容与距离编码的独立对照

2026-09-14。基于当前 `shared_residual`、rank 8、LayerNorm、
`defect_energy_mean` 版本，分别检验两项改动。默认架构不变。

| CLI ablation | 主消息内容 | 距离编码 |
|---|---|---|
| `baseline` | `Linear(h_src)` | 四种有向关系独立编码 |
| `pair_message` | `MLP([h_src, h_dst, e_ij])` | 与 baseline 相同 |
| `shared_distance` | 与 baseline 相同 | 共享编码 + 每种关系一个可学习向量 |

`pair_message` 的 MLP 为 `3H -> H -> H`，中间使用 SiLU。
`e_ij` 是当前边表示：初始由距离 RBF 编码，在 ALIGNN 层内还接收
line graph 的角度更新。关系身份仍通过原有的关系距离编码和 adapter 表达。
没有额外改变 gate、关系内归一化、节点融合、图连接、读出或标签。

`shared_distance` 仅共享初始距离编码器，不共享后续每条边的动态状态。
四种有向关系仍各有一个 H 维向量，初始为零。
节点 embedding、关系内归一化、关系 adapter 和节点融合保持当前实现。
共享编码器初始化自旧编码器 bank 的第一项；构造时保留旧随机数消耗顺序。
两组所有未改变模块的初始权重均与同 seed 基线完全一致。

这不是把模型替换成当前 `alignn_definet`：hetero 仍分别归一化关系消息并融合，
最后执行 `mean(MLP(h_defect))`。当前 DefiNet 的 defect-aware 层使用 marker-pair
gate 后直接求和，最终全节点均值后再执行 MLP。

## 一条命令运行两组

先同步修改后的 HERA 代码。在服务器 `/home/wuhao` 目录运行：

```bash
python -m HERA.main --model alignn --dataset 2dmd_wse2 --mode hetero --r 0 --alignn-hetero-feature-norm layernorm --alignn-hetero-relations shared_residual --alignn-hetero-adapter-rank 8 --alignn-hetero-pooling defect_energy_mean --alignn-hetero-ablation pair_message shared_distance --seed 123 --epochs 500 --device cuda:0 --resume --run-dir HERA/logs/hetero_message_distance_ablation
```

两组依次从头训练，不重复已有 baseline，不运行两项同时开启的组合。
WSe2 按原协议使用 low 的 80%/20% 训练与验证，全部 high 用于测试。
将 `--dataset 2dmd_wse2` 换成 `--dataset 2dmd_mos2 2dmd_wse2` 可运行四组。
两种材料分别训练。将 `--seed 123` 换成多个整数可复核种子稳定性。

结果位于：

```text
HERA/logs/hetero_message_distance_ablation/alignn/2dmd_wse2/hetero/r0/
  features_layernorm/pool_defect_energy_mean/relations_shared_residual_rank8/
    message_pair_mlp/
    distance_shared/
```

每组保存自己的 history、最佳 checkpoint、逐样本预测与 summary。
`--resume` 跳过同目录中已写出最终 TEST 的实验，不恢复中途优化器状态。
合并预测表时，两种架构也使用不同列名。

## 单独选择与兼容

也可通过 `--alignn-hetero-message linear|pair_mlp` 和
`--alignn-hetero-distance independent|shared` 显式选择配置。
显式选项不能和 `--alignn-hetero-ablation` 同时使用，防止两组对照意外混合。
这些选项仅用于 ALIGNN hetero 系列；默认参数和旧结果路径不变。

checkpoint 保存 `hetero_message_mode` 和 `hetero_distance_mode`。
缺少字段的旧 checkpoint 继续使用 linear/independent，可严格恢复旧 state_dict。
新架构改变的参数形状/名称由其保存的配置恢复，不应对旧权重强行套用新配置。

针对性测试覆盖消息内容对目标节点和几何的依赖、共享径向网络及独立类型
向量的梯度、实际 defect mask、单图/批量一致性、无边/无宿主、12 种
配置组合严格恢复、初始化隔离、结果路径和单条 CLI 的两组实验派发。

```bash
python -m unittest HERA.tests.test_hetero_message_distance HERA.tests.test_hetero_alignn_shared HERA.tests.test_hetero_defect_energy_mean
```

短程 CUDA 检查只验证模型运行与梯度，不产生可用于比较优劣的 benchmark MAE。

本次完整 167 项测试通过。两组均在 8 个真实 WSe2 low 结构上完成 3 步 CUDA
优化，并通过额外的 FP16 前向/反向检查，梯度均有限。单图与批量预测最大
差异为 `2.38e-7`。同 seed 原版初始化的 350 个 state_dict 项指纹全部一致；
旧 WSe2 hetero checkpoint 严格恢复后，抽查 high 预测与保存值相差 `3.35e-11`。
完整 64 维、3 ALIGNN + 3 GCN 配置参数量：baseline 679,193，
pair_message 753,305，shared_distance 658,329。
