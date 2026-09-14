# Hetero：跨关系注意力与 sparse 缺陷残差的独立实验

2026-09-14。以 `shared_residual`、rank 8、LayerNorm、
`defect_energy_mean`、r=0 为当前基线。新增两个独立选项，未修改默认架构。

| `--alignn-hetero-ablation` | 主干聚合 | 额外 defect 支路 |
|---|---|---|
| `baseline` | 各关系分别 sigmoid 加权平均 | 无 |
| `cross_relation_attention` | 对所有入边统一逐通道 softmax | 无 |
| `sparse_residual` | 与 baseline 相同 | 12 Å 直接缺陷间通信 |

这两组均保持当前的 `linear` 主消息和 `independent` 初始距离编码。
它们没有叠加之前的 `pair_message` / `shared_distance` 实验。

## 1. 跨关系注意力

设关系为 r，目标节点为 i，特征通道为 c。复用现有关系 gate 的 logits
`l_ijr`（包含 shared 主体和 adapter 修正）与现有 value `v_ijr`。

旧版：

```text
m_ir = sum_j sigmoid(l_ijr) * v_ijr / (sum_j sigmoid(l_ijr) + epsilon)
```

新版：

```text
alpha_ijr,c = exp(l_ijr,c) / sum_{all incoming relations r', neighbors k} exp(l_ikr',c)
m_ir = sum_{j in relation r} alpha_ijr * v_ijr
```

softmax 使用数值稳定的实现，分母跨同一目标节点的所有关系，在 float32
中计算；不同目标节点和不同 batch 样本不混合。每个关系仍保留自己的消息
槽，再交给现有的类型化融合 FFN。不会对这些槽再分别除以各自的权重总和。

这是复用现有 gate 的逐通道注意力，没有新增 score 网络，也不是直接复制
原 AttentionALIGNN 的四头打分网络。shared 参数、adapter、边状态更新、
line graph、6 Å 物理边和能量均值读出保持不变。3 ALIGNN + 3 GCN 层中的
异构原子聚合都使用所选策略，line graph 自身的卷积不变。

64 维默认配置仍为 679,193 个参数。相同 seed 下，所有参数的初始值都与
baseline 一致；预测从一开始就可能不同，因为聚合规则不同。

## 2. Sparse 缺陷残差

保留原有 6 Å 原子/角度主干，在最后一个 GCN 层之后、readout 之前，增加
一次直接 defect-defect 消息更新：

```text
z_ij = [LayerNorm(h_i), LayerNorm(h_j), RBF(distance_ij; cutoff=12 A)]
q_ij = sigmoid(Linear(z_ij)) * MLP(z_ij)
a_i = mean_{j in auxiliary defect neighbors} q_ij
h_i_new = h_i + sigmoid(scale_logit) * W_out(a_i)
y = mean_{actual defects i} readout(h_i_new)
```

MLP 为 `(2H + radial_bins) -> H -> H`，中间使用 SiLU。
支路通过主干 defect 表示接收元素与局部环境信息，并接收自己的距离 RBF；
它没有新增原始 `was_species` 输入，也不等同于复制完整 MEGNet sparse 模型。
按用户要求，支路对经过 gate 的邻居消息做普通算术 mean，以邻居条数为分母。

- 额外边只连接 `pool_type=1` 的真实 defect，包含 vacancy 虚拟节点。
- r>0 标记的邻近 pristine 不会进入额外边或能量均值。即使真实 defect
  位于不同 PyG 节点 store，额外边也按各 store 的正确局部索引连接。
- 使用结构的周期邻居搜索，保留 cutoff 内的非零周期镜像；不添加零距离自环，
  不采用 12 个邻居的数量截断。
- 附加的 `sparse_dd` 边不进入原来的四类物理关系、距离编码器或 line graph。
- `W_out` 无 bias，并从零初始化；`scale_logit=-3`，对应约 0.047 的初始系数。
  初始输出与 baseline 完全一致，第一步输出投影即可得到梯度；投影更新后，
  前面的消息网络和残差系数也会学习。没有邻居的节点保持不变。

64 维默认配置为 709,210 个参数，比 baseline 增加 30,017。
新增支路独立初始化，不改变已有主干和 readout 的初始参数。

## 运行命令

先把修改后的 HERA 代码同步到服务器，在 `/home/wuhao` 运行。
以下是一条命令，顺序运行 WSe2 的两组，不重复已有 baseline：

```bash
python -m HERA.main --model alignn --dataset 2dmd_wse2 --mode hetero --r 0 --alignn-hetero-feature-norm layernorm --alignn-hetero-relations shared_residual --alignn-hetero-adapter-rank 8 --alignn-hetero-pooling defect_energy_mean --alignn-hetero-ablation cross_relation_attention sparse_residual --seed 123 --epochs 500 --device cuda:0 --resume --run-dir HERA/logs/hetero_interaction_ablation
```

同时验证两个材料，用这一条命令运行四组：

```bash
python -m HERA.main --model alignn --dataset 2dmd_mos2 2dmd_wse2 --mode hetero --r 0 --alignn-hetero-feature-norm layernorm --alignn-hetero-relations shared_residual --alignn-hetero-adapter-rank 8 --alignn-hetero-pooling defect_energy_mean --alignn-hetero-ablation cross_relation_attention sparse_residual --seed 123 --epochs 500 --device cuda:0 --resume --run-dir HERA/logs/hetero_interaction_ablation
```

两个材料分别训练。默认维持 low 80%/20% 训练/验证、全部 high 测试的协议。
每项实验独立保存 history、best_checkpoint、test_predictions、summary。
WSe2 的结果目录为：

```text
HERA/logs/hetero_interaction_ablation/alignn/2dmd_wse2/hetero/r0/
  features_layernorm/pool_defect_energy_mean/relations_shared_residual_rank8/
    aggregation_cross_relation_attention/
    defect_residual_sparse_cutoff12/
```

`--resume` 跳过已有最终 TEST 的实验，不是恢复中断的优化器状态。
如果先运行 WSe2 两组，再运行同目录的四组命令，会跳过已完成的 WSe2 两组。

## 显式配置和旧 checkpoint

可用 `--alignn-hetero-aggregation relation_mean|cross_relation_attention`、
`--alignn-hetero-defect-residual none|sparse` 单独选择或组合两项；这些显式开关
不能和 `--alignn-hetero-ablation` 同时使用。`--alignn-hetero-defect-cutoff 10`
可调整额外缺陷边半径，并允许和 ablation sweep 同时使用；只有启用支路的
实验使用该半径，相应输出目录也会包含它。

checkpoint 保存 `hetero_aggregation_mode`、`hetero_defect_residual` 和可选
`hetero_defect_cutoff`。缺少字段时默认 `relation_mean` / `none` / 12 Å。
旧参数键和路径保持兼容。加载带支路的 checkpoint 时，需要用其配置重新
构图，缺少附加边会明确报错，不会悄悄关闭支路。

## 验证

- 初版完整 175 项 unittest 通过，其中 8 项新增测试覆盖关系相对权重、softmax
  稳定性、空关系、真实 defect mask、8 Å 边、周期镜像、批量索引、零残差
  初始化、支路梯度和几何依赖、能量均值、严格恢复及 CLI 两组派发。
- 原有 baseline 的 350 项 state_dict 初始化指纹完全一致。
- 初版两组在 MoS2 / WSe2 各 8 个真实 low 结构上分别完成 3 步 CUDA 优化，并
  通过 FP16 前向/反向检查；各额外检查了两个含 24 defects 的 high 结构前向。
- 单图与批量预测最大差异约 `7.15e-7`，本地峰值 GPU 分配内存约 2.32 GiB。
- sparse 支路改为 mean 后，重新通过上述 8 项交互测试，包含梯度、批量预测
  和 checkpoint 恢复；此前短程 CUDA 记录对应初版 max 支路。

这些是运行与数值检查，没有完成 500 epoch 训练，不能用于判断哪组 MAE 更好。
可复现短程检查见 `logs/hetero_validation/interaction_cuda_smoke.py` 和 `.json`。

```bash
python -m unittest HERA.tests.test_hetero_interaction_ablation HERA.tests.test_hetero_message_distance HERA.tests.test_hetero_alignn_shared HERA.tests.test_hetero_defect_energy_mean
```
