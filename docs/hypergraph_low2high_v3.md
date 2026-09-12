# low→high 的 attention 优势与 hypergraph v3

2026-09-11 更新：第一版 v3 **分层 attention pooling** 已完成训练，high MAE 为
0.250031，劣于 v2 和 attention。后续默认的 **defect_mean** 尚无完整结果。
详见 [本次结果诊断](hypergraph_v3_completed_diagnosis.md)。以下保留原设计分析和
实现记录，不能把其中的预期或旧实验数字当作新版性能证据。

现已加入固定 `defect_mean` 的 `none / local / local_global` 超图更新对照。
运行命令与具体区别见 [消息传递消融说明](hypergraph_update_ablation.md)。

本次依据本地 `logs/2dmd_mos2` 的已有结果分析，并实现
`defect_global_attention_v3`。新版已完成工程验证，**尚未完成完整训练，
不能声称其测试 MAE 已经超过 attention**。以下误差均来自旧实验，版本设计
没有用 high 标签训练或选择 checkpoint。

## 后续直接检查 checkpoint 的发现

进一步检查标签定义：全部 5933 个 MoS₂ low 和 500 个 high 样本满足
`formation_energy_per_site = formation_energy / N_defects`，两组最大绝对残差分别
为 `1.33e-15`、`8.88e-16`。这里的 site 指缺陷位点，目标实际是每个 defect 的
平均形成能，分母并非全图原子数。

固定已训练的 attention checkpoint，随机种子 123 选取 3 个 low 样本和每档
defect 数量各 1 个 high 样本（共 8 个）。这些样本的最终 global attention
几乎全部落在 defect 节点，defect 内最大/最小权重比不超过 `1.000002`。
保持所有节点表示与预测 MLP 不变，仅把最终 attention pooling 换成
`mean(x[defect_mask])`，8 个样本的预测变化均小于 `1e-6`（反标准化后的目标单位）。
记录见 `logs/hypergraph_v3_validation/attention_pool_probe.json`。

因此这些样本上的最终读出实际上接近 **defect mean pooling**：先让物理消息
传递把宿主环境写入 defect 表示，再对 defect 表示取平均。这与标签的归一化
方式一致，是目前比“attention 会挑重要原子”更具体的解释。它并不意味着
pristine 节点没有作用，也不意味着先平均表示再用非线性 MLP 严格等于平均
各 defect 的标量能量。8 个样本的 pooling 替换不是完整测试集消融，局部
attention、LayerNorm 和读出各自的性能贡献尚未独立量化。

这也限制了对 v3 的预期：更复杂的 pooling 未必更优。现在已经把明确的
defect mean pooling 接为新训练的默认读出，并保留前一版分层 attention 供对照。

## 已核实的现象

比较对象为 ALIGNN、`2dmd_mos2`、seed 123，同一组 500 个 high 浓度测试结构。
attention 与旧 hypergraph checkpoint 保存的 split 完全相同，均使用 64 维隐藏层、
3 个 ALIGNN block、3 个 GCN block、6 Å 物理 cutoff、12 个邻居上限，
train batch size 为 8。目标字段为数据集的 `formation_energy_per_site`。

| 模型 | 最低 low 验证 MAE | high 测试 MAE | high 平均有符号误差（预测−真实） |
|---|---:|---:|---:|
| Attention ALIGNN | 0.002407 | 0.056992 | +0.045310 |
| Hetero ALIGNN r0 | 0.002169 | 0.116433 | −0.094062 |
| Hypergraph v2 | 0.002069 | 0.168732 | −0.108163 |

旧 hypergraph 在 low 上拟合得很好，但向 high 转移时偏低估。这个结果支持
“分布外泛化存在问题”的判断，不能仅用增加参数或训练轮数解释。

数据集 low 的 5933 个标签中，1、2、3 个 defect 分别有 4、127、5802 个结构；
high 的 4、9、14、19、24 个 defect 各有 100 个结构。这是明显的缺陷数量外推。
数量取自 descriptors 的 `defects` 列，并通过 `descriptor_id` 与 targets、
`source_id` 与预测表做一对一关联。

| high 中 defect 数量 | Attention MAE | Hypergraph v2 MAE |
|---:|---:|---:|
| 4 | 0.018423 | 0.083998 |
| 9 | 0.038688 | 0.208256 |
| 14 | 0.050758 | 0.200767 |
| 19 | 0.075142 | 0.189655 |
| 24 | 0.101948 | 0.160983 |

Attention 在这五档上均更好，但自身误差也随数量增加而上升。因此全 defect
超边有合理的改进空间，尤其是学习多个局部缺陷环境之间的关系。

原始证据位于：

- `logs/2dmd_mos2/model_comparison_predictions.csv`
- `logs/2dmd_mos2/alignn/2dmd_mos2/attention/seed123_history.csv`
- `logs/2dmd_mos2/alignn/2dmd_mos2/hypergraph/per_defect_neighborhood_v2/seed123_history.csv`
- 上述两个目录中的 `seed123_best_checkpoint.pth`

## 为什么 attention 可能更适合这次外推

1. **局部消息有选择性。** Attention 的权重取决于原子表示、键信息与
   defect/pristine 类型；物理距离和 ALIGNN 角度特征仍然参与计算。
   它能调整不同邻居的贡献，而非只依赖统一的聚合规则。
2. **最终读出也是 attention。** `AtomTypeGlobalAttentionReadout` 根据节点、
   类型和图上下文学习权重。v2 则先在超边内 mean，再在相同类型的超边间 mean，
   并拼接全图 mean。稀有但重要的 defect 环境可能被平均掉。
3. **归一化是实际存在的混杂因素。** 已保存的 attention checkpoint 没有任何
   BatchNorm running mean；v2 有 23 组。当前 attention 使用 LayerNorm，
   v2 的物理骨干仍使用 BatchNorm。在缺陷比例变化时，沿用 low 训练分布的
   running statistics 可能有害。这是需要消融验证的机制，尚未被单独证明。
4. **v2 的 core 超边是 singleton。** 每个 defect 自己构成一个 core 超边，
   本身不提供 defect 之间的信息交换。defect 之间只能经物理键或共享局部 pristine
   间接联系。新增全 defect 超边可以让每层都交换缺陷集合的信息。
5. **旧超图 attention 的归一化不等同于两次标准 attention。** 本机 PyG 2.7.0
   的 `HypergraphConv` 在节点 softmax 后还应用超边大小倒数，并在回传时重用
   同一组权重、再应用节点度数倒数。对于孤立的、大小为 k 的超边、相同节点值
   和均匀 attention，忽略 bias 后的两次传播结果带有 `1/k²` 缩放。
   后续 LayerNorm 会改变其影响，不能把这个比例直接解释成最终预测误差；但
   不同大小超边的相对消息权重仍值得关注。实现可核对
   [PyG 官方源码](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/hypergraph_conv.html)。

这些是结合代码与数据的机制假设；单 seed 的多项架构比较并非因果消融。

## 本次实现

### 构图

- `E_defect = {所有 defect 节点}`，每个图恰好一条，单 defect 时允许 singleton。
- 每个 defect 保留自己的 `E_local(d) = {d} ∪ {半径内的 pristine}`。
  不把其他 defect 塞入它的局部边；不同 local 边允许共享 pristine。
- 取消远处 pristine 的全局超边。远处原子继续保留物理键、节点特征和背景读出，
  在超图分支中保持原样。混合骨干允许它们通过物理消息影响缺陷；纯 hypergraph
  的 defect mean 模式没有这条物理路径，因此远处 pristine 不参与预测。
- 保留 `region_type` 的三个类别：defect、near pristine、far pristine。
  修正无超边关联的节点在自动推断时被错误判为 defect 的情况。

### 消息传递

复用 CGCNN、MEGNet、ALIGNN 已有的原子类型 attention 物理模块。
HyperALIGNN 的节点、键、角度嵌入和物理层统一使用 LayerNorm。

新超图 block 用当前 defect center、当前超边节点均值与超边类型产生 query。
先在每条超边内部做 node→edge softmax，再针对每个节点的入射超边做独立的
edge→node softmax，不再额外除以大小或度数。两方向均使用多头 query/key/value。
消息与 FFN 的残差分别有可学习门，初始 `sigmoid(-2)≈0.119`，控制新增分支的扰动。

### Pooling

当前默认是 `--hypergraph-pooling defect_mean`，用于直接验证 checkpoint
观察到的读出机制：

```text
物理 attention + 全 defect 超边 + 每个 defect 的 local 超边
                           ↓
                  最终节点表示 h_i
                           ↓
        g = sum(h_i for i in defect) / N_defect
                           ↓
                   预测 MLP → 平均 DFE
```

每个 defect 节点只计一次，即便同时属于 global 和 local 超边。共享 pristine
不会增加平均的分母。最终 pooling 不拼接全图、local、pristine、显式浓度比例、
Set2Set 或 state 的额外读出分支。MEGNet 的 state 仍参与前面的消息传递。
64 维 HyperALIGNN 使用 `64→128→64→1` 的预测 MLP。
平均的是节点表示，之后才经过非线性 MLP；不是先对每个 defect 预测标量再平均。
该版本没有改变数据标签，也没有让预测随 defect 数量直接成倍增长。

选择 `--hypergraph-pooling hierarchical_attention` 可恢复前一版读出：
先生成固定长度的三组表示：

1. **Defect 表示 D：** 在全 defect 超边内，根据当前缺陷集合学习权重。
2. **局部环境表示 L：** 每个 local 内先用 center 和环境作为上下文学习节点权重，
   再用 D 作为上下文学习不同 local 的权重。避免一个大邻域仅因为节点更多而
   自动压过另一个小邻域。
3. **Pristine 背景 P：** 所有 pristine 节点直接做一次由 D 条件化的 attention。
   共享 pristine 在背景读出中只出现一次，背景读出不会把消息回传成全局超边。

进一步加入三个有界的结构比例：

`f = [N_defect/N_sites, N_near/N_sites, N_shared_pristine/N_sites]`

其中 shared 指属于超过一个 local 超边的 pristine，`N_sites` 包括图中的 vacancy
占位节点。最终 `R = concat(D,L,P) + W_f f`，`W_f` 零初始化、随训练学习。
这些比例保留 softmax 会丢失的浓度和重叠信息；不直接用原子数或 defect 数乘
预测值，以免擅自对 `formation_energy_per_site` 施加总能量可加性假设。

在 hierarchical_attention 模式下，CGCNN/ALIGNN 还拼接一组 atom-type global
attention 表示；MEGNet 还拼接 Set2Set 节点/键读出与 state。纯 hypergraph
还拼接 state embedding。这些额外分支仅保留在该对照方案中。

### 版本与适用范围

`get_config(..., 'hypergraph')` 和 `hypergraph_was` 默认选择 v3 + defect_mean。
纯 hypergraph、HyperCGCNN、HyperMEGNet、HyperALIGNN 均已接通。
原始 attention 模型、数据切分、标签和优化器协议不变。

`--hypergraph-schema per_defect_neighborhood_v2` 保留旧实现。
底层类直接构造、以及缺少 schema 的旧 checkpoint config，仍默认 v2，以保持
历史调用兼容；直接调用新类时应显式传入 `hypergraph_schema`，或对
`RegionHypergraphInteraction` 传入 `schema`。旧 v3 config 没有 pooling 字段时，
按 hierarchical_attention 还原，不会自动变成 defect_mean。底层新模型需显式
传入 `hypergraph_pooling='defect_mean'`；Interaction 对应参数为 `pooling`。
v2 保留 `region_mean`，不支持 v3 的两个 pooling 选项；CLI 会提前拒绝不兼容组合。
schema 或 pooling 不同的参数不能直接互换，需重新训练。
主训练入口和 native LOO 均支持两个参数。defect_mean 新结果进入 schema 下的
`pool_defect_mean/` 子目录；hierarchical_attention 保留旧 v3 路径。native LOO
的 run label 同样带 pooling 后缀，预测汇总脚本也识别这个后缀。

## 验证与运行

- 122 项 unittest 全部通过，包括两种 pooling、旧模型、数据切分和预测脚本回归。
- 新测试覆盖变长超边 batch、跨图污染拒绝、无关联 far 节点、无键图、无 pristine、
  两方向 softmax 归一化、重复相同节点不引起基数缩放、节点/超边置换不变性、
  共享 pristine 去重、四种骨干梯度和批处理一致性。
- 本地真实 v2 checkpoint 完成 strict restore。
- 前一版 hierarchical_attention 的检查：8 个 seed 123 随机抽取的真实 low 结构，
  HyperALIGNN v3 在 RTX 5060 Ti
  上执行 10 步优化；损失与梯度有限，MSE 从 10.2111 到 1.7652。
  实测峰值分配显存约 2.26 GiB；混合精度反向通过；成批/逐图预测最大差异
  `4.77e-7`。这只是小样本工程检查，不能用于判断迁移性能。
  原始记录：`logs/hypergraph_v3_validation/real_low_cuda_smoke.json`。
- 当前 defect_mean 版在同样 8 个 low 结构上完成 10 步 GPU 优化，MSE 从
  10.0877 到 2.0208，梯度与混合精度反向均有限。批处理/逐图预测最大差异为
  `2.38e-7`，峰值分配显存约 2.26 GiB。这里只检查工程行为，不能把小样本的
  10 步 loss 与上一版当作性能对比。记录为
  `logs/hypergraph_v3_validation/defect_mean_cuda_smoke.json`。

从 HERA 的父目录运行完整实验：

```powershell
Set-Location C:\Users\User\Desktop
& 'C:\Users\User\.conda\envs\hera\python.exe' -m HERA.main --model alignn --dataset 2dmd_mos2 --mode attention hypergraph --hypergraph-schema defect_global_attention_v3 --hypergraph-pooling defect_mean --hypergraph-radius 3.0 --seed 123 --epochs 500 --device cuda:0 --run-dir HERA/logs/2dmd_mos2_hypergraph_mean
```

该命令显式选择 v3 + defect_mean。将 pooling 改为 `hierarchical_attention` 可跑
分层读出对照。重跑 v2 时，将 schema 改为 `per_defect_neighborhood_v2`，
删除 pooling 参数（或改为 `region_mean`），并更换 `--run-dir`。
推广验证可用 `--seed 123 42 7`，并将 dataset 换成 `2dmd_wse2`。
Checkpoint 和早停仍只由 low 验证集决定，high 仅用于最终评估。

尚需完成的实验：完整训练后的同 seed 对比、多 seed 方差、按 defect 数量报告误差；
若要证明某一机制有效，还应分别消融全 defect 超边、分层 pooling 和物理 attention/
LayerNorm。本次 v3 将这些改动作为一个候选版本实现，并未声称某一项独立有效。
全 defect 超边本身不含显式 defect 间距离，几何仍依赖物理骨干和局部邻域；
新增交互与读出也增加参数量，可能过拟合。后续比较需要同时观察 low 验证误差与
最终 high 误差，不能依据 high 误差反复选择架构后仍称其为独立测试。
