# Hetero 的 low→high 泛化：关系共享与邻域组成

## 后续取得的实际训练日志

通过用户已连接的 VS Code 确认了 `SSH: 158.132.183.167` 和指定目录。
界面上的 hetero 结果位于 `hetero/r0/features_layernorm/pool_defect_mean`。
随后用户提供两份完整 history 文本，并明确第一份是 attention、第二份是 hetero。
本次已分析这两份日志；尚未读取该远端 checkpoint 的内部配置或逐样本预测。

| 指标 | Attention | Hetero |
|---|---:|---:|
| 最佳 low 验证 MAE | 0.002183 | 0.001807 |
| 最佳验证 epoch | 286 | 272 |
| 该 epoch 的训练 MAE | 0.001475 | 0.001664 |
| 完成 epochs | 336 | 322 |
| 最后 50 epochs 验证 MAE 中位数 | 0.0024635 | 0.0021005 |
| 最终记录的 high 测试 MAE | 0.051590 | 0.129310 |

单位按本项目 MoS₂ 标签为 eV/defect。Hetero 最佳域内验证误差降低 17.2%，
域外测试误差却为 attention 的 2.506 倍。末段验证曲线也支持其域内优势，
并非仅一次特别低的验证点。两者均在最佳 epoch 后完成 50 个额外 epoch，
与本项目提前停止规则相符；日志没有支持“hetero 尚未训练好”的明显证据。

这次结果说明全程 LN/defect mean 方向尚未解决迁移问题，但由于没有同次实验
中逐项开关对照，不能把整体误差归因于其中某一个改动，也不能据此判定 LN
或 defect mean 单独有害。两份 history 各只有一个最终 TEST 数字，无法判断
更早 checkpoint 的 OOD 表现，也无法推断高估/低估方向或具体缺陷分组误差。

下一步优先保持物理图、LN、defect mean 与预测头一致，仅比较关系参数共享
方式，再单独比较关系间的 attention/计数信息。以下局部邻域统计提供设计
动机，并不替代这些因果消融。

记录：[指标 JSON](../logs/hetero_validation/supplied_remote_history_comparison.json)、
[曲线图](../logs/hetero_validation/supplied_remote_history_comparison.png)、
[分析脚本](../logs/hetero_validation/analyze_supplied_remote_histories.py)。

## 初次 SSH 访问记录（早于上述 UI 和用户日志）

用户指定的远端是 `wuhao@158.132.183.167`，结果目录是
`HERA/logs/2dmd_mos2_attention_hetero_hypergraph_mean/alignn/2dmd_mos2`。
当前 SSH 能连接服务，但普通批处理认证及显式使用现有 codex_158 密钥均被
拒绝，返回 `Permission denied (publickey,password)`。当时的分析没有读取
那次远端训练的指标、checkpoint 或代码版本；后续取得的日志见上节。

以下结论来自本地实际实现与 MoS₂ 数据。目录名含 mean 不足以证明远端使用了
哪种 pooling 或归一化；须以保存的 checkpoint config 和参数为准。

## 为什么域内好不能证明能外推

low 的训练和随机验证几乎都是三缺陷结构；high 为每图 4/9/14/19/24 个缺陷。
两组主要变化是缺陷密度、类型化局部邻域与相互作用组合；参考晶格位点数不必
增加，不能简单解释为图的总节点数变大。低浓度验证误差反映了相似局部环境
的插值，并不检验高缺陷配位的外推。

本次从本地 seed 123 checkpoint 的固定切分中，无放回抽取 256 个 low
训练结构、128 个 low 验证结构，并检查全部 500 个 high 结构。使用本地
当前 r=0、物理 cutoff=6 Å、每类型邻居上限 12 的转换器。dd 配位按模型
实际接收的 dd 入边统计，各 defect 等权：

| 指标 | low train（256 图） | low val（128 图） | high test（500 图） |
|---|---:|---:|---:|
| 每个 defect 平均 dd 邻居数 | 0.345 | 0.352 | 2.842 |
| 没有 dd 邻居的 defect 比例 | 70.4% | 69.0% | 8.9% |
| dd 邻居超过 2 个的 defect 比例 | 0% | 0% | 53.8% |
| 图中完全没有 dd 边的比例 | 59.0% | 57.8% | 5.4% |
| 最大 dd 入度 | 2 | 2 | 10 |

按 high 缺陷数分组，平均 dd 入度由 4-defect 的 0.595 增加到 24-defect 的
4.009；24-defect 组中 80.6% 的 defect 有超过两个 dd 邻居。这直接证实了
当前图表示的局部关系分布发生变化，但还未通过训练消融证明哪项模型设计
贡献了多少误差。尤其不能把这些统计当作远端 checkpoint 的性能结果。

复现：[统计脚本](../logs/hetero_validation/relation_shift_probe.py)、
[完整统计](../logs/hetero_validation/relation_shift_probe.json)、
[逐结构统计](../logs/hetero_validation/relation_shift_probe.csv)。

本地最新 HeteroALIGNN 默认全程 LayerNorm，最终 defect mean pooling。若远端
也是这版，BatchNorm running statistics 和直接的 pristine pooling 已被去除，
剩余误差不能继续全归因于这两点。

当前实现仍为两类节点独立 embedding、四种有向关系独立距离 embedding/
消息参数，以及按目的节点类型独立的融合 FFN。line graph 更新共享参数。
关系内部计算为：

\[
m_{i,r}=\frac{\sum_{j\in N_r(i)}\sigma_{ij,r}W_rh_j}
{\sum_{j\in N_r(i)}\sigma_{ij,r}+10^{-6}},\quad
h_i'=h_i+\mathrm{LN}\bigl(\mathrm{FFN}_{t(i)}([h_i,m_{i,r_1},m_{i,r_2}])\bigr).
\]

因此有两个值得分开检验的假设：

1. **关系独立参数降低了稀有关系的数据效率。** low 中稀疏的 dd 分支无法像
   共享消息网络一样直接利用所有关系训练消息主体；到了 high，需要处理更多
   dd 邻居及训练未覆盖的组合。attention 的消息主体共享，类型进入权重计算。
   这提供了可能更好的迁移偏置，但尚不是独立参数导致失败的因果证明。
2. **关系内归一化弱化了关系之间的邻居数量比例。** 固定其他输入，重复相同
   的 dd 消息时，dd 的加权平均基本不变，融合 FFN 也没有直接收到 dd 的
   邻居数或 gate 总量。attention 在全部入边上归一化，新增一种类型的邻居会
   改变它相对于另一种类型的总权重。attention 仍不是完整计数器，其他图层
   也可能间接编码计数，不能据此声称 hetero 完全不知道浓度。

合成检查：把一个关系中的同一条关联复制四次，固定输入与权重，当前
`HeteroRelationConv` 归一化聚合最大差异仅 `2.15e-6`，来自分母 epsilon。
它验证该聚合的计数性质，不代表真实原子重叠构型或训练后的性能比较。

相关代码：`models/alignn.py` 的 `HeteroRelationConv`、
`_hetero_relation_update`、`AtomTypeAttentionGatedGraphConv`，以及
`models/modules.py` 的 `RelationFusionUpdate`。

## 优先实验

1. 固定物理边、半径、LayerNorm、defect mean 读出和预测头，先只改变参数
   共享：完全共享、共享主体加小型关系修正、当前完全独立。关系修正从小幅
   输出开始，四类关系仍可表达差异；距离 embedding 也纳入共享对照。可采用
   `m_r = m_shared + lambda_r * delta_m_r` 或少量共享基矩阵。
2. 在确定共享方式后，单独比较现有的关系内归一化与跨全部邻居的 attention。
   若保留关系分支，可另试有空关系 mask 的融合门控，并输入固定物理半径内
   的局部缺陷占比/有效配位数。门控只看两个均值本身不能恢复已丢掉的数量。
3. 读出另做消融：当前 `MLP(mean(h_d))` 对比 `mean(MLP(h_d))`。后者更贴近
   每缺陷贡献的平均；在只含局部消息且邻域互不影响的情况下具有更明确的
   可加性。仅有结构总标签时，分配到各 defect 的能量不唯一，不应解释成
   已识别的真实单缺陷能量。
4. 用最近缺陷距离、dd 配位数、缺陷种类/团簇对验证集分组，检查低误差是否
   只来自常见孤立环境。如果允许增加标签，补充近邻多缺陷构型比延长训练
   更直接。拼接两个结构后简单相加标签不能生成真实近距离缺陷相互作用。

不建议一次改完以上因素后仅比较总 MAE。先固定一次 seed/split 做参数共享
对照，再验证多 seed 和独立的最终测试集。反复用于架构选择的 high 结果应
当视为探索性评估，不能再充当完全未接触的最终泛化证据。

R-GCN 原文用共享基矩阵缓解稀有关系过拟合，支持参数共享这一设计动机，
但并未证明本项目或 low→high 一定获益：
[Modeling Relational Data with Graph Convolutional Networks](https://arxiv.org/html/1703.06103v4)。
局部结构分布变化可导致 GNN 泛化失败的理论和实验参见
[From Local Structures to Size Generalization in Graph Neural Networks](https://proceedings.mlr.press/v139/yehudai21a.html)。
后者研究图尺寸迁移，对本项目的相关性在于局部环境分布变化，而非声称两项
任务完全相同。
