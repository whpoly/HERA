**HERA hetero / hypergraph 思路与实现审查（2026-09-10）**

结论：当前 hetero 和 hypergraph 都有真实、成立的图学习实现。尚未成立的是更强的科学主张，例如“超边因此表示真实多体相互作用”“四种关系对应四种物理作用机制”“复杂结构一定改善浓度外推”。需要分别检验形式、物理归纳偏置和实验归因。

本次检查当前代码、新旧配置、已有 MoS₂ 结果和原始标签；运行了 56 项相关单元测试及三个合成反例，并补充一个四参数组成基线。没有进行新的 GNN 全量训练。由于未指定要对照的具体论文，下文采用 PyG 异构图、HGNN、AllSet、ALIGNN 和 Crystal Hypergraph Convolutional Networks 作为参照。

**形式上到底是什么。**

| 项目 | 当前实现 | 与常见定义的关系 |
|---|---|---|
| Hetero | 两类节点，四种有向关系 aa、dd、ad、da；关系独立参数，按关系聚合后用节点类型对应的 FFN 融合 | 成立的区域/角色异构消息传递；异构类型不必具有不同的特征维数 |
| Hypergraph v2 | 每个缺陷一条 singleton core、一条局部超边，另有远场 pristine 超边；使用 PyG HypergraphConv | 成立的超图；singleton 本身不增加节点之间的通信 |
| Hypergraph v3 | 一条所有缺陷共享的 global 超边，每个缺陷一条局部超边；两次独立归一化的节点→超边→节点 attention | 属于基于关联关系和集合聚合的超图网络；是特定设计，不是对 AllSet 的逐行复现 |
| HyperALIGNN 等混合模式 | 物理图更新与超图更新交替 | 几何信息可以先由物理骨干写入节点表示，再被超图使用 |
| 纯 `--model hypergraph` | 原子特征、区域类型、超边成员关系进入网络 | 当前没有连续距离/角度输入；默认 defect_mean 下远处 pristine 没有到预测的路径 |

异构图按节点/关系类型选择消息函数，符合 [PyG 官方异构图说明](https://pytorch-geometric.readthedocs.io/en/latest/notes/heterogeneous.html)。HGNN 使用关联矩阵定义传播，[AllSet](https://arxiv.org/abs/2106.13264) 则用两阶段多重集合函数组织超图计算；你的 v3 与后一类思路更接近。参见 [HGNN 原论文](https://arxiv.org/abs/1809.09401)。

一个超边也可以表示为一个辅助节点，并建立“原子—超边”的二部图。你的计算可以按这种方式实现，这不否定它是超图，但不能据此声称超图拥有任何普通消息传递框架都无法表达的能力。区别要落在集合划分、几何输入、参数共享与聚合函数上。

材料超图论文中的一种更具体实现，是三元超边带角度特征、配位 motif 超边带局部形状描述符。你的超边类型 embedding 和由成员节点生成的上下文没有显式提供这种 motif 几何。这个区别比是否调用名为 HypergraphConv 的层更重要。参见 [Crystal Hypergraph Convolutional Networks](https://www.nature.com/articles/s41524-025-01826-9)。原始 [ALIGNN](https://arxiv.org/abs/2106.01829) 已经通过线图引入键角，因此不能把“用了超边”本身当作首次引入高阶几何的证据。

相关实现：[节点/关系与超边构造](/C:/Users/User/Desktop/HERA/data/converters.py:149)、[v3 消息传递](/C:/Users/User/Desktop/HERA/models/hypergraph.py:33)、[异构关系融合](/C:/Users/User/Desktop/HERA/models/alignn.py:421)。

**最值得处理的建模和实验问题。**

1. **全 defect 超边提供全局通信，但没有显式的缺陷间几何。**

   构造是 `E_global = D`，不依据任意两个 defect 的距离建立不同关系。它将任意位置的缺陷集合压缩到有限维表示，再回传各节点。物理骨干能够间接提供几何，因此不能说混合模型完全不懂距离；但这条全局分支本身没有距离衰减、方向关系或明确的远距离解耦约束。它更适合被描述为“缺陷集合的全局上下文”，而不是已经验证的长程相互作用模型。

   建议在完全相同骨干下对照：无 global、一个缺陷虚拟节点、带周期距离的 defect–defect 边、当前 global 超边。若要讨论物理作用范围，还应检查相同局部环境下改变缺陷间隔的响应。

2. **局部超边排除其他 defect，是有影响的假设。**

   当前 `E_local(d) = {d} ∪ {半径内 pristine}`。相邻的第二个 defect 不会直接进入这条局部超边。共享宿主和物理边仍可能传递它的信息，但局部集合不再是完整配位环境。对缺陷团簇、高浓度替位、相邻空位尤其需要验证这个选择。

   建议对照允许半径内全部位点加入、同时用中心标记保持“这个局部环境属于哪个缺陷”。此时必须显式保留中心 ID/关联角色，因为当前 `hyperedge_context` 会把超边内所有 defect 的表示平均为 centers，不能只改成员列表就沿用现有中心语义。半径 3 Å 也应当作为超参数或材料相关配位假设来验证。

3. **纯 hypergraph 存在确定的几何不可辨识性。**

   我将合成结构中一条 defect–host 距离从 1.0 Å 改为 2.5 Å，保持成员不越过 3 Å 边界。输入原子特征、超边关联与预测完全相同。由于纯模型不接收物理距离/角度，这个问题不能靠多训练修复。

   这项结论限定于纯模型和超图分支的直接输入，不等于 HyperALIGNN 有相同的整体退化。若把纯模型作为完整几何模型使用，应增加关联上的距离 RBF、明确的角度/配位描述，或保留物理骨干。

4. **hetero 的 r 扫描可能同时改变实际图边。**

   `_neighbor_lists` 对每个中心按邻居 `type` 分别截断。`mark_local_region` 又随 r 改变 type，所以“物理 cutoff 不变”不能推出“保留的邻居不变”。在合成结构上，固定 cutoff=4、每类上限=2，仅将 r 从 0 改为 2，边数从 28 变成 24。

   因而 r 消融混合了区域标签、边采样和潜在 pooling 的作用。建议先固定几何邻居表，再按 r 给同一组节点/边赋类型；或者把按类型采样明确当作单独实验因素。full/full_x 与 hetero/attention 的比较同样要核对实际节点、X 占位、类型标记及边集合，不能只核对 cutoff 和 max_neighbors 数值。

   代码位置：[类型相关截断](/C:/Users/User/Desktop/HERA/data/converters.py:260)、[区域标记](/C:/Users/User/Desktop/HERA/data/structure_utils.py:69)。

5. **异构类型的语义和参数量需要解释清楚。**

   r=0 的 defect 类是实际缺陷；r>0 则是缺陷与附近正常位点的并集。后者应称 defect-region/local-region，不能把 dd 解释成纯粹的缺陷—缺陷关系。`pool_type` 保留实际缺陷，新 HeteroALIGNN 最终按该标记读出，这部分已正确处理。物理层的类型选择则仍按区域划分。

   四种独立消息函数在形式上合理，但“不同消息函数”不意味着四种独立力或物理键。ad/da 独立也不会自动违反牛顿第三定律，因为这些是隐空间消息，并非预测的力。

   容量与数据效率是实际问题：建议比较“共享消息函数 + 类型 embedding”“共享主体 + 小型关系修正”和完全独立参数；同时报告参数量与各关系在 low/high 的出现率、邻居数、空关系比例。当前 HeteroALIGNN 的 line-graph 更新使用共享卷积，所以应描述为原子/键关系异构，不能声称每种角关系都有独立参数。

6. **关系均值、attention 和 defect mean 有数量信息与能量结构方面的取舍。**

   按关系平均/归一化有助于避免 host 数量淹没少量 defect，但相同消息复制多次时，归一化输出可能不变。当前 defect_mean 也不显式提供缺陷浓度或局部重叠度。物理节点表示、邻域变化仍可能编码这些量，因此“信息可能不足”应通过消融验证，而不是断言网络完全不知道浓度。

   已重新核验全部 MoS₂ 标签：`formation_energy_per_site = formation_energy / N_defects`，low/high 最大残差均在 1.4e-15 内。因此对缺陷平均有合理依据。但当前是 `MLP(mean(h_d))`，一般不等于 `mean(MLP(h_d))`。前者是灵活的结构级回归；后者配合局部消息才更接近逐缺陷贡献的平均。即使换成后者，若保留全局耦合也不能自动保证远距离可加性。

   建议以同样表示比较两种读出，以及 `组成基线 + 学习交互残差`。如需浓度特征，优先用有明确意义的缺陷/晶格位点比例；二维材料应谨慎使用包含任意真空厚度的体积归一化。不要为了修复计数丢失而直接把每缺陷能量乘上缺陷数。

7. **周期边界正确计算最短距离，还不等于完整的周期超图。**

   当前局部超边按晶胞内唯一索引和最短距离取成员，不记录同一宿主的多个周期像。对半径足够小的大超胞通常没有这个重复像问题；对小晶胞则可能改变配位重数。

   合成 4×20×20 Å 晶胞沿 x 原样复制三次，表示同一周期结构。原超边大小为 3，复制后每个局部超边大小为 4；未训练纯模型的预测差约 2.93e-6。差值本身没有物理精度意义，结构不等价才是反例。不能据此断言目前 8×8 的 2DMD 已发生同样错误。

   建议增加晶胞平移/原子置换/超胞复制一致性检查；需要小晶胞支持时显式保存周期像关联和几何，或清楚限定半径与晶胞大小的适用范围。

8. **缺陷身份需要参考结构，预处理目前偏向理想初始晶格。**

   vacancy 的 X 占位和 WAS 是合理的参考位点表示；X 周围的距离边及角度应称几何消息边，而不是实际化学键。默认 atom features 为当前元素，空位是零向量；被移除/替换的原元素由 WAS 显式提供，普通版只能部分依靠环境推断。

   `strucure_to_dict` 按三位小数分数坐标建字典，一些 2DMD 转换按这些坐标精确匹配参考晶格；`add_was` 也只保留匹配到的参考位点。当前加载器使用 `initial/*.cif`，与这个假设相符。不能无检查地推广到任意弛豫结构、重构、间隙缺陷或位点迁移，否则可能错误识别空位或丢失原子。若未来数据包含不同电荷态或形成能参考条件，也需审计这些条件是否已进入输入。

   代码位置：[坐标匹配](/C:/Users/User/Desktop/HERA/data/structure_utils.py:18)、[WAS 映射](/C:/Users/User/Desktop/HERA/data/structure_utils.py:145)、[初始结构与标签读取](/C:/Users/User/Desktop/HERA/data/datasets.py:219)。

**已有结果支持什么。**

同一 seed=123、同一已保存 checkpoint 切分：4746 low 训练、1187 low 验证、500 high 测试。新增基线只使用四种已知缺陷的数量比例，以训练集 OLS 拟合四个系数，无截距、无调参、不使用几何。它使用包括原位点元素在内的缺陷身份，适合作为信息充分的组成参考，不是对普通无 WAS 输入的严格等信息消融。

| 模型 | low 验证 MAE | high 测试 MAE（eV/defect） |
|---|---:|---:|
| 本次四参数组成基线 | 0.047379 | 0.131824 |
| 旧 HeteroALIGNN r0 | 0.002169 | 0.116433 |
| 旧 Hypergraph v2 | 0.002069 | 0.168732 |
| 旧 Attention ALIGNN | 0.002407 | 0.056992 |

GNN 的 low 误差显著更小，因此不能说它们只学了四种缺陷的线性组成。但是旧 v2 的 high 误差高于这个简单参考，现有结果不足以支持“超图改善浓度迁移”。旧 hetero 在 high 上略好于该参考，attention 改善更明显。旧模型的 low 验证值来自仓库已有分析；high 值本次从预测 CSV 重新计算。上述结果均不能替代当前全 LayerNorm/defect_mean hetero 或 v3 的完整训练结果。

low 的 5933 条中，1/2/3 个 defect 分别有 4/127/5802 条；high 的 4/9/14/19/24 个 defect 各 100 条。验证集仍主要是三缺陷，因此 low 的 0.002 级误差不检验高浓度下的计数、重叠和交互外推。随机种子只重复训练或低浓度切分，不能增加独立高浓度样本数。

现有 split 将 high 留作测试、scaler 只在训练集拟合，这两项设计正确。不过，若在反复查看同一 high 结果后改变架构，该 high 集应被视为探索性评估集；最后的泛化主张需要未用于架构选择的新材料、新浓度或预留结构组。不能因为 high 未进入反向传播，就认为设计过程没有接触测试信息。

**建议按这个次序补证据。**

1. 固定同一物理图、原子/参考信息、LayerNorm、读出和训练协议，建立共享消息函数基线；保留本次组成基线。
2. 只改变是否使用关系独立参数，验证 hetero 的独立贡献；再单独比较区域半径和采样方式。
3. 在同一物理 attention 骨干上比较：关闭超图、仅 local、仅 global、local+global。关闭时保留/控制区域 embedding 和预测头，避免引入新的混杂。
4. 用虚拟节点和带周期距离的 defect–defect 图检验 global 超边的必要性；再测试局部超边是否应包含其他缺陷、是否需要显式几何。
5. 单独比较平均表示、平均逐缺陷输出与组成残差读出；统计按缺陷数、最近缺陷距离、局部重叠、缺陷种类分组的误差，结合多 seed 和独立测试集。

现在可以稳妥地写：**“基于缺陷区域的关系消息传递，以及结合物理骨干的缺陷局部/全局超图消息传递。”** 不宜提前写成已证明的多体能量分解、唯一必要的高阶表示或普遍优于同构图的模型。

**复现与检查记录。**

- [复现脚本](/C:/Users/User/Desktop/HERA/logs/concept_audit_20260910/audit.py)、[原始结果 JSON](/C:/Users/User/Desktop/HERA/logs/concept_audit_20260910/results.json)、[组成基线 high 逐样本预测](/C:/Users/User/Desktop/HERA/logs/concept_audit_20260910/composition_baseline_test_predictions.csv)。合成探针使用随机初始化，只检验表示性质，不代表训练后误差。
- 本次运行 `test_hetero_alignn_relations`、`test_hetero_megnet_relations`、`test_hetero_physical_edges`、`test_hetero_alignn_transfer`、`test_hypergraph`、`test_hypergraph_v3`、`test_neighbor_type_cap`，共 56 项通过。这证明被测实现性质，不能替代物理正确性或泛化实验。
- 从 HERA 目录运行：`& 'C:\Users\User\.conda\envs\hera\python.exe' logs\concept_audit_20260910\audit.py`。脚本只拟合小型组成基线并写自己的审查输出，不训练或修改 GNN。
