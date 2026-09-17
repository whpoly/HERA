# Hetero 与 attention 的 checkpoint 解释对照

## 结论

**Hetero 提供了更清晰的关系类别；这不等于 GNNExplainer 的节点解释更稳定或更可信。** 每张图同时给出预测、解释保真度和跨随机种子的稳定性。Shared hetero 还可以输出逐缺陷的潜在 readout 值，其均值严格还原模型预测，但这些值不是可唯一确定的物理缺陷能量。

本次完成 MoS₂、WSe₂ 各 5 个预先固定的 high-density 结构（4、9、14、19、24 缺陷），合计 3 个结构/模型组合。所有预测模型使用现有 checkpoint，未重新训练；优化的仅是解释掩码。

- 打开 `viewer.html`：切换训练组、材料、缺陷数和解释方式；悬停节点查看位点 ID、元素、类型、掩码不确定性及单点干预结果。
- `figures/`：每个对照的 PNG 和 PDF，可用于 PPT；`quality_overview.png` 汇总解释质量。
- 各训练组 `cases/`：逐节点 CSV、3 次掩码 NPZ、含重要性的 ASE/OVITO extxyz、逐样本 JSON。`quality_metrics.csv` 为完整数值表。

## 对照口径

1. **mixed_low**：采用本次新下载的 mixed-low attention、independent hetero，与 mixed-low shared hetero 对照；训练/验证/测试 ID 及顺序、标签 scaler 均已核对相同。新下载的 attention 是 BatchNorm 兼容版本，下载时间不代表网络架构更新。它保存的 4.052 meV 属于 low 测试集，不能与 high 测试集的 51.59 meV 混用。
2. **mos2_single**：较新的全程 LayerNorm attention（checkpoint 保存的 high MAE 56.992 meV）与同一 MoS₂ 划分的 independent hetero。未找到可核验的 MoS₂-only shared 权重，不用 mixed 权重代替该列。
3. **wse2_single**：全程 LayerNorm attention（46.233 meV）、independent hetero、shared hetero；训练划分相同。
4. 各组使用完全相同的对应 high 结构、位点顺序、缺陷标签和坐标。不同模型的构图/归一化/readout 可能不同，因此这是**现有 checkpoint 的比较**，无法单独归因于参数共享。全程保留各模型自己的边，未改成全缺陷连接图。
5. 新 attention 51.59 meV 那次仍需对应 checkpoint 才能建立准确关联；不把其他权重改名成该次运行。

## 解释质量实测

下面的 MAE 仅来自每种材料选定的 5 个结构，不是完整 500 个结构的测试 MAE。掩码保真误差越小代表更接近原模型输出；排名相关、Jaccard 越高代表重复性越好。Top>随机/Bottom 表示清零前 10% 高分节点造成的输出变化超过随机/低分节点的样本数。

|训练组|材料|模型|样本数|样本 MAE / meV|掩码保真误差 / meV|排名相关|Top10% Jaccard|Top>随机|Top>Bottom|
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
|mixed_low|MoS2|Attention|1|212.67|0.42|0.936|0.519|1/1|1/1|
|mixed_low|MoS2|Independent hetero|1|1.01|5.24|0.837|0.213|0/1|1/1|
|mixed_low|MoS2|Shared hetero|1|3.81|1.26|0.580|0.937|1/1|1/1|

软掩码能重现预测，并不能单独证明节点排名正确：在 LayerNorm 和冗余信息存在时，很多不同掩码可能保留同一预测。若 Top10% 稳定性低，或高分节点的清零效果不优于低分/随机，就不宜把颜色最亮的位点当成已验证的物理机制。

## 图的读法

- **GNNExplainer**：解释冻结模型的预测，而非 DFT 标签；每个原始晶格位点一个标量掩码。所有模型共用 0–1 色标，无逐图 min-max 放大。展示 3 个 seed 的均值，CSV 中保留各次值及标准差。
- **单节点干预**：逐个将初始节点 embedding 清零，记录输出变化。图用绝对变化，悬停保留正负号；每个结构的多个模型共用色标，使用 log(1+x) 以显示数量级差异。它提供无需随机优化的模型敏感度核对，但不等于删除真实原子的能量。
- **关系类别**：aa=普通→普通，ad=普通→缺陷，da=缺陷→普通，dd=缺陷→缺陷。分别在全部消息层清零该关系的节点消息分子，保留门控分母、边更新和几何，比较预测变化。它不等于删键，四个变化也不能相加当作总能量。
- **逐缺陷 readout**：shared energy-mean 模型的每个实际缺陷都有一个潜在标量，满足预测 = 这些值的平均数。可组织“哪些缺陷的内部输出偏高”的叙述；不可视为真实单缺陷形成能，也不可与 GNNExplainer 分数混用。
- **Attention native 权重**：另外展示全图 pooling 权重。Attention 本身也使用节点类型信息，故所有模型的图都标注普通节点、替位和空位；不会只给 hetero 加标签而制造可读性优势。权重本身不是已验证的因果归因。

## 方法与验证

采用 [GNNExplainer](https://arxiv.org/abs/1903.03894) 的 [PyG 实现](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/explain/algorithm/gnn_explainer.html)，`explanation_type=model`，node mask=object，不优化 edge mask。150 epochs、lr=0.03、size=0.01、entropy=0.001，seed=123/456/789；每个位点同一 mean 正则，不按节点类型重复平均。超参数对所有模型相同，本次未进行大规模超参数搜索。

掩码施加在初始的 64 维 learned embedding 上。原始 92 维空位特征全为零，直接乘原始特征会让空位无论如何都不受掩码影响；改在 embedding 层干预后，测试确认空位掩码存在梯度。模型参数全部冻结且 eval 模式，结构、节点类型、边和距离固定。原始模型与 wrapper 未加掩码的输出逐例一致；预测器权重在解释前后逐张量不变；关系 hook 遇到异常也会清理；逐缺陷 readout 的均值已校验还原预测。

样本选择为每个缺陷数中 SHA256(`20260917:source_id`) 最小的一个，与模型误差和热图外观无关。随机干预重复 12 次（5%、10%、20% 节点比例）。这是小样本可视化与机制诊断，不是用户可读性实验，也不是完整的因果验证。

## 复现

在 HERA 的父目录运行（已有 ML 环境）：

```powershell
python -m HERA.explain.checkpoint_comparison --output HERA/results/hetero_explainer_20260917/mixed_low
python -m HERA.explain.checkpoint_comparison --cohort single_material --materials MoS2 --models attention hetero_old --output HERA/results/hetero_explainer_20260917/mos2_single
python -m HERA.explain.checkpoint_comparison --cohort single_material --materials WSe2 --output HERA/results/hetero_explainer_20260917/wse2_single
python -m HERA.explain.enrich_comparison HERA/results/hetero_explainer_20260917/mixed_low HERA/results/hetero_explainer_20260917/mos2_single HERA/results/hetero_explainer_20260917/wse2_single
python -m HERA.explain.render_comparison HERA/results/hetero_explainer_20260917
python -m unittest HERA.tests.test_checkpoint_explainer -v
```

## 权重来源

- **mixed_low / Attention**：`C:\Users\User\Desktop\HERA\logs\bench_2dmd_low_all_models\alignn\2dmd_low\attention\seed123_best_checkpoint.pth`；epoch 222；SHA256 `3d4cb96eb2bd6de93a78df4ad0b2576a8687f62b23378bd2de6651fd46118826`；归一化配置 `{"hetero_node_norm": "layernorm", "alignn_feature_normalization": "batchnorm", "alignn_legacy_residual_norm": true}`。
- **mixed_low / Independent hetero**：`C:\Users\User\Desktop\HERA\logs\bench_2dmd_low_all_models\alignn\2dmd_low\hetero\r0\norm_layernorm\seed123_best_checkpoint.pth`；epoch 339；SHA256 `7cbc2f241ac269cf3d8eaa912d8e798d9e3e990efc155ad8667ba07679343e56`；归一化配置 `{"hetero_node_norm": "layernorm"}`。
- **mixed_low / Shared hetero**：`C:\Users\User\Desktop\HERA\logs\hetero_energy_mean_benchmark\alignn\2dmd_low\hetero\r0\features_layernorm\pool_defect_energy_mean\relations_shared_residual_rank8\seed123_best_checkpoint.pth`；epoch 303；SHA256 `aaf4995019b6dbe70d4da0c5aa65ff9eefead9f37e49f4b64c7cc1a048548a49`；归一化配置 `{"hetero_node_norm": "layernorm", "hetero_feature_norm": "layernorm"}`。
