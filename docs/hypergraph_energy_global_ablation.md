# HyperALIGNN：逐缺陷贡献读出与 global_only 对照

2026-09-13。增加两个独立选项，保留现有默认配置与旧 checkpoint 的解释方式。

| 选项 | 含义 |
|---|---|
| `--hypergraph-pooling defect_mean` | 原版：`MLP(mean(h_defect))` |
| `--hypergraph-pooling defect_energy_mean` | 新版：`mean(MLP(h_defect))` |
| `--hypergraph-updates local_global` | 局部超边与全 defect 超边都参与更新 |
| `--hypergraph-updates global_only` | 只有全 defect 超边参与超图更新 |

`none` 和 `local` 仍可使用。更新消融选项仅用于 v3 HyperALIGNN。
两种 mean 读出均可与四种更新方式组合。新的逐缺陷读出也接入其他现有
v3 hypergraph 主干，但以下实验固定 `--model alignn`。

## 实现

两种读出都使用相同的 `64 -> 128 -> 64 -> 1` MLP，主干、物理邻居图、
归一化和预测头参数量相同。同一随机种子下，全部初始权重相同。
新读出先按真实 defect mask 取节点，再应用 MLP，最后按每张图的真实
缺陷数平均。不会重复计算同时属于局部和全局超边的 defect，也不会直接
平均 pristine 节点。标签、目标标准化、损失和 checkpoint 选择规则不变。
每个 defect 的输出是通过结构级标签学出的潜在贡献，没有新增单缺陷标签。

`global_only` 在 node-to-edge 和 edge-to-node 两次 softmax 之前过滤掉
局部超边的关联，因此局部超边不会进入任一 attention 的分母。
超图块只更新实际 defect，pristine 节点在该块内保持恒等路径。完整物理图
仍然参与 ALIGNN/GCN 更新，pristine 信息仍可通过物理消息传递影响预测。
完整超边元数据和区域 embedding 保留，确保切换更新方式不改变节点标记
和物理图输入。超图块内的 FFN 与残差门仍按原来的规则作用于活跃节点。

默认仍为 `defect_mean`，不传 updates 时仍为 `local_global`。
旧 v3 缺少 pooling 字段时仍恢复 `hierarchical_attention`；旧 v2 仍恢复
`region_mean`，不接受新的读出和更新消融选项。

## 四组训练

同步修改的代码后，在服务器的 `/home/wuhao`（HERA 的父目录）运行：

```bash
for pooling in defect_mean defect_energy_mean; do
  python -m HERA.main \
    --model alignn --dataset 2dmd_mos2 --mode hypergraph \
    --hypergraph-schema defect_global_attention_v3 \
    --hypergraph-pooling "$pooling" \
    --hypergraph-updates local_global global_only \
    --hypergraph-radius 3.0 --seed 123 --epochs 500 --device cuda:0 \
    --resume --run-dir HERA/logs/2dmd_mos2_hypergraph_energy_global || break
done
```

这会依次运行四组，而不是同时占用四份 GPU 显存。`--resume` 跳过已经写出
最终 TEST 结果的任务；它不表示从中途 epoch 恢复优化器状态。

结果目录为：

```text
HERA/logs/2dmd_mos2_hypergraph_energy_global/alignn/2dmd_mos2/hypergraph/
  defect_global_attention_v3/
    pool_defect_mean/
      updates_local_global/
      updates_global_only/
    pool_defect_energy_mean/
      updates_local_global/
      updates_global_only/
```

每个目录分别保存 `seed123_history.csv`、`seed123_best_checkpoint.pth`
和 `seed123_test_predictions.csv`。Native LOO 标签和结果汇总列也按两种
读出及四种更新模式分别保存。

固定 updates 比较两种 pooling，判断读出的影响；固定 pooling 比较
`global_only` 和 `local_global`，判断已有全局消息时局部更新的增益。
训练前固定实验方案和种子，checkpoint 继续由 low 验证集选择。

## 验证

完整 160 项单元测试通过，覆盖 global-only 两次归一化隔离、跨 defect
梯度路径、超图块中 pristine 恒等路径、物理主干中 pristine 梯度、
逐缺陷贡献与正确计数、批量一致性、空边/单缺陷、旧配置恢复及独立目录。

四个目标组合还在合成结构上完成了各 3 步 CUDA FP16 优化，loss 和梯度
均有限。使用完整 64 维、3 ALIGNN + 3 GCN 配置，参数量均为 659,217；
单图与批量预测最大差异为 5.96e-8。没有运行完整训练，尚无新版本的
low-to-high 性能结论。

```bash
python -m unittest HERA.tests.test_hypergraph_energy_global HERA.tests.test_hypergraph_update_ablation
```
