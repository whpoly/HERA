# HyperALIGNN：固定 defect mean pooling 的消息传递消融

2026-09-11。为检查此前 hypergraph 变差是否与超图消息传递有关，增加
`--hypergraph-updates`，在相同 v3 架构中选择三种更新方式：

| 参数 | 超图更新 |
|---|---|
| `none` | 跳过整个超图更新块，包括其 FFN |
| `local` | 仅通过每个 defect 的局部超边更新 |
| `local_global` | 局部超边 + 所有 defect 共享的一条全局超边 |

三者使用相同的物理距离/角度图、attention ALIGNN 主干、LayerNorm、区域
embedding 和预测头。最终仅对**实际 defect 节点的表示取平均**，再经过
`64 → 128 → 64 → 1` 的 MLP。没有额外的 pristine/global pooling 或浓度分数
读出分支。这里是先平均表示再预测，不是先逐 defect 预测标量能量再平均。

`none` 是这个 HyperALIGNN 架构内部的基线；仍保留区域 embedding 和相同
预测头，因此不等同于原来的 `--mode attention`。所有模式都实例化相同模块，
同一随机种子下初始权重一致；`none` 不执行的超图块不会获得梯度。

`local` 在 node→edge 和 edge→node 的 softmax **之前**排除全局超边的关联，
全局超边不会参与消息或归一化。完整超边元数据仍保留，用于识别真实 defect
和局部中心，避免同时改变 pooling 的节点集合。局部超边仍包含中心 defect
及其 3 Å 内 pristine 节点，不包含其他 defect；本次没有同时改变局部拓扑规则。
没有远场 pristine 超边；远场原子仍参与原来的物理图消息传递。

## 运行

在 `HERA` 的父目录、已安装项目依赖的 Python 环境中执行：

```powershell
python -m HERA.main --model alignn --dataset 2dmd_mos2 --mode hypergraph --hypergraph-schema defect_global_attention_v3 --hypergraph-pooling defect_mean --hypergraph-updates none local local_global --hypergraph-radius 3.0 --seed 123 --epochs 500 --device cuda:0 --run-dir HERA/logs/2dmd_mos2_hypergraph_ablation
```

命令依次训练三个版本，沿用相同的 low→high 划分、训练设置和 low 验证集
checkpoint 选择规则，最多 500 epochs，并按现有规则提前停止。只训练局部版时，
将 `--hypergraph-updates none local local_global` 改成 `--hypergraph-updates local`。

每个版本的 checkpoint、history 和 test_predictions 分别保存在：

```text
HERA/logs/2dmd_mos2_hypergraph_ablation/alignn/2dmd_mos2/hypergraph/
  defect_global_attention_v3/pool_defect_mean/
    updates_none/
    updates_local/
    updates_local_global/
```

这三个选项目前仅支持 HyperALIGNN 的 v3 + `defect_mean` 组合，也接入了
`native_initial_relaxed_leave_one_out` 入口。不传该参数时保留原有更新方式和
结果路径；旧 checkpoint 缺少该字段时也按旧逻辑恢复。

## 验证及解释范围

自动测试覆盖：关闭块时恒等映射、局部模式隔离不相交缺陷环境、两次
softmax 均排除全局关联、相同初始化与输入、实际 defect mean 读出、批处理
一致性、反向传播、checkpoint 恢复及结果路径隔离。完整测试命令：

```powershell
python -m unittest discover -s HERA/tests
```

验证结果：完整测试共 139 项通过；三个版本均在 8 个真实 low 结构上完成
5 步 CUDA 训练，loss 和梯度有限，混合精度前向/反向正常，单样本与批量
预测的最大差异不超过 `4.77e-7`。记录在
`logs/hypergraph_v3_validation/update_ablation_cuda_smoke.json`。

此版本是为定位问题建立的对照，尚无完整 low→high 性能结果。此前
high MAE 0.250031 属于分层 attention pooling 的旧 v3，不能当作这里任何
一个新训练版本的结果。短 GPU 检查只能验证代码可训练，不能证明泛化改善。
