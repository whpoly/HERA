# Native Hypergraph：local-only 与扩大半径的后续实验

2026-09-30（香港时间）。状态：500 轮训练和完整测试核验已于 19:36 完成，选择验证 MAE 最优的第 492 轮权重；Test MAE 0.109406 eV，RMSE 0.407796 eV。结果和逐样本分析见[专项结果文档](C:/Users/User/Desktop/HERA/results/native_hypergraph_local_r6_seed123_prepared_20260930/report.md)。未创建定时任务。

本次按 **local-only、6 Å** 测试：关闭 global defect 超边的消息传递，将局部超边半径由 3 Å 增加到 6 Å。本次运行已固定为 6 Å。

| 设置 | 已完成基线 | 已完成新实验 |
|---|---|---|
| 超图更新 | local_global | local |
| local 超边半径 | 3 Å | 6 Å |
| 距离图 cutoff | 6 Å | 6 Å |
| readout | defect_energy_mean | defect_energy_mean |
| 输入 / 骨干 | WAS184，hidden64，3 ALIGNN + 3 GCN | 相同 |
| 划分 | strict，seed123，1818/604/604 | 与基线相同且顺序核对通过 |
| 训练 / 权重选择 | 500 轮，AdamW，验证 MAE | 相同 |
| Test RMSE | 0.429421 eV | 0.407796 eV |

## 关闭 global 的实际含义

复用已有 `hypergraph_updates=local` 实现。在 node→hyperedge 和 hyperedge→node 两次 attention 归一化之前，均排除 global 的关联；global 不产生消息，也不参与 attention 分母。

当前 v3 数据结构保留 global 关联作为“哪些节点是实际缺陷”的元数据，供中心识别与 defect readout 使用。因此这里是从有效消息传递中移除 global，并非从序列化图数据中删除它。这样不需要同时重写缺陷身份识别及旧权重的加载规则。

局部超边仍包含中心缺陷与半径内正常宿主，远处宿主继续通过距离/键角骨干传递信息。扩大半径会扩大局部超边成员，也会使相应宿主的超图区域标签由 far 改为 near；不会把 `local_radius` 改成 6，该字段仍为 0，保持物理构图的缺陷/宿主标记协议。

## 配置与运行方式

已准备的协议：[protocol.json](C:/Users/User/Desktop/HERA/results/native_hypergraph_local_r6_seed123_prepared_20260930/protocol.json)。

本次由 `run_native_hypergraph_job` 监督脚本启动训练和结束后的核验，实际训练参数如下（已经完成，请勿重复提交到原结果目录）：

```powershell
& 'C:/Users/User/.conda/envs/hera/python.exe' -u -m HERA.scripts.run_native_hypergraph_paired --hypergraph-updates local --hypergraph-radius 6 --output 'C:/Users/User/Desktop/HERA/results/native_hypergraph_local_r6_seed123_prepared_20260930'
```

命令末尾加 `--prepare-only` 仅核对协议与划分，不训练。新半径或更新模式必须指定独立 `--output`；已有目录协议不一致会报错，已有不完整训练由保护选项阻止覆盖。

这次同时改变 global 更新和 local 半径，结果检验两项合并后的效果。如果要分别归因，可补齐 `local、3 Å` 和 `local_global、6 Å` 对照；本次没有启动或安排这些对照。

## 已完成的检查

- 16 项现有超图更新与逐缺陷读出测试通过，包括 global 在两次 softmax 中均被排除、局部消息隔离和读出一致性。
- 使用本次完整模型配置做合成结构前向与反向检查：半径从 3 Å 改到 6 Å 后，4.5 Å 处的宿主进入局部超边；物理图节点特征、边、距离及方向向量相同。
- 全部 6 个超图更新块的活跃 global 关联数均为 0；readout 仍只读取一个实际缺陷；输出与梯度均有限。
- 默认配置与已完成的 3 Å local_global 协议一致；新配置只有 `hypergraph_radius` 和 `hypergraph_updates` 两个字段变化。

检查记录：[smoke.json](C:/Users/User/Desktop/HERA/results/native_hypergraph_local_r6_seed123_prepared_20260930/smoke.json)。这些实现检查与最终精度结果分开保存，完整 MAE/RMSE 及逐样本分析已写入专项结果文档。启动前还通过脚本编译、旧完整结果的只读审计，以及新报告不会替换旧 3 Å 结果行的内存检查。
