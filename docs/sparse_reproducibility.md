# MEGNet sparse 可复现性检查

在 HERA 的父目录运行，`--checkpoint` 指向一份本项目生成且可信的 sparse 权重。
脚本使用它的配置、数据划分和 scaler，分别为每个 device 启动两个独立 Python 进程。

```bash
python -m HERA.check_sparse_reproducibility --checkpoint HERA/logs/bench_2dmd_low_all_models/alignn/2dmd_low/sparse/seed123_best_checkpoint.pth --device cuda:0 cpu --steps 100 --samples 32
```

上面的 checkpoint 路径来自已有 high_test_predictions 的记录；文件移动过时需替换路径。
数据默认读取 HERA 同级的 `dataset/2d-materials-point-defects-all`，可用 `--data-root` 覆盖。
输出保存到 `HERA/logs/sparse_reproducibility/<时间和唯一编号>/report.json`。
已有 benchmark 和 checkpoint 不会被覆盖。

## 检查内容

1. 对比代码、依赖、设备、atom_init、checkpoint、样本 ID、图张量和 scaler。
2. 加载同一 checkpoint，对选定的 low 验证样本重复推理三次，再跨进程比较。
3. 从固定 seed 的全新初始权重开始，核对权重 hash 和每一步 batch 顺序。
4. 比较每一步前向结果、loss、梯度和更新后的权重，报告第一次不同的步骤。
5. 报告首步梯度、最终权重、验证预测的最大绝对差。

这是一项短程定位实验：默认只取 checkpoint 中前 32 个 train 和前 16 个 val 样本，
采用 FP32、原始配置的固定初始学习率，不启用学习率调度、early stopping 或 AMP。
它不等同于重跑完整训练，子集验证 MAE 也不是模型性能 benchmark。
Python 哈希种子在子进程启动前固定为 0；模型和 DataLoader 使用 checkpoint 保存的 seed。

## 如何解读

- `EXACT_SHORT_RUN`：本设备上两次检查的所有被比较数值逐位相同；不保证更长训练或跨设备也相同。
- `NUMERICAL_DIFFERENCES`：输入控制相同，但推理或训练数值不同。先看 `first_training_difference`，
  再看 `tensor_differences` 的大小。它不等同于已证实存在有意义的最终 MAE 波动。
- `INPUT_OR_ENVIRONMENT_MISMATCH`：先解决图、顺序、初始化或环境差异。
- `ERROR`：查对应 worker JSON 和日志；严格确定性模式可能遇到不支持的算子。

默认 `--determinism strict` 对 PyTorch 已知的不支持确定性实现的算子报错。
`--determinism warn` 可用于对照现有 main.py 的警告模式。
第三方算子可能不受这一开关完整控制，严格模式没有报错也不能替代实际数值比较。
成功完成检查返回 0（包括检测出数值差异），执行错误返回 1。

## 完整实验的要求

短程检查之后，完整训练仍需固定代码、依赖、输入图、split、batch 顺序和初始权重，
比较所有 epoch 的验证曲线、学习率、early stopping、best epoch 和最终逐样本预测。
同 seed 的完整重跑用于检查可复现性；不同 seed 的多次运行用于衡量统计稳定性。
两者用途不同。不要根据最终 high 测试集反复选择 seed 或 checkpoint。
跨机器还应使用固定 checkpoint、相同输入图对比逐样本预测，并预先设定可接受误差。

参考：[PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html)。

## 本地验证（2026-09-15）

7 项诊断测试和 3 项原 sparse 配置测试通过。使用本地 mixed checkpoint、32 个训练样本、
16 个验证样本、每进程 100 步，严格模式下的两个独立进程对照结果：

- GPU（RTX 5060 Ti，PyTorch 2.11.0+cu128 / PyG 2.7.0）：所有输入控制一致，
  固定 checkpoint 推理一致；首步梯度最大差 `7.45058e-09`，100 步后权重最大差
  `3.12454e-05`，验证预测最大差 `1.85966e-05`。
- CPU（单线程）：上述推理和 100 步训练均逐位一致。

报告位于 `logs/sparse_reproducibility/20260915_021549_556655/report.json`。
这将该次短程实验的数值分歧定位到 GPU 反向计算，但未定位具体算子，
也没有证明它足以解释历史 high MAE 从 0.085 到 0.191 的差距。
