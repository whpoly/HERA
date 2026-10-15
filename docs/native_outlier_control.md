# Native 离群值控制对照

后续已按用户要求改为先筛选完整数据集、冻结清单，再重新 split 的
[`filtered_dataset` 实验](native_filtered_dataset.md)。下面保留较早的
`outlier_control` 协议，其 valid/test 不删样本。

2026-10-12。已完成实现、真实数据清单核验和 80 项相关测试；未启动新的 benchmark 训练。

## 新发现

在当前 strict 的 1818 条训练数据中，按同一缺陷构型系列的训练标签中位数和 MAD 检查，
有 43 条高于下面固定的统计阈值。其中 **36 条本身低于 15 eV**，因此统一设置
15 eV 上限无法涵盖这一类“相对于本系列偏高”的样本。

43 条中有 36 条 interstitial、6 条 antisite、1 条 vacancy；文件序号分布是
POSCAR0 32 条、POSCAR1 9 条、POSCAR2 2 条。这个分布是筛查结果，不是筛选条件。
这些可能包含有效的未弛豫、高应变结构，不能据此认定为错误 DFT 标签。

## 两项独立控制

**同系列上尾筛选**：去掉文件名中的 POSCAR 序号定义配置系列，先取 strict 已保留的
训练样本。系列至少有 4 条训练记录时，只拟合一次：

```text
m = median(E_train)
sigma_MAD = 1.4826 * median(abs(E_train - m))
upper = m + max(6 * sigma_MAD, 1 eV)
```

仅从训练集排除 `E > upper` 的样本，边界保留。训练样本不足的系列不执行这项排除。
不迭代重拟合，不按 POSCAR 序号或模型预测残差删除，不叠加全局 15 eV 上限。
该阈值沿用已有联合筛选中的 6 倍 MAD、1 eV 下限，但本对照不要求同时满足短键条件。
因此它是更强的统计上尾控制，不是已证实的物理有效性边界，也不是原文筛选复现。
seed123 的 293 个训练系列中有 168 个满足拟合样本数。

**Huber 损失**：对每一个训练样本的预测残差使用稳健损失，转折点固定为实际
**1 eV**。它在大残差区间按线性增长，降低极端残差相对于 MSE 的梯度影响。
[PyTorch HuberLoss 定义](https://docs.pytorch.org/docs/2.14/generated/torch.nn.HuberLoss.html)

为了与原 MSE 的小误差段一致，实际训练采用标准化空间的 `2 * Huber`。
若训练集目标标准差为 s，标准化残差为 r，则转折点 d = 1 eV / s：

```text
L(r) = r^2                         if |r| <= d
       2*d*|r| - d^2               otherwise
```

这保留小残差的 MSE 数值和梯度，并限制大残差对模型输出的导数。
它不保证网络所有参数的梯度有统一上界，也不截断真实标签或预测值。
常量目标的标准差回退与现有 Scaler 一致。历史 `smooth_l1` 和 `huber` 配置别名仍保持
原来在标准化空间 beta=1 的含义；新配置名称为 `huber_ev`。

## 实验矩阵

Hetero-WAS no_dd 和 WAS_X 各运行下列四组，共 **8 次训练**：

| 目录 | 训练样本，seed123 | 训练损失 |
|---|---:|---|
| strict | 1818 | MSE |
| family_mad | 1775 | MSE |
| strict_huber | 1818 | Huber，1 eV |
| family_mad_huber | 1775 | Huber，1 eV |

所有组保留同样的 **604 条验证、604 条测试**及原顺序。架构、64 维隐藏层、原有
readout、优化器配置、batch16、500 epochs 和验证 MAE 选 checkpoint 的方式不变。
每个训练子集拟合自己的目标 scaler，同一子集的两个模型、两种损失必须使用一致的
scaler。筛选后每 epoch 的优化器更新次数略少；这是数据筛选对照的一部分。

汇总同时报告完整集、训练统计规则标记的高能组、未标记组的 MAE/RMSE。
seed123 验证中有 13 条、测试中有 17 条被诊断标记，**它们全部保留并计入完整分数**。
未标记组包括训练参照不足或未见过的系列，不应视为已确认的高质量数据。
全部 8 组都报告，不按测试分数挑选阈值或删除点。

Huber 可能改善多数样本的 MAE，同时使尾部 RMSE 变差；减少训练上尾样本也可能
削弱对有效高能构型的覆盖。是否采用，需要比较同一完整测试集上的实际结果。
此前已经检查过该数据集的测试误差，因此这是探索性对照，不是独立确认实验。

## 服务器运行

同步 `main.py`、`training/losses.py`、`training/trainer.py` 和
`scripts/run_native_filter_benchmark.py`，在 **HERA 父目录**运行：

```bash
python -m HERA.scripts.run_native_filter_benchmark --suite outlier_control --seed 123 --epochs 500 --device cuda:0 --run-dir HERA/logs/native_outlier_control_v1
```

命令自动冻结清单，顺序执行八次训练，再用各 checkpoint 的权重和 scaler 做完整评估。
输出 `filter_comparison.csv` / `filter_comparison.md`、`filter_evaluation.json`、
各模型逐结构预测，以及 `seed123_family_mad_review.csv`。

`--dry-run` 只打印计划；`--prepare-only` 只生成/核验清单；`--evaluate-only`
仅在已完成训练后重算。不同实验协议需要不同运行目录。普通重跑会跳过兼容的已完成
训练，未完成训练按现有 `--resume` 规则从头重启。

原来的 `--suite energy_scope`（默认）仍保留原 15 eV 对照及目录身份。
本实验未改动 strict 默认规则、旧权重或 IMP2D interstitial 筛选。

## 已验证

- 80 项相关测试通过，包括 Huber 的物理单位、梯度限幅、小误差段与 MSE 一致、
  Trainer 实际反传分支、历史损失兼容、CLI 参数和不同损失的 resume 拒绝。
- 系列阈值只拟合训练标签；改变留出标签不会改变训练选择或拟合阈值。
- 八组模拟 checkpoint 汇总验证了完整样本数、共有诊断子组、MAE/RMSE 和 scaler 身份。
- 真实源数据哈希与 strict 缓存一致；四组清单保留相同验证/测试 ID 和顺序。

本地真实数据准备记录位于 `results/native_outlier_control_20261012_prepared`，
其中 `preparation_summary.json` 是筛选计数，**不是新的训练准确率结果**。
