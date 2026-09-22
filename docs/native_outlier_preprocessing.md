# Native 离群样本预处理

## 当前使用：严格版，剔除 44 个异常样本

按最新要求，切回完整严格版：剔除 **44** 个异常样本，保留 **3026** 个；不限制 POSCAR 序号。train/val/test 分别剔除 24/10/10 个，保留 1818/604/604 个。清单仍为 `results/native_clean_strict/manifest.json`。此前 POSCAR0/1 专用清单保留作为可选版本，本次命令不使用它。

在 **HERA 父目录**执行，先同步本次代码。现在只需一条命令，读取 native 数据时自动筛查，使用默认 batch size 运行全部指定非 hypergraph 模式，hetero 为 no_dd：

```bash
python -m HERA.main --model alignn --dataset native --mode full full_x attention attention_was hetero hetero_was was_x definet definet_was --alignn-hetero-dd drop --r 0 --native-preprocessing reference_v1 --alignn-hetero-feature-norm layernorm --alignn-hetero-node-norm layernorm --alignn-hetero-relations shared_residual --alignn-hetero-adapter-rank 8 --alignn-hetero-pooling defect_energy_mean --seed 123 --epochs 500 --device cuda:0 --atom-init HERA/atom_init.json --native-outlier-filter strict --run-dir HERA/logs/alignn_native_clean_strict --compact-logs --resume
```

同目录补测 no_aa_dd 时，把训练命令中的 `--mode` 改为 `--mode hetero hetero_was`，再加 `--alignn-hetero-aa drop`。

`--native-outlier-filter strict` 首次读取 native 原始数据时构建清单，直接保存在实验根目录的 `native_filter_manifest.json`；逐项原因和计数写到同目录的 `native_filter_seed123_removed.csv`、`native_filter_summary.csv` 等文件，不增加目录层级。同一次运行的所有模型/模式共享一次筛查结果。

再次运行会校验完整原始标签表和全部源 CIF 哈希，复用保存的清单，不重新拟合阈值。源数据、规则或所需 seed/CV 协议不匹配时停止，避免误用旧结果。它也能复用此前 `--native-filter-manifest` 已复制进同一实验目录的严格版清单；自动与手动方式生成相同的实验身份。

自动入口和 `--native-filter-manifest` 二选一。前者不需要预先运行脚本；后者仍可用于自定义阈值或 POSCAR 序号限制。`standard` 对应 v2 基础规则，当前选择为 `strict`。

日志根目录与这份严格版清单对应；不混用完整数据或 POSCAR0/1 专用版本的实验目录。`--resume` 跳过同配置下已完成的 seed；未完成、只有历史日志的 seed 从头训练，不是恢复优化器状态。以下保留 v2 基础版说明及严格版规则。

## 基础版 v2（保留用于复现）

进一步检查后的严格版使用 `--profile strict`，输出 `results/native_clean_strict`；使用方法见本文末尾。下文的 v2 数量和命令保留，便于复现上一版。

`scripts/preprocess_native_outliers.py` 生成固定的样本清单；`main.py --native-filter-manifest` 读取该清单。原始 CIF、标签 CSV 及旧实验目录保留。没有此参数时，数据加载与划分仍使用原来的流程。

基础版清单为 `results/native_clean_v2/manifest.json`（seed 123）。它是统计筛选后的 benchmark 子集，不代表被筛掉的 DFT 标签均已证实错误。早期 `native_clean_v1` 是开发中的粗分组草稿，未用于训练，不用于正式比较。

## 过滤规则

1. 先按原来的 seed/CV 方法划分完整数据，保存每个原始样本的归属及顺序。
2. 结构检查使用 ASE 读取原始 CIF：排除无效标签、无法读取或无效的几何、元素/原子编号/缺陷组成不一致、非完整占据，以及最短周期性原子间距小于 **1.0 Å** 的样本。
3. 能量按 **材料 + 缺陷类别 + 加入元素 + 移除元素** 分组。例如 BN 的 B 间隙与 N 间隙分别统计，AlN 的 Al 替 N 与 N 替 Al 分别统计。相同化学组成的不同间隙位置可以进入同一组。
4. 每组只用原始训练集中通过结构检查的标签拟合阈值。至少需要 **8** 个训练样本；更少或训练中没有该组时，仅执行结构检查，不借用验证/测试标签拟合。
5. 设训练组第一、第三四分位数为 Q1、Q3，IQR = Q3−Q1，保留区间为：

   `Q1 − 3 × max(IQR, 1 eV) ≤ E ≤ Q3 + 3 × max(IQR, 1 eV)`

6. 将冻结后的阈值一致应用到 train、val、test。只删除各原划分内不满足规则的样本，保留样本顺序，不重新 shuffle。

四分位数用线性插值。1 eV 的 IQR 下限避免一组近乎恒定的已弛豫能量产生过窄区间。上下界均包含在保留区间内。

规则不读取模型预测误差，也不按 POSCAR 序号删除。序号只出现在检查表中。统计离群点仍可能是有效的高能或稀有构型；完整数据成绩与筛选后成绩对应不同任务，不能用后者的下降直接宣称泛化改善。

## 本地生成结果（seed 123）

| 划分 | 原始 | 保留 | 排除 |
|---|---:|---:|---:|
| train | 1842 | 1827 | 15 |
| val | 614 | 606 | 8 |
| test | 614 | 607 | 7 |
| 合计 | 3070 | 3040 | 30 |

`275-SnC-C_Sn-POSCAR6`（最短 C–C 距离约 0.800 Å、标签 136.324 eV）属于训练集，按几何规则排除。其余选择及逐项原因见 `seed123_removed.csv`。

所有模型必须使用同一个清单。清单包含源标签、原始 CIF 的 SHA256、各划分保留/排除 ID、冻结阈值和规则。清单的整体 SHA256 会进入实验 config；保留样本的原文件或标签发生变化时会报错，避免无声混用。

## 可直接执行的命令

以下命令在 **HERA 的父目录**执行，例如 Linux 的 `/home/wuhao`。数据路径与现有项目一致，为 `dataset/Dataset_1/Dataset_1/A_rich/Neutral`。需把本次新增/修改的代码同步到训练机器。

### 1. 在训练机器生成清单

```bash
python -m HERA.scripts.preprocess_native_outliers --seed 123
```

默认输出 `HERA/results/native_clean_v2`，只预处理，不训练、不删除原始文件。

### 2. 默认 batch size，重训 attention_was 与 hetero_was no_dd

```bash
python -m HERA.main --model alignn --dataset native --mode attention_was hetero_was --alignn-hetero-dd drop --r 0 --native-preprocessing reference_v1 --alignn-hetero-feature-norm layernorm --alignn-hetero-node-norm layernorm --alignn-hetero-relations shared_residual --alignn-hetero-adapter-rank 8 --alignn-hetero-pooling defect_energy_mean --seed 123 --epochs 500 --device cuda:0 --atom-init HERA/atom_init.json --native-filter-manifest HERA/results/native_clean_v2/manifest.json --run-dir HERA/logs/alignn_native_clean_v2 --compact-logs --resume
```

当前 native 默认训练 batch size 为 **16**，测试 batch size 为 **1**；命令使用配置默认值。这次只改变样本清单，保留原有模型设置。

使用新根目录区分完整数据和筛选数据；内部仍为紧凑布局 `alignn/native/attention_was`、`alignn/native/hetero_was_no_dd`。

### 3. 同一新目录补测 no_aa_dd

```bash
python -m HERA.main --model alignn --dataset native --mode hetero hetero_was --alignn-hetero-aa drop --alignn-hetero-dd drop --r 0 --native-preprocessing reference_v1 --alignn-hetero-feature-norm layernorm --alignn-hetero-node-norm layernorm --alignn-hetero-relations shared_residual --alignn-hetero-adapter-rank 8 --alignn-hetero-pooling defect_energy_mean --seed 123 --epochs 500 --device cuda:0 --atom-init HERA/atom_init.json --native-filter-manifest HERA/results/native_clean_v2/manifest.json --run-dir HERA/logs/alignn_native_clean_v2 --compact-logs --resume
```

compact 布局会按关系变体使用 `hetero_no_aa_dd`、`hetero_was_no_aa_dd`。

如果需要 native 的全部非 hypergraph 模式，将第 2 条命令的 `--mode` 参数替换为：

```text
--mode full full_x attention attention_was hetero hetero_was was_x definet definet_was
```

## Resume 与版本

- `--resume` 跳过同配置、同清单下已经有有效最终结果的 seed。
- 未完成且只有训练历史时，会从头训练该 seed；它不恢复优化器状态或接着旧 epoch 训练。
- 对损坏或身份不匹配的 checkpoint 仍报错。不要拿完整数据的旧 checkpoint 作为筛选后实验的完成结果。
- 本次命令没有 `--protect-existing`，允许未完成实验重新开始。
- 改阈值、seed 列表或 CV 协议后，生成另一个 `--output`，并使用另一个训练 `--run-dir`；脚本不会覆盖不同内容的已有清单。
- 多 seed 示例：`python -m HERA.scripts.preprocess_native_outliers --seed 123 42 --output HERA/results/native_clean_seed123_42`。训练命令相应使用相同 seed 列表和新清单。每个 seed 只用自己的 train 拟合阈值。
- 与 semi/imp2d 一起请求时，该参数仅作用于 native；本次没有为另外两个数据集定义离群规则。

## 验证与输出

本次相关测试覆盖训练/留出集隔离、不同缺陷元素分组、保留原划分及顺序、输入身份变化、加载时过滤、CLI 参数传递及旧日志兼容。

- `manifest.json`：训练入口读取的完整固定清单。
- `structure_quality.csv`：全部样本几何检查及 POSCAR 序号。
- `seed123_kept.csv` / `seed123_removed.csv`：保留/排除名单与原因。
- `summary.csv`：各划分计数。
- `loader_verification.json`：本地真实数据加载核对，比较保留顺序与两个旧 checkpoint 的原始划分。

POSCAR1 的独立分析见 `results/native_outlier_audit_20260922/poscar1_analysis.md`。

## 严格版：进一步筛查

严格版保留 v2 的全部排除规则，额外检查：

1. **按元素尺寸衡量过短距离。** 对所有真实原子对计算周期性最小镜像距离与两元素共价半径之和的比值 `d/(r_i+r_j)`；任意一对小于 **0.80** 时排除。该检查可以发现距离大于 1 Å、但相对于原子尺寸仍强烈压缩的构型。不是只检查绝对距离最短的那一对，也不包含虚拟 vacancy 节点。
2. **孤立原子。** 对每个原子找归一化距离最近的邻居；若仍大于 **1.80**，按明显脱离其余结构的规则排除。
3. **同一缺陷位置系列的能量。** 去除文件名 POSCAR 序号后形成相同系列，至少有 **4** 个通过几何检查的原始训练样本才拟合 Q1/Q3；继续使用 `Q1−3×max(IQR,1 eV)` 至 `Q3+3×max(IQR,1 eV)`。例如 BN 的 `N_i_B` 与 `N_i_neut` 不混合。序号不作为过滤条件。
4. **中等压缩与高能的联合证据。** 在上述训练系列中计算最小距离比的中位数 q_med、能量中位数 E_med，以及能量稳健尺度 `sigma=1.4826×median(|E−E_med|)`。同时满足以下三项才额外排除：`q<0.90`、`q<0.90×q_med`、`E>E_med+max(6×sigma,1 eV)`。这用于识别少数高能点撑宽四分位区间的情况；稳定短键系列如果距离没有相对收缩，就不会因这条规则被删除。

距离检查采用 [ASE 的共价半径表](https://ase.gitlab.io/ase/ase/data.html)。0.80、0.90、1.80 和能量统计阈值是本实验明确指定的筛查参数，**不是 ASE 或文献给出的普适物理有效性界限**。强压缩构型仍可能有正确的 DFT 能量；因此严格版属于 filtered benchmark，不能标为“所有错误标签均已修正”。

v2 的分组阈值仍按原来的几何检查拟合；严格版的新增检查作为额外排除条件。因此相同数据和 seed 下，严格版保留集一定是 v2 保留集的子集。`seed123_removed.csv` 同时列出各样本的实际距离比、对应训练统计量和阈值，便于逐项复核。

### 结构—能量对应的额外核查

脚本同时比较同一系列的坐标，按原始原子编号对齐，处理周期性最小镜像并消除整体平移。若 RMS 位移不超过 0.015 Å、最大位移不超过 0.05 Å，但标签相差至少 1 eV，列入 `geometry_energy_conflicts.csv`。也记录相邻可用文件的几何与能量变化、同系列晶胞/元素编号变化。

这部分会跨完整系列查看记录，**仅供诊断，不参与训练样本删除或阈值拟合**，从而避免用验证/测试标签影响训练选择。它不穷举旋转、原子置换或晶体对称等价。原始电子收敛记录仍需 OUTCAR/vasprun.xml 等文件才能核验。

### 可运行命令

在 HERA 父目录生成严格版清单：

```bash
python -m HERA.scripts.preprocess_native_outliers --profile strict --seed 123
```

默认 batch size 重训 native 的全部指定非 hypergraph 模式，hetero 使用 no_dd：

```bash
python -m HERA.main --model alignn --dataset native --mode full full_x attention attention_was hetero hetero_was was_x definet definet_was --alignn-hetero-dd drop --r 0 --native-preprocessing reference_v1 --alignn-hetero-feature-norm layernorm --alignn-hetero-node-norm layernorm --alignn-hetero-relations shared_residual --alignn-hetero-adapter-rank 8 --alignn-hetero-pooling defect_energy_mean --seed 123 --epochs 500 --device cuda:0 --atom-init HERA/atom_init.json --native-filter-manifest HERA/results/native_clean_strict/manifest.json --run-dir HERA/logs/alignn_native_clean_strict --compact-logs --resume
```

若只比较两个 WAS 模型，使用 `--mode attention_was hetero_was`。同一新目录补测 no_aa_dd 时使用 `--mode hetero hetero_was --alignn-hetero-aa drop`，其余参数保持相同。
