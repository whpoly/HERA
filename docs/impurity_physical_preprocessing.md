# imp2d / semi 物理质量检查与读取时过滤

日期：2026-09-24。原始文件保留；排除发生在训练读取时。没有训练模型或根据预测误差筛样本。

## imp2d 已执行的最终预处理

**已生成 `HERA/logs/alignn_physical_clean/imp2d_filter_manifest.json`。**
本次使用下载补齐后的主 `id_prop.csv`：**10,985 个唯一 ID、10,985 个非空 CIF**。
全部实际 CIF 的周期几何与全部标签均已逐条匹配原始 ASE 数据库。

沿用旧代码的 −10 < Eform < 10 eV 范围得到 **10,302 条起始样本**，
本次按物理/元数据规则隔离 **619 条**，最终保留 **9,683 条**。
另有 683 条本就处于旧能量范围之外，单独记录，不计入新增隔离数量。

| seed123 划分 | 起始数量 | 隔离 | 保留 |
|---|---:|---:|---:|
| train | 6,181 | 371 | 5,810 |
| val | 2,060 | 134 | 1,926 |
| test | 2,061 | 114 | 1,947 |

619 条的排除原因：475 条最终阶段 `abs(conv2) > 0.05 eV`，
154 条 `extension_factor > 2`、超出局部点缺陷研究范围，3 条能量元数据不一致。
前两类重叠 13 条，因此是 **475 + 154 − 13 + 3 = 619**。
实际训练队列没有因严重原子重叠阈值产生额外排除。
短键、弱结合与弛豫后位点迁移本身只作复查提示。

清单 SHA256：`48e80b4a0df65c0e7cf770ac4a2fe973368132c2d669296154a16493bd438bae`。
163 个实际下载 CIF 代表不同宿主、ads/int 位点及同元素杂质，
full/full_x/hetero/attention 的节点数、唯一缺陷标记、WAS 和来源 ID 检查全部通过。

结果与清单：

- `results/imp2d_preprocessing_result.json`：最终统计与验证状态。
- `logs/alignn_physical_clean/imp2d_filter_seed123_removed.csv`：619 条隔离记录及原 split。
- `logs/alignn_physical_clean/imp2d_filter_outside_original_cohort.csv`：旧能量窗口外的 683 条，另附实际几何/元数据检查。
- `results/impurity_download_verification/summary.json`：本次完整下载的最终核验；旧下载状态另存为 `summary_before_complete_download.json`。

## semi 已执行的最终预处理

**已生成并验证 `HERA/logs/alignn_physical_clean/semi_filter_manifest.json`。**
主 CSV 13,966 行，按显式旧来源规则得到 12,104 条起始样本；
其中 **154 条标签冲突样本隔离，保留 11,950 条**。

| seed123 划分 | 起始数量 | 隔离 | 保留 |
|---|---:|---:|---:|
| train | 7,262 | 84 | 7,178 |
| val | 2,421 | 37 | 2,384 |
| test | 2,421 | 33 | 2,388 |

这 154 条组成 **77 对完全相同的 CIF**：文件 SHA256 相同、材料/缺陷类型/杂质元素相同，
但形成能相差 **0.08477～1.79113 eV**。全部属于 CdSe 的 Mn、Sn、As、Li、Mg 杂质组。
例如 `1444-CdSe-M_i_B-Mn-POSCAR6.cif` 与 `1080-CdSe-M_i_B-Mn-POSCAR6.cif`，
结构内容完全相同，标签分别为 −0.204007188 与 1.587126653 eV。

规则采用固定 **0.05 eV** 的标签一致性容差，并隔离冲突组的所有成员；
没有基于模型误差筛选，没有从冲突标签中挑较低或更好预测的值。
这表明当前输入不能区分这些目标，不能由此断定哪次 DFT 计算错误。
需要原始计算条件、总能量和参考化学势才能确认应恢复哪个标签。
该检查在固定 split 的筛选步骤中对原始记录应用，未拟合 train/val/test 分布阈值。

起始范围之外的 1,862 行单独记录：1,182 个缺 CIF，202 个因缺 `GeSn.vasp` 无法加载，478 个非外来杂质。
它们不计入上述 154 条质量隔离。
12,104 条起始样本另有 217 个短键提示和 6 个弱结合/孤立提示，没有仅凭这些提示删除。

清单 SHA256：`0f39f683727a022678ba94c72991f795cd11acbbf2b57011fca5d035f254fb83`。
再次读取原始 CSV/CIF/host 后缓存验证通过。此前只做几何筛查的临时清单保留为
`semi_filter_manifest_geometry_only.json`，训练自动使用最终的 `semi_filter_manifest.json`。
71 项回归测试通过；154 个真实材料/缺陷类型组合的 WAS/hetero/attention 输入核对通过。
没有训练模型。

## 新下载文件复核

最新读取已确认 semi 目录内 **12,784 个 CIF 全部非空**，两个 CSV 也已下载。
主 CSV 共 **13,966 行**，其中 **1,182 个 POSCAR0 没有对应 CIF**。
这些 ID 恰好等于 `id_prop_A_rich_final.csv` 的名单；该文件的每个标签都等于主 CSV 相应配置组的最低能量，
并非原 POSCAR0 对应的能量。1,129 行标签相差超过 1e-4 eV，因此不能直接合并/替换。

Git 历史 `a27c29c` 之前的 semi 加载器使用 `except Exception: continue` 跳过缺失 CIF 和 host。
新增 **`--semi-source-policy legacy_available`** 明确复现这一可用文件/host 范围，分别记录：
缺 CIF、缺 host、不属于外来杂质范围。这些均不称为物理离群值。
默认 `complete` 仍会要求完整来源，不会自动忽略缺文件。
随后在该起始队列上划分 train/val/test，再应用物理筛选；没有对照历史 checkpoint 核验 split 身份。
缺失文件或 host 日后出现，冻结清单会拒绝复用，要求新 run-dir。

IMP2D 最新下载已补齐：**10,985 个 CIF 全部非空，主 `id_prop.csv` 为 361,596 字节、10,985 条唯一 ID**。
标签全部与本地原始 ASE 数据库一致。实际训练过滤清单已经生成，最终数量见本文开头。

主 CSV 中 **683 条在既有 −10 < Eform < 10 eV 队列之外**；它们是旧代码已经排除的样本，不计入新增物理隔离。
683 条的实际 CIF 均已解析并与原始数据库周期几何核对：319 条命中物理规则，364 条通过当前规则。
因此，能量窗口外不等于物理错误；保留旧窗口是为了维持现有 benchmark 队列。

父目录 `prepared_data.csv` 是另一份 9,823 条子集，不能覆盖主 CSV。
本轮使用完整的 `imp2d/imp2d/id_prop.csv`，不使用该替代名单，也不重新排序。
空文件曾导致检查受阻；现在该下载问题已经解决。

下载核验工具（只检查，不训练、覆盖或恢复原始文件）：

```bash
python -m HERA.scripts.verify_impurity_download
```

结果写入 `results/impurity_download_verification/`，包含每个文件的字节数、SHA256、
标签/几何对应状态，以及独立的缺失文件清单和物理排除候选清单。
`results/semi_download_verification/` 保留早期部分下载时的扫描记录。
semi 的最终结果使用 `logs/alignn_physical_clean/semi_filter_*` 清单与表格。

## 已完成的实际检查

### imp2d

检查了本地原始 ASE 数据库的 **17,364 条结构**。数据库 SHA256：
`3a71db999b477112da248dcf762c4384e455689953679d58b3d71a91e7148fc4`。
首次审计时实际训练 CSV/CIF 缺失；现在下载已补齐，实际训练清单结果见前文。
以下是数据库统计，不能直接当作远端训练集的删除数量。

| 数据范围 | 原数量 | 排除 | 保留 |
|---|---:|---:|---:|
| 全数据库（包含没有有限形成能的记录） | 17,364 | 6,372 | 10,992 |
| 有限形成能 | 14,667 | 3,675 | 10,992 |
| 原代码的 −10 < Eform < 10 eV 范围 | 13,011 | 2,507 | 10,504 |
| 上一行再限定数据库 `converged=True` | 10,302 | **619** | **9,683** |

上一次 WAS 审计使用最后一行的 10,302 条。那次确认了缺陷标记和特征构建，未确认 DFT 数值质量。
若实际 CSV 对应这一子集，新规则的 619 条排除由以下原因组成：

| 原因 | 数量（可重叠） |
|---|---:|
| 最终阶段 `abs(conv2) > 0.05 eV` | 475 |
| `extension_factor > 2`，超出局部点缺陷建模范围 | 154 |
| 总能量、参考能与形成能不一致 | 3 |
| 前两类共同命中 | 13 |

因此是 `475 + 154 − 13 + 3 = 619`。没有样本因为严重原子重叠阈值被额外排除。
按初始位点分：ads 6,949 → 6,644；int 3,353 → 3,039。

**关键发现：本地数据库全部 17,364 条的 `converged` 布尔值都恰好与 `abs(conv1) <= 0.05` 一致。**
这是本次逐条统计得到的事实；没有据此断言数据库生成代码的实现。
它不能替代对最终阶段 `conv2` 的检查。官方字段定义区分了第一阶段 `conv1/en1` 和最终 `conv2/en2`。
来源：[IMP2D 官方数据库说明](https://cmr.fysik.dtu.dk/imp2d/imp2d.html)。

例如 `NiSe2_Sc_int0`：标记为已收敛，`conv1=-0.004441 eV`，但 `conv2=-9.14814 eV`；
形成能 −6.178 eV 本身在旧范围内，因此旧能量范围并不能识别这种质量问题。

三条能量记录异常为 `C3Nb4_Ti_int1`、`C3Nb4_Ti_int2`、`CNb2O2_Re_ads0`。
它们保存的 `en1/en2/conv1/conv2` 都是零，形成能却不满足保存的能量关系。
处理为隔离待核对，没有把 `en2=0` 代入公式“修复”标签，也没有认定几何结构一定错误。

### semi

实际可用来源与最终过滤结果见本文开头。33 个现有 host VASP 文件通过组成检查和严重重叠筛查；
仍缺 `GeSn.vasp`，因此显式历史来源策略将对应 202 条记录列为来源不可用，不视为坏材料。
主 CSV 引用但没有 CIF 的 1,182 条 POSCAR0 也单独记录。
没有原始 DFT 收敛日志或原子受力，几何和标签一致性筛查不能替代 DFT 质量验证。

## 物理规则与适用边界

### 自动排除

1. 非有限目标、无效晶胞/坐标/元素、无法用于当前图模型的部分占据。
2. 单杂质组成不符；semi 替位后不能唯一还原被替换的宿主元素。
3. 周期边界下存在 `d_ij / (r_cov_i + r_cov_j) < 0.5` 的严重重叠。
   这是保守的工程筛查阈值，**不是普适的成键定律**。使用 ASE 元素共价半径，并包括同一原子的周期镜像。
4. imp2d 最终 `abs(conv2) > 0.05 eV` 或缺少该值。直接检查最终阶段；初始未收敛而最终收敛的记录可保留。
5. imp2d 膨胀因子 `XF > 2`，或该值无效。筛选的是适合局部点缺陷研究的结构，不能由此认定所有重构结构均不真实。
6. imp2d 保存的形成能不满足 `Eform = en2 − hostenergy − dopant_chemical_potential`，容差 `1e−4 eV`。
7. semi 在相同材料/缺陷类型/杂质元素内，完全相同 CIF 的目标相差超过 0.05 eV：整组隔离为标签冲突。

论文采用最终离子能量收敛 0.05 eV 的分析口径，并讨论 XF 超过 2 的大幅重构；XF 是掺杂体系弛豫后与弛豫前厚度的比。
来源：[原始论文，分类参数与方法部分](https://arxiv.org/html/2207.05353v2)。`conv2` 的单位是 eV，不是力的 eV/Å。

### 仅标记复查，保留样本

- `0.5 <= 最短归一化距离 < 0.8`；不同元素、键级和配位的合理键长会不同。
- 在 `1.8 × 共价半径和` 范围内没有邻居的原子；弱吸附可能真实存在。
- 初始 ads/int 位点与最终深度分类不同；弛豫迁移是可发生的物理过程。
- 单纯形成能偏高、偏低或为负；不使用 IQR、模型误差、test 表现来决定删除。
- semi 缺少收敛元数据：标记 `DFT_convergence_unverified_no_metadata`。几何和组成检查通过不代表 DFT 已收敛。

在上述 9,683 条保留 IMP2D 样本中，有 740 条短键提示、672 条位点迁移提示、52 条弱结合/孤立提示。
这些计数可重叠，均没有仅凭提示自动排除。

## 一条训练命令：读取时检查并过滤

### native、semi、imp2d 三组一起运行

在 HERA 的父目录执行。三个过滤器都由 `main` 在构图和训练前调用；无需提前运行清洗脚本。
沿用 native strict 44 条方案，semi 隔离 154 条标签冲突，imp2d 隔离 619 条物理/元数据异常。
这些数量对应本次核验的数据文件与 seed 123；来源改变时冻结清单会阻止混用。

```bash
python -m HERA.main --model alignn --dataset native semi imp2d --mode full full_x attention attention_was hetero hetero_was was_x definet definet_was --native-preprocessing reference_v1 --semi-preprocessing reference_v1 --imp2d-preprocessing reference_v1 --native-outlier-filter strict --semi-quality-filter physical --semi-source-policy legacy_available --imp2d-quality-filter physical --alignn-hetero-dd drop --r 0 --alignn-hetero-feature-norm layernorm --alignn-hetero-node-norm layernorm --alignn-hetero-relations shared_residual --alignn-hetero-adapter-rank 8 --alignn-hetero-pooling defect_energy_mean --seed 123 --epochs 500 --device cuda:0 --atom-init HERA/atom_init.json --run-dir HERA/logs/alignn_physical_clean --compact-logs --resume
```

使用默认 batch size；上述模式不含 hypergraph，hetero/hetero_was 使用 no_dd。
semi、imp2d 会自动跳过与 full 重复的 full_x，native 保留 full_x。
原始数据库自动依次查找 `dataset/imp2d/imp2d.db`、`dataset/imp2d/imp2d/imp2d.db`，最后查找 HERA 本地审计缓存。
在 `/home/wuhao` 执行时即使用 `/home/wuhao/dataset/imp2d` 下的数据库，无需设置 `--imp2d-source-db`。
DFT 收敛/能量分项来自原始 `imp2d.db`；CIF 和 `id_prop.csv` 本身不含这些元数据。
`--resume` 跳过已经验证完成的结果，未完成的 split 从头训练；当前实现不是逐 epoch 断点续训。

补跑之前的 no_aa_dd 时，保留同一 run-dir，将 `--mode` 改成 `hetero hetero_was`，
并加入 `--alignn-hetero-aa drop`（已有的 `--alignn-hetero-dd drop` 保留）。
compact 布局会使用同级的 `hetero_no_aa_dd` / `hetero_was_no_aa_dd` 目录。

### 当前已下载的 semi，可直接使用

以下命令使用显式历史可用来源范围，复用已在 `HERA/logs/alignn_physical_clean` 生成的 seed123 过滤清单：

```bash
python -m HERA.main --model alignn --dataset semi --mode full attention attention_was hetero hetero_was was_x definet definet_was --semi-preprocessing reference_v1 --semi-quality-filter physical --semi-source-policy legacy_available --alignn-hetero-dd drop --r 0 --alignn-hetero-feature-norm layernorm --alignn-hetero-node-norm layernorm --alignn-hetero-relations shared_residual --alignn-hetero-adapter-rank 8 --alignn-hetero-pooling defect_energy_mean --seed 123 --epochs 500 --device cuda:0 --atom-init HERA/atom_init.json --run-dir HERA/logs/alignn_physical_clean --compact-logs --resume
```

### semi 与 imp2d 两组一起运行

在 HERA 的父目录执行，机器上需要有完整 semi/imp2d 训练 CSV/CIF、semi host 和原始 `imp2d.db`。
数据库按上述默认目录自动查找，不必在命令中设置路径。

```bash
python -m HERA.main --model alignn --dataset imp2d semi --mode full attention attention_was hetero hetero_was was_x definet definet_was --imp2d-preprocessing reference_v1 --semi-preprocessing reference_v1 --imp2d-quality-filter physical --semi-quality-filter physical --semi-source-policy legacy_available --alignn-hetero-dd drop --r 0 --alignn-hetero-feature-norm layernorm --alignn-hetero-node-norm layernorm --alignn-hetero-relations shared_residual --alignn-hetero-adapter-rank 8 --alignn-hetero-pooling defect_energy_mean --seed 123 --epochs 500 --device cuda:0 --atom-init HERA/atom_init.json --run-dir HERA/logs/alignn_physical_clean --compact-logs --resume
```

使用默认 batch size，不包含 hypergraph；hetero 是 no_dd。两个数据集没有空位，`full_x` 与 `full` 重复。
若一起做 native，将 dataset 改成 `native imp2d semi`，再加
`--native-preprocessing reference_v1 --native-outlier-filter strict`，即保留之前的 native strict 44 条方案。

首次运行在同一个 run-dir 根目录生成：

- `<dataset>_filter_manifest.json`：冻结规则、原始/保留 split、文件及数据库校验值。
- `<dataset>_filter_structure_quality.csv`：所有 CSV 记录，包括原 benchmark 范围外的记录。
- `<dataset>_filter_seed123_removed.csv` / `kept.csv`：实际 split 内排除与保留清单。
- `<dataset>_filter_summary.csv`：各 split 数量。
- `semi_filter_outside_original_cohort.csv`：显式原加载范围之外的来源，独立于物理排除清单。
- `semi_filter_label_conflicts.csv`：154 条完全相同输入对应冲突标签的记录。
- `semi_filter_geometry_review.csv`：短键/弱结合复查名单。

训练结果仍为 `<run-dir>/alignn/<dataset>/<mode>/`，关系消融沿用既有同级模式目录规则。
原始 benchmark 队列先按旧协议分 train/val/test，再在每个 split 内移除；所有模式共用同一个清单。
`--resume` 复用已验证完成的结果；规则、原始标签/结构/数据库改变时拒绝混用旧配置。
先前 native strict 的代码和规则没有改动。

**IMP2D 旧的 −10～10 eV 范围仍保留以维持 benchmark 队列与 split 身份；它是历史选择，不是物理合理性的充分条件。**
数据库审计也单独列出了无此范围限制的结果。实际 CSV 已经排除的样本不会被重新加入。
对实际训练 CIF 与 ASE 库逐条检查能量、元素、晶胞和周期几何对应；版本不一致会停止，避免套用错记录的收敛数据。

清洗后的 test 集已经变化，模型之间应在同一清洗版本上比较；不能把 MAE 的变化全部归因为模型改进。

## 审计与验证

仅检查、输出报告，不训练：

```bash
python -m HERA.scripts.audit_impurity_quality --imp2d-db HERA/tmp/imp2d_audit/imp2d.db --data-root dataset --output HERA/results/impurity_quality
```

完整数据可用时，此命令也会建立实际训练 CSV/CIF 的过滤清单；缺失时明确记录 unavailable。

- 原始数据库的 17,364 条几何、组成和元数据已全部扫描。
- 额外将 93 条真实数据库记录导出 CIF（覆盖 87 个 host/初始位点类型组），验证周期几何匹配、读取时过滤和缓存复用；保留 78 条、隔离 15 条。
  这批是验证用的数据库导出件，不是用户下载的训练 CSV。
- 71 项规则/CLI/下载核验/旧数据路径/compact/resume 回归测试通过。
- 报告：`results/impurity_quality/summary.json`；逐条结果：`imp2d_database_quality.csv`；真实导出件验证：`roundtrip_verification.json`。
- 局限：semi 使用显式可用来源范围；两个数据集均没有原始 DFT 迭代日志或原子受力，无法证明电子/力收敛和标签绝对正确。
