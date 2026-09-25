# 直接从 IMP2D db 筛选和训练

2026-09-25。启用 `--imp2d-source db --imp2d-quality-filter physical` 后，
imp2d 的结构、标签和物理筛查元数据全部来自原始 ASE 数据库，读取时不需要 CIF 或 CSV。
没有启动模型训练。

最新选择为固定 `−10 < DFE < 10 eV`，已实现并实测保留 10,475 条。
见 [当前数据版本和命令](imp2d_10ev_v1.md)。

后续按母体审查版本已接入：`--imp2d-host-filter reviewed_v1` 排除 Ti2CO2，
保留 10,715 条。见 [新数据集和直接运行命令](imp2d_host_clean_v1.md)。
本页下文仍记录不加该选项时的全范围基线。

## 能量范围复查：全库队列仍需审查

当前 DB 路径没有额外截断形成能。10,946 条输入完整、通过现有筛查的记录中，
形成能为 **−176.47023939～153.222900105 eV**；5%～95% 分位为 **−2.00475～7.52072 eV**。
通过现有几何、最终能量变化和能量分项一致性检查，不代表能量参考本身已验证正确。

其中 471 条位于旧的严格区间 `−10 < E_form < 10 eV` 之外：232 条不大于 −10 eV，
239 条不小于 10 eV。保留的 CO2Ti2（Ti2CO2）231 条全部位于 −176.47～−156.16 eV，
并共用 `hostenergy = −545.94530853 eV`，呈现整组偏移，需进一步核对母体参考计算；
目前不能仅据这些元数据确定错误原因。

若另行沿用旧能量区间，当前队列会剩 **10,475 条、43 种母体材料**，
实际范围为 **−9.86772122～9.98396280 eV**。这是 benchmark 范围限制，不能证明区间外均为错误数据。
**这一额外限制尚未应用到代码或冻结清单；以下命令仍使用全范围 10,946 条队列，
不应把它当作已完成标签可靠性审查的最终 benchmark。**
统计见 `results/imp2d_database_energy_range.json`。

## 实际筛查结果

| 阶段 | 条数 |
|---|---:|
| 官方原始 db | 17,364 |
| 通过物理/能量元数据筛查 | 10,992 |
| 其中缺少可靠同元素缺陷原子标记，暂不用于统一模型比较 | 46 |
| **通过现有筛查且模型输入完整（标签仍需上述复查）** | **10,946** |

物理排除 6,372 条。规则沿用最终 `abs(conv2) <= 0.05 eV`、`0 < extension_factor <= 2`、
有限形成能、能量分项一致性、组成与严重原子重叠检查。
已因 DFT 元数据不合格的记录不再重复执行昂贵的几何邻居扫描。
**不再使用旧 CSV 名单、−10～10 eV 范围或数据库第一阶段 `converged` 布尔值作为额外筛选条件。**

46 条不是被判为物理坏数据，而是同元素杂质的位置尚缺可靠标记。
例如在母体中加入同种元素，仅凭化学元素无法唯一识别应该被模型标记的缺陷原子。
它们保存在 `imp2d_filter_unverified_defect_identity.csv`，没有猜测最后一个原子就是缺陷。

随代码提供的 `data/imp2d_self_defect_indices.json` 包含 339 条针对固定 db 版本的已验证原子索引。
这些索引均逐个与此前核验过的 CIF 位点标签比较了元素与周期坐标，最大距离误差约 1.35e-7 Å。
当前物理通过的 359 条同元素杂质中，313 条被这份表覆盖，另外 46 条单独留出。
运行时只读取 db 和随代码提供的索引表，不依赖这些 CIF。

## imp2d 按结构分组划分

在可训练队列内发现 **100 对等价/近似等价结构**，使用固定 0.001 Å 容差，
将相连的等价结构分成同一组；当前对应 **10,846 个结构组**。
同一组只进入一个集合。能量差异仅作复查提示，不用于选择较低标签或决定划分。

seed 123 的实际划分：

| 集合 | 保留数量 |
|---|---:|
| train | 6,598 |
| val | 2,199 |
| test | 2,149 |

分组后数量接近 60/20/20，不强行拆开同组样本凑比例。
原 db 的不合格记录也保留在审计清单中，各自作为单独组记录其排除去向。
此版本是新的样本范围和划分，不能与旧 9,683 条 CIF 队列的 MAE 直接当作同一 benchmark 比较。

semi 仍使用之前的物理/标签筛查与按样本划分，没有应用这里的结构分组。
native 仍是用户选择的 strict 44 条方案。

## 一条命令：三个数据集

在 HERA 的父目录运行，用户服务器上为 `/home/wuhao`。
数据库依次从 `dataset/imp2d/imp2d.db`、`dataset/imp2d/imp2d/imp2d.db` 或已有审计缓存查找。
不存在时自动下载官方固定版本到 `dataset/imp2d/imp2d.db` 并验证 SHA256；后续复用。

```bash
python -m HERA.main --model alignn --dataset native semi imp2d --mode full full_x attention attention_was hetero hetero_was was_x definet definet_was --native-preprocessing reference_v1 --semi-preprocessing reference_v1 --imp2d-preprocessing reference_v1 --native-outlier-filter strict --semi-quality-filter physical --semi-source-policy legacy_available --imp2d-source db --imp2d-quality-filter physical --alignn-hetero-dd drop --r 0 --alignn-hetero-feature-norm layernorm --alignn-hetero-node-norm layernorm --alignn-hetero-relations shared_residual --alignn-hetero-adapter-rank 8 --alignn-hetero-pooling defect_energy_mean --seed 123 --epochs 500 --device cuda:0 --atom-init HERA/atom_init.json --run-dir HERA/logs/alignn_physical_db --compact-logs --resume
```

默认 batch size；hetero/hetero_was 是 no_dd，不含 hypergraph。
semi/imp2d 会跳过与 full 重复的 full_x，native 保留 full_x。
`--resume` 复用验证完成的结果；未完成的 split 从头训练。
使用新的 `alignn_physical_db` 根目录以区分旧 CSV/CIF 队列，子目录仍是 `alignn/dataset/mode`。

只跑 imp2d 时使用：

```bash
python -m HERA.main --model alignn --dataset imp2d --mode full attention attention_was hetero hetero_was was_x definet definet_was --imp2d-preprocessing reference_v1 --imp2d-source db --imp2d-quality-filter physical --alignn-hetero-dd drop --r 0 --alignn-hetero-feature-norm layernorm --alignn-hetero-node-norm layernorm --alignn-hetero-relations shared_residual --alignn-hetero-adapter-rank 8 --alignn-hetero-pooling defect_energy_mean --seed 123 --epochs 500 --device cuda:0 --atom-init HERA/atom_init.json --run-dir HERA/logs/alignn_physical_db --compact-logs --resume
```

要继续 no_aa_dd 消融，在对应命令中将 mode 改为 `hetero hetero_was`，再加 `--alignn-hetero-aa drop`。
使用同一个 run-dir，输出保存在同级的关系消融模式目录。

## 输出和验证

- `logs/alignn_physical_db/imp2d_filter_manifest.json`：来源、规则、结构组和固定 split。
- `imp2d_filter_seed123_kept.csv` / `imp2d_filter_seed123_removed.csv`：可训练与排除记录。
- `imp2d_filter_unverified_defect_identity.csv`：单独留出的 46 条。
- `results/imp2d_database_preprocessing_result.json`：完整统计和读取验证。
- `results/imp2d_database_split_audit/summary.json`：重新匹配结构后的划分检查。

新增 DB 路径、自动下载及原有 impurity 路径的 32 项相关测试通过。
测试包括禁止 CSV/CIF 读取、有限高能样本保留、原始最终收敛判据、分组隔离、五折划分、
检查点数据身份检查，以及 full/hetero/attention 的缺陷位置和 WAS 标记。
原始数据库的能量/几何筛查和结构分组已实际执行，缓存再次验证通过；模型尚未训练。
