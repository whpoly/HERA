# imp2d / semi：缺陷定位、真实 WAS 和数据完整性检查

## 结论

两套旧 loader 都使用 `skip_was=True`，没有生成原占位标签，因此旧 WAS 输入实际是
`[E(current), E(current)]`。新 WAS 模式默认采用各自的 `reference_v1`：

| 数据 / 缺陷 | WAS | 定位依据 |
|---|---|---|
| imp2d 吸附原子 `ads*` | 0（原位置为空） | 外来元素；同元素杂质采用已有 CIF 标签表 |
| imp2d 间隙原子 `int*` | 0 | 同上 |
| semi 替位 `M_A` / `M_B` | 被替换的母体元素 Z | 找出唯一外来原子，由母体元素计数恢复缺失元素 |
| semi 间隙 `M_i_A/B/neut` | 0 | 唯一外来原子，并检查移除后恢复母体化学计量 |
| 两套数据的普通原子 | 自身 Z | 保留原子身份 |

`E(0)` 为 92 维零向量，WAS 输入仍为 184 维。imp2d 的 WAS 提供“原位置为空”这一标记，
并不提供额外的被替换元素；异构图本来已有缺陷类型，因此也不能预先认定新 WAS 必然改善 MAE。

## imp2d：真实原始结构检查通过，训练 CSV 尚未对齐

本地训练路径 `dataset/imp2d/imp2d/id_prop.csv` 缺失。
使用现有 `HERA/tmp/imp2d_audit/imp2d.db`，按 `converged=True`、有限形成能、
`-10 < eform < 10` 选出 10,302 个结构进行检查。该选择不是对远端训练 CSV 的复现承诺。

| 项目 | 数量 |
|---|---:|
| 原始数据库条目 | 17,364 |
| 本次检查结构 | 10,302 |
| 外来元素杂质 | 9,963 |
| 同元素杂质 | 339 |
| 吸附 / 间隙结构 | 6,949 / 3,353 |
| 验证节点 | 434,977 |
| CIF 写入—读入检查 | 608 |

所有 10,302 个结构都保留全部原子和坐标，标记一个缺陷，新 WAS 只有缺陷节点的两半输入不同；
attention 与 hetero 的元素、几何、类型及 WAS 一致。旧输入在全部这些结构上都是重复当前元素。
在本次原始库检查范围内，imp2d 未出现 native 的“误用最后一个原子”或空位节点替换真实原子的问题。

现有 `imp2d_self_defect_sites.json` 的 339 项完整覆盖上述已收敛同元素样本，
全部经过 CIF 往返检查，标签存在且对应元素正确。能量范围内另外 73 个同元素条目均未收敛，
因此不能把它们缺少定位表记录直接算作训练数据的定位错误。
本次复核标签覆盖和程序选取，不是重新人工标注全部同元素原子的历史身份。

原始库有 2,697 个条目的形成能非有限值；旧 `<= -10 or >= 10` 判断不会排除 NaN。
现改为训练清单含 NaN/Inf/不可解析目标时明确报错，避免带着无效标签训练。
**是否影响过旧实验，取决于当时导出的 CSV 是否已筛掉这些记录，现有本地资料不能确定。**

## semi：已修复代码问题，尚不能完成真实数据验证

本地 `dataset/Dataset_1/Dataset_1/Neutral/Neutral/`：

- `id_prop_A_rich.csv` 为 0 字节；12,784 个 CIF 也全部为 0 字节。
- 文件名覆盖 34 种材料；只有 33 个母体 VASP 文件，缺少 `GeSn.vasp`。
- 有 202 个 GeSn CIF 文件名，但 CSV 不可读，不能据此断言它们都在实际训练清单中。

semi 原来的缺陷类型通过“该元素不属于母体元素集合”确定，不依赖读入后的原子排序；
这个逻辑适用于当前 loader 选择的外来元素杂质。新版本继续采用这个身份判据，
要求每个结构恰好有一个与文件名匹配的外来原子。

替位通过母体化学计量和当前原子计数唯一确定原元素，不假设 A/B 对应哪个元素，
不拿弛豫后的坐标与原胞精确匹配。计数、杂质身份或缺陷标签不一致时停止。
新路径保留全部原子，避免旧 `strucure_to_dict` 按三位小数坐标去重；
目前没有真实 semi 结构，不能量化旧代码是否实际丢过原子。

另一个明确问题是旧 loader 用 `except Exception: continue` 跳过坏 CIF / 缺失母体文件。
现已改成带文件名的明确错误，防止训练样本集合悄悄变化。原有“仅保留外来元素缺陷”的选择条件不变。

已用人工构造的替位、间隙、同元素杂质、顺序打乱等结构验证代码、配对图、有限梯度及权重恢复；
**这些测试不能代替完整 semi 数据验证**。需要完整 CSV、CIF 和母体文件后再运行下述审计。

## 结果隔离与运行命令

新配置记录 `imp2d_preprocessing` / `semi_preprocessing`，新路径增加
`imp2d_reference_v1` / `semi_reference_v1`。历史 checkpoint/history 保留，汇总增加带版本的新行。
默认只有 WAS 模式启用新版；普通模式保留旧预处理，以继续复用旧结果。
配对实验应显式指定相同版本。首次训练新语义需要新模型，旧 WAS 权重不是这些模型的结果。

从 HERA 父目录，先只审计，不训练（默认优先使用实际 CSV/CIF；缺失时才尝试本地 imp2d 原始库）：

```bash
python -m HERA.scripts.audit_impurity_was --data-root dataset
```

在完整数据机器上运行配对 baseline：

```bash
python -m HERA.scripts.run_hetero_relation_benchmark --dataset imp2d semi --mode hetero hetero_was --variant baseline --imp2d-preprocessing reference_v1 --semi-preprocessing reference_v1 --seed 123 11 1245 --epochs 500 --device cuda:0 --run-dir HERA/logs/hetero_relation_native_semi_imp2d
```

`--dry-run` 只打印任务和数据可用性，不训练，也不替代上面的逐结构审计。
如需三套数据一起比较，在 `--dataset` 中加 `native`，并加 `--native-preprocessing reference_v1`。
如需删除 DD，改为 `--variant no_dd`。如需复用旧 WAS 结果，用对应数据的 `--*-preprocessing legacy`。
原有 `--resume --protect-existing` 机制继续生效。

本次未启动完整训练，没有新 MAE。审计报告为 `results/impurity_was_audit_20260920.json`。
