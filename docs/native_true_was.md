# Native：真实 WAS 与配对对照（2026-09-20）

## 现在输入的是什么

`attention_was`、`hetero_was` 的节点输入为

\[
x_i = E(Z_i^{\mathrm{now}}) \mathbin{\Vert} E(Z_i^{\mathrm{was}}),\quad E(0)=\mathbf{0}.
\]

两部分各 92 维，合计 184 维。`was` 表示该位置在产生缺陷前是什么。

| 节点 | now | was |
|---|---|---|
| 普通 Zn | Zn | Zn |
| Zn 空位 `V_Zn` | X（0） | Zn |
| `O_Zn` 替位 | O | Zn |
| `Zn_i_A` 间隙原子 | Zn | 空位点（0） |

旧 native loader 没有 `was` 标注，实际输入是 `[E(now), E(now)]`，X 是两个零向量。
新 native WAS 配置默认使用 `native_preprocessing=reference_v1`，必须有真实标签，缺失时直接报错。
`was_x`、`hypergraph_was`、`definet_was` 等 native WAS 模式也使用这个版本。
后续检查已补全 semi/imp2d 的原占位逻辑并按数据集隔离版本；imp2d 原始库检查通过，
semi 仍缺少有效的本地结构数据，详见 [imp2d/semi 检查报告](imp2d_semi_was_audit_20260920.md)。

## 同时修正的 native 缺陷定位

CIF 的原子标签（如 `Zn64`）保留原始编号，而 `Structure.from_file` 会按元素排列原子。
因此不能拿读入后的最后一个原子当作缺陷。新版本检查全部标签编号唯一、连续，
按最大原始编号定位替位/间隙原子，并检查其元素与文件名一致；不匹配时停止。

空位按 native 数据的中心化约定在分数坐标 `(0.5, 0.5, 0.5)` 增加 X，
保留 CIF 中的所有真实原子。旧 hetero/attention 空位路径会拿 X 替换最后一个真实原子。
正常原子的 WAS 不随 `r` 扩展局部区域而改变；hetero 的 `pool_type` 仍只标记实际缺陷。

数据中 GaSb/InSb 的 `Ab_i_A` 文件名被明确映射为 `Sb_i_A`，且仍需通过 CIF 元素 Sb 校验。
这两个已知拼写问题以外的未知元素/不合法编号不会被猜测或静默跳过。

全量检查本地 3,070 个 CIF：531 空位、391 替位、2,148 间隙原子；198,628 个图节点都有 WAS。
两种图使用相同缺陷位置和 WAS。旧代码在 172 个替位、1,093 个间隙结构上选错缺陷原子；
531 个空位图各恢复一个之前被误删的真实原子。
这些数字验证输入构造，不代表测试集 MAE，也不能证明另一台机器用了相同版本。

## 如何比较、如何保留旧结果

| 设置 | 普通模式 | WAS 模式 | 输出目录 |
|---|---|---|---|
| 未指定版本 | 历史预处理 | `reference_v1` | 普通模式沿用原目录；新 WAS 加版本子目录 |
| `--native-preprocessing reference_v1` | 修正缺陷位置/X，92 维输入 | 相同图，真实 WAS，184 维输入 | 都加 `native_reference_v1` |
| `--native-preprocessing legacy` | 历史预处理 | 历史 WAS 回退 | 历史目录 |

**配对比较必须给两组都指定 `reference_v1`。** 否则差异同时包含 WAS 信息与缺陷定位修正。
仅从新模型与旧 MAE 的差值不能得出 WAS 收益。

新路径示意：

```text
baseline/alignn/native/hetero/.../native_reference_v1/seed123_best_checkpoint.pth
baseline/alignn/native/hetero_was/.../native_reference_v1/seed123_best_checkpoint.pth
```

旧 checkpoint/history 不覆盖，汇总保留旧行并增加带版本的新行。
`--resume --protect-existing` 只复用同一路径已完成的 seed；未完成且已有产物的训练不会自动重启。
新语义首次运行需要新训练，不能把旧 WAS checkpoint 当作新模型继续评估。
手动加载 checkpoint 时，数据 loader 必须接收保存的 `config.get('native_preprocessing')`；
旧 checkpoint 缺少该字段，对应 legacy。模型与图的显式版本不一致会报错。

## 运行命令

在 HERA 的父目录，激活 HERA 的 Python 环境执行。下面同时比较 attention 和 shared hetero 的 WAS，
统一 batch size、seed、数据划分；hetero 使用 baseline 的 AA/DD keep、共享关系残差和缺陷能量均值池化。

```bash
python -m HERA.main --model alignn --dataset native --mode attention attention_was hetero hetero_was --r 0 --native-preprocessing reference_v1 --alignn-hetero-node-norm layernorm --alignn-hetero-feature-norm layernorm --alignn-hetero-relations shared_residual --alignn-hetero-adapter-rank 8 --alignn-hetero-pooling defect_energy_mean --batch-size 8 --test-batch-size 1 --seed 123 11 1245 --epochs 500 --device cuda:0 --atom-init HERA/atom_init.json --run-dir HERA/logs/hetero_relation_native_semi_imp2d/baseline --resume --protect-existing
```

只运行之前的 hetero benchmark，两组均采用修正后的 native 图：

```bash
python -m HERA.scripts.run_hetero_relation_benchmark --dataset native --mode hetero hetero_was --variant baseline --native-preprocessing reference_v1 --seed 123 11 1245 --epochs 500 --device cuda:0 --run-dir HERA/logs/hetero_relation_native_semi_imp2d
```

加 `--dry-run` 可先检查第二条命令。要测试删除 DD，把 `--variant baseline` 改成 `--variant no_dd`。
如只增加真实 WAS 而继续保留/复用历史普通 hetero，则使用 `--mode hetero_was`；但那不构成配对消融。

仅检查数据、标签和输入，不训练：

```bash
python -m HERA.scripts.validate_native_was
```

报告：`results/native_was_reference_v1_validation.json`。
新增测试覆盖三类缺陷、CIF 重排、区域扩展、配对图、前向/反向传播、严格权重恢复、旧结果保留和新路径隔离。
本次实现没有启动完整 benchmark，也没有产生新 MAE。
