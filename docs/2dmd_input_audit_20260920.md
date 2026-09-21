# 2DMD 数据与预处理审计（2026-09-20）

## 结论与检查范围

MoS2/WSe2 的现有 `2dmd_mos2`、`2dmd_wse2` 输入没有发现 native 的两类问题：
按读取后最后一个原子错误定位缺陷、用空位 X 替换仍然存在的真实原子。
2DMD 通过母体超胞位置识别缺陷，WAS 已经来自对应母体元素。
此前为 native/semi/imp2d 引入的 `reference_v1` 没有切换 2DMD 的预处理。

本次没有修改训练、构图、原始数据或 checkpoint，也没有重新训练/评估 MAE。
新增了独立审计脚本和本报告。

检查当前本地全部 14,866 个 `initial/*.cif`：

- 用 ASE 读取原子与母体超胞，以周期坐标匹配独立确定空位、替位和原占位元素。
  对 P1 文件直接使用 ASE CIFBlock 构建完整原子列表，每个 descriptor 的首个样本
  再与标准 `ase.io.read` 交叉核对。
- 每个 CIF 与训练使用的 pymatgen 读取结果比对；检查三位小数坐标是否碰撞、
  是否有真实原子无法匹配母体、是否存在非法目标或缺失文件。
- 全量调用现有 hetero 预处理（r=0），核对 2,750,272 个节点的数量、当前元素、
  WAS、缺陷类型、实际缺陷池化标记，以及 92/184 维输入特征。
- 对各子集每个 descriptor 的首个样本，共 929 个结构，额外验证 full、full_x、
  attention、sparse 四种表示。
- 检查三组 seed 的 low→high 划分，以及五个现有 WSe2 checkpoint 的真实划分和 scaler。

这不是逐边张量、所有半径、旋转/平移等价结构、全部历史代码版本或 DFT 目标物理准确性的全面审计。

## 全量输入检查

| 子集 | 结构数 | hetero 输入检查失败 | descriptor 与 CIF 缺陷类型不一致 |
|---|---:|---:|---:|
| MoS2 low | 5,933 | 0 | 0 |
| WSe2 low | 5,933 | 0 | 0 |
| MoS2 high | 500 | 0 | 0 |
| WSe2 high | 500 | 0 | 0 |
| BP high | 500 | 0 | 0 |
| GaSe high | 500 | 0 | 0 |
| hBN high | 500 | 0 | 114 |
| InSe high | 500 | 0 | 0 |

所有目标均为有限数，清单所需 CIF 全部存在；各子集没有重复源 ID。
三位小数键没有造成真实原子碰撞或丢失；实际空位都增加了 X，实际替位被标记为缺陷。
WAS 特征的前 92 维等于普通模式输入，后 92 维对应母体元素；普通模式没有读取 WAS。

按固定母体格点的占位指纹检查，各子集内没有相同占位重复，MoS2/WSe2 的 low 与 high
之间也没有相同占位。该指纹忽略真空厚度，未枚举所有平移、旋转或其他对称等价。

## 已发现的问题

### 1. 旧 `--dataset vacancy` 入口按模式筛选不同样本

当前 CLI 为每个 mode 单独加载所需表示。旧 `load_data_vacancy` 对含替位的结构：

- full/full_x/sparse 保留；
- hetero/attention 的 `get_hetero_vacancy` 返回 `None`，随后样本被过滤。

本地清单共 14,866 个结构，其中实际纯空位结构 1,622 个。因此这个旧入口下上述模型
可能训练/测试在不同样本集合上，不能直接当作公平的模型比较。
已用真实 MoS2 替位样本 `6141cf0fe689ecc4c43cdd4b` 复现。
若一次直接调用 loader 构建多种表示，过滤交集行为不同；这里指出的是当前 CLI 的逐模式加载路径。

`2dmd_low`、`2dmd_high`、`2dmd_mos2`、`2dmd_wse2` 使用支持替位的构图入口，
没有此项筛选差异。`vacancy_mos2`/`vacancy_wse2` 在构图前对所有模式统一选择纯空位。
本次未更改旧入口，以免悄悄改变已有实验定义。

### 2. 分数坐标三位小数舍入会改变几何

`strucure_to_dict` 用三位小数匹配格点，`add_was` 又用该坐标重建节点。
这对普通模式也生效，因为这些表示同样经过 `add_was`。

| 子集 | 最大真实原子坐标偏移（Å） |
|---|---:|
| MoS2 low | 0.014961 |
| WSe2 low | 0.016665 |
| MoS2 high | 0.015590 |
| WSe2 high | 0.017531 |
| 全部材料最大值（InSe high） | 0.020833 |

本次没有发现由舍入导致的错配、丢原子或错误 WAS，但它确实改变了距离和角度。
没有做保持原坐标的配对重训，不能断言 MAE 影响为零。
若修正，应作为独立预处理版本保存，保留现有 checkpoint 对应的历史几何。

### 3. hBN 的 descriptor 不能逐条当作真实缺陷标签

hBN high 的 114/500 条记录中，描述表的缺陷元素组合与 CIF/母体差分不一致。
例如 `BN_B62C2N63_91fe1ce6-6a19-4a26-8caa-82caf4ca6ac2`：

- 描述表：1 个 N 空位、2 个 B→C 替位；
- 结构差分：1 个 B 空位、1 个 B→C 替位、1 个 N→C 替位。

二者总组成相同，具体缺陷位置类型不同。当前图和 WAS 来自 CIF/母体差分，检查通过；
直接用 descriptor 细分缺陷种类会受到影响。不能据此认定 DFT 能量标签错误。
这批不一致没有改变此次清单的纯空位/非纯空位计数。

## low→high 与现有 checkpoint

对 seed 123、11、1245：

| 任务 | Train | Validation | 固定 high test |
|---|---:|---:|---:|
| `2dmd_mos2` | 4,746 | 1,187 | 500 |
| `2dmd_wse2` | 4,746 | 1,187 | 500 |
| `vacancy_mos2` | 623 | 156 | 4 |
| `vacancy_wse2` | 623 | 156 | 3 |

训练和验证仅来自 low，各集合源 ID 不重叠，high 测试集合不随 seed 改变。
`2dmd_low` 与 `2dmd_high` 单独训练使用各自内部随机划分，不能把它们当作自动完成的 low→high 实验。

还检查了本地 WSe2 seed123 的五个 checkpoint：本地 sparse 复现（保存 MAE
0.04128376298）、full、attention、原 hetero、shared hetero energy-mean。
五者训练/验证/测试源 ID 集合完全相同，均为 4,746/1,187/500；高浓度只在测试集，
保存 scaler 的均值和标准差均与其训练集目标一致。
这里读取的是已有 checkpoint 元数据，没有再次推理或验证所保存 MAE 的计算。

## 复查

从 HERA 的父目录运行，只审计、不训练：

```bash
python -m HERA.scripts.audit_2dmd_inputs --workers 4
```

完整输入报告：`results/2dmd_input_audit.json`。
已有权重的本次只读核对：`results/2dmd_saved_split_audit.json`。
相关现有单元测试 18 项通过，涵盖数据入口、浓度划分与 low-checkpoint 推理选择。
