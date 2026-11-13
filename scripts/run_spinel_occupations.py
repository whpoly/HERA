"""Kasamatsu et al. dilute cation occupations predicting relaxed DFT energies."""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile
from tempfile import TemporaryDirectory
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spglib
import torch
from ase import Atoms
from ase.io import read, write

from ..data.datasets import dataset_index_for_mode, init_elem_embedding
from ..data.spinel_antisites import ROOT, RESULTS, first_shell_masks
from .alignn_benchmark_support import persist_contract
from .analyze_spinel_antisites import compare, shell_baseline, verified_fits
from .run_spinel_antisites import MODES, structures, train_fold


SCHEMA = 'abics_mgal2o4_dilute_numbered_holdout_v1'
DIGEST = '7406caa4069f0bded6e1a5d086ae246ff88c4f2d6345980f271a1b165d9fd97a'
ARCHIVE = ROOT.parent / 'dataset' / 'spinel_abics' / 'MgAl2O4_data.tar.gz'
OUT = RESULTS / 'abics_ideal_cations'
DOI = 'https://doi.org/10.1063/5.0096645'
SOURCE = ('https://isspns-gitlab.issp.u-tokyo.ac.jp/abics-dev/abics-gallery/-/blob/'
          '202e5a0ed16903246a3e3877e1240b0e1110359a/paper/arxiv_2022/MgAl2O4data/data.tar.gz')
ORDER_THRESHOLD = 0.020
AUTHOR_FILE = 'author_MgAl2O4_data.tar.gz'


def read_xsf(payload):
    lines = payload.decode().splitlines()
    match = re.fullmatch(r'# total energy =\s*([-+\d.eE]+) eV', lines[0])
    if match is None:
        raise ValueError('Missing DFT total energy in XSF header')
    start = lines.index('PRIMVEC') + 1
    cell = np.loadtxt(lines[start:start+3])
    start = lines.index('PRIMCOORD') + 1
    table = [s.split() for s in lines[start+1:] if s.strip()]
    if lines[start].split() != ['192', '1'] or len(table) != 192:
        raise ValueError('Unexpected XSF structure')
    z = np.array([{'Mg': 12, 'Al': 13}[r[0]] for r in table], dtype=np.uint8)
    x = np.array([r[1:4] for r in table], dtype=float) @ np.linalg.inv(cell)
    grid = np.rint(x * 16).astype(int)
    if (np.max(np.abs(x * 16 - grid)) > 1e-6 or Counter(z) != {12: 64, 13: 128}
            or not np.isfinite(float(match[1]))):
        raise ValueError('Composition, energy or ideal-lattice coordinates differ from the publication')
    grid %= 16
    order = np.lexsort((grid[:, 2], grid[:, 1], grid[:, 0]))
    return cell, grid[order], z[order], float(match[1])


def parent_sites(cell, grid):
    # The shorter octahedral-octahedral contact has six neighbors on each
    # octahedral site and zero on a tetrahedral site.
    frac = grid / 16
    delta = frac[:, None, :] - frac[None, :, :]
    d = np.linalg.norm((delta - np.rint(delta)) @ cell, axis=2)
    coordination = ((d > 0.001) & (d < 3.05)).sum(axis=1)
    if Counter(coordination) != {0: 64, 6: 128}:
        raise ValueError('Cation topology does not match 64 tetrahedral and 128 octahedral sites')
    was = np.where(coordination == 6, 13, 12)
    sym = spglib.get_symmetry((cell, frac, was), symprec=0.0001)
    by_pos = {tuple(v): j for j, v in enumerate(grid)}
    permutations = []
    for rot, tr in zip(sym['rotations'], sym['translations'], strict=True):
        moved = (grid @ rot.T + np.rint(tr * 16).astype(int)) % 16
        perm = [by_pos[tuple(v)] for v in moved]
        if not np.array_equal(was, was[perm]):
            raise ValueError('Normal cation sublattice changed under symmetry')
        permutations.append(perm)
    perms = np.unique(np.array(permutations, dtype=np.int16), axis=0)
    if len(perms) != 1536:
        raise ValueError('Incomplete cation symmetry group')
    return was, perms


def source_payload(archive):
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            payload = z.read(AUTHOR_FILE)
    else:
        payload = archive.read_bytes()
    if hashlib.sha256(payload).hexdigest() != DIGEST:
        raise ValueError('The author archive differs from the pinned version')
    return payload


def collect(archive):
    payload = source_payload(archive)
    records, atoms, raw, first, energy_ranges = [], {}, {}, {}, {}
    reference = None
    with tarfile.open(fileobj=io.BytesIO(payload), mode='r:gz') as tar:
        members = {int(re.fullmatch(r'data/structure\.(\d+)\.xsf', m.name)[1]): m
                   for m in tar if m.isfile()}
        if sorted(members) != list(range(1800)):
            raise ValueError('Incomplete public source cohort')
        for idx, member in sorted(members.items()):
            contents = tar.extractfile(member).read()
            cell, grid, z, e = read_xsf(contents)
            if reference is None:
                reference = (cell, grid)
                was, perms = parent_sites(cell, grid)
                parent = Atoms(numbers=was, scaled_positions=grid/16, cell=cell, pbc=True)
                shells = first_shell_masks(parent)
            if not np.array_equal(grid, reference[1]) or not np.allclose(cell, reference[0], rtol=0, atol=1e-8):
                raise ValueError('Cation lattice differs between records')
            k = int(np.sum((was == 13) & (z == 12)))
            if k != int(np.sum((was == 12) & (z == 13))):
                raise ValueError('Unbalanced antisite species')
            signature = hashlib.sha256(min(s.tobytes() for s in z[perms])).hexdigest()
            rep = first.setdefault(signature, member.name)
            extrema = energy_ranges.setdefault(signature, [e, e])
            extrema[:] = [min(extrema[0], e), max(extrema[1], e)]
            dilute = 1 <= k <= 4
            r = dict(source_id=member.name, source_index=idx, k_each=k, inversion=k/64,
                     energy_eV_cell=e, representative_id=rep, occupation_sha256=signature,
                     is_representative=(rep == member.name), in_dilute_domain=dilute)
            defects = z != was
            for kind, contacts in shells.items():
                r[f'antisite_contacts_{kind}'] = int(contacts[np.ix_(defects, defects)].sum())
            records.append(r)
            if dilute:
                raw[member.name] = contents
                if r['is_representative']:
                    a = Atoms(numbers=z, scaled_positions=grid/16, cell=cell, pbc=True)
                    a.new_array('was_Z', was.copy())
                    a.new_array('is_antisite', (z != was).astype(int))
                    a.info.update(source_id=member.name, k_each=k, relaxed_dft_eV_448_atom_cell=e)
                    atoms[member.name] = a
    cohort = [r for r in records if r['in_dilute_domain'] and r['is_representative']]
    ids = {part: [] for part in ('train', 'val', 'test')}
    for r in cohort:
        i = r['source_index']
        part = 'train' if i < 900 else ('val' if i < 1500 else 'test')
        ids[part].append(r['source_id'])
    ids = {p: sorted(v) for p, v in ids.items()}
    if len(first) != 1756 or [len(ids[p]) for p in ('train','val','test')] != [51, 23, 22]:
        raise ValueError('Unexpected symmetry-deduplicated acquisition blocks')
    return records, cohort, ids, atoms, raw, max(b-a for a,b in energy_ranges.values())


def write_archive(path, records, cohort, ids, atoms, raw, source=ARCHIVE):
    stream = io.StringIO()
    write(stream, [atoms[r['source_id']] for r in cohort], format='extxyz')
    xyz = stream.getvalue()
    for row, frame in zip(cohort, read(io.StringIO(xyz), index=':', format='extxyz'), strict=True):
        original = atoms[row['source_id']]
        if (frame.info['source_id'] != row['source_id'] or frame.info['relaxed_dft_eV_448_atom_cell'] != row['energy_eV_cell']
                or not np.array_equal(frame.numbers, original.numbers)
                or not np.allclose(frame.positions, original.positions, rtol=0, atol=1e-8)
                or not np.array_equal(frame.arrays['was_Z'], original.arrays['was_Z'])
                or not np.array_equal(frame.arrays['is_antisite'], original.arrays['is_antisite'])):
            raise ValueError('Ideal-cation XYZ round-trip failed')
    readme = f"""Kasamatsu et al. MgAl2O4, DOI {DOI}
Original XSF archive: {SOURCE}
The author explicitly predicts the energy of the entire DFT-relaxed 448-atom crystal
from the ideal Mg/Al occupations. Each released XSF contains 192 cations, Mg64Al128.
The 256 oxygen sites are omitted. Do not treat these as relaxed all-atom coordinates
or divide the total energy by 192 to obtain an energy per atom of MgAl2O4.

{AUTHOR_FILE} is the complete byte-preserved 1,800-file upstream archive.
records_all.csv lists all 1,800 author records and their symmetry representatives.
The dilute domain is defined by 1 through 4 Al atoms on 64 tetrahedral Mg sites,
with the same number of Mg atoms on octahedral Al sites. Original XSF files in
this domain are in data/. ideal_cations.extxyz contains 96 symmetry-distinct dilute
structures with per-site was_Z and is_antisite. The reference DFT total energy
in eV per 448-atom cell is stored in relaxed_dft_eV_448_atom_cell in the XYZ header.

The first (lowest numbered) file represents each occupation class under the 1,536
symmetry permutations of the parent lattice, irrespective of energy. Earlier numbered
structures enter training (0-899), validation (900-1499), or the fixed test set
(1500-1799). Each of the 22 test occupations is absent from the preceding blocks.
The 75 comparisons pair distinct held-out occupations at exactly the same inversion.
The antisite_contacts columns count near-neighbor pairs of antisites on each type
of cation-cation first shell of the ideal parent.
"""
    with TemporaryDirectory(dir=path.parent) as tmp:
        candidate = Path(tmp)/path.name
        with zipfile.ZipFile(candidate, 'w', compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr(AUTHOR_FILE, source_payload(source))
            for name, contents in raw.items():
                z.writestr(name, contents)
            z.writestr('README.txt', readme)
            z.writestr('records_all.csv', pd.DataFrame(records).to_csv(index=False))
            z.writestr('ideal_cations.extxyz', xyz)
            z.writestr('split.json', json.dumps(ids, indent=2) + '\n')
        with zipfile.ZipFile(candidate) as z:
            if (z.testzip() is not None or hashlib.sha256(z.read(AUTHOR_FILE)).hexdigest() != DIGEST
                    or any(z.read(k) != v for k, v in raw.items())):
                raise ValueError('Original XSF did not survive archive round trip')
        candidate.replace(path)


def plot_gaps(path, pairs, results, shell_mae):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.4, 4.55), gridspec_kw={'width_ratios': [1, 1.2]})
    bound = 1.07 * pairs.loc[pairs['mode'].isin(['hetero','full']), ['dft_gap_eV_cell', 'predicted_gap_eV_cell']].abs().max().max()
    ax.plot([-bound, bound], [-bound, bound], color='#717680', lw=1, ls='--')
    for mode, color, marker in [('full', '#7389a0', 'o'), ('hetero', '#19736a', '^')]:
        data = pairs[pairs['mode'] == mode]
        ax.scatter(data.dft_gap_eV_cell, data.predicted_gap_eV_cell, color=color, marker=marker,
                   label=mode, alpha=.63, s=22, edgecolors='white', linewidths=.25)
    ax.set(xlim=(-bound, bound), ylim=(-bound, bound), xlabel='DFT energy gap (eV/cell)',
           ylabel='Predicted energy gap (eV/cell)', aspect='equal')
    ax.set_title('Ideal cations to relaxed energies', fontsize=11, loc='left')
    ax.legend(frameon=False)
    bars = results[['mode','gap_MAE_eV_cell']].copy()
    bars.loc[len(bars)] = ['first-shell linear', shell_mae]
    bars.sort_values('gap_MAE_eV_cell', ascending=False, inplace=True)
    colors = ['#19736a' if m == 'hetero' else '#b5babf' if m in ('inversion_only','first-shell linear')
              else '#7389a0' for m in bars['mode']]
    bx.barh(bars['mode'], bars.gap_MAE_eV_cell, color=colors, height=.7)
    bx.set(xlabel='Energy-gap MAE (eV / 448-atom cell)')
    bx.set_title('Equal-count pairs on 22 held-out structures', fontsize=11, loc='left')
    for axis in (ax,bx):
        axis.spines[['top','right']].set_visible(False)
        axis.tick_params(labelsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=185, bbox_inches='tight')
    plt.close(fig)


def analyze(args):
    s, p, audits = verified_fits(args.run_dir, args.seed, ORDER_THRESHOLD)
    results = compare(s, p, structures_expected=22, pairs_by_k={2:15, 3:45, 4:15},
                      order_threshold=ORDER_THRESHOLD)
    with zipfile.ZipFile(args.results / 'abICS_MgAl2O4_dilute.zip') as z, z.open('records_all.csv') as f:
        records = pd.read_csv(f)
    labels = records.set_index('source_id').energy_eV_cell
    if not np.allclose(s.dft_eV_cell.to_numpy(), labels.loc[s.source_id].to_numpy(), atol=1e-7, rtol=0):
        raise ValueError('Held-out energies differ from author archive')
    folds = json.loads((args.run_dir/'data_contract.json').read_text(encoding='utf8'))['partitions']
    shell, coef = shell_baseline(records[records.is_representative & records.in_dilute_domain], folds)
    for frame, name in ((s, 'oof_structures'), (p, 'oof_energy_gaps'), (audits, 'training_audits'),
                        (results, 'model_comparison'), (shell, 'first_shell_gaps'), (coef, 'first_shell_coefficients')):
        frame.to_csv(args.results / (name + '.csv'), index=False)
    plot_gaps(args.results / 'heldout_energy_gaps.png', p, results, shell.abs_gap_error_eV_cell.mean())
    text = ['# MgAl₂O₄ 稀反位占位与弛豫后总能', '',
            'Kasamatsu 等人的目标是用理想晶格的占位预测完整晶体弛豫后的 DFT 能量。物理超胞为 '
            'Mg₆₄Al₁₂₈O₂₅₆，共 448 个原子，作者发布的 XSF 输入为 192 个阳离子位点。'
            f'氧坐标未包含在发布的模型输入中。[论文]({DOI}) · [作者数据]({SOURCE})', '',
            '原始 1,800 个文件对应 1,756 个对称不等价占位。本文的稀反位范围为 64 个四面体位中有 1–4 个 Al 反位，'
            '八面体中的 Mg 反位数与之相同。四组依次有 5、31、29、31 个不等价构型。'
            '同样化学式且反位数相同的 DFT 能量跨度依次为 0.2021、0.5471、0.5376、0.8620 eV/448 原子超胞。', '',
            '每个占位类别保留编号最小的原文件。按原文件编号分成 0–899（训练）、900–1499（验证）和 '
            '1500–1799（测试）。96 个不等价排布依次分为 51、23 和 22 个。测试集包含 6 个 k=2、10 个 '
            'k=3 和 6 个 k=4 构型，k 是每类反位的原子数，共形成 75 对同 k 的比较。'
            '编号划分只依赖发布文件，不假定不同编号的构型在统计上相互独立。', '',
            f'所有 ALIGNN 变体从头训练，权重种子 {args.seed}，使用同一批坐标、数据划分和训练集内按 k 得到的能量均值。'
            '训练上限 500 轮，验证 MAE 早停耐心 50 轮，微批量为 2、梯度累积 4 步。排序统计采用 '
            '0.020 eV/超胞的 DFT 能隙门槛。原档案中对称等价排布的最大能量差为 0.016 eV/超胞。', '',
            '| 模型 | 同 k 能量差 MAE（eV/448 原子超胞） | 总能 MAE（eV/超胞） | 排序正确/可分对 |',
            '|---|---:|---:|---:|']
    for r in results.itertuples(index=False):
        order = f'{r.correctly_ordered}/{r.informative_pairs}' if r.mode != 'inversion_only' else '全部预测平局'
        text.append(f'| {r.mode} | {r.gap_MAE_eV_cell:.5f} | {r.abs_energy_MAE_eV_cell:.5f} | {order} |')
    comparable = shell[shell.dft_gap_eV_cell.abs() > ORDER_THRESHOLD]
    shell_correct = (comparable.predicted_gap_eV_cell * comparable.dft_gap_eV_cell > 0).sum()
    text.extend(['', '用相同训练集拟合三类第一壳层反位接触数的线性物理对照，'
                 f'能量差 MAE 为 {shell.abs_gap_error_eV_cell.mean():.5f} eV/448 原子超胞，'
                 f'排序正确 {shell_correct}/{len(comparable)}。', '',
                 'model_comparison.csv 同时给出以 hetero 为基准的误差差、按 k 分层、按留出结构重采样的 '
                 '95% 区间。区间条件为本次训练和固定划分，75 对构型共享 22 个测试结构。', '',
                 'abICS_MgAl2O4_dilute.zip 包含作者的完整 1,800 条原始档案，以及稀反位域的原始 XSF、'
                 '96 个不等价占位的扩展 XYZ、全部原文件能量与对称分组、训练/验证/测试 ID。', '',
                 '在 HERA 的父目录可用以下命令从数据包复现。', '',
                 '```text', 'python -m HERA.scripts.run_spinel_occupations --archive HERA/results/spinel_antisites_20261110/abics_ideal_cations/abICS_MgAl2O4_dilute.zip --run-dir HERA/logs/abics_reproduction --results HERA/results/abics_reproduction',
                 '```', ''])
    (args.results / 'results.md').write_text('\n'.join(text), encoding='utf8')
    print(results[['mode','gap_MAE_eV_cell','correctly_ordered']].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=ARCHIVE)
    parser.add_argument('--run-dir', type=Path, default=ROOT/'logs'/SCHEMA)
    parser.add_argument('--results', type=Path, default=OUT)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--analyze-only', action='store_true')
    parser.add_argument('--mode', nargs='+', choices=MODES, default=list(MODES))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--cpu-threads', type=int, default=2)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.analyze_only:
        analyze(args)
        return
    torch.set_num_threads(args.cpu_threads if args.device == 'cpu' else 1)
    records, cohort, ids, atoms, raw, spread = collect(args.archive)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    args.results.mkdir(parents=True, exist_ok=True)
    persist_contract(args.run_dir/'data_contract.json',
                     dict(schema=SCHEMA, modes=list(MODES), publication=DOI, source=SOURCE, archive_sha256=DIGEST,
                          domain='Dilute cation inversion, 1 <= k_each <= 4 of 64 tetrahedral sites',
                          input='Released ideal 192-cation occupations, fixed sites rounded to 1/16 fractional grid',
                          target='Relaxed DFT total energy of physical 448-atom cell, centered on train mean per k',
                          primary='Signed energy gap MAE at fixed k on the final unseen source block',
                          order_threshold_eV_cell=ORDER_THRESHOLD, data_rows=1800, distinct_occupations=1756,
                          max_equivalent_energy_spread_eV_cell=spread, dilute_unique=len(cohort), partitions=[ids],
                          selection='minimum validation structure MAE', train_batch_size=2, grad_accum_steps=4,
                          rule='First source index per symmetry orbit. Train 0-899, val 900-1499, test 1500-1799.'))
    bundle = args.results/'abICS_MgAl2O4_dilute.zip'
    if args.prepare_only or not bundle.exists():
        write_archive(bundle, records, cohort, ids, atoms, raw, source=args.archive)
    if args.prepare_only:
        return
    init_elem_embedding(ROOT/'atom_init.json')
    lookup = {r['source_id']: r for r in cohort}
    offsets = {k: float(np.mean([lookup[s]['energy_eV_cell'] for s in ids['train'] if lookup[s]['k_each']==k]))
               for k in (1,2,3,4)}
    targets = torch.tensor([r['energy_eV_cell'] - offsets[r['k_each']] for r in cohort], dtype=torch.float32)
    for mode in args.mode:
        dataset = [None]*4
        dataset[dataset_index_for_mode(mode)] = structures(mode, cohort, atoms, args.archive, member_prefix='')
        train_fold(mode, 0, ids, lookup, offsets, targets, dataset, args, schema=SCHEMA, digest=DIGEST,
                   batch_size=2, accumulation=4)
    if set(args.mode) == set(MODES):
        analyze(args)


if __name__ == '__main__':
    main()
