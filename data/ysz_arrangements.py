"""Published Y2Zr30O63 fluorite defect triplets and relaxed DFT total energies."""
from collections import Counter, defaultdict
import csv
import hashlib
import io
from itertools import combinations
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import warnings
import zipfile

import numpy as np
import spglib
from ase.io import read, write

from ..arrangement_sources import ROOT, SOURCES

ARCHIVE = ROOT.parent / 'dataset' / 'ysz_parkes' / 'jp5b02031_si_001.zip'
DIGEST = '83b80d929f90f83121355670f07d03b0a81ada364a0118853a4c3e678ee3647c'
SOURCE = 'https://doi.org/10.1021/acs.jpca.5b02031.s001'
PAPER = 'https://doi.org/10.1021/acs.jpca.5b02031'
TITLE = '钇稳定氧化锆 Zr₃₀Y₂O₆₃'
INTERPRETATION = ('在 32 个阳离子位和 64 个氧位的萤石超胞内固定两个 Y 替位和一个氧空位，'
                  '从初始占位预测弛豫 DFT 能量。28 条原始记录中，构型 21 和 22 经母晶格对称操作等价，'
                  '两个标签取均值后只计一次。')
SCHEMA = SOURCES['ysz'].schema
RESULTS = SOURCES['ysz'].results
BUNDLE = 'YSZ_fixed_composition_DFT.zip'
SOURCE_MEMBER = 'jp5b02031_si_001.zip'
SPLIT_SEED = 20261110


def original_payload(archive):
    payload = Path(archive).read_bytes()
    if hashlib.sha256(payload).hexdigest() != DIGEST:
        with zipfile.ZipFile(io.BytesIO(payload)) as curated:
            payload = curated.read(SOURCE_MEMBER)
    if hashlib.sha256(payload).hexdigest() != DIGEST:
        raise ValueError('YSZ supplement differs from the pinned author release')
    return payload


def cell_atoms(raw):
    # Xx is the release's empty oxygen site, including in the relaxed .cell.
    # ASE spells a dummy atom X. Do not pass the raw energy-rank ID to the model.
    text = re.sub(r'(?m)^(\s*)Xx(?=\s)', r'\1X', raw.decode())
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning, module=r'ase\.(io|calculators)\.castep')
        atoms = read(io.StringIO(text), format='castep-cell')
    atoms.calc = None
    return atoms


def site_grid(atoms):
    frac = atoms.get_scaled_positions()
    discrete = np.rint(8 * frac)
    if np.max(np.abs(8 * frac - discrete)) > 1e-5:
        raise ValueError('Initial positions do not coincide with the fluorite sites')
    return discrete.astype(int) % 8


def parent_operations(parent):
    grid = site_grid(parent)
    by_grid = {tuple(q): i for i, q in enumerate(grid)}
    symmetry = spglib.get_symmetry((parent.cell.array, grid / 8, parent.numbers), symprec=1e-4)
    perms = []
    for rotation, tr in zip(symmetry['rotations'], symmetry['translations'], strict=True):
        perm = [by_grid[tuple(q)] for q in (grid @ rotation.T + np.rint(8 * tr).astype(int)) % 8]
        if not np.array_equal(parent.numbers[perm], parent.numbers):
            raise ValueError('Invalid fluorite parent operation')
        perms.append(perm)
    perms = np.unique(perms, axis=0)
    if len(by_grid) != 96 or len(perms) != 1536:
        raise ValueError('Unexpected parent symmetry')
    return by_grid, perms


def orbit(perms, vacancy, solutes):
    triples = np.column_stack([perms[:, vacancy], np.sort(perms[:, solutes], axis=1)])
    order = np.lexsort((triples[:, 2], triples[:, 1], triples[:, 0]))
    return tuple(int(x) for x in triples[order[0]])


def prepare(archive=ARCHIVE):
    payload = original_payload(archive)
    base = 'Associated_Content/'
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        parent = cell_atoms(z.read(base + 'Parent_Materials/c-ZrO2.cell'))
        grid = site_grid(parent)
        parent = parent[np.lexsort((grid[:, 2], grid[:, 1], grid[:, 0]))]
        if Counter(parent.get_chemical_symbols()) != {'Zr': 32, 'O': 64}:
            raise ValueError('Unexpected host')
        by_grid, perms = parent_operations(parent)
        lines = z.read(base + 'Energies_and_Force_Constants_Matrix/YSZ_DFT_Relaxed_Total_Energies').decode().splitlines()
        energies = {}
        for line in lines[1:]:
            if line.strip():
                name, value = line.split()
                if name in {'c-ZrO2_formula_unit', 'Y2O3_formula_unit'}:
                    continue
                matched = re.fullmatch(r'YSZ_(\d+)_relaxed\.cell', name)
                if not matched:
                    raise ValueError(f'Unknown entry in the energy table: {name}')
                idx = int(matched[1])
                if idx in energies or not np.isfinite(float(value)):
                    raise ValueError('Invalid energy record')
                energies[idx] = float(value)
        if set(energies) != set(range(1, 29)):
            raise ValueError('Incomplete energy table')
        groups = defaultdict(list)
        for idx, value in energies.items():
            initial = cell_atoms(z.read(f'{base}YSZ_Unrelaxed_Structures/YSZ_{idx}_unrelaxed.cell'))
            final = cell_atoms(z.read(f'{base}YSZ_Relaxed_Structures/YSZ_{idx}_relaxed.cell'))
            for a in (initial, final):
                if Counter(a.get_chemical_symbols()) != {'Zr': 30, 'Y': 2, 'O': 63, 'X': 1}:
                    raise ValueError(f'Assigned composition changed in structure {idx}')
            if not np.allclose(initial.cell.array, parent.cell.array, atol=1e-8, rtol=0):
                raise ValueError('Not the same initial supercell')
            site = np.array([by_grid[tuple(q)] for q in site_grid(initial)])
            if len(set(site)) != 96:
                raise ValueError('Not a bijective site assignment')
            nominal = initial.numbers.copy()
            nominal[nominal == 0] = 8
            nominal[nominal == 39] = 40
            if not np.array_equal(nominal, parent.numbers[site]):
                raise ValueError(f'Unknown defect type in {idx}')
            key = orbit(perms, site[initial.numbers == 0][0], site[initial.numbers == 39])
            groups[key].append((idx, value))
    records, atoms, raw_rows = [], {}, []
    for key, members in sorted(groups.items()):
        vacancy, y1, y2 = key
        sid = 'ysz-' + hashlib.sha256(bytes(key)).hexdigest()[:16]
        a = parent.copy()
        old_z = a.numbers.copy()
        a.numbers[[y1, y2]] = 39
        r1, r2 = sorted(parent.get_distances(vacancy, [y1, y2], mic=True))
        vacancy_frac = parent.get_scaled_positions()[vacancy].copy()
        a.new_array('was_Z', old_z)
        a.new_array('defect_site', (a.numbers != old_z).astype(int))
        a.new_array('parent_site', np.arange(len(parent)))
        del a[vacancy]
        values = [x[1] for x in members]
        target = float(np.mean(values))
        stratum = int(np.isclose([r1, r2], parent.cell.lengths()[0] * np.sqrt(3) / 8).sum())
        row = dict(source_id=sid, group='Zr30Y2O63', stratum=stratum, energy_eV_cell=target,
                   y1_v_A=float(r1), y2_v_A=float(r2), y_y_A=float(parent.get_distance(y1, y2, mic=True)),
                   vacancy_parent_site=vacancy, y_parent_site_1=y1, y_parent_site_2=y2,
                   author_ids=','.join(str(i) for i, _ in sorted(members)), source_label_spread_eV=max(values)-min(values))
        records.append(row)
        for idx, value in members:
            raw_rows.append(dict(author_id=idx, source_id=sid, published_DFT_total_eV=value))
        a.info.update(source_id=sid, vacancy_fractional=vacancy_frac, vacancy_was_Z=8,
                      relaxed_DFT_total_eV=target)
        atoms[sid] = a
    z_site = np.flatnonzero(parent.numbers == 40)
    fixed_o = np.flatnonzero(parent.numbers == 8)[0]
    possible = {orbit(perms, fixed_o, pair) for pair in combinations(z_site, 2)}
    if (len(possible) != 33 or len(groups) != 27 or not set(groups) <= possible or len(atoms) != 27
            or sorted(tuple(sorted(i for i, _ in v)) for v in groups.values() if len(v)>1) != [(21, 22)]):
        raise ValueError('Unexpected coverage or duplicated occupational class')
    return records, atoms, sorted(raw_rows, key=lambda x: x['author_id']), payload


def partitions(records):
    rng = np.random.default_rng(SPLIT_SEED)
    folds = [[], [], []]
    by_stratum = defaultdict(list)
    for r in records:
        by_stratum[r['stratum']].append(r['source_id'])
    # Balance vacancy coordination without inspecting any DFT energy or author rank.
    cursor = 0
    for k, ids in sorted(by_stratum.items()):
        for sid in rng.permutation(sorted(ids)).tolist():
            folds[cursor % 3].append(sid)
            cursor += 1
    result = []
    ids = {r['source_id'] for r in records}
    for test in folds:
        rest = ids - set(test)
        valid = set(rng.permutation(sorted(rest))[:4])
        part = dict(train=sorted(rest-valid), val=sorted(valid), test=sorted(test))
        if [len(v) for v in part.values()] != [14, 4, 9]:
            raise ValueError('Incorrect YSZ fold')
        result.append(part)
    if Counter(k for f in result for k in f['test']) != Counter(ids):
        raise ValueError('Not a partition of the 27 distinct configurations')
    return result


def write_bundle(path, records, atoms, raw, payload, folds):
    readme = f"""Y2Zr30O63 defect triplets in yttria-stabilized zirconia
Parkes et al. (2015), {PAPER}
Original associated content: {SOURCE}

The embedded author ZIP contains all 28 unrelaxed and 28 relaxed CASTEP structures
and their relaxed supercell DFT total energies (eV). This benchmark uses the
published ideal occupations as input. All configurations have two Y on Zr sites
and one charge-compensating O vacancy. Xx in the source denotes the empty site.
The 95 real atoms are in configurations.extxyz; vacancy_fractional locates the
missing oxygen. was_Z and defect_site give the parent species and substitution mask.
Poses and atomic ordering are standardized under the ordered parent's 1536 symmetry
operations. There are 27 published occupation classes among 33 possible in this
supercell. IDs 21 and 22 are symmetry equivalent. Their two DFT energies are retained
in published_labels.csv and their mean is used once as the class label.

The same chemical composition occurs in every train, validation and test set.
folds.json covers each class in one test fold. Compare only energies predicted by
the same held-out model. Author IDs are ranked by energy and never enter features
or membership. extended XYZ uses relaxed_DFT_total_eV in the frame metadata.
"""
    write_arrangement_bundle(path, records, atoms, raw, payload, folds, SOURCE_MEMBER, DIGEST, readme)


def write_arrangement_bundle(path, records, atoms, raw, payload, folds, source_member, digest, readme):
    frames = [atoms[r['source_id']] for r in records]
    stream = io.StringIO()
    write(stream, frames, format='extxyz')
    xyz = stream.getvalue()
    for a, b in zip(frames, read(io.StringIO(xyz), ':', format='extxyz'), strict=True):
        if (a.info['source_id'] != b.info['source_id'] or a.info['relaxed_DFT_total_eV'] != b.info['relaxed_DFT_total_eV']
                or not np.array_equal(a.numbers, b.numbers) or not np.allclose(a.positions, b.positions, atol=1e-8, rtol=0)
                or not np.array_equal(a.arrays['defect_site'], b.arrays['defect_site'])
                or not np.array_equal(a.arrays['was_Z'], b.arrays['was_Z'])
                or any(k in a.info and not np.allclose(a.info[k], b.info[k], atol=1e-8, rtol=0)
                       for k in ('vacancy_fractional', 'vacancy_fractionals'))):
            raise ValueError('Extended XYZ round trip failed')
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as temp:
        tmp = Path(temp)/path.name
        with zipfile.ZipFile(tmp, 'w', compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr(source_member, payload)
            z.writestr('configurations.extxyz', xyz)
            z.writestr('folds.json', json.dumps(folds, indent=2)+'\n')
            z.writestr('README.txt', readme)
            for name, rows in [('occupations.csv', records), ('published_labels.csv', raw)]:
                buffer = io.StringIO(newline='')
                writer = csv.DictWriter(buffer, fieldnames=rows[0])
                writer.writeheader()
                writer.writerows(rows)
                z.writestr(name, buffer.getvalue())
        with zipfile.ZipFile(tmp) as check:
            if hashlib.sha256(check.read(source_member)).hexdigest() != digest:
                raise ValueError('Original archive changed in prepared package')
        tmp.replace(path)
