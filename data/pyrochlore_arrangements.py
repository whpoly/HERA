"""All 97 equimolar B-site occupation classes of the La2(Zr,Sn)2O7 pyrochlore."""
from collections import Counter
import hashlib
import io
from pathlib import Path
import re
import zipfile

import numpy as np
import spglib
from ase.io import read

from ..arrangement_sources import SOURCES
from .hzo_vacancy_sites import site_stratified_partitions
from .ysz_arrangements import ROOT, write_arrangement_bundle


SOURCE_FILE = 'La16Zr8Sn8O56_author_files.zip'
ARCHIVE = ROOT.parent/'dataset'/'pyrochlore_nmr'/SOURCE_FILE
DIGEST = 'f79f403613c1f6d8cf64f17d83e66fa31ce1903cfad4bb7eb1688af54fe480b8'
SOURCE = 'https://github.com/shamgom/NMR_Solid_Solutions/tree/26f9d03a17c863da0d10096328ac0bedc255a98f'
PAPER = 'https://doi.org/10.1039/D5SC04550A'
TITLE = '烧绿石 La₁₆Zr₈Sn₈O₅₆ 中八个 Sn 替位的排布'
INTERPRETATION = ('以 La₂Zr₂O₇ 为母相，在 16 个 B 位固定八个 Sn 替位。12,870 种占位分成 97 个对称独立排布。'
                  '全部排布均有已公开的初始结构和 CASTEP 弛豫能量，逐项核对了作者的能量表。')
SCHEMA = SOURCES['pyrochlore'].schema
RESULTS = SOURCES['pyrochlore'].results
BUNDLE = 'La16Zr8Sn8O56_fixed_composition_DFT.zip'
SPLIT_SEED = 20261110
BASE = 'Code_and_data/figure_3_Chemical_potential/can/'
CASTEP = 'Code_and_data/figure_8_NMR_spectra_Exp_versus_ML/pure_dft_spectra/castep_files/Zr08/'


def prepare(archive=ARCHIVE):
    payload = Path(archive).read_bytes()
    if hashlib.sha256(payload).hexdigest() != DIGEST:
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            payload = z.read(SOURCE_FILE)
    if hashlib.sha256(payload).hexdigest() != DIGEST:
        raise ValueError('Pyrochlore source snapshot differs from the pinned dataset')
    records, geometries, raw_rows = [], {}, []
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        parent = read(io.StringIO(z.read(BASE+'n00/c00001.vasp').decode()), format='vasp')
        metals = np.flatnonzero(parent.numbers == 40)
        if Counter(parent.get_chemical_symbols()) != {'La':16, 'Zr':16, 'O':56}:
            raise ValueError('Unexpected La2Zr2O7 parent')
        frac = parent.get_scaled_positions()
        grid = np.rint(frac[metals]*4).astype(int) % 4
        by_grid = {tuple(r):i for i,r in enumerate(grid)}
        sym = spglib.get_symmetry((parent.cell.array, frac, parent.numbers), symprec=1e-3)
        perms = np.unique([[by_grid[tuple(k)] for k in (grid @ rot.T + np.rint(4*t).astype(int)) % 4]
                           for rot,t in zip(sym['rotations'],sym['translations'],strict=True)], axis=0)
        energies = np.loadtxt(io.BytesIO(z.read(BASE+'n08/ENERGIES')))
        log = [list(map(int,l.split())) for l in z.read(BASE+'n08/OUTSOD').decode().splitlines()[2:] if l.strip()]
        if len(energies) != 97 or len(log) != len(energies) or not np.isfinite(energies).all():
            raise ValueError('Incomplete author enumeration or DFT table')
        common_cell = None
        for (i, degeneracy, *sites), total in zip(log,energies,strict=True):
            raw = read(io.StringIO(z.read(BASE+f'n08/c{i:05d}.vasp').decode()), format='vasp')
            if Counter(raw.get_chemical_symbols()) != {'La':16,'Zr':8,'Sn':8,'O':56}:
                raise ValueError(f'Composition mismatch in source configuration {i}')
            if common_cell is None:
                common_cell = raw.cell.copy()
                parent.set_cell(common_cell, scale_atoms=True)
                b_dist = parent.get_all_distances(mic=True)[np.ix_(metals,metals)]
                shell_dist = np.unique(np.round(b_dist,4))[1:]
            if not np.allclose(raw.cell.array, common_cell.array, atol=1e-8):
                raise ValueError('Initial cells differ between occupations')
            q = raw.get_scaled_positions()
            delta = q[:,None,:] - frac[None,:,:]
            lengths = np.linalg.norm(delta-np.rint(delta),axis=2)
            assignment = lengths.argmin(axis=1)
            restored = raw.numbers.copy()
            restored[restored == 50] = 40
            if (len(set(assignment)) != 88 or lengths.min(axis=1).max() > 1e-6
                    or not np.array_equal(restored, parent.numbers[assignment])
                    or sorted(assignment[raw.numbers == 50]) != sorted(metals[np.array(sites)-1])):
                raise ValueError(f'Occupation numbers do not match the author POSCAR {i}')
            output = z.read(CASTEP + f'{i:03d}.castep').decode()
            scf = re.findall(r'Total energy corrected for finite basis set\s*=\s*([-\d.]+) eV', output)
            if not scf or abs(float(scf[-1]) - total) > 1e-6:
                raise ValueError(f'CASTEP output and energy table differ for {i}')
            images = np.unique(np.sort(perms[:,np.array(sites)-1],axis=1),axis=0)
            if len(images) != degeneracy:
                raise ValueError(f'Incorrect symmetry degeneracy for {i}')
            occ = images[0]
            sid = 'pyro-' + hashlib.sha256(bytes(occ.tolist())).hexdigest()[:16]
            a = parent.copy()
            a.new_array('was_Z', a.numbers.copy())
            a.numbers[metals[occ]] = 50
            a.new_array('defect_site', (a.numbers == 50).astype(int))
            a.new_array('parent_site', np.arange(len(a)))
            a.info.update(source_id=sid, relaxed_DFT_total_eV=float(total))
            d = b_dist[np.ix_(occ,occ)][np.triu_indices(8,k=1)]
            counts = [int(np.isclose(d, value, atol=1e-4).sum()) for value in shell_dist]
            if len(counts) != 3 or sum(counts) != 28:
                raise ValueError('Invalid Sn-Sn shell counts')
            records.append(dict(source_id=sid, group='La16Zr8Sn8O56', stratum=counts[0],
                                energy_eV_cell=float(total), author_id=i, degeneracy=degeneracy,
                                first_shell_Sn_pairs=counts[0], second_shell_Sn_pairs=counts[1],
                                third_shell_Sn_pairs=counts[2], parent_Sn_sites=','.join(map(str,metals[occ]))))
            geometries[sid] = a
            raw_rows.append(dict(author_id=i, source_id=sid, reported_total_eV=total, castep_total_eV=scf[-1]))
    if len(geometries) != 97 or sum(r['degeneracy'] for r in records) != 12870:
        raise ValueError('Missing or duplicate symmetry classes')
    return records, geometries, raw_rows, payload


def partitions(records):
    return site_stratified_partitions(records, SPLIT_SEED, n_validation=12)


def write_bundle(path, records, atoms, raw, payload, folds):
    readme = f"""Equimolar B-site order in La16Zr8Sn8O56
Grau-Crespo et al. {PAPER}
Original authors' source tree {SOURCE}
The embedded ZIP preserves the original bytes of 97 POSCAR inputs, 97 CASTEP
outputs, parent, enumeration and DFT energy table, plus Git blob identifiers.
Eight Sn substitutions in the La2Zr2O7 parent form 97 symmetry-inequivalent
arrangements among C(16,8) microstates. Every class is included, weighted equally.
Structures in configurations.extxyz use equivalent standardized poses on the
authors' common starting cell. Each target is the relaxed total energy (eV)
corrected for finite basis set, checked against the CASTEP output for that ID.
Each Sn is marked as a substitution on a Zr site. Oxygen sites intrinsic to the
parent are identical in all configurations. Test folds balance the number of
nearest-neighbor Sn-Sn pairs before using any DFT labels.
"""
    write_arrangement_bundle(path,records,atoms,raw,payload,folds,SOURCE_FILE,DIGEST,readme)
