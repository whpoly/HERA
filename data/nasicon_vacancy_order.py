"""Na/vacancy orderings in a common Na8V8P12O48 supercell from Park et al."""
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import spglib
from ase import Atoms
from ase.io import read
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
from scipy.optimize import linear_sum_assignment

from ..arrangement_sources import SOURCES
from .hzo_vacancy_sites import site_stratified_partitions
from .ysz_arrangements import ROOT, write_arrangement_bundle


SOURCE_FILE = 'paper_NM23072487B_NVPsolidsolution-V3.zip'
ARCHIVE = ROOT.parent/'dataset'/'nasicon_park'/SOURCE_FILE
DIGEST = '1dcdd7c468c560fb5e2795f6f757d054b96b7695d21e904dda8ce28bc5352527'
PAPER = 'https://doi.org/10.1038/s41563-024-02023-7'
SOURCE = 'https://doi.org/10.5281/zenodo.13985508'
TITLE = 'NASICON 正极 Na₈V₈P₁₂O₄₈ 中的八个钠空位'
INTERPRETATION = ('在 Na₂V₂(PO₄)₃ 的同一个双倍超胞中，16 个可占据的 Na 位固定占据 8 个。'
                  '模型由作者给出的初始 Na/空位排布预测该排布弛豫后的 DFT 总能。69 个初始排布均对称不等价。'
                  '标签包含作者计算中伴随 Na 排布的 V 电子态和结构弛豫效应。')
SCHEMA = SOURCES['nasicon'].schema
RESULTS = SOURCES['nasicon'].results
BUNDLE = 'Na2V2_PO4_3_vacancy_order_DFT.zip'
SPLIT_SEED = 20261110
CELL = 'SCEL2_1_2_1_0_0_0'


def site_key(z, frac):
    return int(z), tuple(np.round(frac % 1,5) % 1)


def prepare(archive=ARCHIVE):
    payload = Path(archive).read_bytes()
    if hashlib.sha256(payload).hexdigest() != DIGEST:
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            payload = z.read(SOURCE_FILE)
    if hashlib.sha256(payload).hexdigest() != DIGEST:
        raise ValueError('NASICON archive differs from the published V3 release')
    frames = []
    matcher = StructureMatcher(ltol=.04,stol=.12,angle_tol=3.,primitive_cell=False,scale=False)
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        prim = json.loads(z.read(next(n for n in z.namelist() if n.endswith('/prim.json'))))
        for n in sorted(z.namelist()):
            if f'/{CELL}/' not in n or not n.endswith('/calctype.default/properties.calc.json'):
                continue
            d = json.loads(z.read(n))
            formula = dict(zip(d['atom_type'], d['atoms_per_type'],strict=True))
            if formula != {'Na':8,'V':8,'P':12,'O':48}:
                continue
            base = n.rsplit('calctype.default/',1)[0]
            initial = read(io.StringIO(z.read(base+'POS').decode()),format='vasp')
            endpoint = Structure(d['relaxed_lattice'],np.repeat(d['atom_type'],d['atoms_per_type']),d['relaxed_basis'])
            vasp = read(io.StringIO(z.read(base+'calctype.default/relaxed_structure.vasp').decode()),format='vasp')
            if not matcher.fit(endpoint, Structure(vasp.cell.array,vasp.numbers,vasp.get_scaled_positions())):
                raise ValueError(f'The independently supplied endpoint does not match its CASM record at {base}')
            fmax = float(np.linalg.norm(d['relaxed_forces'],axis=1).max())
            if (d['coord_mode'] != 'direct' or Counter(initial.get_chemical_symbols()) != formula
                    or len(endpoint) != 76 or not np.isfinite([fmax,d['relaxed_energy']]).all() or fmax > .0101):
                raise ValueError(f'Invalid initial structure, forces or DFT label at {base}')
            frames.append((initial, endpoint, float(d['relaxed_energy']), base.split('/')[-2], fmax))
    if len(frames) != 69:
        raise ValueError('Unexpected size of the fixed-x=2 cohort')
    if len(matcher.group_structures([f[1] for f in frames])) != len(frames):
        raise ValueError('Relaxed endpoint duplicates must be grouped before evaluating held-out structures')
    sites = sorted({site_key(z,f) for a,*_ in frames for z,f in zip(a.numbers,a.get_scaled_positions(),strict=True)})
    numbers = np.array([z for z,_ in sites])
    frac = np.array([f for _,f in sites])
    cell = frames[0][0].cell.array
    if (Counter(numbers.tolist()) != {11:16,23:8,15:12,8:48}
            or not all(np.allclose(cell, f[0].cell.array, atol=1e-8, rtol=0) for f in frames)):
        raise ValueError('The initial cells do not share one full parent lattice')
    parent = Atoms(numbers=numbers,scaled_positions=frac,cell=cell,pbc=True)
    na = np.flatnonzero(numbers == 11)
    by_site = {k:i for i,k in enumerate(sites)}
    primitive = np.array([b['coordinate'] for b in prim['basis'] if b['occupant_dof'] == ['Na','Va']])
    primitive_delta = (frac[na] @ cell @ np.linalg.inv(prim['lattice_vectors']))[:,None,:] - primitive[None,:,:]
    primitive_dist = np.linalg.norm(primitive_delta-np.rint(primitive_delta),axis=2)
    is_na1 = primitive_dist.argmin(axis=1)<2
    if len(primitive) != 8 or sum(is_na1) != 4 or primitive_dist.min(axis=1).max() > 1e-4:
        raise ValueError('Na-site assignments disagree with the eight-site primitive model')
    sym = spglib.get_symmetry((cell,frac,numbers),symprec=.003)
    permutations = []
    for rot,tr in zip(sym['rotations'],sym['translations'],strict=True):
        delta = (frac[na]@rot.T+tr)[:,None,:] - frac[na][None,:,:]
        distances = np.linalg.norm((delta-np.rint(delta))@cell,axis=2)
        left,right = linear_sum_assignment(distances)
        if distances[left,right].max() > .003 or not np.array_equal(is_na1, is_na1[right]):
            raise ValueError('Invalid parent-site symmetry')
        permutations.append(right)
    perms = np.unique(permutations,axis=0)
    distances = parent.get_all_distances(mic=True)[np.ix_(na,na)]
    nearest = distances[distances>0].min()
    records, geometries, raw = [], {}, []
    for initial, _, energy, author, fmax in frames:
        occ = [np.searchsorted(na, by_site[site_key(11,f)]) for f in initial.get_scaled_positions()[initial.numbers==11]]
        vacant = np.setdiff1d(np.arange(16),occ)
        canonical = np.unique(np.sort(perms[:,vacant],axis=1),axis=0)[0]
        sid = 'nvp-' + hashlib.sha256(bytes(canonical.tolist())).hexdigest()[:16]
        a = parent.copy()
        a.new_array('was_Z',a.numbers.copy())
        a.new_array('defect_site',np.zeros(len(a),dtype=int))
        a.new_array('parent_site',np.arange(len(a)))
        a.info.update(source_id=sid,vacancy_fractionals=frac[na[canonical]].copy(),
                      vacancy_was_Z=11,relaxed_DFT_total_eV=energy)
        del a[na[canonical]]
        n_na1 = int(4-sum(is_na1[canonical]))
        bonds = distances[np.ix_(canonical,canonical)][np.triu_indices(8,k=1)]
        records.append(dict(source_id=sid,group='Na8V8P12O48',stratum=n_na1,energy_eV_cell=energy,
                            author_config=author,occupied_Na1=n_na1,first_shell_vac_pairs=int((bonds<nearest+.02).sum()),
                            max_DFT_force_eV_A=fmax,vacant_parent_sites=','.join(map(str,na[canonical]))))
        geometries[sid] = a
        raw.append(dict(source_id=sid,casm_configuration=f'{CELL}/{author}',relaxed_DFT_energy_eV=energy,
                        max_DFT_force_eV_A=fmax))
    if len(geometries) != 69 or len(perms) != 8:
        raise ValueError('Unexpected initial occupation duplicates or cell symmetry')
    return records, geometries, raw, payload


def partitions(records):
    return site_stratified_partitions(records, SPLIT_SEED, n_validation=8)


def write_bundle(path, records, atoms, raw, payload, folds):
    readme = f"""Fixed sodium-vacancy order at Na2V2(PO4)3
Park et al. {PAPER}
Zenodo V3 {SOURCE}
The original author ZIP is embedded without modification. The input occupations
come from POS for {CELL}, choosing every Na8V8P12O48 row. There are 16
potential Na sites as verified against the published prim.json, 8 occupied and
8 vacant per cell. Frames contain the 76 physical atoms in standardized ideal
occupations. The 8 vacancy_fractionals coordinates locate the missing Na sites.
Coordinates and energies of relaxed endpoints are paired within each author's
properties.calc.json. Reported maximum DFT forces are below 0.01 eV/Angstrom.
All 69 ideal occupations are inequivalent under the parent-cell symmetry.
All endpoints are distinct at StructureMatcher tolerances ltol=0.04, stol=0.12,
angle_tol=3 degrees, scale=False and primitive_cell=False. The electronic spin
state can respond to each occupation in DFT. No electronic or relaxation data
are used as input features. Folds stratify by the number of occupied Na1 sites.
"""
    write_arrangement_bundle(path,records,atoms,raw,payload,folds,SOURCE_FILE,DIGEST,readme)
