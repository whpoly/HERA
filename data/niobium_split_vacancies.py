"""Relaxed, fixed-charge Nb split-vacancy complexes from Kavanagh's PBEsol dataset."""
from collections import Counter
import hashlib
import io
from pathlib import Path
import zipfile

import numpy as np
from ase import Atoms
from ase.geometry import find_mic
from ase.io import read
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure

from ..arrangement_sources import SOURCES
from .hzo_vacancy_sites import site_stratified_partitions
from .ysz_arrangements import ROOT, write_arrangement_bundle


SOURCE_FILE = 'split_vacancy.zip'
ARCHIVE = ROOT.parent/'dataset'/'kavanagh_split_vacancy'/SOURCE_FILE
DIGEST = '6c105d36132adc009bfe6d9c6c2c863244cf5fa9ef926531de8f3c586bf5bf61'
SOURCE = 'https://s3.echo.stfc.ac.uk/ml-peg-data/inputs/defects/split_vacancy/split_vacancy.zip'
PAPER = 'https://doi.org/10.1088/2515-7655/ade916'
ORIGINAL = 'https://doi.org/10.5281/zenodo.15838864'
TITLE = 'TiNb₃O₆ 中的 Nb 分裂空位'
INTERPRETATION = ('固定 Ti₂₄Nb₇₁O₁₄₄、超胞电荷 −3 和 PBEsol 泛函，比较两个 Nb 空位与一个 Nb 间隙原子组成的复合缺陷。'
                  '输入为作者的 DFT 弛豫坐标。38 条标为稳定分裂空位的原始记录中，3 条还含额外的氧空位。'
                  '其余 35 条对应 21 个不同弛豫终态。同一终态重复计算的能量取均值。')
INPUT_DESCRIPTION = ('Use DFT-relaxed positions, species, and geometric reference-site labels. The DFT energy, '
                     'screening key, oxidation states and charge are never supplied as input features.')
SCHEMA = SOURCES['nb_split'].schema
RESULTS = SOURCES['nb_split'].results
BUNDLE = 'TiNb3O6_relaxed_Nb_split_vacancies_DFT.zip'
SPLIT_SEED = 20261110
PREFIX = 'split_vacancy/pbesol/TiNb3O6-mp-29699/'


def site_distances(frac, reference, cell):
    # The shortest few distances assign the site in this large hexagonal cell.
    # Include adjacent periodic images for the skew in the first two vectors.
    delta = frac[:,None,:] - reference[None,:,:]
    delta -= np.rint(delta)
    distances = np.linalg.norm(delta@cell,axis=2)
    for dx,dy in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,-1)]:
        distances = np.minimum(distances, np.linalg.norm((delta+[dx,dy,0])@cell,axis=2))
    return distances


def geometry_id(atoms):
    frac = np.round(atoms.get_scaled_positions(),7) % 1
    order = np.lexsort((frac[:,2],frac[:,1],frac[:,0],atoms.numbers))
    return hashlib.sha256(np.column_stack((atoms.numbers[order],frac[order])).astype('<f8').tobytes()).hexdigest()


def prepare(archive=ARCHIVE):
    payload = Path(archive).read_bytes()
    if hashlib.sha256(payload).hexdigest() != DIGEST:
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            payload = z.read(SOURCE_FILE)
    if hashlib.sha256(payload).hexdigest() != DIGEST:
        raise ValueError('The split-vacancy benchmark archive differs from the pinned version')
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        def xyz(path,index=':'):
            return read(io.StringIO(z.read(PREFIX+path).decode()),index,format='extxyz')
        parent = xyz('supercell.xyz',-1)
        starts = xyz('Nb/split_vacancy_initial.xyz')
        finals = xyz('Nb/split_vacancy_ref.xyz')
    if len(finals) != 38 or Counter(parent.get_chemical_symbols()) != {'Ti':24,'Nb':72,'O':144}:
        raise ValueError('Unexpected host or number of split-vacancy candidates')
    pf = parent.get_scaled_positions()
    host_nb = np.flatnonzero(parent.numbers==41)
    accepted, tags, raw = [], {}, []
    for i,(a,b) in enumerate(zip(starts,finals,strict=True)):
        if (a.info['key'] != b.info['key'] or a.info['ref_energy'] != b.info['ref_energy']
                or a.info['ref_total_charge'] != -3 or b.info['ref_total_charge'] != -3
                or not np.isfinite(b.info['ref_energy']) or Counter(b.get_chemical_symbols()) != {'Ti':24,'Nb':71,'O':144}
                or not np.allclose(b.cell.array,parent.cell.array,atol=1e-4,rtol=0)):
            raise ValueError(f'Inconsistent relaxed reference {i}')
        frac = b.get_scaled_positions()
        cell = b.cell.array
        nb = np.flatnonzero(b.numbers==41)
        distances = site_distances(frac[nb],pf[host_nb],cell)
        closest = distances.min(axis=1)
        mobile = np.flatnonzero(closest>1)
        if len(mobile) != 1 or (closest[np.arange(len(nb))!=mobile[0]] > .4).any():
            raise ValueError(f'Cannot uniquely locate the displaced Nb in {i}')
        lattice_nb = np.delete(distances.argmin(axis=1), mobile)
        empty = np.setdiff1d(np.arange(len(host_nb)),lattice_nb)
        if len(empty) != 2:
            raise ValueError('Incorrect number of empty Nb sites')
        ti = site_distances(frac[b.numbers==22],pf[parent.numbers==22],cell).argmin(axis=1)
        if len(np.unique(ti)) != 24:
            raise ValueError('Additional Ti defect present')
        oxygen = site_distances(frac[b.numbers==8],pf[parent.numbers==8],cell).argmin(axis=1)
        extra_oxygen_vacancies = 144 - len(np.unique(oxygen))
        raw.append(dict(source_frame=i,ref_energy_eV=float(b.info['ref_energy']),total_charge=-3,
                        extra_oxygen_vacancies=extra_oxygen_vacancies,source_id=''))
        if extra_oxygen_vacancies:
            continue
        accepted.append(Structure(cell,b.numbers,frac,properties={'frame':i}))
        tags[i] = (int(nb[mobile[0]]),host_nb[empty])
    matcher = StructureMatcher(ltol=.02,stol=.12,angle_tol=2.,primitive_cell=False,scale=False)
    groups = matcher.group_structures(accepted)
    if len(accepted) != 35 or len(groups) != 21:
        raise ValueError('Unexpected number of pure Nb triplets or distinct relaxed minima')
    records, geometries = [], {}
    for group in groups:
        members = [s.properties['frame'] for s in group]
        index = min(members,key=lambda i:(geometry_id(finals[i]),i))
        ref = finals[index]
        key = 'nbsv-'+geometry_id(ref)[:16]
        interstitial, vacant = tags[index]
        energies = [raw[i]['ref_energy_eV'] for i in members]
        if np.ptp(energies) > .0011:
            raise ValueError('Supposedly equivalent DFT minima disagree in energy')
        target = float(np.mean(energies))
        frac = ref.get_scaled_positions()
        cell = ref.cell.array
        vfrac = pf[vacant]
        _, dvv = find_mic((vfrac[1]-vfrac[0])@cell,cell,pbc=True)
        _, div = find_mic((vfrac-frac[interstitial])@cell,cell,pbc=True)
        # Strip the source energy, oxidation states and electrostatic screening key.
        a = Atoms(numbers=ref.numbers,scaled_positions=frac,cell=cell,pbc=True)
        previous = a.numbers.copy()
        previous[interstitial] = 0
        a.new_array('was_Z',previous)
        a.new_array('defect_site',(previous==0).astype(int))
        order = np.lexsort((frac[:,2],frac[:,1],frac[:,0],a.numbers))
        a = a[order]
        a.info.update(source_id=key,relaxed_DFT_total_eV=target,vacancy_fractionals=vfrac,vacancy_was_Z=41)
        for i in members:
            raw[i]['source_id'] = key
        records.append(dict(source_id=key,group='Ti24Nb71O144_q-3',stratum=int(np.floor(dvv)),
                            energy_eV_cell=target,vacancy_distance_A=float(dvv),
                            mean_interstitial_vacancy_distance_A=float(np.mean(div)),
                            reference_frames=','.join(map(str,sorted(members))),label_spread_eV=float(np.ptp(energies))))
        geometries[key] = a
    if len(geometries) != len(records):
        raise ValueError('Nonunique relaxed configuration IDs')
    return records, geometries, raw, payload


def partitions(records):
    return site_stratified_partitions(records, SPLIT_SEED, n_validation=4)


def write_bundle(path, records, atoms, raw, payload, folds):
    readme = f"""DFT-relaxed Nb split vacancies in TiNb3O6
Kavanagh: {PAPER}
Authors' source data: {ORIGINAL}
ML-PEG benchmark: https://ddmms.github.io/ml-peg/user_guide/benchmarks/defects.html#split-vacancy
Byte-pinned source: {SOURCE}
Choose the single material/cation group with the largest number of stable
split-vacancy records in the PBEsol reference set, TiNb3O6-mp-29699/Nb.
All 38 relaxed frames have 239 real atoms and total charge -3. Assign each
ordinary Nb to its nearest parent Nb site. One atom in each record is at least
1.67 Angstrom from the nearest Nb site, all other Nb atoms are within 0.28
Angstrom. This gives two empty Nb sites and one real interstitial Nb. The
three reference frames where an additional oxygen site is empty belong to a
different defect content. The remaining 35 frames cover 21 relaxed minima
(StructureMatcher ltol=.02, stol=.12, angle_tol=2, no scaling or reduction).
Use one geometry representative selected by coordinate hash and the mean of
its class's DFT energies. The largest repeated-label spread is below 1.04 meV.
The two vacancy_fractionals coordinates refer to the relaxed cell. was_Z=0
marks a previously empty position, now occupied by the interstitial Nb.
Structure, species, and defect labels alone enter the model. The atomic
coordinates in configurations.extxyz are DFT-relaxed. Reported energies are
per-supercell PBEsol totals, not absolute charged-defect formation energies.
All fitted models use the same fixed charge, cell contents and functional.
"""
    write_arrangement_bundle(path,records,atoms,raw,payload,folds,SOURCE_FILE,DIGEST,readme)
