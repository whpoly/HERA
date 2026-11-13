"""Fixed half-loading hydrogen interstitial arrangements in the Nb–H system."""
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

import numpy as np
from ase import Atoms
from ase.io import read
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure

from ..arrangement_sources import ROOT, SOURCES
from .hzo_vacancy_sites import site_stratified_partitions
from .ysz_arrangements import write_arrangement_bundle


SOURCE_FILE = 'DFTforCALPHAD.collection_calculation_pub.json'
ARCHIVE = ROOT.parent/'dataset'/'hydride_hannappel'/SOURCE_FILE
DIGEST = 'fb6dfcd52c40f5740872326fcccc215d4ce3900a95e86bd96293838ccb94de6a'
PAPER = 'https://doi.org/10.1038/s41524-025-01902-0'
SOURCE = 'https://github.com/PHannapp/PCT-Simulation-Toolkit/tree/e10c96245b6bd7dbe972d53103c805727c88f605'
TITLE = 'Nb₁₆H₈ 中八个间隙氢的排布'
INTERPRETATION = ('选取从 bcc Nb 母相出发、H/Nb 固定为 1/2 的全部 20 个独立弛豫构型。'
                  '每个超胞含 16 个 Nb 和 8 个间隙 H。输入为 VASP CONTCAR，标签为作者公布的 VASP '
                  '电子自由能 F（TOTEN），逐条核对最终 OSZICAR。')
INPUT_DESCRIPTION = ('Use relaxed CONTCAR coordinates, elements and the hydrogen interstitial mask. '
                     'Energetic, force and electronic properties do not enter the model inputs.')
SCHEMA = SOURCES['nb_h8'].schema
RESULTS = SOURCES['nb_h8'].results
BUNDLE = 'Nb16H8_hydrogen_arrangements_DFT.zip'
SPLIT_SEED = 20261110


def source_atoms(text):
    lines = text.splitlines()
    species = [token.split('/')[0].split('_')[0].capitalize() for token in lines[5].split()]
    if species != ['Nb','H'] or list(map(int,lines[6].split())) != [16,8]:
        raise ValueError('Expected explicit Nb16 and H8 entries in the VASP structure')
    lines[5] = 'Nb H'
    return read(io.StringIO('\n'.join(lines)+'\n'), format='vasp')


def prepare(archive=ARCHIVE):
    payload = Path(archive).read_bytes()
    if hashlib.sha256(payload).hexdigest() != DIGEST:
        with zipfile.ZipFile(io.BytesIO(payload)) as curated:
            payload = curated.read(SOURCE_FILE)
    if hashlib.sha256(payload).hexdigest() != DIGEST:
        raise ValueError('Metal-hydride calculation collection differs from the pinned author data')
    source = [r for r in json.loads(payload) if r.get('system_name') == 'BCC' and r.get('n_H') == 8
              and r.get('type') == 'endmember' and r['POSCAR'].splitlines()[5].split() == ['NB','H']]
    if (sorted(r['n_POSCAR'] for r in source) != list(range(20))
            or len({r['VASP']['INCAR'] for r in source}) != 1 or len({r['VASP']['KPOINTS'] for r in source}) != 1):
        raise ValueError('Expected the complete twenty-configuration single-settings BCC Nb16H8 cohort')
    records, geometries, raw, endpoints = [], {}, [], []
    reference = None
    for r in sorted(source,key=lambda obj:obj['n_POSCAR']):
        v = r['VASP']
        initial = source_atoms(r['POSCAR'])
        relaxed = source_atoms(v['CONTCAR'])
        f = relaxed.get_scaled_positions()
        if not np.isfinite(f).all() or Counter(relaxed.get_chemical_symbols()) != {'Nb':16,'H':8}:
            raise ValueError('Invalid final composition or geometry')
        fixed = sorted(tuple(x) for x in np.round(initial.get_scaled_positions()[initial.numbers==41],6))
        if reference is None:
            reference = initial.cell.array.copy(), fixed
        if fixed != reference[1] or not np.allclose(initial.cell.array,reference[0],atol=1e-7,rtol=0):
            raise ValueError('Initial Nb frameworks differ')
        free = re.findall(r'\bF=\s*([-+\d.Ee]+)',v['OSZICAR'])
        e0 = re.findall(r'\bE0=\s*([-+\d.Ee]+)',v['OSZICAR'])
        energy = float(v['energy'])
        forces = np.asarray(v['forces'],dtype=float)
        if forces.shape != (24,3) or not np.isfinite(forces).all():
            raise ValueError('Force array does not match the 24 real atoms')
        fmax = np.linalg.norm(forces,axis=1).max()
        if not free or not e0 or not np.isfinite(energy) or abs(float(free[-1])-energy)>5.1e-6 or fmax>.0501:
            raise ValueError('Published total does not agree with the relaxed VASP run')
        order = np.lexsort((f[:,2],f[:,1],f[:,0],relaxed.numbers))
        a = Atoms(numbers=relaxed.numbers[order],scaled_positions=f[order],cell=relaxed.cell,pbc=True)
        coords = np.column_stack((a.numbers,np.round(a.get_scaled_positions(),7)))
        key = 'nbh8-'+hashlib.sha256(np.round(a.cell.array,7).astype('<f8').tobytes()+
                                      coords.astype('<f8').tobytes()).hexdigest()[:16]
        is_h = a.numbers==1
        previous = a.numbers.copy()
        previous[is_h] = 0
        a.new_array('was_Z',previous)
        a.new_array('defect_site',is_h.astype(int))
        a.info.update(source_id=key,relaxed_DFT_total_eV=energy)
        distances = a.get_all_distances(mic=True)[np.ix_(is_h,is_h)][np.triu_indices(8,k=1)]
        close = int((distances<2.5).sum())
        records.append(dict(source_id=key,group='Nb16H8',stratum=close,energy_eV_cell=energy,
                            closest_HH_A=float(distances.min()),relaxed_volume_A3=float(a.get_volume()),
                            hydrogen_pairs_within_2p5_A=close,source_run=r['n_POSCAR'],max_force_eV_A=float(fmax),
                            electronic_E0_eV=float(e0[-1])))
        raw.append(dict(source_id=key,mongodb_id=r['_id']['$oid'],n_POSCAR=r['n_POSCAR'],
                        published_VASP_energy_eV=energy,final_OSZICAR_F_eV=float(free[-1]),
                        final_OSZICAR_E0_eV=float(e0[-1])))
        geometries[key] = a
        endpoints.append(Structure(a.cell.array,a.numbers,a.get_scaled_positions()))
    matcher = StructureMatcher(ltol=.03,stol=.11,angle_tol=2.,primitive_cell=False,scale=False)
    if len(geometries)!=20 or len(matcher.group_structures(endpoints))!=20:
        raise ValueError('Relaxed hydride structures must be unique before splitting')
    return records, geometries, raw, payload


def partitions(records):
    return site_stratified_partitions(records,SPLIT_SEED,n_validation=4)


def write_bundle(path, records, atoms, raw, payload, folds):
    readme = f"""Relaxed Nb16H8 arrangements at fixed H/Nb = 1/2
Hannappel et al. {PAPER}
Source collection at the pinned Git commit {SOURCE}
The entire original calculation collection is embedded byte-for-byte. Select all
20 endmember records with system_name BCC, Nb16 host and n_H=8 (n_POSCAR 0..19).
The published initial POSCARs have the same Nb framework. Each VASP CONTCAR gives
16 Nb and 8 real interstitial H atoms. The labels are the published VASP.energy
values (electronic free energy F or TOTEN in eV per cell); all agree with the
final OSZICAR F within 0.0000051 eV. The extrapolated E0 is recorded separately.
INCAR and KPOINTS are identical across this cohort and the maximum atomic forces
are below 0.05 eV/Angstrom. There are 20 inequivalent relaxed structures at
StructureMatcher ltol=.03, stol=.11, angle_tol=2, scale=False, primitive_cell=False.
The structures in configurations.extxyz are the relaxed coordinates. H is marked
as an interstitial (was_Z=0). No ghost is added. The source run index, energies,
forces, and electronic structure are never model inputs. Test folds balance
the number of H-H pairs within 2.5 Angstrom without inspecting the energy labels.
"""
    write_arrangement_bundle(path,records,atoms,raw,payload,folds,SOURCE_FILE,DIGEST,readme)
