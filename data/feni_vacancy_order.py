"""Equiatomic-parent Fe–Ni vacancy jump endpoints and chemical antisite order."""
from collections import Counter
import hashlib
import io
from itertools import combinations, permutations, product
import json
from pathlib import Path, PurePosixPath
import re
import zipfile

import numpy as np
from ase import Atoms
from ase.units import Bohr, Hartree

from ..arrangement_sources import ROOT, SOURCES
from .ysz_arrangements import write_arrangement_bundle


SOURCE_FILE = 'FeNi_equiatomic_NEB_endpoints.zip'
ARCHIVE = ROOT.parent/'dataset'/'feni_vacancy_fisher'/SOURCE_FILE
ORIGINAL_FILE = 'Fe-Ni_DFT_Barriers.zip'
ORIGINAL_MD5 = '09cfb34e4efcc3237527dc03e7e24aa1'
DIGEST = 'f6e75bd33df4771ffcf38507430d6d82e2a8ac17293c794b47fe3144fa34cdc1'
PAPER = 'https://doi.org/10.1103/4q1c-97bp'
SOURCE = 'https://doi.org/10.5281/zenodo.17895774'
TITLE = '等原子比 FeNi 母相中的空位跃迁与化学有序度'
INTERPRETATION = ('32 条作者公开的蒙特卡洛等原子比 FeNi 母相空位跃迁各有两个固定胞弛豫端点。'
                  '按实际 Fe₁₅Ni₁₆ 和 Fe₁₆Ni₁₅ 分别计算能量差，每组成有 32 个不同占位。'
                  '相对于 L1₀ 有序参考，把占错亚晶格的 Fe/Ni 和一个空位标记为缺陷。'
                  '30 个对称不等价的完整母排布按母体分折，同一路径的两个端点始终一起留出。')
INPUT_DESCRIPTION = ('Use CASTEP relaxed endpoint species and positions plus antisite and vacancy labels '
                     'relative to the closest layered L10 variant of the published vacancy-free parent. '
                     'Parent order uses chemical occupations only. Energy, spin, forces, path coordinate '
                     'and annealing temperature are never passed to a graph.')
MATCHED_PATHS = ('Secondary endpoint energy-gap MAE over all 32 author-designated vacancy hops, using '
                 'the same fitted weights as the primary all-within-composition analysis. Report PRO minus REA. '
                 'Neither NEB barriers nor interior images enter fitting or selection.')
SCHEMA = SOURCES['feni'].schema
RESULTS = SOURCES['feni'].results
BUNDLE = 'FeNi_vacancy_antisite_endpoints_DFT.zip'
SPLIT_SEED = 20261110
PARENT_ROOT = 'Fe-Ni_DFT_Barriers/01_structure_generation/04_monte_carlo/fe50ni50/032_atom'
PATH_ROOT = 'Fe-Ni_DFT_Barriers/03_barrier_calculations/fe50ni50'
FCC = np.array([p for p in product(range(4), repeat=3) if sum(p) % 2 == 0])
SITE_INDEX = np.full((4,4,4),-1)
SITE_INDEX[tuple(FCC.T)] = np.arange(32)
OPERATIONS = np.array([SITE_INDEX[tuple(((FCC[:,axis]*sign+shift)%4).T)]
                       for axis in permutations(range(3)) for sign in product((-1,1),repeat=3) for shift in FCC])
ELEMENTS = {'Fe': 26, 'Ni': 28}


def source_extract(original, destination=ARCHIVE):
    """Keep byte-identical endpoint blocks and their inputs from the MD5-pinned Zenodo ZIP."""
    with Path(original).open('rb') as handle:
        checksum = hashlib.md5()
        for chunk in iter(lambda:handle.read(1024*1024), b''):
            checksum.update(chunk)
    if checksum.hexdigest() != ORIGINAL_MD5:
        raise ValueError('Fe-Ni Zenodo V2 archive checksum differs')
    output, provenance = {}, []
    with zipfile.ZipFile(original) as z:
        files = sorted(p for p in z.namelist() if p.startswith(PATH_ROOT+'/') and p.endswith('/neb.ts'))
        if len(files) != 32:
            raise ValueError('Unexpected number of equiatomic-parent Monte Carlo NEB records')
        for file in files:
            folder = PurePosixPath(file).parent
            temp = folder.parent.parent.name
            sample = folder.name.removeprefix('barrier')
            parent = f'{PARENT_ROOT}/{temp}/grids/{sample}.0.xyz'
            key = f'{temp}/{sample}'
            original_ts = z.read(file)
            first_interior = re.search(rb'(?m)^[ \t]*TST[ \t]+\d',original_ts)
            if first_interior is None:
                raise ValueError(f'Missing interior-image boundary in {file}')
            output[key+'/endpoints.ts'] = original_ts[:first_interior.start()]
            for member in ('neb.cell','neb_farm001.castep'):
                output[key+'/'+member] = z.read(str(folder/member))
            if str(folder/'neb.param') in z.namelist():
                output[key+'/neb.param'] = z.read(str(folder/'neb.param'))
            output[key+'/parent.xyz'] = z.read(parent)
            provenance.append(dict(key=key, path=file, parent=parent, ts_sha256=hashlib.sha256(original_ts).hexdigest(),
                                   prefix_bytes=first_interior.start()))
    output['provenance.json'] = (json.dumps(dict(original=ORIGINAL_FILE, md5=ORIGINAL_MD5, files=provenance),
                                           indent=2)+'\n').encode()
    destination = Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(destination,'w') as z:
        for name, data in sorted(output.items()):
            info = zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0))
            info.create_system = 0
            info.external_attr = 32
            z.writestr(info,data)
    return hashlib.sha256(destination.read_bytes()).hexdigest()


def canonical(occupation):
    return min(arr.tobytes() for arr in np.asarray(occupation,dtype=np.uint8)[OPERATIONS])


def ordered_reference(parent):
    # L10 has three possible axes and two layer origins. Break equal-order ties by
    # the parent+reference occupation under all 1,536 symmetries of this FCC supercell.
    candidates = [np.where(FCC[:,axis]%2 == phase,26,28) for axis in range(3) for phase in (0,1)]
    keys = [(np.count_nonzero(ref != parent), canonical(2*(parent==28).astype(np.uint8)+(ref==28)))
            for ref in candidates]
    best = min(keys)
    if keys.count(best) != 1:
        raise ValueError('Ambiguous ordered reference after symmetry reduction')
    return candidates[keys.index(best)], best[0]


def site_indices(positions, cell):
    frac = positions @ np.linalg.inv(cell)
    integer = np.rint(4*frac)
    indices = SITE_INDEX[tuple((integer.astype(int)%4).T)]
    if np.max(abs(4*frac-integer))>.25 or (indices<0).any() or len(set(indices))!=len(indices):
        raise ValueError('Could not uniquely assign atoms to FCC sites')
    return indices


def frame(text, cell_input, phase, block_name):
    match = re.search(r'(?ms)^\s*'+phase+r'\s+1\s+[^\n]+\n(.+?)(?=^\s*PRO\s+\d|\Z)',text)
    if match is None:
        raise ValueError(f'Missing {phase} endpoint')
    fields = {}
    for line in match[1].splitlines():
        if '<--' in line:
            values, tag = line.split('<--')
            fields.setdefault(tag.strip(),[]).append(values.split())
    energy = np.array(fields['E'],dtype=float).ravel()*Hartree
    cell = np.array(fields['h'],dtype=float)*Bohr
    symbols = [line[0] for line in fields['R']]
    numbers = np.array([ELEMENTS[s] for s in symbols])
    pos = np.array([line[2:] for line in fields['R']],dtype=float)*Bohr
    forces = np.array([line[2:] for line in fields['F']],dtype=float)*Hartree/Bohr
    if (len(energy)!=2 or not np.isfinite(energy).all() or np.ptp(energy)>1e-6 or len(numbers)!=31
            or forces.shape!=(31,3) or not np.isfinite(forces).all() or not np.allclose(cell,7.14*np.eye(3),atol=1e-5,rtol=0)):
        raise ValueError('Inconsistent fixed-cell endpoint geometry or energy')
    indices = site_indices(pos, cell)
    occupation = np.zeros(32,dtype=np.uint8)
    occupation[indices] = numbers
    hole = np.flatnonzero(occupation==0).item()
    # The independent CASTEP input specifies these same endpoints in Angstrom.
    content = re.search(r'(?im)^\s*%block\s+'+block_name+r'\s*\n([\s\S]*?)%endblock',cell_input)[1]
    rows = [line.split() for line in content.splitlines() if line.split() and line.split()[0] in ELEMENTS]
    coords = np.array([r[1:4] for r in rows],dtype=float)
    check_idx = site_indices(coords,cell)
    change = (coords[np.argsort(check_idx)]-pos[np.argsort(indices)]) @ np.linalg.inv(cell)
    if (len(rows)!=31 or not np.array_equal(occupation[check_idx],[ELEMENTS[r[0]] for r in rows])
            or np.max(abs((change-np.rint(change))@cell))>2e-5):
        raise ValueError('Stored endpoint does not match independent neb.cell coordinates')
    return Atoms(numbers=numbers,positions=pos,cell=cell,pbc=True), occupation, indices, hole, float(energy[0]), float(np.linalg.norm(forces,axis=1).max())


def prepare(archive=ARCHIVE):
    payload = Path(archive).read_bytes()
    if hashlib.sha256(payload).hexdigest() != DIGEST:
        with zipfile.ZipFile(io.BytesIO(payload)) as curated:
            payload = curated.read(SOURCE_FILE)
    if hashlib.sha256(payload).hexdigest()!=DIGEST:
        raise ValueError('Published FeNi NEB endpoint extract differs from the pinned archive')
    records, geometries, raw = [], {}, []
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        provenance = json.loads(z.read('provenance.json'))
        if provenance['md5']!=ORIGINAL_MD5 or len(provenance['files'])!=32:
            raise ValueError('Invalid fixed-composition source manifest')
        for source in provenance['files']:
            key = source['key']
            lines = z.read(key+'/parent.xyz').decode().splitlines()
            rows = [line.split() for line in lines[2:] if line.strip()]
            positions = np.array([r[1:4] for r in rows],dtype=float)
            original = np.zeros(32,dtype=np.uint8)
            original[site_indices(positions,7.14*np.eye(3))] = [ELEMENTS[r[0]] for r in rows]
            if Counter(original)!={26:16,28:16}:
                raise ValueError('Unexpected parent alloy composition')
            reference, parent_antisites = ordered_reference(original)
            parent_id = hashlib.sha256(canonical(original)).hexdigest()[:16]
            ts = z.read(key+'/endpoints.ts').decode()
            cell_input = z.read(key+'/neb.cell').decode()
            calculation = z.read(key+'/neb_farm001.castep').decode()
            settings = [r'using functional\s*:\s*Perdew Burke Ernzerhof',r'treating system as spin-polarized',
                        r'plane wave basis set cut-off\s*:\s*800\.0000\s+eV',
                        r'smearing width\s*:\s*0\.5000E-01\s+eV',r'MP grid size for SCF calculation is\s+8\s+8\s+8']
            if any(re.search(setting,calculation) is None for setting in settings):
                raise ValueError(f'Inconsistent electronic structure settings for {key}')
            end_occupations = []
            for phase,block in [('REA','positions_abs'),('PRO','positions_abs_product')]:
                a, occ, sites, hole, energy, force = frame(ts,cell_input,phase,block)
                end_occupations.append(occ)
                if phase=='REA' and np.any(original[sites]!=a.numbers):
                    raise ValueError('Reactant occupations differ from the published intact MC host')
                previous = reference[sites]
                a.new_array('was_Z',previous)
                a.new_array('defect_site',(previous!=a.numbers).astype(int))
                stable = np.argsort(sites)
                a = a[stable]
                sid = 'feni-'+hashlib.sha256(canonical(occ)).hexdigest()[:16]
                a.info.update(source_id=sid,vacancy_fractional=FCC[hole]/4,
                              vacancy_was_Z=int(reference[hole]),relaxed_DFT_total_eV=energy)
                delta = (FCC-FCC[hole]+2)%4-2
                shell = (delta**2).sum(axis=1)==2
                all_delta = (FCC[:,None]-FCC[None,:]+2)%4-2
                near = (all_delta**2).sum(axis=2)==2
                fe = int((occ==26).sum())
                group = f'Fe{fe}Ni{31-fe}'
                records.append(dict(source_id=sid,group=group,parent_id=parent_id,path_id=key,endpoint=phase,
                                    energy_eV_cell=energy,parent_antisites=int(parent_antisites),
                                    n_antisites=int(a.arrays['defect_site'].sum()),vacancy_first_shell_Fe=int((occ[shell]==26).sum()),
                                    Fe_Ni_nearest_pairs=int(near[np.ix_(occ==26,occ==28)].sum()),max_force_eV_A=force))
                raw.append(dict(source_id=sid,parent_id=parent_id,path=source['path'],endpoint=phase,
                                DFT_endpoint_energy_eV=energy,published_parent=source['parent']))
                if sid in geometries:
                    raise ValueError('Symmetry-equivalent vacancy endpoints must be grouped before splitting')
                geometries[sid]=a
            start, end = end_occupations
            changed = np.flatnonzero(start!=end)
            if (len(changed)!=2 or set(start[changed])!=set(end[changed]) or 0 not in start[changed]
                    or sum((((FCC[changed[0]]-FCC[changed[1]]+2)%4)-2)**2)!=2):
                raise ValueError('Endpoint occupations must differ by exactly one nearest-neighbour vacancy jump')
    if (Counter(r['group'] for r in records)!={'Fe15Ni16':32,'Fe16Ni15':32}
            or len({r['parent_id'] for r in records})!=30):
        raise ValueError('Unexpected coverage of vacancy endpoints or parent orbits')
    return records,geometries,raw,payload


def partitions(records):
    from sklearn.model_selection import StratifiedGroupKFold
    rows = sorted(records,key=lambda r:r['source_id'])
    ids = np.array([r['source_id'] for r in rows])
    groups = np.array([r['group'] for r in rows])
    parents = np.array([r['parent_id'] for r in rows])
    splitter = StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=SPLIT_SEED)
    folds = []
    for i,(rest,test) in enumerate(splitter.split(ids,groups,parents)):
        inner = StratifiedGroupKFold(n_splits=4,shuffle=True,random_state=SPLIT_SEED+1+i)
        tr, va = next(inner.split(ids[rest],groups[rest],parents[rest]))
        subsets = [rest[tr],rest[va],test]
        for ix,part in enumerate(subsets):
            if (any((groups[part]==g).sum()<2 for g in set(groups))
                    or any(set(parents[part])&set(parents[other]) for other in subsets[:ix])):
                raise ValueError('Partition separates a common parent or lacks same-formula pairs')
        folds.append({key:sorted(ids[part].tolist()) for key,part in zip(('train','val','test'),subsets,strict=True)})
    return folds


def inventory_diagnostic(records,atoms,folds):
    """Geometry-only comparison at fixed vacancy sublattice and numbers of each antisite."""
    composition = {r['source_id']:r['group'] for r in records}
    inventory = {}
    for sid,a in atoms.items():
        previous = a.arrays['was_Z']
        inventory[sid] = [int(((a.numbers==26)&(previous==28)).sum()),
                          int(((a.numbers==28)&(previous==26)).sum()),int(a.info['vacancy_was_Z'])]
    pairs = [dict(fold=i,group=composition[a],id_a=a,id_b=b)
             for i,f in enumerate(folds) for a,b in combinations(f['test'],2)
             if composition[a]==composition[b] and inventory[a]==inventory[b]]
    if len(pairs)!=31 or len({sid for p in pairs for sid in (p['id_a'],p['id_b'])})!=40:
        raise ValueError('Unexpected number of fixed-inventory comparisons')
    return dict(definition='Same actual composition, Fe-on-Ni count, Ni-on-Fe count and vacant L10 sublattice. '
                           'Use the original all-pair validation and fixed models. This is a secondary analysis.',
                inventory_fields=['Fe_on_Ni','Ni_on_Fe','vacancy_reference_atomic_number'],
                source_inventories=inventory,pairs=pairs)


def write_bundle(path,records,atoms,raw,payload,folds):
    readme = f"""DFT endpoint energies for vacancy jumps in Fe-Ni from equiatomic parent alloys
Fisher et al., {PAPER}. Authors' version 2: {SOURCE}.
All 32 available neb.ts paths under 03_barrier_calculations/fe50ni50 are included.
There are eight paths from each of 400, 600, 1200 and 3000 K chemical sampling.
Those temperatures identify sampled chemical occupations; the DFT energies are
not thermal averages. The source 300 K subdirectory contains no neb.ts.
The electronic-structure results in fields REA and PRO give 64 fixed-cell CASTEP
energies in Hartree. Convert using ASE Hartree and Bohr. All cells are 7.14
Angstrom cubic, all structures have 31 real atoms and one FCC vacancy. Initial
and final coordinates match positions_abs and positions_abs_product in neb.cell.
The maximum endpoint atomic force is below 0.009 eV/Angstrom. Interior NEB
images and migration barriers are excluded. Each path has exactly one nearest
neighbour interchange. Member {SOURCE_FILE} includes exact byte prefixes from
neb.ts, original neb.cell, neb_farm001.castep, available neb.param and
the published intact MC parent xyz. One path lacks a per-path neb.param.
All 32 calculation outputs verify spin-polarized PBE, 800 eV, 0.05 eV
smearing and an 8x8x8 k mesh. The extract is distributed
with original file hashes and paths in provenance.json. Regenerate the extract
with source_extract() and the official original archive, MD5 {ORIGINAL_MD5}.

For each intact 32-site Monte Carlo parent select its closest layered L10 FeNi
decoration from all six orientations and layer origins. Ties use the joint
parent+reference occupation under FCC symmetry and resolve uniquely. Both ends
use that same reference; an antisite is an atom on the opposite Fe/Ni sublattice.
The vacancy carries the atomic number of its reference site. At high disorder
this is a chosen chemical-order reference, not an assertion of dilute defects.
The 64 occupied configurations are all distinct under the 1536 cubic FCC
rotations/reflections and translations; the 32 source paths have 30 distinct
pristine parent occupations. Each parent is assigned to exactly one test fold
and is never split across train, validation and test. Primary comparisons are
within Fe15Ni16 or within Fe16Ni15 and within a held-out fold. A separate
secondary diagnostic uses only the 32 author-designated reactant/product pairs.
"""
    write_arrangement_bundle(path,records,atoms,raw,payload,folds,SOURCE_FILE,DIGEST,readme)
