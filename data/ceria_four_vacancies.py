"""Four surface/subsurface oxygen vacancies at fixed Ce64O124 in CeO2(111)."""
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

import numpy as np
import spglib
from ase import Atoms
from ase.geometry import find_mic
from scipy.spatial import cKDTree

from ..arrangement_sources import ROOT, SOURCES
from .hzo_vacancy_sites import site_stratified_partitions
from .ysz_arrangements import write_arrangement_bundle


PAPER = 'https://doi.org/10.1039/D4NR04591B'
SOURCE = 'https://doi.org/10.17172/NOMAD/2024.05.08-1'
RAW = 'https://nomad-lab.eu/backend/api/v1/uploads/YgBxN9FQSFmR5biHFhGy3Q/raw/data?compress=true'
SOURCE_FILE = 'CeO2_111_four_vacancy_VASP.zip'
ARCHIVE = ROOT.parent/'dataset'/'ceria_vacancy_zhang'/SOURCE_FILE
DIGEST = '6431eaa5026fa3c3552b355c16849bd5a4eccc885284306aaf241ce24093256a'
SCHEMA = SOURCES['ceria4'].schema
RESULTS = SOURCES['ceria4'].results
BUNDLE = 'Ce64O124_four_oxygen_vacancy_arrangements_DFT.zip'
SPLIT_SEED = 20261110
TITLE = 'CeO₂(111) 表面和次表面的四个氧空位'
INTERPRETATION = ('固定 Ce₆₄O₁₂₄，使用最高空位数的全部 49 次收敛 VASP 计算。按保持表面方向的母晶对称性，'
                  '共有 34 种四氧空位排布。同一排布存在多个电子局域态时，标签取作者实际采样中最低的弛豫 DFT 总能。'
                  '模型只使用理想母晶和四个位点。多数排布只采样了一个电子态，所取能量不保证是电子基态。')
INPUT_DESCRIPTION = ('Only ideal Ce and O sites and the four vacant O sites enter model inputs. The published '
                     'electronic occupation, local moments and DFT-relaxed geometry are excluded. Target: '
                     'lowest published converged DFT total for each ionic vacancy arrangement.')


def array(field):
    shape, dtype, values = field.get('array',field)['__ndarray__']
    return np.asarray(values,dtype=dtype).reshape(shape)


def wrapped(frac):
    return np.round(np.mod(frac,1.),10)%1.


def source_extract(original,destination=ARCHIVE):
    """Canonical byte snapshot of every four-vacancy calculation and the pristine reference."""
    files = {}
    with zipfile.ZipFile(original) as z:
        ids = sorted(int(n.split('/')[1]) for n in z.namelist() if re.fullmatch(r'train/\d+/geom.json',n))
        if ids!=list(range(1,222)):
            raise ValueError('Expected all 221 published CeO2(111) calculation IDs')
        chosen = defaultdict(list)
        for i in ids:
            name = f'train/{i}/geom.json'
            data = z.read(name)
            nums = array(json.loads(data)['1']['numbers'])
            n_vac = int((nums==0).sum())
            if n_vac not in (0,4):
                continue
            chosen[n_vac].append(i)
            files[name] = data
            for entry in [f'train/{i}/energy.dat',f'DFT/output/{i}/OUTCAR',f'DFT/output/{i}/CONTCAR']:
                files[entry] = z.read(entry)
        if len(chosen[4])!=49 or len(chosen[0])!=1:
            raise ValueError('Incomplete fixed-composition selection')
        for name in ['plat.json','DFT/input/4v/INCAR','DFT/input/4v/KPOINTS']:
            files[name] = z.read(name)
    manifest = dict(doi=SOURCE,public_archive=RAW,parent=chosen[0][0],four_vacancies=chosen[4],
                    file_sha256={k:hashlib.sha256(v).hexdigest() for k,v in sorted(files.items())})
    files['provenance.json'] = (json.dumps(manifest,indent=2)+'\n').encode()
    destination = Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(destination,'w') as out:
        for name,value in sorted(files.items()):
            info = zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0))
            info.create_system = 0
            info.external_attr = 32
            out.writestr(info,value)
    return hashlib.sha256(destination.read_bytes()).hexdigest()


def symmetry(cell,frac,types):
    # Keep the fixed lower face and the exposed upper face distinct.
    _, planes = np.unique(np.round(frac[:,2],3),return_inverse=True)
    ops = spglib.get_symmetry((cell,frac,types+1000*planes),symprec=.004)
    tree = cKDTree(frac,boxsize=1.)
    maps = []
    for rotation,shift in zip(ops['rotations'],ops['translations'],strict=True):
        distances,idx = tree.query(wrapped(frac@rotation.T+shift))
        if distances.max()>.0005 or len(set(idx))!=len(frac):
            raise ValueError('Could not map a slab symmetry back to the parent sites')
        maps.append(idx)
    transforms = np.unique(maps,axis=0)
    if len(transforms)!=96:
        raise ValueError('Expected 16 lateral translations and six surface point operations')
    return tree,transforms


def relaxed_vacancies(contcar,frac,parent_types):
    lines = contcar.splitlines()
    cell = np.array([line.split()[:3] for line in lines[2:5]],dtype=float)*float(lines[1])
    counts = [int(x) for x in lines[6].split()]
    idx = 8 if lines[7].strip().lower().startswith('s') else 7
    if lines[5].split()!=['O','Ce'] or counts!=[124,64] or not lines[idx].strip().lower().startswith('d'):
        raise ValueError('Unexpected relaxed composition or VASP coordinate convention')
    oxygen = np.array([s.split()[:3] for s in lines[idx+1:idx+1+124]],dtype=float)@cell
    sites = np.flatnonzero(parent_types==8)
    shifts = np.indices((3,3,3)).reshape(3,-1).T-1
    images = ((shifts[:,None,:]+frac[sites][None,:,:])@cell).reshape(-1,3)
    delta, nearest = cKDTree(images).query(oxygen)
    assigned = sites[nearest%len(sites)]
    if len(np.unique(assigned))!=124 or delta.max()>1.3:
        raise ValueError('Relaxation changed the occupied oxygen lattice')
    return np.setdiff1d(sites,assigned)


def prepare(archive=ARCHIVE):
    payload = Path(archive).read_bytes()
    if hashlib.sha256(payload).hexdigest()!=DIGEST:
        with zipfile.ZipFile(io.BytesIO(payload)) as bundle:
            payload = bundle.read(SOURCE_FILE)
    if hashlib.sha256(payload).hexdigest()!=DIGEST:
        raise ValueError('CeO2 four-vacancy author source snapshot differs from the pinned data')
    classes, raw = defaultdict(list), []
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        manifest = json.loads(z.read('provenance.json'))
        ref = json.loads(z.read(f"train/{manifest['parent']}/geom.json"))['1']
        cell, types = array(ref['cell']),array(ref['numbers'])
        frac = wrapped(array(ref['positions'])@np.linalg.inv(cell))
        if Counter(types)!={8:128,58:64} or len(manifest['four_vacancies'])!=49:
            raise ValueError('Unexpected pristine oxide slab or four-vacancy selection')
        tree,ops = symmetry(cell,frac,types)
        for i in manifest['four_vacancies']:
            reference = json.loads(z.read(f'train/{i}/geom.json'))['1']
            n = array(reference['numbers'])
            dist,idx = tree.query(wrapped(array(reference['positions'])@np.linalg.inv(cell)))
            if (Counter(n)!={8:124,0:4,58:56,57:8} or np.max(dist)>.001 or len(set(idx))!=192
                    or not np.array_equal(types[idx],np.where((n==8)|(n==0),8,58))):
                raise ValueError(f'Invalid vacancy / polaron occupation record {i}')
            occupation = np.empty(192,dtype=np.uint8)
            occupation[idx] = n
            vacancies = np.flatnonzero(occupation==0)
            final_vacancies = relaxed_vacancies(z.read(f'DFT/output/{i}/CONTCAR').decode(),frac,types)
            if not np.array_equal(vacancies,final_vacancies):
                raise ValueError(f'Vacancy sites changed in the relaxation of record {i}')
            outcar = z.read(f'DFT/output/{i}/OUTCAR').decode()
            signatures = [r'ISPIN\s*=\s*2',r'ENCUT\s*=\s*400\.0 eV',r'GGA\s*=\s*PE\s+GGA type',
                          r'LDAUU\s*=\s*0\.0\s+5\.0',r'NKPTS\s*=\s*1\s+k-points',
                          r'reached required accuracy',r'General timing and accounting']
            if any(re.search(k,outcar) is None for k in signatures):
                raise ValueError(f'DFT settings/completion mismatch for record {i}')
            e0 = float(re.findall(r'energy\(sigma->0\)\s*=\s*([-\d.]+)',outcar)[-1])
            free = float(re.findall(r'free\s+energy\s+TOTEN\s*=\s*([-\d.]+)',outcar)[-1])
            published = float(z.read(f'train/{i}/energy.dat'))
            if not np.isfinite([e0,free,published]).all() or abs(e0-free)>1e-6:
                raise ValueError(f'Invalid final DFT energy for record {i}')
            mask = np.asarray(occupation==0,dtype=np.uint8)
            orbit = min(k.tobytes() for k in mask[ops])
            source_id = 'ce4v-'+hashlib.sha256(orbit).hexdigest()[:16]
            classes[orbit].append((i,e0))
            raw.append(dict(author_id=i,source_id=source_id,OUTCAR_E0_eV=e0,OUTCAR_TOTEN_eV=free,
                            published_CE_energy_eV=published))
    if len(classes)!=34 or np.ptp([r['OUTCAR_E0_eV']-r['published_CE_energy_eV'] for r in raw])>.0101:
        raise ValueError('Vacancy orbits or independent DFT/CE label cross-check failed')
    records, atoms = [], {}
    oxygen_layers = np.unique(np.round(frac[types==8,2],3))
    for orbit, calculations in sorted(classes.items()):
        sid = 'ce4v-'+hashlib.sha256(orbit).hexdigest()[:16]
        vacant = np.frombuffer(orbit,dtype=np.uint8).astype(bool)
        vfrac = frac[vacant]
        depths = np.round(vfrac[:,2],3)
        if not np.isin(depths,oxygen_layers[-3:]).all():
            raise ValueError('Expected all vacancies in one of the top three oxygen layers')
        top = int((depths==oxygen_layers[-1]).sum())
        second = int((depths==oxygen_layers[-2]).sum())
        left,right = np.triu_indices(4,k=1)
        _,separations = find_mic((vfrac[right]-vfrac[left])@cell,cell,pbc=True)
        lowest = min(calculations,key=lambda pair:(pair[1],pair[0]))
        a = Atoms(numbers=types[~vacant],scaled_positions=frac[~vacant],cell=cell,pbc=True)
        a.new_array('was_Z',a.numbers.copy())
        a.new_array('defect_site',np.zeros(len(a),dtype=int))
        a.info.update(source_id=sid,vacancy_fractionals=vfrac,vacancy_was_Z=8,relaxed_DFT_total_eV=lowest[1])
        records.append(dict(source_id=sid,group='Ce64O124',stratum=f'{top},{second}',energy_eV_cell=lowest[1],
                            top_layer_vacancies=top,second_layer_vacancies=second,closest_O_vacancies_A=float(min(separations)),
                            source_ids=','.join(str(j) for j,_ in calculations),lowest_sampled_id=lowest[0],
                            electronic_samples=len(calculations),sampled_energy_span_eV=float(np.ptp([e for _,e in calculations]))))
        atoms[sid]=a
    return records,atoms,raw,payload


def partitions(records):
    return site_stratified_partitions(records,SPLIT_SEED,n_validation=5)


def write_bundle(path,records,atoms,raw,payload,folds):
    readme = f"""Ce64O124: four oxygen vacancies in a CeO2(111) slab
Zhang et al., {PAPER}. NOMAD author collection: {SOURCE}. Original download: {RAW}
All 49 four-vacancy calculations at the largest vacancy count in the 221-record
collection are included. The fixed 4x4 parent slab has 64 Ce and 128 oxygen sites.
Each physical DFT CONTCAR has 64 Ce and 124 O atoms. In the authors' ideal geom.json
files 0 denotes an oxygen vacancy and 57 is an electronic-occupation marker for a
localized electron on Ce, not a chemical La dopant. The arrangement models use
the four oxygen vacancies only; all real cerium atoms use Z=58. No electronic
locations, energy labels, relaxed coordinates, or site moments enter the graph.
The ideal slab symmetries preserve the upper face and each atomic layer. Under
16 translations and six surface point operations, 49 calculations reduce to 34
distinct ionic vacancy arrangements. The relaxed oxygen atoms still map one-to-one
to the occupied parent sites in all 49 DFT calculations. Use the lowest published
relaxed DFT E(sigma->0) for each vacancy motif. This is a minimum over the authors'
sampled electronic states. Each selected label can be traced to its original OUTCAR
and CONTCAR; all calculations finished VASP ionic relaxation, used spin-polarized
PBE+U (Ce U=5 eV), 400 eV, and the Gamma point. The CE training labels round to
two decimal places; at this fixed composition their differences agree with the DFT
totals to within 0.010 eV. Train on the unrounded DFT energies in eV per supercell.
The atomistic input is the ideal parent oxygen-occupancy pattern. Vacancies are
in the outer three oxygen layers. The split balances top and second layer vacancy
counts without energy labels. The deterministic embedded snapshot contains the
original inputs and outputs for all 49 cases and the pristine reference. Raw
files and their SHA-256 hashes are listed in its provenance.json.
"""
    write_arrangement_bundle(path,records,atoms,raw,payload,folds,SOURCE_FILE,DIGEST,readme)
