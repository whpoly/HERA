"""Published PBE labels and reference-site mapping for the BC/SV complexes in Si and Ge."""
from collections import Counter
from functools import lru_cache
import hashlib
import io
from itertools import product
import json
from pathlib import Path
import re
import tempfile
import urllib.request
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT.parent / 'dataset' / 'sige_defects'
SCHEMA = 'sige_bc_sv_pbe_v1'
LABELS_SHA256 = 'f5196cf34d9abe9d61fd84fe1567d82fb8ad5f5c4b3f2699948e6abb4ce05843'
FILES = {
    'cm6b03368_si_001.zip': ('https://ndownloader.figshare.com/files/7264205', '3622ee91e81f3ee1c700127d0b0b6355'),
    'cm6b03368_si_002.pdf': ('https://ndownloader.figshare.com/files/7264202', '5f8c16aabb5da7de8663512096e36e48'),
}


def ensure_sources(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name,(url,digest) in FILES.items():
        path = directory / name
        if path.exists():
            if hashlib.md5(path.read_bytes()).hexdigest() != digest:
                raise ValueError(f'Unrecognized source {path}')
            continue
        with urllib.request.urlopen(url, timeout=90) as resp:
            data = resp.read()
        if hashlib.md5(data).hexdigest()!=digest:
            raise ValueError(f'Unexpected source digest from {url}')
        with tempfile.NamedTemporaryFile(dir=directory, delete=False) as handle:
            handle.write(data)
            temporary=Path(handle.name)
        temporary.replace(path)


def extract_labels(directory):
    from pypdf import PdfReader
    reader = PdfReader(str(Path(directory)/'cm6b03368_si_002.pdf'))
    result = []
    for period, page in enumerate((6,8,10,12,14,16),1):
        lines=[line.strip().replace('−','-') for line in reader.pages[page-1].extract_text().splitlines()]
        for host,start,end in [('Si','(a) Silicon PBE','(b) Germanium PBE'),
                               ('Ge','(b) Germanium PBE','(c) Silicon HSE06')]:
            table=lines[lines.index(start)+1:lines.index(end)]
            elements=table[0].split()
            if not all(re.fullmatch('[A-Z][a-z]?',s) for s in elements):raise ValueError((page,table[0]))
            for kind in ('BC','SV'):
                matches=[line for line in table if line.startswith(kind+' ')]
                if len(matches)!=1:raise ValueError((page,host,kind))
                values=[float(v) for v in matches[0].split()[1:]]
                if len(values)!=len(elements):raise ValueError((page,host,kind,len(values),len(elements)))
                for element,value in zip(elements,values):
                    result.append(dict(host=host,dopant=element,kind=kind,period=period,
                                       formation_enthalpy_ev=value,pdf_page=page))
    assert len(result)==292 and len({(r['host'],r['dopant'],r['kind']) for r in result})==292
    (Path(directory)/'pbe_labels.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')
    return result


@lru_cache(maxsize=1)
def diamond_grids():
    """Diamond translations modulo the common 2 x 2 x 2 conventional supercell."""
    from ase.build import bulk
    template=bulk('Si','diamond',a=1.,cubic=True).repeat((2,2,2))
    lattice=np.rint(template.get_scaled_positions()*8).astype(int)%8
    seen=set()
    candidates=[]
    for shift in product(range(8),repeat=3):
        sites=(lattice+shift)%8
        key=tuple(sorted(map(tuple,sites.tolist())))
        if key not in seen:
            seen.add(key)
            candidates.append(np.asarray(key,dtype=float)/8.)
    return candidates


def map_complex(atoms, host, dopant):
    if host==dopant:raise ValueError('host and nominal dopant are the same element')
    if Counter(atoms.get_chemical_symbols())!={host:62,dopant:1}:
        raise ValueError(f'unexpected composition {atoms.get_chemical_formula()}')
    if not np.allclose(atoms.cell.angles(),90.) or not np.allclose(atoms.cell.lengths(),atoms.cell.lengths()[0]):
        raise ValueError('expected a cubic conventional supercell')
    symbols=np.array(atoms.get_chemical_symbols())
    scaled=atoms.get_scaled_positions(wrap=True)
    f=scaled[symbols==host]
    cell=atoms.cell.array
    primitive_shift=np.angle(np.mean(np.exp(2j*np.pi*f*8),axis=0))/(16*np.pi)
    best=None
    for grid in diamond_grids():
        origin=primitive_shift
        for _ in range(3):
            delta=f[:,None,:]-grid[None,:,:]-origin
            delta-=np.rint(delta)
            distance=np.linalg.norm(delta@cell,axis=-1)
            assigned=distance.argmin(axis=1)
            origin=origin+np.median(delta[np.arange(len(f)),assigned],axis=0)
        delta=f[:,None,:]-grid[None,:,:]-origin
        delta-=np.rint(delta)
        distance=np.linalg.norm(delta@cell,axis=-1)
        assigned=distance.argmin(axis=1)
        score=float(np.mean(distance[np.arange(len(f)),assigned]**2))
        if best is None or score<best[0]:
            best=(score,grid+origin,assigned,distance)
    _,sites,assigned,distance=best
    if len(set(assigned))!=62:
        raise ValueError('host atoms do not uniquely match 62 parent sites')
    nn=float(atoms.cell.lengths()[0]*np.sqrt(3)/8)
    max_dist=float(distance.min(axis=1).max())
    if max_dist>=nn/2:
        raise ValueError(f'host assignment exceeds half the first-neighbor spacing ({max_dist:.3f} A)')
    missing=np.setdiff1d(np.arange(64),assigned)
    sites=sites[missing]%1.
    d=sites[0]-sites[1]
    if abs(np.linalg.norm((d-np.rint(d))@cell)-nn)>0.02:
        raise ValueError('the missing parent sites are not a nearest-neighbor pair')
    dopant_index=int(np.flatnonzero(symbols==dopant)[0])
    displacement=scaled[dopant_index]-sites
    dopant_dist=np.linalg.norm((displacement-np.rint(displacement))@cell,axis=1)
    onsite=int(np.argmin(dopant_dist)) if dopant_dist.min()<0.30*nn else None
    remaining=[s.tolist() for j,s in enumerate(sites) if j!=onsite]
    return dict(dopant_index=dopant_index,dopant_occupies_host_site=onsite is not None,
                vacancy_frac=remaining, host_max_displacement_A=max_dist,
                impurity_to_two_parent_sites_A=sorted(dopant_dist.tolist()))


def prepare(directory=DATA_ROOT):
    from ase.io import read
    directory=Path(directory)
    ensure_sources(directory)
    label_file=directory/'pbe_labels.json'
    if not label_file.exists():
        extract_labels(directory)
    raw_labels=label_file.read_bytes().replace(b'\r\n',b'\n')
    # Text-mode files use CRLF on Windows; pin the published numeric content across platforms.
    if hashlib.sha256(raw_labels).hexdigest()!=LABELS_SHA256:
        raise ValueError('Extracted enthalpies differ from the pinned PBE tables')
    labels=json.loads(raw_labels)
    records,excluded={},[]
    with zipfile.ZipFile(directory/'cm6b03368_si_001.zip') as archive:
        for row in labels:
            h,z,k=row['host'],row['dopant'],row['kind']
            name=f'{h}/period{row["period"]}/{k}/{h}-{k}-{z}.cif'
            try:
                geometry=archive.read(name)
                if not geometry.strip():
                    raise ValueError('empty published CIF')
                atoms=read(io.BytesIO(geometry),format='cif')
                mapping=map_complex(atoms,h,z)
            except (ValueError,KeyError) as exc:
                excluded.append(dict(source=name,reason=str(exc)))
                continue
            records.setdefault((h,z),[]).append(dict(**row, source=name,**mapping))
    complete=[]
    for key,group in sorted(records.items()):
        if {r['kind'] for r in group}=={'BC','SV'}:complete.extend(group)
        else:excluded.extend(dict(source=r['source'],reason='paired geometry unavailable') for r in group)
    manifest=dict(schema=SCHEMA, source_doi_structure='10.1021/acs.chemmater.6b03368.s001',
                  source_doi_labels='10.1021/acs.chemmater.6b03368.s002',
                  label_source_md5=FILES['cm6b03368_si_002.pdf'][1],
                  labels_sha256=LABELS_SHA256,
                  target='published PBE defect formation enthalpy in eV/complex',
                  site_rule='Unique nearest periodic matches for all 62 host atoms. Two adjacent host sites are missing. An impurity within 0.30 host-neighbor spacings of a missing site occupies it; otherwise it is interstitial.',
                  records=complete,excluded=excluded)
    (directory/(SCHEMA+'.json')).write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n',encoding='utf8')
    return manifest

