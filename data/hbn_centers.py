"""Neutral monolayer triplet complexes from the hBN Defects Database."""
from collections import defaultdict
from functools import lru_cache
from hashlib import sha256
import io
from itertools import product
from pathlib import Path
import sqlite3
import tempfile
import urllib.request

import numpy as np
from ase.io.cif import parse_cif
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
import spglib

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT.parent / 'dataset' / 'hbn_color_centers'
REVISION = '3b177bb94bf49d06af831df4a97ff5ab71067e67'
DIGEST = 'edf84d8f88d5f693129473d4f4a57c663e1107e014afcaaca88b03267fae4861'
SCHEMA = 'hbn_neutral_monolayer_triplet_v1'
TRIPLET_ENERGY = 'Emission properties: Ground-state total energy (eV)'


def database(root):
    path = Path(root).resolve() / 'hbn_defects_database.db'
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f'https://raw.githubusercontent.com/QCS-Theory/hBN-database/{REVISION}/{path.name}'
        with urllib.request.urlopen(url, timeout=120) as response:
            raw = response.read()
        if sha256(raw).hexdigest() != DIGEST:
            raise ValueError('Unknown database from the public release')
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            handle.write(raw)
            temporary = Path(handle.name)
        temporary.replace(path)
    if sha256(path.read_bytes()).hexdigest() != DIGEST:
        raise ValueError('Local database differs from the pinned release')
    return path


@lru_cache(maxsize=2)
def parent(orientation):
    # 7 x 7 honeycomb. +/- handles the B/N lateral orientation in the source CIF.
    basis = np.asarray(list(product(range(7), range(7))), dtype=float) / 7
    delta = np.array([1., -1.]) * orientation / 21
    sites = np.vstack([basis, (basis + delta) % 1.])
    numbers = np.repeat([5, 7], 49)
    frac = np.column_stack((sites, np.full(98, 0.5)))
    lattice = np.array([[1.,0.,0.],[-0.5,np.sqrt(.75),0.],[0.,0.,1.]])
    operations = spglib.get_symmetry((lattice, frac, numbers), symprec=1e-5)
    tree = cKDTree(frac % 1, boxsize=1.)
    permutations = []
    for rot, translation in zip(operations['rotations'], operations['translations']):
        dist, ids = tree.query((frac @ rot.T + translation) % 1.)
        if dist.max()>1e-6 or len(set(ids))!=98 or not np.array_equal(numbers[ids],numbers):
            raise ValueError('Invalid symmetry operation on pristine hBN')
        permutations.append(ids)
    return sites, numbers, np.unique(permutations,axis=0)


def site_mapping(a):
    xy = a.get_scaled_positions(wrap=True)[:,:2]
    if len(a) < 90 or not np.allclose(a.cell.cellpar(),[17.5279,17.5279,15.,90.,90.,120.],atol=.03):
        raise ValueError(f'Unexpected monolayer cell {a.cell.cellpar()}')
    phase = {z:np.angle(np.mean(np.exp(2j*np.pi*7*xy[a.numbers==z]),axis=0))/(14*np.pi) for z in (5,7)}
    shift = (phase[7]-phase[5])*7
    shift -= np.rint(shift)
    orientation = 1 if shift[0]>0 else -1
    if np.max(np.abs(shift - np.array([1.,-1.])*orientation/3))>.07:
        raise ValueError('Could not determine B/N sublattice orientation')
    sites, numbers, permutations = parent(orientation)
    sites = (sites+phase[5])%1.
    cell = a.cell.array[:2,:2]
    diff = xy[:,None,:]-sites[None,:,:]
    diff -= np.rint(diff)
    distances = np.linalg.norm(diff @ cell, axis=-1)
    bond = a.cell.lengths()[0]/(7*np.sqrt(3))
    cutoff = .56*bond
    # An atom farther away than the cutoff is free to occupy an interstitial.
    costs = np.column_stack([distances, np.full((len(a),len(a)),cutoff)])
    rows, where = linear_sum_assignment(costs)
    if not np.array_equal(rows,np.arange(len(a))):raise ValueError('Incomplete site assignment')
    mapped = where < len(sites)
    previous = np.zeros(len(a),dtype=int)
    previous[mapped] = numbers[where[mapped]]
    occupation = np.zeros(len(sites),dtype=np.uint8)
    occupation[where[mapped]] = a.numbers[mapped]
    missing = np.flatnonzero(occupation==0)
    if len(where[mapped])!=len(set(where[mapped])):raise ValueError('Nonunique site assignment')
    height = float(np.median(a.get_scaled_positions()[:,2]))
    vacancies = [[*sites[i].tolist(),height] for i in missing]
    # Orient both grids identically before canonicalization. Inversion maps the
    # minus orientation to the plus orientation on the numbered parent grid.
    if orientation<0:
        plus, _, permutations = parent(1)
        tree=cKDTree(plus%1.,boxsize=1.)
        d,flip=tree.query((-parent(-1)[0])%1.)
        if d.max()>1e-6:raise ValueError('Cannot align pristine orientations')
        aligned=np.zeros_like(occupation);aligned[flip]=occupation;occupation=aligned
    occ=min(vec.tobytes() for vec in occupation[permutations])
    extra=bytes(sorted(int(n) for n in a.numbers[~mapped]))
    signature=sha256(occ+extra).hexdigest()
    return dict(was=previous.tolist(),type=(previous!=a.numbers).tolist(),vacancy_frac=vacancies,
                vacancy_species=numbers[missing].tolist(),signature=signature,
                defect_markers=int((previous!=a.numbers).sum()+len(missing)),
                offsite_atoms=int((~mapped).sum()))


def load_records(root=DATA_ROOT):
    path=database(root)
    with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row
        rows=db.execute('SELECT * FROM updated_data WHERE Host=? AND [Charge state]=? AND [Spin multiplicity]=? ORDER BY Defect,rowid',
                        ('monolayer',0,'triplet')).fetchall()
    grouped=defaultdict(list)
    for row in rows:grouped[row['Defect']].append(row)
    records=[]; geometries={}
    for name, candidates in grouped.items():
        energies=np.array([r[TRIPLET_ENERGY] for r in candidates],dtype=float)
        cifs=[r['structure_ground_triplet'] for r in candidates]
        if not np.isfinite(energies).all() or max(energies)>=0 or np.ptp(energies)>1e-4:
            raise ValueError(f'{name}: missing or conflicting ground-state energies')
        if None in cifs or len(set(cifs))!=1:
            raise ValueError(f'{name}: missing or conflicting triplet ground-state coordinates')
        block,=list(parse_cif(io.BytesIO(cifs[0])))
        a=block.get_unsymmetrized_structure()
        a.pbc=True
        record=dict(source=name,formula=a.get_chemical_formula(),energy=float(energies[0]),
                    source_rows=len(candidates),**site_mapping(a))
        if not record['defect_markers']:raise ValueError(f'{name}: no defect assigned')
        records.append(record);geometries[name]=a
    if len(records)!=93 or sum(r['source_rows'] for r in records)!=112:
        raise ValueError('Unexpected size of the published neutral monolayer triplet cohort')
    return records,geometries
