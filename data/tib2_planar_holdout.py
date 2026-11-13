"""Read the VASP-labelled active configurations in the Du et al. TiB2 MTP."""
from collections import defaultdict
import csv
from dataclasses import dataclass
import hashlib
import io
from pathlib import Path
import re
import zipfile

from ase import Atoms
from ase.io import read
import numpy as np
from pymatgen.core import Structure
from pymatgen.core.periodic_table import DummySpecies
from scipy.optimize import linear_sum_assignment

from .datasets import representation_for_mode

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT/'results/tib2_du_source_audit_20261111/TiB2_public_VASP_snapshots.zip'
DIGEST = '3b3a78ff36a4db85778418be47087b85730df83184a0ea97763b37047de6a041'
SOURCE = 'https://doi.org/10.48436/6w3n6-att84'
SCHEMA = 'tib2_du_planar_holdout_v1'
TEST = ['tib2-004', 'tib2-011', 'tib2-016']
SEED = 123


def phase(frac, repeats):
    return np.angle(np.exp(2j*np.pi*repeats*frac).mean())/(2*np.pi*repeats)


def mic(difference):
    return difference - np.rint(difference)


def assignment(sites, atoms, cell):
    vectors = mic(sites[:, None, :] - atoms[None, :, :]) @ cell
    cost = np.einsum('ijk,ijk->ij', vectors, vectors)
    si, ai = linear_sum_assignment(cost)
    return si, ai, np.sqrt(cost[si, ai])


def prism_centers(atoms):
    """Locate both B sublattices with the six Ti atoms of each parent prism.

    Each of the six prismatic Ti rows may glide along c at a planar fault.
    Following its own c phase also preserves the ordinary AlB2 parent when
    the rows share a phase. Only the Ti coordinates enter this construction.
    """
    cell = atoms.cell.array
    ti = atoms.get_scaled_positions()[atoms.numbers == 22]
    j, i = np.indices((6, 5)).reshape(2, -1)
    xy = np.stack(((i + .5*(j%2))/5, j/6), axis=1)
    dy = phase(ti[:, 1], 6)
    row = np.rint((ti[:, 1]-dy)*6).astype(int) % 6
    dx = phase(ti[:, 0] - (row%2)*.1, 5)
    if len(ti) != 240 or not all(np.any(row == n) for n in range(6)):
        raise ValueError('Expected the 240-Ti parent with six prismatic rows')
    dz = np.array([phase(ti[row == n, 2], 8) for n in range(6)])
    # Unwrap at the largest empty arc. Rows sharing a c phase must stay
    # together even when their noisy coordinates straddle half a period.
    phases = np.sort((8*dz) % 1.)
    gaps = np.diff(np.r_[phases, phases[0]+1])
    start = phases[(int(gaps.argmax())+1) % len(phases)]
    dz = (start + ((8*dz-start+1e-12) % 1.))/8
    parents = np.empty((8, 30, 3))
    parents[:, :, :2] = xy + [dx, dy]
    parents[:, :, 2] = np.arange(8)[:, None]/8 + dz[j]
    si, ai, ti_error = assignment(parents.reshape(-1, 3), ti, cell)
    site_index = np.empty(240, dtype=int)
    site_index[si] = ai
    site_index = site_index.reshape(8, 30)

    centers_xy = np.concatenate([xy + [.1, 1/18], xy + [0, 1/9]])
    # Use the undistorted two-dimensional triangular Ti net for topology.
    delta_xy = mic(centers_xy[:, None, :] - xy[None, :, :])
    dsq = ((delta_xy * [5, 3*np.sqrt(3)])**2).sum(axis=-1)
    triangles = np.argsort(dsq, axis=1)[:, :3]
    bottom = site_index[:, triangles]
    top = np.roll(site_index, -1, axis=0)[:, triangles]
    corner_ids = np.concatenate([bottom, top], axis=2).reshape(480, 6)
    return centers_from_ti(atoms, corner_ids), corner_ids, ti_error


def centers_from_ti(atoms, corner_ids):
    ti = atoms.get_scaled_positions()[atoms.numbers == 22]
    corners = ti[corner_ids]
    origin = corners[:, :1, :]
    return (origin[:, 0] + mic(corners-origin).mean(axis=1)) % 1.


@dataclass
class Snapshot:
    sid: str
    atoms: Atoms
    energy: float
    n_ti: int
    n_b: int
    original_cell: bool
    vacancies: np.ndarray
    audit: dict

    @property
    def composition(self):
        return (self.n_ti, self.n_b)

    @property
    def size(self):
        return len(self.atoms)


def load(archive=ARCHIVE):
    with zipfile.ZipFile(archive) as z:
        raw = z.read('active_set.cfg')
        index = list(csv.DictReader(io.StringIO(z.read('snapshot_index.csv').decode('utf-8-sig'))))
        # Published random-vacancy starting cells. In this series the source
        # retains atom IDs through loading and cleavage. Transport the same
        # missing sites with their six Ti neighbours in every strained frame.
        templates = {}
        for name, published_md5 in [('TiB1_5_Vac', '280f3cef45bcf542874327ca5dc592e2'),
                                    ('TiB1_7_Vac', '071e37b98a8a5b911f6c2c224af2f8ae')]:
            raw_template = z.read(f'templates/{name}.vasp')
            if hashlib.md5(raw_template).hexdigest() != published_md5:
                raise ValueError(f'TU Wien template checksum differs for {name}')
            a = read(io.StringIO(raw_template.decode()), format='vasp')
            centers, corner_ids, _ = prism_centers(a)
            occupied, obs, _ = assignment(centers, a.get_scaled_positions()[a.numbers == 5], a.cell.array)
            templates[len(a)] = a, corner_ids, occupied, obs
    if hashlib.sha256(raw).hexdigest() != DIGEST:
        raise ValueError('VASP source digest mismatch')
    blocks = re.findall(rb'(?m)^BEGIN_CFG\r?\n[\s\S]*?^END_CFG[ \t]*\r?$', raw)
    if len(blocks) != 120 or len(index) != len(blocks):
        raise ValueError('Incomplete VASP source')
    result = {}
    for num, (block, info) in enumerate(zip(blocks, index), 1):
        sid = f'tib2-{num:03d}'
        lines = [line.strip() for line in block.decode().splitlines()]
        n = int(lines[lines.index('Size')+1])
        start = next(i for i, line in enumerate(lines) if line.startswith('AtomData:')) + 1
        table = np.array([line.split()[:5] for line in lines[start:start+n]], dtype=float)
        species = table[:, 1].astype(int)
        if set(species) != {0, 1} or not np.array_equal(table[:, 0], np.arange(1, n+1)):
            raise ValueError(f'Unrecognized atom types in {sid}')
        atoms = Atoms(numbers=np.choose(species, [5, 22]), positions=table[:, 2:5])
        has_cell = 'Supercell' in lines
        if has_cell:
            ci = lines.index('Supercell')+1
            atoms.cell = np.array([line.split() for line in lines[ci:ci+3]], dtype=float)
        else:
            # The source cluster has no periodic images. All images of this
            # padded box lie beyond the network's radial cutoff.
            atoms.center(vacuum=8.)
        atoms.pbc = True
        n_ti = int((atoms.numbers == 22).sum())
        n_b = n-n_ti
        energy = float(lines[lines.index('Energy')+1])
        if (num, n, n_ti, n_b, energy, has_cell) != (
                int(info['block_1based']), int(info['n_atoms']), int(info['n_Ti']), int(info['n_B']),
                float(info['total_energy_eV']), info['has_cell'] == 'True') or not any(
                    line.split() == ['Feature', 'EFS_by', 'VASP'] for line in lines):
            raise ValueError(f'Source and index disagree: {sid}')
        vacs = np.zeros((0, 3))
        audit = {}
        if n_ti == 240 and n_b < 480:
            b = atoms.get_scaled_positions()[atoms.numbers == 5]
            if n in templates:
                a, corner_ids, occupied, obs = templates[n]
                if not np.array_equal(atoms.numbers, a.numbers):
                    raise ValueError(f'Source atom ordering differs from the random-vacancy parent in {sid}')
                diff = mic(atoms.get_scaled_positions() - a.get_scaled_positions())
                shift = np.angle(np.exp(2j*np.pi*diff).mean(axis=0))/(2*np.pi)
                ti_dist = np.linalg.norm(mic(diff[atoms.numbers == 22]-shift) @ a.cell.array, axis=1)
                if np.median(ti_dist) > 2.:
                    raise ValueError(f'Ti atom IDs do not register to the parent in {sid}')
                centers = centers_from_ti(atoms, corner_ids)
                b_dist = np.linalg.norm(mic(centers[occupied]-b[obs]) @ atoms.cell.array, axis=1)
                method = 'published_random_parent_atom_ids'
            else:
                centers, _, ti_dist = prism_centers(atoms)
                occupied, _, b_dist = assignment(centers, b, atoms.cell.array)
                method = 'snapshot_Ti_prisms'
            missing = np.setdiff1d(np.arange(480), occupied)
            vacs = centers[missing]
            # Each of six prismatic row intervals contains 80 parent B sites.
            gap_rows = (missing % 30)//5
            audit = dict(site_registration=method, ti_rms_A=float(np.sqrt(np.mean(ti_dist**2))), ti_max_A=float(ti_dist.max()),
                         b_rms_A=float(np.sqrt(np.mean(b_dist**2))), b_max_A=float(b_dist.max()),
                         empty_by_row=','.join(map(str, np.bincount(gap_rows, minlength=6))))
        result[sid] = Snapshot(sid, atoms, energy, n_ti, n_b, has_cell, vacs, audit)
    if [r.sid for r in result.values() if r.composition == (240, 320)] != TEST:
        raise ValueError('The holdout does not contain every Ti240B320 snapshot')
    return result


def make_structures(records, mode):
    rep = representation_for_mode(mode)
    structures = []
    for s in records.values():
        n = s.size
        vacant = s.vacancies if rep != 'full' else []
        markers = [False]*n + [True]*len(vacant)
        props = dict(type=markers, was=s.atoms.numbers.tolist()+[5]*len(vacant))
        if rep == 'hetero':
            # A vacancy-free parent or free cluster has no registered vacancy.
            # Use its real atoms for readout while retaining ordinary atom nodes.
            props['pool_type'] = markers if len(vacant) else [True]*n
        coords = s.atoms.get_scaled_positions()
        if len(vacant):
            coords = np.vstack((coords, vacant))
        st = Structure(s.atoms.cell.array, s.atoms.numbers.tolist()+[DummySpecies()]*len(vacant),
                       coords, site_properties=props)
        st.source_id = st.source_name = s.sid
        st.source_path = f'{ARCHIVE}!active_set.cfg#{s.sid}'
        st.material = st.defect_family = f'Ti{s.n_ti}B{s.n_b}'
        structures.append(st)
    return structures


def partition(records):
    """Hold out complete cell/strain groups for bulk, composition groups for clusters."""
    families = defaultdict(lambda: defaultdict(list))
    for s in records.values():
        if s.sid in TEST:
            continue
        if s.n_ti == 240:
            key = tuple(s.atoms.cell.array.round(5).ravel())
            family = f'bulk-B{s.n_b}'
        else:
            key = s.composition
            family = 'cluster'
        families[family][key].append(s.sid)
    val = set()
    rng = np.random.default_rng(2031)
    for family in sorted(families):
        groups = list(families[family].values())
        rng.shuffle(groups)
        target = round(.15*sum(map(len, groups)))
        # Find a collection of whole groups nearest the target partition size.
        subsets = {0: []}
        for group in groups:
            for size, prior in list(subsets.items()):
                subsets.setdefault(size+len(group), prior+group)
        candidates = [size for size in subsets if 0 < size < sum(map(len, groups))]
        if candidates:
            val.update(subsets[min(candidates, key=lambda x: (abs(x-target), x))])
    ids = {'train': [sid for sid in records if sid not in val and sid not in TEST],
           'val': [sid for sid in records if sid in val], 'test': list(TEST)}
    if sum(map(len, ids.values())) != 120 or set(ids['train']) & set(ids['test']):
        raise ValueError('Incomplete or overlapping split')
    return ids
