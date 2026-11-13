"""Fang et al. MgAl2O4 inversion structures and DFT energies, DOI 10.1063/5.0146056."""
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import zipfile

import numpy as np
import spglib
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.geometry import cellpar_to_cell
from ase.io import read, write
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
REVISION = 'f9a41f127501065577cac4db65f796531785e1e9'
SOURCE = f'https://github.com/YIF27/Dataset-of-Spinel/tree/{REVISION}/dataset_MgAl2O4'
DOI = 'https://doi.org/10.1063/5.0146056'
DIGEST = '0e26271dbd6acc971dda4ae24d8cb1c24427076dd4c59ba2b92ffeb1fd3caa45'
ARCHIVE = ROOT.parent / 'dataset' / 'spinel_fang' / f'Dataset-of-Spinel-{REVISION}.zip'
RESULTS = ROOT / 'results' / 'spinel_antisites_20261110'
SCHEMA = 'spinel_mgal2o4_antisite_cv_v1'
NORMAL = 'CONTCAR-normal.json'
STRATA = (2, 4, 6)
SPLIT_SEED = 20261110


def read_source(archive):
    raw = {}
    with zipfile.ZipFile(archive) as zipped:
        for name in zipped.namelist():
            key = '/dataset_MgAl2O4/'
            if key in '/' + name and name.endswith('.json'):
                source_id = ('/' + name).split(key, 1)[1]
                if source_id in raw:
                    raise ValueError(f'Duplicate source path {source_id}')
                raw[source_id] = zipped.read(name)
    digest = hashlib.sha256()
    for source_id, payload in sorted(raw.items()):
        digest.update(source_id.encode() + b'\0' + payload)
    if digest.hexdigest() != DIGEST:
        raise ValueError('MgAl2O4 JSON files differ from the pinned author release')
    return raw


def to_atoms(payload):
    data = json.loads(payload)
    cell = cellpar_to_cell([data['lattice'][key] for key in ('a', 'b', 'c', 'alpha', 'beta', 'gamma')])
    sites = data['sites']
    if any(len(s['species']) != 1 or s['species'][0]['occu'] != 1 for s in sites):
        raise ValueError('Partial or disordered occupancy')
    atoms = Atoms([s['species'][0]['element'] for s in sites], cell=cell, pbc=True,
                  scaled_positions=[s['abc'] for s in sites])
    if Counter(atoms.symbols) != {'Mg': 8, 'Al': 16, 'O': 32}:
        raise ValueError('Unexpected supercell composition')
    atoms.calc = SinglePointCalculator(atoms, energy=float(data['energy']))
    return atoms


def parent_permutations(parent):
    frac = parent.get_scaled_positions()
    symmetry = spglib.get_symmetry((parent.cell.array, frac, parent.numbers), symprec=0.002)
    tree = cKDTree(frac % 1, boxsize=1)
    permutations = []
    for rotation, translation in zip(symmetry['rotations'], symmetry['translations'], strict=True):
        distance, indices = tree.query((frac @ rotation.T + translation) % 1)
        if (distance.max() > 0.001 or len(set(indices)) != len(parent)
                or not np.array_equal(parent.numbers, parent.numbers[indices])):
            raise ValueError('Invalid symmetry permutation of ordered MgAl2O4')
        permutations.append(indices)
    result = np.unique(permutations, axis=0)
    if len(result) != 192:
        raise ValueError('Incomplete normal-spinel symmetry group')
    return result


def nominal_k(source):
    if source == NORMAL:
        return 0
    if source == 'CONTCAR-inverse.json':
        return 8
    return round(8 * float(re.search(r'CONTCAR-x-([0-9.]+)-json/', source).group(1)))


def first_shell_masks(parent):
    """Contact pairs on the periodic normal parent, counted once per site pair."""
    distance = parent.get_all_distances(mic=True)
    z = parent.numbers
    shells = {}
    for a, b, label in ((12, 12, 'tet_tet'), (12, 13, 'tet_oct'), (13, 13, 'oct_oct')):
        allowed = (z[:, None] == a) & (z[None, :] == b) & (distance > 0.01)
        shortest = np.min(distance[allowed])
        mask = allowed & (distance < shortest + 0.1)
        shells[label] = np.triu(mask, k=1) if a == b else mask
    tetrahedra = int(np.count_nonzero(z == 12))
    if [int(mask.sum()) for mask in shells.values()] != [2*tetrahedra, 12*tetrahedra, 6*tetrahedra]:
        raise ValueError('Unexpected cation coordination on normal parent')
    return shells


def collect(archive=ARCHIVE):
    raw = read_source(archive)
    geometries = {name: to_atoms(payload) for name, payload in raw.items()}
    parent = geometries[NORMAL]
    parent_frac = parent.get_scaled_positions()
    symmetry = parent_permutations(parent)
    shells = first_shell_masks(parent)
    records = []
    groups = defaultdict(list)
    for name, atoms in sorted(geometries.items()):
        delta = atoms.get_scaled_positions()[:, None, :] - parent_frac[None, :, :]
        distance = np.linalg.norm((delta - np.rint(delta)) @ atoms.cell.array, axis=-1)
        atom_index, site = linear_sum_assignment(distance)
        shift = float(distance[atom_index, site].max())
        # Each atom's nearest parent site must also be its unique global match.
        if shift > 0.35 or not np.array_equal(site, distance.argmin(axis=1)):
            raise ValueError(f'Ambiguous match to parent lattice in {name}')
        was = parent.numbers[site]
        if not np.array_equal(was == 8, atoms.numbers == 8):
            raise ValueError(f'Oxygen changed sublattice in {name}')
        k = int(np.sum((was == 12) & (atoms.numbers == 13)))
        if k != int(np.sum((was == 13) & (atoms.numbers == 12))):
            raise ValueError(f'Unbalanced Mg/Al exchange in {name}')
        occupation = np.empty(len(parent), dtype=np.uint8)
        occupation[site] = atoms.numbers
        signature = hashlib.sha256(min(row.tobytes() for row in occupation[symmetry])).hexdigest()
        energy = atoms.get_potential_energy()
        record = dict(source_id=name, formula='Mg8Al16O32', energy_eV_cell=energy,
                      delta_to_normal_eV_cell=round(energy - parent.get_potential_energy(), 8),
                      atoms=56, k_each=k, antisite_sites=2*k, inversion=k/8, nominal_k=nominal_k(name),
                      occupation_sha256=signature, max_parent_displacement_A=shift)
        antisite = occupation != parent.numbers
        for label, mask in shells.items():
            record[f'antisite_contacts_{label}'] = int(mask[np.ix_(antisite, antisite)].sum())
        atoms.info.update(source_id=name, antisites_each_kind=k, inversion_fraction=k/8)
        atoms.new_array('parent_site_index', site.copy())
        atoms.new_array('was_Z', was.copy())
        atoms.new_array('is_antisite', (was != atoms.numbers).astype(int))
        records.append(record)
        groups[signature].append(record)
    for rows in groups.values():
        exemplar = min(rows, key=lambda r: (r['nominal_k'] != r['k_each'], r['source_id']))
        for r in rows:
            r['representative_id'] = exemplar['source_id']
            r['is_representative'] = r is exemplar
    distinct = [r for r in records if r['is_representative']]
    if Counter(r['k_each'] for r in distinct) != {0: 1, 2: 14, 4: 18, 6: 19, 8: 1}:
        raise ValueError('Unexpected distinct occupations')
    return raw, geometries, records


def partitions(cohort):
    rng = np.random.default_rng(SPLIT_SEED)
    groups = {k: sorted(r['source_id'] for r in cohort if r['k_each'] == k) for k in STRATA}
    chunks = {k: np.array_split(rng.permutation(ids), 3) for k, ids in groups.items()}
    folds = []
    for i in range(3):
        fold = {part: [] for part in ('train', 'val', 'test')}
        for k in STRATA:
            test = chunks[k][i].tolist()
            remaining = sorted(set(groups[k]) - set(test))
            reordered = rng.permutation(remaining).tolist()
            nval = max(1, round(0.2 * len(remaining)))
            fold['test'].extend(test)
            fold['val'].extend(reordered[:nval])
            fold['train'].extend(reordered[nval:])
        folds.append({part: sorted(ids) for part, ids in fold.items()})
    ids = {r['source_id'] for r in cohort}
    for fold in folds:
        all_ids = sum(fold.values(), [])
        if len(all_ids) != len(ids) or set(all_ids) != ids:
            raise ValueError('Missing or duplicated occupation in fold')
    if Counter(s for fold in folds for s in fold['test']) != Counter(ids):
        raise ValueError('Each configuration must be held out exactly once')
    return folds


def write_bundle(path, raw, geometries, records, folds):
    distinct = [r for r in records if r['is_representative']]
    structures = io.StringIO()
    write(structures, [geometries[r['source_id']] for r in distinct], format='extxyz')
    xyz = structures.getvalue()
    # Check energies, mapping, and geometry through the common ASE entry point.
    frames = read(io.StringIO(xyz), index=':', format='extxyz')
    for frame, row in zip(frames, distinct, strict=True):
        original = geometries[row['source_id']]
        if (frame.info['source_id'] != row['source_id']
                or abs(frame.get_potential_energy() - row['energy_eV_cell']) > 1e-8
                or not np.array_equal(frame.numbers, original.numbers)
                or not np.allclose(frame.positions, original.positions, atol=1e-8, rtol=0)
                or not np.allclose(frame.cell.array, original.cell.array, atol=1e-9, rtol=0)
                or any(not np.array_equal(frame.arrays[key], original.arrays[key])
                       for key in ('was_Z', 'parent_site_index', 'is_antisite'))):
            raise ValueError(f"Extended XYZ round trip failed for {row['source_id']}")
    table = io.StringIO(newline='')
    writer = csv.DictWriter(table, fieldnames=records[0].keys())
    writer.writeheader()
    writer.writerows(records)
    readme = f"""MgAl2O4 cation antisite arrangements from Fang, Ohodnicki and Wang (2023)
{DOI}
Source files and version: {SOURCE}

dataset_MgAl2O4/ contains the 62 original, byte-preserved JSON files.
records.csv links all entries to their source IDs, total DFT energies in eV per 56-atom
Mg8Al16O32 cell, arrangement signature, and representative. delta_to_normal_eV_cell
uses the energy of CONTCAR-normal.json. k_each counts Al on tetrahedral Mg sites,
equal to Mg on octahedral Al sites. There are 2*k_each antisite atoms. Inversion = k_each/8.
structures.extxyz contains the 53 symmetry-distinct configurations, including both end
members. Read with ase.io.read('structures.extxyz', index=':') and get_potential_energy().
Per-atom arrays give the zero-based normal-parent site index, its original atomic number
(was_Z), and the antisite mask. Oxygen sites and relaxed coordinates are present.

At k_each = 2, 4, 6 the distinct structures span 0.58251, 0.70361, 1.46756 eV/cell
respectively. In the x=0.50 folder, CONTCAR-8 and CONTCAR-13 have energies
-396.30981 and -395.60620 eV/cell. Both have four Al on tetrahedral Mg sites and
four Mg on octahedral Al sites. The former has one nearest-neighbor pair of
octahedral Mg antisites, the latter has four. These contacts are counted once per
pair using the first octahedral cation shell of the periodic normal parent (~2.889 A).
The publication relates cation ordering to metal-oxygen-metal environments and to
the thermodynamic equilibrium degree of cation inversion.
The three antisite_contacts columns in records.csv count such first-shell contacts
between tetrahedral-tetrahedral, tetrahedral-octahedral, and octahedral-octahedral
antisite atoms on the ordered parent. Their shell radii are ~3.538, 3.387 and 2.889 A.

Parent sites were matched to the ordered cell with a periodic nearest-site assignment.
The 192 symmetries of the ordered reference collapse equivalent chemical occupations.
A representative has a correct source-folder inversion label when possible, then the
alphabetically first relative source path. No energy was used to choose one.
CONTCAR-x-0.25-json/CONTCAR-20.json has k_each=6 when read from its coordinates.
Recalculated counts and source folder labels are both recorded.

folds.json defines three train/validation/test partitions of 51 unique intermediate
arrangements (14, 18, 19 in k_each=2, 4, 6). The ordered and fully inverse references
are excluded from training and evaluation. Fold construction uses counts and IDs only.
Compare pairs only within the same test fold and at equal k_each. This evaluates
site arrangements in relaxed structures. No DFT forces or new DFT calculations are included.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as tmp:
        candidate = Path(tmp) / path.name
        with zipfile.ZipFile(candidate, 'w', compression=zipfile.ZIP_DEFLATED) as out:
            for source, payload in sorted(raw.items()):
                out.writestr('dataset_MgAl2O4/' + source, payload)
            out.writestr('records.csv', table.getvalue())
            out.writestr('structures.extxyz', xyz)
            out.writestr('folds.json', json.dumps(folds, indent=2) + '\n')
            out.writestr('README.txt', readme)
        if read_source(candidate) != raw:
            raise ValueError('Source data changed in prepared bundle')
        candidate.replace(path)
