"""BCC Fe solute-vacancy pair shells from Kartamyshev et al., Mendeley Data v2."""
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.etree import ElementTree as ET
import zipfile

import numpy as np
from ase.build import bulk
from ase.data import atomic_numbers
from ase.io import read, write


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'fe_x_vacancy_shells_element_cv_v1'
ARCHIVE = ROOT.parent/'dataset'/'fe_solute_vacancy_kartamyshev'/'zbm4wjmvny_v2.zip'
RESULTS = ROOT/'results'/'fe_solute_vacancy_20261110'
SOURCE = 'https://doi.org/10.17632/zbm4wjmvny.2'
MASTER = 'a91f7e1e7e2461c0827e21002b7c1b8da19f3079cd522f0cb6d39b24010ae411'
CHECK_CSV = 'e916ec4e766e511ccae5588d3a036f7676407090ead4ecbf9e6b88c5e0603d99'
SPLIT_SEED = 20261110
LATTICE_A = 2.87  # ASE reference bcc Fe. Used only for the ideal, reconstructed input.


def source_bytes(archive):
    with zipfile.ZipFile(archive) as z:
        def get(suffix):
            keys = [n for n in z.namelist() if ('/'+n).endswith('/'+suffix)]
            if len(keys) != 1:
                raise ValueError(f'Expected one source member {suffix}')
            return z.read(keys[0])
        master = get('basic/energies.xlsx')
        source_csv = get('sol, vac/vac1/gbrt_v1_database.csv')
        supplement = get('Supplementary_materials.pdf')
    if (hashlib.sha256(master).hexdigest() != MASTER
            or hashlib.sha256(source_csv).hexdigest() != CHECK_CSV):
        raise ValueError('Energy workbook or CSV differs from the public release')
    return master, source_csv, supplement


def binding_table(master, source_csv):
    # Read only the published worksheet and cached numerical cells. This keeps
    # the benchmark independent of Excel and of version-specific spreadsheet packages.
    ns = {'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    rid = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id'
    with zipfile.ZipFile(io.BytesIO(master)) as x:
        workbook = ET.fromstring(x.read('xl/workbook.xml'))
        sheet = next(s for s in workbook.findall('.//m:sheet', ns) if s.get('name')=='Substitution-vacancy')
        links = ET.fromstring(x.read('xl/_rels/workbook.xml.rels'))
        target = next(r.get('Target') for r in links if r.get('Id')==sheet.get(rid))
        part = target.lstrip('/') if target.startswith('/') else 'xl/'+target
        strings = []
        if 'xl/sharedStrings.xml' in x.namelist():
            strings = [''.join(t.itertext()) for t in ET.fromstring(x.read('xl/sharedStrings.xml'))]
        cells = []
        for row in ET.fromstring(x.read(part)).findall('.//m:sheetData/m:row', ns):
            fields = {}
            for c in row.findall('m:c', ns):
                key = c.attrib['r'].rstrip('0123456789')
                val = c.findtext('m:v', '', ns)
                if c.get('t') == 's':
                    val = strings[int(val)]
                elif c.get('t') == 'inlineStr':
                    val = ''.join(c.find('m:is', ns).itertext())
                if val:
                    fields[key] = val
            if fields:
                cells.append((int(row.get('r')), fields))
    if [cells[0][1][k] for k in 'BCD'] != ['1nn','2nn','3nn']:
        raise ValueError('Unexpected shell headers')
    values = {c['A']:(n, [float(c[k]) for k in 'BCD']) for n,c in cells[1:]}
    if len(values) != 46 or len(cells) != 47 or values['Fe'][1] != [0.,0.,0.]:
        raise ValueError('Unexpected master workbook cohort or Fe reference')
    auxiliary = list(csv.DictReader(io.StringIO(source_csv.decode('utf-8-sig'))))
    if len(auxiliary) != 40:
        raise ValueError('Unexpected redundant CSV row count')
    for r in auxiliary:
        if [float(r[f'Vacancy_{j}']) for j in (1,2,3)] != values[r['Elements']][1]:
            raise ValueError('CSV and workbook disagree for '+r['Elements'])
    return values


def prepare(archive=ARCHIVE, vacancy_rotation=None):
    raw = source_bytes(archive)
    table = binding_table(*raw[:2])
    parent = bulk('Fe', 'bcc', a=LATTICE_A, cubic=True).repeat((3,3,3))
    center = np.array([1.,1.,1.])
    def site_at(f):
        d = np.linalg.norm(parent.positions - LATTICE_A*f, axis=1)
        j = int(np.argmin(d))
        if d[j] > 1e-6:
            raise ValueError('Nominal BCC site absent')
        return j
    solute_site = site_at(center)
    rotation = np.eye(3) if vacancy_rotation is None else np.asarray(vacancy_rotation)
    if (rotation.shape != (3,3) or not np.array_equal(rotation, np.rint(rotation))
            or not np.allclose(rotation.T@rotation,np.eye(3)) or not np.isclose(np.linalg.det(rotation),1)):
        raise ValueError('Expected a proper cubic symmetry operation')
    directions = (np.array([.5,.5,.5]), np.array([1.,0.,0.]), np.array([1.,1.,0.]))
    shells = [site_at(center+rotation@d) for d in directions]
    distances = parent.get_distances(solute_site, range(len(parent)), mic=True)
    radii = distances[shells]
    if (len(parent) != 54 or not np.allclose(radii, LATTICE_A*np.array([np.sqrt(3)/2,1.,np.sqrt(2)]))
            or [int(np.isclose(distances,r).sum()) for r in radii] != [8,6,12]):
        raise ValueError('Ideal cell does not reproduce the three BCC coordination shells')
    records, atoms = [], {}
    for element, (row, energies) in table.items():
        if element == 'Fe':
            continue  # Tabulated zero reference: a parent Fe atom is not a second defect.
        for j, (vac, energy, radius) in enumerate(zip(shells,energies,radii,strict=True),start=1):
            a = parent.copy()
            a[solute_site].symbol = element
            a.new_array('parent_site', np.arange(len(a)))
            a.new_array('was_Z', parent.numbers.copy())
            a.new_array('is_solute', (a.numbers != parent.numbers).astype(int))
            vfrac = parent.get_scaled_positions()[vac].copy()
            del a[vac]
            sid = f'{element}/{j}NN'
            if Counter(a.get_chemical_symbols()) != {'Fe':52,element:1} or not np.isfinite(energy):
                raise ValueError(f'Invalid structure or energy for {sid}')
            records.append(dict(source_id=sid, solute=element, shell=j, formula=f'Fe52{element}',
                                binding_eV=energy, distance_A=float(radius), workbook_row=row,
                                workbook_col='BCD'[j-1], vacancy_parent_site=vac, solute_parent_site=solute_site))
            a.info.update(source_id=sid, shell=j, dft_binding_eV=energy, vacancy_fractional=vfrac,
                          formula_with_vacancy=f'Fe52{element}+V_Fe')
            atoms[sid] = a
    return records, atoms, raw


def partitions(records):
    elements = sorted({r['solute'] for r in records},key=atomic_numbers.get)
    rng = np.random.default_rng(SPLIT_SEED)
    periods = np.array([2,10,18,36,54,86,118])
    held = [[],[],[]]
    cursor = 0
    for p in range(7):
        group = [e for e in elements if int(np.searchsorted(periods,atomic_numbers[e]))==p]
        for e in rng.permutation(group).tolist():
            held[cursor % 3].append(e)
            cursor += 1
    splits = []
    for test in held:
        remaining = sorted(set(elements)-set(test),key=atomic_numbers.get)
        val = set(rng.permutation(remaining)[:6])
        parts = {'train':set(remaining)-val,'val':val,'test':set(test)}
        ids = {p:sorted(r['source_id'] for r in records if r['solute'] in es) for p,es in parts.items()}
        if [len(v) for v in ids.values()] != [72,18,45]:
            raise ValueError('Unbalanced or incomplete elemental holdout')
        splits.append(ids)
    if Counter(s for f in splits for s in f['test']) != Counter(r['source_id'] for r in records):
        raise ValueError('Some occupations are repeated or missing from the test folds')
    return splits


def write_bundle(path, records, atoms, raw, splits):
    out = io.StringIO()
    write(out,[atoms[r['source_id']] for r in records], format='extxyz')
    xyz = out.getvalue()
    for r, a in zip(records, read(io.StringIO(xyz),index=':',format='extxyz'),strict=True):
        orig=atoms[r['source_id']]
        if (a.info['source_id'] != r['source_id'] or a.info['dft_binding_eV'] != r['binding_eV']
                or not np.allclose(a.positions,orig.positions,atol=1e-8,rtol=0)
                or not np.array_equal(a.numbers,orig.numbers)
                or not np.array_equal(a.arrays['is_solute'],orig.arrays['is_solute'])
                or not np.allclose(a.info['vacancy_fractional'], orig.info['vacancy_fractional'])):
            raise ValueError('XYZ round-trip changed a label or a defect')
    table=io.StringIO(newline='')
    writer=csv.DictWriter(table,fieldnames=records[0])
    writer.writeheader()
    writer.writerows(records)
    readme=f"""BCC iron, substitutional solute and Fe vacancy. Kartamyshev et al., Mendeley Data v2
{SOURCE}
Author source files in this zip are byte-preserved. See basic/energies.xlsx, worksheet
Substitution-vacancy (B:D), for DFT binding energies in eV. Each row has 1NN, 2NN and
3NN arrangements. The 40-row CSV agrees on all 120 overlapping values. The master sheet
adds six solutes. The all-zero Fe host reference is excluded, leaving 45 genuine solutes.

structures.extxyz contains 135 RECONSTRUCTED IDEAL initial inputs, not DFT-relaxed coordinates.
The published method uses a 3x3x3 conventional bcc cell. The common ideal input uses the ASE
reference lattice constant a=2.87 angstrom. Each configuration contains 52 Fe, one solute,
and one missing Fe site (53 real atoms and 54 parent sites). Its missing-site fractional
coordinate is in vacancy_fractional. The per-atom is_solute flag marks the substituted atom.
Read with ase.io.read('structures.extxyz',index=':'); energy is frame.info['dft_binding_eV'].

Higher binding energy is more stable. At fixed solute, relaxed total-energy differences obey
E(B)-E(A) = binding(A)-binding(B) because the isolated-defect reference terms cancel.
No absolute DFT total energies, magnetic moments or relaxed displacements are supplied as
model inputs. folds.json puts all three arrangements of a solute together. Across the three
test folds every solute is unseen during its own training/validation and is tested once.
The source release is CC BY 4.0. Retain author attribution when sharing.
"""
    path.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as tmp:
        candidate=Path(tmp)/path.name
        with zipfile.ZipFile(candidate,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for n, b in zip(('basic/energies.xlsx','sol, vac/vac1/gbrt_v1_database.csv','Supplementary_materials.pdf'),raw,strict=True):
                z.writestr(n,b)
            z.writestr('records.csv',table.getvalue())
            z.writestr('structures.extxyz',xyz)
            z.writestr('folds.json',json.dumps(splits,indent=2))
            z.writestr('README.txt',readme)
        if source_bytes(candidate) != raw:
            raise ValueError('Source files changed in bundled archive')
        candidate.replace(path)
