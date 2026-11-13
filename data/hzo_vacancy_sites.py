"""Neutral oxygen vacancies in the fixed equimolar Hf/Zr parent of Taper et al."""
from collections import Counter, defaultdict
import hashlib
import io
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import spglib
from ase.io import read

from ..arrangement_sources import SOURCES
from .ysz_arrangements import ROOT, write_arrangement_bundle


ARCHIVE = ROOT.parent/'dataset'/'hzo_taper'/'Data.zip'
DIGEST = 'e6fd7f7aa4ed5ba2154fcf06dd9808d8c643637576b8b489887780942a0c4ea9'
SOURCE = 'https://doi.org/10.17863/CAM.121316'
PAPER = 'https://journals.aps.org/prb/abstract/10.1103/vz64-n5xh'
TITLE = '铁电 Hf₁₆Zr₁₆O₆₃ 中的中性氧空位'
INTERPRETATION = ('固定作者图 2(c,f) 使用的 Hf/Zr 阳离子排布，依次移除 64 个不同氧位中的一个，'
                  '从弛豫前的空位位置和母相环境预测中性缺陷的 DFT 内能。每个构型均为 Hf₁₆Zr₁₆O₆₃。'
                  '移动的是一个氧空位，Hf 和 Zr 在不同构型中保持原位。')
SCHEMA = SOURCES['hzo'].schema
RESULTS = SOURCES['hzo'].results
BUNDLE = 'HZO_equimolar_vacancy_DFT.zip'
SPLIT_SEED = 20261110
CIF = 'Apollo deposit/Main text/Structures/Fig2c_f_0.50_HZO.cif'
TABLE = 'Apollo deposit/Main text/Fig_2/Data_Fig2b_g.csv'


def prepare(archive=ARCHIVE):
    payload = Path(archive).read_bytes()
    if hashlib.sha256(payload).hexdigest() != DIGEST:
        with zipfile.ZipFile(io.BytesIO(payload)) as curated:
            payload = curated.read('Data.zip')
    if hashlib.sha256(payload).hexdigest() != DIGEST:
        raise ValueError('Cambridge archive differs from the pinned source')
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        parent = read(io.StringIO(z.read(CIF).decode()), format='cif', store_tags=True)
        cols = ['Concentration', 'vacancy', 'Neutral Defect energy', 'Bulk_energy']
        table = pd.read_csv(z.open(TABLE), usecols=cols)
    table = table[table.Concentration == 'OUTCAR_0.5'].drop_duplicates().sort_values('vacancy')
    if (table.vacancy.tolist() != list(range(32, 96)) or table.Bulk_energy.nunique() != 1
            or not np.isfinite(table[['Neutral Defect energy', 'Bulk_energy']]).all().all()):
        raise ValueError('Expected one neutral DFT total per author vacancy site')
    symbols = parent.get_chemical_symbols()
    labels = parent.info['_atom_site_label']
    frac = parent.get_scaled_positions()
    raw_frac = np.array([parent.info['_atom_site_fract_' + axis] for axis in 'xyz']).T
    if (Counter(symbols) != {'Hf':16, 'Zr':16, 'O':64} or symbols[32:] != ['O']*64
            or labels != [f'{s}{i}' for i, s in enumerate(symbols)] or not np.allclose(frac, raw_frac, atol=1e-8)
            or not np.all(parent.pbc)):
        raise ValueError('CIF label order or coordinate assignment is ambiguous')
    sym = spglib.get_symmetry_dataset((parent.cell.array, frac, parent.numbers), symprec=1e-3)
    if sym is None or len(set(sym.equivalent_atoms[32:])) != 64:
        raise ValueError('Symmetry-equivalent oxygen sites must be grouped before splitting')
    parent.info.clear()
    distances = parent.get_all_distances(mic=True)[:, :32]
    records, geometries, published = [], {}, []
    for site, total, bulk in table[['vacancy','Neutral Defect energy','Bulk_energy']].itertuples(index=False, name=None):
        # Oxygen sites have three or four nearest cations, separated from the next shell.
        nbr = np.flatnonzero(distances[site] < 2.55)
        n_zr = int((parent.numbers[nbr] == 40).sum())
        sid = 'hzo50-' + hashlib.sha256(np.asarray(frac[site],dtype='<f8').tobytes()).hexdigest()[:16]
        a = parent.copy()
        a.new_array('was_Z', a.numbers.copy())
        a.new_array('defect_site', np.zeros(len(a), dtype=int))
        a.new_array('parent_site', np.arange(len(a)))
        del a[site]
        a.info.update(source_id=sid, vacancy_fractional=frac[site].copy(), vacancy_was_Z=8,
                      relaxed_DFT_total_eV=total)
        records.append(dict(source_id=sid, group='Hf16Zr16O63', stratum=f'{len(nbr)}-{n_zr}',
                            energy_eV_cell=total, vacancy_parent_site=site, first_shell_cations=len(nbr),
                            first_shell_Zr=n_zr, first_shell_Hf=len(nbr)-n_zr))
        geometries[sid] = a
        published.append(dict(source_id=sid, vacancy=site, neutral_defect_total_eV=total,
                              pristine_total_eV=bulk, parent_cif=CIF))
    if len(geometries) != 64 or len({r['stratum'] for r in records}) != 9:
        raise ValueError('Missing independent sites or local cation coordination environments')
    return records, geometries, published, payload


def partitions(records):
    folds = site_stratified_partitions(records, SPLIT_SEED, n_validation=8)
    if [len(f['test']) for f in folds] != [22,21,21]:
        raise ValueError('Wrong test coverage for the 64 inequivalent vacancies')
    return folds


def site_stratified_partitions(records, seed, n_validation):
    rng = np.random.default_rng(seed)
    strata = defaultdict(list)
    for r in records:
        strata[r['stratum']].append(r['source_id'])
    tests = [[], [], []]
    cursor = 0
    for key in sorted(strata):
        for sid in rng.permutation(sorted(strata[key])):
            tests[cursor % 3].append(str(sid))
            cursor += 1
    all_ids = {r['source_id'] for r in records}
    folds = []
    for test in tests:
        rest = all_ids - set(test)
        validation = set(rng.permutation(sorted(rest))[:n_validation].tolist())
        folds.append(dict(train=sorted(rest-validation), val=sorted(validation), test=sorted(test)))
    if Counter(s for f in folds for s in f['test']) != Counter(all_ids):
        raise ValueError('Incorrect coverage of symmetry-distinct configurations')
    return folds


def write_bundle(path, records, atoms, raw, payload, folds):
    readme = f"""Taper et al. Equimolar ferroelectric Hf0.5Zr0.5O2 oxygen vacancies
Paper {PAPER}
Cambridge data {SOURCE} (CC BY 4.0); Data.zip is the original author archive.
Use {CIF} and the Neutral Defect energy in {TABLE} for OUTCAR_0.5.
All 64 distinct O sites in this cation arrangement are represented exactly once.
Repeated Fermi levels, oxygen chemical potentials and charged labels do not add geometries.
Input is the published 96-site pristine parent with one specified oxygen removed.
The 95 real atoms are in configurations.extxyz. vacancy_fractional marks the empty site.
Hf and Zr are physical cations in the mixed host. Only the vacancy is flagged as
an explicit defect. The target is the author DFT total in eV, stored as
relaxed_DFT_total_eV. The reference terms in the formation energies are common
to all these neutral Hf16Zr16O63 structures and cancel from their differences.
The 2.55 Angstrom cation shell defines three- and four-coordinated oxygen sites.
Test folds balance these shell categories and the number of Zr neighbors without
using any energy labels. Each test site is evaluated once per model and seed.
"""
    write_arrangement_bundle(path, records, atoms, raw, payload, folds, 'Data.zip', DIGEST, readme)
