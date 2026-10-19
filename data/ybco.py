"""Pinned YBCO CP2K structures and an explicit fixed-reference energy task.

These are unrelaxed configuration formation energies, not the relaxed defect
energies plotted in the source paper, and not defect-pair interaction energies.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import urllib.request

import numpy as np

SOURCE_COMMIT = 'cfbb8a59f6657bc3f31fbadb9e06b254e6a43fbf'
SOURCE_URL = (f'https://media.githubusercontent.com/media/nicdieugenio/'
              f'YBCO_ACE_MACE/{SOURCE_COMMIT}/YBCO_MACE/ybco_mace.xyz')
SOURCE_SHA256 = 'f5fe4ce2a8e477ad5fc98eaf4dc98fe75cbd28c0f60ba0e39ce769c14679a73b'
SOURCE_SIZE = 237748083
DEFAULT_ROOT = Path(__file__).resolve().parents[2] / 'dataset' / 'ybco'
SCHEMA = 'ybco_unrelaxed_fixed_reference_v1'
HOST_LABEL = 'Ba2YCu3O7_vol0_strain0_jiggle0'
PAPER_URL = 'https://arxiv.org/html/2511.22592v1'
# Table 4, DFT column, per O atom, NON-spin-polarized O2. The public
# training XYZ has no O2 reference calculation; retain the published precision.
OXYGEN_MU_EV = -433.99
LABEL_RE = re.compile(r'^(?P<family>.+)_vol(?P<volume>-?[\d.]+)'
                      r'_strain(?P<strain>\d+)_jiggle(?P<jiggle>\d+)$')
ELEMENTS = ('Y', 'Ba', 'Cu', 'O')


def file_sha256(path):
    result = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def verify_source(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'{path}: first run with --download --prepare-only')
    if path.stat().st_size != SOURCE_SIZE or file_sha256(path) != SOURCE_SHA256:
        raise ValueError('YBCO source does not match the pinned release (size/SHA256). '
                         'A Git LFS pointer is not the XYZ data file.')
    return path


def download_source(root=DEFAULT_ROOT):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    target = root / 'ybco_mace.xyz'
    if target.exists():
        return verify_source(target)
    temporary = root / 'ybco_mace.xyz.part'
    print(f'Downloading {SOURCE_SIZE / 1e6:.1f} MB from the author repository...', flush=True)
    request = urllib.request.Request(SOURCE_URL, headers={'User-Agent': 'HERA-YBCO/1'})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open('wb') as out:
        while chunk := response.read(1024 * 1024):
            out.write(chunk)
    verify_source(temporary)
    temporary.replace(target)
    return target


def parse_family(label):
    match = LABEL_RE.fullmatch(label)
    if not match:
        return None
    family = match['family']
    if not re.fullmatch(r'(Y|Ba|Cu[12]?|O[1-4])_vac|'
                        r'(Y|Ba|Cu|O)_int_[1-7]|'
                        r'(Y|Ba|Cu)_(Y|Ba|Cu[12])_antisite', family):
        return None
    return family


def count_delta(atoms, host):
    current = Counter(atoms.get_chemical_symbols())
    reference = Counter(host.get_chemical_symbols())
    return {symbol: current[symbol] - reference[symbol] for symbol in ELEMENTS}


def expected_delta(family):
    delta = dict.fromkeys(ELEMENTS, 0)
    if family.endswith('_vac'):
        delta[re.sub(r'\d+$', '', family.split('_')[0])] = -1
    elif '_int_' in family:
        delta[family.split('_')[0]] = 1
    else:
        added, removed, _ = family.split('_')
        delta[added] += 1
        delta[re.sub(r'\d+$', '', removed)] -= 1
    return delta


def formation_energy(energy, host_energy, delta, chemical_potentials):
    """delta = N(defect) - N(host); one vacancy adds +mu to Edef-Ehost."""
    return float(energy - host_energy - sum(delta[s] * chemical_potentials[s] for s in ELEMENTS))


def reference_mapping(atoms, host, family, tolerance=1.0):
    """Species-wise periodic assignment at the reference geometry.

    Each family retains source atom order. Matching is done once, then every
    member is checked for order and displacement consistency before use.
    Distances are measured in the fixed host cell to remove homogeneous strain.
    """
    from scipy.optimize import linear_sum_assignment
    if count_delta(atoms, host) != expected_delta(family):
        raise ValueError(f'{family}: source composition disagrees with its defect label')
    if not np.allclose(host.cell.array, np.diag(host.cell.lengths()), atol=1e-7):
        raise ValueError('Expected the pinned orthorhombic host reference')
    actual_symbols = np.array(atoms.get_chemical_symbols())
    host_symbols = np.array(host.get_chemical_symbols())
    actual_frac = atoms.get_scaled_positions(wrap=True)
    host_frac = host.get_scaled_positions(wrap=True)
    was = atoms.numbers.copy()
    defect = np.zeros(len(atoms), dtype=bool)
    matched_actual, matched_host = set(), set()
    max_distance = 0.0
    for symbol in ELEMENTS:
        ai = np.flatnonzero(actual_symbols == symbol)
        hi = np.flatnonzero(host_symbols == symbol)
        df = actual_frac[ai, None, :] - host_frac[None, hi, :]
        distance = np.linalg.norm((df - np.rint(df)) @ host.cell.array, axis=-1)
        rows, cols = linear_sum_assignment(distance)
        max_distance = max(max_distance, float(distance[rows, cols].max(initial=0)))
        matched_actual.update(ai[rows].tolist())
        matched_host.update(hi[cols].tolist())
    if max_distance > tolerance:
        raise ValueError(f'{family}: ambiguous reference mapping ({max_distance:.3f} A)')
    missing = sorted(set(range(len(host))) - matched_host)
    extra = sorted(set(range(len(atoms))) - matched_actual)
    if family.endswith('_vac'):
        if len(missing) != 1 or extra:
            raise ValueError(f'{family}: expected one missing site')
    elif '_int_' in family:
        if missing or len(extra) != 1:
            raise ValueError(f'{family}: expected one inserted atom')
        was[extra[0]] = 0
        defect[extra[0]] = True
    else:
        if len(missing) != 1 or len(extra) != 1:
            raise ValueError(f'{family}: expected one substitution')
        df = actual_frac[extra[0]] - host_frac[missing[0]]
        if np.linalg.norm((df - np.rint(df)) @ host.cell.array) > tolerance:
            raise ValueError(f'{family}: substituted atom is not near the missing host site')
        was[extra[0]] = int(host.numbers[missing[0]])
        defect[extra[0]] = True
        missing = []
    return {'was': was.tolist(), 'defect': defect.tolist(),
            'vacancy_frac': host_frac[missing].tolist(),
            'vacancy_z': host.numbers[missing].tolist(),
            'max_mapping_distance_angstrom': max_distance}


def prepare(root=DEFAULT_ROOT, source=None):
    """Audit the release; export real atoms plus reference annotations to ASE DB."""
    from ase.db import connect
    from ase.io import iread
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    source = verify_source(source or root / 'ybco_mace.xyz')
    print('Reading and auditing the pinned XYZ with ASE...', flush=True)
    frames = list(iread(source, format='extxyz'))
    labelled = {}
    for index, atoms in enumerate(frames):
        labelled.setdefault(atoms.info.get('label', ''), []).append(index)

    def unique_reference(label, symbols):
        indices = labelled.get(label, [])
        if len(indices) != 1:
            raise ValueError(f'Missing or ambiguous reference: {label}')
        index = indices[0]
        atoms = frames[index]
        if set(atoms.get_chemical_symbols()) != set(symbols):
            raise ValueError(f'Reference composition mismatch: {label}')
        return index, atoms

    host_index, host = unique_reference(HOST_LABEL, ELEMENTS)
    host_energy = float(host.info['REF_energy'])
    references = {'host': {'frame': host_index, 'label': HOST_LABEL,
                            'energy_ev': host_energy, 'natoms': len(host),
                            'max_force_ev_per_angstrom': float(np.linalg.norm(host.arrays['REF_forces'], axis=1).max())}}
    mu = {'O': OXYGEN_MU_EV}
    for symbol in ('Y', 'Ba', 'Cu'):
        index, atoms = unique_reference(f'{symbol}_vol0_strain0_jiggle0', [symbol])
        mu[symbol] = float(atoms.info['REF_energy']) / len(atoms)
        references[symbol] = {'frame': index, 'label': atoms.info['label'],
                              'energy_ev': float(atoms.info['REF_energy']), 'natoms': len(atoms)}
    references['O'] = {'source': PAPER_URL, 'table': 4, 'column': 'DFT',
                       'mu_ev_per_atom': OXYGEN_MU_EV, 'rounding_ev': 0.01,
                       'reference': 'non-spin-polarized O2; not isolated atomic E0'}
    families = {}
    for index, atoms in enumerate(frames):
        family = parse_family(atoms.info.get('label', ''))
        if family:
            families.setdefault(family, []).append(index)
    maps, exemplars, rejected = {}, {}, []
    for family, indices in families.items():
        def exemplar_order(i):
            m = LABEL_RE.fullmatch(frames[i].info['label'])
            return int(m['jiggle']), int(m['strain']), abs(float(m['volume'])), i
        index = min(indices, key=exemplar_order)
        try:
            maps[family] = reference_mapping(frames[index], host, family)
        except ValueError as exc:
            rejected.extend({'source_frame': i, 'family': family, 'reason': str(exc)} for i in indices)
            continue
        exemplars[family] = index

    database = root / f'{SCHEMA}.db'
    # Build at a separate path; an existing complete preparation stays readable
    # until the new database and checksum manifest have been completed.
    import tempfile
    with tempfile.NamedTemporaryFile(dir=root, suffix='.db', delete=False) as handle:
        temporary = Path(handle.name)
    rows = []
    try:
        with connect(temporary) as db:
            for family, indices in families.items():
                if family not in maps:
                    continue
                exemplar = frames[exemplars[family]]
                mapping = maps[family]
                for index in indices:
                    atoms = frames[index]
                    if not np.array_equal(atoms.numbers, exemplar.numbers):
                        rejected.append({'source_frame': index, 'family': family,
                                         'reason': 'family atom order changed'})
                        continue
                    df = atoms.get_scaled_positions() - exemplar.get_scaled_positions()
                    displacement = np.linalg.norm((df - np.rint(df)) @ host.cell.array, axis=1).max()
                    if displacement > 1.0:
                        rejected.append({'source_frame': index, 'family': family,
                                         'reason': f'family displacement {displacement:.3f} A exceeds 1.0 A'})
                        continue
                    delta = count_delta(atoms, host)
                    if delta != expected_delta(family):
                        raise ValueError(f'Frame {index}: unexpected composition')
                    energy = float(atoms.info['REF_energy'])
                    if not (np.isfinite(energy) and np.isfinite(atoms.positions).all()
                            and np.isfinite(atoms.cell.array).all() and atoms.get_volume() > 0
                            and atoms.pbc.all()):
                        raise ValueError(f'Invalid energy, coordinates or cell: frame {index}')
                    target = formation_energy(energy, host_energy, delta, mu)
                    sid = f'frame{index:04d}'
                    dbid = db.write(atoms, data={'mapping': mapping}, source_id=sid,
                                    source_frame=index, family=family, source_label=atoms.info['label'],
                                    formation_energy=target)
                    rows.append({'source_id': sid, 'source_frame': index, 'db_id': dbid,
                                 'family': family, 'label': atoms.info['label'], 'natoms': len(atoms),
                                 'energy_ev': energy, 'target_ev': target,
                                 'max_force_ev_per_angstrom': float(np.linalg.norm(atoms.arrays['REF_forces'], axis=1).max())})
        temporary.replace(database)
    finally:
        if temporary.exists():
            temporary.unlink()
    rows.sort(key=lambda r: r['source_frame'])
    manifest = {'schema': SCHEMA, 'source_url': SOURCE_URL, 'source_commit': SOURCE_COMMIT,
                'source_sha256': SOURCE_SHA256, 'source_path': str(source.resolve()),
                'database_sha256': file_sha256(database), 'source_frames': len(frames),
                'selected_frames': len(rows), 'excluded_frames': len(frames) - len(rows),
                'selection': 'labelled single vacancies, interstitials and antisites only',
                'target': 'unrelaxed_fixed_reference_configuration_formation_energy',
                'units': 'eV per configuration (one point defect)',
                'formula': 'E_configuration - E_host - sum((N_configuration-N_host)*mu)',
                'limitations': ['Includes strain and displacement energy relative to a fixed host.',
                                'Host reference itself is not certified force-converged.',
                                'O chemical potential is rounded from paper Table 4.',
                                'No charged-defect or thermodynamic phase-stability corrections.',
                                'Not relaxed defect formation energies or defect interaction energies.'],
                'references': references, 'chemical_potentials_ev': mu,
                'candidate_frames': sum(len(indices) for indices in families.values()),
                'families': dict(Counter(r['family'] for r in rows)),
                'excluded_mapping_records': rejected,
                'mapping_exemplar_frames': exemplars, 'records': rows}
    manifest['mapping_policy'] = {
        'method': 'species-wise periodic assignment in fixed host cell',
        'max_assignment_distance_angstrom': 1.0,
        'max_family_displacement_angstrom': 1.0,
        'selection_uses_targets': False,
        'unmapped_families_or_frames': 'excluded from every model equally',
    }
    manifest['graph_mapping_sha256'] = hashlib.sha256(
        json.dumps({'maps': maps, 'policy': manifest['mapping_policy']}, sort_keys=True).encode()
    ).hexdigest()
    manifest_path = root / f'{SCHEMA}.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(f'Prepared {len(rows)}/{len(frames)} frames, {len(manifest["families"])} defect families: {manifest_path}', flush=True)
    return manifest


def read_prepared(root=DEFAULT_ROOT):
    root = Path(root)
    path = root / f'{SCHEMA}.json'
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if manifest['schema'] != SCHEMA or manifest['source_sha256'] != SOURCE_SHA256:
        raise ValueError('Unsupported YBCO preparation')
    if file_sha256(root / f'{SCHEMA}.db') != manifest['database_sha256']:
        raise ValueError('Prepared YBCO database changed; regenerate preparation')
    return manifest


def grouped_indices(records, seed):
    from sklearn.model_selection import train_test_split
    families = sorted({r['family'] for r in records})
    if len(families) < 5:
        raise ValueError('Need at least five defect families for grouped 60/20/20 splits')
    train, held = train_test_split(families, test_size=0.4, random_state=seed)
    val, test = train_test_split(held, test_size=0.5, random_state=seed)
    return {part: [i for i, r in enumerate(records) if r['family'] in set(groups)]
            for part, groups in zip(('train', 'val', 'test'), (train, val, test))}


def load_structures(root, records, mode, local_cutoff=0):
    from ase.db import connect
    from pymatgen.core import Structure
    from pymatgen.core.periodic_table import DummySpecies
    from .structure_utils import mark_hetero_region_if_needed
    result = []
    with connect(Path(root) / f'{SCHEMA}.db') as db:
        for record in records:
            row = db.get(record['db_id'])
            if (row.source_id != record['source_id'] or row.family != record['family']
                    or row.source_label != record['label']
                    or not np.isclose(row.formation_energy, record['target_ev'], rtol=0, atol=1e-9)):
                raise ValueError('YBCO manifest and prepared database disagree')
            atoms = row.toatoms()
            mapping = row.data['mapping']
            structure = Structure(atoms.cell.array, atoms.numbers, atoms.positions,
                                  coords_are_cartesian=True,
                                  site_properties={'was': mapping['was'], 'type': mapping['defect']})
            if mode != 'full':
                for frac, z in zip(mapping['vacancy_frac'], mapping['vacancy_z']):
                    structure.append(DummySpecies(), frac, properties={'was': z, 'type': True})
            if mode in ('hetero', 'hetero_was'):
                structure = mark_hetero_region_if_needed(structure, 'alignn_hetero', local_cutoff)
            structure.source_id = record['source_id']
            structure.source_name = record['label']
            structure.source_path = str(Path(root) / f'{SCHEMA}.db')
            structure.material = 'YBa2Cu3O7'
            structure.defect_family = record['family']
            result.append(structure)
    return result
