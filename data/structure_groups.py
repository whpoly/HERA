"""Group repeated periodic structures without using their target energies."""
from collections import Counter, defaultdict
import itertools

from ase.geometry import find_mic
import numpy as np
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.io.ase import AseAtomsAdaptor


def anchor_distances(atoms):
    counts = Counter(atoms.numbers)
    number, count = min(counts.items(), key=lambda item: (item[1], item[0]))
    if count != 1:
        return None
    anchor = np.flatnonzero(atoms.numbers == number)[0]
    distances = find_mic(atoms.positions-atoms.positions[anchor], atoms.cell, atoms.pbc)[1]
    return np.concatenate([np.sort(distances[atoms.numbers == z]) for z in sorted(counts)])


def equivalent(left, right, tolerance):
    length_scale = ((left.volume+right.volume)/(2*len(left)))**(1/3)
    matcher = StructureMatcher(ltol=1e-6 if tolerance >= 1e-3 else 1e-8,
                               stol=tolerance/length_scale,
                               angle_tol=1e-4 if tolerance >= 1e-3 else 1e-6,
                               primitive_cell=False, scale=False, attempt_supercell=False)
    return bool(matcher.fit(left, right, symmetric=True, skip_structure_reduction=True))


def assign_structure_groups(records, atoms_by_id, tolerance=1e-3):
    """Return deterministic connected components; rejected rows are singletons."""
    parent = {r['source_id']: r['source_id'] for r in records}
    def find(sid):
        while parent[sid] != sid:
            parent[sid] = parent[parent[sid]]
            sid = parent[sid]
        return sid
    candidates = defaultdict(list)
    for row in records:
        if row['quality_reasons']:
            continue
        sid = row['source_id']
        atoms = atoms_by_id[sid]
        key = (row['material'], row['impurity'], tuple(sorted(Counter(atoms.numbers).items())))
        candidates[key].append((sid, AseAtomsAdaptor.get_structure(atoms), anchor_distances(atoms)))
    pair_count = 0
    for index, members in enumerate(candidates.values(), 1):
        for (a, sa, fa), (b, sb, fb) in itertools.combinations(members, 2):
            if abs(sa.volume-sb.volume) > 5e-6*max(sa.volume, sb.volume):
                continue
            if fa is not None and fb is not None and np.max(np.abs(fa-fb)) > 2*tolerance+1e-4:
                continue
            if equivalent(sa, sb, tolerance):
                ra, rb = find(a), find(b)
                parent[max(ra, rb)] = min(ra, rb)
                pair_count += 1
        if index % 500 == 0:
            print(f'IMP2D structure grouping: {index}/{len(candidates)} families, '
                  f'{pair_count} equivalent pairs', flush=True)
    return {sid: find(sid) for sid in parent}, pair_count
