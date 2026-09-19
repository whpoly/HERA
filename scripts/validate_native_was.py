"""Validate true native WAS on all local CIFs without training a model."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re
import warnings

import numpy as np
from pymatgen.core import Structure
from pymatgen.core.periodic_table import DummySpecies

from ..data.converters import AtomFeaturesExtractor
from ..data.datasets import init_elem_embedding, tag_structure_source
from ..data.native_was import NATIVE_REFERENCE_VERSION, parse_native_defect
from ..data.structure_utils import convert_to_sparse_native


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-list', type=Path,
                        default=ROOT.parent / 'dataset/Dataset_1/Dataset_1/A_rich/Neutral/id_prop_A_rich.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/native_was_reference_v1_validation.json')
    args = parser.parse_args()
    init_elem_embedding(ROOT / 'atom_init.json')
    from ..data.datasets import elem_embedding
    extractor = AtomFeaturesExtractor('was_species', 'alignn_hetero_was', NATIVE_REFERENCE_VERSION)
    counts = Counter()
    wrong_legacy = Counter()
    examples = {}
    nodes = 0
    with args.data_list.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.reader(stream))
    for number, row in enumerate(rows, start=1):
        filename = row[0]
        kind, current_z, previous_z = parse_native_defect(filename)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Issues encountered while parsing CIF:.*')
            raw = Structure.from_file(args.data_list.parent / filename)
        tag_structure_source(raw, args.data_list.parent / filename, filename)
        graphs = []
        for mode in ('hetero', 'attention'):
            graphs.append(convert_to_sparse_native(
                raw, 'vacancy' if kind == 'vacancy' else 'others', 1, f'alignn_{mode}',
                None, skip_was=True, local_cutoff=0, native_preprocessing=NATIVE_REFERENCE_VERSION))
        hetero, attention = graphs
        assert len(hetero) == len(raw) + int(kind == 'vacancy'), filename
        assert hetero.species[:len(raw)] == raw.species, filename
        np.testing.assert_array_equal(hetero.frac_coords[:len(raw)], raw.frac_coords)
        assert hetero.site_properties['type'] == attention.site_properties['type'], filename
        assert hetero.site_properties['was'] == attention.site_properties['was'], filename
        features = extractor.convert(hetero)
        np.testing.assert_array_equal(features, extractor.convert(attention))
        defect_indices = [i for i, site in enumerate(hetero) if site.properties['pool_type']]
        assert len(defect_indices) == 1, filename
        index = defect_indices[0]
        defect = hetero[index]
        assert defect.properties['was'] == previous_z, filename
        assert (0 if isinstance(defect.specie, DummySpecies) else defect.specie.Z) == current_z, filename
        if kind != 'vacancy':
            # Independent check: highest original CIF index, not sorted site order.
            expected = max(range(len(raw)), key=lambda i: int(re.search(r'\d+$', raw[i].label)[0]))
            assert index == expected, filename
            if expected != len(raw) - 1:
                wrong_legacy[kind] += 1
        else:
            np.testing.assert_array_equal(defect.frac_coords, [.5, .5, .5])
        np.testing.assert_array_equal(features[index, 92:], elem_embedding[previous_z] if previous_z else np.zeros(92))
        normal = [i for i in range(len(hetero)) if i != index]
        np.testing.assert_array_equal(features[normal, :92], features[normal, 92:])
        assert not np.array_equal(features[index, :92], features[index, 92:]), filename
        counts[kind] += 1
        nodes += len(hetero)
        examples.setdefault(kind, {'source': filename, 'defect_index': index,
            'defect_label': defect.label, 'current_z': current_z, 'was_z': previous_z,
            'raw_atoms': len(raw), 'graph_nodes': len(hetero)})
        if number % 500 == 0:
            print(f'Validated {number}/{len(rows)} structures', flush=True)
    report = {'native_preprocessing': NATIVE_REFERENCE_VERSION, 'training_performed': False,
              'structures': len(rows), 'defect_counts': dict(counts), 'graph_nodes': nodes,
              'was_coverage': 1.0, 'paired_attention_hetero_structures_identical': True,
              'legacy_nonvacancy_wrong_defect_site_counts': dict(wrong_legacy),
              'vacancy_graphs_with_restored_real_atom': counts['vacancy'], 'examples': examples,
              'scope': 'Input labels/geometry/features only; not a trained test MAE.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    print(f'Saved: {args.output}', flush=True)


if __name__ == '__main__':
    main()
