"""Train HERA's ALIGNN modes from scratch on held-out Sr–O Schottky arrangements."""
import argparse
import csv
import hashlib
import json
import tempfile
import urllib.request
from collections import Counter
from itertools import product
from pathlib import Path

import numpy as np
import torch
from ase.io import iread
from pymatgen.core import Structure
from pymatgen.core.periodic_table import DummySpecies
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

from ..config.defaults import ALIGNN_MODES, get_config
from ..data.datasets import dataset_index_for_mode, init_elem_embedding, representation_for_mode
from ..data.structure_utils import preserve_pool_type
from ..main import set_seed, train_single_mode, write_test_predictions
from ..training.results import ensure_compact_run_config
from ..training.trainer import MEGNetTrainer, load_trusted_checkpoint
from .alignn_benchmark_support import audit_checkpoint, fixed_split, persist_contract

ROOT = Path(__file__).resolve().parents[1]
SOURCE_URL = (
    'https://raw.githubusercontent.com/nickhine/lr_mlip_benchmark/'
    'f1cb07f513d08d63b53f2d94ca0baf29429c65e1/STO_SrO_Schottky_DFT/'
    'datasets/STO_SrO_vac_train.xyz'
)
SOURCE_SHA256 = '2fab63e0da0df49788c52c261e77b9f564e40c9774eb4588e89bd40e227212a2'
FULL = 'Schottky_relax_full'
SINGLE_POINT = 'Schottky_relax_FM_DFT_SP'
MODES = tuple(mode for mode in ALIGNN_MODES if mode not in ('hypergraph', 'hypergraph_was'))
DATASET = 'srtio3_schottky_3510'
SCHEMA = 'srtio3_3510_distance_split_v1'
DEFAULT_RUN = ROOT / 'logs' / 'srtio3_3510_distance_split_fp32_v1'
SPECIES = {'Sr': 149, 'Ti': 150, 'O': 449}


def source_file(root):
    root.mkdir(parents=True, exist_ok=True)
    target = root / 'STO_SrO_vac_train.xyz'
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest() != SOURCE_SHA256:
            raise ValueError(f'Unexpected source content in {target}')
        return target
    with tempfile.NamedTemporaryFile(dir=root, delete=False) as out:
        temporary = Path(out.name)
    try:
        with urllib.request.urlopen(SOURCE_URL, timeout=90) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != SOURCE_SHA256:
            raise ValueError('Source SHA256 does not match the pinned authors’ XYZ')
        temporary.write_bytes(data)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def locate_vacancies(atoms):
    """Register an ideal periodic perovskite on the intact Ti sublattice."""
    repeats = np.array([3, 5, 10])
    lengths = atoms.cell.lengths()
    if not np.allclose(atoms.cell.angles(), 90.0) or Counter(atoms.get_chemical_symbols()) != SPECIES:
        raise ValueError('Expected the orthorhombic Sr149Ti150O449 supercell')
    symbols = np.asarray(atoms.get_chemical_symbols())
    frac = atoms.get_scaled_positions(wrap=True)
    harmonic = np.exp(2j * np.pi * frac[symbols == 'Ti'] * repeats).mean(axis=0)
    if np.min(np.abs(harmonic)) < 0.9:
        raise ValueError('Ti positions no longer register to the parent lattice')
    origin = np.angle(harmonic) / (2 * np.pi)
    cells = np.array(list(product(*(range(n) for n in repeats))))
    vacancies = {}
    for element, basis in (('Ti', [[0, 0, 0]]), ('Sr', [[.5, .5, .5]]),
                           ('O', [[.5, 0, 0], [0, .5, 0], [0, 0, .5]])):
        grid = np.vstack([((cells + origin + site) / repeats) % 1 for site in basis])
        cart = (grid * lengths) % lengths
        dist, ids = cKDTree(cart, boxsize=lengths).query((frac[symbols == element] * lengths) % lengths)
        if np.max(dist) >= 1.0 or len(np.unique(ids)) != len(ids):
            raise ValueError(f'Ambiguous periodic site assignment for {element}')
        missing = np.setdiff1d(np.arange(len(grid)), ids)
        if len(missing) != int(element != 'Ti'):
            raise ValueError(f'Unexpected count of missing {element} sites')
        if len(missing):
            vacancies[element] = grid[missing[0]].tolist()
    diff = np.subtract(vacancies['Sr'], vacancies['O'])
    distance = np.linalg.norm((diff - np.rint(diff)) @ atoms.cell.array)
    if abs(distance - float(atoms.info['SrO_dist'])) > 1e-4:
        raise ValueError('Mapped vacancies disagree with the published separation')
    return vacancies


def read_cohort(path):
    records, atoms_by_id = [], {}
    for i, atoms in enumerate(iread(path, index=':', format='extxyz')):
        tag = atoms.info.get('data_type')
        if atoms.info.get('supercell') != 3510 or tag not in (FULL, SINGLE_POINT):
            continue
        if int(atoms.info['vac_pair_count']) != 1:
            raise ValueError('Expected exactly one Sr–O vacancy pair')
        group = int(atoms.info['frame'])
        source_id = f'xyz_{i}'
        energy = float(atoms.get_potential_energy())
        if not np.isfinite(energy):
            raise ValueError(f'Non-finite energy for {source_id}')
        records.append(dict(source_id=source_id, index=i, frame=group, kind=tag,
                            distance=float(atoms.info['SrO_dist']), energy_ev_cell=energy,
                            vacancies=locate_vacancies(atoms)))
        atoms_by_id[source_id] = atoms
    full = {r['frame']: r for r in records if r['kind'] == FULL}
    if set(full) != set(range(1, 22)) or len(records) != 42:
        raise ValueError('Expected 21 complete distance groups in the pinned release')
    for frame, relaxed in full.items():
        siblings = [r for r in records if r['frame'] == frame]
        if ({r['kind'] for r in siblings} != {FULL, SINGLE_POINT}
                or len(siblings) != 2 or any(abs(r['distance'] - relaxed['distance']) > 1e-5 for r in siblings)):
            raise ValueError(f'Mismatched geometry preparations for frame {frame}')
    return records, atoms_by_id


def make_partition(records, seed):
    """Keep both preparations together; span the separation range in every partition."""
    relaxed = sorted((r for r in records if r['kind'] == FULL), key=lambda r: r['distance'])
    anchor = relaxed[0]
    rng = np.random.default_rng(seed)
    blocks = np.array([r['frame'] for r in relaxed[1:]]).reshape(5, 4)
    no_validation = int(rng.integers(len(blocks)))
    train, val, test = {anchor['frame']}, set(), set()
    for block_id, block in enumerate(blocks):
        selected = rng.permutation(block).tolist()
        test.add(selected.pop())
        if block_id != no_validation:
            val.add(selected.pop())
        train.update(selected)
    groups = dict(train=sorted(train), val=sorted(val), test=sorted(test))
    ids = {p: [r['source_id'] for r in records if r['frame'] in selected and
               (p == 'train' or r['kind'] == FULL)] for p, selected in groups.items()}
    assert [len(ids[p]) for p in ('train', 'val', 'test')] == [24, 4, 5]
    return groups, ids, anchor


def build_structures(records, atoms_by_id, source, mode):
    structs = []
    representation = representation_for_mode(mode)
    for r in records:
        a = atoms_by_id[r['source_id']]
        st = Structure(a.cell.array, a.numbers, a.positions, coords_are_cartesian=True,
                       site_properties={'was': a.numbers.tolist(), 'type': [False] * len(a)})
        if representation != 'full':
            for element, z in (('Sr', 38), ('O', 8)):
                st.append(DummySpecies(), r['vacancies'][element], properties={'was': z, 'type': True})
        if representation == 'hetero':
            st = preserve_pool_type(st)  # r=0: the region and the actual-defect mask coincide
        st.source_id = r['source_id']
        st.source_name = f"frame_{r['frame']}_{r['kind']}"
        st.source_path = str(source)
        st.material = 'SrTiO3'
        st.defect_family = 'V_Sr+V_O'
        structs.append(st)
    return structs


def splitter(ids):
    return fixed_split(ids, SCHEMA, 'held-out Sr–O distances')


def scores(predictions):
    truth = np.array([float(r['target']) for r in predictions]) * 2.0
    predicted = np.array([float(r['prediction']) for r in predictions]) * 2.0
    i, j = np.triu_indices(len(truth), 1)
    td, pd = truth[i] - truth[j], predicted[i] - predicted[j]
    comparable = np.abs(td) > 1e-8
    concordance = np.sign(td[comparable]) * np.sign(pd[comparable])
    correlation = float(spearmanr(truth, predicted).statistic) if np.ptp(predicted) else None
    return dict(test_mae_ev_per_defect=float(np.mean(np.abs(truth - predicted)) / 2),
                pairwise_mae_ev_per_cell=float(np.mean(np.abs(td - pd))),
                pairwise_correct=int((concordance > 0).sum()), pairwise_tied=int((concordance == 0).sum()),
                pairwise_total=int(comparable.sum()), spearman=correlation)


def write_comparison(run_dir, contract):
    """Join predictions for the same holdout; keep all quantities in eV per cell here."""
    records = {r['source_id']: r for r in contract['records']}
    ids = contract['membership']['test']
    train_mean = np.mean([records[i]['target_ev_per_defect'] for i in contract['membership']['train']])
    rows = [dict(source_id=i, frame=records[i]['frame'], distance_angstrom=records[i]['distance'],
                 dft_ev_cell=2 * records[i]['target_ev_per_defect'], train_mean_ev_cell=2 * train_mean)
            for i in ids]
    for mode in MODES:
        directory = run_dir / mode
        if not (directory / 'metrics.json').is_file():
            continue
        with (directory / f"seed{contract['seed']}_test_predictions.csv").open(newline='') as handle:
            predicted = list(csv.DictReader(handle))
        if [r['source_id'] for r in predicted] != ids:
            raise ValueError(f'{mode}: test source IDs differ from the benchmark')
        for row, prediction in zip(rows, predicted):
            if not np.isclose(2 * float(prediction['target']), row['dft_ev_cell'], atol=1e-7):
                raise ValueError(f'{mode}: target differs for {row["source_id"]}')
            row[mode + '_ev_cell'] = 2 * float(prediction['prediction'])
    with (run_dir / 'heldout_comparison.csv').open('w', newline='', encoding='utf8') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def evaluate_fp32(args):
    """Evaluate the unchanged validation-selected weights with full-precision inference."""
    contract = json.loads((args.run_dir / 'data_contract.json').read_text(encoding='utf8'))
    source = source_file(args.data_root)
    records, atoms = read_cohort(source)
    by_id = {r['source_id']: r for r in records}
    test_ids = contract['membership']['test']
    test = [by_id[i] for i in test_ids]
    anchor = by_id[contract['anchor']]['energy_ev_cell']
    targets = torch.tensor([(r['energy_ev_cell'] - anchor) / 2 for r in test], dtype=torch.float32)
    seed = contract['seed']
    init_elem_embedding(ROOT / 'atom_init.json')
    output = args.run_dir / 'fp32'
    for mode in dict.fromkeys(args.mode):
        checkpoint = load_trusted_checkpoint(args.run_dir / mode / f'seed{seed}_best_checkpoint.pth',
                                             map_location='cpu')
        expected = {'model_name': 'alignn', 'dataset_name': DATASET, 'mode': mode}
        if (any(checkpoint[k] != v for k, v in expected.items())
                or [s['source_id'] for s in checkpoint['split']['test_sources']] != test_ids
                or checkpoint['config']['benchmark_source_sha256'] != SOURCE_SHA256):
            raise ValueError(f'{mode}: inconsistent checkpoint or test membership')
        set_seed(seed)
        trainer = MEGNetTrainer(checkpoint['config'], args.device, seed)
        trainer.use_amp = False
        trainer.scaler.load_state_dict(checkpoint['scaler'])
        structures = build_structures(test, atoms, source, mode)
        _, predicted = trainer.predict_structures(structures, targets, checkpoint['model'],
                                                  return_predictions=True)
        directory = output / mode
        file = write_test_predictions(str(directory), seed, structures, targets, predicted)
        with open(file, newline='') as handle:
            metric = dict(mode=mode, seed=seed, **scores(list(csv.DictReader(handle))))
        metric.update(predicted_span_ev_per_cell=float(np.ptp(predicted) * 2),
                      best_epoch=checkpoint['best_epoch'], epochs_completed=checkpoint['epochs_completed'],
                      best_val_mae_ev_per_defect=checkpoint['best_val_mae'])
        (directory / 'metrics.json').write_text(json.dumps(metric, indent=2, allow_nan=False) + '\n', encoding='utf8')
        print(metric, flush=True)
    metrics = [json.loads((output / m / 'metrics.json').read_text(encoding='utf8')) for m in MODES
               if (output / m / 'metrics.json').is_file()]
    with (output / 'summary.csv').open('w', newline='', encoding='utf8') as handle:
        writer = csv.DictWriter(handle, fieldnames=metrics[0])
        writer.writeheader()
        writer.writerows(metrics)
    write_comparison(output, contract)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ROOT.parent / 'dataset' / 'srtio3_schottky')
    parser.add_argument('--run-dir', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--mode', nargs='+', choices=MODES, default=list(MODES))
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--comparison-only', action='store_true', help='Join completed predictions without training')
    parser.add_argument('--score-fp32', action='store_true', help='Rescore this run’s selected checkpoints without training')
    parser.add_argument('--resume', action='store_true', help='Skip completed modes; restart incomplete modes with fresh weights')
    args = parser.parse_args()
    if args.score_fp32:
        evaluate_fp32(args)
        return
    if args.comparison_only:
        contract = json.loads((args.run_dir / 'data_contract.json').read_text(encoding='utf8'))
        write_comparison(args.run_dir, contract)
        return
    if args.epochs < 1 or args.patience < 1:
        parser.error('epochs and patience must be positive')
    source = source_file(args.data_root)
    records, atoms = read_cohort(source)
    groups, ids, anchor = make_partition(records, args.seed)
    for r in records:
        # Subtract the large FHI-aims total energies before casting to float32.
        r['target_ev_per_defect'] = (r['energy_ev_cell'] - anchor['energy_ev_cell']) / 2.0
    used = set().union(*ids.values())
    records = [r for r in records if r['source_id'] in used]
    args.run_dir.mkdir(parents=True, exist_ok=True)
    contract = dict(schema=SCHEMA, source_url=SOURCE_URL, source_sha256=SOURCE_SHA256,
                    target='(DFT total energy minus nearest fully-relaxed training anchor) / 2',
                    units='eV per missing site', anchor=anchor['source_id'], seed=args.seed,
                    groups=groups, membership=ids, records=records)
    persist_contract(args.run_dir / 'data_contract.json', contract)
    print('Distance groups:', groups, flush=True)
    print('Samples:', {p: len(v) for p, v in ids.items()}, '; anchor:', anchor['source_id'], flush=True)
    if args.prepare_only:
        return
    init_elem_embedding(ROOT / 'atom_init.json')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        parser.error('CUDA unavailable; select --device cpu')
    targets = torch.tensor([r['target_ev_per_defect'] for r in records], dtype=torch.float32)
    for mode in dict.fromkeys(args.mode):
        config = get_config('alignn', '2dmd_low', mode)
        config['model'].update(train_batch_size=1, test_batch_size=1, local_radius=0.0)
        config['optim'].update(amp=False, early_stopping_patience=args.patience)
        config.update(target_name='relative_pair_energy_per_defect', target_units='eV/defect',
                      benchmark_schema=SCHEMA, benchmark_seed=args.seed, benchmark_max_epochs=args.epochs,
                      benchmark_source_sha256=SOURCE_SHA256)
        directory = args.run_dir / mode
        directory.mkdir(parents=True, exist_ok=True)
        ensure_compact_run_config(directory, dict(layout='compact_v1', model_name='alignn',
                                                  dataset_name=DATASET, mode=mode, run_label=mode,
                                                  config=config))
        if not args.resume and any(directory.glob('*_history.csv')):
            raise ValueError(f'{directory} already contains history; use --resume or another run directory')
        structures = build_structures(records, atoms, source, mode)
        dataset = [None] * 4
        dataset[dataset_index_for_mode(mode)] = structures
        print(f'Starting ALIGNN {mode} from random initialization', flush=True)
        train_single_mode(mode, config, dataset, targets, [args.seed], args.epochs, args.device,
                          'alignn', DATASET, log_dir=str(directory), resume=args.resume,
                          split_iterator=splitter(ids))
        with (directory / f'seed{args.seed}_test_predictions.csv').open(newline='') as handle:
            predictions = list(csv.DictReader(handle))
        if [r['source_id'] for r in predictions] != ids['test']:
            raise ValueError(f'{mode}: test membership changed')
        audit = audit_checkpoint(directory, config, args.seed, ids)
        metric = dict(mode=mode, seed=args.seed, **scores(predictions), **audit,
                      predicted_span_ev_per_cell=2 * float(np.ptp([float(r['prediction']) for r in predictions])))
        (directory / 'metrics.json').write_text(json.dumps(metric, indent=2, allow_nan=False) + '\n', encoding='utf8')
        metrics = [json.loads((args.run_dir / m / 'metrics.json').read_text(encoding='utf8')) for m in MODES
                   if (args.run_dir / m / 'metrics.json').is_file()]
        with (args.run_dir / 'summary.csv').open('w', newline='', encoding='utf8') as handle:
            writer = csv.DictWriter(handle, fieldnames=metric)
            writer.writeheader()
            writer.writerows(metrics)
        print(metric, flush=True)
    write_comparison(args.run_dir, contract)


if __name__ == '__main__':
    main()

