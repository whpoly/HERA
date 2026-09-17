"""Add deterministic single-site embedding interventions to completed results."""
import argparse
import json
import warnings
from pathlib import Path
import numpy as np
import torch
from .checkpoint_comparison import ROOT, init_elem_embedding, load_models, prepare_case, json_write


def enrich(directory, device='cuda:0'):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    cohort = manifest['protocol']['cohort']
    trainers, provenance = load_models(list(manifest['checkpoints']), device, cohort, manifest['cases'][0]['material'])
    for name, record in provenance.items():
        if record['sha256'] != manifest['checkpoints'][name]['sha256']:
            raise ValueError('Checkpoint changed since explanation run')
    records = []
    for case in manifest['cases']:
        wrappers, _, _ = prepare_case(case, trainers, device, manifest['protocol']['mask_space'])
        for name, wrapper in wrappers.items():
            path = directory / 'cases' / case['key'] / f'{name}.json'
            record = json.loads(path.read_text(encoding='utf-8'))
            if 'single_site_ablation' not in record:
                with torch.no_grad():
                    base = float(wrapper.physical(wrapper.x).item())
                    if abs(base-record['prediction']) > 1e-5:
                        raise ValueError('Prediction changed during replay')
                    delta = []
                    for i in range(len(wrapper.x)):
                        masked = wrapper.x.clone()
                        masked[i] = 0
                        delta.append(1000*(float(wrapper.physical(masked).item())-base))
                record['single_site_ablation'] = {
                    'method': 'Zero one initial node embedding at a time; keep all geometry and edges',
                    'signed_delta_mev': delta,
                    'caveat': 'Conditional model sensitivity; not physical atom removal or an additive energy decomposition',
                }
                json_write(path, record)
                print(f'ENRICHED {directory.name} {case["key"]} {name} maximum_abs_delta={np.max(np.abs(delta)):.3f} meV', flush=True)
            records.append(record)
    json_write(directory / 'results.json', records)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', nargs='+')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    torch.set_num_threads(2)
    warnings.filterwarnings('ignore', message='.*Issues encountered while parsing CIF.*')
    init_elem_embedding(ROOT / 'atom_init.json')
    for directory in args.directories:
        enrich(directory, args.device)
