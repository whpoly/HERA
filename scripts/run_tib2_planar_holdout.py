"""Scratch ALIGNN comparison on the complete Ti240B320 external holdout."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from ..config.defaults import get_config
from ..data import tib2_planar_holdout as source
from ..data.datasets import dataset_index_for_mode, init_elem_embedding
from ..main import train_single_mode
from ..training.results import ensure_compact_run_config
from .alignn_benchmark_support import audit_checkpoint, fixed_split, persist_contract
from .alignn_convergence import convergence_audit
from .run_sige_complexes import MODES, save_csv

LOGS = source.ROOT/'logs'/source.SCHEMA
RESULTS = source.ROOT/'results/tib2_planar_holdout_20261111'
PATIENCE = 100
GUARD = 1000


def references(records, ids):
    train = [records[s] for s in ids['train']]
    fractions = np.array([np.array(s.composition)/s.size for s in train])
    energies = np.array([s.energy/s.size for s in train])
    mu, _, rank, _ = np.linalg.lstsq(fractions, energies, rcond=None)
    if rank != 2:
        raise ValueError('Singular training energy reference')
    offsets = {s.sid: float(s.composition @ mu) for s in records.values()}
    labels = torch.tensor([(s.energy-offsets[s.sid])/s.size for s in records.values()], dtype=torch.float32)
    return mu.tolist(), offsets, labels


def write_protocol(records, ids, mu):
    RESULTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    contract = dict(schema=source.SCHEMA, source=source.SOURCE, source_sha256=source.DIGEST,
                    split=ids, models=list(MODES), scratch_seed=source.SEED,
                    labels='VASP snapshot energy; residual in eV per physical atom',
                    reference_fit='equal-weight least squares on training energy per atom vs Ti, B atom fractions',
                    mu_Ti_B_eV=mu, initialization='independent random initialization for every mode, same seed',
                    validation='MAE in eV per real atom over the entire internal validation set',
                    stopping=dict(patience=PATIENCE, relative_improvement_percent=.5, initial_guard=GUARD,
                                  guard_reached='repeat from same random initialization with double guard'),
                    model_preset='HERA ALIGNN och, no local cropping, 3 ALIGNN and 3 GCN blocks, fp32',
                    microbatch_size=1, gradient_accumulation=4, mode_selection='run all ten independently',
                    vacancy_inputs='480 parent B prisms of 240 Ti; missing sites recorded as B vacancies. '
                                  'Random-vacancy starting template atom IDs carry the sites through strain. '
                                  'For the external composition register the Ti prisms directly using only its geometry. '
                                  'Unregistered cutouts and stoichiometric bulk have no explicit vacancy markers. '
                                  'Heterogeneous defect readout averages real atoms in those frames.',
                    holdout_types=['Type-I','Type-II','Type-II'],
                    primary_pairs=[[source.TEST[0],t] for t in source.TEST[1:]],
                    order='strictly positive E(Type-II)-E(Type-I) for each of the two comparisons',
                    primary_ranking='mean absolute error over the two signed gap predictions in eV/cell',
                    tie_tolerance_eV=1.e-8)
    persist_contract(RESULTS/'data_contract.json', contract)
    inventory = []
    for s in records.values():
        inventory.append(dict(snapshot=s.sid,partition=next(p for p in ids if s.sid in ids[p]),
                              n_ti=s.n_ti,n_b=s.n_b,n_physical_atoms=s.size,explicit_source_cell=s.original_cell,
                              n_vacancy_nodes=len(s.vacancies),energy_eV_cell=s.energy,
                              site_registration=s.audit.get('site_registration',''),
                              ti_rms_A=s.audit.get('ti_rms_A',''),b_rms_A=s.audit.get('b_rms_A',''),
                              b_max_A=s.audit.get('b_max_A',''),missing_by_prism_row=s.audit.get('empty_by_row','')))
    save_csv(RESULTS/'partition_and_site_audit.csv',inventory)


def fit(mode, records, ids, mu, offsets, labels, device):
    dataset = [None]*4
    dataset[dataset_index_for_mode(mode)] = source.make_structures(records, mode)
    base = LOGS/mode
    seed = source.SEED
    budget = GUARD
    while True:
        folder = base if budget == GUARD else base/f'extended_{budget}'
        config = get_config('alignn', 'och', mode)
        config['model'].update(train_batch_size=1, test_batch_size=1, local_radius=0.)
        config['optim'].update(amp=False, grad_accum_steps=4, early_stopping_patience=PATIENCE)
        config.update(benchmark_schema=source.SCHEMA, source_sha256=source.DIGEST,
                      target_name='VASP_energy_minus_training_reference_per_real_atom', target_units='eV/atom',
                      mu_Ti_B_eV=mu, max_epochs=budget)
        ensure_compact_run_config(folder, dict(layout='compact_v1',model_name='alignn',
                                              dataset_name=source.SCHEMA,mode=mode,run_label=mode,config=config))
        train_single_mode(mode,config,dataset,labels,[seed],budget,device,'alignn',source.SCHEMA,
                          log_dir=str(folder),resume=True,split_iterator=fixed_split(ids,source.SCHEMA,'planar holdout'))
        convergence = convergence_audit(folder,seed)
        if convergence['stopped_on_validation_plateau']:
            break
        budget *= 2
    audit = audit_checkpoint(folder,config,seed,ids,'best_validation_MAE_eV_atom')
    with (folder/f'seed{seed}_test_predictions.csv').open(newline='') as f:
        raw = list(csv.DictReader(f))
    if [r['source_id'] for r in raw] != source.TEST:
        raise ValueError(f'Unexpected external test membership for {mode}')
    rows = []
    for p in raw:
        sid = p['source_id']
        s = records[sid]
        target = float(p['target'])*s.size + offsets[sid]
        pred = float(p['prediction'])*s.size + offsets[sid]
        if abs(target-s.energy) > .0001 or not np.isfinite(pred):
            raise ValueError(f'Target mismatch or nonfinite prediction in {sid}')
        rows.append(dict(mode=mode,snapshot=sid,n_real_atoms=s.size,dft_eV_cell=s.energy,pred_eV_cell=pred))
    save_csv(base/'heldout_energies.csv',rows)
    pairs = []
    for ii in rows[1:]:
        a = rows[0]
        truth = ii['dft_eV_cell']-a['dft_eV_cell']
        predicted = ii['pred_eV_cell']-a['pred_eV_cell']
        error = predicted-truth
        pairs.append(dict(mode=mode,type_I=a['snapshot'],type_II=ii['snapshot'],DFT_gap_eV=truth,
                          predicted_gap_eV=predicted,signed_error_eV=error,absolute_error_eV=abs(error),
                          correct_order=bool(truth*predicted>0)))
    save_csv(base/'heldout_pairs.csv',pairs)
    results = dict(mode=mode,seed=seed,selected_checkpoint=str(folder/f'seed{seed}_best_checkpoint.pth'),
                   device=device,**(convergence | audit),pairs=pairs)
    (base/'planar_metrics.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n',encoding='utf8')


def summarize():
    rows, all_pairs = [], []
    for mode in MODES:
        path = LOGS/mode/'planar_metrics.json'
        if not path.is_file():
            raise FileNotFoundError(f'Outstanding converged fit for {mode}')
        report = json.loads(path.read_text(encoding='utf8'))
        if not report['stopped_on_validation_plateau'] or not report['updated_trainable_tensors']:
            raise ValueError(f'Incomplete validation plateau or random-weight audit for {mode}')
        pairs = report['pairs']
        all_pairs.extend(pairs)
        mae = float(np.mean([p['absolute_error_eV'] for p in pairs]))
        rows.append(dict(mode=mode,pred_gap_011_eV=pairs[0]['predicted_gap_eV'],
                         pred_gap_016_eV=pairs[1]['predicted_gap_eV'],order_correct=sum(p['correct_order'] for p in pairs),
                         gap_MAE_eV_cell=mae,gap_MAE_meV_real_atom=mae*1000/560,
                         best_epoch=report['best_epoch'],stop_epoch=report['epochs_completed'],
                         best_validation_MAE_eV_atom=report['best_validation_MAE_eV_atom']))
    rows.sort(key=lambda r:r['gap_MAE_eV_cell'])
    for row in rows:
        row['rank'] = 1+sum(r['gap_MAE_eV_cell'] < row['gap_MAE_eV_cell']-1.e-8 for r in rows)
    save_csv(RESULTS/'ranking.csv',rows)
    save_csv(RESULTS/'all_heldout_pairs.csv',all_pairs)
    print(json.dumps(rows,indent=2),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', nargs='+', choices=MODES, default=list(MODES))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    if args.summarize_only:
        summarize()
        return
    torch.set_num_threads(3)
    records = source.load()
    ids = source.partition(records)
    mu, offsets, labels = references(records,ids)
    write_protocol(records,ids,mu)
    if args.prepare_only:
        return
    init_elem_embedding(source.ROOT/'atom_init.json')
    for mode in dict.fromkeys(args.mode):
        print(f'Starting {mode} on {args.device} with a random initialization',flush=True)
        fit(mode,records,ids,mu,offsets,labels,args.device)
    if all((LOGS/m/'planar_metrics.json').is_file() for m in MODES):
        summarize()


if __name__ == '__main__':
    main()
