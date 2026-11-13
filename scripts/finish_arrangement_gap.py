"""Complete all prespecified discovery fits and any triggered three-seed confirmation."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

from ..arrangement_sources import SOURCES


def call(module, args, log, env=None):
    command = [sys.executable, '-u', '-m', f'HERA.scripts.{module}', *args]
    print(' '.join(command), flush=True)
    with log.open('a', encoding='utf8') as output:
        if env is not None and env.get('CUDA_LAUNCH_BLOCKING')=='1':
            output.write('Restarting the unfinished fold with synchronous CUDA kernel reporting.\n')
            output.flush()
        subprocess.run(command, cwd=Path(__file__).resolve().parents[2], stdout=output,
                       stderr=subprocess.STDOUT, check=True, env=env)


def recover_fold(dataset, mode, fold):
    spec = SOURCES[dataset]
    contract = json.loads((spec.logs/'data_contract.json').read_text(encoding='utf8'))
    seed = contract['initial_screen_seed']
    if mode not in contract['model_names'] or fold not in range(len(contract['partitions'])):
        raise ValueError('Invalid recovery mode or fold')
    base = spec.logs/f'fold{fold}'/f'seed{seed}'/mode
    if (base/'metrics.json').exists():
        return
    if not (base/f'seed{seed}_history.csv').exists():
        raise ValueError('A diagnostic restart requires an interrupted training history')
    args = ['--dataset',dataset,'--device','cuda:0','--batch-size',str(contract['batch_size']),
            '--seed',str(seed),'--mode',mode,'--fold',str(fold),'--resume']
    call('run_arrangement_gap',args,spec.logs/f'seed{seed}_campaign.log',
         env=dict(os.environ,CUDA_LAUNCH_BLOCKING='1'))


def confirmed_rank(dataset):
    spec = SOURCES[dataset]
    table = spec.results/'model_comparison.csv'
    if not table.is_file():
        return None
    with table.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 10:
        return None
    ordinary = next(row for row in rows if row['mode']=='hetero')
    rank = int(ordinary['rank'])
    if rank == 1:
        protocol = json.loads((spec.logs/'data_contract.json').read_text(encoding='utf8'))
        if int(ordinary['n_seeds']) != 1+len(protocol['confirmation']['seeds']):
            return None
    return rank


def qualifying_first(dataset):
    """A first-place search result must resolve energy differences and beat all distinct alternatives."""
    spec = SOURCES[dataset]
    with (spec.results/'model_comparison.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    with (spec.results/'physical_control_pairs.csv').open(newline='') as handle:
        gaps = [abs(float(row['dft_gap_eV'])) for row in csv.DictReader(handle)]
    tolerance = json.loads((spec.logs/'data_contract.json').read_text(encoding='utf8'))['tie_tolerance_eV']
    scores = {r['mode']:float(r['gap_MAE_eV_cell']) for r in rows}
    competing = [value for mode,value in scores.items() if mode not in ('hetero','hetero_fixed_pool')]
    return scores['hetero']+tolerance < min(sum(gaps)/len(gaps),min(competing))


def has_lower_screen_score(dataset):
    """A completed paired comparison can exclude first place while other fits continue."""
    folder = SOURCES[dataset].logs
    contract = json.loads((folder/'data_contract.json').read_text(encoding='utf8'))
    seed = contract['initial_screen_seed']
    def score(mode):
        error_sum, pairs = 0., 0
        for fold, membership in enumerate(contract['partitions']):
            directory = folder/f'fold{fold}'/f'seed{seed}'/mode
            marker = directory/'metrics.json'
            if not marker.is_file():
                return None
            result = json.loads(marker.read_text(encoding='utf8'))
            with (directory/'test_structures.csv').open(newline='') as handle:
                ids = [row['source_id'] for row in csv.DictReader(handle)]
            if (ids != membership['test'] or result['fold'] != fold or result['mode'] != mode
                    or result['seed'] != seed or not result['stopped_on_validation_plateau']
                    or result['updated_trainable_tensors'] < 1):
                raise ValueError(f'Invalid completed fold in {directory}')
            error_sum += result['n_pairs'] * result['test_gap_MAE_eV']
            pairs += result['n_pairs']
        return error_sum / pairs
    ordinary = score('hetero')
    if ordinary is None:
        return False
    for mode in contract['model_names']:
        if mode in ('hetero','hetero_fixed_pool'):
            continue
        candidate = score(mode)
        if candidate is not None and candidate < ordinary-contract['tie_tolerance_eV']:
            return True
    return False


def finish(dataset):
    spec = SOURCES[dataset]
    folder = spec.logs
    contract = json.loads((folder/'data_contract.json').read_text(encoding='utf8'))
    order = ['hetero','attention_was','hetero_was','full','full_x','attention','was_x',
             'hetero_fixed_pool','definet','definet_was']
    modes = contract['model_names']
    seed0 = contract['initial_screen_seed']
    extra = contract['confirmation']['seeds']
    if set(order) != set(modes):
        raise ValueError('Training queue differs from the full non-hypergraph roster')
    for seed in [seed0, *extra]:
        complete = [folder/f'fold{f}'/f'seed{seed}'/m/'metrics.json' for m in modes for f in range(3)]
        if not all(path.exists() for path in complete):
            args = ['--dataset',dataset,'--device','cuda:0','--batch-size',str(contract['batch_size']),
                    '--seed',str(seed),'--mode',*order]
            if any((folder/f'fold{f}'/f'seed{seed}').exists() for f in range(3)):
                args.append('--resume')
            call('run_arrangement_gap', args, folder/f'seed{seed}_campaign.log')
        if seed in (seed0, extra[-1]):
            seeds = [seed0] if seed == seed0 else [seed0,*extra]
            call('analyze_arrangement_gap', ['--dataset',dataset,'--seeds',*map(str,seeds)],
                 folder/'analysis_campaign.log')
            with (spec.results/'model_comparison.csv').open(newline='') as handle:
                ordinary = next(row for row in csv.DictReader(handle) if row['mode']=='hetero')
            print(f"{dataset} seeds={seeds} ordinary hetero rank={int(ordinary['rank'])} "
                  f"gap MAE={float(ordinary['gap_MAE_eV_cell']):.6f}", flush=True)
            if int(ordinary['rank']) != 1:
                return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('datasets', choices=SOURCES, nargs='+')
    parser.add_argument('--wait-pid', type=int)
    parser.add_argument('--retry-fold', nargs=2, metavar=('MODE','FOLD'),
                        help='Restart a previously interrupted discovery fit with synchronous CUDA reporting')
    parser.add_argument('--retry-only', action='store_true', help='Return after checking the recovered fit')
    parser.add_argument('--after-negative', choices=SOURCES, nargs='+',
                        help='Start only if every named cohort completes without confirmed first place')
    args = parser.parse_args()
    if args.retry_fold and len(args.datasets)!=1:
        parser.error('--retry-fold takes exactly one dataset')
    if args.retry_only and not args.retry_fold:
        parser.error('--retry-only requires --retry-fold')
    if args.wait_pid:
        try:
            process = psutil.Process(args.wait_pid)
            job = process.cmdline()
            if not any(module in job for module in ('HERA.scripts.run_arrangement_gap','HERA.scripts.finish_arrangement_gap')):
                raise ValueError('PID is not an arrangement benchmark worker or its coordinator')
            process.wait()
        except psutil.NoSuchProcess:
            pass
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[name] = '1'
    if args.after_negative:
        if set(args.datasets) & set(args.after_negative):
            parser.error('A dataset cannot wait on its own outcome')
        print(f'Waiting for complete rankings on {args.after_negative}', flush=True)
        while True:
            ranks = [confirmed_rank(d) for d in args.after_negative]
            if any(r==1 and qualifying_first(d) for d,r in zip(args.after_negative,ranks)):
                print('An earlier dataset has confirmed first place. Conditional experiment skipped.', flush=True)
                return
            if all(r is not None for r in ranks):
                print(f'Completed ranks {dict(zip(args.after_negative,ranks))}', flush=True)
                break
            if any(r is not None for r in ranks) and all(
                    r is not None or has_lower_screen_score(d) for d,r in zip(args.after_negative,ranks)):
                print('Earlier cohorts have converged competitors below ordinary hetero. '
                      'One full slate is complete and the remaining comparisons continue.', flush=True)
                break
            time.sleep(45)
    for dataset in dict.fromkeys(args.datasets):
        if args.retry_fold:
            recover_fold(dataset,args.retry_fold[0],int(args.retry_fold[1]))
            if args.retry_only:
                return
        finish(dataset)


if __name__ == '__main__':
    main()
