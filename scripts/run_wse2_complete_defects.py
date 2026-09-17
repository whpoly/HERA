"""Train the complete-defect ablation with an existing WSe2 baseline's protocol.

Run from HERA's parent: python -m HERA.scripts.run_wse2_complete_defects.
The existing training loop selects checkpoints using only low validation data.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd
import torch

from ..config.defaults import alignn_hetero_run_components
from ..data.datasets import init_elem_embedding, load_dataset
from ..main import iter_concentration_transfer_splits, train_single_mode, write_mode_summary
from ..training.trainer import load_trusted_checkpoint


ROOT = Path(__file__).resolve().parents[1]
BASELINE = (ROOT / 'logs/hetero_energy_mean_benchmark/alignn/2dmd_wse2/hetero/r0/'
            'features_layernorm/pool_defect_energy_mean/relations_shared_residual_rank8/'
            'seed123_best_checkpoint.pth')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-checkpoint', type=Path, default=BASELINE)
    parser.add_argument('--run-dir', type=Path, default=ROOT / 'logs/hetero_complete_defects_wse2')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    args.run_dir = args.run_dir.resolve()
    args.baseline_checkpoint = args.baseline_checkpoint.resolve()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    def status(phase, **extra):
        record = dict(phase=phase, pid=os.getpid(), elapsed_seconds=time.time()-started, **extra)
        (args.run_dir / 'status.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
        print(json.dumps(record), flush=True)

    try:
        os.chdir(ROOT.parent)
        torch.set_num_threads(1)
        baseline = load_trusted_checkpoint(args.baseline_checkpoint, 'cpu')
        assert baseline['dataset_name'] == '2dmd_wse2'
        assert baseline['task'] == 'alignn_hetero'
        config = copy.deepcopy(baseline['config'])
        assert config['model'].get('hetero_defect_connectivity', 'physical') == 'physical'
        assert config['model'].get('local_radius') == 0
        config['model']['hetero_defect_connectivity'] = 'complete'
        seed = int(baseline['split']['seed'])
        epochs = int(baseline['epochs'])
        init_elem_embedding(ROOT / 'atom_init.json')
        status('loading', seed=seed, epochs=epochs, baseline_mae=baseline['test_mae'])
        dataset = load_dataset('2dmd_wse2', 'alignn', local_cutoff=0, representations={'hetero'})
        split = next(iter_concentration_transfer_splits(dataset[1], dataset[-1], [seed]))
        for part in ('train', 'val', 'test'):
            current = [s.source_id for s in split[f'{part}_X']]
            previous = [s['source_id'] for s in baseline['split'][f'{part}_sources']]
            if current != previous:
                raise ValueError(f'{part} source order differs from saved baseline; refusing unmatched comparison')
        parts = alignn_hetero_run_components(config['model'])
        output = args.run_dir.joinpath('alignn', '2dmd_wse2', 'hetero', 'r0', *parts)
        output.mkdir(parents=True, exist_ok=True)
        label = 'hetero_r0_' + '_'.join(parts)
        status('training', output=str(output), split_counts={p: len(split[p+'_X']) for p in ('train', 'val', 'test')},
               baseline_mae=baseline['test_mae'], config=config)
        losses = train_single_mode('hetero', config, dataset, dataset[-1], [seed], epochs,
                                   args.device, 'alignn', '2dmd_wse2', log_dir=str(output),
                                   run_label=label, resume=True)
        write_mode_summary(str(output / 'summary.txt'), 'alignn', '2dmd_wse2', label,
                           losses, epochs, [seed], config, radius_label='r0')
        candidate = load_trusted_checkpoint(output / f'seed{seed}_best_checkpoint.pth', 'cpu')
        old = pd.read_csv(args.baseline_checkpoint.with_name(f'seed{seed}_test_predictions.csv'))
        new = pd.read_csv(output / f'seed{seed}_test_predictions.csv')
        paired = old.merge(new, on='source_id', suffixes=('_baseline', '_complete'), validate='one_to_one')
        assert len(paired) == len(old) == len(new) == 500
        np.testing.assert_allclose(paired.target_baseline, paired.target_complete, rtol=0, atol=1e-8)
        paired['error_reduction'] = paired.absolute_error_baseline - paired.absolute_error_complete
        paired.to_csv(args.run_dir / 'paired_predictions.csv', index=False)
        result = {'baseline_mae': float(baseline['test_mae']), 'complete_mae': float(candidate['test_mae']),
                  'baseline_val_mae': baseline['best_val_mae'], 'complete_val_mae': candidate['best_val_mae'],
                  'seed': seed, 'test_samples': len(paired), 'same_ordered_splits': True,
                  'epochs_completed': candidate['epochs_completed'], 'best_epoch': candidate['best_epoch'],
                  'improved_samples': int(paired.error_reduction.gt(0).sum()),
                  'baseline_checkpoint': str(args.baseline_checkpoint), 'complete_checkpoint': str(output / f'seed{seed}_best_checkpoint.pth')}
        result['relative_mae_improvement_percent'] = 100 * (1-result['complete_mae']/result['baseline_mae'])
        (args.run_dir / 'comparison.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        report = ('# WSe2 low → high: complete defect edges\n\n'
                  f'Seed {seed}; identical ordered train/validation/test sources; 500 high test samples.\n\n'
                  '| Variant | High MAE (eV/defect) | Best low validation MAE |\n|---|---:|---:|\n'
                  f"| Physical baseline | {result['baseline_mae']:.6f} | {result['baseline_val_mae']:.6f} |\n"
                  f"| Complete defect edges | {result['complete_mae']:.6f} | {result['complete_val_mae']:.6f} |\n\n"
                  f"Relative MAE improvement: {result['relative_mae_improvement_percent']:.2f}%. "
                  f"Best epoch {result['best_epoch']}; completed {result['epochs_completed']} epochs.\n\n"
                  'Only graph connectivity changed. The original 0–6 Å radial basis is retained, '
                  'so added long edges have weak radial features. All main-graph edges enter the ALIGNN line graph. '
                  'This is a single-seed comparison to a saved baseline; hardware/software differences may affect training.\n')
        (args.run_dir / 'comparison.md').write_text(report, encoding='utf-8')
        status('complete', **result)
    except BaseException:
        status('failed', error=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
