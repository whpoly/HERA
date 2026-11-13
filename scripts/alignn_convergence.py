"""Read the saved stopping condition and verify it against each validation learning curve."""
import csv

import numpy as np

from ..main import is_meaningful_relative_improvement
from ..training.trainer import load_trusted_checkpoint


def convergence_audit(folder, seed):
    best = load_trusted_checkpoint(folder/f'seed{seed}_best_checkpoint.pth', map_location='cpu')
    with (folder/f'seed{seed}_history.csv').open(newline='') as f:
        hist = [r for r in csv.DictReader(f) if r['val_mae']]
    n = best['epochs_completed']
    if [int(r['epoch']) for r in hist] != list(range(1,n+1)):
        raise ValueError(f'Incomplete learning curve: {folder}')
    ve = [float(r['val_mae']) for r in hist]
    best_epoch = best['best_epoch']
    if abs(ve[best_epoch-1]-best['best_val_mae']) > 1e-6 or abs(min(ve)-best['best_val_mae']) > 1e-6:
        raise ValueError(f'Best checkpoint did not come from the minimum validation MAE: {folder}')
    reference = float('inf')
    meaningful_epoch = 0
    for epoch, value in enumerate(ve, start=1):
        if is_meaningful_relative_improvement(value, reference, best['early_stopping_min_delta_percent']):
            reference = value
            meaningful_epoch = epoch
    if not np.isfinite(ve).all() or not np.isfinite([float(r['train_mae']) for r in hist]).all():
        raise ValueError(f'Nonfinite learning curve: {folder}')
    return dict(best_epoch=best_epoch, epochs_completed=n, epoch_budget=best['epochs'],
                validation_patience=best['early_stopping_patience'],
                min_relative_improvement_pct=best['early_stopping_min_delta_percent'],
                stopped_on_validation_plateau=bool(best['stopped_early']),
                trailing_epochs_from_rounded_log=n-meaningful_epoch,
                train_MAE_at_best=float(hist[best_epoch-1]['train_mae']), val_MAE_at_best=best['best_val_mae'],
                final_train_MAE=float(hist[-1]['train_mae']), final_val_MAE=ve[-1])


def plot_learning_curves(panels, seed, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ncols = min(3, len(panels))
    nrows = (len(panels)+ncols-1)//ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6*ncols, 2.9*nrows), squeeze=False)
    for axis, (label, folder, report) in zip(axes.flat, panels):
        with (folder/f'seed{seed}_history.csv').open(newline='') as f:
            rows = [r for r in csv.DictReader(f) if r['val_mae']]
        epochs = [int(r['epoch']) for r in rows]
        axis.plot(epochs, [max(1e-7,float(r['train_mae'])) for r in rows], color='#527396', label='Training', lw=1)
        axis.plot(epochs, [max(1e-7,float(r['val_mae'])) for r in rows], color='#166a60', label='Validation', lw=1.2)
        axis.axvline(report['best_epoch'], color='#686b72', ls='--', lw=.9, label='Selected epoch')
        axis.set(yscale='log',xlabel='Epoch',ylabel='MAE (eV)',title=label)
        axis.spines[['top','right']].set_visible(False)
        axis.tick_params(labelsize=8)
    for axis in list(axes.flat)[len(panels):]:
        axis.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False)
    fig.tight_layout(rect=(0,.13/nrows,1,1))
    fig.savefig(path,dpi=175,bbox_inches='tight')
    plt.close(fig)
