"""Shared checks for fixed, source-ID-grouped HERA experiments."""
import json

import numpy as np

from ..main import set_seed
from ..training.trainer import MEGNetTrainer, load_trusted_checkpoint


def persist_contract(path, data):
    if path.exists():
        if json.loads(path.read_text(encoding='utf8')) != data:
            raise ValueError(f'Existing benchmark definition differs: {path}')
        return
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf8')


def fixed_split(membership, schema, description):
    def iterate(data, targets, seeds, cv5=False):
        if cv5:
            raise ValueError('This experiment uses a prespecified group holdout')
        by_id = {s.source_id: (s, y) for s, y in zip(data, targets)}
        for seed in seeds:
            split = dict(display=f'seed={seed} {description}', logger_id=seed,
                         explain_id=f'seed_{seed}', seed=seed, ood_split={'schema': schema})
            for part, keys in membership.items():
                split[part+'_X'] = [by_id[k][0] for k in keys]
                split[part+'_y'] = [by_id[k][1] for k in keys]
            yield split
    return iterate


def audit_checkpoint(directory, config, seed, membership, val_field='best_val_mae_ev_per_defect'):
    """Verify that the selected weights changed and the recorded membership is exact."""
    best=load_trusted_checkpoint(directory/f'seed{seed}_best_checkpoint.pth',map_location='cpu')
    if best['config']!=config or best['split']['seed']!=seed:
        raise ValueError(f'{directory}: best checkpoint does not match the run')
    for part, ids in membership.items():
        if [s['source_id'] for s in best['split'][f'{part}_sources']]!=ids:
            raise ValueError(f'{directory}: checkpoint {part} membership changed')
    set_seed(seed)
    initial=MEGNetTrainer(config,'cpu',seed).model
    changes=[(p.detach()-best['model'][name]).abs().max().item()
             for name,p in initial.named_parameters() if p.requires_grad]
    if not np.all(np.isfinite(changes)):
        raise ValueError(f'{directory}: non-finite checkpoint parameters')
    audit=dict(best_epoch=best['best_epoch'],epochs_completed=best['epochs_completed'],
               trainable_tensors=len(changes),updated_trainable_tensors=int(np.count_nonzero(changes)),
               max_weight_change=max(changes),**{val_field:best['best_val_mae']})
    (directory/'training_audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf8')
    if not audit['updated_trainable_tensors']:
        raise RuntimeError(f'{directory}: all trainable weights equal their initialization')
    return audit

