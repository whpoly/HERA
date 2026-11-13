"""Replay saved 2DMD checkpoints using compatible archived HERA implementations."""
import argparse
import ast
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

import numpy as np
import pandas as pd
import torch
from pymatgen.core import Structure

REPO = Path(__file__).resolve().parents[1]
ARCHIVE = '76e58b477bf304bcf56cb469a7dff000574fd7b9'
LOW_ARCHIVE = 'd2d7dfd649e3271d9fd150de3659bdeb7b29fda9'
DEFAULT_RUN = REPO / 'logs' / 'bench_2dmd_high_all_models' / 'alignn' / '2dmd_high'
DEFAULT_OUTPUT = REPO / 'results' / 'hetero_multidefect_validation_20261109' / '2dmd_high'


def load_test_data(root, sources, density):
    tables = {}
    hosts = {}
    records = []
    for source in sources:
        sid = source['source_id']
        subset=density+'_density_defects'
        sid_path = source['source_path'].replace('\\', '/').split('/'+subset+'/')[-1]
        folder = sid_path.split('/')[0]
        directory = root / subset / folder
        if folder not in tables:
            tables[folder] = (pd.read_csv(directory/'targets.csv.gz',index_col=0),
                              pd.read_csv(directory/'descriptors.csv',index_col=0))
        targets, descriptors = tables[folder]
        target = targets.loc[sid]
        desc_id = target['descriptor_id']
        desc = descriptors.loc[desc_id]
        host = desc['base']
        if host not in hosts:
            unit_path = (root/f'{host}.cif' if density=='high' else
                         root/'low_density_defects'/'MoS2'/'unit_cells'/f'{host}.cif')
            hosts[host] = Structure.from_file(unit_path, primitive=False)
        defects = ast.literal_eval(desc['defects'])
        n = len(defects)
        y = float(target['formation_energy_per_site'])
        if not np.isclose(float(target['formation_energy']) / n, y, rtol=0, atol=1e-8):
            raise ValueError(f'{sid}: inconsistent DFT target or defect count')
        f = directory / 'initial' / f'{sid}.cif'
        st = Structure.from_file(f)
        st.source_id, st.source_name, st.source_path = sid, sid, str(f)
        records.append((st, hosts[host], ast.literal_eval(desc['cell']),
                        dict(source_id=sid, host=host, descriptor=desc_id, defect_count=n,
                             composition=str(desc['description']), target_ev_per_defect=y)))
    return records


def audit_original_training(args):
    from HERA.main import set_seed
    from HERA.training.trainer import MEGNetTrainer
    membership=json.loads((args.output/'membership.json').read_text())
    rows=[]
    for mode in args.mode:
        files=list((args.checkpoints/mode).rglob('seed123_best_checkpoint.pth'))
        if not files:continue
        if len(files)!=1:raise ValueError(f'{mode}: ambiguous checkpoint')
        checkpoint=torch.load(files[0],map_location='cpu',weights_only=False)
        if checkpoint['mode']!=mode:raise ValueError('Incorrect archived model')
        for part,ids in membership.items():
            if [r['source_id'] for r in checkpoint['split'][part+'_sources']]!=ids:
                raise ValueError(f'{mode}: inconsistent archived {part} IDs')
        set_seed(123)
        trainer=MEGNetTrainer(checkpoint['config'],'cpu',123)
        delta=[(p.detach()-checkpoint['model'][name]).abs().max().item()
               for name,p in trainer.model.named_parameters() if p.requires_grad]
        history=pd.read_csv(files[0].with_name('seed123_history.csv'))
        numbered=pd.to_numeric(history.epoch,errors='coerce').dropna().astype(int).tolist()
        if numbered!=list(range(1,len(numbered)+1)) or len(history[history.epoch=='TEST'])!=1:
            raise ValueError(f'{mode}: incomplete or noncontiguous historical learning curve')
        if not np.isfinite(delta).all() or not np.count_nonzero(delta):
            raise ValueError(f'{mode}: stored weights show no finite update')
        rows.append(dict(mode=mode,seed=123,epochs_in_curve=len(numbered),
                         trainable_tensors=len(delta),updated_trainable_tensors=int(np.count_nonzero(delta)),
                         max_abs_weight_change=max(delta),first_val_mae=float(history.iloc[0].val_mae),
                         minimum_val_mae=float(pd.to_numeric(history.val_mae,errors='coerce').min()),
                         checkpoint_test_mae=checkpoint['test_mae'],archive_revision=args.revision,
                         history=str(files[0].with_name('seed123_history.csv').relative_to(REPO))))
        del trainer,checkpoint
    pd.DataFrame(rows).to_csv(args.output/'original_fits.csv',index=False)
    print('Audited archived fits:',[(r['mode'],r['epochs_in_curve']) for r in rows],flush=True)


def run(args):
    from HERA.data.datasets import init_elem_embedding
    from HERA.data.structure_utils import convert_to_sparse_2dmd_high
    from HERA.main import set_seed
    from HERA.training.trainer import MEGNetTrainer

    init_elem_embedding(REPO/'atom_init.json')
    if args.training_audit_only:
        audit_original_training(args)
        return
    ref = None
    records = None
    args.output.mkdir(parents=True,exist_ok=True)
    summary = []
    for mode in args.mode:
        source_mode='hetero' if args.fixed_pool_check else mode
        folder = args.checkpoints/source_mode
        checkpoints = list(folder.rglob('seed123_best_checkpoint.pth'))
        if not checkpoints:
            summary.append(dict(mode=mode,status='no completed checkpoint'))
            continue
        if len(checkpoints)!=1:
            raise ValueError(f'{mode}: ambiguous checkpoints')
        checkpoint = torch.load(checkpoints[0],map_location='cpu',weights_only=False)
        if (checkpoint['dataset_name'],checkpoint['model_name'],checkpoint['mode']) != ('2dmd_'+args.density,'alignn',source_mode):
            raise ValueError(f'{mode}: incorrect checkpoint identity')
        ids = {part:[s['source_id'] for s in checkpoint['split'][part+'_sources']] for part in ('train','val','test')}
        if ref is None:
            ref=ids
            expected_sizes = [1800,600,600] if args.density=='high' else [7118,2373,2373]
            assert [len(ref[p]) for p in ('train','val','test')]==expected_sizes
            assert len(set().union(*ref.values()))==sum(expected_sizes)
            records=load_test_data(args.data_root,checkpoint['split']['test_sources'],args.density)
            (args.output/'membership.json').write_text(json.dumps(ref,indent=2)+'\n',encoding='utf8')
        if ids!=ref:raise ValueError(f'{mode}: ordered source IDs differ')
        config=copy.deepcopy(checkpoint['config'])
        if args.fixed_pool_check:
            if config['model'].get('local_radius') not in (None,0,0.):
                raise ValueError('The pooling check is specific to the zero local radius')
            config['task']='alignn_hetero_fixed_pool'
            config['model']['fixed_pooling']=True
        set_seed(123)
        model=MEGNetTrainer(config,args.device,seed=123)
        model.model.load_state_dict(checkpoint['model'],strict=True)
        model.scaler.load_state_dict(checkpoint['scaler'])
        rep='hetero' if mode.startswith('hetero') else ('attention' if mode.startswith(('attention','definet')) else ('full' if mode=='full' else 'full_x'))
        file=args.output/(mode+'_predictions.csv')
        if not file.exists():
            with file.with_suffix('.partial').open('w',newline='',encoding='utf8') as out:
                writer=None
                for start in range(0,len(records),args.chunk):
                    chunk=records[start:start+args.chunk]
                    # Both archived 2DMD loaders used this converter for mixed vacancies and substitutions.
                    graphs=[convert_to_sparse_2dmd_high(st,h,c,'alignn_'+rep,None,False,False,
                             local_cutoff=config['model'].get('local_radius')) for st,h,c,_ in chunk]
                    targets=torch.tensor([m['target_ev_per_defect'] for _,_,_,m in chunk],dtype=torch.float32)
                    _,pred=model.predict_structures(graphs,targets,checkpoint['model'],return_predictions=True)
                    for (_,_,_,meta),p in zip(chunk,np.asarray(pred).reshape(-1),strict=True):
                        row=dict(**meta,prediction_ev_per_defect=float(p))
                        if writer is None:writer=csv.DictWriter(out,fieldnames=row);writer.writeheader()
                        writer.writerow(row)
                    out.flush()
            file.with_suffix('.partial').replace(file)
        data=pd.read_csv(file)
        if data.source_id.tolist()!=ref['test']:raise ValueError(f'{mode}: cached predictions have different IDs')
        errors=np.abs(data.prediction_ev_per_defect-data.target_ev_per_defect)
        if not np.isfinite(errors).all():raise ValueError(f'{mode}: nonfinite prediction')
        hist=pd.read_csv(checkpoints[0].with_name('seed123_history.csv'))
        tests=hist[hist.epoch.astype(str)=='TEST']
        if len(tests)!=1:
            raise ValueError(f'{mode}: checkpoint-matched history has {len(tests)} final test rows')
        row=dict(mode=mode,status='replayed',checkpoint=str(checkpoints[0].relative_to(REPO)),
                 checkpoint_sha256=hashlib.sha256(checkpoints[0].read_bytes()).hexdigest(),
                 archive_revision=args.revision,checkpoint_mae=float(checkpoint['test_mae']),
                 history_mae=float(tests.iloc[-1].test_mae),replayed_mae=float(errors.mean()))
        if args.fixed_pool_check:
            baseline=pd.read_csv(DEFAULT_OUTPUT.parent/('2dmd_'+args.density)/'hetero_predictions.csv')
            if baseline.source_id.tolist()!=data.source_id.tolist():raise ValueError('Original and fixed-pool cases differ')
            delta=np.abs(baseline.prediction_ev_per_defect-data.prediction_ev_per_defect)
            row.update(status='same trained weights, alternate pooling',reference_mode=source_mode,
                       cases=len(delta),max_prediction_difference_ev=float(delta.max()),
                       mean_prediction_difference_ev=float(delta.mean()))
        print('REPLAY RESULT',row,flush=True);summary.append(row)
        del model,checkpoint
        if args.device.startswith('cuda'):torch.cuda.empty_cache()
        (args.output/'replay.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf8')
    (args.output/'replay.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,default=REPO.parent/'dataset'/'2d-materials-point-defects-all')
    parser.add_argument('--density',choices=['high','low'],default='high')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--checkpoints',type=Path)
    parser.add_argument('--mode',nargs='+',default=['hetero','full_x','attention','definet','hetero_was','definet_was','full','was_x','attention_was'])
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--chunk',type=int,default=32)
    parser.add_argument('--fixed-pool-check',action='store_true',
                        help='Evaluate the archived zero-radius hetero weights using the fixed-pool forward pass')
    parser.add_argument('--training-audit-only',action='store_true',
                        help='Check the original training histories and compare selected weights to random initialization')
    args=parser.parse_args()
    if args.fixed_pool_check and args.training_audit_only:
        parser.error('The fixed-pool check uses the original hetero fit; audit training separately')
    args.revision=ARCHIVE if args.density=='high' else LOW_ARCHIVE
    if args.fixed_pool_check:args.mode=['hetero_fixed_pool']
    suffix='_fixed_pool_check' if args.fixed_pool_check else ''
    args.output=args.output or DEFAULT_OUTPUT.parent/('2dmd_'+args.density+suffix)
    args.checkpoints=args.checkpoints or (REPO/'logs'/('bench_2dmd_'+args.density+'_all_models')/'alignn'/('2dmd_'+args.density))
    if args.chunk<1:parser.error('chunk must be positive')
    paths=['__init__.py','main.py','atom_init.json',':(glob)config/*.py',':(glob)data/*.py',
           ':(glob)models/*.py',':(glob)training/*.py',':(glob)utils/*.py']
    with tempfile.TemporaryDirectory(prefix='hera_2dmd_replay_') as tmp:
        archive=subprocess.check_output(['git','archive','--format=zip',args.revision,'--',*paths],cwd=REPO)
        with zipfile.ZipFile(io.BytesIO(archive)) as z:z.extractall(Path(tmp)/'HERA')
        sys.path.insert(0,tmp)
        run(args)


if __name__=='__main__':
    main()

