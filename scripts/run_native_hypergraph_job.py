"""One-off local HyperALIGNN run, followed by a full prediction audit and report."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/native_hypergraph_paired_20260929'


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def save(state, state_path):
    temporary = state_path.with_name(state_path.name + '.tmp')
    temporary.write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
    temporary.replace(state_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    out = args.output.resolve()
    protocol = json.loads((out / 'protocol.json').read_text(encoding='utf-8'))
    if (out / 'metrics.json').exists():
        raise FileExistsError('Metrics already exist; refusing to repeat training.')
    state_path = out / 'run_state.json'
    existing = [name for name in ('run_state.json', 'train.log', 'seed123_history.csv',
                                 'seed123_best_checkpoint.pth', 'finalize.log')
                if (out / name).exists()]
    if existing:
        raise FileExistsError(f'Run artifacts already exist; refusing to overwrite: {existing}')
    model = protocol['config']['model']
    training_command = [sys.executable, '-u', '-m', 'HERA.scripts.run_native_hypergraph_paired',
                        '--output', str(out), '--hypergraph-radius', str(model['hypergraph_radius']),
                        '--hypergraph-updates', model['hypergraph_updates']]
    state = {'status': 'starting', 'pid': os.getpid(), 'started_at': utc_now(),
             'output': str(out), 'epochs': protocol['epochs'], 'seed': protocol['seed'],
             'hypergraph_radius': model['hypergraph_radius'],
             'hypergraph_updates': model['hypergraph_updates'], 'command': training_command}
    save(state, state_path)
    env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    try:
        with (out / 'train.log').open('x', encoding='utf-8') as log:
            train = subprocess.Popen(
                training_command,
                cwd=ROOT.parent, env=env, stdout=log, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            state.update(status='training', training_pid=train.pid)
            save(state, state_path)
            state['training_returncode'] = train.wait()
            state['training_finished_at'] = utc_now()
            state['status'] = 'auditing' if state['training_returncode'] == 0 else 'training_failed'
            save(state, state_path)
        with (out / 'finalize.log').open('x', encoding='utf-8') as log:
            audit = subprocess.run(
                [sys.executable, '-u', '-m', 'HERA.scripts.finalize_native_hypergraph_paired',
                 '--output', str(out)],
                cwd=ROOT.parent, env=env, stdout=log, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            state['audit_returncode'] = audit.returncode
        state['status'] = 'complete' if state['training_returncode'] == 0 and state['audit_returncode'] == 0 else 'failed'
        state['finished_at'] = utc_now()
        save(state, state_path)
        return 0 if state['status'] == 'complete' else 1
    except BaseException:
        state.update(status='supervisor_failed', finished_at=utc_now(), error=traceback.format_exc())
        save(state, state_path)
        raise


if __name__ == '__main__':
    sys.exit(main())
