"""Merge saved HERA relation benchmarks. This entry point never trains models."""

import argparse
from collections import Counter
from pathlib import Path
import sys

# Direct-file execution selects this checkout, avoiding another installed HERA.
ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ''):
    sys.path.insert(0, str(ROOT.parent))

from HERA.training.results import merge_benchmark_results, read_checkpoint_result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=ROOT / 'logs/hetero_relation_native_semi_imp2d')
    parser.add_argument('--variant', nargs='+', choices=('baseline', 'no_aa', 'no_dd', 'no_aa_dd', 'references'),
                        default=['baseline', 'no_dd'])
    args = parser.parse_args(argv)
    root = args.run_dir.resolve()
    variants = list(dict.fromkeys(args.variant))
    print(f'MERGE ONLY: no training or test evaluation. Script: {Path(__file__).resolve()}', flush=True)
    print(f'Result root: {root}', flush=True)
    if not any((root / variant).is_dir() for variant in variants):
        parser.error(f'No saved variant directories under {root}. '
                     'Pass the common parent of baseline/ and no_dd/. No training was started.')
    # The folder name is provenance, not evidence that DD was kept or removed.
    records = []
    for variant in variants:
        counts = Counter()
        for path in sorted((root / variant).rglob('seed*_best_checkpoint.pth')):
            try:
                record = read_checkpoint_result(path)
            except Exception as exc:
                parser.error(f'Cannot verify checkpoint configuration: {path}\n{exc}\nNo training was started.')
            counts[(record['dataset_name'], record['mode'], record['aa'], record['dd'],
                    record['relation_variant'])] += 1
            records.append(record)
        for (dataset, mode, aa, dd, actual), count in counts.items():
            print(f'  folder={variant} | dataset={dataset} | mode={mode} | '
                  f'saved AA={aa}, DD={dd} => {actual} | checkpoints={count}', flush=True)
        if (root / variant).is_dir() and not counts:
            print(f'  folder={variant}: no checkpoints to audit; merging saved text/CSV metrics only.', flush=True)
    try:
        merge_benchmark_results(root, variants, checkpoint_results=records)
    except ValueError as exc:
        parser.error(f'{exc}\nNo training was started.')


if __name__ == '__main__':
    main()
