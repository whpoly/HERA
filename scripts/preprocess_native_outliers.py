"""Create auditable native outlier manifests without modifying source data or training.

Statistics are fitted on each original training split and frozen for validation
and test. Prediction errors never participate; optional POSCAR indices restrict
which samples are eligible for exclusion without changing fitted thresholds.
"""
import argparse
import json
import math
from pathlib import Path

import pandas as pd

from ..data.native_filter import DEFAULT_POLICY, manifest_identity
from ..data.native_quality import audit_family_geometry
from ..data.native_preprocessing import inspect_source, write_manifest_tables, build_native_manifest, native_filter_policy
from ..main import iter_train_val_test_splits, parse_seed_values


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT.parent/'dataset/Dataset_1/Dataset_1/A_rich/Neutral')
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--profile', choices=['standard', 'strict'], default='standard',
                        help='strict adds element-scaled distances and train-only same-family energy screening')
    parser.add_argument('--seed', dest='seeds', nargs='+', default=['123'])
    parser.add_argument('--cv5', action='store_true')
    parser.add_argument('--filter-poscar-indices', type=int, nargs='+', default=None,
                        help='Only exclude abnormal samples with these POSCAR indices, e.g. 0 1; retain all other indices')
    parser.add_argument('--min-distance', type=float, default=DEFAULT_POLICY['min_distance_angstrom'])
    parser.add_argument('--iqr-multiplier', type=float, default=DEFAULT_POLICY['iqr_multiplier'])
    parser.add_argument('--iqr-floor-ev', type=float, default=DEFAULT_POLICY['iqr_floor_ev'])
    parser.add_argument('--min-group-train-samples', type=int, default=DEFAULT_POLICY['min_group_train_samples'])
    args = parser.parse_args()
    seeds = parse_seed_values(args.seeds, parser)
    if args.cv5 and len(seeds) != 1:
        parser.error('--cv5 requires exactly one seed')
    if any(not math.isfinite(x) or x < 0 for x in (args.min_distance, args.iqr_multiplier, args.iqr_floor_ev)) or args.iqr_multiplier == 0 or args.min_group_train_samples < 4:
        parser.error('Use finite nonnegative cutoffs, a positive IQR multiplier and at least 4 training samples per group')
    policy = {**native_filter_policy(args.profile), 'min_distance_angstrom': args.min_distance, 'iqr_multiplier': args.iqr_multiplier,
              'iqr_floor_ev': args.iqr_floor_ev, 'min_group_train_samples': args.min_group_train_samples}
    if args.filter_poscar_indices is not None:
        if min(args.filter_poscar_indices) < 0:
            parser.error('--filter-poscar-indices requires nonnegative indices')
        policy['filter_poscar_indices'] = sorted(set(args.filter_poscar_indices))
    if args.output is None:
        args.output = ROOT/'results'/('native_clean_strict' if args.profile == 'strict' else 'native_clean_v2')
        if args.filter_poscar_indices is not None:
            args.output = args.output.with_name(args.output.name+'_poscar'+'_'.join(map(str, policy['filter_poscar_indices'])))
    geometry_cache = {} if args.profile == 'strict' else None
    manifest = build_native_manifest(args.data_dir, args.seeds, args.cv5, policy,
                                     iter_train_val_test_splits, geometry_cache=geometry_cache)
    rows = manifest['records']
    path = args.output/'manifest.json'
    encoded = json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False)+'\n'
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != manifest:
        raise RuntimeError(f'Different preprocessing already exists in {args.output}; choose a new --output to preserve its experiment identity')
    args.output.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open('x', encoding='utf-8') as handle:
            handle.write(encoded)
    summaries = write_manifest_tables(manifest, args.output)
    if geometry_cache is not None:
        print('Checking same-family geometry/energy consistency (diagnostic only)...', flush=True)
        pairs, anomalies = audit_family_geometry(rows, geometry_cache, policy['deep_checks'])
        pair_frame = pd.DataFrame(pairs, columns=['family', 'left_id', 'right_id', 'adjacent_available_frames',
                                                'rms_angstrom', 'max_angstrom', 'energy_change_ev', 'near_geometry_energy_conflict'])
        pair_frame.to_csv(args.output/'family_geometry_pairs.csv', index=False)
        pair_frame[pair_frame.near_geometry_energy_conflict == True].to_csv(args.output/'geometry_energy_conflicts.csv', index=False)
        pd.DataFrame(anomalies, columns=['family', 'reason', 'n_frames']).to_csv(args.output/'family_index_anomalies.csv', index=False)
        print(f'Geometry/energy conflicts: {int(pair_frame.near_geometry_energy_conflict.sum())}; '
              f'families with changed cell/indexed species: {len(anomalies)}', flush=True)
    print(pd.DataFrame(summaries).to_string(index=False), flush=True)
    print(f'Manifest: {path}\nIdentity: {manifest_identity(manifest)["sha256"]}\nNo structures were deleted and no model was trained.', flush=True)


if __name__ == '__main__':
    main()
