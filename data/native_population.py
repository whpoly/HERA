"""Screen one seed-independent native population, then create fresh splits."""
import copy
import json
import math
from pathlib import Path
import statistics

import pandas as pd

from .native_filter import (PREFILTER_SCHEMA, apply_filter_fences, fit_filter_fences,
                            validate_manifest, verify_source_file, verify_source_table)
from .native_preprocessing import inspect_source, native_filter_policy
from .native_quality import configuration_family


POPULATION_SCHEMA = 'native_population_filter_v1'


def population_policy(max_energy_ev=None, exclude_poscar0=False):
    policy = dict(
        order='filter_then_split',
        strict_policy=native_filter_policy('strict'),
        strict_fit_scope='all raw quality-eligible samples before any split',
        family_mad=dict(min_samples=4, mad_sigma_multiplier=6.0, min_excess_ev=1.0,
                        mad_normalization=1.4826, direction='upper only', refit_iterations=1),
        family_mad_fit_scope='all strict-retained samples before any split',
        split_protocol='seeded random 60/20/20 on the frozen retained population',
        interpretation='Label-dependent dataset curation; not confirmed bad DFT data or paper reproduction')
    if max_energy_ev is not None:
        if not math.isfinite(float(max_energy_ev)):
            raise ValueError('Population energy upper bound must be finite')
        policy.update(energy_upper_bound_ev=float(max_energy_ev),
                      energy_bound_stage='after strict and family MAD; before splitting',
                      energy_bound_rule='retain E <= upper; no lower energy bound')
    if exclude_poscar0:
        policy.update(exclude_poscar0=True,
                      poscar0_source_rule='exclude filenames containing the exact token -POSCAR0-',
                      poscar0_exclusion_stage='after strict, family MAD and optional energy cap; before splitting')
    return policy


def validate_population(population):
    cap = population.get('policy', {}).get('energy_upper_bound_ev')
    exclude_poscar0 = population.get('policy', {}).get('exclude_poscar0', False)
    expected_schema = ('native_population_filter_v3' if exclude_poscar0 else
                       'native_population_filter_v2' if cap is not None else POPULATION_SCHEMA)
    if population.get('schema') != expected_schema or population.get('policy') != population_policy(cap, exclude_poscar0):
        raise ValueError('Native population filter policy differs; choose a new run directory')
    ids = [r['source_id'] for r in population['records']]
    if len(ids) != len(set(ids)) or any('/' in sid or '\\' in sid or sid in ('', '.', '..') for sid in ids):
        raise ValueError('Native population requires unique plain source filenames')
    for name in ('strict_kept', 'kept'):
        selected = population[name]
        selected_set = set(selected)
        if not selected or [sid for sid in ids if sid in selected_set] != selected:
            raise ValueError('Native population retained IDs must be a nonempty ordered subset')
    if not set(population['kept']) <= set(population['strict_kept']):
        raise ValueError('MAD-retained population must be a subset of strict-retained samples')
    before_poscar0 = population['pre_poscar0_kept'] if exclude_poscar0 else population['kept']
    if exclude_poscar0:
        before_set = set(before_poscar0)
        if [sid for sid in population['strict_kept'] if sid in before_set] != before_poscar0:
            raise ValueError('POSCAR0 exclusion must follow the frozen population screening')
        if population['kept'] != [sid for sid in before_poscar0 if '-POSCAR0-' not in sid]:
            raise ValueError('Native population retained IDs violate the POSCAR0 exclusion')
    if cap is not None:
        uncapped = population['uncapped_kept']
        uncapped_set = set(uncapped)
        if [sid for sid in population['strict_kept'] if sid in uncapped_set] != uncapped:
            raise ValueError('Energy cap must follow the frozen strict and MAD selection')
        by_id = {r['source_id']: r for r in population['records']}
        if before_poscar0 != [sid for sid in uncapped if by_id[sid]['target'] <= cap]:
            raise ValueError('Native population retained IDs violate the inclusive energy upper bound')
    excluded = population['excluded']
    kept_set = set(population['kept'])
    if ([r['source_id'] for r in excluded] != [sid for sid in ids if sid not in kept_set]
            or any(not r.get('reasons') for r in excluded)):
        raise ValueError('Native population exclusions must exactly account for removed sources')
    if len(population['kept']) < 3:
        raise ValueError('Native population requires at least three retained samples to split')
    json.dumps(population, allow_nan=False)
    return population


def screen_population(records, max_energy_ev=None, exclude_poscar0=False):
    """No seed or split enters any filtering stage."""
    policy = population_policy(max_energy_ev, exclude_poscar0)
    max_energy_ev = policy.get('energy_upper_bound_ev')
    ids = [r['source_id'] for r in records]
    strict = policy['strict_policy']
    bounds, family_bounds = fit_filter_fences(records, ids, strict)
    strict_kept, excluded = apply_filter_fences(records, ids, strict, bounds, family_bounds, scope='population')
    by_id = {r['source_id']: r for r in records}
    groups = {}
    for sid in strict_kept:
        groups.setdefault(configuration_family(sid), []).append(by_id[sid]['target'])
    mad = policy['family_mad']
    fences = {}
    for family, values in sorted(groups.items()):
        fence = dict(n_population=len(values), status='insufficient_population_samples')
        if len(values) >= mad['min_samples']:
            median = statistics.median(values)
            sigma = mad['mad_normalization'] * statistics.median(abs(value-median) for value in values)
            fence.update(status='fitted', median_ev=median, mad_sigma_ev=sigma,
                         upper_ev=median+max(mad['mad_sigma_multiplier']*sigma, mad['min_excess_ev']))
        fences[family] = fence
    removed = {r['source_id']: r for r in excluded}
    kept = []
    for sid in strict_kept:
        fence = fences[configuration_family(sid)]
        if fence['status'] == 'fitted' and by_id[sid]['target'] > fence['upper_ev']:
            removed[sid] = dict(source_id=sid, reasons=['above_strict_population_family_mad_upper'])
        else:
            kept.append(sid)

    cap_metadata = {}
    if max_energy_ev is not None:
        cap_metadata['uncapped_kept'] = kept.copy()
        for sid in kept:
            if by_id[sid]['target'] > max_energy_ev:
                removed[sid] = dict(source_id=sid, reasons=['above_population_energy_upper_bound'])
        kept = [sid for sid in kept if by_id[sid]['target'] <= max_energy_ev]
    if exclude_poscar0:
        cap_metadata['pre_poscar0_kept'] = kept.copy()
        for sid in kept:
            if '-POSCAR0-' in sid:
                removed[sid] = dict(source_id=sid, reasons=['excluded_population_poscar0'])
        kept = [sid for sid in kept if '-POSCAR0-' not in sid]

    def population_counts(fitted):
        # Shared strict code keeps its historical train-fitted metadata names.
        result = copy.deepcopy(fitted)
        for fence in result.values():
            fence['n_population'] = fence.pop('n_train')
            if fence['status'] == 'insufficient_train_samples':
                fence['status'] = 'insufficient_population_samples'
        return result

    return validate_population(dict(
        schema=('native_population_filter_v3' if exclude_poscar0 else
                'native_population_filter_v2' if max_energy_ev is not None else POPULATION_SCHEMA),
        policy=policy, records=copy.deepcopy(records), **cap_metadata,
        strict_kept=strict_kept, kept=kept, excluded=[removed[sid] for sid in ids if sid in removed],
        group_fences=population_counts(bounds), family_fences=population_counts(family_bounds),
        family_mad_fences=fences))


def prepare_population(output, data_dir, max_energy_ev=None, exclude_poscar0=False):
    """Freeze curation independently of the later random split and training."""
    output, data_dir = Path(output), Path(data_dir)
    path = output / 'native_population_filter.json'
    source = pd.read_csv(data_dir / 'id_prop_A_rich.csv', header=None)
    if path.exists():
        population = validate_population(json.loads(path.read_text(encoding='utf-8')))
        if population['policy'] != population_policy(max_energy_ev, exclude_poscar0):
            raise ValueError('Native population policy differs; choose a new run directory')
        verify_source_table(population, source)
        for row in population['records']:
            source_path = data_dir / row['source_id']
            if row['sha256'] is not None:
                verify_source_file(source_path, row)
            elif source_path.exists():
                raise ValueError(f'Cannot verify cached source: {source_path}; choose a new run directory')
        print(f'Reusing verified pre-split population: {path}', flush=True)
    else:
        ids = source[0].astype(str).tolist()
        if len(ids) != len(set(ids)) or any('/' in sid or '\\' in sid or sid in ('', '.', '..') for sid in ids):
            raise ValueError('Native source table requires unique plain filenames')
        records = []
        for index, row in enumerate(source.itertuples(index=False, name=None), 1):
            records.append(inspect_source(data_dir / str(row[0]), row[1], population_policy(max_energy_ev, exclude_poscar0)['strict_policy']))
            if index % 500 == 0:
                print(f'Inspected {index}/{len(source)} structures before splitting', flush=True)
        population = screen_population(records, max_energy_ev, exclude_poscar0)
        output.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8') as handle:
            json.dump(population, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write('\n')
    write_population_tables(population, output)
    print(f'Population before split: raw={len(population["records"])}, '
          f'strict={len(population["strict_kept"])}, retained={len(population["kept"])}', flush=True)
    if max_energy_ev is not None:
        after_cap = population['pre_poscar0_kept'] if exclude_poscar0 else population['kept']
        print(f'Inclusive energy bound: E <= {max_energy_ev:g} eV; '
              f'{len(population["uncapped_kept"])-len(after_cap)} additional removals after MAD', flush=True)
    if exclude_poscar0:
        print(f'POSCAR0 excluded before splitting: '
              f'{len(population["pre_poscar0_kept"])-len(population["kept"])} additional removals', flush=True)
    return population


def split_population(population, seeds, split_iterator):
    """Only the frozen retained IDs are passed to the existing 60/20/20 splitter."""
    validate_population(population)
    ids = population['kept']
    splits = []
    for split in split_iterator(ids, [0]*len(ids), seeds, cv5=False):
        membership = {part: list(split[part+'_X']) for part in ('train', 'val', 'test')}
        splits.append({**{key: split[key] for key in ('display', 'logger_id', 'seed', 'explain_id')},
                       'original': membership, 'kept': copy.deepcopy(membership),
                       'excluded': {part: [] for part in membership}})
    return validate_manifest(dict(
        schema=PREFILTER_SCHEMA, dataset='native', cv5=False, policy=copy.deepcopy(population['policy']),
        records=copy.deepcopy(population['records']), splits=splits,
        population_filter={key: copy.deepcopy(value) for key, value in population.items()
                           if key not in ('records', 'policy')}))


def write_population_tables(population, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    excluded = {r['source_id']: r['reasons'] for r in population['excluded']}
    strict = set(population['strict_kept'])
    rows = []
    for row in population['records']:
        sid = row['source_id']
        fence = population['family_mad_fences'].get(configuration_family(sid), {})
        rows.append(dict(source_id=sid, target_ev=row['target'], strict_retained=sid in strict,
                         retained=sid not in excluded, exclusion_reasons=';'.join(excluded.get(sid, [])),
                         family=configuration_family(sid), n_population=fence.get('n_population'),
                         median_ev=fence.get('median_ev'), mad_sigma_ev=fence.get('mad_sigma_ev'),
                         upper_ev=fence.get('upper_ev')))
    pd.DataFrame(rows).to_csv(output / 'population_filter_review.csv', index=False)
    by_id = {r['source_id']: r for r in population['records']}
    pd.DataFrame([(sid, by_id[sid]['target']) for sid in population['kept']]).to_csv(
        output / 'filtered_id_prop_A_rich.csv', index=False, header=False)
