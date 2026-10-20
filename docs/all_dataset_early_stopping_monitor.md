# Five-dataset benchmark with an early-stopping comparison

Run from HERA's parent directory after syncing the updated source to the server:

```bash
python -m HERA.scripts.run_all_dataset_benchmark --seed 11 42 1245 --epochs 500 --early-stopping-patience 50 --early-stopping-min-delta-percent 0.5 --device cuda:0
```

This sequentially runs native, semi, imp2d, 2dmd_low and 2dmd_high. Each selected
mode trains for all 500 epochs. Hypergraph and Hypergraph-WAS are excluded.
Full-X is skipped for semi/imp2d, where it duplicates Full. The default suite
has 48 dataset/mode combinations and 144 training runs across three seeds.
The loss remains MSE; Huber is optional with `--point-loss huber_ev
--huber-delta-ev 1.0`. Nothing trains locally merely by preparing this command.

## What the two evaluations mean

- The first time validation MAE fails to improve meaningfully for `patience`
  epochs, test and save the checkpoint with the smallest validation MAE seen
  up to that point. This happens at most once per seed and mode.
- Training then continues from its current weights and optimizer state.
  The intermediate evaluation restores weights, buffers, module modes,
  validation-data references and Python/NumPy/Torch RNG states.
- After the full budget, test the checkpoint with the smallest validation MAE
  over the entire run, as in the existing benchmark. This is **not** necessarily
  the last epoch's weights.
- Test metrics never choose checkpoints, reset patience or drive the scheduler.
  The intermediate result is a reporting comparison only.
- If the stopping condition never triggers, the early result is absent, not
  zero and not a copy of the final result.

`--early-stopping-min-delta-percent 0.5` means a relative improvement greater
than 0.5%, not an absolute improvement of 0.5 eV. The best checkpoint still
tracks every strict validation improvement, even one below this threshold.

The general training CLI also supports the feature:

```text
--early-stopping-patience 50 --early-stopping-min-delta-percent 0.5 --early-stopping-monitor-only
```

Without `--early-stopping-monitor-only`, positive patience retains the original
behavior of actually stopping training. Patience zero disables stopping and
cannot be combined with monitor-only mode.

## Dataset and model settings

| Dataset | Data policy | Train batch | Hetero settings |
|---|---|---:|---|
| native | Global strict/MAD screening, E ≤ 15 eV, exclude POSCAR0, then new seeded splits | 16 | LayerNorm features/nodes, shared_residual rank 8, defect_energy_mean, no DD |
| semi | physical screening, legacy_available source policy | 64 | Same as native |
| imp2d | Original DB, physical screening, reviewed host exclusions, −10 < E < 10, relaxed interstitial only | 64 | Same as native |
| 2dmd_low | Existing low-concentration random-split cohort | 8 | BatchNorm features, LayerNorm nodes, independent relations, type_mean, keep DD |
| 2dmd_high | Existing high-concentration random-split cohort | 16 | Same as 2dmd_low |

All use ALIGNN hidden size 64, cutoff 6 Å, at most 12 neighbors per target
node type, three ALIGNN blocks and three GCN blocks. These are within-dataset
random-split tests, not low-to-high transfer tests.

Native preparation is automatic and precedes training. Its new directory is
`<run-dir>/native_preparation`. To reuse the previous preparation for the same
three seeds and epoch budget, add:

```text
--native-preparation-dir HERA/logs/native_filter_le15_no_poscar0_3seeds
```

The preparation code verifies the source hashes, filtering policy and requested
split identity. Incompatible preparation or training configurations fail rather
than silently mixing results.

## Outputs and restart behavior

Default output root: `HERA/logs/all5_3seeds_early_stop_monitor`.
Each `<root>/alignn/<dataset>/<mode>/` contains:

- `seed11_early_stopping_checkpoint.pth`: best validation weights before the trigger.
- `seed11_early_stopping_test_predictions.csv`: intermediate held-out predictions.
- `seed11_best_checkpoint.pth`: best validation weights over the complete run.
- `seed11_test_predictions.csv`: final held-out predictions.
- `seed11_evaluation_comparison.json`: exact trigger/best epochs and both MAEs.
- `seed11_history.csv`: `EARLY_STOP_TEST` is intermediate; only `TEST` marks completion.

The root's `early_stopping_comparison.md` and `.json` summarize each seed and
mean/sample-standard-deviation across seeds. Early means include triggered
seeds only; paired deltas compare the same seeds. Final means include all
completed seeds.

`--resume` is included in each training invocation. Matching completed runs
are skipped; interrupted runs restart from epoch one, rather than restoring
optimizer progress. Previous intermediate artifacts are archived as `.bak`.
Use the new output root: old non-monitored results cannot supply missing early
snapshots or resume as this different protocol.

`--dry-run` prints commands without preparing data, training or creating files.
`--summary-only` reads saved results without training. Supply the same
dataset/seed/epoch/patience arguments when summarizing a customized run.
