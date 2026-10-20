# Five-dataset benchmark with an early-stopping comparison

Run from HERA's parent directory after syncing the updated source to the server:

```bash
python -m HERA.scripts.run_all_dataset_benchmark --seed 11 42 1245 --epochs 500 --early-stopping-patience 50 --early-stopping-min-delta-percent 0.5 --device cuda:0
```

This sequentially runs native, semi, imp2d, 2dmd_low and 2dmd_high. Each selected
mode trains for all 500 epochs. Hypergraph and Hypergraph-WAS are excluded.
Full-X is skipped for semi/imp2d, where it duplicates Full. At r=0,
Hetero-fixed-pool duplicates Hetero and is also skipped. The default suite
has 43 distinct dataset/mode combinations and 129 training runs across three seeds.
The loss remains MSE; Huber is optional with `--point-loss huber_ev
--huber-delta-ev 1.0`. Nothing trains locally merely by preparing this command.

## Automatic thresholds without extra training

Use `train_mad` to compute thresholds directly from training labels, with no
reference-model initialization, forward/backward passes or extra training epochs:

```bash
python -m HERA.scripts.run_all_dataset_benchmark --seed 11 42 1245 --epochs 500 --point-loss huber_ev --huber-auto-calibrate --huber-calibration-method train_mad --early-stopping-patience 50 --early-stopping-min-delta-percent 0.5 --device cuda:0 --native-preparation-dir HERA/logs/native_filter_le15_no_poscar0_3seeds --run-dir HERA/logs/all5_3seeds_huber_train_mad_v1
```

For each dataset and outer split seed, this computes:

```text
scale = median(abs(train_y - median(train_y))) / 0.6744897501960817
delta = max(1e-6, 1.345 * scale)
```

The multiplier is configurable with `--huber-calibration-multiplier`. The value
1.345 is a starting heuristic here, not a validated optimum for label-based
thresholds. The rule estimates **label spread**, not prediction-error/noise
spread. Between-material energy differences can therefore make delta too large,
leaving most prediction errors in the MSE region. No MAE improvement is promised.

If MAD is zero, use IQR/1.3489795003921634, then population standard deviation,
then the positive floor. These fallbacks preserve invariance to shifts in the
energy zero. The manifest records the method used and the training-label median,
quartiles, range, scale and delta.

Only the training labels determine the threshold. All modes within the same
dataset/seed share the frozen value; different seeds may have different values.
Thresholds use the original target units (eV, or eV/site for 2DMD). There are no
fold predictions or reference fits; `--huber-calibration-folds` and
`--huber-calibration-epochs` do not apply and can be omitted.

The preparation stage still reads structures through the existing data loader
and split/filter code to preserve sample membership. This has data-reading
overhead, but it does not train a model. The 129 formal benchmark runs are
unchanged: 500 epochs, monitor-only early stopping, no Hypergraph, DD dropped
for native/semi/imp2d Hetero modes and retained only for 2DMD-low/high.

The existing manifest/config/checkpoint checks and threshold summary apply.
Use a new run directory rather than mixing this method with OOF or manual delta
results. OOF remains the default method for compatibility with earlier commands;
the explicit `--huber-calibration-method train_mad` selects the no-extra-training
method above.

## Automatic Huber calibration using OOF reference fits

After syncing the source to the server, run this from HERA's parent directory:

```bash
python -m HERA.scripts.run_all_dataset_benchmark --seed 11 42 1245 --epochs 500 --point-loss huber_ev --huber-auto-calibrate --huber-calibration-folds 3 --huber-calibration-epochs 200 --early-stopping-patience 50 --early-stopping-min-delta-percent 0.5 --device cuda:0 --native-preparation-dir HERA/logs/native_filter_le15_no_poscar0_3seeds --run-dir HERA/logs/all5_3seeds_huber_auto_v1
```

This prepares the filtered native cohort, then calibrates and benchmarks each
of native, semi, interstitial imp2d, 2DMD-low and 2DMD-high. Native/semi/imp2d
Hetero modes drop DD; only the two 2DMD datasets keep DD. Hypergraph is excluded.

For **each dataset and each outer split seed**, calibration uses every retained
training sample once as an out-of-fold prediction target. Outer validation and
test labels never enter calibration or reference-model selection:

1. Partition the outer training set into three seeded folds. Known equivalent
   structure groups in the IMP2D manifest stay together. Other samples use their
   source identities as groups; this does not discover new structural duplicates.
2. Fit the fixed ALIGNN/WAS-X reference using MSE on the remaining folds. Reserve
   approximately 10% of that fitting pool for inner validation. Fit the target
   scaler on the fitting samples only; use inner validation MAE for scheduling
   and checkpoint selection. The OOF fold is not used for either.
3. Train each reference for 200 epochs, then predict its OOF fold using the
   best inner-validation checkpoint. Combine the three held-out residual sets.
4. Set `delta = max(1e-6, 1.345 * median(abs(r - median(r))) / 0.6744897501960817)`.
   If MAD is exactly zero, fall back to median absolute residual, then RMS
   residual, then the positive floor. Record the selected method.
5. Freeze that threshold and use it for **all benchmark modes on that same
   dataset and split**. Different seeds can produce different thresholds.
   Formal benchmark models start fresh; reference weights are not reused.

Thresholds are in the original target's units: eV for native/semi/imp2d and
eV per site for the existing 2DMD targets. No target normalization needs to
be supplied manually. The loss remains the existing `2 * Huber`.

This adds 45 reference fits (5 datasets × 3 seeds × 3 folds), followed by the
129 benchmark runs. `--huber-calibration-epochs`, `--huber-calibration-folds`,
and `--huber-calibration-multiplier` (default 1.345) control calibration.
The method is a reproducible scale heuristic, not a guarantee of optimal delta
or improved MAE: the reference has less fitting data and a shorter budget than
the final models. Do not choose these settings from test results.

`--huber-auto-calibrate` cannot be combined with manual delta flags.
The final training budget and monitor-only early-stopping behavior are unchanged.

Calibration artifacts live in `<run-dir>/huber_calibration/<dataset>/`:

- `protocol.json`: reference config, seeds, folds, epoch budget and scale rule.
- `seed11/fold1.json`: split identity, fit/inner-validation/OOF indices,
  selected epoch, history and physical-unit predictions, protected by a checksum.
- `seed11/oof_predictions.json`: source identities, targets, predictions and residuals.
- `manifest.json`: frozen per-seed thresholds, diagnostics and provenance hash.

Each mode's config and checkpoints carry the same calibration identity; each
checkpoint/evaluation JSON also records its resolved delta. Training checks
ordered split memberships and training labels against calibration before fitting.
This verifies sample/label alignment, not equality of graph tensors across modes.
The aggregate comparison reports the thresholds and verifies shared identities.

Reruns verify and reuse completed reference folds. An interrupted reference fold
restarts, while completed folds are retained. Completed benchmark runs are skipped.
Changed calibration settings or data identities are rejected. Use a fresh
`--run-dir` for a different experiment. Adding `--dry-run` previews every stage
without calibration/training or file writes. For `--summary-only`, retain the
same auto-calibration and protocol flags.

## Per-dataset Huber thresholds

`--huber-delta-ev-by-dataset DATASET=EV ...` overrides the global
`--huber-delta-ev` separately for each selected dataset. The flag requires
`--point-loss huber_ev`. All modes and seeds within a dataset share its threshold.
Unspecified datasets use the global value (default 1 eV). Values must be finite
and positive; duplicate or unselected dataset names are rejected.

For example, the following is a **prospective experiment**, not a set of
validated optimal thresholds:

```bash
python -m HERA.scripts.run_all_dataset_benchmark --seed 11 42 1245 --epochs 500 --early-stopping-patience 50 --early-stopping-min-delta-percent 0.5 --point-loss huber_ev --huber-delta-ev-by-dataset native=1 semi=0.1 imp2d=1 2dmd_low=0.01 2dmd_high=0.1 --device cuda:0 --native-preparation-dir HERA/logs/native_filter_le15_no_poscar0_3seeds --run-dir HERA/logs/all5_3seeds_huber_per_dataset_v1
```

Native keeps the 1 eV threshold from the existing paired experiment. The other
thresholds are exploratory starting values at different error scales; no
completed comparison establishes that they improve these datasets. They are
explicit CLI choices, not new defaults or automatic estimates from held-out
labels. Choose any subsequent tuning using validation MAE, keep the tuning
budget comparable across models, and report test MAE and RMSE for matched seeds.
The runner does not search thresholds automatically.

The threshold acts on the **prediction residual**, not the target energy itself,
and does not filter or delete samples. The loss remains `2 * Huber` in normalized
target space, with `delta_scaled = delta_eV / training_target_std`. Small errors
retain the existing MSE value and gradient. Lowering delta also reduces the
gradient cap for large errors; it is not simply an MAE switch.

Resolved thresholds appear in the launch output, `benchmark_protocol.json`,
the per-mode training config/checkpoints, and the aggregate JSON/Markdown
report. Changing a threshold requires a new `--run-dir`; both training and
`--summary-only` reject a different saved protocol. Equivalent uniform
thresholds remain compatible with existing global-Huber protocol files.

### What the existing evidence supports

The local native experiment at `logs/native_filter_then_split_le15_no_poscar0_v4/`
used the same filtered population, seed 123, and delta 1 eV:

| Model | MSE test MAE | Huber test MAE | MSE test RMSE | Huber test RMSE |
|---|---:|---:|---:|---:|
| Hetero-WAS | 0.050772 | 0.049216 | 0.161694 | 0.166580 |
| WAS-X | 0.058039 | 0.055427 | 0.152409 | 0.142423 |

MAE fell about 3.1% and 4.5%, respectively, but Hetero-WAS RMSE worsened.
This is one-seed evidence for those two models on native, not evidence of a
universal improvement or of an improvement for plain Hetero.

[PyTorch's HuberLoss documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.HuberLoss.html)
explains the quadratic/linear transition and reduced sensitivity to outliers.
That mechanism does not guarantee lower held-out MAE for these graph models.
Use paired MSE/Huber runs on the same data splits and three seeds to assess
whether any gain persists. Current results do not establish statistical
significance.

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
dataset/seed/epoch/patience/loss/threshold arguments when summarizing a customized run.
