# Native training-filter benchmark

The latest requested protocol filters the entire population first, freezes it,
then creates fresh 60/20/20 splits. Use `--suite filtered_dataset`; see
[the all-splits experiment](native_filtered_dataset.md). Earlier protocols
below retain their original evaluation populations.

2026-10-12. Server experiment prepared; no new model has been trained locally.

The follow-up `--suite outlier_control` tests train-only same-family MAD
filtering crossed with MSE/physical-unit Huber, for both models (eight runs).
See [the outlier-control experiment](native_outlier_control.md).
The default `energy_scope` suite below retains its original protocol and paths.

## Why test further filtering?

The paper explicitly removes notable parity-plot outliers and retrains in
section 5.2, for **Datasets 1+2+3+4 combined**. The resulting neutral A-rich
test RMSE is 0.32 eV (Table 3). Native-only 0.15 eV is Table 2, a different
experiment; the text does not provide the IDs/thresholds needed to reproduce
its exact training population. The authors also include intermediate and
unoptimized structures deliberately. Large prediction error or an early
POSCAR index therefore does not establish a bad DFT label.

[Paper](https://arxiv.org/html/2309.06423v2) ·
[Author notebook](https://github.com/msehabibur/defect_GNN_gen_1/blob/main/Codes/Defect_GNN_Notebook.ipynb)

The public native labels span a wider range than the paper's stated 0–15 eV.
Some notebook analyses use a 15 eV bound, but these do not establish the
native Table 2 preprocessing. This benchmark tests **one fixed upper-bound
hypothesis**, not a faithful reproduction of the paper's outlier removal.
Negative energies remain eligible; the condition is exactly `E_f < 15 eV`.

## Existing-checkpoint sensitivity (no retraining)

The current strict policy already removes 44 of 3070 structures. Its seed123
split is 1818 train / 604 validation / 604 test. Re-scoring the cached fixed
checkpoints on a restricted population gives:

| Model | Val RMSE: all 604 | Val RMSE: <15, 583 | Test RMSE: all 604 | Test RMSE: <15, 573 |
|---|---:|---:|---:|---:|
| Hetero-WAS no_dd | 0.230631 | 0.234721 | 0.373926 | 0.324894 |
| WAS_X | 0.188490 | 0.191691 | 0.394296 | 0.309982 |

Units are eV. The changed scores measure a changed evaluation population,
not a learned-model improvement. Restriction even reverses the test RMSE
ordering of these two models and slightly increases both validation RMSEs.
Thus lower test RMSE alone cannot validate a cleaning rule.

The largest 20 Hetero-WAS test errors contribute 91.06% of squared error.
However, some large-error structures have ordinary minimum covalent-distance
ratios: `42-CdO-O_Cd-POSCAR0-Neutral.cif` has E_f=8.96 eV, absolute error
2.445 eV and minimum d/(r_i+r_j)=1.042; `96-BP-B_i_B-POSCAR0-Neutral.cif`
has E_f=7.531 eV, absolute error 2.343 eV and ratio 1.031. These observations
do not establish atomic overlap or an erroneous energy. OUTCAR/vasprun.xml
convergence records are unavailable locally. The existing same-family
near-geometry/energy-conflict diagnostic found no flagged pairs; it is not
an exhaustive symmetry-equivalence or DFT-convergence check.

Audit values: `results/native_filter_reassessment_20261012/summary.json`.
The earlier full paper/data comparison is
`results/native_paper_gap_20260929/report.md`.

## Four-run server experiment

Use two training populations for each of Hetero-WAS no_dd and WAS_X:

| Population | Train, seed123 | Validation | Test |
|---|---:|---:|---:|
| strict | 1818 | 604 | 604 |
| strict + train-only E_f < 15 eV | 1729 | 604 | 604 |

Only the training restriction changes. Original membership and source order,
strict thresholds, hidden width 64, readout, optimizer settings, batch size 16,
500 epochs and full-strict-validation-MAE checkpoint selection are preserved.
Each training population fits its own target scaler from its own training
labels. Fewer training samples mean fewer optimizer updates at the same epoch
budget. Both models share each population and scaler.

Every checkpoint is evaluated on the identical full validation/test sets,
then reported separately for E_f < 15 and E_f >= 15. MAE and RMSE are both
computed from saved per-structure predictions. Compare the two training
populations **within the same evaluation scope** to assess retraining.
The >=15 group becomes energy extrapolation for the restricted-training model;
its errors remain included in the full score.

This is an exploratory experiment motivated by an already inspected dataset,
not an independent confirmation of an optimal filtering threshold. No test
residual determines which training sample is retained. All four runs are
reported; the script does not select a winner using test scores or search
additional bounds. It does not alter the default strict filter or existing
benchmark directories. This native experiment retains all native defect types;
the separate IMP2D interstitial-only protocol remains unchanged.

Synchronize the new script with the existing HERA code, activate the normal
training environment, and run from **HERA's parent directory**:

```bash
python -m HERA.scripts.run_native_filter_benchmark --seed 123 --epochs 500 --device cuda:0 --run-dir HERA/logs/native_training_filter_v1
```

The command prepares both frozen manifests, runs four trainings sequentially
on the specified GPU, then evaluates and writes:

- `filter_protocol.json`: fixed experiment settings;
- `strict/native_filter_manifest.json` and `train_lt15/native_filter_manifest.json`;
- per-model checkpoints and `seed123_filter_predictions.csv`;
- `filter_comparison.csv` / `filter_comparison.md`: all validation/test MAE/RMSE scopes;
- `filter_evaluation.json`: complete metrics, checkpoint hashes and source identity.

Use `--dry-run` to print commands without writes, preprocessing, inference or
training. Use `--prepare-only` to generate and verify manifests only.
Use `--evaluate-only` with the same arguments after all four trainings finish.
Re-running the main command skips compatible completed trainings through the
existing `--resume` behavior; interrupted runs restart rather than restore
optimizer state. Changed protocols require a new output directory.

Local verification: 30 tests passed across the new experiment, existing native
filter and auto-filter tests. The real-data prepared manifests preserve the
604/604 held-out IDs. An evaluator smoke check on three existing test structures
per model reproduced both cached checkpoints' predictions exactly on CPU;
it did not train a model or measure new model accuracy.
