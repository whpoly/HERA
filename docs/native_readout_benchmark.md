# Native: defect / global / defect + global readout

2026-09-29. This is a prepared server experiment. No local training or measured
readout ranking has been produced by this change.

## What changes

All three variants use HeteroALIGNN, with the same node/edge construction,
shared-residual messages (rank 8), LayerNorm, AA kept, DD dropped and r=0.
The new options do not change existing model defaults or old checkpoints.

| `--alignn-hetero-pooling` | Input to prediction | Head at hidden=64 |
|---|---|---|
| `defect_energy_mean` | Mean of per-actual-defect predictions | 64 -> 64 -> 32 -> 1 |
| `global_mean` | Mean of all final node embeddings | 64 -> 64 -> 32 -> 1 |
| `defect_global_mean` | Concatenated actual-defect mean and all-node mean | 128 -> 64 -> 32 -> 1 |

Native has one actual defect per structure, so `defect_energy_mean` and
`defect_mean` are equivalent on this dataset. The first variant preserves the
current native baseline's setting. This experiment does not compare those two
equivalent operations.

Global means are calculated over the union of node stores, with equal weight
per node. They include the same X vacancy placeholders already present in the
heterograph. They are not equally weighted means of the two store means.
The defect branch uses `pool_type`, excluding pristine sites promoted into a
larger defect region. Concatenation keeps this independent defect branch.

All backbone parameters start identically for the same seed. Defect-only and
global-only also start with identical head parameters. Concatenation changes
the head input width and adds 4096 parameters; the head's initialization changes
accordingly. This is a practical readout comparison, not a parameter-count-
matched attribution of every difference to embedding information.

## Server command

Sync the changed files to the server, activate the existing HERA environment,
and execute from the **parent directory of HERA**, where the normal relative
`dataset/Dataset_1/Dataset_1/A_rich/Neutral` path is available:

```bash
python -m HERA.scripts.run_native_readout_benchmark --seed 123 --epochs 500 --device cuda:0 --run-dir HERA/logs/native_readout_benchmark
```

This runs three fresh, sequential readout experiments for ordinary `hetero`.
Defaults follow the current native v4 protocol: batch 16, validation/test batch
1, 500 epochs, no early stopping, AdamW and validation-based LR scheduling.
The best epoch is still selected by validation MAE. Set
`--early-stopping-patience 50` to enable early stopping consistently for all
variants in a **new output root**.

The runner prepares one strict filter manifest in the experiment root, using
the original seed-specific splits and training-fitted thresholds. Every
variant receives that same frozen manifest and uses `reference_v1` preprocessing.
On the currently audited seed-123 sources this retains 1818/604/604
train/validation/test samples. Actual counts are recorded in the results.

To test genuine WAS inputs too, append `--mode hetero hetero_was` and use a new
output root. Results and recommendations are separate for the two feature modes.
For three seeds and both modes (18 training runs):

```bash
python -m HERA.scripts.run_native_readout_benchmark --mode hetero hetero_was --seed 123 11 1245 --epochs 500 --device cuda:0 --run-dir HERA/logs/native_readout_benchmark_multiseed
```

Preview commands without preprocessing or training:

```bash
python -m HERA.scripts.run_native_readout_benchmark --dry-run
```

## Outputs and interpretation

The runner isolates outputs under each pooling name, then standard compact
directories such as `global_mean/alignn/native/hetero_no_dd/`. It writes:

- `readout_protocol.json`: experiment settings; incompatible reuse is rejected.
- `native_filter_manifest.json`: common frozen preprocessing/splits.
- `readout_comparison.csv`: paired per-seed validation/test MAE, epochs and counts.
- `readout_comparison.md`: means, test standard deviations and validation-selected readout.
- `readout_selection.json`: machine-readable recommendation and limits.

Before ranking, it checks ordered train/validation/test IDs, non-readout config,
filter identity and training budget across all three variants for each seed.
Missing, mismatched or invalid results stop comparison. The recommendation
minimizes **mean validation MAE**, never test MAE. Test results are reported for
all predeclared alternatives. A one-seed comparison is preliminary; multiple
seeds vary both initialization and split in the current native protocol.
The random structure split measures within-dataset interpolation and does not
establish unseen-material generalization.

Re-running the same command skips completed runs. Incomplete training restarts
from scratch, following the existing `--resume` behavior; optimizer continuation
is not available. Changing seeds, modes, epochs or batch size requires a new
experiment root. To regenerate summaries without loading datasets or training,
pass the original modes/seeds/output root plus `--summary-only`.

## Local verification

Only CPU synthetic forward/backward checks and mocked CLI/report checks are
used locally. They cover node weighting, actual-defect masks, empty stores,
batch isolation, host gradients, identical backbone initialization, checkpoint
restoration, command dispatch, split/config mismatch rejection and validation-
only selection. These checks do not provide native accuracy measurements.

```bash
python -m unittest HERA.tests.test_hetero_global_readout HERA.tests.test_native_readout_benchmark HERA.tests.test_hetero_defect_energy_mean HERA.tests.test_hetero_alignn_transfer HERA.tests.test_compact_logs
```
