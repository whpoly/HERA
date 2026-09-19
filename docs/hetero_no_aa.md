# HeteroALIGNN without the atom-to-atom relation

2026-09-17. `--alignn-hetero-aa keep` retains the original graph and model;
`--alignn-hetero-aa drop` removes `('atom', 'aa', 'atom')`. Omitting the flag,
including in old saved configs, means `keep`.

## What changes

- Remove the entire aa edge store, its distance encoder, and its relation
  message network or residual adapter. The shared message core remains.
- ALIGNN's line graph contains only retained bonds. Angles incident to aa
  bonds disappear as well. This is graph removal, not a message mask.
- Each ordinary-node fusion receives only the da relation instead of aa and
  da, so its input projection is smaller. Defect-node fusion keeps ad and dd.
- Keep every node, its features, its pooling mask, and the exact selected
  ad/da/dd edges, distances, vectors and ordering. Neighbor selection happens
  before aa filtering; its quota is not reassigned to other relations.
- Ordinary atoms can still affect defects through ad, and the D -> A -> D
  path remains. Nodes with no retained neighbors keep their root updates.

The flag acts on graph node types. Use `--r 0` for A = ordinary site and
D = actual defect. With larger r, D also contains atoms in the defect region;
removing aa then means removing edges within the remaining atom store.

This option supports independent, shared and shared_residual HeteroALIGNN.
The commands below use the discussed shared_residual rank-8 model with
LayerNorm and `defect_energy_mean`. At width 64, 3 ALIGNN + 3 GCN blocks:

| Variant | Parameters |
|---|---:|
| Keep aa | 679,193 |
| Drop aa | 606,819 |

The reduction of 72,374 parameters is 7,040 for the aa distance encoder,
16,182 for six aa adapters, and 49,152 for the smaller atom fusion inputs.
The new model must be trained from scratch: the old full model's state dict
has extra keys and larger fusion weights. Each version restores its own
checkpoint strictly using the saved `hetero_aa_mode`.

## Mixed-low training, then both material high tests

This matches the training domain of the preceding concentration-average
explanations. Run in PowerShell from HERA's parent directory. Both variants
use seed 123, the same split and optimizer protocol, and up to 500 epochs
with the existing validation-based early stopping. The default batch sizes
are 8 for training and 1 for validation/testing, matching the saved baseline.

```powershell
Set-Location C:/Users/User/Desktop
$py = 'C:/Users/User/.conda/envs/hera/python.exe'

# Original: four relations
& $py -m HERA.main --model alignn --dataset 2dmd_low --mode hetero --r 0 --alignn-hetero-feature-norm layernorm --alignn-hetero-pooling defect_energy_mean --alignn-hetero-relations shared_residual --alignn-hetero-adapter-rank 8 --alignn-hetero-aa keep --seed 123 --epochs 500 --device cuda:0 --run-dir HERA/logs/hetero_aa_ablation/keep --resume
if ($LASTEXITCODE -ne 0) { throw 'Original model training failed' }

# New: ad, da, dd only
& $py -m HERA.main --model alignn --dataset 2dmd_low --mode hetero --r 0 --alignn-hetero-feature-norm layernorm --alignn-hetero-pooling defect_energy_mean --alignn-hetero-relations shared_residual --alignn-hetero-adapter-rank 8 --alignn-hetero-aa drop --seed 123 --epochs 500 --device cuda:0 --run-dir HERA/logs/hetero_aa_ablation/drop --resume
if ($LASTEXITCODE -ne 0) { throw 'No-aa model training failed' }

# Evaluate BOTH saved variants on MoS2 and WSe2 high test sets
& $py -m HERA.predict_2dmd_low_checkpoints --checkpoint-root HERA/logs/hetero_aa_ablation --model alignn --mode hetero --material mos2 wse2 --seed 123 --device cuda:0 --test-batch-size 1
if ($LASTEXITCODE -ne 0) { throw 'High-test evaluation failed' }
```

On another machine, use its HERA parent directory and activated environment's
`python` instead of this machine's `$py` path. Sync the changed code first.
Do not change `keep` to `drop` inside an existing checkpoint config and try to
reuse its weights; these commands create and train two distinct architectures.

Training roots are separate, so the overall summaries, histories, checkpoints
and predictions of both versions are retained. Drop-aa also gets a
`relations_no_aa` path component and run-label suffix. Existing logs outside
this new run directory are untouched. `--resume` skips splits with a completed
`TEST` history row. Incomplete splits restart from epoch 1; their old history
is backed up. It does not restore optimizer/epoch state.

High-test summaries (both models, 500 structures per material) are written to:

```text
HERA/logs/hetero_aa_ablation/high_test_predictions/2dmd_mos2/test_summary.csv
HERA/logs/hetero_aa_ablation/high_test_predictions/2dmd_wse2/test_summary.csv
```

## Optional: separate material-specific low-to-high training

If comparing models trained separately for each material, replace
`--dataset 2dmd_low` in BOTH training commands with
`--dataset 2dmd_mos2 2dmd_wse2` and use new roots such as
`HERA/logs/hetero_aa_single_material/keep` and `.../drop`.
These dataset modes automatically evaluate their own fixed high test set at
the end of training; do not use the mixed-low checkpoint evaluator for them.
They are a different training domain from the concentration-average figures.

## Validation and interpretation

77 related unit/integration tests passed, including exact preservation of
ad/da/dd, removal of aa bonds and their incident angles, empty-edge graphs,
batched versus separate predictions, gradients, all three parameter-sharing
modes, both aggregation modes, both distance encoders, strict restoration,
legacy keep behavior and separate CLI result paths.

On RTX 5060 Ti, both full-size versions passed two optimizer steps on four
real structures: MoS2 and WSe2, each with 4 and 24 defects. Predictions were
consistent between batched and separate evaluation, and gradients were finite.
The existing mixed-low checkpoint strictly restored; its predictions differed
from the previous results by at most 2.39e-7 eV/defect. Details are saved in
`results/hetero_no_aa_validation_20260917.json`.

No full retraining or new accuracy benchmark has been run for this change.
A small signed mean intervention effect can include positive/negative
cancellation. Moreover, the previous intervention retained aa edge updates
and geometry, while this version removes them. Whether aa can be omitted
without hurting accuracy must be determined from the retrained test results.
