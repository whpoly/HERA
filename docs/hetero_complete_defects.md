# HeteroALIGNN: connect every actual defect pair

2026-09-15. Opt-in topology ablation:
`--alignn-hetero-defect-connectivity complete`.

- Start with the existing 6 Å physical graph and per-type neighbor cap.
- For every pair of distinct actual defects (`pool_type=1`, including vacancy
  dummy sites), append any missing directed edge in both directions. Added
  pairs bypass both the distance cutoff and the neighbor cap.
- Use periodic minimum-image distances and Cartesian displacement vectors.
  No new self-loops are added. Existing periodic image edges and their order
  remain intact, so this is completion of a periodic multigraph, not removal
  of existing image edges.
- At r=0 these are ordinary `dd` edges. They enter all normal ALIGNN
  atom/bond updates and the line graph's angle updates. This differs from the
  separate, final-layer `sparse_residual` ablation.
- Only actual defects are completed; nearby pristine sites promoted by r>0
  are excluded. Node features, model parameters, readout and loss do not change.
- ALIGNN recomputes radial features from the vectors using the baseline's
  40 Gaussian centers spanning 0–6 Å (sigma = 6/39 Å). Features of edges
  well beyond 6 Å approach zero. This first
  experiment isolates graph completion, without simultaneously changing the
  radial basis. Long-range distance discrimination would require another ablation.

Missing saved fields mean `physical`, preserving old graph conversion and
checkpoint behavior. New checkpoints retain the connectivity setting. Output
paths append `defect_edges_complete`, preventing baseline resume collisions.

## Standard training command

Run from HERA's parent directory:

```bash
python -m HERA.main --model alignn --dataset 2dmd_wse2 --mode hetero --r 0 --alignn-hetero-feature-norm layernorm --alignn-hetero-pooling defect_energy_mean --alignn-hetero-relations shared_residual --alignn-hetero-adapter-rank 8 --alignn-hetero-defect-connectivity complete --seed 123 --epochs 500 --device cuda:0 --resume --run-dir HERA/logs/hetero_complete_defects_wse2
```

WSe2 has 4,746 low training samples, 1,187 low validation samples, and 500
fixed high test samples. High targets do not select epochs or hyperparameters.
Default training uses batch 8, FP32, AdamW and the existing patience-50 early
stopping rule with 0.5% relative validation improvement.

## Matched local experiment

The existing seed-123 baseline under `logs/hetero_energy_mean_benchmark/`
has high MAE **0.05218848 eV/defect**, best low validation MAE 0.00356877,
best epoch 243 and early stopping at epoch 293.

```bash
python -m HERA.scripts.run_wse2_complete_defects
```

This runner copies the saved baseline configuration, changes only connectivity,
and checks the ordered train/validation/test source IDs before training. It
uses the existing trainer and writes `status.json`, `comparison.json`,
`comparison.md` and `paired_predictions.csv` under
`logs/hetero_complete_defects_wse2/`. The checkpoint, history, prediction CSV
and summary use the standard nested model directory. Use
`--baseline-checkpoint` and `--run-dir` to override the local paths.
Runtime/environment differences from the saved baseline may affect the
single-seed comparison. Resume skips completed runs; it does not restore
interrupted optimizer progress.

## Completed result (2026-09-16)

The seed-123 run completed after 262 epochs, selecting epoch 258 by low
validation MAE. Runtime was 13.33 hours. Ordered train/validation/test source
IDs matched the saved baseline; all 500 high samples were evaluated.

| Variant | Best low validation MAE | WSe2 high MAE (eV/defect) |
|---|---:|---:|
| Physical baseline | 0.00356877 | 0.05218848 |
| Complete defect edges | 0.00300696 | 0.06351766 |

High MAE increased by **21.71%** (+0.01132918 eV/defect), despite improved low
validation MAE. Only 159/500 high samples had lower absolute error. This
single-seed topology ablation did not improve low-to-high transfer. The retained
short-range radial basis described above limits what this result establishes
about other possible complete-graph designs.

Artifacts: `logs/hetero_complete_defects_wse2/comparison.md`, `comparison.json`
and `paired_predictions.csv`. The standard nested result directory retains
the full history, best checkpoint and per-sample test predictions.

## Validation

All 192 repository unit tests pass, including new tests for completion beyond
12 neighbors, no duplicated existing pairs, periodic skew-cell geometry,
promoted pristine exclusion, no-defect/single-defect graphs, batched indices,
finite gradients, unchanged model initialization, saved config reconstruction,
separate output paths and CLI dispatch.
An additional runner test verifies final report generation and rejection of a
mismatched validation split without launching benchmark training.

Full-size CUDA checks used eight low WSe2 structures and two high structures
with 24 defects. The latter changed from 92/86 physical dd edges to 552 edges
each. Three low-only FP32 optimization steps passed. Batched versus individual
predictions differed by at most 2.38e-7. Peak allocated GPU memory was 2.39 GiB.
The model still has 679,193 parameters. The engineering check does not provide
a trained high-test MAE; read the completed experiment's comparison report.
