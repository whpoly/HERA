# HeteroALIGNN: predict each defect contribution before averaging

2026-09-13. New opt-in readout `defect_energy_mean` computes
`mean(readout(h_i))` over actual defects. Existing `defect_mean` computes
`readout(mean(h_i))` and remains the default.

Both options use the same shared per-node MLP, `64 -> 64 -> 32 -> 1`, with
the same parameter shapes and initialization. The message backbone, graph
construction, normalization, labels, target scaler and training loss stay the
same. The graph target remains formation energy divided by defect count.
The new head learns latent contributions from graph labels, without requiring
or claiming unique per-defect energy labels. Target inverse scaling still
happens once on the graph prediction through the existing trainer.

The actual-defect `pool_type` mask is applied before the head. Pristine sites
included in an enlarged defect region do not enter the readout directly.
Each graph uses its own actual defect count; empty-defect graphs are rejected.
All relation modes (`independent`, `shared`, `shared_residual`) are supported.
Saved configs restore their selected readout; old defaults are preserved.

Run from the parent directory of HERA after syncing the modified code:

```bash
python -m HERA.main --model alignn --dataset 2dmd_mos2 --mode hetero --r 0 --alignn-hetero-feature-norm layernorm --alignn-hetero-relations shared_residual --alignn-hetero-pooling defect_energy_mean --seed 123 --epochs 500 --device cuda:0 --run-dir HERA/logs/2dmd_mos2_hetero_readout_ablation
```

For the matched old readout, replace `defect_energy_mean` with `defect_mean`.
If the preceding run used pure `shared`, keep `--alignn-hetero-relations shared`
in both commands. Changing the relation mode at the same time would no longer
isolate the readout effect.

Results are separated under `pool_defect_energy_mean` and `pool_defect_mean`,
including native run labels and combined prediction columns. The change is
an ablation, not evidence of improved low-to-high accuracy.

Tests cover nonlinear composition and gradient weighting, identical backbone
initialization, actual-defect masks, batches with unequal defect counts,
host-message gradients, empty relations, restored configs and separate paths.

Validation: all 153 unit tests passed. Both shared modes also passed two CUDA
FP16 forward/backward optimizer steps on synthetic graphs using the full
64-dimensional, 3 ALIGNN + 3 GCN configuration. Batched versus single-graph
prediction differences were at most 2.98e-8. No full training was run.
