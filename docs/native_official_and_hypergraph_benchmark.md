# Native: official ALIGNN and paired HyperALIGNN

This is an independent reproduction of Rahman et al., DOI 10.1063/5.0176333,
not inference from the authors' native-only trained weights. The exact Table 2
native checkpoint, split IDs, dependency lockfile and training seed have not
been identified in the published repository.

The user authorized local execution on 2026-09-29. The workstation has an
RTX 5060 Ti 16 GB. Jobs are sequential; original HERA results are preserved.

## Experiments

| Run | Data/split | Configuration |
|---|---|---|
| Official raw | All 3070 published neutral A-rich rows; Python shuffle seed123, 1842/614/614 | Official ALIGNN 4+4, hidden256, embedding64, cutoff8, neighbors12, distance80, angles40, batch16, 150epochs |
| HyperALIGNN paired | Exact `alignn_v4/hetero_was_no_dd` IDs: 1818/604/604 | WAS184, hidden64, 3+3, cutoff6, neighbors12, distance40, batch16, 500epochs; copies hetero optimizer and scaler protocol |
| Official paired | Same 1818/604/604 IDs as HERA | Same official configuration as raw |

The published label table and every original CIF were verified byte-identical
to the local inputs. Official raw uses no target threshold or outlier removal.
The paired experiments use the already frozen strict filter, identity
`55c62880e3e0806eb3ca08c6ab2552b3ce6231fe698a41cb68614d032be1de1b`.

HyperALIGNN uses `defect_global_attention_v3`, `defect_energy_mean`,
`local_global` updates and radius3 Å. These are hypergraph-specific choices;
the comparison does not imply its messages are identical to hetero messages.
It trains with MSE, selects validation MAE, and disables early stopping,
matching the reference. All held-out samples contribute to both MAE and RMSE.

## Official implementation and compatibility

Official source: https://github.com/usnistgov/alignn/tree/d1415cf824a7d6edd7a044446ae7ddbfdf55316a
(release v2023.08.01, prior to the paper). The precise version used by the
authors is uncertain: their example `version` points to a 2021 commit that
does not support several configuration fields in that same example.

The wrapper rejects changed tracked upstream source. It imports the official
DGL graph builder, model and trainer; it does not substitute HERA's ALIGNN.
The full model has 4,026,753 parameters.

The local `.venvs/alignn-paper` inherits the existing HERA Python/PyTorch
runtime without changing it. Added packages include jarvis-tools2023.8.10,
pydantic1.10.26, pytorch-ignite0.5.2 and DGL2.2.1+cu121. Exact runtime package
versions are recorded per run. This is not the authors' historical software
environment. `--disable-graphbolt` bypasses only the unused GraphBolt import,
whose binary is incompatible with PyTorch2.11; native DGL CUDA kernels are
used unchanged. The JARVIS native CIF parser is selected explicitly because
cif2cell is absent. Raw defective supercells are not replaced with HERA WAS
or defect-labelled representations.

The original trainer selects best validation MAE and drops the final partial
validation batch. Both behaviors are preserved and documented. Evaluation
after training covers the **complete** validation/test sets, for both the
best-validation-MAE checkpoint and the final epoch. Neither is selected by
test results. This avoids accidentally presenting final-epoch predictions as
best-checkpoint predictions. A two-epoch 64-structure smoke run verifies the
training/checkpoint/scoring path; it is not a benchmark result.

## Commands and outputs

Prepare manifests without training, from HERA:

```powershell
python scripts/run_official_alignn_native.py prepare --run-dir results/official_alignn_native_20260929
```

Official training, in the isolated environment:

```powershell
.venvs/alignn-paper/Scripts/python.exe -u scripts/run_official_alignn_native.py train --manifest results/official_alignn_native_20260929/raw_seed123/manifest.json --alignn-source tmp/official_alignn_native/source --epochs 150 --disable-graphbolt
```

HyperALIGNN, from HERA's parent using the HERA environment:

```powershell
python -u -m HERA.scripts.run_native_hypergraph_paired
```

`results/official_alignn_native_20260929/queue_status.json` tracks the submitted
local queue. Each official run writes `training/metrics.json`, predictions,
checkpoints, history and provenance. HyperALIGNN writes equivalent artifacts
under `results/native_hypergraph_paired_20260929`.

One seed can test whether this configuration reaches the claimed scale on
these data. It cannot establish or refute the exact published value without
the original split, weights and preprocessing details.

## Prepared follow-up (not trained)

The user requested a native variant with global defect messages disabled and
a larger local hyperedge radius. A local-only, radius6 Å protocol is prepared
in a separate output directory. See
[the follow-up protocol](native_hypergraph_local_radius_ablation.md)
for its precise masking semantics, validation and future training command.
