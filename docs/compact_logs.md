# Compact experiment directories

Add `--compact-logs` to `HERA.main` to write each mode directly into
`<run-dir>/<model>/<dataset>/<mode>/`. For example:

```text
HERA/logs/alignn_ref_v1/
  alignn/
    native/
      hetero/
        config.json
        seed123_history.csv
        seed123_best_checkpoint.pth
        summary.txt
      hetero_was/
      attention/
      ...
    semi/
    imp2d/
```

The radius, normalization, relation settings and preprocessing version are
stored in `config.json`, together with the full run label and training protocol.
Seeds share a mode directory and retain their individual filenames. Dataset
and model summaries still include the full configuration labels.

## ALIGNN on native, semi and imp2d, without hypergraph

From HERA's parent directory, in the training environment on the machine
holding the complete datasets:

```bash
python -m HERA.main --model alignn --dataset native semi imp2d --mode full full_x attention attention_was hetero hetero_was was_x definet definet_was --r 0 --native-preprocessing reference_v1 --semi-preprocessing reference_v1 --imp2d-preprocessing reference_v1 --alignn-hetero-feature-norm layernorm --alignn-hetero-node-norm layernorm --alignn-hetero-relations shared_residual --alignn-hetero-adapter-rank 8 --alignn-hetero-pooling defect_energy_mean --batch-size 8 --test-batch-size 1 --seed 123 11 1245 --epochs 500 --device cuda:0 --atom-init HERA/atom_init.json --run-dir HERA/logs/alignn_ref_v1 --compact-logs --resume --protect-existing
```

This selects 75 dataset/mode/seed tasks: nine native modes and eight modes each
for semi and imp2d, over three seeds. `full_x` duplicates `full` without
vacancies and is skipped for semi and imp2d. At `r=0`, `hetero_fixed_pool` is
identical to `hetero`, so it is omitted. All three datasets use `reference_v1`
preprocessing for ordinary and WAS modes.

## Resume and configuration changes

- Repeat the same command to skip completed seeds and run missing ones.
  Additional seeds can be added in the same directory.
- `--resume` reuses completed results; it does not restore an interrupted
  optimizer or epoch. `--protect-existing` stops on incomplete saved runs.
- A changed configuration or training protocol in the same mode directory is
  rejected. Use a different `--run-dir` for a different experiment.
- Select one radius, normalization and ablation per mode. For sweeps, use
  separate run roots or omit `--compact-logs` to retain configuration directories.
- Existing directories without `config.json` are not adopted or moved. To
  resume old nested runs, keep their original command and layout. Use a new
  root for a fresh compact run.

The relation benchmark runner also accepts `--compact-logs`; it retains the
`baseline`, `no_dd` and other selected variant directories above the model
directory. Without this flag, the existing detailed layout is unchanged.
