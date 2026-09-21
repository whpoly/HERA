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

ALIGNN relation ablations use sibling mode directories: `hetero_no_dd`,
`hetero_was_no_dd`, `hetero_no_aa`, or `hetero_no_aa_dd`. Their saved model mode
is still `hetero`/`hetero_was`; the suffix only separates output files.

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

## Add no_dd in the same run root

Keep `--run-dir HERA/logs/alignn_ref_v1` and add `--alignn-hetero-dd drop`.
After updating the code, baseline and no_dd can coexist without extra levels:

```text
HERA/logs/alignn_ref_v1/alignn/native/
  hetero/             existing DD=keep
  hetero_was/         existing DD=keep
  hetero_no_dd/       new DD=drop
  hetero_was_no_dd/   new DD=drop
  attention/         existing results
```

The same arrangement applies to semi and imp2d. To run only the two no_dd modes:

```bash
python -m HERA.main --model alignn --dataset native semi imp2d --mode hetero hetero_was --r 0 --native-preprocessing reference_v1 --semi-preprocessing reference_v1 --imp2d-preprocessing reference_v1 --alignn-hetero-feature-norm layernorm --alignn-hetero-node-norm layernorm --alignn-hetero-relations shared_residual --alignn-hetero-adapter-rank 8 --alignn-hetero-pooling defect_energy_mean --alignn-hetero-dd drop --batch-size 8 --test-batch-size 1 --seed 123 11 1245 --epochs 500 --device cuda:0 --atom-init HERA/atom_init.json --run-dir HERA/logs/alignn_ref_v1 --compact-logs --resume --protect-existing
```

Alternatively, retain the full non-hypergraph mode list from the previous
command and add `--alignn-hetero-dd drop`: completed matching full/attention/etc.
results are reused in place, and only missing requested tasks are trained.
Keeping DD and removing DD are different architectures, so completed baseline
hetero weights cannot serve as completed no_dd results. Dataset/model summaries
retain both variants with their full configuration labels.

For compatibility, an earlier compact experiment that already stored DD=drop
under plain `hetero/` or `hetero_was/` keeps that location when the corresponding
suffixed directory does not yet exist. Its manifest must identify the same
AA/DD variant, and the full configuration checks still apply. No files are moved.

## Resume and configuration changes

- Repeat the same command to skip completed seeds and run missing ones.
  Additional seeds can be added in the same directory.
- `--resume` reuses completed results; it does not restore an interrupted
  optimizer or epoch. `--protect-existing` stops on incomplete saved runs.
- AA/DD variants use separate sibling mode directories as above. Other changed
  configurations or training protocols in the same mode directory are rejected;
  use a different `--run-dir` for those experiments.
- Select one radius, normalization and ablation per mode. For sweeps, use
  separate run roots or omit `--compact-logs` to retain configuration directories.
- Existing directories without `config.json` are not adopted or moved. To
  resume old nested runs, keep their original command and layout. Use a new
  root for a fresh compact run.

The relation benchmark runner also accepts `--compact-logs`; it retains the
`baseline`, `no_dd` and other selected variant directories above the model
directory. Without this flag, the existing detailed layout is unchanged.
