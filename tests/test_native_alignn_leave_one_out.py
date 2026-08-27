import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from HERA.native_initial_relaxed_leave_one_out import (
    add_native_targets,
    aggregate_alignn_layernorm_comparison,
    aggregate_overall_mae,
    build_alignn_layernorm_comparison,
    eligible_materials,
    expand_leave_one_out_runs,
    masks_for_material,
    reset_discriminative_optimizer,
)
from HERA.native_ood_case_study import model_mode_display
from HERA.training.trainer import load_trusted_checkpoint


class NativeAlignnLeaveOneOutTests(unittest.TestCase):
    def test_trusted_checkpoint_loader_accepts_numpy_scaler_values(self):
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "checkpoint.pth"
            torch.save(
                {"model": {}, "scaler": {"mean": np.float64(1.0), "std": np.float64(2.0)}},
                path,
            )

            checkpoint = load_trusted_checkpoint(path, map_location="cpu")

        self.assertEqual(float(checkpoint["scaler"]["mean"]), 1.0)
        self.assertEqual(float(checkpoint["scaler"]["std"]), 2.0)

    def test_discriminative_finetuning_updates_backbone_and_head_at_different_rates(self):
        model = nn.Module()
        model.backbone = nn.Linear(4, 4)
        model.readout = nn.Sequential(nn.Linear(4, 2), nn.SiLU(), nn.Linear(2, 1))
        trainer = type("Trainer", (), {})()
        trainer.model = model
        trainer.config = {"optim": {"weight_decay": 1e-4}}

        head_name, n_backbone, n_head = reset_discriminative_optimizer(
            trainer, backbone_lr=1e-5, head_lr=1e-4
        )

        self.assertEqual(head_name, "readout")
        self.assertGreater(n_backbone, 0)
        self.assertGreater(n_head, 0)
        self.assertTrue(all(parameter.requires_grad for parameter in model.backbone.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in model.readout.parameters()))
        self.assertEqual(trainer.optimizer.param_groups[0]["name"], "backbone")
        self.assertEqual(trainer.optimizer.param_groups[0]["lr"], 1e-5)
        self.assertEqual(trainer.optimizer.param_groups[1]["name"], "prediction_head")
        self.assertEqual(trainer.optimizer.param_groups[1]["lr"], 1e-4)

    def test_native_targets_are_not_relabelled_and_only_final_structure_is_tested(self):
        metadata = pd.DataFrame(
            [
                {"material": "A", "defect_group": "A-g1", "configuration": "POSCAR0", "file": "a0"},
                {"material": "A", "defect_group": "A-g1", "configuration": "POSCAR1", "file": "a1"},
                {"material": "A", "defect_group": "A-g1", "configuration": "POSCAR2", "file": "a2"},
                {"material": "A", "defect_group": "A-g2", "configuration": "POSCAR0", "file": "a3"},
                {"material": "B", "defect_group": "B-g1", "configuration": "POSCAR0", "file": "b0"},
                {"material": "B", "defect_group": "B-g1", "configuration": "POSCAR1", "file": "b1"},
            ]
        )
        targets = np.array([5.0, 4.0, 3.0, 2.0, 8.0, 7.0])

        attached = add_native_targets(metadata, targets)

        np.testing.assert_allclose(attached["raw_target"], targets)
        self.assertEqual(attached.index[attached["is_final_relaxed"]].tolist(), [2, 5])
        masks = masks_for_material(attached, "A")
        self.assertEqual(masks["train_other"].tolist(), [4, 5])
        self.assertEqual(masks["finetune_poscar0"].tolist(), [0, 3])
        self.assertEqual(masks["test_final"].tolist(), [2])

        materials, table = eligible_materials(attached)
        self.assertEqual(materials, ["A", "B"])
        a_row = table[table["material"].eq("A")].iloc[0]
        self.assertEqual(int(a_row["n_initial"]), 2)
        self.assertEqual(int(a_row["n_final_structures"]), 1)

    def test_overall_mae_is_weighted_by_final_structure_count(self):
        summary = pd.DataFrame(
            [
                {
                    "material": "A",
                    "model": "alignn",
                    "mode": "full",
                    "model_mode": "Full ALIGNN",
                    "protocol": "direct__final_test",
                    "protocol_display": "Direct",
                    "n_test": 1,
                    "mae": 0.2,
                },
                {
                    "material": "B",
                    "model": "alignn",
                    "mode": "full",
                    "model_mode": "Full ALIGNN",
                    "protocol": "direct__final_test",
                    "protocol_display": "Direct",
                    "n_test": 3,
                    "mae": 0.4,
                },
            ]
        )

        overall = aggregate_overall_mae(summary)

        self.assertEqual(int(overall.iloc[0]["n_predictions"]), 4)
        self.assertAlmostEqual(float(overall.iloc[0]["overall_mae"]), 0.35)

    def test_builds_isolated_full_and_hetero_layernorm_runs(self):
        runs = expand_leave_one_out_runs(
            "alignn",
            ["full", "hetero"],
            radii=None,
            norm_values=["layernorm"],
        )

        self.assertEqual(
            [run["label"] for run in runs],
            ["full", "hetero_r0_norm_layernorm"],
        )
        self.assertEqual(runs[1]["config"]["model"]["hetero_node_norm"], "layernorm")
        self.assertEqual(runs[0]["config"]["task"], "alignn_full")
        self.assertEqual(runs[1]["config"]["task"], "alignn_hetero")

    def test_layernorm_label_is_human_readable(self):
        self.assertEqual(
            model_mode_display("alignn", "hetero_r0_norm_layernorm"),
            "Hetero (LayerNorm) ALIGNN",
        )

    def test_pairs_and_aggregates_alignn_mae(self):
        rows = []
        for material, baseline, hetero in (("GaN", 0.4, 0.3), ("AlN", 0.2, 0.25)):
            common = {
                "material": material,
                "model": "alignn",
                "protocol": "direct__final_test",
                "seed": 123,
                "rmse": 0.0,
                "ground_state_mae": 0.0,
                "top1_accuracy": 1.0,
                "ndcg": 1.0,
            }
            rows.append({**common, "mode": "full", "mae": baseline, "node_normalization": ""})
            rows.append(
                {
                    **common,
                    "mode": "hetero_r0_norm_layernorm",
                    "mae": hetero,
                    "node_normalization": "layernorm",
                }
            )

        comparison = build_alignn_layernorm_comparison(pd.DataFrame(rows))
        self.assertEqual(len(comparison), 2)
        gan = comparison[comparison["material"].eq("GaN")].iloc[0]
        self.assertAlmostEqual(gan["hetero_minus_alignn_mae"], -0.1)
        self.assertAlmostEqual(gan["hetero_relative_improvement_mae_percent"], 25.0)
        self.assertEqual(gan["mae_winner"], "HeteroALIGNN + LayerNorm")

        aggregate = aggregate_alignn_layernorm_comparison(comparison)
        self.assertEqual(int(aggregate.iloc[0]["n_pairs"]), 2)
        self.assertAlmostEqual(aggregate.iloc[0]["alignn_mae_mean"], 0.3)
        self.assertAlmostEqual(aggregate.iloc[0]["hetero_layernorm_mae_mean"], 0.275)
        self.assertAlmostEqual(aggregate.iloc[0]["hetero_mae_win_rate"], 0.5)

    def test_does_not_pair_non_layernorm_hetero_run(self):
        frame = pd.DataFrame(
            [
                {
                    "material": "GaN",
                    "model": "alignn",
                    "mode": "full",
                    "protocol": "p",
                    "seed": 123,
                    "mae": 0.4,
                    "node_normalization": "",
                },
                {
                    "material": "GaN",
                    "model": "alignn",
                    "mode": "hetero_r0_norm_batchnorm",
                    "protocol": "p",
                    "seed": 123,
                    "mae": 0.3,
                    "node_normalization": "batchnorm",
                },
            ]
        )
        self.assertTrue(build_alignn_layernorm_comparison(frame).empty)


if __name__ == "__main__":
    unittest.main()
