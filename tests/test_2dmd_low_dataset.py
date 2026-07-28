import unittest
from unittest.mock import ANY, call, patch

from HERA.config.defaults import VALID_DATASETS, get_config
from HERA.data.datasets import load_data_2dmd_low
from HERA.main import default_modes_for_model, modes_for_dataset


class TwoDMDLowDatasetTests(unittest.TestCase):
    def test_megnet_sparse_runs_first_for_all_2dmd_datasets(self):
        default_modes = default_modes_for_model("megnet")
        self.assertEqual(default_modes[0], "sparse")

        for dataset in ("vacancy", "2dmd_low", "2dmd_high"):
            with self.subTest(dataset=dataset):
                self.assertEqual(
                    modes_for_dataset(default_modes, dataset)[0],
                    "sparse",
                )
                self.assertEqual(
                    modes_for_dataset(
                        ["full", "attention", "sparse"],
                        dataset,
                    ),
                    ["sparse", "full", "attention"],
                )

        self.assertNotIn("sparse", modes_for_dataset(default_modes, "och"))

    def test_dataset_is_registered_with_low_density_batch_defaults(self):
        self.assertIn("2dmd_low", VALID_DATASETS)
        self.assertEqual(
            get_config("cgcnn", "2dmd_low", "full")["model"]["train_batch_size"],
            8,
        )
        self.assertEqual(
            get_config("megnet", "2dmd_low", "sparse")["model"]["train_batch_size"],
            8,
        )

    @patch("HERA.data.datasets.convert_to_sparse_2dmd_high")
    @patch("HERA.data.datasets.CifParser")
    @patch("HERA.data.datasets.get_prepared")
    def test_loader_uses_only_low_density_mos2_and_wse2(
        self,
        get_prepared,
        cif_parser,
        convert,
    ):
        def add_sample(path, prepared, is_high=False):
            material = path.rsplit("/", 1)[-1]
            prepared["id"].append(material)
            prepared["structure"].append(f"{material}-structure")
            prepared["base"].append(material)
            prepared["cell"].append("[1, 1, 1]")
            prepared["target"].append(1.0)
            prepared["weight"].append(3.7165)

        get_prepared.side_effect = add_sample
        cif_parser.return_value.get_structures.return_value = ["unit-cell"]
        convert.side_effect = lambda structure, *_args, **_kwargs: structure

        dataset_full, hetero, attention, sparse, targets = load_data_2dmd_low(
            "cgcnn",
            representations=["full"],
        )

        root = "dataset/2d-materials-point-defects-all/low_density_defects"
        self.assertEqual(
            get_prepared.call_args_list,
            [
                call(f"{root}/MoS2", ANY),
                call(f"{root}/WSe2", ANY),
            ],
        )
        self.assertEqual(dataset_full, ["MoS2-structure", "WSe2-structure"])
        self.assertIsNone(hetero)
        self.assertIsNone(attention)
        self.assertIsNone(sparse)
        self.assertEqual(targets.tolist(), [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
