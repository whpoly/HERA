import unittest

from HERA.config.defaults import VALID_MODES, get_config
from HERA.data.datasets import (
    dataset_index_for_mode,
    representation_for_mode,
)


class MEGNetSparseReproductionTests(unittest.TestCase):
    def test_sparse_mode_is_registered(self):
        self.assertIn("sparse", VALID_MODES)
        self.assertEqual(representation_for_mode("sparse"), "sparse")
        self.assertEqual(dataset_index_for_mode("sparse"), 3)

    def test_original_sparse_model_uses_current_training_protocol(self):
        config = get_config("megnet", "vacancy", "sparse")
        model = config["model"]
        optim = config["optim"]

        self.assertEqual(config["task"], "megnet_sparse")
        self.assertEqual(model["atom_features"], "werespecies")
        self.assertEqual(model["cutoff"], 12)
        self.assertEqual(model["edge_embed_size"], 40)
        self.assertEqual(model["vertex_aggregation"], "max")
        self.assertEqual(model["global_aggregation"], "max")
        self.assertEqual(model["embedding_size"], 64)
        self.assertEqual(model["nblocks"], 3)
        self.assertEqual(model["train_batch_size"], 8)
        self.assertEqual(model["test_batch_size"], 1)
        self.assertNotIn("max_neighbors", model)

        self.assertEqual(optim["optimizer"], "AdamW")
        self.assertEqual(optim["weight_decay"], 1e-4)
        self.assertNotIn("scheduler_monitor", optim)
        self.assertEqual(optim["early_stopping_patience"], 50)
        self.assertEqual(optim["early_stopping_min_delta_percent"], 0.5)

        high_model = get_config("megnet", "2dmd_high", "sparse")["model"]
        self.assertEqual(high_model["train_batch_size"], 16)
        self.assertEqual(high_model["test_batch_size"], 1)
        low_model = get_config("megnet", "2dmd_low", "sparse")["model"]
        self.assertEqual(low_model["train_batch_size"], 8)
        self.assertEqual(low_model["test_batch_size"], 1)

    def test_sparse_mode_is_limited_to_original_scope(self):
        for model, dataset in (
            ("cgcnn", "vacancy"),
            ("alignn", "2dmd_high"),
            ("megnet", "native"),
        ):
            with self.subTest(model=model, dataset=dataset):
                with self.assertRaisesRegex(
                    ValueError,
                    "MEGNET_SPARSE reproduction",
                ):
                    get_config(model, dataset, "sparse")


if __name__ == "__main__":
    unittest.main()
