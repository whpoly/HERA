import unittest

from HERA.config.defaults import get_config


class ModelConfigDefaultsTests(unittest.TestCase):
    def test_cgcnn_uses_matched_train_batch_and_neighbor_cap(self):
        for mode in ("full", "hetero", "attention", "definet"):
            with self.subTest(mode=mode):
                model = get_config("cgcnn", "och", mode)["model"]
                self.assertEqual(model["train_batch_size"], 64)
                self.assertEqual(model["max_neighbors"], 12)
                self.assertEqual(model["test_batch_size"], 1)

    def test_megnet_uses_matched_train_batch_and_neighbor_cap(self):
        for mode in ("full", "hetero", "attention"):
            with self.subTest(mode=mode):
                model = get_config("megnet", "och", mode)["model"]
                self.assertEqual(model["train_batch_size"], 64)
                self.assertEqual(model["max_neighbors"], 12)
                self.assertEqual(model["test_batch_size"], 1)

    def test_megnet_capacity_is_matched_by_representation(self):
        for mode in ("full", "full_x", "attention", "was_x", "attention_was"):
            with self.subTest(mode=mode):
                model = get_config("megnet", "och", mode)["model"]
                self.assertEqual(model["embedding_size"], 32)

        for mode in ("hetero", "hetero_fixed_pool", "hetero_was"):
            with self.subTest(mode=mode):
                model = get_config("megnet", "och", mode)["model"]
                self.assertEqual(model["embedding_size"], 32)

    def test_alignn_memory_specific_defaults_are_unchanged(self):
        model = get_config("alignn", "och", "full")["model"]
        self.assertEqual(model["train_batch_size"], 64)
        self.assertEqual(model["test_batch_size"], 1)
        self.assertEqual(model["max_neighbors"], 12)


if __name__ == "__main__":
    unittest.main()
