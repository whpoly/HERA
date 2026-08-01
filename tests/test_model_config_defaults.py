import unittest

from HERA.config.defaults import VALID_MODES, get_config


class ModelConfigDefaultsTests(unittest.TestCase):
    def test_removed_hetero_bidir_mode_is_not_configurable(self):
        self.assertNotIn("hetero_bidir", VALID_MODES)
        with self.assertRaisesRegex(ValueError, "Unknown mode 'hetero_bidir'"):
            get_config("alignn", "och", "hetero_bidir")

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

    def test_low_density_vacancy_datasets_use_batch_size_eight(self):
        for dataset in ("vacancy", "2dmd_low"):
            for model_name, mode in (
                ("cgcnn", "full"),
                ("megnet", "full"),
                ("alignn", "full"),
                ("definet", "attention"),
            ):
                with self.subTest(dataset=dataset, model=model_name):
                    model = get_config(model_name, dataset, mode)["model"]
                    self.assertEqual(model["train_batch_size"], 8)

    def test_memory_limited_datasets_use_batch_size_sixteen(self):
        for dataset in ("2dmd_high", "native"):
            for model_name, mode in (
                ("cgcnn", "full"),
                ("megnet", "hetero"),
                ("alignn", "attention"),
                ("definet", "attention"),
            ):
                with self.subTest(dataset=dataset, model=model_name, mode=mode):
                    model = get_config(model_name, dataset, mode)["model"]
                    self.assertEqual(model["train_batch_size"], 16)
                    self.assertEqual(model["test_batch_size"], 1)

    def test_other_batch_defaults_are_preserved(self):
        expectations = (
            ("och", "cgcnn", "full", 64, 1),
            ("och", "megnet", "full", 64, 1),
            ("native", "alignn", "full", 16, 1),
            ("och", "definet", "attention", 64, 1),
        )
        for dataset, model_name, mode, train_batch, test_batch in expectations:
            with self.subTest(dataset=dataset, model=model_name):
                model = get_config(model_name, dataset, mode)["model"]
                self.assertEqual(model["train_batch_size"], train_batch)
                self.assertEqual(model["test_batch_size"], test_batch)

    def test_primary_models_share_optimizer_scheduler_and_early_stopping(self):
        fields = (
            "optimizer",
            "weight_decay",
            "lr_initial",
            "scheduler",
            "factor",
            "patience",
            "threshold",
            "min_lr",
            "early_stopping_patience",
            "early_stopping_min_delta_percent",
        )
        optimizers = {
            model: get_config(model, "och", "full")["optim"]
            for model in ("cgcnn", "megnet", "alignn")
        }
        for field in fields:
            with self.subTest(field=field):
                values = {optim[field] for optim in optimizers.values()}
                self.assertEqual(values, {optimizers["alignn"][field]})
        self.assertEqual(optimizers["cgcnn"]["optimizer"], "AdamW")
        self.assertEqual(optimizers["cgcnn"]["weight_decay"], 1e-4)
        self.assertEqual(optimizers["cgcnn"]["early_stopping_patience"], 50)
        self.assertEqual(
            optimizers["megnet"]["early_stopping_min_delta_percent"],
            0.5,
        )


if __name__ == "__main__":
    unittest.main()
