import argparse
import unittest

from HERA.main import (
    ALL_BENCHMARK_SEEDS,
    DEFAULT_SEED,
    parse_seed_values,
)


class SeedOptionTests(unittest.TestCase):
    def setUp(self):
        self.parser = argparse.ArgumentParser(add_help=False)

    def test_default_is_seed_123_only(self):
        self.assertEqual(DEFAULT_SEED, 123)
        self.assertEqual(parse_seed_values(None, self.parser), [123])

    def test_explicit_seed_is_parsed(self):
        self.assertEqual(parse_seed_values(["42"], self.parser), [42])

    def test_all_expands_to_standard_ten_seeds(self):
        self.assertEqual(
            parse_seed_values(["all"], self.parser),
            ALL_BENCHMARK_SEEDS,
        )
        self.assertEqual(len(ALL_BENCHMARK_SEEDS), 10)

    def test_explicit_seed_list_is_parsed(self):
        self.assertEqual(
            parse_seed_values(["42", "123"], self.parser),
            [42, 123],
        )

    def test_all_cannot_be_combined_with_an_explicit_seed(self):
        with self.assertRaises(SystemExit):
            parse_seed_values(["all", "42"], self.parser)


if __name__ == "__main__":
    unittest.main()
