"""Adaptive search tests without TensorFlow, datasets, or model training."""
import contextlib
import copy
import csv
import io
import json
import math
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import run_paper
from paper_afos.adaptive import generation_probabilities
from paper_afos.search import next_population, random_chromosome, validate

ROOT = Path(__file__).resolve().parents[1]


class AdaptiveProbabilityTests(unittest.TestCase):
    def setUp(self):
        self.config = run_paper.load_config(ROOT / "paper_parameters_adaptive.json")
        self.search = self.config["search"]

    def history(self, scores, seed=0):
        return [dict(seed=seed, generation=g, best_score=score)
                for g, score in enumerate(scores, start=1)]

    def test_only_adaptive_settings_differ_from_baseline(self):
        baseline = run_paper.load_config(ROOT / "paper_parameters.json")
        self.config["search"].pop("adaptive_probabilities")
        self.assertEqual(self.config, baseline)

    def test_improvement_degree_and_bounds(self):
        history = self.history([1.0, 1.0, 1.0])
        for score, pc, pm in [(0.9, 1.0, 0.8), (1.0, 1.0, 0.8),
                              (1.0025, 0.95, 0.65), (1.005, 0.9, 0.5),
                              (1.0075, 0.85, 0.35), (1.01, 0.8, 0.2), (2.0, 0.8, 0.2)]:
            with self.subTest(score=score):
                info = generation_probabilities(self.search, history, 0, 4, score)
                self.assertAlmostEqual(info["crossover_probability"], pc)
                self.assertAlmostEqual(info["mutation_probability"], pm)
                self.assertEqual(info["fitness_reference_generation"], 1)
                self.assertEqual(info["fitness_reference_score"], 1.0)
                self.assertEqual(info["best_so_far_score"], max(1.0, score))
        self.assertEqual(self.search["crossover_probability"], 0.8)

    def test_warmup_seed_isolation_and_past_only(self):
        other_seed = self.history([100.0] * 10, seed=1)
        future = [dict(seed=0, generation=9, best_score=1000.0)]
        for generation in (1, 2, 3):
            info = generation_probabilities(self.search, other_seed + future, 0, generation, 1.0)
            self.assertEqual(info["probability_mode"], "warmup")
            self.assertEqual(info["best_so_far_score"], 1.0)
            self.assertEqual((info["crossover_probability"], info["mutation_probability"]), (0.8, 0.2))
        info = generation_probabilities(self.search, other_seed + future + self.history([1, 1, 1]), 0, 4, 1.005)
        self.assertAlmostEqual(info["crossover_probability"], 0.9)
        with self.assertRaisesRegex(ValueError, "Missing reference"):
            generation_probabilities(self.search, other_seed, 0, 4, 1.0)

    def test_rolling_reference_is_cumulative_best_and_zero_is_safe(self):
        info = generation_probabilities(self.search, self.history([1, 0.8, 0.9, 0.9]), 0, 5, 1.005)
        self.assertEqual(info["fitness_reference_generation"], 2)
        self.assertEqual(info["fitness_reference_score"], 1.0)
        self.assertAlmostEqual(info["fitness_relative_improvement"], 0.005)
        self.assertAlmostEqual(info["mutation_probability"], 0.5)
        for invalid in (0, None, math.inf, math.nan):
            info = generation_probabilities(self.search, self.history([invalid] * 3), 0, 4, invalid)
            self.assertEqual(info["fitness_relative_improvement"], 0.0)
            self.assertEqual(info["crossover_probability"], 1.0)
            self.assertEqual(info["mutation_probability"], 0.8)
        info = generation_probabilities(self.search, self.history([0, 0, 0]), 0, 4, 1.0)
        self.assertTrue(math.isfinite(info["fitness_relative_improvement"]))
        self.assertEqual(info["mutation_probability"], 0.2)

    def test_fixed_mode_preserves_population_and_rng_sequence(self):
        baseline = run_paper.load_config(ROOT / "paper_parameters.json")["search"]
        disabled = copy.deepcopy(self.search)
        disabled["adaptive_probabilities"]["enabled"] = False
        for config in (baseline, disabled):
            for seed in range(10):
                rng = random.Random(seed)
                population = [random_chromosome(rng) for _ in range(30)]
                expected_rng = random.Random()
                expected_rng.setstate(rng.getstate())
                info = generation_probabilities(config, self.history([1, 1, 1]), 0, 4, 1.0)
                current = dict(config, crossover_probability=info["crossover_probability"],
                               mutation_probability=info["mutation_probability"])
                actual = next_population(population, list(range(30)), current, rng)
                expected = next_population(population, list(range(30)), baseline, expected_rng)
                self.assertEqual(actual, expected)
                self.assertEqual(rng.getstate(), expected_rng.getstate())

    def test_adaptive_population_keeps_allocation(self):
        rng = random.Random(0)
        population = [random_chromosome(rng) for _ in range(30)]
        for score in (1.0, 1.005, 1.02):
            info = generation_probabilities(self.search, self.history([1, 1, 1]), 0, 4, score)
            config = dict(self.search, crossover_probability=info["crossover_probability"],
                          mutation_probability=info["mutation_probability"])
            for _ in range(10):
                population, counts = next_population(population, list(range(30)), config, rng)
                self.assertEqual(len(population), 30)
                self.assertEqual(sum(counts.values()), 30)
                self.assertEqual(counts["parents"], 6)
                self.assertIn(counts["crossover"], (0, 2, 4, 6))
                if score == 1.0:
                    self.assertEqual(counts["crossover"], 6)
                self.assertIn(counts["mutation"], (0, 1))
                self.assertGreaterEqual(counts["random"], 17)
                for chromosome in population:
                    validate(chromosome)

    def test_invalid_config_is_rejected_before_training(self):
        cases = [("enabled", "true"), ("window_generations", 0), ("window_generations", 1.5),
                 ("window_generations", True), ("improvement_target", 0),
                 ("improvement_target", math.nan), ("epsilon", 0), ("epsilon", math.inf),
                 ("max_crossover_probability", 0.7), ("max_crossover_probability", 1.1),
                 ("max_crossover_probability", math.nan), ("max_crossover_probability", "1.0"),
                 ("max_mutation_probability", 0.1), ("max_mutation_probability", 1.1),
                 ("max_mutation_probability", "0.8"), ("unknown_option", 1)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            for field, value in cases:
                with self.subTest(field=field, value=value):
                    config = copy.deepcopy(self.config)
                    config["search"]["adaptive_probabilities"][field] = value
                    path.write_text(json.dumps(config), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        run_paper.load_config(path)
            config = copy.deepcopy(self.config)
            del config["search"]["adaptive_probabilities"]["epsilon"]
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaises(ValueError):
                run_paper.load_config(path)


class AdaptiveSearchIntegrationTests(unittest.TestCase):
    def test_probabilities_are_applied_logged_and_resume_identically(self):
        config = run_paper.load_config(ROOT / "paper_parameters_adaptive.json")
        config["search"].update(population_size=6, generations=8, seeds=[0, 1], replications=2,
                                selected_parents=2, max_crossover_children=2,
                                max_mutation_children=1, minimum_random_individuals=1)
        config["post_search"]["top_functions"] = 1
        original_config = copy.deepcopy(config)
        initial = dict(seed_index=0, generation=1, population=None, evaluations=[], generations=[],
                       selected=[], retraining=[], search_complete=False, retraining_complete=False)
        locations = {run_paper.training_seed(seed, generation, individual): (seed, generation, individual)
                     for seed in (0, 1) for generation in range(1, 9) for individual in range(6)}
        current_seed = [None]
        interruption = [None]
        used_rates = []
        fake_tf = ModuleType("tensorflow")
        fake_tf.keras = SimpleNamespace(backend=SimpleNamespace(clear_session=lambda: None),
                                       utils=SimpleNamespace(set_random_seed=lambda value: current_seed.__setitem__(0, value)))
        fake_model = ModuleType("paper_afos.model")
        fake_model.build_model = lambda *args, **kwargs: SimpleNamespace(save=lambda *a, **k: None)

        def train(*args, **kwargs):
            seed, generation, individual = locations[current_seed[0]]
            if interruption[0] == "evaluation" and (seed, generation, individual) == (0, 5, 2):
                interruption[0] = None
                raise InterruptedError("after adaptive generation")
            score = [1.0, 1.0, 1.0, 1.005, 1.02, 1.02, 1.02, 1.02][generation - 1]
            return dict(train_loss=1.0, train_accuracy=0.5, validation_loss=0.5 / score,
                        validation_accuracy=0.5, epochs_completed=15, status="completed"), []

        fake_model.train_candidate = train
        real_write_json = run_paper.write_json

        def write_checkpoint(path, value):
            if (interruption[0] == "checkpoint" and Path(path).name == "checkpoint.json"
                    and value["seed_index"] == 0 and len(value["generations"]) == 4):
                interruption[0] = None
                raise InterruptedError("before adaptive generation commit")
            return real_write_json(path, value)

        def generate(population, scores, search_config, rng):
            used_rates.append((search_config["crossover_probability"], search_config["mutation_probability"]))
            return next_population(population, scores, search_config, rng)

        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), \
                patch.dict(sys.modules, {"tensorflow": fake_tf, "paper_afos.model": fake_model}), \
                patch("run_paper.write_json", side_effect=write_checkpoint), \
                patch("run_paper.next_population", side_effect=generate):
            root = Path(directory)
            expected = copy.deepcopy(initial)
            run_paper.search(root / "full", expected, config, (None, None), (None, None), 0, False)
            self.assertEqual(len(expected["evaluations"]), 96)
            self.assertEqual(len(used_rates), 14)
            self.assertAlmostEqual(used_rates[3][0], 0.9)
            self.assertAlmostEqual(used_rates[3][1], 0.5)
            self.assertEqual(used_rates[4], (0.8, 0.2))
            self.assertEqual(used_rates[7], (0.8, 0.2))  # New seed warmup.
            for row, rates in zip([r for r in expected["generations"] if r["next_generation"]], used_rates):
                self.assertEqual((row["crossover_probability"], row["mutation_probability"]), rates)
                self.assertEqual(row["next_generation"], row["generation"] + 1)
            final = expected["generations"][-1]
            self.assertIsNone(final["crossover_probability"])
            self.assertIsNone(final["next_generation"])
            self.assertEqual(final["probability_mode"], "search_complete")
            with (root / "full/results/generation_history.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 16)
            self.assertEqual(rows[3]["next_generation"], "5")
            self.assertAlmostEqual(float(rows[3]["fitness_relative_improvement"]), 0.005)
            self.assertEqual(rows[-1]["crossover_probability"], "")
            for failure in ("evaluation", "checkpoint"):
                with self.subTest(failure=failure):
                    interruption[0] = failure
                    target = root / failure
                    with self.assertRaises(InterruptedError):
                        run_paper.search(target, copy.deepcopy(initial), config, (None, None), (None, None), 0, False)
                    resumed = json.loads((target / "checkpoint.json").read_text(encoding="utf-8"))
                    run_paper.search(target, resumed, config, (None, None), (None, None), 0, False)
                    self.assertEqual(resumed, expected)
                    self.assertEqual((target / "results/generation_history.csv").read_bytes(),
                                     (root / "full/results/generation_history.csv").read_bytes())
        self.assertEqual(config, original_config)


if __name__ == "__main__":
    unittest.main()
