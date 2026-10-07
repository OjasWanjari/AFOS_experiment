import copy
import inspect
import math
import random
import tempfile
import unittest
from pathlib import Path

from paper_afos.search import (BINARY, UNARY, crossover, expression, fitness_score,
                               mutate, next_population, random_chromosome, select_top, validate)
from run_paper import load_config

CONFIG = Path(__file__).resolve().parents[1] / "paper_parameters.json"


class RecordingTests(unittest.TestCase):
    def test_epoch_csv_survives_interruption_and_retry(self):
        from paper_afos.recording import EpochCSV, read_results
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = dict(epoch=1, loss=1.0, accuracy=0.5, val_loss=math.inf, val_accuracy=0.4)
            with self.assertRaises(KeyboardInterrupt):
                with EpochCSV(root, "search_0_1") as record:
                    record(row)
                    # Read while training is still in progress.
                    self.assertEqual(read_results(record.path)[0]["epoch"], 1)
                    self.assertIn("inf", record.path.read_text())
                    raise KeyboardInterrupt()
            with EpochCSV(root, "search_0_1") as record:
                record(dict(row, val_loss=0.8))
            self.assertEqual(len(list((root / "histories" / "attempts").glob("*.csv"))), 1)
            self.assertEqual(len(read_results(record.path)), 1)
            self.assertIn("training_failed", (root / "results" / "events.csv").read_text())

    def test_plot_only_after_completion_and_skip_identical_resume(self):
        from unittest.mock import patch
        from run_paper import finish_outputs
        cfg = load_config(CONFIG)
        state = dict(evaluations=[], generations=[], selected=[], retraining=[],
                     search_complete=True, retraining_complete=False)
        with tempfile.TemporaryDirectory() as directory, patch("run_paper.plots") as plot:
            root = Path(directory)
            def render(root):
                (root / "plots").mkdir(exist_ok=True)
                (root / "plots" / "test.png").write_bytes(b"test")
            plot.side_effect = render
            with self.assertRaises(ValueError):
                finish_outputs(root, state, cfg, "all")
            plot.assert_not_called()
            finish_outputs(root, state, cfg, "search")
            finish_outputs(root, state, cfg, "search")
            self.assertEqual(plot.call_count, 1)
            state["retraining_complete"] = True
            state["generations"] = [dict(seed=0, generation=1, best_score=1.0)]
            finish_outputs(root, state, cfg, "all")
            finish_outputs(root, state, cfg, "all")
            self.assertEqual(plot.call_count, 2)

    def test_failed_plot_is_retried_without_completion_marker(self):
        from unittest.mock import patch
        from run_paper import finish_outputs
        state = dict(evaluations=[], generations=[], selected=[], retraining=[],
                     search_complete=True, retraining_complete=True)
        with tempfile.TemporaryDirectory() as directory, patch("run_paper.plots", side_effect=RuntimeError("plot error")):
            root = Path(directory)
            with self.assertRaises(RuntimeError):
                finish_outputs(root, state, load_config(CONFIG), "all")
            self.assertFalse((root / "plots_complete.json").exists())


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(CONFIG)
        self.rng = random.Random(9)

    def test_configuration_and_population_accounting(self):
        self.assertEqual(len(UNARY), 19)
        self.assertEqual(len(BINARY), 11)
        s = self.config["search"]
        self.assertEqual(s["population_size"] * s["generations"] * s["replications"], 6000)
        population = [random_chromosome(self.rng) for _ in range(30)]
        for probability, expected in [(0, (0, 0, 24)), (1, (6, 1, 17))]:
            s = dict(s, crossover_probability=probability, mutation_probability=probability)
            original = copy.deepcopy(population)
            child, counts = next_population(population, list(range(30)), s, self.rng)
            self.assertEqual(len(child), 30)
            self.assertEqual((counts["crossover"], counts["mutation"], counts["random"]), expected)
            self.assertEqual(child[:6], list(reversed(population[-6:])))
            self.assertEqual(population, original)
            for chromosome in child:
                validate(chromosome)

    def test_operators_preserve_types_and_change_one_gene(self):
        for _ in range(300):
            a, b = random_chromosome(self.rng), random_chromosome(self.rng)
            for chromosome in crossover(a, b, self.rng):
                validate(chromosome)
            mutant = mutate(a, self.rng)
            self.assertEqual(sum(x != y for x, y in zip(a, mutant)), 1)
            validate(mutant)

    def test_fitness_handles_invalid_values(self):
        self.assertEqual(fitness_score(0.5, 0.8), 1.6)
        for loss in (None, 0, -1, math.nan, math.inf):
            self.assertEqual(fitness_score(loss, 0.8), 0)

    def test_selection_uses_validation_not_test_or_score(self):
        rows = [dict(seed=0, evaluation_count=i, chromosome=[0, 0, i],
                     activation_expression=str(i), score=100-i,
                     validation_accuracy=0.5+i/10, validation_loss=1.0,
                     test_accuracy=1-i/10, status="completed") for i in range(3)]
        rows.append(dict(rows[2], seed=1, validation_accuracy=0.6))
        rows.append(dict(rows[0], activation_expression="invalid", status="nonfinite", validation_accuracy=1.0))
        selected = select_top(rows, 2)
        self.assertEqual([r["activation_expression"] for r in selected], ["2", "1"])

    def test_expression_shapes(self):
        self.assertEqual(expression([10, 0, 2]), "(x*erf(exp(x)))")
        self.assertEqual(expression([1, 15, 18]), "(erf(x)*log(1+exp(x)))")
        self.assertIn("sigmoid", expression([9, 0, 2]))


class TensorFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tensorflow as tf
        tf.config.set_visible_devices([], "GPU")
        cls.tf = tf

    def test_interrupted_search_resumes_without_duplicate_evaluations(self):
        import contextlib
        import io
        import json
        from unittest.mock import patch
        from run_paper import search
        cfg = load_config(CONFIG)
        cfg["search"].update(population_size=6, generations=2, seeds=[0, 1], replications=2,
                             selected_parents=2, max_crossover_children=2,
                             max_mutation_children=1, minimum_random_individuals=1)
        cfg["post_search"]["top_functions"] = 1
        initial = dict(seed_index=0, generation=1, population=None, evaluations=[], generations=[],
                       selected=[], retraining=[], search_complete=False, retraining_complete=False)
        class FakeModel:
            def save(self, *args, **kwargs):
                pass
        calls = [0]
        def train(*args, **kwargs):
            calls[0] += 1
            if calls[0] == 3:
                raise InterruptedError("simulated interruption")
            return dict(train_loss=1.0, train_accuracy=0.5, validation_loss=1.1,
                        validation_accuracy=0.4, epochs_completed=15, status="completed"), [{"epoch": 15}]
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), \
                patch("paper_afos.model.build_model", return_value=FakeModel()), \
                patch("paper_afos.model.train_candidate", side_effect=train):
            root = Path(directory)
            with self.assertRaises(InterruptedError):
                search(root, copy.deepcopy(initial), cfg, (None,None), (None,None), 0, False)
            resumed = json.loads((root / "checkpoint.json").read_text())
            self.assertEqual(len(resumed["evaluations"]), 2)
            search(root, resumed, cfg, (None,None), (None,None), 0, False)
            expected = copy.deepcopy(initial)
            search(root / "uninterrupted", expected, cfg, (None,None), (None,None), 0, False)
            self.assertEqual(resumed, expected)
            self.assertEqual(len(resumed["evaluations"]), 24)
            self.assertEqual(len(resumed["generations"]), 4)

    def test_all_primitive_formulas(self):
        import numpy as np
        from paper_afos.model import evaluate_tree
        x = 0.4
        unary_expected = [x, -x, math.exp(x), abs(x), math.exp(-x), min(x, 0), max(x, 0),
                          math.sin(x), math.cos(x), math.sinh(x), math.tanh(x), math.asin(x), math.atan(x),
                          math.asinh(x), math.atanh(x), math.erf(x), 1/(1+math.exp(-x)), x*math.erf(x), math.log1p(math.exp(x))]
        for name, expected in zip(UNARY, unary_expected):
            self.assertAlmostEqual(float(evaluate_tree(("u", name), self.tf.constant(x))), expected, places=6)
        # Negative b distinguishes nested min/max and sigmoid argument ordering.
        b = -x
        binary_expected = [x+b, x*b, max(x,b), min(x,b), max(max(x,0),b), max(min(x,0),b),
                           min(max(x,0),b), min(min(x,0),b), x*math.exp(b), x/(1+math.exp(-b)), x*math.erf(b)]
        for name, expected in zip(BINARY, binary_expected):
            node = ("b", name, ("u", "x"), ("u", "neg"))
            self.assertAlmostEqual(float(evaluate_tree(node, self.tf.constant(x))), expected, places=6)
        self.assertTrue(math.isnan(float(evaluate_tree(("u", "asin"), self.tf.constant(2.0)))))

    def test_table2_and_model_roundtrip(self):
        import numpy as np
        from paper_afos.model import build_model
        tf = self.tf
        model = build_model(load_config(CONFIG), chromosome=[2, 0, 6])
        conv = [l for l in model.layers if isinstance(l, tf.keras.layers.Conv2D)]
        dense = [l for l in model.layers if isinstance(l, tf.keras.layers.Dense)]
        pool = [l for l in model.layers if isinstance(l, tf.keras.layers.MaxPooling2D)]
        dropout = [l for l in model.layers if isinstance(l, tf.keras.layers.Dropout)]
        self.assertEqual([l.filters for l in conv], [28, 32, 64, 64, 128, 128])
        self.assertEqual([l.units for l in dense], [10])
        self.assertEqual([tuple(l.output_shape[1:]) for l in pool], [(16,16,32), (8,8,64), (4,4,128)])
        self.assertEqual(len(dropout), 3)
        self.assertAlmostEqual(float(model.optimizer.learning_rate.numpy()), 0.01)
        inputs = np.ones((1,32,32,3), dtype="float32")
        expected = model(inputs, training=False).numpy()
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "model")
            model.save(path)
            loaded = tf.keras.models.load_model(path)
            np.testing.assert_allclose(loaded(inputs, training=False), expected, rtol=1e-5)

    def test_train_has_no_test_input_and_early_stop_threshold(self):
        from paper_afos.model import train_candidate
        from types import SimpleNamespace
        self.assertNotIn("test", inspect.signature(train_candidate).parameters)
        class FakeModel:
            def __init__(self, accuracy, nan=False):
                self.calls = []
                self.accuracy, self.nan = accuracy, nan
            def fit(self, *args, **kwargs):
                self.calls.append(kwargs)
                if self.nan:
                    kwargs["callbacks"][0].failed = True
                n = kwargs["epochs"] - kwargs.get("initial_epoch", 0)
                return SimpleNamespace(history={"loss": [1.0]*n, "accuracy": [0.5]*n,
                                                "val_loss": [math.nan if self.nan else 1.0]*n,
                                                "val_accuracy": [self.accuracy]*n})
        cfg = load_config(CONFIG)
        for accuracy, epochs, status in [(0.25, 1, "early_stopped"), (0.26, 15, "completed")]:
            model = FakeModel(accuracy)
            metrics, _ = train_candidate(model, (None,None), (None,None), cfg, verbose=0)
            self.assertEqual(metrics["epochs_completed"], epochs)
            self.assertEqual(metrics["status"], status)
        metrics, _ = train_candidate(FakeModel(0.8, True), (None,None), (None,None), cfg, verbose=0)
        self.assertEqual(metrics["status"], "nonfinite")
        model = FakeModel(0.01)
        metrics, _ = train_candidate(model, (None,None), (None,None), cfg, search=False, verbose=0)
        self.assertEqual(metrics["epochs_completed"], 50)
        self.assertEqual(metrics["status"], "completed")

    def test_epoch_callback_streams_across_both_fit_calls(self):
        from types import SimpleNamespace
        from paper_afos.model import train_candidate
        observed = []
        class FakeModel:
            def fit(self, *args, **kwargs):
                history = {k: [] for k in ("loss", "accuracy", "val_loss", "val_accuracy")}
                for epoch in range(kwargs.get("initial_epoch", 0), kwargs["epochs"]):
                    logs = dict(loss=1.0, accuracy=0.5, val_loss=1.1, val_accuracy=0.4)
                    for callback in kwargs["callbacks"]:
                        callback.on_epoch_end(epoch, logs)
                    # It has been recorded before model.fit returns.
                    self_outer.assertEqual(observed[-1]["epoch"], epoch + 1)
                    for key in history:
                        history[key].append(logs[key])
                return SimpleNamespace(history=history)
        self_outer = self
        metrics, history = train_candidate(FakeModel(), (None, None), (None, None),
                                           load_config(CONFIG), verbose=0, on_epoch=observed.append)
        self.assertEqual([r["epoch"] for r in observed], list(range(1, 16)))
        self.assertEqual(observed, history)


if __name__ == "__main__":
    unittest.main()
