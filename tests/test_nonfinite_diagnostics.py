"""Small CPU fixtures test instrumentation, not the original GPU failure."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tensorflow as tf

from diagnose_retraining import run_training


class DiagnosticsTest(unittest.TestCase):
    def run_fixture(self, kind, mode):
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(7)

        @tf.custom_gradient
        def broken_backward(x):
            def grad(dy):
                return dy / (x - x)
            return tf.identity(x), grad

        class Operation(tf.keras.layers.Layer):
            def call(self, x):
                if kind == "forward":
                    return tf.sqrt(-tf.abs(x)-1.)
                if kind == "backward":
                    return broken_backward(x)
                return x

        model = tf.keras.Sequential([
            tf.keras.layers.Input((2,)),
            tf.keras.layers.Dense(2, kernel_initializer="ones"),
            Operation(),
            tf.keras.layers.Dense(2, activation="softmax"),
        ])
        model.compile(optimizer=tf.keras.optimizers.SGD(.01, momentum=.9),
                      loss="categorical_crossentropy", metrics=["accuracy"], jit_compile=False)
        x = np.ones((4, 2), np.float32)
        y = np.eye(2, dtype=np.float32)[[0, 1, 0, 1]]
        if kind == "backward":
            self.assertTrue(np.isfinite(model(x).numpy()).all())
        plan = {"config": {"training": {"batch_size": 2}, "post_search": {"epochs": 1}}}
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            result = run_training(model, (x,y), (x,y), (x,y), plan,
                                  {"trial_id": "fixture"}, output, mode)
            self.assertEqual(json.loads((output / "result.json").read_text())["status"], result["status"])
            events = [json.loads(line) for line in (output / "batches.jsonl").read_text().splitlines()]
            self.assertTrue(any(e["event"] == "batch_begin" for e in events))
            stats = json.loads((output / "variables.json").read_text())
            with np.load(output / "failure_or_final_variables.npz") as saved:
                self.assertEqual(set(saved.files), {s["key"] for s in stats})
            error = (output / "error.txt").read_text() if (output / "error.txt").exists() else ""
            return result, error, events

    def test_finite_check_completes(self):
        result, error, events = self.run_fixture("finite", "check")
        self.assertEqual(result["status"], "not_reproduced")
        self.assertEqual(error, "")
        self.assertEqual({e["phase"] for e in events}, {"train", "validation", "test"})

    def test_forward_check_stops(self):
        result, error, _ = self.run_fixture("forward", "check")
        self.assertEqual(result["status"], "numerics_check_triggered")
        self.assertIn("Sqrt", error)

    def test_finite_forward_bad_backward_stops(self):
        result, error, _ = self.run_fixture("backward", "check")
        self.assertEqual(result["status"], "numerics_check_triggered")
        self.assertIn("RealDiv", error)
        self.assertIn("custom_gradient.py", error)

    def test_replay_preserves_nonfinite_stop(self):
        result, _, _ = self.run_fixture("forward", "replay")
        self.assertEqual(result["status"], "nonfinite")


if __name__ == "__main__":
    unittest.main()
