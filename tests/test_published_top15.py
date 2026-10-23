"""Published formulas, immutable comparison inputs, and environment safeguards."""
import copy
import json
import math
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from paper_afos.published_top15 import catalog_rows, prepare_published_run
from paper_afos.retraining import read_plan, sha256, write_json
from paper_afos.runtime_environment import environment_mismatches
from run_paper import load_config

ROOT = Path(__file__).resolve().parents[1]


def fixture(root):
    reference = root / "reference"
    snapshot = reference / "source_snapshot"
    sources = [ROOT / "run_paper.py", ROOT / "retrain_paper.py", *sorted((ROOT / "paper_afos").glob("*.py"))]
    for source in sources:
        target = snapshot / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    (reference / "split_indices.npz").write_bytes(b"fixture: identical split bytes")
    runtime = dict(device="gpu", jit_compile=False, memory_growth=True, save_models=True)
    trial = dict(trial_id="discovered_01_seed_0", function="discovered_01", seed=0)
    plan = dict(format_version=1, smoke=False, config=load_config(ROOT / "paper_parameters.json"),
                runtime=runtime, trials=[trial],
                source_sha256={str(p.relative_to(snapshot)).replace("\\", "/"): sha256(p) for p in snapshot.rglob("*.py")},
                split_sha256=sha256(reference / "split_indices.npz"))
    write_json(reference / "plan.json", plan)
    (reference / "plan.sha256").write_text(sha256(reference / "plan.json"))
    attempt = reference / "trials/discovered_01_seed_0/attempt_0001"
    write_json(attempt / "result.json", dict(trial, status="completed", attempt="attempt_0001"))
    environment = dict(python="3.10.12", platform="Linux-test", packages={"tensorflow": "2.21.0"},
                       tensorflow="2.21.0", build=dict(cuda_version="12.5.1", cudnn_version="9", is_cuda_build=True),
                       devices=["GPU:0"], runtime=runtime, environment={"TF_XLA_FLAGS": "--tf_xla_auto_jit=0", "XLA_FLAGS": None})
    write_json(attempt / "environment.json", environment)
    (attempt / "worker.log").write_text("device: 0, name: NVIDIA GeForce RTX 4080 SUPER, pci bus id: test\n")
    return reference


class PlanTests(unittest.TestCase):
    def test_reference_is_copied_without_reselection_or_parameter_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = fixture(root)
            old = read_plan(reference)
            plan = prepare_published_run(root / "output", reference)
            self.assertEqual(plan["config"], old["config"])
            self.assertEqual(plan["runtime"], old["runtime"])
            self.assertEqual(plan["split_sha256"], old["split_sha256"])
            self.assertEqual(len(plan["trials"]), 75)
            self.assertEqual(len({t["trial_id"] for t in plan["trials"]}), 75)
            for function in catalog_rows():
                self.assertEqual([t["seed"] for t in plan["trials"] if t["function"] == function["function"]], list(range(5)))
            self.assertEqual(plan["reference_environment"]["gpu_names"], ["NVIDIA GeForce RTX 4080 SUPER"])
            for rel in ("paper_afos/model.py", "paper_afos/search.py", "paper_afos/recording.py", "retrain_paper.py"):
                self.assertEqual(plan["source_sha256"][rel], old["source_sha256"][rel])
            self.assertEqual(read_plan(root / "output"), plan)
            # A moved prepared folder needs no access to the original reference.
            moved = root / "moved"
            (root / "output").rename(moved)
            reference.rename(root / "reference_unavailable")
            result = subprocess.run([sys.executable, str(moved / "source_snapshot/retrain_published_top15.py"),
                                     "--output", str(moved), "--resume", "--plan-only"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_changed_split_and_existing_output_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = fixture(root)
            prepare_published_run(root / "output", reference)
            with self.assertRaisesRegex(ValueError, "already exists"):
                prepare_published_run(root / "output", reference)
            (reference / "split_indices.npz").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "split changed"):
                prepare_published_run(root / "new", reference)

    def test_planning_does_not_import_tensorflow(self):
        subprocess.run([sys.executable, "-c", "import paper_afos.published_top15, sys; assert 'tensorflow' not in sys.modules"],
                       cwd=ROOT, check=True)

    def test_mismatched_environment_stops_before_data_loading(self):
        from paper_afos.retrain_worker import run_worker
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = prepare_published_run(root / "output", fixture(root))
            actual = copy.deepcopy(plan["reference_environment"])
            self.assertEqual(environment_mismatches(actual, actual), [])
            for key, value in (("packages", {}), ("gpu_names", ["another GPU"]), ("platform", "Windows-test")):
                changed = copy.deepcopy(actual)
                changed[key] = value
                self.assertTrue(environment_mismatches(actual, changed))
            actual["tensorflow"] = "different"
            attempt = root / "output/trials/paper_01_seed_0/attempt_0001"
            with patch("paper_afos.retrain_worker.inspect_environment", return_value=actual):
                self.assertEqual(run_worker(root / "output", "paper_01_seed_0", attempt), 1)
            result = json.loads((attempt / "result.json").read_text())
            self.assertEqual(result["status"], "runtime_error")
            self.assertIn("no training started", result["error"])
            self.assertIsNone(result["test_accuracy"])
            self.assertFalse((attempt / "histories").exists())


class FormulaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import tensorflow as tf
        except ImportError:
            raise unittest.SkipTest("TensorFlow is not installed")
        tf.config.set_visible_devices([], "GPU")

    def test_all_fifteen_match_independent_table3_formulas_and_have_finite_gradients(self):
        import numpy as np
        import tensorflow as tf
        from paper_afos.model import evaluate_tree
        from paper_afos.search import tree
        e, a, t = math.erf, math.atan, math.tanh
        s = lambda x: math.log1p(math.exp(x))
        sigmoid = lambda x: 1 / (1 + math.exp(-x))
        r = lambda x: max(x, 0)
        # Direct transcription independent of chromosome indices/evaluator.
        expected = [lambda x: e(x)*s(x), lambda x: -x*e(math.exp(-x)),
                    lambda x: x*e(math.exp(x)), lambda x: x*e(math.exp(x))**2,
                    lambda x: max(x, min(x, 0))*e(min(x, 0)), lambda x: r(x)*e(a(x))-x,
                    lambda x: s(x)*e(e(x)), lambda x: (t(x)+r(x))*sigmoid(s(x)*e(a(x))),
                    lambda x: s(x)*t(x), lambda x: max(math.sin(x)*sigmoid(math.sin(x)), x*e(x)+t(x)),
                    lambda x: (t(x)+r(x))*sigmoid(s(x)*e(math.sinh(x))), lambda x: x*e(x)+math.sin(x),
                    lambda x: max(x, e(x)), lambda x: r(x)*a(x)-x, lambda x: max(s(x)*t(x), e(x))]
        values = [-8., -3., -1., -.1, 0., .1, 1., 3., 8.]
        for row, formula in zip(catalog_rows(), expected):
            with self.subTest(function=row["function"]):
                x = tf.constant(values, dtype=tf.float64)
                with tf.GradientTape() as tape:
                    tape.watch(x)
                    y = evaluate_tree(tree(row["chromosome"]), x)
                np.testing.assert_allclose(y.numpy(), [formula(v) for v in values], rtol=1e-10, atol=1e-10)
                self.assertTrue(np.isfinite(tape.gradient(y, x).numpy()).all())

    def test_eelu2_saved_model_round_trip(self):
        import numpy as np
        import tensorflow as tf
        from paper_afos.model import PaperActivation
        model = tf.keras.Sequential([tf.keras.Input(shape=(1,)), PaperActivation(catalog_rows()[1]["chromosome"])])
        x = np.array([[-2.], [0.], [2.]], dtype="float32")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.keras"
            model.save(path)
            restored = tf.keras.models.load_model(path)
            np.testing.assert_array_equal(model(x).numpy(), restored(x).numpy())


if __name__ == "__main__":
    unittest.main()
