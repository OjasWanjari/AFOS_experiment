"""Selection, isolation, failure persistence, and immutable resume contracts."""
import copy
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from paper_afos.retraining import (latest_result, make_trials, pool_candidates, prepare_run,
                                  read_plan, summarize, write_csv, write_json)
from paper_afos.search import BINARY, UNARY, expression
from retrain_paper import execute
from run_paper import load_config

ROOT = Path(__file__).resolve().parents[1]


def row(seed, evaluation, chromosome, accuracy=0.7, **overrides):
    return dict(seed=seed, evaluation_count=evaluation, chromosome=chromosome,
                activation_expression=expression(chromosome), status="completed", epochs_completed=15,
                train_loss=0.5, train_accuracy=0.8, validation_loss=0.6, validation_accuracy=accuracy,
                **overrides)


def source(root, name, seed, rows, config, split=b"identical-split"):
    run = root / name
    write_json(run / "manifest.json", dict(config=config, unary_indices=list(UNARY), binary_indices=list(BINARY)))
    write_csv(run / "results" / f"evaluation_results_seed_{seed}.csv", rows)
    with zipfile.ZipFile(run / "split_indices.npz", "w") as archive:
        for key in ("train_indices", "validation_indices"):
            archive.writestr(f"{key}.npy", split)
    return run


class PoolTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT / "paper_parameters.json")
        self.config["search"].update(seeds=[0, 1], replications=2)
        self.config["post_search"]["top_functions"] = 2

    def sources(self, root):
        source(root, "seed0", 0, [row(0, 1, [2, 0, 6], 0.75), row(0, 2, [1, 0, 16], 0.8)], self.config)
        source(root, "seed1", 1, [row(1, 1, [1, 0, 16], 0.9), row(1, 2, [9, 15, 9], 0.85)], self.config)

    def test_pool_unique_expression_ranked_by_validation_not_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.sources(root)
            source(root, "partial", 1, [row(1, 1, [2, 0, 6], 0.99, test_accuracy=1.0)], self.config)
            selected, audit, _, _ = pool_candidates(root, self.config)
            self.assertEqual([r["validation_accuracy"] for r in selected], [0.9, 0.85])
            self.assertEqual([r["seed"] for r in selected], [1, 1])
            self.assertNotIn("test_accuracy", selected[0])
            self.assertEqual(audit[1]["recorded"], 2)
            self.assertEqual(len(audit[1]["alternatives"]), 1)

    def test_exclude_numeric_runtime_and_incomplete_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.sources(root)
            bad = [row(1, n, [9, 15, 9], 0.999) for n in range(1, 5)]
            bad[0]["validation_loss"] = float("nan")
            bad[1]["status"] = "nonfinite"
            bad[2]["epochs_completed"] = 0
            bad[3]["epochs_completed"] = 14
            source(root, "longer", 1, bad, self.config)
            selected, _, warnings, _ = pool_candidates(root, self.config)
            self.assertEqual([r["seed"] for r in selected], [0, 0])
            self.assertTrue(any("zero-epoch" in w for w in warnings))

    def test_ambiguous_duplicate_trajectory_requires_override(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.sources(root)
            other = source(root, "other", 1, [row(1, 1, [2, 0, 6], 0.99), row(1, 2, [9, 15, 9], 0.8)], self.config)
            with self.assertRaisesRegex(ValueError, "Ambiguous"):
                pool_candidates(root, self.config)
            selected, _, _, _ = pool_candidates(root, self.config, {1: other})
            self.assertEqual(selected[0]["validation_accuracy"], 0.99)

    def test_reject_missing_seed_split_or_schema_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.sources(root)
            path = root / "seed1" / "manifest.json"
            manifest = json.loads(path.read_text())
            manifest["unary_indices"].reverse()
            write_json(path, manifest)
            with self.assertRaisesRegex(ValueError, "primitive"):
                pool_candidates(root, self.config)
            source(root, "seed1", 1, [row(1, 1, [9, 15, 9])], self.config, split=b"different")
            with self.assertRaisesRegex(ValueError, "different train/validation"):
                pool_candidates(root, self.config)
            config = copy.deepcopy(self.config)
            config["search"]["seeds"].append(2)
            # Verify missing-seed rejection with the matching split restored.
            source(root, "seed1", 1, [row(1, 1, [9, 15, 9])], self.config)
            with self.assertRaisesRegex(ValueError, "required seed 2"):
                pool_candidates(root, config)

    def test_frozen_plan_and_sources_are_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.sources(root / "inputs")
            config_file = root / "config.json"
            write_json(config_file, self.config)
            output = root / "output"
            plan = prepare_run(output, root / "inputs", config_file, {}, device="cpu")
            self.assertEqual(read_plan(output), plan)
            self.assertEqual(len(plan["trials"]), (2 + 6) * 5)
            # Changes in the working tree cannot affect a frozen child process.
            with (output / "source_snapshot" / "paper_afos" / "model.py").open("a") as stream:
                stream.write("\n# modified\n")
            with self.assertRaisesRegex(ValueError, "Frozen source"):
                read_plan(output)

    def test_planning_module_does_not_import_tensorflow(self):
        subprocess.run([sys.executable, "-c",
                        "import paper_afos.retraining, sys; assert 'tensorflow' not in sys.modules"], cwd=ROOT, check=True)


class AccountingTests(unittest.TestCase):
    def plan(self):
        config = load_config(ROOT / "paper_parameters.json")
        selected = [dict(row(3, 1, [2, 0, 6]), rank=1)]
        return dict(trials=make_trials(selected, config)[:5], warnings=[], smoke=False)

    def test_incomplete_replications_do_not_get_a_success_only_mean(self):
        with tempfile.TemporaryDirectory() as directory:
            root, plan = Path(directory), self.plan()
            for trial in plan["trials"][:4]:
                write_json(root / "trials" / trial["trial_id"] / "attempt_0001" / "result.json",
                           dict(trial, status="completed", test_accuracy=0.8))
            summarize(root, plan)
            from paper_afos.recording import read_results
            summary = read_results(root / "results" / "retraining_summary.csv")[0]
            self.assertIsNone(summary["test_accuracy_mean"])
            trial = plan["trials"][-1]
            write_json(root / "trials" / trial["trial_id"] / "attempt_0001" / "result.json",
                       dict(trial, status="completed", test_accuracy=0.7))
            summarize(root, plan)
            summary = read_results(root / "results" / "retraining_summary.csv")[0]
            self.assertAlmostEqual(summary["test_accuracy_mean"], 0.78)
            self.assertAlmostEqual(summary["test_accuracy_std"], 0.04472135955)

    def test_worker_crash_stops_and_explicit_retry_preserves_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            root, plan = Path(directory), self.plan()
            with patch("retrain_paper.subprocess.Popen") as popen:
                popen.return_value.wait.return_value = 137
                self.assertEqual(execute(root, plan), 1)
                self.assertEqual(popen.call_count, 1)
                command = popen.call_args.args[0]
                self.assertEqual(command[0], sys.executable)
                self.assertIn("source_snapshot", command[1])
            self.assertEqual(latest_result(root, plan["trials"][0])["status"], "worker_crash")
            with patch("retrain_paper.subprocess.Popen") as popen:
                self.assertEqual(execute(root, plan), 1)
                popen.assert_not_called()
                popen.return_value.wait.return_value = 137
                self.assertEqual(execute(root, plan, retry_errors=True), 1)
            self.assertEqual(len(list((root / "trials" / plan["trials"][0]["trial_id"]).glob("attempt_*"))), 2)
            self.assertFalse((root / "run.lock").exists())

    def test_numeric_failure_is_not_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            root, plan = Path(directory), self.plan()
            for trial in plan["trials"]:
                write_json(root / "trials" / trial["trial_id"] / "attempt_0001" / "result.json",
                           dict(trial, status="nonfinite", test_accuracy=None))
            with patch("retrain_paper.subprocess.Popen") as popen:
                self.assertEqual(execute(root, plan, retry_errors=True), 2)
                popen.assert_not_called()


class WorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import tensorflow as tf
        except ImportError:
            raise unittest.SkipTest("TensorFlow is not installed")
        tf.config.set_visible_devices([], "GPU")

    def test_saved_split_is_used_and_overlapping_indices_are_rejected(self):
        import numpy as np
        from paper_afos.model import load_data
        config = load_config(ROOT / "paper_parameters.json")
        config["dataset"].update(train_size=3, validation_size=2)
        images = np.stack([np.full((32, 32, 3), i, dtype="uint8") for i in range(5)])
        labels = np.arange(5).reshape(-1, 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.npz"
            np.savez(path, train_indices=np.array([4, 0, 2]), validation_indices=np.array([3, 1]))
            with patch("tensorflow.keras.datasets.cifar10.load_data", return_value=((images, labels), (images[:1], labels[:1]))):
                train, validation, _, _ = load_data(config, split_file=path)
                np.testing.assert_array_equal(np.argmax(train[1], axis=1), [4, 0, 2])
                np.testing.assert_array_equal(np.argmax(validation[1], axis=1), [3, 1])
                np.testing.assert_allclose(train[0][:, 0, 0, 0], np.array([4, 0, 2]) / 255)
                np.savez(path, train_indices=np.array([4, 0, 2]), validation_indices=np.array([2, 1]))
                with self.assertRaisesRegex(ValueError, "partition"):
                    load_data(config, split_file=path)

    def test_save_failure_retains_test_metrics_and_nonfinite_test_skips_save(self):
        from unittest.mock import Mock
        from paper_afos.retrain_worker import run_worker
        config = load_config(ROOT / "paper_parameters.json")
        trial = make_trials([dict(row(0, 1, [2, 0, 6]), rank=1)], config)[0]
        plan = dict(config=config, smoke=True, trials=[trial], runtime=dict(device="cpu", save_models=True))
        metrics = dict(status="completed", epochs_completed=50, train_loss=0.5,
                       train_accuracy=0.8, validation_loss=0.6, validation_accuracy=0.7)
        for loss, expected_status in [(0.7, "completed"), (float("nan"), "nonfinite_test")]:
            with self.subTest(loss=loss), tempfile.TemporaryDirectory() as directory:
                attempt = Path(directory) / "attempt_0001"
                model = Mock()
                model.evaluate.return_value = (loss, 0.75)
                model.save.side_effect = OSError("simulated disk failure")
                with patch("paper_afos.retrain_worker.read_plan", return_value=plan), \
                        patch("paper_afos.model.load_data", return_value=((None,None), (None,None), (None,None), None)), \
                        patch("paper_afos.model.build_model", return_value=model), \
                        patch("paper_afos.model.train_candidate", return_value=(metrics, [])) as train:
                    code = run_worker(directory, trial["trial_id"], attempt, quiet=True)
                result = json.loads((attempt / "result.json").read_text())
                self.assertEqual(result["status"], expected_status)
                self.assertEqual(result["test_accuracy"], 0.75)
                self.assertFalse(train.call_args.kwargs["search"])
                if expected_status == "completed":
                    self.assertEqual(code, 1)
                    self.assertEqual(result["model_save_status"], "error")
                else:
                    self.assertEqual(code, 0)
                    self.assertIsNone(result["test_loss"])
                    model.save.assert_not_called()


if __name__ == "__main__":
    unittest.main()
