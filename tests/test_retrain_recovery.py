import ast
import json
from pathlib import Path
import tempfile
import unittest

from prepare_retrain_recovery import disable_xla, fix_save_path, prepare, source_hashes


RUNNER = '''def search(root, model, seed):
    model.save(str(root / "models" / f"best_search_seed_{seed}"), include_optimizer=True)

def retrain(root, model, name, seed):
    model.save(str(root / "models" / f"{name}_seed_{seed}"), include_optimizer=True)
'''
MODEL = '''def build_model(model):
    model.compile(optimizer="sgd", loss="mse")
    return model
'''


class RecoveryTests(unittest.TestCase):
    def fixture(self, root):
        snapshot = root / "source_snapshot"
        package = snapshot / "paper_afos"
        package.mkdir(parents=True)
        (snapshot / "run_paper.py").write_text(RUNNER, encoding="utf-8")
        (package / "model.py").write_text(MODEL, encoding="utf-8")
        (package / "__init__.py").write_text("", encoding="utf-8")
        config = {"post_search": {"top_functions": 1}, "smoke_test": False}
        (snapshot / "paper_parameters.json").write_text(json.dumps(config), encoding="utf-8")
        (root / "manifest.json").write_text(json.dumps(dict(
            config=config, source_sha256=source_hashes(snapshot))), encoding="utf-8")
        (root / "checkpoint.json").write_text(json.dumps(dict(
            search_complete=True, retraining_complete=False, selected=[{"rank": 1}],
            evaluations=[{"score": 1.2}], retraining=[{"function": "relu", "seed": 0}]
        )), encoding="utf-8")
        (root / "models").mkdir()
        (root / "models" / "best_search_seed_0.keras").write_bytes(b"existing-model")

    def test_migration_preserves_original_and_checkpoint_and_records_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, target = Path(tmp) / "original", Path(tmp) / "recovery"
            self.fixture(source)
            original = {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()}
            prepare(source, target, no_xla=True)
            for relative, data in original.items():
                self.assertEqual((source / relative).read_bytes(), data)
            self.assertEqual((source / "checkpoint.json").read_bytes(), (target / "checkpoint.json").read_bytes())
            self.assertEqual((target / "models" / "best_search_seed_0.keras").read_bytes(), b"existing-model")
            provenance = target / "retrain_recovery_provenance"
            self.assertEqual((provenance / "original_manifest.json").read_bytes(), original[Path("manifest.json")])
            self.assertEqual((provenance / "original_source_snapshot" / "run_paper.py").read_bytes(),
                             original[Path("source_snapshot/run_paper.py")])
            manifest = json.loads((target / "manifest.json").read_text())
            self.assertEqual(manifest["source_sha256"], source_hashes(target / "source_snapshot"))
            self.assertEqual(manifest["retrain_recovery"]["existing_retraining_records"], 1)
            patched = (target / "source_snapshot" / "run_paper.py").read_text()
            class Recorder:
                def save(self, path, **kwargs):
                    self.paths.append(path)
            recorder = Recorder()
            recorder.paths = []
            namespace = {}
            exec(compile(patched, "runner", "exec"), namespace)
            namespace["search"](target, recorder, 0)
            for name in ("discovered_01", "discovered_02", "relu"):
                namespace["retrain"](target, recorder, name, 0)
            self.assertEqual(len(set(recorder.paths)), 4)
            self.assertTrue(all(p.endswith(".keras") for p in recorder.paths))
            self.assertIn("jit_compile=False", (target / "source_snapshot" / "paper_afos" / "model.py").read_text())

    def test_save_only_repair_keeps_model_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, target = Path(tmp) / "original", Path(tmp) / "recovery"
            self.fixture(source)
            prepare(source, target)
            relative = Path("source_snapshot/paper_afos/model.py")
            self.assertEqual((source / relative).read_bytes(), (target / relative).read_bytes())

    def test_refuses_incomplete_or_modified_source_without_creating_destination(self):
        for problem in ("incomplete", "source_changed", "config_changed", "unknown_save"):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as tmp:
                source, target = Path(tmp) / "original", Path(tmp) / "recovery"
                self.fixture(source)
                if problem == "incomplete":
                    (source / "checkpoint.json").write_text('{"search_complete": false}')
                elif problem == "source_changed":
                    (source / "source_snapshot" / "run_paper.py").write_text(RUNNER + "# edited\n")
                elif problem == "config_changed":
                    (source / "source_snapshot" / "paper_parameters.json").write_text('{}')
                else:
                    (source / "source_snapshot" / "run_paper.py").write_text(RUNNER.replace('{name}_seed_{seed}', 'unexpected_{seed}'))
                    path = source / "manifest.json"
                    manifest = json.loads(path.read_text())
                    manifest["source_sha256"] = source_hashes(source / "source_snapshot")
                    path.write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):
                    prepare(source, target)
                self.assertFalse(target.exists())

    def test_refuses_existing_or_nested_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "original"
            self.fixture(source)
            other = Path(tmp) / "existing"
            other.mkdir()
            for target in (source, source / "nested", other):
                with self.assertRaises(ValueError):
                    prepare(source, target)

    def test_wrong_retrain_name_is_corrected_without_changing_search(self):
        wrong = RUNNER.replace('f"{name}_seed_{seed}"', 'f"best_search_seed_{seed}.keras"')
        patched = fix_save_path(wrong, "retrain", 'f"{name}_seed_{seed}.keras"')
        self.assertIn('f"best_search_seed_{seed}"', patched)
        self.assertIn('f"{name}_seed_{seed}.keras"', patched)

    def test_xla_patch_handles_existing_keyword_and_trailing_comma(self):
        for source in (MODEL, MODEL.replace('loss="mse"', 'loss="mse",'),
                       MODEL.replace('loss="mse"', 'loss="mse", jit_compile="auto"')):
            patched = disable_xla(source)
            ast.parse(patched)
            self.assertEqual(patched.count("jit_compile="), 1)
            self.assertIn("jit_compile=False", patched)


if __name__ == "__main__":
    unittest.main()
