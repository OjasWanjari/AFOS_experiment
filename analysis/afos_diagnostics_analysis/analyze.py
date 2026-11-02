"""Audit the returned AFOS diagnostic logs without importing TensorFlow or training."""
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent / "AFOS_diagnostics"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def main():
    audit = {"input": "analysis/AFOS_diagnostics", "groups": {}, "attempts": [], "source_sha256": {}}
    for group in ("published", "previous"):
        root = ROOT / group
        plan, diagnostic = read(root / "plan.json"), read(root / "diagnostic_plan.json")
        for name in ("plan", "diagnostic_plan"):
            assert digest(root / f"{name}.json") == (root / f"{name}.sha256").read_text().strip()
        assert digest(root / "plan.json") == diagnostic["reference_plan_sha256"]
        for name, value in plan["source_sha256"].items():
            assert digest(root / "source_snapshot" / name) == value, name
        for name, value in diagnostic["diagnostic_source_sha256"].items():
            assert digest(root / name) == value, name
        assert digest(root / "split_indices.npz") == plan["split_sha256"]
        trials = {t["trial_id"]: t for t in plan["trials"]}
        present = []
        for path in sorted((root / "diagnostics").glob("*/*/*/result.json")):
            folder = path.parent
            result = read(path)
            assert result["trial_id"] == folder.parent.parent.name
            assert result["mode"] == folder.parent.name
            environment = read(folder / "environment_check.json")
            assert environment == {"matched": True, "differences": []}
            stats = read(folder / "variables.json")
            by_kind = {"model": {"size": 0, "nan": 0, "posinf": 0, "neginf": 0, "max_abs_finite": None},
                       "momentum": {"size": 0, "nan": 0, "posinf": 0, "neginf": 0, "max_abs_finite": None}}
            with np.load(folder / "failure_or_final_variables.npz", allow_pickle=False) as arrays:
                assert set(arrays.files) == {s["key"] for s in stats}
                for stat in stats:
                    a = arrays[stat["key"]]
                    finite = np.isfinite(a)
                    assert list(a.shape) == stat["shape"] and a.size == stat["size"]
                    assert int(finite.sum()) == stat["finite_count"]
                    maximum = float(np.abs(a[finite]).max()) if finite.any() else None
                    assert maximum == stat["max_abs_finite"]
                    kind = "momentum" if stat["name"].endswith("_momentum") else "model" if not stat["name"].startswith("SGD/") else None
                    if kind:
                        counts = by_kind[kind]
                        counts["size"] += int(a.size)
                        counts["nan"] += int(np.isnan(a).sum())
                        counts["posinf"] += int(np.isposinf(a).sum())
                        counts["neginf"] += int(np.isneginf(a).sum())
                        if maximum is not None:
                            counts["max_abs_finite"] = max(counts["max_abs_finite"] or 0, maximum)
            batches = jsonl(folder / "batches.jsonl")
            train = [b for b in batches if b["phase"] == "train" and b["event"] == "batch_end"]
            bad = [b for b in train if any(not math.isfinite(float(v)) for v in b["metrics"].values())]
            error = (folder / "error.txt").read_text() if (folder / "error.txt").exists() else ""
            node = re.search(r"Detected at node (.+?) defined at", error)
            op = re.search(r'output 0 of graph op "(\w+)"', error)
            values = re.search(r"Tensor had (.+?) values", error)
            log = (folder / "worker.log").read_text()
            item = {"path": folder.relative_to(ROOT).as_posix(), "result": result,
                    "expression": trials[result["trial_id"]]["activation_expression"],
                    "environment_matched": environment["matched"],
                    "detected_node": node.group(1) if node else None,
                    "detected_op": op.group(1) if op else None,
                    "detected_values": values.group(1) if values else None,
                    "atan_gradient_stack": "_AtanGrad" in error,
                    "train_batches_ended": len(train), "first_nonfinite_train_metric": bad[0] if bad else None,
                    "last_train_batches": train[-5:], "saved_variables": by_kind,
                    "epochs_logged": len(jsonl(folder / "epochs.jsonl")),
                    "layout_warning": "layout failed" in log,
                    "oom_marker": any(s in log for s in ("CUDA_ERROR_OUT_OF_MEMORY", "RESOURCE_EXHAUSTED", "ResourceExhaustedError"))}
            audit["attempts"].append(item)
            present.append(result["trial_id"])
        expected = {f["trial_id"] for f in diagnostic["failures"]}
        audit["groups"][group] = {"planned_failed_trials": len(expected), "observed_unique_trials": len(set(present)),
                                   "missing_trials": sorted(expected - set(present)), "hash_checks_passed": True}
    for path in sorted(ROOT.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix not in (".pyc", ".pyo"):
            audit["source_sha256"][path.relative_to(ROOT).as_posix()] = digest(path)
    audit["status_counts"] = dict(Counter(a["result"]["status"] for a in audit["attempts"]))
    audit["note"] = "Saved variables are post-stop snapshots, not pre-failure checkpoints; detected node is not proof of the globally earliest nonfinite operation."
    (OUT / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"attempts": len(audit["attempts"]), "statuses": audit["status_counts"], "groups": audit["groups"]}, indent=2))
    for a in audit["attempts"]:
        print(a["path"], json.dumps(a["saved_variables"]))


if __name__ == "__main__":
    main()
