"""Fixed Table 3 candidates, in published test-mean order (not search order).

Reference: user-provided arXiv:2409.04915v1, printed page 16, Tables 3/4.
Use the mathematical expression for EELU-2, not Table 4's duplicated code.
Softplus and all other primitives use the reference run's saved implementation.
"""
import copy
import json
import re
import shutil
from pathlib import Path

from .retraining import latest_result, now, read_plan, sha256, summarize, write_csv, write_json
from .runtime_environment import environment_mismatches
from .search import expression


# EELU-3 is encoded as (x * erf(exp(x))) * erf(exp(x)), mathematically x*erf(exp(x))**2.
# Preserve the native operator evaluator, without adding clipping or custom gradients.
CATALOG = (
    ([1, 15, 18], "EELU-4", 82.74, 0.14),
    ([10, 1, 4], "EELU-2", 82.67, 0.30),
    ([10, 0, 2], "EELU-1", 82.66, 0.29),
    ([10, 10, 0, 2, 2], "EELU-3", 82.48, 0.61),
    ([10, 2, 0, 5, 5], "", 82.46, 0.54),
    ([0, 10, 6, 12, 1], "", 82.44, 0.37),
    ([10, 18, 15], "", 82.40, 0.47),
    ([9, 0, 10, 10, 6, 18, 12], "", 82.38, 0.39),
    ([1, 18, 10], "", 82.29, 0.18),
    ([2, 9, 0, 7, 7, 17, 10], "", 82.27, 0.35),
    ([9, 0, 10, 10, 6, 18, 9], "", 82.26, 0.13),
    ([0, 17, 7], "", 82.25, 0.47),
    ([2, 0, 15], "", 82.19, 0.51),
    ([0, 1, 6, 12, 1], "", 82.15, 0.29),
    ([2, 1, 18, 10, 15], "", 82.15, 0.33),
)


def catalog_rows():
    return [dict(function=f"paper_{rank:02d}", paper_table3_row=rank, eelu_name=name,
                 chromosome=list(chromosome), baseline=None, activation_expression=expression(chromosome),
                 paper_test_mean_percent=mean, paper_test_std_percent=std)
            for rank, (chromosome, name, mean, std) in enumerate(CATALOG, 1)]


def reference_environment(reference, plan):
    expected, records = None, []
    for trial in plan["trials"]:
        result = latest_result(reference, trial)
        if not result or result["status"] != "completed":
            continue
        attempt = reference / "trials" / trial["trial_id"] / result["attempt"]
        path = attempt / "environment.json"
        environment = json.loads(path.read_text(encoding="utf-8"))
        # Older records lack GPU model names, but TensorFlow logged them.
        if "gpu_names" not in environment:
            log = (attempt / "worker.log").read_text(encoding="utf-8", errors="replace")
            environment["gpu_names"] = re.findall(r"name: ([^\r\n]+?), pci bus id:", log)
        if environment["runtime"] != plan["runtime"]:
            raise ValueError(f"Runtime differs from reference plan: {path}")
        if environment["runtime"]["device"] == "gpu" and not environment["gpu_names"]:
            raise ValueError(f"Cannot verify reference GPU model from {path} or worker.log")
        if expected is not None:
            differences = environment_mismatches(expected, environment)
            if differences:
                raise ValueError(f"Reference trials have different environments: {path}: {differences}")
        else:
            expected = environment
        records.append(dict(trial_id=trial["trial_id"], environment_sha256=sha256(path)))
    if expected is None:
        raise ValueError("Reference needs at least one completed trial with environment.json and worker.log")
    return expected, records


def prepare_published_run(output, reference):
    output, reference = Path(output).resolve(), Path(reference).resolve()
    if output.exists():
        raise ValueError(f"Output already exists: {output}; use --resume or a new directory")
    old = read_plan(reference)
    config = copy.deepcopy(old["config"])
    if old["smoke"] or old.get("experiment") == "published_top15":
        raise ValueError("Reference must be the real pooled-candidate retraining run")
    if (config["post_search"]["seeds"] != [0, 1, 2, 3, 4]
            or config["post_search"]["epochs"] != 50):
        raise ValueError("Reference must use seeds 0..4 and 50 epochs")
    if old["runtime"].get("jit_compile") is not False or old["runtime"].get("memory_growth") is not True:
        raise ValueError("Reference must use JIT disabled and GPU memory growth enabled")
    environment, environment_records = reference_environment(reference, old)
    # Fail rather than encode the functions with a different primitive schema.
    source_root = Path(__file__).resolve().parents[1]
    reference_search = reference / "source_snapshot/paper_afos/search.py"
    if reference_search.read_text(encoding="utf-8") != (source_root / "paper_afos/search.py").read_text(encoding="utf-8"):
        raise ValueError("Reference chromosome/evaluator schema differs from this implementation")
    selected = catalog_rows()
    trials = [dict(row, seed=seed, trial_id=f"{row['function']}_seed_{seed}")
              for row in selected for seed in config["post_search"]["seeds"]]
    output.mkdir(parents=True)
    snapshot = output / "source_snapshot"
    # Copy only hash-verified source, never cached code or previous model weights.
    for rel in old["source_sha256"]:
        target = snapshot / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(reference / "source_snapshot" / rel, target)
    # Only the worker's environment recording/checking changes. Model, data loader,
    # optimizer, stopping rule, evaluator, epoch recorder and coordinator stay frozen.
    additions = ("paper_afos/retrain_worker.py", "paper_afos/runtime_environment.py",
                 "paper_afos/published_top15.py", "retrain_published_top15.py")
    for rel in additions:
        shutil.copy2(source_root / rel, snapshot / rel)
    shutil.copy2(reference / "split_indices.npz", output / "split_indices.npz")
    preserved = {rel: digest for rel, digest in old["source_sha256"].items() if rel not in additions}
    for rel, digest in preserved.items():
        if sha256(snapshot / rel) != digest:
            raise ValueError(f"Reference computation source changed: {rel}")
    plan = dict(format_version=1, experiment="published_top15", created_at=now(), smoke=False,
                config=config, selected=selected, trials=trials, runtime=copy.deepcopy(old["runtime"]),
                warnings=["Same recorded environment does not guarantee bitwise GPU determinism; original driver version was not recorded."],
                selection_policy="Fixed Table 3 rows from arXiv:2409.04915v1; no new search or test-based selection.",
                reference=dict(path=str(reference), plan_sha256=sha256(reference / "plan.json"),
                               computation_source_sha256=preserved, environments=environment_records),
                reference_environment=environment,
                source_sha256={str(p.relative_to(snapshot)).replace("\\", "/"): sha256(p)
                               for p in snapshot.rglob("*.py")},
                split_sha256=sha256(output / "split_indices.npz"))
    write_json(output / "plan.json", plan)
    (output / "plan.sha256").write_text(sha256(output / "plan.json") + "\n", encoding="ascii")
    write_csv(output / "selected_top15.csv", selected)
    write_json(output / "reference_audit.json", plan["reference"])
    summarize(output, plan)
    return plan
