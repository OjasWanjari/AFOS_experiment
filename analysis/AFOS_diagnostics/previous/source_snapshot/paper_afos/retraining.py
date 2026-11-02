"""Planning and accounting for pooled retraining (no TensorFlow import)."""
import copy
import hashlib
import json
import math
import shutil
import statistics
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .recording import read_results
from .search import BINARY, UNARY, expression, select_top
# These helpers are standard-library-only; importing them never starts a search.
from run_paper import clean, load_config, write_csv, write_json

RETRYABLE = {"runtime_error", "oom", "interrupted", "worker_crash", "running"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def split_signature(path):
    # Ignore ZIP timestamps/compression, but require identical saved NPY arrays.
    with zipfile.ZipFile(path) as archive:
        names = {"train_indices.npy", "validation_indices.npy"}
        if set(archive.namelist()) != names:
            raise ValueError(f"Unexpected split archive: {path}")
        return {name: hashlib.sha256(archive.read(name)).hexdigest() for name in sorted(names)}


def check_source_config(source, target, path):
    for key in ("training", "dataset", "cnn", "chromosome_schema"):
        if source[key] != target[key]:
            raise ValueError(f"Incompatible {key} in {path}")
    # Seed subsets vary between machines; the search protocol must not vary.
    for key, value in target["search"].items():
        if key not in ("seeds", "replications") and source["search"][key] != value:
            raise ValueError(f"Incompatible search.{key} in {path}")


def pool_candidates(results_root, config, overrides=None):
    """Choose one trajectory per seed by coverage, then rank valid expressions."""
    results_root = Path(results_root).resolve()
    overrides = overrides or {}
    paths = sorted(results_root.rglob("evaluation_results_seed_*.csv"))
    expected = set(config["search"]["seeds"])
    sources = {seed: [] for seed in expected}
    for path in paths:
        if path.parent.name != "results":
            continue
        seed = int(path.stem.removeprefix("evaluation_results_seed_"))
        if seed not in expected:
            raise ValueError(f"Unexpected search seed {seed}; include it in the config or narrow --results-root")
        rows = read_results(path)
        counts = [r["evaluation_count"] for r in rows]
        if (not rows or any(r["seed"] != seed for r in rows)
                or any(type(n) is not int for n in counts)
                or sorted(counts) != list(range(1, len(rows) + 1))):
            raise ValueError(f"Rows must contain one contiguous trajectory starting at evaluation 1: {path}")
        sources[seed].append((path, rows))
    if set(overrides) - expected:
        raise ValueError("--source override has a seed absent from the config")
    pooled, audit, warnings = [], [], []
    reference_split = None
    split_path = None
    for seed in sorted(expected):
        available = sources[seed]
        if not available:
            raise ValueError(f"No search results found for required seed {seed} under {results_root}")
        if seed in overrides:
            chosen = [item for item in available if item[0].parent.parent == Path(overrides[seed]).resolve()]
            if len(chosen) != 1:
                raise ValueError(f"--source {seed} does not identify a run below --results-root")
        else:
            longest = max(len(rows) for _, rows in available)
            chosen = [item for item in available if len(item[1]) == longest]
            if len({sha256(path) for path, _ in chosen}) > 1:
                raise ValueError(f"Ambiguous equal-length trajectories for seed {seed}; use --source {seed}=RUN_DIRECTORY")
        path, rows = chosen[0]
        run = path.parent.parent
        manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        check_source_config(manifest["config"], config, run)
        if (manifest.get("unary_indices") != list(UNARY)
                or manifest.get("binary_indices") != list(BINARY)):
            raise ValueError(f"Incompatible primitive ordering: {run}")
        signature = split_signature(run / "split_indices.npz")
        if reference_split is not None and signature != reference_split:
            raise ValueError("Cannot pool results with different train/validation splits")
        reference_split, split_path = signature, run / "split_indices.npz"
        expected_count = config["search"]["population_size"] * config["search"]["generations"]
        if len(rows) > expected_count:
            raise ValueError(f"Too many evaluations in {path}")
        eligible = []
        for row in rows:
            if expression(row["chromosome"]) != row["activation_expression"]:
                raise ValueError(f"Chromosome/expression mismatch: {path}, evaluation {row['evaluation_count']}")
            values = [row.get(key) for key in ("train_loss", "train_accuracy", "validation_loss", "validation_accuracy")]
            if (row["status"] == "completed"
                    and row["epochs_completed"] == config["training"]["epochs"]
                    and all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)
                    and row["validation_loss"] >= 0 and row["train_loss"] >= 0
                    and 0 <= row["validation_accuracy"] <= 1 and 0 <= row["train_accuracy"] <= 1):
                # Whitelist fields: never carry test metrics into selection.
                eligible.append({key: row[key] for key in (
                    "seed", "evaluation_count", "chromosome", "activation_expression", "status",
                    "epochs_completed", "validation_loss", "validation_accuracy")})
        zero_epoch = sum(r.get("epochs_completed") == 0 for r in rows)
        if len(rows) < expected_count:
            warnings.append(f"Seed {seed}: only {len(rows)}/{expected_count} search evaluations recorded.")
        if zero_epoch:
            warnings.append(f"Seed {seed}: {zero_epoch} zero-epoch records; these are excluded. Search coverage is compromised.")
        audit.append(dict(seed=seed, selected_source=str(run), csv_sha256=sha256(path),
                          manifest_sha256=sha256(run / "manifest.json"),
                          recorded=len(rows), expected=expected_count, eligible=len(eligible),
                          status_counts=dict(Counter(r["status"] for r in rows)), zero_epoch_records=zero_epoch,
                          alternatives=[dict(path=str(p.parent.parent), recorded=len(rs), csv_sha256=sha256(p))
                                        for p, rs in available if p != path]))
        pooled.extend(eligible)
    selected = select_top(pooled, config["post_search"]["top_functions"])
    if len(selected) != config["post_search"]["top_functions"]:
        raise ValueError(f"Only {len(selected)} eligible distinct expression strings; cannot fill the requested top list")
    return selected, audit, warnings, split_path


def make_trials(selected, config, smoke=False):
    functions = [dict(function=f"discovered_{r['rank']:02d}", chromosome=r["chromosome"], baseline=None,
                      activation_expression=r["activation_expression"], search_seed=r["seed"],
                      search_evaluation=r["evaluation_count"], search_validation_accuracy=r["validation_accuracy"])
                 for r in selected]
    functions += [dict(function=name, chromosome=None, baseline=name, activation_expression=name)
                  for name in config["post_search"]["baselines"]]
    if smoke:
        functions = [functions[0], dict(function="relu", chromosome=None, baseline="relu", activation_expression="relu")]
    return [dict(f, seed=seed, trial_id=f"{f['function']}_seed_{seed}")
            for f in functions for seed in config["post_search"]["seeds"]]


def prepare_run(output, results_root, config_path, overrides, device="gpu", smoke=False, save_models=True):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError(f"Output already exists: {output}. Use --resume or a new directory.")
    config = load_config(config_path)
    if not config["search"]["seeds"] or not config["post_search"]["seeds"]:
        raise ValueError("Search and retraining seed lists must not be empty")
    if len(set(config["post_search"]["baselines"])) != len(config["post_search"]["baselines"]):
        raise ValueError("Baseline names must be unique")
    selected, audit, warnings, split_path = pool_candidates(results_root, config, overrides)
    execution_config = copy.deepcopy(config)
    if smoke:
        execution_config["post_search"].update(epochs=1, seeds=[0], replications=1)
    trials = make_trials(selected, execution_config, smoke)
    output.mkdir(parents=True)
    shutil.copy2(split_path, output / "split_indices.npz")
    source_root = Path(__file__).resolve().parents[1]
    snapshot = output / "source_snapshot"
    files = [source_root / "retrain_paper.py", source_root / "run_paper.py"]
    files += sorted((source_root / "paper_afos").glob("*.py"))
    for source in files:
        target = snapshot / source.relative_to(source_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    hashes = {str(path.relative_to(snapshot)).replace("\\", "/"): sha256(path)
              for path in snapshot.rglob("*.py")}
    plan = dict(format_version=1, created_at=now(), smoke=smoke, config=execution_config,
                selection_config=config, selected=selected, sources=audit, warnings=warnings,
                selection_policy="Best completed validation accuracy per exact expression string; ties: loss, seed, evaluation.",
                runtime=dict(device=device, jit_compile=False, memory_growth=True, save_models=save_models),
                source_sha256=hashes, split_sha256=sha256(output / "split_indices.npz"), trials=trials)
    write_json(output / "plan.json", plan)
    (output / "plan.sha256").write_text(sha256(output / "plan.json") + "\n", encoding="ascii")
    write_csv(output / "selected_top15.csv", selected)
    write_json(output / "selection_audit.json", dict(sources=audit, warnings=warnings))
    summarize(output, plan)
    return plan


def read_plan(output):
    output = Path(output)
    if sha256(output / "plan.json") != (output / "plan.sha256").read_text().strip():
        raise ValueError("Frozen plan changed. Create a new output directory for a different experiment.")
    plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    if plan["format_version"] != 1:
        raise ValueError("Unsupported plan format")
    for rel, digest in plan["source_sha256"].items():
        if sha256(output / "source_snapshot" / rel) != digest:
            raise ValueError(f"Frozen source changed: {rel}")
    if sha256(output / "split_indices.npz") != plan["split_sha256"]:
        raise ValueError("Frozen dataset split changed")
    return plan


def attempts_for(output, trial_id):
    return sorted((Path(output) / "trials" / trial_id).glob("attempt_*/result.json"))


def latest_result(output, trial):
    attempts = attempts_for(output, trial["trial_id"])
    return json.loads(attempts[-1].read_text(encoding="utf-8")) if attempts else None


def summarize(output, plan):
    rows = []
    for trial in plan["trials"]:
        row = latest_result(output, trial)
        rows.append(dict(trial, **{k: v for k, v in (row or {"status": "pending"}).items() if k not in trial}))
    write_csv(Path(output) / "results" / "retraining_results.csv", rows)
    summary = []
    for function in dict.fromkeys(t["function"] for t in plan["trials"]):
        group = [r for r in rows if r["function"] == function]
        good = [r["test_accuracy"] for r in group if r["status"] == "completed"
                and isinstance(r.get("test_accuracy"), (int, float)) and math.isfinite(r["test_accuracy"])]
        complete = len(good) == len(group)
        summary.append(dict(function=function, expected=len(group), completed=len(good),
                            status="complete" if complete else "incomplete",
                            test_accuracy_mean=statistics.mean(good) if complete else None,
                            test_accuracy_std=statistics.stdev(good) if complete and len(good) > 1 else None,
                            model_save_errors=sum(r.get("model_save_status") == "error" for r in group)))
    write_csv(Path(output) / "results" / "retraining_summary.csv", summary)
    write_json(Path(output) / "status.json", dict(updated_at=now(), smoke=plan["smoke"],
               planned=len(rows), status_counts=dict(Counter(r["status"] for r in rows)),
               all_evaluations_completed=all(r["status"] == "completed" for r in rows),
               model_save_errors=sum(r.get("model_save_status") == "error" for r in rows),
               selection_warnings=plan["warnings"]))
    return rows
