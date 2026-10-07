"""Run the paper's CIFAR-10 base-CNN experiment. See PAPER_REPRODUCTION.md."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import shutil
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

from paper_afos.search import (BINARY, UNARY, expression, fitness_score,
                               next_population, random_chromosome, select_top)
from paper_afos.recording import EpochCSV, event, read_results


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    # On Windows a reader or virus scanner can briefly hold a file without
    # delete sharing. Preserve atomic checkpoints, but tolerate a short lock.
    for attempt in range(20):
        try:
            temp.replace(path)
            break
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.1)


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v
                             for k, v in clean(row).items()})


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    s, t, p = config["search"], config["training"], config["post_search"]
    for section, field in [(s, "population_size"), (s, "generations"), (t, "epochs"),
                           (t, "batch_size"), (p, "epochs"), (p, "top_functions")]:
        if type(section[field]) is not int or section[field] < 1:
            raise ValueError(f"{field} must be a positive integer")
    for section in (s, p):
        if len(section["seeds"]) != section["replications"] or len(set(section["seeds"])) != len(section["seeds"]):
            raise ValueError("Unique seeds must match replications")
        if any(type(seed) is not int or not 0 <= seed < 2**31 for seed in section["seeds"]):
            raise ValueError("Seeds must be nonnegative 31-bit integers")
    if not 2 <= s["selected_parents"] <= s["population_size"]:
        raise ValueError("At least two parents are needed")
    for field in ("crossover_probability", "mutation_probability"):
        if not 0 <= s[field] <= 1:
            raise ValueError(f"Invalid {field}")
    if s["max_crossover_children"] % 2 or s["max_crossover_children"] < 0 or s["max_mutation_children"] < 0:
        raise ValueError("Invalid child counts")
    if s["population_size"] - s["selected_parents"] - s["max_crossover_children"] - s["max_mutation_children"] < s["minimum_random_individuals"]:
        raise ValueError("GA counts exceed population allocation")
    if config["chromosome_schema"] != "paper_v1_19_unary_11_binary_fig2":
        raise ValueError("Unsupported chromosome schema")
    if config["cnn"]["filters"] != [28, 32, 64, 64, 128, 128]:
        raise ValueError("Expected the Table 2 CNN")
    if not 0 <= config["cnn"]["dropout_rate"] < 1:
        raise ValueError("Invalid dropout")
    if config["dataset"]["train_size"] + config["dataset"]["validation_size"] != 50000 or config["dataset"]["test_size"] != 10000:
        raise ValueError("Invalid CIFAR-10 split")
    if s["fitness"] != "validation_accuracy / validation_loss" or p["selection_metric"] != "validation_accuracy":
        raise ValueError("Selection must use validation metrics")
    # Do not silently ignore edits to descriptive but fixed protocol fields.
    expected = [(s, "selection", "ranking"), (s, "crossover", "single_point"),
                (s, "mutation", "single_point"), (t, "optimizer", "SGD"),
                (config["dataset"], "name", "CIFAR-10"),
                (t, "loss", "categorical_crossentropy"), (p, "evaluation_split", "test"),
                (p, "retrain_from_scratch", True), (t, "stop_on_undefined_loss", True)]
    if any(section[key] != value for section, key, value in expected):
        raise ValueError("Unsupported protocol change")
    if t["learning_rate"] <= 0 or not 0 <= t["momentum"] <= 1 or not 0 <= t["first_epoch_accuracy_threshold"] <= 1:
        raise ValueError("Invalid training hyperparameters")
    if not set(p["baselines"]) <= {"relu", "swish", "mish", "gelu", "serf", "elu"}:
        raise ValueError("Unsupported baseline")
    return config


def training_seed(seed, generation, individual):
    return int.from_bytes(hashlib.sha256(f"{seed}/{generation}/{individual}".encode()).digest()[:4], "little") % (2**31)


def save_search_outputs(root, state, config):
    rows = state["evaluations"]
    best = []
    for seed in config["search"]["seeds"]:
        selected = [r for r in rows if r["seed"] == seed]
        write_csv(root / "results" / f"evaluation_results_seed_{seed}.csv", selected)
        valid = [r for r in selected if r["status"] != "nonfinite"]
        if valid:
            best.append(max(valid, key=lambda r: r["score"]))
        write_csv(root / "results" / f"top3_seed_{seed}.csv", select_top(selected, 3, metric="score"))
    write_csv(root / "results" / "best_by_seed.csv", best)
    write_csv(root / "results" / "generation_history.csv", state["generations"])
    # Rebuild derived files even if a crash happened just after checkpoint commit.
    write_csv(root / "results" / "selected_top15.csv", state["selected"])
    write_csv(root / "results" / "retraining_results.csv", state["retraining"])


def search(root, state, config, train, validation, verbose, smoke):
    import tensorflow as tf
    from paper_afos.model import build_model, train_candidate
    s = config["search"]
    while state["seed_index"] < len(s["seeds"]):
        seed = s["seeds"][state["seed_index"]]
        generation = state["generation"]
        if state["population"] is None:
            rng = random.Random(seed)
            state["population"] = [random_chromosome(rng) for _ in range(s["population_size"])]
            if smoke:
                state["population"][0] = [2, 0, 6]  # max(x,relu(x)) = ReLU
        done = [r for r in state["evaluations"] if r["seed"] == seed and r["generation"] == generation]
        for individual in range(len(done), s["population_size"]):
            chromosome = state["population"][individual]
            tf.keras.backend.clear_session()
            fit_seed = training_seed(seed, generation, individual)
            tf.keras.utils.set_random_seed(fit_seed)
            model = build_model(config, chromosome=chromosome)
            print(f"seed={seed} generation={generation}/{s['generations']} individual={individual+1}/{s['population_size']} {expression(chromosome)}", flush=True)
            evaluation_count = (generation - 1) * s["population_size"] + individual + 1
            with EpochCSV(root, f"search_{seed}_{evaluation_count}") as record:
                metrics, history = train_candidate(model, train, validation, config,
                                                   verbose=verbose, on_epoch=record)
            score = fitness_score(metrics["validation_loss"], metrics["validation_accuracy"])
            if metrics["status"] == "nonfinite":
                score = 0.0
            row = clean(dict(seed=seed, generation=generation,
                             evaluation_count=(generation - 1) * s["population_size"] + individual + 1,
                             training_seed=fit_seed, chromosome=chromosome,
                             activation_expression=expression(chromosome), score=score, **metrics))
            prior = [r["score"] for r in state["evaluations"] if r["seed"] == seed and r["status"] != "nonfinite"]
            if row["status"] != "nonfinite" and (not prior or score > max(prior)):
                model.save(str(root / "models" / f"best_search_seed_{seed}"), include_optimizer=True)
            state["evaluations"].append(row)
            done.append(row)
            write_json(root / "checkpoint.json", state)
            # Refresh both raw and derived CSVs after every completed evaluation.
            save_search_outputs(root, state, config)
            event(root, "evaluation_completed", f"search_{seed}_{evaluation_count}", json.dumps(row))
            print(f"status={row['status']} validation_accuracy={row['validation_accuracy']} score={score}", flush=True)
            del model
        entry = dict(seed=seed, generation=generation, best_score=max(r["score"] for r in done),
                     best_validation_accuracy=max((r["validation_accuracy"] for r in done if r["validation_accuracy"] is not None), default=None))
        if generation < s["generations"]:
            state["population"], counts = next_population(state["population"], [r["score"] for r in done], s,
                                                         random.Random(training_seed(seed, generation, 999999)))
            entry.update(counts)
            state["generation"] += 1
        else:
            state["population"] = None
            state["seed_index"] += 1
            state["generation"] = 1
        state["generations"].append(entry)
        write_json(root / "checkpoint.json", state)
        save_search_outputs(root, state, config)
    state["search_complete"] = True
    state["selected"] = select_top(state["evaluations"], config["post_search"]["top_functions"])
    write_json(root / "checkpoint.json", state)
    write_csv(root / "results" / "selected_top15.csv", state["selected"])


def retrain(root, state, config, train, validation, test, verbose):
    import tensorflow as tf
    from paper_afos.model import build_model, train_candidate
    if not state["search_complete"]:
        raise ValueError("Finish search before retraining")
    if len(state["selected"]) < config["post_search"]["top_functions"]:
        raise ValueError("Too few distinct completed candidates for the requested top functions; inspect search results")
    candidates = [(f"discovered_{r['rank']:02d}", r["chromosome"], None, r["activation_expression"])
                  for r in state["selected"]]
    candidates += [(name, None, name, name) for name in config["post_search"]["baselines"]]
    for name, chromosome, baseline, formula in candidates:
        for seed in config["post_search"]["seeds"]:
            if any(r["function"] == name and r["seed"] == seed for r in state["retraining"]):
                continue
            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(seed)
            model = build_model(config, chromosome=chromosome, baseline=baseline)
            print(f"Retrain {name} seed={seed} epochs={config['post_search']['epochs']}", flush=True)
            with EpochCSV(root, f"{name}_{seed}") as record:
                metrics, history = train_candidate(model, train, validation, config,
                                                   search=False, verbose=verbose, on_epoch=record)
            test_loss, test_accuracy = None, None
            if metrics["status"] == "completed":
                test_loss, test_accuracy = model.evaluate(*test, batch_size=config["training"]["batch_size"], verbose=0)
                if not math.isfinite(test_loss) or not math.isfinite(test_accuracy):
                    metrics["status"] = "nonfinite_test"
                else:
                    model.save(str(root / "models" / f"{name}_seed_{seed}"), include_optimizer=True)
            row = clean(dict(function=name, seed=seed, chromosome=chromosome,
                             activation_expression=formula, test_loss=test_loss,
                             test_accuracy=test_accuracy, **metrics))
            state["retraining"].append(row)
            write_json(root / "checkpoint.json", state)
            write_csv(root / "results" / "retraining_results.csv", state["retraining"])
            summarize(root, state, config)
            event(root, "retraining_completed", f"{name}_{seed}", json.dumps(row))
            del model
    state["retraining_complete"] = True
    write_json(root / "checkpoint.json", state)


def summarize(root, state, config):
    rows = []
    for name in dict.fromkeys(r["function"] for r in state["retraining"]):
        trials = [r for r in state["retraining"] if r["function"] == name]
        good = [r for r in trials if r["status"] == "completed" and r["test_accuracy"] is not None]
        complete = len(good) == config["post_search"]["replications"]
        row = dict(function=name, activation_expression=trials[0]["activation_expression"],
                   expected_runs=config["post_search"]["replications"], observed_runs=len(trials),
                   successful_runs=len(good), complete=complete)
        # Do not hide divergent runs by averaging only survivors.
        for metric in ("train_loss", "validation_loss", "test_loss", "train_accuracy", "validation_accuracy", "test_accuracy"):
            values = [r[metric] for r in good]
            row[metric + "_mean"] = statistics.mean(values) if complete else None
            row[metric + "_std"] = statistics.stdev(values) if complete and len(values) > 1 else None
        rows.append(row)
    write_csv(root / "results" / "comparison_summary.csv", rows)
    return rows


def plots(root):
    # All charts are generated from persisted CSVs, never in-memory training history.
    state = dict(evaluations=[row for path in sorted((root / "results").glob("evaluation_results_seed_*.csv"))
                              for row in read_results(path)],
                 selected=read_results(root / "results" / "selected_top15.csv"))
    os.environ.setdefault("MPLCONFIGDIR", str(root / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    target = root / "plots"
    target.mkdir(exist_ok=True)
    for seed in sorted({r["seed"] for r in state["evaluations"]}):
        records = [r for r in state["evaluations"] if r["seed"] == seed]
        for xkey in ("evaluation_count", "generation"):
            fig, ax = plt.subplots()
            for metric in ("train_loss", "validation_loss"):
                best, points = math.inf, {}
                for r in records:
                    if r[metric] is not None and math.isfinite(r[metric]):
                        best = min(best, r[metric])
                    points[r[xkey]] = best if math.isfinite(best) else float("nan")
                ax.plot(list(points), list(points.values()), label=metric)
            ax.set(xlabel=xkey, ylabel="Best-so-far loss", title=f"Paper base CNN / seed {seed}")
            ax.legend()
            fig.tight_layout()
            fig.savefig(target / f"seed_{seed}_{xkey}.png", dpi=150)
            plt.close(fig)
    if state["selected"]:
        import numpy as np
        import tensorflow as tf
        from paper_afos.model import PaperActivation
        fig, axes = plt.subplots(math.ceil(len(state["selected"]) / 3), 3,
                                 figsize=(15, 3.5 * math.ceil(len(state["selected"]) / 3)), squeeze=False)
        xs = np.linspace(-5, 5, 501, dtype="float32")
        curve_rows = []
        for ax, row in zip(axes.flat, state["selected"]):
            ys = PaperActivation(row["chromosome"])(tf.constant(xs)).numpy()
            curve_rows.extend(dict(rank=row["rank"], chromosome=row["chromosome"],
                                   activation_expression=row["activation_expression"], x=float(x), y=float(y))
                              for x, y in zip(xs, ys))
            ys[~np.isfinite(ys)] = np.nan
            ax.plot(xs, ys)
            ax.axhline(0, color="grey", linewidth=0.5)
            ax.axvline(0, color="grey", linewidth=0.5)
            ax.set_title(f"Rank {row['rank']} / {row['chromosome']}", fontsize=9)
            ax.set(xlabel="x", ylabel="f(x)")
        for ax in list(axes.flat)[len(state["selected"]):]:
            ax.set_visible(False)
        fig.tight_layout()
        fig.savefig(target / "selected_activation_functions.png", dpi=150)
        plt.close(fig)
        write_csv(root / "results" / "activation_curve_samples.csv", curve_rows)
    comparison = read_results(root / "results" / "comparison_summary.csv")
    if comparison:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, metric in zip(axes, ("test_accuracy", "test_loss")):
            rows = [r for r in comparison if r[metric + "_mean"] is not None]
            ax.bar([r["function"] for r in rows], [r[metric + "_mean"] for r in rows],
                   yerr=[r[metric + "_std"] or 0 for r in rows], capsize=3)
            ax.set_ylabel(metric + " (mean +/- sample SD)")
            ax.tick_params(axis="x", labelrotation=90)
        fig.suptitle("Completed functions only (see comparison_summary.csv for failures)")
        fig.tight_layout()
        fig.savefig(target / "retraining_comparison.png", dpi=150)
        plt.close(fig)


def finish_outputs(root, state, config, stage):
    """One plotting batch per completed workload; identical resumes skip it."""
    if not state["search_complete"] or (stage != "search" and not state["retraining_complete"]):
        raise ValueError("Cannot plot before all requested runs have finished")
    save_search_outputs(root, state, config)
    summarize(root, state, config)
    inputs = sorted((root / "results").glob("*.csv"))
    digest = hashlib.sha256()
    for path in inputs:
        if path.name not in ("events.csv", "activation_curve_samples.csv"):
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
    fingerprint = digest.hexdigest()
    marker = root / "plots_complete.json"
    if marker.exists():
        previous = json.loads(marker.read_text(encoding="utf-8"))
        if previous["csv_sha256"] == fingerprint and all((root / p).exists() for p in previous["files"]):
            print("Plots already up to date; skipping.", flush=True)
            return
    event(root, "plotting_started", detail=stage)
    print("All requested runs finished. Generating plots from CSV once.", flush=True)
    plots(root)
    write_json(marker, dict(csv_sha256=fingerprint, stage=stage,
                            files=[p.relative_to(root).as_posix() for p in sorted((root / "plots").glob("*.png"))]))
    event(root, "plotting_completed", detail=stage)


class Tee:
    def __init__(self, original, log):
        self.original, self.log = original, log

    def write(self, value):
        self.original.write(value)
        self.log.write(value)
        self.log.flush()
        return len(value)

    def flush(self):
        self.original.flush()
        if not self.log.closed:
            self.log.flush()

    def close(self):
        # Logging handlers may close their stream at interpreter shutdown.
        # The owner, not the handler, manages both underlying streams.
        self.flush()

    def isatty(self):
        return False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("paper_parameters.json"))
    parser.add_argument("--stage", choices=("all", "search", "retrain"), default="all")
    parser.add_argument("--output", type=Path, help="New directory; refuses an existing directory")
    parser.add_argument("--resume", type=Path, help="Existing paper run with checkpoint")
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without loading TensorFlow/data")
    parser.add_argument("--smoke", action="store_true", help="Synthetic tiny integration check, not scientific results")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    if args.resume and args.output:
        parser.error("--resume and --output are mutually exclusive")
    config = load_config(args.config)
    if args.smoke:
        config["search"].update(population_size=6, generations=1, replications=1, seeds=[0], selected_parents=2,
                                max_crossover_children=2, max_mutation_children=1, minimum_random_individuals=1)
        config["training"].update(epochs=1, first_epoch_accuracy_threshold=-1)
        config["post_search"].update(top_functions=1, epochs=1, replications=1, seeds=[0], baselines=["relu"])
    config["smoke_test"] = args.smoke
    count = config["search"]["population_size"] * config["search"]["generations"] * config["search"]["replications"]
    if args.dry_run:
        print(json.dumps(dict(config=config, search_evaluations=count,
                              retraining_runs=(config["post_search"]["top_functions"] + len(config["post_search"]["baselines"])) * config["post_search"]["replications"]), indent=2))
        return
    import tensorflow as tf
    from paper_afos.model import build_model, load_data
    if args.device == "cpu":
        tf.config.set_visible_devices([], "GPU")
    gpus = tf.config.get_visible_devices("GPU")
    if args.device == "gpu" and not gpus:
        raise RuntimeError("No GPU detected. Fix the environment or explicitly use --device cpu")
    if args.resume:
        root = args.resume.resolve()
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if manifest["config"] != config:
            raise ValueError("Resume config differs from manifest; use the same config and --smoke setting")
        state = json.loads((root / "checkpoint.json").read_text(encoding="utf-8"))
        current_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files()}
        if current_hashes != manifest["source_sha256"]:
            raise ValueError("Source changed since this run; restore its source_snapshot before resuming")
    else:
        if args.stage == "retrain":
            parser.error("--stage retrain requires --resume")
        root = (args.output or Path("runs") / ("paper_smoke_" if args.smoke else "paper_") / datetime.now().strftime("%Y%m%d_%H%M%S_%f")).resolve()
        root.mkdir(parents=True, exist_ok=False)
        versions = {}
        for package in ("tensorflow", "tensorflow-cpu", "tensorflow-directml-plugin", "keras", "numpy", "scikit-learn", "matplotlib"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                pass
        snapshot = root / "source_snapshot"
        snapshot.mkdir()
        for path in source_files():
            destination = snapshot / path.relative_to(Path(__file__).parent)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
        write_json(snapshot / "paper_parameters.json", config)
        manifest = dict(config=config, created_at=datetime.now().astimezone().isoformat(),
                        python=sys.version, platform=platform.platform(), packages=versions,
                        devices=[str(d) for d in tf.config.get_visible_devices()],
                        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files()},
                        unary_indices=list(UNARY), binary_indices=list(BINARY),
                        note="Synthetic data; not paper results" if args.smoke else "See implementation_notes for assumptions")
        write_json(root / "manifest.json", manifest)
        state = dict(seed_index=0, generation=1, population=None, evaluations=[], generations=[],
                     selected=[], retraining=[], search_complete=False, retraining_complete=False)
        write_json(root / "checkpoint.json", state)
    original_stdout, original_stderr = sys.stdout, sys.stderr
    log = (root / "run.log").open("a", encoding="utf-8", buffering=1)
    sys.stdout, sys.stderr = Tee(original_stdout, log), Tee(original_stderr, log)
    print(f"Output: {root}\nSearch evaluations: {count}\nDevice: {args.device}", flush=True)
    try:
        event(root, "run_started", detail=args.stage)
        train, validation, test, split = load_data(config, args.smoke)
        if split is not None:
            import numpy as np
            np.savez_compressed(root / "split_indices.npz", **split)
        model = build_model(config, baseline="relu")
        lines = []
        model.summary(print_fn=lines.append)
        (root / "model_summary.txt").write_text("\n".join(lines), encoding="utf-8")
        del model
        (root / "models").mkdir(exist_ok=True)
        save_search_outputs(root, state, config)
        if args.stage in ("all", "search") and not state["search_complete"]:
            search(root, state, config, train, validation, 0 if args.quiet else 1, args.smoke)
        if args.stage in ("all", "retrain"):
            retrain(root, state, config, train, validation, test, 0 if args.quiet else 1)
        finish_outputs(root, state, config, args.stage)
        write_json(root / "status.json", dict(status="completed", stage=args.stage,
                                              search_complete=state["search_complete"],
                                              retraining_complete=state["retraining_complete"],
                                              failed_retraining_runs=sum(r["status"] != "completed" for r in state["retraining"])))
        print(f"Finished: {root}", flush=True)
        event(root, "run_completed", detail=args.stage)
    except BaseException as exc:
        event(root, "run_interrupted" if isinstance(exc, KeyboardInterrupt) else "run_failed", detail=repr(exc))
        write_json(root / "status.json", dict(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                                              error=repr(exc)))
        raise
    finally:
        sys.stdout, sys.stderr = original_stdout, original_stderr
        log.close()


def source_files():
    base = Path(__file__).parent
    return [Path(__file__), *sorted((base / "paper_afos").glob("*.py"))]


if __name__ == "__main__":
    main()
