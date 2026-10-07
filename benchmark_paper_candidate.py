"""Time exactly one CIFAR-10 candidate using the paper search protocol."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
from datetime import datetime

from paper_afos.search import expression, fitness_score
from run_paper import Tee, load_config, training_seed, write_csv, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chromosome", type=int, nargs="+", default=[10, 0, 2])
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("KERAS_HOME", str(Path(__file__).parent / ".keras"))
    log = (root / "run.log").open("w", encoding="utf-8", buffering=1)
    original_out, original_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = Tee(original_out, log), Tee(original_err, log)
    start = time.perf_counter()
    result = dict(started_at=datetime.now().astimezone().isoformat(), status="initializing",
                  chromosome=args.chromosome, activation_expression=expression(args.chromosome),
                  selection_note="A preselected valid search candidate (default EELU-1), not a random population sample")
    write_json(root / "progress.json", result)
    try:
        import tensorflow as tf
        from paper_afos.model import build_model, load_data, train_candidate
        config = load_config(Path(__file__).with_name("paper_parameters.json"))
        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            raise RuntimeError("GPU is required for this benchmark")
        result.update(config=config, tensorflow_version=tf.__version__,
                      devices=[str(d) for d in gpus], initialization_seconds=time.perf_counter()-start)
        t = time.perf_counter()
        train, validation, test, split = load_data(config)
        del test, split
        result["data_loading_seconds"] = time.perf_counter()-t
        result["train_images"], result["validation_images"] = len(train[0]), len(validation[0])
        t = time.perf_counter()
        tf.keras.utils.set_random_seed(training_seed(0, 1, 0))
        model = build_model(config, chromosome=args.chromosome)
        result["model_build_seconds"] = time.perf_counter()-t
        result["parameters"] = model.count_params()
        epoch_times = []
        def progress():
            # Progress is advisory: a Windows reader/file scanner may temporarily
            # deny atomic replacement. Never abort expensive training for this.
            try:
                write_json(root / "progress.json", result)
            except PermissionError:
                print("Progress file is temporarily locked; continuing training", flush=True)
        class Timer(tf.keras.callbacks.Callback):
            def on_epoch_begin(self, epoch, logs=None):
                self.epoch_start = time.perf_counter()
                self.epoch = epoch
                result.update(status="training", current_epoch=epoch+1, completed_batches=0)
                progress()

            def on_epoch_end(self, epoch, logs=None):
                seconds = time.perf_counter()-self.epoch_start
                epoch_times.append(dict(epoch=epoch+1, seconds=seconds, **{k: float(v) for k,v in (logs or {}).items()}))
                result["epochs"] = epoch_times
                print(f"TIMING epoch={epoch+1} train+validation={seconds:.3f}s", flush=True)
                progress()
        original_fit = model.fit
        def timed_fit(*args, **kwargs):
            kwargs["callbacks"] = [*kwargs.get("callbacks", []), Timer()]
            return original_fit(*args, **kwargs)
        model.fit = timed_fit
        print(f"Benchmark chromosome={args.chromosome}: {result['activation_expression']}", flush=True)
        print("CIFAR-10: train=45000, validation=5000; test not evaluated", flush=True)
        t = time.perf_counter()
        metrics, history = train_candidate(model, train, validation, config, verbose=2)
        result["training_seconds"] = time.perf_counter()-t
        result["metrics"] = metrics
        result["score"] = fitness_score(metrics["validation_loss"], metrics["validation_accuracy"])
        model.fit = original_fit
        t = time.perf_counter()
        if metrics["status"] != "nonfinite":
            model.save(str(root / "model"), include_optimizer=True)
        result["model_save_seconds"] = time.perf_counter()-t
        result["evaluation_seconds_without_data_loading"] = result["model_build_seconds"] + result["training_seconds"]
        result["total_wall_seconds"] = time.perf_counter()-start
        result["status"] = "completed"
        result["finished_at"] = datetime.now().astimezone().isoformat()
        write_csv(root / "epoch_times.csv", epoch_times)
        write_csv(root / "history.csv", history)
        write_json(root / "result.json", result)
        progress()
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    except BaseException as exc:
        result.update(status="failed", error=repr(exc), total_wall_seconds=time.perf_counter()-start)
        write_json(root / "progress.json", result)
        raise
    finally:
        sys.stdout, sys.stderr = original_out, original_err
        log.close()


if __name__ == "__main__":
    main()
