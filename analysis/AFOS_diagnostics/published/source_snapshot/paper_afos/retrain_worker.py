"""One fresh process, one model, one retraining seed."""
import math
import os
import time
import traceback
from pathlib import Path

from .recording import EpochCSV
from .retraining import now, read_plan, write_json
from .runtime_environment import environment_mismatches, inspect_environment


def run_worker(output, trial_id, attempt, quiet=False):
    output, attempt = Path(output).resolve(), Path(attempt).resolve()
    plan = read_plan(output)
    trial = next(t for t in plan["trials"] if t["trial_id"] == trial_id)
    result = dict(trial, status="running", started_at=now(), pid=os.getpid(),
                  attempt=attempt.name, test_accuracy=None, test_loss=None)
    write_json(attempt / "result.json", result)
    started = time.monotonic()
    try:
        environment = inspect_environment(plan["runtime"])
        write_json(attempt / "environment.json", environment)
        if "reference_environment" in plan:
            differences = environment_mismatches(plan["reference_environment"], environment)
            write_json(attempt / "environment_check.json", dict(matched=not differences, differences=differences))
            if differences:
                raise RuntimeError("Reference environment mismatch; no training started: " + "; ".join(differences))
        import tensorflow as tf
        from .model import build_model, load_data, train_candidate
        tf.keras.utils.set_random_seed(trial["seed"])
        train, validation, test, _ = load_data(plan["config"], smoke=plan["smoke"],
                                              split_file=output / "split_indices.npz")
        model = build_model(plan["config"], chromosome=trial["chromosome"], baseline=trial["baseline"],
                            jit_compile=False)
        print(f"Retrain {trial_id}: {trial['activation_expression']}", flush=True)
        with EpochCSV(attempt, trial_id) as record:
            metrics, _ = train_candidate(model, train, validation, plan["config"], search=False,
                                         verbose=0 if quiet else 2, on_epoch=record)
        result.update(metrics)
        if result["status"] == "completed":
            if result["epochs_completed"] != plan["config"]["post_search"]["epochs"]:
                raise RuntimeError("Training ended before the planned epoch count")
            loss, accuracy = model.evaluate(*test, batch_size=plan["config"]["training"]["batch_size"], verbose=0)
            result.update(test_loss=float(loss), test_accuracy=float(accuracy))
            if not all(math.isfinite(float(v)) for v in (loss, accuracy)):
                result["status"] = "nonfinite_test"
            # Commit test metrics before attempting model serialization.
            write_json(attempt / "result.json", result)
            if result["status"] == "completed" and plan["runtime"]["save_models"]:
                try:
                    model.save(str(attempt / "model.keras"))
                    result["model_save_status"] = "saved"
                except Exception as exc:
                    # A serialization failure must not erase a valid evaluation.
                    result.update(model_save_status="error", model_save_error=repr(exc))
                    (attempt / "save_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            else:
                result["model_save_status"] = "not_saved"
    except KeyboardInterrupt:
        result.update(status="interrupted", error="KeyboardInterrupt")
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        oom = any(word in message.lower() for word in ("out of memory", "resourceexhausted", "resource_exhausted"))
        result.update(status="oom" if oom else "runtime_error", error=message)
        (attempt / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        traceback.print_exc()
    finally:
        result.update(finished_at=now(), elapsed_seconds=time.monotonic() - started)
        write_json(attempt / "result.json", result)
    return 0 if result["status"] in ("completed", "nonfinite", "nonfinite_test") and result.get("model_save_status") != "error" else 1
