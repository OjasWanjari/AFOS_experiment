"""Replay failed AFOS trials with frozen code; optionally stop at the first bad op.

See DIAGNOSE_RETRAINING_JA.md. Diagnostic outputs never replace official results.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

_snapshot = Path(__file__).resolve().parent / "source_snapshot"
if _snapshot.is_dir():
    sys.path.insert(0, str(_snapshot))

from paper_afos.retraining import latest_result, now, read_plan, sha256, write_json


def prepare(reference, output):
    reference, output = Path(reference).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("Output exists. Use --resume or another directory.")
    plan = read_plan(reference)
    if plan["smoke"]:
        raise ValueError("A real retraining reference is required")
    failed = []
    for trial in plan["trials"]:
        result = latest_result(reference, trial)
        if result and result["status"] in ("nonfinite", "nonfinite_test"):
            failed.append(dict(trial_id=trial["trial_id"], status=result["status"],
                               epochs_completed=result["epochs_completed"]))
    if not failed:
        raise ValueError("No recorded nonfinite failures in reference")
    # Obtain recorded environment from an actual trial, including GPU model.
    sample = next(t for t in plan["trials"] if latest_result(reference, t))
    saved = latest_result(reference, sample)
    attempt = reference / "trials" / sample["trial_id"] / saved["attempt"]
    environment = json.loads((attempt / "environment.json").read_text())
    if "gpu_names" not in environment:
        import re
        environment["gpu_names"] = re.findall(r"name: ([^\r\n]+?), pci bus id:", (attempt / "worker.log").read_text(errors="replace"))
    if environment["runtime"]["device"] == "gpu" and not environment["gpu_names"]:
        raise ValueError("Reference GPU name is missing")
    if environment["runtime"] != plan["runtime"]:
        raise ValueError("Reference runtime differs from plan")
    output.mkdir(parents=True)
    for name in ("plan.json", "plan.sha256", "split_indices.npz"):
        shutil.copy2(reference / name, output / name)
    for rel in plan["source_sha256"]:
        target = output / "source_snapshot" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(reference / "source_snapshot" / rel, target)
    # The original source snapshot remains byte-identical and hash-verified.
    shutil.copy2(Path(__file__), output / "diagnose_retraining.py")
    helper = Path(__file__).parent / "paper_afos/runtime_environment.py"
    shutil.copy2(helper, output / "diagnostic_environment.py")
    diagnostic = dict(created_at=now(), reference=str(reference), failures=failed,
                      environment=environment, reference_plan_sha256=sha256(output / "plan.json"),
                      diagnostic_source_sha256={name: sha256(output / name) for name in
                                               ("diagnose_retraining.py", "diagnostic_environment.py")})
    write_json(output / "diagnostic_plan.json", diagnostic)
    (output / "diagnostic_plan.sha256").write_text(sha256(output / "diagnostic_plan.json"))
    return diagnostic


def verify(output):
    plan = read_plan(output)
    if sha256(output / "diagnostic_plan.json") != (output / "diagnostic_plan.sha256").read_text().strip():
        raise ValueError("Diagnostic plan was modified")
    diagnostic = json.loads((output / "diagnostic_plan.json").read_text())
    if sha256(output / "plan.json") != diagnostic["reference_plan_sha256"]:
        raise ValueError("Reference plan was modified")
    for name, digest in diagnostic["diagnostic_source_sha256"].items():
        if sha256(output / name) != digest:
            raise ValueError(f"Diagnostic code changed: {name}")
    return plan, diagnostic


def run_training(model, train, validation, test, plan, trial, output, mode):
    import numpy as np
    import tensorflow as tf
    from paper_afos.model import train_candidate
    context = dict(phase="setup", epoch=None, batch=None)
    class TraceBatches(tf.keras.callbacks.Callback):
        def record(self, event, logs=None):
            # Strings preserve NaN/Inf without producing invalid JSON.
            values = {k: str(v) for k,v in (logs or {}).items()}
            with (output / "batches.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(dict(context, event=event, metrics=values)) + "\n")
        def on_epoch_begin(self, epoch, logs=None):
            context.update(epoch=epoch+1, phase="train", batch=None)
            self.record("epoch_begin")
        def on_train_batch_begin(self, batch, logs=None):
            context.update(phase="train", batch=batch)
            self.record("batch_begin")
        def on_train_batch_end(self, batch, logs=None):
            self.record("batch_end", logs)
        def on_test_batch_begin(self, batch, logs=None):
            context.update(phase="test" if context.get("final_test") else "validation", batch=batch)
            self.record("batch_begin")
        def on_test_batch_end(self, batch, logs=None):
            self.record("batch_end", logs)
    trace = TraceBatches()
    original_fit = model.fit
    def fit_with_trace(*args, **kwargs):
        kwargs["callbacks"] = list(kwargs.get("callbacks") or []) + [trace]
        return original_fit(*args, **kwargs)
    model.fit = fit_with_trace
    # Keep normal graph execution and the original training routine. Do not
    # switch to eager, replace SGD, or change the shuffle/dropout policy.
    if mode == "check":
        tf.debugging.enable_check_numerics(stack_height_limit=50, path_length_limit=120)
    result = dict(trial_id=trial["trial_id"], mode=mode, started_at=now())
    try:
        def epoch(row):
            with (output / "epochs.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({k:str(v) for k,v in row.items()}) + "\n")
        metrics, _ = train_candidate(model, train, validation, plan["config"], search=False, verbose=2, on_epoch=epoch)
        result.update(metrics)
        if metrics["status"] == "completed":
            if metrics["epochs_completed"] != plan["config"]["post_search"]["epochs"]:
                raise RuntimeError("Training ended before the planned epoch count")
            context.update(final_test=True, phase="test", batch=None)
            loss, accuracy = model.evaluate(*test, batch_size=plan["config"]["training"]["batch_size"], callbacks=[trace], verbose=0)
            result.update(test_loss=float(loss), test_accuracy=float(accuracy),
                          status="not_reproduced" if np.isfinite([loss,accuracy]).all() else "nonfinite_test")
    except Exception as exc:
        message = str(exc)
        checked = mode == "check" and "CheckNumerics" in message
        result.update(status="numerics_check_triggered" if checked else "runtime_error", error=repr(exc))
        (output / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        traceback.print_exc()
    finally:
        # A captured nonfinite intermediate is not automatically a corrupt final
        # loss: saturated exp->erf chains can contain Inf with finite outputs.
        if mode == "check":
            tf.debugging.disable_check_numerics()
        model.fit = original_fit
        result.update(context=context, finished_at=now())
        arrays, stats = {}, []
        try:
            variables = list(model.weights) + list(model.optimizer.variables)
            for i, variable in enumerate(variables):
                value = variable.numpy()
                arrays[f"variable_{i:03d}"] = value
                finite = np.isfinite(value)
                stats.append(dict(key=f"variable_{i:03d}", name=getattr(variable,"path",variable.name),
                                  shape=list(value.shape), finite_count=int(finite.sum()), size=int(value.size),
                                  max_abs_finite=float(np.max(np.abs(value[finite]))) if finite.any() else None))
            np.savez_compressed(output / "failure_or_final_variables.npz", **arrays)
            write_json(output / "variables.json", stats)
        except Exception as exc:
            result["variable_capture_error"] = repr(exc)
        write_json(output / "result.json", result)
    return result


def worker(output, trial_id, attempt, mode):
    plan, diagnostic = verify(output)
    trial = next(t for t in plan["trials"] if t["trial_id"] == trial_id)
    try:
        # Frozen helper is separate from the unchanged training source snapshot.
        from diagnostic_environment import inspect_environment, environment_mismatches
        env = inspect_environment(plan["runtime"])
        write_json(attempt / "environment.json", env)
        differences = environment_mismatches(diagnostic["environment"], env)
        write_json(attempt / "environment_check.json", dict(matched=not differences, differences=differences))
        if differences:
            raise RuntimeError("Environment mismatch; no training started: " + "; ".join(differences))
        import tensorflow as tf
        from paper_afos.model import build_model, load_data
        tf.keras.utils.set_random_seed(trial["seed"])
        train, validation, test, _ = load_data(plan["config"], split_file=output / "split_indices.npz")
        model = build_model(plan["config"], chromosome=trial["chromosome"], baseline=trial["baseline"], jit_compile=False)
        result = run_training(model, train, validation, test, plan, trial, attempt, mode)
        return 0 if result["status"] in ("nonfinite", "nonfinite_test", "numerics_check_triggered", "not_reproduced") else 1
    except Exception as exc:
        write_json(attempt / "result.json", dict(trial_id=trial_id, status="runtime_error", error=repr(exc)))
        (attempt / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        traceback.print_exc()
        return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--trial", action="append", default=[])
    parser.add_argument("--mode", choices=("replay", "check"), default="check")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--attempt", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.resume == bool(args.reference_run):
        parser.error("Use --reference-run for preparation, or --resume for an existing plan")
    try:
        if not args.resume:
            prepare(args.reference_run, output)
        plan, diagnostic = verify(output)
        available = {f["trial_id"] for f in diagnostic["failures"]}
        if set(args.trial)-available:
            raise ValueError("Requested trial is not a recorded nonfinite failure")
        if args.worker:
            return worker(output, args.trial[0], args.attempt, args.mode)
        if args.plan_only:
            print(f"Frozen diagnostic plan: {output}; {len(available)} failed trials")
            print("\n".join(sorted(available)))
            return 0
        if not args.trial:
            raise ValueError("Specify --trial ID (repeatable) to limit the diagnostic workload")
        lock = output / "diagnostic.lock"
        with lock.open("x") as stream:
            stream.write(str(os.getpid()))
        try:
            for trial_id in dict.fromkeys(args.trial):
                folder = output / "diagnostics" / trial_id / args.mode
                folder.mkdir(parents=True, exist_ok=True)
                attempt = folder / f"attempt_{len(list(folder.glob('attempt_*')))+1:04d}"
                attempt.mkdir()
                command = [sys.executable, str(output / "diagnose_retraining.py"), "--output", str(output),
                           "--resume", "--worker", "--trial", trial_id, "--mode", args.mode, "--attempt", str(attempt)]
                print(f"{trial_id} ({args.mode}): {attempt / 'worker.log'}", flush=True)
                with (attempt / "worker.log").open("w", encoding="utf-8") as log:
                    child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                    try:
                        code = child.wait()
                    except KeyboardInterrupt:
                        child.terminate()
                        child.wait()
                        raise
                if code:
                    return code
                print((attempt / "result.json").read_text(), flush=True)
        finally:
            lock.unlink()
        return 0
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    sys.exit(main())
